"""
Training Script

Full training pipeline:
1. Compute attention weights from corpus
2. Freeze attention weights
3. Train MLP layers
4. Save checkpoints, metrics, and final model
"""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from config import ModelConfig, TrainingConfig, RunConfig
from transformer import SparseAttentionTransformer, create_model_from_weight_dicts
from weight_loader import WeightComputer
from data_loader import (
    create_dataloaders, 
    collect_token_batches,
    save_test_data,
)


class TrainingRun:
    """
    Manages a complete training run including:
    - Directory setup
    - Weight computation
    - Model training
    - Checkpointing
    - Metrics logging
    """
    
    def __init__(
        self,
        run_config: RunConfig,
        run_dir: Optional[str] = None,
        use_fineweb: bool = True,
        local_texts: Optional[List[str]] = None,
    ):
        self.run_config = run_config
        self.model_config = run_config.model
        self.training_config = run_config.training
        self.use_fineweb = use_fineweb
        self.local_texts = local_texts
        
        # Setup device
        if run_config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif run_config.device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Setup run directory
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_dir = f"runs/{timestamp}"
        
        self.run_dir = Path(run_dir)
        self._setup_directories()
        
        # Will be initialized during training
        self.model: Optional[SparseAttentionTransformer] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None
        self.scaler = None
        
        # Metrics
        self.metrics_history: List[Dict] = []
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _setup_directories(self):
        """Create run directory structure."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.snapshots_dir = self.run_dir / "snapshots"
        self.snapshots_dir.mkdir(exist_ok=True)
        
        self.measurements_dir = self.run_dir / "measurements"
        self.measurements_dir.mkdir(exist_ok=True)
        
        self.completed_dir = self.run_dir / "completed"
        self.completed_dir.mkdir(exist_ok=True)
        
        self.test_data_dir = self.run_dir / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Save config
        self.run_config.save(str(self.run_dir / "config.json"))
        print(f"Run directory: {self.run_dir}")
    
    def compute_attention_weights(self) -> Dict:
        """Compute attention weights from corpus."""
        print("\n" + "=" * 60)
        print("Phase 1: Computing attention weights from corpus")
        print("=" * 60)
        
        computer = WeightComputer(
            self.model_config,
            self.training_config,
            fineweb_subset=self.run_config.fineweb_subset,
        )
        
        computer.compute_all_weights(
            from_fineweb=self.use_fineweb,
            local_texts=self.local_texts,
            orthogonalize=True,
        )
        
        # Save weights
        weights_path = str(self.run_dir / "attention_weights.npz")
        computer.save_weights(weights_path)
        
        return computer.get_all_layer_weights()
    
    def create_model(self, attention_weights: Dict) -> SparseAttentionTransformer:
        """Create model with computed attention weights."""
        print("\n" + "=" * 60)
        print("Phase 2: Creating model")
        print("=" * 60)
        
        self.model = SparseAttentionTransformer(
            self.model_config,
            attention_weights,
        )
        
        self.model = self.model.to(self.device)
        self.model.freeze_attention()
        
        # Print parameter counts
        total = self.model.get_num_params()
        trainable = self.model.get_num_params(trainable_only=True)
        
        print(f"Model created:")
        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")
        print(f"  Frozen parameters: {total - trainable:,}")
        
        return self.model
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.training_config.learning_rate,
            betas=self.training_config.betas,
            weight_decay=self.training_config.weight_decay,
        )
        
        # Cosine schedule with warmup
        def lr_lambda(step):
            if step < self.training_config.warmup_steps:
                return step / self.training_config.warmup_steps
            progress = (step - self.training_config.warmup_steps) / (
                self.training_config.max_steps - self.training_config.warmup_steps
            )
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Mixed precision scaler (for CUDA)
        if self.device.type == "cuda":
            self.scaler = GradScaler()
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""
        self.model.train()
        
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            with autocast():
                loss, _ = self.model.compute_loss(batch)
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, _ = self.model.compute_loss(batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.max_grad_norm
            )
            
            self.optimizer.step()
        
        self.scheduler.step()
        self.global_step += 1
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            loss, logits = self.model.compute_loss(batch)
            
            # Count tokens (excluding first position which has no target)
            num_tokens = batch.numel() - batch.size(0)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))  # Cap to avoid overflow
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
    
    def save_snapshot(self, name: str = None):
        """Save model snapshot."""
        if name is None:
            name = f"step_{self.global_step}"
        
        path = self.snapshots_dir / f"{name}.pt"
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }, path)
        
        print(f"Saved snapshot: {path}")
    
    def save_metrics(self, metrics: Dict):
        """Save metrics to JSON."""
        metrics["step"] = self.global_step
        metrics["timestamp"] = datetime.now().isoformat()
        
        self.metrics_history.append(metrics)
        
        # Save individual metric file
        path = self.measurements_dir / f"metrics_{self.global_step}.json"
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save full history
        history_path = self.measurements_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def save_final_model(self):
        """Save final completed model."""
        # Save model
        model_path = self.completed_dir / "model.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "final_step": self.global_step,
        }, model_path)
        
        # Save config
        self.run_config.save(str(self.completed_dir / "config.json"))
        
        # Copy eval script template
        eval_template = self.run_dir / "eval.py"
        self._write_eval_script(eval_template)
        
        print(f"Saved final model to {self.completed_dir}")
    
    def _write_eval_script(self, path: Path):
        """Write evaluation script."""
        script = '''#!/usr/bin/env python3
"""
Evaluation script for trained model.
Run from the run directory.
"""

import sys
import torch
from pathlib import Path

# Add model directory to path
model_dir = Path(__file__).parent.parent.parent / "model"
sys.path.insert(0, str(model_dir))

from config import ModelConfig, RunConfig
from transformer import SparseAttentionTransformer
from data_loader import load_test_data

def main():
    run_dir = Path(__file__).parent
    
    # Load config
    config = RunConfig.load(str(run_dir / "config.json"))
    
    # Load test data
    test_data = load_test_data(str(run_dir / "test_data" / "test_tokens.pt"))
    
    # Load model
    checkpoint = torch.load(run_dir / "completed" / "model.pt", map_location="cpu")
    
    # Create model (need attention weights)
    weights_path = run_dir / "attention_weights.npz"
    if weights_path.exists():
        from weight_loader import WeightComputer
        computer = WeightComputer.load_weights(str(weights_path))
        attention_weights = computer.get_all_layer_weights()
    else:
        print("Warning: No attention weights found, using random")
        attention_weights = {}
    
    model = SparseAttentionTransformer(config.model, attention_weights)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Evaluate
    total_loss = 0
    total_tokens = 0
    
    for seq in test_data:
        input_ids = torch.tensor([seq])
        with torch.no_grad():
            loss, _ = model.compute_loss(input_ids)
        total_loss += loss.item() * (len(seq) - 1)
        total_tokens += len(seq) - 1
    
    avg_loss = total_loss / total_tokens
    perplexity = 2.718 ** avg_loss
    
    print(f"Test Results:")
    print(f"  Sequences: {len(test_data)}")
    print(f"  Tokens: {total_tokens}")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")

if __name__ == "__main__":
    main()
'''
        with open(path, 'w') as f:
            f.write(script)
    
    def train(self):
        """Full training loop."""
        print("\n" + "=" * 60)
        print("Phase 3: Training MLP layers")
        print("=" * 60)
        
        # Create data loaders
        train_loader, test_loader, test_sequences = create_dataloaders(
            self.training_config,
            self.model_config,
            fineweb_subset=self.run_config.fineweb_subset,
            use_fineweb=self.use_fineweb,
            local_texts=self.local_texts,
        )
        
        # Save test data
        if test_sequences:
            save_test_data(test_sequences, str(self.test_data_dir / "test_tokens.pt"))
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Training loop
        print(f"\nStarting training for {self.training_config.max_steps} steps...")
        
        running_loss = 0.0
        num_batches = 0
        
        while self.global_step < self.training_config.max_steps:
            for batch in train_loader:
                if self.global_step >= self.training_config.max_steps:
                    break
                
                loss = self.train_step(batch)
                running_loss += loss
                num_batches += 1
                
                # Logging
                if self.global_step % self.training_config.log_interval == 0:
                    avg_loss = running_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                    running_loss = 0.0
                    num_batches = 0
                
                # Evaluation
                if self.global_step % self.training_config.eval_interval == 0:
                    metrics = self.evaluate(test_loader)
                    metrics["train_loss"] = loss
                    metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
                    
                    print(f"  Eval: loss={metrics['loss']:.4f}, ppl={metrics['perplexity']:.2f}")
                    
                    self.save_metrics(metrics)
                    
                    # Track best
                    if metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = metrics["loss"]
                        self.save_snapshot("best")
                
                # Snapshots
                if self.global_step % self.training_config.snapshot_interval == 0:
                    self.save_snapshot()
        
        # Final evaluation and save
        print("\nTraining complete!")
        
        final_metrics = self.evaluate(test_loader)
        final_metrics["final"] = True
        self.save_metrics(final_metrics)
        
        print(f"Final: loss={final_metrics['loss']:.4f}, ppl={final_metrics['perplexity']:.2f}")
        
        self.save_snapshot("final")
        self.save_final_model()
    
    def run(self):
        """Execute full training pipeline."""
        # Phase 1: Compute weights
        attention_weights = self.compute_attention_weights()
        
        # Phase 2: Create model
        self.create_model(attention_weights)
        
        # Phase 3: Train
        self.train()
        
        print("\n" + "=" * 60)
        print(f"Training complete! Results saved to: {self.run_dir}")
        print("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Sparse Attention Transformer")
    parser.add_argument("--num-chunks", type=int, default=100, help="Training data chunks")
    parser.add_argument("--test-chunks", type=int, default=20, help="Test data chunks")
    parser.add_argument("--max-steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-n", type=int, default=12, help="Max n-gram size")
    parser.add_argument("--max-s", type=int, default=11, help="Max skip value")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--subset", type=str, default="sample-10BT", help="FineWeb subset")
    parser.add_argument("--local", action="store_true", help="Use local test data")
    parser.add_argument("--run-dir", type=str, default=None, help="Run directory")
    
    args = parser.parse_args()
    
    # Create configs
    model_config = ModelConfig()
    training_config = TrainingConfig(
        num_chunks=args.num_chunks,
        test_chunks=args.test_chunks,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_n=args.max_n,
        max_s=args.max_s,
    )
    run_config = RunConfig(
        model=model_config,
        training=training_config,
        fineweb_subset=args.subset,
        device=args.device,
    )
    
    # Local test data
    local_texts = None
    if args.local:
        local_texts = [
            "The quick brown fox jumps over the lazy dog. " * 1000,
            "Machine learning enables computers to learn from data. " * 1000,
            "Natural language processing is a fascinating field. " * 1000,
        ]
    
    # Run training
    run = TrainingRun(
        run_config,
        run_dir=args.run_dir,
        use_fineweb=not args.local,
        local_texts=local_texts,
    )
    run.run()


if __name__ == "__main__":
    main()


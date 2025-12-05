"""
PicoGPT Comparison Training

Trains both sparse (frozen attention) and trainable models for comparison.
Outputs metrics, training curves, and final evaluation.
"""

import sys
from pathlib import Path
import json
import math
from datetime import datetime
from typing import List, Dict, Optional
import time

import torch
import torch.optim as optim
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Components" / "method"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Components" / "model"))

from pico_config import PicoModelConfig, PicoTrainingConfig
from pico_sparse import PicoGPTSparse, compute_pico_weights
from pico_trainable import PicoGPTTrainable


class PicoTrainer:
    """Trainer for PicoGPT models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: PicoModelConfig,
        training_config: PicoTrainingConfig,
        name: str,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.training_config = training_config
        self.name = name
        self.device = device
        
        # Only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=training_config.learning_rate,
            betas=training_config.betas,
            weight_decay=training_config.weight_decay,
        )
        
        # Cosine schedule with warmup
        def lr_lambda(step):
            if step < training_config.warmup_steps:
                return step / max(training_config.warmup_steps, 1)
            progress = (step - training_config.warmup_steps) / max(
                training_config.max_steps - training_config.warmup_steps, 1
            )
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        self.metrics_history = []
        self.global_step = 0
    
    def train_step(self, batch: torch.Tensor) -> float:
        """Single training step."""
        self.model.train()
        batch = batch.to(self.device)
        
        self.optimizer.zero_grad()
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
    def evaluate(self, batches: List[torch.Tensor]) -> Dict[str, float]:
        """Evaluate on batches."""
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0
        
        for batch in batches:
            batch = batch.to(self.device)
            loss, _ = self.model.compute_loss(batch)
            num_tokens = batch.numel() - batch.size(0)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
        }
    
    def log_metrics(self, train_loss: float, eval_metrics: Dict):
        """Log metrics."""
        metrics = {
            "step": self.global_step,
            "train_loss": train_loss,
            "eval_loss": eval_metrics["loss"],
            "eval_perplexity": eval_metrics["perplexity"],
            "lr": self.scheduler.get_last_lr()[0],
        }
        self.metrics_history.append(metrics)
        return metrics


def create_data_batches(
    texts: List[str],
    seq_len: int,
    batch_size: int,
    max_batches: int,
) -> List[torch.Tensor]:
    """Create data batches from texts using character-level tokenization."""
    from pico_tokenizer import PicoTokenizer
    
    tokenizer = PicoTokenizer()
    
    # Tokenize all texts (character-level)
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))
    
    # Create sequences
    sequences = []
    for i in range(0, len(all_tokens) - seq_len + 1, seq_len):
        sequences.append(all_tokens[i:i + seq_len])
        if len(sequences) >= max_batches * batch_size:
            break
    
    # Create batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        if len(batch_seqs) == batch_size:
            batch = torch.tensor(batch_seqs, dtype=torch.long)
            batches.append(batch)
    
    return batches


def run_comparison(
    config: PicoModelConfig = None,
    training_config: PicoTrainingConfig = None,
    texts: List[str] = None,
    output_dir: str = None,
    device: str = "cpu",
    skip_corpus: bool = False,
    seed: int = 42,
):
    """
    Run comparison training between sparse and trainable models.
    
    Uses character-level tokenization (~80 vocab) for tractable full algorithm.
    
    Args:
        skip_corpus: If True, skip corpus computation (for debugging only)
        seed: Random seed for reproducibility
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if config is None:
        config = PicoModelConfig()
    if training_config is None:
        training_config = PicoTrainingConfig()
    
    # Default test texts
    if texts is None:
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 200,
            "Machine learning models learn patterns from data. " * 200,
            "Natural language processing enables text understanding. " * 200,
            "Deep learning has transformed artificial intelligence. " * 200,
            "Transformers use attention mechanisms for sequence modeling. " * 200,
        ]
    
    # Setup device
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    if skip_corpus:
        print("Debug mode: Using random weights (skipping corpus computation)")
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"pico_comparison_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    # Create data batches
    print("\nCreating data batches...")
    train_batches = create_data_batches(
        texts,
        config.max_seq_len,
        training_config.batch_size,
        max_batches=training_config.num_chunks,
    )
    
    test_batches = create_data_batches(
        texts[-1:],  # Use last text for testing
        config.max_seq_len,
        training_config.batch_size,
        max_batches=training_config.test_chunks,
    )
    
    print(f"Train batches: {len(train_batches)}")
    print(f"Test batches: {len(test_batches)}")
    
    # ===== Create Sparse Model (corpus-derived weights) =====
    print("\n" + "=" * 60)
    print("Creating Sparse Model (corpus-derived frozen attention)")
    print("=" * 60)
    
    attention_weights, base_embeddings = compute_pico_weights(config, training_config, texts, skip_corpus=skip_corpus)
    print(f"Base embeddings shape: {base_embeddings.shape} (will be FROZEN in model)")
    sparse_model = PicoGPTSparse(config, attention_weights, base_embeddings=base_embeddings)
    
    sparse_total = sparse_model.get_num_params()
    sparse_trainable = sparse_model.get_num_params(trainable_only=True)
    print(f"Sparse Model - Total: {sparse_total:,}, Trainable: {sparse_trainable:,}")
    
    # ===== Create Zero Model (ablation baseline) =====
    print("\n" + "=" * 60)
    print("Creating Zero Model (all-zero frozen attention)")
    print("=" * 60)
    
    # Create zero attention weights (same structure, all zeros)
    zero_weights = {}
    for layer_idx in range(config.n_layers):
        W_Q = torch.zeros(config.n_heads, config.embedding_dim, config.head_dim)
        W_K = torch.zeros(config.n_heads, config.embedding_dim, config.head_dim)
        W_V = torch.zeros(config.n_heads, config.head_dim, config.embedding_dim)
        W_O = torch.zeros(config.embedding_dim, config.n_heads * config.head_dim)
        zero_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
    
    zero_model = PicoGPTSparse(config, zero_weights, base_embeddings=base_embeddings)
    
    zero_total = zero_model.get_num_params()
    zero_trainable = zero_model.get_num_params(trainable_only=True)
    print(f"Zero Model - Total: {zero_total:,}, Trainable: {zero_trainable:,}")
    
    # ===== Create Trainable Model =====
    print("\n" + "=" * 60)
    print("Creating Trainable Model (all weights trainable)")
    print("=" * 60)
    
    trainable_model = PicoGPTTrainable(config)
    
    trainable_total = trainable_model.get_num_params()
    trainable_trainable = trainable_model.get_num_params(trainable_only=True)
    print(f"Trainable Model - Total: {trainable_total:,}, Trainable: {trainable_trainable:,}")
    
    # ===== Train All Models =====
    print("\n" + "=" * 60)
    print("Training all three models")
    print("=" * 60)
    
    sparse_trainer = PicoTrainer(sparse_model, config, training_config, "sparse", device)
    zero_trainer = PicoTrainer(zero_model, config, training_config, "zero", device)
    trainable_trainer = PicoTrainer(trainable_model, config, training_config, "trainable", device)
    
    print(f"\nTraining for {training_config.max_steps} steps...")
    
    batch_idx = 0
    
    for step in range(training_config.max_steps):
        # Get batch (cycle through)
        batch = train_batches[batch_idx % len(train_batches)]
        batch_idx += 1
        
        # Train step
        sparse_loss = sparse_trainer.train_step(batch)
        zero_loss = zero_trainer.train_step(batch)
        trainable_loss = trainable_trainer.train_step(batch)
        
        # Logging
        if (step + 1) % training_config.log_interval == 0:
            print(f"Step {step + 1}: Sparse={sparse_loss:.4f}, Zero={zero_loss:.4f}, Trainable={trainable_loss:.4f}")
        
        # Evaluation
        if (step + 1) % training_config.eval_interval == 0:
            sparse_eval = sparse_trainer.evaluate(test_batches)
            zero_eval = zero_trainer.evaluate(test_batches)
            trainable_eval = trainable_trainer.evaluate(test_batches)
            
            sparse_trainer.log_metrics(sparse_loss, sparse_eval)
            zero_trainer.log_metrics(zero_loss, zero_eval)
            trainable_trainer.log_metrics(trainable_loss, trainable_eval)
            
            print(f"  Eval - Sparse: loss={sparse_eval['loss']:.4f}, ppl={sparse_eval['perplexity']:.2f}")
            print(f"  Eval - Zero: loss={zero_eval['loss']:.4f}, ppl={zero_eval['perplexity']:.2f}")
            print(f"  Eval - Trainable: loss={trainable_eval['loss']:.4f}, ppl={trainable_eval['perplexity']:.2f}")
    
    # ===== Final Evaluation =====
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    final_sparse = sparse_trainer.evaluate(test_batches)
    final_zero = zero_trainer.evaluate(test_batches)
    final_trainable = trainable_trainer.evaluate(test_batches)
    
    print(f"\nSparse Model (corpus-derived):")
    print(f"  Loss: {final_sparse['loss']:.4f}")
    print(f"  Perplexity: {final_sparse['perplexity']:.2f}")
    print(f"  Trainable params: {sparse_trainable:,}")
    
    print(f"\nZero Model (ablation baseline):")
    print(f"  Loss: {final_zero['loss']:.4f}")
    print(f"  Perplexity: {final_zero['perplexity']:.2f}")
    print(f"  Trainable params: {zero_trainable:,}")
    
    print(f"\nTrainable Model:")
    print(f"  Loss: {final_trainable['loss']:.4f}")
    print(f"  Perplexity: {final_trainable['perplexity']:.2f}")
    print(f"  Trainable params: {trainable_trainable:,}")
    
    # ===== Save Results =====
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)
    
    results = {
        "config": config.to_dict(),
        "training_config": training_config.to_dict(),
        "sparse_model": {
            "total_params": sparse_total,
            "trainable_params": sparse_trainable,
            "final_loss": final_sparse["loss"],
            "final_perplexity": final_sparse["perplexity"],
            "history": sparse_trainer.metrics_history,
        },
        "zero_model": {
            "total_params": zero_total,
            "trainable_params": zero_trainable,
            "final_loss": final_zero["loss"],
            "final_perplexity": final_zero["perplexity"],
            "history": zero_trainer.metrics_history,
        },
        "trainable_model": {
            "total_params": trainable_total,
            "trainable_params": trainable_trainable,
            "final_loss": final_trainable["loss"],
            "final_perplexity": final_trainable["perplexity"],
            "history": trainable_trainer.metrics_history,
        },
    }
    
    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save models
    torch.save(sparse_model.state_dict(), output_path / "sparse_model.pt")
    torch.save(zero_model.state_dict(), output_path / "zero_model.pt")
    torch.save(trainable_model.state_dict(), output_path / "trainable_model.pt")
    
    print(f"\nResults saved to {output_path}")
    
    # ===== Summary =====
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Metric':<25} {'Sparse':<12} {'Zero':<12} {'Trainable':<12}")
    print("-" * 65)
    print(f"{'Total Parameters':<25} {sparse_total:<12,} {zero_total:<12,} {trainable_total:<12,}")
    print(f"{'Trainable Parameters':<25} {sparse_trainable:<12,} {zero_trainable:<12,} {trainable_trainable:<12,}")
    print(f"{'Final Loss':<25} {final_sparse['loss']:<12.4f} {final_zero['loss']:<12.4f} {final_trainable['loss']:<12.4f}")
    print(f"{'Final Perplexity':<25} {final_sparse['perplexity']:<12.2f} {final_zero['perplexity']:<12.2f} {final_trainable['perplexity']:<12.2f}")
    
    # Show improvement of sparse over zero
    if final_zero['perplexity'] > 0:
        improvement = (final_zero['perplexity'] - final_sparse['perplexity']) / final_zero['perplexity'] * 100
        print(f"\nSparse model perplexity is {improvement:.1f}% {'better' if improvement > 0 else 'worse'} than Zero baseline")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare PicoGPT models")
    parser.add_argument("--max-steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Skip corpus computation (use random weights for debugging)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    config = PicoModelConfig()
    training_config = PicoTrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_interval=20,
        eval_interval=100,
    )
    
    run_comparison(
        config=config,
        training_config=training_config,
        output_dir=args.output,
        device=args.device,
        skip_corpus=args.fast,
        seed=args.seed,
    )


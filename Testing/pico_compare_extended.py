"""
Extended PicoGPT Comparison Training

Compares 5 model variants:
1. Sparse (original with learned position embeddings)
2. Sinusoidal (sinusoidal position encoding + skip-n masking)
3. RoPE (rotary position encoding + skip-n masking)
4. Trainable (baseline with SGD-like training)
5. Zero (ablation baseline with zero attention weights)
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
sys.path.insert(0, str(Path(__file__).parent / "Position correction attempts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Components" / "method"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Components" / "model"))

from pico_config import PicoModelConfig, PicoTrainingConfig
from pico_sparse import PicoGPTSparse, compute_pico_weights
from pico_trainable import PicoGPTTrainable

# Import position encoding variants
sys.path.insert(0, str(Path(__file__).parent / "Position correction attempts"))
from pico_sparse_sinusoidal import PicoGPTSparseSinusoidal
from pico_sparse_rope import PicoGPTSparseRoPE


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


def run_extended_comparison(
    config: PicoModelConfig = None,
    training_config: PicoTrainingConfig = None,
    texts: List[str] = None,
    output_dir: str = None,
    device: str = "cpu",
    skip_corpus: bool = False,
    seed: int = 42,
):
    """
    Run extended comparison training with 5 model variants.
    
    Models:
    1. Sparse (original with learned position embeddings)
    2. Sinusoidal (sinusoidal position encoding + skip-n masking)
    3. RoPE (rotary position encoding + skip-n masking)
    4. Trainable (baseline with all weights trainable)
    5. Zero (ablation baseline with zero attention weights)
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
        output_dir = f"pico_extended_comparison_{timestamp}"
    
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
    
    # Compute shared attention weights and embeddings
    print("\n" + "=" * 60)
    print("Computing shared attention weights from corpus")
    print("=" * 60)
    
    attention_weights, base_embeddings = compute_pico_weights(
        config, training_config, texts, skip_corpus=skip_corpus
    )
    print(f"Base embeddings shape: {base_embeddings.shape}")
    
    # ===== Create Model 1: Sparse (Original) =====
    print("\n" + "=" * 60)
    print("Model 1: Sparse (Original with learned position embeddings)")
    print("=" * 60)
    
    sparse_model = PicoGPTSparse(config, attention_weights, base_embeddings=base_embeddings)
    sparse_total = sparse_model.get_num_params()
    sparse_trainable = sparse_model.get_num_params(trainable_only=True)
    print(f"Total: {sparse_total:,}, Trainable: {sparse_trainable:,}")
    
    # ===== Create Model 2: Sinusoidal =====
    print("\n" + "=" * 60)
    print("Model 2: Sinusoidal (Sinusoidal position encoding + skip-n masking)")
    print("=" * 60)
    
    sinusoidal_model = PicoGPTSparseSinusoidal(config, attention_weights, base_embeddings=base_embeddings)
    sinusoidal_total = sinusoidal_model.get_num_params()
    sinusoidal_trainable = sinusoidal_model.get_num_params(trainable_only=True)
    print(f"Total: {sinusoidal_total:,}, Trainable: {sinusoidal_trainable:,}")
    
    # ===== Create Model 3: RoPE =====
    print("\n" + "=" * 60)
    print("Model 3: RoPE (Rotary position encoding + skip-n masking)")
    print("=" * 60)
    
    rope_model = PicoGPTSparseRoPE(config, attention_weights, base_embeddings=base_embeddings)
    rope_total = rope_model.get_num_params()
    rope_trainable = rope_model.get_num_params(trainable_only=True)
    print(f"Total: {rope_total:,}, Trainable: {rope_trainable:,}")
    
    # ===== Create Model 4: Trainable (Baseline) =====
    print("\n" + "=" * 60)
    print("Model 4: Trainable (All weights trainable - SGD baseline)")
    print("=" * 60)
    
    trainable_model = PicoGPTTrainable(config)
    trainable_total = trainable_model.get_num_params()
    trainable_trainable = trainable_model.get_num_params(trainable_only=True)
    print(f"Total: {trainable_total:,}, Trainable: {trainable_trainable:,}")
    
    # ===== Create Model 5: Zero (Ablation) =====
    print("\n" + "=" * 60)
    print("Model 5: Zero (Zero attention weights - ablation baseline)")
    print("=" * 60)
    
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
    print(f"Total: {zero_total:,}, Trainable: {zero_trainable:,}")
    
    # ===== Train All Models =====
    print("\n" + "=" * 60)
    print("Training all 5 models")
    print("=" * 60)
    
    trainers = {
        "sparse": PicoTrainer(sparse_model, config, training_config, "sparse", device),
        "sinusoidal": PicoTrainer(sinusoidal_model, config, training_config, "sinusoidal", device),
        "rope": PicoTrainer(rope_model, config, training_config, "rope", device),
        "trainable": PicoTrainer(trainable_model, config, training_config, "trainable", device),
        "zero": PicoTrainer(zero_model, config, training_config, "zero", device),
    }
    
    print(f"\nTraining for {training_config.max_steps} steps...")
    
    batch_idx = 0
    start_time = time.time()
    
    for step in range(training_config.max_steps):
        # Get batch (cycle through)
        batch = train_batches[batch_idx % len(train_batches)]
        batch_idx += 1
        
        # Train step for all models
        losses = {}
        for name, trainer in trainers.items():
            losses[name] = trainer.train_step(batch)
        
        # Logging
        if (step + 1) % training_config.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(f"Step {step + 1}/{training_config.max_steps} ({steps_per_sec:.2f} steps/s):")
            print(f"  Sparse: {losses['sparse']:.4f} | Sinusoidal: {losses['sinusoidal']:.4f} | RoPE: {losses['rope']:.4f}")
            print(f"  Trainable: {losses['trainable']:.4f} | Zero: {losses['zero']:.4f}")
        
        # Evaluation
        if (step + 1) % training_config.eval_interval == 0:
            print(f"\nEvaluation at step {step + 1}:")
            for name, trainer in trainers.items():
                eval_metrics = trainer.evaluate(test_batches)
                trainer.log_metrics(losses[name], eval_metrics)
                print(f"  {name.capitalize():12s}: loss={eval_metrics['loss']:.4f}, ppl={eval_metrics['perplexity']:.2f}")
            print()
    
    # ===== Final Evaluation =====
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    final_results = {}
    for name, trainer in trainers.items():
        final_results[name] = trainer.evaluate(test_batches)
    
    # Print results
    print(f"\n{'Model':<15} {'Loss':<10} {'Perplexity':<12} {'Trainable Params':<18}")
    print("-" * 60)
    
    model_params = {
        "sparse": (sparse_total, sparse_trainable),
        "sinusoidal": (sinusoidal_total, sinusoidal_trainable),
        "rope": (rope_total, rope_trainable),
        "trainable": (trainable_total, trainable_trainable),
        "zero": (zero_total, zero_trainable),
    }
    
    for name in ["sparse", "sinusoidal", "rope", "trainable", "zero"]:
        metrics = final_results[name]
        _, trainable = model_params[name]
        print(f"{name.capitalize():<15} {metrics['loss']:<10.4f} {metrics['perplexity']:<12.2f} {trainable:<18,}")
    
    # ===== Save Results =====
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)
    
    results = {
        "config": config.to_dict(),
        "training_config": training_config.to_dict(),
        "models": {}
    }
    
    for name, trainer in trainers.items():
        total, trainable = model_params[name]
        results["models"][name] = {
            "total_params": total,
            "trainable_params": trainable,
            "final_loss": final_results[name]["loss"],
            "final_perplexity": final_results[name]["perplexity"],
            "history": trainer.metrics_history,
        }
    
    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save models
    for name, trainer in trainers.items():
        torch.save(trainer.model.state_dict(), output_path / f"{name}_model.pt")
    
    print(f"\nResults saved to {output_path}")
    
    # ===== Comparison Analysis =====
    print("\n" + "=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Compare position encoding variants
    print("\nPosition Encoding Variants (all use frozen attention):")
    print(f"  Original (learned):  loss={final_results['sparse']['loss']:.4f}, ppl={final_results['sparse']['perplexity']:.2f}")
    print(f"  Sinusoidal (fixed):  loss={final_results['sinusoidal']['loss']:.4f}, ppl={final_results['sinusoidal']['perplexity']:.2f}")
    print(f"  RoPE (rotary):       loss={final_results['rope']['loss']:.4f}, ppl={final_results['rope']['perplexity']:.2f}")
    
    # Find best position encoding
    pos_variants = {"sparse": final_results['sparse']['perplexity'],
                    "sinusoidal": final_results['sinusoidal']['perplexity'],
                    "rope": final_results['rope']['perplexity']}
    best_pos = min(pos_variants, key=pos_variants.get)
    print(f"\n  → Best position encoding: {best_pos.upper()} (ppl={pos_variants[best_pos]:.2f})")
    
    # Compare against baselines
    print("\nBaseline Comparisons:")
    zero_ppl = final_results['zero']['perplexity']
    trainable_ppl = final_results['trainable']['perplexity']
    
    for name in ["sparse", "sinusoidal", "rope"]:
        ppl = final_results[name]['perplexity']
        vs_zero = (zero_ppl - ppl) / zero_ppl * 100
        vs_trainable = (trainable_ppl - ppl) / trainable_ppl * 100
        print(f"  {name.capitalize():12s}: {vs_zero:+.1f}% vs Zero, {vs_trainable:+.1f}% vs Trainable")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extended PicoGPT comparison with position encoding variants")
    parser.add_argument("--max-steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--fast", action="store_true", help="Skip corpus computation (use random weights)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    config = PicoModelConfig()
    training_config = PicoTrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_interval=20,
        eval_interval=100,
    )
    
    run_extended_comparison(
        config=config,
        training_config=training_config,
        output_dir=args.output,
        device=args.device,
        skip_corpus=args.fast,
        seed=args.seed,
    )

"""
Complete PicoGPT Comparison Training - All 6 Variants

Compares 6 model variants:
1. Sparse (original with learned position embeddings)
2. Sinusoidal (sinusoidal position encoding + causal masking)
3. Sinusoidal G_k (sinusoidal + G_k correction for positional bias)
4. RoPE (rotary position encoding + causal masking)
5. Trainable (baseline with SGD-like training)
6. Zero (ablation baseline with zero attention weights)
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
from pico_sparse_sinusoidal import PicoGPTSparseSinusoidal
from pico_sparse_rope import PicoGPTSparseRoPE
from pico_sparse_sinusoidal_gk import PicoGPTSparseGkCorrected
from pico_method_gk import compute_pico_attention_weights_gk_corrected


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


def run_complete_comparison(
    config: PicoModelConfig = None,
    training_config: PicoTrainingConfig = None,
    texts: List[str] = None,
    output_dir: str = None,
    device: str = "cpu",
    skip_corpus: bool = False,
    seed: int = 42,
):
    """Run complete comparison training with 6 model variants."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if config is None:
        config = PicoModelConfig()
    if training_config is None:
        training_config = PicoTrainingConfig()
    
    # Load real data from FineWeb (not toy texts!)
    if texts is None:
        print("\nLoading FineWeb data...")
        
        if not skip_corpus:
            # Load real text from FineWeb
            from datasets import load_dataset
            
            dataset = load_dataset(
                "HuggingFaceFW/fineweb",
                name="sample-10BT",
                split="train",
                streaming=True,
            )
            
            # Collect 100 text samples from FineWeb
            texts = []
            for i, example in enumerate(dataset):
                texts.append(example["text"])
                if i >= 99:  # Get 100 text samples
                    break
            
            print(f"Loaded {len(texts)} text samples from FineWeb")
            print(f"Sample lengths (first 5): {[len(t) for t in texts[:5]]}")
            
            # Split into: corpus computation (70), training (20), test (10)
            # This ensures test set is truly held-out
            corpus_texts = texts[:70]  # For computing attention weights
            train_texts = texts[70:90]  # For training
            test_texts = texts[90:100]  # Held-out test set
            
            print(f"\nData split:")
            print(f"  Corpus (weight computation): {len(corpus_texts)} samples")
            print(f"  Training: {len(train_texts)} samples")
            print(f"  Test (held-out): {len(test_texts)} samples")
        else:
            # Fast mode - use toy texts
            print("Fast mode: using toy texts")
            corpus_texts = [
                "The quick brown fox jumps over the lazy dog. " * 200,
                "Machine learning models learn patterns from data. " * 200,
            ] * 35  # 70 samples
            train_texts = [
                "Natural language processing enables text understanding. " * 200,
                "Deep learning has transformed artificial intelligence. " * 200,
            ] * 10  # 20 samples
            test_texts = [
                "Transformers use attention mechanisms for sequence modeling. " * 200,
            ] * 10  # 10 samples
    else:
        # If texts provided, split them
        total = len(texts)
        corpus_texts = texts[:int(total * 0.7)]
        train_texts = texts[int(total * 0.7):int(total * 0.9)]
        test_texts = texts[int(total * 0.9):]
        print(f"\nData split (from provided texts):")
        print(f"  Corpus: {len(corpus_texts)} samples")
        print(f"  Training: {len(train_texts)} samples")
        print(f"  Test: {len(test_texts)} samples")
    
    # Setup device
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"pico_complete_comparison_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}")
    
    # Create data batches
    print("\nCreating data batches...")
    train_batches = create_data_batches(
        train_texts, config.max_seq_len, training_config.batch_size,
        max_batches=training_config.num_chunks,
    )
    test_batches = create_data_batches(
        test_texts, config.max_seq_len, training_config.batch_size,
        max_batches=training_config.test_chunks,
    )
    print(f"Train batches: {len(train_batches)}, Test batches: {len(test_batches)}")
    
    # Compute shared attention weights FROM CORPUS SET ONLY (not training/test)
    print("\n" + "=" * 60)
    print("Computing attention weights from corpus (held-out from train/test)")
    print("=" * 60)
    
    attention_weights, base_embeddings = compute_pico_weights(
        config, training_config, corpus_texts, skip_corpus=skip_corpus
    )
    
    # Compute G_k-corrected weights FROM CORPUS SET ONLY
    print("\n" + "=" * 60)
    print("Computing G_k-corrected attention weights (from corpus set)")
    print("=" * 60)
    
    gk_weights, gk_embeddings = compute_pico_attention_weights_gk_corrected(
        texts=corpus_texts,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        embedding_dim=config.embedding_dim,
        head_dim=config.head_dim,
        max_n=training_config.max_n,
        max_s=training_config.max_s,
        verbose=True,
    )
    
    # Create all 6 models
    print("\n" + "=" * 60)
    print("Creating all 6 models")
    print("=" * 60)
    
    models = {}
    
    # 1. Sparse (original)
    print("\n1. Sparse (learned position embeddings)")
    models['sparse'] = PicoGPTSparse(config, attention_weights, base_embeddings)
    
    # 2. Sinusoidal
    print("2. Sinusoidal (fixed sinusoidal encoding)")
    models['sinusoidal'] = PicoGPTSparseSinusoidal(config, attention_weights, base_embeddings)
    
    # 3. Sinusoidal G_k
    print("3. Sinusoidal G_k (G_k-corrected)")
    models['sinusoidal_gk'] = PicoGPTSparseGkCorrected(config, gk_weights, gk_embeddings)
    
    # 4. RoPE
    print("4. RoPE (rotary encoding)")
    models['rope'] = PicoGPTSparseRoPE(config, attention_weights, base_embeddings)
    
    # 5. Trainable
    print("5. Trainable (SGD baseline)")
    models['trainable'] = PicoGPTTrainable(config)
    
    # 6. Zero
    print("6. Zero (ablation baseline)")
    zero_weights = {}
    for layer_idx in range(config.n_layers):
        zero_weights[layer_idx] = (
            torch.zeros(config.n_heads, config.embedding_dim, config.head_dim),
            torch.zeros(config.n_heads, config.embedding_dim, config.head_dim),
            torch.zeros(config.n_heads, config.head_dim, config.embedding_dim),
            torch.zeros(config.embedding_dim, config.n_heads * config.head_dim),
        )
    models['zero'] = PicoGPTSparse(config, zero_weights, base_embeddings)
    
    # Print parameter counts
    for name, model in models.items():
        total = model.get_num_params()
        trainable = model.get_num_params(trainable_only=True)
        print(f"  {name:15s}: Total={total:,}, Trainable={trainable:,}")
    
    # Train all models
    print("\n" + "=" * 60)
    print(f"Training all 6 models for {training_config.max_steps} steps")
    print("=" * 60)
    
    trainers = {name: PicoTrainer(model, config, training_config, name, device)
                for name, model in models.items()}
    
    batch_idx = 0
    start_time = time.time()
    
    for step in range(training_config.max_steps):
        batch = train_batches[batch_idx % len(train_batches)]
        batch_idx += 1
        
        losses = {name: trainer.train_step(batch) for name, trainer in trainers.items()}
        
        if (step + 1) % training_config.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"\nStep {step + 1}/{training_config.max_steps} ({(step+1)/elapsed:.2f} steps/s):")
            print(f"  Sparse: {losses['sparse']:.4f} | Sin: {losses['sinusoidal']:.4f} | Sin_Gk: {losses['sinusoidal_gk']:.4f}")
            print(f"  RoPE: {losses['rope']:.4f} | Train: {losses['trainable']:.4f} | Zero: {losses['zero']:.4f}")
        
        if (step + 1) % training_config.eval_interval == 0:
            print(f"\nEvaluation at step {step + 1}:")
            for name, trainer in trainers.items():
                eval_metrics = trainer.evaluate(test_batches)
                trainer.log_metrics(losses[name], eval_metrics)
                print(f"  {name:15s}: loss={eval_metrics['loss']:.4f}, ppl={eval_metrics['perplexity']:.2f}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    final_results = {name: trainer.evaluate(test_batches) 
                    for name, trainer in trainers.items()}
    
    print(f"\n{'Model':<18} {'Loss':<10} {'Perplexity':<12}")
    print("-" * 45)
    for name in ['sparse', 'sinusoidal', 'sinusoidal_gk', 'rope', 'trainable', 'zero']:
        metrics = final_results[name]
        print(f"{name:<18} {metrics['loss']:<10.4f} {metrics['perplexity']:<12.2f}")
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results")
    print("=" * 60)
    
    results = {
        "config": config.to_dict(),
        "training_config": training_config.to_dict(),
        "models": {}
    }
    
    for name, trainer in trainers.items():
        results["models"][name] = {
            "total_params": models[name].get_num_params(),
            "trainable_params": models[name].get_num_params(trainable_only=True),
            "final_loss": final_results[name]["loss"],
            "final_perplexity": final_results[name]["perplexity"],
            "history": trainer.metrics_history,
        }
    
    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    for name, trainer in trainers.items():
        torch.save(trainer.model.state_dict(), output_path / f"{name}_model.pt")
    
    print(f"\nResults saved to {output_path}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("COMPARISON ANALYSIS")
    print("=" * 60)
    
    print("\nPosition Encoding Variants (frozen attention):")
    for name in ['sparse', 'sinusoidal', 'sinusoidal_gk', 'rope']:
        ppl = final_results[name]['perplexity']
        print(f"  {name:15s}: {ppl:.2f} ppl")
    
    best_frozen = min(['sparse', 'sinusoidal', 'sinusoidal_gk', 'rope'],
                     key=lambda x: final_results[x]['perplexity'])
    print(f"\n  → Best: {best_frozen.upper()} ({final_results[best_frozen]['perplexity']:.2f} ppl)")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete PicoGPT comparison (6 models)")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = PicoModelConfig()
    training_config = PicoTrainingConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        log_interval=20,
        eval_interval=100,
    )
    
    run_complete_comparison(
        config=config,
        training_config=training_config,
        output_dir=args.output,
        device=args.device,
        skip_corpus=args.fast,
        seed=args.seed,
    )

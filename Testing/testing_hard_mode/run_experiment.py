"""
Hard Mode Experiment Runner

Runs rigorous comparison with:
- Multiple random seeds
- Strict train/val/test splits
- Same hyperparameters for all models
- Full training curve logging
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List
import numpy as np
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ExperimentConfig
from data_loader import DataSplits
from trainer import Trainer, TrainingCurve

from pico_config import PicoModelConfig, PicoTrainingConfig
from pico_sparse import PicoGPTSparse
from pico_trainable import PicoGPTTrainable
from pico_method import compute_pico_attention_weights


def create_models(
    config: ExperimentConfig,
    attention_weights: Dict,
    base_embeddings: torch.Tensor,
    seed: int,
) -> Dict[str, torch.nn.Module]:
    """Create all three models with the same seed for fair comparison."""
    
    # Model config (shared)
    model_config = PicoModelConfig(
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        embedding_dim=config.embedding_dim,
        head_dim=config.head_dim,
        mlp_dim=config.mlp_dim,
        vocab_size=config.vocab_size,
        max_seq_len=config.max_seq_len,
        mlp_dropout=config.dropout,
        embed_dropout=config.dropout,
        attn_dropout=config.dropout,
        layer_norm_eps=config.layer_norm_eps,
    )
    
    # Set seed before creating each model for identical MLP/embedding init
    torch.manual_seed(seed)
    np.random.seed(seed)
    sparse_model = PicoGPTSparse(model_config, attention_weights, base_embeddings=base_embeddings)
    
    # Zero model with same MLP/embedding init
    torch.manual_seed(seed)
    np.random.seed(seed)
    zero_weights = {}
    for layer_idx in range(config.n_layers):
        W_Q = torch.zeros(config.n_heads, config.embedding_dim, config.head_dim)
        W_K = torch.zeros(config.n_heads, config.embedding_dim, config.head_dim)
        W_V = torch.zeros(config.n_heads, config.head_dim, config.embedding_dim)
        W_O = torch.zeros(config.embedding_dim, config.n_heads * config.head_dim)
        zero_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
    zero_model = PicoGPTSparse(model_config, zero_weights, base_embeddings=base_embeddings)
    
    # Trainable model with same init
    torch.manual_seed(seed)
    np.random.seed(seed)
    trainable_model = PicoGPTTrainable(model_config)
    
    return {
        "sparse": sparse_model,
        "zero": zero_model,
        "trainable": trainable_model,
    }


def run_single_seed(
    config: ExperimentConfig,
    data: DataSplits,
    attention_weights: Dict,
    base_embeddings: torch.Tensor,
    seed: int,
    device: torch.device,
    verbose: bool = True,
) -> Dict:
    """Run experiment for a single seed."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}")
        print(f"{'='*60}")
    
    # Create models
    models = create_models(config, attention_weights, base_embeddings, seed)
    
    # Create trainers
    trainers = {}
    for name, model in models.items():
        trainers[name] = Trainer(model, config, name, device)
    
    # Train all models
    curves = {}
    for name, trainer in trainers.items():
        if verbose:
            print(f"\nTraining {name} model...")
        curves[name] = trainer.train(
            data.train_batches,
            data.val_batches,
            verbose=verbose,
        )
    
    # Final evaluation on HELD-OUT test set
    if verbose:
        print(f"\n--- Final Test Set Evaluation (seed={seed}) ---")
    
    results = {"seed": seed, "curves": {}, "final": {}}
    
    for name, trainer in trainers.items():
        final = trainer.get_final_metrics(
            data.train_batches,
            data.val_batches,
            data.test_batches,
        )
        results["curves"][name] = curves[name].to_dict()
        results["final"][name] = final
        
        if verbose:
            print(f"  {name}: test_loss={final['test']['loss']:.4f}, test_ppl={final['test']['perplexity']:.2f}")
    
    return results


def run_multi_seed_experiment(
    config: ExperimentConfig = None,
    output_dir: str = None,
    device: str = "cpu",
    verbose: bool = True,
):
    """
    Run full multi-seed experiment.
    
    This is the main entry point for rigorous evaluation.
    """
    if config is None:
        config = ExperimentConfig()
    
    # Setup device
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Device: {device}")
    print(f"Seeds: {config.seeds}")
    
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"hard_mode_results_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config.save(str(output_path / "config.json"))
    
    # Create data splits (use seed=0 for data, separate from model seeds)
    print("\nCreating data splits...")
    data = DataSplits(config, data_seed=0)
    
    # Compute corpus-derived attention weights ONCE (deterministic)
    print("\nComputing corpus-derived attention weights...")
    training_config = PicoTrainingConfig(max_n=config.max_n, max_s=config.max_s)
    attention_weights, base_embeddings = compute_pico_attention_weights(
        texts=data.get_train_corpus(),
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        embedding_dim=config.embedding_dim,
        head_dim=config.head_dim,
        max_n=config.max_n,
        max_s=config.max_s,
        verbose=verbose,
    )
    print(f"Base embeddings shape: {base_embeddings.shape} (will be FROZEN in models)")
    
    # Run all seeds
    all_results = []
    for seed in config.seeds:
        result = run_single_seed(
            config, data, attention_weights, base_embeddings, seed, device, verbose
        )
        all_results.append(result)
        
        # Save intermediate results
        with open(output_path / f"seed_{seed}_results.json", 'w') as f:
            json.dump(result, f, indent=2)
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("MULTI-SEED RESULTS SUMMARY")
    print("=" * 60)
    
    # Collect test perplexities
    test_ppls = {"sparse": [], "zero": [], "trainable": []}
    for result in all_results:
        for model_name in test_ppls.keys():
            ppl = result["final"][model_name]["test"]["perplexity"]
            test_ppls[model_name].append(ppl)
    
    # Compute statistics
    summary = {}
    print(f"\n{'Model':<15} {'Mean PPL':>12} {'Std PPL':>12} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    
    for model_name, ppls in test_ppls.items():
        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls)
        min_ppl = np.min(ppls)
        max_ppl = np.max(ppls)
        
        summary[model_name] = {
            "mean": mean_ppl,
            "std": std_ppl,
            "min": min_ppl,
            "max": max_ppl,
            "all": ppls,
        }
        
        print(f"{model_name:<15} {mean_ppl:>12.2f} {std_ppl:>12.2f} {min_ppl:>10.2f} {max_ppl:>10.2f}")
    
    # Statistical comparison
    sparse_mean = summary["sparse"]["mean"]
    sparse_std = summary["sparse"]["std"]
    zero_mean = summary["zero"]["mean"]
    zero_std = summary["zero"]["std"]
    trainable_mean = summary["trainable"]["mean"]
    trainable_std = summary["trainable"]["std"]
    
    print("\n--- Statistical Comparison ---")
    
    # Sparse vs Zero
    sparse_upper = sparse_mean + sparse_std
    zero_lower = zero_mean - zero_std
    if sparse_upper < zero_lower:
        print(f"✓ Sparse ({sparse_mean:.2f}±{sparse_std:.2f}) is SIGNIFICANTLY better than Zero ({zero_mean:.2f}±{zero_std:.2f})")
        print(f"  (Sparse upper bound {sparse_upper:.2f} < Zero lower bound {zero_lower:.2f})")
    else:
        print(f"? Sparse ({sparse_mean:.2f}±{sparse_std:.2f}) vs Zero ({zero_mean:.2f}±{zero_std:.2f}) - variance bands overlap")
    
    # Sparse vs Trainable
    sparse_upper = sparse_mean + sparse_std
    trainable_lower = trainable_mean - trainable_std
    if sparse_upper < trainable_lower:
        print(f"✓ Sparse ({sparse_mean:.2f}±{sparse_std:.2f}) is SIGNIFICANTLY better than Trainable ({trainable_mean:.2f}±{trainable_std:.2f})")
    elif sparse_mean < trainable_mean:
        print(f"~ Sparse ({sparse_mean:.2f}±{sparse_std:.2f}) is better than Trainable ({trainable_mean:.2f}±{trainable_std:.2f}) but variance overlaps")
    else:
        print(f"✗ Sparse ({sparse_mean:.2f}±{sparse_std:.2f}) is NOT better than Trainable ({trainable_mean:.2f}±{trainable_std:.2f})")
    
    # Improvement percentages
    print(f"\n--- Improvements ---")
    sparse_vs_zero = (zero_mean - sparse_mean) / zero_mean * 100
    sparse_vs_trainable = (trainable_mean - sparse_mean) / trainable_mean * 100
    print(f"Sparse vs Zero: {sparse_vs_zero:+.1f}%")
    print(f"Sparse vs Trainable: {sparse_vs_trainable:+.1f}%")
    
    # Save summary
    full_summary = {
        "config": config.to_dict(),
        "seeds": config.seeds,
        "summary": summary,
        "all_results": all_results,
    }
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(full_summary, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return full_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hard Mode Experiment")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    config = ExperimentConfig(max_steps=args.steps)
    
    run_multi_seed_experiment(
        config=config,
        output_dir=args.output,
        device=args.device,
        verbose=not args.quiet,
    )


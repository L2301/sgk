"""
Results Analysis and Plotting

Generates:
1. Training curve plots with variance bands
2. Early stopping analysis
3. Statistical significance tests
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Try to import matplotlib, but don't fail if not available
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plots will be skipped")


def load_results(results_dir: str) -> Dict:
    """Load experiment results."""
    results_path = Path(results_dir)
    
    with open(results_path / "summary.json", 'r') as f:
        return json.load(f)


def extract_curves(results: Dict) -> Dict[str, Dict[str, List]]:
    """
    Extract training curves for all models across all seeds.
    
    Returns:
        {model_name: {metric_name: [[seed1_values], [seed2_values], ...]}}
    """
    all_curves = {"sparse": {}, "zero": {}, "trainable": {}}
    
    for seed_result in results["all_results"]:
        for model_name in all_curves.keys():
            curve = seed_result["curves"][model_name]
            
            for metric in ["steps", "train_losses", "val_losses", "val_perplexities"]:
                if metric not in all_curves[model_name]:
                    all_curves[model_name][metric] = []
                all_curves[model_name][metric].append(curve[metric])
    
    return all_curves


def compute_curve_stats(curves: Dict) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute mean and std for each curve."""
    stats = {}
    
    for model_name, model_curves in curves.items():
        stats[model_name] = {}
        
        # Assume all seeds have same steps
        stats[model_name]["steps"] = np.array(model_curves["steps"][0])
        
        for metric in ["train_losses", "val_losses", "val_perplexities"]:
            values = np.array(model_curves[metric])
            stats[model_name][f"{metric}_mean"] = np.mean(values, axis=0)
            stats[model_name][f"{metric}_std"] = np.std(values, axis=0)
    
    return stats


def plot_training_curves(
    stats: Dict,
    output_path: str = "training_curves.png",
    metric: str = "val_perplexities",
    title: str = "Validation Perplexity over Training",
):
    """
    Plot training curves with variance bands.
    
    Shows mean ± std for each model.
    """
    if not HAS_MATPLOTLIB:
        print(f"Skipping plot (matplotlib not available): {output_path}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"sparse": "#2ecc71", "zero": "#e74c3c", "trainable": "#3498db"}
    labels = {"sparse": "Sparse (corpus-derived)", "zero": "Zero (ablation)", "trainable": "Trainable (baseline)"}
    
    for model_name, model_stats in stats.items():
        steps = model_stats["steps"]
        mean = model_stats[f"{metric}_mean"]
        std = model_stats[f"{metric}_std"]
        
        color = colors.get(model_name, "#333333")
        label = labels.get(model_name, model_name)
        
        # Plot mean line
        ax.plot(steps, mean, label=label, color=color, linewidth=2)
        
        # Plot variance band
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Validation Perplexity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Better y-axis
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_loss_curves(
    stats: Dict,
    output_path: str = "loss_curves.png",
):
    """Plot both train and val loss curves."""
    if not HAS_MATPLOTLIB:
        print(f"Skipping plot (matplotlib not available): {output_path}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"sparse": "#2ecc71", "zero": "#e74c3c", "trainable": "#3498db"}
    labels = {"sparse": "Sparse", "zero": "Zero", "trainable": "Trainable"}
    
    for idx, (metric, title) in enumerate([
        ("train_losses", "Training Loss"),
        ("val_losses", "Validation Loss"),
    ]):
        ax = axes[idx]
        
        for model_name, model_stats in stats.items():
            steps = model_stats["steps"]
            mean = model_stats[f"{metric}_mean"]
            std = model_stats[f"{metric}_std"]
            
            color = colors.get(model_name, "#333333")
            label = labels.get(model_name, model_name)
            
            ax.plot(steps, mean, label=label, color=color, linewidth=2)
            ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
        
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved: {output_path}")


def early_stopping_analysis(stats: Dict) -> Dict:
    """
    Analyze: If you stop early, do you still win?
    
    Compares perplexity at different checkpoints.
    """
    steps = stats["sparse"]["steps"]
    
    results = []
    
    for i, step in enumerate(steps):
        sparse_ppl = stats["sparse"]["val_perplexities_mean"][i]
        zero_ppl = stats["zero"]["val_perplexities_mean"][i]
        trainable_ppl = stats["trainable"]["val_perplexities_mean"][i]
        
        sparse_wins_zero = sparse_ppl < zero_ppl
        sparse_wins_trainable = sparse_ppl < trainable_ppl
        
        results.append({
            "step": step,
            "sparse_ppl": sparse_ppl,
            "zero_ppl": zero_ppl,
            "trainable_ppl": trainable_ppl,
            "sparse_beats_zero": sparse_wins_zero,
            "sparse_beats_trainable": sparse_wins_trainable,
        })
    
    return results


def print_early_stopping_analysis(analysis: List[Dict]):
    """Print early stopping analysis."""
    print("\n--- Early Stopping Analysis ---")
    print("Does Sparse win at each checkpoint?")
    print(f"\n{'Step':<8} {'Sparse':<10} {'Zero':<10} {'Trainable':<10} {'vs Zero':<10} {'vs Train':<10}")
    print("-" * 60)
    
    for row in analysis:
        vs_zero = "✓" if row["sparse_beats_zero"] else "✗"
        vs_train = "✓" if row["sparse_beats_trainable"] else "✗"
        print(f"{row['step']:<8} {row['sparse_ppl']:<10.2f} {row['zero_ppl']:<10.2f} {row['trainable_ppl']:<10.2f} {vs_zero:<10} {vs_train:<10}")
    
    # Summary
    total = len(analysis)
    wins_zero = sum(1 for r in analysis if r["sparse_beats_zero"])
    wins_trainable = sum(1 for r in analysis if r["sparse_beats_trainable"])
    
    print(f"\nSparse beats Zero: {wins_zero}/{total} checkpoints ({wins_zero/total*100:.1f}%)")
    print(f"Sparse beats Trainable: {wins_trainable}/{total} checkpoints ({wins_trainable/total*100:.1f}%)")


def analyze_results(results_dir: str, output_dir: str = None):
    """
    Full analysis of experiment results.
    """
    if output_dir is None:
        output_dir = results_dir
    
    output_path = Path(output_dir)
    
    print(f"Analyzing results from: {results_dir}")
    
    # Load results
    results = load_results(results_dir)
    
    # Extract and compute curve statistics
    curves = extract_curves(results)
    stats = compute_curve_stats(curves)
    
    # Generate plots
    plot_training_curves(
        stats,
        str(output_path / "perplexity_curves.png"),
        metric="val_perplexities",
        title="Validation Perplexity (mean ± std across seeds)",
    )
    
    plot_loss_curves(
        stats,
        str(output_path / "loss_curves.png"),
    )
    
    # Early stopping analysis
    early_analysis = early_stopping_analysis(stats)
    print_early_stopping_analysis(early_analysis)
    
    # Save analysis (convert numpy types to Python native)
    def convert_to_native(obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(i) for i in obj]
        return obj
    
    analysis_output = convert_to_native({
        "early_stopping": early_analysis,
        "final_summary": results["summary"],
    })
    
    with open(output_path / "analysis.json", 'w') as f:
        json.dump(analysis_output, f, indent=2)
    
    print(f"\nAnalysis complete! Results in {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("results_dir", type=str, help="Path to results directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    analyze_results(args.results_dir, args.output)


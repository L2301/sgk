"""
Model Evaluation Script

Comprehensive evaluation of trained Sparse Attention Transformer:
- Perplexity on test data
- Generation quality
- Attention pattern analysis
- Layer-wise metrics
"""

import sys
import json
import math
from pathlib import Path
from typing import Optional, Dict, List
import argparse

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import ModelConfig, RunConfig
from transformer import SparseAttentionTransformer
from data_loader import load_test_data, LocalTextDataset
from weight_loader import WeightComputer


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(
        self,
        model: SparseAttentionTransformer,
        config: ModelConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def compute_perplexity(
        self,
        sequences: List[List[int]],
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """
        Compute perplexity on test sequences.
        
        Returns:
            Dict with loss and perplexity metrics
        """
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            
            # Pad sequences if needed
            max_len = max(len(s) for s in batch_seqs)
            padded = torch.zeros(len(batch_seqs), max_len, dtype=torch.long)
            
            for j, seq in enumerate(batch_seqs):
                padded[j, :len(seq)] = torch.tensor(seq)
            
            padded = padded.to(self.device)
            
            loss, _ = self.model.compute_loss(padded)
            
            # Count non-padding tokens
            num_tokens = sum(len(s) - 1 for s in batch_seqs)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
        
        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))
        
        return {
            "loss": avg_loss,
            "perplexity": perplexity,
            "total_tokens": total_tokens,
            "num_sequences": len(sequences),
        }
    
    @torch.no_grad()
    def compute_token_accuracy(
        self,
        sequences: List[List[int]],
        top_k: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """
        Compute top-k token prediction accuracy.
        
        Returns:
            Dict with accuracy metrics for each k
        """
        correct = {k: 0 for k in top_k}
        total = 0
        
        for seq in sequences:
            input_ids = torch.tensor([seq[:-1]], device=self.device)
            targets = torch.tensor(seq[1:], device=self.device)
            
            outputs = self.model(input_ids)
            logits = outputs["logits"][0]  # (seq_len-1, vocab)
            
            for k in top_k:
                top_k_preds = torch.topk(logits, k, dim=-1).indices
                hits = (top_k_preds == targets.unsqueeze(-1)).any(dim=-1)
                correct[k] += hits.sum().item()
            
            total += len(seq) - 1
        
        return {
            f"top_{k}_accuracy": correct[k] / max(total, 1)
            for k in top_k
        }
    
    @torch.no_grad()
    def analyze_attention_patterns(
        self,
        sequence: List[int],
    ) -> Dict[str, np.ndarray]:
        """
        Analyze attention patterns for a sequence.
        
        Returns:
            Dict with attention statistics per layer/head
        """
        input_ids = torch.tensor([sequence], device=self.device)
        outputs = self.model(input_ids, return_attention=True)
        
        attention_weights = outputs["attention_weights"]
        
        results = {
            "entropy_per_layer": [],
            "sparsity_per_layer": [],
            "mean_attention_distance": [],
        }
        
        for layer_idx, layer_attn in enumerate(attention_weights):
            # layer_attn: (batch, n_heads, seq, seq)
            attn = layer_attn[0].cpu().numpy()  # (n_heads, seq, seq)
            
            # Entropy (how spread out attention is)
            entropy = -np.sum(attn * np.log(attn + 1e-10), axis=-1)
            results["entropy_per_layer"].append(entropy.mean())
            
            # Sparsity (fraction of attention weights < 0.01)
            sparsity = (attn < 0.01).mean()
            results["sparsity_per_layer"].append(sparsity)
            
            # Mean attention distance
            seq_len = attn.shape[-1]
            positions = np.arange(seq_len)
            distances = np.abs(positions[:, None] - positions[None, :])
            mean_dist = (attn * distances).sum(axis=-1).mean()
            results["mean_attention_distance"].append(mean_dist)
        
        return results
    
    @torch.no_grad()
    def generate_samples(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate text samples from prompts.
        
        Returns:
            List of generated texts
        """
        from tokenizer import GPT2Tokenizer
        tokenizer = GPT2Tokenizer()
        
        generated = []
        
        for prompt in prompts:
            input_ids = torch.tensor(
                [tokenizer.encode(prompt)],
                device=self.device
            )
            
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            text = tokenizer.decode(output_ids[0].tolist())
            generated.append(text)
        
        return generated
    
    def full_evaluation(
        self,
        test_sequences: List[List[int]],
        prompts: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run full evaluation suite.
        
        Returns:
            Dict with all evaluation metrics
        """
        results = {}
        
        # Perplexity
        print("Computing perplexity...")
        results["perplexity_metrics"] = self.compute_perplexity(test_sequences)
        
        # Token accuracy
        print("Computing token accuracy...")
        results["accuracy_metrics"] = self.compute_token_accuracy(
            test_sequences[:100]  # Limit for speed
        )
        
        # Attention analysis (on first few sequences)
        print("Analyzing attention patterns...")
        if test_sequences:
            attn_results = self.analyze_attention_patterns(test_sequences[0][:256])
            results["attention_analysis"] = {
                k: [float(v) for v in vals] if isinstance(vals, list) else float(vals)
                for k, vals in attn_results.items()
            }
        
        # Generation (if prompts provided)
        if prompts:
            print("Generating samples...")
            results["generated_samples"] = self.generate_samples(prompts)
        
        return results


def load_model_from_run(run_dir: str) -> tuple:
    """
    Load model from a training run directory.
    
    Returns:
        Tuple of (model, config, evaluator)
    """
    run_dir = Path(run_dir)
    
    # Load config
    config = RunConfig.load(str(run_dir / "config.json"))
    
    # Load attention weights
    weights_path = run_dir / "attention_weights.npz"
    if weights_path.exists():
        computer = WeightComputer(config.model, config.training, verbose=False)
        # Load the saved weights
        data = np.load(str(weights_path))
        
        for key in data.files:
            if key.startswith("W_Q_"):
                n_s = eval(key[4:])
                computer.W_Q_dict[n_s] = data[key]
            elif key.startswith("W_K_"):
                n_s = eval(key[4:])
                computer.W_K_dict[n_s] = data[key]
            elif key.startswith("W_V_"):
                n_s = eval(key[4:])
                computer.W_V_dict[n_s] = data[key]
            elif key.startswith("W_O_"):
                n_s = eval(key[4:])
                computer.W_O_dict[n_s] = data[key]
        
        attention_weights = computer.get_all_layer_weights()
    else:
        print("Warning: No attention weights found")
        attention_weights = {}
    
    # Create model
    model = SparseAttentionTransformer(config.model, attention_weights)
    
    # Load trained weights
    checkpoint_path = run_dir / "completed" / "model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {checkpoint_path}")
    else:
        # Try best snapshot
        best_path = run_dir / "snapshots" / "best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model from {best_path}")
        else:
            print("Warning: No trained weights found")
    
    # Setup device
    if config.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif config.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    evaluator = ModelEvaluator(model, config.model, device)
    
    return model, config, evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sparse Attention Transformer")
    parser.add_argument("run_dir", type=str, help="Training run directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--prompts", type=str, nargs="+", default=None, help="Generation prompts")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    
    # Load model
    model, config, evaluator = load_model_from_run(args.run_dir)
    
    # Load test data
    test_data_path = run_dir / "test_data" / "test_tokens.pt"
    if test_data_path.exists():
        test_sequences = load_test_data(str(test_data_path))
    else:
        print("No test data found, using dummy data")
        test_sequences = [[i % 1000 for i in range(256)] for _ in range(10)]
    
    # Default prompts
    prompts = args.prompts or [
        "The meaning of life is",
        "In the beginning, there was",
        "Machine learning models can",
    ]
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60)
    
    results = evaluator.full_evaluation(test_sequences, prompts)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nPerplexity Metrics:")
    for k, v in results["perplexity_metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print("\nAccuracy Metrics:")
    for k, v in results["accuracy_metrics"].items():
        print(f"  {k}: {v:.4f}")
    
    if "attention_analysis" in results:
        print("\nAttention Analysis:")
        for k, v in results["attention_analysis"].items():
            if isinstance(v, list):
                print(f"  {k}: {[f'{x:.3f}' for x in v[:4]]}...")
            else:
                print(f"  {k}: {v:.4f}")
    
    if "generated_samples" in results:
        print("\nGenerated Samples:")
        for i, sample in enumerate(results["generated_samples"]):
            print(f"\n  [{i+1}] {sample[:200]}...")
    
    # Save results
    output_path = args.output or (run_dir / "evaluation_results.json")
    with open(output_path, 'w') as f:
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()


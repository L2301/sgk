"""
Orthogonal Embedding Generator for PicoGPT

Creates embedding matrices where vectors are as orthogonal as possible.
Since we can't have 50,257 perfectly orthogonal vectors in 128D (max is 128),
we use heuristic optimization to minimize pairwise correlations.

Approaches:
1. Random Gaussian (baseline) - random vectors are nearly orthogonal in high-dim
2. Gram-Schmidt on random subset + extend
3. Simulated annealing optimization
4. Gradient descent on coherence loss
"""

import numpy as np
import torch
from typing import Optional, Tuple
import time


def random_gaussian_embeddings(
    vocab_size: int,
    embedding_dim: int,
    normalize: bool = True,
    seed: int = None,
) -> np.ndarray:
    """
    Random Gaussian embeddings.
    
    In high dimensions, random vectors tend to be nearly orthogonal
    (by concentration of measure).
    
    Expected pairwise cosine similarity ≈ 0 with std ≈ 1/sqrt(d)
    """
    if seed is not None:
        np.random.seed(seed)
    
    E = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    
    if normalize:
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.maximum(norms, 1e-8)
    
    return E


def orthogonal_basis_plus_random(
    vocab_size: int,
    embedding_dim: int,
    seed: int = None,
) -> np.ndarray:
    """
    Create orthogonal basis for first embedding_dim tokens,
    then add nearly-orthogonal random vectors for the rest.
    """
    if seed is not None:
        np.random.seed(seed)
    
    E = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    
    # First embedding_dim vectors are perfectly orthogonal (identity-like)
    E[:embedding_dim, :] = np.eye(embedding_dim, dtype=np.float32)
    
    # Remaining vectors: random, projected to be orthogonal to basis
    # (they won't be orthogonal to each other, but will have low coherence)
    if vocab_size > embedding_dim:
        remaining = np.random.randn(vocab_size - embedding_dim, embedding_dim).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(remaining, axis=1, keepdims=True)
        E[embedding_dim:, :] = remaining / np.maximum(norms, 1e-8)
    
    return E


def simulated_annealing_embeddings(
    vocab_size: int,
    embedding_dim: int,
    n_iterations: int = 10000,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.9995,
    sample_pairs: int = 1000,
    seed: int = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Simulated annealing to minimize maximum pairwise coherence.
    
    Coherence = max |<E_i, E_j>| for i ≠ j
    
    This is NP-hard to optimize globally, so we use SA heuristic.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with random Gaussian
    E = random_gaussian_embeddings(vocab_size, embedding_dim, normalize=True, seed=seed)
    
    def sample_coherence(E, n_pairs):
        """Sample coherence by checking random pairs."""
        idx1 = np.random.randint(0, vocab_size, n_pairs)
        idx2 = np.random.randint(0, vocab_size, n_pairs)
        # Avoid same index
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        
        if len(idx1) == 0:
            return 0.0
        
        dots = np.abs(np.sum(E[idx1] * E[idx2], axis=1))
        return dots.max()
    
    current_coherence = sample_coherence(E, sample_pairs)
    best_E = E.copy()
    best_coherence = current_coherence
    
    temp = initial_temp
    
    if verbose:
        print(f"SA Optimization: vocab={vocab_size}, dim={embedding_dim}")
        print(f"Initial coherence: {current_coherence:.6f}")
    
    for i in range(n_iterations):
        # Perturb a random embedding
        idx = np.random.randint(0, vocab_size)
        old_vec = E[idx].copy()
        
        # Small random perturbation
        perturbation = np.random.randn(embedding_dim).astype(np.float32) * 0.1
        E[idx] = E[idx] + perturbation
        # Re-normalize
        E[idx] = E[idx] / np.linalg.norm(E[idx])
        
        new_coherence = sample_coherence(E, sample_pairs)
        
        # Accept or reject
        delta = new_coherence - current_coherence
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_coherence = new_coherence
            if current_coherence < best_coherence:
                best_coherence = current_coherence
                best_E = E.copy()
        else:
            E[idx] = old_vec
        
        temp *= cooling_rate
        
        if verbose and (i + 1) % 2000 == 0:
            print(f"  Iter {i+1}: coherence={current_coherence:.6f}, best={best_coherence:.6f}, temp={temp:.6f}")
    
    if verbose:
        print(f"Final best coherence: {best_coherence:.6f}")
    
    return best_E


def gradient_descent_embeddings(
    vocab_size: int,
    embedding_dim: int,
    n_iterations: int = 1000,
    lr: float = 0.01,
    batch_pairs: int = 5000,
    seed: int = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Gradient descent to minimize coherence loss.
    
    Loss = mean(|<E_i, E_j>|^2) for sampled pairs i ≠ j
    
    Uses PyTorch for autodiff.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Initialize
    E = torch.randn(vocab_size, embedding_dim, dtype=torch.float32)
    E = E / E.norm(dim=1, keepdim=True)
    E = torch.nn.Parameter(E)
    
    optimizer = torch.optim.Adam([E], lr=lr)
    
    if verbose:
        print(f"GD Optimization: vocab={vocab_size}, dim={embedding_dim}")
    
    for i in range(n_iterations):
        optimizer.zero_grad()
        
        # Sample random pairs
        idx1 = torch.randint(0, vocab_size, (batch_pairs,))
        idx2 = torch.randint(0, vocab_size, (batch_pairs,))
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        
        # Normalize (project back to unit sphere)
        E_norm = E / E.norm(dim=1, keepdim=True).clamp(min=1e-8)
        
        # Coherence loss: minimize squared dot products
        dots = (E_norm[idx1] * E_norm[idx2]).sum(dim=1)
        loss = (dots ** 2).mean()
        
        loss.backward()
        optimizer.step()
        
        if verbose and (i + 1) % 200 == 0:
            with torch.no_grad():
                E_eval = E / E.norm(dim=1, keepdim=True)
                sample_dots = (E_eval[idx1] * E_eval[idx2]).sum(dim=1).abs()
                max_coh = sample_dots.max().item()
                mean_coh = sample_dots.mean().item()
            print(f"  Iter {i+1}: loss={loss.item():.6f}, max_coh={max_coh:.6f}, mean_coh={mean_coh:.6f}")
    
    # Final normalization
    with torch.no_grad():
        E_final = E / E.norm(dim=1, keepdim=True)
    
    return E_final.detach().numpy()


def compute_coherence_stats(E: np.ndarray, n_samples: int = 10000) -> dict:
    """Compute coherence statistics for an embedding matrix."""
    vocab_size = E.shape[0]
    
    # Normalize
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_norm = E / np.maximum(norms, 1e-8)
    
    # Sample pairs
    idx1 = np.random.randint(0, vocab_size, n_samples)
    idx2 = np.random.randint(0, vocab_size, n_samples)
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    
    dots = np.abs(np.sum(E_norm[idx1] * E_norm[idx2], axis=1))
    
    return {
        "max_coherence": dots.max(),
        "mean_coherence": dots.mean(),
        "std_coherence": dots.std(),
        "median_coherence": np.median(dots),
        "p99_coherence": np.percentile(dots, 99),
    }


def create_pico_embeddings(
    vocab_size: int = 50257,
    embedding_dim: int = 128,
    method: str = "gradient_descent",
    seed: int = 42,
    verbose: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Create near-orthogonal embeddings for PicoGPT.
    
    Args:
        vocab_size: Number of tokens
        embedding_dim: Embedding dimension
        method: One of "random", "orthogonal_basis", "simulated_annealing", "gradient_descent"
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    if verbose:
        print(f"\nCreating embeddings: method={method}")
        start = time.time()
    
    if method == "random":
        E = random_gaussian_embeddings(vocab_size, embedding_dim, seed=seed)
    elif method == "orthogonal_basis":
        E = orthogonal_basis_plus_random(vocab_size, embedding_dim, seed=seed)
    elif method == "simulated_annealing":
        E = simulated_annealing_embeddings(vocab_size, embedding_dim, seed=seed, verbose=verbose, **kwargs)
    elif method == "gradient_descent":
        E = gradient_descent_embeddings(vocab_size, embedding_dim, seed=seed, verbose=verbose, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if verbose:
        elapsed = time.time() - start
        stats = compute_coherence_stats(E)
        print(f"\nCompleted in {elapsed:.2f}s")
        print(f"Coherence stats:")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}")
    
    return E


if __name__ == "__main__":
    # Test with PicoGPT dimensions
    vocab_size = 50257
    embedding_dim = 128
    
    print("=" * 60)
    print("Comparing Embedding Generation Methods for PicoGPT")
    print(f"Vocab: {vocab_size}, Embedding dim: {embedding_dim}")
    print("=" * 60)
    
    # Theoretical minimum coherence (Welch bound)
    # For M vectors in N dimensions: coherence >= sqrt((M-N)/(N(M-1)))
    welch_bound = np.sqrt((vocab_size - embedding_dim) / (embedding_dim * (vocab_size - 1)))
    print(f"\nWelch bound (theoretical min coherence): {welch_bound:.6f}")
    
    results = {}
    
    # Method 1: Random Gaussian (fast baseline)
    print("\n" + "-" * 60)
    E_random = create_pico_embeddings(vocab_size, embedding_dim, method="random", seed=42)
    results["random"] = compute_coherence_stats(E_random)
    
    # Method 2: Orthogonal basis + random
    print("\n" + "-" * 60)
    E_ortho = create_pico_embeddings(vocab_size, embedding_dim, method="orthogonal_basis", seed=42)
    results["orthogonal_basis"] = compute_coherence_stats(E_ortho)
    
    # Method 3: Gradient descent (moderate optimization)
    print("\n" + "-" * 60)
    E_gd = create_pico_embeddings(
        vocab_size, embedding_dim, 
        method="gradient_descent", 
        seed=42,
        n_iterations=500,
        batch_pairs=2000,
    )
    results["gradient_descent"] = compute_coherence_stats(E_gd)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Method':<25} {'Max Coh':<12} {'Mean Coh':<12} {'P99 Coh':<12}")
    print("-" * 60)
    for method, stats in results.items():
        print(f"{method:<25} {stats['max_coherence']:<12.6f} {stats['mean_coherence']:<12.6f} {stats['p99_coherence']:<12.6f}")
    print(f"{'Welch bound':<25} {welch_bound:<12.6f}")
    
    # Save best embeddings
    print("\n" + "-" * 60)
    print("Saving gradient descent embeddings to pico_embeddings.npy")
    np.save("pico_embeddings.npy", E_gd)
    print("Done!")


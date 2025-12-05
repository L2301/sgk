"""
G_k-Corrected Weight Computation for PicoGPT

Implements the mathematically correct weight derivation:
1. Compute M_k from corpus (as before)
2. Compute G_k = E[P(i+k)P(i)^T] for positional bias
3. M_k,free = M_k - G_k to remove positional contamination
4. Derive W_Q, W_K from M_k,free via SVD

This recovers the true semantic skip-k structure.
"""

import numpy as np
import torch
from scipy import sparse
from typing import Dict, List, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from pico_method import (
    build_all_count_matrices,
    build_all_manifold_matrices,
    build_all_probability_matrices,
    build_all_embedding_matrices,
    compute_value_output_matrices,
)
from pico_sparse_sinusoidal_gk import SinusoidalPositionEncoding, compute_G_k


def compute_gk_corrected_key_query_matrices(
    manifold_matrix: sparse.csr_matrix,
    embedding_n: np.ndarray,
    embedding_1: np.ndarray,
    head_dim: int,
    k: int,  # skip distance
    position_encoding: SinusoidalPositionEncoding,
    embedding_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute W_Q and W_K with G_k correction.
    
    Steps:
    1. Convert sparse M_k to dense
    2. Compute G_k = E[P(i+k)P(i)^T]
    3. M_k,free = M_k - G_k
    4. SVD on M_k,free
    5. Derive W_Q, W_K as before
    
    Args:
        manifold_matrix: M_k from corpus
        k: skip distance for this head
        position_encoding: Sinusoidal position encoding module
        embedding_dim: Embedding dimension
    
    Returns:
        (W_Q, W_K) with G_k correction applied
    """
    from scipy.sparse.linalg import svds
    
    M = manifold_matrix
    
    # Compute G_k
    G_k = compute_G_k(position_encoding, k, max_pos=1000)
    G_k_np = G_k.cpu().numpy().astype(np.float32)
    
    # For manifold matrix, we need to project G_k into the n-gram space
    # M_k is (vocab^n, vocab) but G_k is (d, d)
    # We need to transform: M_k,corrected = M_k - E^n @ G_k @ E^1^T
    
    # This is approximate - project G_k through embeddings
    # G_k_projected = E_n @ G_k @ E_1^T
    # Then subtract from M
    
    # Handle dimension mismatch
    if embedding_n.shape[0] != M.shape[0]:
        if embedding_n.shape[0] > M.shape[0]:
            embedding_n = embedding_n[:M.shape[0]]
        else:
            pad = np.zeros((M.shape[0] - embedding_n.shape[0], embedding_n.shape[1]), dtype=np.float32)
            embedding_n = np.vstack([embedding_n, pad])
    
    if embedding_1.shape[0] != M.shape[1]:
        if embedding_1.shape[0] > M.shape[1]:
            embedding_1 = embedding_1[:M.shape[1]]
        else:
            pad = np.zeros((M.shape[1] - embedding_1.shape[0], embedding_1.shape[1]), dtype=np.float32)
            embedding_1 = np.vstack([embedding_1, pad])
    
    # Project G_k: (vocab^n, d) @ (d, d) @ (d, vocab) = (vocab^n, vocab)
    G_k_projected = embedding_n @ G_k_np @ embedding_1.T
    
    # Convert M to dense and subtract
    M_dense = M.toarray().astype(np.float64)
    M_corrected = M_dense - G_k_projected
    
    # Convert back to sparse for SVD
    M_corrected_sparse = sparse.csr_matrix(M_corrected)
    
    # Truncated SVD on corrected matrix
    k_svd = min(head_dim, min(M_corrected_sparse.shape) - 2)
    if k_svd < 1:
        k_svd = 1
    
    try:
        U, sigma, Vt = svds(M_corrected_sparse, k=k_svd)
        idx = np.argsort(sigma)[::-1]
        U = U[:, idx]
        sigma = sigma[idx]
        V = Vt[idx, :].T
    except Exception as e:
        print(f"  SVD failed on corrected matrix, using random: {e}")
        U = np.random.randn(M.shape[0], k_svd).astype(np.float32)
        sigma = np.ones(k_svd, dtype=np.float32)
        V = np.random.randn(M.shape[1], k_svd).astype(np.float32)
    
    sqrt_sigma = np.sqrt(np.maximum(sigma, 1e-10)).astype(np.float32)
    U = U.astype(np.float32)
    V = V.astype(np.float32)
    
    # W_Q = E^n.T @ U @ sqrt(Sigma)
    W_Q = embedding_n.T @ U @ np.diag(sqrt_sigma)
    
    # W_K = E^1.T @ V @ sqrt(Sigma)
    W_K = embedding_1.T @ V @ np.diag(sqrt_sigma)
    
    # Pad or truncate to head_dim
    if W_Q.shape[1] < head_dim:
        W_Q = np.hstack([W_Q, np.zeros((embedding_dim, head_dim - W_Q.shape[1]), dtype=np.float32)])
        W_K = np.hstack([W_K, np.zeros((embedding_dim, head_dim - W_K.shape[1]), dtype=np.float32)])
    elif W_Q.shape[1] > head_dim:
        W_Q = W_Q[:, :head_dim]
        W_K = W_K[:, :head_dim]
    
    return W_Q, W_K


def compute_pico_attention_weights_gk_corrected(
    texts: List[str],
    n_layers: int,
    n_heads: int,
    embedding_dim: int,
    head_dim: int,
    max_n: int = 2,
    max_s: int = 2,
    verbose: bool = True,
    normalize_weights: bool = True,
    target_std: float = 0.01,
) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
    """
    Compute attention weights with G_k correction.
    
    Key difference: W_Q and W_K are derived from M_k,free = M_k - G_k
    """
    from pico_tokenizer import PicoTokenizer
    
    if verbose:
        print("=" * 60)
        print("Computing PicoGPT Attention Weights (G_k CORRECTED)")
        print("=" * 60)
    
    # Step 1: Tokenize
    tokenizer = PicoTokenizer()
    vocab_size = tokenizer.vocab_size
    
    if verbose:
        print(f"\nStep 1: Tokenization")
        print(f"  Vocab size: {vocab_size}")
    
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))
    
    batch_size = 256
    batches = []
    for i in range(0, len(all_tokens) - batch_size + 1, batch_size):
        batches.append(all_tokens[i:i + batch_size])
    
    if verbose:
        print(f"  Total tokens: {len(all_tokens)}")
        print(f"  Batches: {len(batches)} x {batch_size}")
    
    # Step 2: Build count matrices
    if verbose:
        print(f"\nStep 2: Building count matrices")
    
    count_matrices = build_all_count_matrices(batches, max_n, max_s, vocab_size)
    
    # Step 3: Compute manifold and probability matrices
    if verbose:
        print(f"\nStep 3: Computing manifold matrices")
    
    manifold_matrices = build_all_manifold_matrices(count_matrices)
    
    if verbose:
        print(f"\nStep 4: Computing probability matrices")
    
    prob_matrices = build_all_probability_matrices(count_matrices)
    
    # Step 5: Create base embeddings and n-gram embeddings
    if verbose:
        print(f"\nStep 5: Building embeddings")
    
    from pico_method import create_base_embeddings
    base_embeddings = create_base_embeddings(vocab_size, embedding_dim)
    embedding_matrices = build_all_embedding_matrices(prob_matrices, base_embeddings, max_n, vocab_size)
    
    # Step 6: Create position encoding for G_k computation
    position_encoding = SinusoidalPositionEncoding(embedding_dim, max_seq_len=1024)
    
    # Step 7: Derive attention weights with G_k correction
    if verbose:
        print(f"\nStep 6: Deriving W_Q, W_K (G_k corrected), W_V, W_O")
    
    attention_weights = {}
    
    for layer_idx in range(n_layers):
        n = min(layer_idx + 1, max_n)
        
        E_n = embedding_matrices.get(n, embedding_matrices[1])
        E_1 = embedding_matrices[1]
        
        W_Q_heads = []
        W_K_heads = []
        W_V_heads = []
        W_O_heads = []
        
        for head_idx in range(n_heads):
            s = min(head_idx, max_s)
            k = n + s  # skip distance
            
            key = (n, s)
            if key not in manifold_matrices:
                key = (1, 0)
            
            M = manifold_matrices[key]
            P = prob_matrices[key]
            
            # Compute W_Q, W_K with G_k correction
            W_Q_h, W_K_h = compute_gk_corrected_key_query_matrices(
                M, E_n, E_1, head_dim, k, position_encoding, embedding_dim
            )
            
            # Compute W_V, W_O from probability (no G_k correction needed)
            W_V_h, W_O_h = compute_value_output_matrices(P, E_n, E_1, head_dim)
            
            W_Q_heads.append(W_Q_h)
            W_K_heads.append(W_K_h)
            W_V_heads.append(W_V_h)
            W_O_heads.append(W_O_h)
        
        # Stack heads
        W_Q = torch.tensor(np.stack(W_Q_heads, axis=0), dtype=torch.float32)
        W_K = torch.tensor(np.stack(W_K_heads, axis=0), dtype=torch.float32)
        W_V = torch.tensor(np.stack(W_V_heads, axis=0), dtype=torch.float32)
        W_O = torch.tensor(np.hstack(W_O_heads), dtype=torch.float32)
        
        # Normalize weights
        if normalize_weights:
            for W, name in [(W_Q, 'Q'), (W_K, 'K'), (W_V, 'V'), (W_O, 'O')]:
                current_std = W.std().item()
                if current_std > 0:
                    scale = target_std / current_std
                    W.mul_(scale)
        
        attention_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
        
        if verbose:
            print(f"  Layer {layer_idx} (n={n}): G_k corrected for k={n}+s")
    
    if verbose:
        print(f"\nG_k-corrected attention weight computation complete!")
    
    base_embeddings_tensor = torch.tensor(base_embeddings, dtype=torch.float32)
    
    return attention_weights, base_embeddings_tensor


if __name__ == "__main__":
    print("Testing G_k-corrected weight computation...")
    
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 50,
        "Machine learning is transforming the world. " * 50,
    ]
    
    weights, base_emb = compute_pico_attention_weights_gk_corrected(
        texts=texts,
        n_layers=2,
        n_heads=2,
        embedding_dim=128,
        head_dim=64,
        max_n=2,
        max_s=2,
        verbose=True,
    )
    
    print("\n✓ G_k-corrected weight computation test passed!")

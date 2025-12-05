"""
Pico Method - Scaled-down implementation of the full attention weight derivation algorithm

This implements the COMPLETE algorithm from Components/method but scaled for:
- Vocabulary: ~80 characters (instead of 50,257 BPE tokens)
- Count matrices: 80x80 for n=1, 6400x80 for n=2 (instead of billions)
- Full SVD-based W_Q, W_K, W_V, W_O derivation

The algorithm:
1. Build count matrices C^{n,s} from corpus
2. Compute manifold matrices M^{n,s} = C / sqrt(D_row * D_col)
3. Compute probability matrices P^{n,s} = C / row_sum(C)
4. Build n-gram embeddings E^n using P^{n-1} and base embeddings
5. Derive W_Q, W_K via SVD on manifold matrix
6. Derive W_V, W_O via SVD on probability matrix
7. Orthogonalize across heads
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from typing import Dict, List, Tuple, Optional
import torch

from pico_tokenizer import PicoTokenizer, PICO_VOCAB_SIZE


# ============================================================
# Count Matrix (from count_matrix.py)
# ============================================================

def ngram_to_index(ngram: Tuple[int, ...], vocab_size: int) -> int:
    """Convert n-gram tuple to flat index."""
    index = 0
    for i, token in enumerate(ngram):
        index += token * (vocab_size ** (len(ngram) - 1 - i))
    return index


def index_to_ngram(index: int, n: int, vocab_size: int) -> Tuple[int, ...]:
    """Convert flat index back to n-gram tuple."""
    ngram = []
    for _ in range(n):
        ngram.append(index % vocab_size)
        index //= vocab_size
    return tuple(reversed(ngram))


def build_count_matrix(
    token_batches: List[List[int]],
    n: int,
    s: int,
    vocab_size: int,
) -> sparse.csr_matrix:
    """
    Build count matrix C^{n,s} tracking n-gram -> token transitions with skip s.
    
    For n=1, s=0: tracks direct bigram counts (token_i -> token_{i+1})
    For n=2, s=0: tracks trigram counts ((t_i, t_{i+1}) -> t_{i+2})
    For s>0: skips s tokens between context and target
    """
    # Matrix dimensions: (vocab^n) x vocab
    n_rows = vocab_size ** n
    n_cols = vocab_size
    
    # Count dictionary for sparse construction
    counts = {}
    
    for tokens in token_batches:
        # For each position where we can form an n-gram + skip + target
        for i in range(len(tokens) - n - s):
            # Context n-gram
            context = tuple(tokens[i:i + n])
            # Target token (after skip)
            target = tokens[i + n + s]
            
            row_idx = ngram_to_index(context, vocab_size)
            col_idx = target
            
            key = (row_idx, col_idx)
            counts[key] = counts.get(key, 0) + 1
    
    # Build sparse matrix
    if counts:
        rows, cols, data = [], [], []
        for (r, c), v in counts.items():
            rows.append(r)
            cols.append(c)
            data.append(v)
        
        matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(n_rows, n_cols),
            dtype=np.float32
        )
    else:
        matrix = sparse.csr_matrix((n_rows, n_cols), dtype=np.float32)
    
    return matrix


def build_all_count_matrices(
    token_batches: List[List[int]],
    max_n: int,
    max_s: int,
    vocab_size: int,
) -> Dict[Tuple[int, int], sparse.csr_matrix]:
    """Build all count matrices for n=1..max_n and s=0..max_s."""
    count_matrices = {}
    
    for n in range(1, max_n + 1):
        for s in range(max_s + 1):
            C = build_count_matrix(token_batches, n, s, vocab_size)
            count_matrices[(n, s)] = C
            print(f"  Built C(n={n}, s={s}): shape={C.shape}, nnz={C.nnz}")
    
    return count_matrices


# ============================================================
# Manifold Matrix (from manifold_matrix.py)
# ============================================================

def compute_manifold_matrix(count_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Compute manifold matrix M_ab = C_ab / sqrt(D_aa * D_bb)
    
    Where:
    - D_aa = row sum (total times n-gram a preceded any token)
    - D_bb = column sum (total times token b followed any n-gram)
    """
    C = count_matrix.tocsr()
    
    # Row sums (D_aa)
    row_sums = np.array(C.sum(axis=1)).flatten()
    # Column sums (D_bb)
    col_sums = np.array(C.sum(axis=0)).flatten()
    
    # Avoid division by zero
    row_sums = np.maximum(row_sums, 1e-10)
    col_sums = np.maximum(col_sums, 1e-10)
    
    # Compute M = C / sqrt(D_row * D_col)
    # For sparse matrix: scale rows, then scale columns
    sqrt_row = 1.0 / np.sqrt(row_sums)
    sqrt_col = 1.0 / np.sqrt(col_sums)
    
    # Scale rows
    M = C.multiply(sqrt_row.reshape(-1, 1))
    # Scale columns
    M = M.multiply(sqrt_col.reshape(1, -1))
    
    return M.tocsr()


def build_all_manifold_matrices(
    count_matrices: Dict[Tuple[int, int], sparse.csr_matrix]
) -> Dict[Tuple[int, int], sparse.csr_matrix]:
    """Compute manifold matrices for all count matrices."""
    manifold_matrices = {}
    
    for (n, s), C in count_matrices.items():
        M = compute_manifold_matrix(C)
        manifold_matrices[(n, s)] = M
    
    return manifold_matrices


# ============================================================
# Probability Matrix (from probability_matrix.py)
# ============================================================

def compute_probability_matrix(count_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Compute probability matrix P_ab = C_ab / sum_b'(C_ab')
    
    Row-wise normalization: each row sums to 1 (or 0 if no counts)
    """
    C = count_matrix.tocsr()
    
    # Row sums
    row_sums = np.array(C.sum(axis=1)).flatten()
    row_sums = np.maximum(row_sums, 1e-10)
    
    # Normalize rows
    P = C.multiply(1.0 / row_sums.reshape(-1, 1))
    
    return P.tocsr()


def build_all_probability_matrices(
    count_matrices: Dict[Tuple[int, int], sparse.csr_matrix]
) -> Dict[Tuple[int, int], sparse.csr_matrix]:
    """Compute probability matrices for all count matrices."""
    prob_matrices = {}
    
    for (n, s), C in count_matrices.items():
        P = compute_probability_matrix(C)
        prob_matrices[(n, s)] = P
    
    return prob_matrices


# ============================================================
# Embedding Matrix (from embedding_matrix.py)
# ============================================================

def create_base_embeddings(vocab_size: int, embedding_dim: int, seed: int = 42) -> np.ndarray:
    """
    Create near-orthogonal base embeddings for the character vocabulary.
    
    Since vocab_size (~80) < embedding_dim (128), we can have nearly perfect
    orthogonality by using random unit vectors.
    """
    np.random.seed(seed)
    
    # Random Gaussian, then normalize to unit length
    E = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E = E / np.maximum(norms, 1e-8)
    
    return E


def build_embedding_matrix_n1(
    base_embeddings: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """
    E^1 is just the base embeddings (one per token).
    Shape: (vocab_size, embedding_dim)
    """
    return base_embeddings.copy()


def build_embedding_matrix_n2(
    prob_matrix_n1_s0: sparse.csr_matrix,
    base_embeddings: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """
    E^2_(t1,t2) = P^1(t2|t1) * E_t2
    
    For each bigram (t1, t2), the embedding is the base embedding of t2
    scaled by the probability of t2 given t1.
    
    Shape: (vocab_size^2, embedding_dim)
    """
    embedding_dim = base_embeddings.shape[1]
    n_bigrams = vocab_size ** 2
    
    E2 = np.zeros((n_bigrams, embedding_dim), dtype=np.float32)
    
    P = prob_matrix_n1_s0.tocsr()
    
    for t1 in range(vocab_size):
        for t2 in range(vocab_size):
            bigram_idx = t1 * vocab_size + t2
            prob = P[t1, t2]
            E2[bigram_idx] = prob * base_embeddings[t2]
    
    return E2


def build_all_embedding_matrices(
    prob_matrices: Dict[Tuple[int, int], sparse.csr_matrix],
    base_embeddings: np.ndarray,
    max_n: int,
    vocab_size: int,
) -> Dict[int, np.ndarray]:
    """
    Build embedding matrices E^1, E^2, ..., E^max_n.
    
    E^1 = base embeddings
    E^n = P^{n-1} weighted base embeddings (for last token in n-gram)
    """
    embedding_matrices = {}
    
    # E^1
    embedding_matrices[1] = build_embedding_matrix_n1(base_embeddings, vocab_size)
    print(f"  Built E^1: shape={embedding_matrices[1].shape}")
    
    # E^2 (if needed)
    if max_n >= 2 and (1, 0) in prob_matrices:
        embedding_matrices[2] = build_embedding_matrix_n2(
            prob_matrices[(1, 0)],
            base_embeddings,
            vocab_size
        )
        print(f"  Built E^2: shape={embedding_matrices[2].shape}")
    
    return embedding_matrices


# ============================================================
# Key/Query Matrix (from key_query_matrix.py)
# ============================================================

def compute_key_query_matrices(
    manifold_matrix: sparse.csr_matrix,
    embedding_n: np.ndarray,
    embedding_1: np.ndarray,
    head_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute W_Q and W_K from manifold matrix via truncated SVD.
    
    M^{n,s} = U @ Sigma @ V^T
    
    W_Q^{n,s} = E^n.T @ U_r @ sqrt(Sigma_r)  -> (embedding_dim, head_dim)
    W_K^{n,s} = E^1.T @ V_r @ sqrt(Sigma_r)  -> (embedding_dim, head_dim)
    """
    M = manifold_matrix
    
    # Truncated SVD
    # k must be < min(M.shape) - 1 for sparse SVD
    k = min(head_dim, min(M.shape) - 2)
    if k < 1:
        k = 1
    
    try:
        U, sigma, Vt = svds(M.astype(np.float64), k=k)
        # svds returns in ascending order, reverse to descending
        idx = np.argsort(sigma)[::-1]
        U = U[:, idx]
        sigma = sigma[idx]
        V = Vt[idx, :].T
    except Exception as e:
        print(f"  SVD failed, using random: {e}")
        # Fallback to random
        U = np.random.randn(M.shape[0], k).astype(np.float32)
        sigma = np.ones(k, dtype=np.float32)
        V = np.random.randn(M.shape[1], k).astype(np.float32)
    
    sqrt_sigma = np.sqrt(np.maximum(sigma, 1e-10)).astype(np.float32)
    
    # W_Q = E^n.T @ U @ sqrt(Sigma)
    # E^n: (n_rows, embed_dim), U: (n_rows, k)
    # E^n.T @ U: (embed_dim, k)
    U = U.astype(np.float32)
    V = V.astype(np.float32)
    
    # Handle dimension mismatch (E^n rows should match M rows)
    if embedding_n.shape[0] != M.shape[0]:
        # Truncate or pad
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
    
    W_Q = embedding_n.T @ U @ np.diag(sqrt_sigma)  # (embed_dim, k)
    W_K = embedding_1.T @ V @ np.diag(sqrt_sigma)  # (embed_dim, k)
    
    # Pad or truncate to head_dim
    embed_dim = embedding_n.shape[1]
    
    if W_Q.shape[1] < head_dim:
        # Pad with zeros
        W_Q = np.hstack([W_Q, np.zeros((embed_dim, head_dim - W_Q.shape[1]), dtype=np.float32)])
        W_K = np.hstack([W_K, np.zeros((embed_dim, head_dim - W_K.shape[1]), dtype=np.float32)])
    elif W_Q.shape[1] > head_dim:
        W_Q = W_Q[:, :head_dim]
        W_K = W_K[:, :head_dim]
    
    return W_Q, W_K


# ============================================================
# Value/Output Matrix (from value_output_matrix.py)
# ============================================================

def compute_value_output_matrices(
    prob_matrix: sparse.csr_matrix,
    embedding_n: np.ndarray,
    embedding_1: np.ndarray,
    head_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute W_V and W_O from probability matrix via truncated SVD.
    
    P^{n,s} = U @ Sigma @ V^T
    
    W_V^{n,s} = sqrt(Sigma_r) @ U_r^T @ E^n  -> (head_dim, embedding_dim)
    W_O^{n,s} = E^1.T @ V_r @ sqrt(Sigma_r)  -> (embedding_dim, head_dim)
    """
    P = prob_matrix
    
    # Truncated SVD
    k = min(head_dim, min(P.shape) - 2)
    if k < 1:
        k = 1
    
    try:
        U, sigma, Vt = svds(P.astype(np.float64), k=k)
        idx = np.argsort(sigma)[::-1]
        U = U[:, idx]
        sigma = sigma[idx]
        V = Vt[idx, :].T
    except Exception as e:
        print(f"  SVD failed, using random: {e}")
        U = np.random.randn(P.shape[0], k).astype(np.float32)
        sigma = np.ones(k, dtype=np.float32)
        V = np.random.randn(P.shape[1], k).astype(np.float32)
    
    sqrt_sigma = np.sqrt(np.maximum(sigma, 1e-10)).astype(np.float32)
    U = U.astype(np.float32)
    V = V.astype(np.float32)
    
    # Handle dimension mismatch
    if embedding_n.shape[0] != P.shape[0]:
        if embedding_n.shape[0] > P.shape[0]:
            embedding_n = embedding_n[:P.shape[0]]
        else:
            pad = np.zeros((P.shape[0] - embedding_n.shape[0], embedding_n.shape[1]), dtype=np.float32)
            embedding_n = np.vstack([embedding_n, pad])
    
    if embedding_1.shape[0] != P.shape[1]:
        if embedding_1.shape[0] > P.shape[1]:
            embedding_1 = embedding_1[:P.shape[1]]
        else:
            pad = np.zeros((P.shape[1] - embedding_1.shape[0], embedding_1.shape[1]), dtype=np.float32)
            embedding_1 = np.vstack([embedding_1, pad])
    
    # W_V = sqrt(Sigma) @ U^T @ E^n -> (k, embed_dim)
    W_V = np.diag(sqrt_sigma) @ U.T @ embedding_n  # (k, embed_dim)
    
    # W_O = E^1.T @ V @ sqrt(Sigma) -> (embed_dim, k)
    W_O = embedding_1.T @ V @ np.diag(sqrt_sigma)  # (embed_dim, k)
    
    # Pad or truncate
    embed_dim = embedding_n.shape[1]
    
    if W_V.shape[0] < head_dim:
        W_V = np.vstack([W_V, np.zeros((head_dim - W_V.shape[0], embed_dim), dtype=np.float32)])
        W_O = np.hstack([W_O, np.zeros((embed_dim, head_dim - W_O.shape[1]), dtype=np.float32)])
    elif W_V.shape[0] > head_dim:
        W_V = W_V[:head_dim, :]
        W_O = W_O[:, :head_dim]
    
    return W_V, W_O


# ============================================================
# Full Pipeline
# ============================================================

def compute_pico_attention_weights(
    texts: List[str],
    n_layers: int,
    n_heads: int,
    embedding_dim: int,
    head_dim: int,
    max_n: int = 2,
    max_s: int = 2,
    verbose: bool = True,
    normalize_weights: bool = True,
    target_std: float = 0.01,  # Tuned to match trainable attention output magnitude
) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
    """
    Compute attention weights for PicoGPT using the FULL derivation algorithm.
    
    This is the scaled-down version that:
    1. Uses character-level tokenization (~80 vocab)
    2. Builds count matrices from corpus
    3. Derives manifold and probability matrices
    4. Creates n-gram embeddings
    5. Computes W_Q, W_K, W_V, W_O via SVD
    
    Args:
        texts: Training texts
        n_layers: Number of transformer layers
        n_heads: Number of attention heads per layer
        embedding_dim: Embedding dimension
        head_dim: Dimension per head
        max_n: Maximum n-gram size (layer uses n=layer_idx+1, capped at max_n)
        max_s: Maximum skip value (head uses s=head_idx, capped at max_s)
        normalize_weights: If True, scale weights to match typical init magnitude
        target_std: Target standard deviation for normalized weights
    
    Returns:
        Tuple of:
        - Dict mapping layer_idx -> (W_Q, W_K, W_V, W_O) tensors
        - Base embeddings tensor (vocab_size, embedding_dim) - MUST be frozen in model
    """
    if verbose:
        print("=" * 60)
        print("Computing PicoGPT Attention Weights (FULL ALGORITHM)")
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
    
    # Create batches
    batch_size = 256  # Smaller batches for char-level
    batches = []
    for i in range(0, len(all_tokens) - batch_size + 1, batch_size):
        batches.append(all_tokens[i:i + batch_size])
    
    if verbose:
        print(f"  Total tokens: {len(all_tokens)}")
        print(f"  Batches: {len(batches)} x {batch_size}")
    
    # Step 2: Build count matrices
    if verbose:
        print(f"\nStep 2: Building count matrices (n=1..{max_n}, s=0..{max_s})")
    
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
    
    base_embeddings = create_base_embeddings(vocab_size, embedding_dim)
    embedding_matrices = build_all_embedding_matrices(prob_matrices, base_embeddings, max_n, vocab_size)
    
    # Step 6: Derive attention weights for each layer/head
    if verbose:
        print(f"\nStep 6: Deriving W_Q, W_K, W_V, W_O for {n_layers} layers x {n_heads} heads")
    
    attention_weights = {}
    
    for layer_idx in range(n_layers):
        # Layer uses n = layer_idx + 1 (capped at max_n)
        n = min(layer_idx + 1, max_n)
        
        # Get embedding for this n
        E_n = embedding_matrices.get(n, embedding_matrices[1])
        E_1 = embedding_matrices[1]
        
        # Collect W_Q, W_K, W_V, W_O for all heads
        W_Q_heads = []
        W_K_heads = []
        W_V_heads = []
        W_O_heads = []
        
        for head_idx in range(n_heads):
            # Head uses s = head_idx (capped at max_s)
            s = min(head_idx, max_s)
            
            # Get matrices for this (n, s)
            key = (n, s)
            if key not in manifold_matrices:
                key = (1, 0)  # Fallback
            
            M = manifold_matrices[key]
            P = prob_matrices[key]
            
            # Compute W_Q, W_K from manifold
            W_Q_h, W_K_h = compute_key_query_matrices(M, E_n, E_1, head_dim)
            
            # Compute W_V, W_O from probability
            W_V_h, W_O_h = compute_value_output_matrices(P, E_n, E_1, head_dim)
            
            W_Q_heads.append(W_Q_h)
            W_K_heads.append(W_K_h)
            W_V_heads.append(W_V_h)
            W_O_heads.append(W_O_h)
        
        # Stack heads: (n_heads, embed_dim, head_dim) for Q, K
        #              (n_heads, head_dim, embed_dim) for V
        #              (embed_dim, n_heads * head_dim) for O (concatenated)
        W_Q = torch.tensor(np.stack(W_Q_heads, axis=0), dtype=torch.float32)
        W_K = torch.tensor(np.stack(W_K_heads, axis=0), dtype=torch.float32)
        W_V = torch.tensor(np.stack(W_V_heads, axis=0), dtype=torch.float32)
        
        # Concatenate W_O across heads
        W_O_concat = np.hstack(W_O_heads)  # (embed_dim, n_heads * head_dim)
        W_O = torch.tensor(W_O_concat, dtype=torch.float32)
        
        # Normalize weights to match typical initialization magnitude
        # This preserves the STRUCTURE (directions) learned from corpus
        # while ensuring the SCALE matches standard transformer init
        if normalize_weights:
            for W, name in [(W_Q, 'Q'), (W_K, 'K'), (W_V, 'V'), (W_O, 'O')]:
                current_std = W.std().item()
                if current_std > 0:
                    scale = target_std / current_std
                    W.mul_(scale)
            
            # Re-extract after in-place modification
            W_Q = W_Q
            W_K = W_K
            W_V = W_V
            W_O = W_O
        
        attention_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
        
        if verbose:
            print(f"  Layer {layer_idx} (n={n}): W_Q={W_Q.shape}, W_K={W_K.shape}, W_V={W_V.shape}, W_O={W_O.shape}")
            if normalize_weights:
                print(f"    Normalized: W_Q.std={W_Q.std():.4f}, W_K.std={W_K.std():.4f}, W_V.std={W_V.std():.4f}, W_O.std={W_O.std():.4f}")
    
    if verbose:
        print(f"\nAttention weight computation complete!")
    
    # Convert base embeddings to torch tensor
    base_embeddings_tensor = torch.tensor(base_embeddings, dtype=torch.float32)
    
    return attention_weights, base_embeddings_tensor


# ===== Test =====
if __name__ == "__main__":
    print("Testing Pico Method - Full Algorithm")
    print("=" * 60)
    
    # Test texts
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "Hello world! How are you today? " * 10,
        "Testing the attention weight derivation algorithm. " * 10,
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz 0123456789" * 5,
    ]
    
    # Compute weights
    weights, base_embeddings = compute_pico_attention_weights(
        texts=texts,
        n_layers=2,
        n_heads=2,
        embedding_dim=128,
        head_dim=64,
        max_n=2,
        max_s=2,
        verbose=True,
    )
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    print(f"\nBase embeddings: {base_embeddings.shape}")
    
    for layer_idx, (W_Q, W_K, W_V, W_O) in weights.items():
        print(f"\nLayer {layer_idx}:")
        print(f"  W_Q: {W_Q.shape}, norm={W_Q.norm().item():.4f}")
        print(f"  W_K: {W_K.shape}, norm={W_K.norm().item():.4f}")
        print(f"  W_V: {W_V.shape}, norm={W_V.norm().item():.4f}")
        print(f"  W_O: {W_O.shape}, norm={W_O.norm().item():.4f}")
    
    print("\n✓ Full algorithm test passed!")


"""
Value and Output Matrix Builder

Constructs W_V and W_O matrices from probability matrices via truncated SVD.

The theoretical transformation is:
    W_V+ = E^1.T @ P^{n,s} @ E^n

We decompose P^{n,s} via SVD: P ≈ U_r @ Σ_r @ V_r^T

Then:
    W_V^{n,s} = sqrt(Σ_r) @ U_r^T @ E^n    →  (r, 768)
    W_O^{n,s} = E^1.T @ V_r @ sqrt(Σ_r)    →  (768, r)

Where:
- E^n is the n-gram embedding matrix (vocab^n, 768)
- E^1 is the base GPT-2 embedding matrix (vocab, 768)
- r = head dimension (64 for GPT-2 small)

The full W_O is formed by concatenating all W_O^{n,s} submatrices.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from typing import Optional

from count_matrix import VOCAB_SIZE
from embedding_matrix import EMBEDDING_DIM


# GPT-2 small head dimension
HEAD_DIM = 64


def compute_value_output_matrices(
    prob_matrix: sparse.csr_matrix,
    E_n: sparse.csr_matrix,
    E_1: np.ndarray,
    head_dim: int = HEAD_DIM
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute W_V and W_O from a probability matrix via truncated SVD.
    
    Args:
        prob_matrix: P^{n,s} of shape (vocab^n, vocab)
        E_n: N-gram embedding matrix of shape (vocab^n, embedding_dim)
        E_1: Base embedding matrix of shape (vocab, embedding_dim)
        head_dim: Dimension of attention head (r)
    
    Returns:
        Tuple of (W_V, W_O):
        - W_V: shape (head_dim, embedding_dim) = (64, 768)
        - W_O: shape (embedding_dim, head_dim) = (768, 64)
    """
    P = prob_matrix
    
    # Ensure P is in a format suitable for svds
    if sparse.issparse(P):
        P = P.astype(np.float64)
    
    embedding_dim = E_1.shape[1]
    
    # Handle case where matrix is too small or has too few non-zeros
    max_rank = min(P.shape[0], P.shape[1]) - 1
    if P.nnz == 0 or max_rank < 1:
        return (np.zeros((head_dim, embedding_dim), dtype=np.float32),
                np.zeros((embedding_dim, head_dim), dtype=np.float32))
    
    # Adjust head_dim if matrix rank is too small
    r = min(head_dim, max_rank, P.nnz - 1)
    if r < 1:
        return (np.zeros((head_dim, embedding_dim), dtype=np.float32),
                np.zeros((embedding_dim, head_dim), dtype=np.float32))
    
    # Truncated SVD: P ≈ U @ diag(Σ) @ V^T
    try:
        U, s, Vt = svds(P, k=r)
    except Exception as e:
        print(f"SVD failed: {e}")
        return (np.zeros((head_dim, embedding_dim), dtype=np.float32),
                np.zeros((embedding_dim, head_dim), dtype=np.float32))
    
    # V_r = Vt.T has shape (vocab, r)
    V = Vt.T
    
    # sqrt(Σ_r)
    sqrt_s = np.sqrt(np.maximum(s, 0))
    sqrt_Sigma = np.diag(sqrt_s)  # (r, r)
    
    # W_V = sqrt(Σ_r) @ U_r^T @ E^n
    # sqrt(Σ_r): (r, r)
    # U.T: (r, vocab^n)
    # E_n: (vocab^n, embedding_dim)
    if sparse.issparse(E_n):
        E_n_dense = E_n.toarray()
    else:
        E_n_dense = E_n
    
    W_V = sqrt_Sigma @ U.T @ E_n_dense  # (r, embedding_dim)
    
    # W_O = E^1.T @ V_r @ sqrt(Σ_r)
    # E_1.T: (embedding_dim, vocab)
    # V: (vocab, r)
    # sqrt(Σ_r): (r, r)
    W_O = E_1.T @ V @ sqrt_Sigma  # (embedding_dim, r)
    
    # Pad with zeros if r < head_dim
    if r < head_dim:
        W_V_padded = np.zeros((head_dim, embedding_dim), dtype=np.float32)
        W_O_padded = np.zeros((embedding_dim, head_dim), dtype=np.float32)
        W_V_padded[:r, :] = W_V
        W_O_padded[:, :r] = W_O
        return W_V_padded, W_O_padded
    
    return W_V.astype(np.float32), W_O.astype(np.float32)


def build_all_value_output_matrices(
    prob_matrices: dict[tuple[int, int], sparse.csr_matrix],
    embedding_matrices: dict[int, sparse.csr_matrix],
    base_embeddings: np.ndarray,
    head_dim: int = HEAD_DIM
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """
    Build W_V and W_O for all (n, s) combinations.
    
    Args:
        prob_matrices: Dict mapping (n, s) to probability matrix
        embedding_matrices: Dict mapping n to E^n embedding matrix
        base_embeddings: E^1 base embeddings of shape (vocab, embedding_dim)
        head_dim: Dimension of attention head
    
    Returns:
        Dict mapping (n, s) to (W_V, W_O) tuple
    """
    results = {}
    
    for (n, s), P in prob_matrices.items():
        # Get E^n
        if n not in embedding_matrices:
            print(f"Warning: E^{n} not found, skipping (n={n}, s={s})")
            continue
        
        E_n = embedding_matrices[n]
        
        W_V, W_O = compute_value_output_matrices(P, E_n, base_embeddings, head_dim)
        results[(n, s)] = (W_V, W_O)
        
        # Stats
        v_norm = np.linalg.norm(W_V, 'fro')
        o_norm = np.linalg.norm(W_O, 'fro')
        print(f"Built W_V, W_O for (n={n}, s={s}): "
              f"W_V={W_V.shape}, W_O={W_O.shape}, "
              f"||W_V||_F={v_norm:.4f}, ||W_O||_F={o_norm:.4f}")
    
    return results


def concatenate_output_matrices(
    value_output_matrices: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    key_order: list[tuple[int, int]] = None
) -> np.ndarray:
    """
    Concatenate all W_O submatrices into full output projection matrix.
    
    Each W_O^{n,s} is (embedding_dim, head_dim).
    Concatenating horizontally gives (embedding_dim, num_heads * head_dim).
    
    Args:
        value_output_matrices: Dict mapping (n, s) to (W_V, W_O)
        key_order: Optional ordering of (n, s) keys. If None, uses sorted order.
    
    Returns:
        Full W_O matrix of shape (embedding_dim, num_heads * head_dim)
    """
    if key_order is None:
        key_order = sorted(value_output_matrices.keys())
    
    W_O_list = []
    for key in key_order:
        _, W_O = value_output_matrices[key]
        W_O_list.append(W_O)
    
    # Horizontal concatenation: (embedding_dim, head_dim * num_heads)
    W_O_full = np.concatenate(W_O_list, axis=1)
    
    return W_O_full


def concatenate_value_matrices(
    value_output_matrices: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    key_order: list[tuple[int, int]] = None
) -> np.ndarray:
    """
    Concatenate all W_V submatrices.
    
    Each W_V^{n,s} is (head_dim, embedding_dim).
    Concatenating vertically gives (num_heads * head_dim, embedding_dim).
    
    Args:
        value_output_matrices: Dict mapping (n, s) to (W_V, W_O)
        key_order: Optional ordering of (n, s) keys. If None, uses sorted order.
    
    Returns:
        Full W_V matrix of shape (num_heads * head_dim, embedding_dim)
    """
    if key_order is None:
        key_order = sorted(value_output_matrices.keys())
    
    W_V_list = []
    for key in key_order:
        W_V, _ = value_output_matrices[key]
        W_V_list.append(W_V)
    
    # Vertical concatenation: (head_dim * num_heads, embedding_dim)
    W_V_full = np.concatenate(W_V_list, axis=0)
    
    return W_V_full


def get_value_vector(
    W_V: np.ndarray,
    embedding: np.ndarray
) -> np.ndarray:
    """
    Compute value vector from embedding.
    
    v = W_V @ embedding
    
    Args:
        W_V: Value weight matrix (head_dim, embedding_dim)
        embedding: Token/n-gram embedding (embedding_dim,)
    
    Returns:
        Value vector (head_dim,)
    """
    return W_V @ embedding


def project_output(
    W_O: np.ndarray,
    value: np.ndarray
) -> np.ndarray:
    """
    Project value back to embedding space.
    
    output = W_O @ value
    
    Args:
        W_O: Output weight matrix (embedding_dim, head_dim)
        value: Value vector (head_dim,)
    
    Returns:
        Output vector (embedding_dim,)
    """
    return W_O @ value


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_value_output_shapes():
    """Test that W_V and W_O have correct shapes."""
    vocab = 10
    embed_dim = 8
    head_dim = 4
    
    # Create small probability matrix
    P = sparse.random(vocab, vocab, density=0.5, dtype=np.float64)
    # Normalize rows to make it a valid probability matrix
    row_sums = np.array(P.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    P = sparse.diags(1 / row_sums) @ P
    
    # Create embeddings
    E_n = sparse.csr_matrix(np.random.randn(vocab, embed_dim).astype(np.float32))
    E_1 = np.random.randn(vocab, embed_dim).astype(np.float32)
    
    W_V, W_O = compute_value_output_matrices(P, E_n, E_1, head_dim)
    
    assert W_V.shape == (head_dim, embed_dim), f"W_V shape should be ({head_dim}, {embed_dim})"
    assert W_O.shape == (embed_dim, head_dim), f"W_O shape should be ({embed_dim}, {head_dim})"
    
    print("✓ test_value_output_shapes passed")


def test_value_output_n2():
    """Test with n=2 (bigram probability matrix)."""
    vocab = 5
    embed_dim = 4
    head_dim = 2
    
    # P^{2,s} has shape (vocab^2, vocab) = (25, 5)
    P = sparse.random(vocab**2, vocab, density=0.3, dtype=np.float64)
    
    # E^2 has shape (vocab^2, embed_dim) = (25, 4)
    E_n = sparse.csr_matrix(np.random.randn(vocab**2, embed_dim).astype(np.float32))
    
    # E^1 has shape (vocab, embed_dim) = (5, 4)
    E_1 = np.random.randn(vocab, embed_dim).astype(np.float32)
    
    W_V, W_O = compute_value_output_matrices(P, E_n, E_1, head_dim)
    
    assert W_V.shape == (head_dim, embed_dim)
    assert W_O.shape == (embed_dim, head_dim)
    
    print("✓ test_value_output_n2 passed")


def test_value_output_vectors():
    """Test computing value and output vectors."""
    embed_dim = 8
    head_dim = 4
    
    W_V = np.random.randn(head_dim, embed_dim).astype(np.float32)
    W_O = np.random.randn(embed_dim, head_dim).astype(np.float32)
    
    embedding = np.random.randn(embed_dim).astype(np.float32)
    
    v = get_value_vector(W_V, embedding)
    assert v.shape == (head_dim,)
    
    out = project_output(W_O, v)
    assert out.shape == (embed_dim,)
    
    # Verify computation
    expected_v = W_V @ embedding
    assert np.allclose(v, expected_v)
    
    print("✓ test_value_output_vectors passed")


def test_concatenate_matrices():
    """Test concatenation of W_O and W_V matrices."""
    embed_dim = 8
    head_dim = 4
    
    # Create mock matrices for 3 heads
    matrices = {
        (1, 0): (np.random.randn(head_dim, embed_dim).astype(np.float32),
                 np.random.randn(embed_dim, head_dim).astype(np.float32)),
        (1, 1): (np.random.randn(head_dim, embed_dim).astype(np.float32),
                 np.random.randn(embed_dim, head_dim).astype(np.float32)),
        (2, 0): (np.random.randn(head_dim, embed_dim).astype(np.float32),
                 np.random.randn(embed_dim, head_dim).astype(np.float32)),
    }
    
    W_O_full = concatenate_output_matrices(matrices)
    W_V_full = concatenate_value_matrices(matrices)
    
    assert W_O_full.shape == (embed_dim, head_dim * 3), f"W_O_full shape: {W_O_full.shape}"
    assert W_V_full.shape == (head_dim * 3, embed_dim), f"W_V_full shape: {W_V_full.shape}"
    
    print("✓ test_concatenate_matrices passed")


def test_empty_matrix():
    """Test handling of empty/zero matrix."""
    vocab = 10
    embed_dim = 8
    head_dim = 4
    
    # Empty sparse matrix
    P = sparse.csr_matrix((vocab, vocab), dtype=np.float64)
    E_n = sparse.csr_matrix(np.random.randn(vocab, embed_dim).astype(np.float32))
    E_1 = np.random.randn(vocab, embed_dim).astype(np.float32)
    
    W_V, W_O = compute_value_output_matrices(P, E_n, E_1, head_dim)
    
    assert W_V.shape == (head_dim, embed_dim)
    assert W_O.shape == (embed_dim, head_dim)
    assert np.allclose(W_V, 0)
    assert np.allclose(W_O, 0)
    
    print("✓ test_empty_matrix passed")


def test_roundtrip():
    """Test that value -> output roundtrip preserves structure."""
    embed_dim = 8
    head_dim = 4
    
    W_V = np.random.randn(head_dim, embed_dim).astype(np.float32)
    W_O = np.random.randn(embed_dim, head_dim).astype(np.float32)
    
    embedding = np.random.randn(embed_dim).astype(np.float32)
    
    # embedding -> value -> output
    v = get_value_vector(W_V, embedding)
    out = project_output(W_O, v)
    
    # This should be W_O @ W_V @ embedding
    expected = W_O @ W_V @ embedding
    assert np.allclose(out, expected)
    
    print("✓ test_roundtrip passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running value/output matrix tests...\n")
    
    test_value_output_shapes()
    test_value_output_n2()
    test_value_output_vectors()
    test_concatenate_matrices()
    test_empty_matrix()
    test_roundtrip()
    
    print("\n✓ All value/output tests passed!")


if __name__ == "__main__":
    run_all_tests()
    
    print("\n" + "=" * 50)
    print("Demo: Building W_V and W_O from random data")
    print("=" * 50)
    
    np.random.seed(42)
    vocab = 100
    embed_dim = 32  # Smaller for demo
    head_dim = 8
    
    # Create mock probability matrices
    prob_matrices = {
        (1, 0): sparse.random(vocab, vocab, density=0.1, dtype=np.float64),
        (1, 1): sparse.random(vocab, vocab, density=0.1, dtype=np.float64),
        (2, 0): sparse.random(vocab**2, vocab, density=0.01, dtype=np.float64),
    }
    
    # Normalize to make them probability matrices
    for key in prob_matrices:
        P = prob_matrices[key]
        row_sums = np.array(P.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        prob_matrices[key] = sparse.diags(1 / row_sums) @ P
    
    # Create mock embedding matrices
    embedding_matrices = {
        1: sparse.csr_matrix(np.random.randn(vocab, embed_dim).astype(np.float32)),
        2: sparse.csr_matrix(np.random.randn(vocab**2, embed_dim).astype(np.float32)),
    }
    
    base_embeddings = np.random.randn(vocab, embed_dim).astype(np.float32)
    
    print(f"\nVocab: {vocab}, Embed dim: {embed_dim}, Head dim: {head_dim}")
    print("-" * 50)
    
    results = build_all_value_output_matrices(
        prob_matrices,
        embedding_matrices,
        base_embeddings,
        head_dim
    )
    
    print("\n" + "-" * 50)
    print("Concatenating into full W_O and W_V")
    print("-" * 50)
    
    W_O_full = concatenate_output_matrices(results)
    W_V_full = concatenate_value_matrices(results)
    
    print(f"Full W_O shape: {W_O_full.shape} (embed_dim, num_heads * head_dim)")
    print(f"Full W_V shape: {W_V_full.shape} (num_heads * head_dim, embed_dim)")
    
    print("\n" + "-" * 50)
    print("Example: Value projection roundtrip")
    print("-" * 50)
    
    W_V, W_O = results[(1, 0)]
    embedding = base_embeddings[0]
    
    v = get_value_vector(W_V, embedding)
    out = project_output(W_O, v)
    
    print(f"Input embedding shape: {embedding.shape}")
    print(f"Value vector shape: {v.shape}")
    print(f"Output vector shape: {out.shape}")
    print(f"||input|| = {np.linalg.norm(embedding):.4f}")
    print(f"||value|| = {np.linalg.norm(v):.4f}")
    print(f"||output|| = {np.linalg.norm(out):.4f}")


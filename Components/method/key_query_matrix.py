"""
Key and Query Matrix Builder

Constructs W_Q and W_K matrices from manifold matrices via truncated SVD.

Given M^{n,s} of shape (vocab^n, vocab):
1. Perform truncated SVD: M ≈ U_r @ Σ_r @ V_r^T
2. W_Q^{n,s} = E^n.T @ U_r @ sqrt(Σ_r)
3. W_K^{n,s} = E^1.T @ V_r @ sqrt(Σ_r)

Where:
- E^n is the n-gram embedding matrix (vocab^n, 768)
- E^1 is the base GPT-2 embedding matrix (vocab, 768)
- r = head dimension (64 for GPT-2 small)

Output shapes:
- W_Q: (embedding_dim, head_dim) = (768, 64)
- W_K: (embedding_dim, head_dim) = (768, 64)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from typing import Optional

from count_matrix import VOCAB_SIZE
from embedding_matrix import EMBEDDING_DIM, load_gpt2_embeddings


# GPT-2 small head dimension
HEAD_DIM = 64


def compute_key_query_matrices(
    manifold_matrix: sparse.csr_matrix,
    E_n: sparse.csr_matrix,
    E_1: np.ndarray,
    head_dim: int = HEAD_DIM
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute W_Q and W_K from a manifold matrix via truncated SVD.
    
    Args:
        manifold_matrix: M^{n,s} of shape (vocab^n, vocab)
        E_n: N-gram embedding matrix of shape (vocab^n, embedding_dim)
        E_1: Base embedding matrix of shape (vocab, embedding_dim)
        head_dim: Dimension of attention head (r)
    
    Returns:
        Tuple of (W_Q, W_K), each of shape (embedding_dim, head_dim)
    """
    M = manifold_matrix
    
    # Ensure M is in a format suitable for svds
    if sparse.issparse(M):
        M = M.astype(np.float64)
    
    # Handle case where matrix is too small or has too few non-zeros
    max_rank = min(M.shape[0], M.shape[1]) - 1
    if M.nnz == 0 or max_rank < 1:
        # Return zero matrices if SVD not possible
        embedding_dim = E_1.shape[1]
        return (np.zeros((embedding_dim, head_dim), dtype=np.float32),
                np.zeros((embedding_dim, head_dim), dtype=np.float32))
    
    # Adjust head_dim if matrix rank is too small
    r = min(head_dim, max_rank, M.nnz - 1)
    if r < 1:
        embedding_dim = E_1.shape[1]
        return (np.zeros((embedding_dim, head_dim), dtype=np.float32),
                np.zeros((embedding_dim, head_dim), dtype=np.float32))
    
    # Truncated SVD: M ≈ U @ diag(Σ) @ V^T
    # svds returns: U (m, r), s (r,), Vt (r, n)
    try:
        U, s, Vt = svds(M, k=r)
    except Exception as e:
        print(f"SVD failed: {e}")
        embedding_dim = E_1.shape[1]
        return (np.zeros((embedding_dim, head_dim), dtype=np.float32),
                np.zeros((embedding_dim, head_dim), dtype=np.float32))
    
    # V_r = Vt.T has shape (vocab, r)
    V = Vt.T
    
    # sqrt(Σ_r) as diagonal matrix
    sqrt_s = np.sqrt(np.maximum(s, 0))  # Ensure non-negative
    sqrt_Sigma = np.diag(sqrt_s)  # (r, r)
    
    # W_Q = E^n.T @ U_r @ sqrt(Σ_r)
    # E_n is sparse (vocab^n, embedding_dim), E_n.T is (embedding_dim, vocab^n)
    # U is (vocab^n, r)
    if sparse.issparse(E_n):
        E_n_dense = E_n.toarray()
    else:
        E_n_dense = E_n
    
    W_Q = E_n_dense.T @ U @ sqrt_Sigma  # (embedding_dim, r)
    
    # W_K = E^1.T @ V_r @ sqrt(Σ_r)
    # E_1 is (vocab, embedding_dim), E_1.T is (embedding_dim, vocab)
    # V is (vocab, r)
    W_K = E_1.T @ V @ sqrt_Sigma  # (embedding_dim, r)
    
    # Pad with zeros if r < head_dim
    embedding_dim = E_1.shape[1]
    if r < head_dim:
        W_Q_padded = np.zeros((embedding_dim, head_dim), dtype=np.float32)
        W_K_padded = np.zeros((embedding_dim, head_dim), dtype=np.float32)
        W_Q_padded[:, :r] = W_Q
        W_K_padded[:, :r] = W_K
        return W_Q_padded, W_K_padded
    
    return W_Q.astype(np.float32), W_K.astype(np.float32)


def build_all_key_query_matrices(
    manifold_matrices: dict[tuple[int, int], sparse.csr_matrix],
    embedding_matrices: dict[int, sparse.csr_matrix],
    base_embeddings: np.ndarray,
    head_dim: int = HEAD_DIM
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """
    Build W_Q and W_K for all (n, s) combinations.
    
    Args:
        manifold_matrices: Dict mapping (n, s) to manifold matrix
        embedding_matrices: Dict mapping n to E^n embedding matrix
        base_embeddings: E^1 base embeddings of shape (vocab, embedding_dim)
        head_dim: Dimension of attention head
    
    Returns:
        Dict mapping (n, s) to (W_Q, W_K) tuple
    """
    results = {}
    
    for (n, s), M in manifold_matrices.items():
        # Get E^n
        if n not in embedding_matrices:
            print(f"Warning: E^{n} not found, skipping (n={n}, s={s})")
            continue
        
        E_n = embedding_matrices[n]
        
        W_Q, W_K = compute_key_query_matrices(M, E_n, base_embeddings, head_dim)
        results[(n, s)] = (W_Q, W_K)
        
        # Stats
        q_norm = np.linalg.norm(W_Q, 'fro')
        k_norm = np.linalg.norm(W_K, 'fro')
        print(f"Built W_Q, W_K for (n={n}, s={s}): "
              f"shape={W_Q.shape}, "
              f"||W_Q||_F={q_norm:.4f}, ||W_K||_F={k_norm:.4f}")
    
    return results


def get_query_vector(
    W_Q: np.ndarray,
    embedding: np.ndarray
) -> np.ndarray:
    """
    Compute query vector from embedding.
    
    q = embedding @ W_Q
    
    Args:
        W_Q: Query weight matrix (embedding_dim, head_dim)
        embedding: Token/n-gram embedding (embedding_dim,)
    
    Returns:
        Query vector (head_dim,)
    """
    return embedding @ W_Q


def get_key_vector(
    W_K: np.ndarray,
    embedding: np.ndarray
) -> np.ndarray:
    """
    Compute key vector from embedding.
    
    k = embedding @ W_K
    
    Args:
        W_K: Key weight matrix (embedding_dim, head_dim)
        embedding: Token embedding (embedding_dim,)
    
    Returns:
        Key vector (head_dim,)
    """
    return embedding @ W_K


def compute_attention_score(
    query: np.ndarray,
    key: np.ndarray,
    scale: bool = True
) -> float:
    """
    Compute attention score between query and key.
    
    score = q · k / sqrt(d_k)
    
    Args:
        query: Query vector (head_dim,)
        key: Key vector (head_dim,)
        scale: Whether to scale by sqrt(head_dim)
    
    Returns:
        Attention score (scalar)
    """
    score = np.dot(query, key)
    if scale:
        score = score / np.sqrt(len(query))
    return score


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_key_query_shapes():
    """Test that W_Q and W_K have correct shapes."""
    vocab = 10
    embed_dim = 8
    head_dim = 4
    
    # Create small manifold matrix
    M = sparse.random(vocab, vocab, density=0.5, dtype=np.float64)
    
    # Create embeddings
    E_n = sparse.csr_matrix(np.random.randn(vocab, embed_dim).astype(np.float32))
    E_1 = np.random.randn(vocab, embed_dim).astype(np.float32)
    
    W_Q, W_K = compute_key_query_matrices(M, E_n, E_1, head_dim)
    
    assert W_Q.shape == (embed_dim, head_dim), f"W_Q shape should be ({embed_dim}, {head_dim})"
    assert W_K.shape == (embed_dim, head_dim), f"W_K shape should be ({embed_dim}, {head_dim})"
    
    print("✓ test_key_query_shapes passed")


def test_key_query_n2():
    """Test with n=2 (bigram manifold matrix)."""
    vocab = 5
    embed_dim = 4
    head_dim = 2
    
    # M^{2,s} has shape (vocab^2, vocab) = (25, 5)
    M = sparse.random(vocab**2, vocab, density=0.3, dtype=np.float64)
    
    # E^2 has shape (vocab^2, embed_dim) = (25, 4)
    E_n = sparse.csr_matrix(np.random.randn(vocab**2, embed_dim).astype(np.float32))
    
    # E^1 has shape (vocab, embed_dim) = (5, 4)
    E_1 = np.random.randn(vocab, embed_dim).astype(np.float32)
    
    W_Q, W_K = compute_key_query_matrices(M, E_n, E_1, head_dim)
    
    assert W_Q.shape == (embed_dim, head_dim)
    assert W_K.shape == (embed_dim, head_dim)
    
    print("✓ test_key_query_n2 passed")


def test_query_key_vectors():
    """Test computing query and key vectors."""
    embed_dim = 8
    head_dim = 4
    
    W_Q = np.random.randn(embed_dim, head_dim).astype(np.float32)
    W_K = np.random.randn(embed_dim, head_dim).astype(np.float32)
    
    embedding = np.random.randn(embed_dim).astype(np.float32)
    
    q = get_query_vector(W_Q, embedding)
    k = get_key_vector(W_K, embedding)
    
    assert q.shape == (head_dim,)
    assert k.shape == (head_dim,)
    
    # Verify computation
    expected_q = embedding @ W_Q
    assert np.allclose(q, expected_q)
    
    print("✓ test_query_key_vectors passed")


def test_attention_score():
    """Test attention score computation."""
    head_dim = 64
    
    q = np.ones(head_dim)
    k = np.ones(head_dim)
    
    score = compute_attention_score(q, k, scale=True)
    expected = np.dot(q, k) / np.sqrt(head_dim)  # 64 / 8 = 8
    
    assert abs(score - expected) < 1e-6
    
    print("✓ test_attention_score passed")


def test_svd_decomposition_property():
    """Test that the SVD decomposition makes sense."""
    vocab = 10
    embed_dim = 6
    head_dim = 3
    
    # Create a simple rank-deficient matrix
    A = np.random.randn(vocab, head_dim)
    B = np.random.randn(head_dim, vocab)
    M_dense = A @ B  # rank at most head_dim
    M = sparse.csr_matrix(M_dense)
    
    E_n = sparse.csr_matrix(np.eye(vocab, embed_dim))
    E_1 = np.eye(vocab, embed_dim)
    
    W_Q, W_K = compute_key_query_matrices(M, E_n, E_1, head_dim)
    
    # W_Q and W_K should capture the structure of M
    # This is a sanity check that SVD completed
    assert not np.allclose(W_Q, 0), "W_Q should not be all zeros"
    assert not np.allclose(W_K, 0), "W_K should not be all zeros"
    
    print("✓ test_svd_decomposition_property passed")


def test_empty_matrix():
    """Test handling of empty/zero matrix."""
    vocab = 10
    embed_dim = 8
    head_dim = 4
    
    # Empty sparse matrix
    M = sparse.csr_matrix((vocab, vocab), dtype=np.float64)
    E_n = sparse.csr_matrix(np.random.randn(vocab, embed_dim).astype(np.float32))
    E_1 = np.random.randn(vocab, embed_dim).astype(np.float32)
    
    W_Q, W_K = compute_key_query_matrices(M, E_n, E_1, head_dim)
    
    # Should return zero matrices without crashing
    assert W_Q.shape == (embed_dim, head_dim)
    assert W_K.shape == (embed_dim, head_dim)
    assert np.allclose(W_Q, 0)
    assert np.allclose(W_K, 0)
    
    print("✓ test_empty_matrix passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running key/query matrix tests...\n")
    
    test_key_query_shapes()
    test_key_query_n2()
    test_query_key_vectors()
    test_attention_score()
    test_svd_decomposition_property()
    test_empty_matrix()
    
    print("\n✓ All key/query tests passed!")


if __name__ == "__main__":
    run_all_tests()
    
    print("\n" + "=" * 50)
    print("Demo: Building W_Q and W_K from random data")
    print("=" * 50)
    
    np.random.seed(42)
    vocab = 100
    embed_dim = 32  # Smaller for demo
    head_dim = 8
    
    # Create mock manifold matrices
    manifold_matrices = {
        (1, 0): sparse.random(vocab, vocab, density=0.1, dtype=np.float64),
        (1, 1): sparse.random(vocab, vocab, density=0.1, dtype=np.float64),
        (2, 0): sparse.random(vocab**2, vocab, density=0.01, dtype=np.float64),
    }
    
    # Create mock embedding matrices
    embedding_matrices = {
        1: sparse.csr_matrix(np.random.randn(vocab, embed_dim).astype(np.float32)),
        2: sparse.csr_matrix(np.random.randn(vocab**2, embed_dim).astype(np.float32)),
    }
    
    base_embeddings = np.random.randn(vocab, embed_dim).astype(np.float32)
    
    print(f"\nVocab: {vocab}, Embed dim: {embed_dim}, Head dim: {head_dim}")
    print("-" * 50)
    
    results = build_all_key_query_matrices(
        manifold_matrices,
        embedding_matrices,
        base_embeddings,
        head_dim
    )
    
    print("\n" + "-" * 50)
    print("Example: Computing attention between two tokens")
    print("-" * 50)
    
    W_Q, W_K = results[(1, 0)]
    
    # Get embeddings for two tokens
    token_a, token_b = 0, 1
    e_a = base_embeddings[token_a]
    e_b = base_embeddings[token_b]
    
    # Compute query and key
    q_a = get_query_vector(W_Q, e_a)
    k_b = get_key_vector(W_K, e_b)
    
    # Attention score
    score = compute_attention_score(q_a, k_b)
    
    print(f"Token {token_a} attending to token {token_b}:")
    print(f"  Query shape: {q_a.shape}, Key shape: {k_b.shape}")
    print(f"  Attention score: {score:.4f}")


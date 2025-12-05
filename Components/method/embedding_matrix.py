"""
Embedding Matrix Builder

Constructs n-gram embedding matrices from probability matrices and GPT-2 embeddings.

For an n-gram (t1, t2, ..., tn):
    E^n_(t1,...,tn) = P^{n-1}(tn | t1,...,t_{n-1}) × E_tn

Where:
- P^{n-1} is the probability matrix for (n-1)-grams
- E_tn is GPT-2's pretrained embedding for the last token

For n=1: E^1 = GPT-2's base embeddings (no scaling)

GPT-2 small: vocab_size=50257, embedding_dim=768
"""

import numpy as np
from scipy import sparse
from typing import Optional
import torch

from count_matrix import VOCAB_SIZE, ngram_to_index, index_to_ngram


# GPT-2 small embedding dimension
EMBEDDING_DIM = 768


def load_gpt2_embeddings() -> np.ndarray:
    """
    Load GPT-2 small's pretrained token embeddings.
    
    Returns:
        numpy array of shape (vocab_size, embedding_dim) = (50257, 768)
    """
    from transformers import GPT2Model
    
    model = GPT2Model.from_pretrained("gpt2")
    embeddings = model.wte.weight.detach().numpy()
    
    return embeddings  # shape: (50257, 768)


def compute_embedding_matrix_n1(
    base_embeddings: np.ndarray
) -> np.ndarray:
    """
    For n=1, the embedding matrix is just the base GPT-2 embeddings.
    
    Args:
        base_embeddings: GPT-2 embeddings of shape (vocab_size, embedding_dim)
    
    Returns:
        Same embeddings (no transformation for n=1)
    """
    return base_embeddings.copy()


def compute_embedding_matrix(
    n: int,
    prob_matrix: sparse.csr_matrix,
    base_embeddings: np.ndarray,
    vocab_size: int = VOCAB_SIZE
) -> sparse.csr_matrix:
    """
    Compute n-gram embedding matrix E^n.
    
    E^n_(t1,...,tn) = P^{n-1}(tn | t1,...,t_{n-1}) × E_tn
    
    Args:
        n: N-gram size (must be >= 2)
        prob_matrix: P^{n-1} probability matrix of shape (vocab^{n-1}, vocab)
        base_embeddings: GPT-2 embeddings of shape (vocab_size, embedding_dim)
        vocab_size: Vocabulary size
    
    Returns:
        Sparse embedding matrix of shape (vocab^n, embedding_dim)
        Only n-grams with P > 0 have non-zero embeddings.
    """
    if n < 2:
        raise ValueError("Use compute_embedding_matrix_n1 for n=1")
    
    embedding_dim = base_embeddings.shape[1]
    num_ngrams = vocab_size ** n
    
    # We'll build the embedding matrix by iterating over non-zero entries in P
    # For each non-zero P[context_idx, last_token], we create an n-gram embedding
    
    # Get the non-zero entries from P
    P_coo = prob_matrix.tocoo()
    
    rows = []  # n-gram indices
    cols = []  # embedding dimensions (will be filled for each row)
    data = []  # scaled embedding values
    
    # For efficient construction, we'll build lists of (ngram_idx, embedding_vector)
    ngram_embeddings = {}
    
    for context_idx, last_token, prob in zip(P_coo.row, P_coo.col, P_coo.data):
        if prob <= 0:
            continue
        
        # Recover the context n-gram (length n-1)
        context = index_to_ngram(context_idx, n - 1, vocab_size)
        
        # Full n-gram is context + last_token
        ngram = context + (last_token,)
        ngram_idx = ngram_to_index(ngram, vocab_size)
        
        # Scaled embedding: P(last_token | context) * E_{last_token}
        scaled_embedding = prob * base_embeddings[last_token]
        
        ngram_embeddings[ngram_idx] = scaled_embedding
    
    # Convert to sparse matrix format
    if not ngram_embeddings:
        return sparse.csr_matrix((num_ngrams, embedding_dim), dtype=np.float32)
    
    # Build COO format
    for ngram_idx, embedding in ngram_embeddings.items():
        for dim, value in enumerate(embedding):
            if abs(value) > 1e-10:  # Only store non-trivial values
                rows.append(ngram_idx)
                cols.append(dim)
                data.append(value)
    
    E = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_ngrams, embedding_dim),
        dtype=np.float32
    )
    
    return E


def build_all_embedding_matrices(
    prob_matrices: dict[tuple[int, int], sparse.csr_matrix],
    base_embeddings: np.ndarray,
    max_n: int,
    s: int = 0,
    vocab_size: int = VOCAB_SIZE
) -> dict[int, sparse.csr_matrix]:
    """
    Build embedding matrices for n=1 to max_n.
    
    Note: E^n depends on P^{n-1}, so we use the s=0 probability matrices
    (or a specified skip value).
    
    Args:
        prob_matrices: Dict mapping (n, s) to probability matrix
        base_embeddings: GPT-2 embeddings
        max_n: Maximum n-gram size
        s: Skip value to use for probability matrices (default 0)
        vocab_size: Vocabulary size
    
    Returns:
        Dict mapping n to embedding matrix E^n
    """
    results = {}
    
    # E^1 is just base embeddings (dense)
    E1 = compute_embedding_matrix_n1(base_embeddings)
    # Convert to sparse for consistency (though it's dense)
    results[1] = sparse.csr_matrix(E1, dtype=np.float32)
    print(f"Built E^1: shape={E1.shape}, dense (base GPT-2 embeddings)")
    
    # E^n for n >= 2
    for n in range(2, max_n + 1):
        # Need P^{n-1}
        p_key = (n - 1, s)
        if p_key not in prob_matrices:
            print(f"Warning: P^{n-1} with s={s} not found, skipping E^{n}")
            continue
        
        P = prob_matrices[p_key]
        E = compute_embedding_matrix(n, P, base_embeddings, vocab_size)
        results[n] = E
        
        nnz_rows = len(set(E.tocoo().row)) if E.nnz > 0 else 0
        print(f"Built E^{n}: shape={E.shape}, "
              f"non-zero n-grams={nnz_rows:,}, "
              f"non-zero entries={E.nnz:,}")
    
    return results


def get_ngram_embedding(
    embedding_matrix: sparse.csr_matrix,
    ngram: tuple[int, ...],
    vocab_size: int = VOCAB_SIZE
) -> np.ndarray:
    """
    Get the embedding for a specific n-gram.
    
    Args:
        embedding_matrix: E^n matrix
        ngram: Tuple of token IDs
        vocab_size: Vocabulary size
    
    Returns:
        Embedding vector of shape (embedding_dim,)
    """
    idx = ngram_to_index(ngram, vocab_size)
    return embedding_matrix.getrow(idx).toarray().flatten()


def get_ngram_embedding_dense(
    n: int,
    ngram: tuple[int, ...],
    prob_matrices: dict[tuple[int, int], sparse.csr_matrix],
    base_embeddings: np.ndarray,
    s: int = 0,
    vocab_size: int = VOCAB_SIZE
) -> np.ndarray:
    """
    Compute embedding for a single n-gram on the fly (without building full matrix).
    
    Args:
        n: N-gram size
        ngram: Tuple of token IDs (length n)
        prob_matrices: Probability matrices dict
        base_embeddings: GPT-2 embeddings
        s: Skip value
        vocab_size: Vocabulary size
    
    Returns:
        Embedding vector
    """
    if len(ngram) != n:
        raise ValueError(f"N-gram length {len(ngram)} doesn't match n={n}")
    
    if n == 1:
        return base_embeddings[ngram[0]]
    
    # Get P^{n-1}(last_token | context)
    context = ngram[:-1]
    last_token = ngram[-1]
    
    P = prob_matrices[(n - 1, s)]
    context_idx = ngram_to_index(context, vocab_size)
    prob = P[context_idx, last_token]
    
    # E^n = P * E_{last_token}
    return prob * base_embeddings[last_token]


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_embedding_n1():
    """Test that E^1 is just the base embeddings."""
    # Create fake embeddings
    vocab = 10
    embed_dim = 4
    base = np.random.randn(vocab, embed_dim).astype(np.float32)
    
    E1 = compute_embedding_matrix_n1(base)
    
    assert E1.shape == base.shape
    assert np.allclose(E1, base)
    
    print("✓ test_embedding_n1 passed")


def test_embedding_n2():
    """Test E^2 computation."""
    vocab = 5
    embed_dim = 3
    
    # Create simple P^1 matrix
    P1_data = np.array([
        [0.5, 0.5, 0, 0, 0],   # token 0 -> 50% to 0, 50% to 1
        [0, 0, 1.0, 0, 0],     # token 1 -> 100% to 2
        [0, 0, 0, 0.3, 0.7],   # token 2 -> 30% to 3, 70% to 4
        [0, 0, 0, 0, 0],       # token 3 -> nothing
        [0, 0, 0, 0, 0],       # token 4 -> nothing
    ], dtype=np.float64)
    P1 = sparse.csr_matrix(P1_data)
    
    # Create base embeddings
    base = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
    ], dtype=np.float32)
    
    E2 = compute_embedding_matrix(n=2, prob_matrix=P1, base_embeddings=base, vocab_size=vocab)
    
    # Check E^2_(0,0) = P^1(0|0) * E_0 = 0.5 * [1,0,0] = [0.5, 0, 0]
    ngram_00 = ngram_to_index((0, 0), vocab)
    e_00 = E2.getrow(ngram_00).toarray().flatten()
    expected_00 = 0.5 * base[0]
    assert np.allclose(e_00, expected_00), f"E^2_(0,0) expected {expected_00}, got {e_00}"
    
    # Check E^2_(0,1) = P^1(1|0) * E_1 = 0.5 * [0,1,0] = [0, 0.5, 0]
    ngram_01 = ngram_to_index((0, 1), vocab)
    e_01 = E2.getrow(ngram_01).toarray().flatten()
    expected_01 = 0.5 * base[1]
    assert np.allclose(e_01, expected_01), f"E^2_(0,1) expected {expected_01}, got {e_01}"
    
    # Check E^2_(1,2) = P^1(2|1) * E_2 = 1.0 * [0,0,1] = [0, 0, 1]
    ngram_12 = ngram_to_index((1, 2), vocab)
    e_12 = E2.getrow(ngram_12).toarray().flatten()
    expected_12 = 1.0 * base[2]
    assert np.allclose(e_12, expected_12), f"E^2_(1,2) expected {expected_12}, got {e_12}"
    
    print("✓ test_embedding_n2 passed")


def test_embedding_zero_prob():
    """Test that zero-probability n-grams have zero embeddings."""
    vocab = 5
    embed_dim = 3
    
    # P^1 with some zeros
    P1_data = np.array([
        [1.0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],  # row 1 is all zeros
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float64)
    P1 = sparse.csr_matrix(P1_data)
    
    base = np.ones((vocab, embed_dim), dtype=np.float32)
    
    E2 = compute_embedding_matrix(n=2, prob_matrix=P1, base_embeddings=base, vocab_size=vocab)
    
    # E^2_(1, x) should be zero for all x since P^1(x|1) = 0
    for last_token in range(vocab):
        ngram_idx = ngram_to_index((1, last_token), vocab)
        e = E2.getrow(ngram_idx).toarray().flatten()
        assert np.allclose(e, 0), f"E^2_(1,{last_token}) should be zero"
    
    print("✓ test_embedding_zero_prob passed")


def test_get_ngram_embedding_dense():
    """Test on-the-fly embedding computation."""
    vocab = 5
    embed_dim = 3
    
    P1_data = np.array([
        [0.5, 0.5, 0, 0, 0],
        [0, 0, 1.0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float64)
    P1 = sparse.csr_matrix(P1_data)
    
    prob_matrices = {(1, 0): P1}
    base = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1],
    ], dtype=np.float32)
    
    # Test n=1
    e1 = get_ngram_embedding_dense(1, (2,), prob_matrices, base, vocab_size=vocab)
    assert np.allclose(e1, base[2])
    
    # Test n=2
    e2 = get_ngram_embedding_dense(2, (0, 1), prob_matrices, base, vocab_size=vocab)
    expected = 0.5 * base[1]
    assert np.allclose(e2, expected)
    
    print("✓ test_get_ngram_embedding_dense passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running embedding matrix tests...\n")
    
    test_embedding_n1()
    test_embedding_n2()
    test_embedding_zero_prob()
    test_get_ngram_embedding_dense()
    
    print("\n✓ All embedding tests passed!")


if __name__ == "__main__":
    run_all_tests()
    
    print("\n" + "=" * 50)
    print("Demo: Loading GPT-2 embeddings")
    print("=" * 50)
    
    try:
        print("Loading GPT-2 embeddings...")
        base_embeddings = load_gpt2_embeddings()
        print(f"Loaded: shape={base_embeddings.shape}, dtype={base_embeddings.dtype}")
        print(f"Embedding dim: {EMBEDDING_DIM}")
        print(f"Sample embedding norm: {np.linalg.norm(base_embeddings[0]):.4f}")
        
        # Show a few token embeddings
        from tokenizer import GPT2Tokenizer
        tokenizer = GPT2Tokenizer()
        
        print("\nSample embeddings:")
        for token_str in ["hello", "world", "the", " the"]:
            tokens = tokenizer.encode(token_str)
            if tokens:
                t = tokens[0]
                e = base_embeddings[t]
                print(f"  '{token_str}' (token {t}): norm={np.linalg.norm(e):.4f}, "
                      f"first 5 dims={e[:5].round(3)}")
    
    except Exception as e:
        print(f"Could not load GPT-2 (install transformers): {e}")
        print("Unit tests still passed with mock data.")


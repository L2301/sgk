"""
Probability Matrix Builder

Constructs probability matrices from count matrices via row normalization:

    P_ab = C_ab / Σ_b' C_ab'

This gives P(target=b | ngram=a) — the conditional probability of 
transitioning to token b given n-gram a.

Each row sums to 1 (forms a probability distribution over next tokens).

Parameters (same as count matrices):
- n: size of the n-gram
- s: skip number
"""

import numpy as np
from scipy import sparse

from count_matrix import (
    build_all_count_matrices,
    VOCAB_SIZE,
    ngram_to_index,
    index_to_ngram,
)


def compute_probability_matrix(
    count_matrix: sparse.csr_matrix,
    epsilon: float = 1e-10
) -> sparse.csr_matrix:
    """
    Convert a count matrix to a probability matrix via row normalization.
    
    P_ab = C_ab / Σ_b' C_ab'
    
    Args:
        count_matrix: Sparse count matrix C of shape (vocab^n, vocab)
        epsilon: Small value to avoid division by zero
    
    Returns:
        Sparse probability matrix P of same shape (rows sum to 1)
    """
    C = count_matrix.astype(np.float64)
    
    # Row sums: Σ_b' C_ab'
    row_sums = np.array(C.sum(axis=1)).flatten()  # shape: (vocab^n,)
    
    # Avoid division by zero
    row_sums = np.maximum(row_sums, epsilon)
    
    # Compute D_row^{-1}
    row_scale = 1.0 / row_sums  # shape: (vocab^n,)
    
    # Apply row scaling: P = D_row^{-1} @ C
    D_row_inv = sparse.diags(row_scale)
    P = D_row_inv @ C
    
    return P.tocsr()


def build_all_probability_matrices(
    count_matrices: dict[tuple[int, int], sparse.csr_matrix],
    epsilon: float = 1e-10
) -> dict[tuple[int, int], sparse.csr_matrix]:
    """
    Convert all count matrices to probability matrices.
    
    Args:
        count_matrices: Dict mapping (n, s) to count matrix
        epsilon: Small value to avoid division by zero
    
    Returns:
        Dict mapping (n, s) to probability matrix
    """
    results = {}
    
    for (n, s), C in count_matrices.items():
        P = compute_probability_matrix(C, epsilon)
        results[(n, s)] = P
        
        # Stats
        nnz = P.nnz
        if nnz > 0:
            data = P.data
            print(f"Built P(n={n}, s={s}): shape={P.shape}, "
                  f"non-zeros={nnz:,}, "
                  f"range=[{data.min():.4f}, {data.max():.4f}]")
        else:
            print(f"Built P(n={n}, s={s}): shape={P.shape}, non-zeros=0")
    
    return results


def build_probability_matrices_from_tokens(
    token_batches: list[list[int]],
    max_n: int,
    max_s: int,
    vocab_size: int = VOCAB_SIZE,
    epsilon: float = 1e-10
) -> tuple[dict[tuple[int, int], sparse.csr_matrix],
           dict[tuple[int, int], sparse.csr_matrix]]:
    """
    Build both count and probability matrices from token batches.
    
    Args:
        token_batches: List of token sequences
        max_n: Maximum n-gram size
        max_s: Maximum skip value
        vocab_size: Vocabulary size
        epsilon: Small value for numerical stability
    
    Returns:
        Tuple of (count_matrices, probability_matrices)
    """
    print("Building count matrices...")
    count_matrices = build_all_count_matrices(token_batches, max_n, max_s, vocab_size)
    
    print("\nBuilding probability matrices...")
    probability_matrices = build_all_probability_matrices(count_matrices, epsilon)
    
    return count_matrices, probability_matrices


def get_probability(
    matrix: sparse.csr_matrix,
    ngram: tuple[int, ...],
    target: int,
    vocab_size: int = VOCAB_SIZE
) -> float:
    """
    Get P(target | ngram).
    
    Args:
        matrix: Probability matrix
        ngram: Tuple of token IDs
        target: Target token ID
        vocab_size: Vocabulary size
    
    Returns:
        Probability P(target | ngram)
    """
    row = ngram_to_index(ngram, vocab_size)
    return matrix[row, target]


def get_distribution(
    matrix: sparse.csr_matrix,
    ngram: tuple[int, ...],
    vocab_size: int = VOCAB_SIZE
) -> np.ndarray:
    """
    Get the full probability distribution P(· | ngram).
    
    Args:
        matrix: Probability matrix
        ngram: Tuple of token IDs
        vocab_size: Vocabulary size
    
    Returns:
        Array of probabilities for all possible next tokens
    """
    row = ngram_to_index(ngram, vocab_size)
    return matrix.getrow(row).toarray().flatten()


def get_top_predictions(
    matrix: sparse.csr_matrix,
    ngram: tuple[int, ...],
    k: int = 10,
    vocab_size: int = VOCAB_SIZE
) -> list[tuple[int, float]]:
    """
    Get the top-k most probable next tokens.
    
    Args:
        matrix: Probability matrix
        ngram: Tuple of token IDs
        k: Number of top predictions
        vocab_size: Vocabulary size
    
    Returns:
        List of (token_id, probability) tuples, sorted by probability descending
    """
    dist = get_distribution(matrix, ngram, vocab_size)
    top_indices = np.argsort(dist)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        prob = dist[idx]
        if prob > 0:
            results.append((int(idx), float(prob)))
    
    return results


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_probability_basic():
    """Test basic probability matrix computation."""
    # Simple count matrix
    C = sparse.csr_matrix(np.array([
        [2, 2, 0],  # row sum = 4
        [1, 2, 1],  # row sum = 4
        [0, 0, 4],  # row sum = 4
    ], dtype=np.float64))
    
    P = compute_probability_matrix(C)
    
    # Row 0: [2/4, 2/4, 0] = [0.5, 0.5, 0]
    assert abs(P[0, 0] - 0.5) < 1e-6, f"P[0,0] should be 0.5, got {P[0,0]}"
    assert abs(P[0, 1] - 0.5) < 1e-6, f"P[0,1] should be 0.5, got {P[0,1]}"
    assert P[0, 2] == 0, f"P[0,2] should be 0, got {P[0,2]}"
    
    # Row 1: [1/4, 2/4, 1/4] = [0.25, 0.5, 0.25]
    assert abs(P[1, 0] - 0.25) < 1e-6
    assert abs(P[1, 1] - 0.5) < 1e-6
    assert abs(P[1, 2] - 0.25) < 1e-6
    
    # Row 2: [0, 0, 1]
    assert abs(P[2, 2] - 1.0) < 1e-6
    
    print("✓ test_probability_basic passed")


def test_rows_sum_to_one():
    """Test that non-zero rows sum to 1."""
    C = sparse.csr_matrix(np.array([
        [3, 1, 2],
        [5, 5, 0],
        [1, 1, 1],
    ], dtype=np.float64))
    
    P = compute_probability_matrix(C)
    
    # Check each row sums to 1
    row_sums = np.array(P.sum(axis=1)).flatten()
    for i, rs in enumerate(row_sums):
        assert abs(rs - 1.0) < 1e-6, f"Row {i} should sum to 1, got {rs}"
    
    print("✓ test_rows_sum_to_one passed")


def test_probability_sparse():
    """Test that sparsity is preserved."""
    row = [0, 0, 1, 2, 2, 2]
    col = [1, 2, 0, 0, 1, 2]
    data = [3, 2, 5, 1, 2, 3]
    C = sparse.csr_matrix((data, (row, col)), shape=(10, 10), dtype=np.float64)
    
    P = compute_probability_matrix(C)
    
    # Should have same sparsity pattern
    assert P.nnz == C.nnz, f"Non-zeros should match: {P.nnz} vs {C.nnz}"
    
    # All probabilities should be in [0, 1]
    assert np.all(P.data >= 0), "All probabilities should be >= 0"
    assert np.all(P.data <= 1), "All probabilities should be <= 1"
    
    print("✓ test_probability_sparse passed")


def test_zero_row_handling():
    """Test that zero rows don't cause NaN."""
    C = sparse.csr_matrix(np.array([
        [1, 2, 0],
        [0, 0, 0],  # zero row
        [0, 0, 3],
    ], dtype=np.float64))
    
    P = compute_probability_matrix(C)
    
    # Should not have NaN or Inf
    assert not np.any(np.isnan(P.data)), "Should not have NaN"
    assert not np.any(np.isinf(P.data)), "Should not have Inf"
    
    # Zero row should remain zero (or very small due to epsilon)
    row1_sum = P.getrow(1).sum()
    assert row1_sum < 1e-6, f"Zero row should stay ~0, got {row1_sum}"
    
    print("✓ test_zero_row_handling passed")


def test_probability_from_counts():
    """Test building probability matrices from count matrices dict."""
    from count_matrix import build_all_count_matrices
    
    vocab = 10
    batches = [[0, 1, 2, 3, 4, 5, 0, 1, 2]]
    
    count_matrices = build_all_count_matrices(batches, max_n=2, max_s=1, vocab_size=vocab)
    prob_matrices = build_all_probability_matrices(count_matrices)
    
    # Should have same keys
    assert set(count_matrices.keys()) == set(prob_matrices.keys())
    
    # Check shapes match
    for key in count_matrices:
        assert count_matrices[key].shape == prob_matrices[key].shape
    
    print("✓ test_probability_from_counts passed")


def test_get_top_predictions():
    """Test getting top predictions."""
    C = sparse.csr_matrix(np.array([
        [10, 5, 3, 1, 1],
    ], dtype=np.float64))
    
    P = compute_probability_matrix(C)
    
    top = get_top_predictions(P, (0,), k=3, vocab_size=5)
    
    # Should be sorted by probability descending
    assert top[0][0] == 0, "Top prediction should be token 0"
    assert top[1][0] == 1, "Second prediction should be token 1"
    assert top[2][0] == 2, "Third prediction should be token 2"
    
    # Check probabilities
    assert abs(top[0][1] - 0.5) < 1e-6  # 10/20
    assert abs(top[1][1] - 0.25) < 1e-6  # 5/20
    assert abs(top[2][1] - 0.15) < 1e-6  # 3/20
    
    print("✓ test_get_top_predictions passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running probability matrix tests...\n")
    
    test_probability_basic()
    test_rows_sum_to_one()
    test_probability_sparse()
    test_zero_row_handling()
    test_probability_from_counts()
    test_get_top_predictions()
    
    print("\n✓ All probability tests passed!")


if __name__ == "__main__":
    run_all_tests()
    
    print("\n" + "=" * 50)
    print("Demo: Building probability matrices from token data")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    vocab = 100
    test_batches = [
        list(np.random.randint(0, vocab, size=500)) for _ in range(3)
    ]
    
    count_matrices, prob_matrices = build_probability_matrices_from_tokens(
        test_batches, max_n=2, max_s=2, vocab_size=vocab
    )
    
    print("\n" + "-" * 50)
    print("Verification: Row sums should be ~1.0")
    print("-" * 50)
    
    for (n, s) in sorted(prob_matrices.keys()):
        P = prob_matrices[(n, s)]
        
        # Get row sums for non-empty rows
        row_sums = np.array(P.sum(axis=1)).flatten()
        non_zero_rows = row_sums[row_sums > 0.5]  # rows with actual data
        
        if len(non_zero_rows) > 0:
            print(f"P(n={n}, s={s}): {len(non_zero_rows)} active rows, "
                  f"row_sum range=[{non_zero_rows.min():.6f}, {non_zero_rows.max():.6f}]")


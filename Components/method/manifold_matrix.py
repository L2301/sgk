"""
Manifold Matrix Builder

Constructs manifold matrices from count matrices via symmetric normalization:

    M_ab = C_ab / sqrt(D_aa * D_bb)

Where:
- D_aa = row sum of row a (total transitions from n-gram a)
- D_bb = column sum of column b (total transitions to token b)

This is equivalent to: M = D_row^{-1/2} @ C @ D_col^{-1/2}

Parameters (same as count matrices):
- n: size of the n-gram
- s: skip number
"""

import numpy as np
from scipy import sparse
from typing import Optional

from count_matrix import (
    build_all_count_matrices,
    VOCAB_SIZE,
    ngram_to_index,
    index_to_ngram,
)


def compute_manifold_matrix(
    count_matrix: sparse.csr_matrix,
    epsilon: float = 1e-10
) -> sparse.csr_matrix:
    """
    Convert a count matrix to a manifold matrix.
    
    M_ab = C_ab / sqrt(D_aa * D_bb)
    
    Args:
        count_matrix: Sparse count matrix C of shape (vocab^n, vocab)
        epsilon: Small value to avoid division by zero
    
    Returns:
        Sparse manifold matrix M of same shape
    """
    C = count_matrix.astype(np.float64)
    
    # D_aa = row sums (how often each n-gram transitions to anything)
    row_sums = np.array(C.sum(axis=1)).flatten()  # shape: (vocab^n,)
    
    # D_bb = column sums (how often each token is transitioned to)
    col_sums = np.array(C.sum(axis=0)).flatten()  # shape: (vocab,)
    
    # Avoid division by zero
    row_sums = np.maximum(row_sums, epsilon)
    col_sums = np.maximum(col_sums, epsilon)
    
    # Compute D_row^{-1/2} and D_col^{-1/2}
    row_scale = 1.0 / np.sqrt(row_sums)  # shape: (vocab^n,)
    col_scale = 1.0 / np.sqrt(col_sums)  # shape: (vocab,)
    
    # Apply scaling: M = D_row^{-1/2} @ C @ D_col^{-1/2}
    # For sparse matrix, we scale rows then columns
    
    # Scale rows: multiply each row i by row_scale[i]
    # Create diagonal matrix for row scaling
    D_row_inv_sqrt = sparse.diags(row_scale)
    
    # Scale columns: multiply each column j by col_scale[j]
    D_col_inv_sqrt = sparse.diags(col_scale)
    
    # M = D_row^{-1/2} @ C @ D_col^{-1/2}
    M = D_row_inv_sqrt @ C @ D_col_inv_sqrt
    
    return M.tocsr()


def build_all_manifold_matrices(
    count_matrices: dict[tuple[int, int], sparse.csr_matrix],
    epsilon: float = 1e-10
) -> dict[tuple[int, int], sparse.csr_matrix]:
    """
    Convert all count matrices to manifold matrices.
    
    Args:
        count_matrices: Dict mapping (n, s) to count matrix
        epsilon: Small value to avoid division by zero
    
    Returns:
        Dict mapping (n, s) to manifold matrix
    """
    results = {}
    
    for (n, s), C in count_matrices.items():
        M = compute_manifold_matrix(C, epsilon)
        results[(n, s)] = M
        
        # Stats
        nnz = M.nnz
        if nnz > 0:
            data = M.data
            print(f"Built M(n={n}, s={s}): shape={M.shape}, "
                  f"non-zeros={nnz:,}, "
                  f"range=[{data.min():.4f}, {data.max():.4f}]")
        else:
            print(f"Built M(n={n}, s={s}): shape={M.shape}, non-zeros=0")
    
    return results


def build_manifold_matrices_from_tokens(
    token_batches: list[list[int]],
    max_n: int,
    max_s: int,
    vocab_size: int = VOCAB_SIZE,
    epsilon: float = 1e-10
) -> tuple[dict[tuple[int, int], sparse.csr_matrix], 
           dict[tuple[int, int], sparse.csr_matrix]]:
    """
    Build both count and manifold matrices from token batches.
    
    Args:
        token_batches: List of token sequences
        max_n: Maximum n-gram size
        max_s: Maximum skip value
        vocab_size: Vocabulary size
        epsilon: Small value for numerical stability
    
    Returns:
        Tuple of (count_matrices, manifold_matrices)
    """
    print("Building count matrices...")
    count_matrices = build_all_count_matrices(token_batches, max_n, max_s, vocab_size)
    
    print("\nBuilding manifold matrices...")
    manifold_matrices = build_all_manifold_matrices(count_matrices, epsilon)
    
    return count_matrices, manifold_matrices


def get_manifold_value(
    matrix: sparse.csr_matrix,
    ngram: tuple[int, ...],
    target: int,
    vocab_size: int = VOCAB_SIZE
) -> float:
    """
    Get the manifold value for a specific n-gram → target.
    
    Args:
        matrix: Manifold matrix
        ngram: Tuple of token IDs
        target: Target token ID
        vocab_size: Vocabulary size
    
    Returns:
        Manifold value M_ab
    """
    row = ngram_to_index(ngram, vocab_size)
    return matrix[row, target]


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_manifold_basic():
    """Test basic manifold matrix computation."""
    # Simple 3x3 count matrix
    C = sparse.csr_matrix(np.array([
        [4, 0, 0],
        [0, 2, 2],
        [0, 2, 2],
    ], dtype=np.float64))
    
    M = compute_manifold_matrix(C)
    
    # Row sums: [4, 4, 4]
    # Col sums: [4, 4, 4]
    # M_00 = 4 / sqrt(4 * 4) = 4 / 4 = 1.0
    # M_11 = 2 / sqrt(4 * 4) = 2 / 4 = 0.5
    
    assert abs(M[0, 0] - 1.0) < 1e-6, f"M[0,0] should be 1.0, got {M[0,0]}"
    assert abs(M[1, 1] - 0.5) < 1e-6, f"M[1,1] should be 0.5, got {M[1,1]}"
    assert abs(M[1, 2] - 0.5) < 1e-6, f"M[1,2] should be 0.5, got {M[1,2]}"
    
    print("✓ test_manifold_basic passed")


def test_manifold_asymmetric():
    """Test with asymmetric row/column sums."""
    # Count matrix where row and column sums differ
    C = sparse.csr_matrix(np.array([
        [6, 2],  # row sum = 8
        [0, 4],  # row sum = 4
    ], dtype=np.float64))
    # col sums: [6, 6]
    
    M = compute_manifold_matrix(C)
    
    # M_00 = 6 / sqrt(8 * 6) = 6 / sqrt(48) ≈ 0.866
    # M_01 = 2 / sqrt(8 * 6) = 2 / sqrt(48) ≈ 0.289
    # M_11 = 4 / sqrt(4 * 6) = 4 / sqrt(24) ≈ 0.816
    
    expected_00 = 6 / np.sqrt(8 * 6)
    expected_01 = 2 / np.sqrt(8 * 6)
    expected_11 = 4 / np.sqrt(4 * 6)
    
    assert abs(M[0, 0] - expected_00) < 1e-6, f"M[0,0] expected {expected_00}, got {M[0,0]}"
    assert abs(M[0, 1] - expected_01) < 1e-6, f"M[0,1] expected {expected_01}, got {M[0,1]}"
    assert abs(M[1, 1] - expected_11) < 1e-6, f"M[1,1] expected {expected_11}, got {M[1,1]}"
    
    print("✓ test_manifold_asymmetric passed")


def test_manifold_sparse():
    """Test that sparsity is preserved."""
    # Sparse count matrix
    row = [0, 1, 2, 3]
    col = [1, 2, 3, 0]
    data = [5, 3, 7, 2]
    C = sparse.csr_matrix((data, (row, col)), shape=(10, 10), dtype=np.float64)
    
    M = compute_manifold_matrix(C)
    
    # Should have same sparsity pattern
    assert M.nnz == C.nnz, f"Non-zeros should match: {M.nnz} vs {C.nnz}"
    
    # All manifold values should be positive (since counts are positive)
    assert np.all(M.data > 0), "All manifold values should be positive"
    
    print("✓ test_manifold_sparse passed")


def test_manifold_from_counts():
    """Test building manifold matrices from count matrices dict."""
    from count_matrix import build_all_count_matrices
    
    vocab = 10
    batches = [[0, 1, 2, 3, 4, 5, 0, 1, 2]]
    
    count_matrices = build_all_count_matrices(batches, max_n=2, max_s=1, vocab_size=vocab)
    manifold_matrices = build_all_manifold_matrices(count_matrices)
    
    # Should have same keys
    assert set(count_matrices.keys()) == set(manifold_matrices.keys())
    
    # Check shapes match
    for key in count_matrices:
        assert count_matrices[key].shape == manifold_matrices[key].shape
    
    print("✓ test_manifold_from_counts passed")


def test_zero_handling():
    """Test that zero rows/columns don't cause NaN."""
    # Matrix with a zero row and zero column
    C = sparse.csr_matrix(np.array([
        [1, 0, 2],
        [0, 0, 0],  # zero row
        [3, 0, 1],
    ], dtype=np.float64))
    # col 1 is zero column
    
    M = compute_manifold_matrix(C)
    
    # Should not have NaN or Inf
    assert not np.any(np.isnan(M.data)), "Should not have NaN"
    assert not np.any(np.isinf(M.data)), "Should not have Inf"
    
    print("✓ test_zero_handling passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running manifold matrix tests...\n")
    
    test_manifold_basic()
    test_manifold_asymmetric()
    test_manifold_sparse()
    test_manifold_from_counts()
    test_zero_handling()
    
    print("\n✓ All manifold tests passed!")


if __name__ == "__main__":
    run_all_tests()
    
    print("\n" + "=" * 50)
    print("Demo: Building manifold matrices from token data")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    vocab = 100
    test_batches = [
        list(np.random.randint(0, vocab, size=500)) for _ in range(3)
    ]
    
    count_matrices, manifold_matrices = build_manifold_matrices_from_tokens(
        test_batches, max_n=2, max_s=2, vocab_size=vocab
    )
    
    print("\n" + "-" * 50)
    print("Comparison: Count vs Manifold")
    print("-" * 50)
    
    for (n, s) in sorted(count_matrices.keys()):
        C = count_matrices[(n, s)]
        M = manifold_matrices[(n, s)]
        
        c_sum = C.sum()
        m_sum = M.sum()
        
        print(f"(n={n}, s={s}): C_sum={c_sum:,.0f}, M_sum={m_sum:.2f}, "
              f"M_max={M.max():.4f}")


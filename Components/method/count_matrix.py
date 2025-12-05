"""
Count Matrix Builder

Builds sparse count matrices tracking n-gram → token transitions.
Matrix C[ngram_idx, token] = count of times ngram was followed by token.

Parameters:
- n: size of the n-gram (n=1 is unigram→token, n=2 is bigram→token, etc.)
- s: skip number (s=0 is adjacent, s=1 skips one token between n-gram and target)

Example with "the big red fox" (tokens: [t0, t1, t2, t3]):
- n=1, s=0: t0→t1, t1→t2, t2→t3
- n=1, s=1: t0→t2, t1→t3
- n=2, s=0: (t0,t1)→t2, (t1,t2)→t3
- n=2, s=1: (t0,t1)→t3
"""

import numpy as np
from scipy import sparse
from typing import Iterator
from collections import defaultdict


# GPT-2 vocabulary size
VOCAB_SIZE = 50257


def ngram_to_index(ngram: tuple[int, ...], vocab_size: int = VOCAB_SIZE) -> int:
    """
    Convert an n-gram tuple to a flat row index.
    
    For n-gram (t0, t1, ..., t_{n-1}):
    index = t0 * vocab^{n-1} + t1 * vocab^{n-2} + ... + t_{n-1}
    
    Args:
        ngram: Tuple of token IDs
        vocab_size: Size of vocabulary
    
    Returns:
        Flat index for the n-gram
    """
    n = len(ngram)
    index = 0
    for i, token in enumerate(ngram):
        index += token * (vocab_size ** (n - 1 - i))
    return index


def index_to_ngram(index: int, n: int, vocab_size: int = VOCAB_SIZE) -> tuple[int, ...]:
    """
    Convert a flat row index back to an n-gram tuple.
    
    Args:
        index: Flat index
        n: Size of n-gram
        vocab_size: Size of vocabulary
    
    Returns:
        Tuple of token IDs
    """
    ngram = []
    for i in range(n):
        divisor = vocab_size ** (n - 1 - i)
        token = index // divisor
        index = index % divisor
        ngram.append(token)
    return tuple(ngram)


def count_transitions_sparse(
    tokens: list[int],
    n: int,
    s: int,
    vocab_size: int = VOCAB_SIZE
) -> dict[tuple[int, int], int]:
    """
    Count n-gram → token transitions in a token sequence.
    Returns sparse representation as dict of {(row, col): count}.
    
    Args:
        tokens: List of token IDs
        n: N-gram size
        s: Skip value
        vocab_size: Vocabulary size
    
    Returns:
        Dictionary mapping (ngram_index, target_token) to count
    """
    counts = defaultdict(int)
    seq_len = len(tokens)
    
    # For each valid starting position
    # n-gram spans [i, i+n-1], target is at i+n+s
    for i in range(seq_len - n - s):
        ngram = tuple(tokens[i:i + n])
        target = tokens[i + n + s]
        
        ngram_idx = ngram_to_index(ngram, vocab_size)
        counts[(ngram_idx, target)] += 1
    
    return dict(counts)


def build_count_matrix(
    token_batches: Iterator[list[int]],
    n: int,
    s: int,
    vocab_size: int = VOCAB_SIZE
) -> sparse.csr_matrix:
    """
    Build a single count matrix from token batches.
    
    Args:
        token_batches: Iterator yielding lists of token IDs
        n: N-gram size
        s: Skip value
        vocab_size: Vocabulary size
    
    Returns:
        Sparse CSR matrix of shape (vocab^n, vocab)
    """
    # Accumulate all counts
    total_counts = defaultdict(int)
    
    for tokens in token_batches:
        batch_counts = count_transitions_sparse(tokens, n, s, vocab_size)
        for key, count in batch_counts.items():
            total_counts[key] += count
    
    # Build sparse matrix
    num_rows = vocab_size ** n
    num_cols = vocab_size
    
    if not total_counts:
        return sparse.csr_matrix((num_rows, num_cols), dtype=np.int64)
    
    rows, cols, data = [], [], []
    for (row, col), count in total_counts.items():
        rows.append(row)
        cols.append(col)
        data.append(count)
    
    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(num_rows, num_cols),
        dtype=np.int64
    )
    
    return matrix


def build_all_count_matrices(
    token_batches: list[list[int]],
    max_n: int,
    max_s: int,
    vocab_size: int = VOCAB_SIZE
) -> dict[tuple[int, int], sparse.csr_matrix]:
    """
    Build count matrices for all combinations of n and s.
    
    Args:
        token_batches: List of token sequences (batches)
        max_n: Maximum n-gram size (inclusive)
        max_s: Maximum skip value (inclusive)
        vocab_size: Vocabulary size
    
    Returns:
        Dictionary mapping (n, s) to sparse count matrix
        Matrix shape for each n: (vocab^n, vocab)
    """
    results = {}
    
    for n in range(1, max_n + 1):
        for s in range(0, max_s + 1):
            # Need to re-iterate over batches for each (n, s)
            matrix = build_count_matrix(iter(token_batches), n, s, vocab_size)
            results[(n, s)] = matrix
            
            # Progress info
            nnz = matrix.nnz
            print(f"Built C(n={n}, s={s}): shape={matrix.shape}, non-zeros={nnz:,}")
    
    return results


def get_count(
    matrix: sparse.csr_matrix,
    ngram: tuple[int, ...],
    target: int,
    vocab_size: int = VOCAB_SIZE
) -> int:
    """
    Get the count for a specific n-gram → target transition.
    
    Args:
        matrix: Count matrix
        ngram: Tuple of token IDs
        target: Target token ID
        vocab_size: Vocabulary size
    
    Returns:
        Count of transitions
    """
    row = ngram_to_index(ngram, vocab_size)
    return matrix[row, target]


def get_top_continuations(
    matrix: sparse.csr_matrix,
    ngram: tuple[int, ...],
    k: int = 10,
    vocab_size: int = VOCAB_SIZE
) -> list[tuple[int, int]]:
    """
    Get the top-k most frequent continuations for an n-gram.
    
    Args:
        matrix: Count matrix
        ngram: Tuple of token IDs
        k: Number of top continuations to return
        vocab_size: Vocabulary size
    
    Returns:
        List of (token_id, count) tuples, sorted by count descending
    """
    row = ngram_to_index(ngram, vocab_size)
    row_data = matrix.getrow(row).toarray().flatten()
    
    # Get top k indices
    top_indices = np.argsort(row_data)[-k:][::-1]
    
    results = []
    for idx in top_indices:
        count = row_data[idx]
        if count > 0:
            results.append((int(idx), int(count)))
    
    return results


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_ngram_indexing():
    """Test n-gram to index conversion and back."""
    vocab = 100  # Small vocab for testing
    
    # Test n=1
    for t in [0, 50, 99]:
        idx = ngram_to_index((t,), vocab)
        recovered = index_to_ngram(idx, 1, vocab)
        assert recovered == (t,), f"n=1 failed: {t} -> {idx} -> {recovered}"
    
    # Test n=2
    for t1 in [0, 50, 99]:
        for t2 in [0, 25, 99]:
            ngram = (t1, t2)
            idx = ngram_to_index(ngram, vocab)
            recovered = index_to_ngram(idx, 2, vocab)
            assert recovered == ngram, f"n=2 failed: {ngram} -> {idx} -> {recovered}"
    
    # Test n=3
    ngram = (10, 20, 30)
    idx = ngram_to_index(ngram, vocab)
    recovered = index_to_ngram(idx, 3, vocab)
    assert recovered == ngram, f"n=3 failed: {ngram} -> {idx} -> {recovered}"
    
    print("✓ test_ngram_indexing passed")


def test_count_transitions_n1_s0():
    """Test basic unigram counting with no skip."""
    # Sequence: [0, 1, 2, 1, 2, 3]
    # Transitions: 0→1, 1→2, 2→1, 1→2, 2→3
    tokens = [0, 1, 2, 1, 2, 3]
    vocab = 10
    
    counts = count_transitions_sparse(tokens, n=1, s=0, vocab_size=vocab)
    
    assert counts[(0, 1)] == 1, "0→1 should be 1"
    assert counts[(1, 2)] == 2, "1→2 should be 2"
    assert counts[(2, 1)] == 1, "2→1 should be 1"
    assert counts[(2, 3)] == 1, "2→3 should be 1"
    assert (3, 0) not in counts, "3→anything shouldn't exist"
    
    print("✓ test_count_transitions_n1_s0 passed")


def test_count_transitions_n1_s1():
    """Test unigram counting with skip=1."""
    # Sequence: [0, 1, 2, 3]
    # Skip-1 transitions: 0→2, 1→3
    tokens = [0, 1, 2, 3]
    vocab = 10
    
    counts = count_transitions_sparse(tokens, n=1, s=1, vocab_size=vocab)
    
    assert counts[(0, 2)] == 1, "0→2 (skip 1) should be 1"
    assert counts[(1, 3)] == 1, "1→3 (skip 1) should be 1"
    assert len(counts) == 2, f"Should have exactly 2 transitions, got {len(counts)}"
    
    print("✓ test_count_transitions_n1_s1 passed")


def test_count_transitions_n2_s0():
    """Test bigram counting with no skip."""
    # Sequence: [0, 1, 2, 3]
    # Bigram→next: (0,1)→2, (1,2)→3
    tokens = [0, 1, 2, 3]
    vocab = 10
    
    counts = count_transitions_sparse(tokens, n=2, s=0, vocab_size=vocab)
    
    idx_01 = ngram_to_index((0, 1), vocab)
    idx_12 = ngram_to_index((1, 2), vocab)
    
    assert counts[(idx_01, 2)] == 1, "(0,1)→2 should be 1"
    assert counts[(idx_12, 3)] == 1, "(1,2)→3 should be 1"
    assert len(counts) == 2, f"Should have exactly 2 transitions, got {len(counts)}"
    
    print("✓ test_count_transitions_n2_s0 passed")


def test_count_transitions_n2_s1():
    """Test bigram counting with skip=1."""
    # Sequence: [0, 1, 2, 3, 4]
    # Bigram skip-1: (0,1)→3, (1,2)→4
    tokens = [0, 1, 2, 3, 4]
    vocab = 10
    
    counts = count_transitions_sparse(tokens, n=2, s=1, vocab_size=vocab)
    
    idx_01 = ngram_to_index((0, 1), vocab)
    idx_12 = ngram_to_index((1, 2), vocab)
    
    assert counts[(idx_01, 3)] == 1, "(0,1)→3 (skip 1) should be 1"
    assert counts[(idx_12, 4)] == 1, "(1,2)→4 (skip 1) should be 1"
    assert len(counts) == 2, f"Should have exactly 2 transitions, got {len(counts)}"
    
    print("✓ test_count_transitions_n2_s1 passed")


def test_build_count_matrix():
    """Test building sparse matrix from multiple batches."""
    vocab = 10
    
    # Two batches
    batch1 = [0, 1, 2, 3]  # 0→1, 1→2, 2→3
    batch2 = [1, 2, 3, 4]  # 1→2, 2→3, 3→4
    
    matrix = build_count_matrix(iter([batch1, batch2]), n=1, s=0, vocab_size=vocab)
    
    assert matrix.shape == (vocab, vocab), f"Shape should be ({vocab}, {vocab})"
    assert matrix[0, 1] == 1, "0→1 should be 1"
    assert matrix[1, 2] == 2, "1→2 should be 2 (appears in both batches)"
    assert matrix[2, 3] == 2, "2→3 should be 2 (appears in both batches)"
    assert matrix[3, 4] == 1, "3→4 should be 1"
    
    print("✓ test_build_count_matrix passed")


def test_build_all_count_matrices():
    """Test building all matrices for max_n and max_s."""
    vocab = 10
    batches = [[0, 1, 2, 3, 4, 5]]
    
    results = build_all_count_matrices(batches, max_n=2, max_s=1, vocab_size=vocab)
    
    # Should have 4 matrices: (1,0), (1,1), (2,0), (2,1)
    assert len(results) == 4, f"Should have 4 matrices, got {len(results)}"
    assert (1, 0) in results
    assert (1, 1) in results
    assert (2, 0) in results
    assert (2, 1) in results
    
    # Check shapes
    assert results[(1, 0)].shape == (vocab, vocab)
    assert results[(1, 1)].shape == (vocab, vocab)
    assert results[(2, 0)].shape == (vocab ** 2, vocab)
    assert results[(2, 1)].shape == (vocab ** 2, vocab)
    
    print("✓ test_build_all_count_matrices passed")


def test_the_big_red_fox():
    """Test with the example from the docstring."""
    # "the big red fox" as tokens [0, 1, 2, 3]
    tokens = [0, 1, 2, 3]  # the=0, big=1, red=2, fox=3
    vocab = 10
    
    # n=1, s=0: 0→1, 1→2, 2→3
    c = count_transitions_sparse(tokens, n=1, s=0, vocab_size=vocab)
    assert c[(0, 1)] == 1 and c[(1, 2)] == 1 and c[(2, 3)] == 1
    assert len(c) == 3
    
    # n=1, s=1: 0→2, 1→3
    c = count_transitions_sparse(tokens, n=1, s=1, vocab_size=vocab)
    assert c[(0, 2)] == 1 and c[(1, 3)] == 1
    assert len(c) == 2
    
    # n=2, s=0: (0,1)→2, (1,2)→3
    c = count_transitions_sparse(tokens, n=2, s=0, vocab_size=vocab)
    idx_01 = ngram_to_index((0, 1), vocab)
    idx_12 = ngram_to_index((1, 2), vocab)
    assert c[(idx_01, 2)] == 1 and c[(idx_12, 3)] == 1
    assert len(c) == 2
    
    # n=2, s=1: (0,1)→3
    c = count_transitions_sparse(tokens, n=2, s=1, vocab_size=vocab)
    assert c[(idx_01, 3)] == 1
    assert len(c) == 1
    
    print("✓ test_the_big_red_fox passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running unit tests...\n")
    
    test_ngram_indexing()
    test_count_transitions_n1_s0()
    test_count_transitions_n1_s1()
    test_count_transitions_n2_s0()
    test_count_transitions_n2_s1()
    test_build_count_matrix()
    test_build_all_count_matrices()
    test_the_big_red_fox()
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
    
    print("\n" + "=" * 50)
    print("Demo with larger test data")
    print("=" * 50)
    
    # Create test batches with repeated patterns
    np.random.seed(42)
    vocab = 100  # Small vocab for demo
    
    # Generate some test data with patterns
    test_batches = [
        list(np.random.randint(0, vocab, size=200)) for _ in range(5)
    ]
    
    # Build all matrices
    matrices = build_all_count_matrices(test_batches, max_n=2, max_s=2, vocab_size=vocab)
    
    print("\nMatrix summary:")
    for (n, s), mat in sorted(matrices.items()):
        total_counts = mat.sum()
        print(f"  C(n={n}, s={s}): {mat.nnz:,} non-zeros, {total_counts:,} total counts")


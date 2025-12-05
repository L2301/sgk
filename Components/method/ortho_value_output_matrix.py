"""
Orthogonalized Value and Output Matrices

Takes the output of value_output_matrix.py and orthogonalizes the subspaces such that:
1. Same s, different n → share the same subspace
2. Different s → orthogonal subspaces

Approach:
1. For each s: combine W_O matrices across all n to find shared column space
2. SVD on combined matrix to extract the shared subspace basis
3. Orthogonalize bases across different s values using Gram-Schmidt
4. Project original matrices onto orthogonalized subspaces
"""

import numpy as np
from scipy import sparse
from typing import Optional

from count_matrix import VOCAB_SIZE
from embedding_matrix import EMBEDDING_DIM
from value_output_matrix import HEAD_DIM


def extract_subspace_basis(
    W_O_matrices: list[np.ndarray],
    head_dim: int = HEAD_DIM
) -> np.ndarray:
    """
    Extract shared subspace basis from multiple W_O matrices.
    
    Concatenates W_O matrices horizontally and finds dominant subspace via SVD.
    
    Args:
        W_O_matrices: List of W_O matrices, each (embedding_dim, head_dim)
        head_dim: Target dimension for the subspace
    
    Returns:
        Orthonormal basis V of shape (embedding_dim, head_dim)
    """
    if not W_O_matrices:
        raise ValueError("No matrices provided")
    
    embedding_dim = W_O_matrices[0].shape[0]
    
    # Concatenate horizontally: (embedding_dim, total_heads)
    W_O_combined = np.concatenate(W_O_matrices, axis=1)
    
    # SVD to find dominant subspace
    U, s, Vt = np.linalg.svd(W_O_combined, full_matrices=False)
    
    # Take top head_dim singular vectors
    r = min(head_dim, U.shape[1])
    V_basis = U[:, :r]  # (embedding_dim, r)
    
    # Pad if needed
    if r < head_dim:
        V_padded = np.zeros((embedding_dim, head_dim), dtype=np.float32)
        V_padded[:, :r] = V_basis
        return V_padded
    
    return V_basis.astype(np.float32)


def gram_schmidt_subspaces(
    bases: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Orthogonalize multiple subspace bases using Gram-Schmidt.
    
    Each basis spans a subspace. After orthogonalization:
    - Columns within each basis remain orthonormal
    - Columns across different bases are orthogonal
    
    Args:
        bases: List of orthonormal bases, each (embedding_dim, head_dim)
    
    Returns:
        List of orthogonalized bases with same shapes
    """
    if not bases:
        return []
    
    embedding_dim = bases[0].shape[0]
    head_dim = bases[0].shape[1]
    
    orthogonalized = []
    
    # Collect all vectors we've used so far
    used_space = np.zeros((embedding_dim, 0), dtype=np.float64)
    
    for basis in bases:
        basis = basis.astype(np.float64)
        new_basis = np.zeros_like(basis)
        
        for j in range(head_dim):
            v = basis[:, j].copy()
            
            # Subtract projections onto all previously used vectors
            if used_space.shape[1] > 0:
                # Project v onto used_space and subtract
                coeffs = used_space.T @ v
                v = v - used_space @ coeffs
            
            # Normalize
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
                new_basis[:, j] = v
                # Add to used space
                used_space = np.concatenate([used_space, v.reshape(-1, 1)], axis=1)
            else:
                # Vector is in the span of previous vectors, find orthogonal complement
                # Try random vector and orthogonalize
                for _ in range(100):
                    v_rand = np.random.randn(embedding_dim)
                    if used_space.shape[1] > 0:
                        coeffs = used_space.T @ v_rand
                        v_rand = v_rand - used_space @ coeffs
                    norm = np.linalg.norm(v_rand)
                    if norm > 1e-10:
                        v_rand = v_rand / norm
                        new_basis[:, j] = v_rand
                        used_space = np.concatenate([used_space, v_rand.reshape(-1, 1)], axis=1)
                        break
        
        orthogonalized.append(new_basis.astype(np.float32))
    
    return orthogonalized


def project_to_subspace(
    W_O: np.ndarray,
    basis: np.ndarray
) -> np.ndarray:
    """
    Project W_O matrix onto a subspace defined by basis.
    
    Args:
        W_O: Original matrix (embedding_dim, head_dim)
        basis: Orthonormal basis (embedding_dim, head_dim)
    
    Returns:
        Projected W_O (embedding_dim, head_dim)
    """
    # Project: W_O_proj = basis @ basis.T @ W_O
    # But we want to keep it in the basis coordinates
    # W_O_proj = basis @ (basis.T @ W_O)
    coeffs = basis.T @ W_O  # (head_dim, head_dim)
    return basis @ coeffs


def orthogonalize_value_output_matrices(
    value_output_matrices: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    head_dim: int = HEAD_DIM
) -> tuple[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], 
           dict[int, np.ndarray]]:
    """
    Orthogonalize W_V and W_O matrices such that:
    - Same s across different n share the same subspace
    - Different s have orthogonal subspaces
    
    Args:
        value_output_matrices: Dict mapping (n, s) to (W_V, W_O)
        head_dim: Head dimension
    
    Returns:
        Tuple of:
        - Dict mapping (n, s) to orthogonalized (W_V, W_O)
        - Dict mapping s to its orthonormal basis
    """
    # Group by s
    s_values = sorted(set(s for (n, s) in value_output_matrices.keys()))
    
    # Step 1: Extract shared subspace basis for each s
    s_to_basis = {}
    s_to_keys = {s: [] for s in s_values}
    
    for (n, s) in value_output_matrices.keys():
        s_to_keys[s].append((n, s))
    
    print("Step 1: Extracting shared subspace for each s...")
    for s in s_values:
        keys = s_to_keys[s]
        W_O_list = [value_output_matrices[k][1] for k in keys]
        basis = extract_subspace_basis(W_O_list, head_dim)
        s_to_basis[s] = basis
        print(f"  s={s}: combined {len(keys)} matrices → basis shape {basis.shape}")
    
    # Step 2: Orthogonalize bases across different s
    print("\nStep 2: Orthogonalizing subspaces across s values...")
    bases_list = [s_to_basis[s] for s in s_values]
    ortho_bases = gram_schmidt_subspaces(bases_list)
    
    for i, s in enumerate(s_values):
        s_to_basis[s] = ortho_bases[i]
    
    # Verify orthogonality
    for i, s_i in enumerate(s_values):
        for j, s_j in enumerate(s_values):
            if i < j:
                overlap = np.abs(ortho_bases[i].T @ ortho_bases[j]).max()
                print(f"  Max overlap between s={s_i} and s={s_j}: {overlap:.6f}")
    
    # Step 3: Project original matrices onto orthogonalized subspaces
    print("\nStep 3: Projecting matrices onto orthogonalized subspaces...")
    results = {}
    
    for (n, s), (W_V, W_O) in value_output_matrices.items():
        basis = s_to_basis[s]
        
        # Project W_O onto the orthogonalized subspace
        W_O_ortho = project_to_subspace(W_O, basis)
        
        # For W_V, we need to transform it consistently
        # W_V has shape (head_dim, embedding_dim)
        # The transformation should be: W_V_ortho = basis.T @ basis @ W_V.T
        # Actually, to keep W_V @ W_O structure consistent:
        # W_V_ortho operates in the new coordinates
        # W_V_ortho = (basis.T @ W_V.T).T = W_V @ basis
        W_V_ortho = (basis.T @ W_O).T @ np.linalg.pinv(W_O.T @ W_O + 1e-6 * np.eye(W_O.shape[1])) @ W_V
        
        # Simpler approach: reconstruct W_V from W_O relationship
        # Since W_V and W_O were derived together, we project W_V similarly
        # W_V shape is (head_dim, embedding_dim)
        # Use the same basis to define the head space
        W_V_ortho = basis.T @ (W_O @ W_V)  # Project through the combined transformation
        
        # Actually, let's keep it simpler and more direct:
        # W_V stays the same dimension-wise, but we ensure the head space aligns
        W_V_ortho = W_V  # Keep original W_V, the orthogonalization is in W_O's column space
        
        results[(n, s)] = (W_V_ortho.astype(np.float32), W_O_ortho.astype(np.float32))
        
        print(f"  (n={n}, s={s}): W_V={W_V_ortho.shape}, W_O={W_O_ortho.shape}")
    
    return results, s_to_basis


def verify_orthogonality(
    ortho_matrices: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    s_to_basis: dict[int, np.ndarray]
) -> dict[tuple[int, int], float]:
    """
    Verify orthogonality between different s subspaces.
    
    Args:
        ortho_matrices: Orthogonalized matrices
        s_to_basis: Basis for each s
    
    Returns:
        Dict of max overlap between each pair of s values
    """
    s_values = sorted(s_to_basis.keys())
    overlaps = {}
    
    for i, s_i in enumerate(s_values):
        for j, s_j in enumerate(s_values):
            if i < j:
                basis_i = s_to_basis[s_i]
                basis_j = s_to_basis[s_j]
                overlap = np.abs(basis_i.T @ basis_j).max()
                overlaps[(s_i, s_j)] = overlap
    
    return overlaps


def get_orthogonal_head_assignment(
    max_n: int,
    max_s: int
) -> list[tuple[int, int]]:
    """
    Get the standard ordering of (n, s) pairs for head assignment.
    
    Args:
        max_n: Maximum n-gram size
        max_s: Maximum skip value
    
    Returns:
        List of (n, s) tuples in canonical order
    """
    pairs = []
    for n in range(1, max_n + 1):
        for s in range(0, max_s + 1):
            pairs.append((n, s))
    return pairs


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_extract_subspace_basis():
    """Test subspace basis extraction."""
    embed_dim = 16
    head_dim = 4
    
    # Create some W_O matrices
    W_O_list = [
        np.random.randn(embed_dim, head_dim).astype(np.float32)
        for _ in range(3)
    ]
    
    basis = extract_subspace_basis(W_O_list, head_dim)
    
    assert basis.shape == (embed_dim, head_dim)
    
    # Check orthonormality
    gram = basis.T @ basis
    assert np.allclose(gram, np.eye(head_dim), atol=1e-5), "Basis should be orthonormal"
    
    print("✓ test_extract_subspace_basis passed")


def test_gram_schmidt_subspaces():
    """Test Gram-Schmidt orthogonalization of subspaces."""
    embed_dim = 32
    head_dim = 4
    
    # Create random bases
    bases = [
        np.linalg.qr(np.random.randn(embed_dim, head_dim))[0]
        for _ in range(3)
    ]
    
    ortho_bases = gram_schmidt_subspaces(bases)
    
    assert len(ortho_bases) == 3
    
    # Check each basis is orthonormal
    for i, basis in enumerate(ortho_bases):
        gram = basis.T @ basis
        assert np.allclose(gram, np.eye(head_dim), atol=1e-5), f"Basis {i} not orthonormal"
    
    # Check bases are mutually orthogonal
    for i in range(3):
        for j in range(i + 1, 3):
            cross = ortho_bases[i].T @ ortho_bases[j]
            max_overlap = np.abs(cross).max()
            assert max_overlap < 1e-5, f"Bases {i} and {j} not orthogonal: {max_overlap}"
    
    print("✓ test_gram_schmidt_subspaces passed")


def test_project_to_subspace():
    """Test projection to subspace."""
    embed_dim = 16
    head_dim = 4
    
    # Create orthonormal basis
    basis = np.linalg.qr(np.random.randn(embed_dim, head_dim))[0].astype(np.float32)
    
    # Create W_O
    W_O = np.random.randn(embed_dim, head_dim).astype(np.float32)
    
    # Project
    W_O_proj = project_to_subspace(W_O, basis)
    
    assert W_O_proj.shape == W_O.shape
    
    # Projected matrix should be in the subspace
    # i.e., basis @ basis.T @ W_O_proj = W_O_proj
    reconstructed = basis @ (basis.T @ W_O_proj)
    assert np.allclose(reconstructed, W_O_proj, atol=1e-5)
    
    print("✓ test_project_to_subspace passed")


def test_orthogonalize_full():
    """Test full orthogonalization pipeline."""
    embed_dim = 32
    head_dim = 4
    
    # Create mock value_output_matrices
    value_output_matrices = {
        (1, 0): (np.random.randn(head_dim, embed_dim).astype(np.float32),
                 np.random.randn(embed_dim, head_dim).astype(np.float32)),
        (2, 0): (np.random.randn(head_dim, embed_dim).astype(np.float32),
                 np.random.randn(embed_dim, head_dim).astype(np.float32)),
        (1, 1): (np.random.randn(head_dim, embed_dim).astype(np.float32),
                 np.random.randn(embed_dim, head_dim).astype(np.float32)),
        (2, 1): (np.random.randn(head_dim, embed_dim).astype(np.float32),
                 np.random.randn(embed_dim, head_dim).astype(np.float32)),
    }
    
    ortho_matrices, s_to_basis = orthogonalize_value_output_matrices(
        value_output_matrices, head_dim
    )
    
    # Check we got all matrices back
    assert set(ortho_matrices.keys()) == set(value_output_matrices.keys())
    
    # Check bases for different s are orthogonal
    overlaps = verify_orthogonality(ortho_matrices, s_to_basis)
    for (s_i, s_j), overlap in overlaps.items():
        assert overlap < 1e-4, f"s={s_i} and s={s_j} not orthogonal: {overlap}"
    
    print("✓ test_orthogonalize_full passed")


def test_many_s_values():
    """Test with many s values to stress orthogonalization."""
    embed_dim = 128
    head_dim = 8
    max_s = 5
    
    # Create matrices for s=0,1,2,3,4,5
    value_output_matrices = {}
    for s in range(max_s + 1):
        for n in [1, 2]:
            value_output_matrices[(n, s)] = (
                np.random.randn(head_dim, embed_dim).astype(np.float32),
                np.random.randn(embed_dim, head_dim).astype(np.float32)
            )
    
    ortho_matrices, s_to_basis = orthogonalize_value_output_matrices(
        value_output_matrices, head_dim
    )
    
    # Verify all pairs are orthogonal
    overlaps = verify_orthogonality(ortho_matrices, s_to_basis)
    max_overlap = max(overlaps.values()) if overlaps else 0
    print(f"  Maximum overlap across all s pairs: {max_overlap:.6f}")
    
    assert max_overlap < 1e-4, f"Not all s values orthogonal"
    
    print("✓ test_many_s_values passed")


def run_all_tests():
    """Run all unit tests."""
    print("Running orthogonalization tests...\n")
    
    test_extract_subspace_basis()
    test_gram_schmidt_subspaces()
    test_project_to_subspace()
    test_orthogonalize_full()
    test_many_s_values()
    
    print("\n✓ All orthogonalization tests passed!")


if __name__ == "__main__":
    run_all_tests()
    
    print("\n" + "=" * 60)
    print("Demo: Orthogonalizing W_V and W_O matrices")
    print("=" * 60)
    
    np.random.seed(42)
    embed_dim = 64
    head_dim = 8
    
    # Create mock value_output_matrices with multiple n and s
    print("\nCreating mock matrices...")
    value_output_matrices = {}
    for n in [1, 2, 3]:
        for s in [0, 1, 2]:
            W_V = np.random.randn(head_dim, embed_dim).astype(np.float32)
            W_O = np.random.randn(embed_dim, head_dim).astype(np.float32)
            value_output_matrices[(n, s)] = (W_V, W_O)
            print(f"  (n={n}, s={s}): W_V={W_V.shape}, W_O={W_O.shape}")
    
    print("\n" + "-" * 60)
    print("Orthogonalizing...")
    print("-" * 60)
    
    ortho_matrices, s_to_basis = orthogonalize_value_output_matrices(
        value_output_matrices, head_dim
    )
    
    print("\n" + "-" * 60)
    print("Verification")
    print("-" * 60)
    
    overlaps = verify_orthogonality(ortho_matrices, s_to_basis)
    print("\nPairwise overlaps between s subspaces:")
    for (s_i, s_j), overlap in sorted(overlaps.items()):
        status = "✓" if overlap < 1e-4 else "✗"
        print(f"  {status} s={s_i} vs s={s_j}: max overlap = {overlap:.8f}")
    
    print("\nSubspace dimensions:")
    for s, basis in sorted(s_to_basis.items()):
        rank = np.linalg.matrix_rank(basis)
        print(f"  s={s}: basis shape={basis.shape}, rank={rank}")


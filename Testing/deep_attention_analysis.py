"""
Deep Attention Analysis

1. Try different normalization scales
2. Analyze why Q and K are so orthogonal
3. Check if SVD derivation is producing expected structure
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, '../Components/method')
from pico_config import PicoModelConfig, PicoTrainingConfig
from pico_sparse import PicoGPTSparse
from pico_method import compute_pico_attention_weights
from pico_tokenizer import PicoTokenizer
from corpus_matrix import CorpusMatrixBuilder


def analyze_svd_structure(texts, config, max_n=2, max_s=3):
    """Analyze the SVD structure before and after weight computation."""
    
    print("\n" + "="*60)
    print("SVD Structure Analysis")
    print("="*60)
    
    # We need to manually run through the pipeline to inspect intermediate results
    from pico_method import (
        build_all_count_matrices,
        build_all_manifold_matrices,
        build_all_probability_matrices,
        create_base_embeddings,
        build_all_embedding_matrices,
        compute_key_query_matrices,
        compute_value_output_matrices,
    )
    from pico_tokenizer import PicoTokenizer as PicoTok
    
    tokenizer = PicoTok()
    vocab_size = tokenizer.vocab_size
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))
    
    batch_size = 256
    batches = []
    for i in range(0, len(all_tokens) - batch_size + 1, batch_size):
        batches.append(all_tokens[i:i + batch_size])
    
    print(f"\nStep 1: Count matrices")
    count_matrices = build_all_count_matrices(batches, max_n, max_s, vocab_size)
    
    print(f"\nStep 2: Manifold matrices")
    manifold_matrices = build_all_manifold_matrices(count_matrices)
    
    print(f"\nStep 3: Probability matrices")
    prob_matrices = build_all_probability_matrices(count_matrices)
    
    print(f"\nStep 4: Embeddings")
    base_embeddings = create_base_embeddings(vocab_size, config.embedding_dim)
    embedding_matrices = build_all_embedding_matrices(prob_matrices, base_embeddings, max_n, vocab_size)
    
    print(f"\nStep 5: Analyzing SVD for Layer 0, Head 0 (n=1, s=0)")
    
    # Get matrices for first layer, first head
    n = 1
    s = 0
    key = (n, s)
    
    M = manifold_matrices[key]
    P = prob_matrices[key]
    E_n = embedding_matrices[1]
    E_1 = embedding_matrices[1]
    
    print(f"\nManifold matrix M^{n,s}:")
    print(f"  Shape: {M.shape}")
    print(f"  Density: {M.nnz / (M.shape[0] * M.shape[1]):.4%}")
    print(f"  Mean: {M.data.mean():.6f}")
    print(f"  Std: {M.data.std():.6f}")
    print(f"  Max: {M.data.max():.6f}")
    print(f"  Min: {M.data.min():.6f}")
    
    # Convert to dense for SVD
    M_dense = M.toarray().astype(np.float32)
    
    print(f"\nPerforming SVD on M...")
    from scipy.sparse.linalg import svds
    U, S, Vt = svds(M_dense, k=min(64, min(M.shape) - 1))
    V = Vt.T
    
    print(f"SVD Results:")
    print(f"  U shape: {U.shape}, V shape: {V.shape}, S shape: {S.shape}")
    print(f"  Top 10 singular values: {S[:10]}")
    print(f"  S[0] / S[-1] (condition): {S[0] / S[-1]:.2f}")
    
    # Check orthogonality of U and V
    U_ortho = np.abs(U.T @ U - np.eye(U.shape[1])).max()
    V_ortho = np.abs(V.T @ V - np.eye(V.shape[1])).max()
    print(f"\nOrthogonality check:")
    print(f"  U^T @ U - I max error: {U_ortho:.6f}")
    print(f"  V^T @ V - I max error: {V_ortho:.6f}")
    
    # Now compute W_Q and W_K
    head_dim = config.head_dim
    sqrt_S = np.sqrt(S[:head_dim])
    
    # W_Q = E^n.T @ U @ sqrt(S)
    # W_K = E^1.T @ V @ sqrt(S)
    
    # Handle dimension matching
    if E_n.shape[0] != M.shape[0]:
        if E_n.shape[0] > M.shape[0]:
            E_n = E_n[:M.shape[0]]
        else:
            pad = np.zeros((M.shape[0] - E_n.shape[0], E_n.shape[1]), dtype=np.float32)
            E_n = np.vstack([E_n, pad])
    
    if E_1.shape[0] != M.shape[1]:
        if E_1.shape[0] > M.shape[1]:
            E_1 = E_1[:M.shape[1]]
        else:
            pad = np.zeros((M.shape[1] - E_1.shape[0], E_1.shape[1]), dtype=np.float32)
            E_1 = np.vstack([E_1, pad])
    
    W_Q_raw = E_n.T @ U[:, :head_dim] @ np.diag(sqrt_S)
    W_K_raw = E_1.T @ V[:, :head_dim] @ np.diag(sqrt_S)
    
    print(f"\nComputed W_Q and W_K:")
    print(f"  W_Q shape: {W_Q_raw.shape}, mean={W_Q_raw.mean():.6f}, std={W_Q_raw.std():.6f}")
    print(f"  W_K shape: {W_K_raw.shape}, mean={W_K_raw.mean():.6f}, std={W_K_raw.std():.6f}")
    
    # Check Q-K similarity
    W_Q_norm = W_Q_raw / (np.linalg.norm(W_Q_raw, axis=0, keepdims=True) + 1e-8)
    W_K_norm = W_K_raw / (np.linalg.norm(W_K_raw, axis=0, keepdims=True) + 1e-8)
    QK_sim = np.mean(np.diag(W_Q_norm.T @ W_K_norm))
    print(f"  W_Q vs W_K cosine similarity: {QK_sim:.6f}")
    
    # Analyze why they might be orthogonal
    print(f"\nAnalyzing orthogonality:")
    print(f"  E_n.T @ E_n (embedding similarity):")
    E_n_sim = E_n @ E_n.T
    print(f"    Mean: {E_n_sim.mean():.6f}, Std: {E_n_sim.std():.6f}")
    print(f"  E_1.T @ E_1 (embedding similarity):")
    E_1_sim = E_1 @ E_1.T
    print(f"    Mean: {E_1_sim.mean():.6f}, Std: {E_1_sim.std():.6f}")
    print(f"  E_n.T @ E_1 (cross embedding similarity):")
    E_cross = E_n[:min(E_n.shape[0], E_1.shape[0])] @ E_1[:min(E_n.shape[0], E_1.shape[0])].T
    print(f"    Mean: {E_cross.mean():.6f}, Std: {E_cross.std():.6f}")
    
    # Check U and V structure
    print(f"\nU and V structure:")
    print(f"  U columns (left singular vectors) norm: {np.linalg.norm(U, axis=0)[:10]}")
    print(f"  V columns (right singular vectors) norm: {np.linalg.norm(V, axis=0)[:10]}")
    
    return {
        'M': M_dense,
        'U': U,
        'V': V,
        'S': S,
        'E_n': E_n,
        'E_1': E_1,
        'W_Q_raw': W_Q_raw,
        'W_K_raw': W_K_raw,
    }


def test_normalization_scales(texts, config, output_dir):
    """Test different normalization scales."""
    
    print("\n" + "="*60)
    print("Testing Normalization Scales")
    print("="*60)
    
    scales = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    tokenizer = PicoTokenizer()
    sample_text = texts[0][:200]
    tokens = tokenizer.encode(sample_text)[:32]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    results = []
    
    for scale in scales:
        print(f"\nTesting scale={scale}...")
        
        # Compute weights with this scale
        attention_weights, base_embeddings = compute_pico_attention_weights(
            texts[:15],
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            embedding_dim=config.embedding_dim,
            head_dim=config.head_dim,
            max_n=2,
            max_s=3,
            verbose=False,
            normalize_weights=True,
            target_std=scale,
        )
        
        # Create model
        torch.manual_seed(42)
        model = PicoGPTSparse(config, attention_weights, base_embeddings=base_embeddings)
        
        # Get attention outputs
        attention_outputs = []
        def attention_hook(module, input, output):
            attn_out, attn_weights = output
            attention_outputs.append(attn_weights.detach())
        
        hooks = [block.attention.register_forward_hook(attention_hook) for block in model.blocks]
        
        with torch.no_grad():
            _ = model(input_ids)
        
        for hook in hooks:
            hook.remove()
        
        # Analyze attention patterns
        layer_entropies = []
        layer_max_attns = []
        qk_similarities = []
        
        for layer_idx, attn_weights in enumerate(attention_outputs):
            attn_weights = attn_weights[0]  # Remove batch
            
            # Entropy
            entropies = []
            for head_idx in range(config.n_heads):
                attn_pattern = attn_weights[head_idx].cpu().numpy()
                for q_pos in range(attn_pattern.shape[0]):
                    probs = attn_pattern[q_pos, :q_pos+1]
                    probs = probs / (probs.sum() + 1e-10)
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    entropies.append(entropy)
            layer_entropies.append(np.mean(entropies))
            
            # Max attention
            max_attns = []
            for head_idx in range(config.n_heads):
                attn_pattern = attn_weights[head_idx].cpu().numpy()
                for q_pos in range(attn_pattern.shape[0]):
                    causal_attn = attn_pattern[q_pos, :q_pos+1]
                    max_attns.append(causal_attn.max())
            layer_max_attns.append(np.mean(max_attns))
            
            # Q-K similarity
            block = model.blocks[layer_idx]
            W_Q = block.attention.W_Q[0]  # First head
            W_K = block.attention.W_K[0]
            embeddings = model.token_embedding(input_ids)[0]
            Q = embeddings @ W_Q
            K = embeddings @ W_K
            Q_norm = Q / (Q.norm(dim=1, keepdim=True) + 1e-8)
            K_norm = K / (K.norm(dim=1, keepdim=True) + 1e-8)
            qk_sim = (Q_norm @ K_norm.T).mean().item()
            qk_similarities.append(qk_sim)
        
        # Compute attention scores magnitude
        block = model.blocks[0]
        W_Q = block.attention.W_Q[0]
        W_K = block.attention.W_K[0]
        embeddings = model.token_embedding(input_ids)[0]
        Q = embeddings @ W_Q
        K = embeddings @ W_K
        scores = Q @ K.T
        score_magnitude = scores.abs().mean().item()
        
        results.append({
            'scale': scale,
            'avg_entropy': np.mean(layer_entropies),
            'avg_max_attn': np.mean(layer_max_attns),
            'avg_qk_sim': np.mean(qk_similarities),
            'score_magnitude': score_magnitude,
            'W_Q_std': attention_weights[0][0].std().item(),
            'W_K_std': attention_weights[0][1].std().item(),
        })
        
        print(f"  Avg entropy: {np.mean(layer_entropies):.4f}")
        print(f"  Avg max attention: {np.mean(layer_max_attns):.4f}")
        print(f"  Q-K similarity: {np.mean(qk_similarities):.4f}")
        print(f"  Score magnitude: {score_magnitude:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scales_list = [r['scale'] for r in results]
    
    axes[0, 0].semilogx(scales_list, [r['avg_entropy'] for r in results], 'o-')
    axes[0, 0].set_xlabel('Normalization Scale (target_std)')
    axes[0, 0].set_ylabel('Average Entropy')
    axes[0, 0].set_title('Attention Entropy vs Scale')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=np.log(32), color='r', linestyle='--', label='Uniform (max)')
    axes[0, 0].legend()
    
    axes[0, 1].semilogx(scales_list, [r['avg_max_attn'] for r in results], 'o-')
    axes[0, 1].set_xlabel('Normalization Scale (target_std)')
    axes[0, 1].set_ylabel('Average Max Attention')
    axes[0, 1].set_title('Max Attention vs Scale')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].semilogx(scales_list, [r['avg_qk_sim'] for r in results], 'o-')
    axes[1, 0].set_xlabel('Normalization Scale (target_std)')
    axes[1, 0].set_ylabel('Q-K Similarity')
    axes[1, 0].set_title('Q-K Similarity vs Scale')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].semilogx(scales_list, [r['score_magnitude'] for r in results], 'o-')
    axes[1, 1].set_xlabel('Normalization Scale (target_std)')
    axes[1, 1].set_ylabel('Attention Score Magnitude')
    axes[1, 1].set_title('Attention Score Magnitude vs Scale')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'normalization_scale_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'normalization_scale_analysis.png'}")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("Normalization Scale Summary")
    print("="*60)
    print(f"{'Scale':<10} {'Entropy':<10} {'Max Attn':<10} {'Q-K Sim':<10} {'Score Mag':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['scale']:<10} {r['avg_entropy']:<10.4f} {r['avg_max_attn']:<10.4f} "
              f"{r['avg_qk_sim']:<10.4f} {r['score_magnitude']:<12.6f}")
    
    return results


def analyze_qk_orthogonality(texts, config):
    """Deep dive into why Q and K are orthogonal."""
    
    print("\n" + "="*60)
    print("Q-K Orthogonality Analysis")
    print("="*60)
    
    # Get the raw SVD results
    svd_results = analyze_svd_structure(texts[:15], config)
    
    W_Q = svd_results['W_Q_raw']
    W_K = svd_results['W_K_raw']
    U = svd_results['U']
    V = svd_results['V']
    S = svd_results['S']
    E_n = svd_results['E_n']
    E_1 = svd_results['E_1']
    
    print(f"\n" + "="*60)
    print("Why are W_Q and W_K orthogonal?")
    print("="*60)
    
    # W_Q = E_n.T @ U @ sqrt(S)
    # W_K = E_1.T @ V @ sqrt(S)
    
    # If W_Q and W_K are orthogonal, then:
    # W_Q.T @ W_K = sqrt(S) @ U.T @ E_n @ E_1.T @ V @ sqrt(S) ≈ 0
    
    print(f"\n1. Checking W_Q.T @ W_K:")
    WQ_WK = W_Q.T @ W_K
    print(f"   Shape: {WQ_WK.shape}")
    print(f"   Mean: {WQ_WK.mean():.6f}")
    print(f"   Max abs: {np.abs(WQ_WK).max():.6f}")
    print(f"   Diagonal: {np.diag(WQ_WK)[:10]}")
    
    print(f"\n2. Breaking it down:")
    print(f"   E_n @ E_1.T:")
    E_cross = E_n[:min(E_n.shape[0], E_1.shape[0])] @ E_1[:min(E_n.shape[0], E_1.shape[0])].T
    print(f"     Shape: {E_cross.shape}")
    print(f"     Mean: {E_cross.mean():.6f}")
    print(f"     Std: {E_cross.std():.6f}")
    
    print(f"\n   U.T @ E_n @ E_1.T @ V:")
    # This is the key: U.T @ (E_n @ E_1.T) @ V
    # Need to match dimensions
    min_rows = min(U.shape[0], E_cross.shape[0])
    min_cols = min(E_cross.shape[1], V.shape[0])
    E_cross_matched = E_cross[:min_rows, :min_cols]
    U_matched = U[:min_rows, :64]
    V_matched = V[:min_cols, :64]
    intermediate = U_matched.T @ E_cross_matched @ V_matched
    print(f"     Shape: {intermediate.shape}")
    print(f"     Mean: {intermediate.mean():.6f}")
    print(f"     Max abs: {np.abs(intermediate).max():.6f}")
    print(f"     Diagonal: {np.diag(intermediate)[:10]}")
    
    print(f"\n3. Checking if U and V are related to E_n and E_1:")
    print(f"   U columns vs E_n rows:")
    min_dim = min(U.shape[0], E_n.shape[0])
    U_E_n_sim = (U[:min_dim, :10].T @ E_n[:min_dim, :]).mean(axis=1)
    print(f"     Mean similarity: {U_E_n_sim.mean():.6f}")
    
    print(f"   V columns vs E_1 rows:")
    min_dim = min(V.shape[0], E_1.shape[0])
    V_E_1_sim = (V[:min_dim, :10].T @ E_1[:min_dim, :]).mean(axis=1)
    print(f"     Mean similarity: {V_E_1_sim.mean():.6f}")
    
    print(f"\n4. The manifold matrix M structure:")
    M = svd_results['M']
    print(f"   M = U @ S @ V.T")
    print(f"   M is symmetric? {np.allclose(M, M.T, atol=1e-6)}")
    print(f"   If M is symmetric, then U ≈ V (up to sign)")
    if np.allclose(M, M.T, atol=1e-6):
        U_V_sim = np.abs((U[:, :10].T @ V[:, :10])).mean(axis=1)
        print(f"   U vs V similarity: {U_V_sim.mean():.6f}")
        print(f"   This means W_Q and W_K use similar directions from U and V")
    
    return svd_results


def main():
    """Run all analyses."""
    
    output_dir = Path("attention_deep_analysis")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Deep Attention Analysis")
    print("="*60)
    
    # Load data
    print("\nLoading FineWeb data...")
    builder = CorpusMatrixBuilder()
    token_batches = list(builder.stream_token_batches(max_batches=20))
    texts = [builder.tokenizer.decode(b) for b in token_batches]
    
    config = PicoModelConfig(n_layers=4, n_heads=4, embedding_dim=256, head_dim=64, mlp_dim=1024)
    
    # Run all analyses
    svd_results = analyze_qk_orthogonality(texts, config)
    norm_results = test_normalization_scales(texts, config, output_dir)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


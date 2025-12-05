"""
Analyze Attention Alignment

Compare what positions SHOULD be attended to (based on count matrices)
vs what positions ARE actually attended to.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, '../Components/method')
from pico_config import PicoModelConfig
from pico_sparse import PicoGPTSparse
from pico_method import compute_pico_attention_weights
from pico_tokenizer import PicoTokenizer
from corpus_matrix import CorpusMatrixBuilder


def get_expected_attention_from_counts(texts, config, max_n=2, max_s=3):
    """Compute what positions SHOULD be attended to based on count matrices."""
    
    from pico_method import build_all_count_matrices, build_all_probability_matrices
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
    
    # Build count matrices
    count_matrices = build_all_count_matrices(batches, max_n, max_s, vocab_size)
    prob_matrices = build_all_probability_matrices(count_matrices)
    
    # For each (n, s), compute expected attention pattern
    # Expected attention: P(token_j | ngram_i) where ngram_i ends at position i
    expected_patterns = {}
    
    for (n, s), P in prob_matrices.items():
        # P is (vocab^n, vocab) - probability of token given n-gram
        # For attention, we want: given position i with n-gram ending there,
        # what positions j should we attend to?
        
        # Actually, the count matrix tells us: ngram -> next token
        # So if we're at position i with n-gram (tokens[i-n+1:i+1]),
        # we should attend to positions where we've seen this n-gram before
        
        # But wait - the attention mechanism is different:
        # - Query at position i asks "what should I attend to?"
        # - Key at position j says "I'm here, attend to me if relevant"
        # - The score is Q_i @ K_j
        
        # The count matrix C(n,s) tells us: ngram -> token transitions
        # This doesn't directly tell us position-to-position attention
        
        # However, we can infer: if token B often follows n-gram A,
        # then when we see n-gram A ending at position i, we should
        # attend to positions where token B appears (or where n-gram A appears)
        
        # Let's compute: for each n-gram, what tokens are likely to follow?
        # Then: when at a position with that n-gram, attend to positions with those tokens
        
        expected_patterns[(n, s)] = P
    
    return expected_patterns, count_matrices, prob_matrices


def analyze_actual_attention(model, tokenizer, sample_text, config, max_seq_len=32):
    """Get actual attention patterns from the model."""
    
    tokens = tokenizer.encode(sample_text)[:max_seq_len]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    attention_outputs = []
    
    def attention_hook(module, input, output):
        attn_out, attn_weights = output
        attention_outputs.append(attn_weights.detach())
    
    hooks = [block.attention.register_forward_hook(attention_hook) for block in model.blocks]
    
    with torch.no_grad():
        _ = model(input_ids)
    
    for hook in hooks:
        hook.remove()
    
    return attention_outputs, tokens


def compute_ngram_at_position(tokens, pos, n):
    """Get the n-gram ending at position pos."""
    if pos < n - 1:
        return None
    return tuple(tokens[pos - n + 1:pos + 1])


def analyze_attention_vs_counts(
    texts,
    config,
    output_dir,
    max_seq_len=32,
):
    """Compare expected vs actual attention."""
    
    print("="*60)
    print("Attention Alignment Analysis")
    print("="*60)
    
    # Get expected patterns from counts
    print("\n1. Computing expected attention from count matrices...")
    expected_patterns, count_matrices, prob_matrices = get_expected_attention_from_counts(
        texts[:15], config
    )
    
    # Get actual attention
    print("\n2. Computing actual attention from model...")
    tokenizer = PicoTokenizer()
    sample_text = texts[0][:500]
    
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
        target_std=0.01,  # Current setting
    )
    
    torch.manual_seed(42)
    model = PicoGPTSparse(config, attention_weights, base_embeddings=base_embeddings)
    
    actual_attentions, tokens = analyze_actual_attention(model, tokenizer, sample_text, config, max_seq_len)
    
    print(f"\n3. Analyzing alignment for sequence of {len(tokens)} tokens")
    print(f"   Sample tokens: {tokens[:20]}")
    
    # For each layer/head, compare expected vs actual
    for layer_idx, attn_weights in enumerate(actual_attentions):
        # Layer uses n = layer_idx + 1 (capped at max_n)
        n = min(layer_idx + 1, 2)
        
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx} (n={n})")
        print(f"{'='*60}")
        
        attn_weights = attn_weights[0]  # Remove batch
        
        for head_idx in range(config.n_heads):
            # Head uses s = head_idx (capped at max_s)
            s = min(head_idx, 3)
            
            print(f"\nHead {head_idx} (s={s}):")
            
            attn_pattern = attn_weights[head_idx].cpu().numpy()  # (seq_len, seq_len)
            
            # Get the probability matrix for this (n, s)
            key = (n, s)
            if key not in prob_matrices:
                key = (1, 0)  # Fallback
            P = prob_matrices[key]
            
            # For each query position, compute expected attention
            alignment_scores = []
            
            for q_pos in range(len(tokens)):
                # Get n-gram ending at q_pos
                ngram = compute_ngram_at_position(tokens, q_pos, n)
                if ngram is None:
                    continue
                
                # Convert n-gram to index
                # ngram is tuple of tokens, e.g., (token1, token2) for n=2
                ngram_idx = 0
                for i, token in enumerate(ngram):
                    ngram_idx += token * (85 ** (len(ngram) - 1 - i))
                
                if ngram_idx >= P.shape[0]:
                    continue
                
                # Get probability distribution over next tokens
                next_token_probs = P[ngram_idx, :].toarray().flatten()
                
                # Expected: attend to positions where likely next tokens appear
                # But we need to look BACKWARDS (causal attention)
                # Only consider positions up to q_pos (causal)
                expected_attention = np.zeros(q_pos + 1)
                
                for k_pos in range(q_pos + 1):  # Causal: only look at past
                    if k_pos < len(tokens):
                        token_at_k = tokens[k_pos]
                        # Probability that this token follows our n-gram
                        prob = next_token_probs[token_at_k]
                        expected_attention[k_pos] = prob
                
                # Normalize expected attention
                if expected_attention.sum() > 0:
                    expected_attention = expected_attention / expected_attention.sum()
                
                # Get actual attention at this query position
                actual_attention = attn_pattern[q_pos, :q_pos+1].copy()  # Causal
                # Both should now have length q_pos+1
                
                # Compare using KL divergence or cosine similarity
                if expected_attention.sum() > 0 and actual_attention.sum() > 0 and len(expected_attention) == len(actual_attention):
                    # KL divergence
                    kl = np.sum(expected_attention * np.log(
                        (expected_attention + 1e-10) / (actual_attention + 1e-10)
                    ))
                    
                    # Cosine similarity
                    cos_sim = np.dot(expected_attention, actual_attention) / (
                        np.linalg.norm(expected_attention) * np.linalg.norm(actual_attention) + 1e-10
                    )
                    
                    alignment_scores.append({
                        'pos': q_pos,
                        'kl': kl,
                        'cos_sim': cos_sim,
                        'expected_max': np.argmax(expected_attention),
                        'actual_max': np.argmax(actual_attention),
                    })
            
            if alignment_scores:
                avg_kl = np.mean([s['kl'] for s in alignment_scores])
                avg_cos = np.mean([s['cos_sim'] for s in alignment_scores])
                max_match = sum(1 for s in alignment_scores if s['expected_max'] == s['actual_max'])
                max_match_ratio = max_match / len(alignment_scores)
                
                print(f"  Average KL divergence: {avg_kl:.4f} (lower = better alignment)")
                print(f"  Average cosine similarity: {avg_cos:.4f} (higher = better alignment)")
                print(f"  Max position matches: {max_match_ratio:.2%}")
                print(f"  Sample: pos 10 expected_max={alignment_scores[10]['expected_max']}, "
                      f"actual_max={alignment_scores[10]['actual_max']}")
            else:
                print(f"  No alignment scores computed (sequence too short or n-gram issues)")
    
    # Now let's analyze WHY there's a mismatch
    print(f"\n{'='*60}")
    print("Root Cause Analysis")
    print(f"{'='*60}")
    
    # The key insight: W_Q and W_K are derived from M via SVD
    # But M captures n-gram -> token co-occurrence
    # This doesn't directly translate to position -> position attention
    
    print("\nThe Problem:")
    print("1. Count matrix C(n,s) captures: ngram -> token transitions")
    print("2. Manifold matrix M(n,s) = normalized C(n,s)")
    print("3. SVD of M gives: M = U @ S @ V.T")
    print("4. W_Q = E^n.T @ U @ sqrt(S)")
    print("5. W_K = E^1.T @ V @ sqrt(S)")
    print("\nBut:")
    print("- M is (vocab^n, vocab) - it's about TOKEN types, not POSITIONS")
    print("- Attention is about POSITION -> POSITION relationships")
    print("- The mapping from token co-occurrence to position attention is LOST")
    print("\nThe attention mechanism computes:")
    print("  score[i,j] = Q_i @ K_j = (E_i @ W_Q) @ (E_j @ W_K)")
    print("  This depends on EMBEDDING similarity, not count matrix structure!")
    
    # Let's check if embeddings encode the n-gram structure
    print("\nChecking if embeddings encode n-gram structure...")
    
    # Get embeddings for sample tokens
    with torch.no_grad():
        sample_ids = torch.tensor([tokens[:10]], dtype=torch.long)
        embeddings = model.token_embedding(sample_ids)[0]  # (10, embed_dim)
    
    # Check if similar n-grams have similar embeddings
    print(f"Embedding norms: {embeddings.norm(dim=1)[:10]}")
    print(f"Embedding similarity (first 5 tokens):")
    emb_sim = embeddings[:5] @ embeddings[:5].T
    print(emb_sim)
    
    print("\nConclusion:")
    print("The count matrices capture TOKEN co-occurrence patterns,")
    print("but the attention mechanism operates on EMBEDDING similarity.")
    print("If embeddings don't encode the n-gram structure, attention won't align!")


def visualize_expected_vs_actual(
    texts,
    config,
    output_dir,
    max_seq_len=32,
):
    """Visualize expected vs actual attention patterns."""
    
    tokenizer = PicoTokenizer()
    sample_text = texts[0][:500]
    tokens = tokenizer.encode(sample_text)[:max_seq_len]
    
    # Get count matrices
    from pico_method import build_all_count_matrices, build_all_probability_matrices
    from pico_tokenizer import PicoTokenizer as PicoTok
    
    pico_tok = PicoTok()
    all_tokens = []
    for text in texts[:15]:
        all_tokens.extend(pico_tok.encode(text))
    
    batches = []
    for i in range(0, len(all_tokens) - 256 + 1, 256):
        batches.append(all_tokens[i:i + 256])
    
    count_matrices = build_all_count_matrices(batches, 2, 3, 85)
    prob_matrices = build_all_probability_matrices(count_matrices)
    
    # Get actual attention
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
        target_std=0.01,
    )
    
    torch.manual_seed(42)
    model = PicoGPTSparse(config, attention_weights, base_embeddings=base_embeddings)
    
    input_ids = torch.tensor([tokens], dtype=torch.long)
    attention_outputs = []
    
    def attention_hook(module, input, output):
        attn_out, attn_weights = output
        attention_outputs.append(attn_weights.detach())
    
    hooks = [block.attention.register_forward_hook(attention_hook) for block in model.blocks]
    with torch.no_grad():
        _ = model(input_ids)
    for hook in hooks:
        hook.remove()
    
    # Create visualization
    fig, axes = plt.subplots(config.n_layers, config.n_heads * 2, 
                             figsize=(6*config.n_heads, 4*config.n_layers))
    if config.n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx in range(config.n_layers):
        n = min(layer_idx + 1, 2)
        attn_weights = attention_outputs[layer_idx][0]
        
        for head_idx in range(config.n_heads):
            s = min(head_idx, 3)
            key = (n, s)
            if key not in prob_matrices:
                key = (1, 0)
            P = prob_matrices[key]
            
            # Expected attention (from counts)
            expected = np.zeros((len(tokens), len(tokens)))
            for q_pos in range(len(tokens)):
                ngram = compute_ngram_at_position(tokens, q_pos, n)
                if ngram is None:
                    continue
                # Convert n-gram to index
                ngram_idx = 0
                for i, token in enumerate(ngram):
                    ngram_idx += token * (85 ** (len(ngram) - 1 - i))
                if ngram_idx < P.shape[0]:
                    next_probs = P[ngram_idx, :].toarray().flatten()
                    for k_pos in range(q_pos + 1):
                        if k_pos < len(tokens):
                            expected[q_pos, k_pos] = next_probs[tokens[k_pos]]
                    if expected[q_pos, :].sum() > 0:
                        expected[q_pos, :] /= expected[q_pos, :].sum()
            
            # Actual attention
            actual = attn_weights[head_idx].cpu().numpy()[:len(tokens), :len(tokens)]
            
            # Plot
            col = head_idx * 2
            im1 = axes[layer_idx, col].imshow(expected, aspect='auto', cmap='Blues', vmin=0, vmax=1)
            axes[layer_idx, col].set_title(f'L{layer_idx}H{head_idx} Expected\n(n={n},s={s})')
            plt.colorbar(im1, ax=axes[layer_idx, col])
            
            im2 = axes[layer_idx, col+1].imshow(actual, aspect='auto', cmap='Blues', vmin=0, vmax=1)
            axes[layer_idx, col+1].set_title(f'L{layer_idx}H{head_idx} Actual')
            plt.colorbar(im2, ax=axes[layer_idx, col+1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'expected_vs_actual_attention.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'expected_vs_actual_attention.png'}")
    plt.close()


def main():
    """Run alignment analysis."""
    
    output_dir = Path("attention_alignment_analysis")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Attention Alignment Analysis")
    print("="*60)
    
    # Load data
    print("\nLoading FineWeb data...")
    builder = CorpusMatrixBuilder()
    token_batches = list(builder.stream_token_batches(max_batches=20))
    texts = [builder.tokenizer.decode(b) for b in token_batches]
    
    config = PicoModelConfig(n_layers=4, n_heads=4, embedding_dim=256, head_dim=64, mlp_dim=1024)
    
    # Run analyses
    analyze_attention_vs_counts(texts, config, output_dir)
    visualize_expected_vs_actual(texts, config, output_dir)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


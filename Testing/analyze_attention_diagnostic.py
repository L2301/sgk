"""
Diagnostic Analysis of Attention Patterns

Deep dive into what the corpus-derived attention is actually doing.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '../Components/method')
from pico_config import PicoModelConfig, PicoTrainingConfig
from pico_sparse import PicoGPTSparse
from pico_method import compute_pico_attention_weights
from pico_tokenizer import PicoTokenizer
from corpus_matrix import CorpusMatrixBuilder


def analyze_attention_entropy(model, tokenizer, sample_text, config, max_seq_len=32):
    """Analyze attention entropy - uniform attention has high entropy."""
    
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
    
    print("\n" + "="*60)
    print("Attention Entropy Analysis")
    print("="*60)
    print("(High entropy = uniform/random, Low entropy = focused)")
    
    for layer_idx, attn_weights in enumerate(attention_outputs):
        # (batch, n_heads, seq_len, seq_len)
        attn_weights = attn_weights[0]  # Remove batch
        
        print(f"\nLayer {layer_idx}:")
        for head_idx in range(config.n_heads):
            attn_pattern = attn_weights[head_idx].cpu().numpy()  # (seq_len, seq_len)
            
            # Compute entropy for each query position
            entropies = []
            for q_pos in range(attn_pattern.shape[0]):
                probs = attn_pattern[q_pos, :q_pos+1]  # Causal mask
                probs = probs / (probs.sum() + 1e-10)  # Normalize
                # Entropy: -sum(p * log(p))
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropies.append(entropy)
            
            avg_entropy = np.mean(entropies)
            max_entropy = np.log(max_seq_len)  # Maximum possible entropy
            normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0
            
            print(f"  Head {head_idx}: avg_entropy={avg_entropy:.4f} "
                  f"(normalized={normalized_entropy:.4f}, 1.0=uniform)")


def analyze_attention_focus(model, tokenizer, sample_text, config, max_seq_len=32):
    """Analyze where attention is actually focusing."""
    
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
    
    print("\n" + "="*60)
    print("Attention Focus Analysis")
    print("="*60)
    print("(What positions does each query attend to most?)")
    
    for layer_idx, attn_weights in enumerate(attention_outputs):
        attn_weights = attn_weights[0]
        
        print(f"\nLayer {layer_idx}:")
        for head_idx in range(config.n_heads):
            attn_pattern = attn_weights[head_idx].cpu().numpy()
            
            # For each query position, find max attention
            max_attentions = []
            for q_pos in range(attn_pattern.shape[0]):
                # Only look at positions <= q_pos (causal)
                causal_attn = attn_pattern[q_pos, :q_pos+1]
                max_k_pos = np.argmax(causal_attn)
                max_val = causal_attn[max_k_pos]
                max_attentions.append((max_k_pos, max_val))
            
            # Analyze patterns
            avg_max_attn = np.mean([v for _, v in max_attentions])
            # How often does it attend to the immediate previous position?
            prev_pos_count = sum(1 for k_pos, _ in max_attentions if k_pos == q_pos - 1)
            prev_pos_ratio = prev_pos_count / len(max_attentions) if max_attentions else 0
            
            print(f"  Head {head_idx}:")
            print(f"    Avg max attention: {avg_max_attn:.4f}")
            print(f"    Attends to prev position: {prev_pos_ratio:.2%}")
            print(f"    Sample max positions: {[k for k, _ in max_attentions[:10]]}")


def compare_query_key_similarity(model, tokenizer, sample_text, config, max_seq_len=32):
    """Compare Q and K embeddings to see if they're similar."""
    
    tokens = tokenizer.encode(sample_text)[:max_seq_len]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    with torch.no_grad():
        embeddings = model.token_embedding(input_ids)  # (batch, seq_len, embed_dim)
        embeddings = embeddings[0]  # (seq_len, embed_dim)
    
    # Get Q and K projections for first layer, first head
    block = model.blocks[0]
    W_Q = block.attention.W_Q[0]  # (embed_dim, head_dim)
    W_K = block.attention.W_K[0]  # (embed_dim, head_dim)
    
    Q = embeddings @ W_Q  # (seq_len, head_dim)
    K = embeddings @ W_K  # (seq_len, head_dim)
    
    # Compute similarity matrix
    similarity = Q @ K.T  # (seq_len, seq_len)
    
    print("\n" + "="*60)
    print("Query-Key Similarity Analysis")
    print("="*60)
    print(f"Q @ K^T shape: {similarity.shape}")
    print(f"Mean similarity: {similarity.mean():.4f}")
    print(f"Std similarity: {similarity.std():.4f}")
    print(f"Max similarity: {similarity.max():.4f}")
    print(f"Min similarity: {similarity.min():.4f}")
    
    # Check if Q and K are similar (would indicate identity-like attention)
    Q_norm = Q / (Q.norm(dim=1, keepdim=True) + 1e-8)
    K_norm = K / (K.norm(dim=1, keepdim=True) + 1e-8)
    cosine_sim = (Q_norm @ K_norm.T).mean()
    print(f"\nAverage cosine similarity (Q, K): {cosine_sim:.4f}")
    print("(High similarity -> attention scores depend mainly on embedding similarity)")


def analyze_weight_structure(attention_weights, base_embeddings):
    """Analyze the structure of the weight matrices."""
    
    print("\n" + "="*60)
    print("Weight Matrix Structure Analysis")
    print("="*60)
    
    for layer_idx, (W_Q, W_K, W_V, W_O) in attention_weights.items():
        print(f"\nLayer {layer_idx}:")
        
        # Check if W_Q and W_K are similar
        W_Q_avg = W_Q.mean(dim=0)  # Average across heads
        W_K_avg = W_K.mean(dim=0)
        
        QK_sim = torch.cosine_similarity(
            W_Q_avg.flatten().unsqueeze(0),
            W_K_avg.flatten().unsqueeze(0)
        ).item()
        print(f"  W_Q vs W_K cosine similarity: {QK_sim:.4f}")
        
        # Check rank (effective dimensionality)
        for name, W in [('W_Q', W_Q_avg), ('W_K', W_K_avg), ('W_V', W_V.mean(dim=0)), ('W_O', W_O)]:
            U, S, V = torch.svd(W)
            # Count significant singular values (> 1% of max)
            threshold = S[0] * 0.01
            rank = (S > threshold).sum().item()
            print(f"  {name} effective rank: {rank}/{min(W.shape)} "
                  f"(top 5 svals: {S[:5].tolist()})")


def main():
    """Run diagnostic analysis."""
    
    print("="*60)
    print("Attention Pattern Diagnostic Analysis")
    print("="*60)
    
    # Load data
    print("\nLoading FineWeb data...")
    builder = CorpusMatrixBuilder()
    token_batches = list(builder.stream_token_batches(max_batches=20))
    texts = [builder.tokenizer.decode(b) for b in token_batches]
    
    config = PicoModelConfig(n_layers=4, n_heads=4, embedding_dim=256, head_dim=64, mlp_dim=1024)
    tokenizer = PicoTokenizer()
    
    # Compute weights
    print("\nComputing attention weights...")
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
    
    # Create model
    torch.manual_seed(42)
    model = PicoGPTSparse(config, attention_weights, base_embeddings=base_embeddings)
    
    # Run analyses
    sample_text = texts[0][:500]
    
    analyze_weight_structure(attention_weights, base_embeddings)
    compare_query_key_similarity(model, tokenizer, sample_text, config)
    analyze_attention_entropy(model, tokenizer, sample_text, config)
    analyze_attention_focus(model, tokenizer, sample_text, config)
    
    print("\n" + "="*60)
    print("Diagnostic Analysis Complete")
    print("="*60)


if __name__ == "__main__":
    main()


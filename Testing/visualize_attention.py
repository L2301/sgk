"""
Visualize Attention Patterns

Analyzes and visualizes the attention patterns from corpus-derived weights.
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


def visualize_attention_weights(
    attention_weights: dict,
    base_embeddings: torch.Tensor,
    config: PicoModelConfig,
    output_dir: Path,
):
    """Visualize the attention weight matrices themselves."""
    
    print("\n" + "="*60)
    print("Visualizing Attention Weight Matrices")
    print("="*60)
    
    fig, axes = plt.subplots(config.n_layers, 4, figsize=(20, 5*config.n_layers))
    if config.n_layers == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx in range(config.n_layers):
        W_Q, W_K, W_V, W_O = attention_weights[layer_idx]
        
        # W_Q: (n_heads, embedding_dim, head_dim)
        # W_K: (n_heads, embedding_dim, head_dim)
        # W_V: (n_heads, head_dim, embedding_dim)
        # W_O: (embedding_dim, n_heads * head_dim)
        
        # Average across heads for visualization
        W_Q_avg = W_Q.mean(dim=0).detach().numpy()  # (embedding_dim, head_dim)
        W_K_avg = W_K.mean(dim=0).detach().numpy()
        W_V_avg = W_V.mean(dim=0).detach().numpy()
        W_O_avg = W_O.detach().numpy()  # (embedding_dim, n_heads * head_dim)
        
        # Plot W_Q
        im1 = axes[layer_idx, 0].imshow(W_Q_avg, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[layer_idx, 0].set_title(f'Layer {layer_idx} - W_Q (avg across heads)\n{W_Q_avg.shape}')
        axes[layer_idx, 0].set_xlabel('Head Dim')
        axes[layer_idx, 0].set_ylabel('Embedding Dim')
        plt.colorbar(im1, ax=axes[layer_idx, 0])
        
        # Plot W_K
        im2 = axes[layer_idx, 1].imshow(W_K_avg, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[layer_idx, 1].set_title(f'Layer {layer_idx} - W_K (avg across heads)\n{W_K_avg.shape}')
        axes[layer_idx, 1].set_xlabel('Head Dim')
        axes[layer_idx, 1].set_ylabel('Embedding Dim')
        plt.colorbar(im2, ax=axes[layer_idx, 1])
        
        # Plot W_V
        im3 = axes[layer_idx, 2].imshow(W_V_avg, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[layer_idx, 2].set_title(f'Layer {layer_idx} - W_V (avg across heads)\n{W_V_avg.shape}')
        axes[layer_idx, 2].set_xlabel('Embedding Dim')
        axes[layer_idx, 2].set_ylabel('Head Dim')
        plt.colorbar(im3, ax=axes[layer_idx, 2])
        
        # Plot W_O
        im4 = axes[layer_idx, 3].imshow(W_O_avg, aspect='auto', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        axes[layer_idx, 3].set_title(f'Layer {layer_idx} - W_O\n{W_O_avg.shape}')
        axes[layer_idx, 3].set_xlabel('Heads * Head Dim')
        axes[layer_idx, 3].set_ylabel('Embedding Dim')
        plt.colorbar(im4, ax=axes[layer_idx, 3])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_weight_matrices.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'attention_weight_matrices.png'}")
    plt.close()


def visualize_attention_patterns(
    model: PicoGPTSparse,
    tokenizer: PicoTokenizer,
    sample_text: str,
    config: PicoModelConfig,
    output_dir: Path,
    max_seq_len: int = 32,
):
    """Visualize actual attention patterns on a sample sequence."""
    
    print("\n" + "="*60)
    print("Visualizing Attention Patterns on Sample Text")
    print("="*60)
    print(f"Sample text: {sample_text[:100]}...")
    
    # Tokenize
    tokens = tokenizer.encode(sample_text)[:max_seq_len]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    # Get attention weights by hooking into the model
    attention_outputs = []
    
    def attention_hook(module, input, output):
        attn_out, attn_weights = output
        attention_outputs.append(attn_weights.detach())
    
    # Register hooks
    hooks = []
    for block in model.blocks:
        hooks.append(block.attention.register_forward_hook(attention_hook))
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize attention patterns for each layer
    n_layers = len(attention_outputs)
    fig, axes = plt.subplots(n_layers, config.n_heads, figsize=(4*config.n_heads, 4*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if config.n_heads == 1:
        axes = axes.reshape(-1, 1)
    
    seq_len = len(tokens)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    
    for layer_idx, attn_weights in enumerate(attention_outputs):
        # attn_weights: (batch, n_heads, seq_len, seq_len)
        attn_weights = attn_weights[0]  # Remove batch dim
        
        for head_idx in range(config.n_heads):
            attn_pattern = attn_weights[head_idx].cpu().numpy()  # (seq_len, seq_len)
            
            ax = axes[layer_idx, head_idx]
            im = ax.imshow(attn_pattern, aspect='auto', cmap='Blues', vmin=0, vmax=1)
            ax.set_title(f'Layer {layer_idx}, Head {head_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Add token labels (every few tokens to avoid clutter)
            step = max(1, seq_len // 10)
            tick_positions = list(range(0, seq_len, step))
            tick_labels = [token_strs[i] if i < len(token_strs) else '' for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=6)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=6)
            
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_patterns.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'attention_patterns.png'}")
    plt.close()


def analyze_weight_statistics(
    attention_weights: dict,
    base_embeddings: torch.Tensor,
    output_dir: Path,
):
    """Analyze statistics of the attention weights."""
    
    print("\n" + "="*60)
    print("Analyzing Weight Statistics")
    print("="*60)
    
    stats = []
    
    for layer_idx, (W_Q, W_K, W_V, W_O) in attention_weights.items():
        stats.append({
            'layer': layer_idx,
            'W_Q_mean': W_Q.mean().item(),
            'W_Q_std': W_Q.std().item(),
            'W_Q_norm': W_Q.norm().item(),
            'W_K_mean': W_K.mean().item(),
            'W_K_std': W_K.std().item(),
            'W_K_norm': W_K.norm().item(),
            'W_V_mean': W_V.mean().item(),
            'W_V_std': W_V.std().item(),
            'W_V_norm': W_V.norm().item(),
            'W_O_mean': W_O.mean().item(),
            'W_O_std': W_O.std().item(),
            'W_O_norm': W_O.norm().item(),
        })
        
        print(f"\nLayer {layer_idx}:")
        print(f"  W_Q: mean={W_Q.mean():.6f}, std={W_Q.std():.6f}, norm={W_Q.norm():.4f}")
        print(f"  W_K: mean={W_K.mean():.6f}, std={W_K.std():.6f}, norm={W_K.norm():.4f}")
        print(f"  W_V: mean={W_V.mean():.6f}, std={W_V.std():.6f}, norm={W_V.norm():.4f}")
        print(f"  W_O: mean={W_O.mean():.6f}, std={W_O.std():.6f}, norm={W_O.norm():.4f}")
    
    # Plot statistics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    layers = [s['layer'] for s in stats]
    
    # Mean values
    axes[0, 0].plot(layers, [s['W_Q_mean'] for s in stats], 'o-', label='W_Q')
    axes[0, 0].plot(layers, [s['W_K_mean'] for s in stats], 's-', label='W_K')
    axes[0, 0].plot(layers, [s['W_V_mean'] for s in stats], '^-', label='W_V')
    axes[0, 0].plot(layers, [s['W_O_mean'] for s in stats], 'd-', label='W_O')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Mean')
    axes[0, 0].set_title('Weight Means by Layer')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std values
    axes[0, 1].plot(layers, [s['W_Q_std'] for s in stats], 'o-', label='W_Q')
    axes[0, 1].plot(layers, [s['W_K_std'] for s in stats], 's-', label='W_K')
    axes[0, 1].plot(layers, [s['W_V_std'] for s in stats], '^-', label='W_V')
    axes[0, 1].plot(layers, [s['W_O_std'] for s in stats], 'd-', label='W_O')
    axes[0, 1].set_xlabel('Layer')
    axes[0, 1].set_ylabel('Std')
    axes[0, 1].set_title('Weight Stds by Layer')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Norms
    axes[1, 0].plot(layers, [s['W_Q_norm'] for s in stats], 'o-', label='W_Q')
    axes[1, 0].plot(layers, [s['W_K_norm'] for s in stats], 's-', label='W_K')
    axes[1, 0].plot(layers, [s['W_V_norm'] for s in stats], '^-', label='W_V')
    axes[1, 0].plot(layers, [s['W_O_norm'] for s in stats], 'd-', label='W_O')
    axes[1, 0].set_xlabel('Layer')
    axes[1, 0].set_ylabel('Frobenius Norm')
    axes[1, 0].set_title('Weight Norms by Layer')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Embedding statistics
    emb_mean = base_embeddings.mean().item()
    emb_std = base_embeddings.std().item()
    emb_norm = base_embeddings.norm().item()
    
    axes[1, 1].bar(['Mean', 'Std', 'Norm'], [abs(emb_mean), emb_std, emb_norm])
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Base Embedding Statistics')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'weight_statistics.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'weight_statistics.png'}")
    plt.close()


def compare_attention_outputs(
    sparse_model: PicoGPTSparse,
    zero_model: PicoGPTSparse,
    tokenizer: PicoTokenizer,
    sample_text: str,
    config: PicoModelConfig,
    output_dir: Path,
    max_seq_len: int = 32,
):
    """Compare attention outputs between sparse and zero models."""
    
    print("\n" + "="*60)
    print("Comparing Sparse vs Zero Attention Outputs")
    print("="*60)
    
    tokens = tokenizer.encode(sample_text)[:max_seq_len]
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    # Get embeddings
    with torch.no_grad():
        sparse_embeds = sparse_model.token_embedding(input_ids)
        zero_embeds = zero_model.token_embedding(input_ids)
    
    # Get attention outputs
    sparse_attn_outputs = []
    zero_attn_outputs = []
    
    def sparse_hook(module, input, output):
        attn_out, _ = output
        sparse_attn_outputs.append(attn_out.detach())
    
    def zero_hook(module, input, output):
        attn_out, _ = output
        zero_attn_outputs.append(attn_out.detach())
    
    # Register hooks
    sparse_hooks = [block.attention.register_forward_hook(sparse_hook) for block in sparse_model.blocks]
    zero_hooks = [block.attention.register_forward_hook(zero_hook) for block in zero_model.blocks]
    
    # Forward pass
    with torch.no_grad():
        _ = sparse_model(input_ids)
        _ = zero_model(input_ids)
    
    # Remove hooks
    for hook in sparse_hooks + zero_hooks:
        hook.remove()
    
    # Compare outputs
    fig, axes = plt.subplots(len(sparse_attn_outputs), 2, figsize=(12, 4*len(sparse_attn_outputs)))
    if len(sparse_attn_outputs) == 1:
        axes = axes.reshape(1, -1)
    
    for layer_idx, (sparse_out, zero_out) in enumerate(zip(sparse_attn_outputs, zero_attn_outputs)):
        # (batch, seq_len, embedding_dim)
        sparse_out = sparse_out[0].cpu().numpy()
        zero_out = zero_out[0].cpu().numpy()
        
        # Plot magnitude per position
        sparse_mag = np.linalg.norm(sparse_out, axis=1)
        zero_mag = np.linalg.norm(zero_out, axis=1)
        
        axes[layer_idx, 0].plot(sparse_mag, 'o-', label='Sparse', alpha=0.7)
        axes[layer_idx, 0].plot(zero_mag, 's-', label='Zero', alpha=0.7)
        axes[layer_idx, 0].set_xlabel('Position')
        axes[layer_idx, 0].set_ylabel('Output Magnitude')
        axes[layer_idx, 0].set_title(f'Layer {layer_idx} - Output Magnitude')
        axes[layer_idx, 0].legend()
        axes[layer_idx, 0].grid(True, alpha=0.3)
        
        # Plot difference
        diff = sparse_out - zero_out
        diff_mag = np.linalg.norm(diff, axis=1)
        axes[layer_idx, 1].plot(diff_mag, 'o-', color='red', alpha=0.7)
        axes[layer_idx, 1].set_xlabel('Position')
        axes[layer_idx, 1].set_ylabel('Difference Magnitude')
        axes[layer_idx, 1].set_title(f'Layer {layer_idx} - Sparse - Zero')
        axes[layer_idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_output_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'attention_output_comparison.png'}")
    plt.close()
    
    # Print summary
    print(f"\nOutput Magnitude Summary:")
    for layer_idx, (sparse_out, zero_out) in enumerate(zip(sparse_attn_outputs, zero_attn_outputs)):
        sparse_mag = sparse_out.norm().item()
        zero_mag = zero_out.norm().item()
        diff_mag = (sparse_out - zero_out).norm().item()
        print(f"  Layer {layer_idx}: Sparse={sparse_mag:.4f}, Zero={zero_mag:.4f}, Diff={diff_mag:.4f}")


def main():
    """Main visualization script."""
    
    # Setup
    output_dir = Path("attention_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("Attention Pattern Visualization")
    print("="*60)
    
    # Load FineWeb data
    print("\nLoading FineWeb data...")
    builder = CorpusMatrixBuilder()
    token_batches = list(builder.stream_token_batches(max_batches=20))
    texts = [builder.tokenizer.decode(b) for b in token_batches]
    
    # Config
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
        verbose=True,
        normalize_weights=True,
        target_std=0.01,
    )
    
    # Create models
    print("\nCreating models...")
    torch.manual_seed(42)
    sparse_model = PicoGPTSparse(config, attention_weights, base_embeddings=base_embeddings)
    
    torch.manual_seed(42)
    zero_weights = {
        i: (torch.zeros(4, 256, 64), torch.zeros(4, 256, 64), 
            torch.zeros(4, 64, 256), torch.zeros(256, 256))
        for i in range(4)
    }
    zero_model = PicoGPTSparse(config, zero_weights, base_embeddings=base_embeddings)
    
    # Visualizations
    visualize_attention_weights(attention_weights, base_embeddings, config, output_dir)
    analyze_weight_statistics(attention_weights, base_embeddings, output_dir)
    
    # Use a sample text for attention patterns
    sample_text = texts[0][:500]  # First 500 chars
    visualize_attention_patterns(sparse_model, tokenizer, sample_text, config, output_dir)
    compare_attention_outputs(sparse_model, zero_model, tokenizer, sample_text, config, output_dir)
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


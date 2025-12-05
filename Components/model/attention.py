"""
Multi-Head Attention with Frozen Weights

Implements causal self-attention where W_Q, W_K, W_V, W_O matrices
are pre-computed from corpus statistics and frozen during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

from config import ModelConfig


class FrozenMultiHeadAttention(nn.Module):
    """
    Multi-head attention with frozen W_Q, W_K, W_V, W_O weights.
    
    Each head h uses weights computed with skip value s=h.
    The layer uses n-gram size n=layer_idx+1.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        W_Q: torch.Tensor,  # (n_heads, embedding_dim, head_dim)
        W_K: torch.Tensor,  # (n_heads, embedding_dim, head_dim)
        W_V: torch.Tensor,  # (n_heads, head_dim, embedding_dim)
        W_O: torch.Tensor,  # (embedding_dim, n_heads * head_dim)
    ):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.embedding_dim = config.embedding_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Register frozen weights as buffers (not parameters)
        # W_Q, W_K: (n_heads, embedding_dim, head_dim)
        # W_V: (n_heads, head_dim, embedding_dim) - note: transposed from typical
        # W_O: (embedding_dim, n_heads * head_dim)
        self.register_buffer('W_Q', W_Q)
        self.register_buffer('W_K', W_K)
        self.register_buffer('W_V', W_V)
        self.register_buffer('W_O', W_O)
        
        # Causal mask will be created on first forward pass
        self.register_buffer('causal_mask', None)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask."""
        # Lower triangular mask: position i can attend to positions <= i
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # Convert to attention mask format: 0 for allowed, -inf for blocked
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
            attention_mask: Optional additional mask
        
        Returns:
            Tuple of (output, attention_weights)
            - output: (batch_size, seq_len, embedding_dim)
            - attention_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Create or resize causal mask
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            self.causal_mask = self._create_causal_mask(seq_len, x.device)
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # Compute Q, K, V for all heads
        # x: (batch, seq, embed) 
        # W_Q: (n_heads, embed, head_dim)
        # Q: (batch, n_heads, seq, head_dim)
        Q = torch.einsum('bse,hed->bhsd', x, self.W_Q)
        K = torch.einsum('bse,hed->bhsd', x, self.W_K)
        
        # W_V has shape (n_heads, head_dim, embed) - maps embed -> head_dim
        # V: (batch, n_heads, seq, head_dim)
        V = torch.einsum('bse,hde->bhsd', x, self.W_V)
        
        # Compute attention scores
        # (batch, n_heads, seq, head_dim) @ (batch, n_heads, head_dim, seq)
        # -> (batch, n_heads, seq, seq)
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) * self.scale
        
        # Apply causal mask
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply additional attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        # (batch, n_heads, seq, seq) @ (batch, n_heads, seq, head_dim)
        # -> (batch, n_heads, seq, head_dim)
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, V)
        
        # Concatenate heads: (batch, seq, n_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection
        # (batch, seq, n_heads * head_dim) @ (n_heads * head_dim, embed)
        # -> (batch, seq, embed)
        output = torch.einsum('bsh,he->bse', attn_output, self.W_O.T)
        
        return output, attn_weights


class FrozenAttentionLayer(nn.Module):
    """
    Complete attention layer with frozen attention weights.
    Includes pre-layer norm (GPT-2 style).
    """
    
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        W_Q: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        W_O: torch.Tensor,
    ):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(
            config.embedding_dim,
            eps=config.layer_norm_eps
        )
        
        self.attention = FrozenMultiHeadAttention(
            config, layer_idx, W_Q, W_K, W_V, W_O
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (batch_size, seq_len, embedding_dim)
            attention_mask: Optional attention mask
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Pre-norm
        normed = self.layer_norm(x)
        
        # Attention with residual
        attn_out, attn_weights = self.attention(normed, attention_mask)
        output = x + attn_out
        
        return output, attn_weights


def create_attention_weights_from_matrices(
    W_Q_dict: Dict[Tuple[int, int], torch.Tensor],
    W_K_dict: Dict[Tuple[int, int], torch.Tensor],
    W_V_dict: Dict[Tuple[int, int], torch.Tensor],
    W_O_dict: Dict[Tuple[int, int], torch.Tensor],
    layer_idx: int,
    n_heads: int,
    embedding_dim: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Organize weight matrices for a single layer.
    
    Layer uses n = layer_idx + 1.
    Each head h uses s = h.
    
    Args:
        W_Q_dict: Dict mapping (n, s) to W_Q matrix
        W_K_dict: Dict mapping (n, s) to W_K matrix
        W_V_dict: Dict mapping (n, s) to W_V matrix
        W_O_dict: Dict mapping (n, s) to W_O matrix
        layer_idx: Layer index (0-based)
        n_heads: Number of attention heads
        embedding_dim: Embedding dimension
        head_dim: Head dimension
    
    Returns:
        Tuple of (W_Q, W_K, W_V, W_O) tensors for the layer
    """
    n = layer_idx + 1  # Layer 0 uses n=1, layer 1 uses n=2, etc.
    
    # Stack heads
    W_Q_heads = []
    W_K_heads = []
    W_V_heads = []
    W_O_heads = []
    
    for h in range(n_heads):
        s = h  # Head h uses skip value s=h
        key = (n, s)
        
        if key in W_Q_dict:
            W_Q_heads.append(torch.from_numpy(W_Q_dict[key]).float())
            W_K_heads.append(torch.from_numpy(W_K_dict[key]).float())
            W_V_heads.append(torch.from_numpy(W_V_dict[key]).float())
            W_O_heads.append(torch.from_numpy(W_O_dict[key]).float())
        else:
            # Initialize with zeros if not available
            print(f"Warning: No weights for (n={n}, s={s}), using zeros")
            W_Q_heads.append(torch.zeros(embedding_dim, head_dim))
            W_K_heads.append(torch.zeros(embedding_dim, head_dim))
            W_V_heads.append(torch.zeros(head_dim, embedding_dim))
            W_O_heads.append(torch.zeros(embedding_dim, head_dim))
    
    # Stack: (n_heads, embedding_dim, head_dim) for Q, K
    W_Q = torch.stack(W_Q_heads, dim=0)
    W_K = torch.stack(W_K_heads, dim=0)
    
    # W_V: (n_heads, head_dim, embedding_dim)
    W_V = torch.stack(W_V_heads, dim=0)
    
    # W_O: concatenate horizontally -> (embedding_dim, n_heads * head_dim)
    W_O = torch.cat(W_O_heads, dim=1)
    
    return W_Q, W_K, W_V, W_O


if __name__ == "__main__":
    # Test the attention module
    config = ModelConfig()
    
    print("Testing FrozenMultiHeadAttention...")
    
    # Create random weights for testing
    W_Q = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
    W_K = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
    W_V = torch.randn(config.n_heads, config.head_dim, config.embedding_dim)
    W_O = torch.randn(config.embedding_dim, config.n_heads * config.head_dim)
    
    # Create attention layer
    attn = FrozenAttentionLayer(config, layer_idx=0, W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.embedding_dim)
    
    output, weights = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    # Verify shapes
    assert output.shape == x.shape, "Output shape mismatch"
    assert weights.shape == (batch_size, config.n_heads, seq_len, seq_len), "Weights shape mismatch"
    
    # Check causality: attention weights should be lower triangular
    # (allowing only attending to past positions)
    for b in range(batch_size):
        for h in range(config.n_heads):
            w = weights[b, h].detach()
            # Upper triangle (excluding diagonal) should be ~0
            upper = torch.triu(w, diagonal=1)
            assert upper.abs().max() < 1e-5, "Causal mask not working"
    
    print("✓ All attention tests passed!")
    
    # Count parameters
    total_params = sum(p.numel() for p in attn.parameters())
    trainable_params = sum(p.numel() for p in attn.parameters() if p.requires_grad)
    frozen_params = sum(b.numel() for b in attn.buffers())
    
    print(f"\nParameter counts:")
    print(f"  Trainable (LayerNorm): {trainable_params:,}")
    print(f"  Frozen (Attention weights): {frozen_params:,}")


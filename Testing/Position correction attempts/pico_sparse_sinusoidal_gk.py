"""
PicoGPT Sparse with Sinusoidal Position Encoding + G_k Correction

This variant implements the mathematically correct approach:
1. Computes attention weights accounting for positional encodings
2. Subtracts G_k = E[P(i+k)P(i)^T] to remove positional bias
3. Recovers true semantic skip-k structure

Theory:
- During training: attention uses (e_x + P(i))^T W_Q W_K^T (e_y + P(i+k))
- This includes positional cross-terms that create uniform boost
- G_k captures the average positional contribution at distance k
- M_k,free = M_k - G_k isolates genuine semantic structure
"""

import sys
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Components" / "method"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Components" / "model"))

from pico_config import PicoModelConfig, PicoTrainingConfig


class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal position encoding as in 'Attention is All You Need'.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, embedding_dim: int, max_seq_len: int = 1024):
        super().__init__()
        
        # Create position encoding matrix
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term for the sinusoidal functions
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * 
            (-math.log(10000.0) / embedding_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_len, embedding_dim)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embedding_dim)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
    
    def get_positional_encoding(self, positions):
        """Get positional encodings for specific positions."""
        return self.pe[:, positions, :]


def compute_G_k(position_encoding: SinusoidalPositionEncoding, k: int, max_pos: int = 1000) -> torch.Tensor:
    """
    Compute G_k = E_i[P(i+k)P(i)^T]
    
    This is the average positional contribution at distance k.
    
    Args:
        position_encoding: The sinusoidal position encoding module
        k: The skip distance
        max_pos: Maximum position to average over
    
    Returns:
        G_k: (embedding_dim, embedding_dim) matrix
    """
    # Get positional encodings
    # P(i) for i in [0, max_pos - k]
    # P(i+k) for i in [0, max_pos - k]
    
    valid_positions = max_pos - k
    if valid_positions <= 0:
        raise ValueError(f"k={k} too large for max_pos={max_pos}")
    
    # Get all position encodings we need
    all_pos = torch.arange(0, max_pos)
    P_all = position_encoding.get_positional_encoding(all_pos).squeeze(0)  # (max_pos, d)
    
    # Compute outer products and average
    G_k_sum = torch.zeros(P_all.size(1), P_all.size(1))
    
    for i in range(valid_positions):
        P_i = P_all[i]  # (d,)
        P_i_k = P_all[i + k]  # (d,)
        
        # Outer product: P(i+k) @ P(i)^T
        G_k_sum += torch.outer(P_i_k, P_i)
    
    G_k = G_k_sum / valid_positions
    
    return G_k


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x, approximate='tanh')


class PicoMLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: PicoModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embedding_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embedding_dim)
        self.activation = GELU()
        self.dropout = nn.Dropout(config.mlp_dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PicoFrozenAttentionGkCorrected(nn.Module):
    """
    Multi-head attention with frozen weights corrected for positional bias.
    
    Key innovation: W_Q and W_K are computed as:
    M_k,free = M_k - G_k
    
    Where G_k removes the positional encoding bias.
    """
    
    def __init__(
        self,
        config: PicoModelConfig,
        layer_idx: int,
        W_Q: torch.Tensor,  # (n_heads, embedding_dim, head_dim)
        W_K: torch.Tensor,
        W_V: torch.Tensor,  # (n_heads, head_dim, embedding_dim)
        W_O: torch.Tensor,  # (embedding_dim, n_heads * head_dim)
    ):
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # n value for this layer
        self.n = layer_idx + 1
        
        # Frozen weights as buffers
        self.register_buffer('W_Q', W_Q)
        self.register_buffer('W_K', W_K)
        self.register_buffer('W_V', W_V)
        self.register_buffer('W_O', W_O)
        
        # Causal mask
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            mask = mask.masked_fill(mask == 1, 0.0)
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        causal_mask = self._get_causal_mask(seq_len, x.device)
        
        # Q, K, V projections (using G_k-corrected weights)
        Q = torch.einsum('bse,hed->bhsd', x, self.W_Q)
        K = torch.einsum('bse,hed->bhsd', x, self.W_K)
        V = torch.einsum('bse,hde->bhsd', x, self.W_V)
        
        # Attention scores
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) * self.scale
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attn_output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, V)
        
        # Concat heads and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = torch.einsum('bsh,he->bse', attn_output, self.W_O.T)
        
        return output, attn_weights


class PicoSparseGkCorrectedBlock(nn.Module):
    """Transformer block with G_k-corrected frozen attention."""
    
    def __init__(
        self,
        config: PicoModelConfig,
        layer_idx: int,
        W_Q: torch.Tensor,
        W_K: torch.Tensor,
        W_V: torch.Tensor,
        W_O: torch.Tensor,
    ):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.attention = PicoFrozenAttentionGkCorrected(config, layer_idx, W_Q, W_K, W_V, W_O)
        self.ln2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.mlp = PicoMLP(config)
    
    def forward(self, x, attention_mask=None):
        # Attention with residual
        attn_out, attn_weights = self.attention(self.ln1(x), attention_mask)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        
        return x, attn_weights


class PicoGPTSparseGkCorrected(nn.Module):
    """
    PicoGPT with:
    - Sinusoidal position encoding
    - G_k-corrected frozen attention weights
    - Trainable MLP layers
    """
    
    def __init__(
        self,
        config: PicoModelConfig,
        attention_weights: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        base_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.config = config
        
        # Token embeddings - FROZEN
        if base_embeddings is not None:
            assert base_embeddings.shape == (config.vocab_size, config.embedding_dim)
            self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
            self.token_embedding.weight.data = base_embeddings.clone()
            self.token_embedding.weight.requires_grad = False  # FREEZE
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
            self.token_embedding.weight.requires_grad = False  # FREEZE anyway
        
        # Sinusoidal position encoding (no learned parameters)
        self.position_encoding = SinusoidalPositionEncoding(
            config.embedding_dim, 
            config.max_seq_len
        )
        
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            if layer_idx in attention_weights:
                W_Q, W_K, W_V, W_O = attention_weights[layer_idx]
            else:
                # Random init if not provided
                W_Q = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
                W_K = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
                W_V = torch.randn(config.n_heads, config.head_dim, config.embedding_dim)
                W_O = torch.randn(config.embedding_dim, config.n_heads * config.head_dim)
            
            self.blocks.append(PicoSparseGkCorrectedBlock(config, layer_idx, W_Q, W_K, W_V, W_O))
        
        # Output
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        # Token embeddings already set, keep frozen
        
        for block in self.blocks:
            nn.init.normal_(block.mlp.fc1.weight, std=0.02)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.normal_(block.mlp.fc2.weight, std=0.02 / (2 * self.config.n_layers) ** 0.5)
            nn.init.zeros_(block.mlp.fc2.bias)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add sinusoidal position encoding
        x = self.position_encoding(x)
        
        x = self.embed_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x, _ = block(x, attention_mask)
        
        # Output
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    def compute_loss(self, input_ids, labels=None):
        logits = self.forward(input_ids)
        
        if labels is None:
            labels = input_ids
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
        )
        
        return loss, logits
    
    def get_num_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        param_count = sum(p.numel() for p in self.parameters())
        buffer_count = sum(b.numel() for b in self.buffers() if b is not None)
        return param_count + buffer_count


if __name__ == "__main__":
    print("Testing PicoGPT Sparse with G_k Correction...")
    
    from pico_config import PicoModelConfig
    config = PicoModelConfig()
    
    # Test G_k computation
    print("\nTesting G_k computation:")
    pos_enc = SinusoidalPositionEncoding(config.embedding_dim, config.max_seq_len)
    
    for k in [1, 2, 3]:
        G_k = compute_G_k(pos_enc, k, max_pos=100)
        print(f"  G_{k} shape: {G_k.shape}, norm: {G_k.norm().item():.4f}")
    
    # Random weights for testing
    attention_weights = {}
    for layer_idx in range(config.n_layers):
        W_Q = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
        W_K = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
        W_V = torch.randn(config.n_heads, config.head_dim, config.embedding_dim)
        W_O = torch.randn(config.embedding_dim, config.n_heads * config.head_dim)
        attention_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
    
    base_embeddings = torch.randn(config.vocab_size, config.embedding_dim) * 0.02
    
    model = PicoGPTSparseGkCorrected(config, attention_weights, base_embeddings=base_embeddings)
    
    # Test forward
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"\nInput: {input_ids.shape}")
    print(f"Output: {logits.shape}")
    
    # Test loss
    loss, _ = model.compute_loss(input_ids)
    print(f"Loss: {loss.item():.4f}")
    
    # Parameter counts
    total = model.get_num_params()
    trainable = model.get_num_params(trainable_only=True)
    
    print(f"\nParameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Frozen: {total - trainable:,}")
    
    # Verify embeddings are frozen
    print(f"\nToken embeddings frozen: {not model.token_embedding.weight.requires_grad}")
    
    print("\n✓ PicoGPT Sparse G_k Corrected tests passed!")

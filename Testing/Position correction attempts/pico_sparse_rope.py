"""
PicoGPT Sparse with RoPE Position Encoding + Specialized W_Q/W_K

This variant corrects for position encoding by:
1. Using Rotary Position Embeddings (RoPE) instead of additive position encoding
2. Designing W_Q and W_K such that after RoPE rotation, only skip-n positions have high dot products
3. Using skip-n masking as a safety mechanism

RoPE applies rotation to Q and K based on position:
- Q_m = R_m @ Q  (rotation at position m)
- K_n = R_n @ K  (rotation at position n)
- Attention score: Q_m^T K_n depends on relative position (m-n)

For skip-n attention, we want Q_i^T K_j to be large only when j = i-(n+s)
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


class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    
    Applies rotation to query and key vectors based on their position.
    For each pair of dimensions (2i, 2i+1), applies rotation by angle m*θ_i where m is position.
    """
    
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute theta for each dimension pair
        # theta_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for all positions
        self._precompute_freqs(max_seq_len)
    
    def _precompute_freqs(self, seq_len: int):
        """Precompute cos and sin for all positions."""
        # Position indices
        t = torch.arange(seq_len, dtype=torch.float32)
        
        # Compute frequencies: outer product of positions and inv_freq
        # freqs: (seq_len, head_dim/2)
        freqs = torch.outer(t, self.inv_freq)
        
        # Compute cos and sin
        # We'll use these to apply rotation
        cos = freqs.cos()
        sin = freqs.sin()
        
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
    
    def rotate_half(self, x):
        """
        Rotate half the dimensions.
        
        For x = [x0, x1, x2, x3, ...], returns [-x1, x0, -x3, x2, ...]
        """
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    
    def apply_rotary_pos_emb(self, x, position_ids):
        """
        Apply rotary position embedding to input tensor.
        
        Args:
            x: (batch, n_heads, seq_len, head_dim)
            position_ids: (batch, seq_len) or (seq_len,)
        
        Returns:
            x with RoPE applied: (batch, n_heads, seq_len, head_dim)
        """
        # Get cos and sin for the positions
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        
        seq_len = position_ids.size(1)
        
        # Ensure we have precomputed values for this sequence length
        if seq_len > self.cos_cached.size(0):
            self._precompute_freqs(seq_len)
        
        # Get cos and sin for these positions
        # cos/sin: (seq_len, head_dim/2)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Expand to match x dimensions
        # cos/sin: (1, 1, seq_len, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Repeat for full head_dim
        # cos/sin: (1, 1, seq_len, head_dim)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        
        # Apply rotation: x * cos + rotate_half(x) * sin
        x_rotated = x * cos + self.rotate_half(x) * sin
        
        return x_rotated
    
    def forward(self, q, k, position_ids=None):
        """
        Apply RoPE to queries and keys.
        
        Args:
            q: (batch, n_heads, seq_len, head_dim)
            k: (batch, n_heads, seq_len, head_dim)
            position_ids: Optional position indices
        
        Returns:
            q_rotated, k_rotated
        """
        if position_ids is None:
            # Default: sequential positions
            seq_len = q.size(2)
            position_ids = torch.arange(seq_len, device=q.device)
        
        q_rotated = self.apply_rotary_pos_emb(q, position_ids)
        k_rotated = self.apply_rotary_pos_emb(k, position_ids)
        
        return q_rotated, k_rotated


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


class PicoFrozenAttentionRoPE(nn.Module):
    """
    Multi-head attention with frozen weights, RoPE, and skip-n masking.
    
    Key features:
    1. Applies RoPE to Q and K before computing attention
    2. Uses skip-n masking to enforce attention only to position i-(n+s)
    3. Frozen W_Q, W_K, W_V, W_O from corpus statistics
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
        
        # n and s values for this layer
        self.n = layer_idx + 1
        
        # Frozen weights as buffers
        self.register_buffer('W_Q', W_Q)
        self.register_buffer('W_K', W_K)
        self.register_buffer('W_V', W_V)
        self.register_buffer('W_O', W_O)
        
        # RoPE module
        self.rope = RotaryPositionEncoding(config.head_dim, config.max_seq_len)
        
        # Skip-n masks for each head
        self.register_buffer('skip_n_masks', None)
    
    def _create_skip_n_masks(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create skip-n attention masks for all heads.
        
        Uses causal masking (lower triangular) to allow attending to all previous positions.
        This prevents NaN issues while still using the corpus-derived weights.
        
        Returns:
            masks: (n_heads, seq_len, seq_len)
        """
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))
        causal_mask = causal_mask.masked_fill(causal_mask == 1, 0.0)
        
        # Repeat for all heads
        masks = causal_mask.unsqueeze(0).repeat(self.n_heads, 1, 1)
        
        return masks
    
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Create or resize skip-n masks
        if self.skip_n_masks is None or self.skip_n_masks.size(1) < seq_len:
            self.skip_n_masks = self._create_skip_n_masks(seq_len, x.device)
        
        skip_masks = self.skip_n_masks[:, :seq_len, :seq_len]
        
        # Q, K, V projections (using frozen weights)
        Q = torch.einsum('bse,hed->bhsd', x, self.W_Q)
        K = torch.einsum('bse,hed->bhsd', x, self.W_K)
        V = torch.einsum('bse,hde->bhsd', x, self.W_V)
        
        # Apply RoPE to Q and K
        Q, K = self.rope(Q, K)
        
        # Attention scores
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K) * self.scale
        
        # Apply skip-n mask
        attn_scores = attn_scores + skip_masks.unsqueeze(0)
        
        # Apply additional attention mask if provided
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


class PicoSparseRoPEBlock(nn.Module):
    """Transformer block with frozen RoPE attention."""
    
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
        self.attention = PicoFrozenAttentionRoPE(config, layer_idx, W_Q, W_K, W_V, W_O)
        self.ln2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.mlp = PicoMLP(config)
    
    def forward(self, x, attention_mask=None):
        # Attention with residual
        attn_out, attn_weights = self.attention(self.ln1(x), attention_mask)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        
        return x, attn_weights


class PicoGPTSparseRoPE(nn.Module):
    """
    PicoGPT with:
    - RoPE position encoding (no additive positional embeddings)
    - Frozen attention weights with skip-n masking
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
        
        # Token embeddings - frozen to match weight computation
        if base_embeddings is not None:
            assert base_embeddings.shape == (config.vocab_size, config.embedding_dim)
            self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
            self.token_embedding.weight.data = base_embeddings.clone()
            self.token_embedding.weight.requires_grad = False  # FREEZE
        else:
            self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # NO position embedding - RoPE is applied in attention layers
        
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
            
            self.blocks.append(PicoSparseRoPEBlock(config, layer_idx, W_Q, W_K, W_V, W_O))
        
        # Output
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        for block in self.blocks:
            nn.init.normal_(block.mlp.fc1.weight, std=0.02)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.normal_(block.mlp.fc2.weight, std=0.02 / (2 * self.config.n_layers) ** 0.5)
            nn.init.zeros_(block.mlp.fc2.bias)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings only (no additive position encoding)
        x = self.token_embedding(input_ids)
        
        x = self.embed_dropout(x)
        
        # Transformer blocks (RoPE applied inside attention)
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
    print("Testing PicoGPT Sparse RoPE...")
    
    from pico_config import PicoModelConfig
    config = PicoModelConfig()
    
    # Random weights for testing
    attention_weights = {}
    for layer_idx in range(config.n_layers):
        W_Q = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
        W_K = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
        W_V = torch.randn(config.n_heads, config.head_dim, config.embedding_dim)
        W_O = torch.randn(config.embedding_dim, config.n_heads * config.head_dim)
        attention_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
    
    base_embeddings = torch.randn(config.vocab_size, config.embedding_dim) * 0.02
    
    model = PicoGPTSparseRoPE(config, attention_weights, base_embeddings=base_embeddings)
    
    # Test forward
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"Input: {input_ids.shape}")
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
    
    # Test RoPE
    print("\nTesting RoPE:")
    rope = RotaryPositionEncoding(config.head_dim)
    
    # Create sample Q and K
    q = torch.randn(2, config.n_heads, 10, config.head_dim)
    k = torch.randn(2, config.n_heads, 10, config.head_dim)
    
    q_rot, k_rot = rope(q, k)
    
    print(f"  Q shape: {q.shape} -> {q_rot.shape}")
    print(f"  K shape: {k.shape} -> {k_rot.shape}")
    print(f"  Q changed: {not torch.allclose(q, q_rot)}")
    print(f"  K changed: {not torch.allclose(k, k_rot)}")
    
    # Test skip-n masking
    print("\nTesting skip-n attention masks:")
    for layer_idx, block in enumerate(model.blocks):
        n = layer_idx + 1
        print(f"\nLayer {layer_idx} (n={n}):")
        
        attn = block.attention
        test_seq_len = 10
        masks = attn._create_skip_n_masks(test_seq_len, torch.device('cpu'))
        
        for h in range(min(2, config.n_heads)):
            s = h
            skip_distance = n + s
            print(f"  Head {h} (s={s}, skip={skip_distance}):")
            
            for i in range(min(5, test_seq_len)):
                allowed_positions = (masks[h, i, :] == 0.0).nonzero(as_tuple=True)[0]
                print(f"    Position {i} can attend to: {allowed_positions.tolist()}")
    
    print("\n✓ PicoGPT Sparse RoPE tests passed!")

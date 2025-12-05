"""
PicoGPT Sparse - Frozen Attention Weights

Scaled-down transformer with:
- Frozen attention weights computed from corpus
- Trainable MLP layers
- 2 layers, 2 heads
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
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "Components" / "method"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Components" / "model"))

from pico_config import PicoModelConfig, PicoTrainingConfig


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


class PicoFrozenAttention(nn.Module):
    """Multi-head attention with frozen weights."""
    
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
        
        # Frozen weights as buffers
        self.register_buffer('W_Q', W_Q)
        self.register_buffer('W_K', W_K)
        self.register_buffer('W_V', W_V)
        self.register_buffer('W_O', W_O)
        
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
        
        # Q, K, V projections
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


class PicoSparseBlock(nn.Module):
    """Transformer block with frozen attention."""
    
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
        self.attention = PicoFrozenAttention(config, layer_idx, W_Q, W_K, W_V, W_O)
        self.ln2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.mlp = PicoMLP(config)
    
    def forward(self, x, attention_mask=None):
        # Attention with residual
        attn_out, attn_weights = self.attention(self.ln1(x), attention_mask)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        
        return x, attn_weights


class PicoGPTSparse(nn.Module):
    """
    PicoGPT with frozen attention weights.
    
    - Attention weights computed from corpus (frozen)
    - MLP layers trainable
    - Embeddings trainable
    """
    
    def __init__(
        self,
        config: PicoModelConfig,
        attention_weights: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        base_embeddings: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.config = config
        
        # Embeddings - MUST match those used to compute attention weights
        if base_embeddings is not None:
            # Use the exact embeddings from weight computation (frozen)
            assert base_embeddings.shape == (config.vocab_size, config.embedding_dim), \
                f"Base embeddings shape {base_embeddings.shape} doesn't match expected {(config.vocab_size, config.embedding_dim)}"
            self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
            self.token_embedding.weight.data = base_embeddings.clone()
            self.token_embedding.weight.requires_grad = False  # FREEZE - must match weight computation
        else:
            # Random init (for testing only)
            self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embedding_dim)
        self.position_embedding.weight.requires_grad = False  # Freeze position embeddings too
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
            
            self.blocks.append(PicoSparseBlock(config, layer_idx, W_Q, W_K, W_V, W_O))
        
        # Output
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        # Initialize
        self._init_weights()
        
        self.register_buffer('position_ids', torch.arange(config.max_seq_len).unsqueeze(0))
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        
        for block in self.blocks:
            nn.init.normal_(block.mlp.fc1.weight, std=0.02)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.normal_(block.mlp.fc2.weight, std=0.02 / (2 * self.config.n_layers) ** 0.5)
            nn.init.zeros_(block.mlp.fc2.bias)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(self.position_ids[:, :seq_len])
        x = self.embed_dropout(token_embeds + position_embeds)
        
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
        # Total includes both parameters AND frozen buffers (attention weights)
        param_count = sum(p.numel() for p in self.parameters())
        buffer_count = sum(b.numel() for b in self.buffers() if b is not None)
        return param_count + buffer_count


def compute_pico_weights(
    config: PicoModelConfig,
    training_config: PicoTrainingConfig,
    local_texts: List[str],
    skip_corpus: bool = False,
) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
    """
    Compute attention weights for PicoGPT using the FULL derivation algorithm.
    
    Uses character-level tokenization (~80 vocab) for tractable computation.
    
    Args:
        skip_corpus: If True, skip corpus computation and use random weights (for debugging)
    
    Returns:
        Tuple of (attention_weights dict, base_embeddings tensor)
    """
    # Fast mode for debugging only
    if skip_corpus:
        print("Computing PicoGPT attention weights (FAST: random weights)...")
        attention_weights = {}
        
        # Create random base embeddings too
        base_embeddings = torch.randn(config.vocab_size, config.embedding_dim) * 0.02
        
        for layer_idx in range(config.n_layers):
            scale = 0.02
            W_Q = torch.randn(config.n_heads, config.embedding_dim, config.head_dim) * scale
            W_K = torch.randn(config.n_heads, config.embedding_dim, config.head_dim) * scale
            W_V = torch.randn(config.n_heads, config.head_dim, config.embedding_dim) * scale
            W_O = torch.randn(config.embedding_dim, config.n_heads * config.head_dim) * scale
            
            attention_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
            print(f"  Layer {layer_idx}: random weights generated")
        
        return attention_weights, base_embeddings
    
    # FULL ALGORITHM using scaled-down pico_method
    from pico_method import compute_pico_attention_weights
    
    return compute_pico_attention_weights(
        texts=local_texts,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        embedding_dim=config.embedding_dim,
        head_dim=config.head_dim,
        max_n=training_config.max_n,
        max_s=training_config.max_s,
        verbose=True,
    )


if __name__ == "__main__":
    print("Testing PicoGPT Sparse...")
    
    config = PicoModelConfig()
    
    # Random weights for testing
    attention_weights = {}
    for layer_idx in range(config.n_layers):
        W_Q = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
        W_K = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
        W_V = torch.randn(config.n_heads, config.head_dim, config.embedding_dim)
        W_O = torch.randn(config.embedding_dim, config.n_heads * config.head_dim)
        attention_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
    
    # Random base embeddings for testing
    base_embeddings = torch.randn(config.vocab_size, config.embedding_dim) * 0.02
    
    model = PicoGPTSparse(config, attention_weights, base_embeddings=base_embeddings)
    
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
    
    print("\n✓ PicoGPT Sparse tests passed!")


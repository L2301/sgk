"""
PicoGPT Trainable - Fully Trainable Baseline

Standard transformer with all weights trainable:
- 2 layers, 2 heads
- All attention weights learned
- Used as baseline comparison for sparse version
"""

import sys
from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from pico_config import PicoModelConfig


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


class PicoTrainableAttention(nn.Module):
    """
    Standard multi-head attention with fully trainable weights.
    """
    
    def __init__(self, config: PicoModelConfig):
        super().__init__()
        
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.embedding_dim = config.embedding_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Trainable projections
        self.W_Q = nn.Linear(config.embedding_dim, config.n_heads * config.head_dim, bias=False)
        self.W_K = nn.Linear(config.embedding_dim, config.n_heads * config.head_dim, bias=False)
        self.W_V = nn.Linear(config.embedding_dim, config.n_heads * config.head_dim, bias=False)
        self.W_O = nn.Linear(config.n_heads * config.head_dim, config.embedding_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        
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
        
        # Project Q, K, V
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)
        
        # Concat heads and output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.W_O(attn_output)
        
        return output, attn_weights


class PicoTrainableBlock(nn.Module):
    """Transformer block with trainable attention."""
    
    def __init__(self, config: PicoModelConfig):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.attention = PicoTrainableAttention(config)
        self.ln2 = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.mlp = PicoMLP(config)
    
    def forward(self, x, attention_mask=None):
        # Attention with residual
        attn_out, attn_weights = self.attention(self.ln1(x), attention_mask)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        
        return x, attn_weights


class PicoGPTTrainable(nn.Module):
    """
    PicoGPT with fully trainable attention weights.
    
    Standard transformer baseline for comparison.
    """
    
    def __init__(self, config: PicoModelConfig):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embedding_dim)
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            PicoTrainableBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying
        
        # Initialize
        self._init_weights()
        
        self.register_buffer('position_ids', torch.arange(config.max_seq_len).unsqueeze(0))
    
    def _init_weights(self):
        # Embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        
        for block in self.blocks:
            # Attention
            nn.init.normal_(block.attention.W_Q.weight, std=0.02)
            nn.init.normal_(block.attention.W_K.weight, std=0.02)
            nn.init.normal_(block.attention.W_V.weight, std=0.02)
            nn.init.normal_(block.attention.W_O.weight, std=0.02 / (2 * self.config.n_layers) ** 0.5)
            
            # MLP
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
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Simple autoregressive generation."""
        self.eval()
        
        for _ in range(max_new_tokens):
            input_truncated = input_ids[:, -self.config.max_seq_len:]
            logits = self.forward(input_truncated)
            next_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


if __name__ == "__main__":
    print("Testing PicoGPT Trainable...")
    
    config = PicoModelConfig()
    model = PicoGPTTrainable(config)
    
    # Test forward
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"Input: {input_ids.shape}")
    print(f"Output: {logits.shape}")
    
    # Test loss
    loss, _ = model.compute_loss(input_ids)
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
    print(f"Generated: {generated.shape}")
    
    # Parameter counts
    total = model.get_num_params()
    trainable = model.get_num_params(trainable_only=True)
    
    print(f"\nParameters:")
    print(f"  Total: {total:,}")
    print(f"  Trainable: {trainable:,}")
    
    # Breakdown
    embed_params = config.vocab_size * config.embedding_dim + config.max_seq_len * config.embedding_dim
    attn_params_per_layer = 4 * config.embedding_dim * config.embedding_dim
    mlp_params_per_layer = 2 * config.embedding_dim * config.mlp_dim + config.mlp_dim + config.embedding_dim
    ln_params_per_layer = 4 * config.embedding_dim
    
    print(f"\nParameter breakdown:")
    print(f"  Embeddings: {embed_params:,}")
    print(f"  Attention (per layer): {attn_params_per_layer:,}")
    print(f"  MLP (per layer): {mlp_params_per_layer:,}")
    print(f"  LayerNorm (per layer): {ln_params_per_layer:,}")
    
    # Test backward
    loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"\n✓ Gradients computed for {grad_count} parameters")
    
    print("\n✓ PicoGPT Trainable tests passed!")


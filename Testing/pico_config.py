"""
PicoGPT Configuration

Scaled-down transformer for testing:
- 2 layers
- 2 heads
- Character-level vocabulary (~80 tokens instead of 50,257)
- Smaller dimensions for fast iteration

This tests the FULL attention weight derivation algorithm at small scale.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

# Import vocabulary size from pico tokenizer
try:
    from pico_tokenizer import PICO_VOCAB_SIZE
except ImportError:
    PICO_VOCAB_SIZE = 81  # Fallback


@dataclass
class PicoModelConfig:
    """Scaled-down model configuration for testing."""
    
    # Architecture - tiny version
    n_layers: int = 12
    n_heads: int = 12
    embedding_dim: int = 768
    head_dim: int = 64  # embedding_dim // n_heads
    
    # MLP
    mlp_dim: int = 3072  # 4 * embedding_dim
    mlp_activation: str = "gelu"
    
    # Vocabulary and sequence (CHARACTER-LEVEL for testing)
    vocab_size: int = PICO_VOCAB_SIZE  # ~80 chars instead of 50,257 BPE tokens
    max_seq_len: int = 256  # Shorter for testing
    
    # Dropout
    mlp_dropout: float = 0.1
    embed_dropout: float = 0.1
    attn_dropout: float = 0.1  # For trainable version
    
    # Layer norm
    layer_norm_eps: float = 1e-5
    
    def __post_init__(self):
        assert self.embedding_dim == self.n_heads * self.head_dim, \
            f"embedding_dim ({self.embedding_dim}) must equal n_heads * head_dim ({self.n_heads * self.head_dim})"
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'PicoModelConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class PicoTrainingConfig:
    """Training configuration for PicoGPT."""
    
    # Data
    num_chunks: int = 20  # Fewer chunks for testing
    test_chunks: int = 5
    batch_size: int = 4
    
    # Optimization
    learning_rate: float = 1e-3  # Higher LR for small model
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_steps: int = 50
    max_steps: int = 1000  # Fewer steps for testing
    
    # Checkpointing
    snapshot_interval: int = 200
    eval_interval: int = 100
    log_interval: int = 20
    
    # N-gram settings (for sparse version)
    max_n: int = 2  # Layer 1 uses n=1, Layer 2 uses n=2
    max_s: int = 1  # Head 0 uses s=0, Head 1 uses s=1
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'PicoTrainingConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        if 'betas' in data and isinstance(data['betas'], list):
            data['betas'] = tuple(data['betas'])
        return cls(**data)


# Default configs
PICO_MODEL = PicoModelConfig()
PICO_TRAINING = PicoTrainingConfig()


if __name__ == "__main__":
    print("PicoGPT Configuration (2 layers, 2 heads):")
    print(f"  Layers: {PICO_MODEL.n_layers}")
    print(f"  Heads per layer: {PICO_MODEL.n_heads}")
    print(f"  Embedding dim: {PICO_MODEL.embedding_dim}")
    print(f"  Head dim: {PICO_MODEL.head_dim}")
    print(f"  MLP dim: {PICO_MODEL.mlp_dim}")
    print(f"  Max sequence length: {PICO_MODEL.max_seq_len}")
    
    print("\nTraining Configuration:")
    print(f"  Chunks for weights: {PICO_TRAINING.num_chunks}")
    print(f"  Max steps: {PICO_TRAINING.max_steps}")
    print(f"  Learning rate: {PICO_TRAINING.learning_rate}")
    print(f"  Max n-gram (n): {PICO_TRAINING.max_n}")
    print(f"  Max skip (s): {PICO_TRAINING.max_s}")
    
    # Estimate parameters
    embed_params = PICO_MODEL.vocab_size * PICO_MODEL.embedding_dim
    pos_params = PICO_MODEL.max_seq_len * PICO_MODEL.embedding_dim
    mlp_params = 2 * PICO_MODEL.embedding_dim * PICO_MODEL.mlp_dim + PICO_MODEL.mlp_dim + PICO_MODEL.embedding_dim
    attn_params = 4 * PICO_MODEL.embedding_dim * PICO_MODEL.embedding_dim  # Q, K, V, O
    layer_params = mlp_params + attn_params + 4 * PICO_MODEL.embedding_dim  # + layer norms
    
    total = embed_params + pos_params + PICO_MODEL.n_layers * layer_params
    print(f"\nEstimated total parameters: ~{total:,}")


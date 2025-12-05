"""
Model Configuration

GPT-2 small architecture hyperparameters and training settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class ModelConfig:
    """GPT-2 small architecture configuration."""
    
    # Architecture
    n_layers: int = 12
    n_heads: int = 12
    embedding_dim: int = 768
    head_dim: int = 64  # embedding_dim // n_heads
    
    # MLP
    mlp_dim: int = 3072  # 4 * embedding_dim
    mlp_activation: str = "gelu"
    
    # Vocabulary and sequence
    vocab_size: int = 50257
    max_seq_len: int = 1024
    
    # Dropout (only for MLP since attention weights are frozen)
    mlp_dropout: float = 0.1
    embed_dropout: float = 0.1
    
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
    def load(cls, path: str) -> 'ModelConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    
    # Data
    num_chunks: int = 100  # Number of 1024-token chunks for weight computation
    test_chunks: int = 20  # Held-out chunks for evaluation
    batch_size: int = 8
    
    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_steps: int = 100
    max_steps: int = 10000
    
    # Checkpointing
    snapshot_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 100
    
    # N-gram settings for attention weights
    max_n: int = 12  # Maximum n-gram size (one per layer)
    max_s: int = 11  # Maximum skip value (one per head)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        # Handle tuple conversion for betas
        if 'betas' in data and isinstance(data['betas'], list):
            data['betas'] = tuple(data['betas'])
        return cls(**data)


@dataclass
class RunConfig:
    """Combined configuration for a training run."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # FineWeb subset
    fineweb_subset: str = "sample-10BT"
    
    # Device
    device: str = "cuda"  # or "mps" for Apple Silicon, "cpu"
    
    def to_dict(self) -> dict:
        return {
            'model': self.model.to_dict(),
            'training': self.training.to_dict(),
            'fineweb_subset': self.fineweb_subset,
            'device': self.device,
        }
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'RunConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            model=ModelConfig(**data['model']),
            training=TrainingConfig(**{
                k: tuple(v) if k == 'betas' and isinstance(v, list) else v
                for k, v in data['training'].items()
            }),
            fineweb_subset=data.get('fineweb_subset', 'sample-10BT'),
            device=data.get('device', 'cuda'),
        )


# Default configurations
GPT2_SMALL = ModelConfig()
DEFAULT_TRAINING = TrainingConfig()
DEFAULT_RUN = RunConfig()


if __name__ == "__main__":
    print("GPT-2 Small Configuration:")
    print(f"  Layers: {GPT2_SMALL.n_layers}")
    print(f"  Heads per layer: {GPT2_SMALL.n_heads}")
    print(f"  Embedding dim: {GPT2_SMALL.embedding_dim}")
    print(f"  Head dim: {GPT2_SMALL.head_dim}")
    print(f"  MLP dim: {GPT2_SMALL.mlp_dim}")
    print(f"  Vocab size: {GPT2_SMALL.vocab_size}")
    print(f"  Max sequence length: {GPT2_SMALL.max_seq_len}")
    
    print("\nDefault Training Configuration:")
    print(f"  Chunks for weights: {DEFAULT_TRAINING.num_chunks}")
    print(f"  Test chunks: {DEFAULT_TRAINING.test_chunks}")
    print(f"  Batch size: {DEFAULT_TRAINING.batch_size}")
    print(f"  Learning rate: {DEFAULT_TRAINING.learning_rate}")
    print(f"  Max steps: {DEFAULT_TRAINING.max_steps}")
    print(f"  Max n-gram (n): {DEFAULT_TRAINING.max_n}")
    print(f"  Max skip (s): {DEFAULT_TRAINING.max_s}")


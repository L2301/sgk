"""
Hard Mode Testing Configuration

Ensures fair comparison:
- Same hyperparameters for ALL models
- No accidental advantages
"""

from dataclasses import dataclass, field, asdict
from typing import List
import json


@dataclass
class ExperimentConfig:
    """Shared configuration for fair comparison."""
    
    # ===== Model Architecture =====
    n_layers: int = 4
    n_heads: int = 4
    embedding_dim: int = 256
    head_dim: int = 64
    mlp_dim: int = 1024
    max_seq_len: int = 128
    
    # Vocab (from pico_tokenizer)
    vocab_size: int = 85
    
    # Regularization
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # ===== Training Hyperparameters (SAME FOR ALL) =====
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_steps: int = 100
    max_steps: int = 500
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 50
    
    # ===== Corpus-derived attention settings =====
    max_n: int = 2  # Cap n-gram size (avoid memory explosion)
    max_s: int = 3  # Skip values (one per head)
    
    # ===== Data Split =====
    # Ensures no leakage between splits
    train_texts: int = 20  # Number of distinct text chunks for training
    val_texts: int = 5     # Held-out for validation during training
    test_texts: int = 5    # NEVER touched until final evaluation
    
    # ===== Multi-seed testing =====
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1000])
    
    def __post_init__(self):
        assert self.embedding_dim == self.n_heads * self.head_dim
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['betas'] = list(d['betas'])
        return d
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        with open(path, 'r') as f:
            data = json.load(f)
        data['betas'] = tuple(data['betas'])
        data['seeds'] = list(data['seeds'])
        return cls(**data)


# Default config
DEFAULT_CONFIG = ExperimentConfig()


if __name__ == "__main__":
    config = ExperimentConfig()
    print("Hard Mode Experiment Configuration")
    print("=" * 50)
    print(f"Model: {config.n_layers} layers, {config.n_heads} heads")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Training steps: {config.max_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Seeds: {config.seeds}")
    print(f"Data splits: train={config.train_texts}, val={config.val_texts}, test={config.test_texts}")


"""
Trainable MLP (Feed-Forward) Layers

Standard GPT-2 style MLP with GELU activation.
These are the only trainable weights in the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit activation.
    GPT-2 uses the approximate version.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate='tanh')


class MLP(nn.Module):
    """
    Feed-forward network: Linear -> GELU -> Linear -> Dropout
    
    Expands embedding_dim to mlp_dim, then projects back.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.fc1 = nn.Linear(config.embedding_dim, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.embedding_dim)
        self.activation = GELU()
        self.dropout = nn.Dropout(config.mlp_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, embedding_dim)
        
        Returns:
            Output tensor (batch_size, seq_len, embedding_dim)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MLPLayer(nn.Module):
    """
    Complete MLP layer with pre-layer norm and residual connection.
    GPT-2 style architecture.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(
            config.embedding_dim,
            eps=config.layer_norm_eps
        )
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (batch_size, seq_len, embedding_dim)
        
        Returns:
            Output tensor (batch_size, seq_len, embedding_dim)
        """
        # Pre-norm
        normed = self.layer_norm(x)
        
        # MLP with residual
        return x + self.mlp(normed)


def count_mlp_parameters(config: ModelConfig) -> int:
    """Count parameters in a single MLP layer."""
    mlp = MLPLayer(config)
    return sum(p.numel() for p in mlp.parameters())


if __name__ == "__main__":
    config = ModelConfig()
    
    print("Testing MLP Layer...")
    
    # Create MLP layer
    mlp_layer = MLPLayer(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, config.embedding_dim)
    
    output = mlp_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verify shape preservation
    assert output.shape == x.shape, "Shape should be preserved"
    
    # Count parameters
    total_params = sum(p.numel() for p in mlp_layer.parameters())
    
    print(f"\nMLP Layer Parameters:")
    print(f"  fc1: {config.embedding_dim} x {config.mlp_dim} = {config.embedding_dim * config.mlp_dim:,}")
    print(f"  fc1 bias: {config.mlp_dim:,}")
    print(f"  fc2: {config.mlp_dim} x {config.embedding_dim} = {config.mlp_dim * config.embedding_dim:,}")
    print(f"  fc2 bias: {config.embedding_dim:,}")
    print(f"  LayerNorm: {2 * config.embedding_dim:,}")
    print(f"  Total: {total_params:,}")
    
    # For full model
    print(f"\nFull model MLP parameters ({config.n_layers} layers):")
    print(f"  {total_params * config.n_layers:,}")
    
    # Test gradient flow
    output.sum().backward()
    
    grad_exists = all(p.grad is not None for p in mlp_layer.parameters())
    print(f"\n✓ Gradient flow: {'OK' if grad_exists else 'FAILED'}")
    
    print("✓ All MLP tests passed!")


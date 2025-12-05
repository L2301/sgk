"""
Sparse Attention Transformer

Full GPT-2 style transformer with:
- Frozen attention weights (W_Q, W_K, W_V, W_O) from corpus statistics
- Trainable MLP layers
- Token and position embeddings
- Causal language modeling head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from config import ModelConfig
from attention import FrozenAttentionLayer, create_attention_weights_from_matrices
from mlp import MLPLayer


class TransformerBlock(nn.Module):
    """
    Single transformer block: Attention -> MLP
    
    Attention weights are frozen, MLP is trainable.
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
        
        self.attention = FrozenAttentionLayer(
            config, layer_idx, W_Q, W_K, W_V, W_O
        )
        self.mlp = MLPLayer(config)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention and MLP.
        
        Args:
            x: Input tensor (batch_size, seq_len, embedding_dim)
            attention_mask: Optional attention mask
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Attention with residual (handled inside)
        x, attn_weights = self.attention(x, attention_mask)
        
        # MLP with residual (handled inside)
        x = self.mlp(x)
        
        return x, attn_weights


class SparseAttentionTransformer(nn.Module):
    """
    Full transformer model with frozen attention and trainable MLPs.
    
    Architecture:
    - Token embeddings (trainable)
    - Position embeddings (trainable)
    - N transformer blocks (attention frozen, MLP trainable)
    - Final layer norm
    - Language model head (tied with token embeddings)
    """
    
    def __init__(
        self,
        config: ModelConfig,
        attention_weights: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ):
        """
        Initialize the transformer.
        
        Args:
            config: Model configuration
            attention_weights: Dict mapping layer_idx to (W_Q, W_K, W_V, W_O) tuple
        """
        super().__init__()
        
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embedding_dim)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.embed_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            if layer_idx in attention_weights:
                W_Q, W_K, W_V, W_O = attention_weights[layer_idx]
            else:
                # Initialize with random weights if not provided
                print(f"Warning: No attention weights for layer {layer_idx}, using random")
                W_Q = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
                W_K = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
                W_V = torch.randn(config.n_heads, config.head_dim, config.embedding_dim)
                W_O = torch.randn(config.embedding_dim, config.n_heads * config.head_dim)
            
            block = TransformerBlock(config, layer_idx, W_Q, W_K, W_V, W_O)
            self.blocks.append(block)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        
        # Language model head (projects to vocab)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Tie weights with token embeddings
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
        
        # Register position ids buffer
        self.register_buffer(
            'position_ids',
            torch.arange(config.max_seq_len).unsqueeze(0)
        )
    
    def _init_weights(self):
        """Initialize trainable weights."""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Position embeddings
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        
        # MLP layers
        for block in self.blocks:
            nn.init.normal_(block.mlp.mlp.fc1.weight, std=0.02)
            nn.init.zeros_(block.mlp.mlp.fc1.bias)
            # Scale output projection by 1/sqrt(n_layers) for residual
            nn.init.normal_(block.mlp.mlp.fc2.weight, std=0.02 / (2 * self.config.n_layers) ** 0.5)
            nn.init.zeros_(block.mlp.mlp.fc2.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            Dict with 'logits' and optionally 'attention_weights'
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = self.position_ids[:, :seq_len]
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        x = self.embed_dropout(token_embeds + position_embeds)
        
        # Process through transformer blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x, attention_mask)
            if return_attention:
                all_attention_weights.append(attn_weights)
        
        # Final norm
        x = self.final_norm(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        result = {'logits': logits}
        if return_attention:
            result['attention_weights'] = all_attention_weights
        
        return result
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute language modeling loss.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            labels: Target token IDs (batch_size, seq_len). If None, uses input_ids shifted.
        
        Returns:
            Tuple of (loss, logits)
        """
        outputs = self.forward(input_ids)
        logits = outputs['logits']
        
        if labels is None:
            # Shift for next-token prediction
            labels = input_ids
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        
        return loss, logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Starting token IDs (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated token IDs (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            input_truncated = input_ids[:, -self.config.max_seq_len:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(input_truncated)
                logits = outputs['logits']
            
            # Get logits for last position
            next_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][:, -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (MLPs, embeddings, norms)."""
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(param)
        return trainable
    
    def get_num_params(self, trainable_only: bool = False) -> int:
        """Count parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def freeze_attention(self):
        """Ensure all attention weights are frozen (they should be buffers anyway)."""
        for block in self.blocks:
            for buffer in block.attention.attention.buffers():
                buffer.requires_grad = False


def create_model_from_weight_dicts(
    config: ModelConfig,
    W_Q_dict: Dict[Tuple[int, int], torch.Tensor],
    W_K_dict: Dict[Tuple[int, int], torch.Tensor],
    W_V_dict: Dict[Tuple[int, int], torch.Tensor],
    W_O_dict: Dict[Tuple[int, int], torch.Tensor],
) -> SparseAttentionTransformer:
    """
    Create model from weight dictionaries.
    
    Args:
        config: Model configuration
        W_Q_dict: Dict mapping (n, s) to W_Q matrix
        W_K_dict: Dict mapping (n, s) to W_K matrix
        W_V_dict: Dict mapping (n, s) to W_V matrix  
        W_O_dict: Dict mapping (n, s) to W_O matrix
    
    Returns:
        Initialized model
    """
    # Organize weights by layer
    attention_weights = {}
    
    for layer_idx in range(config.n_layers):
        W_Q, W_K, W_V, W_O = create_attention_weights_from_matrices(
            W_Q_dict, W_K_dict, W_V_dict, W_O_dict,
            layer_idx, config.n_heads, config.embedding_dim, config.head_dim
        )
        attention_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
    
    return SparseAttentionTransformer(config, attention_weights)


if __name__ == "__main__":
    print("Testing SparseAttentionTransformer...")
    
    config = ModelConfig()
    
    # Create random attention weights for testing
    attention_weights = {}
    for layer_idx in range(config.n_layers):
        W_Q = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
        W_K = torch.randn(config.n_heads, config.embedding_dim, config.head_dim)
        W_V = torch.randn(config.n_heads, config.head_dim, config.embedding_dim)
        W_O = torch.randn(config.embedding_dim, config.n_heads * config.head_dim)
        attention_weights[layer_idx] = (W_Q, W_K, W_V, W_O)
    
    # Create model
    model = SparseAttentionTransformer(config, attention_weights)
    
    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids)
    logits = outputs['logits']
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Test loss computation
    loss, _ = model.compute_loss(input_ids)
    print(f"Loss: {loss.item():.4f}")
    
    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    # Parameter counts
    total_params = model.get_num_params()
    trainable_params = model.get_num_params(trainable_only=True)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Frozen: {frozen_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.1f}%")
    
    # Test backward pass (only trainable params should get gradients)
    loss.backward()
    
    trainable_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    print(f"\n✓ Parameters with gradients: {trainable_with_grad}")
    print("✓ All transformer tests passed!")


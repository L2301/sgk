"""
Weight Loader

Computes attention weights (W_Q, W_K, W_V, W_O) from corpus using
the method pipeline:
1. Tokenize corpus chunks
2. Build count matrices
3. Compute probability and manifold matrices
4. Compute embedding matrices
5. Derive W_Q, W_K from manifold (via SVD)
6. Derive W_V, W_O from probability (via SVD)
7. Orthogonalize across heads
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional
from pathlib import Path

# Add method directory to path
method_dir = Path(__file__).parent.parent / "method"
sys.path.insert(0, str(method_dir))

from tokenizer import GPT2Tokenizer
from corpus_matrix import CorpusMatrixBuilder, BATCH_SIZE
from count_matrix import build_all_count_matrices, VOCAB_SIZE
from probability_matrix import build_all_probability_matrices
from manifold_matrix import build_all_manifold_matrices
from embedding_matrix import (
    load_gpt2_embeddings, 
    build_all_embedding_matrices,
    EMBEDDING_DIM
)
from key_query_matrix import build_all_key_query_matrices, HEAD_DIM
from value_output_matrix import build_all_value_output_matrices
from ortho_value_output_matrix import orthogonalize_value_output_matrices

# Import config
sys.path.insert(0, str(Path(__file__).parent))
from config import ModelConfig, TrainingConfig


class WeightComputer:
    """
    Computes attention weights from corpus data.
    
    Pipeline:
    Corpus -> Tokens -> Count -> Probability/Manifold -> Embedding -> W_Q/K/V/O
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        fineweb_subset: str = "sample-10BT",
        verbose: bool = True,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.fineweb_subset = fineweb_subset
        self.verbose = verbose
        
        self.tokenizer = GPT2Tokenizer()
        self.corpus_builder = CorpusMatrixBuilder()
        
        # Will be populated during compute
        self.token_batches: List[List[int]] = []
        self.count_matrices = {}
        self.probability_matrices = {}
        self.manifold_matrices = {}
        self.embedding_matrices = {}
        self.base_embeddings = None
        
        # Final weight matrices
        self.W_Q_dict = {}
        self.W_K_dict = {}
        self.W_V_dict = {}
        self.W_O_dict = {}
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def load_token_batches(
        self,
        num_batches: int,
        from_fineweb: bool = True,
        local_texts: Optional[List[str]] = None,
    ) -> List[List[int]]:
        """
        Load tokenized data batches.
        
        Args:
            num_batches: Number of 1024-token batches to load
            from_fineweb: Whether to stream from FineWeb
            local_texts: Alternative local texts to use
        
        Returns:
            List of token batches
        """
        self._log(f"Loading {num_batches} token batches...")
        
        if from_fineweb:
            batches = []
            for batch in self.corpus_builder.stream_token_batches(
                subset=self.fineweb_subset,
                max_batches=num_batches
            ):
                batches.append(batch)
                if len(batches) % 10 == 0:
                    self._log(f"  Loaded {len(batches)}/{num_batches} batches")
            self.token_batches = batches
        else:
            # Use local texts
            all_tokens = []
            for text in local_texts:
                all_tokens.extend(self.tokenizer.encode(text))
            
            # Split into batches
            batches = []
            for i in range(0, len(all_tokens) - BATCH_SIZE + 1, BATCH_SIZE):
                batches.append(all_tokens[i:i + BATCH_SIZE])
                if len(batches) >= num_batches:
                    break
            self.token_batches = batches
        
        self._log(f"  Loaded {len(self.token_batches)} batches")
        return self.token_batches
    
    def build_count_matrices(self) -> Dict:
        """Build count matrices for all (n, s) combinations."""
        self._log(f"Building count matrices (max_n={self.training_config.max_n}, max_s={self.training_config.max_s})...")
        
        self.count_matrices = build_all_count_matrices(
            self.token_batches,
            max_n=self.training_config.max_n,
            max_s=self.training_config.max_s,
            vocab_size=VOCAB_SIZE
        )
        
        return self.count_matrices
    
    def build_probability_matrices(self) -> Dict:
        """Build probability matrices from count matrices."""
        self._log("Building probability matrices...")
        
        self.probability_matrices = build_all_probability_matrices(
            self.count_matrices
        )
        
        return self.probability_matrices
    
    def build_manifold_matrices(self) -> Dict:
        """Build manifold matrices from count matrices."""
        self._log("Building manifold matrices...")
        
        self.manifold_matrices = build_all_manifold_matrices(
            self.count_matrices
        )
        
        return self.manifold_matrices
    
    def load_base_embeddings(self) -> np.ndarray:
        """Load GPT-2 pretrained embeddings."""
        self._log("Loading GPT-2 base embeddings...")
        
        self.base_embeddings = load_gpt2_embeddings()
        self._log(f"  Loaded embeddings: {self.base_embeddings.shape}")
        
        return self.base_embeddings
    
    def build_embedding_matrices(self) -> Dict:
        """Build n-gram embedding matrices."""
        self._log("Building embedding matrices...")
        
        if self.base_embeddings is None:
            self.load_base_embeddings()
        
        self.embedding_matrices = build_all_embedding_matrices(
            self.probability_matrices,
            self.base_embeddings,
            max_n=self.training_config.max_n,
            s=0,  # Use s=0 for embedding construction
            vocab_size=VOCAB_SIZE
        )
        
        return self.embedding_matrices
    
    def compute_key_query_matrices(self) -> Tuple[Dict, Dict]:
        """Compute W_Q and W_K from manifold matrices."""
        self._log("Computing W_Q and W_K matrices...")
        
        kq_matrices = build_all_key_query_matrices(
            self.manifold_matrices,
            self.embedding_matrices,
            self.base_embeddings,
            head_dim=HEAD_DIM
        )
        
        # Separate into W_Q and W_K dicts
        for (n, s), (W_Q, W_K) in kq_matrices.items():
            self.W_Q_dict[(n, s)] = W_Q
            self.W_K_dict[(n, s)] = W_K
        
        return self.W_Q_dict, self.W_K_dict
    
    def compute_value_output_matrices(self, orthogonalize: bool = True) -> Tuple[Dict, Dict]:
        """Compute W_V and W_O from probability matrices."""
        self._log("Computing W_V and W_O matrices...")
        
        vo_matrices = build_all_value_output_matrices(
            self.probability_matrices,
            self.embedding_matrices,
            self.base_embeddings,
            head_dim=HEAD_DIM
        )
        
        if orthogonalize:
            self._log("Orthogonalizing W_V and W_O matrices...")
            vo_matrices, _ = orthogonalize_value_output_matrices(
                vo_matrices,
                head_dim=HEAD_DIM
            )
        
        # Separate into W_V and W_O dicts
        for (n, s), (W_V, W_O) in vo_matrices.items():
            self.W_V_dict[(n, s)] = W_V
            self.W_O_dict[(n, s)] = W_O
        
        return self.W_V_dict, self.W_O_dict
    
    def compute_all_weights(
        self,
        num_batches: Optional[int] = None,
        from_fineweb: bool = True,
        local_texts: Optional[List[str]] = None,
        orthogonalize: bool = True,
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Full pipeline: compute all attention weights from corpus.
        
        Args:
            num_batches: Number of token batches (default from training config)
            from_fineweb: Whether to use FineWeb
            local_texts: Alternative local texts
            orthogonalize: Whether to orthogonalize W_V/W_O
        
        Returns:
            Tuple of (W_Q_dict, W_K_dict, W_V_dict, W_O_dict)
        """
        if num_batches is None:
            num_batches = self.training_config.num_chunks
        
        self._log("=" * 60)
        self._log("Computing attention weights from corpus")
        self._log("=" * 60)
        
        # Step 1: Load tokens
        self.load_token_batches(num_batches, from_fineweb, local_texts)
        
        # Step 2: Build count matrices
        self.build_count_matrices()
        
        # Step 3: Build probability and manifold matrices
        self.build_probability_matrices()
        self.build_manifold_matrices()
        
        # Step 4: Load embeddings and build n-gram embeddings
        self.load_base_embeddings()
        self.build_embedding_matrices()
        
        # Step 5: Compute W_Q, W_K from manifold
        self.compute_key_query_matrices()
        
        # Step 6: Compute W_V, W_O from probability
        self.compute_value_output_matrices(orthogonalize)
        
        self._log("=" * 60)
        self._log("Weight computation complete!")
        self._log(f"  W_Q: {len(self.W_Q_dict)} matrices")
        self._log(f"  W_K: {len(self.W_K_dict)} matrices")
        self._log(f"  W_V: {len(self.W_V_dict)} matrices")
        self._log(f"  W_O: {len(self.W_O_dict)} matrices")
        self._log("=" * 60)
        
        return self.W_Q_dict, self.W_K_dict, self.W_V_dict, self.W_O_dict
    
    def get_weights_for_layer(
        self,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get weight tensors for a specific layer.
        
        Layer uses n = layer_idx + 1.
        Head h uses s = h.
        
        Returns:
            Tuple of (W_Q, W_K, W_V, W_O) tensors
        """
        n = layer_idx + 1
        n_heads = self.model_config.n_heads
        embedding_dim = self.model_config.embedding_dim
        head_dim = self.model_config.head_dim
        
        W_Q_heads = []
        W_K_heads = []
        W_V_heads = []
        W_O_heads = []
        
        for h in range(n_heads):
            s = h
            key = (n, s)
            
            if key in self.W_Q_dict:
                W_Q_heads.append(torch.from_numpy(self.W_Q_dict[key]).float())
                W_K_heads.append(torch.from_numpy(self.W_K_dict[key]).float())
                W_V_heads.append(torch.from_numpy(self.W_V_dict[key]).float())
                W_O_heads.append(torch.from_numpy(self.W_O_dict[key]).float())
            else:
                self._log(f"Warning: No weights for (n={n}, s={s}), using zeros")
                W_Q_heads.append(torch.zeros(embedding_dim, head_dim))
                W_K_heads.append(torch.zeros(embedding_dim, head_dim))
                W_V_heads.append(torch.zeros(head_dim, embedding_dim))
                W_O_heads.append(torch.zeros(embedding_dim, head_dim))
        
        # Stack heads
        W_Q = torch.stack(W_Q_heads, dim=0)  # (n_heads, embed, head)
        W_K = torch.stack(W_K_heads, dim=0)
        W_V = torch.stack(W_V_heads, dim=0)  # (n_heads, head, embed)
        W_O = torch.cat(W_O_heads, dim=1)    # (embed, n_heads * head)
        
        return W_Q, W_K, W_V, W_O
    
    def get_all_layer_weights(self) -> Dict[int, Tuple]:
        """Get weights for all layers."""
        weights = {}
        for layer_idx in range(self.model_config.n_layers):
            weights[layer_idx] = self.get_weights_for_layer(layer_idx)
        return weights
    
    def save_weights(self, path: str):
        """Save computed weights to disk."""
        self._log(f"Saving weights to {path}...")
        
        data = {
            'W_Q': {str(k): v for k, v in self.W_Q_dict.items()},
            'W_K': {str(k): v for k, v in self.W_K_dict.items()},
            'W_V': {str(k): v for k, v in self.W_V_dict.items()},
            'W_O': {str(k): v for k, v in self.W_O_dict.items()},
            'model_config': self.model_config.to_dict(),
            'training_config': self.training_config.to_dict(),
        }
        
        np.savez_compressed(path, **{
            f'{matrix_type}_{k}': v 
            for matrix_type, matrices in [
                ('W_Q', self.W_Q_dict),
                ('W_K', self.W_K_dict),
                ('W_V', self.W_V_dict),
                ('W_O', self.W_O_dict),
            ]
            for k, v in matrices.items()
        })
        
        self._log(f"  Saved to {path}")
    
    @classmethod
    def load_weights(cls, path: str) -> 'WeightComputer':
        """Load weights from disk."""
        print(f"Loading weights from {path}...")
        
        data = np.load(path, allow_pickle=True)
        
        computer = cls(ModelConfig(), TrainingConfig(), verbose=False)
        
        # Parse keys and reconstruct dicts
        for key in data.files:
            parts = key.split('_')
            matrix_type = parts[0] + '_' + parts[1]  # W_Q, W_K, etc.
            n_s = eval('_'.join(parts[2:]))  # (n, s) tuple
            
            if matrix_type == 'W_Q':
                computer.W_Q_dict[n_s] = data[key]
            elif matrix_type == 'W_K':
                computer.W_K_dict[n_s] = data[key]
            elif matrix_type == 'W_V':
                computer.W_V_dict[n_s] = data[key]
            elif matrix_type == 'W_O':
                computer.W_O_dict[n_s] = data[key]
        
        print(f"  Loaded {len(computer.W_Q_dict)} weight sets")
        return computer


def compute_weights_for_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    fineweb_subset: str = "sample-10BT",
    from_fineweb: bool = True,
    local_texts: Optional[List[str]] = None,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convenience function to compute all weights needed for a model.
    
    Returns:
        Dict mapping layer_idx to (W_Q, W_K, W_V, W_O) tensors
    """
    computer = WeightComputer(model_config, training_config, fineweb_subset)
    computer.compute_all_weights(
        from_fineweb=from_fineweb,
        local_texts=local_texts,
    )
    return computer.get_all_layer_weights()


if __name__ == "__main__":
    print("Testing WeightComputer with local text...")
    
    # Use local text for testing (no network required)
    test_texts = [
        "The quick brown fox jumps over the lazy dog. " * 100,
        "Hello world! This is a test of the weight computation system. " * 100,
        "Machine learning models can learn patterns from data. " * 100,
    ]
    
    # Small config for testing
    model_config = ModelConfig()
    training_config = TrainingConfig(
        num_chunks=5,
        max_n=3,  # Only test up to trigrams
        max_s=2,  # Only test up to skip-2
    )
    
    computer = WeightComputer(model_config, training_config)
    
    # Compute weights
    W_Q, W_K, W_V, W_O = computer.compute_all_weights(
        num_batches=5,
        from_fineweb=False,
        local_texts=test_texts,
        orthogonalize=True,
    )
    
    # Get weights for first layer
    print("\nGetting weights for layer 0 (n=1)...")
    W_Q_l0, W_K_l0, W_V_l0, W_O_l0 = computer.get_weights_for_layer(0)
    
    print(f"  W_Q shape: {W_Q_l0.shape}")
    print(f"  W_K shape: {W_K_l0.shape}")
    print(f"  W_V shape: {W_V_l0.shape}")
    print(f"  W_O shape: {W_O_l0.shape}")
    
    print("\n✓ WeightComputer test passed!")


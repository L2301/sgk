"""
Data Loader

Streaming data loader for FineWeb with train/test split.
Handles tokenization, batching, and sequence construction.
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Iterator, List, Optional, Tuple
import numpy as np

# Add method directory to path
method_dir = Path(__file__).parent.parent / "method"
sys.path.insert(0, str(method_dir))

from tokenizer import GPT2Tokenizer
from corpus_matrix import BATCH_SIZE

sys.path.insert(0, str(Path(__file__).parent))
from config import ModelConfig, TrainingConfig


class TokenBuffer:
    """Buffer for accumulating tokens from streaming data."""
    
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.buffer: List[int] = []
    
    def add_tokens(self, tokens: List[int]):
        """Add tokens to the buffer."""
        self.buffer.extend(tokens)
    
    def get_sequences(self) -> List[List[int]]:
        """Extract complete sequences from buffer."""
        sequences = []
        while len(self.buffer) >= self.seq_len:
            sequences.append(self.buffer[:self.seq_len])
            self.buffer = self.buffer[self.seq_len:]
        return sequences
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []


class FineWebDataset(IterableDataset):
    """
    Streaming dataset for FineWeb.
    
    Yields sequences of token IDs for language modeling.
    """
    
    def __init__(
        self,
        subset: str = "sample-10BT",
        seq_len: int = 1024,
        max_sequences: Optional[int] = None,
        skip_sequences: int = 0,
    ):
        """
        Initialize the dataset.
        
        Args:
            subset: FineWeb subset to use
            seq_len: Sequence length
            max_sequences: Maximum number of sequences to yield
            skip_sequences: Number of sequences to skip (for train/test split)
        """
        self.subset = subset
        self.seq_len = seq_len
        self.max_sequences = max_sequences
        self.skip_sequences = skip_sequences
        
        self.tokenizer = GPT2Tokenizer()
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over sequences."""
        from datasets import load_dataset
        
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=self.subset,
            split="train",
            streaming=True,
        )
        
        buffer = TokenBuffer(self.seq_len)
        sequences_yielded = 0
        sequences_skipped = 0
        
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            buffer.add_tokens(tokens)
            
            for seq in buffer.get_sequences():
                # Skip sequences for test split
                if sequences_skipped < self.skip_sequences:
                    sequences_skipped += 1
                    continue
                
                yield torch.tensor(seq, dtype=torch.long)
                sequences_yielded += 1
                
                if self.max_sequences and sequences_yielded >= self.max_sequences:
                    return


class LocalTextDataset(Dataset):
    """
    Dataset from local text strings.
    
    For testing without network access.
    """
    
    def __init__(
        self,
        texts: List[str],
        seq_len: int = 1024,
    ):
        self.seq_len = seq_len
        self.tokenizer = GPT2Tokenizer()
        
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenizer.encode(text))
        
        # Create sequences
        self.sequences = []
        for i in range(0, len(all_tokens) - seq_len + 1, seq_len):
            self.sequences.append(all_tokens[i:i + seq_len])
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.sequences[idx], dtype=torch.long)


class TokenDataset(Dataset):
    """
    Dataset from pre-tokenized sequences.
    """
    
    def __init__(self, sequences: List[List[int]]):
        self.sequences = sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def create_dataloaders(
    training_config: TrainingConfig,
    model_config: ModelConfig,
    fineweb_subset: str = "sample-10BT",
    use_fineweb: bool = True,
    local_texts: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader, List[List[int]]]:
    """
    Create train and test data loaders.
    
    Args:
        training_config: Training configuration
        model_config: Model configuration
        fineweb_subset: FineWeb subset
        use_fineweb: Whether to use FineWeb (vs local texts)
        local_texts: Local texts for testing
    
    Returns:
        Tuple of (train_loader, test_loader, test_sequences)
    """
    seq_len = model_config.max_seq_len
    batch_size = training_config.batch_size
    num_train = training_config.num_chunks
    num_test = training_config.test_chunks
    
    if use_fineweb:
        # Streaming datasets
        train_dataset = FineWebDataset(
            subset=fineweb_subset,
            seq_len=seq_len,
            max_sequences=num_train,
            skip_sequences=0,
        )
        
        test_dataset = FineWebDataset(
            subset=fineweb_subset,
            seq_len=seq_len,
            max_sequences=num_test,
            skip_sequences=num_train,  # Skip training sequences
        )
        
        # For iterable datasets, we need to collect test data for saving
        test_sequences = []
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
        )
        
    else:
        # Local text datasets
        if local_texts is None:
            raise ValueError("local_texts required when use_fineweb=False")
        
        full_dataset = LocalTextDataset(local_texts, seq_len)
        
        # Split into train/test
        total = len(full_dataset)
        train_size = min(num_train, int(total * 0.8))
        test_size = min(num_test, total - train_size)
        
        train_sequences = [full_dataset.sequences[i] for i in range(train_size)]
        test_sequences = [full_dataset.sequences[i] for i in range(train_size, train_size + test_size)]
        
        train_dataset = TokenDataset(train_sequences)
        test_dataset = TokenDataset(test_sequences)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    
    return train_loader, test_loader, test_sequences


def collect_token_batches(
    num_batches: int,
    fineweb_subset: str = "sample-10BT",
    batch_size: int = BATCH_SIZE,
    use_fineweb: bool = True,
    local_texts: Optional[List[str]] = None,
) -> List[List[int]]:
    """
    Collect token batches for weight computation.
    
    Args:
        num_batches: Number of batches to collect
        fineweb_subset: FineWeb subset
        batch_size: Tokens per batch
        use_fineweb: Whether to use FineWeb
        local_texts: Local texts for testing
    
    Returns:
        List of token lists
    """
    tokenizer = GPT2Tokenizer()
    batches = []
    
    if use_fineweb:
        from datasets import load_dataset
        
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=fineweb_subset,
            split="train",
            streaming=True,
        )
        
        buffer = []
        
        for example in dataset:
            tokens = tokenizer.encode(example["text"])
            buffer.extend(tokens)
            
            while len(buffer) >= batch_size:
                batches.append(buffer[:batch_size])
                buffer = buffer[batch_size:]
                
                if len(batches) >= num_batches:
                    return batches
    else:
        if local_texts is None:
            raise ValueError("local_texts required when use_fineweb=False")
        
        all_tokens = []
        for text in local_texts:
            all_tokens.extend(tokenizer.encode(text))
        
        for i in range(0, len(all_tokens) - batch_size + 1, batch_size):
            batches.append(all_tokens[i:i + batch_size])
            if len(batches) >= num_batches:
                break
    
    return batches


def save_test_data(sequences: List[List[int]], path: str):
    """Save test sequences to disk."""
    torch.save(sequences, path)
    print(f"Saved {len(sequences)} test sequences to {path}")


def load_test_data(path: str) -> List[List[int]]:
    """Load test sequences from disk."""
    sequences = torch.load(path)
    print(f"Loaded {len(sequences)} test sequences from {path}")
    return sequences


if __name__ == "__main__":
    print("Testing data loaders...")
    
    # Test with local text
    test_texts = [
        "The quick brown fox jumps over the lazy dog. " * 500,
        "Hello world! Machine learning is fascinating. " * 500,
        "Natural language processing enables computers to understand text. " * 500,
    ]
    
    model_config = ModelConfig()
    training_config = TrainingConfig(
        num_chunks=10,
        test_chunks=5,
        batch_size=2,
    )
    
    # Create data loaders
    train_loader, test_loader, test_seqs = create_dataloaders(
        training_config,
        model_config,
        use_fineweb=False,
        local_texts=test_texts,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check a batch
    for batch in train_loader:
        print(f"\nBatch shape: {batch.shape}")
        print(f"Token range: [{batch.min().item()}, {batch.max().item()}]")
        break
    
    # Test token batch collection
    print("\nCollecting token batches...")
    batches = collect_token_batches(
        num_batches=5,
        use_fineweb=False,
        local_texts=test_texts,
    )
    
    print(f"Collected {len(batches)} batches")
    print(f"Tokens per batch: {len(batches[0])}")
    
    print("\n✓ All data loader tests passed!")


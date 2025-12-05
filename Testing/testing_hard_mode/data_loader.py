"""
Data Loader with Strict Train/Val/Test Splits

Ensures NO data leakage:
- Distinct text chunks for each split
- Test set is NEVER used during training or hyperparameter tuning
- Reproducible splits via seed
"""

import sys
from pathlib import Path
from typing import List, Tuple, Dict
import hashlib
import torch
import numpy as np

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent))
from pico_tokenizer import PicoTokenizer


# Diverse text sources (each is a distinct "document")
TEXT_SOURCES = [
    # Literature style
    "The quick brown fox jumps over the lazy dog. A wonderful serenity has taken possession of my entire soul.",
    "It was the best of times, it was the worst of times. The world is full of obvious things.",
    "Call me Ishmael. Some years ago, having little money in my purse, I thought I would sail about.",
    "In the beginning was the Word, and the Word was with meaning, and the Word was structure.",
    "To be or not to be, that is the question. Whether it is nobler in the mind to suffer.",
    
    # Technical style
    "Machine learning models learn patterns from data through optimization of loss functions.",
    "Neural networks consist of layers of interconnected nodes that transform input representations.",
    "Attention mechanisms allow models to focus on relevant parts of the input sequence.",
    "Gradient descent iteratively updates parameters to minimize the training objective.",
    "Backpropagation computes gradients by applying the chain rule through network layers.",
    
    # Conversational style
    "Hello! How are you doing today? I hope everything is going well for you.",
    "What do you think about this approach? It seems quite interesting to me.",
    "Let me explain how this works. First, we need to understand the basics.",
    "That is a great question! The answer involves several key concepts.",
    "Thanks for your help! I really appreciate you taking the time to explain.",
    
    # Scientific style
    "The experiment demonstrates significant improvements over baseline methods.",
    "Results indicate that the proposed approach outperforms existing techniques.",
    "Statistical analysis reveals a strong correlation between variables.",
    "Hypothesis testing confirms the effectiveness of the novel algorithm.",
    "Empirical evaluation shows consistent gains across multiple benchmarks.",
    
    # Narrative style
    "Once upon a time in a distant land, there lived a curious inventor.",
    "The sun rose slowly over the mountains, casting long shadows across the valley.",
    "She walked through the ancient library, fingers tracing the spines of countless books.",
    "The clock struck midnight as the final piece of the puzzle fell into place.",
    "Years passed, seasons changed, but the old oak tree remained standing tall.",
    
    # Mixed patterns
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ. The alphabet contains twenty six letters.",
    "Numbers like 1234567890 appear frequently in technical documentation.",
    "Punctuation marks: commas, periods, exclamation points! Question marks?",
    "Short words: a, an, the, is, it, on, in, to, of, and, for, are, but, not.",
    "Repetition helps learning. Practice makes perfect. Patterns emerge from data.",
]


def get_text_hash(text: str) -> str:
    """Get unique hash for a text chunk."""
    return hashlib.md5(text.encode()).hexdigest()[:8]


def create_expanded_texts(base_texts: List[str], repeat: int = 50) -> List[str]:
    """Expand base texts with variations to create more data."""
    expanded = []
    for text in base_texts:
        # Repeat with slight variations
        expanded.append((text + " ") * repeat)
    return expanded


def split_texts_no_leakage(
    seed: int = 42,
    n_train: int = 20,
    n_val: int = 5,
    n_test: int = 5,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split texts into train/val/test with NO overlap.
    
    Each split gets completely different source texts.
    """
    np.random.seed(seed)
    
    # Shuffle text sources
    indices = np.random.permutation(len(TEXT_SOURCES))
    
    # Ensure we have enough
    total_needed = n_train + n_val + n_test
    if len(TEXT_SOURCES) < total_needed:
        # Cycle through with variations
        extended_sources = []
        for i in range(total_needed):
            base = TEXT_SOURCES[i % len(TEXT_SOURCES)]
            # Add variation
            variation = f" [Variant {i // len(TEXT_SOURCES)}] "
            extended_sources.append(base + variation + base)
        sources = extended_sources
        indices = np.random.permutation(len(sources))
    else:
        sources = TEXT_SOURCES
    
    # Split indices
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]
    
    # Expand texts
    train_texts = create_expanded_texts([sources[i] for i in train_idx])
    val_texts = create_expanded_texts([sources[i] for i in val_idx])
    test_texts = create_expanded_texts([sources[i] for i in test_idx])
    
    return train_texts, val_texts, test_texts


def verify_no_leakage(
    train: List[str],
    val: List[str], 
    test: List[str],
) -> bool:
    """Verify no overlap between splits."""
    train_hashes = set(get_text_hash(t) for t in train)
    val_hashes = set(get_text_hash(t) for t in val)
    test_hashes = set(get_text_hash(t) for t in test)
    
    train_val_overlap = train_hashes & val_hashes
    train_test_overlap = train_hashes & test_hashes
    val_test_overlap = val_hashes & test_hashes
    
    if train_val_overlap:
        print(f"WARNING: Train/Val overlap: {train_val_overlap}")
        return False
    if train_test_overlap:
        print(f"WARNING: Train/Test overlap: {train_test_overlap}")
        return False
    if val_test_overlap:
        print(f"WARNING: Val/Test overlap: {val_test_overlap}")
        return False
    
    return True


def texts_to_batches(
    texts: List[str],
    seq_len: int,
    batch_size: int,
    tokenizer: PicoTokenizer = None,
) -> List[torch.Tensor]:
    """Convert texts to batches of token sequences."""
    if tokenizer is None:
        tokenizer = PicoTokenizer()
    
    # Tokenize all texts
    all_tokens = []
    for text in texts:
        all_tokens.extend(tokenizer.encode(text))
    
    # Create sequences
    sequences = []
    for i in range(0, len(all_tokens) - seq_len + 1, seq_len):
        sequences.append(all_tokens[i:i + seq_len])
    
    # Create batches
    batches = []
    for i in range(0, len(sequences) - batch_size + 1, batch_size):
        batch_seqs = sequences[i:i + batch_size]
        batch = torch.tensor(batch_seqs, dtype=torch.long)
        batches.append(batch)
    
    return batches


class DataSplits:
    """Container for train/val/test data splits."""
    
    def __init__(
        self,
        config,
        data_seed: int = 0,  # Separate seed for data (not model init)
    ):
        self.config = config
        self.tokenizer = PicoTokenizer()
        
        # Create splits with no leakage
        train_texts, val_texts, test_texts = split_texts_no_leakage(
            seed=data_seed,
            n_train=config.train_texts,
            n_val=config.val_texts,
            n_test=config.test_texts,
        )
        
        # Verify no leakage
        assert verify_no_leakage(train_texts, val_texts, test_texts), \
            "Data leakage detected!"
        
        # Convert to batches
        self.train_batches = texts_to_batches(
            train_texts, config.max_seq_len, config.batch_size, self.tokenizer
        )
        self.val_batches = texts_to_batches(
            val_texts, config.max_seq_len, config.batch_size, self.tokenizer
        )
        self.test_batches = texts_to_batches(
            test_texts, config.max_seq_len, config.batch_size, self.tokenizer
        )
        
        # Store raw texts (for corpus-derived weights)
        self.train_texts = train_texts
        self.val_texts = val_texts
        self.test_texts = test_texts
        
        print(f"Data splits created:")
        print(f"  Train: {len(self.train_batches)} batches")
        print(f"  Val: {len(self.val_batches)} batches")
        print(f"  Test: {len(self.test_batches)} batches (HELD OUT)")
    
    def get_train_corpus(self) -> List[str]:
        """Get training texts for corpus-derived attention weights."""
        return self.train_texts


if __name__ == "__main__":
    from config import ExperimentConfig
    
    config = ExperimentConfig()
    data = DataSplits(config, data_seed=0)
    
    print(f"\nSample train batch shape: {data.train_batches[0].shape}")
    print(f"Sample val batch shape: {data.val_batches[0].shape}")
    print(f"Sample test batch shape: {data.test_batches[0].shape}")
    
    # Verify splits are different
    print(f"\nTrain text sample: '{data.train_texts[0][:50]}...'")
    print(f"Val text sample: '{data.val_texts[0][:50]}...'")
    print(f"Test text sample: '{data.test_texts[0][:50]}...'")
    
    print("\n✓ Data loader test passed!")


"""
Corpus to One-Hot Matrix Conversion

Creates a vocab × sequence_length matrix from tokenized text data.
Matrix[v, p] = 1 if token v appears at position p, else 0.

Uses FineWeb dataset with streaming for just-in-time loading.
All data is batched into 1024-token chunks.
"""

import numpy as np
from datasets import load_dataset
from typing import Iterator, Generator

from tokenizer import GPT2Tokenizer


# Standard batch size for all operations
BATCH_SIZE = 1024


class CorpusMatrixBuilder:
    """Builds one-hot token matrices from streaming text data in 1024-token batches."""
    
    def __init__(self):
        self.tokenizer = GPT2Tokenizer()
        self.vocab_size = self.tokenizer.vocab_size  # 50257
        self.batch_size = BATCH_SIZE  # 1024 tokens per batch
    
    def stream_fineweb(self, split: str = "train", subset: str = "sample-10BT") -> Iterator[str]:
        """
        Stream text from FineWeb dataset.
        
        Args:
            split: Dataset split ("train")
            subset: FineWeb subset - "sample-10BT" is a 10B token sample
                    Other options: "sample-100BT", "sample-350BT", or full "CC-MAIN-*"
        
        Yields:
            Text strings from the dataset
        """
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=subset,
            split=split,
            streaming=True  # Just-in-time loading
        )
        
        for example in dataset:
            yield example["text"]
    
    def stream_token_batches(
        self,
        subset: str = "sample-10BT",
        max_batches: int = None
    ) -> Generator[list[int], None, None]:
        """
        Stream tokens from FineWeb in 1024-token batches.
        
        Args:
            subset: FineWeb subset to use
            max_batches: Optional limit on number of batches
        
        Yields:
            List of exactly 1024 token IDs per batch
        """
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=subset,
            split="train",
            streaming=True
        )
        
        token_buffer = []
        batch_count = 0
        
        for example in dataset:
            tokens = self.tokenizer.encode(example["text"])
            token_buffer.extend(tokens)
            
            while len(token_buffer) >= self.batch_size:
                yield token_buffer[:self.batch_size]
                token_buffer = token_buffer[self.batch_size:]
                batch_count += 1
                
                if max_batches and batch_count >= max_batches:
                    return
    
    def stream_matrix_batches(
        self,
        subset: str = "sample-10BT",
        max_batches: int = None
    ) -> Generator[tuple[np.ndarray, int], None, None]:
        """
        Stream one-hot matrices in 1024-token batches.
        
        Args:
            subset: FineWeb subset to use
            max_batches: Optional limit on number of batches
        
        Yields:
            Tuple of (matrix of shape (vocab_size, 1024), batch_index)
        """
        for batch_idx, token_batch in enumerate(self.stream_token_batches(subset, max_batches)):
            matrix = self.build_onehot_matrix(token_batch)
            yield matrix, batch_idx
    
    def build_onehot_matrix(self, tokens: list[int]) -> np.ndarray:
        """
        Build one-hot matrix from token sequence.
        
        Args:
            tokens: List of token IDs
        
        Returns:
            numpy array of shape (vocab_size, sequence_length)
            Matrix[v, p] = 1 if token at position p is v
        """
        sequence_length = len(tokens)
        
        # Initialize sparse-like (all zeros)
        matrix = np.zeros((self.vocab_size, sequence_length), dtype=np.uint8)
        
        # Set one-hot positions
        for position, token_id in enumerate(tokens):
            matrix[token_id, position] = 1
        
        return matrix
    
    def build_from_fineweb(
        self,
        num_tokens: int,
        subset: str = "sample-10BT"
    ) -> np.ndarray:
        """
        Build one-hot matrix directly from FineWeb.
        
        Args:
            num_tokens: Number of tokens to include (determines sequence length)
            subset: FineWeb subset to use
        
        Returns:
            One-hot matrix of shape (vocab_size, num_tokens)
        """
        text_stream = self.stream_fineweb(subset=subset)
        token_stream = self.tokenize_stream(text_stream, max_tokens=num_tokens)
        
        # Collect tokens
        tokens = list(token_stream)
        
        return self.build_onehot_matrix(tokens)
    
    def build_from_texts(self, texts: list[str]) -> np.ndarray:
        """
        Build one-hot matrix from a list of texts.
        
        Args:
            texts: List of text strings
        
        Returns:
            One-hot matrix
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenizer.encode(text))
        
        return self.build_onehot_matrix(all_tokens)


def collect_batches(
    num_batches: int,
    subset: str = "sample-10BT"
) -> list[np.ndarray]:
    """
    Convenience function to collect multiple 1024-token batches.
    
    Args:
        num_batches: Number of 1024-token batches to collect
        subset: FineWeb subset to use
    
    Returns:
        List of matrices, each of shape (vocab_size, 1024)
    """
    builder = CorpusMatrixBuilder()
    matrices = []
    
    for matrix, _ in builder.stream_matrix_batches(subset, max_batches=num_batches):
        matrices.append(matrix)
    
    return matrices


def get_token_at_position(matrix: np.ndarray, position: int) -> int:
    """Extract token ID at a given position from one-hot matrix."""
    return int(np.argmax(matrix[:, position]))


def matrix_to_tokens(matrix: np.ndarray) -> list[int]:
    """Convert entire one-hot matrix back to token list."""
    return [int(np.argmax(matrix[:, i])) for i in range(matrix.shape[1])]


if __name__ == "__main__":
    builder = CorpusMatrixBuilder()
    
    print(f"=== Corpus Matrix Builder ===")
    print(f"Batch size: {BATCH_SIZE} tokens")
    print(f"Vocab size: {builder.vocab_size}")
    print(f"Matrix shape per batch: ({builder.vocab_size}, {BATCH_SIZE})")
    print(f"Memory per batch: {(builder.vocab_size * BATCH_SIZE) / (1024 * 1024):.2f} MB")
    
    # Demo with local text
    print(f"\n--- Local Text Demo ---")
    sample_texts = ["Hello world! This is a test of the tokenizer."]
    matrix = builder.build_from_texts(sample_texts)
    
    print(f"Input: {sample_texts[0]}")
    print(f"Tokens: {matrix.shape[1]}")
    
    # Verify roundtrip
    recovered_tokens = matrix_to_tokens(matrix)
    recovered_text = builder.tokenizer.decode(recovered_tokens)
    print(f"Recovered: {recovered_text}")
    
    # Show first few tokens
    print(f"\nToken breakdown:")
    for pos in range(min(5, matrix.shape[1])):
        token_id = get_token_at_position(matrix, pos)
        token_str = builder.tokenizer.token_to_string(token_id)
        print(f"  [{pos}] token {token_id:5d} = '{token_str}'")
    
    print(f"\n--- To stream FineWeb in 1024-token batches: ---")
    print("for matrix, batch_idx in builder.stream_matrix_batches(max_batches=10):")
    print("    # matrix.shape = (50257, 1024)")
    print("    process(matrix)")


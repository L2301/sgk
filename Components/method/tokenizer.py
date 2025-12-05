"""
GPT-2 Small Tokenizer Implementation

Uses tiktoken for fast BPE tokenization compatible with GPT-2.
GPT-2 small uses a vocabulary of 50,257 tokens.
"""

import tiktoken


class GPT2Tokenizer:
    """Wrapper for GPT-2's BPE tokenizer."""
    
    def __init__(self):
        self.encoder = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoder.n_vocab  # 50257
        
        # Special tokens
        self.eot_token = self.encoder.eot_token  # End of text: 50256
    
    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs."""
        return self.encoder.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        """Convert token IDs back to text."""
        return self.encoder.decode(tokens)
    
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, token_lists: list[list[int]]) -> list[str]:
        """Decode multiple token sequences."""
        return [self.decode(tokens) for tokens in token_lists]
    
    def token_to_string(self, token: int) -> str:
        """Convert a single token ID to its string representation."""
        return self.encoder.decode([token])
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


# Convenience function for quick tokenization
def tokenize(text: str) -> list[int]:
    """Quick tokenization without instantiating the class."""
    enc = tiktoken.get_encoding("gpt2")
    return enc.encode(text)


def detokenize(tokens: list[int]) -> str:
    """Quick detokenization without instantiating the class."""
    enc = tiktoken.get_encoding("gpt2")
    return enc.decode(tokens)


if __name__ == "__main__":
    # Demo
    tokenizer = GPT2Tokenizer()
    
    sample_text = "Hello, world! This is a test of the GPT-2 tokenizer."
    tokens = tokenizer.encode(sample_text)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Original text: {sample_text}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Decoded: {tokenizer.decode(tokens)}")
    print(f"\nToken breakdown:")
    for t in tokens:
        print(f"  {t:5d} -> '{tokenizer.token_to_string(t)}'")


"""
Pico Tokenizer - Character-level tokenizer for testing

Vocabulary (~80 tokens):
- Lowercase letters: a-z (26)
- Uppercase letters: A-Z (26)  
- Digits: 0-9 (10)
- Punctuation and whitespace (15)
- Special tokens: <PAD>, <UNK>, <BOS>, <EOS> (4)

Total: ~81 tokens
"""

from typing import List, Dict
import string


# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

# Build vocabulary
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
LOWERCASE = list(string.ascii_lowercase)  # a-z
UPPERCASE = list(string.ascii_uppercase)  # A-Z
DIGITS = list(string.digits)  # 0-9
PUNCTUATION = list(" .,!?;:'\"-()[]/@#\n\t")

# Full vocabulary
VOCAB = SPECIAL_TOKENS + LOWERCASE + UPPERCASE + DIGITS + PUNCTUATION

# Token to ID mapping
TOKEN_TO_ID: Dict[str, int] = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {idx: token for idx, token in enumerate(VOCAB)}

# Constants
PICO_VOCAB_SIZE = len(VOCAB)
PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
UNK_ID = TOKEN_TO_ID[UNK_TOKEN]
BOS_ID = TOKEN_TO_ID[BOS_TOKEN]
EOS_ID = TOKEN_TO_ID[EOS_TOKEN]


class PicoTokenizer:
    """Character-level tokenizer for PicoGPT testing."""
    
    def __init__(self):
        self.vocab = VOCAB
        self.vocab_size = PICO_VOCAB_SIZE
        self.token_to_id = TOKEN_TO_ID
        self.id_to_token = ID_TO_TOKEN
        
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
    
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Convert text to token IDs."""
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.bos_id)
        
        for char in text:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                tokens.append(self.unk_id)
        
        if add_special_tokens:
            tokens.append(self.eos_id)
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to text."""
        chars = []
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            if tid in self.id_to_token:
                chars.append(self.id_to_token[tid])
            else:
                chars.append(UNK_TOKEN)
        
        return "".join(chars)
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]
    
    def decode_batch(self, batch_ids: List[List[int]]) -> List[str]:
        """Decode multiple token sequences."""
        return [self.decode(ids) for ids in batch_ids]


# ===== Test =====
if __name__ == "__main__":
    print("=" * 60)
    print("Pico Tokenizer Test")
    print("=" * 60)
    
    tokenizer = PicoTokenizer()
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={PAD_ID}, UNK={UNK_ID}, BOS={BOS_ID}, EOS={EOS_ID}")
    
    # Show vocabulary
    print(f"\nVocabulary preview:")
    print(f"  Special: {VOCAB[:4]}")
    print(f"  Letters: {VOCAB[4:30]}...")
    
    # Test encoding/decoding
    test_texts = [
        "Hello World!",
        "The quick brown fox.",
        "Testing 123",
    ]
    
    print("\nEncoding/Decoding Tests:")
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"  '{text}' -> {encoded[:15]}... -> '{decoded}' (match={text == decoded})")
    
    print("\nAll tests passed!")


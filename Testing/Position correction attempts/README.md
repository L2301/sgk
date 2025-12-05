# Position Encoding Correction Attempts

This folder contains two approaches to correct the position encoding issue in frozen attention models.

## Problem

The original frozen attention implementation computes W_Q, W_K, W_V, W_O from corpus statistics using **base token embeddings only**. However, during forward passes, the model adds **positional encodings**, creating a mismatch:

- **Weight computation**: Uses `E_token`
- **Forward pass**: Uses `E_token + E_pos`

This causes the frozen attention weights to not properly capture intended skip-n patterns when positional information is present.

## Solutions

### 1. Sinusoidal Position Encoding + Skip-n Masking
**File**: `pico_sparse_sinusoidal.py`

- Uses standard sinusoidal positional encodings (as in "Attention is All You Need")
- Enforces skip-n attention through explicit masking
- Token at position `i` can **only** attend to position `i-(n+s)` where:
  - `n = layer_idx + 1`
  - `s = head_idx`
- All other positions masked with `-inf` before softmax

**Advantages**:
- Simple and direct
- Well-understood sinusoidal encoding
- Explicit control over attention pattern

### 2. RoPE Position Encoding + Skip-n Masking
**File**: `pico_sparse_rope.py`

- Uses Rotary Position Embeddings (RoPE)
- Applies rotation to Q and K based on position
- Also uses skip-n masking as safety mechanism
- No additive positional embeddings

**Advantages**:
- RoPE encodes relative position information naturally
- More parameter-efficient (no position embedding table)
- Theoretically cleaner for relative position modeling

## Usage

### Testing Both Approaches

```bash
cd "/Users/master/Attention Experiments/Implementing/Testing/Position correction attempts"
python test_position_encoding.py
```

This will:
1. Create test models with both approaches
2. Verify attention patterns match skip-n behavior
3. Compare concentration ratios
4. Report which approach better enforces skip-n patterns

### Using in Training

```python
from pico_sparse_sinusoidal import PicoGPTSparseSinusoidal
from pico_sparse_rope import PicoGPTSparseRoPE
from pico_config import PicoModelConfig

# Compute attention weights (same as before)
attention_weights, base_embeddings = compute_pico_weights(...)

# Create model with sinusoidal encoding
model_sin = PicoGPTSparseSinusoidal(
    config, 
    attention_weights, 
    base_embeddings
)

# OR create model with RoPE
model_rope = PicoGPTSparseRoPE(
    config, 
    attention_weights, 
    base_embeddings
)
```

## Key Differences from Original

| Feature | Original `pico_sparse.py` | Sinusoidal Variant | RoPE Variant |
|---------|---------------------------|-------------------|--------------|
| Position Encoding | Learned additive | Sinusoidal (fixed) | RoPE (rotary) |
| Attention Masking | Causal only | Causal + Skip-n | Causal + Skip-n |
| Position Parameters | `max_seq_len × embed_dim` | 0 (fixed) | 0 (no table) |
| Attention Pattern | Full causal | Skip-n only | Skip-n only |

## Files

- `pico_sparse_sinusoidal.py` - Sinusoidal encoding implementation
- `pico_sparse_rope.py` - RoPE encoding implementation  
- `test_position_encoding.py` - Test and comparison script
- `README.md` - This file

## Next Steps

1. Run tests to verify both implementations work correctly
2. Train both variants on the same data
3. Compare perplexity and loss curves
4. Analyze which approach better captures skip-n patterns in practice
5. Potentially combine insights from both approaches

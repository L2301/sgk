# G_k Correction for Position Encoding

## The Problem

When computing attention weights from corpus with positional encodings, the attention score includes:

```
(e_x + P(i))^T W_Q W_K^T (e_y + P(i+k))
```

Expanding:
```
= e_x^T W_Q W_K^T e_y                    [semantic term]
+ e_x^T W_Q W_K^T P(i+k)                 [token-position cross-term]
+ P(i)^T W_Q W_K^T e_y                   [position-token cross-term]
+ P(i)^T W_Q W_K^T P(i+k)                [position-position term]
```

## What M_k Captures

The manifold matrix M_k = E^+ T_k E approximates:
```
M_k ≈ W_Q W_K^T + (position-induced bias)
```

The bias comes from positional encodings creating uniform boost across all distances.

## The Solution: G_k Correction

Define:
```
G_k = E_i[P(i+k)P(i)^T]
```

This is the **average positional contribution at distance k**.

After subtraction:
```
M_k,free = M_k - G_k
```

Now `M_k,free` contains only genuine skip-k semantic structure, with positional baseline removed.

## Implementation

### 1. Compute G_k

```python
def compute_G_k(position_encoding, k, max_pos=1000):
    # Get P(i) and P(i+k) for all valid positions
    # Average the outer products P(i+k) @ P(i)^T
    G_k_sum = 0
    for i in range(max_pos - k):
        P_i = position_encoding[i]
        P_i_k = position_encoding[i + k]
        G_k_sum += outer(P_i_k, P_i)
    
    return G_k_sum / (max_pos - k)
```

### 2. Correct Manifold Matrix

```python
# Project G_k into token space
G_k_projected = E_n @ G_k @ E_1^T

# Subtract from manifold matrix
M_corrected = M_k - G_k_projected
```

### 3. Derive W_Q, W_K from Corrected Matrix

```python
# SVD on corrected matrix
U, sigma, V^T = SVD(M_corrected)

# Derive weights as before
W_Q = E_n^T @ U @ sqrt(Sigma)
W_K = E_1^T @ V @ sqrt(Sigma)
```

## Files

- `pico_sparse_sinusoidal_gk.py` - Model with G_k-corrected attention
- `pico_method_gk.py` - Weight computation with G_k correction

## Expected Benefit

By removing positional bias, the model should show **sharper peaks at correct skip-k distances** in attention patterns, leading to better semantic structure capture.

# Deep Attention Analysis Summary

## Key Findings

### 1. Normalization Scale Analysis

**Critical Discovery**: The normalization scale has a MASSIVE impact on attention quality!

| Scale | Entropy | Max Attention | Score Magnitude | Quality |
|-------|---------|---------------|-----------------|---------|
| 0.001 | 2.55 (uniform!) | 0.13 | 0.000002 | ❌ Too small |
| 0.01  | 2.24 (nearly uniform) | 0.20 | 0.000192 | ❌ Still too small |
| 0.02  | 1.88 | 0.32 | 0.000769 | ⚠️ Getting better |
| 0.05  | 1.36 | 0.52 | 0.004804 | ✅ Much better |
| 0.1   | 0.71 | 0.74 | 0.019216 | ✅ Good |
| 0.2   | 0.20 | 0.93 | 0.076865 | ✅ Very focused |
| 0.5   | 0.03 | 0.99 | 0.480404 | ✅ Extremely focused |
| 1.0   | 0.01 | 0.99 | 1.921617 | ⚠️ Maybe too large |

**Conclusion**: 
- Current scale (0.01) produces **nearly uniform attention** (entropy 2.24/3.47 = 65% of max)
- Need scale **0.05-0.2** for meaningful attention patterns
- At scale 0.1, attention becomes focused (entropy 0.71, max attention 0.74)

### 2. Q-K Orthogonality Analysis

**Why are Q and K orthogonal?**

1. **Manifold matrix M is NOT symmetric**: `M != M.T`
   - This means U and V are different (not just sign differences)
   - W_Q uses U (left singular vectors)
   - W_K uses V (right singular vectors)
   - Since U ≠ V, W_Q and W_K project into different subspaces

2. **Embeddings are nearly orthogonal**: 
   - E_n @ E_1.T has mean 0.0145, std 0.124
   - This is very low similarity
   - Base embeddings are unit vectors, so they're designed to be orthogonal

3. **U and V are NOT aligned with embeddings**:
   - U columns vs E_n rows: mean similarity -0.0015 (essentially zero)
   - V columns vs E_1 rows: mean similarity -0.0008 (essentially zero)
   - The SVD is finding structure in M, not in the embeddings

4. **W_Q.T @ W_K structure**:
   - Mean: 0.0007 (very small)
   - Max abs: 1.02 (some directions align)
   - Diagonal: [0.004, 0.003, 0.008, ...] (small but non-zero)
   - **The orthogonality comes from U and V being different, not from intentional design**

**Conclusion**: Q and K are orthogonal because:
- The manifold matrix M is asymmetric
- U and V capture different aspects of M
- The embeddings don't align with U/V directions
- This is a **consequence of the derivation**, not necessarily optimal

### 3. SVD Structure Analysis

**SVD Results for M^(1,0)**:
- Shape: (85, 85)
- Density: 22.4% (sparse)
- Top singular values: [0.018, 0.021, 0.023, ...]
- Condition number: 0.02 (very well-conditioned)
- U and V are properly orthogonal (error < 1e-6)

**Computed W_Q and W_K**:
- W_Q: mean=-0.000064, std=0.025244
- W_K: mean=0.000303, std=0.025033
- W_Q vs W_K cosine similarity: 0.120 (low but not zero)

**The SVD is working correctly**, but:
- The singular values are very small (0.018-0.035)
- After sqrt(S) scaling, the weights are small
- The normalization to std=0.01 further reduces them
- **This double-reduction makes attention scores tiny**

## Root Cause Analysis

### Why Sparse ≈ Zero with Frozen Embeddings?

1. **Normalization too small**: Scale 0.01 → attention scores ~0.0002 → nearly uniform attention
2. **Q-K orthogonality**: Low similarity (0.014) → small attention scores
3. **Small singular values**: Original SVD values are small → weights are small
4. **Double reduction**: sqrt(S) reduces magnitude, then normalization reduces again

### The Fix

**Option 1: Increase normalization scale**
- Use scale 0.1-0.2 instead of 0.01
- This should produce focused attention patterns
- Trade-off: May make attention too focused (entropy 0.2-0.7)

**Option 2: Don't normalize, or normalize differently**
- Use raw SVD-derived weights
- Or normalize to match trainable attention magnitude (not fixed std)

**Option 3: Fix the derivation**
- The manifold matrix M might need different normalization
- Or the embedding computation might need adjustment
- Or the SVD should use more dimensions

## Recommendations

1. **Immediate**: Test with normalization scale 0.1-0.2
2. **Short-term**: Analyze if M should be symmetric (maybe use (M + M.T)/2?)
3. **Long-term**: Revisit the theoretical derivation - are we computing the right matrices?

## Files Generated

- `attention_deep_analysis/normalization_scale_analysis.png` - Visualization of scale effects
- All intermediate SVD results logged to console


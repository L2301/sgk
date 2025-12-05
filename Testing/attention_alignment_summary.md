# Why Attention Patterns Don't Align with Count Matrices

## The Fundamental Mismatch

### What Count Matrices Capture
- **C(n,s)** is `(vocab^n, vocab)` - captures **TOKEN co-occurrence**
- Example: "After n-gram 'the cat', token 'sat' appears 50% of the time"
- This is about **TOKEN TYPES**, not **POSITIONS**

### What Attention Computes
- **score[i,j] = Q_i @ K_j = (E_i @ W_Q) @ (E_j @ W_K)**
- This depends on **EMBEDDING similarity** at positions i and j
- This is about **POSITION-to-POSITION** relationships via embeddings

### The Problem
**The mapping from token co-occurrence to position attention is LOST!**

## Evidence

### Alignment Metrics (Layer 0, Head 0)
- **KL divergence**: 0.85 (high = poor alignment)
- **Cosine similarity**: 0.61 (moderate)
- **Max position matches**: 12.9% (very poor!)
- Example: Position 10 expected to attend to position 8, but actually attends to position 0

### Why This Happens

1. **Embeddings are nearly orthogonal**:
   - Embedding similarity matrix shows values ~0.01 (essentially zero)
   - Different tokens have orthogonal embeddings
   - This means `E_i @ E_j ≈ 0` for different tokens

2. **W_Q and W_K don't preserve count structure**:
   - W_Q = E^n.T @ U @ sqrt(S)
   - W_K = E^1.T @ V @ sqrt(S)
   - These project embeddings into SVD space, but embeddings don't encode n-gram relationships

3. **The derivation loses positional information**:
   - Count matrix: "ngram -> token" (type-level)
   - Attention: "position -> position" (instance-level)
   - No mechanism to map "token A follows n-gram B" to "attend to position where A appears"

## The Root Cause

**The count matrices capture TOKEN co-occurrence patterns, but the attention mechanism operates on EMBEDDING similarity. If embeddings don't encode the n-gram structure, attention won't align!**

### Why Embeddings Don't Encode N-gram Structure

1. **Base embeddings are random unit vectors**:
   - Created via `create_base_embeddings()` - random near-orthogonal vectors
   - No relationship to token co-occurrence

2. **N-gram embeddings are scaled by probability**:
   - E^n = P^{n-1} * E_base
   - This scales the embedding, but doesn't change its direction
   - Similar n-grams still have orthogonal embeddings

3. **SVD doesn't help**:
   - SVD finds structure in M (the manifold matrix)
   - But M is about token types, not positions
   - W_Q and W_K project embeddings, but embeddings don't encode position relationships

## What Should Happen (But Doesn't)

**Ideal scenario:**
1. Count matrix says: "After 'the cat', 'sat' appears 50%"
2. When we see 'the cat' at position i, we should attend to positions where 'sat' appears
3. But embeddings for 'the', 'cat', and 'sat' are orthogonal
4. So Q_i @ K_j ≈ 0 for all j, regardless of what tokens are there
5. Attention becomes uniform (or nearly so)

## Solutions (Theoretical)

### Option 1: Embeddings Should Encode Co-occurrence
- Embeddings should be learned to encode: "tokens that co-occur have similar embeddings"
- But we're using frozen random embeddings!

### Option 2: Direct Position-to-Position Mapping
- Instead of deriving W_Q/W_K from count matrices, directly compute position attention
- But this loses the theoretical foundation

### Option 3: Different Derivation
- Maybe the derivation should preserve positional relationships
- Or maybe we need to use the count matrices differently

### Option 4: Learnable Embeddings (But Frozen Attention)
- Allow embeddings to learn, but keep attention frozen
- But this breaks the theoretical foundation (embeddings used to compute attention)

## Current State

- **Layer 0 (n=1)**: Best alignment (cosine ~0.61-0.74, max match 12-35%)
- **Layer 1+ (n=2)**: Worse alignment (cosine ~0.33-0.64, max match 0-20%)
- **Deeper layers**: Even worse (KL divergence increases, cosine decreases)

**Conclusion**: The attention patterns are NOT attending to the positions suggested by the count matrices because:
1. Embeddings don't encode n-gram co-occurrence
2. The derivation maps token types to attention, but attention needs position-to-position
3. The normalization makes attention nearly uniform anyway

## Files Generated

- `attention_alignment_analysis/expected_vs_actual_attention.png` - Side-by-side comparison
- Alignment metrics for all layers/heads logged to console


# Spatial Locality and CPU Cache Memory

## Memory Layout (Row-Major)

NumPy arrays use row-major (C-order) storage by default. This means elements in the same row are stored contiguously in memory:

**Matrix B (4×4):**
```
B[0,0] B[0,1] B[0,2] B[0,3]  ← stored consecutively in memory
B[1,0] B[1,1] B[1,2] B[1,3]  ← next consecutive block
...
```

## Cache Line Behavior

CPUs fetch data in cache lines (typically 64 bytes = 16 float32s). When you access `B[k,j]`, the CPU loads that element plus nearby elements into cache.

## Loop Order Comparison

### ijk (naive_gemm) - BAD for cache

```python
for i in range(M):
    for j in range(N):
        for k in range(K):  # innermost
            s += A[i, k] * B[k, j]  # B access pattern
                             # ↓
```

- **B access pattern:** `B[0,j]`, `B[1,j]`, `B[2,j]`, ... - walking down a COLUMN
- **Problem:** Column elements are K elements apart in memory (stride = N*4 bytes)
- **Result:** Each `B[k,j]` access likely cache miss - fetches a cache line but only uses 1 element

### ikj (cache_optimized) - GOOD for cache

```python
for i in range(M):
    for k in range(K):
        a = A[i, k]
        for j in range(N):  # innermost
            C[i, j] += a * B[k, j]  # B access pattern
                               # →
```

- **B access pattern:** `B[k,0]`, `B[k,1]`, `B[k,2]`, ... - walking across a ROW
- **Benefit:** Row elements are contiguous (stride = 4 bytes)
- **Result:** One cache line fetch gives you ~16 elements - 15 subsequent accesses are cache hits!

Similarly, `C[i,:]` is accessed sequentially (writing to consecutive memory locations).

## Concrete Example

For a 128×128 matrix with float32 (4 bytes each):

**ijk order:**
- 128³ = 2M iterations
- B column access: ~2M cache misses (worst case)

**ikj order:**
- Same 2M iterations
- B row access: ~2M/16 = 125K cache misses (best case with 64-byte lines)

**~16x fewer cache misses → dramatically faster execution!**

This is why your `naive_gemm_cache_optimized` with ikj ordering will be significantly faster than the ijk version, even though they do the same number of FLOPs.

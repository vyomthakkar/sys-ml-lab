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

## Key Insights

Performance isn't just about FLOP count - both versions do exactly 2MN*K operations, but cache-optimized (ikj) should be **2-5x faster** for medium sizes. This demonstrates the **memory wall**: modern CPUs can execute billions of FLOPs/sec, but waiting for data from RAM (cache miss penalty ~100-200 cycles) dominates execution time. The ikj loop order exploits spatial locality to keep data in L1/L2 cache.

Even with this optimization, NumPy will still be **10-50x faster** because it adds:
- **Vectorization** (SIMD instructions processing 4-8 floats at once)
- **Register blocking** (keeping hot data in CPU registers)
- **Multi-threading**

This hierarchy (naive → cache-aware → vectorized → parallel) is the foundation of high-performance computing.

## Why Cache-Optimized is SLOWER in Python

**Important caveat:** The cache optimization (ikj loop order) is **slower in Python** but would be faster in C/C++. Here's why:

### The Problem: Python Overhead

**Naive (ijk) - Fewer Array Accesses:**
```python
for i in range(M):
    for j in range(N):
        s = 0  # Python variable (fast!)
        for k in range(K):
            s += A[i, k] * B[k, j]  # Accumulate in Python variable
        C[i, j] = s  # Write to array ONCE
```
**Array writes to C:** M × N (one per output element)

**Cache-opt (ikj) - More Array Accesses:**
```python
for i in range(M):
    for k in range(K):
        a = A[i, k]  # Python variable
        for j in range(N):
            C[i, j] += a * B[k, j]  # Read C[i,j], modify, write back
```
**Array accesses to C:** M × N × K (read-modify-write K times per element!)

### The Numbers

For 128×128×128:
- **Naive:** 128² = 16,384 writes to C
- **Cache-opt:** 128³ = 2,097,152 read-modify-write operations to C (128x more!)

Each `C[i,j] += ...` in Python involves:
1. NumPy array indexing overhead
2. Reading from C
3. Addition
4. Writing back to C

In pure Python, **NumPy array indexing is expensive** compared to simple variable operations.

### Why This Works in C/C++

In compiled code, the compiler recognizes that `C[i,j]` is reused across the k-loop and **keeps it in a CPU register**. The read-modify-write becomes essentially free. Plus, you get the cache benefits for B and C accessing sequential memory.

### The Lesson

**Language matters for optimization!** Cache-aware algorithms assume compiled code where the compiler can optimize away redundant memory accesses. In Python, the interpreter executes each statement literally - `C[i,j] += x` is a separate array access every time, not optimized away.

This demonstrates why **Python + NumPy is essential for numerical computing:** NumPy's C implementation handles all matrix operations in compiled code where cache optimizations actually work. Your naive Python loop does 2MNK operations with massive overhead per operation. NumPy does the same 2MNK operations in tight, vectorized, cache-optimized C code - hence the 68,000x speedup!

**Takeaway:** Don't optimize Python loops - use NumPy/libraries that drop into C/Fortran where real optimizations matter.

## Arithmetic Intensity: Understanding Performance Bottlenecks

**Arithmetic Intensity (AI)** is the key metric for understanding performance bottlenecks. It's the ratio of FLOPs to bytes moved:

```
AI = operations / data_transfer
```

### For Naive GEMM

- **FLOPs**: 2×M×N×K (the computation work)
- **Memory bytes**: ~2×M×N×K×4 (worst-case data movement for ijk order)
- **AI ≈ 0.25 FLOPs/byte** (less than 1!)

This means you're moving **4 bytes for every 1 operation** - classic memory-bound behavior. The CPU spends most time waiting for data, not computing. Modern CPUs can do 10-100 FLOPs per byte of memory bandwidth, but naive GEMM only does 0.25.

### Performance Classification

- **AI < 1**: **Memory-bound** (more data movement than compute)
- **AI = 1-10**: **Balanced** (memory and compute similar)
- **AI > 10**: **Compute-bound** (more math than data movement)

### Why Blocking/Tiling Matters

This is why **blocking/tiling** is crucial: by reusing data in cache, you can do many operations on the same bytes, increasing AI. Optimized BLAS libraries achieve AI > 10 by:

1. **Blocking**: Process small tiles that fit in L1/L2 cache
2. **Reusing C[i,j]**: Keep output in registers across k-loop
3. **Vectorization**: 4-8 operations per memory load with SIMD

### The Roofline Model: https://claude.ai/chat/28d07648-b72d-4d1e-b518-6d9559b8d904

The relationship between arithmetic intensity and performance follows the **Roofline Model**:

```
Performance = min(Peak_Compute, AI × Memory_Bandwidth)
```

For low AI (< 1), you're limited by memory bandwidth. For high AI (> 10), you're limited by compute throughput. Naive GEMM sits firmly in the memory-bound region, which is why optimizations focus on improving data reuse to increase AI.

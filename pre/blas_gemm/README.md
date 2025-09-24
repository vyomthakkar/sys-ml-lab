# Day 2: BLAS/GEMM Intuition

## Learning Objectives
By the end of this 30-minute session, you will:
- Understand the computational complexity of matrix multiplication (2MNK FLOPs)
- Estimate memory bandwidth requirements for GEMM operations  
- Implement a naive matrix multiplication and measure its performance
- Appreciate why optimized BLAS libraries are essential for performance
- Develop intuition for performance analysis fundamentals

## Project Structure
```
day2_blas_gemm/
├── README.md                 # This file
├── naive_gemm.py            # Core implementation and benchmarks
├── analyze_performance.py   # Performance analysis tools
├── test_correctness.py      # Validation against NumPy
└── results/                 # Output directory for measurements
    ├── timings.txt
    └── analysis.txt
```

## Background: Why This Matters

Matrix multiplication (`C = A @ B`) is the backbone of:
- Neural network forward/backward passes
- Scientific computing workloads  
- Computer graphics transformations
- Signal processing algorithms

**The FLOP Count**: For matrices A(M×K) @ B(K×N) → C(M×N), we need **2MNK** operations:
- MNK multiplications (A[i,k] * B[k,j])
- MNK additions (accumulating sums)

**Memory Access**: We touch M×K + K×N + M×N elements, but with poor locality this can explode.

## Core Assignment (30 minutes)

### Task 1: Implement Naive GEMM (10 minutes)

### Task 2: Predict Before You Run (5 minutes)

**Before running your code**, write down predictions in `results/predictions.txt`:

```
My predictions for 128×128×128 GEMM:

1. FLOPs: ___ million (calculate 2*M*N*K)
2. Naive time: ~___ seconds 
3. NumPy speedup: ___x faster than naive
4. Bottleneck: memory-bound or compute-bound? ___

My reasoning:
- 
- 
```

### Task 3: Run and Analyze (10 minutes)

1. **Run the benchmark**: `python naive_gemm.py`
2. **Compare with predictions**: How close were you?
3. **Calculate arithmetic intensity**: `FLOPs / memory_bytes`
   - If <1, likely memory-bound
   - If >10, likely compute-bound

### Task 4: Quick Performance Analysis (5 minutes)

Create a simple analysis script or add to your main file:

```python
def analyze_performance(results):
    """Analyze whether kernel is memory or compute bound"""
    flops = results['flops']
    memory_bytes = results['memory_bytes']
    naive_time = results['naive_time']
    
    arithmetic_intensity = flops / memory_bytes
    memory_bandwidth_used = memory_bytes / naive_time / 1e9  # GB/s
    
    print(f"\n=== Performance Analysis ===")
    print(f"Arithmetic Intensity: {arithmetic_intensity:.2f} FLOPs/byte")
    print(f"Memory BW used (naive): {memory_bandwidth_used:.2f} GB/s")
    print(f"Peak memory BW estimate: ~50-200 GB/s (typical CPU)")
    
    if arithmetic_intensity < 1:
        print("→ Likely MEMORY-BOUND (more data movement than compute)")
    elif arithmetic_intensity > 10:
        print("→ Likely COMPUTE-BOUND (more math than data movement)")
    else:
        print("→ BALANCED (memory and compute similar)")
```

## Success Criteria

✅ **Basic**: Your naive implementation matches NumPy output (error < 1e-5)  
✅ **Basic**: You can explain why NumPy is 10-100x faster  
✅ **Good**: You correctly predicted the FLOP count and memory usage  
✅ **Great**: You can identify if your naive version is memory or compute bound  

## Common Pitfalls & Tips

⚠️ **Don't go too big**: 256×256 naive GEMM takes ~30 seconds!  
⚠️ **Use float32**: Faster than float64, sufficient precision  
⚠️ **Warmup runs**: First run may be slower due to cache misses  
⚠️ **Memory layout**: NumPy defaults to C-order (row-major)  

💡 **Pro tip**: Try swapping the inner loops (ikj vs ijk vs jik). Cache locality matters!

## Optional Extensions (if you have extra time)

1. **Loop order experiment**: Try `ikj` vs `ijk` loop ordering, measure the difference
2. **Data types**: Compare float32 vs float64 performance  
3. **Blocking/tiling**: Implement a simple blocked version (16×16 tiles)
4. **Profiling**: Use `perf stat` or Python `cProfile` on your naive version

## References & Next Steps

- **BLAS Levels**: Level 1 (vector), Level 2 (matrix-vector), Level 3 (matrix-matrix)
- **Why NumPy is fast**: Uses optimized BLAS (OpenBLAS, Intel MKL, etc.)
- **Tomorrow**: We'll explore memory hierarchy and cache optimization
- **Reading**: "What Every Programmer Should Know About Memory" by Ulrich Drepper

---

*Time budget: Core assignment ~30min, extensions optional*
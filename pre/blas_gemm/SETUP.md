# Day 2 BLAS/GEMM - Quick Start Guide

## 5-Minute Setup

```bash
# 1. Create project directory
mkdir day2_blas_gemm && cd day2_blas_gemm

# 2. Copy the provided files:
# - naive_gemm.py (main implementation)
# - test_correctness.py (validation)
# - setup_check.py (environment check)

# 3. Create results directory
mkdir results

# 4. Test your environment
python setup_check.py

# Expected output:
# ✓ NumPy X.X.X available
# ✓ NumPy configuration looks good  
# ✓ NumPy matrix multiplication working
# ✓ timing module available
# 🎉 Environment ready for Day 2!
```

## 30-Minute Learning Path

### Minutes 0-5: Setup & Theory
- [ ] Complete TODO sections in `naive_gemm.py` 
- [ ] Understand the 2MNK FLOP formula
- [ ] Fill out your predictions in `results/predictions.txt`

### Minutes 5-15: Implementation
- [ ] Run `python test_correctness.py` - fix any bugs
- [ ] Run `python naive_gemm.py` - get your first benchmark
- [ ] Compare actual vs predicted performance

### Minutes 15-25: Analysis
- [ ] Calculate arithmetic intensity for your test cases
- [ ] Determine if naive version is memory or compute bound  
- [ ] Document 3 key learnings in results file

### Minutes 25-30: Wrap-up
- [ ] Try one extension (loop reordering, different sizes, etc.)
- [ ] Note questions for tomorrow's memory hierarchy topic

## Key Formulas to Remember

```
FLOPs = 2 × M × N × K
Memory (naive) ≈ M×K + K×N + M×N elements × sizeof(dtype)
Arithmetic Intensity = FLOPs / Memory Bytes
GFLOP/s = FLOPs / (time in seconds × 1e9)
```

## What Success Looks Like

After 30 minutes, you should be able to:

✅ **Explain** why matrix multiplication takes 2MNK operations  
✅ **Predict** performance bottlenecks using arithmetic intensity  
✅ **Measure** GFLOP/s and memory bandwidth for simple kernels  
✅ **Appreciate** why optimized BLAS libraries matter (10-100x speedup!)  

## Common Issues & Solutions

**Issue**: Naive GEMM too slow on large matrices  
**Solution**: Start with 64×64, only go to 128×128 if you have time  

**Issue**: Import errors with NumPy  
**Solution**: `pip install numpy` or use conda  

**Issue**: Numerical differences between naive and NumPy  
**Solution**: Use `np.allclose()` with reasonable tolerance (~1e-5)  

**Issue**: Confusing performance numbers  
**Solution**: Focus on the relative speedup, not absolute GFLOP/s  

## Tomorrow's Preview

**Day 3: Memory Hierarchy**
- We'll explore *why* NumPy is so much faster
- Learn about cache levels (L1/L2/L3) 
- Implement cache-friendly blocked matrix multiplication
- Use profiling tools like cachegrind

The performance gap you measured today is mostly due to cache locality - tomorrow you'll learn to fix it!

---

**Time commitment**: ~30 minutes core + optional extensions  
**Prerequisites**: Basic Python, NumPy installed  
**Output**: Performance intuition + benchmark data for your learning log
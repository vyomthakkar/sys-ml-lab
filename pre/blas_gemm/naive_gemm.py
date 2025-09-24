import numpy as np
import time
from typing import Tuple

def naive_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Naive matrix multiplication C = A @ B
    A: (M, K), B: (K, N) -> C: (M, N)
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"
    
    # TODO: Implement the triple nested loop
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            s = 0 #local accumulator to reduce repeated writes to C and indexing overhead
            for k in range(K):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

def estimate_flops(M: int, N: int, K: int) -> int:
    """Estimate FLOPs for M×K @ K×N matrix multiplication"""
    # TODO: Return the theoretical FLOP count
    pass

def estimate_memory_bytes(M: int, N: int, K: int, dtype=np.float32) -> int:
    """Estimate memory bytes accessed (naive upper bound)"""
    element_size = np.dtype(dtype).itemsize
    # A: M×K, B: K×N, C: M×N (write once, potentially read multiple times)
    # Worst case: we read A and B completely for each output element
    reads = M * N * (K + K)  # Each C[i,j] reads K elements from A and B
    writes = M * N           # Write each C element once
    return (reads + writes) * element_size

def benchmark_gemm(M: int, N: int, K: int, num_runs: int = 5) -> dict:
    """Benchmark naive vs NumPy GEMM"""
    
    # Generate test matrices
    np.random.seed(42)
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Theoretical estimates
    flops = estimate_flops(M, N, K)
    memory_bytes = estimate_memory_bytes(M, N, K)
    
    results = {
        'M': M, 'N': N, 'K': K,
        'flops': flops,
        'memory_bytes': memory_bytes,
    }
    
    # Benchmark naive implementation
    times_naive = []
    for _ in range(num_runs):
        start = time.perf_counter()
        C_naive = naive_gemm(A, B)
        times_naive.append(time.perf_counter() - start)
    
    results['naive_time'] = min(times_naive)
    results['naive_gflops'] = flops / (results['naive_time'] * 1e9)
    
    # Benchmark NumPy
    times_numpy = []
    for _ in range(num_runs):
        start = time.perf_counter()
        C_numpy = A @ B  # Or np.dot(A, B)
        times_numpy.append(time.perf_counter() - start)
    
    results['numpy_time'] = min(times_numpy)
    results['numpy_gflops'] = flops / (results['numpy_time'] * 1e9)
    results['speedup'] = results['naive_time'] / results['numpy_time']
    
    # Verify correctness
    results['max_error'] = np.max(np.abs(C_naive - C_numpy))
    
    return results

if __name__ == "__main__":
    print("=== BLAS/GEMM Intuition Benchmark ===")
    
    # Test sizes - start small!
    test_sizes = [
        (64, 64, 64),    # Small
        (128, 128, 128), # Medium  
        (256, 256, 256), # Large (will be slow!)
    ]
    
    for M, N, K in test_sizes:
        print(f"\nTesting {M}×{K} @ {K}×{N}:")
        result = benchmark_gemm(M, N, K)
        
        print(f"  FLOPs: {result['flops']:,}")
        print(f"  Memory bytes: {result['memory_bytes']:,}")
        print(f"  Naive time: {result['naive_time']:.4f}s ({result['naive_gflops']:.2f} GFLOP/s)")
        print(f"  NumPy time: {result['numpy_time']:.4f}s ({result['numpy_gflops']:.2f} GFLOP/s)")
        print(f"  Speedup: {result['speedup']:.1f}x")
        print(f"  Max error: {result['max_error']:.2e}")
        
        if result['naive_time'] > 5.0:
            print("  ⚠️  Naive version too slow, stopping here")
            break
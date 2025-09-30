import numpy as np
import time
from typing import Tuple
import matplotlib.pyplot as plt
import os

def naive_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Naive matrix multiplication C = A @ B
    A: (M, K), B: (K, N) -> C: (M, N)
    Optimizations:
    - local accumulator s to reduce indexing overhead
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"
    
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for j in range(N):
            s = 0 #local accumulator to reduce repeated writes to C and indexing overhead
            for k in range(K):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

def naive_gemm_cache_optimized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Cache optimized matrix multiplication C = A @ B
    Optimizations:
    - Uses ikj loop order for better spatial locality
    - Accesses B[k,:] and C[i,:] sequentially in innermost loop (better cache performance)
    - Loads A[i,k] once per iteration
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"
    
    C = np.zeros((M, N), dtype=A.dtype)
    for i in range(M):
        for k in range(K):
            a = A[i, k]
            for j in range(N):
                C[i, j] += a * B[k, j]
    return C


def estimate_flops(M: int, N: int, K: int) -> int:
    """Estimate FLOPs for M×K @ K×N matrix multiplication"""
    return 2*M*N*K

def estimate_memory_bytes(M: int, N: int, K: int, dtype=np.float32) -> int:
    """Estimate memory bytes accessed (naive upper bound)"""
    element_size = np.dtype(dtype).itemsize
    # A: M×K, B: K×N, C: M×N (write once, potentially read multiple times)
    # Worst case: we read A and B completely for each output element
    reads = M * N * (K + K)  # Each C[i,j] reads K elements from A and B
    writes = M * N           # Write each C element once
    return (reads + writes) * element_size

def analyze_performance(results: dict) -> None:
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

def plot_performance(results_list: list, output_path: str = "reports/plot.png") -> None:
    """Plot size vs GFLOP/s for all implementations"""
    # Extract data for plotting
    sizes = [r['M'] for r in results_list]  # Assuming square matrices
    naive_gflops = [r['naive_gflops'] for r in results_list]
    cache_opt_gflops = [r['cache_opt_gflops'] for r in results_list]
    numpy_gflops = [r['numpy_gflops'] for r in results_list]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, naive_gflops, 'o-', label='Naive (ijk)', linewidth=2, markersize=8)
    plt.plot(sizes, cache_opt_gflops, 's-', label='Cache-optimized (ikj)', linewidth=2, markersize=8)
    plt.plot(sizes, numpy_gflops, '^-', label='NumPy (optimized BLAS)', linewidth=2, markersize=8)

    plt.xlabel('Matrix Size (M=N=K)', fontsize=12)
    plt.ylabel('Performance (GFLOP/s)', fontsize=12)
    plt.title('GEMM Performance Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to show all implementations clearly

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Performance plot saved to: {output_path}")
    plt.close()

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

    # Benchmark cache-optimized implementation
    times_cache_opt = []
    for _ in range(num_runs):
        start = time.perf_counter()
        C_cache_opt = naive_gemm_cache_optimized(A, B)
        times_cache_opt.append(time.perf_counter() - start)

    results['cache_opt_time'] = min(times_cache_opt)
    results['cache_opt_gflops'] = flops / (results['cache_opt_time'] * 1e9)
    results['cache_opt_speedup_vs_naive'] = results['naive_time'] / results['cache_opt_time']

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
    results['max_error_naive'] = np.max(np.abs(C_naive - C_numpy))
    results['max_error_cache_opt'] = np.max(np.abs(C_cache_opt - C_numpy))
    
    return results

if __name__ == "__main__":
    print("=== BLAS/GEMM Intuition Benchmark ===")

    # Test sizes - start small!
    test_sizes = [
        (64, 64, 64),    # Small
        (128, 128, 128), # Medium
        (256, 256, 256), # Large (will be slow!)
    ]

    all_results = []  # Collect results for plotting

    for M, N, K in test_sizes:
        print(f"\nTesting {M}×{K} @ {K}×{N}:")
        result = benchmark_gemm(M, N, K)
        all_results.append(result)

        print(f"  FLOPs: {result['flops']:,}")
        print(f"  Memory bytes: {result['memory_bytes']:,}")
        print(f"  Naive (ijk) time: {result['naive_time']:.4f}s ({result['naive_gflops']:.2f} GFLOP/s)")
        print(f"  Cache-opt (ikj) time: {result['cache_opt_time']:.4f}s ({result['cache_opt_gflops']:.2f} GFLOP/s)")
        print(f"  NumPy time: {result['numpy_time']:.4f}s ({result['numpy_gflops']:.2f} GFLOP/s)")
        print(f"  Cache speedup vs naive: {result['cache_opt_speedup_vs_naive']:.1f}x")
        print(f"  NumPy speedup vs naive: {result['speedup']:.1f}x")
        print(f"  Max error (naive): {result['max_error_naive']:.2e}")
        print(f"  Max error (cache-opt): {result['max_error_cache_opt']:.2e}")

        # Analyze performance characteristics
        analyze_performance(result)

        if result['naive_time'] > 5.0:
            print("  ⚠️  Naive version too slow, stopping here")
            break

    # Generate performance plot
    if all_results:
        plot_performance(all_results, output_path="reports/plot.png")
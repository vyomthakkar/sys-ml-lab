# test_correctness.py - Validation suite for naive GEMM
import numpy as np
from naive_gemm import naive_gemm, estimate_flops, estimate_memory_bytes

def test_naive_gemm_correctness():
    """Test naive GEMM against NumPy for correctness"""
    test_cases = [
        (2, 3, 4),    # Tiny
        (5, 5, 5),    # Small square
        (10, 8, 6),   # Rectangular
        (1, 100, 1),  # Edge cases
        (100, 1, 1),
    ]
    
    for M, N, K in test_cases:
        print(f"Testing {M}×{K} @ {K}×{N}...")
        
        # Generate test data
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)
        
        # Compare implementations
        C_naive = naive_gemm(A, B)
        C_numpy = A @ B
        
        # Check shapes
        assert C_naive.shape == (M, N), f"Wrong shape: {C_naive.shape} != {(M, N)}"
        assert C_numpy.shape == (M, N), f"NumPy shape mismatch: {C_numpy.shape}"
        
        # Check numerical accuracy
        max_error = np.max(np.abs(C_naive - C_numpy))
        print(f"  Max error: {max_error:.2e}")
        assert max_error < 1e-5, f"Numerical error too large: {max_error}"
        
        print("  ✓ PASSED")

def test_flop_counting():
    """Verify FLOP counting logic"""
    test_cases = [
        ((2, 2, 2), 16),      # 2*2*2*2 = 16
        ((1, 1, 1), 2),       # 2*1*1*1 = 2  
        ((3, 4, 5), 120),     # 2*3*4*5 = 120
    ]
    
    for (M, N, K), expected_flops in test_cases:
        flops = estimate_flops(M, N, K)
        print(f"FLOP count {M}×{N}×{K}: {flops} (expected {expected_flops})")
        assert flops == expected_flops, f"FLOP count wrong: {flops} != {expected_flops}"
        print("  ✓ PASSED")

def test_memory_estimation():
    """Sanity check memory estimation"""
    M, N, K = 10, 10, 10
    bytes_f32 = estimate_memory_bytes(M, N, K, np.float32)
    bytes_f64 = estimate_memory_bytes(M, N, K, np.float64)
    
    print(f"Memory bytes (float32): {bytes_f32}")
    print(f"Memory bytes (float64): {bytes_f64}")
    
    # float64 should be exactly 2x float32
    assert bytes_f64 == 2 * bytes_f32, "Float64 should be 2x float32"
    print("  ✓ PASSED")

if __name__ == "__main__":
    print("=== Running Correctness Tests ===")
    test_flop_counting()
    test_memory_estimation() 
    test_naive_gemm_correctness()
    print("\n🎉 All tests passed! Your implementation is correct.")




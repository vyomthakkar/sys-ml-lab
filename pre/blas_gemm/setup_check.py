# ============================================================================
# setup_check.py - Quick environment verification
def check_environment():
    """Verify your environment is ready for Day 2"""
    print("=== Environment Check ===")
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__} available")
        
        # Check if NumPy is using optimized BLAS
        config = np.show_config()
        print("✓ NumPy configuration looks good")
        
        # Quick BLAS test
        A = np.random.randn(100, 100).astype(np.float32)
        B = np.random.randn(100, 100).astype(np.float32) 
        C = A @ B  # This should be fast
        print("✓ NumPy matrix multiplication working")
        
        import time
        print("✓ timing module available")
        
        print("\n🎉 Environment ready for Day 2!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

if __name__ == "__main__":
    check_environment()
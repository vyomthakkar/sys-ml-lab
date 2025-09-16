"""
STRIDES: The Complete Mental Model
===================================

Think of computer memory as a giant 1D array - just a long line of bytes.
Tensors are just clever ways to interpret chunks of this 1D memory as 
multi-dimensional arrays. Strides are the "interpretation rules."
"""

import numpy as np
import torch

def section_1_memory_is_1d():
    """
    FUNDAMENTAL TRUTH: Memory is always 1D
    
    When you create a 2D array, it's stored as a 1D sequence in memory.
    The question is: in what order?
    """
    print("="*60)
    print("SECTION 1: MEMORY IS ALWAYS 1D")
    print("="*60)
    
    # Create a 2D array
    arr = np.array([[1, 2, 3],
                    [4, 5, 6]], dtype=np.int32)
    
    print(f"\nLogical view (2D):")
    print(arr)
    
    print(f"\nPhysical memory (1D):")
    # View the same data as 1D without copying
    flat_view = arr.ravel()
    print(f"Memory layout: {flat_view}")
    print(f"Memory addresses are sequential: {[hex(addr) for addr in range(arr.__array_interface__['data'][0], arr.__array_interface__['data'][0] + arr.nbytes, 4)][:6]}")
    
    print("\n💡 Key insight: The 2D array [[1,2,3], [4,5,6]] is stored as [1,2,3,4,5,6] in memory")
    print("   This is called ROW-MAJOR (C-style) ordering - rows are contiguous")

def section_2_what_are_strides():
    """
    STRIDES: The mapping from logical indices to memory locations
    
    Stride[i] = "How many bytes to jump in memory to move 1 step in dimension i"
    """
    print("\n" + "="*60)
    print("SECTION 2: WHAT ARE STRIDES?")
    print("="*60)
    
    arr = np.array([[1, 2, 3],
                    [4, 5, 6]], dtype=np.int32)  # int32 = 4 bytes each
    
    print(f"\nArray shape: {arr.shape}")
    print(f"Array strides (bytes): {arr.strides}")
    print(f"Item size: {arr.itemsize} bytes")
    
    print("\n📐 Stride interpretation:")
    print(f"  Stride[0] = {arr.strides[0]} bytes → To go from row to row, jump {arr.strides[0]} bytes")
    print(f"  Stride[1] = {arr.strides[1]} bytes → To go from col to col, jump {arr.strides[1]} bytes")
    
    print("\n🎯 Let's verify by calculating memory positions:")
    print("  Position of arr[0,0] (value=1): base + 0*12 + 0*4 = base + 0")
    print("  Position of arr[0,1] (value=2): base + 0*12 + 1*4 = base + 4")
    print("  Position of arr[1,0] (value=4): base + 1*12 + 0*4 = base + 12")
    print("  Position of arr[1,1] (value=5): base + 1*12 + 1*4 = base + 16")
    
    print("\n💡 Formula: memory_offset = sum(index[i] * stride[i] for all dimensions)")

def section_3_strides_in_action():
    """
    See how different operations change strides WITHOUT moving data
    """
    print("\n" + "="*60)
    print("SECTION 3: STRIDES IN ACTION")
    print("="*60)
    
    # Start with a 1D array to make it crystal clear
    original = np.arange(12, dtype=np.int32)
    print(f"\nOriginal 1D array: {original}")
    print(f"  Memory: [0,1,2,3,4,5,6,7,8,9,10,11]")
    print(f"  Strides: {original.strides} (move 4 bytes for next element)")
    
    # Reshape to 2D - same memory, different strides!
    reshaped_2d = original.reshape(3, 4)
    print(f"\nReshaped to (3, 4):")
    print(reshaped_2d)
    print(f"  Same memory? {reshaped_2d.base is original}")
    print(f"  Strides: {reshaped_2d.strides}")
    print(f"    → Row stride: {reshaped_2d.strides[0]} bytes (skip 4 elements × 4 bytes)")
    print(f"    → Col stride: {reshaped_2d.strides[1]} bytes (next element)")
    
    # Transpose - same memory, swapped strides!
    transposed = reshaped_2d.T
    print(f"\nTransposed to (4, 3):")
    print(transposed)
    print(f"  Same memory? {transposed.base is original}")
    print(f"  Strides: {transposed.strides}")
    print(f"    → Strides are swapped! Now (4, 16) instead of (16, 4)")
    print(f"    → This means: move 4 bytes for next row, 16 bytes for next col")
    print(f"    → Result: columns of original become rows (without copying!)")

def section_4_contiguous_vs_non_contiguous():
    """
    CONTIGUOUS: Strides decrease from left to right
    This means elements are laid out sequentially in memory order
    """
    print("\n" + "="*60)
    print("SECTION 4: CONTIGUOUS VS NON-CONTIGUOUS")
    print("="*60)
    
    # Contiguous example
    x = torch.arange(12).reshape(3, 4)
    print(f"\nContiguous tensor (3, 4):")
    print(x)
    print(f"  Strides: {x.stride()}")
    print(f"  Is contiguous? {x.is_contiguous()}")
    print(f"  ✓ Strides decrease left→right: 4 > 1")
    
    # Non-contiguous example
    x_transposed = x.t()
    print(f"\nTransposed tensor (4, 3):")
    print(x_transposed)
    print(f"  Strides: {x_transposed.stride()}")
    print(f"  Is contiguous? {x_transposed.is_contiguous()}")
    print(f"  ✗ Strides don't decrease: 1 < 4")
    
    print("\n🎯 Why does contiguous matter?")
    print("  CONTIGUOUS: Reading sequentially = cache-friendly")
    print("    Memory access pattern: [0,1,2,3,4,5,6,7,8,9,10,11]")
    print("  NON-CONTIGUOUS: Jumping around = cache misses")
    print("    Memory access pattern: [0,4,8,1,5,9,2,6,10,3,7,11]")

def section_5_stride_tricks():
    """
    Advanced: How to create views with custom strides
    """
    print("\n" + "="*60)
    print("SECTION 5: STRIDE TRICKS & ZERO-COPY VIEWS")
    print("="*60)
    
    # Create a matrix
    x = np.arange(16, dtype=np.int32).reshape(4, 4)
    print(f"\nOriginal (4, 4):")
    print(x)
    print(f"  Strides: {x.strides}")
    
    # Extract diagonal with stride tricks (no copy!)
    from numpy.lib.stride_tricks import as_strided
    diagonal_view = as_strided(x, shape=(4,), strides=(x.strides[0] + x.strides[1],))
    print(f"\nDiagonal view (custom strides):")
    print(f"  Values: {diagonal_view}")
    print(f"  Stride: {diagonal_view.strides[0]} bytes")
    print(f"  How? Start at [0,0], then jump row+col strides to get [1,1], etc.")
    
    # Sliding window view (powerful for convolutions!)
    window_size = 3
    windows = as_strided(x, 
                         shape=(2, 2, 3, 3),  # 2x2 windows of size 3x3
                         strides=(x.strides[0], x.strides[1], x.strides[0], x.strides[1]))
    print(f"\nSliding 3x3 windows:")
    print(f"  Shape: {windows.shape} (2×2 grid of 3×3 windows)")
    print(f"  First window:\n{windows[0, 0]}")
    print(f"  Window at [0,1]:\n{windows[0, 1]}")
    print(f"  💡 All windows share the same memory!")

def section_6_performance_implications():
    """
    Why strides matter for performance
    """
    print("\n" + "="*60)
    print("SECTION 6: WHY THIS MATTERS FOR PERFORMANCE")
    print("="*60)
    
    import time
    
    size = 1000
    x = np.random.randn(size, size)
    
    # Benchmark row-wise sum (contiguous access)
    start = time.perf_counter()
    for _ in range(100):
        row_sum = x.sum(axis=1)  # Sum across rows
    row_time = time.perf_counter() - start
    
    # Benchmark column-wise sum (strided access)
    start = time.perf_counter()
    for _ in range(100):
        col_sum = x.sum(axis=0)  # Sum across columns  
    col_time = time.perf_counter() - start
    
    print(f"\nMatrix size: {size}×{size}")
    print(f"Row-wise sum (contiguous): {row_time:.4f}s")
    print(f"Column-wise sum (strided): {col_time:.4f}s")
    print(f"Ratio: {col_time/row_time:.2f}x")
    
    print("\n📊 Cache behavior:")
    print("  Row-wise: Accesses [0,1,2,3,...] → Cache lines fully utilized")
    print("  Col-wise: Accesses [0,1000,2000,3000,...] → Cache line per element!")
    
    # Show the fix
    x_transposed = x.T
    x_transposed_copy = np.ascontiguousarray(x_transposed)
    
    start = time.perf_counter()
    for _ in range(100):
        col_sum_fast = x_transposed_copy.sum(axis=1)  # Now it's row-wise!
    col_fast_time = time.perf_counter() - start
    
    print(f"\nColumn-sum after making contiguous: {col_fast_time:.4f}s")
    print(f"Speedup from contiguous: {col_time/col_fast_time:.2f}x")

def section_7_rules_of_thumb():
    """
    Practical rules for predicting tensor operation behavior
    """
    print("\n" + "="*60)
    print("SECTION 7: PRACTICAL RULES OF THUMB")
    print("="*60)
    
    rules = """
    🎯 GOLDEN RULES FOR STRIDES:
    
    1. RESHAPE:
       ✓ Can be a view if tensor is contiguous
       ✓ Will be a copy if non-contiguous
       ✓ Rule: Check tensor.is_contiguous() first
    
    2. TRANSPOSE/PERMUTE:
       ✓ ALWAYS a view (never copies)
       ✓ ALWAYS makes tensor non-contiguous (except special cases)
       ✓ Swaps/reorders strides
    
    3. VIEW:
       ✓ REQUIRES contiguous tensor
       ✓ Fails with error if non-contiguous
       ✓ Use .contiguous() first if needed (but that copies!)
    
    4. SLICING:
       ✓ Basic slicing [a:b] usually preserves contiguity
       ✓ Strided slicing [::2] creates non-contiguous views
       ✓ Fancy indexing with arrays creates copies
    
    5. PERFORMANCE RULES:
       ✓ Contiguous = sequential memory = fast
       ✓ Non-contiguous = jumping in memory = slow
       ✓ Sometimes worth calling .contiguous() before heavy computation
    
    6. HOW TO CHECK:
       NumPy:  arr.flags['C_CONTIGUOUS']
       PyTorch: tensor.is_contiguous()
       Strides: Check if they decrease left-to-right
    
    7. MEMORY SHARING:
       NumPy:  arr.base is not None → it's a view
       PyTorch: tensor.data_ptr() → same pointer = same memory
    """
    print(rules)

def interactive_quiz():
    """
    Test your understanding!
    """
    print("\n" + "="*60)
    print("INTERACTIVE QUIZ - TEST YOUR UNDERSTANDING")
    print("="*60)
    
    print("\n❓ QUESTION 1:")
    print("You have a tensor with shape (10, 20, 30) and strides (600, 30, 1)")
    print("What's the memory offset for element [2, 3, 4]?")
    print("\nWork it out: offset = 2*600 + 3*30 + 4*1 = ?")
    print("Answer: 1200 + 90 + 4 = 1294 bytes from base")
    
    print("\n❓ QUESTION 2:")
    print("A tensor has shape (100, 200) and strides (200, 1)")
    print("After transposing, what will the strides be?")
    print("\nAnswer: (1, 200) - strides are swapped!")
    
    print("\n❓ QUESTION 3:")
    print("You have a non-contiguous tensor from slicing: x[::2, ::2]")
    print("Will x.reshape(new_shape) create a view or copy?")
    print("\nAnswer: COPY - reshape needs contiguous for view")
    
    print("\n❓ QUESTION 4:")
    print("Why is iterating over rows faster than columns in a row-major array?")
    print("\nAnswer: Rows are contiguous in memory → better cache locality")

if __name__ == "__main__":
    print("🧠 STRIDES: THE COMPLETE MENTAL MODEL\n")
    print("This tutorial will build your intuition from scratch.\n")
    
    # Build up the concepts progressively
    section_1_memory_is_1d()
    input("\n[Press Enter to continue...]")
    
    section_2_what_are_strides()
    input("\n[Press Enter to continue...]")
    
    section_3_strides_in_action()
    input("\n[Press Enter to continue...]")
    
    section_4_contiguous_vs_non_contiguous()
    input("\n[Press Enter to continue...]")
    
    section_5_stride_tricks()
    input("\n[Press Enter to continue...]")
    
    section_6_performance_implications()
    input("\n[Press Enter to continue...]")
    
    section_7_rules_of_thumb()
    
    print("\n" + "="*60)
    print("✅ You now understand strides! Ready for Day 1 exercises.")
    print("="*60)
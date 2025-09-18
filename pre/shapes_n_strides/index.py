"""
Day 1: Shapes & Strides - Understanding Tensor Memory Layout
=============================================================

Goal: Predict and understand how tensor operations affect memory layout.
Key Insight: Strides tell you how many bytes to jump to get to the next element
             in each dimension. This determines if an operation is O(1) or O(n).
"""

#https://chatgpt.com/g/g-p-68c8f9d108408191959b2e8544c0e2cc-sys-ml/c/68cc4486-4f6c-8333-92a2-36b4d95af369

"""
Strides describe the step in memory per dimension (bytes in NumPy, elements in PyTorch). 
They tell you whether a layout change is a free view (O(1) metadata) or needs a real data copy (O(n)).
"""

import numpy as np
import torch
import time

def print_tensor_info(name, tensor, is_torch=False):
    """Helper to display tensor shape, strides, and contiguity"""
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    
    if is_torch:
        # PyTorch strides are in elements, not bytes
        print(f"  Strides (elements): {tensor.stride()}")
        print(f"  Contiguous: {tensor.is_contiguous()}")
        print(f"  Data pointer: {tensor.data_ptr()}")
    else:
        # NumPy strides are in bytes
        print(f"  Strides (bytes): {tensor.strides}")
        print(f"  Itemsize: {tensor.itemsize} bytes")
        print(f"  Strides (elements): {tuple(s // tensor.itemsize for s in tensor.strides)}")
        print(f"  C-contiguous: {tensor.flags['C_CONTIGUOUS']}")
        print(f"  Data pointer: {tensor.__array_interface__['data'][0]}")

def exercise_1_reshape():
    """
    Exercise 1: RESHAPE
    Reshape can sometimes be a view (O(1)) or require a copy (O(n))
    
    YOUR TASK: Before running, predict:
    1. Will each reshape be a view or copy?
    2. What will the strides be?
    """
    
    print("="*60)
    print("EXERCISE 1: RESHAPE OPERATIONS")
    print("="*60)
    
    # Create a base tensor
    x = np.arange(24, dtype=np.float32).reshape(4, 6)
    print_tensor_info("Original (4, 6)", x)
    
    # PREDICTION TIME: Will these be views or copies?
    # Write your predictions here:
    # reshape_a: [YOUR PREDICTION]
    # reshape_b: [YOUR PREDICTION]
    # reshape_c: [YOUR PREDICTION]
    
    # Test Case A: Compatible reshape (should be a view)
    reshape_a = x.reshape(2, 12)
    print_tensor_info("Reshape A: (4,6) -> (2,12)", reshape_a)
    print(f"  Same data? {reshape_a.base is x}")
    
    # Test Case B: Another compatible reshape
    reshape_b = x.reshape(8, 3)
    print_tensor_info("Reshape B: (4,6) -> (8,3)", reshape_b)
    print(f"  Same data? {reshape_b.base is x}")
    
    # Test Case C: Transpose then reshape (often requires copy)
    x_transposed = x.T
    print_tensor_info("Transposed (6,4)", x_transposed)
    reshape_c = x_transposed.reshape(8, 3)
    print_tensor_info("Reshape C: transposed -> (8,3)", reshape_c)
    print(f"  Same data as transposed? {reshape_c.base is x_transposed}")
    
    return x, reshape_a, reshape_b, reshape_c

def exercise_2_permute():
    """
    Exercise 2: PERMUTE/TRANSPOSE
    Permutations never copy data, only change stride order
    
    YOUR TASK: Predict the stride pattern after each permutation
    """
    
    print("\n" + "="*60)
    print("EXERCISE 2: PERMUTE/TRANSPOSE OPERATIONS")
    print("="*60)
    
    # 3D tensor for more interesting permutations
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    print_tensor_info("Original (2, 3, 4)", x, is_torch=True)
    
    # PREDICTION TIME: What will the strides be?
    # permute_a strides: [YOUR PREDICTION]
    # permute_b strides: [YOUR PREDICTION]
    # permute_c strides: [YOUR PREDICTION]
    
    # Test Case A: Swap first two dimensions
    permute_a = x.permute(1, 0, 2)  # (3, 2, 4)
    print_tensor_info("Permute A: (0,1,2) -> (1,0,2)", permute_a, is_torch=True)
    
    # Test Case B: Reverse all dimensions
    permute_b = x.permute(2, 1, 0)  # (4, 3, 2)
    print_tensor_info("Permute B: (0,1,2) -> (2,1,0)", permute_b, is_torch=True)
    
    # Test Case C: Complex permutation
    permute_c = x.permute(1, 2, 0)  # (3, 4, 2)
    print_tensor_info("Permute C: (0,1,2) -> (1,2,0)", permute_c, is_torch=True)
    
    # Verify no data was copied
    print(f"\nAll permutations share data: {x.data_ptr() == permute_a.data_ptr() == permute_b.data_ptr()}")
    
    return x, permute_a, permute_b, permute_c

def exercise_3_view():
    """
    Exercise 3: VIEW
    Views require the tensor to be contiguous. Non-contiguous tensors must be copied.
    
    YOUR TASK: Predict which views will work and which will fail
    """
    
    print("\n" + "="*60)
    print("EXERCISE 3: VIEW OPERATIONS")
    print("="*60)
    
    # Start with a contiguous tensor
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    print_tensor_info("Original contiguous (4, 6)", x, is_torch=True)
    
    # PREDICTION TIME: Will these work?
    # view_a: [YOUR PREDICTION - will it work?]
    # view_b: [YOUR PREDICTION - will it work?]
    # view_c: [YOUR PREDICTION - will it work?]
    
    # Test Case A: Simple view on contiguous tensor (should work)
    try:
        view_a = x.view(2, 12)
        print_tensor_info("View A: (4,6) -> (2,12)", view_a, is_torch=True)
        print("  ✓ View A succeeded")
    except:
        print("  ✗ View A failed - tensor not contiguous")
    
    # Test Case B: Make non-contiguous via transpose
    x_transposed = x.t()
    print_tensor_info("Transposed (6, 4) - non-contiguous!", x_transposed, is_torch=True)
    
    try:
        view_b = x_transposed.view(2, 12)
        print_tensor_info("View B: transposed -> (2,12)", view_b, is_torch=True)
        print("  ✓ View B succeeded")
    except:
        print("  ✗ View B failed - tensor not contiguous")
        # Fix with contiguous()
        x_contig = x_transposed.contiguous()
        view_b = x_contig.view(2, 12)
        print_tensor_info("View B (after contiguous): (2,12)", view_b, is_torch=True)
        print(f"  Required copy: {x_contig.data_ptr() != x_transposed.data_ptr()}")
    
    # Test Case C: Slice then view
    x_slice = x[:, ::2]  # Every other column
    print_tensor_info("Sliced [:, ::2] (4, 3)", x_slice, is_torch=True)
    
    try:
        view_c = x_slice.view(12)
        print_tensor_info("View C: slice -> (12,)", view_c, is_torch=True)
        print("  ✓ View C succeeded")
    except:
        print("  ✗ View C failed - slice not contiguous")
    
    return x, x_transposed, x_slice

def performance_implications():
    """
    Demonstrate real performance impact of contiguous vs non-contiguous operations
    """
    
    print("\n" + "="*60)
    print("PERFORMANCE IMPLICATIONS")
    print("="*60)
    
    size = 1000
    iterations = 1000
    
    # Create a large tensor
    x = torch.randn(size, size)
    
    # Contiguous case
    start = time.perf_counter()
    for _ in range(iterations):
        y = x.view(-1)
        _ = y.sum()
    contiguous_time = time.perf_counter() - start
    
    # Non-contiguous case
    x_transposed = x.t()
    start = time.perf_counter()
    for _ in range(iterations):
        y = x_transposed.contiguous().view(-1)
        _ = y.sum()
    non_contiguous_time = time.perf_counter() - start
    
    print(f"\nTensor size: {size}x{size}")
    print(f"Iterations: {iterations}")
    print(f"Contiguous view + sum: {contiguous_time:.4f}s")
    print(f"Non-contiguous (transpose) -> contiguous -> view + sum: {non_contiguous_time:.4f}s")
    print(f"Slowdown factor: {non_contiguous_time/contiguous_time:.2f}x")

def key_insights():
    """
    Document the key learnings from today's exercises
    """
    
    print("\n" + "="*60)
    print("KEY INSIGHTS TO REMEMBER")
    print("="*60)
    
    insights = """
    1. STRIDES determine memory layout:
       - Stride[i] = number of elements to skip for next element in dimension i
       - Contiguous: strides decrease from left to right (C-order)
       
    2. VIEW vs RESHAPE:
       - view: requires contiguous, always O(1), fails if not possible
       - reshape: tries view first, falls back to copy, always succeeds
       
    3. PERMUTE/TRANSPOSE:
       - Never copies data, only reorders strides
       - Result is usually non-contiguous
       - Following ops may need .contiguous() call
       
    4. PERFORMANCE RULES:
       - Contiguous tensors → fast sequential memory access
       - Non-contiguous → cache misses, slow iteration
       - .contiguous() copies data but may be worth it for repeated ops
       
    5. HOW TO PREDICT:
       - Check if strides are decreasing → contiguous
       - After transpose/permute → usually non-contiguous  
       - Slicing with step > 1 → non-contiguous
       - If product of shape = product of original → view possible (if contiguous)
    """
    
    print(insights)
    
    return insights

if __name__ == "__main__":
    # Run all exercises
    print("🚀 DAY 1: SHAPES & STRIDES DEEP DIVE\n")
    
    # Exercise 1: Reshape
    exercise_1_reshape()
    
    # Exercise 2: Permute
    exercise_2_permute()
    
    # Exercise 3: View
    exercise_3_view()
    
    # Performance demo
    performance_implications()
    
    # Summary
    key_insights()
    
    print("\n" + "="*60)
    print("✅ Day 1 Complete! Now write up your learnings in notes/README.md")
    print("="*60)
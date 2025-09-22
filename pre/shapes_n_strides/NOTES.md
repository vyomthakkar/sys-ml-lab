# Quick Reference Card

| Operation       | Creates View?        | Preserves Contiguity? | Strides Change?   |
|-----------------|----------------------|-----------------------|-------------------|
| `reshape`       | If contiguous        | Yes if view           | Yes               |
| `view`          | Always (or errors)   | Yes                   | Yes               |
| `transpose` / `T` | Always             | No (usually)          | Swapped           |
| `permute`       | Always               | No (usually)          | Reordered         |
| `contiguous()`  | Never (copies)       | Makes contiguous      | Reset to optimal  |
| `[::-2] slice`  | Always               | No                    | Multiplied        |
| `[mask] index`  | Never                | Yes                   | Reset             |


# Three Key Formulas to Remember

## Memory Offset Calculation
```
offset = sum(index[i] * stride[i] for i in dims)
```

## Contiguous Stride Formula (Row-Major)
```
stride[i] = product(shape[j] for j in range(i+1, n_dims)) * itemsize
```

## Is Contiguous Check
```
strides[i] >= strides[i+1] for all i (decreasing left to right)
```
- Decreasing strides are a necessary but not sufficient condition for a tensor to be contiguous

- A tensor is contiguous if its strides follow the pattern:
```python
stride[i] = stride[i+1] * size[i+1]
```

### Key Characteristics of Contiguous Tensors

- Last dimension stride = 1: Adjacent elements in the last dimension are next to each other in memory
- No memory gaps: Elements follow each other without skipping memory addresses
- Predictable stride pattern: Each dimension's stride is the product of all subsequent dimension sizes



# Strides: Numpy vs Torch

## Numpy
Strides are in bytes.
E.g: 
```python
x=array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
x.flags['C_CONTIGUOUS']=True
x.strides=(32, 8)
```

## Torch
Strides are in elements.
E.g: 
```python
x=tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
x.is_contiguous()=True
x.stride()=(4, 1)
```

https://claude.ai/chat/7577337f-a873-48e0-9061-2b9378c5ef3f

## When Reshape is a View
### 1. Contiguous tensors with compatible dimensions

- The tensor must be contiguous in memory
- The reshape must preserve the linear order of elements
```python
x = torch.arange(12).reshape(3, 4)  # contiguous
y = x.reshape(2, 6)  # view - just changes shape/strides
y = x.reshape(12)    # view - flattening contiguous is free
y = x.reshape(4, 3)  # view - still preserves element order
```

### 2. Dimension splitting/merging that preserves order
```python
   x = torch.randn(6, 8)
   y = x.reshape(2, 3, 8)   # view - splits first dim
   y = x.reshape(6, 2, 4)   # view - splits second dim
   y = x.reshape(48)        # view - complete flatten
```

## When Reshape Requires a Copy
### 1. Non-contiguous tensors (unless the reshape makes it contiguous again)
```python
x = torch.arange(12).reshape(3, 4)
y = x.T                  # transpose makes it non-contiguous
z = y.reshape(12)        # COPY - can't just change strides
```

### 2. After operations that create non-contiguous views
```python
x = torch.randn(3, 4).t()  # transposed, non-contiguous
y = x.reshape(2, 6)         # COPY - memory order doesn't match
```

### 3. Incompatible memory layouts
```python
x = torch.randn(3, 4).t()  # transposed, non-contiguous
y = x.reshape(2, 6)         # COPY - memory order doesn't match
```

## Quick Check Method
You can predict by examining strides:

- A tensor can be reshaped as a view if the new shape can be achieved with strides that maintain the same element ordering
- If tensor.is_contiguous() is True, most reshapes will be views
- Check with: y.data_ptr() == x.data_ptr() (same memory address = view)

```python
# Verification example
x = torch.randn(4, 6)
y = x.reshape(2, 12)
print(y.data_ptr() == x.data_ptr())  # True = view

x_t = x.t()
z = x_t.reshape(24)  
print(z.data_ptr() == x_t.data_ptr())  # False = copy
```

## Performance Implication
- View: O(1) operation, just metadata change
- Copy: O(n) operation, must allocate new memory and copy all elements
This is why understanding when reshapes are free vs expensive is critical for performance optimization, especially in hot paths or with large tensors.


star: https://claude.ai/chat/4a29a23a-2a01-4cf6-84f3-8dac373a496f

## Torch .permute()

In PyTorch, when you use permute(), both the shape and stride get rearranged according to the same permutation pattern.
Here's what happens:

- Shape: The dimensions are reordered according to your permutation
- Stride: The strides are reordered using the exact same permutation

This is because permute() creates a view of the original tensor rather than copying data. It achieves the dimension reordering by modifying the metadata (shape and stride) while keeping the underlying data in the same memory layout.

```python
import torch

# Create a tensor
x = torch.randn(2, 3, 4)
print(f"Original shape: {x.shape}")        # (2, 3, 4)
print(f"Original stride: {x.stride()}")    # (12, 4, 1)

# Permute dimensions: (0, 1, 2) -> (2, 0, 1)
y = x.permute(2, 0, 1)
print(f"\nPermuted shape: {y.shape}")      # (4, 2, 3)
print(f"Permuted stride: {y.stride()}")    # (1, 12, 4)
```

The stride values tell you how many elements to skip in the underlying storage to move one step along each dimension. After permutation, these skip distances are rearranged to match the new dimension order, which is how PyTorch can present a transposed view without actually moving data in memory.
This is why permute() is memory-efficient but can sometimes lead to non-contiguous tensors, which might require calling .contiguous() before certain operations.

** Key Insight **: When we permute, the underlying data remains the same, but the resultant tensor is now non-contiguous.

## .view() edge case: https://claude.ai/chat/02aa6294-112a-4f8a-af84-9e7b123a7800

### The General Rule
```view()``` works when there exists a valid stride pattern for the target shape that references the same memory locations. It's not about strict contiguity - it's about compatible memory layouts.

The key insight is that ```view()``` doesn't actually require strict contiguity - it requires that the tensor's memory layout can be reinterpreted with the target shape using some valid stride pattern.

### The Edge Case (Example)
```python
def exercise_3_view():

    # ....

    # Test Case C: Slice then view
    x_slice = x[:, ::2]  # Every other column
    print_tensor_info("Sliced [:, ::2] (4, 3)", x_slice, is_torch=True)
    
    try:
        view_c = x_slice.view(12)
        print_tensor_info("View C: slice -> (12,)", view_c, is_torch=True)
        console.print("  ✓ View C succeeded")
    except:
        console.print("  ✗ View C failed - slice not contiguous")
```

#### The Memory Layout Analysis
Original tensor (4, 6):
```Memory: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
Strides: (6, 1)
```
After slicing ```[:, ::2]``` → (4, 3):
```Selected elements: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
Strides: (6, 2)  # Jump 6 for next row, jump 2 for next column
```
The view as (12,):
```Same elements: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]  
Strides: (2,)    # Jump 2 for each consecutive element
```
Why It Works
The sliced tensor with strides ```(6, 2)``` and the viewed tensor with stride ```(2,)``` access exactly the same memory locations in the same order. PyTorch can reinterpret the 2D slice as a 1D view because:

- The elements are evenly spaced (stride 2)
- No memory copying is needed
- The same data pointer is maintained


























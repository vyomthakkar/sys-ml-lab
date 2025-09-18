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


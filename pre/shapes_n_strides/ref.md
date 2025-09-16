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
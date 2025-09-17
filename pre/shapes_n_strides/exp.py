# In IPython/Jupyter:
import torch
import numpy as np

# Experiment 1: When does reshape fail to view?
x = torch.arange(12).reshape(3, 4)
print(f"{x=}")
print(f"{x.is_contiguous()=}")
print(f"{x.stride()=}")
y = x[::2, :]  # Skip every other row
print(f"{y=}")
print(f"{y.is_contiguous()=}")  # False
print(f"{y.stride()=}")
z1 = y.reshape(8)  # This will copy!
print(f"{z1=}")
# z2 = y.view(8)     # This will error!

# Experiment 2: Stride arithmetic
# x = torch.zeros(4, 6, 8)
# print(f"Strides: {x.stride()}")
# If stride[0]=48, stride[1]=8, stride[2]=1
# Moving 1 position in dim 0 skips 48 elements (6*8)
# Moving 1 position in dim 1 skips 8 elements
# Moving 1 position in dim 2 skips 1 element
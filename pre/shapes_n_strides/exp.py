# In IPython/Jupyter:
import torch
import numpy as np
from rich.console import Console
console = Console()


console.rule("TORCH")

# Experiment 1: When does reshape fail to view?
x = torch.arange(12).reshape(3, 4)
console.print(f"{x=}")
console.print(f"{x.is_contiguous()=}")
console.print(f"{x.stride()=}")
y = x[::2, :]  # Skip every other row
console.print(f"{y=}")
console.print(f"{y.is_contiguous()=}")  # False
console.print(f"{y.stride()=}")
z1 = y.reshape(8)  # This will copy!
console.print(f"{z1=}")
try:
    z2 = y.view(8)     # This will error!
except RuntimeError as e:
    console.print(f"Error: {e}")

# Experiment 2: Stride arithmetic
x = torch.zeros(4, 6, 8)
console.print(f"Strides: {x.stride()}")
# If stride[0]=48, stride[1]=8, stride[2]=1
# Moving 1 position in dim 0 skips 48 elements (6*8)
# Moving 1 position in dim 1 skips 8 elements
# Moving 1 position in dim 2 skips 1 element

# Experiment 3: Changes to views affect original (same underlying memory/storage)
x = torch.arange(12).reshape(3, 4)
y = x[::2, :]  # Skip every other row
y[0, 0] = 100
console.print(f"{x=}")
console.print(f"{y=}")
console.print(f"{x.data_ptr()=}")
console.print(f"{y.data_ptr()=}")




##################################### NUMPY ####################################################

console.rule("NUMPY")



# Experiment 1: When does reshape fail to view?
x = np.arange(12).reshape(3, 4)
console.print(f"{x=}")
console.print(f"{x.flags['C_CONTIGUOUS']=}")
console.print(f"{x.strides=}")
y = x[::2, :]  # Skip every other row
console.print(f"{y=}")
console.print(f"{y.flags['C_CONTIGUOUS']=}")  # False
console.print(f"{y.strides=}")
z1 = y.reshape(8)  # This will copy!
console.print(f"{z1=}")
try:
    z2 = y.view(8)     # This will error!
except TypeError as e:
    console.print(f"Error: {e}")

# Experiment 2: Stride arithmetic
x = np.zeros((4, 6, 8))
console.print(f"Strides: {x.strides}")
# If stride[0]=48, stride[1]=8, stride[2]=1
# Moving 1 position in dim 0 skips 48 elements (6*8)
# Moving 1 position in dim 1 skips 8 elements
# Moving 1 position in dim 2 skips 1 element

# Experiment 3: Changes to views affect original (same underlying memory/storage)
x = np.arange(12).reshape(3, 4)
y = x[::2, :]  # Skip every other row
y[0, 0] = 100
console.print(f"{x=}")
console.print(f"{y=}")
console.print(f"{x.__array_interface__['data'][0]=}")
console.print(f"{y.__array_interface__['data'][0]=}")



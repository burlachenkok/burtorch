import datetime
print(datetime.datetime.now())

import time
import torch

print("torch version: ", torch.__version__)

# Define the computation as a TorchScript function
@torch.jit.script
def compute(a: torch.Tensor, b: torch.Tensor):
  c = a + b
  d = a * b + b**3
  c += c + 1
  c += 1 + c + (-a)
  d += d * 2 + (b + a).relu()
  d += 3 * d + (b - a).relu()
  e = c - d
  f = e**2
  g = f / 2.0
  g += 10.0 / f
  return g

# Move to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("Using device:", device)

# Check if `compute` is a TorchScript function
if isinstance(compute, torch.jit.ScriptFunction):
    print("compute is a TorchScript function.")
else:
    print("compute is not a TorchScript function.")

# Timing the loop
s = time.time()

for i in range(200000):
    # Initialize tensors on GPU
    a = torch.tensor(-4.0, requires_grad=True, device=device, dtype=torch.float64)
    b = torch.tensor(2.0, requires_grad=True, device=device, dtype=torch.float64)

    # Perform computation
    g = compute(a, b)

    # Perform backpropagation
    g.backward()

e = time.time()

# Print results
print("Time: ", e - s)
print(f'{g.data:.4f}')
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')

#input()

print(datetime.datetime.now())

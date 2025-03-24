import time
import torch

print("torch version: ", torch.__version__)

# Define the computation as a TorchScript function
@torch.jit.script
def compute(a: torch.Tensor, b: torch.Tensor):
    c = a + b
    d = a * b + b**3
    e = c - d
    f = e**2
    g = f / 2.0
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

for i in range(100 * 1000):
    # Initialize tensors on GPU
    a = torch.tensor(-41.0, requires_grad=True, device=device, dtype=torch.float64)
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

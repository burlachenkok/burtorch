import time
import torch

print("torch version: ", torch.__version__)

# Define the function to be scripted
@torch.jit.script
def perform_operations(a: torch.Tensor, b: torch.Tensor):
    # Perform the calculations
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
    # Return the tensors to use them later in the backward pass
    return a, b, c, d, e, f, g

# Initialize tensors with requires_grad=True to track operations
a = torch.tensor(-4.0, requires_grad=True, dtype=torch.float64)
b = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

# Perform operations using the scripted function
a, b, c, d, e, f, g = perform_operations(a, b)

# Perform backward pass
g.backward()

#==================================================================
# Loop to save the tensor states to disk
s = time.time()
for i in range(5000):

    with open('tensor_state_torch_jit.bin', 'wb') as file_:
        # Save tensors during each iteration, ensuring they are moved back to CPU before saving
        torch.save((
            a.detach().cpu(),
            b.detach().cpu(),
            c.detach().cpu(),
            d.detach().cpu(),
            e.detach().cpu(),
            f.detach().cpu(),
            g.detach().cpu()
        ), file_)
e = time.time()
#==================================================================

print("Time: ", e - s)

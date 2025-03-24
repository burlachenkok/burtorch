import time
import torch

print("torch version: ", torch.__version__)

# Initialize tensors with requires_grad=True to track operations
a = torch.tensor(-4.0, requires_grad=True, dtype=torch.float64)
b = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

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

# Perform backward pass
g.backward()

#==================================================================
# Loop to save the tensor states to disk
s = time.time()
for i in range(5000):

    file_ = open('tensor_state_torch_own.bin', 'wb')
    if True:
      # Save tensors during each iteration
      torch.save((
        a.detach(),
        b.detach(),
        c.detach(),
        d.detach(),
        e.detach(),
        f.detach(),
        g.detach()
      ), file_)
    #file_.flush()
    file_.close()
e = time.time()
#==================================================================

print("Time: ", e - s)

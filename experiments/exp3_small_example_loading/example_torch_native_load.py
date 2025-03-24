import time
import torch
import numpy as np

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

# Loop to save the tensor states to disk
s = time.time()
for i in range(5000):

    byte_data = bytearray()
    file_ = open(f'tensor_state_torch_native.bin', 'rb')

    if True:
      a.data = torch.from_numpy(np.frombuffer(file_.read(8), dtype=np.float64).copy())
      b.data = torch.from_numpy(np.frombuffer(file_.read(8), dtype=np.float64).copy())
      c.data = torch.from_numpy(np.frombuffer(file_.read(8), dtype=np.float64).copy())
      d.data = torch.from_numpy(np.frombuffer(file_.read(8), dtype=np.float64).copy())
      e.data = torch.from_numpy(np.frombuffer(file_.read(8), dtype=np.float64).copy())
      f.data = torch.from_numpy(np.frombuffer(file_.read(8), dtype=np.float64).copy())
      g.data = torch.from_numpy(np.frombuffer(file_.read(8), dtype=np.float64).copy())

    file_.close()

e = time.time()

print("Time: ", e - s)

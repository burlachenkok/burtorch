import time
import torch

print("torch version: ", torch.__version__)

# https://github.com/karpathy/micrograd

s = time.time()

for i in range(100*1000):
  a = torch.tensor(-41.0, requires_grad=True, dtype=torch.float64)
  b = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)

  c = a + b
  d = a * b + b**3
  e = c - d
  f = e**2
  g = f / 2.0
    
  # Perform backpropagation
  g.backward()
  #print("g.dtype", g.dtype)

e = time.time()

print("Time: ", e - s)

print(f'{g.data:.4f}')
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')

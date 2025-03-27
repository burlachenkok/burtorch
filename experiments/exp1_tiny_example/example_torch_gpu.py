import time
import torch


if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")
    for i in range(num_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices available.")

#==========================================================================

s = time.time()

for i in range(100*1000):
  a = torch.tensor(-41.0, device="cuda", requires_grad=True, dtype=torch.float64)
  b = torch.tensor(2.0, device="cuda", requires_grad=True, dtype=torch.float64)

  c = a + b
  d = a * b + b**3
  e = c - d
  f = e**2
  g = f / 2.0
    
  # Perform backpropagation
  g.backward()

e = time.time()

print("Time: ", e - s)

print(f'{g.data:.4f}')
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')

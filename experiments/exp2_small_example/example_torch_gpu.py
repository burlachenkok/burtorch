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

for i in range(20000):
  # Initialize tensors with requires_grad=True to track operations
  a = torch.tensor(-4.0, device="cuda", requires_grad=True, dtype=torch.float64)
  b = torch.tensor(2.0, device="cuda", requires_grad=True, dtype=torch.float64)

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

e = time.time()

print("Time: ", e - s)

print(f'{g.data:.4f}')
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')

#input()


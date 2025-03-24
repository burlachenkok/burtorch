import time, sys
sys.path.append("./..")

from micrograd.engine import Value
# https://github.com/karpathy/micrograd

s = time.time()

for i in range(100*1000):
  a = Value(-41.0)
  b = Value(2.0)
  c = a + b
  d = a * b + b**3
  e = c - d
  f = e**2
  g = f / 2.0
  g.backward()

e = time.time()

print("Time: ", e - s)

print(f'{g.data:.4f}')
print(f'{a.grad:.4f}')
print(f'{b.grad:.4f}')

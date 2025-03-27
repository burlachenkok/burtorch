import datetime
print(datetime.datetime.now())

import time, sys
sys.path.append("./..")


import time
from micrograd.engine import Value
# https://github.com/karpathy/micrograd

s = time.time()

for i in range(200000):
    a = Value(-4.0)
    b = Value(2.0)
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
    g.backward()

e = time.time()

print("Time: ", e - s)

print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db

#input()

print(datetime.datetime.now())

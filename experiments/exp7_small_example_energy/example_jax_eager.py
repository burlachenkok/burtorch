import datetime
print(datetime.datetime.now())

import time
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad

print("JAX version: ", jax.__version__)
jax.config.update("jax_enable_x64", True)

# Define the computation
def compute(a, b):
  c = a + b
  d = a * b + b**3
  c += c + 1
  c += 1 + c + (-a)
  d += d * 2 + jnp.maximum(0, b + a)
  d += 3 * d + jnp.maximum(0, b - a)
  e = c - d
  f = e**2
  g = f / 2.0
  g += 10.0 / f
  return g

# Compile the gradient computation for performance
compute_grad = value_and_grad(compute, argnums=(0, 1))

# Start timing
s = time.time()

for i in range(200000):
    g_value, grads = compute_grad(jnp.float64(-4.0), jnp.float64(2.0))

e = time.time()

# Print results
print("Time: ", e - s)
print(f'{g_value:.4f}')
print(f'{grads[0]:.4f}')
print(f'{grads[1]:.4f}')
print("g_value dtype:", g_value.dtype)

#input()

print(datetime.datetime.now())

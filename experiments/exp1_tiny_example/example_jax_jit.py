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
    e = c - d
    f = e**2
    g = f / 2.0
    return g

# Compile the gradient computation for performance
compute_grad = jax.jit(value_and_grad(compute, argnums=(0, 1)))

# Start timing
s = time.time()

for i in range(100 * 1000):
    g_value, grads = compute_grad(jnp.float64(-41.0), jnp.float64(2.0))

e = time.time()

# Print results
print("Time: ", e - s)
print(f'{g_value:.4f}')
print(f'{grads[0]:.4f}')
print(f'{grads[1]:.4f}')
print("g_value dtype:", g_value.dtype)



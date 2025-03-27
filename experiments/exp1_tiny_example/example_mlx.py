import time
import mlx.core as mx

# Define the computation
def compute(ab):
    a = ab[0]
    b = ab[1] 
    c = a + b
    d = a * b + b**3
    e = c - d
    f = e**2
    g = f / 2.0
    return g.mean()  # Return an array with one element

compute_f = mx.value_and_grad(compute)

# Start timing
s = time.time()

for i in range(100 * 1000):
    ab = mx.array([-41.0, 2.0], dtype=mx.float32)  # Create array 'a', mlx does not support fp64
    gvalue, grad = compute_f(ab)

# End timing
e = time.time()

# Print results
print("Time: ", e - s)

# grads will be a tuple of gradients w.r.t a and b
# Print out gradients
print(f"Value: {gvalue}")
print(f'Gradient w.r.t. a and b: {grad}')

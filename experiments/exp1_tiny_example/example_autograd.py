import time
import autograd.numpy as np
from autograd import grad

print("Autograd")

# Start the timer
s = time.time()

# Define the gradient function once outside the loop
def compute_gradients(a, b):
    c = a + b
    d = a * b + b**3
    e = c - d
    f = e**2
    g = f / 2.0
    return g

# Create the gradient function for a and b

compute_gradients_fun_a = grad(compute_gradients, 0)  # Gradient w.r.t. a
compute_gradients_fun_b = grad(compute_gradients, 1)  # Gradient w.r.t. b

for i in range(100*1000):
    a = np.array(-41.0, dtype=np.float64)
    b = np.array(2.0, dtype=np.float64)
    #print("a.dtype:", a.dtype)
    # Compute the value of g
    g_value = compute_gradients(a, b)
    #print("g_value.dtype:", g_value.dtype)
    
    # Compute the gradients w.r.t a and b
    grad_a = compute_gradients_fun_a(a, b)
    grad_b = compute_gradients_fun_b(a, b)

e = time.time()

# Print results
print("Time: ", e - s)
print(f"Function value g: {g_value:.4f}")
print(f"Gradient w.r.t a: {grad_a:.4f}")
print(f"Gradient w.r.t b: {grad_b:.4f}")

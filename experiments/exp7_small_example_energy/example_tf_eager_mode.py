import datetime
print(datetime.datetime.now())

import time
import tensorflow as tf

print("tf version:", tf.__version__)

# Function to define computation
def compute_gradients(a, b):
    #if tf.executing_eagerly():
    #  print("Eager execution is enabled.")
    #else:
    #  print("Graph execution is enabled.")

    # Compute gradients using GradientTape
    with tf.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        # Operations inside GradientTape will track gradients
        c = a + b
        d = a * b + tf.pow(b, 3)
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + tf.nn.relu(b + a)
        d += 3 * d + tf.nn.relu(b - a)
        e = c - d
        f = tf.pow(e, 2)
        g = f / 2.0
        g += 10.0 / f

    gradients = tape.gradient(g, [a, b])
    return g, gradients

# Start timing
start_time = time.time()

# Define variables
a = tf.Variable(-4.0, dtype=tf.float64)
b = tf.Variable(2.0, dtype=tf.float64)

# Iterate computation 100K times
for i in range(200000):
    g_value, gradients = compute_gradients(a, b)

# End timing
end_time = time.time()

# Print results
print("Time: ", end_time - start_time)
print(f'{g_value:.4f}')
print(f'{gradients[0]:.4f}')
print(f'{gradients[1]:.4f}')

print("GPU Available: ", tf.test.is_gpu_available())

#input()

print(datetime.datetime.now())

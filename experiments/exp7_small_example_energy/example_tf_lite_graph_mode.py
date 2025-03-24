import datetime
print(datetime.datetime.now())

import tensorflow as tf
import numpy as np

print("tf version:", tf.__version__)

# Function to define computation and precompute gradients
@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float64),tf.TensorSpec(shape=(), dtype=tf.float64)])
def compute_function_with_gradients(a, b):

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

    # Compute gradients before conversion
    gradients = tape.gradient(g, [a, b])
    return g, gradients

concrete_func = compute_function_with_gradients.get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,   # Use TensorFlow Lite built-in ops
    tf.lite.OpsSet.SELECT_TF_OPS      # Allow fallback to TensorFlow ops (fixes ReluGrad issue)
]

tflite_model = converter.convert()

# Save the TFLite model
with open("compute_function_with_gradients.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model with gradients conversion successful!")

print(datetime.datetime.now())

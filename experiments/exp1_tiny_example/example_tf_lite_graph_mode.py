import tensorflow as tf
import numpy as np

print("tf version:", tf.__version__)

# Function to define computation and precompute gradients
@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.float32),tf.TensorSpec(shape=(), dtype=tf.float32)])
def compute_function_with_gradients(a, b):

    # Compute gradients using GradientTape
    with tf.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        # Operations inside GradientTape will track gradients
        c = a + b
        d = a * b + tf.pow(b, 3)
        e = c - d
        f = tf.pow(e, 2)
        g = f / 2.0

    # Compute gradients before conversion
    gradients = tape.gradient(g, [a, b])
    return g, gradients

concrete_func = compute_function_with_gradients.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the TFLite model
with open("compute_function_with_gradients.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model with gradients conversion successful!")

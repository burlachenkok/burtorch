import time
import tensorflow as tf
import pickle

print("TensorFlow version: ", tf.__version__)

# Define the forward pass as a tf.function (graph mode)
@tf.function
def forward_pass(a, b):
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + tf.nn.relu(b + a)
    d += 3 * d + tf.nn.relu(b - a)
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    return a, b, c, d, e, f, g

# Initialize tensors with gradient tracking
a = tf.Variable(-4.0, dtype=tf.float64)
b = tf.Variable(2.0, dtype=tf.float64)

# Using graph mode (tf.function)
state = forward_pass(a, b)

# Perform the forward pass (graph execution)
s = time.time()
for i in range(5000):

    # Save tensors during each iteration using pickle
    file_ = open('tensor_state_tf_graph.bin', 'wb')
    if True:
        pickle.dump(state, file_)
    file_.close()

e = time.time()

print("Time: ", e - s)

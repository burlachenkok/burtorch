import time
import tensorflow as tf
import pickle

print("TensorFlow version: ", tf.__version__)

# Initialize tensors with gradient tracking
a = tf.Variable(-4.0, dtype=tf.float64)
b = tf.Variable(2.0, dtype=tf.float64)

# Define the forward pass with operations
with tf.GradientTape(persistent=True) as tape:
    tape.watch(a)
    tape.watch(b)
    
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

# Perform backward pass
gradients = tape.gradient(g, [a, b])

#==================================================================
# Loop to save tensor states to disk
s = time.time()
for i in range(5000):
    file_ = open('tensor_state_tf_eager.bin', 'wb')
    if True:
        # Save tensors during each iteration
        state = (
            a,
            b,
            c,
            d,
            e,
            f,
            g
        )
        pickle.dump(state, file_)
    file_.close()

e = time.time()
#==================================================================

print("Time: ", e - s)

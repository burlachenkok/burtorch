import time
import numpy as np
import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path="compute_function_with_gradients.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



# Start timing
start_time = time.time()

# Set input values
a_value = np.array(-41.0, dtype=np.float32)
b_value = np.array(2.0, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], a_value)
interpreter.set_tensor(input_details[1]['index'], b_value)

# Iterate computation 100K times
for i in range(100*1000):
  # Run
  interpreter.invoke()
  # Get outputs
  output_value = interpreter.get_tensor(output_details[0]['index'])
  gradient_a = interpreter.get_tensor(output_details[1]['index'])
  gradient_b = interpreter.get_tensor(output_details[2]['index'])

# End timing
end_time = time.time()

print("Time: ", end_time - start_time)
print(f"TFLite Function Output: {output_value}")
print(f"Gradient w.r.t. a: {gradient_a}")
print(f"Gradient w.r.t. b: {gradient_b}")

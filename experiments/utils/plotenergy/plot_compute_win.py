import matplotlib.pyplot as plt
import numpy as np

#plt.rcParams["lines.markersize"] = 17
#plt.rcParams["lines.linewidth"] = 5
font_size = 50
plt.rcParams["font.size"] = font_size

# Frameworks and corresponding energy consumed values
frameworks = [
    ' 1. BurTorch, Eager, C++ [CPU]',
    ' 2. TensorFlow, Eager, Python [CPU]',
    ' 3. TensorFlow, Graph, Semi-Python [CPU]',
    ' 4. TFLite, Graph, TFLite Interpr. [CPU]',
    ' 5. Autograd, Eager, Python [CPU]',
    ' 6. Torch, Eager, Python [GPU]',
    ' 7. Torch, Eager, Python [CPU]',
    ' 8. Torch, Graph, TorchScript [CPU]',
    ' 9. Torch, Eager, LibTorch [CPU]',
    '10. JAX, Eager, Python [CPU]',
    '11. JAX, Graph, Semi-Python [CPU]',
    '12. Micrograd, Eager, Python [CPU]',
    '13. In Theory [CPU]'
]

total_compute = [
    0.007, 55.217, 14.469, 0.589, 18.956, 51.380, 10.419, 9.994, 5.300, 291.764, 5.580, 1.590, 0.0004 
]

# Plotting
plt.figure(figsize=(35, 15))
plt.barh([x for x in reversed(frameworks)], [x for x in reversed(total_compute)], color='skyblue')
plt.xscale('log')  # Apply log scale to the x-axis

plt.xlabel('Compute Time (seconds)', fontsize=font_size)
plt.xticks(fontsize=font_size)

plt.tight_layout()
plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
plt.savefig('compute-tiny-win.pdf', format='pdf')

#plt.show()


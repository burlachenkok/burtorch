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
    ' 6. Torch, Eager, Python [CPU]',
    ' 7. Torch, Graph, TorchScript [CPU]',
    ' 8. Torch, Eager, LibTorch [CPU]',
    ' 9. JAX, Eager, Python [CPU]',
    '10. JAX, Graph, Semi-Python [CPU]',
    '11. Micrograd, Eager, Python [CPU]',
    '12. Apple MLX, Eager, Python [CPU]'
]

total_energy = [
    0.011, 163.538, 27.882, 0.408, 29.250, 35.240, 15.440, 14.224, 7.611, 506.464, 11.156, 2.751, 3.072
]

# Plotting
plt.figure(figsize=(35, 15))
plt.barh([x for x in reversed(frameworks)], [x for x in reversed(total_energy)], color='skyblue')
plt.xscale('log')  # Apply log scale to the x-axis

plt.xlabel('Compute Time (seconds)', fontsize=font_size)
plt.xticks(fontsize=font_size)

plt.tight_layout()
plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
plt.savefig('compute-tiny-linux.pdf', format='pdf')

#plt.show()


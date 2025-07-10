import nnfs
import numpy as np
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)
y_reshaped = y.reshape(-1, 1)
combined_data = np.concatenate((X, y_reshaped), axis=1)
output_filename = "../data/points.txt"
np.savetxt(output_filename, combined_data, delimiter=" ", fmt="%.6f %.6f %d")
print(f"Combined data saved to {output_filename}")

weights = 0.01 * np.random.randn(2, 3)
print(weights)

weights = 0.01 * np.random.randn(3, 3)
print(weights)
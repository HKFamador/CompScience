import numpy as np
import matplotlib.pyplot as plt

# Data points
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Calculate means
x_mean = sum(x) / len(x)
y_mean = sum(y) / len(y)

# Initialize slope variables
numerator = 0
denominator = 0

# Compute slope (m)
for i in range(len(x)):
    numerator += (x[i] - x_mean) * (y[i] - y_mean)
    denominator += (x[i] - x_mean) ** 2

m = numerator / denominator  # Slope
b = y_mean - (m * x_mean)  # Intercept

# Predicted y values
y_hat = [m * xi + b for xi in x]

# Print results
print("Slope (m):", m)
print("Intercept (b):", b)
print(f"Regression Equation: y = {m:.2f}x + {b:.2f}")

# Plot data and regression line
plt.scatter(x, y, color="blue", label="Data Points")
plt.plot(x, y_hat, color="red", label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

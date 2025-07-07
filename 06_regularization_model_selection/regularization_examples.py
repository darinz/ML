import numpy as np

# Example parameter vector theta
theta = np.array([1.0, -2.0, 0.0, 3.0])

# Example values for loss and regularization parameter
J_theta = 2.5  # Example original loss
lambda_ = 0.1  # Regularization parameter

# 1. Regularized loss: J_lambda = J(theta) + lambda * R(theta)
# (Assume R_theta is computed below)
# Placeholder for R_theta, will be overwritten by actual regularizer
R_theta = 0.0
J_lambda = J_theta + lambda_ * R_theta
print(f"Regularized loss (with placeholder R_theta): {J_lambda}")

# 2. L2 regularization: R(theta) = 0.5 * ||theta||_2^2
R_theta = 0.5 * np.sum(theta ** 2)
print(f"L2 regularization (0.5 * ||theta||_2^2): {R_theta}")

# 3. L0 "norm": counts nonzero elements in theta
num_nonzero = np.count_nonzero(theta)
print(f"L0 'norm' (number of nonzero elements): {num_nonzero}")

# 4. L1 regularization: R(theta) = ||theta||_1
R_theta = np.sum(np.abs(theta))
print(f"L1 regularization (||theta||_1): {R_theta}")

# 5. L2 regularization (again, for clarity): R(theta) = 0.5 * ||theta||_2^2
R_theta = 0.5 * np.sum(theta ** 2)
print(f"L2 regularization (again, 0.5 * ||theta||_2^2): {R_theta}") 
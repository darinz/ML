"""
Examples for vectorization concepts in neural networks.
- For-loop vs. vectorized implementation
- Broadcasting and tiling
- Worked example: two-layer neural network (vectorized)
"""

import numpy as np

# --- For-loop vs. Vectorized Implementation ---
print("\n--- For-loop vs. Vectorized Implementation ---")
np.random.seed(0)
X = np.random.rand(1000, 4)
W = np.random.randn(3, 4)
b = np.random.randn(3)

# For-loop (slow)
outputs_loop = np.zeros((1000, 3))
for i in range(1000):
    for j in range(3):
        outputs_loop[i, j] = np.dot(W[j], X[i]) + b[j]

# Vectorized (fast)
outputs_vec = X @ W.T + b  # shape: (1000, 3)

print('Difference:', np.abs(outputs_loop - outputs_vec).max())

# --- Broadcasting and Tiling Example ---
print("\n--- Broadcasting and Tiling Example ---")
Z = np.random.randn(5, 3)
b_vec = np.array([1, 2, 3])
print("Broadcasting result:\n", Z + b_vec)
print("Tiling result:\n", np.tile(b_vec, (5, 1)))

# --- Worked Example: Two-Layer Neural Network (Vectorized) ---
print("\n--- Worked Example: Two-Layer Neural Network (Vectorized) ---")
# Example: 200 data points, 4 input features, 3 hidden neurons, 1 output
np.random.seed(42)
X = np.random.rand(200, 4)
true_W1 = np.array([[1.2, -0.7, 0.5, 2.0],
                   [0.3, 1.5, -1.0, 0.7],
                   [2.0, 0.1, 0.3, -0.5]])  # (3, 4)
true_b1 = np.array([0.5, -0.2, 0.1])
H = np.maximum(X @ true_W1.T + true_b1, 0)  # Hidden layer (ReLU)
true_W2 = np.array([1.0, -2.0, 0.5])
true_b2 = 0.3
y = H @ true_W2 + true_b2 + np.random.normal(0, 0.2, size=H.shape[0])

# Initialize parameters
W1 = np.random.randn(3, 4)
b1 = np.zeros(3)
W2 = np.random.randn(3)
b2 = 0.0
lr = 0.05

# Training loop
for epoch in range(600):
    # Forward pass
    Z1 = X @ W1.T + b1
    A1 = np.maximum(Z1, 0)  # ReLU
    y_pred = A1 @ W2 + b2
    # Loss (MSE)
    loss = np.mean((y_pred - y) ** 2)
    # Backpropagation
    grad_y_pred = 2 * (y_pred - y) / len(y)
    grad_W2 = A1.T @ grad_y_pred
    grad_b2 = np.sum(grad_y_pred)
    grad_A1 = np.outer(grad_y_pred, W2)
    grad_Z1 = grad_A1 * (Z1 > 0)
    grad_W1 = grad_Z1.T @ X
    grad_b1 = np.sum(grad_Z1, axis=0)
    # Update
    W1 -= lr * grad_W1
    b1 -= lr * grad_b1
    W2 -= lr * grad_W2
    b2 -= lr * grad_b2
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Visualize predictions vs. true values
try:
    import matplotlib.pyplot as plt
    plt.scatter(y, y_pred, alpha=0.6)
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Two-Layer Fully-Connected Neural Network')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.show()
except ImportError:
    print("matplotlib not installed; skipping plot.") 
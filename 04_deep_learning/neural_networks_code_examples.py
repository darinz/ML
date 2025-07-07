# Practical Code Examples for 7.2 Neural Networks
# Each section below corresponds to a section in 02_neural_networks.md

# --- Single Neuron Regression with ReLU ---
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data: house sizes and prices
np.random.seed(0)
x = np.linspace(500, 3500, 50)  # house sizes
true_w, true_b = 0.3, 50
noise = np.random.normal(0, 30, size=x.shape)
y = np.maximum(true_w * x + true_b + noise, 0)  # true prices, clipped at 0

# Initialize parameters
w, b = 0.1, 0.0
learning_rate = 1e-7

# Training loop (simple gradient descent)
for epoch in range(1000):
    y_pred = np.maximum(w * x + b, 0)  # ReLU activation
    error = y_pred - y
    grad_w = np.mean(error * (x * (y_pred > 0)))
    grad_b = np.mean(error * (y_pred > 0))
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

plt.figure()
plt.scatter(x, y, label='Data')
plt.plot(x, np.maximum(w * x + b, 0), color='red', label='Single Neuron Fit')
plt.xlabel('House Size')
plt.ylabel('Price')
plt.legend()
plt.title('Single Neuron Regression with ReLU')
plt.show()

# --- Two-Layer Neural Network Example ---
np.random.seed(1)
X = np.random.rand(100, 4)  # 100 houses, 4 features each
true_w1 = np.array([2.0, 1.5, 0.5, 1.0])
true_b1 = 0.5
hidden = np.maximum(X @ true_w1 + true_b1, 0)  # First layer (ReLU)
true_w2 = 3.0
true_b2 = 2.0
y = true_w2 * hidden + true_b2 + np.random.normal(0, 0.5, size=hidden.shape)  # Output

w1 = np.random.randn(4)
b1 = 0.0
w2 = 1.0
b2 = 0.0
lr = 0.05

for epoch in range(500):
    z1 = X @ w1 + b1
    a1 = np.maximum(z1, 0)  # ReLU
    y_pred = w2 * a1 + b2
    loss = np.mean((y_pred - y) ** 2)
    grad_y_pred = 2 * (y_pred - y) / len(y)
    grad_w2 = np.sum(grad_y_pred * a1)
    grad_b2 = np.sum(grad_y_pred)
    grad_a1 = grad_y_pred * w2
    grad_z1 = grad_a1 * (z1 > 0)
    grad_w1 = grad_z1.T @ X
    grad_b1 = np.sum(grad_z1)
    w1 -= lr * grad_w1
    b1 -= lr * grad_b1
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2
    if epoch % 100 == 0:
        print(f"[Two-Layer] Epoch {epoch}, Loss: {loss:.4f}")

plt.figure()
plt.scatter(y, y_pred)
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.title('Two-Layer Neural Network Predictions')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# --- Two-Layer Fully-Connected Network Example ---
np.random.seed(42)
X = np.random.rand(200, 4)
true_W1 = np.array([[1.2, -0.7, 0.5, 2.0],
                   [0.3, 1.5, -1.0, 0.7],
                   [2.0, 0.1, 0.3, -0.5]])  # 3 hidden neurons
true_b1 = np.array([0.5, -0.2, 0.1])
H = np.maximum(X @ true_W1.T + true_b1, 0)  # Hidden layer (ReLU)
true_W2 = np.array([1.0, -2.0, 0.5])
true_b2 = 0.3
y = H @ true_W2 + true_b2 + np.random.normal(0, 0.2, size=H.shape[0])

W1 = np.random.randn(3, 4)
b1 = np.zeros(3)
W2 = np.random.randn(3)
b2 = 0.0
lr = 0.05

for epoch in range(600):
    Z1 = X @ W1.T + b1
    A1 = np.maximum(Z1, 0)  # ReLU
    y_pred = A1 @ W2 + b2
    loss = np.mean((y_pred - y) ** 2)
    grad_y_pred = 2 * (y_pred - y) / len(y)
    grad_W2 = A1.T @ grad_y_pred
    grad_b2 = np.sum(grad_y_pred)
    grad_A1 = np.outer(grad_y_pred, W2)
    grad_Z1 = grad_A1 * (Z1 > 0)
    grad_W1 = grad_Z1.T @ X
    grad_b1 = np.sum(grad_Z1, axis=0)
    W1 -= lr * grad_W1
    b1 -= lr * grad_b1
    W2 -= lr * grad_W2
    b2 -= lr * grad_b2
    if epoch % 100 == 0:
        print(f"[Fully-Connected] Epoch {epoch}, Loss: {loss:.4f}")

plt.figure()
plt.scatter(y, y_pred, alpha=0.6)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Two-Layer Fully-Connected Neural Network')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# --- Vectorization Example: For-loop vs. Vectorized Layer ---
np.random.seed(0)
X = np.random.rand(1000, 4)
W = np.random.randn(3, 4)
b = np.random.randn(3)

outputs_loop = np.zeros((1000, 3))
for i in range(1000):
    for j in range(3):
        outputs_loop[i, j] = np.dot(W[j], X[i]) + b[j]

outputs_vec = X @ W.T + b  # shape: (1000, 3)
print('Vectorization difference:', np.abs(outputs_loop - outputs_vec).max())

# --- Multi-layer (Deep) Neural Network Example ---
np.random.seed(123)
X = np.random.rand(300, 4)
true_W1 = np.random.randn(5, 4)
true_b1 = np.random.randn(5)
true_W2 = np.random.randn(3, 5)
true_b2 = np.random.randn(3)
true_W3 = np.random.randn(3)
true_b3 = 0.5
H1 = np.maximum(X @ true_W1.T + true_b1, 0)
H2 = np.maximum(H1 @ true_W2.T + true_b2, 0)
y = H2 @ true_W3 + true_b3 + np.random.normal(0, 0.2, size=H2.shape[0])

W1 = np.random.randn(5, 4)
b1 = np.zeros(5)
W2 = np.random.randn(3, 5)
b2 = np.zeros(3)
W3 = np.random.randn(3)
b3 = 0.0
lr = 0.03

for epoch in range(800):
    Z1 = X @ W1.T + b1
    A1 = np.maximum(Z1, 0)
    Z2 = A1 @ W2.T + b2
    A2 = np.maximum(Z2, 0)
    y_pred = A2 @ W3 + b3
    loss = np.mean((y_pred - y) ** 2)
    grad_y_pred = 2 * (y_pred - y) / len(y)
    grad_W3 = A2.T @ grad_y_pred
    grad_b3 = np.sum(grad_y_pred)
    grad_A2 = np.outer(grad_y_pred, W3)
    grad_Z2 = grad_A2 * (Z2 > 0)
    grad_W2 = grad_Z2.T @ A1
    grad_b2 = np.sum(grad_Z2, axis=0)
    grad_A1 = grad_Z2 @ W2
    grad_Z1 = grad_A1 * (Z1 > 0)
    grad_W1 = grad_Z1.T @ X
    grad_b1 = np.sum(grad_Z1, axis=0)
    W1 -= lr * grad_W1
    b1 -= lr * grad_b1
    W2 -= lr * grad_W2
    b2 -= lr * grad_b2
    W3 -= lr * grad_W3
    b3 -= lr * grad_b3
    if epoch % 200 == 0:
        print(f"[Deep NN] Epoch {epoch}, Loss: {loss:.4f}")

plt.figure()
plt.scatter(y, y_pred, alpha=0.6)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('Multi-layer (Deep) Neural Network')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()

# --- Activation Functions Comparison ---
from scipy.special import erf
z = np.linspace(-4, 4, 200)
relu = np.maximum(0, z)
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
leaky_relu = np.where(z > 0, z, 0.1 * z)
gelu = 0.5 * z * (1 + erf(z / np.sqrt(2)))
softplus = np.log(1 + np.exp(z))

plt.figure(figsize=(8, 5))
plt.plot(z, relu, label='ReLU')
plt.plot(z, sigmoid, label='Sigmoid')
plt.plot(z, tanh, label='Tanh')
plt.plot(z, leaky_relu, label='Leaky ReLU')
plt.plot(z, gelu, label='GELU')
plt.plot(z, softplus, label='Softplus')
plt.legend()
plt.title('Common Activation Functions')
plt.xlabel('z')
plt.ylabel('Activation')
plt.grid(True)
plt.show()

# --- Kernel vs. Neural Network Feature Map Example ---
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=0)

svm = SVC(kernel='rbf', gamma=2)
svm.fit(X, y)
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', max_iter=2000)
mlp.fit(X, y)

xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
Z_mlp = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_svm, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
plt.title('SVM with RBF Kernel (Fixed Feature Map)')
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_mlp, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
plt.title('Neural Network (Learned Feature Map)')
plt.show() 
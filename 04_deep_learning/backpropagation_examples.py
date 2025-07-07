"""
Python Examples for Backpropagation (Section 7.4)
This file contains code snippets for the main equations and calculations in the backpropagation notes.
"""
import numpy as np

# --- 1. Composition of Functions Example (Equation 7.52) ---
def g(z):
    # Example: elementwise square
    return z ** 2

def f(u):
    # Example: sum of elements
    return np.sum(u)

z = np.array([1.0, 2.0, 3.0])
u = g(z)
J = f(u)
print('u:', u)
print('J:', J)

# --- 2. Chain Rule for Vector Functions (Equation 7.53) ---
dJ_du = np.ones_like(u)  # dJ/du for f(u) = sum(u) is 1 for each element
# For g(z) = z**2, dg_j/dz_i = 2*z[i] if i==j else 0 (diagonal)
dg_dz = np.diag(2 * z)
dJ_dz = dg_dz @ dJ_du
print('dJ/dz:', dJ_dz)

# --- 3. Backward for Matrix Multiplication (Equation 7.63) ---
W = np.array([[1, 2], [3, 4]])
z = np.array([5, 6])
v = np.array([1, 1])
# Forward: MM = W @ z
# Backward: dJ/dz = W.T @ v
backward_z = W.T @ v
print('Backward for MM, dJ/dz:', backward_z)

# --- 4. Backward for ReLU Activation (Equation 7.69) ---
def relu(z):
    return np.maximum(0, z)
def relu_prime(z):
    return (z > 0).astype(float)
z = np.array([-1.0, 2.0, 3.0])
v = np.array([0.5, 0.5, 0.5])
dJ_dz = relu_prime(z) * v
print('Backward for ReLU, dJ/dz:', dJ_dz)

# --- 5. Backward for Logistic Loss (Equation 7.70) ---
def sigmoid(t):
    return 1 / (1 + np.exp(-t))
t = np.array([0.2, -1.0, 0.5])
y = np.array([1, 0, 1])
v = np.ones_like(t)
dJ_dt = (sigmoid(t) - y) * v
print('Backward for logistic loss, dJ/dt:', dJ_dt)

# --- 6. Full Forward and Backward Pass for a Simple MLP ---
def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1 - s)

print('\n--- Full Forward and Backward Pass for a Simple MLP ---')
x = np.array([1.0, 2.0])
W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
b1 = np.array([0.0, 0.0])
W2 = np.array([[0.5, -0.5]])
b2 = np.array([0.0])
y_true = np.array([1.0])

# Layer 1
z1 = W1 @ x + b1
print('z1:', z1)
a1 = relu(z1)
print('a1:', a1)
# Layer 2
z2 = W2 @ a1 + b2
print('z2:', z2)
a2 = sigmoid(z2)
print('a2:', a2)
# Loss (mean squared error for demonstration)
loss = 0.5 * (a2 - y_true) ** 2
print('loss:', loss)

# Backward pass
# dL/da2
dloss_da2 = a2 - y_true
# dL/dz2
dloss_dz2 = dloss_da2 * sigmoid_prime(z2)
# dL/dW2
dloss_dW2 = dloss_dz2 * a1
print('dloss/dW2:', dloss_dW2)
# dL/db2
dloss_db2 = dloss_dz2
print('dloss/db2:', dloss_db2)
# dL/da1
dloss_da1 = W2.T @ dloss_dz2
# dL/dz1
dloss_dz1 = dloss_da1 * relu_prime(z1)
# dL/dW1
dloss_dW1 = np.outer(dloss_dz1, x)
print('dloss/dW1:', dloss_dW1)
# dL/db1
dloss_db1 = dloss_dz1
print('dloss/db1:', dloss_db1) 
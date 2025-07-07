import numpy as np

# Mean Squared Error (MSE)
def mse(y_true, y_pred):
    """Compute mean squared error."""
    return np.mean(0.5 * (y_pred - y_true) ** 2)

# Example usage:
y_true = np.array([2, 4])
y_pred = np.array([2, 4])
print('MSE:', mse(y_true, y_pred))  # Output: 0.0

# Sigmoid function
def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

# Example usage:
print('Sigmoid(0):', sigmoid(0))  # Output: 0.5

# Binary cross-entropy (log-loss)
def binary_cross_entropy(y_true, y_pred):
    """Compute binary cross-entropy loss."""
    eps = 1e-15  # to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example usage:
y_true = np.array([1, 0, 1])
y_pred = np.array([0.9, 0.1, 0.8])
print('Binary cross-entropy:', binary_cross_entropy(y_true, y_pred))

# Softmax function
def softmax(logits):
    """Compute softmax probabilities from logits."""
    exps = np.exp(logits - np.max(logits))  # for numerical stability
    return exps / np.sum(exps)

# Example usage:
logits = np.array([2, 1, 0])
print('Softmax:', softmax(logits))  # Output: [0.66524096 0.24472847 0.09003057]

# Categorical cross-entropy (multi-class log-loss)
def categorical_cross_entropy(y_true, y_pred):
    """Compute categorical cross-entropy loss for one-hot y_true and probability y_pred."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))

# Example usage:
y_true = np.array([1, 0, 0])
y_pred = np.array([0.7, 0.2, 0.1])
print('Categorical cross-entropy:', categorical_cross_entropy(y_true, y_pred))

# Gradient descent update
def gradient_descent_update(theta, grad, alpha):
    """Perform a single gradient descent update."""
    return theta - alpha * grad

# Example usage:
theta = np.array([1.0, 2.0])
grad = np.array([0.1, -0.2])
alpha = 0.01
print('GD update:', gradient_descent_update(theta, grad, alpha))

# Mini-batch SGD update
def minibatch_sgd_update(theta, grads, alpha):
    """Perform a mini-batch SGD update. grads is an array of gradients for each example in the batch."""
    return theta - alpha * np.mean(grads, axis=0)

# Example usage:
theta = np.array([1.0, 2.0])
grads = np.array([[0.1, -0.2], [0.05, -0.1]])
alpha = 0.01
print('Mini-batch SGD update:', minibatch_sgd_update(theta, grads, alpha)) 
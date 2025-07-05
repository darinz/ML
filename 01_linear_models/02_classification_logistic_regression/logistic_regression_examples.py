import numpy as np

# Sigmoid (logistic) function
def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))

# Logistic regression hypothesis
def hypothesis(theta, x):
    """Compute the logistic regression hypothesis h_theta(x)."""
    return sigmoid(np.dot(theta, x))

# Derivative of the sigmoid function
def sigmoid_derivative(z):
    """Compute the derivative of the sigmoid function."""
    s = sigmoid(z)
    return s * (1 - s)

# Log-likelihood for logistic regression
def log_likelihood(theta, X, y):
    """Compute the log-likelihood for logistic regression."""
    h = sigmoid(X @ theta)
    return np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient of the log-likelihood
def gradient(theta, X, y):
    """Compute the gradient of the log-likelihood."""
    h = sigmoid(X @ theta)
    return X.T @ (y - h)

# Logistic loss for a single example
def logistic_loss(t, y):
    """Compute the logistic loss for a single example."""
    return y * np.log(1 + np.exp(-t)) + (1 - y) * np.log(1 + np.exp(t))

# Example usage: gradient ascent update (vectorized)
def gradient_ascent_update(theta, X, y, alpha=0.01):
    """Perform one step of gradient ascent for logistic regression."""
    grad = gradient(theta, X, y)
    return theta + alpha * grad

# Example: (uncomment to run)
# X = np.array([[1, 2], [1, 3], [1, 4]])  # Add bias term as first column
# y = np.array([0, 1, 1])
# theta = np.zeros(X.shape[1])
# for i in range(100):
#     theta = gradient_ascent_update(theta, X, y, alpha=0.1)
# print('Learned theta:', theta) 
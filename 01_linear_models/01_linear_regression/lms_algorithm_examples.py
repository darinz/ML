import numpy as np

# Hypothesis function for linear regression
def h_theta(theta, x):
    """Compute the linear hypothesis h_theta(x) = theta^T x."""
    return np.dot(theta, x)

# Cost function for a single example
def cost_single(theta, x, y):
    """Compute the cost for a single example."""
    return 0.5 * (h_theta(theta, x) - y) ** 2

# Gradient for a single example
def gradient_single(theta, x, y):
    """Compute the gradient of the cost for a single example."""
    prediction = h_theta(theta, x)
    error = prediction - y
    grad = error * x
    return grad

# Parameter update for a single example (stochastic)
def update_theta_single(theta, x, y, alpha):
    """Update theta for a single example using SGD."""
    grad = gradient_single(theta, x, y)
    return theta - alpha * grad

# Cost function for a dataset (batch)
def cost_batch(theta, X, y):
    """Compute the mean cost over a dataset."""
    predictions = X @ theta
    return 0.5 * np.mean((predictions - y) ** 2)

# Batch gradient descent update
def batch_gradient_descent(theta, X, y, alpha, num_iters):
    """Perform batch gradient descent for a number of iterations."""
    n = len(y)
    for _ in range(num_iters):
        predictions = X @ theta
        gradient = (X.T @ (y - predictions))
        theta = theta + alpha * gradient
    return theta

# Stochastic gradient descent update
def stochastic_gradient_descent(theta, X, y, alpha, num_epochs):
    """Perform stochastic gradient descent for a number of epochs."""
    n = len(y)
    for epoch in range(num_epochs):
        for i in range(n):
            xi = X[i]
            yi = y[i]
            theta = theta + alpha * (yi - h_theta(theta, xi)) * xi
    return theta

# Mini-batch gradient descent update
def minibatch_gradient_descent(theta, X, y, alpha, batch_size, num_epochs):
    """Perform mini-batch gradient descent for a number of epochs."""
    n = len(y)
    for epoch in range(num_epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for start in range(0, n, batch_size):
            end = start + batch_size
            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]
            predictions = xb @ theta
            gradient = xb.T @ (yb - predictions)
            theta = theta + alpha * gradient
    return theta 
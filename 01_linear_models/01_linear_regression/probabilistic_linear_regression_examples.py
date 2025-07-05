import numpy as np

# 1. Linear Model and Data Generation
np.random.seed(0)
n = 100  # number of data points
p = 2    # number of features

theta_true = np.array([2.0, -3.0])
sigma = 1.0

X = np.random.randn(n, p)
epsilon = np.random.normal(0, sigma, size=n)
y = X @ theta_true + epsilon

# 2. Gaussian Likelihood for a Single Data Point
def gaussian_likelihood(y_i, x_i, theta, sigma):
    """Compute p(y_i | x_i; theta) for the Gaussian model."""
    mu = np.dot(theta, x_i)
    coeff = 1.0 / np.sqrt(2 * np.pi * sigma**2)
    exponent = -((y_i - mu) ** 2) / (2 * sigma**2)
    return coeff * np.exp(exponent)

# Example usage:
p_likelihood = gaussian_likelihood(y[0], X[0], theta_true, sigma)
print(f"Likelihood of first point: {p_likelihood:.4f}")

# 3. Log-Likelihood for the Whole Dataset
def log_likelihood(y, X, theta, sigma):
    n = len(y)
    mu = X @ theta
    ll = -0.5 * n * np.log(2 * np.pi * sigma**2)
    ll -= np.sum((y - mu) ** 2) / (2 * sigma**2)
    return ll

# Example usage:
ll_val = log_likelihood(y, X, theta_true, sigma)
print(f"Log-likelihood at true theta: {ll_val:.2f}")

# 4. Least-Squares Solution (Normal Equations)
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Estimated theta: {theta_hat}")

# 5. Mean Squared Error (Cost Function)
def mean_squared_error(y, X, theta):
    return np.mean((y - X @ theta) ** 2)

mse = mean_squared_error(y, X, theta_hat)
print(f"Mean squared error at estimated theta: {mse:.4f}") 
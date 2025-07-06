import numpy as np
from scipy.stats import multivariate_normal

# Bayes Rule for Posterior
def bayes_posterior(px_y, py, px):
    """
    px_y: likelihood p(x|y) for each class (array)
    py: prior p(y) for each class (array)
    px: evidence p(x) (scalar or array)
    Returns: posterior p(y|x) for each class (array)
    """
    return (px_y * py) / px

# Multivariate Normal Density
def multivariate_normal_density(x, mu, Sigma):
    """
    x: data point (d-dimensional array)
    mu: mean vector (d-dimensional array)
    Sigma: covariance matrix (d x d array)
    Returns: density value at x
    """
    return multivariate_normal.pdf(x, mean=mu, cov=Sigma)

# GDA Parameter Estimation
def gda_fit(X, y):
    """
    X: n x d data matrix
    y: n-dimensional label vector (0 or 1)
    Returns: phi, mu0, mu1, Sigma
    """
    n = X.shape[0]
    phi = np.mean(y == 1)
    mu0 = X[y == 0].mean(axis=0)
    mu1 = X[y == 1].mean(axis=0)
    Sigma = np.zeros((X.shape[1], X.shape[1]))
    for i in range(n):
        mu_yi = mu1 if y[i] == 1 else mu0
        diff = (X[i] - mu_yi).reshape(-1, 1)
        Sigma += diff @ diff.T
    Sigma /= n
    return phi, mu0, mu1, Sigma

# GDA Prediction (Posterior and Class)
def gda_predict(X, phi, mu0, mu1, Sigma):
    """
    X: n x d data matrix
    Returns: predicted class labels (0 or 1)
    """
    p0 = multivariate_normal.pdf(X, mean=mu0, cov=Sigma)
    p1 = multivariate_normal.pdf(X, mean=mu1, cov=Sigma)
    # Compute posteriors (unnormalized)
    post0 = p0 * (1 - phi)
    post1 = p1 * phi
    return (post1 > post0).astype(int)

# Logistic Regression Form (for comparison)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_predict(X, theta):
    """
    X: n x d data matrix
    theta: d-dimensional parameter vector
    Returns: predicted probabilities for class 1
    """
    return sigmoid(X @ theta)

# =====================
# Example Usage & Tests
# =====================
if __name__ == "__main__":
    # Example for bayes_posterior
    px_y = np.array([0.2, 0.6])  # p(x|y=0), p(x|y=1)
    py = np.array([0.7, 0.3])    # p(y=0), p(y=1)
    px = np.sum(px_y * py)       # p(x)
    posterior = bayes_posterior(px_y, py, px)
    print("Bayes posterior:", posterior)

    # Example for multivariate_normal_density
    x = np.array([1.0, 2.0])
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
    density = multivariate_normal_density(x, mu, Sigma)
    print("Multivariate normal density at x:", density)

    # Example for GDA fit and predict
    # Synthetic dataset: 6 points, 2D, 2 classes
    X = np.array([
        [1.0, 2.0],
        [1.2, 1.9],
        [0.8, 2.2],
        [3.0, 3.5],
        [3.2, 3.0],
        [2.8, 3.2]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    phi, mu0, mu1, Sigma = gda_fit(X, y)
    print("GDA fit results:")
    print("phi:", phi)
    print("mu0:", mu0)
    print("mu1:", mu1)
    print("Sigma:\n", Sigma)

    # Predict on training data
    preds = gda_predict(X, phi, mu0, mu1, Sigma)
    print("GDA predictions:", preds)
    print("GDA accuracy:", np.mean(preds == y))

    # Example for logistic regression predict
    # (using random theta for demonstration)
    theta = np.array([0.5, -0.25])
    probs = logistic_regression_predict(X, theta)
    print("Logistic regression predicted probabilities:", probs) 
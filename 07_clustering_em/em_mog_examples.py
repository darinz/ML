import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# --- Initialization ---
def initialize_parameters(X, k, seed=None):
    """
    Randomly initialize the parameters for the mixture of Gaussians.
    Args:
        X: Data array of shape (n_samples, n_features)
        k: Number of clusters
        seed: Random seed (optional)
    Returns:
        phi: Mixing proportions (k,)
        mu: Means (k, n_features)
        Sigma: Covariances (k, n_features, n_features)
    """
    if seed is not None:
        np.random.seed(seed)
    n_samples, n_features = X.shape
    phi = np.ones(k) / k
    mu = X[np.random.choice(n_samples, k, replace=False)]
    Sigma = np.array([np.cov(X, rowvar=False) for _ in range(k)])
    return phi, mu, Sigma

# --- E-step ---
def e_step(X, phi, mu, Sigma):
    """
    E-step: Compute responsibilities (soft assignments).
    Args:
        X: Data array (n_samples, n_features)
        phi: Mixing proportions (k,)
        mu: Means (k, n_features)
        Sigma: Covariances (k, n_features, n_features)
    Returns:
        w: Responsibilities (n_samples, k)
    """
    n_samples, n_features = X.shape
    k = phi.shape[0]
    w = np.zeros((n_samples, k))
    for j in range(k):
        rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
        w[:, j] = phi[j] * rv.pdf(X)
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / w_sum
    return w

# --- M-step ---
def m_step(X, w):
    """
    M-step: Update parameters given responsibilities.
    Args:
        X: Data array (n_samples, n_features)
        w: Responsibilities (n_samples, k)
    Returns:
        phi: Updated mixing proportions (k,)
        mu: Updated means (k, n_features)
        Sigma: Updated covariances (k, n_features, n_features)
    """
    n_samples, n_features = X.shape
    k = w.shape[1]
    phi = w.sum(axis=0) / n_samples
    mu = (w.T @ X) / w.sum(axis=0)[:, None]
    Sigma = np.zeros((k, n_features, n_features))
    for j in range(k):
        X_centered = X - mu[j]
        weighted = w[:, j][:, None] * X_centered
        Sigma[j] = (weighted.T @ X_centered) / w[:, j].sum()
    return phi, mu, Sigma

# --- Log-likelihood ---
def log_likelihood(X, phi, mu, Sigma):
    """
    Compute the log-likelihood of the data under the current parameters.
    """
    n_samples = X.shape[0]
    k = phi.shape[0]
    total = np.zeros(n_samples)
    for j in range(k):
        rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
        total += phi[j] * rv.pdf(X)
    return np.sum(np.log(total))

# --- EM main loop ---
def em_mog(X, k, max_iters=100, tol=1e-4, seed=None, verbose=False):
    """
    Run the EM algorithm for a mixture of Gaussians.
    Args:
        X: Data array (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Convergence tolerance (on log-likelihood)
        seed: Random seed (optional)
        verbose: Print log-likelihood at each step
    Returns:
        phi, mu, Sigma: Estimated parameters
        w: Final responsibilities
        logliks: List of log-likelihoods
    """
    phi, mu, Sigma = initialize_parameters(X, k, seed=seed)
    logliks = []
    for i in range(max_iters):
        w = e_step(X, phi, mu, Sigma)
        phi, mu, Sigma = m_step(X, w)
        ll = log_likelihood(X, phi, mu, Sigma)
        logliks.append(ll)
        if verbose:
            print(f"Iteration {i+1}, log-likelihood: {ll:.4f}")
        if i > 0 and abs(logliks[-1] - logliks[-2]) < tol:
            break
    return phi, mu, Sigma, w, logliks

# --- Example usage ---
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Generate synthetic data
    X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.7, random_state=42)
    k = 3

    # Run EM for mixture of Gaussians
    phi, mu, Sigma, w, logliks = em_mog(X, k, max_iters=100, tol=1e-4, seed=42, verbose=True)

    print("Estimated mixing proportions (phi):", phi)
    print("Estimated means (mu):\n", mu)
    print("Estimated covariances (Sigma):\n", Sigma)

    # Assign each point to the most likely cluster
    labels = np.argmax(w, axis=1)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)
    plt.scatter(mu[:, 0], mu[:, 1], c='red', s=200, marker='X', label='Estimated Means')
    plt.title('EM for Mixture of Gaussians')
    plt.legend()
    plt.show()

    # Plot log-likelihood
    plt.figure()
    plt.plot(logliks, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.title('EM Log-likelihood Convergence')
    plt.show() 
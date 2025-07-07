import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# --- General EM Algorithm Skeleton (using ELBO) ---
def general_em(X, init_params, e_step, m_step, elbo_fn, max_iters=100, tol=1e-4, verbose=False):
    """
    General EM algorithm using user-supplied E-step, M-step, and ELBO functions.
    Args:
        X: Data array (n_samples, n_features)
        init_params: Initial parameters (user-defined structure)
        e_step: Function(X, params) -> Q (latent variable distributions)
        m_step: Function(X, Q) -> params (update parameters)
        elbo_fn: Function(X, Q, params) -> float (compute ELBO)
        max_iters: Maximum number of iterations
        tol: Convergence tolerance (on ELBO)
        verbose: Print ELBO at each step
    Returns:
        params: Final parameters
        Q: Final latent variable distributions
        elbos: List of ELBO values
    """
    params = init_params
    elbos = []
    for i in range(max_iters):
        Q = e_step(X, params)
        params = m_step(X, Q)
        elbo = elbo_fn(X, Q, params)
        elbos.append(elbo)
        if verbose:
            print(f"Iteration {i+1}, ELBO: {elbo:.4f}")
        if i > 0 and abs(elbos[-1] - elbos[-2]) < tol:
            break
    return params, Q, elbos

# --- EM for Gaussian Mixture Model (GMM) ---
def gmm_initialize(X, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n_samples, n_features = X.shape
    phi = np.ones(k) / k
    mu = X[np.random.choice(n_samples, k, replace=False)]
    Sigma = np.array([np.cov(X, rowvar=False) for _ in range(k)])
    return {'phi': phi, 'mu': mu, 'Sigma': Sigma}

def gmm_e_step(X, params):
    phi, mu, Sigma = params['phi'], params['mu'], params['Sigma']
    n_samples, n_features = X.shape
    k = phi.shape[0]
    w = np.zeros((n_samples, k))
    for j in range(k):
        rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
        w[:, j] = phi[j] * rv.pdf(X)
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / w_sum
    return w

def gmm_m_step(X, w):
    n_samples, n_features = X.shape
    k = w.shape[1]
    phi = w.sum(axis=0) / n_samples
    mu = (w.T @ X) / w.sum(axis=0)[:, None]
    Sigma = np.zeros((k, n_features, n_features))
    for j in range(k):
        X_centered = X - mu[j]
        weighted = w[:, j][:, None] * X_centered
        Sigma[j] = (weighted.T @ X_centered) / w[:, j].sum()
    return {'phi': phi, 'mu': mu, 'Sigma': Sigma}

def gmm_elbo(X, w, params):
    phi, mu, Sigma = params['phi'], params['mu'], params['Sigma']
    n_samples, n_features = X.shape
    k = phi.shape[0]
    elbo = 0.0
    for i in range(n_samples):
        for j in range(k):
            if w[i, j] > 0:
                rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
                log_p = np.log(phi[j] + 1e-12) + rv.logpdf(X[i])
                elbo += w[i, j] * (log_p - np.log(w[i, j] + 1e-12))
    return elbo

# --- Usage Example: GMM with EM ---
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    # Generate synthetic data
    X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.7, random_state=42)
    k = 3

    # Initialize parameters
    init_params = gmm_initialize(X, k, seed=42)

    # Run general EM algorithm for GMM
    params, w, elbos = general_em(
        X,
        init_params,
        gmm_e_step,
        gmm_m_step,
        gmm_elbo,
        max_iters=100,
        tol=1e-4,
        verbose=True
    )

    print("Estimated mixing proportions (phi):", params['phi'])
    print("Estimated means (mu):\n", params['mu'])
    print("Estimated covariances (Sigma):\n", params['Sigma'])

    # Assign each point to the most likely cluster
    labels = np.argmax(w, axis=1)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)
    plt.scatter(params['mu'][:, 0], params['mu'][:, 1], c='red', s=200, marker='X', label='Estimated Means')
    plt.title('EM for Gaussian Mixture Model')
    plt.legend()
    plt.show()

    # Plot ELBO
    plt.figure()
    plt.plot(elbos, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('EM ELBO Convergence')
    plt.show() 
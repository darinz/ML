"""
General Expectation-Maximization (EM) Algorithm Implementation

This file implements a general framework for the EM algorithm that can be applied
to various latent variable models with comprehensive examples and detailed explanations.

Key Concepts:
1. EM as a general framework for latent variable models
2. The ELBO (Evidence Lower BOund) as the objective function
3. E-step: Computing variational distributions Q(Z)
4. M-step: Maximizing expected complete log-likelihood
5. Jensen's inequality and the variational principle
6. KL divergence and the gap between ELBO and log-likelihood
7. Convergence properties and initialization strategies

Mathematical Foundation:
- ELBO: L(θ) = E_{Z~Q}[log p(X,Z|θ)] - E_{Z~Q}[log Q(Z)]
- E-step: Q(Z) = argmax_Q L(θ) = p(Z|X,θ)
- M-step: θ = argmax_θ E_{Z~Q}[log p(X,Z|θ)]
- Gap: log p(X|θ) - L(θ) = KL(Q(Z) || p(Z|X,θ))

Based on the concepts from 03_general_em.md
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

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
        verbose: Print progress information
    
    Returns:
        params: Final parameters
        Q: Final latent variable distributions
        elbos: List of ELBO values
        n_iters: Number of iterations performed
    
    Algorithm:
    1. Initialize parameters
    2. Repeat until convergence:
       a. E-step: Compute Q(Z) = argmax_Q ELBO
       b. M-step: Update θ = argmax_θ E_{Z~Q}[log p(X,Z|θ)]
       c. Check convergence (change in ELBO < tolerance)
    
    This general framework can be applied to any latent variable model by
    providing appropriate E-step, M-step, and ELBO functions.
    """
    params = init_params
    elbos = []
    
    for i in range(max_iters):
        # E-step: Compute variational distribution Q(Z)
        Q = e_step(X, params)
        
        # M-step: Update parameters
        params = m_step(X, Q)
        
        # Compute ELBO
        elbo = elbo_fn(X, Q, params)
        elbos.append(elbo)
        
        if verbose:
            print(f"Iteration {i+1}: ELBO = {elbo:.4f}")
        
        # Check convergence
        if i > 0 and abs(elbos[-1] - elbos[-2]) < tol:
            if verbose:
                print(f"Converged after {i+1} iterations")
            break
    
    return params, Q, elbos, i + 1


def demonstrate_jensens_inequality():
    """
    Demonstrate Jensen's inequality and its role in the ELBO.
    
    Jensen's inequality: E[f(X)] ≥ f(E[X]) for concave functions f
    For log function: E[log X] ≤ log E[X]
    """
    print("=== Jensen's Inequality Demonstration ===\n")
    
    # Generate random probabilities
    np.random.seed(42)
    p = np.random.dirichlet([1, 1, 1], size=100)  # Random probability distributions
    
    # Compute E[log p] and log E[p]
    log_p = np.log(p)
    E_log_p = log_p.mean(axis=0)
    E_p = p.mean(axis=0)
    log_E_p = np.log(E_p)
    
    print("Jensen's inequality for log function:")
    print(f"E[log p] = {E_log_p}")
    print(f"log E[p] = {log_E_p}")
    print(f"Inequality holds: {np.all(E_log_p <= log_E_p)}")
    print(f"Gap: {log_E_p - E_log_p}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    x = np.arange(len(E_log_p))
    width = 0.35
    
    plt.bar(x - width/2, E_log_p, width, label='E[log p]', alpha=0.8)
    plt.bar(x + width/2, log_E_p, width, label='log E[p]', alpha=0.8)
    plt.xlabel('Component')
    plt.ylabel('Value')
    plt.title("Jensen's Inequality: E[log p] ≤ log E[p]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return E_log_p, log_E_p


def compute_kl_divergence_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Compute KL divergence between two Gaussian distributions.
    
    Args:
        mu1, sigma1: Mean and std of first Gaussian
        mu2, sigma2: Mean and std of second Gaussian
    
    Returns:
        kl_div: KL divergence KL(N(μ₁,σ₁²) || N(μ₂,σ₂²))
    
    Mathematical Formulation:
    KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2
    """
    kl_div = np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 0.5
    return kl_div


def demonstrate_kl_divergence():
    """
    Demonstrate KL divergence and its properties.
    """
    print("\n=== KL Divergence Demonstration ===\n")
    
    # Create two Gaussian distributions
    mu1, sigma1 = 0, 1
    mu2, sigma2 = 2, 1.5
    
    # Compute KL divergence
    kl_forward = compute_kl_divergence_gaussian(mu1, sigma1, mu2, sigma2)
    kl_backward = compute_kl_divergence_gaussian(mu2, sigma2, mu1, sigma1)
    
    print(f"KL(N(μ₁={mu1},σ₁={sigma1}) || N(μ₂={mu2},σ₂={sigma2})) = {kl_forward:.4f}")
    print(f"KL(N(μ₂={mu2},σ₂={sigma2}) || N(μ₁={mu1},σ₁={sigma1})) = {kl_backward:.4f}")
    print(f"KL divergence is asymmetric: {kl_forward != kl_backward}")
    print(f"Both are non-negative: {kl_forward >= 0 and kl_backward >= 0}")
    
    # Visualize the distributions
    x = np.linspace(-5, 7, 1000)
    p1 = np.exp(-0.5 * ((x - mu1) / sigma1)**2) / (sigma1 * np.sqrt(2 * np.pi))
    p2 = np.exp(-0.5 * ((x - mu2) / sigma2)**2) / (sigma2 * np.sqrt(2 * np.pi))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, p1, 'b-', linewidth=2, label=f'N(μ={mu1}, σ={sigma1})')
    plt.plot(x, p2, 'r-', linewidth=2, label=f'N(μ={mu2}, σ={sigma2})')
    plt.fill_between(x, p1, p2, alpha=0.3, color='gray', label='Difference')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Gaussian Distributions and KL Divergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return kl_forward, kl_backward


# --- GMM Implementation using General EM Framework ---

def gmm_initialize(X, k, seed=None):
    """
    Initialize parameters for Gaussian Mixture Model.
    
    Args:
        X: Data array
        k: Number of components
        seed: Random seed
    
    Returns:
        params: Dictionary with 'phi', 'mu', 'Sigma' keys
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples, n_features = X.shape
    phi = np.ones(k) / k
    mu = X[np.random.choice(n_samples, k, replace=False)]
    Sigma = np.array([np.cov(X, rowvar=False) + 0.1 * np.eye(n_features) for _ in range(k)])
    
    return {'phi': phi, 'mu': mu, 'Sigma': Sigma}


def gmm_e_step(X, params):
    """
    E-step for GMM: Compute responsibilities (variational distribution).
    
    Args:
        X: Data array
        params: Dictionary with 'phi', 'mu', 'Sigma' keys
    
    Returns:
        w: Responsibilities (n_samples, k)
    
    This computes Q(zᵢ = j) = p(zᵢ = j|xᵢ, θ) for all i, j
    """
    phi, mu, Sigma = params['phi'], params['mu'], params['Sigma']
    n_samples, n_features = X.shape
    k = phi.shape[0]
    w = np.zeros((n_samples, k))
    
    for j in range(k):
        try:
            rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
            w[:, j] = phi[j] * rv.pdf(X)
        except:
            w[:, j] = 1e-10
    
    # Normalize
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1e-10
    w = w / w_sum
    
    return w


def gmm_m_step(X, w):
    """
    M-step for GMM: Update parameters given responsibilities.
    
    Args:
        X: Data array
        w: Responsibilities (n_samples, k)
    
    Returns:
        params: Updated parameters
    """
    n_samples, n_features = X.shape
    k = w.shape[1]
    
    phi = w.sum(axis=0) / n_samples
    mu = (w.T @ X) / w.sum(axis=0)[:, None]
    
    Sigma = np.zeros((k, n_features, n_features))
    for j in range(k):
        X_centered = X - mu[j]
        weighted = w[:, j][:, None] * X_centered
        Sigma[j] = (weighted.T @ X_centered) / w[:, j].sum() + 1e-6 * np.eye(n_features)
    
    return {'phi': phi, 'mu': mu, 'Sigma': Sigma}


def gmm_elbo(X, w, params):
    """
    Compute ELBO for GMM.
    
    Args:
        X: Data array
        w: Responsibilities
        params: Parameters
    
    Returns:
        elbo: ELBO value
    
    ELBO = E_{z~Q}[log p(X,Z|θ)] - E_{z~Q}[log Q(Z)]
    """
    phi, mu, Sigma = params['phi'], params['mu'], params['Sigma']
    n_samples, n_features = X.shape
    k = phi.shape[0]
    elbo = 0.0
    
    for i in range(n_samples):
        for j in range(k):
            if w[i, j] > 0:
                try:
                    rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
                    log_p = np.log(phi[j] + 1e-12) + rv.logpdf(X[i])
                    elbo += w[i, j] * (log_p - np.log(w[i, j] + 1e-12))
                except:
                    continue
    
    return elbo


# --- Bernoulli Mixture Model Implementation ---

def bmm_initialize(X, k, seed=None):
    """
    Initialize parameters for Bernoulli Mixture Model.
    
    Args:
        X: Binary data array
        k: Number of components
        seed: Random seed
    
    Returns:
        params: Dictionary with 'phi', 'theta' keys
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples, n_features = X.shape
    phi = np.ones(k) / k
    theta = np.random.beta(2, 2, size=(k, n_features))  # Random probabilities
    
    return {'phi': phi, 'theta': theta}


def bmm_e_step(X, params):
    """
    E-step for BMM: Compute responsibilities.
    
    Args:
        X: Binary data array
        params: Parameters
    
    Returns:
        w: Responsibilities (n_samples, k)
    """
    phi, theta = params['phi'], params['theta']
    n_samples, n_features = X.shape
    k = phi.shape[0]
    w = np.zeros((n_samples, k))
    
    for j in range(k):
        # Compute log-likelihood for component j
        log_likelihood = np.sum(X * np.log(theta[j] + 1e-12) + 
                               (1 - X) * np.log(1 - theta[j] + 1e-12), axis=1)
        w[:, j] = np.log(phi[j] + 1e-12) + log_likelihood
    
    # Convert to probabilities (softmax)
    w = np.exp(w - w.max(axis=1, keepdims=True))
    w = w / w.sum(axis=1, keepdims=True)
    
    return w


def bmm_m_step(X, w):
    """
    M-step for BMM: Update parameters.
    
    Args:
        X: Binary data array
        w: Responsibilities
    
    Returns:
        params: Updated parameters
    """
    n_samples, n_features = X.shape
    k = w.shape[1]
    
    phi = w.sum(axis=0) / n_samples
    theta = (w.T @ X) / w.sum(axis=0)[:, None]
    
    # Ensure probabilities are in [0, 1]
    theta = np.clip(theta, 1e-6, 1 - 1e-6)
    
    return {'phi': phi, 'theta': theta}


def bmm_elbo(X, w, params):
    """
    Compute ELBO for BMM.
    
    Args:
        X: Binary data array
        w: Responsibilities
        params: Parameters
    
    Returns:
        elbo: ELBO value
    """
    phi, theta = params['phi'], params['theta']
    n_samples, n_features = X.shape
    k = phi.shape[0]
    elbo = 0.0
    
    for i in range(n_samples):
        for j in range(k):
            if w[i, j] > 0:
                log_likelihood = np.sum(X[i] * np.log(theta[j] + 1e-12) + 
                                       (1 - X[i]) * np.log(1 - theta[j] + 1e-12))
                log_p = np.log(phi[j] + 1e-12) + log_likelihood
                elbo += w[i, j] * (log_p - np.log(w[i, j] + 1e-12))
    
    return elbo


def demonstrate_gmm_with_general_em():
    """
    Demonstrate GMM using the general EM framework.
    """
    print("\n=== GMM with General EM Framework ===\n")
    
    # Generate data
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
    k = 3
    
    # Initialize parameters
    init_params = gmm_initialize(X, k, seed=42)
    
    # Run general EM algorithm for GMM
    params, w, elbos, n_iters = general_em(
        X,
        init_params,
        gmm_e_step,
        gmm_m_step,
        gmm_elbo,
        max_iters=100,
        tol=1e-4,
        verbose=True
    )
    
    print(f"\nFinal parameters:")
    print(f"Mixing proportions: {params['phi']}")
    print(f"Means:\n{params['mu']}")
    
    # Visualize results
    labels = np.argmax(w, axis=1)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(params['mu'][:, 0], params['mu'][:, 1], c='red', s=200, 
               marker='X', edgecolors='black', linewidth=2, label='Component Means')
    plt.title('GMM Clustering Results')
    plt.legend()
    plt.colorbar(scatter, label='Component')
    
    plt.subplot(1, 2, 2)
    plt.plot(elbos, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('ELBO Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return params, w, elbos


def demonstrate_bmm_with_general_em():
    """
    Demonstrate Bernoulli Mixture Model using the general EM framework.
    """
    print("\n=== Bernoulli Mixture Model with General EM ===\n")
    
    # Generate binary data
    np.random.seed(42)
    n_samples, n_features = 200, 10
    k = 3
    
    # Create true parameters
    true_phi = np.array([0.4, 0.3, 0.3])
    true_theta = np.random.beta(2, 2, size=(k, n_features))
    
    # Generate data
    z_true = np.random.choice(k, size=n_samples, p=true_phi)
    X = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        X[i] = np.random.binomial(1, true_theta[z_true[i]])
    
    # Initialize parameters
    init_params = bmm_initialize(X, k, seed=42)
    
    # Run general EM algorithm for BMM
    params, w, elbos, n_iters = general_em(
        X,
        init_params,
        bmm_e_step,
        bmm_m_step,
        bmm_elbo,
        max_iters=50,
        tol=1e-4,
        verbose=True
    )
    
    print(f"\nFinal parameters:")
    print(f"Mixing proportions: {params['phi']}")
    print(f"Feature probabilities (first 3 features):\n{params['theta'][:, :3]}")
    
    # Visualize results
    labels = np.argmax(w, axis=1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(X.T, cmap='binary', aspect='auto')
    plt.title('Binary Data Matrix')
    plt.xlabel('Sample')
    plt.ylabel('Feature')
    plt.colorbar(label='Value')
    
    plt.subplot(1, 2, 2)
    plt.plot(elbos, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('BMM ELBO Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return params, w, elbos


def demonstrate_elbo_decomposition():
    """
    Demonstrate the decomposition of ELBO into reconstruction and KL terms.
    """
    print("\n=== ELBO Decomposition Demonstration ===\n")
    
    # Generate data
    X, _ = make_blobs(n_samples=100, centers=2, cluster_std=0.5, random_state=42)
    k = 2
    
    # Initialize parameters
    init_params = gmm_initialize(X, k, seed=42)
    
    # Run EM for a few iterations
    params, w, elbos, _ = general_em(
        X, init_params, gmm_e_step, gmm_m_step, gmm_elbo,
        max_iters=10, verbose=False
    )
    
    # Compute ELBO components
    phi, mu, Sigma = params['phi'], params['mu'], params['Sigma']
    n_samples, n_features = X.shape
    
    reconstruction_term = 0
    kl_term = 0
    
    for i in range(n_samples):
        for j in range(k):
            if w[i, j] > 0:
                try:
                    rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
                    log_likelihood = rv.logpdf(X[i])
                    reconstruction_term += w[i, j] * log_likelihood
                    kl_term += w[i, j] * (np.log(phi[j] + 1e-12) - np.log(w[i, j] + 1e-12))
                except:
                    continue
    
    print(f"ELBO decomposition:")
    print(f"Reconstruction term: {reconstruction_term:.4f}")
    print(f"KL divergence term: {kl_term:.4f}")
    print(f"Total ELBO: {reconstruction_term + kl_term:.4f}")
    print(f"From function: {elbos[-1]:.4f}")
    
    # Visualize convergence of components
    plt.figure(figsize=(10, 6))
    plt.plot(elbos, 'bo-', linewidth=2, markersize=6, label='Total ELBO')
    plt.axhline(y=reconstruction_term, color='r', linestyle='--', 
               label=f'Reconstruction term: {reconstruction_term:.4f}')
    plt.axhline(y=kl_term, color='g', linestyle='--', 
               label=f'KL term: {kl_term:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('ELBO and its Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return reconstruction_term, kl_term, elbos


if __name__ == "__main__":
    print("General Expectation-Maximization (EM) Algorithm Implementation")
    print("=" * 70)
    
    # Demonstrate Jensen's inequality
    demonstrate_jensens_inequality()
    
    # Demonstrate KL divergence
    demonstrate_kl_divergence()
    
    # Demonstrate GMM with general EM
    gmm_params, gmm_w, gmm_elbos = demonstrate_gmm_with_general_em()
    
    # Demonstrate BMM with general EM
    bmm_params, bmm_w, bmm_elbos = demonstrate_bmm_with_general_em()
    
    # Demonstrate ELBO decomposition
    recon_term, kl_term, elbos = demonstrate_elbo_decomposition()
    
    print("\n" + "=" * 70)
    print("Summary of Key Concepts Demonstrated:")
    print("1. General EM framework as coordinate ascent on ELBO")
    print("2. Jensen's inequality and its role in variational inference")
    print("3. KL divergence and its properties")
    print("4. E-step: Computing variational distributions Q(Z)")
    print("5. M-step: Maximizing expected complete log-likelihood")
    print("6. ELBO decomposition into reconstruction and KL terms")
    print("7. Application to different models (GMM, BMM)")
    print("8. Convergence properties and initialization strategies")
    print("9. Comparison with specific model implementations") 
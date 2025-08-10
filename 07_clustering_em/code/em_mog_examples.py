"""
Expectation-Maximization (EM) for Mixture of Gaussians Implementation

This file implements the EM algorithm for fitting Gaussian Mixture Models (GMM)
with comprehensive examples and detailed explanations of the underlying concepts.

Key Concepts:
1. Mixture models as latent variable models
2. The EM algorithm as coordinate ascent on the ELBO
3. E-step: Computing responsibilities (soft assignments)
4. M-step: Updating parameters given responsibilities
5. Convergence properties and initialization
6. Comparison with k-means clustering

Mathematical Foundation:
- Generative model: p(x) = Σⱼ₌₁ᵏ πⱼ N(x|μⱼ, Σⱼ)
- Latent variables: zᵢ ∈ {1,2,...,k} indicating component assignment
- Responsibilities: γ(zᵢⱼ) = p(zᵢ = j|xᵢ, θ)
- E-step: γ(zᵢⱼ) = πⱼ N(xᵢ|μⱼ, Σⱼ) / Σₗ πₗ N(xᵢ|μₗ, Σₗ)
- M-step: πⱼ = (1/N) Σᵢ γ(zᵢⱼ), μⱼ = Σᵢ γ(zᵢⱼ)xᵢ / Σᵢ γ(zᵢⱼ)

Based on the concepts from 02_em_mixture_of_gaussians.md
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.mixture import GaussianMixture
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def initialize_parameters(X, k, method='random', seed=None):
    """
    Initialize parameters for the Gaussian Mixture Model.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        k: Number of components
        method: 'random' or 'kmeans'
        seed: Random seed for reproducibility
    
    Returns:
        phi: Mixing proportions (k,)
        mu: Means (k, n_features)
        Sigma: Covariances (k, n_features, n_features)
    
    Concept: Good initialization is crucial for EM convergence. Random initialization
    can lead to poor local optima, while k-means initialization provides better starting points.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_samples, n_features = X.shape
    
    if method == 'random':
        # Random initialization
        phi = np.ones(k) / k  # Uniform mixing proportions
        
        # Random means from data points
        mu = X[np.random.choice(n_samples, k, replace=False)]
        
        # Random covariances based on data covariance
        data_cov = np.cov(X, rowvar=False)
        Sigma = np.array([data_cov + 0.1 * np.eye(n_features) for _ in range(k)])
        
    elif method == 'kmeans':
        # K-means initialization for better starting points
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=1)
        labels = kmeans.fit_predict(X)
        
        # Initialize parameters based on k-means clusters
        phi = np.zeros(k)
        mu = np.zeros((k, n_features))
        Sigma = np.zeros((k, n_features, n_features))
        
        for j in range(k):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                phi[j] = len(cluster_points) / n_samples
                mu[j] = cluster_points.mean(axis=0)
                Sigma[j] = np.cov(cluster_points, rowvar=False) + 0.1 * np.eye(n_features)
            else:
                # Handle empty clusters
                phi[j] = 1.0 / k
                mu[j] = X[np.random.choice(n_samples)]
                Sigma[j] = np.cov(X, rowvar=False) + 0.1 * np.eye(n_features)
    
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return phi, mu, Sigma


def e_step(X, phi, mu, Sigma):
    """
    E-step: Compute responsibilities (soft assignments) for each data point.
    
    Args:
        X: Data array (n_samples, n_features)
        phi: Mixing proportions (k,)
        mu: Means (k, n_features)
        Sigma: Covariances (k, n_features, n_features)
    
    Returns:
        w: Responsibilities (n_samples, k)
    
    Mathematical Formulation:
    γ(zᵢⱼ) = p(zᵢ = j|xᵢ, θ) = πⱼ N(xᵢ|μⱼ, Σⱼ) / Σₗ πₗ N(xᵢ|μₗ, Σₗ)
    
    This step computes the posterior probability of each data point belonging
    to each component, given the current parameter estimates.
    """
    n_samples, n_features = X.shape
    k = phi.shape[0]
    w = np.zeros((n_samples, k))
    
    # Compute unnormalized responsibilities for each component
    for j in range(k):
        try:
            # Create multivariate normal distribution for component j
            rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
            
            # Compute likelihood: πⱼ * N(xᵢ|μⱼ, Σⱼ)
            w[:, j] = phi[j] * rv.pdf(X)
        except:
            # Handle numerical issues with singular covariance
            w[:, j] = 1e-10
    
    # Normalize to get responsibilities (posterior probabilities)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1e-10  # Avoid division by zero
    w = w / w_sum
    
    return w


def m_step(X, w):
    """
    M-step: Update parameters given the responsibilities.
    
    Args:
        X: Data array (n_samples, n_features)
        w: Responsibilities (n_samples, k)
    
    Returns:
        phi: Updated mixing proportions (k,)
        mu: Updated means (k, n_features)
        Sigma: Updated covariances (k, n_features, n_features)
    
    Mathematical Formulation:
    πⱼ = (1/N) Σᵢ γ(zᵢⱼ)
    μⱼ = Σᵢ γ(zᵢⱼ)xᵢ / Σᵢ γ(zᵢⱼ)
    Σⱼ = Σᵢ γ(zᵢⱼ)(xᵢ - μⱼ)(xᵢ - μⱼ)ᵀ / Σᵢ γ(zᵢⱼ)
    
    This step maximizes the expected complete log-likelihood with respect to
    the parameters, given the current responsibilities.
    """
    n_samples, n_features = X.shape
    k = w.shape[1]
    
    # Update mixing proportions
    phi = w.sum(axis=0) / n_samples
    
    # Update means
    mu = (w.T @ X) / w.sum(axis=0)[:, None]
    
    # Update covariances
    Sigma = np.zeros((k, n_features, n_features))
    for j in range(k):
        # Center the data
        X_centered = X - mu[j]
        
        # Weighted covariance
        weighted = w[:, j][:, None] * X_centered
        Sigma[j] = (weighted.T @ X_centered) / w[:, j].sum()
        
        # Add regularization to ensure positive definiteness
        Sigma[j] += 1e-6 * np.eye(n_features)
    
    return phi, mu, Sigma


def compute_log_likelihood(X, phi, mu, Sigma):
    """
    Compute the log-likelihood of the data under the current parameters.
    
    Args:
        X: Data array
        phi: Mixing proportions
        mu: Means
        Sigma: Covariances
    
    Returns:
        log_likelihood: Total log-likelihood
    
    Mathematical Formulation:
    log p(X|θ) = Σᵢ log Σⱼ πⱼ N(xᵢ|μⱼ, Σⱼ)
    """
    n_samples = X.shape[0]
    k = phi.shape[0]
    total = np.zeros(n_samples)
    
    for j in range(k):
        try:
            rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
            total += phi[j] * rv.pdf(X)
        except:
            total += 1e-10
    
    return np.sum(np.log(total + 1e-10))


def compute_elbo(X, w, phi, mu, Sigma):
    """
    Compute the Evidence Lower BOund (ELBO).
    
    Args:
        X: Data array
        w: Responsibilities
        phi: Mixing proportions
        mu: Means
        Sigma: Covariances
    
    Returns:
        elbo: ELBO value
    
    Mathematical Formulation:
    ELBO = E_{z~q}[log p(X,Z|θ)] - E_{z~q}[log q(Z)]
    where q(Z) is the variational distribution defined by responsibilities w.
    """
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


def em_mog(X, k, max_iters=100, tol=1e-4, init_method='kmeans', seed=None, verbose=False):
    """
    Run the EM algorithm for Gaussian Mixture Model.
    
    Args:
        X: Data array (n_samples, n_features)
        k: Number of components
        max_iters: Maximum number of iterations
        tol: Convergence tolerance (on log-likelihood)
        init_method: Initialization method
        seed: Random seed
        verbose: Print progress information
    
    Returns:
        phi, mu, Sigma: Estimated parameters
        w: Final responsibilities
        logliks: List of log-likelihoods
        elbos: List of ELBO values
    
    Algorithm:
    1. Initialize parameters
    2. Repeat until convergence:
       a. E-step: Compute responsibilities
       b. M-step: Update parameters
       c. Check convergence (change in log-likelihood < tolerance)
    """
    # Initialize parameters
    phi, mu, Sigma = initialize_parameters(X, k, method=init_method, seed=seed)
    logliks = []
    elbos = []
    
    for i in range(max_iters):
        # E-step: Compute responsibilities
        w = e_step(X, phi, mu, Sigma)
        
        # M-step: Update parameters
        phi, mu, Sigma = m_step(X, w)
        
        # Compute log-likelihood and ELBO
        ll = compute_log_likelihood(X, phi, mu, Sigma)
        elbo = compute_elbo(X, w, phi, mu, Sigma)
        
        logliks.append(ll)
        elbos.append(elbo)
        
        if verbose:
            print(f"Iteration {i+1}: Log-likelihood = {ll:.4f}, ELBO = {elbo:.4f}")
        
        # Check convergence
        if i > 0 and abs(logliks[-1] - logliks[-2]) < tol:
            if verbose:
                print(f"Converged after {i+1} iterations")
            break
    
    return phi, mu, Sigma, w, logliks, elbos


def assign_clusters(w):
    """
    Assign each data point to the most likely component.
    
    Args:
        w: Responsibilities (n_samples, k)
    
    Returns:
        labels: Hard cluster assignments
    """
    return np.argmax(w, axis=1)


def evaluate_gmm(X, w, phi, mu, Sigma):
    """
    Evaluate GMM quality using multiple metrics.
    
    Args:
        X: Data array
        w: Responsibilities
        phi: Mixing proportions
        mu: Means
        Sigma: Covariances
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Log-likelihood
    log_likelihood = compute_log_likelihood(X, phi, mu, Sigma)
    
    # ELBO
    elbo = compute_elbo(X, w, phi, mu, Sigma)
    
    # Hard assignments
    labels = assign_clusters(w)
    
    # Silhouette score
    try:
        from sklearn.metrics import silhouette_score
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = np.nan
    
    # Component sizes
    component_sizes = w.sum(axis=0)
    
    return {
        'log_likelihood': log_likelihood,
        'elbo': elbo,
        'silhouette_score': silhouette,
        'component_sizes': component_sizes,
        'min_component_size': component_sizes.min(),
        'max_component_size': component_sizes.max()
    }


def visualize_gmm(X, w, phi, mu, Sigma, title="GMM Clustering Results"):
    """
    Visualize GMM results with components and decision boundaries.
    
    Args:
        X: Data array
        w: Responsibilities
        phi: Mixing proportions
        mu: Means
        Sigma: Covariances
        title: Plot title
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Soft assignments (responsibilities)
    plt.subplot(1, 2, 1)
    labels = assign_clusters(w)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    plt.scatter(mu[:, 0], mu[:, 1], c='red', s=200, marker='X', 
               edgecolors='black', linewidth=2, label='Component Means')
    
    # Plot component ellipses
    for j in range(len(mu)):
        try:
            # Create ellipse for component j
            from matplotlib.patches import Ellipse
            eigenvals, eigenvecs = np.linalg.eigh(Sigma[j])
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            ellipse = Ellipse(mu[j], 2*np.sqrt(eigenvals[0]), 2*np.sqrt(eigenvals[1]),
                            angle=angle, alpha=0.3, color=f'C{j}')
            plt.gca().add_patch(ellipse)
        except:
            pass
    
    plt.title(f"{title} - Hard Assignments")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter, label='Component')
    
    # Plot 2: Soft assignments (responsibilities)
    plt.subplot(1, 2, 2)
    # Color points by their maximum responsibility
    max_resp = w.max(axis=1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=max_resp, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    plt.scatter(mu[:, 0], mu[:, 1], c='red', s=200, marker='X', 
               edgecolors='black', linewidth=2, label='Component Means')
    
    plt.title(f"{title} - Soft Assignments (Max Responsibility)")
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter, label='Max Responsibility')
    
    plt.tight_layout()
    plt.show()


def compare_with_kmeans(X, k):
    """
    Compare GMM with k-means clustering.
    
    Args:
        X: Data array
        k: Number of clusters/components
    """
    print("=== Comparing GMM with K-Means ===\n")
    
    # Run GMM
    print("Running GMM...")
    phi, mu, Sigma, w, logliks, elbos = em_mog(X, k, verbose=True)
    gmm_labels = assign_clusters(w)
    gmm_metrics = evaluate_gmm(X, w, phi, mu, Sigma)
    
    # Run k-means
    print("\nRunning k-means...")
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X)
    
    # Evaluate k-means
    from sklearn.metrics import silhouette_score
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    
    # Compare results
    print(f"\n--- Comparison Results ---")
    print(f"GMM Log-likelihood: {gmm_metrics['log_likelihood']:.4f}")
    print(f"GMM Silhouette Score: {gmm_metrics['silhouette_score']:.4f}")
    print(f"K-means Silhouette Score: {kmeans_silhouette:.4f}")
    print(f"GMM Component Sizes: {gmm_metrics['component_sizes']}")
    print(f"K-means Cluster Sizes: {np.bincount(kmeans_labels)}")
    
    # Visualize comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(mu[:, 0], mu[:, 1], c='red', s=200, marker='X', edgecolors='black', linewidth=2)
    plt.title('GMM Clustering')
    plt.colorbar(scatter, label='Component')
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
               c='red', s=200, marker='X', edgecolors='black', linewidth=2)
    plt.title('K-Means Clustering')
    plt.colorbar(scatter, label='Cluster')
    
    plt.tight_layout()
    plt.show()
    
    return gmm_metrics, kmeans_silhouette


def demonstrate_convergence():
    """
    Demonstrate EM convergence properties.
    """
    print("\n=== EM Convergence Demonstration ===\n")
    
    # Generate data
    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)
    k = 3
    
    # Run EM with verbose output
    phi, mu, Sigma, w, logliks, elbos = em_mog(X, k, max_iters=50, verbose=True)
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(logliks, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Log-likelihood')
    plt.title('EM Log-likelihood Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(elbos, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('EM ELBO Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nConverged after {len(logliks)} iterations")
    print(f"Final log-likelihood: {logliks[-1]:.4f}")
    print(f"Final ELBO: {elbos[-1]:.4f}")
    print(f"Total improvement in log-likelihood: {((logliks[-1] - logliks[0]) / abs(logliks[0]) * 100):.1f}%")


def demonstrate_different_datasets():
    """
    Demonstrate GMM on different types of datasets.
    """
    print("\n=== GMM on Different Dataset Types ===\n")
    
    # Create different datasets
    datasets = {
        'Well-separated clusters': make_blobs(n_samples=300, centers=3, 
                                            cluster_std=0.8, random_state=42),
        'Overlapping clusters': make_blobs(n_samples=300, centers=3, 
                                         cluster_std=1.5, random_state=42),
        'Moons': make_moons(n_samples=300, noise=0.1, random_state=42)
    }
    
    for name, (X, y_true) in datasets.items():
        print(f"\n--- {name} ---")
        
        # Run GMM
        k = 3 if 'cluster' in name else 2
        phi, mu, Sigma, w, logliks, elbos = em_mog(X, k, verbose=False)
        
        # Evaluate
        metrics = evaluate_gmm(X, w, phi, mu, Sigma)
        print(f"Log-likelihood: {metrics['log_likelihood']:.2f}")
        print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
        print(f"Component sizes: {metrics['component_sizes']}")
        
        # Visualize
        visualize_gmm(X, w, phi, mu, Sigma, f"GMM: {name}")


if __name__ == "__main__":
    print("Expectation-Maximization for Gaussian Mixture Models")
    print("=" * 60)
    
    # Generate synthetic data
    print("1. Generating synthetic data with 3 Gaussian components...")
    X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.7, random_state=42)
    
    # Run EM for GMM
    print("\n2. Running EM algorithm for GMM...")
    phi, mu, Sigma, w, logliks, elbos = em_mog(X, k=3, max_iters=100, tol=1e-4, 
                                              init_method='kmeans', seed=42, verbose=True)
    
    print(f"\n3. Estimated parameters:")
    print(f"Mixing proportions (phi): {phi}")
    print(f"Means (mu):\n{mu}")
    print(f"Covariances (Sigma):\n{Sigma}")
    
    # Evaluate results
    print("\n4. Evaluating GMM quality...")
    metrics = evaluate_gmm(X, w, phi, mu, Sigma)
    print(f"Log-likelihood: {metrics['log_likelihood']:.4f}")
    print(f"ELBO: {metrics['elbo']:.4f}")
    print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
    print(f"Component sizes: {metrics['component_sizes']}")
    
    # Visualize results
    print("\n5. Visualizing results...")
    visualize_gmm(X, w, phi, mu, Sigma, "GMM Clustering Results")
    
    # Compare with k-means
    compare_with_kmeans(X, k=3)
    
    # Demonstrate convergence
    demonstrate_convergence()
    
    # Demonstrate on different datasets
    demonstrate_different_datasets()
    
    print("\n" + "=" * 60)
    print("Summary of Key Concepts Demonstrated:")
    print("1. GMM as a latent variable model with soft assignments")
    print("2. EM algorithm as coordinate ascent on the ELBO")
    print("3. E-step: Computing responsibilities (posterior probabilities)")
    print("4. M-step: Updating parameters given responsibilities")
    print("5. Convergence properties and initialization strategies")
    print("6. Comparison with k-means (hard vs soft assignments)")
    print("7. Evaluation metrics (log-likelihood, ELBO, silhouette score)")
    print("8. Visualization of component shapes and soft assignments") 
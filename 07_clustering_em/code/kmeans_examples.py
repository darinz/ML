"""
K-Means Clustering Implementation and Examples

This file implements the k-means clustering algorithm with comprehensive examples
and detailed explanations of the underlying concepts.

Key Concepts:
1. K-means as an optimization problem minimizing distortion
2. The two-step iterative algorithm (Assignment + Update)
3. Convergence properties and local optima
4. Initialization strategies (random vs k-means++)
5. Multiple runs for better solutions

Mathematical Foundation:
- Objective: minimize J = Σᵢ₌₁ⁿ Σⱼ₌₁ᵏ rᵢⱼ ||xᵢ - μⱼ||²
- Assignment: rᵢⱼ = 1 if j = argminₖ ||xᵢ - μₖ||², 0 otherwise
- Update: μⱼ = (1/|Cⱼ|) Σᵢ∈Cⱼ xᵢ

Based on the concepts from 01_clustering.md
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def initialize_centroids(X, k, method='random'):
    """
    Initialize centroids using different strategies.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        k: Number of clusters
        method: 'random' or 'kmeans++'
    
    Returns:
        centroids: Array of shape (k, n_features)
    
    Concept: The choice of initial centroids significantly affects the final
    clustering quality due to k-means' sensitivity to local optima.
    """
    if method == 'random':
        # Simple random initialization
        indices = np.random.choice(X.shape[0], k, replace=False)
        return X[indices]
    
    elif method == 'kmeans++':
        # K-means++ initialization for better starting points
        n_samples, n_features = X.shape
        centroids = np.empty((k, n_features))
        
        # Choose first centroid uniformly at random
        centroids[0] = X[np.random.choice(n_samples)]
        
        # Choose remaining centroids with probability proportional to distance²
        for i in range(1, k):
            # Compute distances to nearest existing centroid
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids[:i]], axis=0)
            
            # Probability proportional to distance²
            probs = distances ** 2
            probs /= probs.sum()
            
            # Choose next centroid
            centroids[i] = X[np.random.choice(n_samples, p=probs)]
        
        return centroids
    
    else:
        raise ValueError(f"Unknown initialization method: {method}")


def assign_clusters(X, centroids):
    """
    Assignment step: Assign each data point to the nearest centroid.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        centroids: Array of shape (k, n_features)
    
    Returns:
        labels: Array of shape (n_samples,) with cluster indices
    
    Mathematical Formulation:
    rᵢⱼ = 1 if j = argminₖ ||xᵢ - μₖ||², 0 otherwise
    
    This step minimizes the objective function with respect to the assignments
    while keeping the centroids fixed.
    """
    # Compute distances from each point to each centroid
    # Broadcasting: X[:, np.newaxis] has shape (n_samples, 1, n_features)
    # centroids has shape (k, n_features)
    # Result: (n_samples, k, n_features) -> (n_samples, k) after norm
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    
    # Assign each point to the nearest centroid
    labels = np.argmin(distances, axis=1)
    
    return labels


def update_centroids(X, labels, k):
    """
    Update step: Update centroids as the mean of assigned points.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with cluster indices
        k: Number of clusters
    
    Returns:
        centroids: Updated centroids array of shape (k, n_features)
    
    Mathematical Formulation:
    μⱼ = (1/|Cⱼ|) Σᵢ∈Cⱼ xᵢ
    
    This step minimizes the objective function with respect to the centroids
    while keeping the assignments fixed.
    """
    centroids = np.zeros((k, X.shape[1]))
    
    for j in range(k):
        # Get all points assigned to cluster j
        cluster_points = X[labels == j]
        
        if len(cluster_points) > 0:
            # Update centroid as mean of assigned points
            centroids[j] = cluster_points.mean(axis=0)
        else:
            # Handle empty clusters by reinitializing randomly
            centroids[j] = X[np.random.choice(X.shape[0])]
    
    return centroids


def compute_distortion(X, centroids, labels):
    """
    Compute the distortion (objective function value).
    
    Args:
        X: Data array of shape (n_samples, n_features)
        centroids: Array of shape (k, n_features)
        labels: Array of shape (n_samples,) with cluster indices
    
    Returns:
        distortion: Total distortion (float)
    
    Mathematical Formulation:
    J = Σᵢ₌₁ⁿ ||xᵢ - μ_{cᵢ}||²
    where cᵢ is the cluster assignment for point i
    """
    total_distortion = 0
    for i in range(len(X)):
        cluster_idx = labels[i]
        centroid = centroids[cluster_idx]
        total_distortion += np.sum((X[i] - centroid) ** 2)
    
    return total_distortion


def kmeans(X, k, max_iters=100, tol=1e-4, init_method='kmeans++', verbose=False):
    """
    Run the complete k-means algorithm.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence (change in distortion)
        init_method: Initialization method ('random' or 'kmeans++')
        verbose: Print progress information
    
    Returns:
        centroids: Final centroids
        labels: Final cluster assignments
        distortion_history: List of distortion values at each iteration
        n_iters: Number of iterations performed
    
    Algorithm:
    1. Initialize centroids
    2. Repeat until convergence:
       a. Assignment step: Assign points to nearest centroids
       b. Update step: Update centroids as means of assigned points
       c. Check convergence (change in distortion < tolerance)
    """
    # Initialize centroids
    centroids = initialize_centroids(X, k, method=init_method)
    distortion_history = []
    
    for iteration in range(max_iters):
        # Assignment step
        labels = assign_clusters(X, centroids)
        
        # Update step
        new_centroids = update_centroids(X, labels, k)
        
        # Compute current distortion
        current_distortion = compute_distortion(X, new_centroids, labels)
        distortion_history.append(current_distortion)
        
        if verbose:
            print(f"Iteration {iteration + 1}: Distortion = {current_distortion:.4f}")
        
        # Check convergence
        if iteration > 0:
            distortion_change = abs(distortion_history[-1] - distortion_history[-2])
            if distortion_change < tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
        
        centroids = new_centroids
    
    return centroids, labels, distortion_history, iteration + 1


def best_of_n_runs(X, k, n_runs=10, init_method='kmeans++', verbose=False):
    """
    Run k-means multiple times and return the best result.
    
    Args:
        X: Data array of shape (n_samples, n_features)
        k: Number of clusters
        n_runs: Number of times to run k-means
        init_method: Initialization method
        verbose: Print progress information
    
    Returns:
        best_centroids: Centroids from the best run
        best_labels: Labels from the best run
        best_distortion: Lowest distortion achieved
        all_results: List of all results for analysis
    
    Rationale: K-means can get stuck in local optima. Running multiple times
    with different initializations increases the chance of finding a good solution.
    """
    best_distortion = float('inf')
    best_centroids, best_labels = None, None
    all_results = []
    
    for run in range(n_runs):
        if verbose:
            print(f"Run {run + 1}/{n_runs}")
        
        centroids, labels, distortion_history, n_iters = kmeans(
            X, k, init_method=init_method, verbose=False
        )
        
        final_distortion = distortion_history[-1]
        all_results.append({
            'centroids': centroids,
            'labels': labels,
            'distortion': final_distortion,
            'n_iters': n_iters,
            'distortion_history': distortion_history
        })
        
        if final_distortion < best_distortion:
            best_distortion = final_distortion
            best_centroids = centroids
            best_labels = labels
    
    if verbose:
        print(f"Best distortion achieved: {best_distortion:.4f}")
    
    return best_centroids, best_labels, best_distortion, all_results


def evaluate_clustering(X, labels, centroids):
    """
    Evaluate clustering quality using multiple metrics.
    
    Args:
        X: Data array
        labels: Cluster assignments
        centroids: Cluster centroids
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Distortion (lower is better)
    distortion = compute_distortion(X, centroids, labels)
    
    # Silhouette score (higher is better, range [-1, 1])
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = np.nan  # Can fail with single cluster
    
    # Number of clusters
    n_clusters = len(np.unique(labels))
    
    # Cluster sizes
    cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
    
    return {
        'distortion': distortion,
        'silhouette_score': silhouette,
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes,
        'min_cluster_size': min(cluster_sizes),
        'max_cluster_size': max(cluster_sizes)
    }


def visualize_clustering(X, labels, centroids, title="K-Means Clustering Results"):
    """
    Visualize clustering results with centroids and cluster boundaries.
    
    Args:
        X: Data array
        labels: Cluster assignments
        centroids: Cluster centroids
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Plot data points colored by cluster
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                         s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, 
               marker='X', edgecolors='black', linewidth=2, label='Centroids')
    
    # Add cluster boundaries (Voronoi diagram approximation)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Assign each grid point to nearest centroid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_labels = assign_clusters(grid_points, centroids)
    grid_labels = grid_labels.reshape(xx.shape)
    
    # Plot decision boundaries
    plt.contour(xx, yy, grid_labels, alpha=0.3, colors='black', linewidths=0.5)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def demonstrate_kmeans_concepts():
    """
    Demonstrate key k-means concepts with examples.
    """
    print("=== K-Means Clustering Concepts Demonstration ===\n")
    
    # 1. Generate synthetic data
    print("1. Generating synthetic data with 3 clusters...")
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.8, 
                          random_state=42)
    
    # 2. Compare initialization methods
    print("\n2. Comparing initialization methods...")
    k = 3
    
    # Random initialization
    centroids_random, labels_random, distortion_random, _ = kmeans(
        X, k, init_method='random', verbose=False
    )
    
    # K-means++ initialization
    centroids_plus, labels_plus, distortion_plus, _ = kmeans(
        X, k, init_method='kmeans++', verbose=False
    )
    
    print(f"Random initialization distortion: {distortion_random:.2f}")
    print(f"K-means++ initialization distortion: {distortion_plus:.2f}")
    print(f"Improvement: {((distortion_random - distortion_plus) / distortion_random * 100):.1f}%")
    
    # 3. Multiple runs demonstration
    print("\n3. Running multiple times for better results...")
    best_centroids, best_labels, best_distortion, all_results = best_of_n_runs(
        X, k, n_runs=10, verbose=True
    )
    
    # 4. Evaluate clustering
    print("\n4. Evaluating clustering quality...")
    metrics = evaluate_clustering(X, best_labels, best_centroids)
    print(f"Final distortion: {metrics['distortion']:.2f}")
    print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
    print(f"Cluster sizes: {metrics['cluster_sizes']}")
    
    # 5. Visualize results
    print("\n5. Visualizing results...")
    visualize_clustering(X, best_labels, best_centroids, 
                        "K-Means Clustering (Best of 10 Runs)")
    
    return X, best_labels, best_centroids, metrics


def demonstrate_convergence():
    """
    Demonstrate k-means convergence properties.
    """
    print("\n=== K-Means Convergence Demonstration ===\n")
    
    # Generate data
    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=42)
    k = 3
    
    # Run k-means with verbose output
    centroids, labels, distortion_history, n_iters = kmeans(
        X, k, max_iters=20, verbose=True
    )
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(distortion_history, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Iteration')
    plt.ylabel('Distortion (Objective Function)')
    plt.title('K-Means Convergence')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(len(distortion_history)))
    plt.tight_layout()
    plt.show()
    
    print(f"\nConverged after {n_iters} iterations")
    print(f"Final distortion: {distortion_history[-1]:.4f}")
    print(f"Total improvement: {((distortion_history[0] - distortion_history[-1]) / distortion_history[0] * 100):.1f}%")


def demonstrate_different_datasets():
    """
    Demonstrate k-means on different types of datasets.
    """
    print("\n=== K-Means on Different Dataset Types ===\n")
    
    # Create different datasets
    datasets = {
        'Well-separated clusters': make_blobs(n_samples=300, centers=3, 
                                            cluster_std=0.8, random_state=42),
        'Overlapping clusters': make_blobs(n_samples=300, centers=3, 
                                         cluster_std=1.5, random_state=42),
        'Moons': make_moons(n_samples=300, noise=0.1, random_state=42),
        'Circles': make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
    }
    
    for name, (X, y_true) in datasets.items():
        print(f"\n--- {name} ---")
        
        # Run k-means
        k = 3 if 'cluster' in name else 2
        centroids, labels, distortion, _ = kmeans(X, k, verbose=False)
        
        # Evaluate
        metrics = evaluate_clustering(X, labels, centroids)
        print(f"Distortion: {metrics['distortion']:.2f}")
        print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
        
        # Visualize
        visualize_clustering(X, labels, centroids, f"K-Means: {name}")


if __name__ == "__main__":
    print("K-Means Clustering Implementation and Examples")
    print("=" * 50)
    
    # Run demonstrations
    X, labels, centroids, metrics = demonstrate_kmeans_concepts()
    demonstrate_convergence()
    demonstrate_different_datasets()
    
    print("\n" + "=" * 50)
    print("Summary of Key Concepts Demonstrated:")
    print("1. K-means as an optimization algorithm minimizing distortion")
    print("2. The two-step iterative process (Assignment + Update)")
    print("3. Importance of initialization (random vs k-means++)")
    print("4. Multiple runs to avoid local optima")
    print("5. Convergence properties and stopping criteria")
    print("6. Evaluation metrics (distortion, silhouette score)")
    print("7. Limitations on non-globular cluster shapes") 
import numpy as np

# --- Initialization ---
def initialize_centroids(X, k):
    """
    Randomly select k data points as initial centroids.
    Args:
        X: Data array of shape (n_samples, n_features)
        k: Number of clusters
    Returns:
        centroids: Array of shape (k, n_features)
    """
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

# --- Assignment Step ---
def assign_clusters(X, centroids):
    """
    Assign each data point to the nearest centroid.
    Args:
        X: Data array of shape (n_samples, n_features)
        centroids: Array of shape (k, n_features)
    Returns:
        labels: Array of shape (n_samples,) with cluster indices
    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# --- Update Step ---
def update_centroids(X, labels, k):
    """
    Update centroids as the mean of assigned points.
    Args:
        X: Data array of shape (n_samples, n_features)
        labels: Array of shape (n_samples,) with cluster indices
        k: Number of clusters
    Returns:
        centroids: Updated centroids array of shape (k, n_features)
    """
    centroids = np.zeros((k, X.shape[1]))
    for j in range(k):
        points = X[labels == j]
        if len(points) > 0:
            centroids[j] = points.mean(axis=0)
        else:
            # Reinitialize centroid randomly if no points assigned
            centroids[j] = X[np.random.choice(X.shape[0])]
    return centroids

# --- Full k-means Algorithm ---
def kmeans(X, k, max_iters=100, tol=1e-4):
    """
    Run the full k-means algorithm.
    Args:
        X: Data array of shape (n_samples, n_features)
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Tolerance for convergence
    Returns:
        centroids: Final centroids
        labels: Cluster assignments
    """
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids
    return centroids, labels

# --- k-means++ Initialization ---
def kmeans_plus_plus_init(X, k):
    """
    k-means++ initialization for better starting centroids.
    Args:
        X: Data array of shape (n_samples, n_features)
        k: Number of clusters
    Returns:
        centroids: Array of shape (k, n_features)
    """
    n_samples, n_features = X.shape
    centroids = np.empty((k, n_features))
    centroids[0] = X[np.random.choice(n_samples)]
    for i in range(1, k):
        distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids[:i], axis=2), axis=1)
        probs = distances ** 2
        probs /= probs.sum()
        centroids[i] = X[np.random.choice(n_samples, p=probs)]
    return centroids

# --- Distortion Function ---
def distortion(X, centroids, labels):
    """
    Compute the distortion (sum of squared distances).
    Args:
        X: Data array of shape (n_samples, n_features)
        centroids: Array of shape (k, n_features)
        labels: Array of shape (n_samples,) with cluster indices
    Returns:
        Total distortion (float)
    """
    return np.sum((X - centroids[labels]) ** 2)

# --- Multiple Runs for Best Result ---
def best_of_n_runs(X, k, n_runs=10):
    """
    Run k-means multiple times and return the best result.
    Args:
        X: Data array of shape (n_samples, n_features)
        k: Number of clusters
        n_runs: Number of times to run k-means
    Returns:
        best_centroids: Centroids from the best run
        best_labels: Labels from the best run
        best_J: Lowest distortion achieved
    """
    best_J = float('inf')
    best_centroids, best_labels = None, None
    for _ in range(n_runs):
        centroids, labels = kmeans(X, k)
        J = distortion(X, centroids, labels)
        if J < best_J:
            best_J = J
            best_centroids, best_labels = centroids, labels
    return best_centroids, best_labels, best_J

if __name__ == "__main__":
    # Example usage: Generate synthetic data
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    # Create a dataset with 3 clusters
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
    k = 3

    # Run k-means
    centroids, labels = kmeans(X, k)
    print("Centroids:\n", centroids)
    print("Distortion:", distortion(X, centroids, labels))

    # Plot the results
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.75, marker='X')
    plt.title('K-means Clustering Results')
    plt.show() 
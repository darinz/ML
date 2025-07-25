from sklearn.cluster import KMeans
import numpy as np

def k_means(X, k):
    """
    K-means clustering using scikit-learn.
    Args:
        X: Data matrix (numpy array)
        k: Number of clusters (int)
    Returns:
        clusters: Cluster labels (numpy array, 1-based like MATLAB)
        centroids: Centroid coordinates (numpy array)
    """
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    kmeans.fit(X)
    clusters = kmeans.labels_ + 1  # Convert to 1-based indexing to match MATLAB
    centroids = kmeans.cluster_centers_
    return clusters, centroids 
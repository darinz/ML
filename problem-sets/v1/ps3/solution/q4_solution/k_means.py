import numpy as np
import time
from draw_clusters import draw_clusters

def k_means(X, k):
    m, n = X.shape
    oldcentroids = np.zeros((k, n))
    # Randomly initialize centroids by selecting k random data points
    centroids = X[np.random.choice(m, k, replace=False), :]
    clusters = np.zeros(m, dtype=int)

    while np.linalg.norm(oldcentroids - centroids) > 1e-15:
        oldcentroids = centroids.copy()
        # compute cluster assignments
        for i in range(m):
            dists = np.sum((centroids - X[i, :]) ** 2, axis=1)
            clusters[i] = np.argmin(dists) + 1  # 1-based indexing
        draw_clusters(X, clusters, centroids)
        time.sleep(0.1)
        # compute cluster centroids
        for i in range(1, k + 1):
            if np.any(clusters == i):
                centroids[i - 1, :] = np.mean(X[clusters == i, :], axis=0)
    return clusters, centroids 
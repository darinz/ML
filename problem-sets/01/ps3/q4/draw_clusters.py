import matplotlib.pyplot as plt
import numpy as np

def draw_clusters(X, clusters, centroids):
    plt.clf()
    plt.gca().set_aspect('equal', adjustable='box')
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    markers = ['o', 'o', 'o', 'o', 'o', 'o']
    n_clusters = int(np.max(clusters))
    for i in range(1, n_clusters + 1):
        color = colors[(i-1) % len(colors)]
        marker = markers[(i-1) % len(markers)]
        mask = (clusters == i)
        plt.plot(X[mask, 0], X[mask, 1], color + marker, label=f'Cluster {i}')
    plt.plot(centroids[:, 0], centroids[:, 1], 'kx', markersize=10, label='Centroids')
    plt.legend()
    plt.show() 
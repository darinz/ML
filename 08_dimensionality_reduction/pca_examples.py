"""
Principal Components Analysis (PCA) - Python Examples
This script collects all the code snippets from the PCA notes, organized by section.
Each section can be run independently for educational purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SKPCA
from sklearn.preprocessing import StandardScaler

# ---
# 1. Data Normalization Example
# ---
print("\n--- Data Normalization Example ---")
X = np.array([
    [170, 30],
    [160, 25],
    [180, 35]
], dtype=float)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_normalized = (X - mean) / std
print("Normalized data:\n", X_normalized)

# ---
# 2. Covariance Matrix Calculation
# ---
print("\n--- Covariance Matrix Calculation ---")
cov_matrix = np.cov(X_normalized, rowvar=False)
print("Covariance matrix:\n", cov_matrix)

# ---
# 3. Eigen Decomposition for PCA
# ---
print("\n--- Eigen Decomposition for PCA ---")
eigvals, eigvecs = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eigvals)
print("Eigenvectors (columns):\n", eigvecs)
# Sort eigenvectors by descending eigenvalue
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
eigvals = eigvals[idx]
pc1 = eigvecs[:, 0]
print("First principal component:", pc1)

# ---
# 4. Projecting Data onto Principal Components
# ---
print("\n--- Projecting Data onto Principal Components ---")
k = 1  # or 2 for 2D
W = eigvecs[:, :k]
X_pca = X_normalized @ W
print(f"Data projected onto first {k} principal component(s):\n", X_pca)

# ---
# 5. Visualizing PCA Results (2D)
# ---
print("\n--- Visualizing PCA Results (2D) ---")
W2 = eigvecs[:, :2]
X_pca2 = X_normalized @ W2
plt.figure()
plt.scatter(X_pca2[:, 0], X_pca2[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection (first 2 components)')
plt.grid(True)
plt.show()

# ---
# 6. Example: PCA on a Small Dataset
# ---
print("\n--- PCA on a Small Dataset ---")
X_small = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2, 1.6],
    [1, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])
X_mean = np.mean(X_small, axis=0)
X_std = np.std(X_small, axis=0)
X_norm = (X_small - X_mean) / X_std
cov = np.cov(X_norm, rowvar=False)
eigvals, eigvecs = np.linalg.eig(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
X_pca = X_norm @ eigvecs[:, :2]
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Example: Small Dataset')
plt.grid(True)
plt.show()

# ---
# 7. Using scikit-learn for PCA
# ---
print("\n--- Using scikit-learn for PCA ---")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_small)
pca = SKPCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("PCA components (directions):\n", pca.components_) 
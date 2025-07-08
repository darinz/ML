"""
Independent Components Analysis (ICA) - Python Examples
This script provides code implementations for the main concepts in the ICA notes.
Each section can be run independently for educational purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# ---
# 1. Simulate the Cocktail Party Problem (Mixing Sources)
# ---
print("\n--- Simulating the Cocktail Party Problem ---")
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# Generate independent sources
s1 = np.sin(2 * time)  # Sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Square signal
s3 = np.random.laplace(size=n_samples)  # Non-Gaussian noise
S = np.c_[s1, s2, s3]
S /= S.std(axis=0)  # Standardize data

# Mixing matrix (random)
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
X = S @ A.T  # Generate observations (mixtures)

# ---
# 2. Visualize the Sources and Mixtures
# ---
plt.figure(figsize=(9, 6))
models = [S, X]
names = ['True Sources', 'Observed Mixtures (Microphones)']
for i, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(2, 1, i)
    for sig in model.T:
        plt.plot(sig)
    plt.title(name)
plt.tight_layout()
plt.show()

# ---
# 3. Whitening/Preprocessing (optional, for educational clarity)
# ---
print("\n--- Whitening the Mixtures (PCA Preprocessing) ---")
X_centered = X - X.mean(axis=0)
cov = np.cov(X_centered, rowvar=False)
D, E = np.linalg.eigh(cov)
D = np.diag(1.0 / np.sqrt(D))
X_white = (X_centered @ E) @ D
print("Whitened data shape:", X_white.shape)

# ---
# 4. ICA Algorithm (Gradient Ascent for W, simplified)
# ---
print("\n--- ICA via Gradient Ascent (Simplified) ---")
def g(x):
    return 1 / (1 + np.exp(-x))  # Sigmoid

def g_prime(x):
    gx = g(x)
    return gx * (1 - gx)

# FastICA-like fixed-point iteration for demonstration
W = np.random.randn(3, 3)
alpha = 0.1
for iteration in range(100):
    WX = X_white @ W.T
    gwx = g(WX)
    g_wx = 1 - 2 * gwx
    dW = (g_wx.T @ X_white) / n_samples + np.linalg.inv(W.T)
    W += alpha * dW
    # Decorrelate rows of W (optional, for stability)
    W = np.linalg.inv(np.sqrtm(W @ W.T)) @ W

S_est = X_white @ W.T

# ---
# 5. Visualize the Recovered Sources
# ---
plt.figure(figsize=(9, 6))
plt.subplot(2, 1, 1)
for sig in S.T:
    plt.plot(sig)
plt.title('True Sources')
plt.subplot(2, 1, 2)
for sig in S_est.T:
    plt.plot(sig)
plt.title('Recovered Sources (ICA)')
plt.tight_layout()
plt.show()

# ---
# 6. Using scikit-learn's FastICA
# ---
print("\n--- ICA using scikit-learn's FastICA ---")
ica = FastICA(n_components=3, random_state=0)
S_ica = ica.fit_transform(X)  # Reconstruct signals
A_ica = ica.mixing_  # Estimated mixing matrix

plt.figure(figsize=(9, 6))
plt.subplot(2, 1, 1)
for sig in S.T:
    plt.plot(sig)
plt.title('True Sources')
plt.subplot(2, 1, 2)
for sig in S_ica.T:
    plt.plot(sig)
plt.title('Recovered Sources (FastICA)')
plt.tight_layout()
plt.show() 
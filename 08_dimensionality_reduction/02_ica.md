# Independent components analysis

## Introduction and Motivation

Independent Components Analysis (ICA) is a powerful technique for separating a set of mixed signals into their original, independent sources. While Principal Components Analysis (PCA) finds new axes that maximize variance and decorrelate the data, ICA goes further: it tries to find components that are statistically independent, not just uncorrelated. This makes ICA especially useful for problems where the observed data is a mixture of underlying signals, and we want to recover those original signals.

### The Cocktail Party Problem: An Intuitive Example
Imagine you are at a lively cocktail party. There are several people (say, $` d `$ speakers) all talking at once, and you place $` d `$ microphones around the room. Each microphone records a different mixture of all the voices, depending on how close it is to each speaker. The challenge: can you take the recordings from the microphones and separate out each individual speaker's voice?

This is not just a party trick! Similar problems arise in:
- **Brain imaging:** EEG/MEG sensors record mixtures of brain signals.
- **Finance:** Observed prices are mixtures of underlying market factors.
- **Image processing:** Pixels may be mixtures of different sources of light.

ICA provides a mathematical framework to solve these kinds of problems, where the observed data is a mixture of independent sources.

---

### Python: Simulating the Cocktail Party Problem
```python
import numpy as np
import matplotlib.pyplot as plt
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
```

### Python: Visualizing the Sources and Mixtures
```python
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
```

---

## Whitening/Preprocessing

Before applying ICA, it is common to preprocess the data by whitening (decorrelating and scaling) the mixtures. This step is not strictly necessary for all ICA algorithms, but it often improves convergence and interpretability.

### Python: Whitening the Mixtures (PCA Preprocessing)
```python
X_centered = X - X.mean(axis=0)
cov = np.cov(X_centered, rowvar=False)
D, E = np.linalg.eigh(cov)
D = np.diag(1.0 / np.sqrt(D))
X_white = (X_centered @ E) @ D
print("Whitened data shape:", X_white.shape)
```

---

## ICA Algorithm (Gradient Ascent for W, simplified)

The following is a simplified version of the ICA algorithm using gradient ascent. For practical use, see the FastICA implementation below.

### Python: ICA via Gradient Ascent (Simplified)
```python
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
    # from scipy.linalg import sqrtm
    # W = np.linalg.inv(sqrtm(W @ W.T)) @ W

S_est = X_white @ W.T
```

### Python: Visualizing the Recovered Sources
```python
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
```

---

## Using scikit-learn's FastICA

For practical ICA, use the FastICA implementation from scikit-learn, which is robust and efficient.

### Python: ICA using scikit-learn's FastICA
```python
from sklearn.decomposition import FastICA
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
```

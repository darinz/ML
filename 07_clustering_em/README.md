# Clustering and Expectation-Maximization (EM)

[![Clustering](https://img.shields.io/badge/Clustering-K-means%20%26%20GMM-blue.svg)](https://en.wikipedia.org/wiki/Cluster_analysis)
[![EM](https://img.shields.io/badge/EM-Expectation%20Maximization-green.svg)](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm)
[![VAE](https://img.shields.io/badge/VAE-Variational%20Autoencoder-purple.svg)](https://en.wikipedia.org/wiki/Variational_autoencoder)

Comprehensive implementations of clustering algorithms and EM algorithm, covering k-means, Gaussian Mixture Models, and Variational Auto-Encoders.

## Overview

Materials cover unsupervised learning techniques for discovering patterns and structure in data without labels.

## Materials

### Theory
- **[01_clustering.md](01_clustering.md)** - K-means clustering algorithm and concepts
- **[02_em_mixture_of_gaussians.md](02_em_mixture_of_gaussians.md)** - GMM and EM algorithm details
- **[03_general_em.md](03_general_em.md)** - General EM framework and variational inference
- **[04_variational_auto-encoder.md](04_variational_auto-encoder.md)** - VAE concepts and implementation
- **[05_hands-on_coding.md](05_hands-on_coding.md)** - Practical implementation guide

### Implementation
- **[kmeans_examples.py](kmeans_examples.py)** - Complete k-means implementation with examples
- **[em_mog_examples.py](em_mog_examples.py)** - EM for Gaussian Mixture Models
- **[general_em_examples.py](general_em_examples.py)** - General EM framework with multiple models
- **[variational_auto_encoder_examples.py](variational_auto_encoder_examples.py)** - Complete VAE implementation

### Supporting Files
- **requirements.txt** - Python dependencies
- **environment.yaml** - Conda environment setup
- **img/** - Images and visualizations

## Key Concepts

### K-Means Clustering
**Objective**: Minimize distortion $J = \sum_{i=1}^n ||x_i - \mu_{c_i}||^2$

**Algorithm**:
1. Initialize k centroids
2. Assign points to nearest centroid
3. Update centroids as means
4. Repeat until convergence

### Gaussian Mixture Models (GMM)
**Generative Model**: $p(x) = \sum_{j=1}^k \pi_j \mathcal{N}(x|\mu_j, \Sigma_j)$

**EM Algorithm**:
- **E-step**: Compute responsibilities $\gamma(z_{ij}) = p(z_i = j|x_i, \theta)$
- **M-step**: Update parameters given responsibilities

### General EM Framework
**ELBO**: $\mathcal{L}(\theta) = \mathbb{E}_{Z\sim Q}[\log p(X,Z|\theta)] - \mathbb{E}_{Z\sim Q}[\log Q(Z)]$

**Algorithm**:
1. **E-step**: $Q(Z) = \arg\max_Q$ ELBO
2. **M-step**: $\theta = \arg\max_\theta \mathbb{E}_{Z\sim Q}[\log p(X,Z|\theta)]$

### Variational Auto-Encoders (VAE)
**Generative Model**: $p(x,z) = p(z) \cdot p(x|z; \theta)$

**Approximate Posterior**: $q(z|x; \phi) \approx p(z|x; \theta)$

**ELBO**: $\mathcal{L}(\theta,\phi) = \mathbb{E}_{z\sim q}[\log p(x|z;\theta)] - \text{KL}(q(z|x;\phi) || p(z))$

## Applications

- **Data Exploration**: Discovering natural groupings in data
- **Dimensionality Reduction**: Feature learning and representation
- **Anomaly Detection**: Identifying unusual patterns
- **Image Segmentation**: Grouping similar pixels
- **Document Clustering**: Organizing text collections
- **Generative Modeling**: Creating new data samples

## Getting Started

1. Read `01_clustering.md` for k-means fundamentals
2. Study `02_em_mixture_of_gaussians.md` for GMM and EM
3. Learn `03_general_em.md` for general framework
4. Explore `04_variational_auto-encoder.md` for VAE concepts
5. Use `05_hands-on_coding.md` for practical implementation
6. Run Python examples to see algorithms in action

## Prerequisites

- Basic probability and statistics
- Linear algebra fundamentals
- Python programming and NumPy
- Understanding of optimization concepts

## Installation

```bash
pip install -r requirements.txt
```

Or use conda:
```bash
conda env create -f environment.yaml
```

## Running Examples

```bash
python kmeans_examples.py
python em_mog_examples.py
python general_em_examples.py
python variational_auto_encoder_examples.py
```

## Quick Start Code

```python
# K-Means
from kmeans_examples import kmeans
centroids, labels, distortion_history, n_iters = kmeans(X, k=3)

# GMM with EM
from em_mog_examples import em_mog
phi, mu, Sigma, w, logliks, elbos = em_mog(X, k=3)

# General EM
from general_em_examples import general_em, gmm_initialize, gmm_e_step, gmm_m_step
init_params = gmm_initialize(X, k=3)
params, Q, elbos, n_iters = general_em(X, init_params, gmm_e_step, gmm_m_step)

# VAE
from variational_auto_encoder_examples import VAE
vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
vae.fit(X_train)
samples = vae.sample(n_samples=10)
```

## Method Comparison

| Method | Assignment | Cluster Shape | Initialization Sensitivity | Complexity |
|--------|------------|---------------|---------------------------|------------|
| K-Means | Hard | Spherical | High | Low |
| GMM | Soft | Elliptical | Medium | Medium |
| VAE | Continuous | Arbitrary | Low | High |
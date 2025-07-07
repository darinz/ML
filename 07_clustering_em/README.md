# Clustering and Expectation-Maximization (EM) Algorithms

This section covers fundamental unsupervised learning algorithms for clustering and probabilistic modeling, including k-means clustering, mixture of Gaussians, the general EM algorithm, and variational auto-encoders.

## Overview

Clustering and EM algorithms are essential tools in unsupervised learning for discovering patterns and structure in data without labeled examples. These methods range from simple distance-based clustering to sophisticated probabilistic models with latent variables.

## Table of Contents

1. [K-means Clustering](#k-means-clustering)
2. [EM for Mixture of Gaussians](#em-for-mixture-of-gaussians)
3. [General EM Algorithm](#general-em-algorithm)
4. [Variational Auto-Encoders](#variational-auto-encoders)

---

## K-means Clustering

**File:** `01_clustering.md` | **Implementation:** `kmeans_examples.py`

K-means is one of the most popular and simple clustering algorithms. It partitions data into k clusters by minimizing the within-cluster sum of squares.

### Key Concepts

- **Objective Function**: Minimize $\sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$
- **Algorithm**: Iterative assignment and update steps
- **Convergence**: Guaranteed to converge to a local minimum
- **Initialization**: Sensitive to initial cluster centers

### Algorithm Steps

1. **Initialize**: Choose k cluster centers randomly
2. **Assign**: Assign each point to nearest center
3. **Update**: Recompute cluster centers as means
4. **Repeat**: Until convergence

### Mathematical Formulation

For data points $x_1, x_2, \ldots, x_n$ and k clusters:

```math
\min_{C_1, \ldots, C_k, \mu_1, \ldots, \mu_k} \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
```

where $C_i$ are the clusters and $\mu_i$ are the cluster centers.

### Implementation Features

- K-means algorithm with multiple initialization strategies
- Visualization of clustering results
- Convergence analysis
- Performance metrics (inertia, silhouette score)

---

## EM for Mixture of Gaussians

**File:** `02_em_mixture_of_gaussians.md` | **Implementation:** `em_mog_examples.py`

The EM algorithm for Gaussian Mixture Models (GMM) is a probabilistic approach to clustering that models data as a mixture of several Gaussian distributions.

### Key Concepts

- **Generative Model**: Data generated from mixture of Gaussians
- **Latent Variables**: Cluster assignments for each data point
- **E-step**: Compute posterior probabilities (responsibilities)
- **M-step**: Update model parameters (means, covariances, mixing weights)

### Mathematical Formulation

**Generative Model:**
```math
p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x; \mu_k, \Sigma_k)
```

**E-step (Expectation):**
```math
\gamma_{nk} = \frac{\pi_k \mathcal{N}(x_n; \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n; \mu_j, \Sigma_j)}
```

**M-step (Maximization):**
```math
\mu_k^{new} = \frac{\sum_{n=1}^{N} \gamma_{nk} x_n}{\sum_{n=1}^{N} \gamma_{nk}}
```

### Implementation Features

- Complete EM algorithm for GMM
- Parameter initialization strategies
- Convergence monitoring
- Visualization of mixture components
- Model selection (number of components)

---

## General EM Algorithm

**File:** `03_general_em.md` | **Implementation:** `general_em_examples.py`

The general EM algorithm provides a framework for maximum likelihood estimation in the presence of latent variables, extending beyond mixture models to any probabilistic model with hidden variables.

### Key Concepts

- **Evidence Lower BOund (ELBO)**: Lower bound on log-likelihood
- **Jensen's Inequality**: Foundation for ELBO derivation
- **Alternating Maximization**: E-step and M-step optimization
- **Variational Inference**: Connection to modern probabilistic methods

### Mathematical Formulation

**ELBO:**
```math
\mathcal{L}(q, \theta) = \mathbb{E}_{z \sim q}[\log p(x, z; \theta)] - \mathbb{E}_{z \sim q}[\log q(z)]
```

**E-step:**
```math
q^{new}(z) = \arg\max_q \mathcal{L}(q, \theta^{old})
```

**M-step:**
```math
\theta^{new} = \arg\max_\theta \mathcal{L}(q^{new}, \theta)
```

### Implementation Features

- General EM framework
- ELBO computation and monitoring
- Flexible model specification
- Convergence analysis
- Example with GMM using general framework

---

## Variational Auto-Encoders

**File:** `04_variational_auto-encoder.md` | **Implementation:** `variational_auto_encoder_examples.py`

Variational Auto-Encoders (VAEs) extend EM algorithms to complex models parameterized by neural networks, enabling deep generative modeling with continuous latent variables.

### Key Concepts

- **Neural Network Parameterization**: Complex generative models
- **Posterior Intractability**: Why variational approximation is needed
- **Mean Field Assumption**: Independent latent variables
- **Reparameterization Trick**: Enabling gradient-based optimization
- **Encoder-Decoder Architecture**: Neural network implementation

### Mathematical Formulation

**Generative Model:**
```math
z \sim \mathcal{N}(0, I), \quad x|z \sim \mathcal{N}(g(z; \theta), \sigma^2 I)
```

**Approximate Posterior:**
```math
Q(z|x) = \mathcal{N}(\mu(x; \phi), \sigma^2(x; \psi))
```

**Reparameterization Trick:**
```math
z = \mu(x; \phi) + \sigma(x; \psi) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
```

**ELBO:**
```math
\mathcal{L} = \mathbb{E}_{z \sim Q}[\log p(x|z)] - \text{KL}(Q(z|x) \| p(z))
```

### Implementation Features

- Complete VAE implementation with PyTorch
- Encoder and decoder networks
- Reparameterization trick implementation
- ELBO computation and optimization
- Sample generation and visualization
- Training framework with monitoring

---

## Mathematical Background

### Probability and Statistics

- **Gaussian Distribution**: $\mathcal{N}(x; \mu, \Sigma)$
- **KL Divergence**: $\text{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$
- **Jensen's Inequality**: $f(\mathbb{E}[X]) \geq \mathbb{E}[f(X)]$ for concave $f$

### Optimization

- **Gradient Descent**: Parameter updates via gradients
- **Alternating Optimization**: Coordinate-wise optimization
- **Convergence**: Local optimality guarantees

### Information Theory

- **Entropy**: $H(X) = -\sum p(x) \log p(x)$
- **Mutual Information**: $I(X; Y) = H(X) - H(X|Y)$
- **Evidence Lower BOund**: Connection to variational inference

---

## Implementation Details

### Dependencies

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
```

### File Structure

```
07_clustering_em/
├── README.md                           # This file
├── 01_clustering.md                    # K-means theory
├── kmeans_examples.py                  # K-means implementation
├── 02_em_mixture_of_gaussians.md       # EM for GMM theory
├── em_mog_examples.py                  # EM for GMM implementation
├── 03_general_em.md                    # General EM theory
├── general_em_examples.py              # General EM implementation
├── 04_variational_auto-encoder.md      # VAE theory
├── variational_auto_encoder_examples.py # VAE implementation
└── img/                                # Images and figures
```

### Running Examples

Each implementation file can be run independently:

```bash
# K-means clustering
python kmeans_examples.py

# EM for mixture of Gaussians
python em_mog_examples.py

# General EM algorithm
python general_em_examples.py

# Variational auto-encoder
python variational_auto_encoder_examples.py
```

---

## Applications

### Clustering Applications

- **Customer Segmentation**: Group customers by behavior patterns
- **Image Segmentation**: Partition images into regions
- **Document Clustering**: Organize documents by topic
- **Anomaly Detection**: Identify unusual data points

### Generative Modeling Applications

- **Data Generation**: Create synthetic data samples
- **Dimensionality Reduction**: Learn low-dimensional representations
- **Feature Learning**: Discover meaningful features
- **Data Compression**: Efficient data encoding

### Real-world Examples

- **Recommendation Systems**: User clustering for recommendations
- **Computer Vision**: Image generation and manipulation
- **Natural Language Processing**: Topic modeling and text generation
- **Bioinformatics**: Gene expression clustering

---

## Advanced Topics

### Model Selection

- **Number of Clusters**: Elbow method, silhouette analysis
- **Model Complexity**: AIC, BIC, cross-validation
- **Hyperparameter Tuning**: Learning rates, network architectures

### Scalability

- **Mini-batch Processing**: Large-scale data handling
- **Parallelization**: Multi-core and GPU acceleration
- **Online Learning**: Streaming data adaptation

### Extensions

- **Hierarchical Clustering**: Multi-level clustering
- **Spectral Clustering**: Graph-based clustering
- **Deep Clustering**: Neural network-based clustering
- **Variational Inference**: Modern probabilistic methods

---

## References

### Key Papers

1. **K-means**: MacQueen, J. (1967). "Some Methods for Classification and Analysis of Multivariate Observations"
2. **EM Algorithm**: Dempster, A.P., et al. (1977). "Maximum Likelihood from Incomplete Data via the EM Algorithm"
3. **Variational Auto-Encoders**: Kingma, D.P. & Welling, M. (2013). "Auto-Encoding Variational Bayes"
4. **Variational Inference**: Blei, D.M., et al. (2017). "Variational Inference: A Review for Statisticians"

### Textbooks

- Bishop, C.M. (2006). "Pattern Recognition and Machine Learning"
- Murphy, K.P. (2012). "Machine Learning: A Probabilistic Perspective"
- Goodfellow, I., et al. (2016). "Deep Learning"

### Online Resources

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [PyTorch Distributions](https://pytorch.org/docs/stable/distributions.html)
- [VAE Tutorial](https://arxiv.org/abs/1606.05908)

---

## Contributing

This section is part of a comprehensive machine learning curriculum. Contributions are welcome for:

- Additional algorithms and implementations
- Improved visualizations and examples
- Mathematical derivations and proofs
- Real-world applications and case studies
- Performance optimizations and scalability improvements
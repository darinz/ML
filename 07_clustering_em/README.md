# Clustering and Expectation-Maximization (EM)

This directory contains comprehensive implementations and examples of clustering algorithms and the Expectation-Maximization (EM) algorithm, with detailed explanations of the underlying concepts.

## Overview

The materials in this directory cover:

1. **K-Means Clustering** - A simple but effective clustering algorithm
2. **Gaussian Mixture Models (GMM)** - Probabilistic clustering using EM
3. **General EM Framework** - A flexible implementation for various latent variable models
4. **Variational Auto-Encoders (VAE)** - Deep generative models using variational inference

## Files

### Documentation
- `01_clustering.md` - Comprehensive explanation of k-means clustering
- `02_em_mixture_of_gaussians.md` - Detailed coverage of GMM and EM algorithm
- `03_general_em.md` - General EM framework and variational inference
- `04_variational_auto-encoder.md` - VAE concepts and implementation

### Python Implementations
- `kmeans_examples.py` - Complete k-means implementation with examples
- `em_mog_examples.py` - EM for Gaussian Mixture Models
- `general_em_examples.py` - General EM framework with multiple models
- `variational_auto_encoder_examples.py` - Complete VAE implementation

### Supporting Files
- `requirements.txt` - Python dependencies
- `img/` - Images and visualizations

## Key Concepts

### 1. K-Means Clustering

**Objective**: Minimize the distortion (sum of squared distances)
```
J = Σᵢ₌₁ⁿ ||xᵢ - μ_{cᵢ}||²
```

**Algorithm**:
1. **Initialization**: Choose k initial centroids
2. **Assignment**: Assign each point to nearest centroid
3. **Update**: Recompute centroids as means of assigned points
4. **Repeat** until convergence

**Key Features**:
- Simple and fast
- Sensitive to initialization
- Assumes spherical clusters
- Hard assignments (each point belongs to one cluster)

### 2. Gaussian Mixture Models (GMM)

**Generative Model**:
```
p(x) = Σⱼ₌₁ᵏ πⱼ N(x|μⱼ, Σⱼ)
```

**EM Algorithm**:
- **E-step**: Compute responsibilities γ(zᵢⱼ) = p(zᵢ = j|xᵢ, θ)
- **M-step**: Update parameters given responsibilities

**Key Features**:
- Probabilistic clustering
- Soft assignments (each point has probability of belonging to each cluster)
- Can model elliptical clusters
- More flexible than k-means

### 3. General EM Framework

**ELBO (Evidence Lower BOund)**:
```
L(θ) = E_{Z~Q}[log p(X,Z|θ)] - E_{Z~Q}[log Q(Z)]
```

**Algorithm**:
1. **E-step**: Q(Z) = argmax_Q ELBO
2. **M-step**: θ = argmax_θ E_{Z~Q}[log p(X,Z|θ)]

**Key Features**:
- General framework for latent variable models
- Can be applied to any model with appropriate E-step and M-step functions
- Based on variational inference principles

### 4. Variational Auto-Encoders (VAE)

**Generative Model**:
```
p(x,z) = p(z) * p(x|z; θ)
```

**Approximate Posterior**:
```
q(z|x; φ) ≈ p(z|x; θ)
```

**ELBO**:
```
L(θ,φ) = E_{z~q}[log p(x|z;θ)] - KL(q(z|x;φ) || p(z))
```

**Key Features**:
- Deep generative model
- Neural network encoder and decoder
- Reparameterization trick for gradient flow
- Can generate new samples

## Usage Examples

### Running K-Means

```python
from kmeans_examples import kmeans, demonstrate_kmeans_concepts

# Run k-means on synthetic data
X, labels, centroids, metrics = demonstrate_kmeans_concepts()

# Or run manually
centroids, labels, distortion_history, n_iters = kmeans(X, k=3, verbose=True)
```

### Running GMM with EM

```python
from em_mog_examples import em_mog, demonstrate_gmm_with_general_em

# Run GMM on synthetic data
phi, mu, Sigma, w, logliks, elbos = em_mog(X, k=3, verbose=True)

# Or use the demonstration
gmm_params, gmm_w, gmm_elbos = demonstrate_gmm_with_general_em()
```

### Running General EM

```python
from general_em_examples import general_em, gmm_initialize, gmm_e_step, gmm_m_step, gmm_elbo

# Initialize parameters
init_params = gmm_initialize(X, k=3)

# Run general EM
params, Q, elbos, n_iters = general_em(
    X, init_params, gmm_e_step, gmm_m_step, gmm_elbo, verbose=True
)
```

### Running VAE

```python
from variational_auto_encoder_examples import VAE, demonstrate_vae_training

# Train VAE on data
vae = demonstrate_vae_training()

# Generate samples
samples = vae.sample(n_samples=10)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run examples:
```bash
# K-means
python kmeans_examples.py

# GMM with EM
python em_mog_examples.py

# General EM
python general_em_examples.py

# VAE
python variational_auto_encoder_examples.py
```

## Key Insights

### Comparison of Methods

| Method | Assignment Type | Cluster Shape | Initialization Sensitivity | Complexity |
|--------|----------------|---------------|---------------------------|------------|
| K-Means | Hard | Spherical | High | Low |
| GMM | Soft | Elliptical | Medium | Medium |
| VAE | Continuous | Arbitrary | Low | High |

### When to Use Each Method

- **K-Means**: Simple clustering, spherical clusters, fast computation
- **GMM**: Probabilistic clustering, elliptical clusters, uncertainty quantification
- **General EM**: Custom latent variable models, flexible framework
- **VAE**: Complex data, generative modeling, deep learning applications

### Practical Tips

1. **Initialization**: Use k-means++ for better starting points
2. **Multiple Runs**: Run algorithms multiple times to avoid local optima
3. **Model Selection**: Use cross-validation or information criteria for choosing k
4. **Evaluation**: Use multiple metrics (silhouette score, log-likelihood, etc.)
5. **Visualization**: Always visualize results to understand cluster structure

## Advanced Topics

### Convergence Properties

- **K-Means**: Guaranteed to converge but may reach local optima
- **EM**: Monotonic increase in log-likelihood, may reach local optima
- **VAE**: Gradient-based optimization, may get stuck in local optima

### Theoretical Foundations

- **K-Means**: Lloyd's algorithm, coordinate descent
- **EM**: Jensen's inequality, coordinate ascent on ELBO
- **VAE**: Variational inference, reparameterization trick

### Extensions

- **K-Means**: K-means++, fuzzy c-means, spectral clustering
- **GMM**: Bayesian GMM, infinite GMM, hierarchical GMM
- **EM**: Stochastic EM, online EM, variational EM
- **VAE**: β-VAE, conditional VAE, adversarial VAE

## References

1. Bishop, C. M. (2006). Pattern Recognition and Machine Learning
2. Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective
3. Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes
4. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational Inference: A Review for Statisticians

## Contributing

Feel free to contribute improvements, bug fixes, or additional examples. Please ensure that:

1. Code follows the existing style and structure
2. New functions are well-documented
3. Examples are clear and educational
4. Mathematical formulations are accurate
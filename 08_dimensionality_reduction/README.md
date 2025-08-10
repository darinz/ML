# Dimensionality Reduction

[![PCA](https://img.shields.io/badge/PCA-Principal%20Component%20Analysis-blue.svg)](https://en.wikipedia.org/wiki/Principal_component_analysis)
[![ICA](https://img.shields.io/badge/ICA-Independent%20Component%20Analysis-green.svg)](https://en.wikipedia.org/wiki/Independent_component_analysis)
[![Python](https://img.shields.io/badge/Python-3.7+-yellow.svg)](https://python.org)

Techniques to reduce the number of variables in datasets while preserving relevant information, covering PCA and ICA with comprehensive implementations.

## Overview

Dimensionality reduction helps visualize, interpret, and process high-dimensional data by removing noise, redundancy, and revealing hidden structure.

## Materials

### Theory
- **[01_pca.md](01_pca.md)** - Principal Component Analysis: mathematical foundations, geometric intuition, applications
- **[02_ica.md](02_ica.md)** - Independent Component Analysis: statistical independence, source separation, algorithms
- **[03_hands-on_coding.md](03_hands-on_coding.md)** - Practical implementation guide

### Implementation
- **[code/pca_examples.py](code/pca_examples.py)** - Complete PCA implementation with 10 comprehensive sections
- **[code/ica_examples.py](code/ica_examples.py)** - Complete ICA implementation with 10 detailed sections

### Supporting Files
- **code/requirements.txt** - Python dependencies
- **code/environment.yaml** - Conda environment setup
- **img/** - Figures and diagrams

## Key Concepts

### Principal Components Analysis (PCA)
**Objective**: Find directions of maximum variance in data

**Mathematical Foundation**: 
- Eigenvalue decomposition of covariance matrix
- Principal components are eigenvectors
- Explained variance from eigenvalues

**Algorithm**:
1. Center and scale data
2. Compute covariance matrix
3. Find eigenvectors and eigenvalues
4. Project data onto principal components

### Independent Components Analysis (ICA)
**Objective**: Find statistically independent components

**Linear Mixing Model**: $x = As$ where $s$ are independent sources

**ICA Ambiguities**:
- Permutation: Order of components unknown
- Scaling: Magnitude of components unknown
- Sign: Direction of components unknown

**Algorithm**:
1. Center and whiten data
2. Initialize unmixing matrix
3. Update using gradient ascent or FastICA
4. Extract independent components

## Applications

- **Data Visualization**: Reducing to 2D/3D for plotting
- **Noise Reduction**: Removing irrelevant dimensions
- **Feature Extraction**: Creating new representations
- **Compression**: Reducing storage requirements
- **Source Separation**: Separating mixed signals (ICA)

## Getting Started

1. Read `01_pca.md` for PCA fundamentals
2. Study `02_ica.md` for ICA concepts
3. Use `03_hands-on_coding.md` for practical guidance
4. Run Python examples to see algorithms in action

## Prerequisites

- Linear algebra fundamentals
- Basic probability and statistics
- Python programming and NumPy
- Understanding of optimization concepts

## Installation

```bash
pip install -r code/requirements.txt
```

Or use conda:
```bash
conda env create -f code/environment.yaml
```

## Running Examples

```bash
python code/pca_examples.py
python code/ica_examples.py
```

## Quick Start Code

```python
# PCA
from code.pca_examples import pca_manual
X_reduced, components, explained_variance = pca_manual(X, n_components=2)

# ICA
from code.ica_examples import ica_gradient_ascent
unmixing_matrix, independent_components = ica_gradient_ascent(X, n_components=3)

# Using scikit-learn
from sklearn.decomposition import PCA, FastICA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

ica = FastICA(n_components=3)
X_ica = ica.fit_transform(X)
```

## Method Comparison

| Method | Goal | Assumptions | Applications |
|--------|------|-------------|--------------|
| PCA | Maximize variance | Linear relationships | Visualization, compression |
| ICA | Find independence | Non-Gaussian sources | Source separation, feature extraction | 
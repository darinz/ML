# Generative Learning Algorithms

[![Generative Models](https://img.shields.io/badge/Generative-GDA%20%26%20Naive%20Bayes-blue.svg)](https://en.wikipedia.org/wiki/Generative_model)
[![Bayes Rule](https://img.shields.io/badge/Bayes-Rule-green.svg)](https://en.wikipedia.org/wiki/Bayes%27_theorem)
[![Classification](https://img.shields.io/badge/Classification-Binary%20%26%20Multi-class-purple.svg)](https://en.wikipedia.org/wiki/Statistical_classification)

Generative learning algorithms model the joint distribution $p(x,y)$ by learning $p(x|y)$ and $p(y)$, then use Bayes' rule to compute $p(y|x)$ for classification.

## Overview

**Generative vs. Discriminative Models:**
- **Generative**: Learn data generation process ($p(x|y)$ and $p(y)$)
- **Discriminative**: Learn decision boundary directly ($p(y|x)$)

**Key Advantage:** Generative models can generate new data and work well with limited training data.

## Materials

### Theory
- **[01_gda.md](01_gda.md)** - Gaussian Discriminant Analysis
  - Multivariate normal distribution properties
  - GDA model assumptions and parameter estimation
  - Linear decision boundaries and relationship to logistic regression

- **[02_naive_bayes.md](02_naive_bayes.md)** - Naive Bayes Classification
  - Discrete feature modeling with independence assumption
  - Bernoulli and multinomial variants
  - Laplace smoothing and text classification applications

### Implementation
- **[code/gda_examples.py](code/gda_examples.py)** - GDA implementation with parameter estimation, prediction, and visualization
- **[code/naive_bayes_examples.py](code/naive_bayes_examples.py)** - Naive Bayes with text preprocessing and feature importance

### Visualizations
- **img/** - Gaussian contours, decision boundaries, and parameter estimation results

## Key Concepts

### Bayes' Rule
```math
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
```

### GDA Model (Continuous Features)
```math
y \sim \text{Bernoulli}(\phi)
x|y=0 \sim \mathcal{N}(\mu_0, \Sigma)
x|y=1 \sim \mathcal{N}(\mu_1, \Sigma)
```

**Key Assumptions:**
- Gaussian class-conditionals with shared covariance
- Linear decision boundaries

### Naive Bayes (Discrete Features)
```math
p(x_1, \ldots, x_d|y) = \prod_{j=1}^d p(x_j|y)
```

**Key Assumptions:**
- Conditional independence of features given class
- Discrete features with finite values

## Applications

### GDA
- **Use when**: Continuous features, limited data, interpretable parameters needed
- **Examples**: Medical diagnosis, financial modeling, quality control
- **Strengths**: Interpretable, data efficient, can generate synthetic data
- **Limitations**: Assumes normal distribution, shared covariance constraint

### Naive Bayes
- **Use when**: High-dimensional discrete data, text classification, sparse features
- **Examples**: Spam filtering, sentiment analysis, document classification
- **Strengths**: Fast, scalable, handles missing data, works with little training data
- **Limitations**: Independence assumption often violated, can't capture feature interactions

## Getting Started

1. Read `01_gda.md` for continuous feature modeling
2. Study `02_naive_bayes.md` for discrete feature classification
3. Run examples in Python files to see algorithms in action
4. Explore visualizations in `img/` folder

## Prerequisites

- Probability and statistics basics
- Linear algebra fundamentals
- Python programming
- Understanding of classification concepts 
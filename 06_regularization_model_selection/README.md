# Regularization, Model Selection, and Bayesian Methods

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![NumPy](https://img.shields.io/badge/numpy-%3E=1.18-blue.svg)](https://numpy.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-%3E=0.22-orange.svg)](https://scikit-learn.org/stable/)

Comprehensive guide covering regularization, model selection, cross-validation, and Bayesian statistics in machine learning for building robust, generalizable models.

## Overview

**Why These Topics Matter:**
- **Regularization** prevents overfitting and improves generalization
- **Model Selection** helps choose the right model complexity
- **Cross-Validation** provides reliable performance estimates
- **Bayesian Methods** incorporate uncertainty and prior knowledge

## Materials

### Theory
- **[01_regularization.md](01_regularization.md)** - L1/L2 regularization, elastic net, implicit regularization, practical guidelines
- **[02_model_selection.md](02_model_selection.md)** - Cross-validation methods, Bayesian statistics (MLE, MAP), model comparison

### Implementation
- **[regularization_examples.py](regularization_examples.py)** - L2/L1 regularization, elastic net, coefficient paths, performance curves
- **[model_selection_and_bayes_examples.py](model_selection_and_bayes_examples.py)** - Cross-validation, MLE/MAP estimation, Bayesian inference

### Visualizations
- **img/** - Regularization coefficient paths, bias-variance tradeoff, uncertainty analysis

## Key Concepts

### Regularization Framework
```math
J_\lambda(\theta) = J(\theta) + \lambda R(\theta)
```

**Types:**
- **L2 (Ridge)**: $R(\theta) = \sum_j \theta_j^2$ - Weight decay
- **L1 (LASSO)**: $R(\theta) = \sum_j |\theta_j|$ - Sparsity
- **Elastic Net**: $R(\theta) = \alpha \sum_j |\theta_j| + (1-\alpha) \sum_j \theta_j^2$

### Cross-Validation Strategies
| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| **Hold-out** | Large datasets | Fast, simple | Wastes data |
| **k-Fold** | Medium datasets | Good balance | Computationally expensive |
| **Leave-One-Out** | Small datasets | Most unbiased | Very slow |

### Bayesian vs Frequentist
| Approach | Parameters | Prior | Output | Overfitting Control |
|----------|------------|-------|--------|-------------------|
| **MLE** | Fixed, unknown | None | Best fit | None |
| **MAP** | Fixed, unknown | Yes | Best fit + prior | Regularization |
| **Bayesian** | Random variable | Yes | Average over posterior | Regularization + Uncertainty |

## Applications

- **Overfitting Prevention**: Regularization techniques
- **Feature Selection**: L1 regularization for sparsity
- **Model Comparison**: Cross-validation and Bayesian methods
- **Uncertainty Quantification**: Bayesian inference
- **Hyperparameter Tuning**: Systematic parameter selection

## Getting Started

1. Read `01_regularization.md` for regularization fundamentals
2. Study `02_model_selection.md` for selection strategies
3. Run Python examples to see techniques in action
4. Explore visualizations in `img/` folder

## Prerequisites

- Basic machine learning concepts
- Linear algebra fundamentals
- Probability and statistics basics
- Python programming experience

## Installation

```bash
pip install numpy scikit-learn matplotlib seaborn
```

## Running Examples

```bash
python regularization_examples.py
python model_selection_and_bayes_examples.py
```

## Quick Start Code

```python
# L2 Regularization (Ridge)
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X, y, cv=5)

# L1 Regularization (LASSO)
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)

# Bayesian Ridge
from sklearn.linear_model import BayesianRidge
bayesian_ridge = BayesianRidge()
y_pred, y_std = bayesian_ridge.predict(X_new, return_std=True)
``` 
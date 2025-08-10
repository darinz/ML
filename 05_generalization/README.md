# Generalization in Machine Learning

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Educational](https://img.shields.io/badge/purpose-educational-informational)](https://en.wikipedia.org/wiki/Education)

Comprehensive materials on generalization theory and practice in machine learning, covering bias-variance tradeoff, double descent phenomena, and theoretical complexity bounds.

## Overview

Understanding how and why machine learning models generalize to new data, and what factors influence their performance on unseen examples.

## Materials

### Theory
- **[01_bias-variance_tradeoﬀ.md](01_bias-variance_tradeoﬀ.md)** - MSE decomposition, bias-variance tradeoff, model selection strategies
- **[02_double_descent.md](02_double_descent.md)** - Classical vs modern regimes, interpolation threshold, implicit regularization
- **[03_complexity_bounds.md](03_complexity_bounds.md)** - Concentration inequalities, VC dimension, sample complexity bounds

### Implementation
- **[bias_variance_decomposition_examples.py](bias_variance_decomposition_examples.py)** - Underfitting vs overfitting, bias-variance decomposition, polynomial regression
- **[double_descent_examples.py](double_descent_examples.py)** - Model-wise and sample-wise double descent, regularization effects
- **[complexity_bounds_examples.py](complexity_bounds_examples.py)** - Hoeffding bounds, VC dimension visualization, learning curves

### Reference Materials
- **cs229-notes4_bias-variance.pdf** - CS229 comprehensive theoretical coverage
- **cs229-evaluation_metrics_slides.pdf** - Evaluation framework and bias-variance analysis

### Visualizations
- **img/** - Bias-variance tradeoff plots, double descent curves, VC dimension examples, learning curves

## Key Concepts

### Bias-Variance Tradeoff
```math
\text{MSE} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
```

**Components:**
- **Bias**: $E[h_S(x)] - h^*(x)$ - Systematic error from model assumptions
- **Variance**: $E[(h_S(x) - E[h_S(x)])^2]$ - Variability due to training data

### Double Descent Phenomenon
- **Classical regime**: Bias-variance tradeoff (U-shaped curve)
- **Interpolation threshold**: $n \approx d$ (samples ≈ parameters)
- **Modern regime**: Overparameterized models ($d > n$)

### Sample Complexity Bounds
- **Hoeffding bound**: $P(|\bar{X} - \mu| > \gamma) \leq 2\exp(-2\gamma^2 n)$
- **Sample complexity**: $n \geq \frac{1}{2\gamma^2} \log(\frac{2k}{\delta})$
- **VC dimension**: Linear classifiers in 2D have VC dimension = 3

## Applications

- **Model Selection**: Choosing optimal model complexity
- **Regularization**: Balancing fit and generalization
- **Deep Learning**: Understanding overparameterized models
- **Data Requirements**: Estimating necessary training data size

## Getting Started

1. Read `01_bias-variance_tradeoﬀ.md` for fundamental tradeoffs
2. Study `02_double_descent.md` for modern ML phenomena
3. Learn `03_complexity_bounds.md` for theoretical foundations
4. Run Python examples to see concepts in action
5. Explore visualizations in `img/` folder

## Prerequisites

- Basic probability and statistics
- Linear algebra fundamentals
- Python programming and NumPy
- Understanding of machine learning models

## Installation

```bash
pip install numpy matplotlib scipy scikit-learn
```

## Running Examples

```bash
python bias_variance_decomposition_examples.py
python double_descent_examples.py
python complexity_bounds_examples.py
``` 
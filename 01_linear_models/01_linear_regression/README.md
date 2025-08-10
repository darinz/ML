# Linear Regression

[![Regression](https://img.shields.io/badge/Regression-Linear%20Regression-blue.svg)](https://en.wikipedia.org/wiki/Linear_regression)
[![Least Squares](https://img.shields.io/badge/Least%20Squares-Optimization-green.svg)](https://en.wikipedia.org/wiki/Least_squares)
[![Mathematics](https://img.shields.io/badge/Mathematics-Linear%20Algebra-purple.svg)](https://en.wikipedia.org/wiki/Linear_algebra)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)

## Overview

Linear regression fundamentals including least squares, gradient descent, normal equations, and locally weighted regression.

## Materials

### Core Notes
- **[01_linear_regression.md](01_linear_regression.md)** - Supervised learning framework, hypothesis functions, cost functions
- **[02_lms_algorithm.md](02_lms_algorithm.md)** - Gradient descent variants, learning rates, convergence
- **[03_normal_equations.md](03_normal_equations.md)** - Analytical solution, matrix calculus
- **[04_probabilistic_interpretation.md](04_probabilistic_interpretation.md)** - Statistical foundations, Gaussian noise, maximum likelihood
- **[05_locally_weighted_linear_regression.md](05_locally_weighted_linear_regression.md)** - Non-parametric approaches, kernel weighting
- **[06_hands-on_coding.md](06_hands-on_coding.md)** - Practical implementation guide

### Code Examples
- **[code/linear_regression_examples.py](code/linear_regression_examples.py)** - Basic implementation and visualization
- **[code/lms_algorithm_examples.py](code/lms_algorithm_examples.py)** - Gradient descent variants
- **[code/normal_equations_examples.py](code/normal_equations_examples.py)** - Analytical solutions
- **[code/probabilistic_linear_regression_examples.py](code/probabilistic_linear_regression_examples.py)** - Statistical interpretation
- **[code/locally_weighted_linear_regression_examples.py](code/locally_weighted_linear_regression_examples.py)** - Non-parametric regression

### Interactive Notebook
- **[03_normal_equations.ipynb](03_normal_equations.ipynb)** - Jupyter notebook with interactive examples

### Images
- **img/** - Visualizations including gradient descent, housing prices, and locally weighted regression plots

## Key Concepts

### Mathematical Foundation
- Hypothesis: $h_\theta(x) = \theta^T x$
- Cost function: $J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$
- Normal equations: $\theta = (X^T X)^{-1} X^T y$

### Optimization Methods
- **Batch GD**: $\theta := \theta + \alpha \sum_{i=1}^n (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}$
- **Stochastic GD**: $\theta := \theta + \alpha (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}$

### Non-parametric Methods
- **Locally Weighted Regression**: $w^{(i)} = \exp(-\frac{(x^{(i)} - x)^2}{2\tau^2})$

## Getting Started

1. Read `01_linear_regression.md` for fundamentals
2. Study `02_lms_algorithm.md` for optimization
3. Explore `03_normal_equations.md` for analytical solutions
4. Run Python examples to see concepts in action
5. Use `06_hands-on_coding.md` for practical implementation

## Setup

Install dependencies:
```bash
pip install -r code/requirements.txt
```

Or use conda:
```bash
conda env create -f code/environment.yaml
``` 
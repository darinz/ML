# Classification and Logistic Regression

[![Classification](https://img.shields.io/badge/Classification-Binary%20%26%20Multiclass-blue.svg)](https://en.wikipedia.org/wiki/Statistical_classification)
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-GLM%20Family-green.svg)](https://en.wikipedia.org/wiki/Logistic_regression)
[![Perceptron](https://img.shields.io/badge/Perceptron-Neural%20Network%20Foundation-red.svg)](https://en.wikipedia.org/wiki/Perceptron)
[![Softmax](https://img.shields.io/badge/Softmax-Multiclass%20Classification-purple.svg)](https://en.wikipedia.org/wiki/Softmax_function)
[![Newton's Method](https://img.shields.io/badge/Newton's%20Method-Optimization%20Algorithm-orange.svg)](https://en.wikipedia.org/wiki/Newton's_method)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)

## Overview

Classification fundamentals including binary classification, multi-class classification, perceptron algorithm, and Newton's method optimization.

## Materials

### Core Notes
- **[01_logistic_regression.md](01_logistic_regression.md)** - Binary classification, sigmoid function, gradient ascent, logistic loss
- **[02_perceptron.md](02_perceptron.md)** - Perceptron algorithm, threshold function, convergence theorem, geometric interpretation
- **[03_multi-class_classification.md](03_multi-class_classification.md)** - Softmax function, multinomial logistic regression, cross-entropy loss
- **[04_newtons_method.md](04_newtons_method.md)** - Newton's method, Hessian matrix, quadratic convergence, optimization

### Code Examples
- **[logistic_regression_examples.py](logistic_regression_examples.py)** - Binary classification, sigmoid function, gradient ascent
- **[perceptron_examples.py](perceptron_examples.py)** - Perceptron algorithm, threshold function, convergence analysis
- **[multiclass_softmax_example.py](multiclass_softmax_example.py)** - Multi-class classification, softmax function, temperature scaling
- **[newtons_method_examples.py](newtons_method_examples.py)** - Newton's method, Hessian computation, convergence analysis

### Visualizations
- **img/sigmoid.png** - Sigmoid function visualization
- **img/newtons_method.png** - Newton's method convergence

## Key Concepts

### Binary Classification
- Hypothesis: $h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$
- Sigmoid function: $g(z) = \frac{1}{1 + e^{-z}}$
- Logistic loss: $L(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$

### Perceptron Algorithm
- Threshold function: $g(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$
- Update rule: $\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$
- Convergence: Guaranteed for linearly separable data

### Multi-Class Classification
- Softmax function: $\mathrm{softmax}(t_1, \ldots, t_k) = \left[ \frac{\exp(t_1)}{\sum_{j=1}^k \exp(t_j)}, \ldots, \frac{\exp(t_k)}{\sum_{j=1}^k \exp(t_j)} \right]$
- Cross-entropy loss: $\ell_{ce}((t_1, \ldots, t_k), y) = -\log \left( \frac{\exp(t_y)}{\sum_{i=1}^k \exp(t_i)} \right)$

### Newton's Method
- Update rule: $\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)$
- Hessian matrix: $H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}$
- Quadratic convergence: Faster than gradient ascent

## Getting Started

1. Read `01_logistic_regression.md` for binary classification fundamentals
2. Study `02_perceptron.md` for perceptron algorithm
3. Explore `03_multi-class_classification.md` for multi-class methods
4. Learn `04_newtons_method.md` for advanced optimization
5. Run Python examples to see concepts in action

## Learning Path

- **Beginner**: Binary classification, sigmoid function, gradient ascent
- **Intermediate**: Perceptron algorithm, multi-class classification
- **Advanced**: Newton's method, Hessian analysis, convergence properties
- **Expert**: Numerical stability, temperature scaling, advanced optimization 
# Classification and Logistic Regression

[![Classification](https://img.shields.io/badge/Classification-Binary%20%26%20Multiclass-blue.svg)](https://en.wikipedia.org/wiki/Statistical_classification)
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-GLM%20Family-green.svg)](https://en.wikipedia.org/wiki/Logistic_regression)
[![Evaluation](https://img.shields.io/badge/Evaluation-Metrics%20%26%20Validation-purple.svg)](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Theory](https://img.shields.io/badge/Theory-Practical%20Examples-orange.svg)](https://github.com)

This section explores classification problems and logistic regression, including:

## Materials Included

### Theory and Concepts
- **`01_logistic_regression.md`** - Comprehensive lecture notes covering:
  - Binary classification fundamentals
  - Why linear regression fails for classification
  - Logistic (sigmoid) function and its properties
  - Probabilistic interpretation of logistic regression
  - Likelihood and log-likelihood functions
  - Gradient ascent optimization
  - Logistic loss and logit concepts
  - Mathematical derivations and proofs

### Code Implementation
- **`logistic_regression_examples.py`** - Complete Python implementation including:
  - Sigmoid function and its derivative
  - Logistic regression hypothesis function
  - Log-likelihood computation
  - Gradient calculation
  - Logistic loss function
  - Gradient ascent update rule
  - Ready-to-use example code

### Visual Aids
- **`img/sigmoid.png`** - Visualization of the sigmoid function

## Key Topics Covered

### Binary Classification
- Understanding classification vs. regression problems
- Negative and positive class concepts
- Probability interpretation of class membership

### Logistic Regression Theory
- **Sigmoid Function**: Mathematical properties and implementation
- **Hypothesis Function**: $h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$
- **Probabilistic Framework**: $P(y = 1 \mid x; \theta) = h_\theta(x)$
- **Likelihood Maximization**: Using maximum likelihood estimation
- **Gradient Ascent**: Optimization algorithm for parameter learning

### Mathematical Foundations
- Derivative of sigmoid function: $g'(z) = g(z)(1 - g(z))$
- Log-likelihood function and its optimization
- Logistic loss function and logit concepts
- Gradient computation and update rules

### Practical Implementation
- Vectorized implementations in NumPy
- Gradient ascent optimization
- Model training and parameter estimation
- Code examples with sample data

## Getting Started

1. **Read the Theory**: Start with `01_logistic_regression.md` for comprehensive understanding
2. **Study the Code**: Examine `logistic_regression_examples.py` for implementation details
3. **Run Examples**: Uncomment the example code in the Python file to see logistic regression in action
4. **Visualize**: Check the sigmoid function plot in `img/sigmoid.png`

## Related Sections

- **Linear Regression**: Foundation for understanding linear models
- **Generalized Linear Models**: Extension of logistic regression concepts
- **Regularization**: Ridge and Lasso regression for improved generalization

## Prerequisites

- Basic understanding of linear algebra and calculus
- Familiarity with Python and NumPy
- Knowledge of linear regression concepts
- Understanding of probability and statistics fundamentals

This section provides both theoretical foundations and practical implementation skills for one of the most fundamental classification algorithms in machine learning. 
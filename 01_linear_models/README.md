# Linear Models

[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Linear%20Models-blue.svg)](https://en.wikipedia.org/wiki/Linear_model)
[![Supervised Learning](https://img.shields.io/badge/Supervised%20Learning-Regression%20%26%20Classification-green.svg)](https://en.wikipedia.org/wiki/Supervised_learning)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://python.org)

Comprehensive materials for linear models in machine learning, covering theory, implementation, and practical examples.

## Topics

### **01_linear_regression/** - Linear Regression
- Hypothesis functions, cost functions, and mathematical foundations
- LMS algorithm, gradient descent, and normal equations
- Probabilistic interpretation with Gaussian noise models
- Locally weighted linear regression for non-linear patterns
- House price prediction examples with Portland housing data

**Key Files**: `01_linear_regression.md`, `02_lms_algorithm.md`, `03_normal_equations.md`

### **02_classification_logistic_regression/** - Classification
- Binary classification with logistic regression and sigmoid function
- Perceptron algorithm as neural network foundation
- Multi-class classification with softmax regression
- Newton's method for optimization
- Complete implementations with numerical stability

**Key Files**: `01_logistic_regression.md`, `02_perceptron.md`, `03_multi-class_classification.md`

### **03_generalized_linear_models/** - Generalized Linear Models (GLMs)
- Exponential family distributions as unified framework
- GLM construction with three fundamental assumptions
- Link functions connecting linear predictors to response distributions
- Applications: linear regression, logistic regression, Poisson regression
- Parameter estimation and model diagnostics

**Key Files**: `01_exponential_family.md`, `02_constructing_glm.md`, `constructing_glm_examples.py`

## Learning Path

1. **Linear Regression** (`01_linear_regression/`) - Start with fundamentals
2. **Classification** (`02_classification_logistic_regression/`) - Learn binary and multi-class
3. **GLMs** (`03_generalized_linear_models/`) - Master unified framework

## Key Mathematical Concepts

- **Linear Regression**: $h_\theta(x) = \theta^T x$, $J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$
- **Logistic Regression**: $h_\theta(x) = g(\theta^T x)$ where $g(z) = \frac{1}{1 + e^{-z}}$
- **GLMs**: $p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))$ with $\eta = \theta^T x$

## Applications

- **Regression**: House prices, stock forecasting, demand estimation
- **Classification**: Spam detection, medical diagnosis, image classification
- **GLMs**: Healthcare outcomes, financial risk, ecological modeling

---

*Foundation for understanding linear models in machine learning, from basic regression to advanced GLM frameworks.* 
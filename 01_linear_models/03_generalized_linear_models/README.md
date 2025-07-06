# Generalized Linear Models (GLMs)

[![GLM](https://img.shields.io/badge/GLM-Generalized%20Linear%20Models-blue.svg)](https://en.wikipedia.org/wiki/Generalized_linear_model)
[![Link Functions](https://img.shields.io/badge/Link%20Functions-Transformation-green.svg)](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function)
[![Statistics](https://img.shields.io/badge/Statistics-Exponential%20Family-purple.svg)](https://en.wikipedia.org/wiki/Exponential_family)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Theory](https://img.shields.io/badge/Theory-Practical%20Examples-orange.svg)](https://github.com)

This section introduces **Generalized Linear Models (GLMs)**, a powerful framework that unifies linear regression and logistic regression into a single theoretical framework. GLMs extend the capabilities of linear models to handle various types of response variables and distributions through the elegant mathematics of exponential families.

## 🎯 Learning Objectives

By the end of this section, you will understand:

- **Exponential Family Distributions**: The mathematical foundation that unifies various probability distributions
- **GLM Framework**: How to construct models for different types of response variables
- **Link Functions**: The role of link functions in connecting linear predictors to response distributions
- **Canonical Forms**: Understanding canonical link functions and their properties
- **Applications**: How GLMs apply to regression, classification, and other prediction tasks
- **Parameter Estimation**: Maximum likelihood estimation in the GLM framework

## 📚 Materials Overview

### Theory and Concepts

- **`01_exponential_family.md`**: Comprehensive introduction to exponential family distributions
  - Mathematical formulation and properties
  - Examples: Bernoulli and Gaussian distributions
  - Natural parameters, sufficient statistics, and log partition functions
  - Foundation for understanding GLM construction

- **`02_constructing_glm.md`**: Complete GLM construction methodology
  - Step-by-step recipe for building GLMs
  - Three fundamental assumptions of GLMs
  - Ordinary Least Squares as a GLM
  - Logistic Regression as a GLM
  - Link function selection and interpretation
  - Canonical response and link functions

### Practical Implementation

- **`exponential_family_examples.py`**: Python implementations and examples
  - Generic exponential family probability calculations
  - Bernoulli distribution in exponential family form
  - Gaussian distribution in exponential family form
  - Interactive examples and demonstrations

- **`constructing_glm_examples.py`**: GLM construction examples
  - Ordinary Least Squares implementation
  - Logistic Regression implementation
  - Parameter estimation demonstrations

## 🔑 Key Concepts Covered

### Exponential Family Distributions
The exponential family provides a unified framework for probability distributions:

```math
p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))
```

Where:
- $\eta$ is the **natural parameter**
- $T(y)$ is the **sufficient statistic**
- $a(\eta)$ is the **log partition function**
- $b(y)$ is the **base measure**

### GLM Construction Recipe
1. **Choose Response Distribution**: Select from exponential family (Gaussian, Bernoulli, Poisson, etc.)
2. **Define Linear Predictor**: $\eta = \theta^T x$
3. **Specify Link Function**: Connect $\eta$ to the mean parameter $\mu$
4. **Estimate Parameters**: Use maximum likelihood or other methods

### Three Fundamental Assumptions
1. **Distribution**: $y \mid x; \theta \sim \text{ExponentialFamily}(\eta)$
2. **Prediction Goal**: $h(x) = \mathbb{E}[y|x]$
3. **Linear Relationship**: $\eta = \theta^T x$

### Common GLM Examples

| Model | Response Distribution | Link Function | Canonical Link | Use Case |
|-------|---------------------|---------------|----------------|----------|
| **Linear Regression** | Gaussian | Identity | Yes | Continuous outcomes |
| **Logistic Regression** | Bernoulli | Logit | Yes | Binary classification |
| **Poisson Regression** | Poisson | Log | Yes | Count data |
| **Gamma Regression** | Gamma | Inverse | Yes | Positive continuous data |

## 🚀 Quick Start Examples

### Linear Regression as GLM
```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([2, 3, 4])
model = LinearRegression().fit(X, y)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
```

### Logistic Regression as GLM
```python
from sklearn.linear_model import LogisticRegression

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([0, 1, 1])
model = LogisticRegression().fit(X, y)
print('Coefficients:', model.coef_)
print('Probabilities:', model.predict_proba(X))
```

## 📖 Prerequisites

- Understanding of linear regression and logistic regression
- Basic probability and statistics
- Familiarity with Python and NumPy
- Knowledge of maximum likelihood estimation
- Comfort with mathematical notation and calculus

## 🎓 Learning Path

1. **Start with Exponential Family** (`01_exponential_family.md`)
   - Understand the mathematical foundation
   - Work through the examples in `exponential_family_examples.py`

2. **Learn GLM Construction** (`02_constructing_glm.md`)
   - Master the three assumptions
   - See how OLS and logistic regression fit the framework
   - Practice with `constructing_glm_examples.py`

3. **Apply to Real Problems**
   - Choose appropriate distributions for your data
   - Select canonical link functions
   - Interpret coefficients and predictions

## 🔬 Advanced Topics

After mastering the basics, explore:
- **Non-canonical link functions**: When to use alternatives to canonical links
- **Quasi-likelihood**: Handling overdispersion and model misspecification
- **Mixed models**: Incorporating random effects
- **Bayesian GLMs**: Prior specification and posterior inference
- **Regularization**: Ridge, Lasso, and Elastic Net for GLMs

## 📊 Applications

GLMs are widely used in:
- **Healthcare**: Modeling disease outcomes and treatment effects
- **Finance**: Credit scoring and risk assessment
- **Marketing**: Customer behavior and conversion modeling
- **Ecology**: Species abundance and environmental modeling
- **Economics**: Demand forecasting and policy analysis

## 🔗 Related Sections

- **Linear Regression** (`../01_linear_regression/`): Foundation for understanding linear predictors
- **Logistic Regression** (`../02_classification_logistic_regression/`): Binary classification as a GLM
- **Advanced Topics**: Mixed models, hierarchical GLMs, and Bayesian approaches

## 📝 Notes and Tips

- **Canonical Links**: Always prefer canonical link functions when possible, as they provide optimal statistical properties
- **Model Diagnostics**: Check residuals, goodness-of-fit, and overdispersion
- **Interpretation**: Coefficients represent changes in the linear predictor, not necessarily the response
- **Software**: Use specialized GLM packages (statsmodels, R's glm) for robust inference

---

*This section builds upon the foundational linear models to provide a unified framework for understanding regression and classification problems through the elegant lens of exponential families and link functions. The GLM framework is fundamental to modern statistical modeling and machine learning.* 
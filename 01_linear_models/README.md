# Linear Models

[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Linear%20Models-blue.svg)](https://en.wikipedia.org/wiki/Linear_model)
[![Supervised Learning](https://img.shields.io/badge/Supervised%20Learning-Regression%20%26%20Classification-green.svg)](https://en.wikipedia.org/wiki/Supervised_learning)
[![Educational](https://img.shields.io/badge/Educational-Course%20Materials-orange.svg)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-red.svg)](https://numpy.org)

This section contains comprehensive materials and resources for linear models in machine learning. Each covers fundamental topics with detailed theory, practical implementations, and hands-on examples.

## Learning Objectives

By the end of this section, you will understand:
- **Linear Regression**: Least squares, gradient descent, normal equations, and probabilistic interpretation
- **Classification**: Logistic regression, perceptron algorithm, and multi-class classification with softmax
- **Generalized Linear Models**: Exponential family distributions, link functions, and unified framework
- **Optimization**: Gradient descent, Newton's method, and analytical solutions
- **Implementation**: Python code with NumPy, vectorized operations, and real-world applications

## Topics

### **01_linear_regression/** - Linear Regression Fundamentals
Comprehensive coverage of linear regression including:
- **Theory**: Hypothesis functions, cost functions, and mathematical foundations
- **Optimization**: LMS algorithm, batch/stochastic/mini-batch gradient descent
- **Analytical Solutions**: Normal equations and closed-form solutions
- **Statistical Foundations**: Probabilistic interpretation with Gaussian noise models
- **Advanced Topics**: Locally weighted linear regression for non-linear patterns
- **Practical Examples**: House price prediction with Portland housing data
- **Code Implementation**: Complete Python examples with NumPy vectorization

**Key Files**: `01_linear_regression.md`, `02_lms_algorithm.md`, `03_normal_equations.md`, `04_probabilistic_interpretation.md`, `05_locally_weighted_linear_regression.md`

### **02_classification_logistic_regression/** - Classification and Logistic Regression
Complete treatment of classification problems including:
- **Binary Classification**: Logistic regression with sigmoid function and probabilistic interpretation
- **Perceptron Algorithm**: Historical foundation for neural networks with threshold functions
- **Multi-Class Classification**: Softmax regression for handling multiple classes
- **Advanced Optimization**: Newton's method with quadratic convergence
- **Mathematical Foundations**: Likelihood functions, cross-entropy loss, and gradient computations
- **Practical Applications**: Image classification, text classification, medical diagnosis
- **Code Examples**: Complete implementations with numerical stability considerations

**Key Files**: `01_logistic_regression.md`, `02_perceptron.md`, `03_multi-class_classification.md`, `04_newtons_method.md`

### **03_generalized_linear_models/** - Generalized Linear Models (GLMs)
Comprehensive unified framework extending linear models through:
- **Exponential Family Distributions**: Mathematical foundation unifying probability distributions with detailed derivations
- **GLM Construction**: Three fundamental assumptions and systematic four-step recipe
- **Link Functions**: Connecting linear predictors to response distributions with canonical and non-canonical options
- **Canonical Forms**: Understanding canonical link functions and their optimal properties
- **Applications**: Linear regression, logistic regression, Poisson regression, and Gamma regression as GLMs
- **Parameter Estimation**: Maximum likelihood, IRLS, and gradient descent methods
- **Model Diagnostics**: Comprehensive residual analysis and goodness-of-fit measures
- **Real-World Examples**: Healthcare, finance, marketing, and environmental applications
- **Advanced Topics**: Regularization, mixed models, and Bayesian extensions
- **Code Implementation**: Complete Python frameworks with educational annotations and best practices

**Key Files**: `01_exponential_family.md`, `02_constructing_glm.md`, `exponential_family_examples.py`, `constructing_glm_examples.py`

## Getting Started

### **Recommended Learning Path**

1. **Start with Linear Regression** (`01_linear_regression/`)
   - Read `01_linear_regression.md` for fundamentals
   - Study optimization algorithms in `02_lms_algorithm.md`
   - Explore analytical solutions in `03_normal_equations.md`
   - Understand statistical foundations in `04_probabilistic_interpretation.md`

2. **Move to Classification** (`02_classification_logistic_regression/`)
   - Begin with `01_logistic_regression.md` for binary classification
   - Study `02_perceptron.md` for neural network foundations
   - Learn multi-class classification in `03_multi-class_classification.md`
   - Explore advanced optimization in `04_newtons_method.md`

3. **Master GLMs** (`03_generalized_linear_models/`)
   - Understand exponential families in `01_exponential_family.md`
   - Learn GLM construction in `02_constructing_glm.md`
   - Apply the unified framework to various problems

### **Practical Implementation**
- Run Python code examples in each subfolder
- Experiment with different datasets and parameters
- Visualize results using provided images and plots
- Compare different optimization algorithms
- Practice implementing algorithms from scratch

## Key Mathematical Concepts

### **Linear Regression**
- Hypothesis: $h_\theta(x) = \theta^T x$
- Cost Function: $J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$
- Normal Equations: $\theta = (X^T X)^{-1} X^T y$

### **Logistic Regression**
- Sigmoid Function: $g(z) = \frac{1}{1 + e^{-z}}$
- Hypothesis: $h_\theta(x) = g(\theta^T x)$
- Log-Likelihood: $\ell(\theta) = \sum_{i=1}^m [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$

### **Generalized Linear Models**
- Exponential Family: $p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))$
- Linear Predictor: $\eta = \theta^T x$
- Link Function: $g(\mu) = \eta$

## Applications and Use Cases

- **Regression**: House price prediction, stock price forecasting, demand estimation
- **Classification**: Spam detection, medical diagnosis, image classification
- **GLMs**: Healthcare outcomes, financial risk assessment, ecological modeling
- **Optimization**: Large-scale machine learning, real-time prediction systems

## Related Materials

- **Prerequisites**: Review mathematical foundations in `../00_math_python_numpy_review/`
- **Advanced Topics**: Explore more complex models and algorithms
- **Reference Materials**: Check `../reference/` for additional resources and cheatsheets

---

*This comprehensive collection provides the foundation for understanding linear models in machine learning, from basic regression to advanced GLM frameworks. Each section includes theoretical foundations, practical implementations, and real-world applications.* 
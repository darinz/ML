# Classification and Logistic Regression

[![Classification](https://img.shields.io/badge/Classification-Binary%20%26%20Multiclass-blue.svg)](https://en.wikipedia.org/wiki/Statistical_classification)
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-GLM%20Family-green.svg)](https://en.wikipedia.org/wiki/Logistic_regression)
[![Perceptron](https://img.shields.io/badge/Perceptron-Neural%20Network%20Foundation-red.svg)](https://en.wikipedia.org/wiki/Perceptron)
[![Evaluation](https://img.shields.io/badge/Evaluation-Metrics%20%26%20Validation-purple.svg)](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Theory](https://img.shields.io/badge/Theory-Practical%20Examples-orange.svg)](https://github.com)

This section explores classification problems, logistic regression, and the perceptron learning algorithm, including:

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

- **`02_perceptron.md`** - Detailed exploration of the perceptron algorithm:
  - Historical significance and neural network foundations
  - Threshold function vs. sigmoid function
  - Perceptron learning algorithm and update rule
  - Geometric interpretation and hyperplane separation
  - Perceptron convergence theorem
  - Linear separability and limitations
  - Comparison with logistic regression
  - Python implementation examples

### Code Implementation
- **`logistic_regression_examples.py`** - Complete Python implementation including:
  - Sigmoid function and its derivative
  - Logistic regression hypothesis function
  - Log-likelihood computation
  - Gradient calculation
  - Logistic loss function
  - Gradient ascent update rule
  - Ready-to-use example code

- **`perceptron_examples.py`** - Perceptron algorithm implementation:
  - Threshold function (step function)
  - Perceptron prediction function
  - Perceptron update rule
  - Vectorized implementations
  - Example usage with sample data

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

### Perceptron Learning Algorithm
- **Threshold Function**: $g(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$
- **Hypothesis Function**: $h_\theta(x) = g(\theta^T x)$
- **Update Rule**: $\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$
- **Geometric Interpretation**: Hyperplane separation of classes
- **Convergence Theorem**: Guaranteed convergence for linearly separable data
- **Historical Significance**: Foundation for neural networks and AI

### Mathematical Foundations
- Derivative of sigmoid function: $g'(z) = g(z)(1 - g(z))$
- Log-likelihood function and its optimization
- Logistic loss function and logit concepts
- Gradient computation and update rules
- Perceptron convergence properties

### Practical Implementation
- Vectorized implementations in NumPy
- Gradient ascent optimization for logistic regression
- Perceptron learning algorithm implementation
- Model training and parameter estimation
- Code examples with sample data

## Getting Started

1. **Read the Theory**: Start with `01_logistic_regression.md` for comprehensive understanding of logistic regression
2. **Study Perceptron**: Read `02_perceptron.md` to understand the perceptron algorithm and its historical significance
3. **Study the Code**: Examine `logistic_regression_examples.py` and `perceptron_examples.py` for implementation details
4. **Run Examples**: Uncomment the example code in both Python files to see both algorithms in action
5. **Visualize**: Check the sigmoid function plot in `img/sigmoid.png`

## Related Sections

- **Linear Regression**: Foundation for understanding linear models
- **Generalized Linear Models**: Extension of logistic regression concepts
- **Regularization**: Ridge and Lasso regression for improved generalization

## Prerequisites

- Basic understanding of linear algebra and calculus
- Familiarity with Python and NumPy
- Knowledge of linear regression concepts
- Understanding of probability and statistics fundamentals

This section provides both theoretical foundations and practical implementation skills for fundamental classification algorithms in machine learning, including the historically significant perceptron algorithm that laid the groundwork for modern neural networks. 
# Classification and Logistic Regression

[![Classification](https://img.shields.io/badge/Classification-Binary%20%26%20Multiclass-blue.svg)](https://en.wikipedia.org/wiki/Statistical_classification)
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-GLM%20Family-green.svg)](https://en.wikipedia.org/wiki/Logistic_regression)
[![Perceptron](https://img.shields.io/badge/Perceptron-Neural%20Network%20Foundation-red.svg)](https://en.wikipedia.org/wiki/Perceptron)
[![Softmax](https://img.shields.io/badge/Softmax-Multiclass%20Classification-purple.svg)](https://en.wikipedia.org/wiki/Softmax_function)
[![Newton's Method](https://img.shields.io/badge/Newton's%20Method-Optimization%20Algorithm-orange.svg)](https://en.wikipedia.org/wiki/Newton's_method)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Theory](https://img.shields.io/badge/Theory-Practical%20Examples-cyan.svg)](https://github.com)

This comprehensive section explores classification problems, logistic regression, perceptron learning algorithm, multi-class classification, and Newton's method optimization, providing both theoretical foundations and practical implementations.

## Table of Contents

- [Overview](#overview)
- [Materials Included](#materials-included)
- [Key Topics Covered](#key-topics-covered)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Related Sections](#related-sections)

## Overview

Classification is a fundamental machine learning task where we predict discrete class labels rather than continuous values. This section covers:

- **Binary Classification**: Predicting between two classes using logistic regression
- **Multi-Class Classification**: Handling multiple classes with softmax regression
- **Perceptron Algorithm**: Historical foundation for neural networks
- **Newton's Method**: Advanced optimization technique for logistic regression

## Materials Included

### Theory and Concepts

#### **`01_logistic_regression.md`** - Binary Classification Fundamentals
Comprehensive lecture notes covering:
- Binary classification problem formulation
- Why linear regression fails for classification tasks
- Logistic (sigmoid) function and its mathematical properties
- Probabilistic interpretation of logistic regression
- Likelihood and log-likelihood functions
- Gradient ascent optimization algorithm
- Logistic loss function and logit concepts
- Complete mathematical derivations and proofs

#### **`02_perceptron.md`** - Neural Network Foundation
Detailed exploration of the perceptron algorithm:
- Historical significance and neural network foundations
- Threshold function vs. sigmoid function comparison
- Perceptron learning algorithm and update rule
- Geometric interpretation and hyperplane separation
- Perceptron convergence theorem and proof
- Linear separability and algorithm limitations
- Comparison with logistic regression approaches
- Python implementation examples and demonstrations

#### **`03_multi-class_classification.md`** - Multi-Class Classification
Comprehensive coverage of multi-class classification:
- Multi-class classification fundamentals and motivation
- Softmax function and its mathematical properties
- Multinomial logistic regression theory
- Cross-entropy loss and negative log-likelihood
- Numerical stability considerations and implementation
- Gradient derivation and optimization techniques
- Practical applications and real-world extensions
- Step-by-step examples and implementation guidance

#### **`04_newtons_method.md`** - Advanced Optimization
In-depth coverage of Newton's method for logistic regression:
- Newton's method theory and mathematical foundations
- Hessian matrix computation and properties
- Convergence analysis and quadratic convergence
- Comparison with gradient ascent optimization
- Implementation considerations and numerical stability
- Practical applications and performance characteristics
- Step-by-step algorithm derivation and examples

### Code Implementation

#### **`logistic_regression_examples.py`** - Binary Classification Implementation
Complete Python implementation including:
- Sigmoid function and its derivative computation
- Logistic regression hypothesis function
- Log-likelihood computation and optimization
- Gradient calculation and vectorized operations
- Logistic loss function implementation
- Gradient ascent update rule
- Ready-to-use example code with sample data

#### **`perceptron_examples.py`** - Perceptron Algorithm Implementation
Perceptron algorithm implementation:
- Threshold function (step function) implementation
- Perceptron prediction function
- Perceptron update rule and learning algorithm
- Vectorized implementations for efficiency
- Example usage with sample data and visualization

#### **`multiclass_softmax_example.py`** - Multi-Class Classification Implementation
Multi-class classification implementation:
- Softmax function with numerical stability
- Cross-entropy loss computation
- Multi-class probability prediction
- Example usage with sample data and demonstrations

#### **`newtons_method_examples.py`** - Newton's Method Implementation
Advanced optimization implementation:
- Newton's method algorithm for logistic regression
- Hessian matrix computation and inversion
- Convergence monitoring and stopping criteria
- Comparison with gradient ascent performance
- Numerical stability considerations
- Complete implementation with examples

### Visual Aids

#### **`img/sigmoid.png`** - Sigmoid Function Visualization
Visualization of the sigmoid function showing its S-shaped curve and properties

#### **`img/newtons_method.png`** - Newton's Method Convergence
Visualization of Newton's method convergence showing quadratic convergence behavior

## Key Topics Covered

### Binary Classification
- Understanding classification vs. regression problems
- Negative and positive class concepts and interpretation
- Probability interpretation of class membership
- Decision boundary and classification thresholds

### Logistic Regression Theory
- **Sigmoid Function**: Mathematical properties and implementation
- **Hypothesis Function**: $h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$
- **Probabilistic Framework**: $P(y = 1 \mid x; \theta) = h_\theta(x)$
- **Likelihood Maximization**: Using maximum likelihood estimation
- **Gradient Ascent**: Optimization algorithm for parameter learning
- **Logistic Loss**: $L(\theta) = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$

### Perceptron Learning Algorithm
- **Threshold Function**:

$$
g(z) = \begin{cases} 
1 & \text{if } z \geq 0 \\ 
0 & \text{if } z < 0 
\end{cases}
$$

- **Hypothesis Function**: $h_\theta(x) = g(\theta^T x)$
- **Update Rule**: $\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}$
- **Geometric Interpretation**: Hyperplane separation of classes
- **Convergence Theorem**: Guaranteed convergence for linearly separable data
- **Historical Significance**: Foundation for neural networks and AI

### Multi-Class Classification
- **Softmax Function**: $\mathrm{softmax}(t_1, \ldots, t_k) = \left[ \frac{\exp(t_1)}{\sum_{j=1}^k \exp(t_j)}, \ldots, \frac{\exp(t_k)}{\sum_{j=1}^k \exp(t_j)} \right]$
- **Multinomial Logistic Regression**: $P(y = i \mid x; \theta) = \frac{\exp(\theta_i^\top x)}{\sum_{j=1}^k \exp(\theta_j^\top x)}$
- **Cross-Entropy Loss**: $\ell_{ce}((t_1, \ldots, t_k), y) = -\log \left( \frac{\exp(t_y)}{\sum_{i=1}^k \exp(t_i)} \right)$
- **Numerical Stability**: Subtracting maximum logit before exponentiation
- **Gradient Computation**: $\frac{\partial \ell_{ce}(t, y)}{\partial t_i} = \phi_i - 1\{y = i\}$
- **Applications**: Image classification, text classification, medical diagnosis

### Newton's Method Optimization
- **Update Rule**: $\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)$
- **Hessian Matrix**: $H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}$
- **Quadratic Convergence**: Much faster than gradient ascent
- **Computational Cost**: Higher per iteration but fewer iterations needed
- **Numerical Stability**: Careful handling of matrix inversion
- **Convergence Analysis**: Theoretical guarantees and practical considerations

### Mathematical Foundations
- **Sigmoid Derivative**: $g'(z) = g(z)(1 - g(z))$
- **Log-likelihood Function**: $\ell(\theta) = \sum_{i=1}^m [y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$
- **Gradient Computation**: $\nabla_\theta \ell(\theta) = \frac{1}{m} X^T(h_\theta(X) - y)$
- **Hessian Matrix**: $H = \frac{1}{m} X^T \text{diag}(h_\theta(X) \odot (1 - h_\theta(X))) X$
- **Perceptron Convergence**: Guaranteed for linearly separable data
- **Softmax Gradient**: $\frac{\partial \ell_{ce}(t, y)}{\partial t_i} = \phi_i - 1\{y = i\}$

### Practical Implementation
- **Vectorized Operations**: Efficient NumPy implementations
- **Gradient Ascent**: First-order optimization for logistic regression
- **Newton's Method**: Second-order optimization with quadratic convergence
- **Perceptron Learning**: Online learning algorithm implementation
- **Multi-Class Classification**: Softmax regression with numerical stability
- **Model Training**: Parameter estimation and convergence monitoring
- **Code Examples**: Complete implementations with sample data

## Getting Started

### **Step 1: Read the Theory**
1. Start with `01_logistic_regression.md` for comprehensive understanding of binary classification
2. Read `02_perceptron.md` to understand the perceptron algorithm and its historical significance
3. Study `03_multi-class_classification.md` to learn about softmax and multi-class classification
4. Explore `04_newtons_method.md` to understand advanced optimization techniques

### **Step 2: Study the Code**
1. Examine `logistic_regression_examples.py` for binary classification implementation
2. Review `perceptron_examples.py` for perceptron algorithm implementation
3. Study `multiclass_softmax_example.py` for multi-class classification
4. Analyze `newtons_method_examples.py` for advanced optimization

### **Step 3: Run Examples**
1. Uncomment the example code in all Python files to see the algorithms in action
2. Experiment with different parameters and datasets
3. Compare performance between gradient ascent and Newton's method
4. Visualize the sigmoid function plot in `img/sigmoid.png`
5. Study Newton's method convergence in `img/newtons_method.png`

### **Step 4: Experiment and Extend**
1. Try different learning rates and convergence criteria
2. Implement additional evaluation metrics
3. Apply the algorithms to your own datasets
4. Compare with scikit-learn implementations

## Prerequisites

### Mathematical Background
- **Linear Algebra**: Matrix operations, eigenvectors, eigenvalues
- **Calculus**: Derivatives, gradients, Hessian matrices
- **Probability**: Basic probability theory and Bayes' theorem
- **Statistics**: Maximum likelihood estimation and hypothesis testing

### Programming Skills
- **Python**: Intermediate Python programming skills
- **NumPy**: Array operations and vectorized computations
- **Matplotlib**: Basic plotting and visualization (optional)

### Machine Learning Concepts
- **Linear Regression**: Understanding of linear models and least squares
- **Gradient Descent**: Basic optimization concepts
- **Model Evaluation**: Understanding of training and testing splits

## Related Sections

### **Prerequisites**
- **Linear Regression**: Foundation for understanding linear models
- **Mathematics Review**: Linear algebra, calculus, and probability fundamentals

### **Extensions**
- **Generalized Linear Models**: Extension of logistic regression concepts
- **Regularization**: Ridge and Lasso regression for improved generalization
- **Neural Networks**: Building on perceptron foundations

### **Applications**
- **Computer Vision**: Image classification applications
- **Natural Language Processing**: Text classification tasks
- **Healthcare**: Medical diagnosis and risk prediction

---

This section provides comprehensive theoretical foundations and practical implementation skills for fundamental classification algorithms in machine learning. It covers binary classification with logistic regression, the historically significant perceptron algorithm that laid the groundwork for modern neural networks, multi-class classification using softmax regression, and advanced optimization techniques with Newton's method. The materials are designed to be both mathematically rigorous and practically applicable, with complete code implementations and visual aids to enhance understanding. 
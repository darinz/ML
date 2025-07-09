# Classification and Logistic Regression

[![Classification](https://img.shields.io/badge/Classification-Binary%20%26%20Multiclass-blue.svg)](https://en.wikipedia.org/wiki/Statistical_classification)
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-GLM%20Family-green.svg)](https://en.wikipedia.org/wiki/Logistic_regression)
[![Perceptron](https://img.shields.io/badge/Perceptron-Neural%20Network%20Foundation-red.svg)](https://en.wikipedia.org/wiki/Perceptron)
[![Softmax](https://img.shields.io/badge/Softmax-Multiclass%20Classification-purple.svg)](https://en.wikipedia.org/wiki/Softmax_function)
[![Newton's Method](https://img.shields.io/badge/Newton's%20Method-Optimization%20Algorithm-orange.svg)](https://en.wikipedia.org/wiki/Newton's_method)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Theory](https://img.shields.io/badge/Theory-Practical%20Examples-cyan.svg)](https://github.com)

This comprehensive section explores classification problems, logistic regression, perceptron learning algorithm, multi-class classification, and Newton's method optimization, providing both theoretical foundations and practical implementations with enhanced educational content and complete code examples.

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
- **Enhanced with**: Step-by-step derivations, geometric interpretations, practical considerations, historical context, and comparative analysis

#### **`02_perceptron.md`** - Neural Network Foundation
Detailed exploration of the perceptron algorithm:
- Historical significance and neural network foundations
- Threshold function vs. sigmoid function comparison
- Perceptron learning algorithm and update rule
- Geometric interpretation and hyperplane separation
- Perceptron convergence theorem and proof
- Linear separability and algorithm limitations
- Comparison with logistic regression approaches
- **Enhanced with**: Detailed convergence analysis, practical examples, implementation guidance, and geometric intuition

#### **`03_multi-class_classification.md`** - Multi-Class Classification
Comprehensive coverage of multi-class classification:
- Multi-class classification fundamentals and motivation
- Softmax function and its mathematical properties
- Multinomial logistic regression theory
- Cross-entropy loss and negative log-likelihood
- Numerical stability considerations and implementation
- Gradient derivation and optimization techniques
- Practical applications and real-world extensions
- **Enhanced with**: Temperature scaling, advanced topics, comparative analysis, and implementation best practices

#### **`04_newtons_method.md`** - Advanced Optimization
In-depth coverage of Newton's method for logistic regression:
- Newton's method theory and mathematical foundations
- Hessian matrix computation and properties
- Convergence analysis and quadratic convergence
- Comparison with gradient ascent optimization
- Implementation considerations and numerical stability
- Practical applications and performance characteristics
- **Enhanced with**: Convergence analysis, practical considerations, and advanced optimization topics

### Code Implementation

#### **`logistic_regression_examples.py`** - Binary Classification Implementation
**Comprehensive Python implementation with enhanced features:**
- **Core Functions**: Sigmoid function with numerical stability, hypothesis computation, log-likelihood and cross-entropy loss
- **Optimization**: Gradient computation, gradient ascent algorithm with convergence checking
- **Training Pipeline**: Complete training loop with learning rate scheduling and early stopping
- **Prediction**: Probability prediction and binary classification with customizable thresholds
- **Data Generation**: Synthetic data generation for demonstration and testing
- **Visualization**: Decision boundary plotting and training history visualization
- **Educational Features**: Comprehensive documentation with mathematical formulas, examples, and usage instructions
- **Practical Examples**: Complete demonstration function with data generation, training, evaluation, and visualization
- **Numerical Stability**: Proper handling of edge cases, overflow prevention, and convergence monitoring

#### **`perceptron_examples.py`** - Perceptron Algorithm Implementation
**Enhanced perceptron implementation with educational features:**
- **Core Algorithm**: Threshold function (Heaviside step function), prediction, and learning rule
- **Training Pipeline**: Complete perceptron training with convergence checking and early stopping
- **Data Generation**: Both linearly separable and non-separable data generation for demonstration
- **Convergence Analysis**: Training history tracking and convergence monitoring
- **Visualization**: Decision boundary plotting and training error visualization
- **Comparison Features**: Side-by-side comparison with logistic regression
- **Historical Context**: Background on Frank Rosenblatt's 1958 perceptron
- **Educational Content**: Detailed explanations of algorithm behavior and limitations
- **Practical Demonstrations**: Examples showing convergence on separable data and limitations on non-separable data

#### **`multiclass_softmax_example.py`** - Multi-Class Classification Implementation
**Advanced multi-class classification with comprehensive features:**
- **Softmax Function**: Numerically stable implementation with temperature scaling
- **Loss Functions**: Cross-entropy loss with proper numerical handling
- **Gradient Computation**: Efficient gradient calculation for softmax regression
- **Training Pipeline**: Complete training loop with gradient descent optimization
- **Prediction**: Class prediction and probability estimation
- **Data Generation**: Multi-class synthetic data generation
- **Visualization**: Multi-class decision boundary plotting
- **Advanced Features**: Temperature scaling, numerical stability demonstrations
- **Comparison**: Side-by-side comparison with binary logistic regression
- **Educational Content**: Detailed mathematical explanations and practical examples

#### **`newtons_method_examples.py`** - Newton's Method Implementation
**Comprehensive Newton's method implementation with analysis tools:**
- **Root Finding**: 1D Newton's method for finding function roots
- **Function Maximization**: 1D Newton's method for optimization
- **Logistic Regression**: Multidimensional Newton's method with Hessian computation
- **Hessian Analysis**: Eigenvalue analysis, condition number computation, positive definiteness checking
- **Convergence Analysis**: Training history tracking and convergence monitoring
- **Comparison Tools**: Side-by-side comparison with gradient descent
- **Visualization**: Convergence plots and analysis tools
- **Numerical Stability**: Proper handling of singular Hessians and regularization
- **Educational Content**: Detailed explanations of convergence properties and trade-offs
- **Practical Examples**: Various optimization problems and convergence analysis

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
- **Temperature Scaling**: Controlling probability distribution sharpness
- **Applications**: Image classification, text classification, medical diagnosis

### Newton's Method Optimization
- **Update Rule**: $\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)$
- **Hessian Matrix**: $H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}$
- **Quadratic Convergence**: Much faster than gradient ascent
- **Computational Cost**: Higher per iteration but fewer iterations needed
- **Numerical Stability**: Careful handling of matrix inversion
- **Convergence Analysis**: Theoretical guarantees and practical considerations
- **Hessian Properties**: Eigenvalue analysis and condition number monitoring

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
- **Educational Features**: Comprehensive documentation and practical demonstrations

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
1. **Execute Complete Demonstrations**: Run the main demonstration functions in each Python file:
   - `demonstrate_logistic_regression()` in `logistic_regression_examples.py`
   - `demonstrate_perceptron()` in `perceptron_examples.py`
   - `demonstrate_multiclass_classification()` in `multiclass_softmax_example.py`
   - `demonstrate_newtons_method()` in `newtons_method_examples.py`

2. **Explore Additional Examples**: Each file includes comprehensive examples showing:
   - Basic function usage and mathematical properties
   - Training and convergence analysis
   - Visualization of decision boundaries and training history
   - Comparison between different algorithms

3. **Experiment with Parameters**: Try different:
   - Learning rates and convergence criteria
   - Data generation parameters
   - Model configurations
   - Visualization options

### **Step 4: Experiment and Extend**
1. **Algorithm Comparison**: Compare performance between:
   - Gradient ascent vs. Newton's method for logistic regression
   - Perceptron vs. logistic regression on different datasets
   - Binary vs. multi-class classification approaches

2. **Custom Implementations**: 
   - Apply algorithms to your own datasets
   - Implement additional evaluation metrics
   - Add regularization techniques
   - Extend to different loss functions

3. **Advanced Topics**:
   - Study Hessian properties and numerical stability
   - Explore temperature scaling in softmax
   - Analyze convergence behavior
   - Implement cross-validation

## Prerequisites

### Mathematical Background
- **Linear Algebra**: Matrix operations, eigenvectors, eigenvalues
- **Calculus**: Derivatives, gradients, Hessian matrices
- **Probability**: Basic probability theory and Bayes' theorem
- **Statistics**: Maximum likelihood estimation and hypothesis testing

### Programming Skills
- **Python**: Intermediate Python programming skills
- **NumPy**: Array operations and vectorized computations
- **Matplotlib**: Basic plotting and visualization
- **Scikit-learn**: Basic understanding (for comparisons)

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

## Recent Enhancements

This section has been significantly enhanced with:

### **Educational Improvements**
- **Comprehensive Documentation**: All Python files now include detailed mathematical explanations, step-by-step derivations, and practical examples
- **Enhanced Theory**: Markdown files have been expanded with deeper mathematical explanations, geometric interpretations, and historical context
- **Practical Examples**: Complete demonstration functions showing end-to-end workflows
- **Visualization Tools**: Decision boundary plotting, training history visualization, and convergence analysis

### **Code Quality Enhancements**
- **Numerical Stability**: Proper handling of edge cases, overflow prevention, and convergence monitoring
- **Modular Design**: Functions that can be used independently or as part of complete pipelines
- **Error Handling**: Appropriate error messages, warnings, and convergence checking
- **Performance Optimization**: Vectorized operations and efficient implementations

### **New Features**
- **Synthetic Data Generation**: Functions to create appropriate test datasets for each algorithm
- **Comparison Tools**: Side-by-side comparisons between different algorithms and optimization methods
- **Advanced Analysis**: Hessian analysis, convergence monitoring, and performance evaluation
- **Temperature Scaling**: Advanced softmax features for controlling probability distributions

### **Learning Resources**
- **Step-by-Step Examples**: Detailed examples showing how to use each function
- **Mathematical Background**: Clear explanations of the underlying mathematics
- **Implementation Guidance**: Best practices and practical considerations
- **Historical Context**: Background information on the development of these algorithms

This section now provides a comprehensive, educational, and practical resource for understanding and implementing fundamental classification algorithms in machine learning, with both rigorous theoretical foundations and complete, well-documented code implementations. 
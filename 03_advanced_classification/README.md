# Advanced Classification: Support Vector Machines and Kernel Methods

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-Complete-brightgreen.svg)](README.md)
[![Implementation](https://img.shields.io/badge/Implementation-Complete-orange.svg)](README.md)
[![Topics](https://img.shields.io/badge/Topics-5%20Covered-purple.svg)](README.md)

This directory contains comprehensive implementations and theoretical explanations for Support Vector Machines (SVM) and Kernel Methods, covering the complete SVM pipeline from basic concepts to advanced regularization techniques.

## Table of Contents

- [Directory Structure](#directory-structure)
- [Kernel Methods](#1-kernel-methods)
- [Kernel Properties](#2-kernel-properties)
- [SVM Margins](#3-svm-margins)
- [SVM Optimal Margin](#4-svm-optimal-margin-dual-form)
- [SVM Regularization](#5-svm-regularization-soft-margin)
- [Usage](#usage)
- [Mathematical Framework](#mathematical-framework)
- [Visualization Features](#visualization-features)
- [Key Insights](#key-insights)
- [References](#references)

## Directory Structure

### Theory Documents
- `01_kernel_methods.md` - Comprehensive guide to kernel methods and the kernel trick
- `02_kernel_properties.md` - Properties of kernels and Mercer's theorem
- `03_svm_margins.md` - SVM margins, functional vs geometric margins, and optimal margin classifiers
- `04_svm_optimal_margin.md` - Dual formulation and Lagrange duality for SVM
- `05_svm_regularization.md` - SVM with slack variables and regularization

### Implementation Files
- `kernel_methods_examples.py` - Complete kernel methods implementation with examples
- `kernel_properties_examples.py` - Kernel validation and properties demonstrations
- `svm_margins_equations.py` - SVM margin calculations and Lagrangian implementations
- `svm_optimal_margin_examples.py` - Dual SVM implementation and SMO algorithm
- `svm_regularization_examples.py` - Soft margin SVM with slack variables

### Supporting Files
- `requirements.txt` - Required Python packages
- `img/` - Visualization images for SVM concepts
- `README.md` - This file

---

## 1. Kernel Methods

### Key Concepts

#### 1.1 Feature Maps and Motivation
- **Linear Model Limitation**: When linear models fail to capture non-linear patterns
- **Polynomial Feature Maps**: Mapping input to higher-dimensional polynomial features
- **Curse of Dimensionality**: Exponential growth of feature space with polynomial degree

#### 1.2 LMS with Features
```python
# LMS algorithm with custom feature map
def lms_with_features(X, y, feature_map, learning_rate=0.01, max_iterations=1000):
    # Implementation of gradient descent with feature maps
```

#### 1.3 The Kernel Trick
- **Representer Theorem**: θ can be represented as linear combination of φ(x⁽ⁱ⁾)
- **Kernel Functions**: K(x, z) = ⟨φ(x), φ(z)⟩
- **Computational Efficiency**: O(d) vs O(d³) for explicit feature computation

### Common Kernels

#### Polynomial Kernel
```python
def polynomial_kernel(x, z, degree=3, gamma=1.0, r=1.0):
    return (gamma * np.dot(x, z) + r) ** degree
```

#### RBF (Gaussian) Kernel
```python
def rbf_kernel(x, z, gamma=1.0):
    diff = x - z
    return np.exp(-gamma * np.dot(diff, diff))
```

#### Linear Kernel
```python
def linear_kernel(x, z):
    return np.dot(x, z)
```

### Usage Examples
```python
# Kernelized LMS
beta, K = kernelized_lms(X, y, rbf_kernel)

# Multiple kernel learning
predictions = multiple_kernel_learning(X, y, [linear_kernel, rbf_kernel])

# Hyperparameter tuning
best_params = example_hyperparameter_tuning()
```

---

## 2. Kernel Properties

### Mercer's Theorem
A function K is a valid kernel if and only if for any finite set of points, the corresponding kernel matrix is symmetric positive semi-definite.

### Kernel Validation
```python
def is_valid_kernel(X, kernel_func):
    """Check if kernel function satisfies Mercer's conditions"""
    K = kernel_matrix(X, kernel_func)
    eigvals = np.linalg.eigvalsh(K)
    return np.all(eigvals >= -1e-10)
```

### Kernel Matrix Construction
```python
def kernel_matrix(X, kernel_func):
    """Compute kernel matrix for dataset X"""
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(X[i], X[j])
    return K
```

---

## 3. SVM Margins

### 3.1 Functional vs Geometric Margins

#### Functional Margin
```python
def functional_margin(w, b, x_i, y_i):
    """Functional margin: ŷ⁽ⁱ⁾ = y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b)"""
    return y_i * (np.dot(w, x_i) + b)
```

#### Geometric Margin
```python
def geometric_margin(w, b, x_i, y_i):
    """Geometric margin: γ⁽ⁱ⁾ = y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b)/||w||"""
    norm_w = np.linalg.norm(w)
    return y_i * (np.dot(w, x_i) + b) / norm_w
```

### 3.2 Optimal Margin Classifier
The primal optimization problem:
```math
minimize: (1/2) ||w||²
subject to: y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1, i = 1,...,n
```

### 3.3 Lagrange Duality
```python
def lagrangian(w, b, alpha, X, y):
    """Lagrangian: L(w, b, α) = (1/2)||w||² - Σαᵢ[y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) - 1]"""
    first_term = 0.5 * np.dot(w, w)
    second_term = 0
    for i in range(len(y)):
        constraint = y[i] * (np.dot(w, X[i]) + b) - 1
        second_term -= alpha[i] * constraint
    return first_term + second_term
```

---

## 4. SVM Optimal Margin (Dual Form)

### 4.1 Dual Formulation
The dual optimization problem:
```math
maximize: W(α) = Σαᵢ - (1/2)ΣΣ y⁽ⁱ⁾y⁽ʲ⁾αᵢαⱼ⟨x⁽ⁱ⁾, x⁽ʲ⁾⟩
subject to: αᵢ ≥ 0, i = 1,...,n
           Σαᵢy⁽ⁱ⁾ = 0
```

### 4.2 Support Vectors
- Points with αᵢ > 0 are support vectors
- Only support vectors affect the decision boundary
- w = Σαᵢy⁽ⁱ⁾x⁽ⁱ⁾ (sum over support vectors)

### 4.3 SMO Algorithm
Sequential Minimal Optimization for efficient dual problem solving:

```python
def smo_algorithm(X, y, C=1.0, tol=1e-3, max_iter=1000):
    """SMO algorithm for solving SVM dual problem"""
    # Implementation of SMO with two-variable updates
```

### 4.4 KKT Conditions
```python
def kkt_conditions(alphas, X, y, b, K):
    """Check Karush-Kuhn-Tucker conditions for convergence"""
    # αᵢ = 0 → y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1
    # αᵢ = C → y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≤ 1  
    # 0 < αᵢ < C → y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) = 1
```

---

## 5. SVM Regularization (Soft Margin)

### 5.1 Slack Variables
Introduction of slack variables ξᵢ to handle non-separable data:

```math
minimize: (1/2) ||w||² + C * Σξᵢ
subject to: y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0
```

### 5.2 Regularization Parameter C
- **Small C**: Large margin, more misclassifications (high bias, low variance)
- **Large C**: Small margin, fewer misclassifications (low bias, high variance)

### 5.3 Dual Form with Slack Variables
```math
maximize: W(α) = Σαᵢ - (1/2)ΣΣ y⁽ⁱ⁾y⁽ʲ⁾αᵢαⱼK(x⁽ⁱ⁾, x⁽ʲ⁾)
subject to: 0 ≤ αᵢ ≤ C, Σαᵢy⁽ⁱ⁾ = 0
```

### 5.4 SVMRegularization Class
```python
class SVMRegularization:
    def __init__(self, C=1.0, kernel='linear', tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.tol = tol
    
    def fit(self, X, y):
        """Train SVM using SMO algorithm"""
        
    def predict(self, X):
        """Predict class labels"""
        
    def smo_algorithm(self, X, y, max_iter=1000):
        """SMO implementation with regularization"""
```

---

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running Examples
```bash
# Kernel methods examples
python kernel_methods_examples.py

# Kernel properties validation
python kernel_properties_examples.py

# SVM margins and equations
python svm_margins_equations.py

# SVM optimal margin (dual form)
python svm_optimal_margin_examples.py

# SVM regularization (soft margin)
python svm_regularization_examples.py
```

### Key Functions by Topic

#### Kernel Methods
- `polynomial_kernel()`, `rbf_kernel()`, `linear_kernel()`
- `kernelized_lms()` - Kernelized least mean squares
- `multiple_kernel_learning()` - Learning with multiple kernels
- `kernel_ridge_regression()` - Regularized kernel regression

#### Kernel Properties
- `is_valid_kernel()` - Validate kernel using Mercer's theorem
- `kernel_matrix()` - Compute kernel matrix
- `check_positive_semidefinite()` - Verify PSD property

#### SVM Margins
- `functional_margin()`, `geometric_margin()`
- `min_functional_margin()`, `min_geometric_margin()`
- `lagrangian()` - Lagrangian formulation
- `svm_qp_solver()` - Quadratic programming solver

#### SVM Optimal Margin
- `smo_algorithm()` - Sequential Minimal Optimization
- `kkt_conditions()` - Karush-Kuhn-Tucker conditions
- `compute_w_from_alpha()` - Reconstruct w from dual variables
- `dual_svm_objective()` - Dual objective function

#### SVM Regularization
- `coordinate_ascent_example()` - Optimization demonstration
- `slack_variables_visualization()` - C parameter effects
- `regularization_tradeoff_analysis()` - C vs accuracy analysis
- `margin_calculation()` - Margin computation with regularization

---

## Mathematical Framework

### 1. Primal to Dual Transformation
The journey from primal to dual form enables:
- Kernel trick application
- Efficient optimization via SMO
- Support vector identification

### 2. Kernel Trick Application
Any algorithm expressible in terms of inner products can be kernelized:
- Replace ⟨x, z⟩ with K(x, z)
- Work in high-dimensional feature spaces efficiently
- Handle non-linear decision boundaries

### 3. Regularization Trade-offs
The C parameter balances:
- Margin maximization (generalization)
- Training error minimization (fitting)
- Support vector sparsity

---

## Visualization Features

The implementations include comprehensive visualizations:

1. **Kernel Comparisons**: Different kernel functions and their decision boundaries
2. **Margin Visualization**: Functional vs geometric margins
3. **Support Vectors**: Highlighting support vectors for different C values
4. **Optimization Paths**: Coordinate ascent and SMO convergence
5. **Regularization Effects**: C parameter impact on decision boundaries

---

## Key Insights

1. **Kernel Trick**: Enables efficient computation in high-dimensional spaces
2. **Support Vectors**: Only critical points affect the final classifier
3. **Dual Form**: Enables kernelization and efficient optimization
4. **Regularization**: Balances model complexity with training accuracy
5. **KKT Conditions**: Provide convergence criteria and interpretability

---

## References

- CS229 Lecture Notes on SVM and Kernel Methods
- Platt's SMO Algorithm Paper
- Mercer's Theorem and Kernel Properties
- Support Vector Machines: Theory and Applications
- Kernel Methods in Machine Learning 
# Problem Sets - Series 02

This directory contains the second series of problem sets covering advanced machine learning topics, optimization methods, and deep learning foundations.

## Overview

The problem sets in this series build upon fundamental concepts and introduce more sophisticated techniques in machine learning, including optimization theory, kernel methods, neural networks, and dimensionality reduction.

## Problem Set Contents

### Problem Set 1: Train-Test Splitting, Generalized Least Squares Regression, MAP as Regularization
**Files:** `ps1_problems.md`, `ps1_solution.md`, `q2_d_solution.png`, `q2_solution.png`, `q2_gradient_descent.png`

**Topics Covered:**
- **Biased Test Error Analysis**: Identifying and fixing data leakage in train-test splitting procedures
- **Gradient Descent**: Implementation and analysis of gradient descent algorithms
- **Generalized Least Squares**: Extending linear regression with weighted error terms
- **MAP as Regularization**: Understanding maximum a posteriori estimation as a form of regularization

**Key Concepts:**
- Proper data preprocessing and train-test splitting
- Gradient descent convergence analysis
- Regularization techniques in linear models

### Problem Set 2: Cross-Validation, Lasso, Subgradients, and Convexity
**Files:** `ps2_problems.md`, `ps2_solution.md`, `k-fold.ipynb`, `lasso_cv.ipynb`

**Topics Covered:**
- **K-fold Cross-Validation**: Implementation and analysis of cross-validation techniques
- **Lasso Regression**: L1 regularization and sparse solutions
- **Subgradients**: Theory and applications in non-differentiable optimization
- **Convexity**: Properties, verification methods, and practical implications
- **Gradient Descent Convergence**: Theoretical analysis of convergence rates

**Key Concepts:**
- Model selection and hyperparameter tuning
- Sparse regression and feature selection
- Optimization theory for non-smooth functions
- Convex optimization foundations

### Problem Set 3: Gradient Descent and Stochastic Methods
**Files:** `ps3_problems.md`, `ps3_solution.md`, `PyTorch_Introduction.ipynb`

**Topics Covered:**
- **Gradient Descent Convergence**: Theoretical analysis with Lipschitz continuity
- **Stochastic Gradient Descent (SGD)**: Theory, implementation, and convergence analysis
- **Mini-batching**: Batch processing techniques and variance reduction
- **PyTorch Introduction**: Practical implementation of optimization algorithms

**Key Concepts:**
- Convergence analysis of optimization algorithms
- Stochastic optimization methods
- Computational efficiency trade-offs
- Deep learning framework basics

### Problem Set 4: Kernel Methods and Neural Networks
**Files:** `ps4_problems.md`, `ps4_solution.md`, `PyTorch_NN.ipynb`

**Topics Covered:**
- **Kernel Methods**: Kernel properties, Mercer's theorem, and kernelized regression
- **Representer Theorem**: Proving optimal solutions lie in the span of data points
- **Neural Network Implementation**: Practical PyTorch-based neural network training

**Key Concepts:**
- Kernel trick and feature space transformations
- Theoretical foundations of kernel methods
- Neural network architecture and training
- Practical deep learning implementation

### Problem Set 5: Neural Network Theory and Backpropagation
**Files:** `ps5_problems.md`, `ps5_solution.md`

**Topics Covered:**
- **Chain Rule Applications**: Multi-variate chain rule in neural networks
- **Backpropagation**: Forward and backward pass computations
- **Weight Initialization**: Impact of initialization on gradient flow
- **Neural Network Gradients**: Analytical computation of partial derivatives

**Key Concepts:**
- Automatic differentiation fundamentals
- Neural network architecture analysis
- Gradient flow and vanishing/exploding gradients
- Mathematical foundations of deep learning

### Problem Set 6: Dimensionality Reduction and Matrix Decompositions
**Files:** `ps6_problems.md`, `ps6_solution.md`, `q1_pca.png`, `q1_v2_pca.png`

**Topics Covered:**
- **Principal Component Analysis (PCA)**: Theory, computation, and applications
- **Eigenvalue Decomposition**: Using eigenbasis for matrix operations
- **Singular Value Decomposition (SVD)**: General matrix decomposition techniques
- **Dimensionality Reduction**: Data compression and feature extraction

**Key Concepts:**
- Linear algebra foundations for ML
- Data compression and visualization
- Matrix factorization techniques
- Geometric interpretation of transformations

## Prerequisites

To work through these problem sets, you should be familiar with:
- Linear algebra (eigenvalues, eigenvectors, matrix operations)
- Calculus (partial derivatives, chain rule, optimization)
- Probability and statistics
- Python programming (NumPy, PyTorch)
- Basic machine learning concepts

## File Structure

Each problem set folder contains:
- `psX_problems.md`: Problem statements and theoretical questions
- `psX_solution.md`: Detailed solutions with explanations
- Supporting files: Jupyter notebooks, images, and additional resources

## Getting Started

1. **Review Prerequisites**: Ensure you have the necessary mathematical and programming background
2. **Install Dependencies**: For notebooks, install required packages:
   ```bash
   pip install jupyter numpy matplotlib pytorch ipywidgets
   ```
3. **Start with PS1**: Work through the problem sets sequentially as they build upon each other
4. **Practice Implementation**: Use the provided notebooks to implement and experiment with algorithms

## Learning Objectives

By completing this series, you will:
- Understand advanced optimization techniques in machine learning
- Master theoretical foundations of kernel methods and neural networks
- Develop practical skills in implementing ML algorithms
- Gain proficiency in mathematical analysis of ML algorithms
- Learn to apply dimensionality reduction techniques effectively

## Additional Resources

- Linear algebra review materials in the main course directory
- PyTorch documentation for deep learning implementations
- Optimization theory references for theoretical foundations 
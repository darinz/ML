# Advanced Classification: Kernel Methods and Support Vector Machines

[![Kernel Methods](https://img.shields.io/badge/Kernel-Methods-blue.svg)](https://en.wikipedia.org/wiki/Kernel_method)
[![SVM](https://img.shields.io/badge/SVM-Support%20Vector%20Machines-green.svg)](https://en.wikipedia.org/wiki/Support_vector_machine)
[![Classification](https://img.shields.io/badge/Classification-Non-linear-purple.svg)](https://en.wikipedia.org/wiki/Statistical_classification)

Advanced classification techniques using kernel methods and SVMs to handle complex, non-linear decision boundaries beyond linear models.

## Overview

**Why These Methods Matter:**
- Linear classifiers create only straight-line boundaries
- Real-world data is often not linearly separable
- Kernel methods work in high-dimensional feature spaces efficiently
- SVMs provide theoretical guarantees and optimality

## Materials

### Theory
- **[01_kernel_methods.md](01_kernel_methods.md)** - Feature maps, kernel trick, common kernels (RBF, polynomial, linear)
- **[02_kernel_properties.md](02_kernel_properties.md)** - Positive definite kernels, Mercer's theorem, kernel construction
- **[03_svm_margins.md](03_svm_margins.md)** - Functional/geometric margins, maximum margin classification

### Implementation
- **[code/kernel_methods_examples.py](code/kernel_methods_examples.py)** - Curse of dimensionality, kernel trick, multiple kernel comparison
- **[code/kernel_properties_examples.py](code/kernel_properties_examples.py)** - Positive definiteness testing, Mercer's theorem, kernel PCA
- **[code/svm_margins_equations.py](code/svm_margins_equations.py)** - Linear classifier visualization, margin calculations, SVM optimization
- **[code/svm_optimal_margin_examples.py](code/svm_optimal_margin_examples.py)** - Primal/dual formulations, support vectors, kernel SVM
- **[code/svm_regularization_examples.py](code/svm_regularization_examples.py)** - SMO algorithm, slack variables, KKT conditions

### Visualizations
- **img/** - Decision boundaries, constraints, coordinate ascent, hyperplanes

## Key Concepts

### The Kernel Trick
```python
# Instead of computing explicit features φ(x)
# Compute kernel K(x, z) = ⟨φ(x), φ(z)⟩
K(x, z) = exp(-γ ||x - z||²)  # RBF kernel
```

### SVM Dual Formulation
```python
# Maximize: Σ α_i - ½ Σ Σ α_i α_j y_i y_j K(x_i, x_j)
# Subject to: 0 ≤ α_i ≤ C, Σ α_i y_i = 0
```

### Common Kernels
- **Linear**: $K(x,z) = x^T z$
- **Polynomial**: $K(x,z) = (γx^T z + r)^d$
- **RBF**: $K(x,z) = \exp(-γ||x-z||^2)$
- **Sigmoid**: $K(x,z) = \tanh(γx^T z + r)$

## Applications

### GDA
- **Computer Vision**: Image classification, object detection, face recognition
- **Bioinformatics**: Protein classification, gene expression, drug discovery
- **NLP**: Text classification, sentiment analysis, machine translation
- **Finance**: Credit scoring, fraud detection, portfolio optimization

## Getting Started

1. Read `01_kernel_methods.md` for kernel foundations
2. Study `02_kernel_properties.md` for mathematical properties
3. Learn `03_svm_margins.md` for margin concepts
4. Run Python examples to see algorithms in action
5. Explore visualizations in `img/` folder

## Prerequisites

- Linear algebra and calculus
- Optimization theory (Lagrangian duality)
- Basic probability and statistics
- Understanding of linear classification methods

## Practical Guidelines

### Choosing Kernels
- **Start with RBF**: Works well for most problems
- **Try linear**: For high-dimensional or linearly separable data
- **Use polynomial**: For expected polynomial relationships

### Hyperparameter Tuning
- **C (regularization)**: Trade-off between margin and error
- **γ (RBF bandwidth)**: Controls influence reach of training points

### Performance Considerations
- **Training**: O(n²) for kernel matrix computation
- **Memory**: O(n²) for storing kernel matrix
- **Prediction**: O(n) per prediction (reducible with support vectors) 
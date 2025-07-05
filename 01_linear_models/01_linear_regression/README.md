# Linear Regression

[![Regression](https://img.shields.io/badge/Regression-Linear%20Regression-blue.svg)](https://en.wikipedia.org/wiki/Linear_regression)
[![Least Squares](https://img.shields.io/badge/Least%20Squares-Optimization-green.svg)](https://en.wikipedia.org/wiki/Least_squares)
[![Mathematics](https://img.shields.io/badge/Mathematics-Linear%20Algebra-purple.svg)](https://en.wikipedia.org/wiki/Linear_algebra)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Theory](https://img.shields.io/badge/Theory-Practical%20Examples-orange.svg)](https://github.com)

This section covers the fundamentals of linear regression, including:

- The least squares method
- Model assumptions
- Analytical and computational solutions
- Practical applications and examples
- Exercises and problem sets

## Materials

### Main Notes
- **[01_linear_regression.md](01_linear_regression.md)** - Comprehensive notes covering:
  - **House Price Prediction Example**: Real-world application with Portland housing data
  - **Supervised Learning Fundamentals**: Input features, target variables, training sets
  - **Mathematical Formulation**: Hypothesis functions, parameter notation, vectorized forms
  - **Cost Function Analysis**: Least squares, geometric interpretation, vectorized implementation
  - **Python Code Examples**: Practical implementations with NumPy
  - **Multiple Features**: Extension to multivariate regression

### Code Examples
- **[linear_regression_examples.py](linear_regression_examples.py)** - Complete Python implementation with:
  - **Data Visualization**: Housing data scatter plots and cost function surfaces
  - **Hypothesis Function**: Implementation of h_θ(x) = θ^T x with detailed annotations
  - **Cost Function Methods**: Non-vectorized, vectorized, and MSE implementations
  - **Multiple Features**: Extension to multivariate regression examples
  - **Interactive Examples**: Step-by-step code with inline explanations

### Supporting Materials
- **img/housing_prices.png** - Visualization of house prices vs living area scatter plot
- **img/learning_algorithm.png** - Diagram illustrating the supervised learning process

## Key Concepts Covered

### Supervised Learning Framework
- Training examples: $(x^{(i)}, y^{(i)})$ pairs
- Hypothesis function: $h_\theta(x) = \theta^T x$
- Cost function: $J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$

### Practical Implementation
- Vectorized computations for efficiency
- Mean squared error (MSE) calculations
- Python implementations with NumPy
- Real-world data visualization

### Mathematical Foundations
- Linear algebra concepts
- Optimization principles
- Geometric interpretations
- Notation conventions

## Getting Started

1. Read through the main notes: `01_linear_regression.md`
2. Run the Python code examples to understand the concepts
3. Experiment with the house price prediction example
4. Explore the visualizations in the `img/` directory

Materials include lecture notes, code samples, and practice problems. 
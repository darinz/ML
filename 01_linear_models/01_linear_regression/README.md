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
- Gradient descent optimization algorithms
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

- **[02_lms_algorithm.md](02_lms_algorithm.md)** - Detailed coverage of optimization algorithms:
  - **Gradient Descent Fundamentals**: Mathematical and geometric intuition
  - **LMS Algorithm Derivation**: Step-by-step mathematical derivation of update rules
  - **Batch Gradient Descent**: Full dataset optimization with convergence analysis
  - **Stochastic Gradient Descent (SGD)**: Single-example updates for large datasets
  - **Mini-batch Gradient Descent**: Compromise between batch and stochastic methods
  - **Learning Rate Analysis**: Convergence properties and hyperparameter tuning
  - **Geometric Interpretations**: Visual understanding of optimization trajectories

- **[03_normal_equations.md](03_normal_equations.md)** - Analytical solution approach:
  - **Matrix Derivatives**: Calculus with matrices and vectorized operations
  - **Design Matrix Formulation**: Compact representation of training data
  - **Normal Equations Derivation**: Closed-form solution for optimal parameters
  - **Least Squares Revisited**: Matrix-vector notation and efficient computation
  - **Closed-form Solution**: Direct computation of $θ = (X^T X)^(-1) X^T y$
  - **Comparison with Gradient Descent**: When to use analytical vs iterative methods

- **[04_probabilistic_interpretation.md](04_probabilistic_interpretation.md)** - Statistical foundations and interpretation:
  - **Linear Model Assumption**: $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$ with error terms
  - **Gaussian Noise Model**: IID error distribution $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$
  - **Conditional Distribution**: $p(y^{(i)}|x^{(i)}; \theta)$ as Gaussian distribution
  - **Likelihood Function**: $L(\theta) = \prod_{i=1}^n p(y^{(i)} \mid x^{(i)}; \theta)$
  - **Maximum Likelihood Estimation**: Connection to least squares optimization
  - **Log-Likelihood Derivation**: Mathematical proof that MLE equals least squares
  - **Statistical Justification**: Why least squares is a natural choice under Gaussian assumptions

### Code Examples
- **[linear_regression_examples.py](linear_regression_examples.py)** - Complete Python implementation with:
  - **Data Visualization**: Housing data scatter plots and cost function surfaces
  - **Hypothesis Function**: Implementation of $h_θ(x) = θ^T x$ with detailed annotations
  - **Cost Function Methods**: Non-vectorized, vectorized, and MSE implementations
  - **Multiple Features**: Extension to multivariate regression examples
  - **Interactive Examples**: Step-by-step code with inline explanations

- **[lms_algorithm_examples.py](lms_algorithm_examples.py)** - Optimization algorithm implementations:
  - **Single Example Updates**: Stochastic gradient descent for individual training examples
  - **Batch Gradient Descent**: Full dataset optimization with vectorized computations
  - **Stochastic Gradient Descent**: Random sampling for large-scale optimization
  - **Mini-batch Gradient Descent**: Balanced approach with configurable batch sizes
  - **Gradient Computation**: Efficient vectorized gradient calculations
  - **Parameter Updates**: All three optimization variants with clear implementations

- **[normal_equations_examples.py](normal_equations_examples.py)** - Analytical solution implementations:
  - **Matrix Derivative Examples**: Computing gradients with respect to matrices
  - **Design Matrix Operations**: Efficient matrix-vector computations
  - **Normal Equation Solution**: Direct computation of optimal parameters
  - **Closed-form Theta**: Implementation of $θ = (X^T X)^{-1} X^T y$
  - **Cost Function Verification**: Validation of analytical solutions
  - **Comparison Examples**: Side-by-side comparison with gradient descent results

- **[probabilistic_linear_regression_examples.py](probabilistic_linear_regression_examples.py)** - Statistical interpretation implementations:
  - **Data Generation**: Synthetic data with known true parameters and Gaussian noise
  - **Gaussian Likelihood**: Computing $p(y_i | x_i; \theta)$ for individual data points
  - **Log-Likelihood Function**: Complete dataset likelihood calculation
  - **Maximum Likelihood Estimation**: Connection to least squares solution
  - **Parameter Estimation**: Closed-form solution using normal equations
  - **Model Validation**: Mean squared error computation and verification

### Supporting Materials
- **img/housing_prices.png** - Visualization of house prices vs living area scatter plot
- **img/learning_algorithm.png** - Diagram illustrating the supervised learning process
- **img/gradient_descent.png** - Contour plot showing gradient descent optimization trajectory
- **img/results_from_gradient_descent.png** - Final fitted line with training data visualization

## Key Concepts Covered

### Supervised Learning Framework
- Training examples: $(x^{(i)}, y^{(i)})$ pairs
- Hypothesis function: $h_\theta(x) = \theta^T x$
- Cost function: $J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$

### Optimization Algorithms
- **Batch Gradient Descent**: $\theta := \theta + \alpha \sum_{i=1}^n (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}$
- **Stochastic Gradient Descent**: $\theta := \theta + \alpha (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}$
- **Mini-batch Gradient Descent**: $\theta := \theta + \alpha \frac{1}{m} \sum_{k=1}^m (y^{(k)} - h_\theta(x^{(k)})) x^{(k)}$
- **Normal Equations (Analytical)**: $\theta = (X^T X)^{-1} X^T \vec{y}$

### Practical Implementation
- Vectorized computations for efficiency
- Mean squared error (MSE) calculations
- Python implementations with NumPy
- Real-world data visualization
- Learning rate selection and convergence analysis

### Mathematical Foundations
- Linear algebra concepts
- Optimization principles
- Geometric interpretations
- Notation conventions
- Gradient computation and chain rule applications
- Matrix calculus and derivatives
- Normal equations and closed-form solutions
- **Statistical foundations**: Gaussian distributions, likelihood functions, maximum likelihood estimation
- **Probabilistic interpretation**: Error models, conditional distributions, statistical justification

## Getting Started

1. Read through the main notes: `01_linear_regression.md`
2. Study the optimization algorithms: `02_lms_algorithm.md`
3. Explore the analytical solution: `03_normal_equations.md`
4. **Understand the statistical foundations**: `04_probabilistic_interpretation.md`
5. Run the Python code examples to understand the concepts
6. Experiment with the house price prediction example
7. Explore the visualizations in the `img/` directory
8. Practice with different gradient descent variants
9. Compare iterative vs analytical solutions

Materials include lecture notes, code samples, optimization algorithms, and practice problems. 
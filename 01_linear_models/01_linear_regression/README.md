# Linear Regression

[![Regression](https://img.shields.io/badge/Regression-Linear%20Regression-blue.svg)](https://en.wikipedia.org/wiki/Linear_regression)
[![Least Squares](https://img.shields.io/badge/Least%20Squares-Optimization-green.svg)](https://en.wikipedia.org/wiki/Least_squares)
[![Mathematics](https://img.shields.io/badge/Mathematics-Linear%20Algebra-purple.svg)](https://en.wikipedia.org/wiki/Linear_algebra)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Theory](https://img.shields.io/badge/Theory-Practical%20Examples-orange.svg)](https://github.com)

This section covers the comprehensive fundamentals of linear regression, including:

- The least squares method and mathematical foundations
- Model assumptions and statistical interpretations
- Analytical and computational solutions
- Gradient descent optimization algorithms
- Non-parametric approaches (locally weighted regression)
- Practical applications and real-world examples
- Advanced topics and computational considerations

## Materials

### Main Notes
- **[01_linear_regression.md](01_linear_regression.md)** - Comprehensive notes covering:
  - **Supervised Learning Framework**: Complete introduction to regression vs classification
  - **House Price Prediction Example**: Real-world application with Portland housing data
  - **Mathematical Formulation**: Hypothesis functions, parameter notation, vectorized forms
  - **Cost Function Analysis**: Least squares, geometric interpretation, optimization landscape
  - **Multiple Features**: Extension to multivariate regression with detailed explanations
  - **Practical Considerations**: Feature scaling, model interpretation, and real-world applications

- **[02_lms_algorithm.md](02_lms_algorithm.md)** - Detailed coverage of optimization algorithms:
  - **Gradient Descent Fundamentals**: Mathematical and geometric intuition with step-by-step derivations
  - **LMS Algorithm Derivation**: Complete mathematical derivation of update rules
  - **Batch Gradient Descent**: Full dataset optimization with convergence analysis and visualization
  - **Stochastic Gradient Descent (SGD)**: Single-example updates with convergence properties
  - **Mini-batch Gradient Descent**: Balanced approach with practical implementation guidelines
  - **Learning Rate Analysis**: Convergence properties, hyperparameter tuning, and adaptive methods
  - **Geometric Interpretations**: Visual understanding of optimization trajectories and landscapes
  - **Historical Context**: Development and evolution of gradient descent methods

- **[03_normal_equations.md](03_normal_equations.md)** - Analytical solution approach:
  - **Matrix Calculus**: Comprehensive coverage of matrix derivatives and vectorized operations
  - **Design Matrix Formulation**: Compact representation and efficient computation
  - **Normal Equations Derivation**: Step-by-step mathematical derivation of closed-form solution
  - **Least Squares Revisited**: Matrix-vector notation and computational efficiency
  - **Closed-form Solution**: Direct computation of $θ = (X^T X)^(-1) X^T y$ with numerical stability
  - **Comparison with Gradient Descent**: When to use analytical vs iterative methods
  - **Computational Complexity**: Analysis of time and space requirements
  - **Numerical Stability**: Handling ill-conditioned matrices and regularization

- **[04_probabilistic_interpretation.md](04_probabilistic_interpretation.md)** - Statistical foundations and interpretation:
  - **Linear Model Assumption**: $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$ with detailed error analysis
  - **Gaussian Noise Model**: IID error distribution $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$ and alternatives
  - **Conditional Distribution**: $p(y^{(i)}|x^{(i)}; \theta)$ as Gaussian distribution with full derivation
  - **Likelihood Function**: $L(\theta) = \prod_{i=1}^n p(y^{(i)} \mid x^{(i)}; \theta)$ and its properties
  - **Maximum Likelihood Estimation**: Complete connection to least squares optimization
  - **Log-Likelihood Derivation**: Mathematical proof that MLE equals least squares under Gaussian assumptions
  - **Statistical Justification**: Why least squares is optimal and when assumptions may be violated
  - **Alternative Justifications**: Other theoretical foundations for least squares

- **[05_locally_weighted_linear_regression.md](05_locally_weighted_linear_regression.md)** - Non-parametric approaches:
  - **Limitations of Global Models**: When linear models fail and why local approaches are needed
  - **Bias-Variance Trade-off**: Detailed analysis of model complexity and generalization
  - **Locally Weighted Regression Algorithm**: Complete mathematical formulation and implementation
  - **Gaussian Kernel Weighting**: Distance-based weighting with bandwidth parameter tuning
  - **Parametric vs Non-parametric Learning**: Fundamental differences and when to use each
  - **Curse of Dimensionality**: Challenges in high-dimensional spaces and mitigation strategies
  - **Computational Optimizations**: Efficient implementations and approximate methods
  - **Alternative Kernels**: Epanechnikov, triangular, and other kernel functions
  - **Extensions and Applications**: Real-world use cases and advanced variants

### Code Examples with Comprehensive Annotations
- **[linear_regression_examples.py](linear_regression_examples.py)** - Complete Python implementation with:
  - **Data Visualization**: Housing data scatter plots, cost function surfaces, and optimization landscapes
  - **Hypothesis Function**: Implementation of $h_θ(x) = θ^T x$ with detailed mathematical annotations
  - **Cost Function Methods**: Non-vectorized, vectorized, and MSE implementations with performance analysis
  - **Multiple Features**: Extension to multivariate regression with feature importance analysis
  - **Interactive Examples**: Step-by-step code with inline explanations and mathematical foundations
  - **Real-world Applications**: Housing price prediction with comprehensive analysis and visualization
  - **Performance Metrics**: MSE, MAE, R-squared calculations and interpretation

- **[lms_algorithm_examples.py](lms_algorithm_examples.py)** - Optimization algorithm implementations:
  - **Single Example Updates**: Stochastic gradient descent with detailed convergence analysis
  - **Batch Gradient Descent**: Full dataset optimization with vectorized computations and visualization
  - **Stochastic Gradient Descent**: Random sampling with learning rate scheduling and convergence plots
  - **Mini-batch Gradient Descent**: Balanced approach with configurable batch sizes and performance comparison
  - **Gradient Computation**: Efficient vectorized gradient calculations with mathematical explanations
  - **Parameter Updates**: All three optimization variants with convergence analysis and practical tips
  - **Learning Rate Analysis**: Impact of different learning rates on convergence and stability
  - **Performance Comparison**: Side-by-side analysis of different gradient descent variants

- **[normal_equations_examples.py](normal_equations_examples.py)** - Analytical solution implementations:
  - **Matrix Derivative Examples**: Computing gradients with respect to matrices with step-by-step explanations
  - **Design Matrix Operations**: Efficient matrix-vector computations with mathematical foundations
  - **Normal Equation Solution**: Direct computation of optimal parameters with numerical stability analysis
  - **Closed-form Theta**: Implementation of $θ = (X^T X)^{-1} X^T y$ with verification methods
  - **Cost Function Verification**: Validation of analytical solutions against iterative methods
  - **Comparison Examples**: Side-by-side comparison with gradient descent results and performance analysis
  - **Numerical Stability**: Handling ill-conditioned matrices and regularization techniques
  - **Computational Complexity**: Analysis of time and space requirements for different problem sizes

- **[probabilistic_linear_regression_examples.py](probabilistic_linear_regression_examples.py)** - Statistical interpretation implementations:
  - **Data Generation**: Synthetic data with known true parameters and Gaussian noise for controlled experiments
  - **Gaussian Likelihood**: Computing $p(y_i | x_i; \theta)$ for individual data points with detailed explanations
  - **Log-Likelihood Function**: Complete dataset likelihood calculation with mathematical foundations
  - **Maximum Likelihood Estimation**: Connection to least squares solution with statistical justification
  - **Parameter Estimation**: Closed-form solution using normal equations with uncertainty quantification
  - **Model Validation**: Mean squared error computation, residual analysis, and assumption checking
  - **Noise Impact Analysis**: How different noise levels affect parameter estimation quality
  - **Confidence Intervals**: Uncertainty quantification and statistical inference
  - **Model Assumptions**: Validation and analysis of when assumptions are violated

- **[locally_weighted_linear_regression_examples.py](locally_weighted_linear_regression_examples.py)** - Non-parametric implementations:
  - **Gaussian Kernel Implementation**: Distance-based weighting with bandwidth parameter control
  - **Weight Function Analysis**: Visualization of how weights change with distance and bandwidth
  - **Comparison with Global Methods**: Side-by-side analysis of LWR vs global linear regression
  - **Bandwidth Selection**: Cross-validation for optimal bandwidth parameter tuning
  - **Real-world Applications**: Housing price prediction with local modeling advantages
  - **Computational Complexity**: Analysis of scalability and performance considerations
  - **Curse of Dimensionality**: Demonstration of performance degradation in high dimensions
  - **Advanced Topics**: Alternative kernels, approximate methods, and optimization techniques

### Supporting Materials
- **img/housing_prices.png** - Visualization of house prices vs living area scatter plot
- **img/learning_algorithm.png** - Diagram illustrating the supervised learning process
- **img/gradient_descent.png** - Contour plot showing gradient descent optimization trajectory
- **img/results_from_gradient_descent.png** - Final fitted line with training data visualization

## Key Concepts Covered

### Supervised Learning Framework
- Training examples: $(x^{(i)}, y^{(i)})$ pairs with comprehensive notation
- Hypothesis function: $h_\theta(x) = \theta^T x$ with vectorized implementation
- Cost function: $J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$ with geometric interpretation

### Optimization Algorithms
- **Batch Gradient Descent**: $\theta := \theta + \alpha \sum_{i=1}^n (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}$ with convergence analysis
- **Stochastic Gradient Descent**: $\theta := \theta + \alpha (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}$ with learning rate scheduling
- **Mini-batch Gradient Descent**: $\theta := \theta + \alpha \frac{1}{m} \sum_{k=1}^m (y^{(k)} - h_\theta(x^{(k)})) x^{(k)}$ with batch size optimization
- **Normal Equations (Analytical)**: $\theta = (X^T X)^{-1} X^T \vec{y}$ with numerical stability considerations

### Non-parametric Methods
- **Locally Weighted Regression**: $w^{(i)} = \exp(-\frac{(x^{(i)} - x)^2}{2\tau^2})$ with bandwidth selection
- **Weighted Least Squares**: $\theta = (X^T W X)^{-1} X^T W y$ for local modeling
- **Kernel Functions**: Gaussian, Epanechnikov, and other distance-based weighting schemes
- **Bandwidth Selection**: Cross-validation and adaptive methods for optimal locality

### Practical Implementation
- Vectorized computations for efficiency and numerical stability
- Mean squared error (MSE), mean absolute error (MAE), and R-squared calculations
- Real-world data visualization and interpretation
- Learning rate selection and convergence analysis
- Feature scaling and preprocessing considerations
- Model validation and assumption checking

### Mathematical Foundations
- **Linear Algebra**: Matrix operations, eigenvalues, condition numbers, and numerical stability
- **Optimization Theory**: Convex optimization, gradient methods, and convergence analysis
- **Geometric Interpretations**: Visual understanding of optimization landscapes and decision boundaries
- **Statistical Theory**: Gaussian distributions, likelihood functions, maximum likelihood estimation
- **Probability Theory**: Conditional distributions, error models, and uncertainty quantification
- **Matrix Calculus**: Derivatives with respect to vectors and matrices, chain rule applications
- **Numerical Methods**: Stability analysis, regularization, and computational complexity

### Advanced Topics
- **Regularization**: Ridge regression, Lasso, and elastic net for improved generalization
- **Cross-validation**: Model selection and hyperparameter tuning
- **Feature Engineering**: Polynomial features, interaction terms, and non-linear transformations
- **Model Diagnostics**: Residual analysis, assumption checking, and outlier detection
- **Computational Efficiency**: Approximate methods, parallelization, and scalability considerations

## Getting Started

1. **Read through the main notes**: Start with `01_linear_regression.md` for foundational concepts
2. **Study the optimization algorithms**: Explore `02_lms_algorithm.md` for iterative methods
3. **Explore the analytical solution**: Understand `03_normal_equations.md` for closed-form solutions
4. **Understand the statistical foundations**: Dive into `04_probabilistic_interpretation.md` for theoretical justification
5. **Study non-parametric approaches**: Learn about `05_locally_weighted_linear_regression.md` for local modeling
6. **Run the Python code examples**: Execute each `.py` file to see concepts in action
7. **Experiment with real data**: Use the housing price prediction examples
8. **Explore visualizations**: Study the plots and diagrams in the `img/` directory
9. **Practice with different methods**: Compare gradient descent variants and analytical solutions
10. **Advanced exploration**: Investigate regularization, cross-validation, and model selection

## Learning Path

### Beginner Level
- Start with linear regression fundamentals in `01_linear_regression.md`
- Run basic examples in `linear_regression_examples.py`
- Understand cost functions and optimization objectives

### Intermediate Level
- Study gradient descent algorithms in `02_lms_algorithm.md`
- Implement and compare different optimization methods
- Explore analytical solutions in `03_normal_equations.md`

### Advanced Level
- Understand probabilistic foundations in `04_probabilistic_interpretation.md`
- Study non-parametric methods in `05_locally_weighted_linear_regression.md`
- Implement advanced topics like regularization and cross-validation

### Expert Level
- Explore computational complexity and scalability
- Investigate alternative optimization methods
- Study extensions to other linear models and generalized linear models

## Materials Include
- **Comprehensive lecture notes** with mathematical derivations and practical insights
- **Separate Python code files** with detailed annotations and educational examples
- **Optimization algorithms** with convergence analysis and performance comparisons
- **Real-world applications** with data visualization and interpretation
- **Advanced topics** including regularization, cross-validation, and model selection
- **Interactive examples** with step-by-step explanations and mathematical foundations
- **Performance analysis** with computational complexity and scalability considerations 
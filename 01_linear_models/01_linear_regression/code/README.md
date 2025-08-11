# Linear Regression Code Examples

This directory contains Python code examples that demonstrate the concepts covered in the Linear Regression chapter.

## Files Overview

### Core Demonstration Files

- **`linear_vs_nonlinear_demo.py`** - Demonstrates when linear models work well vs. when they don't
  - Shows linear, quadratic, and exponential relationships
  - Compares R² scores for different relationship types
  - Visualizes model performance

- **`regression_vs_classification_demo.py`** - Compares regression and classification approaches
  - House price prediction (regression) vs. expensive/cheap classification
  - Shows different evaluation metrics (MSE vs. Accuracy)
  - Demonstrates probability outputs for classification

- **`multiple_features_demo.py`** - Linear regression with multiple features
  - Uses living area, bedrooms, and age to predict house prices
  - Shows coefficient interpretation
  - Demonstrates feature importance analysis

- **`loss_functions_demo.py`** - Different loss functions and their properties
  - Squared Error, Absolute Error, and Huber Loss
  - Shows derivatives and penalty structures
  - Compares robustness to outliers

- **`optimization_approaches_demo.py`** - Compares analytical vs. iterative optimization
  - Normal equations vs. gradient descent performance
  - Computational time and memory usage analysis
  - Trade-offs between different optimization methods

- **`gradient_descent_visualization_demo.py`** - Comprehensive gradient descent visualization
  - 2D contour plots with optimization paths
  - Cost convergence analysis from different starting points
  - Learning rate effects on convergence
  - 3D surface plots of cost functions

- **`learning_rate_effects_demo.py`** - Learning rate analysis and effects
  - Parameter convergence for different learning rates
  - Cost function behavior with various step sizes
  - Convergence speed analysis and comparison
  - Guidelines for choosing learning rates

### Utility Files

- **`run_all_demos.py`** - Main script to run all demonstrations
- **`requirements.txt`** - Python package dependencies
- **`environment.yaml`** - Conda environment specification

## Running the Code

### Option 1: Run All Demonstrations

```bash
python run_all_demos.py
```

### Option 2: Run Individual Demonstrations

```bash
# Linear vs. Non-linear relationships
python linear_vs_nonlinear_demo.py

# Regression vs. Classification
python regression_vs_classification_demo.py

# Multiple features
python multiple_features_demo.py

# Loss functions
python loss_functions_demo.py

# Optimization approaches
python optimization_approaches_demo.py

# Gradient descent visualization
python gradient_descent_visualization_demo.py

# Learning rate effects
python learning_rate_effects_demo.py
```

### Option 3: Import and Use in Jupyter Notebook

```python
from linear_vs_nonlinear_demo import demonstrate_linear_vs_nonlinear
from regression_vs_classification_demo import demonstrate_regression_vs_classification
from multiple_features_demo import demonstrate_multiple_features
from loss_functions_demo import demonstrate_loss_functions
from optimization_approaches_demo import demonstrate_optimization_approaches
from gradient_descent_visualization_demo import demonstrate_gradient_descent_visualization
from learning_rate_effects_demo import demonstrate_learning_rate_effects

# Run any demonstration
r2_linear, r2_quadratic, r2_exponential = demonstrate_linear_vs_nonlinear()
```

## Dependencies

Required Python packages:
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `scikit-learn` - Machine learning algorithms

Install with:
```bash
pip install numpy matplotlib scikit-learn
```

Or using conda:
```bash
conda install numpy matplotlib scikit-learn
```

## Code Structure

Each demonstration file follows a consistent structure:

1. **Imports** - Required libraries
2. **Main function** - Contains the demonstration logic
3. **Data generation** - Creates synthetic data for examples
4. **Model fitting** - Trains linear regression models
5. **Analysis** - Evaluates model performance
6. **Visualization** - Creates plots to illustrate concepts
7. **Output** - Prints insights and results

## Key Concepts Demonstrated

- **Linear vs. Non-linear Relationships**: When linear models work and when they don't
- **Problem Types**: Regression vs. Classification differences
- **Multiple Features**: Extending linear regression to multiple input variables
- **Loss Functions**: Different ways to measure prediction errors
- **Model Interpretation**: Understanding coefficients and their meanings
- **Visualization**: Plotting data and model fits
- **Optimization Methods**: Analytical vs. iterative approaches
- **Gradient Descent**: Understanding the optimization algorithm
- **Learning Rates**: How step size affects convergence
- **Convergence Analysis**: Studying optimization behavior

## Learning Objectives

After running these demonstrations, you should understand:

1. The assumptions and limitations of linear regression
2. How to interpret model coefficients
3. The difference between regression and classification problems
4. Why we use squared error as the default loss function
5. How to evaluate model performance
6. The importance of data visualization in understanding models
7. When to use analytical vs. iterative optimization methods
8. How gradient descent works and why it's important
9. The critical role of learning rate in optimization
10. How to analyze convergence behavior and choose appropriate parameters

## Notes

- All demonstrations use synthetic data for clarity
- Random seeds are set for reproducible results
- Code includes detailed comments explaining each step
- Visualizations are designed to illustrate key concepts
- Functions return values that can be used for further analysis

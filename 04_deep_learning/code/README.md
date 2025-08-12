# Deep Learning Code Examples

This directory contains Python code examples that complement the deep learning course materials. Each file focuses on specific concepts and provides hands-on demonstrations.

## Code Files

### Core Concepts

- **`linear_vs_nonlinear_demo.py`** - Demonstrates why non-linear models are necessary by comparing linear and non-linear classifiers on non-linear data
- **`mse_properties_demo.py`** - Shows the properties of Mean Squared Error (MSE) loss function and compares it with Mean Absolute Error (MAE)
- **`loss_functions_comparison.py`** - Compares different regression loss functions (MSE, MAE, Huber) on data with outliers
- **`mse_calculation_demo.py`** - Step-by-step demonstration of MSE calculation with a simple linear model

### Neural Networks and Deep Learning

- **`neural_networks_code_examples.py`** - Comprehensive examples of neural network implementations
- **`backpropagation_examples.py`** - Backpropagation algorithm demonstrations and implementations
- **`modules_examples.py`** - Neural network module implementations and examples
- **`non_linear_models_equations.py`** - Mathematical implementations of non-linear models
- **`vectorization_examples.py`** - Vectorized implementations for efficient computation

### Jupyter Notebooks

- **`Perceptron.ipynb`** - Interactive notebook demonstrating perceptron learning

## Environment Setup

The required dependencies are specified in:
- **`environment.yaml`** - Conda environment file
- **`requirements.txt`** - Pip requirements file

## Usage

Each Python file can be run independently:

```bash
# Example: Run the linear vs nonlinear demonstration
python linear_vs_nonlinear_demo.py

# Example: Run MSE properties demonstration
python mse_properties_demo.py
```

Most files include visualization code that will display plots when run. Make sure you have a display environment set up for matplotlib.

## Key Learning Objectives

1. **Understanding Non-linearity**: See why linear models fail on non-linear data
2. **Loss Functions**: Compare different loss functions and their properties
3. **Model Evaluation**: Learn how to calculate and interpret model performance
4. **Visualization**: Understand model behavior through plots and visualizations

## Dependencies

- numpy
- matplotlib
- scikit-learn
- pandas (for some examples)

Install using:
```bash
pip install -r requirements.txt
```

Or with conda:
```bash
conda env create -f environment.yaml
```

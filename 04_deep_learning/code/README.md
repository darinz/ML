# Deep Learning Code Examples

This directory contains Python code examples that complement the deep learning course materials. Each file focuses on specific concepts and provides hands-on demonstrations.

## Code Files

### Core Concepts

- **`linear_vs_nonlinear_demo.py`** - Demonstrates why non-linear models are necessary by comparing linear and non-linear classifiers on non-linear data
- **`mse_properties_demo.py`** - Shows the properties of Mean Squared Error (MSE) loss function and compares it with Mean Absolute Error (MAE)
- **`loss_functions_comparison.py`** - Compares different regression loss functions (MSE, MAE, Huber) on data with outliers
- **`mse_calculation_demo.py`** - Step-by-step demonstration of MSE calculation with a simple linear model

### Neural Network Fundamentals

- **`neural_network_basics_demo.py`** - Demonstrates basic neural network concepts by comparing different architectures (single layer, one hidden layer, two hidden layers)
- **`single_neuron_demo.py`** - Shows how a single neuron works with different activation functions and house price prediction
- **`linear_vs_nonlinear_activation_demo.py`** - Demonstrates why non-linear activation functions are necessary by comparing linear and non-linear models
- **`activation_functions_demo.py`** - Comprehensive comparison of activation functions (ReLU, Sigmoid, Tanh) and their derivatives
- **`house_price_prediction_demo.py`** - Complete implementation of single neuron for house price prediction with evaluation and visualization

### Neural Network Modules

- **`modular_design_demo.py`** - Demonstrates the power of modular design by comparing simple linear models with modular neural networks
- **`matrix_multiplication_demo.py`** - Shows the fundamental matrix multiplication module for linear transformations
- **`advanced_activation_functions_demo.py`** - Comprehensive comparison of activation functions including ReLU, Sigmoid, Tanh, and GELU
- **`module_composition_demo.py`** - Demonstrates how different modules can be composed to create complex architectures

### Backpropagation and Training

- **`backpropagation_necessity_demo.py`** - Demonstrates why backpropagation is essential by comparing random search with gradient descent
- **`computational_graph_demo.py`** - Shows how computational graphs work with forward and backward passes
- **`automatic_differentiation_demo.py`** - Compares forward mode and reverse mode automatic differentiation

### Vectorization and Computational Efficiency

- **`vectorization_benefits_demo.py`** - Demonstrates dramatic performance benefits of vectorization with benchmarks
- **`basic_vectorization_demo.py`** - Shows basic vectorization concepts using neural network forward pass
- **`matrix_conventions_demo.py`** - Demonstrates column-major vs row-major matrix conventions
- **`broadcasting_demo.py`** - Shows how broadcasting enables efficient operations between different shapes

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
5. **Neural Network Basics**: Understand how neurons work and how they combine to form networks
6. **Activation Functions**: Learn about different activation functions and when to use each
7. **Single Neuron Applications**: See how single neurons can solve real-world problems
8. **Modular Design**: Learn how to build complex architectures from simple, reusable modules
9. **Module Composition**: Understand how to combine modules to create sophisticated neural networks
10. **Backpropagation**: Understand the fundamental algorithm that enables neural network training
11. **Computational Graphs**: Learn how data and gradients flow through neural networks
12. **Automatic Differentiation**: Understand the technology that makes backpropagation work
13. **Vectorization**: Learn how to replace loops with efficient matrix operations
14. **Matrix Conventions**: Understand column-major vs row-major data organization
15. **Broadcasting**: Learn how automatic shape expansion enables efficient operations

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

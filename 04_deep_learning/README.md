# Deep Learning: Foundations and Modern Applications

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-red.svg)](https://scikit-learn.org/)

Comprehensive introduction to deep learning, covering mathematical foundations, neural network architectures, and efficient implementation techniques.

## Overview

Deep learning enables machines to learn complex patterns directly from raw data through hierarchical representations, representing a paradigm shift in artificial intelligence.

## Materials

### Theory
- **[01_non-linear_models.md](01_non-linear_models.md)** - Function composition, loss functions (MSE, Cross-Entropy), activation functions, optimization
- **[02_neural_networks.md](02_neural_networks.md)** - Single neurons, multi-layer architectures, biological inspiration, feature learning
- **[03_modules.md](03_modules.md)** - Matrix multiplication, layer normalization, convolutions, residual connections, module composition

### Implementation
- **[non_linear_models_equations.py](non_linear_models_equations.py)** - Complete `NonLinearModels` class with loss functions, activations, optimization
- **[neural_networks_code_examples.py](neural_networks_code_examples.py)** - Progressive complexity from single neurons to deep networks
- **[modules_examples.py](modules_examples.py)** - All module implementations with scale invariance demonstrations
- **[backpropagation_examples.py](backpropagation_examples.py)** - Step-by-step chain rule, MLP forward/backward passes
- **[vectorization_examples.py](vectorization_examples.py)** - Performance benchmarks, broadcasting, vectorized neural networks

### Visualizations
- **img/** - Activation functions, backpropagation diagrams, neural network architectures

## Key Concepts

### Neural Network Forward Pass
```math
a^{[0]} = x \\
z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]} \\
a^{[l]} = \sigma(z^{[l]})
```

### Backpropagation Algorithm
```math
\frac{\partial J}{\partial W^{[l]}} = \frac{\partial J}{\partial z^{[l]}} \cdot (a^{[l-1]})^T \\
\frac{\partial J}{\partial b^{[l]}} = \sum_i \frac{\partial J}{\partial z_i^{[l]}}
```

### Vectorized Operations
```math
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]} \\
A^{[l]} = \sigma(Z^{[l]})
```

### Loss Functions
**Mean Squared Error**: $J = \frac{1}{2m} \sum_{i=1}^m (y^{(i)} - \hat{y}^{(i)})^2$

**Binary Cross-Entropy**: $J = -\frac{1}{m} \sum_{i=1}^m [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$

## Applications

- **Computer Vision**: Image classification, object detection, medical imaging
- **NLP**: Machine translation, text generation, sentiment analysis
- **Speech & Audio**: Speech recognition, music generation, voice assistants
- **Healthcare**: Disease detection, drug discovery, medical image analysis
- **Robotics**: Game playing, autonomous systems, robot control

## Getting Started

1. Read `01_non-linear_models.md` for mathematical foundations
2. Study `02_neural_networks.md` for basic architectures
3. Learn `03_modules.md` for modern building blocks
4. Run Python examples to see algorithms in action
5. Explore visualizations in `img/` folder

## Prerequisites

- Linear algebra (matrices, vectors, matrix multiplication)
- Calculus (derivatives, chain rule)
- Python programming and NumPy
- Basic machine learning concepts

## Installation

```bash
pip install numpy matplotlib scikit-learn scipy
```

## Running Examples

```bash
python non_linear_models_equations.py
python neural_networks_code_examples.py
python modules_examples.py
python backpropagation_examples.py
python vectorization_examples.py
``` 
# Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

This section covers the fundamentals of deep learning, from basic neural networks to modern architectures and optimization techniques. Deep learning is a subfield of machine learning that focuses on learning data representations through neural networks with many layers, enabling breakthroughs in computer vision, natural language processing, and other domains.

## Table of Contents

1. [Non-Linear Models](#non-linear-models)
2. [Neural Networks](#neural-networks)
3. [Modern Neural Network Modules](#modern-neural-network-modules)
4. [Backpropagation](#backpropagation)
5. [Vectorization](#vectorization)

## Learning Objectives

By the end of this section, you will understand:
- How to formulate supervised learning problems with non-linear models
- The mathematical foundations of neural networks
- Modern neural network architectures and building blocks
- The backpropagation algorithm for efficient gradient computation
- Vectorization techniques for scalable implementations

## Content Overview

### 1. Non-Linear Models (`01_non-linear_models.md`)

**Key Concepts:**
- Moving beyond linear models to capture complex relationships
- Loss functions for regression, binary classification, and multi-class classification
- Mathematical foundations for non-linear supervised learning

**Topics Covered:**
- **Regression Problems**: Mean squared error loss and alternatives
- **Binary Classification**: Logistic function, binary cross-entropy loss
- **Multi-class Classification**: Softmax function, categorical cross-entropy loss

**Files:**
- `01_non-linear_models.md` - Comprehensive theory and mathematical foundations
- `non_linear_models_equations.py` - Practical implementations and examples

### 2. Neural Networks (`02_neural_networks.md`)

**Key Concepts:**
- Building neural networks from simple mathematical operations
- Single neurons to multi-layer architectures
- Biological inspiration and practical applications

**Topics Covered:**
- **Single Neuron Networks**: ReLU activation, housing price prediction
- **Multi-Layer Networks**: Stacking neurons, hierarchical feature learning
- **Fully-Connected Networks**: Matrix formulations and implementations
- **Biological Inspiration**: Connection to biological neural networks

**Files:**
- `02_neural_networks.md` - Theory and mathematical foundations
- `neural_networks_code_examples.py` - Practical implementations and examples

### 3. Modern Neural Network Modules (`03_modules.md`)

**Key Concepts:**
- Modular design of modern neural networks
- Advanced building blocks beyond basic matrix multiplication
- Architectures used in state-of-the-art models

**Topics Covered:**
- **Matrix Multiplication Modules**: Basic building blocks
- **Residual Connections**: ResNet architecture and skip connections
- **Layer Normalization**: Scale-invariant normalization techniques
- **Convolutional Layers**: 1D and 2D convolutions for structured data

**Files:**
- `03_modules.md` - Theory and mathematical foundations
- `modules_examples.py` - Practical implementations and examples

### 4. Backpropagation (`04_backpropagation.md`)

**Key Concepts:**
- Efficient gradient computation for neural networks
- Chain rule applications in deep learning
- Auto-differentiation and computational graphs

**Topics Covered:**
- **Chain Rule Review**: Mathematical foundations for gradient computation
- **Backward Functions**: Computing gradients for basic modules
- **MLP Backpropagation**: Complete algorithm for multi-layer perceptrons
- **Computational Efficiency**: Time complexity analysis

**Files:**
- `04_backpropagation.md` - Comprehensive theory and algorithms
- `backpropagation_examples.py` - Step-by-step implementations

### 5. Vectorization (`05_vectorization.md`)

**Key Concepts:**
- Parallel computation across training examples
- Matrix operations for efficient neural network implementation
- Broadcasting and numerical optimization

**Topics Covered:**
- **Forward Pass Vectorization**: Matrix formulations for multiple examples
- **Backward Pass Vectorization**: Efficient gradient computation
- **Broadcasting**: Automatic dimension handling
- **Implementation Conventions**: Row-major vs column-major formats

**Files:**
- `05_vectorization.md` - Theory and mathematical foundations
- `vectorization_examples.py` - Practical implementations and examples

## Visual Resources

The `img/` directory contains visual aids for understanding the concepts:

- `activation_functions.png` - Common activation functions and their properties
- `backprop_figure.png` - Illustration of the backpropagation algorithm
- `housing_price.png` - Housing price prediction with neural networks
- `mlp_resnet.png` - Comparison of MLP and ResNet architectures
- `nn_housing_diagram.png` - Neural network architecture for housing prediction

## Getting Started

### Prerequisites
- Understanding of linear algebra (matrices, vectors, matrix multiplication)
- Familiarity with calculus (derivatives, chain rule)
- Basic Python programming skills
- NumPy for numerical computations

### Recommended Learning Path
1. Start with **Non-Linear Models** to understand the motivation for deep learning
2. Move to **Neural Networks** to learn the basic building blocks
3. Study **Backpropagation** to understand how neural networks learn
4. Explore **Modern Modules** to see advanced architectures
5. Finish with **Vectorization** to learn efficient implementations

### Running the Examples
All Python files contain runnable examples. Make sure you have NumPy installed:

```bash
pip install numpy matplotlib
```

Then run any example file:
```bash
python neural_networks_code_examples.py
```

## Key Mathematical Concepts

### Neural Network Forward Pass
For a multi-layer perceptron with $L$ layers:

```math
a^{[0]} = x \\
z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]} \\
a^{[l]} = \sigma(z^{[l]})
```

### Backpropagation
The gradient computation follows the chain rule:

```math
\frac{\partial J}{\partial W^{[l]}} = \frac{\partial J}{\partial z^{[l]}} \cdot (a^{[l-1]})^T
```

### Vectorized Operations
For $m$ training examples:
```math
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}
```

## Applications and Impact

Deep learning has revolutionized numerous fields:

- **Computer Vision**: Image classification, object detection, segmentation
- **Natural Language Processing**: Machine translation, text generation, sentiment analysis
- **Speech Recognition**: Voice assistants, transcription services
- **Game Playing**: AlphaGo, OpenAI Five, game AI
- **Medical Diagnosis**: Disease detection, medical image analysis
- **Autonomous Systems**: Self-driving cars, robotics

## Further Reading

### Foundational Papers
- "Deep Learning" by LeCun, Bengio, and Hinton (Nature, 2015)
- "ImageNet Classification with Deep Convolutional Neural Networks" by Krizhevsky et al. (2012)
- "Deep Residual Learning for Image Recognition" by He et al. (2016)

### Books
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- "Neural Networks and Deep Learning" by Michael Nielsen
- "Pattern Recognition and Machine Learning" by Christopher Bishop

### Online Resources
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning)

## Contributing

This section is part of a comprehensive machine learning curriculum. If you find errors or have suggestions for improvements, please contribute by:

1. Creating detailed issue reports
2. Suggesting additional examples or visualizations
3. Improving mathematical explanations
4. Adding practical applications and case studies

---

**Next Steps**: After completing this section, you'll be ready to explore advanced topics like convolutional neural networks, recurrent neural networks, transformers, and practical deep learning applications. 
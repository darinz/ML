# Deep Learning: Hands-On Learning Guide

[![Neural Networks](https://img.shields.io/badge/Neural%20Networks-Deep%20Learning-blue.svg)](https://en.wikipedia.org/wiki/Artificial_neural_network)
[![Backpropagation](https://img.shields.io/badge/Backpropagation-Gradient%20Descent-green.svg)](https://en.wikipedia.org/wiki/Backpropagation)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Hands-on Learning](https://img.shields.io/badge/Learning-Hands--on%20Experience-green.svg)](https://en.wikipedia.org/wiki/Experiential_learning)

## From Linear Models to Deep Neural Networks

We've explored the elegant framework of **deep learning**, which represents a paradigm shift in artificial intelligence by enabling machines to learn complex patterns directly from raw data through hierarchical representations. Deep learning extends beyond linear models to capture intricate, non-linear relationships through neural networks with multiple layers of computation.

However, true understanding comes from **hands-on implementation**. This practical guide will help you translate the theoretical concepts into working code, experiment with different neural network architectures, and develop the intuition needed to apply these powerful algorithms to real-world problems.

## From Computational Efficiency to Hands-On Mastery

We've now explored **vectorization** - the techniques that enable efficient computation by leveraging parallel processing and optimized matrix operations. We've seen how vectorization can dramatically speed up both forward and backward passes, making deep learning practical for real-world applications.

However, while we've established the theoretical foundations and computational techniques, true mastery of deep learning comes from **hands-on implementation**. Understanding the mathematical principles and optimization techniques is essential, but implementing neural networks from scratch, experimenting with different architectures, and applying them to real-world problems is where the concepts truly come to life.

This motivates our exploration of **hands-on coding** - the practical implementation of all the concepts we've learned. We'll put our theoretical knowledge into practice by implementing neural networks from scratch, experimenting with different architectures, and developing the practical skills needed to apply deep learning to real-world problems.

The transition from computational efficiency to practical implementation represents the bridge from theoretical understanding to practical mastery - taking our knowledge of how deep learning works and turning it into working systems that can solve real problems.

In this practical guide, we'll implement complete neural network systems, experiment with different architectures and optimization techniques, and develop the practical skills needed for deep learning applications.

## Learning Objectives

By completing this hands-on learning guide, you will:

1. **Master neural network fundamentals** through interactive implementations from single neurons to deep architectures
2. **Implement backpropagation** and understand gradient flow through networks
3. **Build neural network modules** and understand modern deep learning building blocks
4. **Apply vectorization techniques** for efficient computation
5. **Understand non-linear transformations** and activation functions
6. **Develop intuition for deep learning** through practical experimentation

## Quick Start

### Prerequisites
- Basic Python knowledge (variables, functions, arrays)
- Familiarity with linear algebra (vectors, matrices, matrix multiplication)
- Understanding of calculus (derivatives, chain rule)
- Completion of linear models and classification modules (recommended)

### Estimated Time
- **Setup**: 30 minutes
- **Lesson 1**: 3-4 hours
- **Lesson 2**: 4-5 hours
- **Lesson 3**: 3-4 hours
- **Lesson 4**: 3-4 hours
- **Total**: 14-18 hours

---

## Environment Setup

### Option 1: Using Conda (Recommended)

#### Step 1: Install Miniconda
```bash
# Download Miniconda for your OS
# Windows: https://docs.conda.io/en/latest/miniconda.html
# macOS: https://docs.conda.io/en/latest/miniconda.html
# Linux: https://docs.conda.io/en/latest/miniconda.html

# Verify installation
conda --version
```

#### Step 2: Create Environment
```bash
# Navigate to the deep learning directory
cd 04_deep_learning

# Create a new conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate deep-learning-lesson

# Verify installation
python -c "import numpy, matplotlib, scipy, sklearn; print('All packages installed successfully!')"
```

### Option 2: Using pip

#### Step 1: Create Virtual Environment
```bash
# Navigate to the deep learning directory
cd 04_deep_learning

# Create virtual environment
python -m venv deep-learning-env

# Activate environment
# On Windows:
deep-learning-env\Scripts\activate
# On macOS/Linux:
source deep-learning-env/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import numpy, matplotlib, scipy, sklearn; print('All packages installed successfully!')"
```

### Option 3: Using Jupyter Notebooks

#### Step 1: Install Jupyter
```bash
# After setting up environment above
pip install jupyter notebook

# Launch Jupyter
jupyter notebook
```

#### Step 2: Create New Notebook
```python
# In a new notebook cell, import required packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.datasets import make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
np.random.seed(42)  # For reproducible results
```

---

## Lesson Structure

### Lesson 1: Neural Network Fundamentals (3-4 hours)
**File**: `neural_networks_code_examples.py`

#### Learning Goals
- Understand single neuron models and their mathematical foundations
- Master multi-layer perceptrons and forward propagation
- Implement activation functions and understand their properties
- Build complete neural network training pipelines
- Visualize network behavior and decision boundaries

#### Hands-On Activities

**Activity 1.1: Single Neuron with ReLU**
```python
# Implement single neuron regression with ReLU activation
from neural_networks_code_examples import NeuralNetworkExamples

# Create neural network examples instance
nn = NeuralNetworkExamples()

# Train single neuron with ReLU activation
w, b = nn.single_neuron_regression_relu(visualize=True)

print(f"Learned parameters: w={w:.4f}, b={b:.4f}")

# Key insight: Single neurons can learn non-linear relationships through activation functions
```

**Activity 1.2: Two-Layer Neural Network**
```python
# Implement two-layer neural network for classification
# This demonstrates the power of adding hidden layers

# Train two-layer network
W1, b1, W2, b2 = nn.two_layer_neural_network(visualize=True)

print(f"Layer 1 weights shape: {W1.shape}")
print(f"Layer 1 bias shape: {b1.shape}")
print(f"Layer 2 weights shape: {W2.shape}")
print(f"Layer 2 bias shape: {b2.shape}")

# Key insight: Hidden layers enable learning of complex decision boundaries
```

**Activity 1.3: Activation Functions Comparison**
```python
# Compare different activation functions and their properties
nn.activation_functions_comparison()

# Key insights:
# - ReLU: Most popular, addresses vanishing gradient problem
# - Sigmoid: Smooth, bounded, good for probabilities
# - Tanh: Zero-centered, often better than sigmoid
# - Each has different gradient properties and use cases
```

**Activity 1.4: Fully Connected Network**
```python
# Build a fully connected network with multiple hidden layers
W1, W2, W3, b = nn.fully_connected_network(visualize=True)

print(f"Network architecture:")
print(f"  Input -> Hidden 1: {W1.shape}")
print(f"  Hidden 1 -> Hidden 2: {W2.shape}")
print(f"  Hidden 2 -> Output: {W3.shape}")

# Key insight: Deeper networks can learn more complex representations
```

#### Experimentation Tasks
1. **Experiment with different activation functions**: Try ReLU, Sigmoid, Tanh on the same data
2. **Vary network depth**: Compare 1, 2, 3, and 4 layer networks
3. **Test different learning rates**: Observe convergence behavior
4. **Visualize decision boundaries**: See how networks partition the input space

#### Check Your Understanding
- [ ] Can you explain how a single neuron computes its output?
- [ ] Do you understand why activation functions are necessary?
- [ ] Can you implement forward propagation through multiple layers?
- [ ] Do you see how network depth affects representational power?

---

### Lesson 2: Backpropagation and Gradient Flow (4-5 hours)
**File**: `backpropagation_examples.py`

#### Learning Goals
- Understand function composition and the chain rule
- Master backpropagation algorithm for gradient computation
- Implement automatic differentiation concepts
- Analyze gradient flow through networks
- Build complete forward and backward passes

#### Hands-On Activities

**Activity 2.1: Function Composition**
```python
# Understand how complex functions are built from simple ones
from backpropagation_examples import BackpropagationExamples

# Create backpropagation examples instance
bp = BackpropagationExamples()

# Demonstrate function composition
u, J = bp.function_composition_example()
print(f"Intermediate result u: {u}")
print(f"Final result J: {J}")

# Key insight: Neural networks are compositions of simple functions
```

**Activity 2.2: Chain Rule for Vector Functions**
```python
# Implement chain rule for computing gradients
gradient = bp.chain_rule_vector_functions()
print(f"Gradient ∂J/∂z: {gradient}")

# Key insight: Chain rule enables efficient gradient computation
```

**Activity 2.3: Matrix Multiplication Backward**
```python
# Understand backpropagation through matrix operations
gradient = bp.matrix_multiplication_backward()
print(f"Matrix multiplication gradient: {gradient}")

# Key insight: Matrix operations have specific gradient patterns
```

**Activity 2.4: ReLU Backward Pass**
```python
# Implement backpropagation through ReLU activation
gradient = bp.relu_backward()
print(f"ReLU gradient: {gradient}")

# Key insight: ReLU gradient is simple but powerful
```

**Activity 2.5: Complete MLP Forward-Backward**
```python
# Build complete forward and backward passes for a multi-layer perceptron
loss, gradients = bp.full_mlp_forward_backward()
print(f"Final loss: {loss:.6f}")
print(f"Number of gradient tensors: {len(gradients)}")

# Key insight: Backpropagation computes all gradients in one pass
```

#### Experimentation Tasks
1. **Trace gradient flow**: Follow gradients through different network architectures
2. **Experiment with different loss functions**: Compare gradients for MSE vs cross-entropy
3. **Analyze vanishing/exploding gradients**: See how they affect training
4. **Test automatic differentiation**: Compare with manual gradient computation

#### Check Your Understanding
- [ ] Can you explain the chain rule and its role in backpropagation?
- [ ] Do you understand how gradients flow backward through networks?
- [ ] Can you implement backpropagation for common operations?
- [ ] Do you see why backpropagation is computationally efficient?

---

### Lesson 3: Neural Network Modules (3-4 hours)
**File**: `modules_examples.py`

#### Learning Goals
- Understand modular design principles in neural networks
- Master matrix multiplication and linear transformations
- Implement layer normalization and its properties
- Build convolutional modules for spatial data
- Create residual connections and understand their benefits

#### Hands-On Activities

**Activity 3.1: Matrix Multiplication Module**
```python
# Implement the fundamental matrix multiplication module
from modules_examples import NeuralNetworkModules

# Create modules instance
modules = NeuralNetworkModules()

# Test matrix multiplication
x = np.array([1.0, 2.0])
W = np.array([[1.0, 0.5], [0.5, 1.0]])
b = np.array([0.1, -0.2])

output = modules.matrix_multiplication(x, W, b)
print(f"Input: {x}")
print(f"Weights: {W}")
print(f"Bias: {b}")
print(f"Output: {output}")

# Key insight: Matrix multiplication is the core operation in neural networks
```

**Activity 3.2: Layer Normalization**
```python
# Implement layer normalization for training stability
z = np.array([1.0, 2.0, 3.0, 4.0])
normalized = modules.layer_normalization(z)
print(f"Input: {z}")
print(f"Normalized: {normalized}")
print(f"Mean: {np.mean(normalized):.6f}")
print(f"Std: {np.std(normalized):.6f}")

# Key insight: Normalization helps stabilize training
```

**Activity 3.3: Scale Invariance Demonstration**
```python
# Understand scale invariance properties
modules.demonstrate_scale_invariance()

# Key insight: Scale invariance is a powerful property for generalization
```

**Activity 3.4: Convolutional Modules**
```python
# Implement 1D and 2D convolutions
# 1D convolution
z_1d = np.array([1, 2, 3, 4, 5])
w_1d = np.array([0.5, 1.0, 0.5])
conv_1d = modules.convolution_1d_simple(z_1d, w_1d)
print(f"1D Input: {z_1d}")
print(f"1D Kernel: {w_1d}")
print(f"1D Convolution: {conv_1d}")

# Key insight: Convolutions capture local spatial patterns
```

**Activity 3.5: Residual Connections**
```python
# Implement residual connections for deep networks
def simple_layer(x):
    return 0.5 * x + 1.0

x = np.array([1.0, 2.0, 3.0])
residual_output = modules.residual_connection(x, simple_layer)
print(f"Input: {x}")
print(f"Residual output: {residual_output}")

# Key insight: Residual connections enable training of very deep networks
```

#### Experimentation Tasks
1. **Experiment with different normalization techniques**: Compare layer norm, batch norm, instance norm
2. **Test convolutional kernels**: Try different kernel sizes and patterns
3. **Analyze residual connections**: See how they affect gradient flow
4. **Build module compositions**: Combine different modules into complex architectures

#### Check Your Understanding
- [ ] Can you explain why modular design is important in neural networks?
- [ ] Do you understand the mathematical properties of layer normalization?
- [ ] Can you implement basic convolutional operations?
- [ ] Do you see how residual connections help with deep networks?

---

### Lesson 4: Vectorization and Efficiency (3-4 hours)
**File**: `vectorization_examples.py`

#### Learning Goals
- Understand the importance of vectorization in deep learning
- Master efficient matrix operations for neural networks
- Implement batch processing for training
- Compare vectorized vs loop-based implementations
- Optimize neural network computations

#### Hands-On Activities

**Activity 4.1: Vectorization Comparison**
```python
# Compare vectorized vs loop-based neural network implementations
from neural_networks_code_examples import NeuralNetworkExamples

# Test vectorization performance
speedup = nn.vectorization_comparison()
print(f"Vectorized implementation is {speedup:.1f}x faster")

# Key insight: Vectorization dramatically improves computational efficiency
```

**Activity 4.2: Batch Processing**
```python
# Implement batch processing for efficient training
from vectorization_examples import VectorizationExamples

vec = VectorizationExamples()

# Test batch matrix multiplication
batch_size = 32
input_size = 100
output_size = 50

X_batch = np.random.randn(batch_size, input_size)
W = np.random.randn(output_size, input_size)
b = np.random.randn(output_size)

# Vectorized batch processing
output_batch = vec.batch_matrix_multiplication(X_batch, W, b)
print(f"Batch input shape: {X_batch.shape}")
print(f"Output batch shape: {output_batch.shape}")

# Key insight: Batch processing enables efficient GPU utilization
```

**Activity 4.3: Gradient Computation Optimization**
```python
# Optimize gradient computation through vectorization
gradients = vec.vectorized_gradient_computation()
print(f"Vectorized gradient computation completed")

# Key insight: Vectorized gradients are both faster and more numerically stable
```

#### Experimentation Tasks
1. **Profile different implementations**: Measure performance of vectorized vs loop-based code
2. **Experiment with batch sizes**: Find optimal batch size for your hardware
3. **Test memory efficiency**: Compare memory usage of different approaches
4. **Optimize for specific hardware**: Adapt code for CPU vs GPU

#### Check Your Understanding
- [ ] Can you explain why vectorization is crucial for deep learning?
- [ ] Do you understand how batch processing improves efficiency?
- [ ] Can you implement vectorized neural network operations?
- [ ] Do you see the connection between vectorization and hardware utilization?

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Vanishing/Exploding Gradients
```python
# Problem: Gradients become too small or too large during training
# Solution: Use proper weight initialization and normalization
def xavier_initialization(fan_in, fan_out):
    """Xavier/Glorot initialization for better gradient flow."""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))

def he_initialization(fan_in, fan_out):
    """He initialization for ReLU activations."""
    return np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
```

#### Issue 2: Overfitting in Deep Networks
```python
# Problem: Network memorizes training data but doesn't generalize
# Solution: Add regularization and early stopping
def add_dropout(activations, dropout_rate=0.5):
    """Add dropout for regularization."""
    mask = np.random.binomial(1, 1-dropout_rate, size=activations.shape)
    return activations * mask / (1 - dropout_rate)

def early_stopping(validation_losses, patience=5):
    """Implement early stopping to prevent overfitting."""
    if len(validation_losses) < patience:
        return False
    return all(validation_losses[-i] <= validation_losses[-i-1] 
               for i in range(1, patience+1))
```

#### Issue 3: Numerical Instability
```python
# Problem: Numerical overflow or underflow in computations
# Solution: Use stable implementations and proper scaling
def stable_softmax(x):
    """Numerically stable softmax implementation."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def stable_log_softmax(x):
    """Numerically stable log-softmax implementation."""
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    return x_shifted - np.log(np.sum(np.exp(x_shifted), axis=-1, keepdims=True))
```

#### Issue 4: Slow Training Convergence
```python
# Problem: Network takes too long to converge
# Solution: Use adaptive learning rates and momentum
def adam_optimizer(params, grads, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam optimizer for adaptive learning rates."""
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads ** 2)
    
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    params -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, m, v
```

#### Issue 5: Memory Issues with Large Networks
```python
# Problem: Network consumes too much memory
# Solution: Use gradient checkpointing and memory-efficient operations
def gradient_checkpointing(module, inputs):
    """Save memory by recomputing intermediate activations."""
    # Forward pass without storing intermediate activations
    outputs = module.forward_no_cache(inputs)
    
    # Backward pass with recomputation
    gradients = module.backward_recompute(inputs, outputs)
    
    return outputs, gradients
```

---

## Assessment and Progress Tracking

### Self-Assessment Checklist

#### Neural Network Fundamentals Level
- [ ] I can explain how single neurons compute their outputs
- [ ] I understand the role of activation functions in neural networks
- [ ] I can implement forward propagation through multiple layers
- [ ] I can visualize decision boundaries and network behavior

#### Backpropagation Level
- [ ] I can explain the chain rule and its role in backpropagation
- [ ] I understand how gradients flow backward through networks
- [ ] I can implement backpropagation for common operations
- [ ] I can analyze gradient flow and identify potential issues

#### Neural Network Modules Level
- [ ] I can implement basic neural network modules from scratch
- [ ] I understand the properties of layer normalization
- [ ] I can implement convolutional operations
- [ ] I can build modular neural network architectures

#### Vectorization and Efficiency Level
- [ ] I can implement vectorized neural network operations
- [ ] I understand the importance of batch processing
- [ ] I can optimize neural network computations
- [ ] I can profile and improve code performance

#### Practical Application Level
- [ ] I can design neural network architectures for different problems
- [ ] I can implement complete training pipelines
- [ ] I can handle common training issues and debugging
- [ ] I can apply deep learning to real-world problems

### Progress Tracking

#### Week 1: Neural Network Fundamentals
- **Goal**: Complete Lesson 1
- **Deliverable**: Working neural network implementations with visualizations
- **Assessment**: Can you implement neural networks and explain their behavior?

#### Week 2: Backpropagation and Gradient Flow
- **Goal**: Complete Lesson 2
- **Deliverable**: Complete backpropagation implementation with gradient analysis
- **Assessment**: Can you implement backpropagation and understand gradient flow?

#### Week 3: Neural Network Modules
- **Goal**: Complete Lesson 3
- **Deliverable**: Modular neural network components and architectures
- **Assessment**: Can you build neural network modules and understand their properties?

#### Week 4: Vectorization and Efficiency
- **Goal**: Complete Lesson 4
- **Deliverable**: Optimized neural network implementations
- **Assessment**: Can you implement efficient, vectorized neural network operations?

---

## Extension Projects

### Project 1: Image Classification with CNNs
**Goal**: Build a complete convolutional neural network for image classification

**Tasks**:
1. Implement 2D convolutional layers from scratch
2. Add pooling layers and normalization
3. Build complete CNN architecture
4. Train on image classification dataset
5. Add data augmentation and regularization

**Skills Developed**:
- Convolutional neural networks
- Image processing and computer vision
- Data augmentation techniques
- Model regularization and optimization

### Project 2: Recurrent Neural Networks
**Goal**: Build RNNs for sequential data processing

**Tasks**:
1. Implement RNN cells from scratch
2. Add LSTM and GRU variants
3. Build sequence-to-sequence models
4. Apply to text generation or time series prediction
5. Add attention mechanisms

**Skills Developed**:
- Recurrent neural networks
- Sequential data processing
- Attention mechanisms
- Natural language processing

### Project 3: Autoencoders and Generative Models
**Goal**: Build generative models for data synthesis

**Tasks**:
1. Implement autoencoder architectures
2. Add variational autoencoder (VAE) components
3. Build generative adversarial network (GAN) framework
4. Train on image or text data
5. Evaluate generation quality and diversity

**Skills Developed**:
- Generative models
- Unsupervised learning
- Latent space representations
- Model evaluation and comparison

---

## Additional Resources

### Books
- **"Deep Learning"** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- **"Neural Networks and Deep Learning"** by Michael Nielsen
- **"Pattern Recognition and Machine Learning"** by Christopher Bishop

### Online Courses
- **Coursera**: Deep Learning Specialization by Andrew Ng
- **edX**: Introduction to Deep Learning
- **MIT OpenCourseWare**: Introduction to Deep Learning

### Practice Datasets
- **MNIST**: Handwritten digit classification
- **CIFAR-10**: Color image classification
- **UCI Machine Learning Repository**: Various datasets for practice

### Advanced Topics
- **Transformers**: Attention-based architectures
- **Graph Neural Networks**: Deep learning on graph-structured data
- **Reinforcement Learning**: Deep RL and policy gradients
- **Federated Learning**: Distributed deep learning

---

## Conclusion: The Power of Deep Learning

Congratulations on completing this comprehensive journey through deep learning fundamentals! We've explored the mathematical foundations, practical implementations, and modern techniques that make deep learning one of the most powerful tools in artificial intelligence.

### The Complete Picture

**1. Neural Network Fundamentals** - We started with single neurons and built up to complex deep architectures, understanding how simple mathematical operations can learn complex patterns.

**2. Backpropagation and Gradient Flow** - We mastered the fundamental algorithm that enables training of deep networks through efficient gradient computation.

**3. Neural Network Modules** - We learned how modern deep learning is built using modular components that can be composed to create complex architectures.

**4. Vectorization and Efficiency** - We optimized our implementations for real-world performance and scalability.

### Key Insights

- **Function Composition**: Neural networks are compositions of simple functions
- **Gradient Flow**: Backpropagation enables efficient training of deep networks
- **Modular Design**: Modern deep learning relies on reusable, composable modules
- **Computational Efficiency**: Vectorization and optimization are crucial for practical applications
- **Representational Power**: Deep networks can learn hierarchical representations from data

### Looking Forward

This deep learning foundation prepares you for cutting-edge topics:
- **Computer Vision**: Convolutional neural networks and image understanding
- **Natural Language Processing**: Transformers and language models
- **Reinforcement Learning**: Deep RL and policy optimization
- **Generative Models**: GANs, VAEs, and diffusion models
- **Federated Learning**: Distributed and privacy-preserving deep learning

The principles we've learned here - neural networks, backpropagation, modular design, and efficient computation - will serve you well throughout your machine learning journey.

### Next Steps

1. **Apply deep learning** to your own complex problems
2. **Explore specialized architectures** like CNNs, RNNs, and Transformers
3. **Build a portfolio** of deep learning projects
4. **Contribute to open source** deep learning frameworks
5. **Continue learning** with more advanced deep learning techniques

Remember: Deep learning represents a fundamental shift in how we approach artificial intelligence, enabling machines to learn directly from data. Keep exploring, building, and applying these concepts to new problems!

---

**Previous: [Vectorization](05_vectorization.md)** - Understand how to optimize neural network computation through vectorization techniques.

**Next: [Generalization](../05_generalization/README.md)** - Explore bias-variance tradeoff, overfitting, and generalization in machine learning.

## Environment Files

### requirements.txt
```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0
jupyter>=1.0.0
notebook>=6.4.0
ipykernel>=6.0.0
nb_conda_kernels>=2.3.0
```

### environment.yaml
```yaml
name: deep-learning-lesson
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy>=1.21.0
  - matplotlib>=3.5.0
  - scipy>=1.7.0
  - scikit-learn>=1.0.0
  - pandas>=1.3.0
  - seaborn>=0.11.0
  - jupyter>=1.0.0
  - notebook>=6.4.0
  - pip
  - pip:
    - ipykernel
    - nb_conda_kernels
```

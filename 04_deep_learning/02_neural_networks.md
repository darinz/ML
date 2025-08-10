# Neural Networks: From Single Neurons to Deep Architectures

## Introduction to Neural Networks

Neural networks represent one of the most powerful and flexible approaches to machine learning, capable of learning complex patterns and relationships from data. At their core, neural networks are computational models inspired by biological neural systems, consisting of interconnected processing units (neurons) organized in layers.

### What Are Neural Networks?

Neural networks are mathematical models that can approximate any continuous function given sufficient capacity. They consist of:

1. **Input Layer**: Receives the raw data
2. **Hidden Layers**: Process and transform the data through non-linear operations
3. **Output Layer**: Produces the final prediction or classification

### Key Characteristics

- **Non-linear**: Can model complex, non-linear relationships
- **Universal**: Can approximate any continuous function (Universal Approximation Theorem)
- **Hierarchical**: Learn features at multiple levels of abstraction
- **Adaptive**: Parameters are learned from data through optimization

### Mathematical Foundation

A neural network can be viewed as a composition of functions:

$$
f(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)
$$

Where each $f_i$ represents a layer transformation, and $\circ$ denotes function composition.

## From Mathematical Foundations to Neural Network Architectures

We've now established the **mathematical foundations** of deep learning - understanding why non-linear models are necessary, how loss functions capture different types of learning objectives, and how optimization algorithms enable us to find the best parameters for our models. This theoretical framework provides the foundation for understanding how neural networks work.

However, while we've discussed non-linear models in abstract terms, we need to move from mathematical concepts to concrete **neural network architectures**. The transition from understanding why non-linear models are powerful to actually building them requires us to explore how simple computational units (neurons) can be combined to create complex learning systems.

This motivates our exploration of **neural networks** - the specific architectural framework that implements non-linear models through interconnected layers of neurons. We'll see how the mathematical principles we've established (non-linear transformations, function composition, optimization) translate into concrete neural network designs.

The transition from non-linear models to neural networks represents the bridge from mathematical theory to practical architecture - taking our understanding of why non-linear models work and turning it into a systematic approach for building them.

In this section, we'll explore how individual neurons work, how they can be combined into layers, and how these layers can be stacked to create deep architectures that can learn increasingly complex patterns.

---

## 7.2 From Linear to Non-Linear: The Single Neuron

### The Building Block: The Artificial Neuron

The artificial neuron is the fundamental computational unit of neural networks. It performs three basic operations:

1. **Linear Combination**: $z = w^T x + b$
2. **Non-linear Activation**: $a = \sigma(z)$
3. **Output**: The activated value becomes the neuron's output

### Mathematical Formulation

For a single neuron with input $x \in \mathbb{R}^d$:

$$
z = w^T x + b
a = \sigma(z)
$$

Where:
- $w \in \mathbb{R}^d$ is the weight vector
- $b \in \mathbb{R}$ is the bias term
- $\sigma: \mathbb{R} \rightarrow \mathbb{R}$ is the activation function
- $z$ is the pre-activation (or logit)
- $a$ is the activation (or output)

### Why Non-Linear Activation Functions?

**The Problem with Linear Activations**: If we used $\sigma(z) = z$ (linear activation), then:

$$
f(x) = w_2^T (W_1 x + b_1) + b_2 = (w_2^T W_1) x + (w_2^T b_1 + b_2) = W' x + b'
$$

This reduces to a linear function, losing the power of non-linearity.

**The Solution**: Non-linear activation functions introduce the ability to model complex, non-linear relationships.

### Common Activation Functions

#### 1. Rectified Linear Unit (ReLU)
$$
\sigma(z) = \max(0, z)
$$

**Properties**:
- **Range**: $[0, \infty)$
- **Derivative**: $\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$
- **Advantages**: Simple, computationally efficient, helps with vanishing gradient problem
- **Disadvantages**: Can cause "dying ReLU" problem (neurons stuck at zero)

#### 2. Sigmoid Function
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Properties**:
- **Range**: $(0, 1)$
- **Derivative**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- **Advantages**: Smooth, bounded output, interpretable as probability
- **Disadvantages**: Suffers from vanishing gradient problem

#### 3. Hyperbolic Tangent (tanh)
$$
\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

**Properties**:
- **Range**: $(-1, 1)$
- **Derivative**: $\sigma'(z) = 1 - \sigma(z)^2$
- **Advantages**: Zero-centered, bounded
- **Disadvantages**: Still suffers from vanishing gradient problem

### Single Neuron Example: Housing Price Prediction

Consider predicting house prices based on house size. A single neuron with ReLU activation can model the relationship:

$$
\hat{h}_\theta(x) = \max(w \cdot x + b, 0)
$$

Where:
- $x$ is the house size (square feet)
- $w$ is the price per square foot
- $b$ is the base price
- The ReLU ensures non-negative predictions

**Intuition**: The neuron learns to predict a price that increases linearly with size, but never goes below zero (which makes sense for house prices).

### Mathematical Analysis

**Why ReLU Works Well**:
1. **Non-linearity**: Introduces a "kink" at $x = -b/w$
2. **Sparsity**: Can produce exact zeros, leading to sparse representations
3. **Gradient Flow**: Simple derivative prevents vanishing gradients
4. **Computational Efficiency**: Simple max operation

**Parameter Learning**:
The parameters $w$ and $b$ are learned through gradient descent by minimizing a loss function (e.g., mean squared error):

$$
J(w, b) = \frac{1}{n} \sum_{i=1}^n (y^{(i)} - \hat{h}_\theta(x^{(i)}))^2
$$

---

## Stacking Neurons: Multi-Layer Networks

### The Power of Composition

While a single neuron can model simple non-linear relationships, real-world problems often require more complex functions. By stacking multiple neurons in layers, we can create networks that learn hierarchical representations.

### Mathematical Motivation

**Universal Approximation Theorem**: A neural network with a single hidden layer containing a sufficient number of neurons can approximate any continuous function on a compact domain to arbitrary precision.

**Intuition**: Just as any function can be approximated by a sum of basis functions, any function can be approximated by a combination of non-linear transformations.

### Two-Layer Network Architecture

A two-layer network consists of:
1. **Input Layer**: $x \in \mathbb{R}^d$
2. **Hidden Layer**: $h$ neurons with activations $a_1, a_2, \ldots, a_h$
3. **Output Layer**: Final prediction

#### Mathematical Formulation

For a two-layer network with $h$ hidden neurons:

**Hidden Layer**:
$$
z_j = w_j^T x + b_j, \quad j = 1, 2, \ldots, h
a_j = \sigma(z_j), \quad j = 1, 2, \ldots, h
$$

**Output Layer**:
$$
\hat{y} = w_{out}^T a + b_{out}
$$

Where:
- $w_j \in \mathbb{R}^d$ are the weights for the $j$-th hidden neuron
- $b_j \in \mathbb{R}$ are the biases for the $j$-th hidden neuron
- $a = [a_1, a_2, \ldots, a_h]^T$ is the vector of hidden activations
- $w_{out} \in \mathbb{R}^h$ and $b_{out} \in \mathbb{R}$ are the output layer parameters

#### Vectorized Form

We can write this more compactly using matrix notation:

**Hidden Layer**:
$$
Z = W x + b
A = \sigma(Z)
$$

Where:
- $W \in \mathbb{R}^{h \times d}$ is the weight matrix
- $b \in \mathbb{R}^h$ is the bias vector
- $Z, A \in \mathbb{R}^h$ are the pre-activations and activations

**Output Layer**:
$$
\hat{y} = w_{out}^T A + b_{out}
$$

### Feature Learning Interpretation

Each hidden neuron learns to detect a specific feature or pattern in the input:

1. **Feature Detectors**: Each neuron becomes specialized in recognizing certain input patterns
2. **Feature Combination**: The output layer learns to combine these features for the final prediction
3. **Hierarchical Learning**: Complex features are built from simpler ones

### Example: Housing Price Prediction with Multiple Features

Consider predicting house prices using multiple features: size, bedrooms, location, age.

**Hidden Layer Features**:
- Neuron 1: "Family size indicator" (combines size and bedrooms)
- Neuron 2: "Location premium" (based on zip code)
- Neuron 3: "Maintenance cost" (based on age and size)

**Output Layer**: Combines these features to predict the final price.

### Why Stacking Helps

**Expressiveness**: Each additional layer increases the network's capacity to represent complex functions.

**Mathematical Intuition**: 
- Single neuron: Can create one "kink" or threshold
- Two neurons: Can create two kinks
- $h$ neurons: Can create $h$ kinks, approximating any piecewise linear function
- Multiple layers: Can create exponentially more complex patterns

---

## Biological Inspiration and Analogies

### Connection to Biological Neural Networks

While artificial neural networks are inspired by biological systems, they are simplified mathematical models rather than accurate simulations.

#### Biological Neuron Structure

A biological neuron consists of:
1. **Dendrites**: Receive signals from other neurons
2. **Cell Body**: Processes the signals
3. **Axon**: Transmits signals to other neurons
4. **Synapses**: Connection points where signals are transmitted

#### Artificial vs. Biological Neurons

| Aspect | Biological Neuron | Artificial Neuron |
|--------|-------------------|-------------------|
| Input | Electrical/chemical signals | Numerical values |
| Processing | Complex biochemical processes | Simple mathematical operations |
| Output | Action potential (spike) | Continuous value |
| Learning | Synaptic plasticity | Gradient descent |
| Speed | Milliseconds | Nanoseconds |

### Key Insights from Biology

1. **Connectivity**: Neurons are highly interconnected
2. **Plasticity**: Connections can strengthen or weaken based on activity
3. **Hierarchy**: Information processing occurs in stages
4. **Parallelism**: Many neurons operate simultaneously

### Limitations of the Biological Analogy

1. **Simplification**: Artificial neurons are much simpler than biological ones
2. **Learning**: Biological learning is more complex than gradient descent
3. **Architecture**: Biological networks have more complex connectivity patterns
4. **Purpose**: Artificial networks are designed for mathematical convenience, not biological accuracy

---

## Two-Layer Fully-Connected Neural Networks

### Architecture Overview

A two-layer fully-connected network is the simplest form of a "deep" neural network. It consists of:

1. **Input Layer**: $x \in \mathbb{R}^d$
2. **Hidden Layer**: $m$ neurons with full connectivity
3. **Output Layer**: Final prediction

### Mathematical Formulation

#### Layer-by-Layer Computation

**Layer 1 (Hidden Layer)**:
$$
z_j^{[1]} = (w_j^{[1]})^T x + b_j^{[1]}, \quad j = 1, 2, \ldots, m
a_j^{[1]} = \sigma(z_j^{[1]}), \quad j = 1, 2, \ldots, m
$$

**Layer 2 (Output Layer)**:
$$
z^{[2]} = (w^{[2]})^T a^{[1]} + b^{[2]}
\hat{y} = z^{[2]} \quad \text{(for regression)}
\hat{y} = \sigma(z^{[2]}) \quad \text{(for classification)}
$$

#### Matrix Notation

**Forward Pass**:
$$
Z^{[1]} = W^{[1]} x + b^{[1]}
A^{[1]} = \sigma(Z^{[1]})
Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}
\hat{y} = Z^{[2]}
$$

Where:
- $W^{[1]} \in \mathbb{R}^{m \times d}$: Weight matrix for layer 1
- $b^{[1]} \in \mathbb{R}^m$: Bias vector for layer 1
- $W^{[2]} \in \mathbb{R}^{1 \times m}$: Weight matrix for layer 2
- $b^{[2]} \in \mathbb{R}$: Bias for layer 2

### Parameter Sharing and Efficiency

#### Computational Complexity

- **Forward Pass**: $O(md + m) = O(md)$ operations
- **Memory**: $O(md + m + m + 1) = O(md)$ parameters
- **Expressiveness**: Can represent any function that can be approximated by $m$ basis functions

#### Comparison with Single Layer

| Aspect | Single Neuron | Two-Layer Network |
|--------|---------------|-------------------|
| Parameters | $d + 1$ | $md + m + m + 1$ |
| Expressiveness | Limited | High |
| Training Time | Fast | Slower |
| Overfitting Risk | Low | Higher |

### Training Process

#### Loss Function

For regression:
$$
J(\theta) = \frac{1}{n} \sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2
$$

For classification:
$$
J(\theta) = -\frac{1}{n} \sum_{i=1}^n [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]
$$

#### Gradient Computation

The gradients are computed using backpropagation:

$$
\frac{\partial J}{\partial W^{[2]}} = \frac{1}{n} \sum_{i=1}^n (a^{[1](i)})^T (y^{(i)} - \hat{y}^{(i)})
$$

$$
\frac{\partial J}{\partial W^{[1]}} = \frac{1}{n} \sum_{i=1}^n x^{(i)} (\sigma'(z^{[1](i)}) \odot (W^{[2]})^T (y^{(i)} - \hat{y}^{(i)}))^T
$$

Where $\odot$ denotes element-wise multiplication.

### Practical Considerations

#### Initialization

**Weight Initialization**: Important for training success
- **Xavier/Glorot Initialization**: $W \sim \mathcal{N}(0, \frac{2}{n_{in} + n_{out}})$
- **He Initialization**: $W \sim \mathcal{N}(0, \frac{2}{n_{in}})$ (for ReLU)

**Bias Initialization**: Usually initialized to zero or small positive values

#### Regularization

**L2 Regularization**:
$$
J_{reg}(\theta) = J(\theta) + \frac{\lambda}{2} (\|W^{[1]}\|_F^2 + \|W^{[2]}\|_F^2)
$$

**Dropout**: Randomly set some activations to zero during training

#### Hyperparameter Tuning

- **Number of hidden units**: Start with $m = \sqrt{d}$ or $m = 2d$
- **Learning rate**: Start with 0.01 and adjust based on convergence
- **Batch size**: Balance between memory usage and training stability

---

## Multi-Layer Networks: Going Deeper

### Why Go Deeper?

#### Theoretical Motivation

**Representation Learning**: Deep networks can learn hierarchical representations automatically.

**Parameter Efficiency**: Deep networks can represent complex functions with fewer parameters than shallow networks.

**Feature Hierarchy**: Early layers learn low-level features, later layers learn high-level abstractions.

#### Mathematical Intuition

A deep network with $L$ layers can be written as:

$$
f(x) = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)
$$

Each layer $f_i$ transforms the input into a new representation that becomes the input for the next layer.

### Deep Network Architecture

#### General Formulation

For a network with $L$ layers:

**Layer $l$**:
$$
Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}
A^{[l]} = \sigma^{[l]}(Z^{[l]})
$$

Where:
- $A^{[0]} = x$ (input)
- $\sigma^{[l]}$ is the activation function for layer $l$
- $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$ is the weight matrix
- $b^{[l]} \in \mathbb{R}^{n_l}$ is the bias vector

#### Activation Functions by Layer

- **Hidden Layers**: Usually ReLU or variants
- **Output Layer**: 
  - Regression: Linear (no activation)
  - Binary Classification: Sigmoid
  - Multi-class Classification: Softmax

### Training Deep Networks

#### Challenges

1. **Vanishing/Exploding Gradients**: Gradients can become very small or very large
2. **Overfitting**: More parameters increase risk of overfitting
3. **Computational Cost**: Training time increases with depth
4. **Hyperparameter Tuning**: More parameters to tune

#### Solutions

**Gradient Issues**:
- **Batch Normalization**: Normalize activations within each batch
- **Residual Connections**: Skip connections to help gradient flow
- **Proper Initialization**: Use appropriate weight initialization schemes

**Overfitting**:
- **Regularization**: L2 regularization, dropout
- **Early Stopping**: Stop training when validation loss increases
- **Data Augmentation**: Increase effective dataset size

**Computational Efficiency**:
- **GPU Acceleration**: Use specialized hardware
- **Mini-batch Training**: Process data in batches
- **Optimized Libraries**: Use frameworks like PyTorch, TensorFlow

### Modern Architectures

#### Residual Networks (ResNets)

Add skip connections to help with gradient flow:

$$
A^{[l+1]} = \sigma(Z^{[l+1]} + A^{[l]})
$$

#### Batch Normalization

Normalize activations to stabilize training:

$$
A_{norm}^{[l]} = \frac{A^{[l]} - \mu}{\sqrt{\sigma^2 + \epsilon}}
A^{[l]} = \gamma A_{norm}^{[l]} + \beta
$$

#### Attention Mechanisms

Allow the network to focus on relevant parts of the input:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

---

## Activation Functions Deep Dive

### Why Activation Functions Matter

Activation functions are crucial because they introduce non-linearity, enabling neural networks to learn complex patterns.

### Properties of Good Activation Functions

1. **Non-linearity**: Essential for modeling complex relationships
2. **Differentiability**: Required for gradient-based optimization
3. **Monotonicity**: Helps with optimization stability
4. **Boundedness**: Can help prevent exploding gradients
5. **Computational Efficiency**: Should be fast to compute

### Detailed Analysis of Common Activations

#### ReLU (Rectified Linear Unit)

**Definition**: $\sigma(z) = \max(0, z)$

**Advantages**:
- **Computational Efficiency**: Simple max operation
- **Sparsity**: Can produce exact zeros
- **Gradient Flow**: Simple derivative prevents vanishing gradients
- **Biological Plausibility**: Similar to biological neuron firing

**Disadvantages**:
- **Dying ReLU**: Neurons can get stuck at zero
- **Not Zero-Centered**: Output is always non-negative
- **Not Bounded**: Output can grow arbitrarily large

**Variants**:
- **Leaky ReLU**: $\sigma(z) = \max(\alpha z, z)$ where $\alpha < 1$
- **Parametric ReLU**: $\alpha$ is learned
- **ELU**: $\sigma(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \leq 0 \end{cases}$

#### Sigmoid

**Definition**: $\sigma(z) = \frac{1}{1 + e^{-z}}$

**Advantages**:
- **Bounded**: Output always between 0 and 1
- **Smooth**: Continuous and differentiable everywhere
- **Interpretable**: Can be interpreted as probability

**Disadvantages**:
- **Vanishing Gradient**: Derivative approaches zero for large inputs
- **Not Zero-Centered**: Output is always positive
- **Saturation**: Neurons can get stuck in saturation regions

#### Tanh (Hyperbolic Tangent)

**Definition**: $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$

**Advantages**:
- **Zero-Centered**: Output ranges from -1 to 1
- **Bounded**: Output is always between -1 and 1
- **Smooth**: Continuous and differentiable

**Disadvantages**:
- **Vanishing Gradient**: Still suffers from gradient vanishing
- **Saturation**: Can get stuck in saturation regions

#### GELU (Gaussian Error Linear Unit)

**Definition**: $\sigma(z) = z \cdot \Phi(z)$ where $\Phi$ is the cumulative distribution function of the standard normal distribution

**Advantages**:
- **Smooth**: Continuous and differentiable
- **Non-monotonic**: Can model more complex relationships
- **Performance**: Often performs better than ReLU in practice

**Disadvantages**:
- **Computational Cost**: More expensive to compute
- **Complexity**: More complex than ReLU

### Choosing Activation Functions

#### Guidelines

1. **Hidden Layers**: ReLU is usually a good default choice
2. **Output Layer**: 
   - Regression: Linear (no activation)
   - Binary Classification: Sigmoid
   - Multi-class Classification: Softmax
3. **Special Cases**: Consider alternatives based on specific requirements

#### Empirical Considerations

- **ReLU**: Good default for most cases
- **Leaky ReLU**: If you observe dying ReLU problem
- **Tanh**: If you need bounded outputs
- **GELU**: For transformer-based architectures

---

## Connection to Kernel Methods

### Theoretical Relationship

Neural networks and kernel methods are both approaches to non-linear learning, but they work in fundamentally different ways.

#### Kernel Methods

Kernel methods rely on the "kernel trick" to implicitly map data to high-dimensional spaces:

$$
f(x) = \sum_{i=1}^n \alpha_i K(x, x_i)
$$

Where $K$ is a kernel function measuring similarity between points.

#### Neural Networks

Neural networks learn explicit feature mappings:

$$
f(x) = W^{[L]} \sigma(W^{[L-1]} \sigma(\cdots \sigma(W^{[1]} x + b^{[1]}) \cdots) + b^{[L-1]}) + b^{[L]}
$$

### Key Differences

| Aspect | Kernel Methods | Neural Networks |
|--------|----------------|-----------------|
| Feature Learning | Fixed kernel functions | Learned feature mappings |
| Scalability | Limited by number of training examples | Limited by model capacity |
| Interpretability | Kernel functions are interpretable | Learned features may not be |
| Flexibility | Limited by choice of kernel | Highly flexible architecture |

### Mathematical Connection

#### Neural Tangent Kernel (NTK)

Recent research has shown that in the limit of infinite width, neural networks behave like kernel methods with a specific kernel called the Neural Tangent Kernel.

**Intuition**: As the number of neurons approaches infinity, the network's behavior becomes more predictable and can be characterized by a kernel function.

#### Practical Implications

1. **Understanding**: NTK helps understand why neural networks work
2. **Design**: Can guide architecture design
3. **Training**: Provides insights into optimization behavior
4. **Generalization**: Helps understand generalization properties

---

*This concludes our exploration of neural network fundamentals. In the next sections, we will dive deeper into specific architectures, training algorithms, and practical implementation details.*

## From Neural Network Fundamentals to Modular Design

We've now explored the fundamental building blocks of neural networks - from individual neurons with their activation functions to multi-layer architectures that can learn complex patterns. We've seen how the mathematical principles of non-linear transformations and function composition translate into concrete neural network designs.

However, as neural networks become more complex and are applied to increasingly sophisticated problems, we need to move beyond basic architectures to **modular design principles**. Modern deep learning systems are built using reusable components that can be composed to create complex architectures efficiently.

This motivates our exploration of **neural network modules** - the building blocks that enable us to construct sophisticated architectures systematically. We'll see how common patterns (like fully connected layers, convolutional layers, and attention mechanisms) can be implemented as reusable modules that can be combined in various ways.

The transition from neural network fundamentals to modular design represents the bridge from understanding basic architectures to building practical, scalable systems - taking our knowledge of how neural networks work and turning it into a systematic approach for constructing complex models.

In the next section, we'll explore how to design and implement neural network modules, how to compose them into larger architectures, and how this modular approach enables both flexibility and efficiency in deep learning systems.

---

**Previous: [Non-Linear Models](01_non-linear_models.md)** - Understand the mathematical foundations of deep learning and non-linear models.

**Next: [Neural Network Modules](03_modules.md)** - Learn how to design and implement modular neural network components.

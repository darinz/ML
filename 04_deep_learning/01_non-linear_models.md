# Deep Learning: Foundations and Non-Linear Models

## Introduction to Deep Learning

Deep learning represents a paradigm shift in artificial intelligence, enabling machines to learn complex patterns directly from raw data without explicit feature engineering. Unlike traditional machine learning approaches that rely on hand-crafted features, deep learning models automatically discover hierarchical representations through multiple layers of non-linear transformations.

### What Makes Deep Learning "Deep"?

The term "deep" refers to the multiple layers of processing that data undergoes as it flows through the network. Each layer transforms the input data into increasingly abstract representations:

- **Early layers** learn low-level features (edges, textures, basic patterns)
- **Middle layers** combine these into intermediate concepts (shapes, parts)
- **Later layers** form high-level abstractions (objects, concepts, semantic meaning)

This hierarchical feature learning is inspired by how the human brain processes information through multiple stages of neural processing.

### Historical Context and Breakthroughs

Deep learning's resurgence began in the early 2010s with several key breakthroughs:

1. **AlexNet (2012)**: Demonstrated that deep convolutional neural networks could dramatically outperform traditional methods on ImageNet
2. **Word2Vec (2013)**: Showed how neural networks could learn meaningful word representations
3. **AlphaGo (2016)**: Proved that deep reinforcement learning could master complex strategic games
4. **Transformer Architecture (2017)**: Revolutionized natural language processing with attention mechanisms

### Key Advantages of Deep Learning

1. **Automatic Feature Learning**: No need for domain experts to design features
2. **Scalability**: Performance typically improves with more data and larger models
3. **Transfer Learning**: Pre-trained models can be adapted to new tasks
4. **End-to-End Learning**: Single model can handle complex pipelines
5. **Representation Learning**: Learns useful representations for multiple downstream tasks

### Applications and Impact

Deep learning has transformed numerous fields:

- **Computer Vision**: Image classification, object detection, medical imaging
- **Natural Language Processing**: Machine translation, text generation, question answering
- **Speech Recognition**: Voice assistants, transcription services
- **Reinforcement Learning**: Game playing, robotics, autonomous systems
- **Generative AI**: Image generation, text generation, music composition

---

## 7.1 Supervised Learning with Non-Linear Models

### From Linear to Non-Linear: The Need for Complexity

In traditional supervised learning, we've explored models that are **linear in their parameters**:

1. **Linear Regression**: $h_\theta(x) = \theta^T x$
2. **Linear Classification**: $h_\theta(x) = \theta^T \phi(x)$ where $\phi(x)$ is a feature map

While these models are mathematically elegant and computationally efficient, they suffer from fundamental limitations in expressiveness.

### Limitations of Linear Models

**Mathematical Limitation**: Linear models can only represent functions of the form $f(x) = w^T x + b$, which are straight lines (or hyperplanes in higher dimensions).

**Practical Consequences**:
- Cannot capture non-linear relationships in data
- Cannot solve problems requiring non-linear decision boundaries
- Limited ability to model complex real-world phenomena

**Classic Example - XOR Problem**: The XOR (exclusive OR) function cannot be represented by any linear model:
- Input: $(0,0) \rightarrow 0$, $(0,1) \rightarrow 1$, $(1,0) \rightarrow 1$, $(1,1) \rightarrow 0$
- No single line can separate the classes $(0,1)$ and $(1,0)$ from $(0,0)$ and $(1,1)$

### The Power of Non-Linear Models

Non-linear models can represent complex functions by combining multiple non-linear transformations:

**Polynomial Models**: $h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3 + \ldots$

**Neural Networks**: $h_\theta(x) = \sigma(W_2 \sigma(W_1 x + b_1) + b_2)$

Where $\sigma$ is a non-linear activation function (e.g., ReLU, sigmoid, tanh).

### Universal Approximation Theorem

A fundamental theoretical result states that a neural network with a single hidden layer containing a sufficient number of neurons can approximate any continuous function on a compact domain to arbitrary precision. This provides theoretical justification for the power of neural networks.

### Mathematical Framework

Given training examples $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$, where:
- $x^{(i)} \in \mathbb{R}^d$ is the input feature vector
- $y^{(i)}$ is the target output (real-valued for regression, categorical for classification)
- $h_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^k$ is our non-linear model parameterized by $\theta$

Our goal is to find parameters $\theta$ that minimize a loss function $J(\theta)$ measuring the discrepancy between predictions and true values.

---

### Regression Problems

Regression involves predicting continuous real-valued outputs. This is the most straightforward case for understanding loss functions.

#### Mathematical Formulation

For regression, we have:
- Input: $x^{(i)} \in \mathbb{R}^d$
- Output: $y^{(i)} \in \mathbb{R}$ (continuous real value)
- Model: $h_\theta: \mathbb{R}^d \rightarrow \mathbb{R}$

#### Mean Squared Error (MSE) Loss

The most common loss function for regression is the Mean Squared Error:

```math
J^{(i)}(\theta) = \frac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2
```

The factor of $\frac{1}{2}$ is included for mathematical convenience (it cancels out when taking derivatives).

The total loss over the entire dataset is:

```math
J(\theta) = \frac{1}{n} \sum_{i=1}^n J^{(i)}(\theta) = \frac{1}{2n} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2
```

#### Why Mean Squared Error?

**Statistical Justification**: MSE corresponds to the maximum likelihood estimator under the assumption that the noise in the outputs follows a Gaussian distribution. If we assume $y^{(i)} = h_\theta(x^{(i)}) + \epsilon^{(i)}$ where $\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$, then maximizing the likelihood is equivalent to minimizing MSE.

**Mathematical Properties**:
- **Differentiability**: MSE is differentiable everywhere, making optimization easier
- **Convexity**: When $h_\theta$ is linear in $\theta$, MSE is convex, guaranteeing convergence to global optimum
- **Penalty Structure**: Squaring errors penalizes large errors more heavily than small ones

**Intuition**: The squared term means that an error of 2 units is penalized 4 times more than an error of 1 unit, making the model more sensitive to outliers.

#### Alternative Loss Functions

**Mean Absolute Error (MAE)**:
```math
J(\theta) = \frac{1}{n} \sum_{i=1}^n |h_\theta(x^{(i)}) - y^{(i)}|
```

**Advantages**: Less sensitive to outliers, more robust
**Disadvantages**: Not differentiable at zero, can be harder to optimize

**Huber Loss**: Combines the best of both MSE and MAE:
```math
J^{(i)}(\theta) = \begin{cases}
\frac{1}{2}(h_\theta(x^{(i)}) - y^{(i)})^2 & \text{if } |h_\theta(x^{(i)}) - y^{(i)}| \leq \delta \\
\delta|h_\theta(x^{(i)}) - y^{(i)}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
```

Where $\delta$ is a hyperparameter controlling the transition point.

#### Worked Example

Consider a simple case with 2 data points:
- $x^{(1)} = 1$, $y^{(1)} = 2$
- $x^{(2)} = 2$, $y^{(2)} = 4$

And a linear model $h_\theta(x) = 2x$ (perfect fit):

For the first point: $J^{(1)}(\theta) = \frac{1}{2}(2 \cdot 1 - 2)^2 = \frac{1}{2}(0)^2 = 0$

For the second point: $J^{(2)}(\theta) = \frac{1}{2}(2 \cdot 2 - 4)^2 = \frac{1}{2}(0)^2 = 0$

Total loss: $J(\theta) = \frac{1}{2}(0 + 0) = 0$ (perfect fit)

---

### Binary Classification

Binary classification involves predicting one of two possible outcomes (e.g., spam/not spam, sick/healthy).

#### Mathematical Framework

- Input: $x \in \mathbb{R}^d$
- Output: $y \in \{0, 1\}$ (binary labels)
- Model: $h_\theta: \mathbb{R}^d \rightarrow \mathbb{R}$ (produces a real-valued score)

#### From Scores to Probabilities: The Sigmoid Function

The model produces a real-valued score (logit), but we need a probability. We use the sigmoid function to transform the score:

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

**Properties of the Sigmoid Function**:
- **Range**: $\sigma(z) \in (0, 1)$ for all $z \in \mathbb{R}$
- **Symmetry**: $\sigma(-z) = 1 - \sigma(z)$
- **Monotonicity**: $\sigma'(z) = \sigma(z)(1 - \sigma(z)) > 0$ for all $z$
- **Limits**: $\lim_{z \to -\infty} \sigma(z) = 0$, $\lim_{z \to +\infty} \sigma(z) = 1$

**Intuition**: The sigmoid function "squashes" any real number into the interval $(0,1)$, making it suitable for representing probabilities.

#### Probabilistic Interpretation

We model the conditional probability as:

```math
P(y = 1 \mid x; \theta) = \sigma(h_\theta(x)) = \frac{1}{1 + e^{-h_\theta(x)}}
```

```math
P(y = 0 \mid x; \theta) = 1 - P(y = 1 \mid x; \theta) = \frac{e^{-h_\theta(x)}}{1 + e^{-h_\theta(x)}}
```

This formulation ensures that:
- Probabilities sum to 1: $P(y = 1 \mid x; \theta) + P(y = 0 \mid x; \theta) = 1$
- Probabilities are always positive and less than 1
- The model can express uncertainty through intermediate probability values

#### Binary Cross-Entropy Loss

The loss function for binary classification is the negative log-likelihood:

```math
J^{(i)}(\theta) = -\log P(y^{(i)} \mid x^{(i)}; \theta)
```

For binary classification, this becomes:

```math
J^{(i)}(\theta) = -y^{(i)} \log(\sigma(h_\theta(x^{(i)}))) - (1 - y^{(i)}) \log(1 - \sigma(h_\theta(x^{(i)})))
```

**Why This Loss Function?**

**Information-Theoretic Interpretation**: Cross-entropy measures the difference between the true distribution and the predicted distribution. Minimizing it is equivalent to maximizing the likelihood of the data.

**Mathematical Properties**:
- **Convexity**: When $h_\theta$ is linear in $\theta$, the loss is convex
- **Gradient Properties**: The gradient has a simple form that facilitates optimization
- **Penalty Structure**: Heavily penalizes confident but wrong predictions

**Intuition**: If the true label is 1 but the model predicts a very low probability, the loss becomes very large (due to the log term). This encourages the model to be confident in correct predictions and appropriately uncertain in incorrect ones.

#### Total Loss

The total loss is the average over all training examples:

```math
J(\theta) = \frac{1}{n} \sum_{i=1}^n J^{(i)}(\theta)
```

---

### Multi-Class Classification

Multi-class classification extends binary classification to $k > 2$ classes.

#### Mathematical Framework

- Input: $x \in \mathbb{R}^d$
- Output: $y \in \{1, 2, \ldots, k\}$ (categorical labels)
- Model: $\hat{h}_\theta: \mathbb{R}^d \rightarrow \mathbb{R}^k$ (produces $k$ real-valued scores)

#### From Scores to Probabilities: The Softmax Function

The softmax function generalizes the sigmoid to multiple classes:

```math
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}}
```

**Properties of the Softmax Function**:
- **Output Range**: $\text{softmax}(z)_i \in (0, 1)$ for all $i$
- **Sum to 1**: $\sum_{i=1}^k \text{softmax}(z)_i = 1$
- **Monotonicity**: If $z_i > z_j$, then $\text{softmax}(z)_i > \text{softmax}(z)_j$
- **Translation Invariance**: $\text{softmax}(z + c) = \text{softmax}(z)$ for any constant $c$

**Numerical Stability**: In practice, we compute softmax as:
```math
\text{softmax}(z)_i = \frac{e^{z_i - \max_j z_j}}{\sum_{j=1}^k e^{z_j - \max_j z_j}}
```

This prevents numerical overflow while producing the same result.

#### Probabilistic Interpretation

We model the conditional probability as:

```math
P(y = s \mid x; \theta) = \frac{e^{\hat{h}_\theta(x)_s}}{\sum_{j=1}^k e^{\hat{h}_\theta(x)_j}}
```

Where $\hat{h}_\theta(x)_s$ is the $s$-th component of the model output.

#### Categorical Cross-Entropy Loss

The loss function for multi-class classification is:

```math
J^{(i)}(\theta) = -\log P(y^{(i)} \mid x^{(i)}; \theta) = -\log \left( \frac{e^{\hat{h}_\theta(x^{(i)})_{y^{(i)}}}}{\sum_{j=1}^k e^{\hat{h}_\theta(x^{(i)})_j}} \right)
```

This can be written more compactly as:

```math
J^{(i)}(\theta) = -\hat{h}_\theta(x^{(i)})_{y^{(i)}} + \log \sum_{j=1}^k e^{\hat{h}_\theta(x^{(i)})_j}
```

**Intuition**: The loss encourages the model to assign high probability to the correct class while keeping probabilities for other classes low.

#### Worked Example

Consider a 3-class problem with logits $\hat{h}_\theta(x) = [2, 1, 0]$:

```math
P(y=1 \mid x) = \frac{e^2}{e^2 + e^1 + e^0} = \frac{7.39}{7.39 + 2.72 + 1} \approx 0.665
```

```math
P(y=2 \mid x) = \frac{e^1}{e^2 + e^1 + e^0} = \frac{2.72}{7.39 + 2.72 + 1} \approx 0.245
```

```math
P(y=3 \mid x) = \frac{e^0}{e^2 + e^1 + e^0} = \frac{1}{7.39 + 2.72 + 1} \approx 0.090
```

The model is most confident in class 1, followed by class 2, and least confident in class 3.

---

### Exponential Family Generalization

The loss functions we've discussed can be unified under the framework of exponential family distributions.

#### Exponential Family Distributions

A distribution belongs to the exponential family if its probability density/mass function can be written as:

```math
p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta))
```

Where:
- $\eta$ is the natural parameter
- $T(y)$ is the sufficient statistic
- $b(y)$ is the base measure
- $a(\eta)$ is the log-normalizer

#### Connection to Loss Functions

Many common loss functions arise from the negative log-likelihood of exponential family distributions:

1. **Gaussian Distribution** → Mean Squared Error
2. **Bernoulli Distribution** → Binary Cross-Entropy
3. **Categorical Distribution** → Categorical Cross-Entropy
4. **Poisson Distribution** → Poisson Loss

#### Generalized Linear Models

In this framework, we assume:
- $y \sim \text{Exponential-Family}(\eta)$
- $\eta = h_\theta(x)$ (the natural parameter is a function of the input)

The loss function becomes:
```math
J^{(i)}(\theta) = -\log p(y^{(i)}; h_\theta(x^{(i)}))
```

This provides a unified theoretical foundation for understanding different loss functions and their properties.

---

### Optimization Methods

Training deep learning models requires efficient optimization algorithms to minimize the loss function.

#### Gradient Descent (GD)

The most fundamental optimization algorithm is gradient descent:

```math
\theta := \theta - \alpha \nabla J(\theta)
```

Where:
- $\alpha > 0$ is the learning rate
- $\nabla J(\theta)$ is the gradient of the loss function with respect to the parameters

**Intuition**: The gradient points in the direction of steepest increase. By moving in the opposite direction (negative gradient), we decrease the loss function.

**Mathematical Foundation**: This update rule follows from the first-order Taylor approximation:
```math
J(\theta + \Delta\theta) \approx J(\theta) + \nabla J(\theta)^T \Delta\theta
```

To minimize $J(\theta + \Delta\theta)$, we choose $\Delta\theta = -\alpha \nabla J(\theta)$.

#### Learning Rate Selection

The learning rate $\alpha$ is a crucial hyperparameter:

- **Too Small**: Slow convergence, may get stuck in local minima
- **Too Large**: May overshoot the minimum, causing divergence
- **Optimal**: Balances convergence speed with stability

**Common Strategies**:
- **Fixed Learning Rate**: Simple but requires careful tuning
- **Learning Rate Scheduling**: Gradually decrease $\alpha$ over time
- **Adaptive Methods**: Automatically adjust $\alpha$ based on gradient statistics

#### Variants of Gradient Descent

**Batch Gradient Descent (BGD)**:
- Uses all training examples to compute the gradient
- **Pros**: Stable, deterministic updates
- **Cons**: Computationally expensive for large datasets

**Stochastic Gradient Descent (SGD)**:
- Uses a single randomly selected example per update
- **Pros**: Fast updates, can escape local minima
- **Cons**: Noisy updates, may require more iterations

**Mini-Batch Gradient Descent**:
- Uses a small batch of examples per update
- **Pros**: Balances speed and stability
- **Cons**: Introduces hyperparameter (batch size)

#### Advanced Optimizers

**Momentum**: Adds a velocity term to accelerate convergence:
```math
v := \beta v - \alpha \nabla J(\theta)
\theta := \theta + v
```

**RMSProp**: Adapts learning rates based on gradient magnitudes:
```math
s := \beta s + (1-\beta)(\nabla J(\theta))^2
\theta := \theta - \frac{\alpha}{\sqrt{s + \epsilon}} \nabla J(\theta)
```

**Adam**: Combines momentum and adaptive learning rates:
```math
m := \beta_1 m + (1-\beta_1)\nabla J(\theta)
v := \beta_2 v + (1-\beta_2)(\nabla J(\theta))^2
\theta := \theta - \frac{\alpha}{\sqrt{v + \epsilon}} m
```

#### Comparison of Optimization Methods

| Method | Speed | Memory | Convergence | Robustness |
|--------|-------|--------|-------------|------------|
| BGD | Slow | High | Stable | High |
| SGD | Fast | Low | Noisy | Medium |
| Mini-batch | Medium | Medium | Balanced | High |
| Adam | Fast | Medium | Fast | High |

---

### Practical Deep Learning Workflow

A typical deep learning project involves several key steps:

#### 1. Model Design
- Choose architecture (number of layers, layer types, activation functions)
- Define the mathematical form of $h_\theta(x)$
- Consider inductive biases appropriate for the task

#### 2. Loss Function Selection
- Choose loss function based on problem type (regression vs. classification)
- Consider task-specific requirements (robustness, interpretability)
- Ensure mathematical properties (differentiability, convexity where possible)

#### 3. Optimization Setup
- Select optimizer and hyperparameters
- Design learning rate schedule
- Implement gradient clipping if needed

#### 4. Data Preprocessing
- Normalize/standardize inputs
- Handle missing data
- Split into train/validation/test sets
- Apply data augmentation if applicable

#### 5. Training Process
- Monitor training and validation loss
- Implement early stopping
- Use regularization techniques (dropout, weight decay)
- Track relevant metrics

#### 6. Evaluation and Deployment
- Evaluate on test set
- Analyze model behavior and errors
- Deploy model with appropriate monitoring
- Plan for model updates and maintenance

---

*This concludes our introduction to non-linear models and the mathematical foundations of deep learning. In the next sections, we will explore specific neural network architectures, training algorithms, and practical implementation details.*
# 2.3 Multi-Class Classification

## Introduction and Motivation

### Real-World Applications

In many real-world problems, the task is not just to distinguish between two classes (binary classification), but among three or more possible categories. Here are some compelling examples:

- **Email classification:** spam, personal, work, marketing, newsletters
- **Image recognition:** cat, dog, car, airplane, bird, fish, etc.
- **Handwritten digit recognition:** digits 0 through 9 (MNIST dataset)
- **Medical diagnosis:** healthy, disease A, disease B, disease C, etc.
- **Language identification:** English, Spanish, French, German, Chinese, etc.
- **Product categorization:** electronics, clothing, books, food, etc.
- **Sentiment analysis:** very negative, negative, neutral, positive, very positive

Multi-class classification is essential in machine learning because most practical problems involve more than two possible outcomes. The response variable $y$ can take on any one of $k$ values, so $y \in \{1, 2, \ldots, k\}$.

### Mathematical Framework

In multi-class classification, we have:
- **Input space:** $\mathcal{X} \subseteq \mathbb{R}^d$ (feature vectors)
- **Output space:** $\mathcal{Y} = \{1, 2, \ldots, k\}$ (class labels)
- **Training data:** $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(n)}, y^{(n)})\}$
- **Goal:** Learn a function $h: \mathcal{X} \rightarrow \mathcal{Y}$ that accurately predicts the class label

### Challenges in Multi-Class Classification

1. **Complexity:** More classes mean more complex decision boundaries
2. **Imbalanced Data:** Some classes may have many more examples than others
3. **Computational Cost:** Training time scales with the number of classes
4. **Evaluation:** More complex metrics needed (beyond accuracy)

## The Multinomial Model and Softmax Intuition

### From Binary to Multi-Class

Recall that in binary classification, we often use the logistic (sigmoid) function to map real-valued scores to probabilities. In the multi-class case, we need a function that:
- Outputs a probability for each class
- Ensures all probabilities are non-negative and sum to 1

This is achieved by the **softmax function**, which generalizes the sigmoid function to multiple classes.

### The Softmax Function

Let $x \in \mathbb{R}^d$ be the input features. We introduce $k$ parameter vectors $\theta_1, \ldots, \theta_k$, each in $\mathbb{R}^d$. For each class $i$, we compute a score (logit):

$$
t_i = \theta_i^\top x
$$

The vector $t = (t_1, \ldots, t_k)$ is then passed through softmax:

$$
\mathrm{softmax}(t_1, \ldots, t_k) = \left[ \frac{\exp(t_1)}{\sum_{j=1}^k \exp(t_j)}, \ldots, \frac{\exp(t_k)}{\sum_{j=1}^k \exp(t_j)} \right]
$$

The output is a probability vector $\phi = (\phi_1, \ldots, \phi_k)$, where $\phi_i$ is the probability assigned to class $i$.

### Intuitive Understanding of Softmax

#### Why Exponentiation?

The exponential function $\exp(t_i)$ has several desirable properties:
1. **Always Positive:** $\exp(t_i) > 0$ for any real $t_i$
2. **Monotonic:** Larger $t_i$ leads to larger $\exp(t_i)$
3. **Sensitive to Differences:** Small differences in $t_i$ become amplified

#### Why Normalization?

Dividing by the sum $\sum_{j=1}^k \exp(t_j)$ ensures:
1. **Probability Constraint:** $\sum_{i=1}^k \phi_i = 1$
2. **Non-negative:** $\phi_i \geq 0$ for all $i$
3. **Relative Scale:** Probabilities depend on relative differences between logits

#### Geometric Interpretation

Softmax can be viewed as projecting a $k$-dimensional real vector (the logits) onto the $(k-1)$-dimensional probability simplex. The simplex is the set of all probability distributions over $k$ classes.

### Properties of Softmax

#### Invariance to Translation

Softmax is invariant to adding a constant to all logits:
$$
\mathrm{softmax}(t_1 + c, \ldots, t_k + c) = \mathrm{softmax}(t_1, \ldots, t_k)
$$

This means we can subtract the maximum logit for numerical stability:
$$
\mathrm{softmax}(t_1, \ldots, t_k) = \mathrm{softmax}(t_1 - \max_j t_j, \ldots, t_k - \max_j t_j)
$$

#### Temperature Scaling

We can control the "sharpness" of the distribution by introducing a temperature parameter $\tau$:
$$
\mathrm{softmax}_\tau(t_1, \ldots, t_k) = \left[ \frac{\exp(t_1/\tau)}{\sum_{j=1}^k \exp(t_j/\tau)}, \ldots, \frac{\exp(t_k/\tau)}{\sum_{j=1}^k \exp(t_j/\tau)} \right]
$$

- **$\tau \to 0$:** Approaches one-hot encoding (deterministic)
- **$\tau = 1$:** Standard softmax
- **$\tau \to \infty$:** Approaches uniform distribution

## Probabilistic Model

### Model Definition

Given input $x$, the model predicts:

$$
P(y = i \mid x; \theta) = \phi_i = \frac{\exp(\theta_i^\top x)}{\sum_{j=1}^k \exp(\theta_j^\top x)}
$$

This is a generalization of logistic regression to multiple classes, sometimes called **multinomial logistic regression** or **softmax regression**.

### Parameter Interpretation

- **$\theta_i$:** Parameter vector for class $i$
- **$\theta_i^\top x$:** Score (logit) for class $i$
- **$\phi_i$:** Probability of class $i$

#### Decision Rule

The predicted class is:
$$
\hat{y} = \arg\max_{i} P(y = i \mid x; \theta) = \arg\max_{i} \theta_i^\top x
$$

This shows that the decision boundary between any two classes $i$ and $j$ is linear:
$$
\theta_i^\top x = \theta_j^\top x \implies (\theta_i - \theta_j)^\top x = 0
$$

### Example: 3-Class Classification

Consider a 3-class problem with:
- $\theta_1 = [1, 2]$, $\theta_2 = [0, 1]$, $\theta_3 = [-1, 0]$
- Input $x = [1, 1]$

Then:
- $t_1 = \theta_1^\top x = 1 \cdot 1 + 2 \cdot 1 = 3$
- $t_2 = \theta_2^\top x = 0 \cdot 1 + 1 \cdot 1 = 1$
- $t_3 = \theta_3^\top x = -1 \cdot 1 + 0 \cdot 1 = -1$

Softmax probabilities:
- $\phi_1 = \frac{e^3}{e^3 + e^1 + e^{-1}} \approx 0.88$
- $\phi_2 = \frac{e^1}{e^3 + e^1 + e^{-1}} \approx 0.12$
- $\phi_3 = \frac{e^{-1}}{e^3 + e^1 + e^{-1}} \approx 0.00$

Prediction: Class 1 (highest probability)

## Loss Function: Cross-Entropy and Negative Log-Likelihood

### Likelihood Function

Given $n$ independent training examples, the likelihood is:

$$
L(\theta) = \prod_{i=1}^n P(y^{(i)} \mid x^{(i)}; \theta) = \prod_{i=1}^n \frac{\exp(\theta_{y^{(i)}}^\top x^{(i)})}{\sum_{j=1}^k \exp(\theta_j^\top x^{(i)})}
$$

### Negative Log-Likelihood

Maximizing likelihood is equivalent to minimizing negative log-likelihood:

$$
\ell(\theta) = -\log L(\theta) = \sum_{i=1}^n -\log \left( \frac{\exp(\theta_{y^{(i)}}^\top x^{(i)})}{\sum_{j=1}^k \exp(\theta_j^\top x^{(i)})} \right)
$$

### Cross-Entropy Loss

The cross-entropy loss for a single example is:

$$
\ell_{ce}((t_1, \ldots, t_k), y) = -\log \left( \frac{\exp(t_y)}{\sum_{i=1}^k \exp(t_i)} \right)
$$

#### Intuition

The loss penalizes the model when it assigns low probability to the true class:
- **Correct prediction with high confidence:** Low loss
- **Correct prediction with low confidence:** Higher loss
- **Incorrect prediction with high confidence:** Very high loss
- **Incorrect prediction with low confidence:** Lower loss

#### Example Calculation

For the previous example with $t = [3, 1, -1]$ and true class $y = 2$:
$$
\ell_{ce} = -\log\left(\frac{e^1}{e^3 + e^1 + e^{-1}}\right) = -\log(0.12) \approx 2.12
$$

## Step-by-Step Example

### Complete Example: 3-Class Classification

Suppose we have 3 classes and a single input $x$ with logits $t = (2, 1, 0)$. Let's compute the softmax probabilities step by step:

1. **Compute exponentials:** 
   - $\exp(2) \approx 7.39$
   - $\exp(1) \approx 2.72$
   - $\exp(0) = 1$

2. **Sum the exponentials:** 
   - $7.39 + 2.72 + 1 = 11.11$

3. **Compute probabilities:** 
   - $\phi_1 = 7.39/11.11 \approx 0.665$
   - $\phi_2 = 2.72/11.11 \approx 0.245$
   - $\phi_3 = 1/11.11 \approx 0.090$

4. **Verify probability constraint:** 
   - $0.665 + 0.245 + 0.090 = 1.000$ ✓

If the true class is 2 (indexing from 1), the cross-entropy loss is:
$$
-\log(0.245) \approx 1.40
$$

### Numerical Stability Example

Consider logits $t = [1000, 1001, 1002]$:
- **Naive computation:** $\exp(1000) \approx \infty$ (overflow!)
- **Stable computation:** Subtract max logit first
  - $t' = [1000-1002, 1001-1002, 1002-1002] = [-2, -1, 0]$
  - $\exp(-2) \approx 0.135$, $\exp(-1) \approx 0.368$, $\exp(0) = 1$
  - Probabilities: $[0.090, 0.245, 0.665]$

## Practical Considerations

### Numerical Stability

When computing softmax, subtract the maximum logit from all logits before exponentiating to avoid overflow:

$$
\mathrm{softmax}(t)_i = \frac{\exp(t_i - \max_j t_j)}{\sum_{j=1}^k \exp(t_j - \max_j t_j)}
$$

This is mathematically equivalent but numerically stable.

### Label Encoding

Labels should be integers $1, \ldots, k$ (or $0, \ldots, k-1$ depending on convention). Common approaches:
- **One-hot encoding:** Convert to binary vectors
- **Integer encoding:** Use class indices directly
- **Ordinal encoding:** For ordered classes

### Implementation Considerations

Most ML libraries have built-in softmax and cross-entropy loss functions:
- **NumPy:** `np.softmax()`, `np.log_softmax()`
- **PyTorch:** `torch.softmax()`, `torch.nn.CrossEntropyLoss()`
- **TensorFlow:** `tf.nn.softmax()`, `tf.keras.losses.SparseCategoricalCrossentropy()`
- **scikit-learn:** `LogisticRegression(multi_class='multinomial')`

### Regularization

To prevent overfitting, add regularization terms:
- **L2 regularization:** $\lambda \sum_{i=1}^k \|\theta_i\|_2^2$
- **L1 regularization:** $\lambda \sum_{i=1}^k \|\theta_i\|_1$
- **Dropout:** Randomly zero some inputs during training

## Gradient Derivation and Intuition

### Gradient of Cross-Entropy Loss

The cross-entropy loss has a simple and elegant gradient. For a single example:

$$
\frac{\partial \ell_{ce}(t, y)}{\partial t_i} = \phi_i - 1\{y = i\}
$$

where $1\{y = i\}$ is the indicator function (1 if $y = i$, 0 otherwise).

#### Intuitive Understanding

- **$\phi_i - 1\{y = i\}$:** Difference between predicted and true probability
- **Positive gradient:** Predicted probability too high, decrease it
- **Negative gradient:** Predicted probability too low, increase it
- **Zero gradient:** Perfect prediction

#### Vectorized Form

In vectorized form: $\nabla_t \ell_{ce}(t, y) = \phi - e_y$
where $e_y$ is the one-hot encoding of $y$.

### Gradient for Model Parameters

For the model parameters $\theta_i$:

$$
\frac{\partial \ell(\theta)}{\partial \theta_i} = \sum_{j=1}^n (\phi_i^{(j)} - 1\{y^{(j)} = i\}) x^{(j)}
$$

This form is efficient to compute and is the basis for gradient descent and backpropagation in neural networks.

### Gradient Descent Update

The parameter update rule is:

$$
\theta_i := \theta_i - \alpha \sum_{j=1}^n (\phi_i^{(j)} - 1\{y^{(j)} = i\}) x^{(j)}
$$

where $\alpha$ is the learning rate.

## Applications and Extensions

### Neural Networks

Softmax is used as the final layer in multi-class classification networks:
1. **Hidden layers:** Learn feature representations
2. **Output layer:** Linear transformation + softmax
3. **Training:** Backpropagate gradients through the network

### Computer Vision

- **MNIST:** 10-class digit recognition
- **CIFAR-10:** 10-class image classification
- **ImageNet:** 1000-class image classification
- **Object detection:** Multi-class with bounding boxes

### Natural Language Processing

- **Text classification:** Topic classification, sentiment analysis
- **Part-of-speech tagging:** Grammatical role classification
- **Named entity recognition:** Person, organization, location, etc.
- **Language modeling:** Next word prediction

### Medical Applications

- **Disease diagnosis:** Multiple possible conditions
- **Drug discovery:** Compound classification
- **Medical imaging:** Tissue type classification
- **Patient stratification:** Risk group classification

### Extensions and Variants

#### Hierarchical Softmax

For large numbers of classes ($k \gg 100$), hierarchical softmax organizes classes in a tree structure:
- **Advantages:** Faster training and inference
- **Disadvantages:** Requires hierarchical structure
- **Applications:** Large vocabulary language models

#### Label Smoothing

Replace hard targets with soft targets to improve generalization:
$$
y_i' = (1 - \epsilon) \cdot 1\{y = i\} + \frac{\epsilon}{k}
$$

where $\epsilon$ is a small constant (e.g., 0.1).

#### Multi-Label Classification

When an example can belong to multiple classes simultaneously:
- **Approach:** Use sigmoid activation for each class
- **Loss:** Binary cross-entropy for each class
- **Applications:** Tag prediction, multi-disease diagnosis

#### Ordinal Classification

When classes have a natural ordering:
- **Approach:** Model cumulative probabilities
- **Loss:** Ordinal regression loss
- **Applications:** Rating prediction, severity assessment

## Comparison with Other Approaches

### One-vs-Rest (OvR)

Train $k$ binary classifiers, one for each class:
- **Advantages:** Simple, can use any binary classifier
- **Disadvantages:** Imbalanced training sets, no probability calibration
- **When to use:** Small number of classes, existing binary classifiers

### One-vs-One (OvO)

Train $\binom{k}{2}$ binary classifiers for each pair of classes:
- **Advantages:** Balanced training sets, can handle non-linear boundaries
- **Disadvantages:** More classifiers, complex prediction
- **When to use:** Small number of classes, non-linear boundaries

### Softmax vs. Other Methods

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| **Softmax** | Probabilistic, efficient, convex | Linear boundaries only |
| **OvR** | Simple, flexible | Imbalanced, no calibration |
| **OvO** | Balanced, non-linear | Many classifiers |
| **Decision Trees** | Non-linear, interpretable | No probabilities |
| **Random Forest** | Robust, feature importance | Black box |
| **SVM** | Non-linear (with kernels) | No probabilities |

## Summary

Multi-class classification with softmax and cross-entropy is a foundational technique in machine learning, generalizing logistic regression to multiple classes. Its mathematical simplicity, interpretability, and efficient gradient make it the default choice for many practical problems.

### Key Takeaways

1. **Softmax Function:** Smooth, differentiable generalization of sigmoid
2. **Cross-Entropy Loss:** Natural loss function for probability estimation
3. **Linear Decision Boundaries:** Between any pair of classes
4. **Numerical Stability:** Always subtract max logit before exponentiating
5. **Efficient Gradients:** Simple, interpretable gradient expressions

### Advanced Topics

For more advanced topics, see:
- **Neural networks:** Multi-layer architectures
- **Kernel methods:** Non-linear feature spaces
- **Ensemble methods:** Combining multiple classifiers
- **Calibration:** Improving probability estimates
- **Active learning:** Selecting informative examples

---

> **Note:** There are some ambiguities in naming conventions. Some people call the cross-entropy loss the function that maps the probability vector (the $\phi$ in our language) and label $y$ to the final real number, and call our version of cross-entropy loss softmax-cross-entropy loss. We choose our current naming convention because it's consistent with the naming of most modern deep learning libraries such as PyTorch and Jax.

---

**Previous: [Perceptron Algorithm](02_perceptron.md)** - Learn about the perceptron learning algorithm and its relationship to linear classification.

**Next: [Newton's Method](04_newtons_method.md)** - Explore second-order optimization methods for faster convergence in logistic regression.
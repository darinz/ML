# 2.3 Multi-class classification

## Introduction and Motivation

In many real-world problems, the task is not just to distinguish between two classes (binary classification), but among three or more possible categories. For example:
- **Email classification:** spam, personal, work
- **Image recognition:** cat, dog, car, airplane, etc.
- **Handwritten digit recognition:** digits 0 through 9
- **Medical diagnosis:** healthy, disease A, disease B, ...

Multi-class classification is essential in machine learning because most practical problems involve more than two possible outcomes. The response variable $y$ can take on any one of $k$ values, so $y \in \{1, 2, \ldots, k\}$.

## The Multinomial Model and Softmax Intuition

Recall that in binary classification, we often use the logistic (sigmoid) function to map real-valued scores to probabilities. In the multi-class case, we need a function that:
- Outputs a probability for each class
- Ensures all probabilities are non-negative and sum to 1

This is achieved by the **softmax function**. Geometrically, softmax projects a $k$-dimensional real vector (the logits) onto the $(k-1)$-dimensional probability simplex.

Let $x \in \mathbb{R}^d$ be the input features. We introduce $k$ parameter vectors $\theta_1, \ldots, \theta_k$, each in $\mathbb{R}^d$. For each class $i$, we compute a score (logit):

$$
t_i = \theta_i^\top x
$$

The vector $t = (t_1, \ldots, t_k)$ is then passed through softmax:

$$
\mathrm{softmax}(t_1, \ldots, t_k) = \left[ \frac{\exp(t_1)}{\sum_{j=1}^k \exp(t_j)}, \ldots, \frac{\exp(t_k)}{\sum_{j=1}^k \exp(t_j)} \right]
$$

The output is a probability vector $\phi = (\phi_1, \ldots, \phi_k)$, where $\phi_i$ is the probability assigned to class $i$.

**Why softmax?**
- Exponentiation ensures all outputs are positive.
- Division by the sum ensures they sum to 1.
- The largest logit gets the largest probability, but all classes get some probability mass.

## Probabilistic Model

Given input $x$, the model predicts:

$$
P(y = i \mid x; \theta) = \phi_i = \frac{\exp(\theta_i^\top x)}{\sum_{j=1}^k \exp(\theta_j^\top x)}
$$

This is a generalization of logistic regression to multiple classes, sometimes called **multinomial logistic regression** or **softmax regression**.

## Loss Function: Cross-Entropy and Negative Log-Likelihood

The loss function for training is the **negative log-likelihood** (NLL) of the data under the model, also known as the **cross-entropy loss**:

$$
\ell(\theta) = \sum_{i=1}^n -\log \left( \frac{\exp(\theta_{y^{(i)}}^\top x^{(i)})}{\sum_{j=1}^k \exp(\theta_j^\top x^{(i)})} \right)
$$

Or, using the cross-entropy notation:

$$
\ell_{ce}((t_1, \ldots, t_k), y) = -\log \left( \frac{\exp(t_y)}{\sum_{i=1}^k \exp(t_i)} \right)
$$

**Intuition:**
- The loss penalizes the model when it assigns low probability to the true class.
- If the model is confident and correct, the loss is small; if it is confident and wrong, the loss is large.

## Step-by-Step Example

Suppose we have 3 classes and a single input $x$ with logits $t = (2, 1, 0)$. Compute the softmax probabilities:

1. Compute exponentials: $\exp(2) \approx 7.39$, $\exp(1) \approx 2.72$, $\exp(0) = 1$
2. Sum: $7.39 + 2.72 + 1 = 11.11$
3. Probabilities: $[7.39/11.11, 2.72/11.11, 1/11.11] \approx [0.665, 0.245, 0.090]$

If the true class is 2 (indexing from 1), the cross-entropy loss is:

$$
-\log(0.245) \approx 1.40
$$

## Practical Considerations

- **Numerical stability:** When computing softmax, subtract the maximum logit from all logits before exponentiating to avoid overflow:

$$\mathrm{softmax}(t)_i = \frac{\exp(t_i - \max_j t_j)}{\sum_{j=1}^k \exp(t_j - \max_j t_j)}$$

- **Label encoding:** Labels should be integers $1, \ldots, k$ (or $0, \ldots, k-1$ depending on convention).
- **Implementation:** Most ML libraries (NumPy, PyTorch, TensorFlow, scikit-learn) have built-in softmax and cross-entropy loss functions.

## Gradient Derivation and Intuition

The cross-entropy loss has a simple and elegant gradient. For a single example:

$$
\frac{\partial \ell_{ce}(t, y)}{\partial t_i} = \phi_i - 1\{y = i\}
$$

- The gradient is positive if the predicted probability is higher than the true label indicator, negative otherwise.
- In vectorized form: $\nabla_t \ell_{ce}(t, y) = \phi - e_y$

For the model parameters $\theta_i$:

$$
\frac{\partial \ell(\theta)}{\partial \theta_i} = \sum_{j=1}^n (\phi_i^{(j)} - 1\{y^{(j)} = i\}) x^{(j)}
$$

This form is efficient to compute and is the basis for gradient descent and backpropagation in neural networks.

## Applications and Extensions

- **Neural networks:** Softmax is used as the final layer in multi-class classification networks.
- **Image classification:** e.g., MNIST, CIFAR-10, ImageNet
- **Natural language processing:** text classification, part-of-speech tagging
- **Medical diagnosis, recommender systems, etc.**
- **Extensions:**
  - Hierarchical softmax for large $k$
  - Label smoothing for regularization
  - Multi-label classification (using sigmoid for each class)

## Implementation Example (NumPy)

```python
import numpy as np

def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true):
    n = y_true.shape[0]
    return -np.log(probs[np.arange(n), y_true]).mean()

# Example usage:
logits = np.array([[2, 1, 0]])
probs = softmax(logits)
y_true = np.array([1])  # class index 1 (second class)
loss = cross_entropy_loss(probs, y_true)
print('Probabilities:', probs)
print('Loss:', loss)
```

## Summary

Multi-class classification with softmax and cross-entropy is a foundational technique in machine learning, generalizing logistic regression to multiple classes. Its mathematical simplicity, interpretability, and efficient gradient make it the default choice for many practical problems.

---

> 1. There are some ambiguity in the naming here. Some people call the cross-entropy loss the function that maps the probability vector (the $\phi$ in our language) and label $y$ to the final real number, and call our version of cross-entropy loss softmax-cross-entropy loss. We choose our current naming convention because it's consistent with the naming of most modern deep learning library such as PyTorch and Jax.
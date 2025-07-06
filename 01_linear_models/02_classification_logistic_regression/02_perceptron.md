## 2.2 Digression: the perceptron learning algorithm

The perceptron learning algorithm is a historically significant method in machine learning, providing a foundation for later developments in neural networks and classification. We briefly explore its formulation, intuition, properties, and limitations.

Consider modifying the logistic regression method to "force" it to output values that are either 0 or 1 exactly. To do so, we change the definition of $g$ to be the threshold function:

$$
g(z) = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

If we then let $h_\theta(x) = g(\theta^T x)$ as before but using this modified definition of $g$, and if we use the update rule

$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$

then we have the **perceptron learning algorithm**.

In the 1960s, this "perceptron" was argued to be a rough model for how individual neurons in the brain work. Given its simplicity, it provides a starting point for understanding learning theory and the development of more advanced algorithms.

Geometrically, the perceptron algorithm tries to find a hyperplane (a line in 2D, a plane in 3D, etc.) that separates the data points of two classes. Each update to $\theta$ moves the separating hyperplane so that it better classifies the training examples. If a data point is misclassified, the algorithm adjusts $\theta$ in the direction that would correctly classify that point.

If the data are **linearly separable** (i.e., there exists a hyperplane that perfectly separates the two classes), the perceptron algorithm is guaranteed to find such a hyperplane in a finite number of steps. This is known as the **perceptron convergence theorem**. The number of updates required depends on the margin (the distance between the closest point and the separating hyperplane) and the scale of the data. If the data are **not linearly separable**, the perceptron will never converge; it will keep updating $\theta$ indefinitely as it tries to correct misclassifications that cannot all be fixed by a single linear boundary. In practice, modifications such as limiting the number of iterations or using averaged weights can be used in these cases.

Despite its historical importance and conceptual simplicity, the perceptron has several limitations:
- It does **not provide probabilistic outputs**—only hard class labels (0 or 1).
- It **cannot solve non-linearly separable problems** (such as XOR) with a single layer.
- There is **no underlying cost function** that the perceptron is guaranteed to minimize, unlike logistic regression which maximizes the likelihood of the data.

While the perceptron and logistic regression have similar-looking update rules, logistic regression uses a smooth, differentiable sigmoid function and is derived from a probabilistic model. Logistic regression outputs probabilities and can be interpreted statistically, while the perceptron outputs only binary class labels. Logistic regression minimizes the logistic loss (cross-entropy), while the perceptron does not minimize a well-defined loss function over the whole dataset.

The perceptron, introduced by Frank Rosenblatt in 1958, was one of the earliest models of a neuron and inspired much of the early work in artificial intelligence and neural networks. While its limitations led to a temporary decline in neural network research (the "AI winter"), the perceptron laid the groundwork for modern deep learning, where multi-layer networks can overcome the limitations of the single-layer perceptron.

### Python Implementation

Below are Python code snippets that correspond to the perceptron equations and calculations described above:

```python
import numpy as np

def perceptron_threshold(z):
    """Threshold function: returns 1 if z >= 0, else 0."""
    return 1 if z >= 0 else 0

# Vectorized version for arrays
perceptron_threshold_vec = np.vectorize(perceptron_threshold)

def predict(theta, x):
    """Compute perceptron prediction for input x and weights theta."""
    z = np.dot(theta, x)
    return perceptron_threshold(z)

# Perceptron update rule for a single example
# theta: parameter vector
# x: input feature vector (including bias term)
# y: true label (0 or 1)
# alpha: learning rate

def perceptron_update(theta, x, y, alpha):
    """Update theta using the perceptron learning rule."""
    prediction = predict(theta, x)
    theta = theta + alpha * (y - prediction) * x
    return theta

# Example usage:
# Initialize parameters
alpha = 0.1
theta = np.zeros(3)  # For 2 features + bias
x = np.array([1, 2, 3])  # Example input (with bias term as x[0]=1)
y = 1  # True label

# Perform one update
theta = perceptron_update(theta, x, y, alpha)
print("Updated theta:", theta)
```

This code demonstrates the perceptron threshold function, prediction, and update rule for a single training example. In practice, you would iterate over your dataset, applying the update rule to each example, possibly for multiple epochs.

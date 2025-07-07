# Deep learning

Deep learning is a subfield of machine learning that focuses on learning data representations through neural networks with many layers. It has revolutionized fields such as computer vision, natural language processing, and reinforcement learning. The power of deep learning comes from its ability to automatically extract hierarchical features from raw data, enabling breakthroughs in tasks like image classification, speech recognition, and game playing.

Some key applications of deep learning include:
- Image and speech recognition (e.g., ImageNet, Siri, Google Assistant)
- Natural language processing (e.g., machine translation, chatbots)
- Game playing (e.g., AlphaGo, OpenAI Five)
- Medical diagnosis, self-driving cars, and more

Deep learning models are typically trained using large datasets and powerful hardware (GPUs/TPUs), and they rely on optimization techniques such as stochastic gradient descent and backpropagation.

---

## 7.1 Supervised learning with non-linear models

In the supervised learning setting (predicting $y$ from the input $x$), suppose our model/hypothesis is $h_\theta(x)$. In the past lectures, we have considered the cases when $h_\theta(x) = \theta^T x$ (in linear regression) or $h_\theta(x) = \theta^T \phi(x)$ (where $\phi(x)$ is the feature map). A commonality of these two models is that they are linear in the parameters $\theta$. Next we will consider learning general family of models that are **non-linear in both** the parameters $\theta$ and the inputs $x$. The most common non-linear models are neural networks, which we will define starting from the next section. For this section, it suffices to think $h_\theta(x)$ as an abstract non-linear model.

**Why non-linear models?**
- Linear models are limited in their expressiveness; they can only capture linear relationships between input and output.
- Non-linear models, such as neural networks, can approximate complex functions and decision boundaries, enabling them to solve tasks that are impossible for linear models.
- Example: XOR function cannot be represented by a linear model, but a neural network with a hidden layer can represent it.

**Example of a non-linear model:**
- Polynomial regression: $h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2$
- Neural network: $h_\theta(x) = \sigma(W_2 \sigma(W_1 x + b_1) + b_2)$, where $\sigma$ is a non-linear activation function (e.g., ReLU, sigmoid)

Suppose $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$ are the training examples. We will define the nonlinear model and the loss/cost function for learning it.

---

### Regression problems

For simplicity, we start with the case where the output is a real number, that is, $y^{(i)} \in \mathbb{R}$, and thus the model $h_\theta$ also outputs a real number $h_\theta(x) \in \mathbb{R}$. We define the least square cost function for the $i$-th example $(x^{(i)}, y^{(i)})$ as

```math
J^{(i)}(\theta) = \frac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2,
```

and define the mean-square cost function for the dataset as

```math
J(\theta) = \frac{1}{n} \sum_{i=1}^n J^{(i)}(\theta),
```

**Python code for mean squared error (MSE):**
```python
import numpy as np

def mse(y_true, y_pred):
    return np.mean(0.5 * (y_pred - y_true) ** 2)

# Example usage:
y_true = np.array([2, 4])
y_pred = np.array([2, 4])
print(mse(y_true, y_pred))  # Output: 0.0
```

**Why mean squared error?**
- It corresponds to the maximum likelihood estimator under the assumption of Gaussian noise in the outputs.
- It penalizes large errors more heavily than small errors, making it sensitive to outliers.

**Alternative loss functions:**
- Mean Absolute Error (MAE): $\frac{1}{n} \sum_{i=1}^n |h_\theta(x^{(i)}) - y^{(i)}|$
- Huber loss: Combines MSE and MAE, less sensitive to outliers than MSE.

**Worked example:**
Suppose $x = [1, 2]$, $y = [2, 4]$, and $h_\theta(x) = 2x$. Then $J^{(1)}(\theta) = \frac{1}{2}(2 \ast 1 - 2)^2 = 0 \text{ , } J^{(2)}(\theta) = \frac{1}{2}(2 \ast 2 - 4)^2 = 0 \text{.}$ So $J(\theta) = 0$ (perfect fit).

---

### Binary classification

Next we define the model and loss function for binary classification. Suppose the inputs $x \in \mathbb{R}^d$. Let $h_\theta : \mathbb{R}^d \to \mathbb{R}$ be a parameterized model (the analog of $\theta^T x$ in logistic linear regression). We call the output $h_\theta(x) \in \mathbb{R}$ the logit. Analogous to Section 2.1, we use the logistic function $g(\cdot)$ to turn the logit $h_\theta(x)$ to a probability $h_\theta(x) \in [0, 1]$:

```math
h_\theta(x) = g(h_\theta(x)) = 1/(1 + \exp(-h_\theta(x))).
```

**Python code for sigmoid function:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Example usage:
print(sigmoid(0))  # Output: 0.5
```

**Intuition:**
- The sigmoid function squashes any real-valued input to the range $(0, 1)$, making it suitable for modeling probabilities.
- The output $h_\theta(x)$ can be interpreted as the probability that $y=1$ given $x$ and $\theta$.

**Plot:**
- The sigmoid function is S-shaped, with $g(0) = 0.5$, $g(-\infty) \to 0$, $g(+\infty) \to 1$.

We model the conditional distribution of $y$ given $x$ and $\theta$ by

```math
P(y = 1 \mid x; \theta) = h_\theta(x)
```
```math
P(y = 0 \mid x; \theta) = 1 - h_\theta(x)
```

**Loss function:**

Following the same derivation in Section 2.1 and using the derivation in Remark 2.1.1, the negative log-likelihood loss function is equal to:

```math
J^{(i)}(\theta) = - \log p(y^{(i)} \mid x^{(i)}; \theta) = \ell_{logistic}(h_\theta(x^{(i)}), y^{(i)})
```
(7.4)

**Python code for binary cross-entropy (log-loss):**
```python
def binary_cross_entropy(y_true, y_pred):
    # y_true and y_pred are numpy arrays
    eps = 1e-15  # to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example usage:
y_true = np.array([1, 0, 1])
y_pred = np.array([0.9, 0.1, 0.8])
print(binary_cross_entropy(y_true, y_pred))
```

- This is also called the binary cross-entropy or log-loss.
- It penalizes confident but wrong predictions heavily.
- The loss is minimized when the predicted probability matches the true label.

As done in equation (7.2), the total loss function is also defined as the average of the loss function over individual training examples,
```math
J(\theta) = \frac{1}{n} \sum_{i=1}^n J^{(i)}(\theta).
```

---

### Multi-class classification

Following Section 2.3, we consider a classification problem where the response variable $y$ can take on any one of $k$ values, i.e. $y \in \{1, 2, \ldots, k\}$. Let $\hat{h}_{\theta}$ : $\mathbb{R} ^d \to \mathbb{R} ^k$ be a parameterized model. We call the outputs $\hat{h}_\theta(x) \in \mathbb{R}^k$ the logits. Each logit corresponds to the prediction for one of the $k$ classes. Analogous to Section 2.3, we use the softmax function to turn the logits $\hat{h}_\theta(x)$ into a probability vector with non-negative entries that sum up to 1:

```math
P(y = s \mid x; \theta) = \frac{\exp(\hat{h}_\theta(x)_s)}{\sum_{j=1}^k \exp(\hat{h}_\theta(x)_j)},
```
(7.5)

**Python code for softmax function:**
```python
def softmax(logits):
    exps = np.exp(logits - np.max(logits))  # for numerical stability
    return exps / np.sum(exps)

# Example usage:
logits = np.array([2, 1, 0])
print(softmax(logits))  # Output: [0.66524096 0.24472847 0.09003057]
```

where $\hat{h}_\theta(x)_s$ denotes the $s$-th coordinate of $\hat{h}_\theta(x)$.

**Why softmax?**
- Softmax generalizes the sigmoid to multiple classes, ensuring the outputs are positive and sum to 1.
- Each output can be interpreted as the probability of the corresponding class.

**Worked example:**
Suppose $\hat{h}_\theta(x) = [2, 1, 0]$. Then
$P(y=1|x) = \frac{e^2}{e^2 + e^1 + e^0} \approx 0.665$, $P(y=2|x) \approx 0.245$, $P(y=3|x) \approx 0.090$.

Similarly to Section 2.3, the loss function for a single training example $(x^{(i)}, y^{(i)})$ is its negative log-likelihood:

```math
J^{(i)}(\theta) = - \log p(y^{(i)} \mid x^{(i)}; \theta) = - \log \left( \frac{\exp(\hat{h}_\theta(x^{(i)})_{y^{(i)}})}{\sum_{j=1}^k \exp(\hat{h}_\theta(x^{(i)})_j)} \right).
```
(7.6)

**Python code for categorical cross-entropy (multi-class log-loss):**
```python
def categorical_cross_entropy(y_true, y_pred):
    # y_true is a one-hot vector, y_pred is a probability vector
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred))

# Example usage:
y_true = np.array([1, 0, 0])
y_pred = np.array([0.7, 0.2, 0.1])
print(categorical_cross_entropy(y_true, y_pred))
```

**Cross-entropy loss:**
- Measures the difference between the true label distribution and the predicted distribution.
- Encourages the model to assign high probability to the correct class.

Using the notations of Section 2.3, we can simply write in an abstract way:

```math
J^{(i)}(\theta) = \ell_{ce}(\hat{h}_\theta(x^{(i)}), y^{(i)}).
```
(7.7)

The loss function is also defined as the average of the loss function of individual training examples,
```math
J(\theta) = \frac{1}{n} \sum_{i=1}^n J^{(i)}(\theta).
```

---

### Exponential family generalization

We also note that the approach above can also be generated to any conditional probabilistic model where we have an exponential distribution for $y$, Exponential-family$(y; \eta)$, where $\eta = h_\theta(x)$ is a parameterized nonlinear function of $x$. However, the most widely used situations are the three cases discussed above.

**What is the exponential family?**
- A broad class of probability distributions that includes Gaussian, Bernoulli, Poisson, and many others.
- Many loss functions in machine learning arise from the negative log-likelihood of exponential family distributions.

**Examples:**
- Gaussian (regression), Bernoulli (binary classification), Categorical (multi-class), Poisson (count data)

---

### Optimizers (GD, SGD, Mini-batch SGD)

Commonly, people use gradient descent (GD), stochastic gradient (SGD), or their variants to optimize the loss function $J(\theta)$. GD's update rule can be written as$^2$

```math
\theta := \theta - \alpha \nabla J(\theta)
```
(7.8)

**Python code for a single gradient descent update:**
```python
def gradient_descent_update(theta, grad, alpha):
    return theta - alpha * grad

# Example usage:
theta = np.array([1.0, 2.0])
grad = np.array([0.1, -0.2])
alpha = 0.01
print(gradient_descent_update(theta, grad, alpha))
```

where $\alpha > 0$ is often referred to as the learning rate or step size. Next, we introduce a version of the SGD (Algorithm 1), which is lightly different from that in the first lecture notes.

**Intuition:**
- Gradient descent moves the parameters in the direction that most rapidly decreases the loss.
- The learning rate $\alpha$ controls the step size; too large can cause divergence, too small can slow convergence.

**Variants:**
- **Batch GD:** Uses all data to compute the gradient (slow for large datasets).
- **Stochastic GD:** Uses one example at a time (noisy but fast updates).
- **Mini-batch GD:** Uses a small batch (common in deep learning; balances speed and stability).
- **Momentum, RMSProp, Adam:** Advanced optimizers that adapt the learning rate or use momentum to accelerate convergence.

**Comparison table:**
| Method         | Speed   | Memory | Noise   |
|----------------|---------|--------|---------|
| Batch GD       | Slow    | High   | Low     |
| Stochastic GD  | Fast    | Low    | High    |
| Mini-batch GD  | Medium  | Medium | Medium  |

**Python pseudocode for SGD:**
```python
for epoch in range(num_epochs):
    for x_i, y_i in data:
        grad = compute_gradient(x_i, y_i, theta)
        theta = theta - alpha * grad
```

---

**Algorithm 1 Stochastic Gradient Descent**

1. Hyperparameter: learning rate $\alpha$, number of total iteration $n_{iter}$.
2. Initialize $\theta$ randomly.
3. **for** $i = 1$ to $n_{iter}$ **do**
    1. Sample $j$ uniformly from $\{1, \ldots, n\}$, and update $\theta$ by
        
        ```math
        \theta := \theta - \alpha \nabla_\theta J^{(j)}(\theta)
        ```
        (7.9)

Oftentimes computing the gradient of $B$ examples simultaneously for the parameter $\theta$ can be faster than computing $B$ gradients separately due to hardware parallelization. Therefore, a mini-batch version of SGD is most commonly used in deep learning, as shown in Algorithm 2. There are also other variants of the SGD or mini-batch SGD with slightly different sampling schemes.

**Algorithm 2 Mini-batch Stochastic Gradient Descent**

1. Hyperparameters: learning rate $\alpha$, batch size $B$, # iterations $n_{iter}$.
2. Initialize $\theta$ randomly
3. **for** $i = 1$ to $n_{iter}$ **do**
    1. Sample $B$ examples $j_1, \ldots, j_B$ (without replacement) uniformly from $\{1, \ldots, n\}$, and update $\theta$ by
        
```math
\theta := \theta - \frac{\alpha}{B} \sum_{k=1}^B \nabla_\theta J^{(j_k)}(\theta)
```
(7.10)

**Python code for a mini-batch SGD update:**
```python
def minibatch_sgd_update(theta, grads, alpha):
    # grads: array of gradients for each example in the batch
    return theta - alpha * np.mean(grads, axis=0)

# Example usage:
theta = np.array([1.0, 2.0])
grads = np.array([[0.1, -0.2], [0.05, -0.1]])
alpha = 0.01
print(minibatch_sgd_update(theta, grads, alpha))
```

---

### Practical deep learning workflow

With these generic algorithms, a typical deep learning model is learned with the following steps:
1. **Model design:** Define a neural network parametrization $h_\theta(x)$ (layers, activations, etc.), which we will introduce in Section 7.2.
2. **Gradient computation:** Write the backpropagation algorithm to compute the gradient of the loss function $J^{(i)}(\theta)$ efficiently (see Section 7.4).
3. **Optimization:** Run SGD or mini-batch SGD (or other gradient-based optimizers) with the loss function $J(\theta)$.
4. **Data preprocessing:** Normalize/standardize inputs, handle missing data, and split into train/validation/test sets.
5. **Regularization:** Use techniques like dropout, weight decay, or early stopping to prevent overfitting.
6. **Evaluation:** Monitor training and validation loss/accuracy, and tune hyperparameters as needed.

---

*This concludes the overview of supervised learning with non-linear models. In the next sections, we will introduce neural network architectures, vectorization, and backpropagation in detail.*
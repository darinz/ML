# Classification and logistic regression

Classification is a fundamental task in machine learning where the goal is to assign input data points to one of several predefined categories or classes. Unlike regression, where the output variable is continuous, classification deals with discrete outputs. In binary classification, there are only two possible classes, often labeled as 0 and 1 (or negative and positive). For example, in email spam detection, the input features might represent properties of an email, and the output is 1 if the email is spam and 0 otherwise. The terms **negative class** and **positive class** are used to refer to these two categories, and the output variable is often called the **label**.

## 2.1 Logistic regression

### Why Not Use Linear Regression for Classification?
One might consider using linear regression for classification by thresholding its output, but this approach is problematic. Linear regression can produce predictions less than 0 or greater than 1, which do not make sense for probabilities. Moreover, the relationship between the input features and the probability of class membership is often nonlinear, especially when the classes are not linearly separable. This motivates the need for a model that outputs values strictly between 0 and 1 and models the probability of class membership directly.

### The Logistic (Sigmoid) Function
The **logistic function**, also known as the **sigmoid function**, is defined as:

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

This function has several important properties:
- Its output is always between 0 and 1, making it suitable for modeling probabilities.
- As $z \to \infty$, $g(z) \to 1$; as $z \to -\infty$, $g(z) \to 0$.
- The function is S-shaped (sigmoidal), which allows it to model the smooth transition between classes.

**Python code for the sigmoid function:**
```python
import numpy as np

def sigmoid(z):
    """Compute the sigmoid function."""
    return 1 / (1 + np.exp(-z))
```

In logistic regression, we model the probability that $y = 1$ given $x$ as:

$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

where $\theta$ is the parameter vector.

**Python code for the hypothesis:**
```python
def hypothesis(theta, x):
    """Compute the logistic regression hypothesis h_theta(x)."""
    return sigmoid(np.dot(theta, x))
```

<img src="./img/sigmoid.png" width="300px" />

Notice that $g(z)$ tends towards 1 as $z \to \infty$, and $g(z)$ tends towards 0 as $z \to -\infty$. Moreover, $g(z)$, and hence also $h(x)$, is always bounded between 0 and 1. This is crucial for interpreting the output as a probability. As before, we are keeping the convention of letting $x_0 = 1$, so that $\theta^T x = \theta_0 + \sum_{j=1}^d \theta_j x_j$.

For now, let's take the choice of $g$ as given. Other functions that smoothly increase from 0 to 1 can also be used, but for a couple of reasons that we'll see later (when we talk about GLMs, and when we talk about generative learning algorithms), the choice of the logistic function is a fairly natural one. The logistic function is the canonical link function for the Bernoulli distribution in the framework of generalized linear models (GLMs), which provides a deeper theoretical justification for its use.

#### Derivative of the Sigmoid Function
Before moving on, here's a useful property of the derivative of the sigmoid function, which we write as $g'$:

$$
\begin{align*}
g'(z) &= \frac{d}{dz} \frac{1}{1 + e^{-z}} \\
      &= \frac{1}{(1 + e^{-z})^2} (e^{-z}) \\
      &= \frac{1}{(1 + e^{-z})} \cdot \left(1 - \frac{1}{1 + e^{-z}}\right) \\
      &= g(z)(1 - g(z)).
\end{align*}
$$

This elegant result greatly simplifies the computation of gradients during optimization.

**Python code for the derivative of the sigmoid function:**
```python
def sigmoid_derivative(z):
    """Compute the derivative of the sigmoid function."""
    s = sigmoid(z)
    return s * (1 - s)
```

### Probabilistic Interpretation
Logistic regression provides a probabilistic framework for classification. We interpret the output of the model as the probability that the label is 1 given the input features:

$$
\begin{align*}
P(y = 1 \mid x; \theta) &= h_\theta(x) \\
P(y = 0 \mid x; \theta) &= 1 - h_\theta(x)
\end{align*}
$$

This probabilistic interpretation allows us to use principles from statistics, such as maximum likelihood estimation, to fit the model parameters. It also means that logistic regression can be used not only for hard classification (predicting 0 or 1), but also for estimating the probability of class membership, which is useful in many applications.

### Likelihood and Log-Likelihood
Given a dataset of $n$ independent training examples, the likelihood of the parameters $\theta$ is the probability of observing the data given the model:

$$
L(\theta) = \prod_{i=1}^n p(y^{(i)} \mid x^{(i)}; \theta)
$$

For logistic regression, this becomes:

$$
L(\theta) = \prod_{i=1}^n (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1 - y^{(i)}}
$$

Maximizing the likelihood is equivalent to maximizing the log-likelihood:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^n y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))
$$

The log-likelihood is easier to work with mathematically and numerically, as it turns products into sums and avoids numerical underflow for small probabilities.

**Python code for the log-likelihood:**
```python
def log_likelihood(theta, X, y):
    """Compute the log-likelihood for logistic regression."""
    h = sigmoid(X @ theta)
    return np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
```

### Gradient Ascent for Logistic Regression
To find the parameters $\theta$ that maximize the log-likelihood, we use **gradient ascent** (since we are maximizing, not minimizing). The update rule for each parameter $\theta_j$ is:

$$
\theta_j := \theta_j + \alpha \frac{\partial}{\partial \theta_j} \ell(\theta)
$$

where $\alpha$ is the learning rate. The gradient for a single training example is:

$$
\frac{\partial}{\partial \theta_j} \ell(\theta) = (y - h_\theta(x)) x_j
$$

This update rule has a similar form to the update rule in linear regression, but here $h_\theta(x)$ is a nonlinear function of $\theta^T x$. The similarity in the update rules is a result of the mathematical structure of the models, but the learning problems and interpretations are distinct.

**Python code for the gradient and parameter update:**
```python
def gradient(theta, X, y):
    """Compute the gradient of the log-likelihood."""
    h = sigmoid(X @ theta)
    return X.T @ (y - h)

# Gradient ascent update
alpha = 0.01  # learning rate
# theta = theta + alpha * gradient(theta, X, y)
```

Above, we used the fact that $g'(z) = g(z)(1 - g(z))$. This therefore gives us the stochastic gradient ascent rule

$$
\theta_j := \theta_j + \alpha \left(y^{(i)} - h_\theta(x^{(i)})\right)x_j^{(i)}
$$

If we compare this to the LMS update rule, we see that it looks identical; but this is *not* the same algorithm, because $h_\theta(x^{(i)})$ is now defined as a non-linear function of $\theta^T x^{(i)}$. Nonetheless, it's a little surprising that we end up with the same update rule for a rather different algorithm and learning problem. Is this coincidence, or is there a deeper reason behind this? We'll answer this when we get to GLM models.

### The Logistic Loss and Logit
The **logistic loss** (or log-loss) is another way to express the cost function for logistic regression:

$$
\ell_{\text{logistic}}(t, y) = y \log(1 + \exp(-t)) + (1 - y) \log(1 + \exp(t))
$$

where $t = \theta^T x$ is called the **logit**. The logit represents the unbounded score before applying the sigmoid function. The logistic loss penalizes incorrect predictions more heavily, especially when the model is confident but wrong.

**Python code for the logistic loss:**
```python
def logistic_loss(t, y):
    """Compute the logistic loss for a single example."""
    return y * np.log(1 + np.exp(-t)) + (1 - y) * np.log(1 + np.exp(t))
```

One can verify by plugging in $h_\theta(x) = 1/(1 + e^{-\theta^T x})$ that the negative log-likelihood (the negation of $\ell(\theta)$ in equation (2.1)) can be re-written as
$$
-\ell(\theta) = \ell_{\text{logistic}}(\theta^T x, y)
$$
Oftentimes $\theta^T x$ or $t$ is called the *logit*. Basic calculus gives us that
$$
\frac{\partial \ell_{\text{logistic}}(t, y)}{\partial t} = \frac{1}{1 + \exp(-t)} - y = h_\theta(x) - y
$$
This leads to the same gradient update as before.

Then, using the chain rule, we have that
$$
\frac{\partial}{\partial \theta_j} \ell(\theta) = -\frac{\partial \ell_{\text{logistic}}(t, y)}{\partial t} \cdot \frac{\partial t}{\partial \theta_j} = (y - 1/(1 + \exp(-t))) \cdot x_j = (y - h_\theta(x)) x_j,
$$
which is consistent with the derivation in equation (2.2). We will see this viewpoint can be extended in nonlinear models in Section 7.1.

### Summary and Further Reading
Logistic regression is a foundational algorithm for binary classification. It combines ideas from linear models, probability theory, and optimization. Its probabilistic interpretation makes it a natural choice for many applications, and its loss function and gradient have elegant mathematical properties. For more advanced topics, such as regularization, multiclass classification, and connections to generalized linear models (GLMs), see later sections.


# 1.3 Probabilistic interpretation

When faced with a regression problem, why might linear regression, and specifically why might the least-squares cost function $J$, be a reasonable choice? In this section, we will give a set of probabilistic assumptions, under which least-squares regression is derived as a very natural algorithm.

## Linear Model Assumption

Let us assume that the target variables and the inputs are related via the equation

$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)},
$$

where $y^{(i)}$ is the observed output for the $i$-th data point, $x^{(i)}$ is the corresponding input feature vector, $\theta$ is the parameter vector we wish to learn, and $\epsilon^{(i)}$ is an error term. This model asserts that the relationship between the inputs and outputs is linear, up to some noise or unmodeled effects.

> **Matrix invertibility note:** In the above step, we are implicitly assuming that $X^T X$ is an invertible matrix, where $X$ is the design matrix whose rows are the $x^{(i)}$'s. This can be checked before calculating the inverse. If either the number of linearly independent examples is fewer than the number of features, or if the features are not linearly independent, then $X^T X$ will not be invertible. Even in such cases, it is possible to "fix" the situation with additional techniques, such as regularization (ridge regression), which we skip here for the sake of simplicity.

## Gaussian Noise Model

The error term $\epsilon^{(i)}$ captures either unmodeled effects (such as if there are some features very pertinent to predicting housing price, but that we'd left out of the regression), or random noise. To proceed probabilistically, we further assume that the $\epsilon^{(i)}$ are distributed IID (independently and identically distributed) according to a Gaussian (Normal) distribution with mean zero and some variance $\sigma^2$:

$$
\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)
$$

This is a common assumption in statistics and machine learning, as the Gaussian distribution arises naturally in many contexts (e.g., by the Central Limit Theorem). It also leads to mathematically convenient properties.

The density of $\epsilon^{(i)}$ is given by:

$$
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)
$$

This means that most of the time, the error is close to zero, but larger deviations are possible with exponentially decreasing probability.

## Conditional Distribution of $y^{(i)}$

Given the linear model and the Gaussian noise, the conditional distribution of $y^{(i)}$ given $x^{(i)}$ and $\theta$ is also Gaussian:

$$
p(y^{(i)}|x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)
$$

This says that, for a given input $x^{(i)}$, the output $y^{(i)}$ is most likely to be near $\theta^T x^{(i)}$, but can deviate from it due to noise.

The notation $p(y^{(i)}|x^{(i)}; \theta)$ indicates that this is the probability density of $y^{(i)}$ given $x^{(i)}$ and parameterized by $\theta$. Note that $\theta$ is not a random variable here; it is a parameter to be estimated.

## Likelihood Function

Given $X$ (the design matrix, which contains all the $x^{(i)}$'s) and $\theta$, what is the probability of observing the data $\vec{y}$? The likelihood function is defined as:

$$
L(\theta; X, \vec{y}) = p(\vec{y}|X; \theta)
$$

Assuming the data points are independent given $\theta$, the likelihood factorizes as a product:

$$
L(\theta) = \prod_{i=1}^n p(y^{(i)} \mid x^{(i)}; \theta)
= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)
$$

This function measures how probable the observed data is, as a function of $\theta$.

## Maximum Likelihood Estimation (MLE)

The principle of **maximum likelihood** says that we should choose $\theta$ to maximize the likelihood $L(\theta)$, i.e., to make the observed data as probable as possible under our model. This is a general principle in statistics and machine learning for parameter estimation.

## Log-Likelihood and Connection to Least Squares

Maximizing the likelihood is equivalent to maximizing any strictly increasing function of it, such as the log-likelihood. Taking the logarithm simplifies the product into a sum, which is easier to work with:

$$
\begin{align*}
\ell(\theta) &= \log L(\theta) \\
&= \log \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right) \\
&= \sum_{i=1}^n \log \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right) \\
&= n \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{1}{2\sigma^2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2
\end{align*}
$$

Notice that the first term does not depend on $\theta$, so maximizing $\ell(\theta)$ is equivalent to minimizing the sum of squared errors:

$$
\frac{1}{2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2
$$

This is exactly the least-squares cost function $J(\theta)$ used in linear regression. Thus, under the Gaussian noise assumption, least-squares regression is equivalent to maximum likelihood estimation.

## Summary and Broader Perspective

To summarize: Under the previous probabilistic assumptions on the data, least-squares regression corresponds to finding the maximum likelihood estimate of $\theta$. This is thus one set of assumptions under which least-squares regression can be justified as a very natural method that's just doing maximum likelihood estimation. 

However, the probabilistic assumptions are by no means **necessary** for least-squares to be a perfectly good and rational procedure, and there may—and indeed there are—other natural assumptions that can also be used to justify it. For example, least-squares can also be motivated from a geometric perspective (minimizing the Euclidean distance between predictions and observations), or as a method of moments estimator.

> **Practical note:** In real-world data, the Gaussian noise assumption may not always hold. Outliers, heteroscedasticity (non-constant variance), or non-linear relationships can violate the assumptions. In such cases, alternative models or robust regression techniques may be more appropriate.

## Independence from $\sigma^2$

Note also that, in our previous discussion, our final choice of $\theta$ did not depend on what was $\sigma^2$, and indeed we'd have arrived at the same result even if $\sigma^2$ were unknown. This is because $\sigma^2$ only affects the scaling of the likelihood, not the location of its maximum with respect to $\theta$. We will use this fact again later, when we talk about the exponential family and generalized linear models.

## Python Code: Probabilistic Linear Regression

Below are Python code snippets that illustrate the key equations and calculations from this section. These examples use `numpy` for numerical operations.

### 1. Linear Model and Data Generation
```python
import numpy as np

# Parameters
np.random.seed(0)
n = 100  # number of data points
p = 2    # number of features

# True parameters
theta_true = np.array([2.0, -3.0])
sigma = 1.0

# Generate random features and noise
X = np.random.randn(n, p)
epsilon = np.random.normal(0, sigma, size=n)

# Generate targets according to the linear model
y = X @ theta_true + epsilon
```

### 2. Gaussian Likelihood for a Single Data Point
```python
def gaussian_likelihood(y_i, x_i, theta, sigma):
    """Compute p(y_i | x_i; theta) for the Gaussian model."""
    mu = np.dot(theta, x_i)
    coeff = 1.0 / np.sqrt(2 * np.pi * sigma**2)
    exponent = -((y_i - mu) ** 2) / (2 * sigma**2)
    return coeff * np.exp(exponent)

# Example usage:
p_likelihood = gaussian_likelihood(y[0], X[0], theta_true, sigma)
print(f"Likelihood of first point: {p_likelihood:.4f}")
```

### 3. Log-Likelihood for the Whole Dataset
```python
def log_likelihood(y, X, theta, sigma):
    n = len(y)
    mu = X @ theta
    ll = -0.5 * n * np.log(2 * np.pi * sigma**2)
    ll -= np.sum((y - mu) ** 2) / (2 * sigma**2)
    return ll

# Example usage:
ll_val = log_likelihood(y, X, theta_true, sigma)
print(f"Log-likelihood at true theta: {ll_val:.2f}")
```

### 4. Least-Squares Solution (Normal Equations)
```python
# Closed-form solution for theta (assuming X^T X is invertible)
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Estimated theta: {theta_hat}")
```

### 5. Mean Squared Error (Cost Function)
```python
def mean_squared_error(y, X, theta):
    return np.mean((y - X @ theta) ** 2)

mse = mean_squared_error(y, X, theta_hat)
print(f"Mean squared error at estimated theta: {mse:.4f}")
```

These code snippets demonstrate how the probabilistic interpretation of linear regression translates directly into practical calculations and model fitting in Python.

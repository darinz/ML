# Generalized linear models

So far, we've seen a regression example, and a classification example. In the regression example, we had $y|x; \theta \sim \mathcal{N}(\mu, \sigma^2)$, and in the classification one, $y|x; \theta \sim \text{Bernoulli}(\phi)$, for some appropriate definitions of $\mu$ and $\phi$ as functions of $x$ and $\theta$. In this section, we will show that both of these methods are special cases of a broader family of models, called Generalized Linear Models (GLMs). We will also show how other models in the GLM family can be derived and applied to other classification and regression problems.

## 3.1 The exponential family

To work our way up to GLMs, we will begin by defining exponential family distributions. We say that a class of distributions is in the exponential family if it can be written in the form

```math
p(y; \eta) = b(y) \exp(\eta^T T(y) - a(\eta)) \tag{3.1}
```

```python
import numpy as np

def exponential_family_p(y, eta, T, a, b):
    """
    Generic exponential family probability calculation.
    y: observed value
    eta: natural parameter (can be vector)
    T: sufficient statistic function
    a: log partition function
    b: base measure function
    """
    return b(y) * np.exp(np.dot(eta, T(y)) - a(eta))
```

Here, $\eta$ is called the **natural parameter** (also called the canonical parameter) of the distribution; $T(y)$ is the **sufficient statistic** (for the distributions we consider, it will often be the case that $T(y) = y$); and $a(\eta)$ is the **log partition function**. The quantity $e^{-a(\eta)}$ essentially plays the role of a normalization constant, that makes sure the distribution $p(y; \eta)$ sums/integrates over $y$ to 1.

A fixed choice of $T$, $a$ and $b$ defines a *family* (or set) of distributions that is parameterized by $\eta$; as we vary $\eta$, we then get different distributions within this family.

We now show that the Bernoulli and the Gaussian distributions are examples of exponential family distributions. The Bernoulli distribution with mean $\phi$, written Bernoulli($\phi$), specifies a distribution over $y \in \{0, 1\}$, so that $p(y = 1; \phi) = \phi$; $p(y = 0; \phi) = 1 - \phi$. As we vary $\phi$, we obtain Bernoulli distributions with different means. We now show that this class of Bernoulli distributions, ones obtained by varying $\phi$, is in the exponential family; i.e., that there is a choice of $T$, $a$ and $b$ so that Equation (3.1) becomes exactly the class of Bernoulli distributions.

We write the Bernoulli distribution as:

```math
p(y; \phi) = \phi^y (1 - \phi)^{1-y}
           = \exp\left(y \log \phi + (1-y) \log(1-\phi)\right)
           = \exp\left(\left(\log\left(\frac{\phi}{1-\phi}\right)\right) y + \log(1-\phi)\right)
```

```python
# Bernoulli distribution in exponential family form
import numpy as np

def bernoulli_p(y, phi):
    return phi**y * (1 - phi)**(1 - y)

def bernoulli_exp_family(y, phi):
    eta = np.log(phi / (1 - phi))
    a = np.log(1 + np.exp(eta))
    return np.exp(eta * y - a)

# Example usage:
y = 1
phi = 0.7
print('Bernoulli PMF:', bernoulli_p(y, phi))
print('Exponential family form:', bernoulli_exp_family(y, phi))
```

Thus, the natural parameter is given by $\eta = \log(\phi/(1-\phi))$. Interestingly, if we invert this definition for $\eta$ by solving for $\phi$ in terms of $\eta$, we obtain $\phi = 1/(1 + e^{-\eta})$. This is the familiar sigmoid function! This will come up again when we derive logistic regression as a GLM. To complete the formulation of the Bernoulli distribution as an exponential family distribution, we also have

```math
T(y) = y
```
```python
def T_bernoulli(y):
    return y
```
```math
a(\eta) = -\log(1-\phi) = \log(1 + e^{\eta})
```
```python
def a_bernoulli(eta):
    return np.log(1 + np.exp(eta))
```
```math
b(y) = 1
```
```python
def b_bernoulli(y):
    return 1
```

This shows that the Bernoulli distribution can be written in the form of Equation (3.1), using an appropriate choice of $T$, $a$ and $b$.

Let's now move on to consider the Gaussian distribution. Recall that, when deriving linear regression, the value of $\sigma^2$ had no effect on our final choice of $\theta$ and $h_\theta(x)$. Thus, we can choose an arbitrary value for $\sigma^2$ without changing anything. To simplify the derivation below, let's set $\sigma^2 = 1$.

We then have:

```math
p(y; \mu) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}(y-\mu)^2\right)
           = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}y^2\right) \cdot \exp\left(\mu y - \frac{1}{2}\mu^2\right)
```

```python
# Gaussian distribution in exponential family form
from scipy.stats import norm

def gaussian_p(y, mu):
    return norm.pdf(y, loc=mu, scale=1)

def gaussian_exp_family(y, mu):
    eta = mu
    a = mu**2 / 2
    b = (1 / np.sqrt(2 * np.pi)) * np.exp(-y**2 / 2)
    return b * np.exp(eta * y - a)

# Example usage:
y = 0.5
mu = 1.0
print('Gaussian PDF:', gaussian_p(y, mu))
print('Exponential family form:', gaussian_exp_family(y, mu))
```

Thus, we see that the Gaussian is in the exponential family, with

```math
\eta = \mu
```
```python
def eta_gaussian(mu):
    return mu
```
```math
T(y) = y
```
```python
def T_gaussian(y):
    return y
```
```math
a(\eta) = \mu^2/2 = \eta^2/2
```
```python
def a_gaussian(eta):
    return eta**2 / 2
```
```math
b(y) = (1/\sqrt{2\pi}) \exp(-y^2/2)
```
```python
def b_gaussian(y):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-y**2 / 2)
```

This shows that the Gaussian distribution can also be written in the form of Equation (3.1), using an appropriate choice of $T$, $a$ and $b$.

There are many other distributions that are members of the exponential family: The multinomial (which we'll see later), the Poisson (for modelling count-data; also see the problem set); the gamma and the exponential (for modelling continuous, non-negative random variables, such as time-intervals); the beta and the Dirichlet (for distributions over probabilities); and many more. In the next section, we will describe a general "recipe" for constructing models in which $y$ (given $x$ and $\theta$) comes from any of these distributions.

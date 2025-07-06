## 3.2 Constructing GLMs

Suppose you would like to build a model to estimate the number $y$ of customers arriving in your store (or number of page-views on your website) in any given hour, based on certain features $x$ such as store promotions, recent advertising, weather, day-of-week, etc. We know that the Poisson distribution usually gives a good model for numbers of visitors. Knowing this, how can we come up with a model for our problem? Fortunately, the Poisson is an exponential family distribution, so we can apply a Generalized Linear Model (GLM). In this section, we will describe a method for constructing GLM models for problems such as these.

**Intuition:**
Generalized Linear Models (GLMs) provide a unified framework for modeling a wide variety of prediction problems, including regression and classification. The key insight is that many common distributions (Gaussian, Bernoulli, Poisson, etc.) belong to the exponential family, which allows us to use a common recipe for building models. This recipe involves:
- Choosing a response distribution from the exponential family,
- Defining a linear predictor (a linear combination of input features),
- Specifying a link function that connects the linear predictor to the mean of the response,
- Estimating parameters, typically via maximum likelihood.

GLMs are powerful because they allow us to model different types of data (continuous, binary, counts, etc.) using a consistent approach, and they provide interpretable coefficients and well-understood statistical properties.

More generally, consider a classification or regression problem where we would like to predict the value of some random variable $y$ as a function of $x$. To derive a GLM for this problem, we will make the following three assumptions about the conditional distribution of $y$ given $x$ and about our model:

1. $y \mid x; \theta \sim \text{ExponentialFamily}(\eta)$. I.e., given $x$ and $\theta$, the distribution of $y$ follows some exponential family distribution, with parameter $\eta$.

2. Given $x$, our goal is to predict the expected value of $T(y)$ given $x$. In most of our examples, we will have $T(y) = y$, so this means we would like the prediction $h(x)$ output by our learned hypothesis $h$ to satisfy $h(x) = \mathbb{E}[y|x]$. (Note that this assumption is satisfied in the choices for $h_\theta(x)$ for both logistic regression and linear regression. For instance, in logistic regression, we had $h_\theta(x) = p(y = 1|x; \theta) = 0 \cdot p(y = 0|x; \theta) + 1 \cdot p(y = 1|x; \theta) = \mathbb{E}[y|x; \theta]$.)

3. The natural parameter $\eta$ and the inputs $x$ are related linearly: $\eta = \theta^T x$. (Or, if $y$ is vector-valued, then $\eta_i = \theta_i^T x$.)

```math
\begin{align*}
y \mid x; \theta &\sim \text{ExponentialFamily}(\eta) \\
h(x)\ &=\ \mathbb{E}[y|x] \\
\eta\ &=\ \theta^T x
\end{align*}
```

The third of these assumptions might seem the least well justified of the above, and it might be better thought of as a "design choice" in our recipe for designing GLMs, rather than as an assumption per se. These three assumptions/design choices will allow us to derive a very elegant class of learning algorithms, namely GLMs, that have many desirable properties such as ease of learning. Furthermore, the resulting models are often very effective for modelling different types of distributions over $y$; for example, we will shortly show that both logistic regression and ordinary least squares can both be derived as GLMs.

---

### 3.2.1 Ordinary least squares

To show that ordinary least squares is a special case of the GLM family of models, consider the setting where the target variable $y$ (also called the **response variable** in GLM terminology) is continuous, and we model the conditional distribution of $y$ given $x$ as a Gaussian $\mathcal{N}(\mu, \sigma^2)$. (Here, $\mu$ may depend on $x$.) So, we let the $ExponentialFamily(\eta)$ distribution above be the Gaussian distribution. As we saw previously, in the formulation of the Gaussian as an exponential family distribution, we had $\mu = \eta$. So, we have

```math
\begin{align*}
h_\theta(x) &= \mathbb{E}[y|x; \theta] \\
            &= \mu \\
            &= \eta \\
            &= \theta^T x
\end{align*}
```

The first equality follows from Assumption 2, above; the second equality follows from the fact that $y|x; \theta \sim \mathcal{N}(\mu, \sigma^2)$, and so its expected value is given by $\mu$; the third equality follows from Assumption 1 (and our earlier derivation showing that $\mu = \eta$ in the formulation of the Gaussian as an exponential family distribution); and the last equality follows from Assumption 3.

**Geometric Interpretation:**
Ordinary least squares (OLS) finds the line (or hyperplane) that minimizes the sum of squared vertical distances to the data points. This is equivalent to projecting the data onto the closest point in the subspace defined by the model.

**Link Function:**
The identity function, meaning the mean of $y$ is modeled directly as $\theta^T x$.

**Practical Note:**
OLS is optimal (in the sense of minimum variance unbiased estimation) when the errors are normally distributed and homoscedastic (constant variance). It is also computationally efficient and forms the basis for many extensions in statistics and machine learning.

**Python Example:**
```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([2, 3, 4])
model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)
```

---

### 3.2.2 Logistic regression

We now consider logistic regression. Here we are interested in binary classification, so $y \in \{0, 1\}$. Given that $y$ is binary-valued, it therefore seems natural to choose the Bernoulli family of distributions to model the conditional distribution of $y$ given $x$. In our formulation of the Bernoulli distribution as an exponential family distribution, we had $\phi = 1/(1 + e^{-\eta})$. Furthermore, note that if $y|x; \theta \sim \text{Bernoulli}(\phi)$, then $\mathbb{E}[y|x; \theta] = \phi$. So, following a similar derivation as the one for ordinary least squares, we get:

```math
\begin{align*}
h_\theta(x) &= \mathbb{E}[y|x; \theta] \\
            &= \phi \\
            &= \frac{1}{1 + e^{-\eta}} \\
            &= \frac{1}{1 + e^{-\theta^T x}}
\end{align*}
```

So, this gives us hypothesis functions of the form $h_\theta(x) = 1/(1 + e^{-\theta^T x})$. If you are previously wondering how we came up with the form of the logistic function $1/(1 + e^{-z})$, this gives one answer: Once we assume that $y$ conditioned on $x$ is Bernoulli, it arises as a consequence of the definition of GLMs and exponential family distributions.

**Intuition:**
The log-odds of the probability of $y=1$ is modeled as a linear function of $x$. This means that each feature contributes additively to the log-odds, making the model interpretable and robust.

**Link Function:**
The logit function (inverse of the logistic/sigmoid), which maps probabilities to the real line.

**Geometric Interpretation:**
Logistic regression finds the hyperplane that best separates the two classes in terms of probability. The decision boundary is where $h_\theta(x) = 0.5$.

**Practical Note:**
Logistic regression is robust, interpretable, and forms the basis for more complex classification models. It is widely used in practice for binary classification tasks.

**Python Example:**
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([0, 1, 1])
model = LogisticRegression().fit(X, y)
print(model.coef_, model.intercept_)
print(model.predict_proba(X))
```

To introduce a little more terminology, the function $g$ giving the distribution's mean as a function of the natural parameter ($g(\eta) = \mathbb{E}[T(y); \eta]$) is called the **canonical response function**. Its inverse, $g^{-1}$, is called the **canonical link function**. Thus, the canonical response function for the Gaussian family is just the identity function; and the canonical response function for the Bernoulli is the logistic function.\footnote{Many texts use $g$ to denote the link function, and $g^{-1}$ to denote the response function; but the notation we're using here, inherited from the early machine learning literature, will be more consistent with the notation used in the rest of the class.}

---

**Further Reading & Extensions:**
- GLMs can be extended to other distributions (Poisson for counts, Gamma for positive continuous data, etc.).
- The choice of link function and response distribution is guided by the nature of the data and the scientific question.
- For more, see the `01_exponential_family.md` for mathematical details and `exponential_family_examples.py` for code.

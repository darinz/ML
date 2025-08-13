# Practice 5 Solutions

** Problem 1. If $X$ and $Y$ are independent random variables, which of the following are true?**
*   (a) $\text{Cov}(X, Y) = 0$
*   (b) $E[XY] = E[X]E[Y]$
*   (c) $\text{Var}(XY) = \text{Var}(X)\text{Var}(Y)$
*   (d) $P(X,Y) = P(Y|X)P(X|Y)$

**Correct answers:** (a), (b), (d)

**Explanation:**

**Options (a), (b), and (d) are true** for independent random variables.

**Why (a) is true - Covariance is zero:**

**1. Definition of independence:**
- **X and Y are independent** if P(X,Y) = P(X)P(Y)
- **No relationship** between X and Y

**2. Covariance definition:**

$\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$

**3. For independent variables:**
- **$E[XY] = E[X]E[Y]$** (from option b)
- **Therefore:** $\text{Cov}(X,Y) = E[X]E[Y] - E[X]E[Y] = 0$

**Why (b) is true - Expectation of product:**

**1. Independence property:**
- **E[XY] = E[X]E[Y]** is a fundamental property of independence
- **No correlation** means no linear relationship

**2. Intuitive explanation:**
- **Independent variables** don't influence each other
- **Product expectation** factors into individual expectations

**Why (c) is false - Variance of product:**

**1. Correct formula:**

$\text{Var}(XY) = E[X^2]E[Y^2] - (E[X]E[Y])^2$

**2. This is NOT equal to:**

$\text{Var}(X)\text{Var}(Y) = (E[X^2] - E[X]^2)(E[Y^2] - E[Y]^2)$

**3. Example:**
- **X, Y ~ N(0,1)** independent
- **Var(XY) = 1** (not Var(X)Var(Y) = 1×1 = 1, but this is a special case)

**Why (d) is true - Joint probability:**

**1. For independent variables:**

$P(X,Y) = P(X)P(Y)$

**2. Also:**

$P(Y|X) = P(Y)$ and $P(X|Y) = P(X)$

**3. Therefore:**

$P(X,Y) = P(Y|X)P(X|Y) = P(Y)P(X)$ ✓

**Key insight:** **Independence** implies **no linear relationship** (zero covariance) and **factorization** of expectations and probabilities.

**2. A certain disease affects 2% of the population. A diagnostic test for this disease has the following characteristics:**
*   **Sensitivity (True Positive Rate):** If a person has the disease, the test returns a positive result with probability 0.90.
*   **False Positive Rate:** If a person does not have the disease, the test returns a positive result with probability 0.10.

**If a randomly selected person tests positive, what is the probability that they actually have the disease?**
*   (a) $\frac{11}{58}$
*   (b) $\frac{9}{58}$
*   (c) $\frac{9}{50}$
*   (d) $\frac{49}{58}$

**Correct answers:** (b)

**Explanation:**

**This is a classic Bayes' theorem problem** - finding the probability of having the disease given a positive test result.

**Step-by-step solution:**

**1. Define events:**
- **D** = "Person has the disease" (P(D) = 0.02)
- **T** = "Test result is positive"
- **D^c** = "Person does not have the disease" (P(D^c) = 0.98)

**2. Given information:**
- **Sensitivity:** P(T|D) = 0.90 (90% of diseased people test positive)
- **False Positive Rate:** P(T|D^c) = 0.10 (10% of healthy people test positive)

**3. Apply Bayes' theorem:**

$P(D|T) = \frac{P(T|D)P(D)}{P(T)}$

**4. Calculate P(T) using law of total probability:**

$P(T) = P(T|D)P(D) + P(T|D^c)P(D^c)$

$= (0.90)(0.02) + (0.10)(0.98)$

$= 0.018 + 0.098$

$= 0.116$

**5. Substitute into Bayes' theorem:**

$P(D|T) = \frac{(0.90)(0.02)}{0.116}$

$= \frac{0.018}{0.116}$

$= \frac{18}{116}$

$= \frac{9}{58}$

**6. Intuitive interpretation:**
- **Only 9/58 ≈ 15.5%** of positive test results indicate actual disease
- **Low prevalence** (2%) combined with **high false positive rate** (10%) leads to many false alarms
- **Most positive tests** are false positives due to the large healthy population

**7. Why other options are incorrect:**
- **(a) 11/58:** Incorrect calculation
- **(c) 9/50:** Wrong denominator
- **(d) 49/58:** Much too high

**Key insight:** **Low disease prevalence** combined with **imperfect test accuracy** means most positive results are **false positives**.

**3.**

**The probability mass function of a geometric distribution with unknown parameter $0 < p \le 1$ is**
$$P(X=k|p) = (1-p)^{k-1}p$$
**where $k = 1,2,3,....$ The interpretation of $X$ is that it is the number of independent Bernoulli trials needed to get one success, if each trial has success probability $p$.**

**Given a set of $n$ observations $\{x_1, x_2,..., x_n\}$ from a geometric distribution, derive the Maximum Likelihood Estimate (MLE) $\hat{p}_{MLE}$ for the parameter $p$.**

**Hint: don't forget about the chain rule: for $h(x) = f(g(x))$, $h'(x) = f'(g(x))g'(x)$.**

**Answer:** $\hat{p} = \frac{n}{\sum_{i=1}^{n} x_i}$

**Explanation:**

**The MLE for the geometric distribution parameter p is the reciprocal of the sample mean.**

**Step-by-step derivation:**

**1. Likelihood function:**

$L_n(p) = \prod_{i=1}^{n} P(X=x_i|p) = \prod_{i=1}^{n} (1-p)^{x_i-1}p$

**2. Log-likelihood function:**

$\log L_n(p) = \sum_{i=1}^{n} \log((1-p)^{x_i-1}p)$

$= \sum_{i=1}^{n} [(x_i-1)\log(1-p) + \log(p)]$

$= \sum_{i=1}^{n} (x_i-1)\log(1-p) + \sum_{i=1}^{n} \log(p)$

$= (\sum_{i=1}^{n} x_i - n)\log(1-p) + n \log(p)$

**3. Take derivative with respect to p:**

$\frac{d}{dp}[\log L_n(p)] = -\frac{\sum_{i=1}^{n} x_i - n}{1-p} + \frac{n}{p}$

**4. Set derivative to zero:**

$0 = -\frac{\sum_{i=1}^{n} x_i - n}{1-p} + \frac{n}{p}$

**5. Solve for p:**

$\frac{n}{p} = \frac{\sum_{i=1}^{n} x_i - n}{1-p}$

$n(1-p) = p(\sum_{i=1}^{n} x_i - n)$

$n - np = p\sum_{i=1}^{n} x_i - np$

$n = p\sum_{i=1}^{n} x_i$

$p = \frac{n}{\sum_{i=1}^{n} x_i}$

**6. Verification:**
- **Second derivative:** $\frac{d^2}{dp^2}[\log L_n(p)] < 0$ for $0 < p < 1$
- **Maximum** confirmed at $p = \frac{n}{\sum_{i=1}^{n} x_i}$

**7. Interpretation:**
- **p = n/Σᵢ₌₁ⁿ xᵢ** is the **reciprocal of the sample mean**
- **Geometric distribution** models number of trials until first success
- **MLE** estimates success probability as reciprocal of average trials needed

**Key insight:** **MLE for geometric distribution** is the **reciprocal of the sample mean**, similar to the exponential distribution.

**4. Select All That Apply**

**Which of the following is true about maximum likelihood estimation, in general?**
*   (a) It always produces unbiased parameter estimates.
*   (b) It can be used for continuous probability distributions.
*   (c) It can be used for discrete probability distributions.
*   (d) It maximizes the likelihood of the data given the model parameters.
*   (e) It maximizes the likelihood of the model parameters given the data.

**Correct answers:** (b), (c), (d)

**Explanation:**

**Options (b), (c), and (d) are true** about maximum likelihood estimation.

**Why (a) is false - MLE is not always unbiased:**

**1. Counterexample - Sample variance:**
- **MLE for variance:** $\sigma^2_{\text{MLE}} = \frac{1}{n}\sum(x_i - \mu)^2$
- **Unbiased estimator:** $\sigma^2_{\text{unbiased}} = \frac{1}{n-1}\sum(x_i - \mu)^2$
- **MLE is biased** - underestimates true variance

**2. Other examples:**
- **MLE for normal mean:** unbiased
- **MLE for exponential rate:** unbiased
- **MLE for variance:** biased (Bessel's correction needed)

**Why (b) and (c) are true - Applicable to all distributions:**

**1. Continuous distributions:**
- **Normal distribution** (linear regression)
- **Exponential distribution**
- **Uniform distribution**
- **Any continuous PDF**

**2. Discrete distributions:**
- **Bernoulli distribution** (logistic regression)
- **Poisson distribution**
- **Geometric distribution**
- **Any discrete PMF**

**Why (d) is true - Correct interpretation:**

**1. MLE objective:**

$\max L(\theta) = \max P(\text{data}|\theta)$

- **$\theta$** = model parameters
- **data** = observed data
- **$L(\theta)$** = likelihood function

**2. Frequentist framework:**
- **Parameters are fixed** (not random)
- **Data is random**
- **We maximize P(data|θ)**, not P(θ|data)

**Why (e) is false - Wrong interpretation:**

**1. P(θ|data) is Bayesian:**
- **Requires prior P(θ)**
- **Posterior P(θ|data) = P(data|θ)P(θ)/P(data)**
- **MLE doesn't use priors**

**2. MLE vs MAP:**
- **MLE:** max P(data|θ) (frequentist)
- **MAP:** max P(θ|data) (Bayesian)

**Key insight:** **MLE** is a **frequentist method** that maximizes **data likelihood** and works for **any distribution type**.

**5. Select All That Apply**

**Suppose $A \in \mathbb{R}^{n \times n}$ is a positive semi-definite (PSD) matrix. Which of the following is always true about $A$?**
*   (a) All eigenvalues of $A$ are non-negative.
*   (b) All elements of $A$ are non-negative.
*   (c) $A$ is invertible.
*   (d) $x^T A x \leq 0$ for all $x$.

**Correct answers:** (a)

**Explanation:**

**Only option (a) is always true** for positive semi-definite matrices.

**Why (a) is true - Non-negative eigenvalues:**

**1. Definition of PSD:**
- **A is PSD** if $x^T A x \geq 0$ for all $x \in \mathbb{R}^n$
- **Equivalent condition:** All eigenvalues of A are $\geq 0$

**2. Spectral theorem:**
- **Symmetric matrices** have real eigenvalues
- **PSD matrices** have non-negative eigenvalues
- **Eigendecomposition:** $A = Q\Lambda Q^T$ where $\Lambda \geq 0$

**3. Mathematical proof:**

For eigenvector $v$ with eigenvalue $\lambda$:

$Av = \lambda v$

$v^T Av = \lambda v^T v = \lambda\|v\|^2 \geq 0$

Since $\|v\|^2 > 0$, we must have $\lambda \geq 0$

**Why other options are false:**

**Option (b): All elements non-negative**
- **PSD** is about quadratic forms, not individual elements
- **Counterexample:** A = [2 -1; -1 2] is PSD but has negative elements
- **Individual elements** can be negative

**Option (c): A is invertible**
- **PSD** allows zero eigenvalues
- **Zero eigenvalue** → singular matrix → not invertible
- **Example:** A = [1 0; 0 0] is PSD but not invertible

**Option (d): x^T A x ≤ 0**
- **Wrong direction** - should be ≥ 0
- **This would be** negative semi-definite
- **PSD** means x^T A x ≥ 0 for all x

**4. Examples of PSD matrices:**
- **X^T X** (Gram matrix)
- **Covariance matrices**
- **Correlation matrices**
- **Identity matrix**

**Key insight:** **PSD matrices** have **non-negative eigenvalues** and satisfy **x^T A x ≥ 0** for all vectors x.

**6.**

**Assume we have $X \in \mathbb{R}^{n \times p}$ representing $n$ data points with $p$ features each and $Y \in \mathbb{R}^n$ representing the corresponding outcomes. Using linear regression with no offset/intercept, provide an expression to predict the outcome for a new data point $x_{\text{new}} \in \mathbb{R}^p$ in terms of $X$ and $Y$.**

**Answer:** $\hat{y}_{\text{new}} = x_{\text{new}}^T (X^T X)^{-1} X^T Y$

**Explanation:**

**This is the prediction formula for linear regression without intercept** using the normal equations solution.

**Step-by-step derivation:**

**1. Linear regression model (no intercept):**

$y = Xw + \varepsilon$

where $w \in \mathbb{R}^p$ is the weight vector

**2. Normal equations solution:**

$\hat{w} = (X^T X)^{-1} X^T Y$

**3. Prediction for new data point:**

$\hat{y}_{\text{new}} = x_{\text{new}}^T \hat{w}$

$= x_{\text{new}}^T (X^T X)^{-1} X^T Y$

**4. Why this works:**

**Training phase:**
- **Minimize:** $\|Y - Xw\|^2$
- **Solution:** $\hat{w} = (X^T X)^{-1} X^T Y$
- **Assumes:** $X^T X$ is invertible (full rank)

**Prediction phase:**
- **New input:** $x_{\text{new}} \in \mathbb{R}^p$
- **Prediction:** $\hat{y}_{\text{new}} = x_{\text{new}}^T \hat{w}$
- **Substitute:** $\hat{w}$ from training

**5. Comparison with intercept model:**
- **With intercept:** $\hat{y}_{\text{new}} = [x_{\text{new}}^T, 1] \times [\hat{w}^T, b]^T$
- **Without intercept:** $\hat{y}_{\text{new}} = x_{\text{new}}^T \hat{w}$
- **No bias term** $b$ in the model

**6. Matrix dimensions:**
- **X:** $n \times p$
- **Y:** $n \times 1$
- **$x_{\text{new}}$:** $p \times 1$
- **$\hat{w}$:** $p \times 1$
- **$\hat{y}_{\text{new}}$:** scalar

**Key insight:** **Prediction** is the **dot product** of the new feature vector with the **learned weight vector**.

**7.**

**Suppose you want to use linear regression to fit a weight vector $w \in \mathbb{R}^d$ and an offset/intercept term $b \in \mathbb{R}$ using data points $x_i \in \mathbb{R}^d$. What is the minimum number of data points $n$ required in your training set such that there will be a single unique solution?**

**Answer:** $n = d+1$

**Explanation:**

**The minimum number of data points needed is n = d+1** to ensure a unique solution for linear regression with intercept.

**Step-by-step reasoning:**

**1. Linear regression model with intercept:**

$y_i = x_i^T w + b + \varepsilon_i$

**2. Augmented data matrix:**

$X_{\text{aug}} = [X, 1] \in \mathbb{R}^{n \times (d+1)}$

where 1 is a column of ones

**3. Normal equations:**

$\hat{w}_{\text{aug}} = (X_{\text{aug}}^T X_{\text{aug}})^{-1} X_{\text{aug}}^T Y$

**4. Rank requirement for unique solution:**
- **$X_{\text{aug}}^T X_{\text{aug}}$** must be **invertible**
- **Invertible** requires **full rank**
- **Full rank** requires **$n \geq d+1$**

**5. Why n = d+1 is minimum:**

**If $n < d+1$:**
- **$X_{\text{aug}}$** has more columns than rows
- **$\text{rank}(X_{\text{aug}}) \leq n < d+1$**
- **$X_{\text{aug}}^T X_{\text{aug}}$** is singular
- **No unique solution**

**If $n = d+1$:**
- **$X_{\text{aug}}$** is square $(d+1) \times (d+1)$
- **Full rank** possible (if data is well-conditioned)
- **Unique solution** exists

**6. Geometric interpretation:**
- **d+1 points** in d-dimensional space
- **Can fit** a unique hyperplane through these points
- **Fewer points** → infinite solutions
- **More points** → overdetermined system

**7. Example:**
- **d = 2 features**
- **n = 3 data points** minimum
- **Fits unique plane** through 3 points in 3D space

**Key insight:** **n = d+1** ensures the **augmented data matrix** has **full rank** for a **unique solution**.

**8. One Answer**

**In a regression model, what is the primary purpose of using general basis functions?**
*   (a) Transform nonlinear relationships between features and the target variable into a linear form.
*   (b) Regularize the model to prevent overfitting.
*   (c) Reduce the number of data samples needed for model training.
*   (d) Simplify the model by reducing the number of features.

**Correct answers:** (a)

**Explanation:**

**Option (a) is correct** - basis functions transform nonlinear relationships into linear form.

**Why (a) is correct - Nonlinear to linear transformation:**

**1. Primary purpose:**
- **Capture nonlinear patterns** in data
- **Transform features** into higher-dimensional space
- **Enable linear modeling** of nonlinear relationships
- **Maintain interpretability** of linear models

**2. Mathematical approach:**

Original: $y = f(x)$ (nonlinear)

Basis expansion: $y = w_1\phi_1(x) + w_2\phi_2(x) + \cdots + w_k\phi_k(x)$

where $\phi_i(x)$ are basis functions

**3. Examples of basis functions:**
- **Polynomial:** φ(x) = [1, x, x², x³, ...]
- **Radial:** φ(x) = exp(-||x-c||²/2σ²)
- **Fourier:** φ(x) = [sin(x), cos(x), sin(2x), ...]
- **Spline:** Piecewise polynomial functions

**4. Why other options are incorrect:**

**Option (b): Regularization**
- **Basis functions** don't regularize by themselves
- **Regularization** is separate (L1/L2 penalties)
- **Basis functions** can actually increase overfitting risk

**Option (c): Reduce data requirements**
- **More complex models** typically need more data
- **Higher-dimensional space** requires more samples
- **Curse of dimensionality** effect

**Option (d): Simplify model**
- **Basis functions** increase feature count
- **More parameters** to estimate
- **Higher complexity**, not lower

**5. Practical benefits:**
- **Flexible modeling** of complex patterns
- **Linear optimization** techniques still applicable
- **Feature engineering** for domain knowledge
- **Kernel methods** foundation

**6. Trade-offs:**
- **Increased complexity** (more features)
- **Higher computational cost**
- **Risk of overfitting**
- **Need for regularization**

**Key insight:** **Basis functions** enable **linear models** to capture **nonlinear patterns** through **feature transformation**.

**9. One Answer**

**In regression, when our prediction model is linear-Gaussian, i.e., $y_i \sim N(x_i^T w, \sigma^2)$ for target output $y_i \in \mathbb{R}$ and feature vectors $x_i \in \mathbb{R}^d$, finding the $w$ that maximizes the data likelihood is equivalent to minimizing the average absolute difference between the target output and predicted output.**
*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:**

**This statement is false** - MLE for linear-Gaussian regression minimizes squared differences, not absolute differences.

**Why this is false:**

**1. Linear-Gaussian model:**

$y_i \sim N(x_i^T w, \sigma^2)$

**2. Likelihood function:**

$L(w) = \prod_i \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - x_i^T w)^2}{2\sigma^2}\right)$

**3. Log-likelihood:**

$\log L(w) = -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_i (y_i - x_i^T w)^2$

**4. MLE objective:**

$\max \log L(w) = \min \sum_i (y_i - x_i^T w)^2$

**5. Comparison of loss functions:**

**Squared error (MLE for Gaussian):**

$L_{\text{squared}} = \sum_i (y_i - x_i^T w)^2$

**Absolute error (MLE for Laplace):**

$L_{\text{absolute}} = \sum_i |y_i - x_i^T w|$

**6. Why Gaussian noise leads to squared error:**
- **Gaussian distribution** has exponential decay with squared distance
- **Log-likelihood** contains squared terms
- **MLE** naturally leads to squared error minimization

**7. Alternative noise distributions:**
- **Laplace noise** → absolute error loss
- **Gaussian noise** → squared error loss
- **Poisson noise** → different loss function

**8. Mathematical intuition:**
- **Squared error** penalizes large errors more heavily
- **Absolute error** penalizes all errors equally
- **Gaussian MLE** prefers squared error due to distribution shape

**Key insight:** **Gaussian noise assumption** leads to **squared error loss**, not **absolute error loss**.

**10. Select All That Apply**

**In ridge regression, we obtain $\hat{w}_{\text{ridge}} = (X^T X + \lambda I)^{-1} X^T y$ for $\lambda \geq 0$. Which of the following is true?**
*   (a) $X^T X$ is always invertible.
*   (b) $X^T X + \lambda I$ is always invertible.
*   (c) Increasing $\lambda$ typically adds bias to the model.
*   (d) Increasing $\lambda$ typically adds variance to the model.
*   (e) When $\lambda = 0$, ridge regression reduces to ordinary (unregularized) linear regression.
*   (f) As $\lambda \rightarrow \infty$, $\hat{w}_{\text{ridge}} \rightarrow 0$.

**Correct answers:** (c), (e), (f)

**Explanation:**

**Options (c), (e), and (f) are true** about ridge regression.

**Why (c) is true - Increasing λ adds bias:**

**1. Ridge regression objective:**

$\min \|y - Xw\|^2 + \lambda\|w\|^2$

**2. Effect of increasing λ:**
- **Stronger penalty** on large weights
- **Weights shrink** toward zero
- **Model becomes less flexible**
- **Higher bias, lower variance**

**3. Mathematical intuition:**
- **Large λ** → strong regularization
- **Constrained parameters** → less capacity to fit data
- **Underfitting risk** → increased bias

**Why (e) is true - λ = 0 reduces to OLS:**

**1. When λ = 0:**
```
\hat{w}_{\text{ridge}} = (X^T X + 0I)^{-1} X^T y
                      = (X^T X)^{-1} X^T y
                      = \hat{w}_{\text{OLS}}
```

**2. No regularization:**
- **$\lambda = 0$** means no penalty term
- **Objective:** $\min \|y - Xw\|^2$
- **Same as** ordinary least squares

**Why (f) is true - λ → ∞ shrinks to zero:**

**1. As $\lambda \to \infty$:**
- **Regularization term** dominates
- **Objective:** approximately $\min \lambda\|w\|^2$
- **Solution:** $w \to 0$

**2. Intuitive explanation:**
- **Infinite penalty** on non-zero weights
- **Optimal solution** is w = 0
- **Simplest possible model**

**Why other options are false:**

**Option (a): X^T X always invertible**
- **X^T X** is positive semi-definite
- **Can be singular** if X has null space
- **Not always invertible**

**Option (b): X^T X + λI always invertible**
- **True when λ > 0**
- **False when λ = 0** (reduces to option a)

**Option (d): Increasing λ adds variance**
- **Opposite effect** - λ reduces variance
- **More regularization** → less sensitivity to data
- **Lower variance, higher bias**

**Key insight:** **Ridge regression** trades **bias for variance** through **L2 regularization**.

**11. One Answer**

**You have a dataset with many features. You know a priori that only a small portion of those features are relevant to your prediction problem, but you don't know which are the relevant features. Is it better to use Ridge regression or Lasso regression?**
*   (a) Ridge regression
*   (b) Lasso regression

**Correct answers:** (b)

**Explanation:**

**Lasso regression is better** when you know only a small portion of features are relevant.

**Why Lasso is preferred:**

**1. Sparsity induction:**
- **L1 regularization** can set coefficients exactly to zero
- **L2 regularization** shrinks coefficients but rarely to exactly zero
- **Feature selection** happens automatically with Lasso

**2. Mathematical difference:**

**Lasso (L1):**

$\min \|y - Xw\|^2 + \lambda\|w\|_1$

- **$\|w\|_1 = \sum|w_i|$** (L1 norm)
- **Sharp corners** at axes
- **Can produce exact zeros**

**Ridge (L2):**

$\min \|y - Xw\|^2 + \lambda\|w\|_2^2$

- **$\|w\|_2^2 = \sum w_i^2$** (L2 norm)
- **Smooth surface**
- **Rarely produces exact zeros**

**3. Geometric interpretation:**

**L1 constraint region:**
- **Diamond-shaped** in 2D
- **Sharp corners** touch the axes
- **Optimal solution** often at corners (zeros)

**L2 constraint region:**
- **Circular** in 2D
- **Smooth surface**
- **Optimal solution** rarely on axes

**4. Feature selection capability:**

**Lasso:**
- **Automatic feature selection**
- **Zero coefficients** = irrelevant features
- **Non-zero coefficients** = relevant features
- **Matches a priori knowledge**

**Ridge:**
- **No feature selection**
- **All features** get non-zero weights
- **Assigns meaning** to irrelevant features

**5. Practical example:**
- **100 features, 10 relevant**
- **Lasso:** May select 10-15 features
- **Ridge:** Uses all 100 features with small weights

**6. When to use each:**
- **Lasso:** When you expect sparsity
- **Ridge:** When all features might be relevant
- **Elastic Net:** When you want both properties

**Key insight:** **Lasso's sparsity** makes it ideal for **feature selection** when you know most features are irrelevant.

**12. One Answer**

**Which of the following best explains the effect of Lasso regression on the bias-variance tradeoff?**
*   (a) Lasso regression reduces both bias and variance simultaneously, leading to a more accurate model.
*   (b) Lasso regression reduces bias by shrinking coefficients, often at the expense of increasing variance.
*   (c) Lasso regression reduces variance by shrinking coefficients and can increase bias, especially when some features are dropped entirely from the learned predictor.
*   (d) Lasso regression increases both bias and variance as it enforces sparsity in the learned predictor.

**Correct answers:** (c)

**Explanation:**

**Option (c) is correct** - Lasso reduces variance and can increase bias through coefficient shrinkage and feature selection.

**Why (c) is correct:**

**1. Lasso's effect on model complexity:**
- **Shrinks coefficients** toward zero
- **Sets some coefficients** exactly to zero
- **Reduces effective** number of features
- **Simplifies the model**

**2. Bias-variance tradeoff:**
- **Reduced complexity** → **lower variance**
- **Reduced complexity** → **higher bias**
- **Feature selection** can increase bias if important features are dropped

**3. Mathematical intuition:**

**Variance reduction:**
- **Fewer parameters** to estimate
- **Less sensitive** to training data noise
- **More stable** predictions across datasets
- **Lower overfitting** risk

**Bias increase:**
- **Simpler model** may miss true patterns
- **Dropped features** could be important
- **Underfitting** risk if regularization is too strong

**4. Why other options are incorrect:**

**Option (a): Reduces both bias and variance**
- **Impossible** - bias and variance typically trade off
- **Cannot reduce** both simultaneously
- **Violates** fundamental tradeoff principle

**Option (b): Reduces bias, increases variance**
- **Opposite effect** - Lasso reduces variance
- **Simpler models** have lower variance
- **Wrong direction** of tradeoff

**Option (d): Increases both bias and variance**
- **Incorrect** - Lasso reduces variance
- **Simpler models** are more stable
- **Lower variance** is a key benefit

**5. Practical implications:**
- **Strong L1 penalty** → high bias, low variance
- **Weak L1 penalty** → low bias, high variance
- **Optimal λ** balances the tradeoff

**6. Comparison with Ridge:**
- **Both** reduce variance and increase bias
- **Lasso** can set coefficients to zero
- **Ridge** only shrinks coefficients

**Key insight:** **Lasso** implements the **bias-variance tradeoff** through **L1 regularization** and **feature selection**.

**13. One Answer**

**In prediction, the total expected prediction error can be decomposed into three components: bias squared, variance, and irreducible error. By optimizing the model complexity and increasing the size of the dataset, it is possible to reduce all three components.**
*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:**

**This statement is false** - irreducible error cannot be reduced by any modeling technique.

**Why this is false:**

**1. Error decomposition:**

$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$

**2. What can be reduced:**

**Bias²:**
- **Model complexity** optimization
- **Feature engineering**
- **Algorithm selection**
- **Hyperparameter tuning**

**Variance:**
- **More training data**
- **Regularization**
- **Ensemble methods**
- **Cross-validation**

**3. What cannot be reduced:**

**Irreducible Error:**
- **Inherent noise** in the data
- **Measurement uncertainty**
- **Missing information**
- **Fundamental randomness**

**4. Why irreducible error is irreducible:**

**Definition:**
- **Fundamental uncertainty** in the data generation process
- **Lower bound** on model performance
- **Independent** of model choice or data size
- **Cannot be eliminated** by any algorithm

**5. Examples of irreducible error:**
- **Sensor noise** in measurements
- **Natural variability** in biological systems
- **Unpredictable external factors**
- **Quantum uncertainty** in physical systems

**6. What optimization can do:**
- **Reduce bias** through better model selection
- **Reduce variance** through more data/regularization
- **Cannot touch** irreducible error

**7. Practical implications:**
- **Perfect models** still have irreducible error
- **Performance limits** exist regardless of data size
- **Realistic expectations** about model accuracy

**Key insight:** **Irreducible error** represents the **fundamental limit** on prediction accuracy that **cannot be overcome**.

**14. One Answer**

**Which strategy is most effective for reducing variance in a high-variance, low-bias model?**
*   (a) Increasing the number of training examples.
*   (b) Increasing the model complexity.
*   (c) Decreasing regularization.
*   (d) Removing the features that exhibit high variance across training examples.

**Correct answers:** (a)

**Explanation:**

**Option (a) is correct** - increasing the number of training examples is most effective for reducing variance.

**Why (a) is correct:**

**1. High-variance, low-bias model characteristics:**
- **Complex model** that fits training data well
- **Sensitive** to training data changes
- **Overfitting** to training data
- **Poor generalization** to unseen data

**2. How more training data reduces variance:**
- **Larger sample size** reduces parameter estimation uncertainty
- **More stable** parameter estimates
- **Better generalization** to unseen data
- **Natural regularization** effect

**3. Mathematical intuition:**
- **Variance** $\propto \sigma^2/n$ (for many estimators)
- **Larger $n$** → smaller variance
- **More data** → more stable estimates

**4. Why other options are incorrect:**

**Option (b): Increasing model complexity**
- **Increases variance** by adding more parameters
- **More complex models** are more sensitive to data
- **Opposite effect** of what we want

**Option (c): Decreasing regularization**
- **Increases variance** by allowing more overfitting
- **Less constraint** on model parameters
- **Higher sensitivity** to training data

**Option (d): Removing high-variance features**
- **Feature variance** ≠ model variance
- **Removing features** could increase or decrease model variance
- **No a priori guarantee** of improvement

**5. Alternative variance reduction strategies:**
- **Regularization** (L1/L2 penalties)
- **Ensemble methods** (bagging, random forests)
- **Cross-validation** for model selection
- **Feature selection** (when appropriate)

**6. Practical considerations:**
- **Data collection** can be expensive
- **Quality** of additional data matters
- **Diminishing returns** as n increases
- **Balance** with computational cost

**7. When to use each strategy:**
- **More data:** When available and affordable
- **Regularization:** When data is limited
- **Ensemble methods:** When individual models are unstable
- **Feature selection:** When many irrelevant features exist

**Key insight:** **More training data** is the **most effective** way to reduce variance in high-variance models.

**15. One Answer**

**If your model has high validation loss and high training loss, which action is most appropriate to improve the model?**
*   (a) Increase the model complexity.
*   (b) Increase $k$ in $k$-fold cross-validation.
*   (c) Increase the number of training examples.
*   (d) Apply regularization to reduce overfitting.

**Correct answers:** (a)

**Explanation:**

**Option (a) is correct** - increasing model complexity is appropriate when both training and validation loss are high.

**Why (a) is correct:**

**1. High training and validation loss indicates underfitting:**
- **Model is too simple** to capture data patterns
- **High bias, low variance** situation
- **Poor performance** on both training and test data
- **Model capacity** is insufficient

**2. Underfitting characteristics:**
- **Training loss:** High (model can't fit training data)
- **Validation loss:** High (model can't generalize)
- **Gap between losses:** Small (low variance)
- **Model complexity:** Too low

**3. Why increasing complexity helps:**
- **More parameters** to capture patterns
- **Higher model capacity** to fit data
- **Reduced bias** through better representation
- **Better performance** on both training and validation

**4. Why other options are incorrect:**

**Option (b): Increase k in k-fold CV**
- **Cross-validation** is for model evaluation, not improvement
- **Larger k** doesn't change model performance
- **Doesn't address** the fundamental underfitting issue

**Option (c): Increase training examples**
- **More data** helps with variance, not bias
- **Underfitting** persists regardless of data size
- **Model capacity** is the limiting factor

**Option (d): Apply regularization**
- **Regularization** reduces complexity
- **Underfitting** means model is already too simple
- **Would make the problem worse**

**5. Specific ways to increase complexity:**
- **Add more features** or basis functions
- **Increase polynomial degree** in polynomial regression
- **Add more layers** in neural networks
- **Use more complex** algorithms

**6. When to use other strategies:**
- **More data:** When model is overfitting (high variance)
- **Regularization:** When model is overfitting (high variance)
- **Feature engineering:** When current features are insufficient
- **Different algorithm:** When current algorithm is inappropriate

**7. Monitoring improvement:**
- **Training loss** should decrease
- **Validation loss** should decrease
- **Gap** between losses may increase (higher variance)
- **Overall performance** should improve

**Key insight:** **High training and validation loss** indicates **underfitting**, which requires **increasing model complexity**.

**16.**

**In a project using a customer purchase history dataset with a 60/20/20 train, validation, and test split, the validation accuracy remains consistently lower than the training accuracy. What could be a reason for this?**

**Answer:** Overfitting - the model is learning training data patterns that don't generalize to validation data.

**Explanation:**

**The most likely reason is overfitting** - the model is learning patterns specific to the training data that don't generalize.

**Why overfitting occurs:**

**1. Model complexity vs. data size:**
- **Complex model** relative to available training data
- **Too many parameters** to estimate reliably
- **Model memorizes** training data instead of learning generalizable patterns

**2. Training vs. validation performance gap:**
- **Training accuracy:** High (model fits training data well)
- **Validation accuracy:** Lower (model doesn't generalize)
- **Gap indicates** overfitting

**3. Specific causes in this scenario:**

**Data characteristics:**
- **Customer purchase patterns** may be complex
- **Seasonal effects** or temporal dependencies
- **Individual customer idiosyncrasies** in training data

**Model issues:**
- **Too many features** relative to data size
- **Insufficient regularization**
- **Complex algorithm** for the problem

**4. Other possible reasons:**

**Data leakage:**
- **Validation set** may have different characteristics
- **Temporal split** issues (validation data from different time period)
- **Sampling bias** between train and validation sets

**Feature distribution shift:**
- **Different customer segments** in validation set
- **Different time periods** with different purchase patterns
- **Different data collection** methods

**5. Solutions to address overfitting:**

**Reduce model complexity:**
- **Feature selection** to remove irrelevant features
- **Regularization** (L1/L2 penalties)
- **Simpler algorithms** or fewer parameters

**Increase effective data size:**
- **Data augmentation** techniques
- **Cross-validation** for better model selection
- **Ensemble methods** to reduce variance

**6. Monitoring and diagnosis:**
- **Track training vs. validation** performance over time
- **Use learning curves** to identify overfitting
- **Cross-validation** to get more reliable estimates

**Key insight:** **Consistent gap** between training and validation performance typically indicates **overfitting** due to **model complexity** exceeding **data capacity**.

**17. One Answer**

**A consortium of 10 hospitals have pooled together their Electronic Health Records data and want to build a machine learning model to predict patient prognosis based on patient records in their hospitals. They want to maximize the accuracy of their model across all 10 hospitals and do not plan to deploy their model in other hospitals. How should they split the data into train / validation / test sets?**
*   (a) Leave out data from 1 hospital for the validation set, data from another hospital for the test set, and use the rest for train set.
*   (b) Leave out data from 1 hospital for the validation set, data from another hospital for the test set, and use the rest for train set. After training, add the validation data to the train set and re-train the model on the combined data.
*   (c) Use data from 8 hospitals with the most number of records for training, and use data from the other 2 hospitals for validation and test sets.
*   (d) Mix data from all hospitals, randomly shuffle all the records, and then do the 80/10/10 train/validation/test split.

**Correct answers:** (d)

**Explanation:**

**Option (d) is correct** - mixing data from all hospitals and random splitting avoids overfitting to specific hospital characteristics.

**Why (d) is correct:**

**1. Problem with hospital-specific splits:**
- **Each hospital** has unique characteristics
- **Different patient populations** and demographics
- **Different medical practices** and protocols
- **Different data collection** methods and quality

**2. Why hospital-specific splits cause issues:**

**Options (a) and (b):**
- **Validation/test sets** from single hospitals
- **Hyperparameter tuning** overfits to one hospital's characteristics
- **Model selection** biased toward that hospital's patterns
- **Poor generalization** to other hospitals

**Option (c):**
- **Validation/test** from hospitals with fewer records
- **Small validation set** leads to unreliable hyperparameter selection
- **Test set** may not represent overall hospital population
- **Statistical power** issues with small samples

**3. Benefits of mixed data splitting:**

**Representative samples:**
- **All hospitals** represented in each split
- **Balanced distribution** of hospital characteristics
- **More reliable** performance estimates
- **Better generalization** across hospitals

**Statistical advantages:**
- **Larger validation/test sets** for reliable estimates
- **IID assumption** more likely to hold
- **Reduced variance** in performance estimates
- **More robust** model selection

**4. Practical considerations:**

**Data heterogeneity:**
- **Different hospitals** may have different patient mixes
- **Varying data quality** across institutions
- **Different coding** practices and standards
- **Temporal differences** in data collection

**Model deployment:**
- **Model will be used** across all 10 hospitals
- **Performance should be** representative of all hospitals
- **Mixed splitting** ensures this representation

**5. Alternative approaches (if needed):**
- **Stratified sampling** to ensure hospital representation
- **Cross-validation** with hospital-aware folds
- **Ensemble methods** trained on different hospital subsets

**6. When hospital-specific splits might be appropriate:**
- **Testing generalization** to completely new hospitals
- **Domain adaptation** scenarios
- **Transfer learning** applications

**Key insight:** **Mixed data splitting** ensures **representative performance estimates** and **avoids overfitting** to specific hospital characteristics.

**18.**

**Given the task of determining loan approval for applicants using a predictive model given applicant features such as race, salary, education, etc., is it always best practice to allow the model to use all of the given features? Why or why not?**

**Answer:** No, we should not always use all features. Ethical considerations and potential bias require careful feature selection.

**Explanation:**

**No, we should not always use all features** - ethical considerations and potential bias require thoughtful feature selection.

**Why not to use all features:**

**1. Ethical and legal concerns:**

**Protected attributes:**
- **Race, gender, age** may be legally protected
- **Direct discrimination** is illegal in many contexts
- **Model may learn** discriminatory patterns
- **Fair lending laws** prohibit certain features

**2. Bias and discrimination:**

**Historical bias:**
- **Training data** may reflect historical discrimination
- **Model learns** biased patterns from biased data
- **Perpetuates** existing inequalities
- **Proxy discrimination** through correlated features

**3. Correlation vs. causation:**

**Spurious correlations:**
- **Race and income** may be correlated due to historical factors
- **Model may use race** as a proxy for income
- **Correlation doesn't imply** causation
- **Reinforces** existing biases

**4. Practical considerations:**

**Model interpretability:**
- **Fewer features** often more interpretable
- **Regulatory compliance** requires transparency
- **Stakeholder trust** depends on understanding
- **Debugging** easier with simpler models

**5. Alternative approaches:**

**Feature engineering:**
- **Remove protected attributes** entirely
- **Create fair features** that don't encode bias
- **Use domain knowledge** to select relevant features
- **Feature selection** based on business logic

**Fairness techniques:**
- **Preprocessing** to remove bias
- **In-processing** fairness constraints
- **Post-processing** to ensure fairness
- **Regularization** against unfair predictions

**6. When to include features:**

**Legitimate business need:**
- **Features directly related** to creditworthiness
- **Income, employment history, credit score**
- **Features that predict** ability to repay
- **Features that don't encode** protected attributes

**7. Best practices:**

**Documentation:**
- **Justify** each feature inclusion
- **Document** potential bias sources
- **Monitor** model fairness over time
- **Regular audits** of model decisions

**Testing:**
- **Fairness metrics** (disparate impact, equalized odds)
- **Bias detection** in predictions
- **A/B testing** with different feature sets
- **Stakeholder review** of feature choices

**Key insight:** **Feature selection** in sensitive applications requires **ethical consideration** beyond just **predictive accuracy**.

**19. One Answer**

**You are building a predictive model about users of a website. Suppose that after you train your model on historical user data, the distribution of users shifts dramatically. What can happen if you deploy your machine learning system without addressing this distribution shift?**
*   (a) The model will automatically adapt to new data distributions.
*   (b) The model will generate more diverse predictions, increasing its overall accuracy.
*   (c) The model will maintain its original performance indefinitely regardless of data changes.
*   (d) The model's predictions may become unreliable or biased.

**Correct answers:** (d)

**Explanation:**

**Option (d) is correct** - distribution shift can make model predictions unreliable or biased.

**Why (d) is correct:**

**1. Distribution shift problem:**
- **Training data** comes from one distribution
- **Deployment data** comes from different distribution
- **Model assumptions** no longer hold
- **Performance degradation** inevitable

**2. What happens during distribution shift:**

**Covariate shift:**
- **Feature distributions** change (e.g., user demographics)
- **Label distribution** remains same
- **Model trained** on old user types
- **Poor performance** on new user types

**Label shift:**
- **Target variable distribution** changes
- **Feature distributions** may remain same
- **Model calibrated** for old label distribution
- **Biased predictions** on new data

**3. Why other options are incorrect:**

**Option (a): Automatic adaptation**
- **Models don't adapt** automatically
- **Static parameters** learned during training
- **No online learning** mechanism
- **Requires retraining** to adapt

**Option (b): More diverse predictions**
- **Diversity doesn't imply** accuracy
- **Random predictions** are diverse but useless
- **Distribution shift** typically reduces accuracy
- **Wrong direction** of effect

**Option (c): Maintain performance indefinitely**
- **Models are not** distribution-agnostic
- **Performance depends** on data distribution
- **Assumes** unrealistic model robustness
- **Contradicts** fundamental ML principles

**4. Specific consequences:**

**Performance degradation:**
- **Lower accuracy** on new data
- **Increased error rates**
- **Poor generalization** to new users
- **Unreliable predictions**

**Bias introduction:**
- **Systematic errors** in predictions
- **Unfair treatment** of new user groups
- **Discriminatory outcomes**
- **Loss of trust** in system

**5. Examples of distribution shift:**

**Website user changes:**
- **New user demographics** (age, location)
- **Different usage patterns** (mobile vs. desktop)
- **Seasonal effects** (holiday shopping)
- **Platform changes** (new features, redesign)

**6. Solutions to address distribution shift:**

**Monitoring:**
- **Track performance** over time
- **Detect distribution** changes
- **Alert systems** for performance drops
- **Regular model** evaluation

**Adaptation strategies:**
- **Online learning** for continuous adaptation
- **Transfer learning** to adapt to new domains
- **Domain adaptation** techniques
- **Regular retraining** with new data

**7. Prevention measures:**
- **Robust feature engineering**
- **Domain-invariant** representations
- **Ensemble methods** for stability
- **Conservative model** selection

**Key insight:** **Distribution shift** is a **fundamental challenge** in ML that requires **proactive monitoring** and **adaptation strategies**.

**20. One Answer**

**For a possibly non-convex optimization problem, gradient descent on the full dataset always finds a better solution than stochastic gradient descent.**
*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:**

**This statement is false** - SGD can sometimes find better solutions than full gradient descent for non-convex problems.

**Why this is false:**

**1. Non-convex optimization challenges:**
- **Multiple local minima** exist
- **Global minimum** is hard to find
- **Optimization landscape** is complex
- **Convergence** to local optima is common

**2. How SGD can outperform GD:**

**Escape local minima:**
- **Stochastic noise** helps escape local minima
- **Random perturbations** explore more of the landscape
- **Less likely** to get stuck in poor local optima
- **Better exploration** of solution space

**3. Why GD can get stuck:**

**Deterministic nature:**
- **Always follows** exact gradient direction
- **Can get trapped** in local minima
- **No randomness** to escape poor solutions
- **Sensitive** to initialization

**4. Mathematical intuition:**

**GD update rule:**
```
w_{t+1} = w_t - \eta \nabla L(w_t)
```
- **Exact gradient** from all data
- **Deterministic** path
- **Converges** to nearest local minimum

**SGD update rule:**
```
w_{t+1} = w_t - \eta \nabla L_i(w_t)
```
- **Noisy gradient** from mini-batch
- **Stochastic** path
- **Can escape** local minima

**5. Practical examples:**

**Neural networks:**
- **Highly non-convex** optimization
- **SGD often finds** better solutions
- **GD may converge** to poor local minima
- **SGD's noise** is beneficial

**6. Trade-offs:**

**SGD advantages:**
- **Better exploration** of solution space
- **Escape local minima**
- **Computational efficiency**
- **Scalability** to large datasets

**SGD disadvantages:**
- **Noisy convergence**
- **Requires careful** learning rate tuning
- **May not converge** to exact minimum
- **Higher variance** in final solution

**7. When each is preferred:**

**Use GD when:**
- **Convex optimization** problems
- **Small datasets** that fit in memory
- **Need exact** convergence
- **Computational cost** is not a concern

**Use SGD when:**
- **Non-convex problems** (like neural networks)
- **Large datasets** that don't fit in memory
- **Want to escape** local minima
- **Computational efficiency** is important

**Key insight:** **SGD's stochasticity** can be **advantageous** for **non-convex optimization** by helping escape **local minima**.

**21. Select All That Apply**

**Given the gradient descent algorithm, $w_{t+1} = w_t - \eta \frac{df(w)}{dw} \Big|_{w=w_t}$, which of the following statement is correct regarding the hyperparameter $\eta$?**
*   (a) $\eta$ controls the magnitude of each step.
*   (b) $\eta$ determines the initial value of $w$.
*   (c) A larger $\eta$ guarantees faster convergence to the global minimum.
*   (d) A smaller $\eta$ guarantees faster convergence to the global minimum.

**Correct answers:** (a)

**Explanation:**

**Only option (a) is correct** - η controls the step size in gradient descent.

**Why (a) is correct:**

**1. Learning rate role:**
- **η** is the **learning rate** or **step size**
- **Controls how far** to move in gradient direction
- **Scales the gradient** vector
- **Determines convergence** behavior

**2. Mathematical interpretation:**
```
w_{t+1} = w_t - η∇f(w_t)
```
- **η** multiplies the gradient
- **Larger η** → larger steps
- **Smaller η** → smaller steps
- **Step magnitude** = η × ||∇f(w_t)||

**3. Why other options are incorrect:**

**Option (b): η determines initial value**
- **η and w₀** are set independently
- **Initial value w₀** is chosen separately
- **η only affects** step size, not starting point
- **No relationship** between η and w₀

**Option (c): Larger η guarantees faster convergence**
- **Too large η** can cause overshooting
- **May oscillate** around minimum
- **May diverge** if η is too large
- **No guarantee** of faster convergence

**Option (d): Smaller η guarantees faster convergence**
- **Too small η** leads to slow convergence
- **May get stuck** in local minima
- **Many iterations** needed to converge
- **No guarantee** of faster convergence

**4. Learning rate effects:**

**Too large η:**
- **Overshooting** the minimum
- **Oscillations** around optimal point
- **Divergence** in extreme cases
- **Unstable** convergence

**Too small η:**
- **Slow convergence** to minimum
- **Many iterations** required
- **May get stuck** in local minima
- **Computationally expensive**

**Optimal η:**
- **Fast convergence** without overshooting
- **Stable** optimization path
- **Reaches minimum** efficiently
- **Problem-dependent** choice

**5. Practical considerations:**

**Adaptive learning rates:**
- **Adam, RMSprop** adjust η automatically
- **Learning rate scheduling** reduces η over time
- **Line search** methods find optimal η
- **Grid search** for hyperparameter tuning

**6. Guidelines for choosing η:**
- **Start small** (e.g., 0.01, 0.001)
- **Monitor convergence** behavior
- **Adjust based on** problem characteristics
- **Use validation** to tune η

**Key insight:** **Learning rate η** controls **step size** and **convergence behavior**, but **optimal value** depends on the **specific problem**.

**22. Select All That Apply**

**Which of the following functions are convex?**
*   (a) $f(x) = x^3$
*   (b) $f(x) = \frac{3x(x-1)}{2}$
*   (c) $f(x) = \sin x$, for $x \in [\pi, 2\pi]$
*   (d) $f(x) = \log_{10}(x)$

**Correct answers:** (b), (c)

**Explanation:**

**Options (b) and (c) are convex** on their specified domains.

**Analysis of each function:**

**Option (a): f(x) = x³**
- **f'(x) = 3x²**
- **f''(x) = 6x**
- **f''(x) < 0** for x < 0 (concave)
- **f''(x) > 0** for x > 0 (convex)
- **Not convex** on $\mathbb{R}$ (mixed convexity)

**Option (b): f(x) = 3x(x-1)/2 = (3x² - 3x)/2**
- **f'(x) = (6x - 3)/2 = 3x - 3/2**
- **f''(x) = 3 > 0** for all x
- **Convex** on $\mathbb{R}$ (positive second derivative)

**Option (c): f(x) = sin(x) on [π, 2π]**
- **f'(x) = cos(x)**
- **f''(x) = -sin(x)**
- **On [π, 2π]:** sin(x) ≤ 0
- **Therefore:** f''(x) = -sin(x) ≥ 0
- **Convex** on [π, 2π]

**Option (d): f(x) = log₁₀(x)**
- **f'(x) = 1/(x ln(10))**
- **f''(x) = -1/(x² ln(10)) < 0** for x > 0
- **Concave** on (0, ∞)
- **Not convex**

**3. Visual interpretation:**

**Option (b):**
- **Quadratic function** with positive coefficient
- **U-shaped curve** opening upward
- **Always convex**

**Option (c):**
- **Sine function** on [π, 2π]
- **Curve bends upward** in this interval
- **Convex** in this domain

**4. Why convexity matters:**

**Optimization properties:**
- **Local minima** are global minima
- **Gradient descent** converges to global minimum
- **No local minima traps**
- **Efficient optimization** algorithms

**5. Second derivative test:**
- **f''(x) > 0** → convex
- **f''(x) < 0** → concave
- **f''(x) = 0** → inflection point

**6. Practical implications:**
- **Convex functions** are easier to optimize
- **Guaranteed convergence** to global minimum
- **No initialization** sensitivity
- **Deterministic** optimization path

**Key insight:** **Convexity** is determined by **second derivative sign** and provides **optimization guarantees**.

**23. Which of the following are true about a convex function $f(x): \mathbb{R}^d \rightarrow \mathbb{R}$?**
*   (a) $f(x)$ must be differentiable across its entire domain.
*   (b) $f(x)$ has a unique global minimum.
*   (c) $g(x) = -f(x)$ is also convex.
*   (d) If $f(x)$ is twice differentiable, then $z^T \nabla^2 f(x)z \geq 0$ for all $z \in \mathbb{R}^d$.

**Correct answers:** (d)

**Explanation:**

**Only option (d) is correct** - the Hessian matrix must be positive semi-definite for twice differentiable convex functions.

**Why (d) is correct:**

**1. Second-order condition for convexity:**
- **f is convex** if and only if ∇²f(x) is positive semi-definite
- **z^T ∇²f(x) z ≥ 0** for all z ∈ ℝ^d
- **All eigenvalues** of ∇²f(x) are non-negative
- **This is the** fundamental characterization

**2. Why other options are incorrect:**

**Option (a): Must be differentiable**
- **Counterexample:** f(x) = |x| is convex but not differentiable at x = 0
- **Convex functions** can have non-differentiable points
- **Subdifferential** exists at all points
- **Differentiability** is not required

**Option (b): Unique global minimum**
- **Counterexample:** f(x) = 0 (constant function) has infinitely many global minima
- **Counterexample:** f(x) = x² on [-1,1] has minimum at x = 0, but f(x) = 0 on [0,1] has multiple minima
- **Convex functions** can have multiple connected global minima
- **Uniqueness** requires strict convexity

**Option (c): Negative is also convex**
- **Counterexample:** f(x) = x² is convex, but g(x) = -x² is concave
- **Only true** for linear/affine functions
- **Convexity** is not preserved under negation
- **Negation** flips convexity to concavity

**3. Mathematical details:**

**Hessian condition:**
```
For twice differentiable f: ℝ^d → ℝ
f is convex ⇔ ∇²f(x) is positive semi-definite ∀x
```

**Positive semi-definite matrix:**
- **z^T A z ≥ 0** for all z ∈ ℝ^d
- **All eigenvalues** ≥ 0
- **Symmetric** matrix

**4. Examples:**

**Convex functions:**
- **f(x) = x²** → f''(x) = 2 > 0
- **f(x) = e^x** → f''(x) = e^x > 0
- **f(x) = ||x||²** → ∇²f(x) = 2I (positive definite)

**Non-differentiable convex:**
- **f(x) = |x|** (convex but not differentiable at 0)

**5. Practical implications:**

**Optimization:**
- **Local minima** are global minima
- **Gradient descent** converges to global minimum
- **No local minima traps**
- **Efficient algorithms** available

**Key insight:** **Convexity** is characterized by **positive semi-definite Hessian** for twice differentiable functions, but **differentiability** is not required.

**24. Which of the following have convex objective functions?**
*   (a) Linear regression
*   (b) Linear regression with arbitrary nonlinear basis functions
*   (c) Ridge regression
*   (d) Lasso regression
*   (e) Logistic regression

**Correct answers:** (a), (b), (c), (d), (e)

**Explanation:**

**All options have convex objective functions** - this is a key property of these models.

**Why all are convex:**

**1. Linear regression:**
```
\text{Objective: } \min \|y - Xw\|^2
```
- **Quadratic function** in w
- **Positive semi-definite** Hessian (2X^T X)
- **Convex** for any X

**2. Linear regression with basis functions:**
```
\text{Objective: } \min \|y - \Phi w\|^2
```
where $\Phi = \phi(X)$ are basis functions
- **Still quadratic** in w
- **Same convexity** properties as linear regression
- **Basis transformation** preserves convexity

**3. Ridge regression:**
```
\text{Objective: } \min \|y - Xw\|^2 + \lambda\|w\|^2
```
- **Sum of convex functions** is convex
- **$\|y - Xw\|^2$** is convex (quadratic)
- **$\lambda\|w\|^2$** is convex (quadratic)
- **Convex combination** remains convex

**4. Lasso regression:**
```
\text{Objective: } \min \|y - Xw\|^2 + \lambda\|w\|_1
```
- **$\|y - Xw\|^2$** is convex (quadratic)
- **$\|w\|_1$** is convex (L1 norm)
- **Sum of convex functions** is convex
- **L1 penalty** preserves convexity

**5. Logistic regression:**
```
\text{Objective: } \min \sum[-y_i \log(\sigma(x_i^T w)) - (1-y_i)\log(1-\sigma(x_i^T w))]
```
- **Log-likelihood** of Bernoulli distribution
- **Sigmoid function** $\sigma(z) = 1/(1+e^{-z})$
- **Concave log-likelihood** → convex negative log-likelihood
- **Well-known** to be convex

**6. Mathematical properties:**

**Convexity preservation:**
- **Linear combinations** of convex functions are convex
- **Composition** with linear functions preserves convexity
- **Sum** of convex functions is convex
- **Regularization** terms are typically convex

**7. Practical implications:**

**Optimization guarantees:**
- **Global minimum** exists and is unique (if strictly convex)
- **Gradient descent** converges to global minimum
- **No local minima** traps
- **Efficient algorithms** available

**8. Why convexity matters:**

**Reliable optimization:**
- **Predictable** convergence behavior
- **No initialization** sensitivity
- **Deterministic** optimization path
- **Well-understood** algorithms

**Key insight:** **All these models** have **convex objectives** because they use **linear functions** with **convex loss functions** and **convex regularization**.

**25. Which of the following scenarios are better suited for a logistic regression model over a linear regression model?**
*   (a) Forecasting the price of stocks for the next year, given the price of stocks for the past year.
*   (b) Diagnosing the presence or absence of a rare disease, given a medical x-ray.
*   (c) Predicting what the average global temperature will be in the next year, given historical climate data.
*   (d) Predicting how likely a student is to successfully complete a 4-year college degree, given their high school grades.
*   (e) Predicting the hardness of a material on a scale of 1-10 given the molecular structure of the material.

**Correct answers:** (b), (d)

**Explanation:**

**Options (b) and (d) are classification problems** that are better suited for logistic regression.

**Why (b) and (d) are classification problems:**

**Option (b): Disease diagnosis**
- **Binary outcome:** Disease present (1) or absent (0)
- **Classification task:** Predict discrete categories
- **Logistic regression** outputs probabilities P(disease = 1)
- **Medical diagnosis** is inherently categorical

**Option (d): College completion prediction**
- **Binary outcome:** Complete degree (1) or not (0)
- **Classification task:** Predict success/failure
- **Logistic regression** outputs P(completion = 1)
- **Educational outcomes** are often binary

**Why other options are regression problems:**

**Option (a): Stock price forecasting**
- **Continuous outcome:** Price can be any real number
- **Regression task:** Predict continuous values
- **Linear regression** appropriate for continuous targets
- **Time series** prediction problem

**Option (c): Temperature prediction**
- **Continuous outcome:** Temperature can be any real number
- **Regression task:** Predict continuous values
- **Linear regression** appropriate for continuous targets
- **Climate modeling** problem

**Option (e): Material hardness prediction**
- **Continuous outcome:** Hardness scale 1-10 (continuous)
- **Regression task:** Predict continuous values
- **Linear regression** appropriate for continuous targets
- **Material science** prediction problem

**3. Key differences:**

**Logistic regression:**
- **Binary classification** (0 or 1)
- **Outputs probabilities** P(y = 1)
- **Sigmoid activation** function
- **Log-likelihood** loss function

**Linear regression:**
- **Continuous prediction** (any real number)
- **Outputs continuous** values
- **Linear activation** function
- **Squared error** loss function

**4. When to use each:**

**Use logistic regression for:**
- **Binary classification** problems
- **Probability estimation** tasks
- **Categorical outcomes**
- **Risk assessment** problems

**Use linear regression for:**
- **Continuous prediction** problems
- **Quantity estimation** tasks
- **Numerical outcomes**
- **Forecasting** problems

**5. Model outputs:**

**Logistic regression:**
- **P(y = 1)** ∈ [0, 1]
- **Interpretable** as probability
- **Decision threshold** needed for classification
- **ROC curves** and AUC metrics

**Linear regression:**
- **ŷ** ∈ ℝ (any real number)
- **Direct prediction** of target
- **No threshold** needed
- **R²** and RMSE metrics

**Key insight:** **Logistic regression** is for **binary classification**, while **linear regression** is for **continuous prediction**.

**26. Which of the following statements about classification are true?**

**Recall that the softmax function $\sigma: \mathbb{R}^k \rightarrow (0,1)^k$ takes a vector $z \in \mathbb{R}^k$ and returns a vector $\sigma(z) \in (0,1)^k$ with**
$$\sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)}$$

*   (a) Consider a binary classification setting. If the data is linearly separable, we can use a logistic regression model with quadratic features to avoid overfitting.
*   (b) Because binary logistic regression is a convex optimization problem, it has a closed-form solution.
*   (c) The softmax function is used when we are classifying $k > 2$ classes. When we are classifying only $k = 2$ classes, softmax regression will overfit, so we use binary logistic regression instead.
*   (d) We can use linear regression to solve classification problems, though the model we learn might not be as accurate compared to using logistic/softmax regression.

**Correct answers:** (d)

**Explanation:**

**Only option (d) is correct** - linear regression can be used for classification, though it's not optimal.

**Why (d) is correct:**

**1. Linear regression for classification:**
- **Can predict** binary outcomes (0 or 1)
- **Outputs continuous** values that can be thresholded
- **Threshold** at 0.5 for binary classification
- **Works** but not optimal for classification

**2. Why linear regression works:**
- **Can learn** linear decision boundaries
- **Outputs** can be interpreted as scores
- **Thresholding** converts to binary predictions
- **Simple** and interpretable

**3. Why it's not optimal:**
- **No probability** interpretation
- **Predictions** outside [0,1] range
- **Squared error** not appropriate for classification
- **Poor calibration** for probabilities

**Why other options are incorrect:**

**Option (a): Quadratic features avoid overfitting**
- **Quadratic features** increase model complexity
- **More parameters** to estimate
- **Higher risk** of overfitting
- **Opposite effect** of what's claimed

**Option (b): Closed-form solution for logistic regression**
- **Logistic regression** is convex but has no closed-form solution
- **Requires iterative** optimization (gradient descent, Newton's method)
- **Only linear regression** has closed-form solution among these models
- **Nonlinear** objective function

**Option (c): Softmax overfits for k=2**
- **Softmax** works fine for k=2 (equivalent to logistic regression)
- **No overfitting** issue specific to k=2
- **Logistic regression** is special case of softmax
- **Choice between** them is implementation preference

**4. Comparison of approaches:**

**Linear regression for classification:**
- **Pros:** Simple, interpretable, convex
- **Cons:** No probability output, poor calibration
- **Use when:** Quick prototype, interpretability important

**Logistic regression for classification:**
- **Pros:** Probability output, appropriate loss function
- **Cons:** Requires iterative optimization
- **Use when:** Probability estimates needed

**5. Practical considerations:**

**When to use linear regression:**
- **Quick prototyping**
- **Interpretability** is crucial
- **Computational** constraints
- **Simple** decision boundaries

**When to use logistic regression:**
- **Probability estimates** needed
- **Proper classification** metrics important
- **Model calibration** matters
- **Standard** classification practice

**Key insight:** **Linear regression** can perform **basic classification** but **logistic regression** is **more appropriate** for classification tasks.

# Practice 5 Solutions

**1. If $X$ and $Y$ are independent random variables, which of the following are true?**
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
```
Cov(X,Y) = E[XY] - E[X]E[Y]
```

**3. For independent variables:**
- **E[XY] = E[X]E[Y]** (from option b)
- **Therefore:** Cov(X,Y) = E[X]E[Y] - E[X]E[Y] = 0

**Why (b) is true - Expectation of product:**

**1. Independence property:**
- **E[XY] = E[X]E[Y]** is a fundamental property of independence
- **No correlation** means no linear relationship

**2. Intuitive explanation:**
- **Independent variables** don't influence each other
- **Product expectation** factors into individual expectations

**Why (c) is false - Variance of product:**

**1. Correct formula:**
```
Var(XY) = E[X²]E[Y²] - (E[X]E[Y])²
```

**2. This is NOT equal to:**
```
Var(X)Var(Y) = (E[X²] - E[X]²)(E[Y²] - E[Y]²)
```

**3. Example:**
- **X, Y ~ N(0,1)** independent
- **Var(XY) = 1** (not Var(X)Var(Y) = 1×1 = 1, but this is a special case)

**Why (d) is true - Joint probability:**

**1. For independent variables:**
```
P(X,Y) = P(X)P(Y)
```

**2. Also:**
```
P(Y|X) = P(Y) and P(X|Y) = P(X)
```

**3. Therefore:**
```
P(X,Y) = P(Y|X)P(X|Y) = P(Y)P(X) ✓
```

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
```
P(D|T) = P(T|D)P(D) / P(T)
```

**4. Calculate P(T) using law of total probability:**
```
P(T) = P(T|D)P(D) + P(T|D^c)P(D^c)
     = (0.90)(0.02) + (0.10)(0.98)
     = 0.018 + 0.098
     = 0.116
```

**5. Substitute into Bayes' theorem:**
```
P(D|T) = (0.90)(0.02) / 0.116
       = 0.018 / 0.116
       = 18/116
       = 9/58
```

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
```
L_n(p) = ∏ᵢ₌₁ⁿ P(X=xᵢ|p) = ∏ᵢ₌₁ⁿ (1-p)^(xᵢ-1)p
```

**2. Log-likelihood function:**
```
log L_n(p) = Σᵢ₌₁ⁿ log((1-p)^(xᵢ-1)p)
           = Σᵢ₌₁ⁿ [(xᵢ-1)log(1-p) + log(p)]
           = Σᵢ₌₁ⁿ (xᵢ-1)log(1-p) + Σᵢ₌₁ⁿ log(p)
           = (Σᵢ₌₁ⁿ xᵢ - n)log(1-p) + n log(p)
```

**3. Take derivative with respect to p:**
```
d/dp[log L_n(p)] = -(Σᵢ₌₁ⁿ xᵢ - n)/(1-p) + n/p
```

**4. Set derivative to zero:**
```
0 = -(Σᵢ₌₁ⁿ xᵢ - n)/(1-p) + n/p
```

**5. Solve for p:**
```
n/p = (Σᵢ₌₁ⁿ xᵢ - n)/(1-p)
n(1-p) = p(Σᵢ₌₁ⁿ xᵢ - n)
n - np = pΣᵢ₌₁ⁿ xᵢ - np
n = pΣᵢ₌₁ⁿ xᵢ
p = n/Σᵢ₌₁ⁿ xᵢ
```

**6. Verification:**
- **Second derivative:** d²/dp²[log L_n(p)] < 0 for 0 < p < 1
- **Maximum** confirmed at p = n/Σᵢ₌₁ⁿ xᵢ

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
- **MLE for variance:** σ²_MLE = (1/n)Σ(xᵢ - μ)²
- **Unbiased estimator:** σ²_unbiased = (1/(n-1))Σ(xᵢ - μ)²
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
```
max L(θ) = max P(data|θ)
```
- **θ** = model parameters
- **data** = observed data
- **L(θ)** = likelihood function

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
- **A is PSD** if x^T A x ≥ 0 for all x ∈ ℝ^n
- **Equivalent condition:** All eigenvalues of A are ≥ 0

**2. Spectral theorem:**
- **Symmetric matrices** have real eigenvalues
- **PSD matrices** have non-negative eigenvalues
- **Eigendecomposition:** A = QΛQ^T where Λ ≥ 0

**3. Mathematical proof:**
```
For eigenvector v with eigenvalue λ:
Av = λv
v^T Av = λv^T v = λ||v||² ≥ 0
Since ||v||² > 0, we must have λ ≥ 0
```

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
```
y = Xw + ε
```
where w ∈ ℝ^p is the weight vector

**2. Normal equations solution:**
```
ŵ = (X^T X)^(-1) X^T Y
```

**3. Prediction for new data point:**
```
ŷ_new = x_new^T ŵ
      = x_new^T (X^T X)^(-1) X^T Y
```

**4. Why this works:**

**Training phase:**
- **Minimize:** ||Y - Xw||²
- **Solution:** ŵ = (X^T X)^(-1) X^T Y
- **Assumes:** X^T X is invertible (full rank)

**Prediction phase:**
- **New input:** x_new ∈ ℝ^p
- **Prediction:** ŷ_new = x_new^T ŵ
- **Substitute:** ŵ from training

**5. Comparison with intercept model:**
- **With intercept:** ŷ_new = [x_new^T, 1] × [ŵ^T, b]^T
- **Without intercept:** ŷ_new = x_new^T ŵ
- **No bias term** b in the model

**6. Matrix dimensions:**
- **X:** n × p
- **Y:** n × 1
- **x_new:** p × 1
- **ŵ:** p × 1
- **ŷ_new:** scalar

**Key insight:** **Prediction** is the **dot product** of the new feature vector with the **learned weight vector**.

**7.**

**Suppose you want to use linear regression to fit a weight vector $w \in \mathbb{R}^d$ and an offset/intercept term $b \in \mathbb{R}$ using data points $x_i \in \mathbb{R}^d$. What is the minimum number of data points $n$ required in your training set such that there will be a single unique solution?**

**Answer:** $n = d+1$

**Explanation:**

**The minimum number of data points needed is n = d+1** to ensure a unique solution for linear regression with intercept.

**Step-by-step reasoning:**

**1. Linear regression model with intercept:**
```
y_i = x_i^T w + b + ε_i
```

**2. Augmented data matrix:**
```
X_aug = [X, 1] ∈ ℝ^(n×(d+1))
```
where 1 is a column of ones

**3. Normal equations:**
```
ŵ_aug = (X_aug^T X_aug)^(-1) X_aug^T Y
```

**4. Rank requirement for unique solution:**
- **X_aug^T X_aug** must be **invertible**
- **Invertible** requires **full rank**
- **Full rank** requires **n ≥ d+1**

**5. Why n = d+1 is minimum:**

**If n < d+1:**
- **X_aug** has more columns than rows
- **Rank(X_aug) ≤ n < d+1**
- **X_aug^T X_aug** is singular
- **No unique solution**

**If n = d+1:**
- **X_aug** is square (d+1) × (d+1)
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
```
Original: y = f(x) (nonlinear)
Basis expansion: y = w₁φ₁(x) + w₂φ₂(x) + ... + wₖφₖ(x)
```
where φᵢ(x) are basis functions

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
```
y_i ~ N(x_i^T w, σ²)
```

**2. Likelihood function:**
```
L(w) = ∏ᵢ (1/√(2πσ²)) exp(-(y_i - x_i^T w)²/(2σ²))
```

**3. Log-likelihood:**
```
log L(w) = -n/2 log(2πσ²) - (1/(2σ²)) Σᵢ (y_i - x_i^T w)²
```

**4. MLE objective:**
```
max log L(w) = min Σᵢ (y_i - x_i^T w)²
```

**5. Comparison of loss functions:**

**Squared error (MLE for Gaussian):**
```
L_squared = Σᵢ (y_i - x_i^T w)²
```

**Absolute error (MLE for Laplace):**
```
L_absolute = Σᵢ |y_i - x_i^T w|
```

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
```
min ||y - Xw||² + λ||w||²
```

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
ŵ_ridge = (X^T X + 0I)^(-1) X^T y
       = (X^T X)^(-1) X^T y
       = ŵ_OLS
```

**2. No regularization:**
- **λ = 0** means no penalty term
- **Objective:** min ||y - Xw||²
- **Same as** ordinary least squares

**Why (f) is true - λ → ∞ shrinks to zero:**

**1. As λ → ∞:**
- **Regularization term** dominates
- **Objective:** approximately min λ||w||²
- **Solution:** w → 0

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
```
min ||y - Xw||² + λ||w||₁
```
- **||w||₁ = Σ|wᵢ|** (L1 norm)
- **Sharp corners** at axes
- **Can produce exact zeros**

**Ridge (L2):**
```
min ||y - Xw||² + λ||w||₂²
```
- **||w||₂² = Σwᵢ²** (L2 norm)
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

**Explanation:** The correct answer is (b), False, because irreducible error is irreducible.

**14. One Answer**

**Which strategy is most effective for reducing variance in a high-variance, low-bias model?**
*   (a) Increasing the number of training examples.
*   (b) Increasing the model complexity.
*   (c) Decreasing regularization.
*   (d) Removing the features that exhibit high variance across training examples.

**Correct answers:** (a)

**Explanation:** The correct answer is (a). (b) is incorrect because increasing model complexity usually increases variance. (c) is incorrect because decreasing regularization will usually increase variance. (d) is incorrect because the variance of features is a difference concept than variance of a model—removing the high-variance features could increase or decrease the model variance and there is no way knowing a priori.

**15. One Answer**

**If your model has high validation loss and high training loss, which action is most appropriate to improve the model?**
*   (a) Increase the model complexity.
*   (b) Increase $k$ in $k$-fold cross-validation.
*   (c) Increase the number of training examples.
*   (d) Apply regularization to reduce overfitting.

**Correct answers:** (a)

**Explanation:** If the validation and training losses are both high, it suggests that the model is underfitting (high bias), meaning it is too simple to capture the underlying patterns in the data. Increasing the model complexity

**16.**

**In a project using a customer purchase history dataset with a 60/20/20 train, validation, and test split, the validation accuracy remains consistently lower than the training accuracy. What could be a reason for this?**

**Answer:**

**Explanation:** The validation accuracy is likely lower due to overfitting (the model is complex, variance is high). Overfitting happens when a model learns too much detail and noise from the training data, capturing specific patterns that don't apply to new, unseen data. This makes the model perform well on the training set but poorly on the validation or test sets, as it fails to generalize.

**17. One Answer**

**A consortium of 10 hospitals have pooled together their Electronic Health Records data and want to build a machine learning model to predict patient prognosis based on patient records in their hospitals. They want to maximize the accuracy of their model across all 10 hospitals and do not plan to deploy their model in other hospitals. How should they split the data into train / validation / test sets?**
*   (a) Leave out data from 1 hospital for the validation set, data from another hospital for the test set, and use the rest for train set.
*   (b) Leave out data from 1 hospital for the validation set, data from another hospital for the test set, and use the rest for train set. After training, add the validation data to the train set and re-train the model on the combined data.
*   (c) Use data from 8 hospitals with the most number of records for training, and use data from the other 2 hospitals for validation and test sets.
*   (d) Mix data from all hospitals, randomly shuffle all the records, and then do the 80/10/10 train/validation/test split.

**Correct answers:** (d)

**Explanation:** D is the correct answer, as it is the only approach that avoids overfitting hyperparameters to data from only one or two hospitals. Each hospital may have a different distribution of patients, doctors, outcomes, etc. So we should not expect all data to be IID.

**18.**

**Given the task of determining loan approval for applicants using a predictive model given applicant features such as race, salary, education, etc., is it always best practice to allow the model to use all of the given features? Why or why not?**

**Answer:**

**Explanation:** No, we should not ALWAYS use all the features. In addition to building an accurate model, we also want to build ethically-informed models and this requires us to be thoughtful about what features go into our analyses. For any feature we choose to include, our model may find correlations that are not necessarily causations, that are either coincidental or the result of pre-existing biases. Depending on the most informed choice to make, the best practice may or may not be to include all available features.

**19. One Answer**

**You are building a predictive model about users of a website. Suppose that after you train your model on historical user data, the distribution of users shifts dramatically. What can happen if you deploy your machine learning system without addressing this distribution shift?**
*   (a) The model will automatically adapt to new data distributions.
*   (b) The model will generate more diverse predictions, increasing its overall accuracy.
*   (c) The model will maintain its original performance indefinitely regardless of data changes.
*   (d) The model's predictions may become unreliable or biased.

**Correct answers:** (d)

**Explanation:** Machine learning models can only reliably generalize to data from the same distribution they were trained on; when faced with different distributions, their predictions may become unreliable or biased due to this domain shift, rather than becoming more diverse or accurate.

**20. One Answer**

**For a possibly non-convex optimization problem, gradient descent on the full dataset always finds a better solution than stochastic gradient descent.**
*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:** Gradient descent is not always better than stochastic gradient descent. The variability of SGD can escape local minima more effectively than deterministic gradient descent.

**21. Select All That Apply**

**Given the gradient descent algorithm, $w_{t+1} = w_t - \eta \frac{df(w)}{dw} \Big|_{w=w_t}$, which of the following statement is correct regarding the hyperparameter $\eta$?**
*   (a) $\eta$ controls the magnitude of each step.
*   (b) $\eta$ determines the initial value of $w$.
*   (c) A larger $\eta$ guarantees faster convergence to the global minimum.
*   (d) A smaller $\eta$ guarantees faster convergence to the global minimum.

**Correct answers:** (a)

**Explanation:** $\eta$ controls the step size in the gradient descent algorithm. $\eta$ and $w$ are independently set. A larger $\eta$ may cause model update to overshoot the global minimum. A smaller $\eta$ may cause model to get stuck in local minimum.

**22. Select All That Apply**

**Which of the following functions are convex?**
*   (a) $f(x) = x^3$
*   (b) $f(x) = \frac{3x(x-1)}{2}$
*   (c) $f(x) = \sin x$, for $x \in [\pi, 2\pi]$
*   (d) $f(x) = \log_{10}(x)$

**Correct answers:** (b), (c)

**Explanation:** To determine which functions are convex, we need to examine the second derivative of each function. A function $f(x)$ is convex on an interval if $f''(x) \ge 0$ for all $x$ in that interval. (b) and (c) are the functions satisfy the definition. You can also plot the functions for an informal check of the convexity.

**23. Which of the following are true about a convex function $f(x): \mathbb{R}^d \rightarrow \mathbb{R}$?**
*   (a) $f(x)$ must be differentiable across its entire domain.
*   (b) $f(x)$ has a unique global minimum.
*   (c) $g(x) = -f(x)$ is also convex.
*   (d) If $f(x)$ is twice differentiable, then $z^T \nabla^2 f(x)z \geq 0$ for all $z \in \mathbb{R}^d$.

**Correct answers:** (d)

**Explanation:** The correct answer(s) are: D. A is incorrect—consider the function $f(x) = |x|$. This is convex but is not differentiable at $x=0$. B is incorrect because a convex function may have multiple connected global minima (e.g., the "half-pipes" we discussed when building up ridge regression) or no global minima (e.g., a hyperplane with non-zero slope). C is only true when $f(x)$ is a linear or affine function, but is not true in general (e.g., a bowl is convex, but when you flip it upside down it becomes concave).

**24. Which of the following have convex objective functions?**
*   (a) Linear regression
*   (b) Linear regression with arbitrary nonlinear basis functions
*   (c) Ridge regression
*   (d) Lasso regression
*   (e) Logistic regression

**Correct answers:** (a), (b), (c), (d), (e)

**Explanation:** All the aforementioned models use a linear function to map inputs to outputs, and their objective function is linear.

**25. Which of the following scenarios are better suited for a logistic regression model over a linear regression model?**
*   (a) Forecasting the price of stocks for the next year, given the price of stocks for the past year.
*   (b) Diagnosing the presence or absence of a rare disease, given a medical x-ray.
*   (c) Predicting what the average global temperature will be in the next year, given historical climate data.
*   (d) Predicting how likely a student is to successfully complete a 4-year college degree, given their high school grades.
*   (e) Predicting the hardness of a material on a scale of 1-10 given the molecular structure of the material.

**Correct answers:** (b), (d)

**Explanation:** (b) and (d) are classification problems; the others are more suited to regression problems.

**26. Which of the following statements about classification are true?**

**Recall that the softmax function $\sigma: \mathbb{R}^k \rightarrow (0,1)^k$ takes a vector $z \in \mathbb{R}^k$ and returns a vector $\sigma(z) \in (0,1)^k$ with**
$$\sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)}$$

*   (a) Consider a binary classification setting. If the data is linearly separable, we can use a logistic regression model with quadratic features to avoid overfitting.
*   (b) Because binary logistic regression is a convex optimization problem, it has a closed-form solution.
*   (c) The softmax function is used when we are classifying $k > 2$ classes. When we are classifying only $k = 2$ classes, softmax regression will overfit, so we use binary logistic regression instead.
*   (d) We can use linear regression to solve classification problems, though the model we learn might not be as accurate compared to using logistic/softmax regression.

**Correct answers:** (d)

**Explanation:** Quadratic features can still lead to overfitting, and while some convex optimization problems (like linear regression) have closed-form solutions, others like logistic regression require iterative methods. Softmax regression's complexity depends on implementation, and linear regression can perform basic classification tasks despite not being optimized for this purpose.

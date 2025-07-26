# Problem Set #3 Solutions: Math Review

This document contains solutions to Problem Set #3, focusing on advanced mathematical concepts relevant to machine learning.

## Problem 1: Advanced Linear Algebra

### (a) Matrix Decompositions

**Problem:** Let $`A \in \mathbb{R}^{n \times n}`$ be a symmetric positive definite matrix. Show that $`A`$ can be written as $`A = LL^T`$ where $`L`$ is a lower triangular matrix.

**Solution:**

Since $`A`$ is symmetric positive definite, it has a Cholesky decomposition. We can prove this by induction.

For $`n = 1`$: $`A = [a_{11}]`$ where $`a_{11} > 0`$. Then $`L = [\sqrt{a_{11}}]`$ gives $`A = LL^T`$.

Assume the result holds for $`(n-1) \times (n-1)`$ matrices. For an $`n \times n`$ matrix $`A`$, we can write:

```math
A = \begin{bmatrix}
A_{11} & a_{12} \\
a_{12}^T & a_{nn}
\end{bmatrix}
```

where $`A_{11}`$ is $`(n-1) \times (n-1)`$ and $`a_{12}`$ is $`(n-1) \times 1`$.

By the induction hypothesis, $`A_{11} = L_{11}L_{11}^T`$ where $`L_{11}`$ is lower triangular.

Let $`L = \begin{bmatrix} L_{11} & 0 \\ l_{21}^T & l_{nn} \end{bmatrix}`$ where $`l_{21} = L_{11}^{-1}a_{12}`$ and $`l_{nn} = \sqrt{a_{nn} - l_{21}^T l_{21}}`$.

Then $`A = LL^T`$ as required.

**Explanation:** This proof demonstrates the existence of the Cholesky decomposition for symmetric positive definite matrices, which is crucial for efficient matrix operations in machine learning algorithms.

### (b) Eigenvalue Bounds

**Problem:** Let $`A \in \mathbb{R}^{n \times n}`$ be symmetric with eigenvalues $`\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n`$. Show that for any unit vector $`x`$:

$`\lambda_n \leq x^T A x \leq \lambda_1`$

**Solution:**

Since $`A`$ is symmetric, it has an orthonormal eigenbasis $`\{v_1, v_2, \ldots, v_n\}`$ with corresponding eigenvalues $`\{\lambda_1, \lambda_2, \ldots, \lambda_n\}`$.

Any unit vector $`x`$ can be written as $`x = \sum_{i=1}^n c_i v_i`$ where $`\sum_{i=1}^n c_i^2 = 1`$.

Then:

```math
x^T A x = \left(\sum_{i=1}^n c_i v_i\right)^T A \left(\sum_{j=1}^n c_j v_j\right) = \sum_{i=1}^n \sum_{j=1}^n c_i c_j v_i^T A v_j
```

Since $`Av_i = \lambda_i v_i`$ and $`v_i^T v_j = \delta_{ij}`$:

```math
x^T A x = \sum_{i=1}^n c_i^2 \lambda_i
```

Since $`\sum_{i=1}^n c_i^2 = 1`$ and $`\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n`$, we have:

$`\lambda_n \leq \sum_{i=1}^n c_i^2 \lambda_i \leq \lambda_1`$

**Explanation:** This result is known as the Rayleigh-Ritz theorem and is fundamental in optimization theory, particularly for understanding the behavior of quadratic forms.

## Problem 2: Optimization Theory

### (a) Convexity and Gradient Descent

**Problem:** Let $`f : \mathbb{R}^n \to \mathbb{R}`$ be a convex function with $`L`$-Lipschitz gradient. Show that for gradient descent with step size $`\alpha = \frac{1}{L}`$:

$`f(x_{k+1}) - f(x_k) \leq -\frac{1}{2L} \|\nabla f(x_k)\|_2^2`$

**Solution:**

Since $`f`$ has $`L`$-Lipschitz gradient, we have:

```math
f(y) \leq f(x) + \nabla f(x)^T (y - x) + \frac{L}{2} \|y - x\|_2^2
```

For gradient descent, $`x_{k+1} = x_k - \alpha \nabla f(x_k)`$ with $`\alpha = \frac{1}{L}`$.

Substituting $`x = x_k`$ and $`y = x_{k+1}`$:

```math
f(x_{k+1}) \leq f(x_k) + \nabla f(x_k)^T (x_{k+1} - x_k) + \frac{L}{2} \|x_{k+1} - x_k\|_2^2
```

Since $`x_{k+1} - x_k = -\frac{1}{L} \nabla f(x_k)`$:

```math
f(x_{k+1}) \leq f(x_k) - \frac{1}{L} \|\nabla f(x_k)\|_2^2 + \frac{L}{2} \cdot \frac{1}{L^2} \|\nabla f(x_k)\|_2^2
```

Simplifying:

```math
f(x_{k+1}) \leq f(x_k) - \frac{1}{L} \|\nabla f(x_k)\|_2^2 + \frac{1}{2L} \|\nabla f(x_k)\|_2^2 = f(x_k) - \frac{1}{2L} \|\nabla f(x_k)\|_2^2
```

Therefore:

$`f(x_{k+1}) - f(x_k) \leq -\frac{1}{2L} \|\nabla f(x_k)\|_2^2`$

**Explanation:** This result provides a theoretical guarantee for the convergence rate of gradient descent on convex functions with Lipschitz gradients, which is essential for understanding optimization algorithms in machine learning.

### (b) Lagrange Multipliers

**Problem:** Consider the optimization problem:

```math
\min_{x \in \mathbb{R}^n} f(x) \quad \text{subject to} \quad g_i(x) = 0, \quad i = 1, 2, \ldots, m
```

Show that if $`x^*`$ is a local minimum and the constraint gradients $`\{\nabla g_i(x^*)\}_{i=1}^m`$ are linearly independent, then there exist Lagrange multipliers $`\lambda_1, \lambda_2, \ldots, \lambda_m`$ such that:

$`\nabla f(x^*) + \sum_{i=1}^m \lambda_i \nabla g_i(x^*) = 0`$

**Solution:**

This is a fundamental result in constrained optimization. The proof relies on the fact that at a local minimum $`x^*`$, the gradient of the objective function $`\nabla f(x^*)`$ must be orthogonal to the tangent space of the constraint manifold.

Since the constraint gradients are linearly independent, they form a basis for the normal space at $`x^*`$. Therefore, $`\nabla f(x^*)`$ can be written as a linear combination of the constraint gradients:

```math
\nabla f(x^*) = -\sum_{i=1}^m \lambda_i \nabla g_i(x^*)
```

Rearranging gives the required condition:

$`\nabla f(x^*) + \sum_{i=1}^m \lambda_i \nabla g_i(x^*) = 0`$

**Explanation:** This is the first-order necessary condition for constrained optimization, known as the Karush-Kuhn-Tucker (KKT) conditions. It's essential for understanding optimization problems with constraints, which are common in machine learning.

## Problem 3: Probability and Statistics

### (a) Maximum Likelihood Estimation

**Problem:** Let $`X_1, X_2, \ldots, X_n`$ be i.i.d. random variables from a normal distribution $`\mathcal{N}(\mu, \sigma^2)`$. Find the maximum likelihood estimators for $`\mu`$ and $`\sigma^2`$.

**Solution:**

The likelihood function is:

```math
L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
```

The log-likelihood is:

```math
\ell(\mu, \sigma^2) = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2
```

Taking partial derivatives:

```math
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu)
```

```math
\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2} \sum_{i=1}^n (x_i - \mu)^2
```

Setting these to zero:

$`\hat{\mu} = \frac{1}{n} \sum_{i=1}^n x_i = \bar{x}`$

$`\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2`$

**Explanation:** These are the well-known maximum likelihood estimators for the mean and variance of a normal distribution. The sample mean is unbiased for $`\mu`$, but the sample variance is biased for $`\sigma^2`$ (hence the common use of $`n-1`$ in the denominator for the unbiased estimator).

### (b) Central Limit Theorem

**Problem:** State and prove the Central Limit Theorem for i.i.d. random variables.

**Solution:**

**Statement:** Let $`X_1, X_2, \ldots`$ be i.i.d. random variables with mean $`\mu`$ and finite variance $`\sigma^2`$. Let $`S_n = X_1 + X_2 + \cdots + X_n`$. Then:

```math
\frac{S_n - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
```

as $`n \to \infty`$.

**Proof Sketch:**

The proof uses characteristic functions. Let $`\phi(t)`$ be the characteristic function of $`X_1 - \mu`$:

```math
\phi(t) = \mathbb{E}[e^{it(X_1 - \mu)}] = 1 - \frac{\sigma^2 t^2}{2} + o(t^2)
```

The characteristic function of $`\frac{S_n - n\mu}{\sigma\sqrt{n}}`$ is:

```math
\phi_n(t) = \left[\phi\left(\frac{t}{\sigma\sqrt{n}}\right)\right]^n = \left[1 - \frac{t^2}{2n} + o\left(\frac{t^2}{n}\right)\right]^n
```

Taking the limit as $`n \to \infty`$:

```math
\lim_{n \to \infty} \phi_n(t) = e^{-t^2/2}
```

This is the characteristic function of the standard normal distribution, completing the proof.

**Explanation:** The Central Limit Theorem is one of the most important results in probability theory, explaining why normal distributions appear so frequently in nature and statistics. It's fundamental to many statistical inference procedures.

## Problem 4: Information Theory

### (a) Entropy and Mutual Information

**Problem:** Let $`X`$ and $`Y`$ be discrete random variables. Define the entropy $`H(X)`$ and mutual information $`I(X; Y)`$. Show that $`I(X; Y) \geq 0`$ with equality if and only if $`X`$ and $`Y`$ are independent.

**Solution:**

**Definitions:**

Entropy: $`H(X) = -\sum_x p(x) \log p(x)`$

Mutual Information: $`I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}`$

**Proof that $`I(X; Y) \geq 0`$:**

Using the fact that $`\log x \leq x - 1`$ for $`x > 0`$:

```math
I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} \geq \sum_{x,y} p(x,y) \left(\frac{p(x,y)}{p(x)p(y)} - 1\right)
```

Simplifying:

```math
I(X; Y) \geq \sum_{x,y} \frac{p(x,y)^2}{p(x)p(y)} - \sum_{x,y} p(x,y) = \sum_{x,y} \frac{p(x,y)^2}{p(x)p(y)} - 1
```

By the Cauchy-Schwarz inequality:

```math
\left(\sum_{x,y} \frac{p(x,y)^2}{p(x)p(y)}\right) \left(\sum_{x,y} p(x)p(y)\right) \geq \left(\sum_{x,y} p(x,y)\right)^2 = 1
```

Since $`\sum_{x,y} p(x)p(y) = 1`$, we have $`\sum_{x,y} \frac{p(x,y)^2}{p(x)p(y)} \geq 1`$.

Therefore, $`I(X; Y) \geq 0`$.

**Equality condition:** $`I(X; Y) = 0`$ if and only if $`p(x,y) = p(x)p(y)`$ for all $`x, y`$, which is the definition of independence.

**Explanation:** This result shows that mutual information is a natural measure of dependence between random variables, with zero mutual information characterizing independence. This is fundamental in information theory and has applications in feature selection and dimensionality reduction.

### (b) Kullback-Leibler Divergence

**Problem:** Let $`P`$ and $`Q`$ be probability distributions over the same sample space. Define the Kullback-Leibler divergence $`D_{KL}(P \| Q)`$ and show that $`D_{KL}(P \| Q) \geq 0`$ with equality if and only if $`P = Q`$.

**Solution:**

**Definition:**

```math
D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
```

**Proof that $`D_{KL}(P \| Q) \geq 0`$:**

Using the fact that $`\log x \leq x - 1`$ for $`x > 0`$:

```math
D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \geq \sum_x p(x) \left(\frac{p(x)}{q(x)} - 1\right)
```

Simplifying:

```math
D_{KL}(P \| Q) \geq \sum_x \frac{p(x)^2}{q(x)} - \sum_x p(x) = \sum_x \frac{p(x)^2}{q(x)} - 1
```

By the Cauchy-Schwarz inequality:

```math
\left(\sum_x \frac{p(x)^2}{q(x)}\right) \left(\sum_x q(x)\right) \geq \left(\sum_x p(x)\right)^2 = 1
```

Since $`\sum_x q(x) = 1`$, we have $`\sum_x \frac{p(x)^2}{q(x)} \geq 1`$.

Therefore, $`D_{KL}(P \| Q) \geq 0`$.

**Equality condition:** $`D_{KL}(P \| Q) = 0`$ if and only if $`p(x) = q(x)`$ for all $`x`$, which means $`P = Q`$.

**Explanation:** The Kullback-Leibler divergence is a fundamental measure of the difference between probability distributions. It's widely used in machine learning for variational inference, generative models, and many other applications where we need to compare or approximate probability distributions.


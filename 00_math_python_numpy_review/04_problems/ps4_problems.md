# Problem Set #4: Math Review

In this section, we explore maximum likelihood estimation with more examples of noise densities, review some basics about subspaces in linear algebra, and go over data standardization and normalization.

## Definitions, again!

Norms are instrumental and frequently used. For any n-dimensional vector $`v`$ (i.e., $`v \in \mathbb{R}^n`$), the following norms are defined:

**(a) One-norm ($`\ell_1`$):** $`\|v\|_1 = \sum_{i=1}^n |v_i|`$

**(b) Two-norm ($`\ell_2`$):** $`\|v\|_2 = \sqrt{v^T v} = \sqrt{\sum_{i=1}^n v_i^2}`$

**(c) $\infty$-norm:** $`\|v\|_\infty = \max_i |v_i|`$

**Symmetric Matrices and the Quadratic Form:**

Let $`A \in \mathbb{R}^{n \times n}`$.

**(a)** We have that the matrix $`A`$ is symmetric iff $`A = A^T`$

**(b)** The quadratic form is defined to be $`x^T A x`$ for any vector $`x \in \mathbb{R}^n`$. The matrix $`A`$ is said to be positive semi-definite if $`x^T A x \geq 0`$

**Closure of the Normal and Laplacian under scale and shift:**

**(a) Normal:** If $`X \sim N(\mu_x, \sigma^2)`$, then we have that $`Y = aX + b \sim N(a\mu_x + b, a^2\sigma^2)`$.

**(b) Laplacian:** If $`X \sim \text{Laplace}(\mu, b)`$ then $`kX + c \sim \text{Laplace}(k\mu + c, |k|b)`$

## 1. Maximum Likelihood Estimation

In this section, we formulate maximum likelihood estimation for different noise densities as different minimization problems. Specifically, we'll see how each noise distribution corresponds to a specific objective function.

We consider the linear measurement model (parameterized by $`w`$), $`y_i = x_i^T w + v_i`$ for $`i = 1, 2, \ldots, m`$. The noise $`v_i`$ for different measurements $`(x_i, y_i)`$ are all independent and identically distributed. Under our assumption of a linear model, $`v_i = y_i - x_i^T w`$. Note Per the principle of maximum likelihood estimation, we seek to maximize

```math
\log p_w((x_1, y_1), \ldots, (x_m, y_m)) = \log \prod_{i=1}^m p(y_i - x_i^T w).
```

**(a)** Show that when the noise measurements follow a Gaussian distribution ($`v_i \sim N(0, \sigma^2)`$), the maximum likelihood estimate of $`w`$ is the solution to $`\min_w \|Xw - Y\|_2^2`$. Here each row in $`X`$ corresponds to an $`x_i`$, and each row in $`Y`$ to $`y_i`$.

**(b)** When the noise measurements follow a Laplacian distribution ($`p(z) = (1/2a) \exp(-|z|/a)`$), what is the maximum likelihood estimate of $`w`$? Express your answer as the solution to an optimization problem such as in the previous part.

**(c)** When the noise measurements follow a uniform distribution ($`p(z) = (1/2a)`$ on $`[-a, a]`$), what is the maximum likelihood estimate of $`w`$? Express your answer as a condition to be satisfied by some function of $`w`$.

## 2. Data Normalization/Standardization

Sometimes, our features have very different ranges of values. This is not ideal and can lead to numerical issues (e.g., overflow) and optimization difficulties.

There are two ways to take care of this issue. One is called data normalization, and the other is called data standardization. Sometimes these terms are used interchangeably, but it is important to understand the difference.

Below, $`x_i^{(j)}`$ represents the value of the $`i^{th}`$ feature of the $`j^{th}`$ data point.

### 2.1. Data Standardization

Data standardization is the task of transforming each feature in our dataset to have mean 0 and variance 1. The typical way to do this is using the Z-Score, which is defined as below:

```math
\tilde{x}_i^{(j)} = \frac{x_i^{(j)} - \mu_i}{\sigma_i}
```

Where $`\mu_i`$ is the mean of each feature and $`\sigma_i`$ is the standard deviation of each feature, which are empirically calculated from the data.

**Question:** what should you do when $`\sigma_i = 0`$ for some $`i`$?

### 2.2. Data Normalization

Data normalization refers to the task of rescaling each feature in our dataset to have range $`[0, 1]`$.

One such method to achieve this is min-max scaling:

```math
\tilde{x}_i^{(j)} = \frac{x_i^{(j)} - x_i^{min}}{x_i^{max} - x_i^{min}}
```

Where $`x_i^{min}`$, $`x_i^{max}`$ are the minimum and maximum values of feature $`i`$ in our dataset, respectively.

When training and evaluating your model, you should calculate the parameters for your normalization or standardization function on the training set ONLY!

In other words, if we were using Z-Score, we'd calculate our $`\mu_i, \sigma_i`$ on the training set, and use those same values when standardizing our validation/test data. The same applies to normalization methods, such as min-max scaling, where we want to determine $`x_i^{min}, x_i^{max}`$ from training data. This will likely mean that we may get some values outside $`[0, 1]`$ after rescaling new data, but, given that our training dataset is large enough, this shouldn't be much of an issue (so unseen data likely won't be wildly outside of $`[0, 1]`$).

**Question 1:** Should we always choose $`x_i^{min}`$ and $`x_i^{max}`$ based on train data? Can we sometimes do better? Think about cases when we have some underlying information about data.

**Question 2:** When can values outside of $`[0, 1]`$ range in test set cause issues?

## 3. Linear Algebra Review Part 2

Let $`X \in \mathbb{R}^{m \times n}`$. $`X`$ may not have full rank. We explore properties about the four fundamental subspaces of $`X`$. Please refer to section 3 of the supplementary material for more info on matrix properties and subspaces.

### 3.1. Connections between subspaces of $`X`$

Check the following facts.

**(a)** The rowspace of $`X`$ is the columnspace of $`X^T`$, and vice versa.

**(b)** The nullspace of $`X`$ and the rowspace of $`X`$ are orthogonal complements. This can be written in shorthand as $`Null(X) = Range(X^T)^\perp`$. This is further equivalent to saying $`Range(X^T) = Null(X)^\perp`$.

**(c)** The nullspace of $`X^T`$ is orthogonal to the columnspace of $`X`$. This can be written in shorthand as $`Null(X^T) = Range(X)^\perp`$.

### 3.2. Linear algebra facts for linear regression

We saw in lecture on Linear Regression that the closed form expression for linear regression without an offset involves the term $`(X^T X)^{-1}`$.

**(a)** Is it true that the matrix $`X^T X`$ is always symmetric and positive semidefinite?

**(b)** State and prove the connection between the nullspace of $`X`$ and the nullspace of $`X^T X`$. That is, your statement should look like one of the following: $`Null(X) \subseteq Null(X^T X)`$, or $`Null(X) \supseteq Null(X^T X)`$ or $`Null(X) = Null(X^T X)`$.

**(c)** Is it true that $`X^T X`$ is always invertible?

**(d)** Based on the above fact about the connection between the nullspaces of $`X`$ and $`X^T X`$ and the expression for linear regression without an offset (that we referred to two problems above), justify the use of "tall skinny" data matrix $`X`$ as opposed to a "short wide" matrix $`X`$.

**(e)** The columnspace and rowspace of $`X^T X`$ are the same, and are equal to the rowspace of $`X`$. (Hint: Use the relationship between nullspace and rowspace.)
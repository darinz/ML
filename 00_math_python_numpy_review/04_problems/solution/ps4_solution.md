# Problem Set #4 Solutions: Math Review

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

**Solution:**

When $`v_i \sim N(0, \sigma^2)`$, the density is given by the expression $`p(v) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{v^2}{2\sigma^2}}`$. This implies that the

**MLE of parameter is:**

```math
\hat{w}_{MLE} = \arg \max_w \log p_w((x_1, y_1), \ldots, (x_m, y_m))
```

```math
= \arg \max_w \log \prod_{i=1}^m p(y_i - x_i^T w)
```

```math
= \arg \max_w \sum_{i=1}^m \log p(y_i - x_i^T w) \quad [\log(ab) = \log a + \log b]
```

```math
= \arg \max_w \sum_{i=1}^m \log \left[ \frac{1}{\sqrt{2\pi\sigma^2}} e^{-(y_i - x_i^T w)^2 / (2\sigma^2)} \right]
```

```math
= \arg \max_w \left( m \log \left( \frac{1}{\sqrt{2\pi\sigma^2}} \right) + \sum_{i=1}^m \frac{-(y_i - x_i^T w)^2}{2\sigma^2} \right)
```

```math
= \arg \max_w \sum_{i=1}^m -\frac{1}{2\sigma^2} (y_i - x_i^T w)^2 \quad (\text{constant offset doesn't affect results})
```

```math
= \arg \max_w \sum_{i=1}^m -(y_i - x_i^T w)^2 \quad (\text{constant scalar doesn't affect results})
```

```math
= \arg \min_w \sum_{i=1}^m (y_i - x_i^T w)^2 = \arg \min_w \|Xw - Y\|_2^2
```

Therefore, the maximum likelihood estimate is given by $`\arg \min \|Xw - Y\|_2^2`$, as claimed.

**(b)** When the noise measurements follow a Laplacian distribution ($`p(z) = (1/2a) \exp(-|z|/a)`$), what is the maximum likelihood estimate of $`w`$? Express your answer as the solution to an optimization problem such as in the previous part.

**Solution:**

For $`a > 0`$, with density $`p(z) = (1/2a) \exp(-|z|/a)`$, we follow a similar procedure as in part (a)

```math
\hat{w}_{MLE} = \arg \max_w \log p_w((x_1, y_1), \ldots, (x_m, y_m))
```

```math
= \arg \max_w \log \prod_{i=1}^m p(y_i - x_i^T w)
```

```math
= \arg \max_w \sum_{i=1}^m \log p(y_i - x_i^T w) \quad [\log(ab) = \log a + \log b]
```

```math
= \arg \max_w \sum_{i=1}^m \log \left( \frac{1}{2a} e^{-\frac{|y_i - x_i^T w|}{a}} \right)
```

```math
= \arg \max_w \sum_{i=1}^m \left( \log \left( \frac{1}{2a} \right) - \frac{|y_i - x_i^T w|}{a} \right)
```

```math
= \arg \max_w \sum_{i=1}^m - \frac{|y_i - x_i^T w|}{a} \quad \text{constant offset doesn't affect optimization}
```

```math
= \frac{1}{a} \arg \max_w \sum_{i=1}^m -|y_i - x_i^T w|
```

```math
= \arg \max_w \sum_{i=1}^m -|y_i - x_i^T w| \quad \text{constant offset doesn't affect optimization}
```

```math
= \arg \min_w \sum_{i=1}^m |y_i - x_i^T w| = \arg \min_w \|Xw - Y\|_1
```

Therefore the maximum likelihood estimate of $`w`$ is $`\hat{w} = \arg \min_w \|Xw - Y\|_1`$.

**(c)** When the noise measurements follow a uniform distribution ($`p(z) = (1/2a)`$ on $`[-a, a]`$), what is the maximum likelihood estimate of $`w`$? Express your answer as a condition to be satisfied by some function of $`w`$.

**Solution:**

For uniformly distributed $`v_i`$ on $`[-a, a]`$, the density function is $`p(z) = \frac{1}{2a}`$, we have that

```math
\hat{w}_{MLE} = \arg \max_w \log p_w((x_1, y_1), \ldots, (x_m, y_m))
```

```math
= \arg \max_w \log \prod_{i=1}^m p(y_i - x_i^T w)
```

```math
= \arg \max_w \sum_{i=1}^m \log p(y_i - x_i^T w) \quad [\log(ab) = \log a + \log b]
```

Using an indicator function, we have that this can be simplified into

```math
\hat{w}_{MLE} = \arg \max_w \sum_{i=1}^m \log \left( \frac{1}{2a} \cdot \mathbf{1}\{-a \le y_i - x_i^T w \le a\} \right)
```

```math
= \arg \max_w \sum_{i=1}^m \left( \log \left( \frac{1}{2a} \right) + \log \left( \mathbf{1}\{-a \le y_i - x_i^T w \le a\} \right) \right)
```

```math
= \arg \max_w \sum_{i=1}^m \log \left( \mathbf{1}\{-a \le y_i - x_i^T w \le a\} \right) \quad \text{Constant factor doesn't affect optimization}
```

We now simplify this to be

```math
\hat{w}_{MLE} = \arg \max_w \begin{cases} 0 & -a \le y_i - x_i^T w \le a \\ -\infty & \text{otherwise} \end{cases}
```

We further simplify this to

```math
\hat{w}_{MLE} = \arg \max_w \begin{cases} 0 & \|Y - Xw\|_\infty \le a \\ -\infty & \text{otherwise} \end{cases}
```

Therefore the maximum likelihood estimate is given by any $`w`$ that satisfies $`\|Y - Xw\|_\infty \le a`$ which is the same as $`\|Xw - Y\|_\infty \le a`$.

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

**Solution:**

Having $`\sigma_i = 0`$ for some feature means that the value for that feature is constant in our dataset. If we leave it as 0, we will encounter a divide by 0 error. Since the feature is constant, once we subtract the mean, the new value for the feature will be 0, so we can divide by anything except 0 to avoid this error.

Having $`\sigma_i = 0`$ is rare, and may be a sign something is wrong with your data or code. One specific case to watch out for is appending your bias column of ones before standardizing.

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

**Solution:**
Consider RGB images. These are typically encoded as arrays of shape (3, height, width), with each value being an integer in range $`[0, 255]`$. In this case we should just use $`x_i^{max} = 255`$ and $`x_i^{min} = 0`$ to normalize the data.

In general there can be many cases in which we will know max and/or min values of distribution. Always **examine and visualize data** before transforming it.

**Question 2:** When can values outside of $`[0, 1]`$ range in test set cause issues?

**Solution:**
This might lead to an issue if our model performs any transformations on data that have a limited domain. Consider a model $`f(x) = \log(x)^T w`$. In this case if test datapoint have a value below 0, this code will fail, as log has domain $`[0, \infty)`$.

In general, after you visualize the data, think about what transforms are needed for it to be well behaved. Always pay attention to domains and ranges of each transform since these may lead to NaNs.

## 3. Linear Algebra Review Part 2

Let $`X \in \mathbb{R}^{m \times n}`$. $`X`$ may not have full rank. We explore properties about the four fundamental subspaces of $`X`$. Please refer to section 3 of the supplementary material for more info on matrix properties and subspaces.

### 3.1. Connections between subspaces of $`X`$

Check the following facts.

**(a)** The rowspace of $`X`$ is the columnspace of $`X^T`$, and vice versa.

**Solution:**
The matrix $`X^T`$ is
```math
\begin{pmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{pmatrix}
```
. The rows of $`X`$ are the columns of $`X^T`$, and vice versa.

**(b)** The nullspace of $`X`$ and the rowspace of $`X`$ are orthogonal complements. This can be written in shorthand as $`Null(X) = Range(X^T)^\perp`$. This is further equivalent to saying $`Range(X^T) = Null(X)^\perp`$.

**Solution:**
A vector $`v \in Null(X)`$ if and only if $`Xv = 0`$, which is true if and only if for every row $`X_i`$ of $`X`$, $`(X_i, v) = 0`$. This is precisely the condition that $`v`$ is perpendicular to each row of $`X`$, which is the stated claim.

**(c)** The nullspace of $`X^T`$ is orthogonal to the columnspace of $`X`$. This can be written in shorthand as $`Null(X^T) = Range(X)^\perp`$.

**Solution:**
This is seen by applying the previous result to $`X^T`$.

### 3.2. Linear algebra facts for linear regression

We saw in lecture on Linear Regression that the closed form expression for linear regression without an offset involves the term $`(X^T X)^{-1}`$.

**(a)** Is it true that the matrix $`X^T X`$ is always symmetric and positive semidefinite?

**Solution:**
Yes. Symmetry can be checked by computing the transpose. For any vector $`u`$, we have $`u^T X^T X u = \|X u\|^2 \geq 0`$.

**(b)** State and prove the connection between the nullspace of $`X`$ and the nullspace of $`X^T X`$. That is, your statement should look like one of the following: $`Null(X) \subseteq Null(X^T X)`$, or $`Null(X) \supseteq Null(X^T X)`$ or $`Null(X) = Null(X^T X)`$.

**Solution:**
We have, $`Null(X) = Null(X^T X)`$. Let $`v \in Null(X)`$. Then, one can check that $`X^T X v = 0`$, leading to $`v \in Null(X^T X)`$, which proves $`Null(X) \subseteq Null(X^T X)`$. For the other direction, let $`0 \neq v \in Null(X^T X)`$. Then, $`0 = v^T X^T X v = \|X v\|^2`$, which implies $`v \in Null(X)`$. Therefore, $`Null(X^T X) \subseteq Null(X)`$, which finishes the proof.

**(c)** Is it true that $`X^T X`$ is always invertible?

**Solution:**
No, this isn't always the case. Since $`Null(X) = Null(X^T X)`$ (see the answer to the previous question), the matrix $`X^T X`$ is not invertible if $`X`$ has a non-empty nullspace.

**(d)** Based on the above fact about the connection between the nullspaces of $`X`$ and $`X^T X`$ and the expression for linear regression without an offset (that we referred to two problems above), justify the use of "tall skinny" data matrix $`X`$ as opposed to a "short wide" matrix $`X`$.

**Solution:**
If $`X`$ is "short and wide", it has a non-empty nullspace. Therefore, $`X^T X`$ is not invertible.

**(e)** The columnspace and rowspace of $`X^T X`$ are the same, and are equal to the rowspace of $`X`$. (Hint: Use the relationship between nullspace and rowspace.)

**Solution:**
$`X^T X`$ is symmetric, and from the previous parts, we have $`Row(X^T X) = Col((X^T X)^T) = Col(X^T X)`$. By previous parts again, we have: $`Row(X^T X) = Null(X^T X)^\perp = Null(X)^\perp = Row(X)`$.


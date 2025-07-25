# Problem Set #1 Solutions: Math Review

We encourage you to solve each of the following problems to brush up on your linear algebra and probability.

We strongly suggest you use LaTex to work on problems (not ony is it helpful for Machine Learning, but it is a good skill to learn).

## 1. Gradients and Hessians

**Problem:**

Recall that a matrix $`A \in \mathbb{R}^{n \times n}`$ is **symmetric** if $`A^T = A`$, that is, $`A_{ij} = A_{ji}`$ for all $`i, j`$. Also recall the gradient $`\nabla f(x)`$ of a function $`f : \mathbb{R}^n \to \mathbb{R}`$, which is the $`n`$-vector of partial derivatives

```math
\nabla f(x) = \begin{bmatrix}
\frac{\partial}{\partial x_1} f(x) \\
\vdots \\
\frac{\partial}{\partial x_n} f(x)
\end{bmatrix}
\quad \text{where} \quad x = \begin{bmatrix}
x_1 \\
\vdots \\
x_n
\end{bmatrix}.
```

The Hessian $`\nabla^2 f(x)`$ of a function $`f : \mathbb{R}^n \to \mathbb{R}`$ is the $`n \times n`$ symmetric matrix of twice partial derivatives,

```math
\nabla^2 f(x) = \begin{bmatrix}
\frac{\partial^2}{\partial x_1^2} f(x) & \frac{\partial^2}{\partial x_1 \partial x_2} f(x) & \cdots & \frac{\partial^2}{\partial x_1 \partial x_n} f(x) \\
\frac{\partial^2}{\partial x_2 \partial x_1} f(x) & \frac{\partial^2}{\partial x_2^2} f(x) & \cdots & \frac{\partial^2}{\partial x_2 \partial x_n} f(x) \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2}{\partial x_n \partial x_1} f(x) & \frac{\partial^2}{\partial x_n \partial x_2} f(x) & \cdots & \frac{\partial^2}{\partial x_n^2} f(x)
\end{bmatrix}.
```

(a) Let $`f(x) = \frac{1}{2} x^T A x + b^T x`$, where $`A`$ is a symmetric matrix and $`b \in \mathbb{R}^n`$ is a vector. What is $`\nabla f(x)`$?

(b) Let $`f(x) = g(h(x))`$, where $`g : \mathbb{R} \to \mathbb{R}`$ is differentiable and $`h : \mathbb{R}^n \to \mathbb{R}`$ is differentiable. What is $`\nabla f(x)`$?

(c) Let $`f(x) = \frac{1}{2} x^T A x + b^T x`$, where $`A`$ is symmetric and $`b \in \mathbb{R}^n`$ is a vector. What is $`\nabla^2 f(x)`$?

(d) Let $`f(x) = g(a^T x)`$, where $`g : \mathbb{R} \to \mathbb{R}`$ is continuously differentiable and $`a \in \mathbb{R}^n`$ is a vector. What are $`\nabla f(x)`$ and $`\nabla^2 f(x)`$? *(Hint: your expression for $`\nabla^2 f(x)`$ may have as few as 11 symbols, including ' and parentheses.)*

**Solution:**

### (a) Gradient of Quadratic Function

**Key Concept:** This is a fundamental result in optimization - the gradient of a quadratic function. This form appears frequently in machine learning, particularly in linear regression, ridge regression, and many optimization problems.

**Step-by-step reasoning:**

1. **Break down the function:** $`f(x) = \frac{1}{2} x^T A x + b^T x`$ has two terms:
   - A quadratic term: $`\frac{1}{2} x^T A x`$
   - A linear term: $`b^T x`$

2. **Use known gradient formulas:**
   - For the quadratic term: $`\nabla_x (x^T A x) = (A + A^T)x`$
   - For the linear term: $`\nabla_x (b^T x) = b`$

3. **Apply symmetry condition:** Since $`A`$ is symmetric, $`A = A^T`$, so:
   - $`\nabla_x (x^T A x) = (A + A)x = 2Ax`$

4. **Combine results:**
```math
\nabla f(x) = \frac{1}{2} \cdot 2Ax + b = Ax + b
```

**Why this matters:** This result is crucial for gradient descent algorithms. When you have a quadratic objective function (common in least squares problems), the gradient is linear in $`x`$, making optimization much more tractable.

### (b) Chain Rule for Gradients

**Key Concept:** This is the multivariate chain rule, which is essential for understanding how gradients flow through composite functions - a fundamental concept in backpropagation for neural networks.

**Step-by-step reasoning:**

1. **Understand the composition:** $`f(x) = g(h(x))`$ means we have:
   - An inner function $`h: \mathbb{R}^n \to \mathbb{R}`$
   - An outer function $`g: \mathbb{R} \to \mathbb{R}`$

2. **Apply the chain rule:** For scalar-valued functions, the chain rule states:
```math
\nabla f(x) = g'(h(x)) \nabla h(x)
```

3. **Interpretation:** 
   - $`g'(h(x))`$ is the derivative of the outer function evaluated at $`h(x)`$
   - $`\nabla h(x)`$ is the gradient of the inner function
   - The result is a vector in $`\mathbb{R}^n`$

**Why this matters:** This is the foundation of automatic differentiation and backpropagation. In neural networks, you have many layers of functions, and this chain rule allows you to compute gradients efficiently by propagating them backward through the network.

### (c) Hessian of Quadratic Function

**Key Concept:** The Hessian gives us information about the curvature of the function, which is crucial for understanding optimization behavior and convergence rates.

**Step-by-step reasoning:**

1. **Recall Hessian definition:** The Hessian is the matrix of second derivatives:
```math
[\nabla^2 f(x)]_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
```

2. **Apply to each term:**
   - For the quadratic term: $`\nabla^2_x (x^T A x) = A + A^T`$
   - For the linear term: $`\nabla^2_x (b^T x) = 0`$ (since first derivatives are constants)

3. **Use symmetry:** Since $`A`$ is symmetric, $`A + A^T = 2A`$

4. **Combine with coefficient:** The $`\frac{1}{2}`$ factor cancels the 2:
```math
\nabla^2 f(x) = \frac{1}{2} \cdot 2A + 0 = A
```

**Why this matters:** 
- If $`A`$ is positive definite, the function is strictly convex and has a unique global minimum
- The condition number of $`A`$ determines how well gradient descent will converge
- This is why preconditioning (transforming the problem to make $`A`$ closer to identity) is important in optimization

### (d) Gradient and Hessian of Composition with Linear Function

**Key Concept:** This is a special case of the chain rule that appears frequently in machine learning, particularly in logistic regression, neural networks, and other models where you have a nonlinear function applied to a linear combination of inputs.

**Step-by-step reasoning:**

1. **Set up the problem:** Let $`y = a^T x`$ be the linear combination, so $`f(x) = g(y)`$

2. **Compute gradient using chain rule:**
   - $`\nabla_y g(y) = g'(y)`$ (scalar derivative)
   - $`\nabla_x y = \nabla_x (a^T x) = a`$ (gradient of linear function)
   - Therefore: $`\nabla f(x) = g'(a^T x) \cdot a`$

3. **Compute Hessian:**
   - The Hessian involves second derivatives of the composition
   - Using the chain rule for second derivatives: $`\nabla^2 f(x) = g''(a^T x) \cdot a a^T`$
   - This is because the second derivative of $`g`$ with respect to $`x`$ involves the outer product of the gradient of the inner function

**Why this matters:** 
- This form appears in logistic regression where $`g`$ is the sigmoid function
- The Hessian $`g''(a^T x) a a^T`$ has rank 1, which has important implications for optimization
- This explains why logistic regression can sometimes have convergence issues - the Hessian is not full rank

---

## 2. Positive definite matrices

**Problem:**

A matrix $`A \in \mathbb{R}^{n \times n}`$ is **positive semi-definite** (PSD), denoted $`A \succeq 0`$, if $`A = A^T`$ and $`x^T A x \geq 0`$ for all $`x \in \mathbb{R}^n`$. A matrix $`A`$ is **positive definite**, denoted $`A \succ 0`$, if $`A = A^T`$ and $`x^T A x > 0`$ for all $`x \neq 0`$, that is, all non-zero vectors $`x`$. The simplest example of a positive definite matrix is the identity $`I`$ (the diagonal matrix with 1s on the diagonal and 0s elsewhere), which satisfies $`x^T I x = \|x\|_2^2 = \sum_{i=1}^n x_i^2`$.

(a) Let $`z \in \mathbb{R}^n`$ be an $`n`$-vector. Show that $`A = zz^T`$ is positive semidefinite.

(b) Let $`z \in \mathbb{R}^n`$ be a *non-zero* $`n`$-vector. Let $`A = zz^T`$. What is the null-space of $`A`$? What is the rank of $`A`$?

(c) Let $`A \in \mathbb{R}^{n \times n}`$ be positive semidefinite and $`B \in \mathbb{R}^{m \times n}`$ be arbitrary, where $`m, n \in \mathbb{N}`$. Is $`BAB^T`$ PSD? If so, prove it. If not, give a counterexample with explicit $`A, B`$.

**Solution:**

### (a) Outer Product is Positive Semidefinite

**Key Concept:** Outer products $`zz^T`$ are fundamental building blocks in linear algebra and appear frequently in machine learning (e.g., in covariance matrices, projection matrices, and kernel methods).

**Step-by-step reasoning:**

1. **Understand the structure:** $`A = zz^T`$ is an $`n \times n`$ matrix where $`A_{ij} = z_i z_j`$

2. **Check symmetry:** $`A^T = (zz^T)^T = zz^T = A`$, so $`A`$ is symmetric

3. **Check positive semidefiniteness:** For any vector $`x \in \mathbb{R}^n`$:
```math
x^T A x = x^T (zz^T) x = (x^T z)(z^T x) = (z^T x)^2 \geq 0
```

   The last inequality holds because any real number squared is non-negative.

4. **Conclusion:** Since $`A`$ is symmetric and $`x^T A x \geq 0`$ for all $`x`$, $`A`$ is positive semidefinite.

**Why this matters:** 
- This is why sample covariance matrices $`\frac{1}{n}XX^T`$ are always PSD
- It explains why kernel matrices in kernel methods are PSD
- This property is crucial for ensuring that optimization problems have well-defined solutions

### (b) Properties of Outer Product Matrix

**Key Concept:** Understanding the rank and null space of outer products helps in understanding projection matrices, dimensionality reduction, and the geometry of linear transformations.

**Step-by-step reasoning:**

**Null-space analysis:**
1. **Definition:** The null space of $`A`$ is $`\{x : Ax = 0\}`$
2. **Solve:** $`Ax = 0 \implies zz^T x = 0 \implies z(z^T x) = 0`$
3. **Key insight:** Since $`z \neq 0`$, we must have $`z^T x = 0`$
4. **Conclusion:** The null space consists of all vectors orthogonal to $`z`$

**Rank analysis:**
1. **Recall:** Rank is the dimension of the column space
2. **Column space:** The columns of $`A = zz^T`$ are scalar multiples of $`z`$
3. **Conclusion:** Since all columns are proportional to $`z`$, the rank is 1

**Geometric interpretation:** 
- $`A`$ projects any vector $`x`$ onto the line spanned by $`z`$
- The projection is $`(z^T x)z`$
- Vectors orthogonal to $`z`$ are mapped to zero

**Why this matters:** 
- This explains why projection matrices have rank 1
- It's fundamental to understanding principal component analysis (PCA)
- The rank deficiency explains why some optimization problems can be ill-conditioned

### (c) Congruence Transformation Preserves PSD

**Key Concept:** This is a fundamental property that shows how positive semidefiniteness is preserved under certain transformations. This appears in many contexts, including:
- Change of variables in optimization
- Kernel methods and feature transformations
- Covariance matrix transformations

**Step-by-step reasoning:**

1. **Goal:** Show that $`BAB^T`$ is PSD if $`A`$ is PSD

2. **Check symmetry:** $`(BAB^T)^T = (B^T)^T A^T B^T = BAB^T`$ (since $`A`$ is symmetric)

3. **Check positive semidefiniteness:** For any vector $`y \in \mathbb{R}^m`$:
```math
y^T BAB^T y = (B^T y)^T A (B^T y)
```

4. **Key insight:** Let $`v = B^T y`$. Since $`A`$ is PSD, $`v^T A v \geq 0`$ for all $`v`$

5. **Conclusion:** Therefore, $`y^T BAB^T y \geq 0`$ for all $`y`$, so $`BAB^T`$ is PSD

**Why this matters:** 
- This property is crucial for kernel methods where you transform data via $`K = \Phi \Phi^T`$
- It explains why regularized covariance matrices remain PSD
- It's fundamental to understanding how linear transformations affect the geometry of optimization problems

---

## 3. Eigenvectors, eigenvalues, and the spectral theorem

**Problem:**

The eigenvalues of an $`n \times n`$ matrix $`A \in \mathbb{R}^{n \times n}`$ are the roots of the characteristic polynomial $`p_A(\lambda) = \det(\lambda I - A)`$, which may (in general) be complex. They are also defined as the values $`\lambda \in \mathbb{C}`$ for which there exists a vector $`x \in \mathbb{C}^n`$ such that $`Ax = \lambda x`$. We call such a pair $`(x, \lambda)`$ an *eigenvector, eigenvalue* pair. In this question, we use the notation $`\text{diag}(\lambda_1, \ldots, \lambda_n)`$ to denote the diagonal matrix with diagonal entries $`\lambda_1, \ldots, \lambda_n`$, that is,

```math
\text{diag}(\lambda_1, \ldots, \lambda_n) = \begin{bmatrix}
\lambda_1 & 0 & 0 & \cdots & 0 \\
0 & \lambda_2 & 0 & \cdots & 0 \\
0 & 0 & \lambda_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \lambda_n
\end{bmatrix}.
```

(a) Suppose that the matrix $`A \in \mathbb{R}^{n \times n}`$ is diagonalizable, that is, $`A = T \Lambda T^{-1}`$ for an invertible matrix $`T \in \mathbb{R}^{n \times n}`$, where $`\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)`$ is diagonal. Use the notation $`t^{(i)}`$ for the columns of $`T`$, so that $`T = [t^{(1)} \ \cdots \ t^{(n)}]`$, where $`t^{(i)} \in \mathbb{R}^n`$. Show that $`A t^{(i)} = \lambda_i t^{(i)}`$, so that the eigenvalues/eigenvector pairs of $`A`$ are $`(t^{(i)}, \lambda_i)`$.

A matrix $`U \in \mathbb{R}^{n \times n}`$ is orthogonal if $`U^T U = I`$. The spectral theorem, perhaps one of the most important theorems in linear algebra, states that if $`A \in \mathbb{R}^{n \times n}`$ is symmetric, that is, $`A = A^T`$, then $`A`$ is *diagonalizable by a real orthogonal matrix*. That is, there are a diagonal matrix $`\Lambda \in \mathbb{R}^{n \times n}`$ and orthogonal matrix $`U \in \mathbb{R}^{n \times n}`$ such that $`U^T A U = \Lambda`$, or, equivalently,

```math
A = U \Lambda U^T.
```

Let $`\lambda_i = \lambda_i(A)`$ denote the $`i`$-th eigenvalue of $`A`$.

(b) Let $`A`$ be symmetric. Show that if $`U = [u^{(1)} \ \cdots \ u^{(n)}]`$ is orthogonal, where $`u^{(i)} \in \mathbb{R}^n`$ and $`A = U \Lambda U^T`$, then $`u^{(i)}`$ is an eigenvector of $`A`$ and $`A u^{(i)} = \lambda_i u^{(i)}`$, where $`\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)`$.

(c) Show that if $`A`$ is PSD, then $`\lambda_i(A) \geq 0`$ for each $`i`$.

**Solution:**

### (a) Eigenvectors from Diagonalization

**Key Concept:** This connects the abstract concept of diagonalization with the concrete geometric interpretation of eigenvectors. This is fundamental to understanding how matrices act on vectors and why diagonalization is so powerful.

**Step-by-step reasoning:**

1. **Setup:** We have $`A = T \Lambda T^{-1}`$ where $`T = [t^{(1)} \cdots t^{(n)}]`$

2. **Key insight:** The columns of $`T`$ are the eigenvectors, and the diagonal entries of $`\Lambda`$ are the eigenvalues

3. **Use standard basis:** Let $`e_i`$ be the $`i`$-th standard basis vector (all zeros except 1 in position $`i`$)

4. **Compute $`A t^{(i)}`$:**
```math
A t^{(i)} = T \Lambda T^{-1} t^{(i)}
```

5. **Simplify using orthogonality:** Since $`T^{-1} T = I`$, we have $`T^{-1} t^{(i)} = e_i`$

6. **Apply diagonal matrix:** $`\Lambda e_i = \lambda_i e_i`$ (since $`\Lambda`$ is diagonal)

7. **Final step:** $`T (\lambda_i e_i) = \lambda_i T e_i = \lambda_i t^{(i)}`$

8. **Conclusion:** $`A t^{(i)} = \lambda_i t^{(i)}`$, so $`t^{(i)}`$ is an eigenvector with eigenvalue $`\lambda_i`$

**Why this matters:** 
- This explains why diagonalization is so useful: it decomposes a matrix into its fundamental building blocks
- It shows how to find eigenvectors once you have the diagonalization
- This is the foundation of many numerical algorithms for finding eigenvalues and eigenvectors

### (b) Spectral Theorem and Orthogonal Eigenvectors

**Key Concept:** The spectral theorem is one of the most important results in linear algebra. It tells us that symmetric matrices have a particularly nice structure - they can be diagonalized by orthogonal matrices, which means their eigenvectors form an orthonormal basis.

**Step-by-step reasoning:**

1. **Setup:** We have $`A = U \Lambda U^T`$ where $`U`$ is orthogonal

2. **Key property of orthogonal matrices:** $`U^T U = I`$, so $`U^T u^{(i)} = e_i`$

3. **Compute $`A u^{(i)}`$:**
```math
A u^{(i)} = U \Lambda U^T u^{(i)}
```

4. **Simplify:** $`U^T u^{(i)} = e_i`$ (the $`i`$-th standard basis vector)

5. **Apply diagonal matrix:** $`\Lambda e_i = \lambda_i e_i`$

6. **Final step:** $`U (\lambda_i e_i) = \lambda_i U e_i = \lambda_i u^{(i)}`$

7. **Conclusion:** $`A u^{(i)} = \lambda_i u^{(i)}`$, so $`u^{(i)}`$ is an eigenvector with eigenvalue $`\lambda_i`$

**Why this matters:** 
- This is why principal component analysis (PCA) works - the covariance matrix is symmetric, so it has orthogonal eigenvectors
- It explains why many optimization algorithms work well with symmetric matrices
- The orthogonality of eigenvectors makes many computations much simpler

### (c) Eigenvalues of Positive Semidefinite Matrices

**Key Concept:** This connects the geometric property of positive semidefiniteness (all quadratic forms are non-negative) with the algebraic property of eigenvalues (all eigenvalues are non-negative). This is crucial for understanding optimization and stability.

**Step-by-step reasoning:**

1. **Setup:** Let $`v`$ be an eigenvector of $`A`$ with eigenvalue $`\lambda`$

2. **Eigenvalue equation:** $`A v = \lambda v`$

3. **Multiply by $`v^T`$:** $`v^T A v = \lambda v^T v = \lambda \|v\|^2`$

4. **Key insight:** Since $`A`$ is PSD, $`v^T A v \geq 0`$ for all $`v`$

5. **For non-zero eigenvectors:** If $`v \neq 0`$, then $`\|v\|^2 > 0`$

6. **Conclusion:** Therefore, $`\lambda \geq 0`$

**Why this matters:**
- This explains why covariance matrices always have non-negative eigenvalues
- It's crucial for understanding why some optimization problems are well-posed
- It explains why some numerical algorithms (like Cholesky decomposition) only work for positive definite matrices
- This property is fundamental to understanding the geometry of quadratic forms and ellipsoids

---

## 4. Probability and multivariate Gaussians

**Problem:**

Suppose $`X = (X_1, \ldots, X_n)`$ is sampled from a multivariate Gaussian distribution with mean $`\mu`$ in $`\mathbb{R}^n`$ and covariance $`\Sigma`$ in $`S_+^n`$ (i.e. $`\Sigma`$ is positive semidefinite). This is commonly also written as $`X \sim \mathcal{N}(\mu, \Sigma)`$.

(a) Describe the random variable $`Y = X_1 + X_2 + \ldots + X_n`$. What is the mean and variance? Is this a well known distribution, and if so, which?

(b) Now, further suppose that $`\Sigma`$ is invertible. Find $`\mathbb{E}[X^T \Sigma^{-1} X]`$. (Hint: use the property of trace that $`x^T A x = \text{tr}(x^T A x)`$).

**Solution:**

### (a) Sum of Multivariate Gaussian Components

**Key Concept:** This is a fundamental result about how linear transformations of Gaussian random variables behave. This appears constantly in statistics and machine learning, particularly in:
- Linear combinations of features
- Aggregating multiple measurements
- Understanding how noise propagates through linear systems

**Step-by-step reasoning:**

1. **Understand the setup:** $`X \sim \mathcal{N}(\mu, \Sigma)`$ means:
   - Each component $`X_i`$ is marginally Gaussian
   - The components may be correlated (off-diagonal elements of $`\Sigma`$)

2. **Linear transformation:** $`Y = \sum_{i=1}^n X_i = \mathbf{1}^T X`$ where $`\mathbf{1}`$ is the vector of all ones

3. **Mean calculation:** 
```math
\mathbb{E}[Y] = \mathbb{E}[\sum_{i=1}^n X_i] = \sum_{i=1}^n \mathbb{E}[X_i] = \sum_{i=1}^n \mu_i = \mathbf{1}^T \mu
```

4. **Variance calculation:** 
```math
\text{Var}(Y) = \text{Var}(\sum_{i=1}^n X_i) = \sum_{i=1}^n \sum_{j=1}^n \text{Cov}(X_i, X_j) = \mathbf{1}^T \Sigma \mathbf{1}
```

5. **Distribution:** Since $`Y`$ is a linear combination of jointly Gaussian variables, it is also Gaussian

6. **Conclusion:** $`Y \sim \mathcal{N}(\mathbf{1}^T \mu, \mathbf{1}^T \Sigma \mathbf{1})`$

**Why this matters:** 
- This explains why sums of independent Gaussians are Gaussian
- It shows how correlations affect the variance of sums
- This is fundamental to understanding portfolio theory in finance
- It explains why many aggregate statistics are approximately Gaussian (Central Limit Theorem)

### (b) Expected Value of Quadratic Form

**Key Concept:** This is a fundamental result about the expected value of quadratic forms of Gaussian random variables. This appears in many contexts:
- Mahalanobis distance calculations
- Likelihood functions for Gaussian models
- Understanding the geometry of Gaussian distributions

**Step-by-step reasoning:**

1. **Setup:** We need to find $`\mathbb{E}[X^T \Sigma^{-1} X]`$ where $`X \sim \mathcal{N}(\mu, \Sigma)`$

2. **Key formula:** For any matrix $`A`$ and Gaussian $`X \sim \mathcal{N}(\mu, \Sigma)`$:

```math
\mathbb{E}[X^T A X] = \text{tr}(A \Sigma) + \mu^T A \mu
```

3. **Apply with $`A = \Sigma^{-1}`$:**
```math
\mathbb{E}[X^T \Sigma^{-1} X] = \text{tr}(\Sigma^{-1} \Sigma) + \mu^T \Sigma^{-1} \mu
```

4. **Simplify:** $`\text{tr}(\Sigma^{-1} \Sigma) = \text{tr}(I) = n`$

5. **Conclusion:** $`\mathbb{E}[X^T \Sigma^{-1} X] = n + \mu^T \Sigma^{-1} \mu`$

**Understanding the formula:**
- The term $`n`$ comes from the dimensionality of the space
- The term $`\mu^T \Sigma^{-1} \mu`$ is the squared Mahalanobis distance from the origin to the mean
- This formula shows how the expected value depends on both the dimension and the location of the mean relative to the covariance structure

**Why this matters:** 
- This is crucial for understanding maximum likelihood estimation for Gaussian models
- It explains why the chi-squared distribution appears in many statistical tests
- It's fundamental to understanding the geometry of multivariate normal distributions
- This result is used in many machine learning algorithms that involve Gaussian assumptions
# 1.2 The normal equations

Gradient descent gives one way of minimizing $J$. However, gradient descent requires iterative updates and careful tuning of the learning rate. In contrast, the normal equations approach allows us to directly solve for the optimal parameters in one step, provided the problem is well-posed and the necessary matrix inverses exist. This is especially useful for linear regression problems where the cost function is quadratic and differentiable. In this method, we will minimize $J$ by explicitly taking its derivatives with respect to the $\theta_j$'s, and setting them to zero. To enable us to do this without having to write reams of algebra and pages full of matrices of derivatives, let's introduce some notation for doing calculus with matrices.

## Why Normal Equations?

Before diving into the mathematics, let's understand why we might prefer normal equations over gradient descent:

**Advantages of Normal Equations:**
1. **Exact solution**: No need for iterative optimization
2. **No hyperparameters**: No learning rate to tune
3. **Guaranteed convergence**: Always finds the global minimum (if it exists)
4. **Theoretical insight**: Provides understanding of the optimal solution structure

**Disadvantages of Normal Equations:**
1. **Computational cost**: $O(n^3)$ for matrix inversion vs $O(n^2)$ per iteration for gradient descent
2. **Memory usage**: Requires storing $X^T X$ matrix
3. **Numerical instability**: Matrix inversion can be numerically unstable
4. **Non-invertible matrices**: Fails when $X^T X$ is singular

**When to use each method:**
- **Normal equations**: Small to medium datasets (< 10,000 examples), when you need exact solution
- **Gradient descent**: Large datasets, when approximate solution is acceptable, or when $X^T X$ is singular

## 1.2.1 Matrix derivatives

When working with functions that take matrices as inputs, we generalize the concept of derivatives to matrices. For a function $f : \mathbb{R}^{n \times d} \mapsto \mathbb{R}$ mapping from $n$-by-$d$ matrices to the real numbers, we define the derivative of $f$ with respect to $A$ to be:

$$
\nabla_A f(A) = \begin{bmatrix}
\frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1d}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial A_{n1}} & \cdots & \frac{\partial f}{\partial A_{nd}}
\end{bmatrix}
$$

### Understanding Matrix Derivatives

The gradient of a scalar-valued function with respect to a matrix is itself a matrix, where each entry is the partial derivative of the function with respect to the corresponding entry in the input matrix. This allows us to perform calculus operations in a compact, vectorized form, which is essential for efficient computation in machine learning.

**Key properties:**
1. **Linearity**: $\nabla_A (f(A) + g(A)) = \nabla_A f(A) + \nabla_A g(A)$
2. **Chain rule**: $\nabla_A f(g(A)) = \nabla_{g(A)} f(g(A)) \cdot \nabla_A g(A)$
3. **Transpose rule**: $\nabla_A f(A^T) = (\nabla_{A^T} f(A^T))^T$

### Example: Matrix Derivative Computation

For example, suppose 

$$
A = 
\begin{bmatrix} 
A_{11} & A_{12} \\ 
A_{21} & A_{22} 
\end{bmatrix}
$$

is a 2-by-2 matrix, and the function $f : \mathbb{R}^{2 \times 2} \mapsto \mathbb{R}$ is given by

$$
f(A) = \frac{3}{2}A_{11} + 5A_{12}^2 + A_{21}A_{22}.
$$

Here, $A_{ij}$ denotes the $(i, j)$ entry of the matrix $A$. We then have

$$
\nabla_A f(A) = \begin{bmatrix}
\frac{3}{2} & 10A_{12} \\
A_{22} & A_{21}
\end{bmatrix}.
$$

**Step-by-step computation:**
1. $\frac{\partial f}{\partial A_{11}} = \frac{3}{2}$ (derivative of $\frac{3}{2}A_{11}$)
2. $\frac{\partial f}{\partial A_{12}} = 10A_{12}$ (derivative of $5A_{12}^2$)
3. $\frac{\partial f}{\partial A_{21}} = A_{22}$ (derivative of $A_{21}A_{22}$ with respect to $A_{21}$)
4. $\frac{\partial f}{\partial A_{22}} = A_{21}$ (derivative of $A_{21}A_{22}$ with respect to $A_{22}$)

In this example, we see how to compute the gradient of a function with respect to a matrix. Each entry in the resulting gradient matrix is obtained by differentiating the function with respect to the corresponding entry in $A$. This process is analogous to taking partial derivatives with respect to each variable in multivariable calculus, but extended to matrices.

### Important Matrix Calculus Rules

For our derivation of the normal equations, we'll need these key rules:

1. **Linear term**: $\nabla_x (a^T x) = a$
2. **Quadratic term**: $\nabla_x (x^T A x) = 2A x$ (for symmetric matrix $A$)
3. **Transpose property**: $(A^T)^T = A$
4. **Matrix multiplication**: $(AB)^T = B^T A^T$

These rules will allow us to efficiently compute the gradient of our cost function.

## 1.2.2 Least squares revisited

Armed with the tools of matrix derivatives, let us now proceed to find in closed-form the value of $\theta$ that minimizes $J(\theta)$. We begin by re-writing $J$ in matrix-vectorial notation. The design matrix $X$ is a convenient way to represent all the input features of your training data in a single matrix. Each row corresponds to one training example, and each column corresponds to a feature (including the intercept term if present). This matrix formulation allows us to express the entire dataset and the linear model compactly, making it easier to apply linear algebra techniques.

### The Design Matrix

Given a training set, define the **design matrix** $X$ to be the `n-by-d` matrix (actually `n-by-(d+1)`, if we include the intercept term) that contains the training examples' input values in its rows:

$$
X = \begin{bmatrix}
--- (x^{(1)})^T --- \\
--- (x^{(2)})^T --- \\
\vdots \\
--- (x^{(n)})^T ---
\end{bmatrix}.
$$

**Understanding the design matrix:**
- **Rows**: Each row represents one training example
- **Columns**: Each column represents one feature
- **First column**: Usually all ones (intercept term)
- **Dimensions**: $n \times (d+1)$ where $n$ is number of examples, $d$ is number of features

**Example**: For our housing dataset with living area and bedrooms:
$$X = \begin{bmatrix}
1 & 2104 & 3 \\
1 & 1600 & 3 \\
1 & 2400 & 3 \\
\vdots & \vdots & \vdots
\end{bmatrix}$$

### The Target Vector

Also, let $\vec{y}$ be the $n$-dimensional vector containing all the target values from the training set:

$$
\vec{y} = \begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(n)}
\end{bmatrix}.
$$

The vector $\vec{y}$ stacks all the target values (labels) from the training set into a single column vector. This matches the structure of $X$, so that we can perform matrix operations between $X$ and $\vec{y}$.

**Example**: For our housing dataset:
$$\vec{y} = \begin{bmatrix}
400 \\
330 \\
369 \\
\vdots
\end{bmatrix}$$

### Vectorized Predictions and Errors

Now, since $h_\theta(x^{(i)}) = (x^{(i)})^T \theta$, we can easily verify that

$$
X\theta - \vec{y} = \begin{bmatrix}
(x^{(1)})^T \theta \\
\vdots \\
(x^{(n)})^T \theta
\end{bmatrix} - \begin{bmatrix}
y^{(1)} \\
\vdots \\
y^{(n)}
\end{bmatrix} = \begin{bmatrix}
h_\theta(x^{(1)}) - y^{(1)} \\
\vdots \\
h_\theta(x^{(n)}) - y^{(n)}
\end{bmatrix}.
$$

**Understanding this expression:**
- $X\theta$ computes all predictions at once using matrix multiplication
- $X\theta - \vec{y}$ gives the vector of prediction errors (residuals)
- This is much more efficient than computing each prediction individually

The expression $X\theta$ computes the predicted values for all training examples at once, using matrix multiplication. Subtracting $\vec{y}$ gives the vector of residuals (errors) for each example. This vectorized form is much more efficient than computing each prediction and error individually.

### Vectorized Cost Function

Thus, using the fact that for a vector $z$, we have that $z^T z = \sum_i z_i^2$:

$$
\frac{1}{2}(X\theta - \vec{y})^T (X\theta - \vec{y}) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2 = J(\theta)
$$

**Understanding the vectorized form:**
- $(X\theta - \vec{y})^T (X\theta - \vec{y})$ computes the dot product of the error vector with itself
- This is equivalent to summing the squared errors: $\sum_i (h_\theta(x^{(i)}) - y^{(i)})^2$
- The factor $\frac{1}{2}$ is included for mathematical convenience

The cost function $J(\theta)$ for linear regression is the mean squared error (up to a factor of $1/2$ for convenience in differentiation). The matrix form $\frac{1}{2}(X\theta - \vec{y})^T (X\theta - \vec{y})$ is equivalent to summing the squared errors for all training examples, but is more compact and enables efficient computation and differentiation.

### Computing the Gradient

Finally, to minimize $J$, let's find its derivatives with respect to $\theta$. Hence,

$$
\begin{align*}
\nabla_\theta J(\theta)
    &= \nabla_\theta \frac{1}{2}(X\theta - \vec{y})^T (X\theta - \vec{y}) \\
    &= \frac{1}{2} \nabla_\theta \left( (X\theta)^T X\theta - (X\theta)^T \vec{y} - \vec{y}^T X\theta + \vec{y}^T \vec{y} \right) \\
    &= \frac{1}{2} \nabla_\theta \left( \theta^T X^T X \theta - \vec{y}^T X\theta - \vec{y}^T X\theta \right) \\
    &= \frac{1}{2} \nabla_\theta \left( \theta^T X^T X \theta - 2 (X^T \vec{y})^T \theta \right) \\
    &= \frac{1}{2} \left( 2 X^T X \theta - 2 X^T \vec{y} \right) \\
    &= X^T X \theta - X^T \vec{y}
\end{align*}
$$

**Step-by-step derivation:**

#### Step 1: Expand the quadratic form
$(X\theta - \vec{y})^T (X\theta - \vec{y}) = (X\theta)^T X\theta - (X\theta)^T \vec{y} - \vec{y}^T X\theta + \vec{y}^T \vec{y}$

#### Step 2: Simplify using matrix properties
- $(X\theta)^T = \theta^T X^T$
- $(X\theta)^T \vec{y} = \vec{y}^T X\theta$ (since both are scalars)
- $\vec{y}^T \vec{y}$ is constant with respect to $\theta$

#### Step 3: Apply matrix calculus rules
- $\nabla_\theta (\theta^T X^T X \theta) = 2 X^T X \theta$ (quadratic term rule)
- $\nabla_\theta ((X^T \vec{y})^T \theta) = X^T \vec{y}$ (linear term rule)

#### Step 4: Simplify
The $\frac{1}{2}$ factor cancels the 2's, giving us the final result.

Here, we use properties of matrix calculus to differentiate the cost function with respect to $\theta$. The key steps involve expanding the quadratic form, applying the rules for differentiating with respect to vectors and matrices, and simplifying. The result is a linear equation in $\theta$.

In the third step, we used the fact that $a^T b = b^T a$, and in the fifth step used the facts $\nabla_x b^T x = b$ and $\nabla_x x^T A x = 2A x$ for symmetric matrix $A$ (for more details, see Section 4.3 of "Linear Algebra Review and Reference"). To minimize $J$, we set its derivatives to zero, and obtain the **normal equations**:

$$
X^T X \theta = X^T \vec{y}
$$

### Understanding the Normal Equations

Setting the gradient to zero gives us the condition for optimality. The resulting equation, called the normal equation, is a system of linear equations that can be solved directly for $\theta$ (provided $X^T X$ is invertible). This is the closed-form solution for linear regression.

**Geometric interpretation:**
- $X^T X$ is the Gram matrix, which measures the correlations between features
- $X^T \vec{y}$ is the correlation between features and target
- The normal equations say: "Find $\theta$ such that the predicted correlations match the actual correlations"

**When does this solution exist?**
- When $X^T X$ is invertible (i.e., when $X$ has full column rank)
- This means no feature is a perfect linear combination of other features
- If $X^T X$ is singular, we need regularization or use gradient descent

### The Closed-Form Solution

Thus, the value of $\theta$ that minimizes $J(\theta)$ is given in closed form by the equation

$$
\theta = (X^T X)^{-1} X^T \vec{y}
$$

**Understanding this formula:**
- $(X^T X)^{-1}$ is the inverse of the Gram matrix
- $X^T \vec{y}$ is the correlation between features and target
- The product gives us the optimal parameters

This formula gives the optimal parameters for linear regression in one step. It is derived from the normal equations and uses the inverse of $X^T X$. In practice, this approach is efficient for small to medium-sized datasets, but for very large datasets or when $X^T X$ is not invertible, iterative methods like gradient descent or regularization techniques are preferred.

### Computational Complexity

**Time complexity:**
- Matrix multiplication $X^T X$: $O(nd^2)$
- Matrix inversion $(X^T X)^{-1}$: $O(d^3)$
- Final multiplication: $O(d^2)$
- **Total**: $O(nd^2 + d^3)$

**Space complexity:**
- Storing $X^T X$: $O(d^2)$
- Storing $(X^T X)^{-1}$: $O(d^2)$
- **Total**: $O(d^2)$

**Comparison with gradient descent:**
- **Normal equations**: $O(nd^2 + d^3)$ one-time cost
- **Gradient descent**: $O(nd)$ per iteration, but many iterations needed

### Numerical Stability Considerations

**Potential issues:**
1. **Singular matrices**: $X^T X$ may not be invertible
2. **Ill-conditioned matrices**: Small changes in data cause large changes in solution
3. **Numerical precision**: Matrix inversion can amplify roundoff errors

**Solutions:**
1. **Regularization**: Add $\lambda I$ to $X^T X$ before inverting
2. **QR decomposition**: More numerically stable than direct inversion
3. **SVD decomposition**: Handles singular matrices gracefully

### Practical Implementation

**Direct implementation:**
```python
theta = np.linalg.inv(X.T @ X) @ X.T @ y
```

**More stable implementation:**
```python
theta = np.linalg.solve(X.T @ X, X.T @ y)
```

**With regularization:**
```python
lambda_reg = 0.01
theta = np.linalg.solve(X.T @ X + lambda_reg * np.eye(X.shape[1]), X.T @ y)
```

### Summary

The normal equations provide a beautiful closed-form solution to linear regression. They show us that:

1. **The optimal solution exists** when the features are linearly independent
2. **The solution can be computed directly** without iteration
3. **The solution has a clear geometric interpretation** in terms of correlations
4. **The method is efficient** for small to medium datasets

However, they also have limitations that make gradient descent preferable in many practical scenarios, especially with large datasets or when numerical stability is a concern.

---

**Previous: [LMS Algorithm](02_lms_algorithm.md)** - Learn about gradient descent and the LMS algorithm for optimizing the cost function.

**Next: [Probabilistic Interpretation](04_probabilistic_interpretation.md)** - Understand the probabilistic foundations of linear regression and maximum likelihood estimation.


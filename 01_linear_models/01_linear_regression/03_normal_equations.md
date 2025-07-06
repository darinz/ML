# 1.2 The normal equations

Gradient descent gives one way of minimizing $J$. However, gradient descent requires iterative updates and careful tuning of the learning rate. In contrast, the normal equations approach allows us to directly solve for the optimal parameters in one step, provided the problem is well-posed and the necessary matrix inverses exist. This is especially useful for linear regression problems where the cost function is quadratic and differentiable. In this method, we will minimize $J$ by explicitly taking its derivatives with respect to the $\theta_j$'s, and setting them to zero. To enable us to do this without having to write reams of algebra and pages full of matrices of derivatives, let's introduce some notation for doing calculus with matrices.

## 1.2.1 Matrix derivatives

When working with functions that take matrices as inputs, we generalize the concept of derivatives to matrices. For a function $f : \mathbb{R}^{n \times d} \mapsto \mathbb{R}$ mapping from $n$-by-$d$ matrices to the real numbers, we define the derivative of $f$ with respect to $A$ to be:

$$
\nabla_A f(A) = \begin{bmatrix}
\frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1d}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial A_{n1}} & \cdots & \frac{\partial f}{\partial A_{nd}}
\end{bmatrix}
$$

The gradient of a scalar-valued function with respect to a matrix is itself a matrix, where each entry is the partial derivative of the function with respect to the corresponding entry in the input matrix. This allows us to perform calculus operations in a compact, vectorized form, which is essential for efficient computation in machine learning.

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

In this example, we see how to compute the gradient of a function with respect to a matrix. Each entry in the resulting gradient matrix is obtained by differentiating the function with respect to the corresponding entry in $A$. This process is analogous to taking partial derivatives with respect to each variable in multivariable calculus, but extended to matrices.

**Python Example:**
```python
import numpy as np

def f(A):
    return 1.5 * A[0, 0] + 5 * A[0, 1] ** 2 + A[1, 0] * A[1, 1]

A = np.array([[1.0, 2.0], [3.0, 4.0]])
grad = np.zeros_like(A)
grad[0, 0] = 1.5
grad[0, 1] = 10 * A[0, 1]
grad[1, 0] = A[1, 1]
grad[1, 1] = A[1, 0]
print("Gradient of f at A:\n", grad)
```

By slowly letting the learning rate $\alpha$ decrease to zero as the algorithm runs, it is also possible to ensure that the parameters will converge to the global minimum rather than merely oscillate around the minimum.

## 1.2.2 Least squares revisited

Armed with the tools of matrix derivatives, let us now proceed to find in closed-form the value of $\theta$ that minimizes $J(\theta)$. We begin by re-writing $J$ in matrix-vectorial notation. The design matrix $X$ is a convenient way to represent all the input features of your training data in a single matrix. Each row corresponds to one training example, and each column corresponds to a feature (including the intercept term if present). This matrix formulation allows us to express the entire dataset and the linear model compactly, making it easier to apply linear algebra techniques.

Given a training set, define the **design matrix** $X$ to be the $n$-by-$d$ matrix (actually $n$-by-$(d+1)$, if we include the intercept term) that contains the training examples' input values in its rows:

$$
X = \begin{bmatrix}
--- (x^{(1)})^T --- \\
--- (x^{(2)})^T --- \\
\vdots \\
--- (x^{(n)})^T ---
\end{bmatrix}.
$$

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

The expression $X\theta$ computes the predicted values for all training examples at once, using matrix multiplication. Subtracting $\vec{y}$ gives the vector of residuals (errors) for each example. This vectorized form is much more efficient than computing each prediction and error individually.

**Python Example:**
```python
# X: n x d matrix, theta: d x 1 vector, y: n x 1 vector
X = np.array([[1, 2], [1, 3], [1, 4]])  # Example with intercept
theta = np.array([[0.5], [1.0]])
y = np.array([[2.5], [3.5], [4.5]])

predictions = X @ theta  # Matrix multiplication
residuals = predictions - y
print("Predictions:\n", predictions)
print("Residuals:\n", residuals)
```

Thus, using the fact that for a vector $z$, we have that $z^T z = \sum_i z_i^2$:

$$
\frac{1}{2}(X\theta - \vec{y})^T (X\theta - \vec{y}) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2 = J(\theta)
$$

The cost function $J(\theta)$ for linear regression is the mean squared error (up to a factor of $1/2$ for convenience in differentiation). The matrix form $\frac{1}{2}(X\theta - \vec{y})^T (X\theta - \vec{y})$ is equivalent to summing the squared errors for all training examples, but is more compact and enables efficient computation and differentiation.

**Python Example:**
```python
def cost_function(X, theta, y):
    residuals = X @ theta - y
    return 0.5 * np.sum(residuals ** 2)

J = cost_function(X, theta, y)
print("Cost J(theta):", J)
```

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

Here, we use properties of matrix calculus to differentiate the cost function with respect to $\theta$. The key steps involve expanding the quadratic form, applying the rules for differentiating with respect to vectors and matrices, and simplifying. The result is a linear equation in $\theta$.

In the third step, we used the fact that $a^T b = b^T a$, and in the fifth step used the facts $\nabla_x b^T x = b$ and $\nabla_x x^T A x = 2A x$ for symmetric matrix $A$ (for more details, see Section 4.3 of "Linear Algebra Review and Reference"). To minimize $J$, we set its derivatives to zero, and obtain the **normal equations**:

$$
X^T X \theta = X^T \vec{y}
$$

Setting the gradient to zero gives us the condition for optimality. The resulting equation, called the normal equation, is a system of linear equations that can be solved directly for $\theta$ (provided $X^T X$ is invertible). This is the closed-form solution for linear regression.

**Python Example:**
```python
# Normal equation solution for theta
XTX = X.T @ X
XTy = X.T @ y

theta_closed_form = np.linalg.inv(XTX) @ XTy
print("Closed-form theta:", theta_closed_form)
```

Thus, the value of $\theta$ that minimizes $J(\theta)$ is given in closed form by the equation

$$
\theta = (X^T X)^{-1} X^T \vec{y}
$$

This formula gives the optimal parameters for linear regression in one step. It is derived from the normal equations and uses the inverse of $X^T X$. In practice, this approach is efficient for small to medium-sized datasets, but for very large datasets or when $X^T X$ is not invertible, iterative methods like gradient descent or regularization techniques are preferred.


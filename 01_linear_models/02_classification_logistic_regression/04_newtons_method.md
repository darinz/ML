## 2.4 Newton's Method: Another algorithm for maximizing $\ell(\theta)$

Returning to logistic regression with $g(z)$ being the sigmoid function, let's now talk about a different algorithm for maximizing $\ell(\theta)$.

### 1. Newton's Method: Intuition and Geometric Interpretation

To get us started, let's consider Newton's method for finding a zero of a function. Specifically, suppose we have some function $f : \mathbb{R} \mapsto \mathbb{R}$, and we wish to find a value of $\theta$ so that $f(\theta) = 0$. Here, $\theta \in \mathbb{R}$ is a real number. Newton's method performs the following update:

```math
\theta := \theta - \frac{f(\theta)}{f'(\theta)}.
```

**Geometric intuition:** Newton's method approximates the function $f$ at the current guess $\theta$ by its tangent (a linear function). It then finds where this tangent crosses the $x$-axis (i.e., where it equals zero) and uses that as the next guess. This is typically much faster than simply taking a small step in the direction of the negative gradient (as in gradient descent), especially when close to the solution.

Here's a picture of the Newton's method in action:

<img src="./img/newtons_method.png" width="700px" />

In the leftmost figure, we see the function $f$ plotted along with the line $y = 0$. We're trying to find $\theta$ so that $f(\theta) = 0$; the value of $\theta$ that achieves this is about $1.3$. Suppose we initialized the algorithm with $\theta = 4.5$. Newton's method then fits a straight line tangent to $f$ at $\theta = 4.5$, and solves for where that line evaluates to $0$. (Middle figure.) This gives us the next guess for $\theta$, which is about $2.8$. The rightmost figure shows the result of running one more iteration, which then updates $\theta$ to about $1.8$. After a few more iterations, we rapidly approach $\theta = 1.3$.

### 2. Newton's Method for Maximization

Newton's method gives a way of getting to $f(\theta) = 0$. What if we want to use it to maximize some function $\ell$? The maxima of $\ell$ correspond to points where its first derivative $\ell'(\theta)$ is zero. So, by letting $f(\theta) = \ell'(\theta)$, we can use the same algorithm to maximize $\ell$, and we obtain the update rule:

```math
\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)}.
```

> **Something to think about:** How would this change if we wanted to use Newton's method to minimize rather than maximize a function?

**Minimization:** For minimization, the update is the same, but you want to ensure you are moving towards a minimum (where the second derivative is positive). In practice, the sign of the denominator (the curvature) determines whether you are at a minimum or maximum.

### 3. Advantages and Disadvantages of Newton's Method

**Advantages:**
- **Fast (quadratic) convergence** near the optimum, often requiring fewer iterations than gradient descent.
- **Takes curvature into account:** Uses second-order information (the Hessian), so steps are automatically scaled according to the local geometry of the function.

**Disadvantages:**
- **Computationally expensive:** Each iteration requires computing and inverting the Hessian matrix (second derivatives), which can be costly for high-dimensional problems.
- **Not always globally convergent:** If the initial guess is far from the optimum, Newton's method can diverge or get stuck at saddle points.
- **Requires the Hessian to be invertible:** If the Hessian is singular or nearly singular, the method may fail or become unstable.

### 4. Multidimensional Newton's Method and the Hessian

Lastly, in our logistic regression setting, $\theta$ is vector-valued, so we need to generalize Newton's method to this setting. The generalization of Newton's method to this multidimensional setting (also called the Newton-Raphson method) is given by

```math
\theta := \theta - H^{-1} \nabla_\theta \ell(\theta).
```

Here, $\nabla_\theta \ell(\theta)$ is, as usual, the vector of partial derivatives of $\ell(\theta)$ with respect to the $\theta_i$'s; and $H$ is an $d$-by-$d$ matrix (actually, $d+1$-by-$d+1$, assuming that we include the intercept term) called the **Hessian**, whose entries are given by

```math
H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j}.
```

**Intuition for the Hessian:** The Hessian captures the local curvature of the function. If the Hessian is positive definite, the function is locally convex and Newton's method will move towards a minimum. In logistic regression, the Hessian is typically positive semi-definite, ensuring stable updates.

### 5. Practical Tips and Fisher Scoring

Newton's method typically enjoys faster convergence than (batch) gradient descent, and requires many fewer iterations to get very close to the minimum. One iteration of Newton's can, however, be more expensive than one iteration of gradient descent, since it requires finding and inverting an $d$-by-$d$ Hessian; but so long as $d$ is not too large, it is usually much faster overall.

- **When to use Newton's method:**
  - When the number of parameters is moderate (so the Hessian is not too large to invert).
  - When you need fast, high-precision convergence.
  - When second derivatives are easy to compute (as in logistic regression).
- **Regularization:** Adding a small multiple of the identity matrix to the Hessian (i.e., $H + \lambda I$) can help if the Hessian is nearly singular.
- **Fisher scoring:** When Newton's method is applied to maximize the logistic regression log likelihood function $\ell(\theta)$, the resulting method is also called **Fisher scoring**. In Fisher scoring, the Hessian is replaced by its expected value (the Fisher information matrix), which can improve stability.

### 6. Newton's Method vs. Gradient Descent: Comparison Table

| Feature                | Newton's Method                | Gradient Descent           |
|------------------------|-------------------------------|----------------------------|
| Uses second derivatives| Yes (Hessian)                 | No (only gradient)         |
| Convergence rate       | Quadratic (very fast near opt.)| Linear (slower)            |
| Per-iteration cost     | High (Hessian + inversion)    | Low                        |
| Step size              | Adaptive (curvature-aware)    | Fixed or scheduled         |
| Global convergence     | Not guaranteed                | More robust                |
| Best for               | Small/medium $d$, high accuracy| Large $d$, cheap iterations|

**Summary:** Newton's method is a powerful optimization technique that leverages curvature information for rapid convergence, especially in problems like logistic regression where the Hessian is tractable. However, its computational cost can be prohibitive for very high-dimensional problems, where gradient descent or quasi-Newton methods (like BFGS) may be preferable.

### 7. Python Code Examples for Newton's Method

Below are Python code snippets that implement the key equations and calculations for Newton's method, both in 1D and in the multidimensional case (as used in logistic regression). These examples use NumPy for efficient computation.

#### Newton's Method in 1D

```python
import numpy as np

def newton_1d(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for finding a root of f(x) = 0 in 1D.
    Args:
        f: function f(x)
        df: derivative f'(x)
        x0: initial guess
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    Returns:
        x: estimated root
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ValueError("Derivative too small.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x
```

#### Newton's Method for Maximizing a Function in 1D

```python
def newton_maximize_1d(l, dl, ddl, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for maximizing l(x) in 1D.
    Args:
        l: function l(x)
        dl: first derivative l'(x)
        ddl: second derivative l''(x)
        x0: initial guess
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    Returns:
        x: estimated maximizer
    """
    x = x0
    for i in range(max_iter):
        grad = dl(x)
        hess = ddl(x)
        if abs(hess) < 1e-12:
            raise ValueError("Second derivative too small.")
        x_new = x - grad / hess
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x
```

#### Newton's Method for Logistic Regression (Multidimensional Case)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(theta, X, y):
    h = sigmoid(X @ theta)
    return np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient(theta, X, y):
    h = sigmoid(X @ theta)
    return X.T @ (y - h)

def hessian(theta, X):
    h = sigmoid(X @ theta)
    D = np.diag(h * (1 - h))
    return -X.T @ D @ X

def newton_logistic_regression(X, y, tol=1e-6, max_iter=20):
    """
    Newton's method for logistic regression.
    Args:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    Returns:
        theta: estimated parameters
    """
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    for i in range(max_iter):
        grad = gradient(theta, X, y)
        H = hessian(theta, X)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            # Add regularization if Hessian is singular
            H_reg = H - 1e-4 * np.eye(n_features)
            delta = np.linalg.solve(H_reg, grad)
        theta_new = theta - delta
        if np.linalg.norm(theta_new - theta) < tol:
            return theta_new
        theta = theta_new
    return theta
```

**Usage Example:**
```python
# X: (n_samples, n_features), y: (n_samples,)
# Add intercept term to X if needed
# theta = newton_logistic_regression(X, y)
```

These code snippets demonstrate how to implement Newton's method for root finding, maximization, and logistic regression. For large-scale problems, consider using quasi-Newton methods (like BFGS) or stochastic optimization for efficiency.

# 6.6 Optimal margin classifiers: the dual form

> **Note:** _The equivalence of optimization problem (6.8) and the optimization problem (6.12), and the relationship between the primary and dual variables in equation (6.10) are the most important take home messages of this section._

---
**Intuition:**
The dual form of the SVM optimization problem is not just a mathematical curiosity—it is the key to unlocking the power of SVMs in high-dimensional and non-linear settings. By expressing the problem in terms of inner products, we can later use the "kernel trick" to implicitly map data into much higher-dimensional spaces without ever computing the mapping explicitly. This is what allows SVMs to be so effective for complex, non-linear classification tasks.

Previously, we posed the following (primal) optimization problem for finding the optimal margin classifier:

```math
\begin{align}
\min_{w, b} \quad & \frac{1}{2} \|w\|^2 \\
\text{s.t.} \quad & y^{(i)} (w^T x^{(i)} + b) \geq 1, \quad i = 1, \ldots, n
\end{align}
```

```python
import numpy as np
from scipy.optimize import minimize

def primal_svm_objective(params, X, y):
    """
    Primal SVM objective function
    params: [w, b] where w is the weight vector and b is the bias
    """
    n_features = X.shape[1]
    w = params[:n_features]
    b = params[n_features]
    
    # Objective: (1/2) * ||w||^2
    objective = 0.5 * np.dot(w, w)
    return objective

def primal_svm_constraints(params, X, y):
    """
    Primal SVM constraints: y[i] * (w^T * x[i] + b) >= 1
    Returns negative values for violated constraints
    """
    n_features = X.shape[1]
    w = params[:n_features]
    b = params[n_features]
    
    # Constraints: y[i] * (w^T * x[i] + b) >= 1
    constraints = y * (np.dot(X, w) + b) - 1
    return constraints

def solve_primal_svm(X, y):
    """
    Solve the primal SVM optimization problem
    """
    n_samples, n_features = X.shape
    
    # Initial guess
    w0 = np.zeros(n_features)
    b0 = 0
    params0 = np.concatenate([w0, [b0]])
    
    # Define constraints
    constraints = {
        'type': 'ineq',
        'fun': lambda params: primal_svm_constraints(params, X, y)
    }
    
    # Solve optimization problem
    result = minimize(
        primal_svm_objective, 
        params0, 
        args=(X, y),
        constraints=constraints,
        method='SLSQP'
    )
    
    if result.success:
        w_opt = result.x[:n_features]
        b_opt = result.x[n_features]
        return w_opt, b_opt
    else:
        raise ValueError("Optimization failed")
```

_(6.8)_

We can write the constraints as

```math
g_i(w) = -y^{(i)} (w^T x^{(i)} + b) + 1 \leq 0.
```

```python
def constraint_g(w, b, x_i, y_i):
    """
    Individual constraint function g_i(w, b) = -y_i * (w^T * x_i + b) + 1 <= 0
    """
    return -y_i * (np.dot(w, x_i) + b) + 1

def all_constraints(w, b, X, y):
    """
    All constraint functions for the dataset
    """
    return np.array([constraint_g(w, b, X[i], y[i]) for i in range(len(y))])
```

The points with the smallest margins are exactly the ones closest to the decision boundary; here, these are the three points (one negative and two positive examples) that lie on the dashed lines parallel to the decision boundary. Thus, only three of the $\alpha_i$'s—namely, the ones corresponding to these three training examples—will be non-zero at the optimal solution to our optimization problem. These three points are called the **support vectors** in this problem. The fact that the number of support vectors can be much smaller than the size the training set will be useful later.

---
**Remark:**
In practice, this means that most training points do not affect the final classifier at all! Only the support vectors matter, which makes SVMs very efficient at prediction time.

Let's move on. Looking ahead, as we develop the dual form of the problem, one key idea to watch out for is that we'll try to write our algorithm in terms of only the inner product $\langle x^{(i)}, x^{(j)} \rangle$ (think of this as $x^{(i)T} x^{(j)}$) between points in the input feature space. The fact that we can express our algorithm in terms of these inner products will be key when we apply the kernel trick.

When we construct the Lagrangian for our optimization problem we have:

```math
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^n \alpha_i \left[ y^{(i)} (w^T x^{(i)} + b) - 1 \right].
```

```python
def lagrangian(w, b, alpha, X, y):
    """
    Lagrangian function for SVM
    L(w, b, α) = (1/2) * ||w||^2 - sum(α_i * [y_i * (w^T * x_i + b) - 1])
    """
    # First term: (1/2) * ||w||^2
    first_term = 0.5 * np.dot(w, w)
    
    # Second term: -sum(α_i * [y_i * (w^T * x_i + b) - 1])
    second_term = 0
    for i in range(len(y)):
        constraint = y[i] * (np.dot(w, X[i]) + b) - 1
        second_term -= alpha[i] * constraint
    
    return first_term + second_term
```

_(6.9)_

Note that there's only $\alpha_i$ but no $\alpha_i^*$ Lagrange multipliers, since the problem has only inequality constraints.

---
**Intuition:**
The Lagrangian formulation allows us to incorporate the constraints directly into the objective function, paving the way for duality theory and the use of KKT conditions.

Let's find the dual form of the problem. To do so, we need to first minimize $\mathcal{L}(w, b, \alpha)$ with respect to $w$ and $b$ (for fixed $\alpha$), to get $\theta_D$, which we'll do by setting the derivatives of $\mathcal{L}$ with respect to $w$ and $b$ to zero. We have:

```math
\nabla_w \mathcal{L}(w, b, \alpha) = w - \sum_{i=1}^n \alpha_i y^{(i)} x^{(i)} = 0
```

This implies that

```math
w = \sum_{i=1}^n \alpha_i y^{(i)} x^{(i)}.
```

```python
def compute_w_from_alpha(alpha, X, y):
    """
    Compute w from alpha using equation (6.10)
    w = sum(α_i * y_i * x_i)
    """
    w = np.zeros(X.shape[1])
    for i in range(len(y)):
        w += alpha[i] * y[i] * X[i]
    return w

# Vectorized version
def compute_w_from_alpha_vectorized(alpha, X, y):
    """
    Vectorized computation of w from alpha
    """
    return np.sum((alpha * y)[:, None] * X, axis=0)
```

_(6.10)_

As for the derivative with respect to $b$, we obtain

```math
\frac{\partial}{\partial b} \mathcal{L}(w, b, \alpha) = \sum_{i=1}^n \alpha_i y^{(i)} = 0.
```

```python
def derivative_b_lagrangian(alpha, y):
    """
    Derivative of Lagrangian with respect to b
    ∂L/∂b = sum(α_i * y_i) = 0
    """
    return np.sum(alpha * y)

# Constraint function for optimization
def alpha_constraint(alpha, y):
    """
    Constraint: sum(α_i * y_i) = 0
    """
    return np.sum(alpha * y)
```

_(6.11)_

If we take the definition of $w$ in Equation (6.10) and plug that back into the Lagrangian (Equation 6.9), and simplify, we get

```math
\mathcal{L}(w, b, \alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n y^{(i)} y^{(j)} \alpha_i \alpha_j (x^{(i)T} x^{(j)}) - b \sum_{i=1}^n \alpha_i y^{(i)}.
```

But from Equation (6.11), the last term must be zero, so we obtain

```math
\mathcal{L}(w, b, \alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n y^{(i)} y^{(j)} \alpha_i \alpha_j (x^{(i)T} x^{(j)}).
```

```python
def dual_objective(alpha, X, y):
    """
    Dual objective function W(α)
    W(α) = sum(α_i) - (1/2) * sum(sum(α_i * α_j * y_i * y_j * <x_i, x_j>))
    """
    n_samples = len(y)
    
    # First term: sum(α_i)
    first_term = np.sum(alpha)
    
    # Second term: (1/2) * sum(sum(α_i * α_j * y_i * y_j * <x_i, x_j>))
    second_term = 0
    for i in range(n_samples):
        for j in range(n_samples):
            inner_product = np.dot(X[i], X[j])
            second_term += alpha[i] * alpha[j] * y[i] * y[j] * inner_product
    second_term *= 0.5
    
    return first_term - second_term

# Vectorized version (more efficient)
def dual_objective_vectorized(alpha, X, y):
    """
    Vectorized dual objective function
    """
    # Compute kernel matrix K[i,j] = <x_i, x_j>
    K = np.dot(X, X.T)
    
    # Compute y[i] * y[j] * K[i,j]
    yyK = np.outer(y, y) * K
    
    # Compute objective
    first_term = np.sum(alpha)
    second_term = 0.5 * np.dot(alpha, np.dot(yyK, alpha))
    
    return first_term - second_term
```

Recall that we got to the equation above by minimizing $\mathcal{L}$ with respect to $w$ and $b$. Putting this together with the constraints $\alpha_i \geq 0$ (that we always had) and the constraint (6.11), we obtain the following dual optimization problem:

```math
\begin{align}
\max_{\alpha} \quad & W(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \\
\text{s.t.} \quad & \alpha_i \geq 0, \quad i = 1, \ldots, n \\
& \sum_{i=1}^n \alpha_i y^{(i)} = 0,
\end{align}
```

```python
def solve_dual_svm(X, y):
    """
    Solve the dual SVM optimization problem
    """
    n_samples = len(y)
    
    # Initial guess for alpha
    alpha0 = np.zeros(n_samples)
    
    # Define constraints
    # 1. α_i >= 0 for all i
    bounds = [(0, None) for _ in range(n_samples)]
    
    # 2. sum(α_i * y_i) = 0
    constraints = {
        'type': 'eq',
        'fun': lambda alpha: np.sum(alpha * y)
    }
    
    # Solve optimization problem (maximize dual objective)
    result = minimize(
        lambda alpha: -dual_objective_vectorized(alpha, X, y),  # Negative for maximization
        alpha0,
        constraints=constraints,
        bounds=bounds,
        method='SLSQP'
    )
    
    if result.success:
        alpha_opt = result.x
        w_opt = compute_w_from_alpha_vectorized(alpha_opt, X, y)
        b_opt = compute_b_from_support_vectors(alpha_opt, X, y, w_opt)
        return w_opt, b_opt, alpha_opt
    else:
        raise ValueError("Dual optimization failed")
```

_(6.12)_

---
**Why the dual?**
- The dual problem is often easier to solve, especially when the number of constraints is smaller than the number of variables.
- The dual variables $\alpha_i$ have a direct interpretation: they measure how "important" each training point is for defining the decision boundary.
- The dual form is the gateway to using kernels, which allow SVMs to learn non-linear boundaries efficiently.

You should also be able to verify that the conditions required for $p^\ast = d^\ast$ and the KKT conditions (Equations 6.3–6.7) to hold are indeed satisfied in our optimization problem. Hence, we can solve the dual in lieu of solving the primal problem. Specifically, in the dual problem above, we have a maximization problem in which the parameters are the $\alpha_i$'s. We'll talk later about the specific algorithm that we're going to use to solve the dual problem, but if we are indeed able to solve it (i.e., find the $\alpha$'s that maximize $W(\alpha)$ subject to the constraints), then we can use Equation (6.10) to go back and find the optimal $w$'s as a function of the $\alpha$'s. Having found $w^\ast$, by considering the primal problem, it is also straightforward to find the optimal value for the intercept term $b$ as

```math
b^* = \frac{-\max_{i: y^{(i)} = -1} w^{*T} x^{(i)} + \min_{i: y^{(i)} = 1} w^{*T} x^{(i)}}{2}
```

```python
def compute_b_from_support_vectors(alpha, X, y, w):
    """
    Compute optimal b from support vectors using equation (6.13)
    b* = (-max_{i:y_i=-1} w*^T x_i + min_{i:y_i=1} w*^T x_i) / 2
    """
    # Compute w^T * x for all points
    wx = np.dot(X, w)
    
    # Find negative and positive examples
    neg_idx = np.where(y == -1)[0]
    pos_idx = np.where(y == 1)[0]
    
    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise ValueError("Need both positive and negative examples")
    
    # Compute max for negative examples and min for positive examples
    max_neg = np.max(wx[neg_idx])
    min_pos = np.min(wx[pos_idx])
    
    # Compute b
    b = (-max_neg + min_pos) / 2
    return b

def get_support_vectors(alpha, X, y, tolerance=1e-5):
    """
    Identify support vectors (points with α_i > tolerance)
    """
    support_vector_indices = np.where(alpha > tolerance)[0]
    return support_vector_indices, X[support_vector_indices], y[support_vector_indices]
```

_(6.13)_

---
**Practical note:**
In real SVM implementations, the value of $b$ is often computed using only the support vectors, as these are the points for which the margin is exactly 1.

Before moving on, let's also take a more careful look at Equation (6.10), which gives the optimal value of $w$ in terms of (the optimal value of) $\alpha$. Suppose we've fit our model's parameters to a training set, and now wish to make a prediction at a new point input $x$. We would then calculate $w^T x + b$, and predict $y = 1$ if and only if this quantity is bigger than zero. But using (6.10), this quantity can also be written:

```math
w^T x + b = \left( \sum_{i=1}^n \alpha_i y^{(i)} x^{(i)} \right)^T x + b
```

```python
def svm_decision_function(x, X, y, alpha, b):
    """
    Compute decision function using equation (6.14)
    f(x) = w^T * x + b = sum(α_i * y_i * <x_i, x>) + b
    """
    # Compute inner products between x and all training points
    inner_products = np.dot(X, x)
    
    # Compute decision function
    decision = np.sum(alpha * y * inner_products) + b
    return decision

def svm_predict(x, X, y, alpha, b):
    """
    Make prediction for a new point x
    """
    decision = svm_decision_function(x, X, y, alpha, b)
    return 1 if decision > 0 else -1
```

_(6.14)_

```math
= \sum_{i=1}^n \alpha_i y^{(i)} \langle x^{(i)}, x \rangle + b.
```

```python
def svm_decision_function_kernel(x, X, y, alpha, b, kernel_func=None):
    """
    Compute decision function using kernel trick (equation 6.15)
    f(x) = sum(α_i * y_i * K(x_i, x)) + b
    """
    if kernel_func is None:
        # Linear kernel: K(x_i, x) = <x_i, x>
        kernel_values = np.dot(X, x)
    else:
        # Custom kernel function
        kernel_values = np.array([kernel_func(X[i], x) for i in range(len(X))])
    
    # Compute decision function
    decision = np.sum(alpha * y * kernel_values) + b
    return decision

def svm_predict_kernel(x, X, y, alpha, b, kernel_func=None):
    """
    Make prediction using kernel trick
    """
    decision = svm_decision_function_kernel(x, X, y, alpha, b, kernel_func)
    return 1 if decision > 0 else -1

# Example kernel functions
def linear_kernel(x1, x2):
    """Linear kernel: K(x1, x2) = <x1, x2>"""
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=2, gamma=1.0, coef0=0):
    """Polynomial kernel: K(x1, x2) = (γ * <x1, x2> + r)^d"""
    return (gamma * np.dot(x1, x2) + coef0) ** degree

def rbf_kernel(x1, x2, gamma=1.0):
    """RBF kernel: K(x1, x2) = exp(-γ * ||x1 - x2||^2)"""
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))
```

_(6.15)_

Hence, if we've found the $\alpha_i$'s, in order to make a prediction, we have to calculate a quantity that depends only on the inner product between $x$ and the points in the training set. Moreover, we saw earlier that the $\alpha_i$'s will all be zero except for the support vectors. Thus, many of the terms in the sum above will be zero, and we really need to find only the inner products between $x$ and the support vectors (of which there is often only a small number) in order calculate (6.15) and make our prediction.

```python
def svm_decision_function_support_vectors_only(x, X_sv, y_sv, alpha_sv, b, kernel_func=None):
    """
    Compute decision function using only support vectors (more efficient)
    """
    if kernel_func is None:
        # Linear kernel
        kernel_values = np.dot(X_sv, x)
    else:
        # Custom kernel function
        kernel_values = np.array([kernel_func(X_sv[i], x) for i in range(len(X_sv))])
    
    # Compute decision function using only support vectors
    decision = np.sum(alpha_sv * y_sv * kernel_values) + b
    return decision

def complete_svm_implementation(X, y, kernel_func=None):
    """
    Complete SVM implementation from training to prediction
    """
    # Step 1: Solve dual problem
    w_opt, b_opt, alpha_opt = solve_dual_svm(X, y)
    
    # Step 2: Identify support vectors
    sv_indices, X_sv, y_sv = get_support_vectors(alpha_opt, X, y)
    alpha_sv = alpha_opt[sv_indices]
    
    # Step 3: Create prediction function
    def predict(x):
        return svm_decision_function_support_vectors_only(
            x, X_sv, y_sv, alpha_sv, b_opt, kernel_func
        )
    
    return {
        'w': w_opt,
        'b': b_opt,
        'alpha': alpha_opt,
        'support_vectors': X_sv,
        'support_vector_labels': y_sv,
        'support_vector_alphas': alpha_sv,
        'predict': predict
    }
```

---
**Key insight:**
This is the heart of the kernel trick: if we can compute $\langle x^{(i)}, x \rangle$ efficiently (or replace it with a kernel function $K(x^{(i)}, x)$), we can use SVMs for highly non-linear classification tasks without ever explicitly mapping data to a high-dimensional space.

By examining the dual form of the optimization problem, we gained significant insight into the structure of the problem, and were also able to write the entire algorithm in terms of only inner products between input feature vectors. In the next section, we will exploit this property to apply the kernels to our classification problem. The resulting algorithm, **support vector machines**, will be able to efficiently learn in very high dimensional spaces.
# Problem Set 2

## 1. K-fold Cross-Validation

Implement K-fold Cross-Validation in Python.

## 2. Lasso and CV

Implement Lasso and CV in Python.

## 3. Subgradients

We start with the definition of subgradients before discussing the motivation and its usefulness.
**Definition 1 (subgradients).** A vector $`g \in \mathbb{R}^d`$ is a subgradient of a convex function $`f: D \to \mathbb{R}`$ at $`x \in D \subseteq \mathbb{R}^d`$ if
```math
f(y) \ge f(x) + g^T(y-x) \quad \text{for all } y \in D.
```
One interpretation of subgradient $`g`$ is that the affine function (of $`y`$) $`f(x) + g^T(y-x)`$ is a global underestimator of $`f`$. Note that if a convex function $`f`$ is differentiable at $`x`$ (i.e., $`\nabla f(x)`$ exists), then $`f(y) \ge f(x) + \nabla f(x)^T (y - x)`$ is true for all $`y \in D`$, meaning that $`\nabla f(x)`$ is a subgradient of $`f`$ at $`x`$. But a subgradient can exist even when $`f`$ is not differentiable at $`x``.

### (a) Why are subgradients useful in optimization? If $`g = 0`$ is a subgradient of a function $`f`$ at $`x^*`$, what does it imply?

### (b) What are the subgradients of $`f(x) = \max(x, x^2)`$ at $`0`$, with $`x \in \mathbb{R}`$? (Hint: draw a picture and note that subgradients at a point might not be unique)

### (c) Some important results about subgradients are
*   If $`f`$ is convex, then a subgradient of $`f`$ at $`x \in \text{int}(D)`$ (interior of the domain of $`f`$) always exists.
*   If $`f`$ is convex, then $`f`$ is differentiable at $`x`$ if and only if $`\nabla f(x)`$ is the only subgradient of $`f`$ at $`x`$.
*   A point $`x^*`$ is a global minimizer of a function $`f`$ (not necessarily convex) if and only if $`g = 0`$ is a subgradient of $`f`$ at $`x^*`$.

## 4. Convexity

Convexity is defined for both sets and functions. For today we'll focus on discussing the convexity of functions.
**Definition 2 (convex functions).** A function $`f: \mathbb{R}^d \to \mathbb{R}`$ is convex on a set $`A`$ if for all $`x, y \in A`$ and $`\lambda \in [0, 1]`$:
```math
f(\lambda x + (1 - \lambda)y) \le \lambda f(x) + (1 - \lambda)f(y)
```

When this definition holds with the inequality being reversed, then $`f`$ is said to be concave. From the definition, it is clear that a function $`f`$ is convex if and only if $`-f`$ is concave.

### (a) Why do we care whether a function is convex or not?

### (b) Which of the following functions are convex? (Hint: draw a picture!)
(i) $`|x|`$
(ii) $`\cos(x)`$
(iii) $`x^T x`$

### (c) Can a function be both convex and concave on the same set? If so, give an example. If not, describe why not.

## 5. Practical Methods for Checking Convexity

Using the definition of convexity can be tedious. Here are several basic methods:
*   For a differentiable function, examine if $`f(y) \ge f(x) + \nabla f(x)^T (y-x)`$ for any $`x, y`$ in the domain of $`f`$.
*   For twice differentiable functions, examine if $`\nabla^2 f(x) \ge 0`$ (i.e., the Hessian matrix is positive semidefinite).
*   Nonnegative weighted sum.
*   Composition with affine function.
*   Pointwise maximum and supremum.

Note: More such methods are covered in convex optimization courses or textbooks.

### (a) Prove Jensen's inequality using the characterization of convexity for differentiable functions.

If $`f`$ is differentiable, then $`f`$ is convex if and only if $`f(y) \ge f(x) + \nabla f(x)^T (y-x)`$ for any $`x, y`$ in the domain of $`f`$. A geometric interpretation is that any tangent plane of a convex function $`f`$ must lie entirely below $`f`$. Jensen's inequality states that $`E f(X) \ge f(E(X))`$ when $`f`$ is convex. Prove Jensen's inequality using the other inequality mentioned.

### (b) Show that the objective function in linear regression is convex using the Hessian method.

If $`f`$ is twice differentiable with a convex domain, then $`f`$ is convex if and only if $`\nabla^2 f(x) \ge 0`$ for any $`x`$ in the domain of $`f`$. Use this method to show that the objective function in linear regression is convex.

### (c) Let $`\alpha \ge 0`$ and $`\beta \ge 0`$, and if $`f`$ and $`g`$ are convex, then $`\alpha f`$, $`f + g`$, $`\alpha f + \beta g`$ are all convex.

One application is that when a (possibly complicated) objective function can be expressed as a sum (e.g., the negative log-likelihood function), then showing the convexity of each individual term is typically easier.

### (d) Suppose $`f(\cdot)`$ is convex, then $`g(x) := f(Ax + b)`$ is convex. Use this method to show that $`||Ax + b||_1`$ is convex (in $`x`$), where $`||z||_1 = \sum |z_i|`$.

### (e) Suppose you know that $`f_1`$ and $`f_2`$ are convex functions on a set A. The function $`g(x) := \max\{f_1(x), f_2(x)\}`$ is also convex on A. In general, if $`f(x, y)`$ is convex in $`x`$ for each $`y`$, then $`g(x) := \sup_y f(x, y)`$ is convex. Use this method to show that the largest eigenvalue of a matrix $`X`$, $`\lambda_{\text{Max}}(X)`$, is convex in $`X`$ (Using the definition of convexity would make this question quite difficult).

### (f) Does the same result hold for $`h(x) := \min\{f_1(x), f_2(x)\}`$? If so, give a proof. If not, provide convex functions $`f_1, f_2`$ such that $`h`$ is not convex.

## 6. Gradient Descent

We will now examine gradient descent algorithm and study the effect of learning rate $`\alpha`$ on the convergence of the algorithm. Recall from lecture that Gradient Descent takes on the form of $`x_{t+1} = x_t - \alpha \nabla f`$

### (a) Convergence Rate of Gradient Descent

Assume that $`f: \mathbb{R}^n \to \mathbb{R}`$ is convex and differentiable, and additionally,
```math
||\nabla f(x) - \nabla f(y)|| \le L||x - y|| \text{ for any } x, y
```
I.e., $`\nabla f`$ is Lipschitz continuous with constant $`L > 0`$
Show that:
Gradient descent with fixed step size $`\eta \le \frac{1}{L}`$ satisfies
```math
f(x^{(k)}) - f(x^*) \le \frac{||x^{(0)} - x^*||^2}{2\eta k}
```
I.e., gradient descent has convergence rate $`O(\frac{1}{k})`$

**Hints:**
(i) $`\nabla f`$ is Lipschitz continuous with constant $`L > 0 \to f(y) \le f(x) + \nabla f(x) (y-x) + \frac{L}{2} ||y-x||^2`$ for all $`x, y`$.
(ii) $`f`$ is convex $`\to f(x) \le f(x^*) + \nabla f(x^*) (x-x^*)`$, where $`x^*`$ is the local minima that the gradient descent algorithm is converging to.
(iii) $`2\eta \nabla f(x) (x - x^*) - \eta^2 ||\nabla f(x)||^2 = ||x-x^*||^2 - ||x - \eta \nabla f(x) - x^*||^2`$
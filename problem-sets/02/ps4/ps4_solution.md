# Problem Set 4 Solutions

Let $`\phi: \mathbb{R}^d \to \mathbb{R}^k`$ be a feature map, define $`K`$ as the kernel function, and define $`G`$ to be the kernel matrix of $`\phi`$.

(a) The kernel matrix is symmetric, that is, show $`G_{i,j} = G_{j,i}`$.

(b) The kernel matrix $`G`$ is positive semi-definite, that is, for any column vector $`x, x^T Gx \ge 0`$.

(c) Mercer's theorem: A function $`K: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}`$ is a valid kernel if and only if the corresponding kernel matrix $`G`$ is symmetric and positive definite.

## 1. Kernelized Linear Regression

Recall that the definition of a kernel is the following:

**Definition 1.** A function $`K: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}`$ is a kernel for a map $`\phi`$ if $`K(x, x') = \phi(x) \cdot \phi(x') = \langle\phi(x), \phi(x')\rangle`$ for all $`x, x'`$.

Consider regularized linear regression (without a bias, for simplicity). Our objective to find the optimal parameters $`\hat{w} = \arg \min_w L(W)`$ for a dataset $`(x_i, y_i)_{i=1}^n`$ that minimize the following loss function:

```math
L(w) = \sum_{i=1}^n (w^T x_i - y_i)^2 + \lambda ||w||^2
```

Note that from class, we know there is an optimal $`\hat{w}`$ that lies in the span of the datapoints. Concretely, there exist $`\alpha_1, \dots, \alpha_n \in \mathbb{R}`$ such that $`\hat{w} = \sum \alpha_i x_i`$. Also recall from lecture that the expression of our loss function $`L(w)`$ in terms of the kernel is:

```math
L(w) = ||\mathbf{y} - \mathbf{K}\alpha||_2^2 + \lambda \alpha^T \mathbf{K}\alpha
```

This derivation can be seen here on slide 53.

**(a)** Solve for the optimal $`\hat{\alpha}`$.

**Solution:**
Setting gradient of $`L(w)`$ with respect to $`\alpha`$ equal to 0:
```math
\nabla_\alpha L(w) = 0
```
```math
-2\mathbf{K} (\mathbf{y} - \mathbf{K}\alpha) + 2\lambda \mathbf{K}\alpha = 0
```
```math
-\mathbf{K} (\mathbf{y} - \mathbf{K}\alpha) + \lambda \mathbf{K}\alpha = 0
```
```math
\mathbf{K} (\mathbf{K}\alpha - \mathbf{y} + \lambda \alpha) = 0
```
```math
\mathbf{K} ((\mathbf{K} + \lambda \mathbf{I})\alpha - \mathbf{y}) = 0
```
```math
\mathbf{K}(\mathbf{K} + \lambda \mathbf{I})\alpha = \mathbf{K}\mathbf{y}
```
```math
\hat{\alpha} = (\mathbf{K} + \lambda \mathbf{I})^{-1}\mathbf{y}
```

**(b)** Let us assume that we were using a linear kernel where $`K_{ij} = x_i^T x_j`$. Suppose we have $`\mathbf{X}_{\text{test}}`$ that we want to make prediction for after training on $`\mathbf{X}_{\text{train}}`$. Express the estimate $`\hat{\mathbf{Y}}`$ in terms of $`\mathbf{K}_{\text{train}} = \mathbf{X}_{\text{train}}\mathbf{X}_{\text{train}}^T`$, $`\mathbf{Y}_{\text{train}}`$, $`\mathbf{X}_{\text{train}}`$ and $`\mathbf{X}_{\text{test}}`$. What would the general prediction formula look like if we are not using a linear kernel? Express the solution in terms of $`\mathbf{K}_{\text{train, test}}`$

**Solution:**

**General Solution for Kernel Ridge:**
```math
\hat{Y} = \mathbf{X}_{\text{test}}\hat{w}
```
```math
= \mathbf{X}_{\text{test}}\mathbf{X}_{\text{Train}}^T\hat{\alpha}
```
```math
= \mathbf{X}_{\text{test}}\mathbf{X}_{\text{Train}}^T (\mathbf{K}_{\text{train}} + \lambda\mathbf{I})^{-1}y_{\text{train}}
```
```math
\hat{Y} = \mathbf{K}_{\text{train, test}}\hat{\alpha}
```
Where $`\mathbf{K}_{\text{train,test}} = \mathbf{X}_{\text{test}}\mathbf{X}_{\text{Train}}^T`$

## 2. Proving $`\hat{w} \in \text{Span}(x_1, ..., x_n)`$

We will prove this through contradiction. Assume $`\hat{w} \notin \text{span}(x_1,..., x_n)`$ solves $`\text{arg min}_w L(w)`$. Then, there exists a component of $`\hat{w}`$ that is perpendicular to the span, which we will call $`\hat{w}^\perp`$. Concretely,

```math
\hat{w} = \bar{w} + \hat{w}^\perp
```

Where $`\bar{w} = \sum_i^n \alpha_i x_i`$ is the component of $`\hat{w}`$ in the span of the datapoints.

To show that $`\hat{w}^\perp`$ is part of our optimal parameters, we need to consider both the error term and the regularization term of L(w). Since $`\bar{w}`$ and $`\hat{w}^\perp`$ are perpendicular to each other, their contribution to L(w) can be minimized independently. Let us split the error and regularization terms into their $`\bar{w}`$ and $`\hat{w}^\perp`$ components.

### (a)

First, we will find the optimal hyperparameter selection for the error term of our loss function in terms of $`\bar{w}`$ and $`\hat{w}^\perp`$. Show that $`\hat{w} \cdot x_i = \bar{w} \cdot x_i`$, for every $`\mathbf{x}_i`$. (Hint: what is the relationship of $`\hat{w}^\perp`$ and $`\mathbf{x}_i`$)

**Solution:**
```math
\hat{w} \cdot x_i = (\bar{w} + \hat{w}^\perp) \cdot x_i
```
```math
= \bar{w} \cdot x_i + \hat{w}^\perp \cdot x_i
```
```math
= \bar{w} \cdot x_i + 0
```
```math
= \bar{w} \cdot x_i
```

$`\hat{w}^\perp`$ is perpendicular to each $`\mathbf{x}_i`$

### (b)

We have shown that for the optimal solution, the error term relies only on $`\text{Span}(x_1,...x_n)`$. Let us find the regularization term in terms of $`\bar{w}`$ and $`\hat{w}^\perp`$ and the range of values it can take. Now, show that $`\|\hat{w}\|^2 \ge \|\bar{w}\|^2`$.

**Solution:**
```math
\|\hat{w}\|^2 = \|\bar{w} + \hat{w}^\perp\|^2
```
```math
= (\bar{w} + \hat{w}^\perp)^T (\bar{w} + \hat{w}^\perp)
```
```math
= \bar{w}^T \bar{w} + 2\bar{w}^T \hat{w}^\perp + (\hat{w}^\perp)^T \hat{w}^\perp
```
```math
= \|\bar{w}\|^2 + \|\hat{w}^\perp\|^2
```
```math
\ge \|\bar{w}\|^2
```

as $`\bar{w}^T \hat{w}^\perp = \langle\bar{w}, \hat{w}^\perp\rangle = 0`$

### (c)

We now know the minimum value of the regularization term and what it is equal to with respect to $`\bar{w}`$ and $`\hat{w}^\perp`$. Finally, show that $`\hat{w} \in \text{Span}(x_1,...,x_n)`$. (Hint: Think about the regularization term. What is $`\hat{w}^\perp`$ when the regularization term is minimized?)

**Solution:**
Note that in the loss function, we're trying to minimize the magnitude of w (with the regularization term $`\lambda||w||_2`$). Now note that if $`\forall_i \hat{w}^T x_i = \bar{w}^T x_i`$, and $`\|\hat{w}\|_2 \ge \|\bar{w}\|_2`$, then our optimization will always choose $`w^\perp = 0`$ (as we favor smaller solutions), meaning that $`\hat{w} = \bar{w}`$ and $`\hat{w} \in \text{Span}(x_1, \dots, x_n)$), which completes the contradiction.

## Remark

For running Jupyter Notebook locally, the following are required:
*   Jupyter Notebook [Install] [Document]
*   ipywidgets
*   pytorch
*   matplotlib
*   numpy

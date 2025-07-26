# Problem Set #3: Math Review

This document contains Problem Set #3, focusing on advanced mathematical concepts relevant to machine learning.

## Problem 1: Advanced Linear Algebra

### (a) Matrix Decompositions

**Problem:** Let $`A \in \mathbb{R}^{n \times n}`$ be a symmetric positive definite matrix. Show that $`A`$ can be written as $`A = LL^T`$ where $`L`$ is a lower triangular matrix.

### (b) Eigenvalue Bounds

**Problem:** Let $`A \in \mathbb{R}^{n \times n}`$ be symmetric with eigenvalues $`\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n`$. Show that for any unit vector $`x`$:

$`\lambda_n \leq x^T A x \leq \lambda_1`$

## Problem 2: Optimization Theory

### (a) Convexity and Gradient Descent

**Problem:** Let $`f : \mathbb{R}^n \to \mathbb{R}`$ be a convex function with $`L`$-Lipschitz gradient. Show that for gradient descent with step size $`\alpha = \frac{1}{L}`$:

$`f(x_{k+1}) - f(x_k) \leq -\frac{1}{2L} \|\nabla f(x_k)\|_2^2`$

### (b) Lagrange Multipliers

**Problem:** Consider the optimization problem:

```math
\min_{x \in \mathbb{R}^n} f(x) \quad \text{subject to} \quad g_i(x) = 0, \quad i = 1, 2, \ldots, m
```

Show that if $`x^*`$ is a local minimum and the constraint gradients $`\{\nabla g_i(x^*)\}_{i=1}^m`$ are linearly independent, then there exist Lagrange multipliers $`\lambda_1, \lambda_2, \ldots, \lambda_m`$ such that:

$`\nabla f(x^*) + \sum_{i=1}^m \lambda_i \nabla g_i(x^*) = 0`$

## Problem 3: Probability and Statistics

### (a) Maximum Likelihood Estimation

**Problem:** Let $`X_1, X_2, \ldots, X_n`$ be i.i.d. random variables from a normal distribution $`\mathcal{N}(\mu, \sigma^2)`$. Find the maximum likelihood estimators for $`\mu`$ and $`\sigma^2`$.

### (b) Central Limit Theorem

**Problem:** State and prove the Central Limit Theorem for i.i.d. random variables.

## Problem 4: Information Theory

### (a) Entropy and Mutual Information

**Problem:** Let $`X`$ and $`Y`$ be discrete random variables. Define the entropy $`H(X)`$ and mutual information $`I(X; Y)`$. Show that $`I(X; Y) \geq 0`$ with equality if and only if $`X`$ and $`Y`$ are independent.

### (b) Kullback-Leibler Divergence

**Problem:** Let $`P`$ and $`Q`$ be probability distributions over the same sample space. Define the Kullback-Leibler divergence $`D_{KL}(P \| Q)`$ and show that $`D_{KL}(P \| Q) \geq 0`$ with equality if and only if $`P = Q`$.


# 6.7 Regularization and the Non-separable Case

## Introduction

**Motivation:**
In real-world datasets, perfect linear separability is rare. Outliers, noise, or overlapping classes can make it impossible to find a hyperplane that separates the data without error. SVMs address this by introducing slack variables and regularization, allowing some points to be misclassified or to fall within the margin, while still maximizing the margin as much as possible.

**The Real-World Challenge**:
The derivation of the SVM as presented so far assumed that the data is linearly separable. While mapping data to a high dimensional feature space via $\phi$ does generally increase the likelihood that the data is separable, we can't guarantee that it always will be so. Also, in some cases it is not clear that finding a separating hyperplane is exactly what we'd want to do, since that might be susceptible to outliers.

## From Optimal Classification to Practical Regularization

We've now derived the **optimal margin classifier** and seen how the dual formulation naturally leads to kernelization and reveals the elegant structure of support vectors. The dual form expresses everything in terms of inner products, enabling us to work in high-dimensional feature spaces efficiently through the kernel trick.

However, the optimal margin classifier we've developed assumes that the data is **linearly separable** - that there exists a hyperplane that can perfectly separate all training points. In real-world problems, this assumption is often violated due to noise, outliers, or inherently non-separable data.

This motivates our exploration of **SVM regularization** - extending the optimal margin classifier to handle non-separable data through the introduction of **slack variables** and the **soft margin** formulation. This regularization approach allows us to trade off between margin size and classification error, making SVMs robust to noisy data.

The transition from optimal margin classification to regularization represents the bridge from theoretical perfection to practical robustness - taking our elegant mathematical formulation and adapting it to handle the messy realities of real-world data.

In this section, we'll explore how regularization makes SVMs practical for real-world problems and introduces the crucial hyperparameter $C$ that controls the trade-off between margin size and classification accuracy.

**The Outlier Problem**:
For instance, the left figure below shows an optimal margin classifier, and when a single outlier is added in the upper-left region (right figure), it causes the decision boundary to make a dramatic swing, and the resulting classifier has a much smaller margin.

<img src="./img/smaller_margin.png" width="700px" />

**The Solution**: To make the algorithm work for non-linearly separable datasets as well as be less sensitive to outliers, we reformulate our optimization (using $\ell_1$ **regularization**) as follows:

$$
\min_{w, x, b} \quad \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \xi_i
$$

subject to

$$
y^{(i)} (w^T x^{(i)} + b) \geq 1 - \xi_i, \quad i = 1, \ldots, n
\xi_i \geq 0, \quad i = 1, \ldots, n.
$$

*Implementation details are provided in the accompanying Python examples file.*

---
**Intuition:**
The slack variables $\xi_i$ measure how much each point violates the margin. If $\xi_i = 0$, the point is correctly classified and outside the margin. If $0 < \xi_i < 1$, the point is inside the margin but on the correct side. If $\xi_i > 1$, the point is misclassified. The regularization parameter $C$ controls the trade-off between maximizing the margin and minimizing the total slack (i.e., classification error).

*Implementation details are provided in the accompanying Python examples file.*

## Understanding Slack Variables

**The Slack Variable Concept**:
Thus, examples are now permitted to have (functional) margin less than 1, and if an example has functional margin $1 - \xi_i$ (with $\xi > 0$), we would pay a cost of the objective function being increased by $C \xi_i$. The parameter $C$ controls the relative weighting between the twin goals of making the $\|w\|^2$ small (which we saw earlier makes the margin large) and of ensuring that most examples have functional margin at least 1.

**The Trade-off**:
- **Large $C$**: High penalty for violations, small margin, few misclassifications
- **Small $C$**: Low penalty for violations, large margin, more misclassifications

---
**Practical scenario:**
Suppose you have a dataset of emails, and a few spam emails are mislabeled as non-spam. Without slack variables, a single mislabeled point could force the SVM to create a very poor decision boundary. With slack and regularization, the SVM can "ignore" these outliers to some extent, leading to a more robust classifier.

*Implementation details are provided in the accompanying Python examples file.*

## The Lagrangian Formulation

As before, we can form the Lagrangian:

$$
\mathcal{L}(w, b, \xi, \alpha, r) = \frac{1}{2} w^T w + C \sum_{i=1}^n \xi_i - \sum_{i=1}^n \alpha_i \left[ y^{(i)} (w^T x^{(i)} + b) - 1 + \xi_i \right] - \sum_{i=1}^n r_i \xi_i.
$$

Here, the $\alpha_i$'s and $r_i$'s are our Lagrange multipliers (constrained to be $\geq 0$). We won't go through the derivation of the dual again in detail, but after setting the derivatives with respect to $w$ and $b$ to zero as before, substituting them back in, and simplifying, we obtain the following dual form of the problem:

*Implementation details are provided in the accompanying Python examples file.*

---
**Interpretation:**
The only change from the separable case is the upper bound $C$ on the $\alpha_i$'s. This means that no single point can become "too influential" in determining the decision boundary, which helps control the effect of outliers.

**The Key Changes**:
- **Upper bound**: $\alpha_i \leq C$ (was $\alpha_i \geq 0$ before)
- **Interpretation**: Limits the influence of any single training point
- **Effect**: Prevents overfitting to outliers

As before, we also have that $w$ can be expressed in terms of the $\alpha_i$'s as given in Equation (6.10), so that after solving the dual problem, we can continue to use Equation (6.15) to make our predictions. Note that, somewhat surprisingly, in adding $\ell_1$ regularization, the only change to the dual problem is that what was previously a constraint that $0 \leq \alpha_i$ has now become $0 \leq \alpha_i \leq C$. The calculation for $b^*$ also has to be modified (Equation 6.13 is no longer valid); we'll see the correct formula later.

Also, the KKT dual-complementarity conditions (which in the next section will be useful for testing for the convergence of the SMO algorithm) are:

$$
\alpha_i = 0 \implies y^{(i)} (w^T x^{(i)} + b) \geq 1 \tag{6.16}
$$
$$
\alpha_i = C \implies y^{(i)} (w^T x^{(i)} + b) \leq 1 \tag{6.17}
$$
$$
0 < \alpha_i < C \implies y^{(i)} (w^T x^{(i)} + b) = 1. \tag{6.18}
$$

---
**Remark:**
These KKT conditions are not just theoretical—they are used in practice to check for convergence in SVM solvers like SMO.

**Understanding the KKT Conditions**:
- **$\alpha_i = 0$**: Point is correctly classified and outside the margin
- **$\alpha_i = C$**: Point is either misclassified or on the wrong side of the margin
- **$0 < \alpha_i < C$**: Point is exactly on the margin (support vector)

*Implementation details are provided in the accompanying Python examples file.*

Now, all that remains is to give an algorithm for actually solving the dual problem, which we will do in the next section.

# 6.8 The SMO Algorithm

## Introduction

---
**Why SMO?**
For large datasets, solving the dual SVM problem directly can be computationally expensive, as it involves optimizing over all $\alpha_i$'s simultaneously. The SMO algorithm breaks this large problem into a series of much smaller problems, each involving only two variables at a time, making it highly scalable and efficient.

*Implementation details are provided in the accompanying Python examples file.*

The SMO (sequential minimal optimization) algorithm, due to John Platt, gives an efficient way of solving the dual problem arising from the derivation of the SVM. Partly to motivate the SMO algorithm, and partly because it's interesting in its own right, let's first take another digression to talk about the coordinate ascent algorithm.

## 6.8.1 Coordinate Ascent

### The Basic Idea

Consider trying to solve the unconstrained optimization problem

$$
\max_{\alpha} W(\alpha_1, \alpha_2, \ldots, \alpha_n).
$$

Here, we think of $W$ as just some function of the parameters $\alpha_i$'s, and for now ignore any relationship between this problem and SVMs. We've already seen two optimization algorithms, gradient ascent and Newton's method. The new algorithm we're going to consider here is called **coordinate ascent**:

**The Algorithm**:
Loop until convergence:  {

For $i = 1, \ldots, n$,  {

$$
  \alpha_i := \arg\max_{\tilde{\alpha}_i} W(\alpha_1, \ldots, \alpha_{i-1}, \tilde{\alpha}_i, \alpha_{i+1}, \ldots, \alpha_n).
$$
  }

}

*Implementation details are provided in the accompanying Python examples file.*

---
**Intuition:**
Coordinate ascent is a simple yet powerful optimization technique. By optimizing one variable at a time while holding the others fixed, we can often make steady progress toward the optimum, especially when the function is well-behaved (e.g., convex or quadratic in each variable).

**Why Coordinate Ascent Works**:
- **Simplicity**: Each subproblem is one-dimensional
- **Efficiency**: Can often solve each subproblem analytically
- **Convergence**: Guaranteed to converge for convex functions
- **Scalability**: Works well for high-dimensional problems

Thus, in the innermost loop of this algorithm, we will hold all the variables except for some $\alpha_i$ fixed, and reoptimize $W$ with respect to just the parameter $\alpha_i$. In the version of this method presented here, the inner-loop reoptimizes the variables in order $\alpha_1, \alpha_2, \ldots, \alpha_n, \alpha_1, \alpha_2, \ldots$. (A more sophisticated version might choose other orderings; for instance, we may choose the next variable to update according to which one we expect to allow us to make the largest increase in $W(\alpha)$.)

When the function $W$ happens to be of such a form that the "arg max" in the inner loop can be performed efficiently, then coordinate ascent can be a fairly efficient algorithm.

<img src="./img/coordinate_ascent.png" width="400px"/>

---
**Geometric view:**
The path taken by coordinate ascent is a sequence of axis-aligned steps in the parameter space. Each step moves along one coordinate direction, which can be visualized as moving parallel to one of the axes in a multidimensional space.

## 6.8.2 SMO

### The Problem with Coordinate Ascent for SVMs

We close off the discussion of SVMs by sketching the derivation of the SMO algorithm.

Here's the (dual) optimization problem that we want to solve:

$$
\begin{align}
\max_{\alpha} \quad & W(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \tag{6.19} \\
\text{s.t.} \quad & 0 \leq \alpha_i \leq C, \quad i = 1, \ldots, n \tag{6.20} \\
& \sum_{i=1}^n \alpha_i y^{(i)} = 0. \tag{6.21}
\end{align}
$$

**The Challenge**: Let's say we have set of $\alpha_i$'s that satisfy the constraints (6.20, 6.21). Now, suppose we want to hold $\alpha_2, \ldots, \alpha_n$ fixed, and take a coordinate ascent step and reoptimize the objective with respect to $\alpha_1$. Can we make any progress? The answer is no, because the constraint (6.21) ensures that

$$
\alpha_1 y^{(1)} = - \sum_{i=2}^n \alpha_i y^{(i)}.
$$

Or, by multiplying both sides by $y^{(1)}$, we equivalently have

$$
\alpha_1 = -y^{(1)} \sum_{i=2}^n \alpha_i y^{(i)}.
$$

(This step used the fact that $y^{(1)} \in \{-1, 1\}$, and hence $(y^{(1)})^2 = 1$.) Hence, $\alpha_1$ is exactly determined by the other $\alpha_i$'s, and if we were to hold $\alpha_2, \ldots, \alpha_n$ fixed, then we can't make any change to $\alpha_1$ without violating the constraint (6.21) in the optimization problem.

---
**Key insight:**
This is why SMO always updates two $\alpha$'s at a time: the equality constraint couples all the $\alpha$'s together, so changing one requires changing at least one other to maintain feasibility.

### The SMO Algorithm

Thus, if we want to update some subject of the $\alpha_i$'s, we must update at least two of them simultaneously in order to keep satisfying the constraints. This motivates the SMO algorithm, which simply does the following:

**The SMO Algorithm**:
Repeat till convergence {

1. Select some pair $\alpha_i$ and $\alpha_j$ to update next (using a heuristic that tries to pick the two that will allow us to make the biggest progress towards the global maximum).
2. Reoptimize $W(\alpha)$ with respect to $\alpha_i$ and $\alpha_j$, while holding all the other $\alpha_k$'s ($k \neq i, j$) fixed.

}

*Implementation details are provided in the accompanying Python examples file.*

**Why Two Variables?**:
- **Feasibility**: The equality constraint requires updating at least two variables
- **Efficiency**: Two-variable optimization is still very fast
- **Convergence**: Can make significant progress with each update

### The Two-Variable Optimization

The key reason that SMO is an efficient algorithm is that the update to $\alpha_i, \alpha_j$ can be computed very efficiently. Let's now briefly sketch the main idea for deriving efficient update.

Let's say we currently have some setting of the $\alpha_i$'s that satisfy the constraints (6.20, 6.21), and suppose we've decided to hold $\alpha_3, \ldots, \alpha_n$ fixed, and want to reoptimize $W(\alpha_1, \alpha_2, \ldots, \alpha_n)$ with respect to $\alpha_1$ and $\alpha_2$ (subject to the constraints). From (6.21), we require that

$$
\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = - \sum_{i=3}^n \alpha_i y^{(i)}.
$$

$$
\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta. \tag{6.22}
$$

**The Constraint Line**: We can thus picture the constraints on $\alpha_1$ and $\alpha_2$ as follows:

<img src="./img/constraints.png" width="350px"/>

From the constraints (6.20), we know that $\alpha_1$ and $\alpha_2$ must lie within the box $[0, C] \times [0, C]$ shown. Also plotted is the line $\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta$, on which we know $\alpha_1$ and $\alpha_2$ must lie. Note also that, from these constraints, we know $L \leq \alpha_2 \leq H$; otherwise, $(\alpha_1, \alpha_2)$ can't simultaneously satisfy both the box and the straight line constraint. In this example, $L = 0$. But depending on what the line $\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta$ looks like, this won't always necessarily be the case; but more generally, there will be some lower-bound $L$ and some upper-bound $H$ on the permissible values for $\alpha_2$ that will ensure that $\alpha_1, \alpha_2$ lie within the box $[0, C] \times [0, C]$.

### The Analytical Solution

Using Equation (6.22), we can also write $\alpha_1$ as a function of $\alpha_2$:

$$
\alpha_1 = (\zeta - \alpha_2 y^{(2)}) y^{(1)}.
$$

(Check this derivation yourself; we again used the fact that $y^{(1)} \in \{-1, 1\}$ so that $(y^{(1)})^2 = 1$.) Hence, the objective $W(\alpha)$ can be written

$$
W(\alpha_1, \alpha_2, \ldots, \alpha_n) = W(((\zeta - \alpha_2 y^{(2)}) y^{(1)}, \alpha_2, \ldots, \alpha_n).
$$

Treating $\alpha_3, \ldots, \alpha_n$ as constants, you should be able to verify that this is just some quadratic function in $\alpha_2$. I.e., this can also be expressed in the form $a \alpha_2^2 + b \alpha_2 + c$ for some appropriate $a, b, c$. If we ignore the "box" constraints (6.20) (or, equivalently, that $L \leq \alpha_2 \leq H$), then we can easily maximize this quadratic function by setting its derivative to zero and solving. We'll let $\alpha_2^{\text{new, unclipped}}$ denote the resulting value of $\alpha_2$. You should also be able to convince yourself that if we had instead wanted to maximize $W$ with respect to $\alpha_2$ but subject to the box constraint, then we can find the resulting value optimal simply by taking $\alpha_2^{\text{new, unclipped}}$ and "clipping" it to lie in the $[L, H]$ interval, to get

$$
\alpha_2^{\text{new}} = \begin{cases}
H & \text{if } \alpha_2^{\text{new, unclipped}} > H \\
\alpha_2^{\text{new, unclipped}} & \text{if } L \leq \alpha_2^{\text{new, unclipped}} \leq H \\
L & \text{if } \alpha_2^{\text{new, unclipped}} < L
\end{cases}
$$

*Implementation details are provided in the accompanying Python examples file.*

**The Clipping Step**:
- **Unclipped**: The analytical solution to the quadratic optimization
- **Clipped**: The solution that respects the box constraints
- **Feasibility**: Ensures the solution satisfies all constraints

Finally, having found the $\alpha_2^{\text{new}}$, we can use Equation (6.22) to go back and find the optimal value of $\alpha_1^{\text{new}}$.

### Implementation Details

There are a couple more details that are quite easy but that we'll leave you to read about yourself in Platt's paper: One is the choice of the heuristics used to select the next $\alpha_i, \alpha_j$ to update; the other is how to update $b$ as the SMO algorithm is run.

**The Heuristics**:
- **First choice**: Pick the first $\alpha_i$ that violates KKT conditions
- **Second choice**: Pick the second $\alpha_j$ that maximizes the step size
- **Fallback**: Use random selection if no good pair is found

**Updating $b$**:
- **After each update**: Recompute $b$ using the new $\alpha$ values
- **Using support vectors**: Only use points with $0 < \alpha_i < C$
- **Averaging**: If multiple support vectors, average the computed $b$ values

**Convergence**:
- **KKT conditions**: Check if all KKT conditions are satisfied
- **Tolerance**: Allow small violations (e.g., $\epsilon = 10^{-3}$)
- **Maximum iterations**: Limit the number of iterations

**The Efficiency of SMO**:
- **Analytical solution**: No need for iterative optimization in each step
- **Sparse updates**: Only two variables change at a time
- **Fast convergence**: Typically converges in $O(n)$ iterations
- **Memory efficient**: Only need to store the kernel matrix

**Why SMO is Fast**:
1. **Analytical solution**: No need for iterative optimization in each step
2. **Sparse updates**: Only two variables change at a time
3. **Good heuristics**: Smart choice of variable pairs
4. **Early stopping**: Can stop when KKT conditions are approximately satisfied

**Comparison with Other Methods**:
- **Interior point methods**: More complex, but can be faster for small problems
- **Gradient methods**: Simpler, but slower convergence
- **SMO**: Good balance of simplicity and efficiency

The SMO algorithm has become the standard method for training SVMs, and is implemented in popular libraries like LIBSVM and scikit-learn. Its success is due to its combination of simplicity, efficiency, and robustness.

## From Theoretical Framework to Practical Implementation

We've now built a comprehensive theoretical understanding of **Support Vector Machines** - from the geometric intuition of margins to the mathematical optimization of the dual formulation, and from kernel methods to regularization techniques. This theoretical framework provides the foundation for one of the most powerful classification algorithms in machine learning.

However, true mastery of SVMs comes from **hands-on implementation** and practical application. While understanding the mathematical framework is essential, implementing SVMs from scratch, experimenting with different kernels and regularization parameters, and applying them to real-world problems is where the concepts truly come to life.

The transition from theoretical framework to practical implementation is crucial in advanced classification. While the mathematical foundations provide the structure, implementing these algorithms helps develop intuition, reveals practical challenges, and builds the skills needed for real-world applications. Coding SVMs from scratch forces us to confront the details that theory often abstracts away.

In the next section, we'll put our theoretical knowledge into practice through hands-on coding exercises. We'll implement SVMs from scratch, experiment with different kernels and regularization approaches, and develop the practical skills needed to apply these powerful classification algorithms to real-world problems.

This hands-on approach will solidify our understanding and prepare us for the complex challenges that arise when applying advanced classification techniques in practice.

---

**Previous: [Optimal Margin Classifiers](04_svm_optimal_margin.md)** - Derive the optimal margin classifier and understand the dual formulation that enables kernelization.

**Next: [Hands-on Coding](06_hands-on_coding.md)** - Implement support vector machines from scratch and apply them to real-world classification problems. 
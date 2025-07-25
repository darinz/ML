# Problem Set #2: Kernels, SVMs, and Theory

## 1. Kernel ridge regression

In contrast to ordinary least squares which has a cost function
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)})^2,
```
we can also add a term that penalizes large weights in $`\theta`$. In *ridge regression*, our least squares cost is regularized by adding a term $`\lambda||\theta||^2`$, where $`\lambda > 0`$ is a fixed (known) constant (regularization will be discussed at greater length in an upcoming course lecture). The ridge regression cost function is then
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)})^2 + \frac{\lambda}{2}||\theta||^2.
```

(a) Use the vector notation described in class to find a closed-form expression for the value of $`\theta`$ which minimizes the ridge regression cost function.

**Answer:**  Using the design matrix notation, we can rewrite $`J(\theta)`$ as

```math
J(\theta) = \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}) + \frac{\lambda}{2} \theta^T \theta.
```

Then the gradient is

```math
\nabla_{\theta} J(\theta) = X^T X \theta - X^T \vec{y} + \lambda \theta.
```

Setting the gradient to 0 gives us

```math
0 = X^T X \theta - X^T \vec{y} + \lambda \theta
```
```math
\theta = (X^T X + \lambda I)^{-1} X^T \vec{y}.
```

**Explanation:**

In ridge regression, we start with the regularized least squares cost function:

```math
J(\theta) = \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}) + \frac{\lambda}{2} \theta^T \theta.
```

- The first term measures the sum of squared errors between the predictions $`X\theta`$ and the true values $`\vec{y}`$.
- The second term penalizes large values of $`\theta`$ to prevent overfitting, with $`\lambda`$ controlling the strength of regularization.

To find the minimum, we take the gradient with respect to $`\theta`$:

```math
\nabla_{\theta} J(\theta) = X^T X \theta - X^T \vec{y} + \lambda \theta.
```

- $`X^T X \theta`$ comes from differentiating the quadratic term $`(X\theta - \vec{y})^T (X\theta - \vec{y})`$.
- $`-X^T \vec{y}`$ is from the cross term.
- $`\lambda \theta`$ is from the regularization term.

Setting the gradient to zero gives the normal equations for ridge regression:

```math
0 = X^T X \theta - X^T \vec{y} + \lambda \theta
```

Rearranging terms:

```math
X^T X \theta + \lambda \theta = X^T \vec{y}
```

Factoring $`\theta`$:

```math
(X^T X + \lambda I) \theta = X^T \vec{y}
```

Solving for $`\theta`$ gives the closed-form solution:

```math
\theta = (X^T X + \lambda I)^{-1} X^T \vec{y}.
```

This formula shows how regularization modifies the normal equations by adding $`\lambda I`$ to the matrix being inverted, improving numerical stability and reducing overfitting.

---

(b) Suppose that we want to use kernels to implicitly represent our feature vectors in a high-dimensional (possibly infinite dimensional) space. Using a feature mapping $`\phi`$, the ridge regression cost function becomes
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (\theta^T \phi(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2}||\theta||^2.
```

Making a prediction on a new input $`x_{new}`$ would now be done by computing $`\theta^T \phi(x_{new})`$. Show how we can use the "kernel trick" to obtain a closed form for the prediction on the new input without ever explicitly computing $`\phi(x_{new})`$. You may assume that the parameter vector $`\theta`$ can be expressed as a linear combination of the input feature vectors; i.e., $`\theta = \sum_{i=1}^{m} \alpha_i \phi(x^{(i)})`$ for some set of parameters $`\alpha_i`$.

[Hint: You may find the following identity useful:
```math
(\lambda I + BA)^{-1}B = B(\lambda I + AB)^{-1}.
```
If you want, you can try to prove this as well, though this is not required for the problem.]

**Answer:**  Let $`\Phi`$ be the design matrix associated with the feature vectors $`\phi(x^{(i)})`$. Then from parts (a) and (b),

```math
\begin{align*}
\theta &= (\Phi^T \Phi + \lambda I)^{-1} \Phi^T \vec{y} \\
       &= \Phi^T (\Phi \Phi^T + \lambda I)^{-1} \vec{y} \\
       &= \Phi^T (K + \lambda I)^{-1} \vec{y}.
\end{align*}
```

where $`K`$ is the kernel matrix for the training set (since $`\Phi_{i,j} = \phi(x^{(i)})^T \phi(x^{(j)}) = K_{ij}`$.)
To predict a new value $`y_{\text{new}}`$, we can compute

```math
\begin{align*}
\vec{y}_{\text{new}} &= \theta^T \phi(x_{\text{new}}) \\
              &= \vec{y}^T (K + \lambda I)^{-1} \Phi \phi(x_{\text{new}}) \\
              &= \sum_{i=1}^m \alpha_i K(x^{(i)}, x_{\text{new}}).
\end{align*}
```

where $`\alpha = (K + \lambda I)^{-1} \vec{y}`$. All these terms can be efficiently computing using the kernel function.
To prove the identity from the hint, we left-multiply by $`\lambda(I + BA)`$ and right-multiply by $`\lambda(I + AB)`$ on both sides. That is,

```math
\begin{align*}
(\lambda I + BA)^{-1} B &= B(\lambda I + AB)^{-1} \\
B &= (\lambda I + BA)B(\lambda I + AB)^{-1} \\
B(\lambda I + AB) &= (\lambda I + BA)B \\
\lambda B + BAB &= \lambda B + BAB.
\end{align*}
```

This last line clearly holds, proving the identity.

**Explanation:**

When using kernels, we map the data into a high-dimensional space via $`\phi(x)`$ and express the solution in terms of the kernel matrix $`K`$.

- $`\Phi`$ is the design matrix of mapped features.
- $`K = \Phi \Phi^T`$ is the kernel matrix, where $`K_{ij} = \phi(x^{(i)})^T \phi(x^{(j)})`$.

The solution for $`\theta`$ can be rewritten using the matrix inversion identity:

```math
\theta = \Phi^T (K + \lambda I)^{-1} \vec{y}
```

To predict for a new input $`x_{\text{new}}`$:

```math
\vec{y}_{\text{new}} = \theta^T \phi(x_{\text{new}}) = \vec{y}^T (K + \lambda I)^{-1} \Phi \phi(x_{\text{new}})
```

- $`\Phi \phi(x_{\text{new}})`$ is a vector of kernel evaluations between $`x_{\text{new}}`$ and each training point.
- The prediction can be written as a sum over training points:

```math
\sum_{i=1}^m \alpha_i K(x^{(i)}, x_{\text{new}})
```

where $`\alpha = (K + \lambda I)^{-1} \vec{y}`$.

The matrix identity from the hint is proved by multiplying both sides by $`\lambda(I + AB)`$ and $`\lambda(I + BA)`$ and showing both sides are equal, confirming the equivalence of the two forms.

## 2. $`\ell_2`$ norm soft margin SVMs

In the notes, we saw that if our data is not linearly separable, then we need to modify our support vector machine algorithm by introducing an error margin that must be minimized. Specifically, the formulation we have looked at is known as the $`\ell_1`$ norm soft margin SVM. In this problem we will consider an alternative method, known as the $`\ell_2`$ norm soft margin SVM. This new algorithm is given by the following optimization problem (notice that the slack penalties are now squared):

```math
\min_{w, b, \xi} \quad \frac{1}{2} \|w\|^2 + \frac{C}{2} \sum_{i=1}^m \xi_i^2 \\
\text{s.t.}\quad y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i,\quad i = 1, \ldots, m.
```

(a) Notice that we have dropped the $`\xi_i \geq 0`$ constraint in the $`\ell_2`$ problem. Show that these non-negativity constraints can be removed. That is, show that the optimal value of the objective will be the same whether or not these constraints are present.

**Answer:**  Consider a potential solution to the above problem with some $`\xi < 0`$. Then the constraint $`y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i`$ would also be satisfied for $`\xi_i = 0`$, and the objective function would be lower, proving that this could not be an optimal solution.

**Explanation:**

For part (a):
- The $\ell_2$ soft margin SVM formulation omits the non-negativity constraint $\xi_i \geq 0$.
- If any $\xi_i < 0$, the constraint $y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i$ is still satisfied for $\xi_i = 0$ (since $1 - \xi_i$ increases as $\xi_i$ decreases), and the objective $\frac{C}{2} \sum \xi_i^2$ is strictly smaller for $\xi_i = 0$ than for $\xi_i < 0$.
- Therefore, at the optimum, $\xi_i \geq 0$ always holds, so the constraint can be omitted without changing the solution.

---

(b) What is the Lagrangian of the $`\ell_2`$ soft margin SVM optimization problem?

**Answer:**

```math
\mathcal{L}(w, b, \xi, \alpha) = \frac{1}{2} w^T w + \frac{C}{2} \sum_{i=1}^m \xi_i^2 - \sum_{i=1}^m \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1 + \xi_i],
```

where $`\alpha_i \geq 0`$ for $`i = 1, \ldots, m`$.

**Explanation:**

For part (b):
- The Lagrangian $\mathcal{L}(w, b, \xi, \alpha)$ is constructed by taking the primal objective and subtracting the constraints multiplied by Lagrange multipliers $\alpha_i \geq 0$.
- The first term $\frac{1}{2} w^T w$ is the regularization on $w$.
- The second term $\frac{C}{2} \sum \xi_i^2$ penalizes the slack variables quadratically.
- The third term $-\sum \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1 + \xi_i]$ enforces the margin constraints.

---

(c) Minimize the Lagrangian with respect to $`w`$, $`b`$, and $`\xi`$ by taking the following gradients: $`\nabla_w \mathcal{L}`$, $`\frac{\partial}{\partial b} \mathcal{L}`$, and $`\nabla_\xi \mathcal{L}`$, and then setting them equal to 0. Here, $`\xi = [\xi_1, \xi_2, \ldots, \xi_m]^T`$.

**Answer:**  Taking the gradient with respect to $`w`$, we get

```math
0 = \nabla_w \mathcal{L} = w - \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)},
```

which gives us

```math
w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}.
```

Taking the derivative with respect to $`b`$, we get

```math
0 = \frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^m \alpha_i y^{(i)},
```

giving us

```math
0 = \sum_{i=1}^m \alpha_i y^{(i)}.
```

Finally, taking the gradient with respect to $`\xi`$, we have

```math
0 = \nabla_{\xi} \mathcal{L} = C\xi - \alpha,
```

where $`\alpha = [\alpha_1, \alpha_2, \ldots, \alpha_m]^T`$. Thus, for each $`i = 1, \ldots, m`$, we get

```math
0 = C\xi_i - \alpha_i \implies C\xi_i = \alpha_i.
```

**Explanation:**

For part (c):
- To find the dual, we minimize the Lagrangian with respect to the primal variables $w$, $b$, and $\xi$ by setting their gradients to zero.
- $\nabla_w \mathcal{L} = 0$ gives $w = \sum \alpha_i y^{(i)} x^{(i)}$ (the optimal $w$ is a linear combination of support vectors).
- $\frac{\partial \mathcal{L}}{\partial b} = 0$ gives $\sum \alpha_i y^{(i)} = 0$ (the weighted sum of labels is zero).
- $\nabla_{\xi} \mathcal{L} = 0$ gives $C \xi_i = \alpha_i$ for each $i$ (relating slack and Lagrange multipliers).

---

(d) What is the dual of the $`\ell_2`$ soft margin SVM optimization problem?

**Answer:**  The objective function for the dual is

```math
\begin{align*}
W(\alpha) &= \min_{w, b, \xi} \mathcal{L}(w, b, \xi, \alpha) \\
&= \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m (\alpha_i y^{(i)} x^{(i)})^T (\alpha_j y^{(j)} x^{(j)}) + \frac{1}{2} \sum_{i=1}^m \frac{\alpha_i}{\xi_i} \xi_i^2 \\
&\quad - \sum_{i=1}^m \alpha_i \left[ y^{(i)} \left( \left( \sum_{j=1}^m \alpha_j y^{(j)} x^{(j)} \right)^T x^{(i)} + b \right) - 1 + \xi_i \right] \\
&= -\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} + \frac{1}{2} \sum_{i=1}^m \alpha_i \xi_i \\
&\quad - \left( \sum_{i=1}^m \alpha_i y^{(i)} \right) b + \sum_{i=1}^m \alpha_i - \sum_{i=1}^m \alpha_i \xi_i \\
&= \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} - \frac{1}{2} \sum_{i=1}^m \alpha_i \xi_i \\
&= \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} - \frac{1}{2} \sum_{i=1}^m \frac{\alpha_i^2}{C}.
\end{align*}
```

Then the dual formulation of our problem is

```math
\begin{align*}
\max_{\alpha} \quad & \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} - \frac{1}{2} \sum_{i=1}^m \frac{\alpha_i^2}{C} \\
s.t. \quad & \alpha_i \geq 0, \quad i = 1, \ldots, m \\
& \sum_{i=1}^m \alpha_i y^{(i)} = 0
\end{align*}
```

**Explanation:**

For part (d):
- Substitute the optimal values for $w$, $b$, and $\xi$ back into the Lagrangian to get the dual objective $W(\alpha)$.
- The quadratic terms in $w$ and $\xi$ expand to sums over $\alpha_i$ and $y^{(i)}$.
- The dual problem is to maximize $\sum \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} - \frac{1}{2} \sum \alpha_i^2 / C$ subject to $\alpha_i \geq 0$ and $\sum \alpha_i y^{(i)} = 0$.
- The $\alpha_i^2 / C$ term comes from the quadratic penalty on the slack variables.

## 3. SVM with Gaussian kernel

Consider the task of training a support vector machine using the Gaussian kernel $`K(x, z) = \exp(-\|x - z\|^2 / \tau^2)`$. We will show that as long as there are no two identical points in the training set, we can always find a value for the bandwidth parameter $`\tau`$ such that the SVM achieves zero training error.


(a) Recall from class that the decision function learned by the support vector machine can be written as

```math
f(x) = \sum_{i=1}^m \alpha_i y^{(i)} K(x^{(i)}, x) + b.
```

Assume that the training data $`\{(x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)})\}`$ consists of points which are separated by at least a distance of $`\epsilon`$; that is, $`\|x^{(j)} - x^{(i)}\| \geq \epsilon`$ for any $`i \neq j`$. Find values for the set of parameters $`\{\alpha_1, \ldots, \alpha_m, b\}`$ and Gaussian kernel width $`\tau`$ such that $`x^{(i)}`$ is correctly classified, for all $`i = 1, \ldots, m`$. [Hint: Let $`\alpha_i = 1`$ for all $`i`$ and $`b = 0`$. Now notice that for $`y \in \{-1, +1\}`$ the prediction on $`x^{(i)}`$ will be correct if $`|f(x^{(i)}) - y^{(i)}| < 1`$, so find a value of $`\tau`$ that satisfies this inequality for all $`i`$.]

**Answer:**  First we set $`\alpha_i = 1`$ for all $`i = 1, \ldots, m`$ and $`b = 0`$. Then, for a training example $`\{x^{(i)}, y^{(i)}\}`$, we get

```math
\left| f(x^{(i)}) - y^{(i)} \right| = \left| \sum_{j=1}^m y^{(j)} K(x^{(j)}, x^{(i)}) - y^{(i)} \right|
```
```math
= \left| \sum_{j=1}^m y^{(j)} \exp\left(-\|x^{(j)} - x^{(i)}\|^2 / \tau^2\right) - y^{(i)} \right|
```
```math
= \left| y^{(i)} + \sum_{j \neq i} y^{(j)} \exp\left(\|x^{(j)} - x^{(i)}\|^2 / \tau^2\right) - y^{(i)} \right|
```
```math
= \left| \sum_{j \neq i} y^{(j)} \exp\left(-\|x^{(j)} - x^{(i)}\|^2 / \tau^2\right) \right|
```
```math
\leq \sum_{j \neq i} \left| y^{(j)} \right| \exp\left(-\|x^{(j)} - x^{(i)}\|^2 / \tau^2\right)
```
```math
= \sum_{j \neq i} \exp\left(-\|x^{(j)} - x^{(i)}\|^2 / \tau^2\right)
```
```math
\leq \sum_{j \neq i} \exp(-\epsilon^2 / \tau^2)
```
```math
= (m-1) \exp(-\epsilon^2 / \tau^2).
```

The first inequality comes from repeated application of the triangle inequality $`|a + b| \leq |a| + |b|`$, and the second inequality (1) from the assumption that $`\|x^{(j)} - x^{(i)}\| \geq \epsilon`$ for all $`i \neq j`$. Thus we need to choose a $`\gamma`$ such that

```math
(m-1) \exp(-\epsilon^2 / \tau^2) < 1,
```

or

```math
\tau < \frac{\epsilon}{\log(m-1)}.
```

By choosing, for example, $`\tau = \epsilon / \log m`$ we are done.

**Explanation:**

For part (a):
- We set $\alpha_i = 1$ for all $i$ and $b = 0$ as suggested by the hint.
- The SVM decision function at a training point $x^{(i)}$ is:

```math
f(x^{(i)}) = \sum_{j=1}^m y^{(j)} K(x^{(j)}, x^{(i)})
```
- Subtracting the true label $y^{(i)}$ and taking the absolute value, we expand the sum and separate the $j = i$ term:

```math
|f(x^{(i)}) - y^{(i)}| = |y^{(i)} + \sum_{j \neq i} y^{(j)} \exp(-\|x^{(j)} - x^{(i)}\|^2 / \tau^2) - y^{(i)}|
```
- The $y^{(i)}$ terms cancel, leaving only the sum over $j \neq i$.
- The triangle inequality is used to bound the sum of absolute values by the sum of the absolute values of each term.
- Since $y^{(j)} \in \{-1, +1\}$, $|y^{(j)}| = 1$.
- The assumption $\|x^{(j)} - x^{(i)}\| \geq \epsilon$ for $i \neq j$ allows us to bound each exponential by $\exp(-\epsilon^2 / \tau^2)$.
- There are $m-1$ such terms, so the total is $(m-1) \exp(-\epsilon^2 / \tau^2)$.
- To ensure $|f(x^{(i)}) - y^{(i)}| < 1$, we require $(m-1) \exp(-\epsilon^2 / \tau^2) < 1$.
- Solving for $\tau$ gives $\tau < \epsilon / \log(m-1)$.
- Choosing $\tau = \epsilon / \log m$ is a valid example.

---

(b) Suppose we run a SVM with slack variables using the parameter $`\tau`$ you found in part (a). Will the resulting classifier necessarily obtain zero training error? Why or why not? A short explanation (without proof) will suffice.

**Answer:**  The classifier will obtain zero training error. The SVM without slack variables will always return zero training error if it is able to find a solution, so all that remains to be shown is that there exists at least one feasible point.
Consider the constraint $`y^{(i)}(w^T x^{(i)} + b)`$ for some $`i`$, and let $`b = 0`$. Then

```math
y^{(i)}(w^T x^{(i)} + b) = y^{(i)} \cdot f(x^{(i)}) > 0
```

since $`f(x^{(i)})`$ and $`y^{(i)}`$ have the same sign, and shown above. Therefore, as we choose all the $`\alpha_i`$'s large enough, $`y^{(i)}(w^T x^{(i)} + b) > 1`$, so the optimization problem is feasible.

**Explanation:**

For part (b):
- The SVM with slack variables may not achieve zero training error if the optimization problem allows for nonzero slack (i.e., misclassifications) to reduce the objective.
- However, if the SVM is run without slack variables (i.e., hard margin), and a feasible solution exists, it will achieve zero training error.
- The answer shows that for the constructed $\alpha_i$ and $b$, the margin constraint $y^{(i)}(w^T x^{(i)} + b) > 0$ is satisfied, and by increasing $\alpha_i$ further, the constraint $y^{(i)}(w^T x^{(i)} + b) > 1$ can be satisfied, so the problem is feasible and zero training error is possible.

---

(c) Suppose we run the SMO algorithm to train an SVM with slack variables, under the conditions stated above, using the value of $`\tau`$ you picked in the previous part, and using some arbitrary value of $`C`$ (which you do not know beforehand). Will this necessarily result in a classifier that achieve zero training error? Why or why not? Again, a short explanation is sufficient.

**Answer:**  The resulting classifier will not necessarily obtain zero training error. The $`C`$ parameter controls the trade-off between minimizing the norm of $`w`$ and minimizing the training error (slack). If the $`C`$ parameter is sufficiently small, then the former component will have relatively little contribution to the objective. In this case, a weight vector which has a very small norm but does not achieve zero training error may achieve a lower objective value than one which achieves zero training error. For example, you can consider the extreme case where $`C = 0`$, and the objective is just the norm of $`w`$. In this case, $`w = 0`$ is the solution to the optimization problem regardless of the choise of $`\tau`$, this this may not obtain zero training error.

**Explanation:**

For part (c):
- When using slack variables, the $`C`$ parameter controls the trade-off between minimizing the norm of $`w`$ and minimizing the training error (slack).
- If $`C`$ is very small, the optimizer may prefer a small $`w`$ (even $`w = 0`$) over achieving zero training error, because the penalty for slack is negligible.
- In the extreme case $`C = 0`$, the solution is $`w = 0`$ regardless of $\tau$, which may not classify all points correctly.
- Therefore, with slack variables and arbitrary $`C`$, zero training error is not guaranteed.

## 4. Naive Bayes and SVMs for Spam Classification (Python Version)

In this question you’ll look into the Naive Bayes and Support Vector Machine algorithms for a spam classification problem using Python. Instead of implementing the algorithms yourself, you’ll use the scikit-learn machine learning library (or similar Python libraries). You should use the provided dataset in the `@/q4` directory if possible. The files are in ARFF format, which can be loaded in Python using the `liac-arff` package or by converting to CSV. If you encounter issues with the provided dataset, you may use a public spam dataset that works with Python, such as the [UCI Spambase dataset](https://archive.ics.uci.edu/ml/datasets/spambase) or the built-in datasets in scikit-learn.

**Instructions:**

- Install required packages: `scikit-learn`, `pandas`, and `liac-arff` (if using ARFF files).
- Load the dataset(s) and preprocess as needed (e.g., convert ARFF to pandas DataFrame).
- Use the same train/test split as in the provided files (e.g., `spam_test.arff` for testing, various `spam_train_*.arff` for training).
- If using a different dataset, use its standard train/test split or create your own.

(a) Train a Naive Bayes classifier (e.g., `sklearn.naive_bayes.MultinomialNB`) on the dataset and report the resulting error rates. Evaluate the performance of the classifier using each of the different training files (but each time using the same test file, e.g., `spam_test.arff`). Plot the error rate of the classifier versus the number of training examples.

Answer: See solution in ./q4_solution folder

(b) Repeat the previous part, but using a Support Vector Machine classifier (e.g., `sklearn.svm.SVC`). How does the performance of the SVM compare to that of Naive Bayes?

Answer: See solution in ./q4_solution folder

*Note: If you use a different dataset, clearly state which dataset you used and how you split the data for training and testing.*

## 5. Uniform convergence

In class we proved that for any finite set of hypotheses $`\mathcal{H} = \{h_1, \ldots, h_k\}`$, if we pick the hypothesis $`\hat{h}`$ that minimizes the training error on a set of $`m`$ examples, then with probability at least $`(1 - \delta)`$,

```math
\varepsilon(\hat{h}) \leq \left( \min_i \varepsilon(h_i) \right) + 2 \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}},
```

where $`\varepsilon(h)`$ is the generalization error of hypothesis $`h_i`$. Now consider a special case (often called the realizable case) where we know, a priori, that there is some hypothesis in our class $`\mathcal{H}`$ that achieves zero error on the distribution from which the data is drawn. Then we could obviously just use the above bound with $`\min_i \varepsilon(h_i) = 0`$; however, we can prove a better bound than this.


(a) Consider a learning algorithm which, after looking at $`m`$ training examples, chooses some hypothesis $`\hat{h} \in \mathcal{H}`$ that makes zero mistakes on this training data. (By our assumption, there is at least one such hypothesis, possibly more.) Show that with probability $`1 - \delta`$

```math
\varepsilon(\hat{h}) \leq \frac{1}{m} \log \frac{k}{\delta}.
```

Notice that since we do not have a square root here, this bound is much tighter. [Hint: Consider the probability that a hypothesis with generalization error greater than $`\gamma`$ makes no mistakes on the training data. Instead of the Hoeffding bound, you might also find the following inequality useful: $`(1 - \gamma)^m \leq e^{-\gamma m}`$.]

**Answer:**  Let $`h \in \mathcal{H}`$ be a hypothesis with true error greater than $`\gamma`$. Then

```math
P(\text{``}h \text{ predicts correctly''}) \leq 1 - \gamma,
```

so

```math
P(\text{``}h \text{ predicts correctly } m \text{ times''}) \leq (1 - \gamma)^m \leq e^{-\gamma m}.
```

Applying the union bound,

```math
P(\exists h \in \mathcal{H}, \text{ s.t. } \varepsilon(h) > \gamma \text{ and ``}h \text{ predicts correct } m \text{ times''}) \leq k e^{-\gamma m}.
```

We want to make this probability equal to $`\delta`$, so we set

```math
k e^{-\gamma m} = \delta,
```

which gives us

```math
\gamma = \frac{1}{m} \log \frac{k}{\delta}.
```

This implies that with probability $`1 - \delta`$,

```math
\varepsilon(\hat{h}) \leq \frac{1}{m} \log \frac{k}{\delta}.
```

(b) Rewrite the above bound as a sample complexity bound, i.e., in the form: for fixed $`\delta`$ and $`\gamma`$, for $`\varepsilon(\hat{h}) \leq \gamma`$ to hold with probability at least $`(1 - \delta)`$, it suffices that $`m \geq f(k, \gamma, \delta)`$ (i.e., $`f(\cdot)`$ is some function of $`k`$, $`\gamma`$, and $`\delta`$).

**Answer:**  From part (a), if we take the equation,

```math
k e^{-\gamma m} = \delta
```

and solve for $`m`$, we obtain

```math
m = \frac{1}{\gamma} \log \frac{k}{\delta}.
```

Therefore, for $`m`$ larger than this, $`\varepsilon(\hat{h}) \leq \gamma`$ will hold with probability at least $`1 - \delta`$.
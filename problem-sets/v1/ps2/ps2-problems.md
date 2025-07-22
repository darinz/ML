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


## 2. $`\ell_2`$ norm soft margin SVMs

In class, we saw that if our data is not linearly separable, then we need to modify our support vector machine algorithm by introducing an error margin that must be minimized. Specifically, the formulation we have looked at is known as the $`\ell_1`$ norm soft margin SVM. In this problem we will consider an alternative method, known as the $`\ell_2`$ norm soft margin SVM. This new algorithm is given by the following optimization problem (notice that the slack penalties are now squared):

```math
\min_{w, b, \xi} \quad \frac{1}{2} \|w\|^2 + \frac{C}{2} \sum_{i=1}^m \xi_i^2 \\
\text{s.t.}\quad y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i,\quad i = 1, \ldots, m.
```

(a) Notice that we have dropped the $`\xi_i \geq 0`$ constraint in the $`\ell_2`$ problem. Show that these non-negativity constraints can be removed. That is, show that the optimal value of the objective will be the same whether or not these constraints are present.

(b) What is the Lagrangian of the $`\ell_2`$ soft margin SVM optimization problem?

(c) Minimize the Lagrangian with respect to $`w`$, $`b`$, and $`\xi`$ by taking the following gradients: $`\nabla_w \mathcal{L}`$, $`\frac{\partial}{\partial b} \mathcal{L}`$, and $`\nabla_\xi \mathcal{L}`$, and then setting them equal to 0. Here, $`\xi = [\xi_1, \xi_2, \ldots, \xi_m]^T`$.

(d) What is the dual of the $`\ell_2`$ soft margin SVM optimization problem?

## 3. SVM with Gaussian kernel

Consider the task of training a support vector machine using the Gaussian kernel $`K(x, z) = \exp(-\|x - z\|^2 / \tau^2)`$. We will show that as long as there are no two identical points in the training set, we can always find a value for the bandwidth parameter $`\tau`$ such that the SVM achieves zero training error.


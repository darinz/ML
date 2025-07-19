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


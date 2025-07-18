# Problem Set #1: Supervised Learning

1. **Newton’s method for computing least squares**

   In this problem, we will prove that if we use Newton’s method solve the least squares optimization problem, then we only need one iteration to converge to $`\theta^*`$.

   (a) Find the Hessian of the cost function $`J(\theta) = \frac{1}{2} \sum_{i=1}^m (\theta^T x^{(i)} - y^{(i)})^2`$.

   (b) Show that the first iteration of Newton’s method gives us $`\theta^* = (X^T X)^{-1} X^T \vec{y}`$, the solution to our least squares problem.

2. **Locally-weighted logistic regression**

   In this problem you will implement a locally-weighted version of logistic regression, where we weight different training examples differently according to the query point. The locally-weighted logistic regression problem is to maximize

```math
\ell(\theta) = -\frac{\lambda}{2} \theta^T \theta + \sum_{i=1}^m w^{(i)} \left[ y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right].
```

   The $`-\frac{\lambda}{2} \theta^T \theta`$ here is what is known as a regularization parameter, which will be discussed in a future lecture, but which we include here because it is needed for Newton’s method to perform well on this task. For the entirety of this problem you can use the value $`\lambda = 0.0001`$.

   Using this definition, the gradient of $`\ell(\theta)`$ is given by

```math
\nabla_\theta \ell(\theta) = X^T z - \lambda \theta
```

   where $`z \in \mathbb{R}^m`$ is defined by

```math
z_i = w^{(i)} (y^{(i)} - h_\theta(x^{(i)}))
```

   and the Hessian is given by

```math
H = X^T D X - \lambda I
```

   where $`D \in \mathbb{R}^{m \times m}`$ is a diagonal matrix with

```math
D_{ii} = -w^{(i)} h_\theta(x^{(i)})(1 - h_\theta(x^{(i)}))
```

   For the sake of this problem you can just use the above formulas, but you should try to derive these results for yourself as well.

   Given a query point $`x`$, we choose compute the weights

```math
w^{(i)} = \exp \left( - \frac{\|x - x^{(i)}\|^2}{2\tau^2} \right).
```

Much like the locally weighted linear regression that was discussed in class, this weighting
scheme gives more when the “nearby” points when predicting the class of a new example.
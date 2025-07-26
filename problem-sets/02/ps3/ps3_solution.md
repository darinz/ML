# Problem Set 3 Solutions

## 1. Gradient Descent

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

**Solution:**
**Proof:**
For any positive integer $`k`$, $`x^{(k)} = x^{(k-1)} - \eta \nabla f(x)`$, according to the gradient descent algorithm.
By hint(1), we have
```math
f(x^{(k)}) \le f(x^{(k-1)}) + \nabla f(x^{(k-1)}) (x^{(k)} - x^{(k-1)}) + \frac{L}{2} ||x^{(k)} - x^{(k-1)}||^2
```
```math
= f(x^{(k-1)}) - \eta \nabla f(x^{(k-1)})^2 + \frac{L}{2} \eta^2 \nabla f(x^{(k-1)})^2
```
```math
\le f(x^{(k-1)}) + (-\eta + \frac{L}{2}\eta^2) \nabla f(x^{(k-1)})^2 \quad (\text{Since } \eta \le \frac{1}{L})
```
```math
= f(x^{(k-1)}) - \frac{\eta}{2} \nabla f(x^{(k-1)})^2
```
```math
\le f(x^*) + \nabla f(x^{(k-1)}) (x^{(k-1)} - x^*) - \frac{\eta}{2} \nabla f(x^{(k-1)})^2 \quad (\text{By hint(2)})
```
```math
= f(x^*) + \frac{1}{2\eta} (2\eta \nabla f(x^{(k-1)}) (x^{(k-1)} - x^*) - \eta^2 \nabla f(x^{(k-1)})^2)
```
```math
\le f(x^*) + \frac{1}{2\eta} (||x^{(k-1)} - x^*||^2 - ||x^{(k-1)} - \eta \nabla f(x^{(k-1)}) - x^*||^2) \quad (\text{By hint(3)})
```
```math
= f(x^*) + \frac{1}{2\eta} (||x^{(k-1)} - x^*||^2 - ||x^{(k)} - x^*||^2)
```
Hence, we have
```math
f(x^{(k)}) - f(x^*) \le \frac{1}{2\eta} (||x^{(k-1)} - x^*||^2 - ||x^{(k)} - x^*||^2)
```
Adding up from 1 to k:
```math
\sum_{i=1}^k [f(x^{(i)}) - f(x^*)] \le \sum_{i=1}^k \frac{1}{2\eta} (||x^{(i-1)} - x^*||^2 - ||x^{(i)} - x^*||^2)
```
```math
\sum_{i=1}^k f(x^{(i)}) - kf(x^*) \le \frac{1}{2\eta} (||x^{(0)} - x^*||^2 - ||x^{(k)} - x^*||^2) \le \frac{1}{2\eta} (||x^{(0)} - x^*||^2)
```
Since $`f(x^{(k)}) \le f(x^{(k-1)})`$, $`f(x^{(k)}) \le \frac{1}{k} \sum_{i=1}^k f(x^{(i)})`$
Hence,
```math
f(x^{(k)}) - f(x^*) \le \frac{1}{2k\eta} (||x^{(0)} - x^*||^2)
```

## 2. Stochastic Gradient Descent

Consider minimizing an average of functions:
```math
\min_w \frac{1}{n} \sum_{i=1}^n l_i(w),
```
where $`w`$ is a $`d`$-dimensional vector (or the feature dimension is $`d`$). The minimization of the negative of a log-likelihood function can serve as an example. Recall that the (full) gradient descent step is given by
```math
w^{(t+1)} = w^{(t)} - \eta \cdot \frac{1}{n} \sum_{i=1}^n \nabla l_i(w^{(t)}).
```
The computational cost of a single step here is $`O(dn)`$. To reduce cost, one idea is to just use a subset of all samples to approximate the full gradient. Specifically, consider revising the gradient descent step as follows:
```math
w^{(t+1)} = w^{(t)} - \eta \cdot \nabla l_{I_t}(w^{(t)}),
```
where $`I_t`$ is chosen randomly within $`\{1, 2,..., n\}`$ with equal probabilities. This is called **stochastic gradient descent (SGD)**, and the computational cost of a single step now reduces to $`O(d)`$.

(a) The following two results provide intuitions or foundations for why SGD works.
*   $`E_{I_t} (\nabla l_{I_t} (w^{(t)})) = \frac{1}{n} \sum_{i=1}^n \nabla l_i (w^{(t)}),`$ which is the full gradient. Hence the estimate of gradient is unbiased.
*   Let $`l(w) = \frac{1}{n} \sum_i l_i(w)`$ and $`w^* = \text{arg min}_w l(w)`. Assume $`||w^{(1)} - w^*||^2 \le R`$ and $`\sup_w \max_i ||\nabla l_i(w)||^2 \le G`$. Then
    ```math
    E[l(\bar{w}) - l(w^*)] \le \sqrt{\frac{RG}{T}},
    ```
    where $`\bar{w} := \frac{1}{T} \sum_{t=1}^T w^{(t)}`$. Therefore, the expected error over $`T`$ iterations is $`O(\frac{1}{\sqrt{T}})`. (The proof of this result is provided in the solution part for reference.)

**Solution:**
We first consider deriving an upper bound for $`E(l(w^{(t)}) - l(w^*)):`$
```math
E||w^{(t+1)} - w^*||^2 = E||w^{(t)} - \eta\nabla l_{I_t}(w^{(t)}) - w^*||^2 \\
= E||w^{(t)} - w^*||^2 - 2\eta E(\nabla l_{I_t}(w^{(t)})^T (w^{(t)} - w^*)) + \eta^2 E||\nabla l_{I_t}(w^{(t)})||^2 \\
\le E||w^{(t)} - w^*||^2 - 2\eta E(\nabla l_{I_t}(w^{(t)})^T (w^{(t)} - w^*)) + \eta^2 G \\
\le E||w^{(t)} - w^*||^2 - 2\eta E(l(w^{(t)}) - l(w^*)) + \eta^2 G,
```
where the last inequality holds because of the following:
```math
E(\nabla l_{I_t}(w^{(t)})^T (w^{(t)} - w^*)) \\
= EE [\nabla l_{I_t}(w^{(t)})^T (w^{(t)} - w^*)|I_1, ..., I_{t-1}, w^{(t-1)}] \\
= \frac{1}{n} \sum_i \nabla l_i(w^{(t)})^T (w^{(t)} - w^*) \\
= E\nabla l(w^{(t)})^T (w^{(t)} - w^*) \\
\ge E(l(w^{(t)}) - l(w^*)),
```
where the last inequality holds from the convexity of $`\ell(\cdot)`$. Furthermore, in the right-hand side of starred equality above, the outer expectation is over the variables $`I_1, \dots, I_{t-1}`$ and $`w^{(1)}, \dots, w^{(t-1)}`$, and the inner expectation is over $`I_t, w^{(t)}`$ conditioned on the other variables.

Therefore, we've proved that $`E||w^{(t+1)} - w^*||^2 \le E||w^{(t)} - w^*||^2 - 2\eta E(\ell(w^{(t)}) - \ell(w^*)) + \eta^2 G`$, which implies (from rearrangement) that
```math
E(\ell(w^{(t)}) - \ell(w^*)) \le \frac{1}{2\eta} (E||w^{(t)} - w^*||^2 - E||w^{(t+1)} - w^*||^2 + \eta^2 G).
```
Now note that the convexity of $`\ell`$ and Jensen's inequality ensure that $`\ell(\bar{w}) \le \frac{1}{T} \sum_{t=1}^T \ell(w^{(t)})`$, which implies
```math
E(\ell(\bar{w}) - \ell(w^*)) \le \frac{1}{T} \sum_t E(\ell(w^{(t)}) - \ell(w^*)).
```
From (1) and (2), we have
```math
E(\ell(\bar{w}) - \ell(w^*)) \le \frac{1}{T} \sum_t \left( \frac{1}{2\eta} (E||w^{(t)} - w^*||^2 - E||w^{(t+1)} - w^*||^2 + \eta^2 G) \right)
```
```math
= \frac{1}{2\eta T} (E||w^{(1)} - w^*||^2 - E||w^{(T+1)} - w^*||^2) + \frac{\eta G}{2}
```
```math
\le \frac{1}{2\eta T} E||w^{(1)} - w^*||^2 + \frac{\eta G}{2}
```
```math
\le \frac{R}{2\eta T} + \frac{\eta G}{2}
```
```math
= \sqrt{\frac{RG}{T}},
```
where the last equality holds by choosing $`\eta = \sqrt{\frac{R}{GT}}`$.

(b) What disadvantages can SGD have? How can we balance between the noise in updates and computational cost?

**Solution:**
By treating SGD as noise-injected gradient descent:
```math
\nabla \ell_{I_t}(w^{(t)}) = E_{I_t} \nabla \ell_{I_t}(w^{(t)}) + e_t = \frac{1}{n} \sum_{i=1}^n \ell_i(w^{(t)}) + e_t,
```
where $`e_t`$ represents the noise term and is random, we know that the steps taken towards a minimum can be very noisy because the gradient used in updating involves noise. One way to balance the noise in updates and computational cost is to consider a technique called **mini-batching**, which is employed with SGD.

## 3. Extensions of SGD

(a) Gradient descent requires the full gradient when updating while (standard) SGD utilizes the gradient of one sample when updating. **Mini-batching** is somewhere between the two extremes. That is, we choose a random subset $`I_t \subseteq \{1, ..., n\}`$ with size $`|I_t| = b \ll n`$ in the stochastic gradient descent step:
```math
w^{(t+1)} = w^{(t)} - \eta \cdot \frac{1}{b} \sum_{i \in I_t} \nabla l_i (w^{(t)}).
```

With mini-batching, we have the following results:
*   $`E_{I_t} \left( \frac{1}{b} \sum_{i \in I_t} \nabla l_i (w^{(t)}) \right) = \frac{1}{n} \sum_{i=1}^n \nabla l_i (w^{(t)})`$: we still have an unbiased estimate of the full gradient.
*   Compared to standard SGD, variance of the gradient estimate is reduced approximately by $`\frac{1}{b}`$.
*   Computational cost for each step now becomes $`O(db)`$.
Remark: By matrix computations (computing $`b`$ gradients at a time) and parallelization, we can denoise the estimated gradients without increasing much computational cost (for batch size $`b`$ that is not large).

(b) How should we choose the batch size?

**Solution:**
The choice of the optimal batch size is not an easy question, and there is no standard answer to it. However, we still try to provide some important intuitions regarding the choice of batch size. Firstly, when the objective function (to be minimized) behaves "better" (e.g., Lipschitz continuous, strong convex) than convex functions, the difference in the convergence rates between GD and SGD becomes significant, suggesting a nontrivial gain of having a faster convergence rate and hence we should consider relatively larger batch size. Secondly, a smaller batch size yields less stable gradient estimates, suggesting that we shall employ a fairly small step size/learning rate. An increase in the batch size can be paired with an increase in the step size/learning rate.

(c) Are there other extensions or variants of the basic stochastic gradient descent algorithm?

**Solution:**
Many improvements, which are listed below, on the basic SGD algorithm have been developed and used.
*   Implicit updates (ISGD)
*   Momentum
*   Averaged stochastic gradient descent
*   Adaptive gradient algorithm (AdaGrad)
*   Root Mean Square Propagation (RMSProp)
*   Adaptive Moment Estimation (Adam)
Basically, these methods consider to fine-tune the step size parameter, take previous update magnitude into account, or introduce the second moments of the gradients when updating. For example, Momentum remembers the previous update magnitude so that $`w^{(t)}`$ tends to keep traveling in the same direction, preventing oscillations:
```math
w^{(t+1)} = w^{(t)} - \eta \nabla l_t (w^{(t)}) + \alpha (w^{(t)} - w^{(t-1)}).
```
Adam, as another example, considers to tune to step size with the second moments of the gradients:
```math
w^{(t+1)} = w^{(t)} - \eta G(\nabla l_t^{(t)}, \nabla l_t^{(t-1)}, ..., (\nabla l_t^{(t)})^2, (\nabla l_t^{(t-1)})^2, ...),
```
where $`\nabla l_t^{(t)} = \nabla l_{I_t} (w^{(t)})`$ and $`G`$ is a function that involves element-wise square of all previous gradients.

The paper below provides more details on Adam:
https://arxiv.org/pdf/1412.6980.pdf

# Regularization and model selection

## 9.1 Regularization

Recall that as discussed in Section 8.1, overfitting is typically a result of using too complex models, and we need to choose a proper model complexity to achieve the optimal bias-variance tradeoff. Overfitting happens when a model learns not only the underlying pattern in the data but also the noise, leading to poor performance on new, unseen data. To prevent this, we want our models to be just complex enough to capture the true structure, but not so complex that they memorize the training data.

When the model complexity is measured by the number of parameters, we can vary the size of the model (e.g., the width of a neural net). However, the correct, informative complexity measure of the models can be a function of the parameters (e.g., $`\ell_2`$ norm of the parameters), which may not necessarily depend on the number of parameters. For example, two neural networks might have the same number of parameters, but one might have much larger weights, making it more likely to overfit. In such cases, we will use regularization, an important technique in machine learning, to control the model complexity and prevent overfitting.

Regularization typically involves adding an additional term, called a regularizer and denoted by $`R(\theta)`$ here, to the training loss/cost function. The idea is to penalize models that are too complex, encouraging the learning algorithm to find simpler, more generalizable solutions:

```math
J_\lambda(\theta) = J(\theta) + \lambda R(\theta)
```

```python
# Python code to compute regularized loss
# J(theta): original loss (float)
# R(theta): regularizer (float)
# lambda_: regularization parameter (float)
J_lambda = J_theta + lambda_ * R_theta
```

Here $`J_\lambda`$ is often called the regularized loss, and $`\lambda \geq 0`$ is called the regularization parameter. The regularizer $`R(\theta)`$ is a nonnegative function (in almost all cases). In classical methods, $`R(\theta)`$ is purely a function of the parameter $`\theta`$, but some modern approaches allow $`R(\theta)`$ to depend on the training dataset.\

**Intuitive explanation:**
- $`J(\theta)`$ measures how well the model fits the training data (e.g., mean squared error for regression).
- $`R(\theta)`$ measures the complexity of the model (e.g., the size of the weights).
- $`\lambda`$ controls the tradeoff: a larger $`\lambda`$ means we care more about keeping the model simple, while a smaller $`\lambda`$ means we care more about fitting the training data.

The regularizer $`R(\theta)`$ is typically chosen to be some measure of the complexity of the model $`\theta`$. Thus, when using the regularized loss, we aim to find a model that both fits the data (a small loss $`J(\theta)`$) and has a small model complexity (a small $`R(\theta)`$). The balance between the two objectives is controlled by the regularization parameter $`\lambda`$. When $`\lambda = 0`$, the regularized loss is equivalent to the original loss. When $`\lambda`$ is a sufficiently small positive number, minimizing the regularized loss is effectively minimizing the original loss with the regularizer as the tie-breaker. When the regularizer is extremely large, then the original loss is not effective (and likely the model will have a large bias.)

**Why does this help?**
- Without regularization, a model might fit the training data perfectly but fail to generalize to new data (overfitting).
- With regularization, the model is encouraged to be simpler, which often leads to better generalization.

The most commonly used regularization is perhaps $`\ell_2`$ regularization, where
```math
R(\theta) = \frac{1}{2} \|\theta\|_2^2
```
```python
# Python code for L2 regularization (R(theta) = 0.5 * ||theta||_2^2)
import numpy as np
R_theta = 0.5 * np.sum(theta ** 2)
```
This means we penalize the sum of the squares of the weights. It encourages the optimizer to find a model with small $`\ell_2`$ norm. In deep learning, it's oftentimes referred to as **weight decay**, because gradient descent with learning rate $`\eta`$ on the regularized loss $`R_\lambda(\theta)`$ is equivalent to shrinking/decaying $`\theta`$ by a scalar factor of $`1 - \eta \lambda`$ each step alongside the standard gradient.

**Analogy:** Think of regularization as a way to "keep your model on a leash"—it can still learn, but it can't wander too far into overly complex solutions.

Besides encouraging simpler models, regularization can also impose inductive biases or structures on the model parameters. For example, suppose we had a prior belief that the number of non-zeros in the ground-truth model parameters is small\(^2\)—which is oftentimes called sparsity of the model—we can impose a regularization on the number of non-zeros in $`\theta`$, denoted by $`\|\theta\|_0`$, to leverage such a prior belief. Imposing additional structure of the parameters narrows our search space and makes the complexity of the model family smaller—e.g., the family of sparse models can be thought of as having lower complexity than the family of all models—and thus tends to lead to a better generalization. On the other hand, imposing additional structure may risk increasing the bias. For example, if we regularize the sparsity strongly but no sparse models can predict the label accurately, we will suffer from large bias (analogously to the situation when we use linear models to learn data than can only be represented by quadratic functions in Section 8.1.)

**Sparsity explained:**
- Sparsity means most of the parameters are zero. This is useful when we believe only a few features are truly important.
- $`\|\theta\|_0`$ counts the number of nonzero elements in $`\theta`$.

```python
# Python code for L0 "norm" (counts nonzero elements)
num_nonzero = np.count_nonzero(theta)
```

The sparsity of the parameters is not a continuous function of the parameters, and thus we cannot optimize it with (stochastic) gradient descent. A common relaxation is to use
```math
R(\theta) = \|\theta\|_1
```
```python
# Python code for L1 regularization (R(theta) = ||theta||_1)
R_theta = np.sum(np.abs(theta))
```
as a continuous surrogate.$`^3`$

The $`R(\theta) = \|\theta\|_1`$ (also called LASSO) and
```math
R(\theta) = \frac{1}{2}\|\theta\|_2^2
```
```python
# Python code for L2 regularization (again, for clarity)
R_theta = 0.5 * np.sum(theta ** 2)
```
are perhaps among the most commonly used regularizers for linear models. Other norm and powers of norms are sometimes also used. The $`\ell_2`$ norm regularization is much more commonly used with kernel methods because $`\ell_1`$ regularization is typically not compatible with the kernel trick (the optimal solution cannot be written as functions of inner products of features.)

**Practical note:**
- $`\ell_1`$ regularization encourages sparsity (many weights become exactly zero).
- $`\ell_2`$ regularization encourages small weights (but not exactly zero).
- LASSO (Least Absolute Shrinkage and Selection Operator) is a popular method using $`\ell_1`$ regularization.

In deep learning, the most commonly used regularizer is $`\ell_2`$ regularization or weight decay. Other common ones include dropout (randomly turning off neurons during training), data augmentation (expanding the training set with modified data), regularizing the spectral norm of the weight matrices, and regularizing the Lipschitzness of the model, etc. Regularization in deep learning is an active research area, and it's known that there is another implicit source of regularization, as discussed in the next section.

## 9.2 Implicit regularization effect

The implicit regularization effect of optimizers, or implicit bias or algorithmic regularization, is a new concept/phenomenon observed in the deep learning era. It largely refers to the fact that the optimizers can implicitly impose structures on parameters beyond what has been imposed by the regularized loss.

**What does this mean?**
- Even if we don't explicitly add a regularization term, the way we train our models (the optimizer, learning rate, initialization, etc.) can still influence which solution we end up with.
- This is especially important in deep learning, where the loss surface is highly non-convex and there are many global minima.

In most classical settings, the loss or regularized loss has a unique global minimum, and thus any reasonable optimizer should converge to that global minimum and cannot impose any additional preferences. However, in deep learning, oftentimes the loss or regularized loss has more than one (approximate) global minima, and different optimizers may converge to different global minima. Though these global minima have the same or similar training losses, they may be of different nature and have dramatically different generalization performance. See Figures 9.1 and 9.2 and its caption for an illustration and some experiment results. For example, it's possible that one global minimum gives a much more Lipschitz or sparse model than others and thus has a better test error. It turns out that many commonly-used optimizers (or their components) prefer or bias towards finding global minima of certain properties, leading to a better test performance.

**Why does this matter?**
- Two models can have the same training loss but very different test performance, depending on which minimum the optimizer finds.
- The optimizer's "implicit bias" can help us find solutions that generalize better, even without explicit regularization.

<img src="./img/global_minima.png" width="350px"/>

**Figure 9.1:** An illustration that different global minima of the training loss can have different test performance. (The figure shows two global minima: one with both low training and test loss, and another with low training but higher test loss.)

<img src="./img/neural_networks_trained.png" width="700px"/>

**Figure 9.2:** **Left:** Performance of neural networks trained by two different learning rates schedules on the CIFAR-10 dataset. Although both experiments used exactly the same regularized losses and the optimizers fit the training data perfectly, the models' generalization performance differ much. **Right:** On a different synthetic dataset, optimizers with different initializations have the same training error but different generalization performance.$^4$

**Key takeaway:**
- The choice of optimizer does not only affect minimizing the training loss, but also imposes implicit regularization and affects the generalization of the model. Even if your current optimizer already converges to a small training error perfectly, you may still need to tune your optimizer for a better generalization.

One may wonder which components of the optimizers bias towards what type of global minima and what type of global minima may generalize better. These are open questions that researchers are actively investigating. Empirical and theoretical research have offered some clues and heuristics. In many (but definitely far from all) situations, among those setting where optimization can succeed in minimizing the training loss, the use of larger initial learning rate, smaller initialization, smaller batch size, and momentum appears to help with biasing towards more generalizable solutions. A conjecture (that can be proven in certain simplified case) is that stochasticity in the optimization process helps the optimizer to find flatter global minima (global minima where the curvature of the loss is small), and flat global minima tend to give more Lipschitz models and better generalization. Characterizing the implicit regularization effect formally is still a challenging open research question.

**In summary:**
- Regularization is not just about adding penalty terms to the loss function. The way we train our models (choice of optimizer, learning rate, batch size, etc.) can also have a big impact on generalization.
- Understanding both explicit and implicit regularization is key to building models that perform well on new data.

---

4. The setting is the same as in Woodworth et al. (2020), HaoChen et al. (2020)

2. For linear models, this means the model just uses a few coordinates of the inputs to make an accurate prediction.

3. There has been a rich line of theoretical work that explains why $`\|\theta\|_1`$ is a good surrogate for encouraging sparsity, but it's beyond the scope of this course. An intuition is: assuming the parameter is on the unit sphere, the parameter with smallest $`\ell_1`$ norm also
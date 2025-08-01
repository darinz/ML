# Practice 9 Solutions

**1. One Answer**

In a popular gacha game, the probability of pulling an SSR character on a single pull is 0.6% (P = 0.006). Assume that each pull is independent and follows a Bernoulli distribution. In such games, players often perform a 10-pull, which means making 10 pulls. Each of these 10 pulls is still independent, meaning the probability of getting an SSR in each pull remains 0.006. If you perform a 10-pull, what is the probability of pulling exactly 2 SSRs? You do not need to know what gacha game is to solve this problem.

*   (a) 0
*   (b) $1 - (1 - P)^{10} = 5.84\%$
*   (c) $(1 - P)^9 \times P \times 10 = 5.68\%$
*   (d) $(1 - P)^8 \times P^2 \times (10 \times 9 / 2) = 0.15\%$

**Correct answers:** (d)

**Explanation:** We model each pull as a Bernoulli trial, where the probability of success (pulling an SSR) is P = 0.006. Since a 10-pull consists of 10 independent trials, the number of SSRs obtained follows a Binomial distribution:

$X \sim \text{Binomial}(n = 10, p = 0.006)$

We want to find the probability of pulling exactly 2 SSRs, which is given by the binomial probability mass function (PMF):

$P(X = 2) = \binom{10}{2} P^2 (1 - P)^8$

Computing the combination: $\binom{10}{2} = \frac{10!}{2! \times (10 - 2)!} = \frac{10 \times 9}{2} = 45$

Thus, the probability is: $P(X = 2) = (1 - P)^8 \times P^2 \times 45 = 0.15\%$

Therefore, the correct answer is D.

**Why other options are incorrect:**

**A: 0** - This option would be correct if it were impossible to pull an SSR. However, since the probability of obtaining an SSR is nonzero, this option is incorrect.

**B: $1 - (1 - P)^{10} = 5.84\%$** - This formula calculates the probability of pulling at least 1 SSR, computed as: $P(\text{at least 1 SSR}) = 1 - P(0 \text{ SSRs}) = 1 - (1 - P)^{10}$. This probability includes cases where the player gets 1, 2, 3, ..., or even 10 SSRs.

**C: $(1 - P)^9 \times P \times 10 = 5.68\%$** - This formula calculates the probability of pulling exactly 1 SSR, given by: $P(X = 1) = \binom{10}{1} P^1(1 - P)^9 = 10 \times P \times (1 - P)^9$. Hence, the correct answer remains D.

**2. Select All That Apply**

Below are several statements about Gradient Descent (GD) and Stochastic Gradient Descent (SGD). Which of the following are correct?

*   (a) For GD, each step aims to move along the gradient descent direction at the current point to reduce the value of the objective function.
*   (b) In SGD, each step computes an estimated gradient based on a single sample, introducing randomness, which may not guarantee that the objective function decreases in every step.
*   (c) Suppose you have model $w_t$ at the $t$-th iteration of SGD. The expectation of the direction of the model update for SGD at step $t$ is different from the negative direction of the gradient $-\nabla_w f(w)|_{w=w_t}$.
*   (d) GD requires the full gradient information of the objective function, while SGD only needs the gradient information on a single sample at each step.

**Correct answers:** (a), (b), (d)

**Explanation:** The correct answers are (A), (B), and (D). Below is an explanation of each option.

**(A) Correct:** In Gradient Descent (GD), each step moves in the direction of the negative gradient of the objective function at the current point. This ensures that the function value decreases (assuming an appropriate step size). Mathematically, the update rule for GD is:

$w_{t+1} = w_t - \eta \nabla_w f(w_t)$

where $\eta$ is the learning rate, and $\nabla_w f(w_t)$ is the gradient of the objective function over the entire dataset at iteration $t$. This step guarantees movement in the direction that minimizes the objective function.

**(B) Correct:** In Stochastic Gradient Descent (SGD), instead of computing the exact gradient using the full dataset, an estimated gradient is computed based on a single sample (or a mini-batch). This introduces randomness in the updates, meaning that the objective function might not necessarily decrease in every step. The update rule for SGD can be rewritten as:

$w_{t+1} = w_t - \eta(\nabla_w f(w_t) + \xi_t)$

where $\nabla_w f(w_t)$ is the true full-batch gradient, and $\xi_t$ is a noise term introduced due to the stochastic nature of SGD, representing the deviation from the full gradient when using only one sample. This noise term makes each individual update potentially non-optimal, but in expectation, the updates still align with the true gradient direction over multiple iterations.

**(C) Incorrect:** The expectation of the SGD update direction is equal to the negative full gradient:

$E[\nabla_w f(w_t) + \xi_t] = \nabla_w f(w_t)$

Since SGD approximates the full gradient using randomly sampled data points, its update direction is an unbiased estimate of the true gradient. That is, while each step introduces noise, on average, the update follows the same direction as GD. Therefore, the claim that the expectation is different from $-\nabla_w f(w_t)$ is false.

**(D) Correct:** GD requires the full gradient of the objective function, meaning it computes $\nabla_w f(w)$ over the entire dataset at each step. In contrast, SGD only uses the gradient of a single sample (or a mini-batch), significantly reducing the computational cost per step, especially in large-scale datasets.

Therefore, the correct answers are (A), (B), and (D).

**3.**

In a gacha game, the probability of obtaining an SSR character per pull is $p$, but $p$ is unknown. To estimate $p$, Bob performed 100 pulls and obtained SSRs $k = 3$ times (i.e., 3 successes). Assume that each pull is independent and follows a Bernoulli distribution.

What is the likelihood of this scenario occurring?

Likelihood function: $L(p) = \frac{100 \times 99 \times 98}{6} p^3 (1-p)^{97}$

What is the Maximum Likelihood Estimate (MLE) of $p$ as a fractional number?

MLE: $\hat{p} = \frac{3}{100}$

**Explanation:** Since each pull is independent and follows a Bernoulli distribution, the total number of SSRs obtained follows a Binomial distribution:

$X \sim \text{Binomial}(n = 100, p)$

The likelihood function is given by the binomial probability mass function:

$L(p) = \binom{100}{3} p^3 (1-p)^{97} = \frac{100 \times 99 \times 98}{6} p^3 (1-p)^{97}$

To find the MLE $\hat{p}$, we maximize $\ln L(p)$:

$\ln L(p) = \text{constant} + 3 \ln p + 97 \ln(1-p)$

Taking the derivative and setting it to zero:

$\frac{3}{p} - \frac{97}{1-p} = 0$

Solving for p:

$\hat{p} = \frac{3}{100}$

**4. One Answer**

Suppose you train a linear regression model (without doing feature expansion), i.e., $f_w(x) = wx + b$, to approximate the cubic function $g(x) = 2x^3 + 7x^2 + 4x + 3$. What's the most likely outcome?

*   (a) The model will have low bias and low variance
*   (b) The model will have low bias and high variance
*   (c) The model will have high bias and low variance
*   (d) The model will have high bias and high variance

**Correct answers:** (c)

**Explanation:** The model being linear means the variance is likely to be low. The linear model trying to approximate a cubic function, which is of a higher degree, means that the bias is likely to be high.

**5. One Answer**

Adding more basis functions to a linear regression model always leads to better prediction accuracy on new, unseen data.

*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:** As the complexity of the model increases, the prediction accuracy on new, unseen data (test data) doesn't always get better as the model may overfit.

**6. One Answer**

What datasets from the training/validation/test data split should you utilize during hyperparameter tuning?

*   (a) Training Data
*   (b) Training Data, Validation Data
*   (c) Training Data, Validation Data, Test Data
*   (d) Training Data, Test Data

**Correct answers:** (b)

**Explanation:** You never want to use your Test Data for hyperparameter tuning, as it will bias the model on the test data. The test data should only be used for final evaluation of the model. Instead, for hyperparameter tuning, you want to train your model with different hyperparameters using the Training Data, then evaluate those models on the Validation Data to select the best hyperparameters for the model. (Methods like K-fold Cross Validation)

**7. One Answer**

Consider $u = \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix}$, $v = \begin{bmatrix} -4 \\ 5 \\ 1 \end{bmatrix}$, $w = \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}$. Let $x \in \mathbb{R}^3$. Does there exist unique $a, b, c \in \mathbb{R}$ such that $a \cdot u + b \cdot v + c \cdot w = x$?

*   (a) Yes
*   (b) No
*   (c) Not enough information to determine

**Correct answers:** (a)

**Explanation:** $u, v, w$ are linearly independent, so the function $f(a, b, c) = a \cdot u + b \cdot v + c \cdot w$ is onto $\mathbb{R}^3$ as well as one-to-one. This can be verified by using row reduction on $\begin{bmatrix} u & v & w \end{bmatrix}$ or by noticing the three vectors are orthogonal.

**8.**

Consider data matrix $X \in \mathbb{R}^{n \times d}$, label vector $y \in \mathbb{R}^n$, and regularization parameter $\lambda > 0$. Write the closed form solution for ridge regression.

**Answer:** $(X^T X + \lambda I)^{-1} X^T y$

**Explanation:** $(X^T X + \lambda I)^{-1} X^T y$

**9. One Answer**

Consider a dataset containing three observations for a simple linear regression problem, where $y$ is the dependent variable and $x$ is the independent variable. The dataset is given as follows:

| x | y |
|---|---|
| 1 | 7 |
| 2 | 8 |
| 3 | 9 |

Find the coefficient $\beta_1$ of the linear regression (without bias) $y = \beta_1 x$ using the least squares as loss.

*   (a) $\frac{46}{14}$
*   (b) $\frac{14}{46}$
*   (c) $\frac{50}{14}$
*   (d) $\frac{14}{50}$

**Correct answers:** (c)

**Explanation:** $\beta_1 = \frac{50}{14}$

$\beta_1 = (X^T X)^{-1} (X^T Y) = \frac{1}{14} \times 50$

**10. One Answer**

We can find the solution for LASSO by setting the gradient of the loss to 0 and solving for weight parameter $w$.

*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:** LASSO has no closed form solution, which is why we use gradient descent.

**11. One Answer**

You are building a model to detect fraudulent transactions from a dataset of 100K samples. What would be the most effective way to split and utilize your data?

*   (a) Randomly take an 80-20 data split. Use 80% of the data for training, and 20% for both validation and evaluation.
*   (b) Use the first 80% of the data for training, the next 10% for validation, and the last 10% for evaluation.
*   (c) Randomly make a 80-10-10 data split. Use 80% of the data for training, 10% for validation, and 10% for evaluation.
*   (d) Select a random 80% of the data for training, use the remaining 20% for validation. Evaluate on the training set.

**Correct answers:** (c)

**Explanation:** A is incorrect since the validation and test set should be separate. B is incorrect since the data splits should be randomized. C is the correct as it is the standard data split method. D is incorrect since evaluating on the known train set induces bias.

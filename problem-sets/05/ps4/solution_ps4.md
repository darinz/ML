# Problem Set 4 Solutions

## Problem 1: Short Answers [24 points]

### (a) [5 points] Optimization Update Rule

**Problem:** Given a cost function $J(\theta)$ that we seek to minimize and $\alpha \in \mathbb{R} > 0$, consider the following update rule:

$$\theta^{(t+1)} = \arg\min_{\theta} \left\{ J(\theta^{(t)}) + \nabla_{\theta^{(t)}} J(\theta^{(t)})^T (\theta - \theta^{(t)}) + \frac{1}{2\alpha} \|\theta - \theta^{(t)}\|^2 \right\}$$

**(i) [3 points]** Show that this yields the same $\theta^{(t+1)}$ as the gradient descent update with step size $\alpha$.

**Answer:**

Denote $U(\theta) = J(\theta^{(t)}) + \nabla_{\theta^{(t)}} J(\theta^{(t)})^T (\theta - \theta^{(t)}) + \frac{1}{2\alpha} \|\theta - \theta^{(t)}\|^2$.

To find the minimum over $\theta$, we compute the gradient of $U(\theta)$ w.r.t. $\theta$ and set it to 0:

$$\nabla_{\theta} U(\theta) = 0$$
$$\nabla_{\theta^{(t)}} J(\theta^{(t)}) + \frac{1}{2\alpha} (-2\theta^{(t)} + 2\theta) = 0$$
$$\alpha\nabla_{\theta^{(t)}} J(\theta^{(t)}) - \theta^{(t)} + \theta = 0$$
$$\Rightarrow \theta = \theta^{(t)} - \alpha\nabla_{\theta^{(t)}} J(\theta^{(t)})$$

which is the gradient descent update, as desired.

To confirm this is a minimum, we compute the Hessian $\nabla_{\theta}^2 U = \frac{1}{\alpha}I$ which is positive definite as expected.

**(ii) [2 points]** Provide a sketch (i.e. draw a picture) of the above update for the simplified case where $\theta \in \mathbb{R}$, $J(\theta) = \theta$, and $\theta^{(t)} = 1$. Make sure to clearly label $\theta^{(t)}$, $\theta^{(t+1)}$ and $\alpha$.

**Answer:**

We provide an example sketch for $\alpha = 1$. Note that $\alpha = \theta^{(t)} - \theta^{(t+1)}$ since $\nabla J(\theta) = 1$.

<img src="./q1-a-ii_solution.png" width="250px">

### (b) [4 points] Loss Functions in Binary Classification

**Problem:** In the binary classification setting where $y \in \{-1, +1\}$, the margin is defined as $z = y\theta^T x$, where $\theta$ and $x$ lie in $\mathbb{R}^n$.

Three loss functions are given:
i. zero-one loss: $\varphi_{zo}(z) = \mathbf{1}\{z \le 0\}$
ii. exponential loss: $\varphi_{\exp}(z) = e^{-z}$
iii. hinge loss: $\varphi_{\text{hinge}}(z) = \max\{1 - z, 0\}$

Suppose that the margin $z < 0$ for the current parameters $\theta$.

1. Give the expression for $\frac{\partial}{\partial \theta_k}\varphi(y\theta^T x)$ for each of the given loss functions.
2. Identify which loss would fail to minimize with gradient descent, no matter the step size chosen.

**Answer:**

The expressions for the partial derivatives are:

i. $\frac{\partial}{\partial \theta_k}\varphi_{zo}(y\theta^T x) = 0$
ii. $\frac{\partial}{\partial \theta_k}\varphi_{\exp}(y\theta^T x) = -yx_k e^{-z}$
iii. $\frac{\partial}{\partial \theta_k}\varphi_{\text{hinge}}(y\theta^T x) = -yx_k$

Since the zero-one loss is 0 for margin $z < 0$, no matter the step size our parameter values would remain unchanged, and hence we fail to minimize the loss with gradient descent.

### (c) [5 points] Spam Classification: Naive Bayes vs Boosting

**Problem:** Consider performing spam classification where each e-mail is represented as a vector $\mathbf{x}$ of the same size as the number of words in the vocabulary $|V|$, where $x_i$ is 1 if the e-mail contains word $i$ and 0 otherwise. We saw in class that Naive Bayes with Laplace smoothing is one simple method for performing classification in this setting. For this question, to simplify we set $p(y = 1) = p(y = -1) = 0.5$.

Consider classifying $\mathbf{x}$ by instead using the boosting algorithm with $2|V|$ decision stumps as the weak learners. In this setting, which of the two methods, Naive Bayes or boosting with decision stumps, would you expect to yield lower bias? Explain your reasoning.

**Answer:**

First, note that since $\mathbf{x}$ is a vector of only 0s and 1s, the decision stump thresholds can all be set to any value strictly between 0 and 1 and have the same effect. One possible output of the boosting algorithm would simply be of the form $\text{sign}(\theta^T[\mathbf{x}; \mathbf{x}])$ for $\theta \in \mathbb{R}^{2|V|}$ (where we replace the 0s in $\mathbf{x}$ with -1s).

For each possible word, Naive Bayes learns two parameters, $p(x_j|y = 1)$ and $p(x_j|y = -1)$, and hence also has $2|V|$ parameters (this is crucial for comparing the two classifiers!). The decision rule in log space is also linear: output $\text{sign}(\sum_j \log p(x_j|y = 1) - \sum_j \log p(x_j|y = -1))$. However, Naive Bayes makes the generative modeling assumption that $p(\mathbf{x}|y)$ is modeled by independent word counts. On the other hand, as a discriminative model boosting allows for more possible values of $\theta$, and hence has a larger hypothesis space and should achieve lower bias.

### (d) [4 points] Linear SVM Decision Boundary Changes

**Problem:** Consider a linear SVM classifier trained for binary classification using the hinge loss $L(\theta^T x, y) = \max\{0, 1 - y\theta^T x\}$. For each of the following scenarios, does the optimal decision boundary necessarily remain the same? Explain your reasoning and sketch a picture if helpful. Assume that after performing the action in each scenario, there is still at least one training example in both the positive and negative classes.

i. Remove all examples $(x^{(i)}, y^{(i)})$ with margin $> 1$.
ii. Remove all examples $(x^{(i)}, y^{(i)})$ with margin $< 1$.
iii. Add an $\ell_2$-regularization term $\frac{\lambda}{2}\theta^T\theta = \frac{\lambda}{2} \|\theta\|^2$ to the training loss.
iv. Scale all $x^{(i)}$ by a constant factor $\alpha$.

**Answer:**

i. **Yes;** the loss is not affected by examples with margin $> 1$.

ii. **No;** the loss is affected by these examples and hence we may have different optimal $\theta$.

iii. **No;** the regularization term directly encourages $\theta$ with smaller $\ell_2$-norm, hence changing the decision boundary.

iv. **No;** consider 1-D counter-example with $\alpha = 2$, $x^{(1)}$ at the origin, and $x^{(2)}$ at 1; the decision boundary moves from 0.5 to 1.

### (e) [6 points] Bias-Variance Tradeoff Scenarios

**Problem:** We consider a binary classification task where we have $m$ training examples and our hypothesis $h_\theta(x)$ is parameterized by $\theta$. For each of the following scenarios, select whether we should expect bias and variance to increase or decrease. Explain your reasoning.

**Scenario i:** Project the values of $\theta$ to lie between $-1$ and $1$ after each training update, that is $\theta_j = \min\{1, \max\{-1, \theta_j\}\}$.

**Scenario ii:** Smooth the estimates of our hypotheses by outputting
$h(x) = (1/3) \sum_{x^{(i)} \in N_3(x)} h_\theta(x^{(i)})$,
where $N_3(x)$ are the 3 points in the training set closest to $x$.

**Scenario iii:** Remove one of the feature dimensions of $x$.

**Answer:**

i. Bias should increase and variance should decrease since we're reducing the hypothesis space of the model.

ii. Bias should increase and variance should decrease since smoothing encourages more similar outputs for different examples. For example, consider the extreme case where we smooth by outputting the mean over all $m$ examples; we then have very high bias and 0 variance since we make the same prediction for every input.

iii. Bias should increase and variance should decrease since for the same reason as in (i); the hypothesis space is now a strict subset of the previous space.

## Problem 2: Linear Regression - First Order Convergence for Least Squares

**Problem:** Consider the least squares problem, where we pick $\theta$ to minimize the objective $J(\theta) = \frac{1}{2}(X^T\theta-y)^T(X^T\theta-y)$. The solution to this problem is given by the normal equation, where $\theta = (XX^T)^{-1}Xy$. In Problem Set 1, we showed that a single Newton step will converge to the correct solution. Now we will examine how gradient descent performs on the same problem.

### (a) [4 points] Gradient Descent Update

**Problem:** Find the gradient of $J$ with respect to $\theta$, and write the gradient descent update step for $\theta^{(t+1)}$, given $\theta^{(t)}$ and step size $\alpha$.

**Answer:**

$\nabla_\theta J = XX^T\theta - Xy$; $\theta^{(t+1)} = \theta^{(t)} - \alpha(XX^T\theta^{(t)} - Xy)$

### (b) [8 points] Convergence to Optimal Solution

**Problem:** Show that as $t \to \infty$, $\theta^{(t+1)} \to (XX^T)^{-1}Xy$, for gradient descent with step size $\alpha$ and initial condition $\theta^{(0)} = 0$. You may use the fact that $(\alpha A)^{-1} = \sum_{i=0}^{\infty} (I - \alpha A)^i$ for small $\alpha > 0$, and assume that the choice of $\alpha$ is small enough.

**Answer:**

From (a), we have the gradient descent update:

$$\theta^{(t+1)} = \theta^{(t)} - \alpha XX^T \theta^{(t)} + \alpha Xy$$

$$= (I - \alpha XX^T) \theta^{(t)} + \alpha Xy$$

$$\theta^{(t+1)} = (I - \alpha XX^T)^{t+1} \theta^{(0)} + \sum_{i=1}^{t+1} (I - \alpha XX^T)^{t+1-i} \alpha Xy$$

Given $\theta^{(0)} = 0$ and adjusting the summation index:

$$\theta^{(t+1)} = 0 + \alpha \sum_{i=0}^{t} (I - \alpha XX^T)^i Xy$$

As $t \to \infty$, $\sum_{i=0}^{t} (I - \alpha XX^T)^i = (\alpha XX^T)^{-1}$. Using this, we now have:

$$\theta^{(t+1)} = \alpha \alpha^{-1} (XX^T)^{-1} Xy$$

$$= (XX^T)^{-1} Xy$$

## Problem 3: Generative Models - Gaussian Discriminant Analysis [12 points]

**Problem:** Consider the 1-dimensional Gaussian discriminant analysis model where $x \in \mathbb{R}$ and we assume

$$p(y) = \phi^{1\{y=1\}} (1-\phi)^{1\{y=-1\}}$$

$$p(x|y = -1) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2\sigma^2}(x-\mu_{-1})^2\right)$$

$$p(x|y = 1) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2\sigma^2}(x-\mu_1)^2\right)$$

In this problem we will assume that $\sigma$ is a fixed quantity that we have been given and is therefore not a parameter of the model.

Recall from Problem Set 1 that we can express $p(y|x; \phi, \mu_{-1}, \mu_1)$ in the form

$$p(y|x; \theta) = \frac{1}{1 + \exp(-y(\theta_1 x + \theta_0))}$$

where for the model described above we have,

$$\theta_0 = \frac{1}{2\sigma^2}(\mu_{-1}^2 - \mu_1^2) - \log \frac{1-\phi}{\phi}$$

$$\theta_1 = \frac{1}{\sigma^2}(\mu_1 - \mu_{-1})$$

### (a) [2 points] Joint Log-Likelihood

**Problem:** Write the joint log-likelihood $\ell(\phi, \mu_{-1}, \mu_1) = \log p(x, y; \phi, \mu_{-1}, \mu_1)$ for a single example $(x, y)$.

**Answer:**

$$p(x, y; \phi, \mu_{-1}, \mu_1) = p(y; \phi)p(x|y; \mu_{-1}, \mu_1)$$

$$\log p(x, y; \phi, \mu_{-1}, \mu_1) = \log p(y|\phi) + \log p(x|y; \mu_{-1}, \mu_1)$$

$$= \log(1 - \phi)^{1\{y=-1\}} \log(\phi)^{1\{y=1\}} + \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{1}{2\sigma^2}(x - \mu_y)^2$$

### (b) [7 points] Concavity of Log-Likelihood

**Problem:** Show that the log-likelihood of all training examples $\{(x^{(i)},y^{(i)})\}_{i=1}^m$ is concave (and hence any maximum we find must be the global maximum) by first computing $\frac{\partial^2 \ell}{\partial \phi^2}$, $\frac{\partial^2 \ell}{\partial \mu_{-1}^2}$, and $\frac{\partial^2 \ell}{\partial \mu_1^2}$ for a single example $(x, y)$. Then make an argument that the total log-likelihood is concave. Hint: Recall a function is concave if its Hessian is negative semidefinite. A one-dimensional function $f$ is concave if $f''(x) \le 0$ for all $x$.

**Answer:**

First we show that the log-likelihood is concave for a single $(x, y)$.

$$\frac{\partial \ell}{\partial \phi} = -1\{y = -1\}\frac{1}{1 - \phi} + 1\{y = 1\}\frac{1}{\phi}$$

$$\frac{\partial^2 \ell}{\partial \phi^2} = \begin{cases} -\phi^{-2} & y = 1 \\ -(1 - \phi)^{-2} & y = -1 \end{cases}$$

which is negative for both cases.

$$\frac{\partial \ell}{\partial \mu_y} = \frac{1}{\sigma^2}(x - \mu_y)$$

$$\frac{\partial^2 \ell}{\partial \mu_y^2} = -\frac{1}{\sigma^2}$$

and negative as well.

Since $\phi$ and $\mu_y$ are in separate terms, the Hessian $H$ must be diagonal and negative along the diagonal. Hence $H$ is negative semidefinite, and $\ell$ is concave in both $\phi$ and $\mu_y$. Due to linearity of differentiation, the sum of concave functions is concave, and thus log-likelihood over all training $m$ examples must be concave as well.

### (c) [3 points] Decision Boundary

**Problem:** Derive an expression for the decision boundary for classifying $x$ as either $y = -1$ or $1$.

**Answer:**

We want $p(y = -1|x;\theta) = p(y = 1|x;\theta) = 0.5$ and hence set $\theta_1x + \theta_0 = 0$ where $\theta_1$ and $\theta_0$ are given in the problem statement.

Solving, we find:

$$x = \frac{2\sigma^2 \log \frac{1-\phi}{\phi} + (\mu_1^2 - \mu_{-1}^2)}{2(\mu_1 - \mu_{-1})}$$

Note that setting $p(x|y = -1) = p(x|y = 1)$ does not work, since this does not take into account $p(y)$.


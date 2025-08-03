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


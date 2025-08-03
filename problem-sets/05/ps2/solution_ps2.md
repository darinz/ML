# Problem Set 2 Solutions

## Problem 1: Least Squares

### Problem 1(a) [2 points]

**Problem:** Let $\hat{\theta} = \arg \min_{\theta} J(\theta)$ be the minimizer of the original least squares objective (using the original design matrix $X$). Using the orthonormality assumption, show that $J(\hat{\theta}) = (XX^T \bar{y} - \bar{y})^T (XX^T \bar{y} - \bar{y})$. I.e., show that this is the value of $\min_{\theta} J(\theta)$ (the value of the objective at the minimum).

**Answer:** We know from lecture that the least squares minimizer is $\hat{\theta} = (X^T X)^{-1} X^T \bar{y}$ but because of the orthonormality assumption, this simplifies to $\hat{\theta} = X^T \bar{y}$. Substituting this expression into the normal equation for $J(\theta)$ gives the final expression $J(\hat{\theta}) = (XX^T \bar{y} - \bar{y})^T (XX^T \bar{y} - \bar{y})$.

### Problem 1(b) [5 points]

**Problem:** Now let $\hat{\theta}_{\text{new}}$ be the minimizer for $\tilde{J}(\theta_{\text{new}}) = (\tilde{X}\theta_{\text{new}} - \vec{y})^T(\tilde{X}\theta_{\text{new}} - \vec{y})$. Find the new minimized objective $\tilde{J}(\hat{\theta}_{\text{new}})$ and write this expression in the form: $\tilde{J}(\hat{\theta}_{\text{new}}) = J(\hat{\theta}) + f(X, \vec{v}, \vec{y})$ where $J(\hat{\theta})$ is as derived in part (a) and $f$ is some function of $X, \vec{v}$, and $\vec{y}$.

**Answer:** Just like we had in part (a), the minimizer for the new objective is $\hat{\theta}_{\text{new}} = \tilde{X}^T\vec{y}$. Now we solve for the new minimized objective:

$\tilde{J}(\hat{\theta}_{\text{new}}) = (\tilde{X}\hat{\theta}_{\text{new}} - \vec{y})^T(\tilde{X}\hat{\theta}_{\text{new}} - \vec{y})$
$= (\tilde{X}\tilde{X}^T\vec{y} - \vec{y})^T(\tilde{X}\tilde{X}^T\vec{y} - \vec{y})$
$= ((XX^T + \vec{v}\vec{v}^T)\vec{y} - \vec{y})^T((XX^T + \vec{v}\vec{v}^T)\vec{y} - \vec{y})$
$= ((XX^T\vec{y} - \vec{y}) + \vec{v}\vec{v}^T\vec{y})^T((XX^T\vec{y} - \vec{y}) + \vec{v}\vec{v}^T\vec{y})$
$= (XX^T\vec{y} - \vec{y})^T(XX^T\vec{y} - \vec{y}) + 2(XX^T\vec{y} - \vec{y})^T(\vec{v}\vec{v}^T\vec{y}) + (\vec{v}\vec{v}^T\vec{y})^T(\vec{v}\vec{v}^T\vec{y})$
$= J(\hat{\theta}) + 2(XX^T\vec{y} - \vec{y})^T(\vec{v}\vec{v}^T\vec{y}) + (\vec{v}\vec{v}^T\vec{y})^T(\vec{v}\vec{v}^T\vec{y})$

### Problem 1(c) [6 points]

**Problem:** Prove that the optimal objective value does not increase upon adding a feature to the design matrix. That is, show $\tilde{J}(\hat{\theta}_{\text{new}}) \le J(\hat{\theta})$.

**Answer:** Using the final result of part (b), we can continue simplifying the expression for $J(\hat{\theta}_{\text{new}})$ as follows:
\[
\begin{aligned}
\tilde{J}(\hat{\theta}_{\text{new}}) &= J(\hat{\theta}) + 2(XX^T\tilde{y} - \tilde{y})^T(vv^T\tilde{y}) + (vv^T\tilde{y})^T(vv^T\tilde{y}) \\
&= J(\hat{\theta}) + 2(XX^T\tilde{y})^T(vv^T\tilde{y}) - 2\tilde{y}^T(vv^T\tilde{y}) + (vv^T\tilde{y})^T(vv^T\tilde{y}) \\
&= J(\hat{\theta}) + 2(\tilde{y}^T XX^T vv^T\tilde{y}) - 2(\tilde{y}^T vv^T\tilde{y}) + (\tilde{y}^T vv^T vv^T\tilde{y}) \\
&= J(\hat{\theta}) - \tilde{y}^T vv^T\tilde{y} \\
&= J(\hat{\theta}) - (v^T\tilde{y})^2 \\
&\le J(\hat{\theta})
\end{aligned}
\]
From the third to last equality to the second to last equality, we use the two facts that $X^T v = 0$ and $v^T v = 1$.

The proof is complete.

Note for this problem we also accepted solutions where parts (b) and (c) overlapped.

### Problem 1(d) [3 points]

**Problem:** Does the above result show that if we keep increasing the number of features, we can always get a model that generalizes better than a model with fewer features? Explain why or why not.

**Answer:** The result shows that we can either maintain or decrease the minimized square error objective by adding more features. However, remember that the error objective is computed only on the training samples and not the true data distribution. As a result, reducing training error does not guarantee a reduction in error on the true distribution. In fact, after a certain point adding features will likely lead to overfitting, increasing our generalization error. Therefore, adding features does not actually always result in a model that generalizes better.

## Problem 2: Decision Boundaries for Generative Models

### Problem 2(a) [7 points]

**Problem:** Consider the multinomial event model of Naive Bayes. Our goal in this problem is to show that this is a linear classifier.

For a given text document $x$, let $c_1, \dots, c_V$ indicate the number of times each word (out of $V$ words) appears in the document. Thus, $c_i \in \{0, 1, 2, \dots\}$ counts the occurrences of word $i$. Recall that the Naive Bayes model uses parameters $\phi_y = p(y = 1)$, $\phi_{i|y=1} = p(\text{word i appears in a specific document position } | y = 1)$ and $\phi_{i|y=0} = p(\text{word i appears in a specific document position } | y = 0)$.

We say a classifier is linear if it assigns a label $y = 1$ using a decision rule of the form
$$ \sum_{i=1}^V w_i c_i + b \ge 0 $$
I.e., the classifier predicts "$y = 1$" if $\sum_{i=1}^V w_i c_i + b \ge 0$, and predicts "$y = 0$" otherwise.

Show that Naive Bayes is a linear classifier, and clearly state the values of $w_i$ and $b$ in terms of the Naive Bayes parameters. (Don't worry about whether the decision rule uses "$\ge$" or "$>$.") Hint: consider using log-probabilities.

**Answer:**
The decision boundary for Naive Bayes can be stated as
$$ P(y = 1|c; \Phi) > P(y = 0|c; \Phi) $$
$$ \log p(y = 1|c; \Phi) > \log p(y = 0|c; \Phi) $$
$$ \log p(y = 1|c; \Phi) - \log p(y = 0|c; \Phi) > 0 $$
$$ \log \frac{p(y = 1|c; \Phi)}{p(y = 0|c; \Phi)} > 0 $$
$$ \log \frac{p(y = 1) \prod_{i=1}^V p(E_i|y = 1)^{c_i}}{p(y = 0) \prod_{i=1}^V p(E_i|y = 0)^{c_i}} > 0 $$
$$ \log \frac{p(y = 1)}{p(y = 0)} + \sum_{i=1}^V \log p(E_i|y = 1)^{c_i} - \log p(E_i|y = 0)^{c_i} > 0 $$
$$ \log \frac{p(y = 1)}{p(y = 0)} + \sum_{i=1}^V c_i \log \frac{p(E_i|y = 1)}{p(E_i|y = 0)} > 0 $$
Using the given parameters:
$$ \log \frac{\phi_y}{1 - \phi_y} + \sum_{i=1}^V c_i \log \frac{\phi_{i|y=1}}{\phi_{i|y=0}} > 0 $$
Thus, Naive Bayes is a linear classifier with
$$ w_i = \log \frac{\phi_{i|y=1}}{\phi_{i|y=0}} $$
$$ b = \log \frac{\phi_y}{1 - \phi_y} $$

### Problem 2(b) [7 points]

**Problem:** In Problem Set 1, you showed that Gaussian Discriminant Analysis (GDA) is a linear classifier. In this problem, we will show that a modified version of GDA has a quadratic decision boundary.

Recall that GDA models $p(x|y)$ using a multivariate normal distribution, where $(x|y = 0) \sim \mathcal{N}(\mu_0, \Sigma)$ and $(x|y = 1) \sim \mathcal{N}(\mu_1, \Sigma)$, where we used the same $\Sigma$ for both Gaussians.

For this question, we will instead use two covariance matrices $\Sigma_0, \Sigma_1$ for the two labels. So, $(x|y = 0) \sim \mathcal{N}(\mu_0, \Sigma_0)$ and $(x|y = 1) \sim \mathcal{N}(\mu_1, \Sigma_1)$.

<img src="./q2b_problem.png" width="350">

The model distributions can now be written as:
$p(y) = \phi^y (1 - \phi)^{1-y}$
$p(x|y = 0) = \frac{1}{(2\pi)^{n/2} |\Sigma_0|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_0)^T \Sigma_0^{-1} (x - \mu_0)\right)$
$p(x|y = 1) = \frac{1}{(2\pi)^{n/2} |\Sigma_1|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu_1)^T \Sigma_1^{-1} (x - \mu_1)\right)$

Let's follow a binary decision rule, where we predict $y = 1$ if $p(y = 1|x) \ge p(y = 0|x)$, and $y = 0$ otherwise. Show that if $\Sigma_0 \ne \Sigma_1$, then the separating boundary is quadratic in $x$.
That is, simplify the decision rule "$p(y = 1|x) \ge p(y = 0|x)$" to the form "$x^T Ax + B^T x + C \ge 0$" (supposing that $x \in \mathbb{R}^{n+1}$), for some $A \in \mathbb{R}^{(n+1)\times(n+1)}$, $B \in \mathbb{R}^{n+1}$, $C \in \mathbb{R}$ and $A \ne 0$. Please clearly state your values for $A, B$ and $C$.

**Answer:** Examining the log-probabilities yields:

$\log p(y = 1|x) \ge \log p(y = 0|x)$

$0 \le \log \left(\frac{p(y = 1|x)}{p(y = 0|x)}\right)$

$0 \le \log \left(\frac{p(y = 1)p(x|y = 1)}{p(y = 0)p(x|y = 0)}\right)$

$0 \le \log \left(\frac{\phi}{1 - \phi}\right) - \log \left(\frac{|\Sigma_1|^{1/2}}{|\Sigma_0|^{1/2}}\right) - \frac{1}{2}(x - \mu_1)^T \Sigma_1^{-1}(x - \mu_1) - \frac{1}{2}(x - \mu_0)^T \Sigma_0^{-1}(x - \mu_0)$

$0 \le -\frac{1}{2}x^T(\Sigma_1^{-1} - \Sigma_0^{-1})x - 2(\mu_1^T \Sigma_1^{-1} - \mu_0^T \Sigma_0^{-1})x + \mu_1^T \Sigma_1^{-1} \mu_1 - \mu_0^T \Sigma_0^{-1} \mu_0 + \log \left(\frac{\phi}{1 - \phi}\right) - \log \left(\frac{|\Sigma_1|^{1/2}}{|\Sigma_0|^{1/2}}\right)$

$0 \le x^T\left(\frac{1}{2}(\Sigma_0^{-1} - \Sigma_1^{-1})\right)x + (\mu_1^T \Sigma_1^{-1} - \mu_0^T \Sigma_0^{-1})x + \log \left(\frac{\phi}{1 - \phi}\right) + \log \left(\frac{|\Sigma_0|^{1/2}}{|\Sigma_1|^{1/2}}\right) + \frac{1}{2}(\mu_0^T \Sigma_0^{-1} \mu_0 - \mu_1^T \Sigma_1^{-1} \mu_1)$

From the above, we see that $A = \frac{1}{2}(\Sigma_0^{-1} - \Sigma_1^{-1})$, $B^T = \mu_1^T \Sigma_1^{-1} - \mu_0^T \Sigma_0^{-1}$, and $C = \log\left(\frac{\phi}{1 - \phi}\right) + \log\left(\frac{|\Sigma_0|^{1/2}}{|\Sigma_1|^{1/2}}\right) + \frac{1}{2}(\mu_0^T \Sigma_0^{-1} \mu_0 - \mu_1^T \Sigma_1^{-1} \mu_1)$. Furthermore, $A \ne 0$ since $\Sigma_0 \ne \Sigma_1$ implies that $\Sigma_0^{-1} - \Sigma_1^{-1} \ne 0$. Therefore, the decision boundary is quadratic.

## Problem 3: Generalized Linear Models

### Problem 3(a) i. [5 points]

**Problem:** Show that the geometric distribution is an exponential family distribution. Explicitly specify the components $b(y)$, $\eta$, $T(y)$, and $\alpha(\eta)$, as well as express $\phi$ in terms of $\eta$.

**Answer:**
The given probability mass function:
$$p(y; \phi) = (1 - \phi)^{y-1} \phi$$
$$= \exp\left((y - 1) \log(1 - \phi) + \log \phi\right)$$
$$= \exp\left((\log(1 - \phi))y + \log \phi - \log(1 - \phi)\right)$$
$$= \exp\left((\log(1 - \phi))y - \log \frac{1 - \phi}{\phi}\right)$$

Therefore:
$$b(y) = 1$$
$$\eta = \log(1 - \phi)$$
$$T(y) = y$$
$$\alpha(\eta) = \log \frac{1 - \phi}{\phi}$$
$$\phi = 1 - e^\eta$$

### Problem 3(a) ii. [5 points]

**Problem:** Suppose that we have an IID training set $\{(x^{(i)}, y^{(i)}), i = 1, ..., m\}$ and we wish to model this using a GLM based on a geometric distribution. Find the log-likelihood $\log \prod_{i=1}^{m} p(y^{(i)}|x^{(i)}; \theta)$ defined with respect to the entire training set.

**Answer:**
We calculate the log-likelihood for 1 sample as well as for the entire training set:

$$
\begin{aligned}
\log p(y^{(i)}|x^{(i)}; \theta) &= \log \left( (1 - \phi)^{y^{(i)}-1} \phi \right) \\
&= (y^{(i)} - 1) \log (1 - \phi) + \log \phi \\
&= (\log (1 - \phi))y^{(i)} - \log \frac{1 - \phi}{\phi} \\
&= y^{(i)} \log e^{\eta} - \log \frac{e^{\eta}}{1 - e^{\eta}} \\
&= \eta y^{(i)} - \eta + \log (1 - e^{\eta}) \\
&= \theta^T x^{(i)} y^{(i)} - \theta^T x^{(i)} + \log (1 - \exp (\theta^T x^{(i)})) \\
&= \theta^T x^{(i)} (y^{(i)} - 1) + \log (1 - \exp (\theta^T x^{(i)}))
\end{aligned}
$$

$$
\begin{aligned}
l(\theta) &= \log p(y|x; \theta) \\
&= \log \left( \prod_{i=1}^{m} p(y^{(i)}|x^{(i)}; \theta) \right) \\
&= \sum_{i=1}^{m} \log (p(y^{(i)}|x^{(i)}; \theta)) \\
&= \sum_{i=1}^{m} \left( \theta^T x^{(i)} (y^{(i)} - 1) + \log (1 - \exp (\theta^T x^{(i)})) \right)
\end{aligned}
$$

Observe that since we use the substitution $\eta = \log (1 - \phi) = \theta^T x$ above, it follows that $\forall x, \theta^T x < 0$ in order for the prediction to be valid. This is obviously not feasible as a prediction rule, and so in practice one could use the function $g(\theta^T x) = -\log (\theta^T x)$ to map these linear predictions to a valid interval when dealing with real data.

### Problem 3(b) [6 points]

**Problem:** Derive the Hessian $H$ and the gradient vector of the log likelihood with respect to $\theta$, and state what one step of Newton's method for maximizing the log likelihood would be.

**Answer:** To apply Newton's method, we need to find the gradient and Hessian of the log-likelihood:

$$
\begin{aligned}
\nabla_\theta l(\theta) &= \nabla_\theta \sum_{i=1}^{m} \left(\theta^T x^{(i)} y^{(i)} - \theta^T x^{(i)} + \log (1 - \exp (\theta^T x^{(i)}))\right) \\
&= \sum_{i=1}^{m} \left(x^{(i)} (y^{(i)} - 1) - \frac{x^{(i)} \exp (\theta^T x^{(i)})}{(1 - \exp (\theta^T x^{(i)}))}\right) \\
&= \sum_{i=1}^{m} \left(y^{(i)} - \frac{1}{(1 - \exp (\theta^T x^{(i)}))}\right) x^{(i)}
\end{aligned}
$$

$$
\begin{aligned}
H &= \nabla_\theta (\nabla_\theta l(\theta))^T \\
&= -\nabla_\theta \sum_{i=1}^{m} \frac{1}{(1 - \exp (\theta^T x^{(i)}))} x^{(i)T} \\
&= -\sum_{i=1}^{m} \frac{\exp (\theta^T x^{(i)})}{(1 - \exp (\theta^T x^{(i)}))^2} x^{(i)} x^{(i)T}
\end{aligned}
$$

The Newton's method update rule is then: $\theta := \theta - H^{-1} \nabla_\theta l(\theta)$

### Problem 3(c) [2 points]

**Problem:** Show that the Hessian is negative semi-definite, which implies the optimization objective is concave and Newton's method maximizes log-likelihood.

**Answer:** 
$$z^T H z = - \sum_{i=1}^{m} \frac{\exp(\theta^T x^{(i)})}{(1 - \exp(\theta^T x^{(i)}))^2} z^T x^{(i)} x^{(i)T} z$$
$$= - \sum_{i=1}^{m} \frac{\exp(\theta^T x^{(i)})}{(1 - \exp(\theta^T x^{(i)}))^2} \|z^T x^{(i)}\|^2 \le 0$$

Therefore, $H$ is negative semidefinite, which means $l(\theta)$ is concave. So we are maximizing it.

## Problem 4: Support Vector Regression

### Problem 4(a) [4 points]

**Problem:** Write down the Lagrangian for the optimization problem above. We suggest you use two sets of Lagrange multipliers $\alpha_i$ and $\alpha_i^*$, corresponding to the two inequality constraints (labeled (1) and (2) above), so that the Lagrangian would be written $\mathcal{L}(w, b, \alpha, \alpha^*)$.

**Answer:**
Let $\alpha_i, \alpha_i^* \ge 0$ ($i = 1,\dots,m$) be the Lagrange multiplier for (1)-(4) respectively. Then, the Lagrangian can be written as:

$$
\mathcal{L}(w, b, \alpha, \alpha^*) \\
= \frac{1}{2}\|w\|^2 \\
- \sum_{i=1}^m \alpha_i (\epsilon - y^{(i)} + w^T x^{(i)} + b) \\
- \sum_{i=1}^m \alpha_i^* (\epsilon + y^{(i)} - w^T x^{(i)} - b)
$$

### Problem 4(b) [10 points]

**Problem:** Derive the dual optimization problem. You will have to take derivatives of the Lagrangian with respect to $w$ and $b$.

**Answer:**
First, the dual objective function can be written as:
$$ \theta_D(\alpha, \alpha^*) = \min_{w,b} L(w, b, \alpha, \alpha^*) $$

Now, taking the derivatives of Lagrangian with respect to all primal variables, we have:
$$ \partial_w L = w - \sum_{i=1}^m (\alpha_i - \alpha_i^*)x^{(i)} = 0 $$
$$ \partial_b L = \sum_{i=1}^m (\alpha_i^* - \alpha_i) = 0 $$

Substituting the above two relations back into the Lagrangian, we have:
$$ \theta_D(\alpha, \alpha^*) = \frac{1}{2}\|w\|^2 - \epsilon \sum_{i=1}^m (\alpha_i + \alpha_i^*) + \sum_{i=1}^m y^{(i)}(\alpha_i - \alpha_i^*) + b \sum_{i=1}^m (\alpha_i^* - \alpha_i) + \sum_{i=1}^m (\alpha_i^* - \alpha_i)w^T x^{(i)} $$
Since $\sum_{i=1}^m (\alpha_i^* - \alpha_i) = 0$, the term $b \sum_{i=1}^m (\alpha_i^* - \alpha_i)$ vanishes.
$$ \theta_D(\alpha, \alpha^*) = \frac{1}{2}\|w\|^2 - \epsilon \sum_{i=1}^m (\alpha_i + \alpha_i^*) + \sum_{i=1}^m y^{(i)}(\alpha_i - \alpha_i^*) + \sum_{i=1}^m (\alpha_i^* - \alpha_i)w^T x^{(i)} $$
Substitute $w = \sum_{i=1}^m (\alpha_i - \alpha_i^*)x^{(i)}$:
\begin{align*} &= \frac{1}{2}\left\|\sum_{i=1}^m (\alpha_i - \alpha_i^*)x^{(i)}\right\|^2 - \epsilon \sum_{i=1}^m (\alpha_i + \alpha_i^*) + \sum_{i=1}^m y^{(i)}(\alpha_i - \alpha_i^*) \\ & \quad + \sum_{i=1}^m (\alpha_i^* - \alpha_i)\left(\sum_{j=1}^m (\alpha_j - \alpha_j^*)x^{(j)}\right)^T x^{(i)} \\ &= \frac{1}{2}\left(\sum_{i=1}^m (\alpha_i - \alpha_i^*)x^{(i)}\right)^T\left(\sum_{k=1}^m (\alpha_k - \alpha_k^*)x^{(k)}\right) - \epsilon \sum_{i=1}^m (\alpha_i + \alpha_i^*) + \sum_{i=1}^m y^{(i)}(\alpha_i - \alpha_i^*) \\ & \quad - \sum_{i=1}^m (\alpha_i - \alpha_i^*)\left(\sum_{j=1}^m (\alpha_j - \alpha_j^*)x^{(j)}\right)^T x^{(i)} \\ &= \frac{1}{2} \sum_{i=1}^m \sum_{k=1}^m (\alpha_i - \alpha_i^*)(\alpha_k - \alpha_k^*)x^{(i)T}x^{(k)} - \epsilon \sum_{i=1}^m (\alpha_i + \alpha_i^*) + \sum_{i=1}^m y^{(i)}(\alpha_i - \alpha_i^*) \\ & \quad - \sum_{i=1}^m \sum_{j=1}^m (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)x^{(j)T}x^{(i)} \\ &= -\frac{1}{2} \sum_{i=1,j=1}^m (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)x^{(i)T}x^{(j)} - \epsilon \sum_{i=1}^m (\alpha_i + \alpha_i^*) + \sum_{i=1}^m y^{(i)}(\alpha_i - \alpha_i^*) \end{align*}

Now the dual problem can be formulated as:
$$ \max_{\alpha_i, \alpha_i^*} -\frac{1}{2} \sum_{i=1,j=1}^m (\alpha_i - \alpha_i^*)(\alpha_j - \alpha_j^*)x^{(i)T}x^{(j)} - \epsilon \sum_{i=1}^m (\alpha_i + \alpha_i^*) + \sum_{i=1}^m y^{(i)}(\alpha_i - \alpha_i^*) $$
s.t.
$$ \sum_{i=1}^m (\alpha_i^* - \alpha_i) = 0 $$
$$ \alpha_i, \alpha_i^* \ge 0 $$

### Problem 4(c) [4 points]

**Problem:** Show that this algorithm can be kernelized. For this, you have to show that (i) the dual optimization objective can be written in terms of inner-products of training examples; and (ii) at test time, given a new $x$ the hypothesis $h_{w,b}(x)$ can also be computed in terms of inner products.

**Answer:** This algorithm can be kernelized because when making prediction at $x$, we have:

$$f(w, x) = w^T x + b = \sum_{i=1}^{m} (\alpha_i - \alpha_i^*) x^{(i)T} x + b = \sum_{i=1}^{m} (\alpha_i - \alpha_i^*) k(x^{(i)}, x) + b$$

This shows that predicting function can be written in a kernel form.

## Problem 5: Learning Theory

Suppose you are given a hypothesis $h_0 \in \mathcal{H}$, and your goal is to determine whether $h_0$ has generalization error within $\eta > 0$ of the best hypothesis, $h^* = \arg \min_{h \in \mathcal{H}} \varepsilon(h)$. More specifically, we say that a hypothesis $h$ is $\eta$-optimal if $\varepsilon(h) \le \varepsilon(h^*) + \eta$. Here, we wish to answer the following question:

Given a hypothesis $h_0$, is $h_0$ $\eta$-optimal?

Let $\delta > 0$ be some fixed constant, and consider a finite hypothesis class $\mathcal{H}$ of size $|\mathcal{H}| = k$. For each $h \in \mathcal{H}$, let $\hat{\varepsilon}(h)$ denote the training error of $h$ with respect to some training set of $m$ IID examples, and let $\hat{h} = \arg \min_{h \in \mathcal{H}} \hat{\varepsilon}(h)$ denote the hypothesis that minimizes training error.

Now, consider the following algorithm:

1. Set $\gamma := \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}}$
2. If $\hat{\varepsilon}(h_0) > \hat{\varepsilon}(\hat{h}) + \eta + 2\gamma$, then return NO.
3. If $\hat{\varepsilon}(h_0) < \hat{\varepsilon}(\hat{h}) + \eta - 2\gamma$, then return YES.
4. Otherwise, return UNSURE.

Intuitively, the algorithm works by comparing the training error of $h_0$ to the training error of the hypothesis $\hat{h}$ with the minimum training error, and returns NO or YES only when $\hat{\varepsilon}(h_0)$ is either significantly larger than or significantly smaller than $\hat{\varepsilon}(\hat{h})+\eta$.

### Problem 5(a) [6 points]

**Problem:** First, show that if $\varepsilon(h_0) \le \varepsilon(h^*) + \eta$ (i.e., $h_0$ is $\eta$-optimal), then the probability that the algorithm returns NO is at most $\delta$.

**Answer:** Suppose that $\varepsilon(h_0) \le \varepsilon(h^*) + \eta$. Using the Hoeffding inequality, we have that for

$$ \gamma = \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}} $$

then with probability at least $1 - \delta$,

$$
\begin{aligned}
\hat{\varepsilon}(h_0) &\le \varepsilon(h_0) + \gamma \\
&\le \varepsilon(h^*) + \eta + \gamma \\
&\le \varepsilon(\hat{h}) + \eta + \gamma \\
&\le \hat{\varepsilon}(\hat{h}) + \eta + 2\gamma.
\end{aligned}
$$

Here, the first and last inequalities follow from the fact that under the stated uniform convergence conditions, all hypotheses in $\mathcal{H}$ have empirical errors within $\gamma$ of their true generalization errors. The second inequality follows from our assumption, and the third inequality follows from the fact that $h^*$ minimizes the true generalization error. Therefore, the reverse condition, $\hat{\varepsilon}(h_0) > \hat{\varepsilon}(\hat{h}) + \eta + 2\gamma$, occurs with probability at most $\delta$.

### Problem 5(b) [6 points]

**Problem:** Second, show that if $\varepsilon(h_0) > \varepsilon(h^*) + \eta$ (i.e., $h_0$ is not $\eta$-optimal), then the probability that the algorithm returns YES is at most $\delta$.

**Answer:** Suppose that $\varepsilon(h_0) > \varepsilon(h^*) + \eta$. Using the Hoeffding inequality, we have that for
$$ \gamma = \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}} $$
then with probability at least $1 - \delta$,
$$
\begin{aligned}
\hat{\varepsilon}(h_0) &\ge \varepsilon(h_0) - \gamma \\
&> \varepsilon(h^*) + \eta - \gamma \\
&\ge \hat{\varepsilon}(h^*) + \eta - 2\gamma \\
&\ge \hat{\varepsilon}(\hat{h}) + \eta - 2\gamma.
\end{aligned}
$$
Here, the first and third inequalities follow from the fact that under the stated uniform convergence conditions, all hypotheses in $\mathcal{H}$ have empirical errors within $\gamma$ of their true generalization errors. The second inequality follows from our assumption, and the last inequality follows from the fact that $\hat{h}$ minimizes the empirical error. Therefore, the reverse condition, $\hat{\varepsilon}(h_0) < \hat{\varepsilon}(\hat{h}) + \eta - 2\gamma$ occurs with probability at most $\delta$.

### Problem 5(c) [8 points]

**Problem:** Finally, suppose that $h_0 = h^*$, and let $\eta > 0$ and $\delta > 0$ be fixed. Show that if $m$ is sufficiently large, then the probability that the algorithm returns YES is at least $1 - \delta$.

Hint: observe that for fixed $\eta$ and $\delta$, as $m \to \infty$, we have
$$ \gamma = \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}} \to 0. $$
This means that there are values of $m$ for which $2\gamma < \eta - 2\gamma$.

**Answer:** Suppose that $h_0 = h^*$. Using the Hoeffding inequality, we have that for
$$ \gamma = \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}} $$
then with probability at least $1 - \delta$,
$$ \hat{\varepsilon}(h_0) \le \varepsilon(h_0) + \gamma $$
$$ = \varepsilon(h^*) + \gamma $$
$$ \le \varepsilon(\hat{h}) + \gamma $$
$$ \le \hat{\varepsilon}(\hat{h}) + 2\gamma. $$
Here, the first and last inequalities follow from the fact that under the stated uniform convergence conditions, all hypotheses in $\mathcal{H}$ have empirical errors within $\gamma$ of their true generalization errors. The equality in the second step follows from our assumption, and the inequality in the third step follows from the fact that $h^*$ minimizes the true generalization error. But, observe that for fixed $\eta$ and $\delta$, as $m \to \infty$, we have
$$ \gamma = \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}} \to 0. $$
This implies that for $m$ sufficiently large, $4\gamma < \eta$, or equivalently, $2\gamma < \eta - 2\gamma$. It follows that with probability at least $1 - \delta$, if $m$ is sufficiently large, then
$$ \hat{\varepsilon}(h_0) \le \hat{\varepsilon}(\hat{h}) + \eta - 2\gamma, $$
so the algorithm returns YES.


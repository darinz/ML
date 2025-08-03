# Problem Set 1 Solutions

## Problem 1: Generalized Linear Models

### (a) [10 points] Log-likelihood Concavity

**Problem:** Given a training set $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$, the loglikelihood is given by
$$\ell(\theta) = \sum_{i=1}^m \log p(y^{(i)} | x^{(i)}; \theta).$$

Give a set of conditions on $b(y)$, $T(y)$, and $a(\eta)$ which ensure that the loglikelihood is a concave function of $\theta$ (and thus has a unique maximum). Your conditions must be reasonable, and should be as weak as possible.

**Answer:** The log-likelihood is given by
$$\ell(\theta) = \sum_{k=1}^M \log(b(y)) + \eta^{(k)}T(y) - a(\eta^{(k)})$$

where $\eta^{(k)} = \theta^T x^{(k)}$. Find the Hessian by taking the partials with respect to $\theta_i$ and $\theta_j$:

$$\frac{\partial}{\partial \theta_i} \ell(\theta) = \sum_{k=1}^M T(y)x_i^{(k)} - \frac{\partial}{\partial \eta} a(\eta^{(k)})x_i^{(k)}$$

$$\frac{\partial^2}{\partial \theta_i \partial \theta_j} \ell(\theta) = \sum_{k=1}^M - \frac{\partial^2}{\partial \eta^2} a(\eta^{(k)}) x_i^{(k)} x_j^{(k)}$$

The Hessian is:
$$H = -\sum_{k=1}^{M} \frac{\partial^2}{\partial \eta^2} a(\eta^{(k)}) x^{(k)} x^{(k)T}$$

For any vector $z$:
$$z^T H z = -\sum_{k=1}^{M} \frac{\partial^2}{\partial \eta^2} a(\eta^{(k)}) (z^T x^{(k)})^2$$

If $\frac{\partial^2}{\partial \eta^2} a(\eta) \ge 0$ for all $\eta$, then $z^T H z \le 0$. If H is negative semi-definite, then the original optimization problem is concave.

### (b) [3 points] Normal Distribution Verification

**Problem:** When the response variable is distributed according to a Normal distribution (with unit variance), we have $b(y) = \frac{1}{\sqrt{2\pi}} e^{-\frac{y^2}{2}}$, $T(y) = y$, and $a(\eta) = \frac{\eta^2}{2}$. Verify that the condition(s) you gave in part (a) hold for this setting.

**Answer:**
$$\frac{\partial^2}{\partial \eta^2} a(\eta) = 1 \ge 0.$$

## Problem 2: Bayesian Linear Regression

**Problem:** Consider Bayesian linear regression using a Gaussian prior on the parameters $\theta \in \mathbb{R}^{n+1}$. Thus, in our prior, $\theta \sim N(0, \tau^2 I_{n+1})$, where $\tau^2 \in \mathbb{R}$, and $I_{n+1}$ is the $n+1$-by-$n+1$ identity matrix. Also let the conditional distribution of $y^{(i)}$ given $x^{(i)}$ and $\theta$ be $N(\theta^T x^{(i)}, \sigma^2)$, as in our usual linear least-squares model.$^1$ Let a set of $m$ IID training examples be given (with $x^{(i)} \in \mathbb{R}^{n+1}$). Recall that the MAP estimate of the parameters $\theta$ is given by:
$$\theta_{MAP} = \arg \max_{\theta} \left( \prod_{i=1}^{m} p(y^{(i)}|x^{(i)}, \theta) \right) p(\theta)$$

Find, in closed form, the MAP estimate of the parameters $\theta$. For this problem, you should treat $\tau^2$ and $\sigma^2$ as fixed, known, constants. [Hint: Your solution should involve deriving something that looks a bit like the Normal equations.]

**Answer:**
$$\theta_{MAP} = \arg \max_{\theta} \left( \prod_{i=1}^{m} p(y^{(i)}|x^{(i)}, \theta) \right) p(\theta)$$
$$= \arg \max_{\theta} \log \left[ \left( \prod_{i=1}^{m} p(y^{(i)}|x^{(i)}, \theta) \right) p(\theta) \right]$$
$$= \arg \min_{\theta} \left( -\log p(\theta) - \sum_{i=1}^{m} \log p(y^{(i)}|x^{(i)}, \theta) \right) \quad (1)$$

Substituting expressions for $p(\theta)$ and $p(y^{(i)}|x^{(i)}, \theta)$, and dropping terms that don't affect the optimization, we get:
$$\theta_{MAP} = \arg \min_{\theta} \left( \frac{\sigma^2}{\tau^2} \theta^T \theta + (Y - X\theta)^T (Y - X\theta) \right) \quad (2)$$

In the above expression, $Y$ is an $m$-vector containing the training labels $y^{(i)}$, $X$ is an $m$-by-$n$ matrix with the data $x^{(i)}$, and $\theta$ is our vector of parameters. Taking derivatives with respect to $\theta$, equating to zero, and solving, we get:
$$\theta_{MAP} = (X^T X + \frac{\sigma^2}{\tau^2}I_n)^{-1}X^T Y \quad (3)$$

Observe the similarity between this expression, and the least squares solution derived in the notes.

$^1$Equivalently, $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$, where the $\epsilon^{(i)}$'s are distributed IID $N(0, \sigma^2)$.

## Problem 3: Kernels

**Problem:** In this problem, you will prove that certain functions $K$ give valid kernels. Be careful to justify every step in your proofs. Specifically, if you use a result proved either in the lecture notes or homeworks, be careful to state exactly which result you're using.

### (a) [8 points] Exponential Kernel

**Problem:** Let $K(x, z)$ be a valid (Mercer) kernel over $\mathbb{R}^n \times \mathbb{R}^n$. Consider the function given by
$$K_e(x, z) = \exp(K(x, z)).$$

Show that $K_e$ is a valid kernel. [Hint: There are many ways of proving this result, but you might find the following two facts useful: (i) The Taylor expansion of $e^x$ is given by $e^x = \sum_{j=0}^\infty \frac{1}{j!}x^j$ (ii) If a sequence of non-negative numbers $a_i \ge 0$ has a limit $a = \lim_{i \to \infty} a_i$, then $a \ge 0$.]

**Answer:** Let $K_i(x, z) = \sum_{j=0}^i \frac{1}{j!}K(x, z)^j$. $K_i$ is a polynomial in $K(x, z)$ with positive coefficients. As proved in the homework, $K_i(x, z)$ is also a kernel, so $z^T K_i z \ge 0$. Thus,
$$\lim_{i \to \infty} z^T K_i z \ge 0$$
$$z^T (\lim_{i \to \infty} K_i)z \ge 0$$

Since $\lim_{i \to \infty} K_i = K_e$, $K_e$ is positive semi-definite, and thus a valid kernel.

### (b) [8 points] Gaussian Kernel

**Problem:** The Gaussian kernel is given by the function
$$K(x,z) = e^{-\frac{||x-z||^2}{\sigma^2}},$$

where $\sigma^2 > 0$ is some fixed, positive constant. We said in class that this is a valid kernel, but did not prove it. Prove that the Gaussian kernel is indeed a valid kernel. [Hint: The following fact may be useful. $||x-z||^2 = ||x||^2 - 2x^Tz + ||z||^2$.]

**Answer:** We can rewrite the Gaussian kernel as
$$K(x, z) = e^{-\frac{||x||^2}{\sigma^2}} e^{-\frac{||z||^2}{\sigma^2}} e^{\frac{2}{\sigma^2}x^T z}$$

The first two terms together form a kernel by the fact proved in the homework that $K(x,z) = f(x)f(z)$ is a valid kernel. The third term is $e^{K(x,z)}$, which we've already shown to be a valid kernel. By the result proved in the homework, the product of two kernels is also a kernel.

## Problem 4: One-class SVM

**Problem:** Given an unlabeled set of examples $\{x^{(1)}, \dots, x^{(m)}\}$ the one-class SVM algorithm tries to find a direction $w$ that maximally separates the data from the origin.$^2$ More precisely, it solves the (primal) optimization problem:
$$\min_w \frac{1}{2} w^T w$$
$$\text{s.t.} \quad w^T x^{(i)} \ge 1 \quad \text{for all } i = 1, \dots, m$$

A new test example $x$ is labeled 1 if $w^T x \ge 1$, and 0 otherwise.

### (a) [9 points] Dual Formulation

**Problem:** The primal optimization problem for the one-class SVM was given above. Write down the corresponding dual optimization problem. Simplify your answer as much as possible. In particular, $w$ should not appear in your answer.

**Answer:** The Lagrangian is given by
$$L(w, \alpha) = \frac{1}{2} w^T w + \sum_{i=1}^m \alpha_i (1 - w^T x^{(i)}) \quad (4)$$

Setting the gradient of the Lagrangian with respect to $w$ to zero, we obtain $w = \sum_{i=1}^m \alpha_i x^{(i)}$. It follows that
$$\max_{\alpha \ge 0} \min_w \left( \frac{1}{2} w^T w + \sum_{i=1}^m \alpha_i (1 - w^T x^{(i)}) \right) \quad (5)$$
$$= \max_{\alpha \ge 0} \left( \frac{1}{2} \left( \sum_{i=1}^m \alpha_i x^{(i)} \right)^T \left( \sum_{i=1}^m \alpha_i x^{(i)} \right) + \sum_{i=1}^m \alpha_i \left( 1 - \left( \sum_{i=1}^m \alpha_i x^{(i)} \right)^T x^{(i)} \right) \right) \quad (6)$$
$$= \max_{\alpha \ge 0} \left( \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j x^{(i)^T} x^{(j)} \right) \quad (7)$$

The first equality follows from setting the gradient w.r.t. $w$ equal to zero, and solving for $w$, which gives $w = \sum_{i=1}^m \alpha_i x^{(i)}$ and substituting this expression for $w$. The second equality follows from simplifying the expression.

### (b) [4 points] Kernelization

**Problem:** Can the one-class SVM be kernelized (both in training and testing)? Justify your answer.

**Answer:** Yes. For training we can use the dual formulation, in which only inner products of the data appear. For testing at a point $z$ we just need to evaluate $w^T z = (\sum_{i=1}^m \alpha_i x^{(i)})^T z = \sum_{i=1}^m \alpha_i x^{(i)^T} z$ in which the training data and the test point $z$ only appear in inner products.

### (c) [5 points] SMO-like Algorithm

**Problem:** Give an SMO-like algorithm to optimize the dual. I.e., give an algorithm that in every optimization step optimizes over the smallest possible subset of variables. Also give in closed-form the update equation for this subset of variables. You should also justify why it is sufficient to consider this many variables at a time in each step.

**Answer:** Since we have convex optimization problem with only independent coordinate wise constraints ($\alpha_i \ge 0$), we can optimize iteratively over 1 variable at a time. Optimizing w.r.t. $\alpha_i$ is done by setting
$$\alpha_i = \max \left\{ 0, \frac{1}{K_{i,i}} \left( 1 - \sum_{j \ne i} \alpha_j K_{i,j} \right) \right\}$$

(Set the derivative w.r.t. $\alpha_i$ equal to zero and solve for $\alpha_i$. And take into account the constraint. Here, we defined $K_{i,j} = x^{(i)\text{T}} x^{(j)}$.)

$^2$This turns out to be useful for anomaly detection. In anomaly detection you are given a set of data points that are all considered to be 'normal'. Given these 'normal' data points, the task is to decide for a new data point whether it is also 'normal' or not. Adding slack variables allows for training data that are not necessarily all 'normal'. The most common formulation with slack variables is not the most direct adaptation of the soft margin SVM formulation seen in class. Instead the $\nu$-SVM formulation is often used. This formulation allows you to specify the fraction of outliers (instead of the constant $C$ which is harder to interpret). See the literature for details.

## Problem 5: Uniform Convergence

**Problem:** In this problem, we consider trying to estimate the mean of a biased coin toss. We will repeatedly toss the coin and keep a running estimate of the mean. We would like to prove that (with high probability), after some initial set of $N$ tosses, the running estimate from that point on will always be accurate and never deviate too much from the true value.

More formally, let $X_i \sim \text{Bernoulli}(\phi)$ be IID random variables. Let $\hat{\phi}_n$ be our estimate for $\phi$ after $n$ observations:
$$\hat{\phi}_n = \frac{1}{n} \sum_{i=1}^{n} X_i.$$

We'd like to show that after a certain number of coin flips, our estimates will stay close to the true value of $\phi$. More formally, we'd like to show that for all $\gamma, \delta \in (0,1/2]$, there exists a value $N$ such that
$$P\left( \max_{n \ge N} |\phi - \hat{\phi}_n| > \gamma \right) \le \delta.$$

Show that in order to make the guarantee above, it suffices to have $N = O\left(\frac{1}{\gamma^2} \log\left(\frac{1}{\delta\gamma}\right)\right)$. You may need to use the fact that for $\gamma \in (0,1/2]$, $\log\left(\frac{1}{1-\exp(-2\gamma^2)}\right) = O(\log(\frac{1}{\gamma}))$.

[Hint: Let $A_n$ be the event that $|\phi - \hat{\phi}_n| > \gamma$ and consider taking a union bound over the set of events $A_N, A_{N+1}, A_{N+2}, \dots$]

**Answer:**
$$\text{Pr}(\max_{n \ge N} |\phi - \hat{\phi}_n| > \gamma)$$
$$= \text{Pr}(\bigcup_{n \ge N} \{|\phi - \hat{\phi}_n| > \gamma\})$$
$$\le \sum_{n \ge N} \text{Pr}(|\phi - \hat{\phi}_n| > \gamma)$$
$$\le \sum_{n \ge N} 2e^{-2\gamma^2 n}$$
$$= \frac{2(e^{-2\gamma^2})^N}{1 - e^{-2\gamma^2}}$$

Hence, in order to guarantee that $\text{Pr}(\max_{n \ge N} |\theta - \hat{\theta}_n| > \gamma) \le \delta$, we only need to choose $N$ such that
$$\frac{2(e^{-2\gamma^2})^N}{1 - e^{-2\gamma^2}} \le \delta$$
$$(e^{-2\gamma^2})^N \le \delta(1 - e^{-2\gamma^2})/2$$
$$\log((e^{-2\gamma^2})^N) \le \log(\delta(1 - e^{-2\gamma^2})/2)$$
$$(-2\gamma^2)N \le \log \delta + \log(1 - e^{-2\gamma^2}) - \log 2$$
$$N \ge \frac{1}{2\gamma^2}(-\log \delta - \log(1 - e^{-2\gamma^2}) + \log 2)$$
$$N \ge \frac{1}{2\gamma^2}\left(\log \frac{1}{\delta} + \log\left(\frac{1}{1 - e^{-2\gamma^2}}\right) + \log 2\right)$$

Thus, it is sufficient to have
$$N = O\left(\frac{1}{\gamma^2}\left(\log \frac{1}{\delta} + \log\left(\frac{1}{1 - e^{-2\gamma^2}}\right)\right)\right)$$
$$N = O\left(\frac{1}{\gamma^2}\left(\log \frac{1}{\delta} + \log \frac{1}{\gamma}\right)\right)$$
$$N = O\left(\frac{1}{\gamma^2} \log \frac{1}{\delta\gamma}\right)$$

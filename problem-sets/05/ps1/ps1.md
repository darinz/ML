# Problem Set 1 Problems

## Problem 1: Generalized Linear Models

### (a) [10 points] Log-likelihood Concavity

**Problem:** Given a training set $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$, the loglikelihood is given by
$$\ell(\theta) = \sum_{i=1}^m \log p(y^{(i)} | x^{(i)}; \theta).$$

Give a set of conditions on $b(y)$, $T(y)$, and $a(\eta)$ which ensure that the loglikelihood is a concave function of $\theta$ (and thus has a unique maximum). Your conditions must be reasonable, and should be as weak as possible.

### (b) [3 points] Normal Distribution Verification

**Problem:** When the response variable is distributed according to a Normal distribution (with unit variance), we have $b(y) = \frac{1}{\sqrt{2\pi}} e^{-\frac{y^2}{2}}$, $T(y) = y$, and $a(\eta) = \frac{\eta^2}{2}$. Verify that the condition(s) you gave in part (a) hold for this setting.

## Problem 2: Bayesian Linear Regression

**Problem:** Consider Bayesian linear regression using a Gaussian prior on the parameters $\theta \in \mathbb{R}^{n+1}$. Thus, in our prior, $\theta \sim N(0, \tau^2 I_{n+1})$, where $\tau^2 \in \mathbb{R}$, and $I_{n+1}$ is the $n+1$-by-$n+1$ identity matrix. Also let the conditional distribution of $y^{(i)}$ given $x^{(i)}$ and $\theta$ be $N(\theta^T x^{(i)}, \sigma^2)$, as in our usual linear least-squares model.$^1$ Let a set of $m$ IID training examples be given (with $x^{(i)} \in \mathbb{R}^{n+1}$). Recall that the MAP estimate of the parameters $\theta$ is given by:
$$\theta_{MAP} = \arg \max_{\theta} \left( \prod_{i=1}^{m} p(y^{(i)}|x^{(i)}, \theta) \right) p(\theta)$$

Find, in closed form, the MAP estimate of the parameters $\theta$. For this problem, you should treat $\tau^2$ and $\sigma^2$ as fixed, known, constants. [Hint: Your solution should involve deriving something that looks a bit like the Normal equations.]

$^1$Equivalently, $y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$, where the $\epsilon^{(i)}$'s are distributed IID $N(0, \sigma^2)$.

## Problem 3: Kernels

**Problem:** In this problem, you will prove that certain functions $K$ give valid kernels. Be careful to justify every step in your proofs. Specifically, if you use a result proved either in the lecture notes or homeworks, be careful to state exactly which result you're using.

### (a) [8 points] Exponential Kernel

**Problem:** Let $K(x, z)$ be a valid (Mercer) kernel over $\mathbb{R}^n \times \mathbb{R}^n$. Consider the function given by
$$K_e(x, z) = \exp(K(x, z)).$$

Show that $K_e$ is a valid kernel. [Hint: There are many ways of proving this result, but you might find the following two facts useful: (i) The Taylor expansion of $e^x$ is given by $e^x = \sum_{j=0}^\infty \frac{1}{j!}x^j$ (ii) If a sequence of non-negative numbers $a_i \ge 0$ has a limit $a = \lim_{i \to \infty} a_i$, then $a \ge 0$.]

### (b) [8 points] Gaussian Kernel

**Problem:** The Gaussian kernel is given by the function
$$K(x,z) = e^{-\frac{||x-z||^2}{\sigma^2}},$$

where $\sigma^2 > 0$ is some fixed, positive constant. We said in class that this is a valid kernel, but did not prove it. Prove that the Gaussian kernel is indeed a valid kernel. [Hint: The following fact may be useful. $||x-z||^2 = ||x||^2 - 2x^Tz + ||z||^2$.]

## Problem 4: One-class SVM

**Problem:** Given an unlabeled set of examples $\{x^{(1)}, \dots, x^{(m)}\}$ the one-class SVM algorithm tries to find a direction $w$ that maximally separates the data from the origin.$^2$ More precisely, it solves the (primal) optimization problem:
$$\min_w \frac{1}{2} w^T w$$
$$\text{s.t.} \quad w^T x^{(i)} \ge 1 \quad \text{for all } i = 1, \dots, m$$

A new test example $x$ is labeled 1 if $w^T x \ge 1$, and 0 otherwise.

### (a) [9 points] Dual Formulation

**Problem:** The primal optimization problem for the one-class SVM was given above. Write down the corresponding dual optimization problem. Simplify your answer as much as possible. In particular, $w$ should not appear in your answer.

### (b) [4 points] Kernelization

**Problem:** Can the one-class SVM be kernelized (both in training and testing)? Justify your answer.

### (c) [5 points] SMO-like Algorithm

**Problem:** Give an SMO-like algorithm to optimize the dual. I.e., give an algorithm that in every optimization step optimizes over the smallest possible subset of variables. Also give in closed-form the update equation for this subset of variables. You should also justify why it is sufficient to consider this many variables at a time in each step.

$^2$This turns out to be useful for anomaly detection. In anomaly detection you are given a set of data points that are all considered to be 'normal'. Given these 'normal' data points, the task is to decide for a new data point whether it is also 'normal' or not. Adding slack variables allows for training data that are not necessarily all 'normal'. The most common formulation with slack variables is not the most direct adaptation of the soft margin SVM formulation seen in class. Instead the $\nu$-SVM formulation is often used. This formulation allows you to specify the fraction of outliers (instead of the constant $C$ which is harder to interpret). See the literature for details.

## Problem 5: Uniform Convergence

**Problem:** In this problem, we consider trying to estimate the mean of a biased coin toss. We will repeatedly toss the coin and keep a running estimate of the mean. We would like to prove that (with high probability), after some initial set of $N$ tosses, the running estimate from that point on will always be accurate and never deviate too much from the true value.

More formally, let $X_i \sim \text{Bernoulli}(\phi)$ be IID random variables. Let $\hat{\phi}_n$ be our estimate for $\phi$ after $n$ observations:
$$\hat{\phi}_n = \frac{1}{n} \sum_{i=1}^{n} X_i.$$

We'd like to show that after a certain number of coin flips, our estimates will stay close to the true value of $\phi$. More formally, we'd like to show that for all $\gamma, \delta \in (0,1/2]$, there exists a value $N$ such that
$$P\left( \max_{n \ge N} |\phi - \hat{\phi}_n| > \gamma \right) \le \delta.$$

Show that in order to make the guarantee above, it suffices to have $N = O\left(\frac{1}{\gamma^2} \log\left(\frac{1}{\delta\gamma}\right)\right)$. You may need to use the fact that for $\gamma \in (0,1/2]$, $\log\left(\frac{1}{1-\exp(-2\gamma^2)}\right) = O(\log(\frac{1}{\gamma}))$.

[Hint: Let $A_n$ be the event that $|\phi - \hat{\phi}_n| > \gamma$ and consider taking a union bound over the set of events $A_N, A_{N+1}, A_{N+2}, \dots$]

## Problem 6: Short Answers

The following questions require a true/false accompanied by one sentence of explanation, or a reasonably short answer (usually at most 1-2 sentences or a figure).
To discourage random guessing, one point will be deducted for a wrong answer on multiple choice questions! Also, no credit will be given for answers without a correct explanation.

### (a) [5 points] Decision Boundary with Separate Covariance Matrices

**Problem:** Let there be a binary classification problem with continuous-valued features. In Problem Set #1, you showed if we apply Gaussian discriminant analysis using the same covariance matrix $\Sigma$ for both classes, then the resulting decision boundary will be linear. What will the decision boundary look like if we modeled the two classes using separate covariance matrices $\Sigma_0$ and $\Sigma_1$? (I.e., $x^{(i)}|y^{(i)} = b \sim \mathcal{N}(\mu_b, \Sigma_b)$, for $b = 0$ or $1$.)

### (b) [5 points] Kernel Perceptron Mistakes

**Problem:** Consider a sequence of examples $(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(m)}, y^{(m)})$. Assume that for all $i$ we have $||x^{(i)}|| \le D$ and that the data are linearly separated with a margin $\gamma$. Suppose that the perceptron algorithm makes exactly $(D/\gamma)^2$ mistakes on this sequence of examples. Now, suppose we use a feature mapping $\phi(\cdot)$ to a higher dimensional space and use the corresponding kernel perceptron algorithm on the same sequence of data (now in the higher-dimensional feature space). Then the kernel perceptron (implicitly operating in this higher dimensional feature space) will make a number of mistakes that is

i. strictly less than $(D/\gamma)^2$.
ii. equal to $(D/\gamma)^2$.
iii. strictly more than $(D/\gamma)^2$.
iv. impossible to say from the given information.

### (c) [5 points] Mercer Kernel Construction

**Problem:** Let any $x^{(1)}, x^{(2)}, x^{(3)} \in \mathbb{R}^p$ be given ($x^{(1)} \ne x^{(2)}, x^{(1)} \ne x^{(3)}, x^{(2)} \ne x^{(3)}$). Also let any $z^{(1)}, z^{(2)}, z^{(3)} \in \mathbb{R}^q$ be fixed. Then there exists a valid Mercer kernel $K: \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}$ such that for all $i, j \in \{1,2,3\}$ we have $K(x^{(i)}, x^{(j)}) = (z^{(i)})^T z^{(j)}$. True or False?

### (d) [5 points] Newton's Method Convergence

**Problem:** Let $f: \mathbb{R}^n \to \mathbb{R}$ be defined according to $f(x) = \frac{1}{2}x^T Ax + b^T x + c$, where $A$ is symmetric positive definite. Suppose we use Newton's method to minimize $f$. Show that Newton's method will find the optimum in exactly one iteration. You may assume that Newton's method is initialized with $\vec{0}$.

### (e) [5 points] Boolean Functions VC Dimension

**Problem:** Consider binary classification, and let the input domain be $\mathcal{X} = \{0, 1\}^n$, i.e., the space of all $n$-dimensional bit vectors. Thus, each sample $x$ has $n$ binary-valued features. Let $\mathcal{H}_n$ be the class of all boolean functions over the input space. What is $|\mathcal{H}_n|$ and $VC(\mathcal{H}_n)$?

### (f) [5 points] L1-Regularized SVM on Linearly Separable Data

**Problem:** Suppose an $l_1$-regularized SVM (with regularization parameter $C > 0$) is trained on a dataset that is linearly separable. Because the data is linearly separable, to minimize the primal objective, the SVM algorithm will set all the slack variables to zero. Thus, the weight vector $w$ obtained will be the same no matter what regularization parameter $C$ is used (so long as it is strictly bigger than zero). True or false?

### (g) [5 points] Locally Weighted Linear Regression Bandwidth

**Problem:** Consider using hold-out cross validation (using 70% of the data for training, 30% for hold-out CV) to select the bandwidth parameter $\tau$ for locally weighted linear regression. As the number of training examples $m$ increases, would you expect the value of $\tau$ selected by the algorithm to generally become larger, smaller, or neither of the above? For this problem, assume that (the expected value of) $y$ is a non-linear function of $x$.

### (h) [5 points] Feature Selection Algorithms

**Problem:** Consider a feature selection problem in which the mutual information $MI(x_i, y) = 0$ for all features $x_i$. Also for every subset of features $S_i = \{x_{i_1}, \dots, x_{i_k}\}$ of size $< n/2$ we have $MI(S_i, y) = 0.^3$ However there is a subset $S^*$ of size exactly $n/2$ such that $MI(S^*, y) = 1$. I.e. this subset of features allows us to predict $y$ correctly. Of the three feature selection algorithms listed below, which one do you expect to work best on this dataset?

i. Forward Search.
ii. Backward Search.
iii. Filtering using mutual information $MI(x_i, y)$.
iv. All three are expected to perform reasonably well.

$^3 MI(S_i, y) = \sum_{S_i} \sum_y P(S_i, y) \log(P(S_i, y)/P(S_i) P(y))$, where the first summation is over all possible values of the features in $S_i$.

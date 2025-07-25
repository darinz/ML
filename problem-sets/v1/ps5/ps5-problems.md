# Problem Set #5: topics from ps1 to ps4

### 1. Generalized Linear Models

Recall that generalized linear models assume that the response variable y (conditioned on x) is distributed according to a member of the exponential family:

$`P (y; \eta) = b(y) \exp(\eta^T(y) - a(\eta))`$

where $`\eta = \theta^T x`$. For this problem, we will assume $`\eta \in \mathbb{R}`$.

**(a)** Given a training set $`\{(x^{(i)}, y^{(i)})\}_{i=1}^m`$, the loglikelihood is given by:

```math
\ell(\theta) = \sum_{i=1}^{m} \log p(y^{(i)} |x^{(i)}; \theta)
```

Give a set of conditions on $`b(y)`$, $`T(y)`$, and $`a(\eta)`$ which ensure that the loglikelihood is a concave function of $`\theta`$ (and thus has a unique maximum). Your conditions must be reasonable, and should be as weak as possible.

**(b)** When the response variable is distributed according to a Normal distribution (with unit variance), we have:

$`b(y) = \frac{1}{\sqrt{2\pi}} e^{-y^2 / 2}, \quad T(y) = y, \quad \text{and} \quad a(\eta) = \frac{\eta^2}{2}`$

Verify that the condition(s) you gave in part (a) hold for this setting.

---

### 2. Bayesian Linear Regression

Consider Bayesian linear regression using a Gaussian prior on the parameters $`\theta \in \mathbb{R}^{n+1}`$. Thus, in our prior, $`\theta \sim \mathcal{N}(\vec{0}, \tau^2 I_n)`$, where $`\tau^2 \in \mathbb{R}`$, and $`I_{n+1}`$ is the $`n + 1`$-by-$`n + 1`$ identity matrix. Also let the conditional distribution of $`y^{(i)}`$ given $`x^{(i)}`$ and $`\theta`$ be $`\mathcal{N}(\theta^T x^{(i)}, \sigma^2)`$, as in our usual linear least-squares model.
\[Equivalently, $`y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}`$, where the $`\epsilon^{(i)}`$’s are distributed IID $`\mathcal{N}(0, \sigma^2)`$]

Let a set of m IID training examples be given (with $`x^{(i)} \in \mathbb{R}^{n+1}`$). Recall that the MAP estimate of the parameters $`\theta`$ is given by:

```math
\theta_{\text{MAP}} = \arg \max_\theta \left( \prod_{i=1}^{m} p(y^{(i)}|x^{(i)}, \theta) \right) p(\theta)
```

Find, in closed form, the MAP estimate of the parameters $`\theta`$. For this problem, you should treat $`\tau^2`$ and $`\sigma^2`$ as fixed, known, constants.

\[Hint: Your solution should involve deriving something that looks a bit like the Normal equations.]

---

### 3. Kernels

In this problem, you will prove that certain functions $`K`$ give valid kernels. Be careful to justify every step in your proofs. Specifically, if you use a result proved either in the lecture notes or homeworks, be careful to state exactly which result you’re using.

**(a)** Let $`K(x, z)`$ be a valid (Mercer) kernel over $`\mathbb{R}^n \times \mathbb{R}^n`$. Consider the function given by:

$`K_e(x, z) = \exp(K(x, z))`$

Show that $`K_e`$ is a valid kernel.

\[Hint: There are many ways of proving this result, but you might find the following two facts useful:
(i) The Taylor expansion of $`e^x`$ is given by $`e^x = \sum_{j=0}^{\infty} \frac{1}{j!}x^j`$
(ii) If a sequence of non-negative numbers $`a_i \geq 0`$ has a limit $`a = \lim_{i \to \infty} a_i`$, then $`a \geq 0`$]

**(b)** The Gaussian kernel is given by the function:

$`K(x, z) = e^{ - \frac{ \|x - z\|^2 }{\sigma^2} }`$

where $`\sigma^2 > 0`$ is some fixed, positive constant. We said in class that this is a valid kernel, but did not prove it. Prove that the Gaussian kernel is indeed a valid kernel.

\[Hint: The following fact may be useful. $`\|x - z\|^2 = \|x\|^2 - 2x^T z + \|z\|^2`$]

---

### 4. One-class SVM

Given an unlabeled set of examples $`\{x^{(1)}, \ldots, x^{(m)}\}`$ the one-class SVM algorithm tries to find a direction $`w`$ that maximally separates the data from the origin.

More precisely, it solves the (primal) optimization problem:

```math
\min_w \quad \frac{1}{2} w^\top w \quad \text{s.t.} \quad w^\top x^{(i)} \geq 1 \quad \forall i = 1, \ldots, m
```

A new test example x is labeled 1 if $`w^\top x \geq 1`$, and 0 otherwise.

**(a)** The primal optimization problem for the one-class SVM was given above. Write down the corresponding dual optimization problem. Simplify your answer as much as possible. In particular, $`w`$ should not appear in your answer.

**(b)** Can the one-class SVM be kernelized (both in training and testing)? Justify your answer.

**(c)** Give an SMO-like algorithm to optimize the dual. I.e., give an algorithm that in every optimization step optimizes over the smallest possible subset of variables. Also give in closed-form the update equation for this subset of variables. You should also justify why it is sufficient to consider this many variables at a time in each step.

---

### 5. Uniform Convergence

In this problem, we consider trying to estimate the mean of a biased coin toss. We will repeatedly toss the coin and keep a running estimate of the mean. We would like to prove that (with high probability), after some initial set of $`N`$ tosses, the running estimate from that point on will always be accurate and never deviate too much from the true value.

More formally, let $`X_i \sim \text{Bernoulli}(\phi)`$ be IID random variables. Let $`\hat{\phi}_n`$ be our estimate for $`\phi`$ after n observations:

```math
\hat{\phi}_n = \frac{1}{n} \sum_{i=1}^{n} X_i
```

We’d like to show that after a certain number of coin flips, our estimates will stay close to the true value of $`\phi`$. More formally, we’d like to show that for all $`\gamma, \delta \in (0, 1/2]`$, there exists a value $`N`$ such that:

```math
P\left( \max_{n \geq N} |\phi - \hat{\phi}_n| > \gamma \right) \leq \delta
```

Show that in order to make the guarantee above, it suffices to have:

```math
N = O\left( \frac{1}{\gamma^2} \log\left(\frac{1}{\delta \gamma}\right) \right)
```

You may need to use the fact that for $`\gamma \in (0, 1/2]`$,
$`\log\left( \frac{1}{1 - \exp(-2\gamma^2)} \right) = O\left( \log\left( \frac{1}{\gamma} \right) \right)`$

\[Hint: Let $`A_n`$ be the event that $`|\phi - \hat{\phi}_n| > \gamma`$ and consider taking a union bound over the set of events $`A_n, A_{n+1}, A_{n+2}, \ldots`$]

---

### 6. Short Answers

These questions require a true/false with explanation or short answer (1–2 sentences or a figure). Wrong multiple choice answers deduct 1 point. No credit without correct explanation.

**(a)** Binary classification with continuous features. Gaussian discriminant analysis with same covariance $`\Sigma`$ gives linear boundary. What happens if we use different covariances $`\Sigma_0`$ and $`\Sigma_1`$?

**(b)** Perceptron makes $`(D/\gamma)^2`$ mistakes. Kernel perceptron now used. Which is true?

i. strictly less than $`(D/\gamma)^2`$
ii. equal to $`(D/\gamma)^2`$
iii. strictly more than $`(D/\gamma)^2`$
iv. impossible to say from the given information

**(c)** Given $`x^{(1)}, x^{(2)}, x^{(3)} \in \mathbb{R}^p`$ and $`z^{(1)}, z^{(2)}, z^{(3)} \in \mathbb{R}^q`$, there exists a valid Mercer kernel $`K`$ such that $`K(x^{(i)}, x^{(j)}) = (z^{(i)})^\top z^{(j)}`$.
True or False?

**(d)** $`f(x) = x^\top A x + b^\top x + c`$ where $`A`$ is symmetric positive definite. Show Newton's method finds optimum in one iteration (start at $`\vec{0}`$).

**(e)** For binary classification $`X = \{0, 1\}^n`$, class $`H_n`$ of all boolean functions. What is $`|H_n|`$ and $`\text{VC}(H_n)`$?

**(f)** $`\ell_1`$-regularized SVM on linearly separable data: will optimal $`w`$ be same regardless of $`C > 0`$?
True or False?

**(g)** Using 70/30 train/hold-out to select $`\tau`$ for locally weighted regression. As $`m`$ increases, expect $`\tau`$ to:

* increase
* decrease
* neither

Assume $`\mathbb{E}[y]`$ is non-linear in $`x`$.

**(h)** Mutual information $`MI(x_i, y) = 0`$ for all $`x_i`$ and subsets $`S_i`$ of size < $`n/2`$. But one subset $`S^*`$ of size = $`n/2`$ gives $`MI(S^*, y) = 1`$. Which feature selection works best?

i. Forward Search

ii. Backward Search

iii. Filtering using $`MI(x_i, y)`$

iv. All three perform well
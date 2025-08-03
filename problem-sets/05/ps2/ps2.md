# Problem Set 2

## Problem 1: Least Squares

### Problem 1(a) [2 points]

**Problem:** Let $\hat{\theta} = \arg \min_{\theta} J(\theta)$ be the minimizer of the original least squares objective (using the original design matrix $X$). Using the orthonormality assumption, show that $J(\hat{\theta}) = (XX^T \bar{y} - \bar{y})^T (XX^T \bar{y} - \bar{y})$. I.e., show that this is the value of $\min_{\theta} J(\theta)$ (the value of the objective at the minimum).

### Problem 1(b) [5 points]

**Problem:** Now let $\hat{\theta}_{\text{new}}$ be the minimizer for $\tilde{J}(\theta_{\text{new}}) = (\tilde{X}\theta_{\text{new}} - \vec{y})^T(\tilde{X}\theta_{\text{new}} - \vec{y})$. Find the new minimized objective $\tilde{J}(\hat{\theta}_{\text{new}})$ and write this expression in the form: $\tilde{J}(\hat{\theta}_{\text{new}}) = J(\hat{\theta}) + f(X, \vec{v}, \vec{y})$ where $J(\hat{\theta})$ is as derived in part (a) and $f$ is some function of $X, \vec{v}$, and $\vec{y}$.

### Problem 1(c) [6 points]

**Problem:** Prove that the optimal objective value does not increase upon adding a feature to the design matrix. That is, show $\tilde{J}(\hat{\theta}_{\text{new}}) \le J(\hat{\theta})$.

### Problem 1(d) [3 points]

**Problem:** Does the above result show that if we keep increasing the number of features, we can always get a model that generalizes better than a model with fewer features? Explain why or why not.

## Problem 2: Decision Boundaries for Generative Models

### Problem 2(a) [7 points]

**Problem:** Consider the multinomial event model of Naive Bayes. Our goal in this problem is to show that this is a linear classifier.

For a given text document $x$, let $c_1, \dots, c_V$ indicate the number of times each word (out of $V$ words) appears in the document. Thus, $c_i \in \{0, 1, 2, \dots\}$ counts the occurrences of word $i$. Recall that the Naive Bayes model uses parameters $\phi_y = p(y = 1)$, $\phi_{i|y=1} = p(\text{word i appears in a specific document position } | y = 1)$ and $\phi_{i|y=0} = p(\text{word i appears in a specific document position } | y = 0)$.

We say a classifier is linear if it assigns a label $y = 1$ using a decision rule of the form
$$ \sum_{i=1}^V w_i c_i + b \ge 0 $$
I.e., the classifier predicts "$y = 1$" if $\sum_{i=1}^V w_i c_i + b \ge 0$, and predicts "$y = 0$" otherwise.

Show that Naive Bayes is a linear classifier, and clearly state the values of $w_i$ and $b$ in terms of the Naive Bayes parameters. (Don't worry about whether the decision rule uses "$\ge$" or "$>$.") Hint: consider using log-probabilities.

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

## Problem 3: Generalized Linear Models

### Problem 3(a) i. [5 points]

**Problem:** Show that the geometric distribution is an exponential family distribution. Explicitly specify the components $b(y)$, $\eta$, $T(y)$, and $\alpha(\eta)$, as well as express $\phi$ in terms of $\eta$.

### Problem 3(a) ii. [5 points]

**Problem:** Suppose that we have an IID training set $\{(x^{(i)}, y^{(i)}), i = 1, ..., m\}$ and we wish to model this using a GLM based on a geometric distribution. Find the log-likelihood $\log \prod_{i=1}^{m} p(y^{(i)}|x^{(i)}; \theta)$ defined with respect to the entire training set.

### Problem 3(b) [6 points]

**Problem:** Derive the Hessian $H$ and the gradient vector of the log likelihood with respect to $\theta$, and state what one step of Newton's method for maximizing the log likelihood would be.

### Problem 3(c) [2 points]

**Problem:** Show that the Hessian is negative semi-definite, which implies the optimization objective is concave and Newton's method maximizes log-likelihood.

## Problem 4: Support Vector Regression

### Problem 4(a) [4 points]

**Problem:** Write down the Lagrangian for the optimization problem above. We suggest you use two sets of Lagrange multipliers $\alpha_i$ and $\alpha_i^*$, corresponding to the two inequality constraints (labeled (1) and (2) above), so that the Lagrangian would be written $\mathcal{L}(w, b, \alpha, \alpha^*)$.

### Problem 4(b) [10 points]

**Problem:** Derive the dual optimization problem. You will have to take derivatives of the Lagrangian with respect to $w$ and $b$.

### Problem 4(c) [4 points]

**Problem:** Show that this algorithm can be kernelized. For this, you have to show that (i) the dual optimization objective can be written in terms of inner-products of training examples; and (ii) at test time, given a new $x$ the hypothesis $h_{w,b}(x)$ can also be computed in terms of inner products.

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

### Problem 5(b) [6 points]

**Problem:** Second, show that if $\varepsilon(h_0) > \varepsilon(h^*) + \eta$ (i.e., $h_0$ is not $\eta$-optimal), then the probability that the algorithm returns YES is at most $\delta$.

### Problem 5(c) [8 points]

**Problem:** Finally, suppose that $h_0 = h^*$, and let $\eta > 0$ and $\delta > 0$ be fixed. Show that if $m$ is sufficiently large, then the probability that the algorithm returns YES is at least $1 - \delta$.

Hint: observe that for fixed $\eta$ and $\delta$, as $m \to \infty$, we have
$$ \gamma = \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}} \to 0. $$
This means that there are values of $m$ for which $2\gamma < \eta - 2\gamma$.

## Problem 6: Short answers

The following questions require a reasonably short answer (usually at most 2-3 sentences or a figure, though some questions may require longer or shorter explanations). To discourage random guessing, one point will be deducted for a wrong answer on true/false or multiple choice questions! Also, no credit will be given for answers without a correct explanation.

### Problem 6(a) [3 points]

**Problem:** You have an implementation of Newton's method and gradient descent. Suppose that one iteration of Newton's method takes twice as long as one iteration of gradient descent. Then, this implies that gradient descent will converge to the optimal objective faster. True/False?

### Problem 6(b) [3 points]

**Problem:** A stochastic gradient descent algorithm for training logistic regression with a fixed learning rate will always converge to exactly the optimal setting of the parameters $\theta^* = \arg \max_{\theta} \prod_{i=1}^{m} p(y^{(i)}|x^{(i)}; \theta)$, assuming a reasonable choice of the learning rate. True/False?

### Problem 6(c) [3 points]

**Problem:** Given a valid kernel $K(x, y)$ over $\mathbb{R}^m$, is $K_{norm}(x, y) = \frac{K(x,y)}{\sqrt{K(x,x)K(y,y)}}$ a valid kernel?

### Problem 6(d) [3 points]

**Problem:** Consider a 2 class classification problem with a dataset of inputs $\{x^{(1)} = (-1,-1), x^{(2)} = (-1, +1), x^{(3)} = (+1,-1), x^{(4)} = (+1,+1)\}$. Can a linear SVM (with no kernel trick) shatter this set of 4 points?

### Problem 6(e) [3 points]

**Problem:** The vector of learned weights $w$ for linear hypotheses of the form $h(x) = w^T x + b$ is always perpendicular to the separating hyperplane. True/False? Justify your answer.

### Problem 6(f) [3 points]

**Problem:** Let $\mathcal{H}$ be a set of classifiers with a VC dimension of 5. Consider a set of 5 training examples $\{(x^{(1)}, y^{(1)}), \dots, (x^{(5)}, y^{(5)}) \}$. Now we select a classifier $h^*$ from $\mathcal{H}$ by minimizing the classification error on the training set. Which one of the following is true?

i. $x^{(5)}$ will certainly be classified correctly (i.e. $h^*(x^{(5)}) = y^{(5)}$)
ii. $x^{(5)}$ will certainly be classified incorrectly (i.e. $h^*(x^{(5)}) \ne y^{(5)}$)
iii. We cannot tell

Briefly justify your answer.

### Problem 6(g) [6 points]

**Problem:** Suppose you would like to use a linear regression model in order to predict the price of houses. In your model, you use the features $x_0 = 1$, $x_1 = \text{size in square meters}$, $x_2 = \text{height of roof in meters}$. Now, suppose a friend repeats the same analysis using exactly the same training set, only he represents the data instead using features $x'_0 = 1$, $x'_1 = x_1$, and $x'_2 = \text{height in cm (so } x'_2 = 100x_2)$.

#### Problem 6(g) i. [3 points]

**Problem:** Suppose both of you run linear regression, solving for the parameters via the Normal equations. (Assume there are no degeneracies, so this gives a unique solution to the parameters.) You get parameters $\theta_0, \theta_1, \theta_2$; your friend gets $\theta'_0, \theta'_1, \theta'_2$. Then $\theta'_0 = \theta_0, \theta'_1 = \theta_1, \theta'_2 = \frac{1}{100}\theta_2$. True/False?

#### Problem 6(g) ii. [3 points]

**Problem:** Suppose both of you run linear regression, initializing the parameters to 0, and compare your results after running just *one* iteration of batch gradient descent. You get parameters $\theta_0, \theta_1, \theta_2$; your friend gets $\theta'_0, \theta'_1, \theta'_2$. Then $\theta'_0 = \theta_0, \theta'_1 = \theta_1, \theta'_2 = \frac{1}{100}\theta_2$. True/False?


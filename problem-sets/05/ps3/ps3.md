# Problem Set 3

## 1. [26 points] Short answers

The following questions require a reasonably short answer (usually at most 2-3 sentences or a figure, though some questions may require longer or shorter explanations).

**To discourage random guessing, one point will be deducted for a wrong answer on true/false or multiple choice questions! Also, no credit will be given for answers without a correct explanation.**

### (a) [6 points] 

Suppose you are fitting a fixed dataset with $m$ training examples using linear regression, $h_\theta(x) = \theta^T x$, where $\theta, x \in \mathbb{R}^{n+1}$. After training, you realize that the variance of your model is relatively high (i.e. you are overfitting). For the following methods, indicate true if the method can mitigate your overfitting problem and false otherwise. Briefly explain why.

#### i. [3 points] 
Add additional features to your feature vector.

#### ii. [3 points] 
Impose a prior distribution on $\theta$, where the distribution of $\theta$ is of the form $N(0, \tau^2 I)$, and we derive $\theta$ via maximum a posteriori estimation.

### (b) [3 points] 

Choosing the parameter C is often a challenge when using SVMs. Suppose we choose $C$ as follows: First, train a model for a wide range of values of $C$. Then, evaluate each model on the test set. Choose the $C$ whose model has the best performance on the test set. Is the performance of the chosen model on the test set a good estimate of the model's generalization error?

### (c) [11 points] 

For the following, provide the VC-dimension of the described hypothesis classes and briefly explain your answer.

#### i. [3 points] 
Assume $\mathcal{X} = \mathbb{R}^2$. $\mathcal{H}$ is a hypothesis class containing a single hypothesis $h_1$ (i.e. $\mathcal{H} = \{h_1\}$)

#### ii. [4 points] 
Assume $\mathcal{X} = \mathbb{R}^2$. Consider $\mathcal{A}$ to be the set of all convex polygons in $\mathcal{X}$. $\mathcal{H}$ is the class of all hypotheses $h_P(x)$ (for $P \in \mathcal{A}$) such that

$$h_P(x) = \begin{cases} 
1 & \text{if } x \text{ is contained within polygon } P \\
0 & \text{otherwise}
\end{cases}$$

**Hint:** Points on the edges or vertices of $P$ are included in $P$.

#### iii. [4 points] 
$\mathcal{H}$ is the class of hypotheses $h_{(a,b)}(x)$ such that each hypothesis is represented by a single open interval in $\mathcal{X} = \mathbb{R}$ as follows:

$$h_{(a,b)}(x) = \begin{cases} 
1 & \text{if } a < x < b \\
0 & \text{otherwise}
\end{cases}$$

### (d) [3 points] 

Consider a sine function $f(x) = \sin(x)$ such that $x \in [-\pi, \pi]$. We use two different hypothesis classes such that $\mathcal{H}_0$ contains all constant hypotheses of the form, $h(x) = b$ and $\mathcal{H}_1$ contains all linear hypotheses of the form $h(x) = ax + b$. Consider taking a very large number of training sets, $S_i, i = 1, \dots, N$ such that each $S_i$ contains only two points $\{(x_1, y_1), (x_2, y_2)\}$ sampled iid from $f(x)$. In other words, each $(x, y)$ pair is drawn from a distribution such that $y = f(x) = \sin(x)$ is satisfied. We train a model from each hypothesis class using each training set such that we have a collection of $N$ models from each class. We then compute a mean-squared error between each model and the function $f(x)$.

It turns out that the average expected error of all models from $\mathcal{H}_0$ is significantly lower than the average expected error of models from $\mathcal{H}_1$ even though $\mathcal{H}_1$ is a more complex hypothesis class. Using the concepts of bias and variance, provide an explanation for why this is the case.

### (e) [3 points] 

In class when we discussed the decision boundary for logistic regression $h_\theta(x) = g(\theta^T x)$, we did not require an explicit intercept term because we could define $x_0 = 1$ and let $\theta_0$ be the intercept. When discussing SVMs, we dropped this convention and had $h_{w,b}(x) = g(w^T x+b)$ with $b$ as an explicit intercept term. Consider an SVM where we now write $h_w(x) = g(w^T x)$ and define $x_0 = 1$ such that $w_0$ is the intercept. If the primal optimization objective remains $\frac{1}{2}||w||^2$, can we change the intercept in this way without changing the decision boundary found by the SVM? Justify your answer.

## 2. [10 + 3 Extra Credit points] More Linear Regression

In our homework, we saw a variant of linear regression called locally-weighted linear regression. In the problem below, we consider a regularized form of locally-weighted linear regression where we favor smaller parameter vectors by adding a complexity penalty term to the cost function. Additionally, we consider the case where we are trying to predict multiple outputs for each training example. Our dataset is:

$$S = \{(x^{(i)},y^{(i)}), i = 1, ..., m\}, x^{(i)} \in \mathbb{R}^n, y^{(i)} \in \mathbb{R}^p$$

Thus for each training example, $y^{(i)}$ is a real-valued vector with $p$ entries. We wish to use a linear model to predict the outputs by specifying the parameter matrix $\theta$, where $\theta \in \mathbb{R}^{n \times p}$. You can assume $x^{(i)}$ contains the intercept term (i.e. $x_0 = 1$). The cost function for this model is:

$$J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{p} w^{(i)} \left( (\theta^T x^{(i)})_j - y_j^{(i)} \right)^2 + \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{p} (\theta_{ij})^2 \quad (1)$$

As before, $w^{(i)}$ is the "weight" for a specific training example $i$.

### (a) [2 points] 

Show that $J(\theta)$ can be written as

$$J(\theta) = \frac{1}{2} tr \left( (X\theta - Y)^T W (X\theta - Y) \right) + \frac{1}{2} tr(\theta^T \theta)$$

### (b) [5 points] 

Derive a closed form expression for the minimizer $\theta^*$ that minimizes $J(\theta)$ from part (a).

### (c) [3 points] 

Given the dataset $S$ above, which of the following cost functions will lead to higher accuracy on the training set? Briefly explain why this is the case. If there is insufficient information, explain what details are needed to make a decision.

#### i. 
$J_1(\theta) = \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{p} ((\theta^T x^{(i)})_j - y_j^{(i)})^2$

#### ii. 
$J_2(\theta) = \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{p} ((\theta^T x^{(i)})_j - y_j^{(i)})^2 + \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{p} (\theta_{ij})^2$

#### iii. 
$J_3(\theta) = \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{p} ((\theta^T x^{(i)})_j - y_j^{(i)})^2 + 100 \sum_{i=1}^{n} \sum_{j=1}^{p} (\theta_{ij})^2$

### (d) [3 Extra Credit points] 

Suppose we want to weight the regularization penalty on a per element basis. For this problem, we use the following cost function:

$$J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{p} w^{(i)} \left( \left(\theta^T x^{(i)}\right)_j - y_j^{(i)} \right)^2 + \frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{p} \left((\Gamma\theta)_{ij}\right)^2 \quad (2)$$

Here, $\Gamma \in \mathbb{R}^{n \times n}$ where $\Gamma_{ij} > 0$ for all $i, j$. Derive a closed form solution for $J(\theta)$ and $\theta^*$ using this new cost function.

## 3. [17 points] Generalized Linear Models

In class we showed that the Gaussian distribution is in the Exponential Family. However, a simplification we made to make the derivation easier was to set the variance term $\sigma^2 = 1$. This problem will investigate a more general form for the Exponential Family. First, recall that the Gaussian distribution can be written as follows:

$$p(y|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma}}\exp\left\{-\frac{1}{2\sigma^2}(y-\mu)^2\right\} \quad (3)$$

### (a) [6 points] 

Show that the Gaussian distribution (without assuming unit variance) is an exponential family distribution. In particular, please specify $b(y)$, $\eta$, $T(y)$, $a(\eta)$. Recall that the standard form for the exponential family is given by

$$p(y; \eta) = b(y)\exp\{\eta^T T(y) - a(\eta)\} \quad (4)$$

**Hint:** since $\sigma^2$ is now a variable, $\eta$ and $T(y)$ will now be two dimensional vectors; for consistent notation denote $\eta = [\eta_1 \quad \eta_2]^T$. For full credit, please ensure $a(\eta)$ is expressed in terms of $\eta_1$ and $\eta_2$.

### (b) [4 points] 

Assume an Independent and Identically Distributed (IID) training set given by $\{(x^{(i)}, y^{(i)}), i = 1,..., m\}$. Starting with an expression for $p(y; \eta)$, derive the general expression for the Hessian of the log-likelihood. The log-likelihood function is explicitly given as $\ell(\theta) = \sum_{i=1}^{m} \log p(y^{(i)}|x^{(i)}; \theta)$. The final answer for the Hessian should be expressed in terms of $x$, $\eta_1$, and $\eta_2$.

### (c) [5 points] 

Using your result from the part (b), show that the Hessian is negative semi-definite, i.e., $z^T H z \leq 0$.

### (d) [2 points] 

It turns out there is a more general definition for the exponential family given by

$$p(y; \eta, \tau) = b(a, \tau)\exp\left\{\frac{\eta^\top T(y) - a(\eta)}{c(\tau)}\right\}$$

In particular $c(\tau)$ is the dispersion function, where $\tau$ is called the *dispersion parameter*. Show that the Gaussian distribution can be written in this more general form with $c(\tau) = \sigma^2$.

## 4. [17 points] Naive Bayes and Logistic Regression

For this entire problem assume that the input features $x_j, j = 1, ..., n$ are discrete binary-valued variables such that $x_j \in \{0,1\}$ and $x = [x_1 x_2 ... x_n]$. For each training example $x^{(i)}$, assume that the output target variable $y^{(i)} \in \{0,1\}$.

### (a) [2 points] 

Consider the Naive Bayes model, given the above context. This model can be parameterized by $\phi_{j|y=0} = p(x_j = 1|y = 0)$, $\phi_{j|y=1} = p(x_j = 1|y = 1)$ and $\phi_y = p(y = 1)$. Write down the expression for $p(y = 1|x)$ in terms of $\phi_{j|y=0}, \phi_{j|y=1}$, and $\phi_y$.

### (b) [7 points] 

Show that the conditional likelihood expression you obtained in part (a) can be simplified to the same form as the hypothesis for logistic regression:

$$p(y = 1|x) = \frac{1}{1 + e^{-\theta^T x}} \quad (5)$$

**Hint:** Modify the definition of x to include the intercept term $x_0 = 1$.

### (c) [6 points] 

#### i. [2 points] 

Two Naive Bayes classifiers are trained independently on $S_1$ and $S_2$.

**Dataset $S_1$:**
* Consists of $m$ training examples of the form $\{(x^{(i)}, y^{(i)}), i = 1, \dots, m\}$.
* Each $x^{(i)} \in \mathbb{R}^{n+1}$.
* Satisfies the Naive Bayes assumption: $p(x_1, \dots, x_n|y) = \prod_{j=1}^{n} p(x_j|y)$.

**Dataset $S_2$:**
* Also consists of $m$ training examples of the form $\{(x^{(i)}, y^{(i)}), i = 1, \dots, m\}$.
* Each $x^{(i)} \in \mathbb{R}^{n+2}$.
* Contains the same $n$ conditionally-independent features as $S_1$, plus an additional feature $x_{n+1}$ such that $x_{n+1} = x_n$.
* Each $x^{(i)}$ contains an intercept term $x_0 = 1$.

Test data is generated according to the true distribution: $p(x_1, \dots, x_n, y) = p(x_1, \dots, x_n, x_{n+1}, y) = p(y)p(x_1, \dots, x_n|y)$, where $x_{n+1} = x_n$. Will the test error of the classifier trained on $S_1$ be larger or smaller than that trained on $S_2$? Assume that $m$ (the number of training examples) is very large. Briefly justify your answer.

#### ii. [4 points] 

Now we will look at a similar situation regarding how logistic regression is affected by copies of features. In order to simplify the math, let's assume a more basic case where $S_1$ still has $m$ training examples, but now has one feature $x_1$. $S_2$ has $m$ training examples but has two features $x_1$ and $x_2$ where $x_2 = x_1$. The logistic regression model trained on $S_1$ therefore has associated parameters $\{\theta_0, \theta_1\}$ and the model trained on $S_2$ has parameters $\{\theta_0, \theta_1, \theta_2\}$. Here, $\theta_0$ is associated with the intercept term $x_0 = 1$. Testing data is generated the same way (from the original true distribution). How will the error of the classifier trained on $S_1$ compare to that of the classifier trained on $S_2$? For this question you need to prove your result mathematically. (Hint: compare the forms of the log-likelihood for each classifier)

### (d) [2 points] 

In general, if we assume that the number of training examples $m$ is very large, which classifier will have a lower generalization error? Briefly justify why.
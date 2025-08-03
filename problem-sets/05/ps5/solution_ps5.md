# Problem Set 5 Solutions

## Problem 1: Short Answers

### 1. [24 points] Short answers

Questions require reasonably short answer (usually at most 2-3 sentences or a figure for each question part, though some may require longer or shorter explanations). To discourage random guessing, one point will be deducted for a wrong answer on true/false or multiple choice questions. No credit will be given for answers without a correct explanation.

#### (a) [5 points]

Given a cost function $J(\theta)$ that we seek to minimize and $\alpha \in \mathbb{R} > 0$, consider the following update rule:

$$\theta^{(t+1)} = \arg\min_{\theta} \left\{ J(\theta^{(t)}) + \nabla_{\theta^{(t)}} J(\theta^{(t)})^T (\theta - \theta^{(t)}) + \frac{1}{2\alpha} \|\theta - \theta^{(t)}\|^2 \right\}$$

##### (i) [3 points]

Show that this yields the same $\theta^{(t+1)}$ as the gradient descent update with step size $\alpha$.

**Answer:**

Denote $U(\theta) = J(\theta^{(t)}) + \nabla_{\theta^{(t)}} J(\theta^{(t)})^T (\theta - \theta^{(t)}) + \frac{1}{2\alpha} \|\theta - \theta^{(t)}\|^2$.

To find the minimum over $\theta$, we compute the gradient of $U(\theta)$ w.r.t. $\theta$ and set it to 0:

$$\nabla_{\theta} U(\theta) = 0$$
$$\nabla_{\theta^{(t)}} J(\theta^{(t)}) + \frac{1}{2\alpha} (-2\theta^{(t)} + 2\theta) = 0$$
$$\alpha\nabla_{\theta^{(t)}} J(\theta^{(t)}) - \theta^{(t)} + \theta = 0$$
$$\Rightarrow \theta = \theta^{(t)} - \alpha\nabla_{\theta^{(t)}} J(\theta^{(t)})$$

which is the gradient descent update, as desired.

To confirm this is a minimum, we compute the Hessian $\nabla_{\theta}^2 U = \frac{1}{\alpha}I$ which is positive definite as expected.

##### (ii) [2 points]

Provide a sketch (i.e. draw a picture) of the above update for the simplified case where $\theta \in \mathbb{R}$, $J(\theta) = \theta$, and $\theta^{(t)} = 1$. Make sure to clearly label $\theta^{(t)}$, $\theta^{(t+1)}$ and $\alpha$.

**Answer:**

We provide an example sketch for $\alpha = 1$. Note that $\alpha = \theta^{(t)} - \theta^{(t+1)}$ since $\nabla J(\theta) = 1$.

## Problem 2: Supervised Learning Methods

### A supervised learning method that would likely not work:

**Answer:** Naive Bayes - assumption of independence between features is not helpful when the data is strongly correlated.

*[Scatter plot showing two classes of data points: blue circles and red 'x' marks distributed across the plot with clusters in bottom-left and top-right, and red 'x' marks concentrated in central region. The two classes are intermingled, indicating they are not linearly separable.]*

### A supervised learning method that would likely work:

**Answer:** Kernel perceptron - Can separate around the curve of the graph using the kernel trick.

*[Same scatter plot as above but with a black curved decision boundary that effectively separates most red 'x' marks (within the curve) from blue circles (outside the curve).]*

### A supervised learning method that would likely not work:

**Answer:** Linear classification - Data is not linearly separable.

## Problem 3: Gaussian Discriminant Analysis

### A supervised learning method that would likely work:

**Answer:** Gaussian Discriminant Analysis - Has room for error, identifies main two clusters with overlap.

*[Scatter plot with x-axis -10 to 10, y-axis 0 to 5, showing blue circles clustered in region x=0 to 4, y=2 to 4, and red 'x's clustered where x=-2 to 1, y=2 to 3.5, with overlap in central region.]*

### A supervised learning method that would likely not work:

**Answer:** We accepted essentially anything reasonable here. A method that attempts to classify every example perfectly might fail, for example, because the data clusters overlap.

*[Same scatter plot as above but with two large overlapping black ellipses representing Gaussian distributions and a curved black decision boundary line separating the regions defined by the ellipses.]*

## Problem 4: Loss Functions

### Figure 1: An easy to separate dataset

*[Scatter plot titled "Figure 1: An easy to separate dataset" with x-axis -10 to 20, y-axis 0 to 5, showing 8 red 'x' data points clustered in region x=5 to 9, y=2 to 3.5, and 10 blue circular data points clustered in region x=11 to 15, y=2 to 3.5. A solid blue vertical line at x=10 perfectly separates the classes.]*

and negative are o's) using the loss functions

$$L_1(\theta^T x, y) = \frac{1}{2}(\theta^T x - y)^2$$

$$L_2(\theta^T x, y) = [1 - y\theta^T x]_+ = \max\{0, 1 - y\theta^T x\}$$

For the given dataset, we plot the line $\{x \in \mathbb{R}^2 : x^T\theta^* = 0\}$, where $\theta^*$ is the minimizer of the average losses (empirical risks) $J_1(\theta) = \frac{1}{m}\sum_{i=1}^m L_1(\theta^T x, y)$ and $J_2(\theta) = \frac{1}{m}\sum_{i=1}^m L_2(\theta^T x, y)$. (For the given dataset, the same $\theta$ is optimal for each.)

#### i. [1 points]

What are the names of the loss functions?

**Answer:** $L_1$ is the least-squares loss or squared error, $L_2$ is the hinge loss (or SVM loss).

#### (b) [3 points]

*[Scatter plot titled "Figure 2: A new point" with x-axis -10 to 20, y-axis 0 to 5, showing a cluster of approximately six red 'x' marks in region x=5 to 8, y=2 to 3.5, and a cluster of approximately eight blue 'o' marks in region x=11.5 to 14.5, y=2 to 3.5. There is also a single isolated red 'x' mark at coordinates (-5, 1).]*

We add new data point with positive label at the point $(-5, 1)$, as in Fig. 2.

**ii. [3 points]** Could the classifying line change for either of the loss functions? Briefly explain why. Draw (your best estimate of) the new classification boundaries, and clearly label the lines with the corresponding loss functions.

**Answer:** For the hinge loss, nothing changes—the point is classified perfectly and suffers no loss. For the squared error, the loss changes substantially, because we try to assign it a prediction close to 1.

## Problem 4: VC-dimension, Empirical Risk, and Generalization

### 4. [11 points] VC-dimension, Empirical Risk, and Generalization

#### (c) [2 points]

Suppose we have two collections of hypotheses, $H_1$ and $H_2$, and we fit them on a training set to give $\hat{h}_1$ and $\hat{h}_2$ solving

$$\hat{h}_1 = \arg\min_{h \in H_1} \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\{h(x^{(i)}) \neq y^{(i)}\} \quad \text{and} \quad \hat{h}_2 = \arg\min_{h \in H_2} \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\{h(x^{(i)}) \neq y^{(i)}\}.$$

We have $VC(H_1) < VC(H_2)$. Which of $\hat{h}_1$ and $\hat{h}_2$ will have lower training error?

**Answer:** It depends. Neither will necessarily be lower, because the hypothesis classes could be completely unrelated.

#### (d) [3 points]

Give an example of a class of hypotheses $H$ and a distribution on $(x, y)$, where $x \in \mathbb{R}$ and $y \in \{-1, 1\}$, such that there always exists $h \in H$ with

$$\frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\{h(x^{(i)}) \neq y^{(i)}\} < .01 \quad \text{and} \quad P(h(X) \neq Y) > .99$$

no matter the training set size $m$.

**Answer:** Here is one example; many others are possible. If we take $H$ to be the class of all functions $h : \mathbb{R} \to \{-1,1\}$, and then let $P$ be a distribution with $x \sim N(0, 1)$ and $Y = \text{sign}(x)$. We can always find a function $h \in H$ perfectly classifying the training data but for all $x \notin \{x^{(1)}, \dots, x^{(m)}\}$ predicting $h(x) = -\text{sign}(x)$. Thus $P(h(X) \neq Y) = 1$ but the empirical error is always 0.

#### (e) [3 points]

You are given the choice of two loss functions for a binary classification problem: the exponential and logistic losses,

$$L(\theta^T x, y) = \exp(-y\theta^T x) \quad \text{or} \quad L(\theta^T x, y) = \log(1 + \exp(-y\theta^T x)).$$

The label $y^{(i)}$ is incorrect for about 10% (the precise number is unimportant) of the training data $\{(x^{(i)},y^{(i)})\}_{i=1}^m$. You will choose a hypothesis by minimizing

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} L(\theta^T x^{(i)}, y^{(i)})$$

for one of the two losses. Which loss is more likely to have better generalization performance? Justify your answer.

**Answer:** The logistic loss. It is less sensitive to mistaken labels, as it grows only linearly for mis-classifications rather than exponentially. So the exponential loss will work very hard to classify mis-labeled examples correctly.

#### (f) [3 points]

Instead of minimizing the average loss on a training set, John decides to minimize the maximal loss on your training set for a classification problem with $y \in \{-1,1\}$. He has training data $x^{(i)} \in \mathbb{R}$, and he will learn a linear classifier $\theta x^{(i)}$ by finding $\theta \in \mathbb{R}$. He decides to minimize

$$J_{\max}(\theta) = \max_{i \in \{1,...,m\}} \log(1 + \exp(-y^{(i)}x^{(i)}\theta))$$

Examples 1 and 2 in his dataset satisfy $x^{(1)} < 0$ and $x^{(2)} < 0$, but $y^{(1)} \neq y^{(2)}$.

Is his idea to minimize the maximal loss a good one? Why or why not? [Hint: The answer is not 'It depends.']

**Answer:** No, it's a bad idea. The minimizing $\theta$ will be $\theta = 0$, as this is the only way to guarantee that

$$\max_{i} \log(1 + \exp(-y^{(i)} x^{(i)}\theta)) \le \log 2,$$

which is attained at $\theta = 0$.

## Problem 2: Exponential Families and Generative Models

### 2. [7 points] Exponential families and generative models

We have a problem with $k$ categories, $y \in \{1,..., k\}$, and we make the generative assumption that $x$ conditional on $y$ follows the exponential family distribution

$$p(x | y; \eta) = b(x) \exp (\eta_y^T T(x) – A(\eta; y))$$

where $\eta_y \in \mathbb{R}^n$ for $y = 1,..., k$. Also assume that we have prior probabilities

$$p(y) = \pi_y > 0 \text{ for } y = 1,..., k.$$

Show that the distribution of $y$ conditional on $x$ follows the multinomial logistic model. That is, show that there are $\theta_y \in \mathbb{R}^n$ (for $y = 1,..., k$) and $\theta^{(0)} \in \mathbb{R}^k$ such that

$$p(y | x) = \frac{\exp (\theta_y^T T(x) + \theta_y^{(0)})}{\sum_{l=1}^k \exp (\theta_l^T T(x) + \theta_l^{(0)})}$$

Describe explicitly what the values of $\theta$ and $\theta^{(0)}$ are as a function of $\eta$, $\pi$, and A.

**Answer:** We use Bayes' rule, which gives

\begin{align*}
p(y | x) &= \frac{p(x | y)p(y)}{\sum_{l=1}^k p(x | l)p(l)} \\
&= \frac{b(x) \exp(\eta_y^T T(x) – A(\eta; y))\pi_y}{\sum_{l=1}^k b(x) \exp(\eta_l^T T(x) – A(\eta; l))\pi_l} \\
&= \frac{\exp(\eta_y^T T(x) – A(\eta; y) + \log \pi_y)}{\sum_{l=1}^k \exp(\eta_l^T T(x) – A(\eta; l) + \log \pi_l)}
\end{align*}

Now, let $\theta_l = \eta_l$ and $\theta_l^{(0)} = -A(\eta; l) + \log \pi_l$, which gives

$$p(y | x) = \frac{\exp (\theta_y^T T(x) + \theta_y^{(0)})}{\sum_{l=1}^k \exp (\theta_l^T T(x) + \theta_l^{(0)})}$$

as desired.

## Problem 3: Local Polynomial Regression

### 3. [15 points] Local Polynomial Regression

We have a training set:
$S = \{(x^{(i)},y^{(i)}), i = 1,..., m\}$ where $x^{(i)} \in \mathbb{R}^n, y^{(i)} \in \mathbb{R}$.

Assume $x^{(i)}$ contains the intercept term (i.e. $x_0^{(i)} = 1$ for all $i$). Consider the following regression model:
$y = \theta^{(1)T} x + \theta^{(2)T} x^2 + \dots + \theta^{(p-1)T} x^{p-1} + \theta^{(p)T} x^p$

where $\theta^{(p)}$ denotes the $p^{th}$ parameter vector and where $x^p$ denotes element-wise exponentiation (i.e. each element of $x$ is raised to the $p^{th}$ power). The cost function for this model is:
$$J(\theta^{(1)}, \dots, \theta^{(p)}) = \frac{1}{2} \sum_{i=1}^{m} w^{(i)} \left(\sum_{k=1}^{p} \theta^{(k)T} x^{(i)k} - y^{(i)}\right)^2$$

As before, $w^{(i)}$ is the "weight" for a specific training example $i$.

#### (a) [3 points]

Show that $J(\theta^{(1)}, \dots, \theta^{(p)})$ can be written as:
$$J(\theta) = \frac{1}{2} \text{tr}[(X\theta - y)^T W(X\theta - y)]$$

Using $\theta^{(1)}, \dots, \theta^{(p)}$, you need to define a vector $\theta$ and matrices $X$ and $W$ such that the transformation is possible. Clearly state the dimensions of these variables.

**Answer:** Define $\theta$ to be the concatenation of each $\theta^{(1)}, \dots, \theta^{(p)}$, that is:
$\theta = (\theta_1^{(1)}, \dots, \theta_n^{(1)}, \theta_1^{(2)}, \dots, \theta_n^{(2)}, \dots, \theta_1^{(p-1)}, \dots, \theta_n^{(p-1)}, \theta_1^{(p)}, \dots, \theta_n^{(p)})$

Similarly, we redefine $x^{(i)}$ such that:
$x^{(i)} = (x_1^{(i)1}, \dots, x_n^{(i)1}, x_1^{(i)2}, \dots, x_n^{(i)2}, \dots, x_1^{(i)p-1}, \dots, x_n^{(i)p-1}, x_1^{(i)p}, \dots, x_n^{(i)p})$

We define $W$ to be a diagonal matrix with $W_{ii} = w^{(i)}$, similar to the problem set.
The dimensions are:
* $\theta$ is a vector of size $pn$
* $X$ is the design matrix of size $m \times pn$
* $W$ is a matrix of size $m \times m$

Some students included the bias term such that $x^{(i)} \in \mathbb{R}^{n+1}$. This resulted in the dimensions:
$\theta \in \mathbb{R}^{pn+p}$
$X \in \mathbb{R}^{m \times (pn+p)}$
$W \in \mathbb{R}^{m \times m}$

We accepted both sets of answers as correct.

#### (b) [2 points]

Let $\theta \in \mathbb{R}^N$. Define $\Gamma \in \mathbb{R}^{N_0 \times N}$ to be any matrix. Suppose we add a term $P(\theta)$ to our cost function:
$$P(\theta) = \frac{1}{2} \sum_{i=1}^{N_0} (\Gamma\theta)_i^2$$
Show that $P(\theta)$ can be written as
$$P(\theta) = \frac{1}{2} \text{tr}((\Gamma\theta)^T (\Gamma\theta)) = \frac{1}{2} \|\Gamma\theta\|_2^2$$

**Answer:** The definition of $\|u\|_2^2 = \sum_{i=1}^n u_i^2$. So $\sum_{i=1}^{N_0} (\Gamma\theta)_i^2 = \|\Gamma\theta\|_2^2$, which is $(\Gamma\theta)^T (\Gamma\theta) \in \mathbb{R}_+$. And $\text{tr}(a) = a$ for any scalar.

#### (c) [4 points]

Our final cost function is:
$$J(\theta) = \frac{1}{2} \text{tr}[(X\theta - y)^T W(X\theta - y)] + \frac{1}{2} \text{tr}((\Gamma\theta)^T (\Gamma\theta)) \quad (1)$$
Derive a closed form expression for the minimizer $\theta^*$ that minimizes $J(\theta)$ as shown in Equation (1).

**Answer:** We compute the gradient of the first and second term separately. We start with the first term, $J_1(\theta) = \frac{1}{2} \text{tr} ((X\theta - y)^T W(X\theta - y))$.

$$\nabla_{\theta} J_1(\theta) = \nabla_{\theta} \frac{1}{2} \text{tr} ((X\theta - y)^T W(X\theta - y))$$
$$= \nabla_{\theta} \frac{1}{2} \text{tr} (\theta^T X^T WX\theta - \theta^T X^T Wy - y^T WX\theta - y^T Wy)$$
$$= \frac{1}{2} \nabla_{\theta} [\text{tr}(\theta^T X^T WX\theta) - \text{tr}(\theta^T X^T Wy) - \text{tr}(y^T WX\theta) - \text{tr}(y^T Wy)]$$
$$= \frac{1}{2} \nabla_{\theta} [\text{tr}(\theta^T X^T WX\theta) - 2\text{tr}(y^T WX\theta) - \text{tr}(y^T Wy)]$$
$$= \frac{1}{2} (X^T WX\theta - 2X^T Wy + X^T WX\theta)$$
$$= X^T WX\theta - X^T Wy$$

Now we find the gradient of the second term, $J_2(\theta) = \frac{1}{2} \text{tr}((\Gamma\theta)^T (\Gamma\theta))$:

$$\nabla_{\theta} J_2(\theta) = \nabla_{\theta} \frac{1}{2} \text{tr}((\Gamma\theta)^T (\Gamma\theta))$$
$$= \nabla_{\theta} \frac{1}{2} \text{tr}(\theta^T \Gamma^T \Gamma\theta)$$
$$= \frac{1}{2} (2\Gamma^T \Gamma\theta)$$
$$= \Gamma^T \Gamma\theta$$

Combining the gradient of both terms gives us the final gradient:

$$\nabla_\theta J(\theta) = \nabla_\theta J_1(\theta) + \nabla_\theta J_2(\theta) = X^T W X \theta - X^T W y + \Gamma^T \Gamma \theta$$

We can then set $\nabla_\theta J(\theta)$ equal to zero and find the optimal $\theta^*$:

$$0 = X^T W X \theta - X^T W y + \Gamma^T \Gamma \theta$$
$$X^T W y = X^T W X \theta + \Gamma^T \Gamma \theta$$
$$X^T W y = (X^T W X + \Gamma^T \Gamma) \theta$$
$$\theta^* = (X^T W X + \Gamma^T \Gamma)^{-1} X^T W y$$

Aside: Let $I$ be the $n \times n$ identity matrix. If $\Gamma = \alpha I$ for some $\alpha > 0$, the above technique is known as \textit{ridge regression} or $l_2$ regularization.

#### (d) [2 points]

If we want to maximize the training accuracy, what is the optimal value of $\Gamma$ (if any)? In 1-2 sentences, justify your answer.

**Answer:** Choose $\Gamma = 0$, as this means we choose $\theta$ exclusively to minimize the training error.

#### (e) [2 points]

If we want to maximize the test accuracy, what is the optimal value of $\Gamma$ (if any)? In 1-2 sentences, justify your answer.

**Answer:** It depends. There is no particular value that is guaranteed to minimize test accuracy.

#### (f) [2 points]

So far, we used a regression model containing polynomial representations of the input. Our polynomial model contains $\theta^{(1)T} x$ as a term which is the same as our "standard" linear model of $y = \theta^T x$. However, our polynomial model can express higher-order relationships while our standard model cannot. In 2-4 sentences, explain when and why we should \textit{not} use the polynomial model.

**Answer:** Consider a linearly separable dataset (i.e. the original, non-polynomial $x^{(i)}$'s are good enough to predict $y^{(i)}$). Including polynomial terms is unnecessary, since we know the dataset can be modeled without them. If $p$ is large, this will increase the VC dimension which can lead to overfitting. In this case, we should not use the polynomial model.

## Problem 4: Online Gradient Descent

### 4. [14 points] Online (not stochastic) gradient descent

In this question, we explore a variant of stochastic gradient descent known as online gradient descent. A cost function $c: \mathbb{R}^n \to \mathbb{R}$ is convex if

$$c(\lambda\theta + (1 - \lambda)\bar{\theta}) \le \lambda c(\theta) + (1 - \lambda)c(\bar{\theta})$$

for all $\theta, \bar{\theta} \in \mathbb{R}^n$. A differentiable convex function $c$ satisfies

$$c(\bar{\theta}) \ge c(\theta) + \nabla c(\theta)^T (\bar{\theta} - \theta) \quad \text{for all } \bar{\theta} \in \mathbb{R}^n.$$

*[Graph titled "Figure 3: A convex function and its linear approximation at the point $\theta$" showing a black solid curve representing a convex function labeled $c(w)$ at a point on the curve, a dashed black line tangent to the curve at the point labeled $c(w)$, another point on the curve labeled $c(v)$ to the left of $c(w)$, a vertical dotted line extending downwards from $c(v)$ to the dashed tangent line, and the point on the dashed tangent line directly below $c(v)$ labeled $c(w) + \nabla c(w)(v - w)$.]*

**Figure 3:** A convex function and its linear approximation at the point $\theta$.

In online convex optimization, the learner receives (sequentially) a sequence of convex functions $c_1, c_2, c_3, \dots$, and at iteration $t$ makes the online gradient update

$$\theta^{(t+1)} = \theta^{(t)} - \alpha g^{(t)} \quad \text{where } g^{(t)} = \nabla c_t(\theta^{(t)}).$$

Here $\alpha > 0$ is a scalar stepsize, and we assume that all cost functions $c_t$ are differentiable. The goal is to not suffer too much cumulative loss $\sum_{t=1}^T c_t(\theta^{(t)})$.

#### (a) [4 points]

Prove that with the update (2), for any $\theta \in \mathbb{R}^n$,

$$\frac{1}{2} \|\theta^{(t+1)} - \theta\|_2^2 \leq \frac{1}{2} \|\theta^{(t)} - \theta\|_2^2 - \alpha(c_t(\theta^{(t)}) - c_t(\theta)) + \frac{\alpha^2}{2} \|g^{(t)}\|_2^2.$$

**Answer:** We expand $\theta^{(t+1)}$ in terms of $\theta^{(t)}$, getting

$$\frac{1}{2} \|\theta^{(t+1)} - \theta\|_2^2 = \frac{1}{2} \|\theta^{(t)} - \alpha g^{(t)} - \theta\|_2^2$$

$$= \frac{1}{2} \|\theta^{(t)} - \theta\|_2^2 - \alpha g^{(t)T} (\theta^{(t)} - \theta) + \frac{\alpha^2}{2} \|g^{(t)}\|_2^2.$$

Then use convexity to see that

$$-g^{(t)T} (\theta^{(t)} - \theta) = -\nabla c_t(\theta^{(t)})^T (\theta^{(t)} - \theta) = \nabla c_t(\theta^{(t)})^T (\theta - \theta^{(t)}) \leq c_t(\theta) - c_t(\theta^{(t)}).$$

We substitute to obtain

$$\frac{1}{2} \|\theta^{(t+1)} - \theta\|_2^2 \leq \frac{1}{2} \|\theta^{(t)} - \theta\|_2^2 + \alpha(c_t(\theta) - c_t(\theta^{(t)})) + \frac{\alpha^2}{2} \|g^{(t)}\|_2^2.$$

#### (b) [4 points]

After $T$ iterations of online gradient descent (2), the regret of the learner with respect to a fixed $\theta \in \mathbb{R}^n$ is

$$\text{Reg}_T(\theta) := \sum_{t=1}^T [c_t(\theta^{(t)}) - c_t(\theta)].$$

Using the result of part (a), show that

$$\text{Reg}_T(\theta) = \sum_{t=1}^T [c_t(\theta^{(t)}) - c_t(\theta)] \leq \frac{1}{2\alpha} \|\theta^{(1)} - \theta\|_2^2 + \frac{\alpha}{2} \sum_{t=1}^T \|g^{(t)}\|_2^2.$$

**Answer:** Rearrange the result of part (a) to find that

$$c_t(\theta^{(t)}) - c_t(\theta) \leq \frac{1}{2\alpha} \left[\|\theta^{(t)} - \theta\|_2^2 - \|\theta^{(t+1)} - \theta\|_2^2 + \alpha^2 \|g^{(t)}\|_2^2\right].$$

Sum this expression from $t=1$ to $t=T$, which gives

$$\sum_{t=1}^T [c_t(\theta^{(t)}) - c_t(\theta)] \leq \frac{1}{2\alpha} \left[\|\theta^{(1)} - \theta\|_2^2 - \|\theta^{(T+1)} - \theta\|_2^2\right] + \frac{\alpha}{2} \sum_{t=1}^T \|g^{(t)}\|_2^2$$

$$\leq \frac{1}{2\alpha} \|\theta^{(1)} - \theta\|_2^2 + \frac{\alpha}{2} \sum_{t=1}^T \|g^{(t)}\|_2^2.$$

#### (c) [4 points]

Suppose we guarantee that the functions $c_t$ have bounded gradients, that is, $\|g^{(t)}\|_2 \leq G$ for all $t$. Give a stepsize $\alpha$, which may depend on $\|\theta\|_2$ and $G$, such that if $\theta^{(1)} = 0$, we can guarantee

$$\text{Reg}_T(\theta) \leq G \|\theta\|_2 \sqrt{T}.$$

That is, the average regret $\frac{1}{T}\text{Reg}_T(\theta) = O(1/\sqrt{T})$ for any vector $\theta$.

**Answer:** Set $\alpha = \frac{\|w\|_2}{G\sqrt{T}}$ and get

$$\text{Reg}_T(w) \le \frac{\|w\|_2^2}{2\alpha} + \frac{\alpha TG^2}{2} = \frac{\|w\|_2 G}{2\sqrt{T}} + \frac{G\|w\|_2}{2\sqrt{T}} = G\|w\|_2\sqrt{T}.$$

#### (d) [2 points]

Show that if $y \in \{-1,1\}$ and $x$ satisfies $\|x\|_2 \le G$, then the gradient of the logistic loss (the logistic loss is $L(\theta^T x, y) = \log(1 + \exp(-y\theta^T x)))$ has $\ell_2$-norm bounded by $G$.

**Answer:** Taking derivatives of logistic regression, we have

$$\nabla_\theta \log(1 + \exp(-y\theta^T x)) = - \frac{1}{1 + e^{y\theta^T x}} (yx).$$

Taking the $\ell_2$-norm of this, we have

$$\|\nabla_\theta \log(1 + \exp(-y\theta^T x))\|_2 = \underbrace{\frac{1}{1 + e^{y\theta^T x}}}_{\le 1} \|x\|_2 \le \|x\|_2 \le G.$$

## Problem 5: Kernels via Randomization

### 5. [17 points] Kernels via randomization

You have seen how using kernels can allow efficient predictions by using the representer theorem, and the kernel trick allows us to automatically incorporate nonlinearities in supervised learning problems via the kernel function $K$. A difficulty with kernels is their time complexity: if we form the kernel (Gram) matrix $G$,$^1$ defined by

$$G_{ji} = G_{ij} = K(x^{(i)}, x^{(j)}), \quad G \in \mathbb{R}^{m \times m},$$

then storing $G$ requires space $O(m^2)$, inverting it requires time $O(m^3)$, and making new predictions on an unseen point $x$ requires time $m \cdot T$, where $T$ is the amount of time to compute $K(x, x^{(i)})$. One way around this is via randomization.

Suppose that the raw input attributes $x \in \mathcal{X}$, and let $W$ be some other space (you may assume that $W = \mathbb{R}$). Let $\phi: \mathcal{X} \times W \to \mathbb{R}$ be an arbitrary function, and let $P$ be a probability distribution on the space $W$. Define the function

$$K_P(x, z) := \mathbb{E}[\phi(x, W)\phi(z, W)] \quad \text{for } x, z \in \mathcal{X}, \quad (3)$$

where the subscript $P$ denotes that $W$ is sampled according to $P$ (i.e. the expectation is taken over $W \sim P$).

#### (a) [4 points]

Is the function $K_P$ a valid (Mercer) kernel? If so, prove this. If not, give a counterexample.

**Answer:** The function $K_P$ is indeed a Mercer kernel. Indeed, define the Gram matrix $G$ by $G_{ij} = K_P(x^{(i)}, x^{(j)})$. Then it is clear that $G_{ij} = G_{ji}$, and for any vector $v \in \mathbb{R}^m$, we have

\begin{align*}
v^T G v &= \sum_{i,j} v_i v_j G_{ij} \\
&= \sum_{i,j} v_i \mathbb{E}[\phi(x^{(i)}, W)\phi(x^{(j)}, W)]v_j \\
&= \mathbb{E}\left[\sum_{i,j} v_i \phi(x^{(i)}, W)\phi(x^{(j)}, W)v_j\right] \\
&= \mathbb{E}\left[\left(\sum_{i=1}^m \phi(x^{(i)}, W)v_i\right)^2\right] \ge 0.
\end{align*}

In particular, the matrix $G$ is positive semidefinite, so that $K_P$ is a valid kernel.

---
$^1$We use $G$ in this problem so as not to confuse it with $K$, the kernel function

#### (b) [4 points]

A natural idea is to approximate $K_P$ by random sampling. We take $N$ i.i.d. samples $W_l \sim P$, calling them $W_1, W_2, \dots, W_N$, and we define

$$\hat{K}(x, z) := \frac{1}{N} \sum_{l=1}^{N} \phi(x, W_l)\phi(z, W_l).$$

Suppose we know that $\phi(x, w) \in [-1,1]$ for all $x \in \mathcal{X}$ and all $w \in \mathcal{W}$. For a fixed pair $x, z \in \mathcal{X}$, give an upper bound on the probability that $\hat{K}$ is far from $K_P$, that is, give a bound decreasing to 0 exponentially in $N$ on

$$P(|\hat{K}(x, z) - K_P(x, z)| \ge \epsilon)$$

that is valid for all $\epsilon \ge 0$.

**Answer:** We use Hoeffding's inequality. In particular, if we define the random variable

$$Z_l = \phi(x, W_l)\phi(z, W_l)$$

then $Z_l \in [-1, 1]$, and $E[Z_l] = E[\phi(x, W)\phi(z, W)] = K_P(x, z)$. Hoeffding's inequality then implies that

$$P(\hat{K}(x, z) - K_P(x, z) \ge \epsilon) = P\left(\frac{1}{N} \sum_{l=1}^{N} (Z_l - E[Z_l]) \ge \epsilon\right)$$

$$\le \exp\left(-\frac{2N\epsilon^2}{(1+1)^2}\right) = \exp\left(-\frac{N\epsilon^2}{2}\right)$$

and similarly for the event that $\hat{K}(x, z) - K_P(x, z) \le -\epsilon$. Thus

$$P(|\hat{K}(x, z) - K_P(x, z)| \ge \epsilon) \le 2 \exp\left(-\frac{N\epsilon^2}{2}\right).$$

#### (c) [4 points]

Continue to assume that $\phi(x, w) \in [-1,1]$ for all $x, w$. Suppose we have a training set $\{x^{(i)}\}_{i=1}^m$ of size $m$. Give a sample size $N^*$ such that if we take $N \ge N^*$ samples of $W$ we are guaranteed that with probability at least $1 - \delta$, we have

$$|\hat{K}(x^{(i)}, x^{(j)}) - K_P(x^{(i)}, x^{(j)})| \le \epsilon$$

for all pairs $i, j \in \{1,...,m\}$. Written differently, if $\hat{G}_{ij} = \hat{K}(x^{(i)}, x^{(j)})$ and $G_{ij} = K_P(x^{(i)}, x^{(j)})$, guarantee that $\max_{i,j} |\hat{G}_{ij} - G_{ij}| \le \epsilon$.

**Answer:**

As the kernel functions $\hat{K}$ and $K_P$ are symmetric, we need concern ourselves only with indices $i \le j$, of which there are $\frac{m(m+1)}{2}$. By the union bound, we have

$$P \left( \max_{i,j} |\hat{G}_{ij} - G_{ij}| \ge \epsilon \right) \le \sum_{i \le j}^m P \left( |\hat{K}(x^{(i)}, x^{(j)}) - K_P(x^{(i)}, x^{(j)})| \ge \epsilon \right)$$

$$\le m(m + 1) \exp \left( -\frac{N\epsilon^2}{2} \right)$$

by part (b). Setting this value equal to $\delta$, we solve to obtain

$$\delta = m(m + 1) \exp \left( -\frac{N\epsilon^2}{2} \right) \quad \text{iff} \quad \frac{N\epsilon^2}{2} = \log \frac{m(m + 1)}{\delta},$$

so that it is sufficient that we have

$$N \ge \frac{2 \log \frac{m^2+m}{\delta}}{\epsilon^2}$$

samples of $W$ to guarantee good approximation for all pairs $x^{(i)}, x^{(j)}$.

#### (d) [5 points]

Assume that you have $N$ i.i.d. samples $W_1, \dots, W_N \stackrel{\text{iid}}{\sim} P$ and a training set $\{(x^{(i)}, y^{(i)})\}_{i=1}^m$ for a binary classification problem, with $y^{(i)} \in \{-1, 1\}$, and loss $L: \mathbb{R} \times \{-1, 1\} \to \mathbb{R}$. In the usual kernelized supervised learning setting, we would make predictions on a new datapoint $x$ using $\sum_{i=1}^m K_P(x, x^{(i)})\alpha_i$, and if
$G = [G^{(1)} \dots G^{(m)}] \in \mathbb{R}^{m \times m}$, $G^{(i)} \in \mathbb{R}^m$
is the Gram matrix, we would choose $\alpha$ by minimizing
$$J_\lambda(\alpha) = \frac{1}{m} \sum_{i=1}^m L(G^{(i)T}\alpha, y^{(i)}) + \frac{\lambda}{2}\alpha^T G \alpha. \quad (4)$$
Using your $N$ samples $W_1, \dots, W_N$, how can we reverse the kernel trick? That is, (i) write down a supervised learning problem with optimization variable $\theta \in \mathbb{R}^N$ that approximates problem (4), (ii) describe how, when given a new datapoint $x$, you can make a prediction on that datapoint, and (iii) give a bound on the runtime of making a prediction on a new datapoint $x$.

**Answer:**

(i) We define the vector

$$\hat{\phi}(x) := \begin{bmatrix} \phi(x, W_1) \\ \vdots \\ \phi(x, W_N) \end{bmatrix}.$$

Then we let $\theta \in \mathbb{R}^N$, and write the regularized risk

$$J_\lambda(\theta) = \frac{1}{m} \sum_{i=1}^m L(\hat{\phi}(x^{(i)})^T\theta, y^{(i)}) + \frac{\lambda}{2}\|\theta\|_2^2.$$

The representer theorem tells us that this is equivalent to minimizing

$$\hat{J}_\lambda(\alpha) = \frac{1}{m} \sum_{i=1}^m L((\hat{G}^{(i)})^T\alpha, y^{(i)}) + \frac{\lambda}{2}\alpha^T \hat{G} \alpha,$$

which approximates $J_\lambda(\alpha)$ because we have $\hat{G} \approx G$ by sampling.

(ii) We make predictions on a new point $x$ via

$$h_\theta(x) = \hat{\phi}(x)^T \theta = \sum_{l=1}^N \phi(x, W_l)\theta_l.$$

(iii) The runtime of the predictions is $O(NT_\phi)$, where $T_\phi$ is the time to compute $\phi(x, W)$.
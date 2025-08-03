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

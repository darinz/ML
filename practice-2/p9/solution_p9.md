# Problem Set 9 Solutions

## Problem 1: Irreducible Error

**1 point**

**Question:** For a given model, irreducible error can be decreased by improving the model's complexity and increasing the amount of training data.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

**Explanation:** 
You can't reduce irreducible error.

## Problem 2: Neural Network Overfitting

**1 point**

**Question:** You're training a classifier model using a neural network built from scratch in PyTorch. You are unable to decide on the depth of the neural network, so you decide to make the network as deep as possible. Despite achieving low training loss, your model performs poorly on the XOR test data.

Why? Choose the most accurate explanation.

**Options:**
- a) The neural network is too complex and has too high of a bias squared error.
- b) The neural network is too complex and has too high of a variance error.
- c) The neural network is unable to learn non-linearities since XOR data is not linearly separable.
- d) We need to make the neural network even deeper to capture the complex relationship in the XOR dataset.

**Correct Answer:** b) The neural network is too complex and has too high of a variance error.

**Explanation:** 
The deep neural network is too complex and fits the training data too well (overfitting) resulting in a low bias squared error but fails to generalize as a result of high variance error.

## Problem 3: Leave-One-Out Cross-Validation

**2 points**

**Question:** As dataset sizes increase, would you be more or less inclined to use leave-one-out cross-validation (LOOCV)? Provide reasoning to support your answer.

**Answer:** [Student response area]

**Explanation:** 
For larger datasets, leave-one-out cross validation becomes an extremely expensive process.

## Problem 4: K-Fold Cross-Validation Calculations

**1 point**

**Question:** You are fine-tuning a model with parameters $\alpha$, $\beta$, and $\gamma$, and have decided to perform 7-fold cross-validation to get the best set of hyperparameters. You have 5 candidate values for $\alpha$, 3 candidate values for $\beta$, and 2 candidate values for $\gamma$. How many validation errors will you be calculating in total?

**Options:**
- a) Cannot be determined.
- b) 10
- c) 96
- d) 210
- e) 30

**Correct Answer:** d) 210

**Explanation:** 
$5 \times 3 \times 2 \times 7 = 210$

## Problem 5: Maximum Likelihood Estimation - Exponential Distribution

**3 points**

**Question:** You are analyzing the time until failure for a set of lightbulbs. The data represents the number of months each bulb lasted before failing and is given as follows: $x_1, x_2, x_3, x_4$. Assuming these times are modeled as being drawn from an exponential distribution. Derive the maximum likelihood estimate (MLE) of the rate parameter $\lambda$ of this distribution. You must show your work.

**Recall:** The probability density function (PDF) for the exponential distribution is $f(x|\lambda) = \lambda e^{-\lambda x}$ for $x \ge 0$.

**Hint:** You should not have $n$ in your final answer.

**Answer:** $\lambda = \underline{\hspace{2cm}}$

**Explanation:** 
The answer is $\frac{4}{\sum_{i=1}^{4} x_i}$.

**Detailed Solution:**

First, we want to calculate the likelihood function $L(x_1,..., x_n|\lambda)$ below.

$L(x_1,..., x_n|\lambda) = P(x_1|\lambda) \cdot P(x_2|\lambda) \cdot ... \cdot P(x_n|\lambda) = \lambda e^{-\lambda x_1} \cdot \lambda e^{-\lambda x_2} \cdot ... \cdot \lambda e^{-\lambda x_n}$

$= \lambda^n \cdot e^{-\lambda(x_1+x_2+...+x_n)}$

Now, we calculate the log-likelihood function:

$lnL(x_1,..., x_n|\lambda) = ln(\lambda^n \cdot e^{-\lambda(x_1+x_2+...+x_n)}) = n \cdot ln(\lambda) - \lambda(x_1 + x_2 + ... + x_n)$

To find the argmax of $\lambda$ (and thus the MLE) of this log-likelihood expression, we need to take it's derivative with respect to $\lambda$ and set it equal to 0.

$\frac{d}{d\lambda}lnL(x_1,..., x_n|\lambda) = \frac{d}{d\lambda}(n \cdot ln(\lambda) - \lambda(x_1 + x_2 + ... + x_n)) = \frac{n}{\lambda} - (x_1+x_2+ ... + x_n) = 0$

$\implies \lambda = \frac{n}{x_1+x_2+...+x_n} = \frac{n}{\sum_{i=1}^{n} x_i}$

Thus, the MLE here of $\lambda$ is given by $\lambda = \frac{4}{\sum_{i=1}^{4} x_i}$.

## Problem 6: Convex Set Operations

**1 point**

**Question:** Which of the following can be convex?

**Options:**
- a) The intersection of non-convex sets
- b) The intersection of convex sets
- c) The union of non-convex sets
- d) The union of convex sets

**Correct Answer:** All of them (a, b, c, d)

**Explanation:** 
The answer is all of them.

- For the intersection of non-convex sets, the intersection of two five-pointed stars can be convex.
- For the intersection of convex sets, just consider two circles that are on top of each other.
- For the union of non-convex sets, just consider a circle that is split into two non-convex sets.
- For the intersection of convex sets, just consider two circles that are on top of each other.

## Problem 7: Gradient Descent Convergence

**1 point**

**Question:** For convex optimization objectives, taking a gradient step using full-batch GD ensures that your loss shrinks.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

**Explanation:** 
The answer is False.

Even for convex optimization objectives, if the learning rate is too high, there is a real probability of overshooting the global minima.

## Problem 8: Neural Network Activation Functions

**1 point**

**Question:** You are building a multi-class classifier using a deep neural network. You notice that your network is training slowly and that the gradients are diminishing quickly. Which activation function for the hidden layers of your network should you switch to, in order to avoid these issues?

**Options:**
- a) $f(x_i) = \frac{1}{1+e^{-x_i}}$
- b) $f(x_i) = \max(0, x_i)$
- c) $f(x_i) = x_i$
- d) $f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$

**Correct Answer:** b) $f(x_i) = \max(0, x_i)$

**Explanation:** 
Sigmoid ($f(x_i) = \frac{1}{1+e^{-x_i}}$) can cause vanishing gradients and hence can cause slow learning.
ReLU ($f(x_i) = \max(0, x_i)$) avoids saturation.
Having only linear layers reduces the network to a linear one.
Softmax ($f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$) should be used in the output layer, but not the hidden layers of the network.

## Problem 9: Neural Network Depth and Training Loss

**1 point**

**Question:** If two neural networks differ only in the number of hidden layers, the deeper network will always achieve a lower training loss given the same training data.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

## Problem 10: Reducing Overfitting in Neural Networks

**1 point**

**Question:** Snoopy is training a neural network to classify birds into 'Woodstock' and 'Not Woodstock'. A plot of the training and validation accuracy for the neural network model during the training process is provided.

<img src="./img/q10_problem.png" width="350px">

**Figure 6: Snoopy's Training Plot**
- **Training Accuracy:** Starts around 70% at 2.5 epochs, steadily increases, and reaches approximately 100% accuracy by 15.0 epochs, remaining high thereafter.
- **Validation Accuracy:** Starts around 62% at 2.5 epochs, shows some fluctuations, increases to about 70% by 7.5 epochs, and then continues to slowly increase, reaching approximately 78% by 20.0 epochs.
- **Key Observation:** A significant and increasing gap is observed between the training accuracy and validation accuracy, particularly after about 7.5 epochs, where the training accuracy continues to rise sharply while the validation accuracy plateaus or increases very slowly. This indicates a clear case of overfitting.

Which of the following actions could Snoopy take to help reduce the difference between training and validation accuracy?

**Options:**
- a) Increase the amount of training data
- b) Apply regularization techniques
- c) Reduce the complexity of the model (e.g., use fewer layers or units)
- d) Train for more epochs without making other changes
- e) Decrease the learning rate

**Correct Answer:** a), b), c)

## Problem 11: Feature Selection (LASSO vs. PCA)

**1 point**

**Question:** Although both LASSO and PCA can be used for feature selection, they differ in their approach. Specifically, LASSO sets some weight coefficients to 0 and selects a subset of the original features, whereas PCA selects features that minimize variance and creates a linear combinations of the original features.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

**Explanation:** 
PCA selects features that capture the most variance, and produces a linear combination of the original features.

## Problem 12: Logistic Loss Minimization Objective

**1 point**

**Question:** What is the minimization objective for logistic loss? Here $\hat{y}$ is the prediction, and $y$ is the ground truth label.

**Options:**
- a) $\log(1 + e^{-y\hat{y}})$
- b) $1 + \log(e^{-y\hat{y}})$
- c) $1 + e^{-y\hat{y}}$
- d) $1 + \log(e^{y\hat{y}})$

**Correct Answer:** a) $\log(1 + e^{-y\hat{y}})$

**Explanation:** 
In this classification setting we are attempting to maximize the probability $P(y_i|x_i)$.

Within the logistic regression problem setting, we set $P(y_i|x_i)$ to be equal to $\sigma(y_i w^T x_i)$ where $\sigma(z)$ is the sigmoid function $\frac{1}{1+e^{-z}}$.

If we are attempting to maximize the probability of $P(y_i|x_i)$, this is an equivalent objective to the minimization of the negative logprob.

We therefore have the minimization objective become $-\log(\sigma(y_i w^T x_i))$.

Since $\hat{y}$ is our prediction, it is equivalent to $w^T x_i$.

Finally, using the definition of $\sigma(\cdot)$ and reducing $-\log(\sigma(y\hat{y}))$ gives us $\log(1+e^{-y\hat{y}})$.

## Problem 13: L-infinity Norm Regularization

**1 point**

**Question:** The L-$\infty$ norm is represented as $|| \cdot ||_\infty$ and is calculated for a vector $x \in \mathbb{R}^d$ as $||x||_\infty = \max_i(|x_i|)$. What happens to the parameters in $w$ if we optimize for a standard linear regression objective with L-$\infty$ regularization?

**Options:**
- a) There will be lots of parameters within $w$ that are the same/similar absolute value.
- b) $w$ will be very sparse.
- c) $w$ will not be very sparse.
- d) Not enough information given to determine any of the above.

**Correct Answer:** a), c)

**Explanation:** 
The L-$\infty$ ball in parameter space is a square who's most protruding points are where the absolute values of the parameters are equivalent (corners of the square centered at the origin). Therefore A is correct. We know $w$ will not be sparse because the protruding points of the L-$\infty$ ball are not on the origin. Therefore C is also correct. Because A and C are correct, neither B nor D can be correct.

## Problem 14: K-means Clustering

**1 point**

**Question:** True/False: In k-means, increasing the value of $k$ never worsens the model's performance on training data.

**Options:**
- a) True
- b) False

**Correct Answer:** a) True

**Explanation:** 
Increasing $k$ so that it is equal to $n$ will make it so there is one cluster centroid per data point. This will perfectly fit the training data with zero training loss.

## Problem 15: Principal Component Analysis (PCA)

**1 point**

**Question:** Which of the following statements about PCA are true?

**Options:**
- a) The first principal component corresponds to the eigenvector of the covariance matrix with the smallest eigenvalue.
- b) If all the singular values are equal, PCA will not find a meaningful lower-dimensional representation.
- c) The principal components are the eigenvectors of the covariance matrix of the data.
- d) The reconstruction error of the recovered data points with a rank-q PCA strictly decreases as we increase q for all datasets.

**Correct Answer:** b), c)

**Explanation:** 
- A is false since the first principal component corresponds to the eigenvector with the largest eigenvalue.
- B is correct since if all the singular values are equal, the variance is equally distributed across all directions so PCA won't find a meaningful lower-dimensional representation.
- C is also correct since we find the eigenvalue decomposition of the covariance matrix. It isn't guaranteed that PCA will reduce the dimensionality, for example if all principal components are chosen.

## Problem 16: Singular Value Decomposition (SVD)

**1 point**

**Question:** Consider the $2 \times 2$ matrix:

$A = \begin{bmatrix} 3 & 4 \\ 0 & 0 \end{bmatrix}$

Let the Singular Value Decomposition (SVD) of A be given by: $A = U\Sigma V^T$

where U and V are orthogonal matrices, and $\Sigma$ is a diagonal matrix containing the singular values of A. Which of the following statements are correct?

**Options:**
- a) The rank of A is 1.
- b) The nonzero singular value of A is 5.
- c) The columns of V must be $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$.
- d) The columns of V form an orthonormal basis for $\mathbb{R}^2$.

**Correct Answer:** a), b), d)

**Explanation:** 
- **(A) True:** The rank of A is the number of nonzero singular values. Since the second row of A is entirely zero, its rank is **1**.
- **(B) True:** The singular values of A are given by the square roots of the eigenvalues of $A^T A$:

$A^T A = \begin{bmatrix} 9 & 12 \\ 12 & 16 \end{bmatrix}$

## Problem 17: Kernel Methods

**1 point**

**Question:** True/False: In kernel methods, we use a kernel function $k(x, x')$ to implicitly map input data into a feature space with different dimensions without explicitly computing the transformation. If we choose a linear kernel $k(x,x') = x^T x'$, then this is equivalent to mapping data into an infinite-dimensional feature space.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

**Explanation:** 
The statement is False because the linear kernel $k(x,x') = x^T x'$ does not map the data into an infinite-dimensional feature space. Instead, it corresponds to the original input space (i.e., the feature map $\phi(x)$ is simply $x$ itself).

In contrast, nonlinear kernels, such as the Gaussian (RBF) kernel:

$k(x, x') = \exp\left(-\frac{||x - x'||^2}{2\sigma^2}\right)$

correspond to an infinite-dimensional feature space. This is because an RBF kernel can be expressed as an infinite sum of polynomial terms in a Taylor expansion.

Thus, the error in the statement is that the linear kernel is incorrectly claimed to map data into an infinite-dimensional space, when in reality, it remains in the original finite-dimensional space.

## Problem 18: Kernel Properties

**1 point**

**Question:** Which of the following statements about kernels is/are true?

**Options:**
- a) The kernel trick is a technique for computing the coordinates in a high-dimensional space.
- b) If the kernel matrix $K$ is symmetric, it is always a valid kernel.
- c) Eigenvalues of a valid kernel matrix must always be non-negative.
- d) The kernel trick eliminates the need for regularization.

**Correct Answer:** c)


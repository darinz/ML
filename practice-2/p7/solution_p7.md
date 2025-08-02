# Practice 2 Problem 7 Solution

## Problem 1
**Question:** In the context of logistic regression, which of the following statements is true about the interpretation of the model coefficients?

**Options:**
(a) The coefficients represent the change in the log odds of the dependent variable for a one-unit change in the predictor variable, holding all other variables constant.
(b) The coefficients represent the change in the dependent variable for a one-unit change in the predictor variable, holding all other variables constant.
(c) The coefficients are directly proportional to the probability of the dependent variable being 1.
(d) The coefficients represent the probability that the predictor variable will be present when the dependent variable is 1.

**Correct answer:** (a)

## Problem 2
**Question:** You are working on a machine learning project to classify emails as either spam (1) or not spam (0) using logistic regression. The model has been trained based on emails with labels and several features, including the frequency of specific keywords. For a particular new email, the model's output of the log-odds is 0.4. Given the model's output, which of the following options best describes its classification of the email?

**Options:**
(a) The email is classified as not spam because a positive log-odds score indicates a higher likelihood of the email belonging to the negative class (not spam).
(b) The email is classified as spam because the log-odds score is positive, indicating that the odds of the email being spam are greater than the odds of it not being spam.
(c) The email is classified as not spam because the probability of being spam is less than 0.5.
(d) The email is classified as spam because the probability of it being spam is positive.

**Correct answer:** (b)

**Explanation:** Note that the log-odds is positive, so $P(Y = +1|X = x) > P(Y = 0|X = x)$.

## Problem 3
**Question:** In the context of logistic regression used for binary classification, which of the following statements is true?

**Options:**
(a) The model directly outputs class labels (0 or 1)
(b) The model's optimization has a closed-form solution.
(c) The model produces a linear decision boundary with respect to the features.
(d) The model uses the softmax function to output class probabilities.

**Correct answer:** (c)

## Problem 4
**Question:** Which are key properties of the Radial Basis Function kernel?

**Options:**
(a) It works best when features take on categorical values.
(b) It relies on the distance between points in the original feature space.
(c) It relies on the distance between points in infinite-dimensional space.
(d) It implicitly maps to an infinite-dimensional feature space.
(e) It identifies hyperplanes in an infinite-dimensional space.

**Correct answers:** (b), (d)

## Problem 5
**Question:** Which of the following is not a valid kernel?

**Options:**
(a) $K(x, x') = \frac{1}{\sqrt{2\pi}} \exp(-\frac{1}{2}\|x - x'\|_2^2)$
(b) $K(x,x') = -\frac{1}{\sqrt{2\pi}} \exp(-\frac{1}{2}\|x - x'\|_2^2)$
(c) $K(x,x') = x^T x'$
(d) $K(x, x') = 1$

**Correct answer:** (b)

**Explanation:** Recall that kernels must be positive semidefinite: $K(x,x') \ge 0$ for all $x,x'$. This is not true for $K(x,x') = -\frac{1}{\sqrt{2\pi}} \exp(-\frac{1}{2}\|x - x'\|_2^2)$.

## Problem 6
**Question:** Which of the following statements about the "kernel trick" are true in the context of machine learning algorithms?

**Options:**
(a) It provides an efficient method for computing and representing high-dimensional feature expansions.
(b) It implicitly maps to a high-dimensional feature space.
(c) It eliminates the need for regularization.
(d) It can only be used in regression prediction settings.

**Correct answers:** (b)

**Explanation:** We gave all students full credit for this question because after the exam we realized that choice (b) was too ambiguous. Our initial intention was to grade (b) as correct. Our rationale was: (a) is incorrect because a major motivation for the kernel trick is that it avoids ever needing to compute or represent high-dimensional feature expansions. (b) is how the kernel trick is typically (always?) used in practice and how the kernel trick was always used in lecture and HWs. However, one could technically map to a lower-dimensional feature space. This generally does not make sense to do from the perspective of computational cost. (c) is incorrect—we frequently need regularization with kernel methods, as we saw that kernel regression perfectly fits the training data when no ridge penalty is included. (d) is incorrect—we discussed kernel extensions to PCA in class, and there are many other extensions we did not discuss, e.g., support vector machines for classification.

## Problem 7
**Question:** Consider the kernel ridge regression problem.

$$\hat{w} = \arg\min_w \frac{1}{n} \sum_{i=1}^n (y_i - \phi(x_i)^\text{T} w)^2 + \lambda \Vert w \Vert^2 \quad \text{becomes} \quad \hat{\alpha} = \arg\min_\alpha \Vert Y - K\alpha \Vert_2^2 + \lambda \alpha^\text{T} K \alpha$$

Let $\phi(x): \mathbb{R}^d \to \mathbb{R}^p$ be the feature mapping the kernel matrix $K$ is with respect to. Let $n$ be the number of samples we have. Which of the following statements is true?

**Options:**
(a) Ridge regression can only be kernelized assuming $\alpha \in \text{span}\{x_1, x_2,..., x_n\}$ where $x_i \in \mathbb{R}^d$ denotes the $i$th training sample
(b) When $n \ll p$, the kernel method will be more computationally efficient than using regular ridge regression.
(c) There is no closed-form solution if $K$ is positive definite.
(d) The optimal $\hat{w}$ can be obtained after solving for the optimal $\hat{\alpha}$ even though $w$ is not explicitly included in the optimization problem

**Correct answers:** (b), (d)

## Problem 8
**Question:** Assume we have $n$ samples from some distribution $P_X$, and wish to estimate the variance of $P_X$, as well as compute a confidence interval on the variance. If $n = 1$ and we draw only a single datapoint $X_1 = 2$ from $P_X$, which of the following are true?

**Options:**
(a) The bootstrap estimate of the variance is 0.
(b) The bootstrap estimate of the variance is 2.
(c) The bootstrap cannot be applied when we only have $n = 1$ samples.
(d) The bootstrap is likely to give a very poor estimate of the variance in this setting.

**Correct answers:** (a), (d)

**Explanation:** Given $n$ samples from a distribution, the bootstrap estimate of the variance is calculated by drawing some number of samples with replacement from this data, computing the variance on these samples, then repeating, and averaging the variance values to get a final estimate. When $n = 1$, however, we will always sample the same point, and so the variance will always be 0, regardless of the distribution's true variance. Thus, in this case, while the bootstrap estimate is well-defined, it is very inaccurate.

## Problem 9
**Question:** Which of the following statements about the bootstrap method are true?

**Options:**
(a) It requires a large sample size to be effective and cannot be used effectively with small datasets.
(b) It involves repeatedly sampling with replacement from a dataset to create samples and then calculating the statistic of interest on each sample.
(c) Bootstrap methods can only be applied to estimate the mean of a dataset and do not apply to other statistics like median or variance.
(d) One of the advantages is that it does not make strong parametric assumptions about the distribution of the data.
(e) It can be used to estimate confidence intervals for almost any statistic, regardless of the original data distribution.

**Correct answers:** (b), (d), (e)

## Problem 10
**Question:** Which of the following are advantages of using random forests over decision trees?

**Options:**
(a) The optimal decision tree cannot be efficiently computed, but the optimal random forest can.
(b) Random forests typically have smaller variance than decision trees.
(c) Random forests typically have smaller bias than decision trees.
(d) Random forests are less prone to overfitting compared to decision trees.

**Correct answers:** (b), (d)

**Explanation:** Decision trees can overfit to the data and have high variance, as the decision criteria may very often capture noise, and be very sensitive. In contrast, random forests typically have lower variance, as they are trained on smaller subsets of features and bootstrapped data, thereby reducing the sensitivity to any particular feature of data point.

## Problem 11
**Question:** Which of the following is true about $k$-nearest neighbors (KNN)?

**Options:**
(a) KNN works best with high dimensional data.
(b) When $k=1$, the training error is always less than or equal to the test error.
(c) The computational cost of making a prediction on a new test point increases with the size of the training dataset.
(d) The effectiveness of KNN is independent of the distance metric used.

**Correct answers:** (b), (c)

## Problem 12
**Question:** For $k$-nearest neighbors (KNN), changing $k$ will affect:

**Options:**
(a) Bias
(b) Variance
(c) Both bias and variance
(d) Neither bias nor variance

**Correct answer:** (c)

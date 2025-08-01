# Practice 2 Solutions

## Problem 1

**1. Which of the following is the definition of irreducible error in machine learning?**
*   (a) The error that cannot be eliminated by any model
*   (b) The error that is caused by overfitting to the training data
*   (c) The error that is caused by underfitting to the testing data
*   (d) All of the above

**Correct answers:** (a)

**2. What is the general model for $P(Y = 1|X = x,\theta)$ in logistic regression, where $X = (X_0, X_1,..., X_n)$ is the features, $Y$ is the predictions, and $\theta$ is the parameters? Assume that a bias term has already been appended to $X$ (i.e., $X_0 = 1$).**
*   (a) $P(Y = 1|X = x, \theta) = \frac{1}{1+e^{-\theta^T x}}$
*   (b) $P(Y = 1|X = x, \theta) = \theta^T x$
*   (c) $P(Y = 1|X = x, \theta) = \log(1 + e^{-\theta^T x})$
*   (d) $P(Y = 1|X = x, \theta) = \log(1 + e^{\theta^T x})$

**Correct answers:** (a)

**3. Two realtors are creating machine learning models to predict house costs based on house traits (i.e. house size, neighborhood, school district, etc.) trained on the same set of houses, using the same model hyperparameters. Realtor A includes 30 different housing traits in their model. Realtor B includes 5 traits in their model. Which of the following outcomes is most likely?**
*   (a) Realtor B's model has higher variance and lower bias than Realtor A's model
*   (b) Realtor A's model has higher variance than Realtor B's model and without additional information, we cannot know which model has a higher bias
*   (c) Realtor A's model has higher variance and lower bias than Realtor B's model
*   (d) Realtor A's model has higher variance and higher bias than Realtor B's model

**Correct answers:** (b)

## Problem 2

**4. When $L(w,b) = \sum_{i=1}^{n}(y_i - (w^T x_i + b))^2$ is used as a loss function to train a model, which of the following is true?**
*   (a) It minimizes the sum of the absolute differences between observed and predicted values.
*   (b) It maximizes the correlation coefficient between the independent and dependent variables.
*   (c) It requires the use of gradient descent optimization to find the best-fit line.
*   (d) It minimizes the sum of the squared difference between observed and predicted values.

**Correct answers:** (d)

**5. True/False: As the value of the regularization term coefficient in Ridge Regression increases, the sensitivity of predictions to inputs decreases.**
*   (a) True
*   (b) False

**Correct answers:** (a)

**6. Which of the following statements about logistic regression is true?**
*   (a) The loss function of logistic regression without regularization is convex, and the loss function of logistic regression with L2 regularization is convex.
*   (b) Neither the loss function of logistic regression without regularization is convex nor the loss function of logistic regression with L2 regularization is convex.
*   (c) The loss function of logistic regression without regularization is convex, but the loss function of logistic regression with L2 regularization is non-convex.
*   (d) The loss function of logistic regression without regularization is non-convex, but the loss function of logistic regression with L2 regularization is convex.

**Correct answers:** (a)

## Problem 3

**7. Which of the following is NOT an assumption of logistic regression?**
*   (a) The output target is binary.
*   (b) The input features can be continuous or categorical.
*   (c) The residual errors are normally distributed.

**Correct answers:** (c)

**Explanation:** Note: Option A was also accepted as a correct answer as logistic regression can refer to multi-class logistic regression.

**8. Suppose we've split a dataset into train, validation, and test sets; trained a regression model on the train set; and found the optimal value for a regularization constant $\lambda$. Select all of the regression methods for which adding the validation set into the train set and retraining can change the optimal value for $\lambda$.**
*   (a) LASSO regression
*   (b) Ridge regression

**Correct answers:** (a), (b)

**9. Suppose that we want to estimate the ideal parameter $\theta^*$ for $p(x, y, \theta)$ given a set of observations $\{x_i, y_i\}$. Which of the following is a key assumption made when using $\hat{\theta}_{MLE} = \arg \max_{\theta} \sum_i \log(p(x_i, y_i|\theta_i))$ for Maximum Likelihood Estimation (MLE) to estimate the model parameter?**
*   (a) The data is normally distributed.
*   (b) The data is independent and identically distributed (i.i.d.).
*   (c) The data contains no outliers.
*   (d) The data is linearly separable.

**Correct answers:** (b)

## Problem 4

**10. Provide one advantage and one disadvantage of Stochastic Gradient Descent (SGD) over Gradient Descent (GD).**

**Answer:**

**Explanation:** One possible upside: SGD is much faster than GD. One possible downside: Because of stochasticity in SGD, optimizing with SGD can result in a lot of noise in training metrics, making it hard to find a stopping point.

## Problem 5

**11. Assume a simple linear model $Y = \beta_1 X$. For simplicity, no intercept is considered. Given the following dataset:**

$X = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$

$Y = \begin{pmatrix} 3 \\ 5 \\ 7 \end{pmatrix}$

**(a) (1 point) Compute the least squares estimate of $\beta_1$ without any regularization. You may leave your answer as a fraction, if necessary.**

**Answer:** $\hat{\beta}_1 = \frac{17}{7}$

**(b) (1 point) Using Lasso Regression (equation 11) with a penalty term $\alpha = 2$, would $\beta_1$ increase or decrease? Provide a short explanation.**

**Answer:** Decrease

**Explanation:**
1. For the simple linear model without regularization:
$\hat{\beta}_1 = \frac{\sum_{i=1}^{3} X_i Y_i}{\sum_{i=1}^{3} X_i^2} = \frac{3(1) + 5(2) + 7(3)}{1^2 + 2^2 + 3^2} = \frac{3 + 10 + 21}{1 + 4 + 9} = \frac{34}{14} = \frac{17}{7}$

2. Now, if $\beta_1$ is positive and greater than zero, the L1 penalty will encourage the coefficient to shrink towards zero. In other words, the Lasso regularization "penalizes" larger coefficients, pushing them towards zero. So, given the same data and a positive $\alpha$, the coefficient $\beta_1$ in Lasso regression will always be less than or equal to its value in simple linear regression without regularization.

## Problem 6

**12. Suppose you're given a scatter plot of a dataset, and the pattern appears to be a periodic wave-like curve that repeats itself at regular intervals.**

<img src="./scatter_plot.png" width="450px">

**Which of the following basis functions might be most appropriate to capture the relationship between $x$ and $y$ for this dataset?**

*   (a) Polynomial basis functions: $\phi(x) = \{1, x, x^2, x^3, ...\}$
*   (b) Radial basis functions: $\phi(x) = \exp(-\lambda||x - c||^2)$
*   (c) Fourier basis functions: $\phi(x) = \{1, \sin(\omega x), \cos(\omega x), \sin(2\omega x), \cos(2\omega x), ...\}$
*   (d) Logarithmic basis function: $\phi(x) = \log(x)$
*   (e) Exponential basis function: $\phi(x) = \exp(\lambda x)$

**Correct answers:** (c)

**Explanation:** Fourier basis functions are particularly suitable for capturing periodic wave-like patterns in data.

**13. Which of the following statements about convexity is true?**

*   (a) If $f(x)$ is convex, then $g(x) = \frac{1}{3}f(x)$ is also convex
*   (b) If $f(x)$ is convex, then gradient descent on minimizing $f(x)$ will always reach global minimum
*   (c) If $f(x)$ is convex, then $f(x)$ is everywhere differentiable

**Correct answers:** (a)

## Problem 7

**14. What are the unbiased maximum likelihood estimates (MLE) for the parameters $(\mu, \sigma)$ of a univariate Gaussian distribution, given a dataset of $n$ independently sampled 1-dimensional data points $X = \{x_1, ..., x_n\}$ and the sample mean $\bar{x}$?**

*   (a) $\hat{\mu}_{MLE} = \bar{x}$, $\hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^n x_i$
*   (b) $\hat{\mu}_{MLE} = \bar{x}$, $\hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu}_{MLE})^2$
*   (c) $\hat{\mu}_{MLE} = \bar{x}$, $\hat{\sigma}^2_{MLE} = \frac{1}{n-1} \sum_{i=1}^n (x_i - \hat{\mu}_{MLE})^2$
*   (d) $\hat{\mu}_{MLE} = \frac{1}{n} \bar{x}$, $\hat{\sigma}^2_{MLE} = \frac{1}{n-1} \sum_{i=1}^n (x_i - \hat{\mu}_{MLE})^2$

**Correct answers:** (c)

**15. True/False: When performing gradient descent, decreasing the learning rate enough will slow down convergence but will eventually guarantee you arrive at the global minimum.**

*   (a) True
*   (b) False

**Correct answers:** (b)

**16. Which of the following functions is strictly convex over its entire domain?**

*   (a) $f(x) = -x^2$
*   (b) $f(x) = x^3$
*   (c) $f(x) = \ln(x)$
*   (d) $f(x) = e^x$

**Correct answers:** (d)

## Problem 8

**17. Which of the following is true about a validation set and how it is used?**

*   (a) The validation set allows us to estimate how a model would perform on unseen data
*   (b) When deciding to use a validation set, you do not need a separate test set
*   (c) After hyperparameter tuning, the validation set is always added back into the training set before training the final model
*   (d) The validation set allows us to train a model quicker by decreasing the size of our training data set

**Correct answers:** (a)

## Problem 9

**18. (2 points) Suppose we have the function**

$$f(x) = \begin{cases} 1 - e^{-\frac{1}{x^2}} & x \neq 0 \\ 1 & x = 0 \end{cases}$$

<img src="./function.png" width="450px">

**(a) (1 point) Suppose that we perform gradient descent starting at $x_0 = 0$ with step size $\eta = 1$. What is the asymptotic behavior of gradient descent given by Equation 12?**

$$x_{n+1} = x_n - \eta f'(x_n) \quad (12)$$

**Answer:** The gradient descent will be stationary at $x=0$.

**(b) (1 point) Now suppose that $x_0 \sim \mathcal{N}(0, \epsilon)$ for some small $\epsilon$. What is the behavior then?**

**Answer:** For $x_0 \neq 0$, the gradient descent will head towards $\text{sign}(x_0) \infty$ very slowly.

**Explanation:** For $x_0 = 0$ gradient descent is stationary and for $x_0 \neq 0$ it will head towards $\text{sign}(x_0) \infty$ very slowly.

# Problem Set 8 Solutions

## Problem 1: Irreducible Error

**Question:** Which of the following is the cause/reason for irreducible error?

**Options:**

a) Stochastic label noise

b) Very few data points

c) Nonlinear relationships in the data

d) Insufficient model complexity

**Correct Answer:** (a)

**Explanation:** A is correct. Stochastic label noise is what drives irreducible error. See lecture 4 slides. In essence, irreducible error comes from randomness that cannot be modeled since there is no deeper pattern to it. B and D are wrong because fewer data points and insufficient model complexity are responsible for reducible error. C is wrong because nonlinear relationships in the data don't have anything to do with irreducible error.

## Problem 2: Neural Network Bias and Variance

**Problem:** Saket unfortunately did not learn from the midterm and still has not attended lecture. He is now given the task of training 3 neural networks with increasing complexity on a regression task:

- Model A: 1 hidden layer with 10 neurons.
- Model B: 2 hidden layers with 50 neurons each.
- Model C: 10 hidden layers with 100 neurons each.

After training and evaluating these models on an appropriately split dataset with train and test splits, you find the following MSEs:

- Model A: train MSE = 2.5, test MSE = 2.6
- Model B: train MSE = 0.1, test MSE = 0.2
- Model C: train MSE = 0.01, test MSE = 1.3

Saket only knows about bias and variance, so based on the model architectures and train/test MSE losses, choose the best relative bias/variance estimates for each of the models.

**Correct Answer:** 
- Model A: High bias, Low variance
- Model B: Low bias, Low variance  
- Model C: Low bias, High variance

**Explanation:** Due to the simpler architecture and high MSEs, A likely underfits. B achieves low but similar train/test MSEs so probably has a good balance. C has a low train MSE but a high test MSE so is probably overfitting, which matches the likely overcomplex architecture.

## Problem 3: K-fold Cross-validation

**Question:** Explain one upside and one downside of using a high K in K-fold cross validation.

**Upside:** You get a more accurate estimate of your test error, possibly making hyperparameter selection more accurate.

**Downside:** A higher K means more folds and therefore much more compute/time needed to find the right hyperparameters. A higher K also means each validation set has fewer data points. This will result in higher variability in the results across different folds.

## Problem 4: Training and Validation Loss Analysis

**Question:** You are training a model and get the following plot for your training and validation loss.

**Plot Description:** A line plot titled "Loss" with "Number of Epochs" on the x-axis (ranging from 2 to 20) and "Loss" on the y-axis (ranging from 0.0 to 1.0). There are two lines:
- **train (solid blue line):** Starts at a loss of 1.0 at epoch 2, rapidly decreases to near 0.0 by epoch 7, and remains very low (close to 0.0) up to epoch 20.
- **validation (dashed orange line):** Starts at a loss of 1.0 at epoch 2, decreases to a minimum loss of approximately 0.4 at epoch 7, and then steadily increases to about 0.75 by epoch 20.

**Question:** Which of the following statements are true?

**Options:**

a) The model has high bias and low variance.

b) The large gap between training and validation loss indicates underfitting.

c) Training for more epochs will eventually decrease validation loss.

d) The model might be too complex for the dataset.

e) The model is likely memorizing the training data.

**Correct Answers:** (d), (e)

**Explanation:** This is a classic example of overfitting, which is caused when we have too complex of a model and it ends up memorizing the training set. Overfitting means the model has low bias and high variance. Thus, the only correct options are D and E.

## Problem 5: Maximum Likelihood Estimation

**Question:** Which of the following models that we studied in class use maximum likelihood estimation?

**Options:**

a) Linear regression with Gaussian noise model

b) Principal Components Analysis

c) Gaussian Mixture Models

d) Neural Network trained to do classification with softmax cross entropy loss

**Correct Answers:** (a), (c), (d)

**Explanation:** 
a) True: Linear regression with Gaussian noise model is true because you maximize the likelihood of the data under a linear model which assumes a gaussian distribution on errors
b) False: PCA does not use MLE because it does not define a probabilistic distribution for the data, it just uses linear algebra to find vectors that explain a lot of variance in the data
c) True: Gaussian Mixture Models define a probability distribution that is a mixture of Gaussians and then find the parameters by maximizing likelihood under that model
d) True: NNs with softmax define a probability distribution over the classification labels and try to maximize it with cross entropy

## Problem 6: Maximum Likelihood Estimation for Coin Flips

**Question:** Yann, a strict frequentist statistician, observes 5 flips of a possibly uneven coin. Here are the outcomes:
1. Heads
2. Tails
3. Heads
4. Heads
5. Tails

Based on these observations, Yann uses maximum likelihood estimation to determine the most likely outcome of the next coin toss. What does he predict will happen?

**Options:**

a) Heads

b) Tails

c) Both are equally likely

d) It hits Marco in the head

**Correct Answer:** (a)

**Explanation:** There were 3 Heads and 2 Tails. Based on these observations, the estimated probability of Heads is $\frac{3}{5} = 0.6$, which is greater than the estimated probability of Tails ($\frac{2}{5} = 0.4$). Therefore, Heads is the most likely outcome.

## Problem 7: Convex Functions and Optimization

**Problem:** Let $f: \mathbb{R}^d \to \mathbb{R}$ be differentiable everywhere, such that $f(y) \ge f(x) + \nabla f(x)^T (y-x)$ for all $x, y \in \mathbb{R}^d$. Suppose there exists a unique $x_* \in \mathbb{R}^d$ such that $\nabla_x f(x_*) = 0$.

**Part (a):** $x_*$ is a:

a) Minimizer of $f$

b) Maximizer of $f$

c) Saddle point of $f$

d) Not enough information to determine any of the above

**Part (b):** Suppose we are unable to solve for $x_*$ in closed-form. Briefly outline a procedure for finding $x_*$.

**Correct Answer:** (a)

**Explanation:** 
- Part (a): $f$ is convex, so $x_*$ must be a minimizer of $f$.
- Part (b): Gradient descent

## Problem 8: Gradient Descent Convergence

**Question:** Which of the following is true, given the optimal learning rate?

**Clarification:** All options refer to convex loss functions that have a minimum bound / have a minimum value.

**Options:**

a) For convex loss functions, stochastic gradient descent is guaranteed to eventually converge to the global optimum while gradient descent is not.

b) For convex loss functions, both stochastic gradient descent and gradient descent will eventually converge to the global optimum.

c) Stochastic gradient descent is always guaranteed to converge to the global optimum of a loss function.

d) For convex loss functions, gradient descent with the optimal learning rate is guaranteed to eventually converge to the global optimum point while stochastic gradient descent is not.

**Correct Answer:** (d)

**Explanation:** Due to the noisy updates of SGD, it is not guaranteed to converge at the minimum but for instance, cycle close to it whereas batch gradient descent alleviates this and is guaranteed to reach the minimum given appropriate step size.

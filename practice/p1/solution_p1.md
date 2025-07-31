# Practice Problem Set 1 Solutions

Consider the following matrix $X$ with four data points in $\mathbb{R}^2$. We would like to use PCA to find a rank-1 linear representation of these data.

$X = \begin{bmatrix} 4 & 1 \\ 5 & 4 \\ 1 & 2 \\ 1 & 0 \end{bmatrix}$

**Figure 2:** These plots depict data points from the four-sample dataset $X$.

## Question 1
Which line in Figure 2 represents the direction of the first principal component of $X - \mu$, where $X \in \mathbb{R}^{n \times d}$ the vector $\mu \in \mathbb{R}^d$ is the featurewise mean of $X$?

(A) Plot 1  
(B) Plot 2  
(C) Plot 3  
(D) Plot 4  

**Solution:** The answer is (A).

## Question 2
Which of the following statements about kernels is false?

(A) Kernel feature vectors $\phi(x_i)$ can be infinite-dimensional  
(B) Kernels methods scale well to large datasets because the size of the kernel matrix does not depend on the size of the dataset  
(C) Kernel matrices store the results of inner products of the data's features computed in a higher-dimensional space  
(D) Kernels allow otherwise linear models to find non-linear decision boundaries  

**Solution:** The answer is (B).

## Question 3
Suppose you have a logistic regression model for spam detection, using a dataset with a binary outcome that indicates whether an email is spam (1) or not spam (0). The predictor variables $x_1$, $x_2$, and $x_3$ are boolean values (0 or 1) that indicate whether the email contains the words "free", "order", and "homework", respectively. The model has four parameters: weights $w_1$, $w_2$, $w_3$, and offset $b$. You find that emails containing the words "free" and "order" have a higher probability of being spam, while emails containing the word "homework" have a lower probability of being spam. Given this information, which of the following signs is most likely for the weights $w_1$, $w_2$, and $w_3$?

(A) All positive  
(B) All negative  
(C) $w_1$ and $w_2$ are positive, $w_3$ is negative  
(D) $w_1$ and $w_2$ are negative, $w_3$ is positive  

## Question 4
**True/False:** Solving the k-means objective is an unsupervised learning problem.

(A) True  
(B) False  

**Solution:** The answer is (A).

## Question 5
Which of the following is typical for decision trees trained to have 0 training error?

(A) High bias, low variance  
(B) High bias, high variance  
(C) Low bias, high variance  
(D) Low bias, low variance  

**Solution:** The answer is (C).

## Question 6
When is PCA ineffective?

(A) When data has an orthogonal underlying structure.  
(B) When the data's underlying low-dimensional structure is non-linear.  
(C) When the data is standardized.  
(D) When data visualisation is needed.  

**Solution:** The answer is (B).

## Question 7
The kernel matrix $K$ is not

(A) Symmetric  
(B) Square  
(C) Positive semi-definite (defined as $x^T Kx > 0$ for every nonzero column vector $x$; a necessary condition for this is that $K$ has nonnegative eigenvalues).  
(D) Elementwise positive  

**Solution:** The answer is (D).

## Question 8
**True/False:** Ridge regression's optimal parameters $\hat{w} = (X^T X + \lambda I)^{-1} X^T y$ are a linear combination of the data points $x_i$ in $X$.

(A) True  
(B) False  

**Solution:** The answer is (A).

## Question 9
**True/False:** Solving the k-means objective with Lloyd's algorithm (shown in lecture) will always converge to the global optimum of the k-means objective.

(A) True  
(B) False  

**Solution:** The answer is (B).

## Question 10
When might it be appropriate to use ridge regression instead of (unregularized) least square regression?

(A) When the data is linearly separable.  
(B) When the number of predictor variables is very large relative to the number of observations ($d > n$).  
(C) When there are categorical or one-hot features in the input dataset.  
(D) When the data is not standardized.  

**Solution:** The answer is (B).

## Question 11
**True/False:** For PCA, the objective function can equivalently be thought of as (1) variance-maximization or (2) reconstruction error-minimization.

(A) True  
(B) False  

**Solution:** The answer is (A).

## Question 12
Which of the following activation functions can be used in the output layer of a neural network if we wish to predict the probabilities of $k$ classes $\hat{p} = (p_1, p_2,..., p_k)$ such that sum of $\hat{p}$ over all $k$ equals to 1? (Assume $k \ge 2$.)

(A) Tanh  
(B) Leaky ReLU  
(C) Sigmoid  
(D) Softmax  

**Solution:** The answer is (D).

## Question 13
**True/False:** For decision tree algorithms, small perturbation in the training data can result in large differences in the resulting classifiers.

(A) True  
(B) False  

**Solution:** The answer is (A).

## Question 14
**True/False:** The unfairness with respect to race of a model trained on a dataset can be completely resolved by removing race as a feature.

(A) True  
(B) False  

**Solution:** The answer is (B).

## Question 15
For the following code snippet on the bootstrap algorithm, determine whether it is correct or select the function that is possibly buggy.

```python
1 import random
2
3 def sample(data):
4     sample = random.sample(data, len(data)) # samples len(data) elements without replacement
5     return sample
6
7 def mean(numbers):
8     return sum(numbers) / len(numbers)
9
10 def bootstrap(data, num_samples):
11     samples = [sample(data) for _ in range(num_samples)]
12     sample_means = [mean(sample) for sample in samples]
13     return sample_means
14
15 data = [1, 2, 3, 4, 5]
16 bootstrap_means = bootstrap(data, 10)
```

(A) Code is correct.  
(B) sample function is buggy.  
(C) mean function is buggy.  
(D) bootstrap function is buggy (assuming sample function and mean function is correct).  

**Solution:** The answer is (B).

## Question 16
**True/False:** The bootstrap method can be applied to other statistics, not just variance.

(A) True  
(B) False  

**Solution:** The answer is (A).

## Question 17
**True/False:** The cluster centers that are calculated during each iteration of Lloyd's algorithm are always actual data points.

(A) True  
(B) False  

**Solution:** The answer is (B).

## Question 18
What kind of method can be used to tune models and hyperparameter selection so as to optimize the bias-variance tradeoff?

(A) Bootstrap.  
(B) $k$-means.  
(C) Cross validation.  
(D) All of the above.  

**Solution:** The answer is (C).

## Question 19
**True/False:** The expected error on unseen samples is at least the irreducible error.

(A) True  
(B) False  

**Solution:** The answer is (A).

## Question 20
Which of the following methods would not help when a model suffers from high bias?

(A) Add more input features.  
(B) Standardizing the data (to have mean 0, variance 1).  
(C) Decrease regularization.  
(D) Increase the complexity of the hypothesis class.  

**Solution:** The answer is (B).

## Question 21
Which of the following would be the most appropriate loss function to use when training a neural network on a multi-class classification problem?

(A) Mean Absolute Error  
(B) Mean Squared Error  
(C) Cross Entropy  
(D) Hinge loss  

**Solution:** The answer is (C).

## Question 22
Which of the following does not increase the complexity of a neural network?

(A) Adding more layers  
(B) Increasing the hidden layer size  
(C) Reducing the strength of the regularizer  
(D) Reducing the learning rate  

**Solution:** The answer is (D).


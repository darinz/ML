# Practice 6 Solutions

**1. In a machine learning classification problem, you have a dataset with two classes: Positive (P) and Negative (N). The probability of a randomly selected sample being Negative is 0.6. The probability of a correct classification given that the sample is Positive is 0.8, and the probability of a correct classification given that the sample is Negative is 0.6. What is the probability that a randomly selected sample is Positive given that it has been classified as Positive?**

*   (a) $\frac{4}{7}$
*   (b) $\frac{8}{17}$
*   (c) $\frac{4}{5}$
*   (d) $\frac{4}{15}$

**Correct answers:** (a)

**2. What is NOT true in the following statements?**

**The optimal weight $\hat{W}$ is given by the formula:**
$$\hat{W} = (X^T X + \lambda I)^{-1} X^T Y$$

**where:**
*   $X = [x_1 \cdots x_n]^T \in \mathbb{R}^{n \times d}$
*   $Y = [y_1 \cdots y_n]^T \in \mathbb{R}^{n \times k}$

*   (a) When $\lambda > 0$, the matrix $X^T X + \lambda I$ is invertible.
*   (b) The identity $I$ is a $d \times d$ matrix.
*   (c) When $\lambda = 0$, the matrix is not full-rank, so there is no solution for Ridge Regression.
*   (d) If we apply a unitary transform $U \in \mathbb{R}^{d \times d}$ ($U^T U = I$) on the input $X$ and output $Y$ to get another dataset $(UX, UY)$, the new estimated weight would still be $\hat{W}$.

**Correct answers:** (c)

**The next two questions:**

A fresh graduate of CSE 446 is helping a biologist friend model the relationship between the concentration $y$ of amino acid Arginine in blood plasma and time $x$ in hours after interacting with a reagent. The experiment measured concentration within 3 distinct time blocks (A, B, C):

*   **A:** time $x = 0$ to around 6 hours (represented by circles)
*   **B:** time $x$ around 6 hours to $x$ around 12 hours (represented by squares)
*   **C:** time $x$ around 12 hours to $x$ around 16 hours (represented by the symbol 'x')

<img src="./experiment.png" width="350px">

**3. Based on the scatter plot above, which of the following statements is most likely to be true?**

*   (a) The relationship between $x$ and $y$ is linear across all time blocks.
*   (b) The relationship between $x$ and $y$ is non-linear and follows a piecewise pattern.
*   (c) There is no relationship between $x$ and $y$.
*   (d) The relationship between $x$ and $y$ is exponential.

**Correct answers:** (b)

**4. If you were to fit a linear regression model to this data, which of the following would be the most appropriate approach?**

*   (a) Fit a single linear model to all the data points.
*   (b) Fit separate linear models for each time block (A, B, C).
*   (c) Use polynomial regression with degree 2 or higher.
*   (d) Use logistic regression.

**Correct answers:** (b)

**5. Consider a binary classification problem where you want to predict whether a customer will buy a product (class 1) or not (class 0). You have the following confusion matrix:**

|                | Predicted 0 | Predicted 1 |
|----------------|-------------|-------------|
| **Actual 0**   | 80          | 20          |
| **Actual 1**   | 10          | 90          |

**What is the precision of this classifier?**

*   (a) 0.75
*   (b) 0.80
*   (c) 0.82
*   (d) 0.90

**Correct answers:** (c)

**6. In the context of the same confusion matrix from question 5, what is the recall (sensitivity) of the classifier?**

*   (a) 0.75
*   (b) 0.80
*   (c) 0.82
*   (d) 0.90

**Correct answers:** (d)

**7. A machine learning model has a training accuracy of 95% and a validation accuracy of 70%. This is most likely an example of:**

*   (a) Underfitting
*   (b) Overfitting
*   (c) Good generalization
*   (d) Data leakage

**Correct answers:** (b)

**8. Which of the following regularization techniques is most effective for feature selection?**

*   (a) L1 regularization (Lasso)
*   (b) L2 regularization (Ridge)
*   (c) Dropout
*   (d) Early stopping

**Correct answers:** (a)

**9. In cross-validation, what is the main advantage of using k-fold cross-validation over leave-one-out cross-validation?**

*   (a) It's computationally faster
*   (b) It provides better estimates of model performance
*   (c) It's more robust to outliers
*   (d) It requires less data

**Correct answers:** (a)

**10. Which of the following is NOT a valid reason for standardizing features before training a machine learning model?**

*   (a) To ensure all features have the same scale
*   (b) To improve convergence speed of gradient descent
*   (c) To make the model more interpretable
*   (d) To increase the model's accuracy

**Correct answers:** (d)

**11. In a neural network, what is the primary purpose of the activation function?**

*   (a) To increase the number of parameters
*   (b) To introduce non-linearity into the model
*   (c) To reduce computational complexity
*   (d) To normalize the input data

**Correct answers:** (b)

**12. Which of the following loss functions is most appropriate for binary classification?**

*   (a) Mean Squared Error (MSE)
*   (b) Binary Cross-Entropy
*   (c) Mean Absolute Error (MAE)
*   (d) Hinge Loss

**Correct answers:** (b)

**13. What is the main difference between Stochastic Gradient Descent (SGD) and Batch Gradient Descent?**

*   (a) SGD uses momentum while Batch GD does not
*   (b) SGD updates parameters using a single sample while Batch GD uses all samples
*   (c) SGD is faster but less accurate than Batch GD
*   (d) SGD requires more memory than Batch GD

**Correct answers:** (b)

**14. In the context of machine learning, what does the term "bias" refer to?**

*   (a) The difference between predicted and actual values
*   (b) The systematic error that occurs when a model is too simple
*   (c) The random error in the data
*   (d) The difference between training and validation performance

**Correct answers:** (b)

**15. Which of the following is a valid approach to handle imbalanced datasets?**

*   (a) Always use accuracy as the evaluation metric
*   (b) Use techniques like SMOTE or class weights
*   (c) Remove samples from the majority class
*   (d) Increase the learning rate

**Correct answers:** (b)

**16. What is the primary purpose of the validation set in machine learning?**

*   (a) To train the final model
*   (b) To evaluate model performance during training
*   (c) To test the final model
*   (d) To preprocess the data

**Correct answers:** (b)

**17. In the context of decision trees, what does "pruning" refer to?**

*   (a) Adding more branches to the tree
*   (b) Removing branches to prevent overfitting
*   (c) Changing the splitting criteria
*   (d) Increasing the depth of the tree

**Correct answers:** (b)

**18. Which of the following is NOT a common hyperparameter for a neural network?**

*   (a) Learning rate
*   (b) Number of layers
*   (c) Number of epochs
*   (d) Number of features

**Correct answers:** (d)

**19. What is the main advantage of using ensemble methods like Random Forest?**

*   (a) They are always more accurate than single models
*   (b) They reduce overfitting through averaging
*   (c) They are computationally faster than single models
*   (d) They require less data than single models

**Correct answers:** (b)

**20. In the context of clustering, what does the "elbow method" help determine?**

*   (a) The optimal number of clusters
*   (b) The best clustering algorithm
*   (c) The distance metric to use
*   (d) The initialization method

**Correct answers:** (a)


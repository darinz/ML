# Problem Set 3 Solutions

## Problem 1

If we have $n$ data points and $d$ features, we store $nd$ values in total. We can use principal component analysis to store an approximate version of this dataset in fewer values overall. If we use the first $q$ principal components of this data, how many values do we need to approximate the original demeaned dataset? Justify your answer.

**Answer:** $qd + qn$

**Explanation:** The answer is $qd + qn$. The first term is due to the fact that we store $q$ principal components each in $\mathbb{R}^d$. We also store $q$ coefficients for each of the principal components for each of the $n$ data points. Justification must be correct and must match answer to receive credit.

## Problem 2

Suppose we have a multilayer perceptron (MLP) model with 17 neurons in the input layer, 25 neurons in the hidden layer and 10 neuron in the output layer. What is the size of the weight matrix between the hidden layer and output layer?

(a) $25 \times 17$

(b) $10 \times 25$

(c) $25 \times 10$

(d) $17 \times 10$

**Correct answers:** (b), (c)

**Explanation:** Both options b and c were accepted for this problem.

## Problem 3

Recall that a kernel function $K(x,x')$ is a metric of the similarity between two input feature vectors $x$ and $x'$. In order to be a valid kernel function, $K(x,x') = \phi(x)^T \phi(x')$ for some arbitrary feature mapping function $\phi(x)$. Which of the following is **not** a valid kernel function for input features $x, x' \in \mathbb{R}^2$?

(a) $(x^T x')^2$

(b) $3x^T x'$

(c) $x^T x'$

(d) All of the above are valid

**Correct answers:** (d)

**Explanation:** The answer is (D). Note that $x, x' \in \mathbb{R}^2$ for this problem.

For (A), $(x^T x')^2 = (x_1x_1' + x_2x_2')^2 = x_1^2x_1'^2 + 2x_1x_1'x_2x_2' + x_2^2x_2'^2 = \begin{bmatrix} x_1^2 \\ \sqrt{2}x_1x_2 \\ x_2^2 \end{bmatrix}^T \begin{bmatrix} x_1'^2 \\ \sqrt{2}x_1'x_2' \\ x_2'^2 \end{bmatrix}$

$\phi(x)^T \phi(x')$.

For (B), $3x^T x' = (\sqrt{3}x)^T (\sqrt{3}x') = \phi(x)^T \phi(x')$, where $\phi(x) = \sqrt{3}x$.

For (C), $\phi(x) = x$.

Since all are valid, the answer is (D).

## Problem 4

Consider the following figure. Which shape is not convex?

<img src="./shape.png" width="600px">

(a) I.

(b) II.

(c) III.

(d) IV.

**Correct answers:** (b)

## Problem 5

What is the typical effect of increasing the penalty ($\lambda$) in the ridge regression loss function? Select all that apply.

(a) It increases the bias of the model.

(b) It decreases the bias of the model.

(c) It increases the variance of the model.

(d) It decreases the variance of the model.

**Correct answers:** (a), (d)

## Problem 6

Suppose we are performing linear regression using a non-linear basis expansion $\Phi$. Which of the following statements is true about the learned predictor?

(a) It is a linear function of the inputs and a linear function of the weights.

(b) It is a linear function of the inputs and a non-linear function of the weights.

(c) It is a non-linear function of the inputs and a linear function of the weights.

(d) It is a non-linear function of the inputs and a non-linear function of the weights.

**Correct answers:** (c)

## Problem 7

Which of the following is true about the k-nearest neighbors (KNN) algorithm?

(a) It is a parametric model.

(b) It learns a nonlinear decision boundary between classes.

(c) It requires a separate training phase and testing phase for prediction.

(d) It typically requires longer training compared to other ML algorithms.

**Correct answers:** (b)

## Problem 8

Which of the following statements about the first principal component is true?

(a) If we add Gaussian noise to a feature in the input matrix $X$, the first principal component remains unchanged.

(b) The first principal component is equivalent to the eigenvector corresponding to the largest eigenvalue of the input matrix $X$.

(c) The first principal component is the vector direction which maximizes the variance of the input.

(d) The first principal component corresponds to the most influential feature for prediction.

**Correct answers:** (c)

## Problem 9

Leave-one-out cross-validation (LOOCV) is a special case of k-fold cross-validation where:

(a) The training set contains all but one sample, and the remaining sample is used for testing.

(b) The training set contains only one sample, and the remaining sample is used for testing.

(c) The training set contains exactly one sample from each class, and the remaining samples are used for testing.

(d) The training set contains one sample from each fold, and the remaining folds are used for testing.

**Correct answers:** (a)

## Problem 10

Which of the following statements accurately compare or contrast bootstrapping and cross-validation? Select all that apply.

(a) Bootstrapping and cross-validation both train models on subsets of the training data.

(b) In cross-validation, there is no overlap between the subsets each model trains on.

(c) Bootstrapping and cross-validation are both methods to estimate prediction error.

(d) In bootstrapping, each model is trained on the same number of data points as the original training set, unlike cross-validation.

(e) In cross-validation, each learned model is evaluated on non-overlapping subsections of the original training set, unlike bootstrapping.

**Correct answers:** (a), (c), (d), (e)

**Explanation:** The answer is (A), (C), (D), (E). (B) is not true for k-fold cross validation for any $k > 2$.

## Problem 11

Which of the following are true about a twice-differentiable function $f: \mathbb{R}^d \to \mathbb{R}$? Select all that apply.

(a) $f$ is convex if $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$ for all $x, y$ in the domain of $f$ and $\lambda \in [0, 1]$.

(b) $f$ is convex if $\nabla^2 f(x) \ge 0$ for all $x$ in the domain of $f$.

(c) $f$ is convex if the set $\{(x, t) \in \mathbb{R}^{d+1} : f(x) \le t\}$ is convex.

**Correct answers:** (b), (c)

## Problem 12

Which of the following statements about random forests are true? Select all that apply.

(a) Random forests reduce overfitting by aggregating predictions from multiple trees.

(b) Random forests reduce overfitting by having each tree in the forest use a subset of all the data features.

(c) Random forests can handle a larger number of features compared to individual decision trees.

(d) Random forests provide better interpretability and understanding of the underlying relationships in the data than individual decision trees.

**Correct answers:** (a), (b)

## Problem 13

Consider a training dataset with samples $(x_i, y_i)$, where $x_i \in \mathbb{R}^d$ and $y_i \in \{0,1\}$. Suppose that $P_X$ is supported everywhere in $\mathbb{R}^d$ and $P(Y=1 | X=x)$ is smooth everywhere. Which of the following statements is true about 1-nearest neighbor classification as the number of training samples $n \rightarrow \infty$?

(a) The error of 1-NN classification approaches infinity.

(b) The error of 1-NN classification is at most twice the Bayes error rate.

(c) The error of 1-NN classification is at most the Bayes error rate.

(d) The error of 1-NN classification approaches zero.

**Correct answers:** (b)

## Problem 14

Which of the following statements about Pooling layers in convolutional neural networks (CNNs) are true? Select all that apply.

(a) A $2 \times 2$ pooling layer has 4 parameters.

(b) Pooling layers never change the height and width of the output image.

(c) For a max-pooling layer, the gradients with respect to some inputs will always be zero.

(d) Pooling layers do not change the depth of the output image.

**Correct answers:** (c), (d)

## Problem 15

Which of the following statements about SVMs are true? Select all that apply.

(a) SVMs are only applicable to binary classification problems.

(b) SVMs cannot be applied to non-linearly separable data.

(c) SVMs are a form of supervised learning.

(d) SVMs are primarily used for regression tasks.

**Correct answers:** (a), (c)

## Problem 16

In Gaussian mixture models (GMMs), which of the following statements is false?

(a) GMMs assume that the data points within each component follow a Gaussian distribution.

(b) GMMs can be used for clustering.

(c) The number of components in a GMM must be equal to the number of features in the dataset.

**Correct answers:** (c)

## Problem 17

True/False: If $X$ is a matrix in $\mathbb{R}^{n \times m}$, $X^T X$ is always invertible.

(a) True

(b) False

**Correct answers:** (b)

## Problem 18

Consider the dataset pictured below. The features of each datapoint are given by its position. So the datapoint $(0,1)$ appears at position $(0,1)$. The ground truth label of the datapoint is given by its shape, either a circle or square. You have a test set of datapoints, shown with no fill, and a train set of data, shown with a grey fill.

<img src="./datapoints.png" width="500px">

**Dataset Visualization:**
A 2D scatter plot with horizontal and vertical axes intersecting at the origin.
- **Top-left quadrant:** Contains a grey-filled square labeled "Train" and a white-filled circle labeled "Test".
- **Top-right quadrant:** Contains a grey-filled circle labeled "Train" and a white-filled circle labeled "Test".
- **Bottom-left quadrant:** Contains a grey-filled circle labeled "Train" and a white-filled circle labeled "Test".
- **Bottom-right quadrant:** Contains a grey-filled circle labeled "Train" and a white-filled circle labeled "Test".

True/False: KNN with $K = 1$ has higher train accuracy than with $K = 4$.

(a) True

(b) False

**Correct answers:** (a)

**Explanation:** (This question was thrown out during the Spring 2023 exam.)

## Problem 19

True/False: Consider the dataset from the previous problem. KNN with $K = 1$ has higher test accuracy than with $K = 4$.

(a) True

(b) False

**Correct answers:** (b)

## Problem 20

Consider two neural networks, A and B, trained on 100x100 images to predict 5 classes.
- Neural network A consists of a single linear layer followed by a softmax output activation.
- Neural network B consists of a sequence of layers with dimensions 128, 512, and 32, respectively, followed by a softmax output activation.

Both networks are trained using an identical procedure (e.g., batch size, learning rate, epochs, etc.), and neither contains hidden activations.

(a) A will outperform B

(b) B will outperform A

(c) A and B will perform roughly the same.

**Correct answers:** (c)

**Explanation:** NN B has no hidden activations, thus making it virtually identical to A.

## Problem 21

The probability of seeing data $D$ from a Gaussian distribution is given by:

$P(D|\mu, \sigma) = \left(\frac{1}{\sigma\sqrt{2\pi}}\right)^n \prod_{i=1}^n e^{-\frac{(x_i-\mu)^2}{2\sigma^2}}$

Which of the following statements are true about the MLEs $\hat{\mu}_{MLE}$ and $\hat{\sigma}^2_{MLE}$ from this distribution? Select all that apply.

(a) $\hat{\mu}_{MLE}$ is dependent on $\hat{\sigma}^2_{MLE}$

(b) $\hat{\sigma}^2_{MLE}$ is dependent on $\hat{\mu}_{MLE}$

(c) $\hat{\mu}_{MLE}$ is a biased estimator

(d) $\hat{\sigma}^2_{MLE}$ is a biased estimator

**Correct answers:** (b), (d)

## Problem 22

True/False: The bootstrap method samples a dataset with replacement.

(a) True

(b) False

**Correct answers:** (a)

## Problem 23

What does the PyTorch optimizer's `step()` function do when training neural networks?

(a) Adjust the network's weights based on the gradients

(b) Randomly initializing the network's weights.

(c) Sets all the network's gradients to zero to prepare it for backpropagation

(d) Compute the gradients of the network based on the error between predicted and actual outputs.

**Correct answers:** (a)

## Problem 24

Below is a simple computation graph with inputs $x$ and $y$ with an initial computation of $z = xy$ before the unknown path to final loss $L$. A forward propagation pass has been completed with values $x = 3$ and $y = 4$, and the upstream gradient is given as $\partial L/\partial z = 5$. Complete the backpropagation pass by filling in the scalar answers to boxes $\partial L/\partial x$ and $\partial L/\partial y$.

<img src="./computation_graph.png" width="500px">

**Computation Graph:**
- Input $x = 3$ and $y = 4$
- Computation: $z = xy = 12$
- Upstream gradient: $\partial L/\partial z = 5$
- Find: $\partial L/\partial x$ and $\partial L/\partial y$

**Correct answers:** $\partial L/\partial x = 20$ and $\partial L/\partial y = 15$

**Explanation:** $\partial L/\partial x = 20$ and $\partial L/\partial y = 15$

## Problem 25

What are some ways to reduce overfitting in a neural network?

**Explanation:** Several methods to reduce overfitting include:
- Training on more data (or augmenting the dataset).
- Applying regularization.
- Using dropout layers.
- Decreasing model complexity by removing layers or changing layer sizes.

## Problem 26

True/False: Suppose you set up and train a neural network on a classification task and converge to a final loss value. Keeping everything in the training process the exact same (e.g. learning rate, optimizer, epochs). It is possible to reach a lower loss value by ONLY changing the network initialization.

(a) True

(b) False

**Correct answers:** (a)

**Explanation:** Changing initialization can lead to a lower loss because neural networks are non-convex, meaning different initializations can converge to different (and potentially better) local minima.

## Problem 27

Why should ridge regression not be used for feature selection solely based on coefficient magnitude thresholds?

**Explanation:** Selecting features based on coefficient magnitudes alone can lead to ignoring multicollinearity. Features with small magnitudes might be highly correlated with other features, and removing them could worsen model performance.

## Problem 28

4 Deep Neural Network models are trained on a classification task, and below are the plots of their losses:

<img src="./dnn.png" width="500px">

**DenseNet-121 Loss**

(Plot showing Training Loss (blue solid line) decreasing from ~0.63 to ~0.57, and Validation Loss (red dashed line) decreasing initially from ~0.62 to ~0.60, then slightly increasing to ~0.61 over 10 epochs.)

**VGG19 Loss**

(Plot showing Training Loss (blue solid line) decreasing from ~0.65 to ~0.47, and Validation Loss (red dashed line) decreasing initially from ~0.63 to ~0.61, then increasing significantly from epoch 5 to ~0.68 by epoch 10 over 10 epochs.)

**InceptionV3 Loss**

(Plot showing Training Loss (blue solid line) decreasing from ~0.63 to ~0.57, and Validation Loss (red dashed line) fluctuating but generally staying around ~0.60, with a slight increase towards the end over 10 epochs.)

**EfficientNetv2 Loss**

(Plot showing Training Loss (blue solid line) decreasing from ~0.65 to ~0.58, and Validation Loss (red dashed line) fluctuating significantly but generally staying around ~0.60, with a peak around epoch 8 over 10 epochs.)

Based on these plots, which model is overfitting?

(a) DenseNet-121

(b) VGG19

(c) InceptionV3

(d) EfficientNetv2

**Correct answers:** (b)
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

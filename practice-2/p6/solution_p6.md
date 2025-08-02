# Problem Set 6 Solutions

## Problem 1

**One Answer** Let $L_i(w)$ be the loss of parameter $w$ corresponding to a sample point $X_i$ with label $y_i$. The update rule for stochastic gradient descent with step size $\eta$ is

(a) $w_{\text{new}} \leftarrow w - \eta \nabla_{X_i} L_i(w)$

(b) $w_{\text{new}} \leftarrow w - \eta \sum_{i=1}^n \nabla_{X_i} L_i(w)$

(c) $w_{\text{new}} \leftarrow w - \eta \nabla_w L_i(w)$

(d) $w_{\text{new}} \leftarrow w - \eta \sum_{i=1}^n \nabla_w L_i(w)$

**Correct answers:** (c)

---

## Problem 2

**One Answer** Suppose data $x_1, \dots, x_n$ is drawn from an exponential distribution $\text{exp}(\lambda)$ with PDF $p(x|\lambda) = \lambda \text{exp}(-\lambda x)$. Find the maximum likelihood for $\lambda$?

(a) $\lambda = \frac{n}{\sum_{i=1}^n x_i}$

(b) $\lambda = \sum_{i=1}^n x_i$

(c) $\lambda = \frac{\sum_{i=1}^n x_i}{n}$

(d) $\lambda = \log(\sum_{i=1}^n x_i)$

**Correct answers:** (a)

**Explanation:** $\log(p(\text{data}|\lambda)) = n\log(\lambda) - \lambda \sum_{i=1}^n x_i$

$\frac{d(\log(p(\text{data}|\lambda)))}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0$

---

## Problem 3

**One Answer** Aman and Ed built a model on their data with two regularization hyperparameters $\lambda$ and $\gamma$. They have 4 good candidate values for $\lambda$ and 3 possible values for $\gamma$, and they are wondering which $\lambda, \gamma$ pair will be the best choice. If they were to perform five-fold cross-validation, how many validation errors would they need to calculate?

(a) 12

(b) 17

(c) 24

(d) 60

**Correct answers:** (d)

---

## Problem 4

**One Answer** Which of the following is most indicative of a model overfitting?

(a) Low bias, low variance.

(b) Low bias, high variance.

(c) High bias, low variance.

**Correct answers:** (b)

---

## Problem 5

**One Answer** In k-fold cross-validation, what is the primary advantage of setting k to a higher value (e.g., k=10) compared to a lower value (e.g., k=2)?

(a) It increases the accuracy of the model on unseen data.

(b) It provides a more reliable estimate of model performance.

(c) It reduces computational time.

(d) It eliminates the need for a separate test set.

**Correct answers:** (b)

---

## Problem 6

**One Answer** Two realtors are creating machine learning models to predict house costs based on house traits (i.e. house size, neighborhood, school district, etc.) trained on the same set of houses, using the same model hyperparameters. Realtor A includes 30 different housing traits in their model. Realtor B includes 5 traits in their model. Which of the following outcomes is most likely?

(a) Realtor B's model has higher variance and lower bias than Realtor A's model.

(b) Realtor A's model has higher variance than Realtor B's model and without additional information, we cannot know which model has a higher bias.

(c) Realtor A's model has higher variance and lower bias than Realtor B's model.

(d) Realtor A's model has higher variance and higher bias than Realtor B's model.

**Correct answers:** (b)

---

## Problem 7

**Select All** Suppose we have $N$ data points $x_1, x_2, \dots, x_N$ that $x_i \in \mathbb{R}^d$. Define $X \in \mathbb{R}^{N \times d}$ such that $X_{i,j} = (x_i)_j$, $\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$, and $\mathbf{1}_N = (1,1,\dots,1)^T \in \mathbb{R}^N$. Which of the following are true about principal components analysis (PCA)?

(a) The principal components are eigenvectors of the centered data matrix $X - \mathbf{1}_N \bar{x}^T$.

(b) The principal components are right singular vectors of the centered data matrix.

(c) The principal components are eigenvectors of the sample covariance matrix $\sum_{i=1}^N (x_i - \bar{x})(x_i - \bar{x})^T$.

(d) Applying a rigid rotation matrix $Q$ (i.e., $QQ^T = Q^T Q = I$) to $X$ will not change the principal components' directions.

**Correct answers:** (b), (c)

**Explanation:** (a) They are eigenvectors of $(X - \mathbf{1}_N \bar{x}^T)^T (X - \mathbf{1}_N \bar{x}^T)$. (d) The directions change by $Q$.

---

## Problem 8

**Select All** In the context of singular value decomposition (SVD) $A = U \Sigma V^T$, which of the following statements are correct?

(a) The columns of $U$ are called left singular vectors and form an orthonormal basis for the range of $A$, while the columns of $V$ are called right singular vectors and form an orthonormal basis for the range of $A^T$.

(b) For any $A$ that is real and symmetric, we have $U = V$.

(c) For a square matrix $A$, the singular values of $A$ are the absolute values of the eigenvalues of $A$.

(d) Singular values are always non-negative real numbers.

**Correct answers:** (a), (d)

**Explanation:** (c) Consider $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$, then eigenvalues are $\lambda_1 = \lambda_2 = 1$. However, $AA^T = \begin{bmatrix} 2 & 1 \\ 1 & 1 \end{bmatrix}$, then singular values are $\sigma_{1,2} = \frac{\sqrt{3 \pm \sqrt{5}}}{2}$.

---

## Problem 9

**Select All** Which of the following statements about matrix completion are correct?

(a) It may not perform well when the real-world data is not inherently low-rank or when the pattern of missing observations is not random.

(b) The purpose of matrix completion is to estimate missing entries in a partially observed matrix.

(c) Matrix completion is only applicable for square matrices.

**Correct answers:** (a), (b)

**Explanation:** (c) No such restriction.

---

## Problem 10

**One Answer** Consider a feature map $\phi : \mathbb{R}^2 \to \mathbb{R}^4$ defined as:

$\phi \left( \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} \right) = \begin{bmatrix} x_1^2 \\ x_2^2 \\ x_1 x_2 \\ x_2 x_1 \end{bmatrix}$

What is the corresponding kernel function $K$ for $\phi$?

(a) $K: \mathbb{R}^2 \times \mathbb{R}^2 \to \mathbb{R}$ and $K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^\top \mathbf{x}')^2$.

(b) $K: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ and $K(x, x') = x^4 + x'^4 + 2x^2x'^2$.

(c) $K: \mathbb{R}^2 \times \mathbb{R}^2 \to \mathbb{R}$ and $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$.

(d) $K: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ and $K(x,x') = x^2 + x'^2$.

**Correct answers:** (a)

---

## Problem 11

**One Answer** In the context of kernel methods, what does the "kernel trick" refer to?

(a) Adding an extra kernel layer to the end of a neural network.

(b) A technique for explicitly computing the coordinates in a high-dimensional space.

(c) A method for computing the inner products in a high-dimensional feature space without explicitly mapping data to that space.

(d) A technique for speeding up the convergence of gradient descent.

**Correct answers:** (c)


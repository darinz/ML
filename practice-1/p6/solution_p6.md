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

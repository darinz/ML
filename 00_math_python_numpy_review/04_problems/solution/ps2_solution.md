# Problem Set #2 Solutions: Math Review

## Definitions

### Norms

Norms are incredibly useful, and they show up quite often! For any vector v that is n-dimensional, i.e. $`v \in \mathbb{R}^n`$, we have the following

(a) One-norm ($`l_1`$): $`\|v\|_1 = \sum_{i=1}^n |v_i|`$

(b) Two-norm ($`l_2`$): $`\|v\|_2 = \sqrt{v^T v} = \sqrt{\sum_{i=1}^n v_i^2}`$  

(c) $`\infty`$-norm: $`\|v\|_\infty = \max_i |v_i|`$

### Symmetric Matrices and the Quadratic Form

Let us define a matrix $`A \in \mathbb{R}^{n \times n}`$.

(a) We have that the matrix A is symmetric iff $`A = A^T`$

(b) The quadratic form is defined to be $`x^T A x`$ for any vector $`x \in \mathbb{R}^n`$. The matrix A is said to be positive semi-definite if $`x^T A x \geq 0`$

## 1. Probability Review: PDF, CDF and Expectation

The Cumulative Density Function (CDF) $`F_X : \mathbb{R} \rightarrow [0, 1]`$ of a random variable X is defined as $`P(X \leq x)`$. The Probability Density Function (PDF), $`f_X : \mathbb{R} \rightarrow \mathbb{R}_{\geq 0}`$, of the same random variable is defined as $`f_X(x) = \frac{d}{dx} F_X(x)`$.

Note that the CDF can be computed from the PDF, and vice versa; e.g. $`F_X = \int_{-\infty}^{x} f(x)dx`$.

We can use these functions to directly compute the expectation of random variables, since the expectation is defined in terms of the PDF:  
$`E(X) = \int_{-\infty}^{\infty} x \cdot f_X(x)dx`$

These functions can also be used to compute the distribution of any one-to-one transformation $`g(\cdot)`$ of the random variable:  
$`E(g(X)) = \int_{-\infty}^{\infty} g(x) \cdot f_X(x)dx`$

Note: this section focuses on the continuous case, but equivalent formulations hold in the discrete case by replacing integration with summation.

### (a)

You start on the 2nd floor of CSE1, and then make a random choice:

- With probability $`p_1`$ you run up 2 flights of stairs.  
- With probability $`p_2`$ you run up 1 flight of stairs.  
- With probability $`p_3`$ you walk down 1 flight of stairs.  

Where $`p_1 + p_2 + p_3 = 1`$

You will do two iterations of your exercise scheme (with each draw being independent). Let X be the floor you’re on at the end of your exercise routine. Recall you start on floor 2.

#### (i) Let $`Y`$ be the difference between your ending floor and your starting floor in one iteration. What is $`E[Y]`$ (in terms of $`p_1, p_2, p_3`$)?

Solution:

Recall for a random variable $`X, E[X] = \sum _i x_i \cdot  p_i`$

So, $`E[Y] = 2 \cdot p_1 + 1 \cdot p_2 + (-1) \cdot p_3`$

#### (ii) What is $`E[X]`$? (use your answer from the previous part)

Solution:

Since we start at floor 2, we can take 2 and add the difference ($`E[Y]`$) twice to get our expected floor
at the end of the routine.

$`E[X] = 2 + E[Y] + E[Y] = 2 + 2E[Y]`$

#### (iii) You change your scheme: instead of doing two independent iterations, you decide the second iteration of your regimen will just use the same random choice as your first (in particular they are no longer independent!). Does $`E[X]`$ change?

**Solution:**

No! We can say using the same choice as the first will effectively double Y, thus by linearity of expectation, $`E[X] = 2 + E[2Y] = 2 + 2E[Y]`$

**Fact 1:**  
Let $`X_{(j)}`$ denote the jth order statistic in a sample of i.i.d. random variables; that is, the jth element when the items are sorted in increasing order $`X_{(1)} \leq X_{(2)} \leq \ldots \leq X_{(n)}`$.

The PDF of $`X_{(j)}`$ is given by:

```math
f_{X_{(j)}}(x) = \frac{n!}{(n-j)!(j-1)!} [F(x)]^{j-1}[1-F(x)]^{n-j} f(x)
```

### (b)

When a sample of $`2n + 1`$ i.i.d. random variables is observed, the $`(n+1)^{st}`$ smallest is called the sample median. If a sample of size 3 from a uniform distribution over $`[0, 1]`$ is observed, find the probability that the sample median is between $`1/4`$ and $`3/4`$. Hint: use Fact 1

**Solution:**

We will use Fact 1. To apply Fact 1, we can note that $`n = 3, j = 2`$ and

$`f_X(x) = \begin{cases} 
0 & \text{if } x < 0 \\
1 & \text{if } 0 \leq x \leq 1 \\
0 & \text{if } x \geq 1
\end{cases}`$

$`F_X(x) = \begin{cases} 
0 & \text{if } x < 0 \\
x & \text{if } 0 \leq x \leq 1 \\
1 & \text{if } x \geq 1
\end{cases}`$

We can use the PDF, which we compute via the above equations to compute the probability that the median lies in the specified range:

$`P(1/4 \leq X_{(2)} \leq 3/4) = \int_{1/4}^{3/4} f_{X_{(2)}}(x)dx`$

$`= 6 \int_{1/4}^{3/4} (x)(1-x)dx`$ (Using Fact 1 with $`n = 3, j = 2`$)

$`= 6 \left[\frac{x^2}{2} - \frac{x^3}{3}\right]_{x=1/4}^{x=3/4}`$

$`= \frac{11}{16}`$

## 2. Linear Algebra Review

Let $`X \in \mathbb{R}^{m \times n}`$. $X$ may not have full rank. We explore properties about the four fundamental subspaces of $X$.

### 2.1 Summation vs Matrix form

#### (a)

Let $`w \in \mathbb{R}^n`$ and $`Y \in \mathbb{R}^m`$. Let $`x_i^T`$ denote each row in $`X`$ and $`y_i`$ in $`Y`$. Show $`\|Xw - Y\|_2^2 = \sum_{i=1}^m (x_i^T w - y_i)^2`$

**Solution:**

$`Xw - Y`$ is a vector in $`\mathbb{R}^m`$, and its $`i`$-th component is $`(x_i^T w - y_i)`$.

For any vector $`P`$:
- $`\|P\|_2 = \sqrt{\sum P_i^2}`$
- $`P^T P = P \cdot P = \sum P_i^2`$

Therefore, $`\|P\|_2^2 = \sum P_i^2 = P^T P`$

Substituting $`P = Xw - Y`$:
$`\|Xw - Y\|_2^2 = \sum_{i=1}^m (x_i^T w - y_i)^2`$

#### (b)

Let $`L(w) = \|Xw - Y\|_2^2`$. What is $`\nabla_w L(w)`$? (Hint: You can use either summation or matrix form from first sub-problem)

**Matrix Form:**

$`\nabla_w L(w) = \nabla_w \|Xw - Y\|_2^2`$

$`= \nabla_w (Xw - Y)^T (Xw - Y)`$

$`= X^T (Xw - Y) + X^T (Xw - Y)`$

$`= 2X^T (Xw - Y)`$

**Summation Form: For an element $`w_j`$**

$`\frac{\partial L(w)}{\partial w_j} = \frac{\partial}{\partial w_j} \sum_{i=1}^m (x_i^T w - y_i)^2`$

$`= \frac{\partial}{\partial w_j} \sum_{i=1}^m \left( \left(\sum_{k=1}^n x_{ik} w_k\right) - y_i \right)^2`$

$`= \sum_{i=1}^m \frac{\partial}{\partial w_j} \left( \left(\sum_{k=1}^n x_{ik} w_k\right) - y_i \right)^2`$

$`= \sum_{i=1}^m 2 \left( \left(\sum_{k=1}^n x_{ik} w_k\right) - y_i \right) \frac{\partial}{\partial w_j} \left(\sum_{k=1}^n x_{ik} w_k\right)`$ [chain rule]

$`= \sum_{i=1}^m 2 \left( \left(\sum_{k=1}^n x_{ik} w_k\right) - y_i \right) x_{ij}`$

Note that on line 4, when evaluating $`\frac{\partial}{\partial w_j} \sum_{k=1}^n x_{ik} w_k`$, the summation can be decomposed to $`\frac{\partial}{\partial w_j} \left( \left(\sum_{k \neq j} x_{ik} w_k\right) + x_{ij} w_j \right)`$. The partial derivative of $`\sum_{k \neq j} x_{ik} w_k`$ will evaluate to 0, since it is not in terms of $`w_j`$. The partial derivative of $`x_{ij} w_j`$ will evaluate to $`x_{ij}`$.

Therefore, $`\nabla_w L(w)`$ is a column vector with $`n`$ components, where the $`j`$-th component is:
$`\sum_{i=1}^m 2 \left( \left(\sum_{k=1}^n x_{ik} w_k\right) - y_i \right) x_{ij}`$

This expanded summation form is equivalent to the matrix form $`2X^T (Xw - Y)`$.



### 2.2 Subspaces of X

Determine the rowspace, columnspace, nullspace, and rank of the matrix $`X`$.

Given:  
$`X = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}`$

**Rowspace:** the span (i.e., the set of all linear combinations) of the rows of $`X`$. Therefore, in this example, it is the subspace of vectors of the form $`(1x + 4y, 2x + 5y, 3x + 6y)`$ for all $`x`$ and $`y`$.

**Columnspace (a.k.a. Range(X)):** is the span of the columns of $`X`$. In this example, it is the subspace of vectors of the form $`(1x + 2y + 3z, 4x + 5y + 6z)`$ for all $`x, y,`$ and $`z`$.

**Nullspace (a.k.a. Null(X)):** is the set of vectors $`v`$ such that $`Xv = 0`$. In this example, the nullspace is the subspace spanned by $`(1, -2, 1)`$.

**Rank:** The matrix $`X`$ can be reduced to the form $`\begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & 2 \end{pmatrix}`$. This matrix has submatrix $`\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}`$, which has rank 2. Observe that the third column, $`\begin{pmatrix} -1 \\ 2 \end{pmatrix}`$, is in the columnspace of this first submatrix.

### 2.3 Connections between subspaces

Check the following facts regarding connections between subspaces:

**(a) The rowspace of X is the columnspace of X^T, and vice versa.**

**Solution:** The matrix $`X^T`$ is $`\begin{pmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{pmatrix}`$. The rows of $`X`$ are the columns of $`X^T`$, and vice versa.

**(b) The nullspace of X and the rowspace of X are orthogonal complements. This can be written in shorthand as Null(X) = Range(X^T)⊥. This is further equivalent to saying Range(X^T) = Null(X)⊥.**

**Solution:** A vector $`v \in \text{Null}(X)`$ if and only if $`Xv = 0`$, which is true if and only if for every row $`X_i`$ of $`X`$, $`\langle X_i, v \rangle = 0`$. This is precisely the condition that $`v`$ is perpendicular to each row of $`X`$, which is the stated claim.

**(c) The nullspace of X^T is orthogonal to the columnspace of X. This can be written in shorthand as Null(X^T) = Range(X)⊥.**

### 2.4 Facts in regression

(a) $`X^T X`$ is symmetric and PSD  
(b) $`\text{Null}(X) = \text{Null}(X^T X)`$  
(c) $`X^T X`$ is not always invertible  
(d) If X is short and wide, $`X^T X`$ not invertible  
(e) $`\text{Col}(X^T X) = \text{Row}(X)`$


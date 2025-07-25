# Problem Set #2: Math Review

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

You will do two iterations of your exercise scheme (with each draw being independent). Let X be the floor you're on at the end of your exercise routine. Recall you start on floor 2.

#### (i) Let $`Y`$ be the difference between your ending floor and your starting floor in one iteration. What is $`E[Y]`$ (in terms of $`p_1, p_2, p_3`$)?

#### (ii) What is $`E[X]`$? (use your answer from the previous part)

#### (iii) You change your scheme: instead of doing two independent iterations, you decide the second iteration of your regimen will just use the same random choice as your first (in particular they are no longer independent!). Does $`E[X]`$ change?

**Fact 1:**  
Let $`X_{(j)}`$ denote the jth order statistic in a sample of i.i.d. random variables; that is, the jth element when the items are sorted in increasing order $`X_{(1)} \leq X_{(2)} \leq \ldots \leq X_{(n)}`$.

The PDF of $`X_{(j)}`$ is given by:

```math
f_{X_{(j)}}(x) = \frac{n!}{(n-j)!(j-1)!} [F(x)]^{j-1}[1-F(x)]^{n-j} f(x)
```

### (b)

When a sample of $`2n + 1`$ i.i.d. random variables is observed, the $`(n+1)^{st}`$ smallest is called the sample median. If a sample of size 3 from a uniform distribution over $`[0, 1]`$ is observed, find the probability that the sample median is between $`1/4`$ and $`3/4`$. Hint: use Fact 1

## 2. Linear Algebra Review

Let $`X \in \mathbb{R}^{m \times n}`$. $X$ may not have full rank. We explore properties about the four fundamental subspaces of $X$.

### 2.1 Summation vs Matrix form

#### (a)

Let $`w \in \mathbb{R}^n`$ and $`Y \in \mathbb{R}^m`$. Let $`x_i^T`$ denote each row in $`X`$ and $`y_i`$ in $`Y`$. Show $`\|Xw - Y\|_2^2 = \sum_{i=1}^m (x_i^T w - y_i)^2`$

#### (b)

Let $`L(w) = \|Xw - Y\|_2^2`$. What is $`\nabla_w L(w)`$? (Hint: You can use either summation or matrix form from first sub-problem)

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

**(b) The nullspace of X and the rowspace of X are orthogonal complements. This can be written in shorthand as Null(X) = Range(X^T)⊥. This is further equivalent to saying Range(X^T) = Null(X)⊥.**

**(c) The nullspace of X^T is orthogonal to the columnspace of X. This can be written in shorthand as Null(X^T) = Range(X)⊥.**

### 2.4 Linear algebra facts for linear regression

We saw in lecture on Linear Regression that the closed form expression for linear regression without an offset involves the term $`(X^T X)^{-1}`$.

**(a)** Is it true that the matrix $`X^T X`$ is always symmetric and positive semidefinite?

**(b)** State and prove the connection between the nullspace of $`X`$ and the nullspace of $`X^T X`$. That is, your statement should look like one of the following: $`\text{Null}(X) \subseteq \text{Null}(X^T X)`$, or $`\text{Null}(X) \supseteq \text{Null}(X^T X)`$ or $`\text{Null}(X) = \text{Null}(X^T X)`$.

**(c)** Is it true that $`X^T X`$ is always invertible?

**(d)** Based on the above fact about the connection between the nullspaces of $`X`$ and $`X^T X`$ and the expression for linear regression without an offset (that we referred to two problems above), justify the use of "tall skinny" data matrix $`X`$ as opposed to a "short wide" matrix $`X`$.

**(e)** The columnspace and rowspace of $`X^T X`$ are the same, and are equal to the rowspace of $`X`$. (Hint: Use the relationship between nullspace and rowspace.)
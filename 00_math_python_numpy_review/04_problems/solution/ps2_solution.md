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

Show $`\|Xw - Y\|_2^2 = \sum_{i=1}^m (x_i^T w - y_i)^2`$

**Solution:**  
$`\|Xw - Y\|_2^2 = (Xw - Y)^T (Xw - Y) = \sum (x_i^T w - y_i)^2`$

#### (b)

What is $`\nabla_w L(w)`$ where $`L(w) = \|Xw - Y\|_2^2`$?

Matrix form:  
$`\nabla_w L(w) = 2X^T (Xw - Y)`$

Summation form:  
$`\nabla_w L(w) = \left[ \sum 2(x^T w - y) x_j \right]_{j=1}^n = 2X^T (Xw - Y)`$

### 2.2 Subspaces of X

Given:  
$`X = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}`$

- Rowspace: span of rows  
- Colspace: span of columns  
- Nullspace: solution to $`Xv = 0`$, which gives span of $`(1,-2,1)`$  
- Rank: 2

### 2.3 Connections between subspaces

(a) Rowspace of X is colspace of $`X^T`$ and vice versa  
(b) Null(X) = Row(X)$`^\perp`$  
(c) Null($`X^T`$) = Col(X)$`^\perp`$

### 2.4 Facts in regression

(a) $`X^T X`$ is symmetric and PSD  
(b) $`\text{Null}(X) = \text{Null}(X^T X)`$  
(c) $`X^T X`$ is not always invertible  
(d) If X is short and wide, $`X^T X`$ not invertible  
(e) $`\text{Col}(X^T X) = \text{Row}(X)`$

---

## Topics Covered

- Probability Review: PDF, CDF and Expectation  
- Linear Algebra Review  
  - Summation vs Matrix Form  
  - Subspaces of X  
  - Subspace Connections  
  - Regression and $`X^T X`$
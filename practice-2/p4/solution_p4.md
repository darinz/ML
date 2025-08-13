# Practice 2 Problem 4 Solutions

## Problem 1

Both forward and backward passes are a part of the backpropagation algorithm.

(a) True

(b) False

**Solution:** The solution is (a).

**Explanation:**

The correct answer is **(a) - True**. Here's the detailed explanation:

**Backpropagation Algorithm Components:**

**1. Forward Pass:**
- Computes the output of the network for given inputs
- Propagates activations from input layer to output layer
- Calculates the loss/error at the output layer

**2. Backward Pass:**
- Computes gradients of the loss with respect to all parameters
- Propagates gradients backward from output layer to input layer
- Uses the chain rule to efficiently compute all partial derivatives

**Mathematical Framework:**

**Forward Pass:**
For a neural network with $L$ layers:
$$a^{(l)} = f^{(l)}(W^{(l)}a^{(l-1)} + b^{(l)})$$

where:
- $a^{(l)}$ is the activation at layer $l$
- $W^{(l)}$ and $b^{(l)}$ are weights and biases
- $f^{(l)}$ is the activation function

**Backward Pass:**
$$\delta^{(l)} = \frac{\partial L}{\partial a^{(l)}} = \frac{\partial L}{\partial a^{(l+1)}} \frac{\partial a^{(l+1)}}{\partial a^{(l)}}$$

**Why Both Are Essential:**

1. **Forward Pass Purpose:**
   - Computes predictions: $\hat{y} = f(x; \theta)$
   - Calculates loss: $L(\hat{y}, y)$
   - Stores intermediate activations for gradient computation

2. **Backward Pass Purpose:**
   - Computes gradients: $\nabla_\theta L$
   - Updates parameters: $\theta \leftarrow \theta - \alpha \nabla_\theta L$
   - Enables learning through gradient descent

**Algorithm Flow:**
```
1. Forward Pass:
   - Input: x
   - Compute: a^(1), a^(2), ..., a^(L)
   - Output: ŷ and loss L

2. Backward Pass:
   - Input: ∂L/∂ŷ
   - Compute: ∂L/∂W^(L), ∂L/∂W^(L-1), ..., ∂L/∂W^(1)
   - Output: All gradients

3. Parameter Update:
   - Update: W^(l) ← W^(l) - α ∂L/∂W^(l)
```

**Historical Context:**
The term "backpropagation" originally referred to the backward pass, but modern usage includes both forward and backward passes as part of the complete algorithm.

**Conclusion:**
Both forward and backward passes are essential components of the backpropagation algorithm, making **(a) True** the correct answer.

## Problem 2

Which of the following is the best option that can be done to reduce a model's bias?

(a) Add more input features.

(b) Standardize/normalize the data.

(c) Add regularization.

(d) Collect more data.

**Solution:** The solution is (a).

**Explanation:**

The correct answer is **(a) - Add more input features**. Here's the detailed explanation:

**Understanding Model Bias:**

**Definition of Bias:**
Bias is the difference between the expected prediction of our model and the true value:
$$\text{Bias} = \mathbb{E}[\hat{f}(x)] - f(x)$$

where $\hat{f}(x)$ is our model's prediction and $f(x)$ is the true underlying function.

**Why Adding Features Reduces Bias:**

**1. Increased Model Capacity:**
- More features provide more information to the model
- The model can capture more complex patterns in the data
- Higher-dimensional feature space allows for more sophisticated decision boundaries

**2. Mathematical Intuition:**
- With more features, the model can represent more complex functions
- The hypothesis space becomes larger and more expressive
- The model can better approximate the true underlying relationship

**3. Example:**
Consider predicting house prices:
- **Low-dimensional model**: Only square footage → limited predictive power
- **High-dimensional model**: Square footage + bedrooms + bathrooms + location + age + condition → much better predictions

**Why Other Options Don't Reduce Bias:**

**Option (b) - Standardize/normalize the data:**
- **Effect**: Changes the scale of features, not the model's capacity
- **Impact on Bias**: Minimal effect on bias
- **Purpose**: Helps with optimization and numerical stability

**Option (c) - Add regularization:**
- **Effect**: Increases bias by constraining the model
- **Impact on Bias**: **Increases bias** (this is the trade-off for reducing variance)
- **Purpose**: Prevents overfitting by making the model simpler

**Option (d) - Collect more data:**
- **Effect**: Reduces variance, not bias
- **Impact on Bias**: Minimal effect on bias
- **Purpose**: Makes estimates more stable and reliable

**Bias-Variance Tradeoff:**

The relationship can be expressed as:
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Adding Features:**
- **Bias**: Decreases (model becomes more flexible)
- **Variance**: May increase (risk of overfitting)
- **Total Error**: May decrease if bias reduction outweighs variance increase

**Practical Considerations:**

**When to Add Features:**
- Model is underfitting (high bias, low variance)
- Domain knowledge suggests important features are missing
- Feature engineering can create informative derived features

**When Not to Add Features:**
- Model is already overfitting (low bias, high variance)
- Features are highly correlated (multicollinearity)
- Computational cost is prohibitive

**Feature Engineering Examples:**
- **Polynomial features**: $x^2, x^3, xy$
- **Interaction terms**: $x_1 \times x_2$
- **Domain-specific features**: Day of week, season, etc.
- **Transformed features**: $\log(x), \sqrt{x}$

**Conclusion:**
Adding more input features is the most direct way to reduce model bias by increasing the model's capacity to capture complex patterns in the data.

## Problem 3

Draw the maximum margin separating boundary between the hollow and filled points.

<img src="img/q3.png" width="450px">

A Cartesian coordinate system is shown with an x-axis ranging from 0 to 4 and a y-axis ranging from 0 to 4. Grid lines are present at 0.5 unit intervals.

There are two types of points plotted:

**Hollow points (circles):**
- $(1, 2)$
- $(2, 3)$
- $(2, 4)$
- $(3, 3)$
- $(4, 3.5)$

**Filled points (solid dots):**
- $(0, 0)$
- $(1, 0.25)$
- $(2, 0.5)$
- $(2, 1)$
- $(3, 0.5)$

**Explanation:** The solution for part 2 is (red = actual, purple=acceptable):

<img src="img/q3_answer.png" width="350px">

A second Cartesian coordinate system is shown, identical in scale and points to the first.

All the hollow and filled points are plotted as described above.

A red line, representing the "actual" maximum margin separating boundary, is drawn. This line appears to pass through approximately $(0, 0.5)$, $(1, 1)$, $(2, 1.5)$, $(3, 2)$, and $(4, 2.5)$. The equation of this line can be approximated as $y = 0.5x + 0.5$.

A purple shaded band, representing the "acceptable" region for the separating boundary, surrounds the red line. The lower boundary of this purple band appears to pass through approximately $(0, 0.25)$, $(1, 0.75)$, $(2, 1.25)$, $(3, 1.75)$, and $(4, 2.25)$, which can be approximated as $y = 0.5x + 0.25$. The upper boundary of the purple band appears to pass through approximately $(0, 0.75)$, $(1, 1.25)$, $(2, 1.75)$, $(3, 2.25)$, and $(4, 2.75)$, which can be approximated as $y = 0.5x + 0.75$. The red line is centered within this purple band.

The hollow points are all above the upper boundary of the purple band, and the filled points are all below the lower boundary of the purple band, indicating a clear separation.

## Problem 4

Fix a kernel $K$ and corresponding feature map $\phi$. True/False: One can train and evaluate a kernelized SVM (with this kernel) in polynomial time only if $\phi(x)$ runs in polynomial time for every $x$.

(a) True

(b) False

**Extra credit:** explain your answer.

**Correct answers:** (b)

**Explanation:**

The correct answer is **(b) - False**. Here's the detailed explanation:

**Kernelized SVM Training and Evaluation:**

**Key Insight:**
You can train and evaluate a kernelized SVM in polynomial time **even if** $\phi(x)$ runs in exponential time or is computationally expensive.

**Why This Is Possible:**

**1. The Kernel Trick:**
- Kernelized SVM never explicitly computes $\phi(x)$
- Instead, it works directly with the kernel function $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$
- The kernel function can be computed efficiently without computing $\phi(x)$

**2. Mathematical Framework:**

**Primal Form (with explicit feature mapping):**
$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^n \xi_i$$
$$\text{subject to } y_i(w^T \phi(x_i) + b) \geq 1 - \xi_i$$

**Dual Form (using kernel trick):**
$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
$$\text{subject to } 0 \leq \alpha_i \leq C, \sum_{i=1}^n \alpha_i y_i = 0$$

**3. Computational Complexity:**

**Training:**
- **With kernel trick**: $O(n^3)$ for solving the quadratic programming problem
- **Without kernel trick**: $O(d^3)$ where $d$ is the dimension of $\phi(x)$
- If $d$ is exponential, the primal form becomes intractable

**Prediction:**
- **With kernel trick**: $O(n_s)$ where $n_s$ is the number of support vectors
- **Without kernel trick**: $O(d)$ for computing $w^T \phi(x)$

**4. Examples of Efficient Kernels:**

**Polynomial Kernel:**
$$K(x_i, x_j) = (x_i^T x_j + c)^d$$
- **Feature mapping**: $\phi(x)$ has $\binom{d + n}{d}$ dimensions (exponential in $d$)
- **Kernel computation**: $O(n)$ time
- **SVM training**: $O(n^3)$ time (independent of $d$)

**RBF Kernel:**
$$K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$$
- **Feature mapping**: $\phi(x)$ is infinite-dimensional
- **Kernel computation**: $O(n)$ time
- **SVM training**: $O(n^3)$ time

**5. Why the Statement is False:**

The statement claims that polynomial-time training/evaluation requires polynomial-time feature mapping. However:

- **Training**: Uses dual form with kernel matrix, complexity $O(n^3)$
- **Evaluation**: Uses kernel function directly, complexity $O(n_s)$
- **Feature mapping**: Never computed explicitly

**6. Practical Implications:**

**Advantages of Kernel Trick:**
- Can work with infinite-dimensional feature spaces
- Computationally efficient for many kernels
- Enables non-linear decision boundaries

**Limitations:**
- Memory requirements: $O(n^2)$ for kernel matrix
- Training time: $O(n^3)$ becomes prohibitive for large datasets
- Need to choose appropriate kernel function

**Conclusion:**
The kernel trick allows polynomial-time training and evaluation of kernelized SVMs even when the feature mapping $\phi(x)$ is computationally expensive or infinite-dimensional. Therefore, the statement is **False**.

## Problem 5

Consider a data matrix $X \in \mathbb{R}^{n \times d}$. What is the smallest upper bound on $\operatorname{rank}(X)$ which holds for every $X$?

**Answer:** $\operatorname{rank}(X) \le \min(n, d)$

**Explanation:**

The answer is **$\operatorname{rank}(X) \le \min(n, d)$**. Here's the detailed explanation:

**Matrix Rank Fundamentals:**

**Definition of Rank:**
The rank of a matrix $X$ is the maximum number of linearly independent rows (or columns) in the matrix.

**Key Properties:**
1. **Row rank = Column rank**: For any matrix, the number of linearly independent rows equals the number of linearly independent columns
2. **Rank bounds**: The rank cannot exceed the number of rows or columns

**Mathematical Proof:**

**1. Row Rank Bound:**
- The rank cannot exceed the number of rows $n$
- If rank $> n$, we would have more than $n$ linearly independent rows, which is impossible

**2. Column Rank Bound:**
- The rank cannot exceed the number of columns $d$
- If rank $> d$, we would have more than $d$ linearly independent columns, which is impossible

**3. Combined Bound:**
Since rank $\leq n$ AND rank $\leq d$, we have:
$$\operatorname{rank}(X) \leq \min(n, d)$$

**Examples:**

**Case 1: $n < d$ (more columns than rows)**
- Example: $X \in \mathbb{R}^{3 \times 5}$
- Maximum possible rank: $\min(3, 5) = 3$
- The rank cannot exceed 3 because there are only 3 rows

**Case 2: $d < n$ (more rows than columns)**
- Example: $X \in \mathbb{R}^{5 \times 3}$
- Maximum possible rank: $\min(5, 3) = 3$
- The rank cannot exceed 3 because there are only 3 columns

**Case 3: $n = d$ (square matrix)**
- Example: $X \in \mathbb{R}^{4 \times 4}$
- Maximum possible rank: $\min(4, 4) = 4$
- The rank cannot exceed 4

**Why This is the Smallest Upper Bound:**

**1. Tightness:**
- For any $n$ and $d$, there exist matrices that achieve rank $= \min(n, d)$
- Example: Identity matrix padded with zeros

**2. No Smaller Bound:**
- If we claimed rank $\leq k$ where $k < \min(n, d)$, we would be wrong
- There exist matrices with rank $= \min(n, d)$

**3. Universality:**
- This bound holds for **every** matrix $X \in \mathbb{R}^{n \times d}$
- It doesn't depend on the specific values in the matrix

**Mathematical Verification:**

**Full Rank Matrix:**
Consider a matrix with $\min(n, d)$ linearly independent rows/columns:
- If $n \leq d$: Take first $n$ columns of identity matrix, pad with zeros
- If $d \leq n$: Take first $d$ rows of identity matrix, pad with zeros
- This matrix has rank $= \min(n, d)$

**Rank Deficiency:**
Any matrix with fewer than $\min(n, d)$ linearly independent rows/columns will have rank $< \min(n, d)$

**Practical Implications:**

**1. Dimensionality Reduction:**
- PCA can reduce dimensions to at most $\min(n, d)$
- SVD produces at most $\min(n, d)$ singular values

**2. Linear Independence:**
- Maximum number of linearly independent features: $\min(n, d)$
- Maximum number of linearly independent samples: $\min(n, d)$

**3. Computational Efficiency:**
- Matrix operations are most efficient when rank is close to $\min(n, d)$
- Low-rank matrices can be compressed efficiently

**Conclusion:**
The smallest upper bound on the rank of any matrix $X \in \mathbb{R}^{n \times d}$ is **$\operatorname{rank}(X) \leq \min(n, d)$**.

## Problem 6

Consider a kernel matrix $P$ that is given by $P_{ij} = \langle\phi(x_i), \phi(x_j)\rangle$ for a kernel map $\phi$, inner product $\langle\cdot, \cdot\rangle$, and data samples $x_i, x_j \in \mathbb{R}^d$. Write the closed-form solution for the $\hat{\alpha}$ that minimizes the loss function $L(\alpha) = \|y - P\alpha\|_2^2 + \lambda\alpha^T P\alpha$.

**Answer:** $\hat{\alpha} = (P + \lambda I)^{-1}y$

**Explanation:**

The answer is **$\hat{\alpha} = (P + \lambda I)^{-1}y$**. Here's the detailed derivation:

**Problem Setup:**
We want to minimize the loss function:
$$L(\alpha) = ||y - P\alpha||_2^2 + \lambda\alpha^T P\alpha$$

where:
- $P_{ij} = \langle\phi(x_i), \phi(x_j)\rangle$ is the kernel matrix
- $y$ is the target vector
- $\lambda$ is the regularization parameter
- $\alpha$ is the coefficient vector

**Mathematical Derivation:**

**Step 1: Expand the Loss Function**
$$L(\alpha) = ||y - P\alpha||_2^2 + \lambda\alpha^T P\alpha$$
$$L(\alpha) = (y - P\alpha)^T(y - P\alpha) + \lambda\alpha^T P\alpha$$
$$L(\alpha) = y^T y - y^T P\alpha - \alpha^T P^T y + \alpha^T P^T P\alpha + \lambda\alpha^T P\alpha$$

**Step 2: Simplify Using Kernel Matrix Properties**
Since $P$ is symmetric (kernel matrix), $P^T = P$:
$$L(\alpha) = y^T y - 2y^T P\alpha + \alpha^T P^2\alpha + \lambda\alpha^T P\alpha$$
$$L(\alpha) = y^T y - 2y^T P\alpha + \alpha^T(P^2 + \lambda P)\alpha$$

**Step 3: Take the Gradient**
$$\nabla_\alpha L(\alpha) = -2P^T y + 2(P^2 + \lambda P)\alpha$$
$$\nabla_\alpha L(\alpha) = -2P y + 2P(P + \lambda I)\alpha$$

**Step 4: Set Gradient to Zero**
$$-2P y + 2P(P + \lambda I)\alpha = 0$$
$$P y = P(P + \lambda I)\alpha$$

**Step 5: Solve for $\alpha$**
Assuming $P$ is invertible (which it is for positive definite kernels):
$$y = (P + \lambda I)\alpha$$
$$\alpha = (P + \lambda I)^{-1}y$$

**Verification:**

**1. Check that this is a minimum:**
The Hessian is:
$$\nabla_\alpha^2 L(\alpha) = 2P(P + \lambda I)$$

Since $P$ is positive semi-definite and $\lambda > 0$, this is positive definite, confirming we have a minimum.

**2. Alternative Derivation Using Matrix Calculus:**

**Vector Calculus Rules:**
- $\nabla_x ||Ax - b||_2^2 = 2A^T(Ax - b)$
- $\nabla_x x^T Ax = (A + A^T)x$

**Applying to Our Problem:**
$$\nabla_\alpha L(\alpha) = \nabla_\alpha ||P\alpha - y||_2^2 + \nabla_\alpha \lambda\alpha^T P\alpha$$
$$\nabla_\alpha L(\alpha) = 2P^T(P\alpha - y) + \lambda(P + P^T)\alpha$$
$$\nabla_\alpha L(\alpha) = 2P(P\alpha - y) + 2\lambda P\alpha$$
$$\nabla_\alpha L(\alpha) = 2P^2\alpha - 2P y + 2\lambda P\alpha$$
$$\nabla_\alpha L(\alpha) = 2P(P + \lambda I)\alpha - 2P y$$

Setting to zero:
$$2P(P + \lambda I)\alpha = 2P y$$
$$(P + \lambda I)\alpha = y$$
$$\alpha = (P + \lambda I)^{-1}y$$

**Interpretation:**

**1. Ridge Regression Analogy:**
This is similar to ridge regression in the primal space:
- **Primal**: $\hat{w} = (X^T X + \lambda I)^{-1}X^T y$
- **Dual**: $\hat{\alpha} = (P + \lambda I)^{-1}y$

**2. Regularization Effect:**
- The $\lambda I$ term adds regularization
- As $\lambda \to \infty$, $\alpha \to 0$ (strong regularization)
- As $\lambda \to 0$, $\alpha \to P^{-1}y$ (no regularization)

**3. Computational Complexity:**
- **Training**: $O(n^3)$ for matrix inversion
- **Prediction**: $O(n)$ for computing $\sum_{i=1}^n \alpha_i K(x_i, x_{\text{new}})$

**Conclusion:**
The closed-form solution is **$\hat{\alpha} = (P + \lambda I)^{-1}y$**.

## Problem 7

You have a batch of size $N$ $256 \times 256$ RGB images as your input. The input tensor your neural network has the shape $(N, 3, 256, 256)$. You pass your input through a convolutional layer like below:

`Conv2d(in_channels=3, out_channels=28, kernel_size=9, stride=1, padding=1)`

What is the shape of your output tensor?

Answer: (____, ____, ____, ____)

**Explanation:**

The answer is **$(N, 28, 250, 250)$**. Here's the detailed explanation:

**Understanding Convolutional Layer Output Dimensions:**

**Input Tensor Shape:**
- **Batch size**: $N$ (number of images)
- **Channels**: $3$ (RGB: Red, Green, Blue)
- **Height**: $256$ pixels
- **Width**: $256$ pixels
- **Shape**: $(N, 3, 256, 256)$

**Convolutional Layer Parameters:**
- **Input channels**: $3$ (matches input)
- **Output channels**: $28$ (number of filters)
- **Kernel size**: $9 \times 9$
- **Stride**: $1$ (step size)
- **Padding**: $1$ (zero padding)

**Output Dimension Calculation:**

**Formula for Output Size:**
$$\text{Output Size} = \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} + 1$$

**Height Calculation:**
$$\text{Output Height} = \frac{256 - 9 + 2 \times 1}{1} + 1 = \frac{256 - 9 + 2}{1} + 1 = \frac{249}{1} + 1 = 250$$

**Width Calculation:**
$$\text{Output Width} = \frac{256 - 9 + 2 \times 1}{1} + 1 = \frac{256 - 9 + 2}{1} + 1 = \frac{249}{1} + 1 = 250$$

**Step-by-Step Verification:**

**Step 1: Input Dimensions**
- Height: $256$ pixels
- Width: $256$ pixels
- Channels: $3$ (RGB)

**Step 2: Kernel Dimensions**
- Kernel size: $9 \times 9$
- This means the kernel covers $9$ pixels in both height and width

**Step 3: Padding Effect**
- Padding: $1$ adds $1$ pixel on each side
- Effective input height: $256 + 2 = 258$
- Effective input width: $256 + 2 = 258$

**Step 4: Convolution Operation**
- Kernel slides over the padded input
- Number of valid positions: $258 - 9 + 1 = 250$
- This applies to both height and width

**Step 5: Output Shape**
- **Batch size**: $N$ (unchanged)
- **Channels**: $28$ (number of output filters)
- **Height**: $250$
- **Width**: $250$
- **Final shape**: $(N, 28, 250, 250)$

**Mathematical Intuition:**

**Why Output Size Decreases:**
- Kernel size $9 \times 9$ means each output pixel depends on a $9 \times 9$ region
- Without padding, output would be $256 - 9 + 1 = 248$
- With padding $1$, we add $2$ pixels total, so output is $248 + 2 = 250$

**Padding Purpose:**
- **No padding**: Output size decreases with each convolution layer
- **With padding**: Can maintain or control output size
- **Full padding**: Output size equals input size

**Visual Example:**

**Input Image (256×256):**
```
┌─────────────────────────────────────┐
│                                     │
│          256 × 256 pixels           │
│                                     │
└─────────────────────────────────────┘
```

**After Padding (258×258):**
```
┌─────────────────────────────────────┐
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │
│ 0 ┌─────────────────────────────┐ 0 │
│ 0 │      256 × 256 pixels      │ 0 │
│ 0 └─────────────────────────────┘ 0 │
│ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 │
└─────────────────────────────────────┘
```

**Kernel (9×9):**
```
┌─────────────────┐
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ │
└─────────────────┘
```

**Output (250×250):**
```
┌─────────────────────────────────────┐
│                                     │
│          250 × 250 pixels           │
│                                     │
└─────────────────────────────────────┘
```

**General Formula:**

**For any convolutional layer:**
$$\text{Output Size} = \frac{\text{Input Size} - \text{Kernel Size} + 2 \times \text{Padding}}{\text{Stride}} + 1$$

**Common Cases:**
- **Same padding**: Output size = Input size
- **Valid padding**: Output size = Input size - Kernel size + 1
- **Full padding**: Output size = Input size + Kernel size - 1

**Computational Considerations:**

**Memory Usage:**
- Input: $N \times 3 \times 256 \times 256 = 196,608N$ elements
- Output: $N \times 28 \times 250 \times 250 = 1,750,000N$ elements
- Memory increase: ~8.9x

**Parameters:**
- Each filter: $3 \times 9 \times 9 = 243$ parameters
- Total parameters: $28 \times 243 = 6,804$ parameters

**Conclusion:**
The output tensor shape is **$(N, 28, 250, 250)$**, where the spatial dimensions are reduced from $256 \times 256$ to $250 \times 250$ due to the $9 \times 9$ kernel, but increased from $248 \times 248$ to $250 \times 250$ due to the padding of $1$.

## Problem 8

For ridge regression, how will the bias and variance in our estimate $\hat{w}$ change as the number of training examples $N$ increases? Assume the regularization parameter $\lambda$ is fixed.

(a) $\downarrow$ bias, $\uparrow$ variance

(b) same bias, $\downarrow$ variance

(c) same bias, $\uparrow$ variance

(d) $\downarrow$ bias, $\downarrow$ variance

(e) same bias, same variance

**Correct answers:** (b)

**Explanation:**

The correct answer is **(b) - same bias, $\downarrow$ variance**. Here's the detailed explanation:

**Understanding Ridge Regression:**

**Ridge Regression Model:**
Ridge regression adds L2 regularization to linear regression:
$$\hat{w} = \arg\min_w \frac{1}{N} \sum_{i=1}^N (y_i - w^T x_i)^2 + \lambda ||w||_2^2$$

**Closed-Form Solution:**
$$\hat{w} = (X^T X + \lambda I)^{-1} X^T y$$

**Effect of Increasing Training Examples:**

**Mathematical Analysis:**

**1. Bias Analysis:**
- **Bias**: $\text{Bias} = \mathbb{E}[\hat{w}] - w_{\text{true}}$
- **With regularization**: $\mathbb{E}[\hat{w}] = (X^T X + \lambda I)^{-1} X^T X w_{\text{true}}$
- **As $N \to \infty$**: $X^T X$ dominates $\lambda I$, but bias remains due to regularization
- **Result**: Bias stays approximately the same (determined by $\lambda$)

**2. Variance Analysis:**
- **Variance**: $\text{Var}(\hat{w}) = \sigma^2 (X^T X + \lambda I)^{-1} X^T X (X^T X + \lambda I)^{-1}$
- **As $N$ increases**: $X^T X$ becomes more stable and well-conditioned
- **Effect**: Variance decreases due to more stable estimates

**Intuitive Understanding:**

**Why Bias Stays the Same:**
1. **Regularization Effect**: The $\lambda ||w||_2^2$ term introduces bias regardless of data size
2. **Shrinkage**: Ridge regression shrinks coefficients toward zero
3. **Fixed $\lambda$**: The regularization strength doesn't change with $N$

**Why Variance Decreases:**
1. **More Data**: More training examples provide more stable estimates
2. **Better Conditioning**: $X^T X$ becomes more well-conditioned with more data
3. **Reduced Noise**: Averaging over more examples reduces estimation noise

**Mathematical Verification:**

**Bias Calculation:**
$$\text{Bias} = \mathbb{E}[\hat{w}] - w_{\text{true}} = (X^T X + \lambda I)^{-1} X^T X w_{\text{true}} - w_{\text{true}}$$
$$\text{Bias} = -(\lambda I)(X^T X + \lambda I)^{-1} w_{\text{true}}$$

**Variance Calculation:**
$$\text{Var}(\hat{w}) = \sigma^2 (X^T X + \lambda I)^{-1} X^T X (X^T X + \lambda I)^{-1}$$

**As $N$ increases:**
- $X^T X$ becomes larger and more stable
- The inverse $(X^T X + \lambda I)^{-1}$ becomes smaller
- Variance decreases due to more stable estimates

**Visual Example:**

**Small Dataset ($N = 10$):**
```
Coefficient estimates: [2.1, -1.8, 3.2, 0.9, -2.3]
Variance: High (estimates vary a lot)
Bias: Moderate (due to regularization)
```

**Large Dataset ($N = 1000$):**
```
Coefficient estimates: [2.05, -1.95, 3.15, 0.95, -2.25]
Variance: Low (estimates are stable)
Bias: Same (still due to regularization)
```

**Comparison with Other Options:**

**Option (a) - $\downarrow$ bias, $\uparrow$ variance:**
- **Problem**: This describes underfitting or removing regularization
- **Issue**: More data typically doesn't increase variance
- **Result**: Incorrect for ridge regression

**Option (c) - same bias, $\uparrow$ variance:**
- **Problem**: More data should decrease variance, not increase it
- **Issue**: This would indicate poor model behavior
- **Result**: Incorrect

**Option (d) - $\downarrow$ bias, $\downarrow$ variance:**
- **Problem**: Ridge regression bias is determined by $\lambda$, not $N$
- **Issue**: Bias doesn't decrease with more data in ridge regression
- **Result**: Partially correct but wrong about bias

**Option (e) - same bias, same variance:**
- **Problem**: Variance should decrease with more data
- **Issue**: Ignores the stabilizing effect of more training examples
- **Result**: Incorrect

**Practical Implications:**

**1. Model Selection:**
- Ridge regression is robust to overfitting
- More data improves stability without changing bias
- Good choice when you have many features

**2. Hyperparameter Tuning:**
- $\lambda$ controls the bias-variance tradeoff
- Larger $\lambda$: More bias, less variance
- Smaller $\lambda$: Less bias, more variance

**3. Data Collection:**
- More data always helps reduce variance
- But doesn't eliminate regularization bias
- Consider collecting more data vs. adjusting $\lambda$

**Bias-Variance Tradeoff in Ridge Regression:**

**Mathematical Relationship:**
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Effect of $N$:**
- **Bias**: Determined by $\lambda$, not $N$
- **Variance**: Decreases with $N$
- **Total Error**: Decreases with $N$ (due to variance reduction)

**Optimal $\lambda$:**
- Depends on the true underlying relationship
- Can be found using cross-validation
- Balances bias and variance for given dataset size

**Conclusion:**
As the number of training examples $N$ increases in ridge regression with fixed $\lambda$, the **bias remains the same** (determined by regularization) while the **variance decreases** (due to more stable estimates from more data).

## Problem 9

Suppose you have a data matrix $X \in \mathbb{R}^{10,000 \times 10,000}$ where $x_{ij} \sim \text{iid } N(0, \sigma^2)$ for each $i, j \in [10,000]$ and you want to understand how many principal components are needed to have reconstruction error $\le 5/10,000$. What would be an efficient way to answer this question?

Answer: ________

**Explanation:** Accept SVD, or anything that refers to .eig/other packages. Kudos (+1)? if they also mention how to use these results (namely, look at the reconstruction error for each $d$ and pick the min $d$ with reconstruction error below the quantity. If they explain why this is the better choice (e.g, that this is likely a full-rank matrix so we'll need an overwhelming majority of our features for that level of reconstruction error), another +1. We don't accept the power method.

## Problem 10

What method can be described as a resampling method used to estimate population parameters by repeatedly sampling from a dataset?

(a) Power method

(b) Bootstrapping

(c) k-means

(d) SVD

**Correct answers:** (b)

## Problem 11

Let $A \in \mathbb{R}^{m \times m}$ and $x$ in $\mathbb{R}^m$. What is $\nabla_x x^T A x$?

Answer: $\nabla_x x^T A x = \rule{5cm}{0.15mm}$

**Explanation:** The solution is $(A + A^T)x$.

## Problem 12

What is the biggest advantage of k-fold cross-validation over Leave-one-out (LOO) cross-validation?

(a) It provides a more accurate estimation of model performance

(b) Prevents overfitting

(c) Easier to compute

(d) Minimizes impact from sample size

**Correct answers:** (c)

## Problem 13

What is the expression for logistic loss? Here $\hat{y}$ is a prediction, and $y$ is the corresponding ground truth label.

(a) $\log(1+e^{-y\hat{y}})$

(b) $-\log(1+e^{-y\hat{y}})$

(c) $1 + e^{-y\hat{y}}$

(d) $\log(1+e^{y\hat{y}})$

**Solution:** The solution is (a).

## Problem 14

Suppose that you have a convolutional neural network with the following components:
1. One 2D-convolutional layer with two 2x2 kernels, stride 2, and no zero-padding
2. A max pooling layer of size 2x2 with stride 2.
3. One 2D-convolutional layer with one 1x1 kernel, stride 1, and no zero-padding
Suppose you propagate the input below (left) through the CNN with the following kernel weights. Assume there are no bias terms.

<img src="img/q14.png" width="450px">

**Input:**
A $4 \times 4$ matrix labeled "Input":
$$
\begin{pmatrix}
1 & 3 & 0 & 3 \\
2 & 0 & 1 & 4 \\
7 & 1 & 6 & 2 \\
5 & 2 & 5 & 0
\end{pmatrix}
$$
Below the matrix, it is labeled "$4 \times 4$".

**Layer 1 Kernel 1:**
A $2 \times 2$ matrix labeled "Layer 1 Kernel 1":
$$
\begin{pmatrix}
-1 & 1 \\
-1 & 1
\end{pmatrix}
$$
Below the matrix, it is labeled "$2 \times 2$".

**Layer 1 Kernel 2:**
A $2 \times 2$ matrix labeled "Layer 1 Kernel 2":
$$
\begin{pmatrix}
1 & 1 \\
-1 & -1
\end{pmatrix}
$$
Below the matrix, it is labeled "$2 \times 2$".

**Layer 2 Kernel 1:**
A 3D block representing a kernel, with a '1' on its top face and a '1' on its bottom face. This visually implies a 1x1 kernel operating on two input channels (one '1' for each channel). Below it, it is labeled "$1 \times 2$".

What is the output of this network given the current weights and input?

(a) 0

(b) 4.5

(c) 8

(d) 9

**Correct answers:** (d)

## Problem 15

True/False: Given a set of points in a $d$-dimensional space, using PCA to reduce the dataset to $d' < d$ dimensions will **always** lead to loss of information.

(a) True

(b) False

**Correct answers:** (b)

## Problem 16

True/False: The bootstrap method can be applied to both regression and classification questions.

(a) True

(b) False

**Correct answers:** (a)

## Problem 17

Which of the following techniques can be helpful in reducing the original dimensions of input data? Select **all** that apply.

(a) L1 Regularization (LASSO)

(b) L2 Regularization (Ridge)

(c) Principal Component Analysis (PCA)

(d) $k$-means Clustering

**Correct answers:** (a), (c)


## Problem 18

True/False: Given a dataset $X$ in a $d$-dimensional space, using PCA to project $X$ onto $d_1 < d_2 < d$ dimensions leads to the $d_1$ dimensional projection to being a subspace of the $d_2$-dimensional projection.

(a) True

(b) False

**Correct answers:** (a)

## Problem 19

Shade in the region where decision boundaries that lie inside it have equal training error.

<img src="img/q19_problem.png" width="450px">

A Cartesian coordinate system is shown with an x-axis ranging from 0 to 4 and a y-axis ranging from 0 to 4. Grid lines are present at 0.5 unit intervals.

There are two types of points plotted:

**Hollow points (circles):**
- $(1, 2)$
- $(2, 3)$
- $(2, 4)$
- $(3, 3)$
- $(4, 3.5)$

**Filled points (solid dots):**
- $(0, 0)$
- $(1, 0.25)$
- $(2, 0.5)$
- $(2, 1)$
- $(3, 0.5)$

**Explanation:** The solution for part 1 is:

<img src="img/q19_solution.png" width="450px">

A Cartesian coordinate system is shown with an x-axis ranging from 0 to 4 and a y-axis ranging from 0 to 4. Grid lines are present. The same hollow and filled points as in the problem description are plotted.

A region is shaded in gray. This shaded region is bounded by two dashed lines:
- The upper dashed line passes through the points $(1, 2)$ and $(4, 3.5)$. Its equation is approximately $y = 0.5x + 1.5$.
- The lower dashed line passes through the points $(0, 0)$ and $(3, 0.5)$. Its equation is approximately $y = \frac{1}{6}x$.

The shaded region represents the area between these two dashed lines, inclusive of the lines themselves.

## Problem 20

Which of the following features could allow a logistic regression model to perfectly classify all data points in the following figure? Select all that apply.


A Cartesian coordinate system is shown with an x-axis labeled 'X' ranging from -3 to 3 and a y-axis labeled 'y' ranging from -3 to 3. Major grid lines are present at integer values on both axes, and minor grid lines are present at 0.5 unit intervals.

<img src="img/q20_problem.png" width="450px" >

There are two types of data points:
- **Crosses (x):** These points are distributed widely across the entire plot area, forming an outer region. They are present in all four quadrants.
- **Solid Circles (•):** These points are clustered tightly around the origin, primarily within the region where X is approximately between -0.5 and 0.5, and Y is approximately between -0.5 and 0.5. This cluster of solid circles forms an inner region, completely surrounded by the crosses.

(a) $|x_i|, |y_i|$

(b) $x_i + y_i, x_i - y_i$

(c) $x_i^2, y_i^2$

(d) $x_i^3, y_i^3$

**Correct answers:** (a), (c)

## Problem 21

**Extra credit:** Suppose that we have $x_1, x_2, \dots, x_{2n}$ are independent and identically distributed realizations from the Laplacian distribution, the density of which is described by

$$f(x | \theta) = \frac{1}{2}e^{-|x-\theta|}$$

Find the M.L.E of $\theta$. Note that for this problem you may find the sign function useful, the definition of which is as follows

$$\operatorname{sign}(x) = \begin{cases} +1 & x \ge 0 \\ -1 & x < 0 \end{cases}$$

**Answer:**

**Explanation:** The solution is $\hat{\theta} \in [x_n, x_{n+1}]$

## Problem 22

SVM models that use slack variables have higher bias compared to SVM models that do not use slack variables.

(a) equal

(b) lower

(c) higher

**Correct answers:** (c)

## Problem 23

The following expression for $\hat{\Theta}_{2}$ will appear twice in this exam. Consider a distribution X with unknown mean and variance $\sigma^{2}$. We define the population variance to be as follows

$\hat{\Theta}_{2}=\frac{1}{n}(\sum_{i=1}^{n}(x_{i}-\hat{\Theta}_{1})^{2})$ for $\hat{\Theta}_{1}=\frac{1}{n}\sum_{i=1}^{n}x_{i}$

What is the expected value of $\Theta_{2}$?

**Answer:**

**Explanation:** The solution is $\hat{\Theta}_{2}=(1-\frac{1}{n})\sigma^{2}$

## Problem 24

Which of the following statements about kernels is/are true? Select all that apply.

(a) A kernel feature map $\phi(x):\mathbb{R}^{d}\longrightarrow\mathbb{R}^{k}$ always maps to higher dimensional space (i.e., $k>d$.

(b) Kernel matrices depend on the size of the dataset.

(c) Kernel matrices are square.

(d) Kernel matrices are used for data dimensionality reduction.

**Correct answers:** (b), (c)

## Problem 25

Both LASSO and PCA can be used for feature selection. Which of the following statements are true? Select all that apply.

(a) LASSO selects a subset (not necessarily a strict subset) of the original features

(b) If you use the kernel trick, principal component analysis and LASSO are equivalent learning "techniques"

(c) PCA produces features that are linear combinations of the original features

(d) PCA is a supervised learning algorithm

**Correct answers:** (a), (c)

## Problem 26

Consider a dataset X where row $X_{i}$ corresponds to a complete medical record of an individual $i\in[n].$ Suppose the first column of X contains each patient's name, and no other column contains their name.

True/False: Removing the first column from X gives a dataset $X_{.,2:d}$ where no individual (row) is unique.

(a) True

(b) False

**Correct answers:** (b)

## Problem 27

True/False: The number of clusters k is a hyperparameter for Lloyd's Algorithm for k-means clustering.

(a) True

(b) False

**Correct answers:** (a)

## Problem 28

You are using Lloyd's algorithm (the algorithm described in class) to perform k-means clustering on a small dataset.

The following figure depicts the data and cluster centers for an iteration of the algorithm.

Dataset samples are denoted by markers and cluster centers are denoted by markers x.

<img src="img/q28_problem_1.png" width="350px">

Which of the following depicts the best estimate of the cluster center positions after the next single iteration of Lloyd's algorithm?

<img src="img/q28_problem_2.png" width="650px">

Hint: a single iteration refers to both update steps.

(a) Plot A

(b) Plot B

(c) Plot C

(d) Plot D

**Correct answers:** (b)

[Image 2]

[Image 3]

[Image 4]

[Image 5]

## Problem 29

Which of the following loss functions are convex? Select all that apply.

(a) 1-0 loss.

(b) Squared loss (MSE).

(c) Sigmoid loss.

(d) Logistic loss.

(e) Hinge loss.

**Correct answers:** (b), (d), (e)

## Problem 30

In neural networks, the activation functions sigmoid, ReLU, and tanh all

(a) always output values between 0 and 1.

(b) are applied only to the output units.

(c) are essential for learning non-linear decision boundaries.

(d) are needed to speed up the gradient computation during backpropagation (compared to not using activation functions at all).

**Correct answers:** (c)

## Problem 31

Consider a neural network with 8 layers trained on a dataset of 800 samples with a batch size of 10. How many forward passes through the entire network are needed to train this model for 5 epochs?

**Answer:**

**Explanation:** 400

## Problem 32

k-means refers to optimizing which of the following objectives? Here $\mu_{C(j)}$ is the mean of the cluster that $x_{j}$ belongs to. m is the number of points.

(a) $F(\mu,C)=\sum_{j=1}^{m}||\mu_{C(j)}-x_{j}||_{2}^{2}$

(b) $F(\mu,C)=min_{j=1}^{m}||\mu_{C(j)}-x_{j}||_{2}^{2}$

(c) $F(\mu,C)=\sum_{j=1}^{m}||\mu_{C(j)}-x_{j}||_{2}$

(d) $F(\mu,C)=max_{j=1}^{m}||\mu_{C(j)}-x_{j}||_{2}^{2}$

**Correct answers:** (a)

## Problem 33

Which of the following statements about choosing L1 regularization (LASSO) over L2 regularization (Ridge) are true? Select all that apply.

(a) LASSO (L1) learns model weights faster than Ridge regression (L2).

(b) L1 regularization can help us identify which features are important for a certain task.

(c) L1 regularization usually achieves lower generalization error.

(d) If the feature space is large, evaluating models trained with L1 regularization is more computationally efficient.

**Correct answers:** (b), (d)

## Problem 34

**Extra Credit:** Consider one of the "semi-fresh" datasets $\hat{X}$ generated using the bootstrap method for a dataset X, where n is large and $X_{i}\sim_{iid}\mathcal{D}$ Let $f_{X}$ be the model trained on X. $err(f_{X},\hat{X})$ is a/an

(a) unbiased estimate

(b) slightly biased upwards

(c) slightly biased downwards

of $err_{\mathcal{D}}(f_{X}).$

(d) very biased estimate (either upwards or downwards), to the point where this value by itself is not useful.

**Correct answers:** (c)

## Problem 35

Consider a nearest neighbor classifier that chooses the label for a test point to be the label of its nearest neighboring training example. What is its leave-one-out cross-validated error for the data in the following figure?

()"+" and "-" indicate labels of the points).

<img src="img/q35_problem.png" width="350px">

**Answer:**

**Explanation:** The solution is 2/5

## Problem 36

Consider the following scatter plots of a data matrix X with four data points in $\mathbb{R}^{2}.$ Choose the plot whose line represents the direction of the first principal component of $X-\mu,$ where $X\in\mathbb{R}^{n\times d}$ the vector $\mu\in\mathbb{R}^{d}$ is the featurewise mean of X.

<img src="img/q36_problem.png" width="550px">

(a) Plot 1

(b) Plot 2

(c) Plot 3

(d) Plot 4

**Correct answers:** (c)

## Problem 37

Suppose that a model finds that towns with more children tend to have higher rates of poverty compared to towns with fewer children. Upon seeing this, a local mayor suggests that children be banished from the town in order to reduce poverty. What is the flaw of this reasoning?

(a) The reasoning is correct.

(b) We cannot make policy decisions based on a machine learning model.

(c) Correlation does not imply equal causation.

**Correct answers:** (c)

## Problem 38

Consider the following neural network with weights shown in the image below. Every hidden neuron uses the ReLU activation function, and there is no activation function on the output neuron. Assume there are no bias terms. What is the output of this network with the input $x=(1,2)?$ Give a numerical answer.

<img src="img/q38_problem.png" width="450px">

**Answer:**

**Explanation:** The answer is -3.

## Problem 39

Suppose you have a data matrix $X\in\mathbb{R}^{n\times10,000}$ and you want the 3 principal components of X. What is an efficient algorithm to compute these?

**Answer:**

**Explanation:** Accept "the power method", or skinny SVD, (I'll also accept anything that refers to eig/other packages). We won't accept SVD.

## Problem 40

In PCA, the following words go together (draw lines to match the words on the left with the words on the right)

<img src="img/q40_problem.png" width="350px">

**Explanation:**

<img src="img/q40_solution.png" width="350px">

## Problem 41

The following expression for $\hat{\Theta}_{2}$ will appear twice in this exam. Consider a distribution X with unknown mean and variance $\sigma^{2}$ We define the population variance to be as follows

Is $\hat{\Theta}_{2}$ unbiased?

(a) Yes

(b) No

$\hat{\Theta}_{2}=\frac{1}{n}(\sum_{i=1}^{n}(x_{i}-\hat{\Theta}_{1})^{2})$ for $\hat{\Theta}_{1}=\frac{1}{n}\sum_{i=1}^{n}x_{i}$

**Correct answers:** (b)

## Problem 42

Which of the following shapes are convex? Select all that apply.

<img src="img/q42_problem.png" width="450px">

(a) Shape A.

(b) Shape B.

(c) Shape C.

(d) Shape D.

(e) Shape E.

**Correct answers:** (a)

## Problem 43

Given a dataset X in a d-dimensional space, using PCA to project X onto $d1 < d2 < d$ dimensions leads to the d1 dimensional projection to have higher compared to the d2-dimensional projection.

**Answer:**

**Explanation:** Reconstruction error, or average distance from the original points to their projections. Also accept mathematical notation for these.

## Problem 44

What are support vectors in an SVM without slack?

(a) The data points that don't fall into a specific classification.

(b) The most important features in the dataset.

(c) The data points on the margin of the SVM.

(d) All points within the dataset are considered support vectors.

**Correct answers:** (c)

## Problem 45

While training a neural network for a classification task, you realize that there isn't a significant change to the weights of the first few layers between iterations. What could NOT be a reason for this?

(a) The model is stuck in a local minimum.

(b) The network is very wide.

(c) The weights of the network are all zero.

(d) The learning rate is very small.

**Correct answers:** (b)

## Problem 46

Let $\eta(X)$ be an unknown function relating random variables X and Y , D be a dataset consisting of sample pairs (xi, yi) drawn iid from the probability distribution PXY , and $\hat{f}_D$ an estimator of $\eta$. Draw lines to match the expressions on the left with the words on the right.

<img src="img/q46_problem.png" width="450px">

**Explanation:**

<img src="img/q46_solution.png" width="450px">

## Problem 47

Given differentiable functions $f(x) : \mathbb{R} \to \mathbb{R}$ and $g(x) : \mathbb{R} \to \mathbb{R}$, which of the following statements is false?

(a) if f(x) is concave, then -f(x) is convex.

(b) if f(x) and g(x) are convex, then $h(x) := \max(f(x), g(x))$ is also convex.

(c) if f(x) and g(x) are convex, then $h(x) := \min(f(x), g(x))$ is also convex.

(d) f(x) can be both convex and concave on the same domain.

**Correct answers:** (c)

## Problem 48

Let A be an $n \times n$ matrix. Which of the following statements is true?

(a) If A is invertible, then $A^T$ is invertible

(b) If A is PSD, then A is invertible

(c) If A is symmetric, then A is invertible

(d) None of these answers.

**Correct answers:** (a)
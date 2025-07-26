# Problem Set 6 Solutions

## 1. Principal Component

Consider the following dataset, which is represented as three points in R². Note that in this problem we will not demean the dataset.

**Dataset Matrix:**
```
[ 1   2  ]
[ 1.5 3  ]
[ 6   12 ]
```

**Scatter Plot:**
A 2D scatter plot shows the three data points.
*   **X-axis:** Ranges from 1 to 6, with major ticks at 1, 2, 3, 4, 5, 6.
*   **Y-axis:** Ranges from 2 to 12, with major ticks at 2, 4, 6, 8, 10, 12.
*   **Plotted Points:** Three blue circular markers are visible:
    *   One at coordinates (1, 2)
    *   One at coordinates (1.5, 3)
    *   One at coordinates (6, 12)
These points appear to lie perfectly on a straight line passing through the origin.

### (a)

What is the first principal component vector, $`v_1`$?

**Solution:**
Each point has second coordinate twice the first, so every point is on the line $`y = 2x`$, or equivalently is a multiple of the vector $`[1, 2]`$.
That direction, normalized, is the first principal component, so $`v_1 = [1/\sqrt{5}, 2/\sqrt{5}] \approx [0.45, 0.89]`$.

### (b)

What is the second principal component, $`v_2`$?

**Solution:**
Since every data is in the span of the first principal component, any unit norm vector perpendicular to $`v_1`$ is an acceptable choice. One such vector is $`[-2/\sqrt{5}, 1/\sqrt{5}] \approx [-0.89, 0.45]`$.

### (c)

If we use only the first principal component to compress the dataset, what will the representation of each point be?

**Solution:**
The first point is $`\sqrt{5}v_1`$, the second one is $`1.5 \cdot \sqrt{5}v_1`$, and the third one is $`6 \cdot \sqrt{5}v_1`$.

### (d)

Will this representation be lossy, or perfectly preserve the dataset?

**Solution:**
In this particular dataset, we perfectly preserve this dataset (the points are all multiples of $`v_1`$).

**Answer the same questions for the following, slightly larger dataset:**

**Dataset (a 6x2 matrix):**
```
[ 1    1  ]
[ 1.5  1.5]
[ -2   2  ]
[ 4   -4  ]
[ 6   -6  ]
[ 2    2  ]
```

### (a)

What is the first principal component vector, $`v_1`$?

**Solution:**
Notice that every point is either a multiple of $`[1, 1]`$ or $`[1, -1]`$, so some of those must be our principal component. The norms of the multiples of $`[1, -1]`$ are much larger, so $`[1/\sqrt{2}, -1/\sqrt{2}]`$ is $`v_1`$.

### (b)

What is the second principal component, $`v_2`$?

**Solution:**
We need a vector perpendicular to $`v_1`$, which can best describe our remaining data. Since we're in two dimensions, we don't have choices after we chose the first principal component. Therefore, the second principal component is $`[1/\sqrt{2}, 1/\sqrt{2}]`$.

### (c)

If we use only the first principal component to compress the dataset, what will the representation of each point be?

**Solution:**
Data points 1, 2, and 6 are all perpendicular to $`v_1`$, so are represented as $`[0,0]`$ (i.e. $`0 \cdot v_1`$). The other points are multiples of $`v_1`$, which are $`-2\sqrt{2}v_1`$, $`4\sqrt{2}v_1`$, and $`6\sqrt{2}v_1`$, respectively.

### (d)

Will this representation be lossy, or perfectly preserve the data?

**Solution:**
The data representation is lossy. Points 1, 2, and 6 have lost information.

## 2. Using the Eigenbasis

A useful fact about symmetric $`n \times n`$ matrices is that they have a set of eigenvectors $`u_1, ..., u_n`$ that satisfy three properties:
*   $`||u_i||_2 = 1`$ (each eigenvector has a unit norm).
*   $`u_i^T u_j = 0, \forall i \neq j`$ (eigenvectors are orthogonal to each other).
*   $`u_1, ..., u_n`$ form a basis of $`\mathbb{R}^n`$.

This fact is useful because it simplifies proofs by allowing one to think about vectors in terms of their "eigenbasis" components instead of standard basis components. As a trivial example, we'll show how to calculate $`Ax`$ for a vector $`x`$ without performing direct matrix multiplication.

### (a)

Consider the matrix $`A = \begin{bmatrix} 4 & -1 \\ -1 & 4 \end{bmatrix}`$. Verify that $`u_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}`$ and $`u_2 = \begin{bmatrix} -1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}`$ are eigenvectors and meet the definitions. Find the eigenvalues associated with $`u_1`$ and $`u_2`$.

**Solution:**
*   They are eigenvectors: $`Au_1 = \begin{bmatrix} 4/\sqrt{2} - 1/\sqrt{2} \\ -1/\sqrt{2} + 4/\sqrt{2} \end{bmatrix} = 3u_1`$ and $`Au_2 = 5u_2`$ by a similar calculation.
*   They are unit norm: $`u_1^T u_1 = (1/\sqrt{2})^2 + (1/\sqrt{2})^2 = 1/2 + 1/2 = 1`$. The calculation for $`u_2`$ is similar.
*   They are orthogonal: $`u_1^T u_2 = (1/\sqrt{2})(-1/\sqrt{2}) + (1/\sqrt{2})(1/\sqrt{2}) = -1/2 + 1/2 = 0`$
*   They form a basis (since they're 2 linearly independent vectors in $`\mathbb{R}^2`$)

### (b)

Since $`\{u_1, u_2\}`$ are a basis, we can write any vector as a linear combination of them. Write $`x = \begin{bmatrix} -1/\sqrt{2} \\ 3/\sqrt{2} \end{bmatrix}`$ in this basis.

**Solution:**
$`x^T u_1 = -1/2 + 3/2 = 1`$. $`x^T u_2 = 1/2 + 3/2 = 2`$. So $`x = u_1 + 2u_2`$

### (c)

Based on the eigenvectors and eigenvalues your found in part a, diagonalize A, i.e, find matrix U and D such that $`UDU^T = A`$ where all entries of D are 0 except for the ones on the diagonal.

**Solution:**
*   $`U = \begin{bmatrix} 1/\sqrt{2} & -1/\sqrt{2} \\ 1/\sqrt{2} & 1/\sqrt{2} \end{bmatrix}`$
*   $`D = \begin{bmatrix} 3 & 0 \\ 0 & 5 \end{bmatrix}`$

### (d)

Use the decomposition and the eigenvalues you calculated in the previous parts to calculate Ax without doing matrix-vector multiplication.

**Solution:**
```math
Ax = A (u_1 + 2u_2) = Au_1 + 2Au_2 = 3u_1 + 10u_2 = \begin{bmatrix} -7/\sqrt{2} \\ 13/\sqrt{2} \end{bmatrix}
```

This method of calculating a matrix vector product won't actually be more computationally efficient - but it's what's "really" happening when you do the multiplication, so this will be useful intuition under certain circumstances. Expressing vectors in an eigenbasis is also a useful proof technique, as we'll see in some later problems.

## 3. Singular Value Decomposition - Proofs

Recall that if we have a symmetric, square matrix $`A \in \mathbb{R}^{n \times n}`$, we can eigen-decompose it in the form of $`A = USU^T`$, where the columns of $`U`$ are eigenvectors of $`A`$ with lengths of 1, and the diagonal of $`S`$ is the list of eigenvalues corresponding to those eigenvectors.

Now, for a more general case, where $`A`$ is a data matrix with the dimension of $`\mathbb{R}^{n \times d}`$, there is still a way to decompose it: $`A = USV^T`$, where $`U \in \mathbb{R}^{n \times n}`$, $`S`$ is a rectangular diagonal matrix and $`S \in \mathbb{R}^{n \times d}`$, and $`V \in \mathbb{R}^{d \times d}`$. It is called Singular Value Decomposition (SVD).

### (a)

Let A have SVD $`USV^T`$. Show $`AA^T`$ has the columns of $`U`$ as eigenvectors with associated eigenvalues $`S^2`$.

**Solution:**
We have $`A = USV^T`$ then:
```math
AA^T = USV^T (USV^T)^T
     = USV^T ((V^T)^T S^T U^T)
     = USV^T V S^T U^T
     = USISU^T
     = US^2U^T
```
Since we can diagonalize $`AA^T`$ into $`US^2U^T`$, it has eigenvectors that are columns of $`U`$ and associated eigenvalues $`S^2`$.

### (b)

Let A have SVD $`USV^T`$. Show $`A^T A`$ has the columns of $`V`$ as eigenvectors with associated eigenvalues $`S^2`$.

**Solution:**
We have $`A = USV^T`$ then:
```math
A^T A = (USV^T)^T USV^T
      = VS^T U^T USV^T
      = VS^T ISV^T
      = VSISV^T
      = VS^2V^T
```
Since we can diagonalize $`A^T A`$ into $`VS^2V^T`$, it has eigenvectors that are columns of $`V`$ and associated eigenvalues $`S^2`$.

### (c)

Show that for a matrix $`A`$, given $`AA^T = US^2U^T`$ and $`A^T A = VS^2V^T`$, then $`A = USV^T`$. In other words, show that for any vector $`x \in \mathbb{R}^d`$, $`Ax = USV^T x`$.

**Solution:**
Let $`\{v_1, v_2, ..., v_n\}`$ be the rows of $`V^T`$. They are orthogonal and unit norm. Any $`x \in \mathbb{R}^d`$ can be written as $`x = \sum_{i=1}^d \alpha_i v_i`$.

**First Derivation:**
```math
USV^T x = USV^T \sum_{i=1}^d \alpha_i v_i
= \sum_{i=1}^d \alpha_i USV^T v_i
= \sum_{i=1}^d \alpha_i US e_i
= \sum_{i=1}^d \alpha_i U \lambda_i e_i
= \sum_{i=1}^d \alpha_i \lambda_i u_i
```

**Intermediate Explanation:**
*   $`v_i`$ is an eigenvector of $`A^T A`$, so $`A^T A v_i = \lambda_i^2 v_i`$ (equation 1).
*   Multiplying equation (1) by $`A`$ on both sides: $`(AA^T) A v_i = \lambda_i^2 A v_i`$, so $`A v_i`$ is an eigenvector of $`AA^T`$.
*   Multiplying equation (1) by $`v_i^T`$ on both sides: $`v_i^T A^T A v_i = \lambda_i^2 v_i^T v_i`$, which is $`||A v_i||^2 = \lambda_i^2 ||v_i||^2`$.
*   So the length of vector $`A v_i`$ is $`\lambda_i`$.
*   Normalize the vector $`A v_i`$: $`\frac{A v_i}{\lambda_i} = u_i`$.
*   So $`\lambda_i u_i = A v_i`$.

**Second Derivation:**
Plug it back into the formula:
```math
USV^T x = \sum_{i=1}^d \alpha_i \lambda_i u_i
= \sum_{i=1}^d \alpha_i A v_i
= A \sum_{i=1}^d \alpha_i v_i
= Ax
```

## 4. Convolutional Neural Networks

### (a)

Discuss the advantages of a convolutional layer compared to a fully connected one.

**Solution:**
Convolutional layers are more flexible than fully connected ones since not all input neurons affect all output neurons. In addition, the number of weights per layer is smaller than that of linear layers, which would ease computation with high-dimensional data.

### (b)

Discuss the advantages of maxpooling in CNN.

**Solution:**
Pooling layers are used to downsample feature maps, which make processing more efficient by reducing the number of parameters to learn.

## 5. Shapes in Convolutional Neural Networks

Understanding data shapes is crucial when designing convolutional neural networks. This problem set aims to provide experience and intuition regarding why CNNs require fewer parameters compared to fully connected layers.

**Shape of a convolutional layer / maxpooling output:**
For a $`n \times n`$ input, $`f \times f`$ filter, padding $`p`$, and stride $`s`$:
```math
o = \frac{n - f + 2p}{s} + 1
```

### (a)

Pytorch `Conv2d` will represent 2D convolution and `MaxPool2d` will represent 2D max pooling. Given an initial input tensor shape of `(N, 3, 64, 64)` (a batch of `N` 64x64 RGB images), determine the new shape of the tensor after each operation. Note that activations are omitted as they don't change data shape.

**Solution:**

**1. Operation:** `Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)`
**Solution:** `(N, 16, 64, 64)`

**2. Operation:** `MaxPool2d(kernel_size=2, stride=2, padding=0)`
**Solution:** `(N, 16, 32, 32)`

**3. Operation:** `Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)`
**Solution:** `(N, 32, 30, 30)`

**4. Operation:** `MaxPool2d(kernel_size=2, stride=2, padding=1)`
**Solution:** `(N, 32, 16, 16)`

**5. Operation:** `Conv2D(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0)`
**Solution:** `(N, 8, 16, 16)`

### (b)

For each convolutional and fully connected layer, compute the number of parameters. For convolutional layers, also compute the number of parameters for a hypothetical fully connected layer mapping from the flattened input channels to the flattened output channels. Parameter calculations can be left as products and additions.

**Solution:**

**Layer 1: `Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)`**
*   **Conv Parameters:** $`3 * 16 * 3 * 3 + 16 = 448`$
*   **Fully Connected Parameters (hypothetical):** $`3 * 64 * 64 * 16 * 64 * 64 + 16 * 64 * 64 = 805306512`$

**Layer 2: `MaxPool2d(kernel_size=2, stride=2, padding=0)`**
*   **Parameters:** $`0`$

**Layer 3: `Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)`**
*   **Conv Parameters:** $`16 * 32 * 3 * 3 + 32 = 4640`$
*   **Fully Connected Parameters (hypothetical):** $`16 * 32 * 32 * 32 * 30 * 30 + 32 * 30 * 30 = 471888000`$

**Layer 4: `MaxPool2d(kernel_size=2, stride=2, padding=1)`**
*   **Output Shape:** $`(N, 32, 16, 16)`$

**Layer 5: `Conv2D(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0)`**
*   **Conv Parameters:** $`32 * 8 + 8 = 264`$
*   **Fully Connected Parameters (hypothetical):** $`32 * 16 * 16 * 8 * 16 * 16 + 8 * 16 * 16 = 16779264`$

**Layer 6: `Conv2D(in_channels=8, out_channels=4, kernel_size=5, stride=1, padding=0)`**
*   **Conv Parameters:** $`8 * 4 * 5 * 5 + 4 = 804`$
*   **Fully Connected Parameters (hypothetical):** $`8 * 16 * 16 * 4 * 12 * 12 + 4 * 12 * 12 = 1180224`$

**Layer 7: `Flatten`**
*   **Parameters:** $`0`$

**Layer 8: `Linear(in_features=576, out_features=10)`**
*   **Parameters:** $`576 * 10 + 10 = 5770`$


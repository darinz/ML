# 5.4 Properties of Kernels

## Introduction

In the previous section, we started with an explicitly defined feature map $\phi$, which induces the kernel function $K(x, z) \triangleq \langle \phi(x), \phi(z) \rangle$. Then we saw that the kernel function is so intrinsic that as long as the kernel function is defined, the whole training algorithm can be written entirely in the language of the kernel without referring to the feature map $\phi$, so can the prediction of a test example $x$ (equation (5.12)).

**The Key Insight**: The kernel function is more fundamental than the feature map. We can work entirely with kernels without ever explicitly computing features.

Therefore, it would be tempting to define other kernel functions $K(\cdot, \cdot)$ and run the algorithm (5.11). Note that the algorithm (5.11) does not need to explicitly access the feature map $\phi$, and therefore we only need to ensure the existence of the feature map $\phi$, but do not necessarily need to be able to explicitly write $\phi$ down.

**The Fundamental Question**: What kinds of functions $K(\cdot, \cdot)$ can correspond to some feature map $\phi$? In other words, can we tell if there is some feature mapping $\phi$ so that $K(x, z) = \phi(x)^T \phi(z)$ for all $x, z$?

If we can answer this question by giving a precise characterization of valid kernel functions, then we can completely change the interface of selecting feature maps $\phi$ to the interface of selecting kernel function $K$. Concretely, we can pick a function $K$, verify that it satisfies the characterization (so that there exists a feature map $\phi$ that $K$ corresponds to), and then we can run update rule (5.11). The benefit here is that we don't have to be able to compute $\phi$ or write it down analytically, and we only need to know its existence.

We will answer this question at the end of this subsection after we go through several concrete examples of kernels.

## Concrete Examples of Kernels

### Example 1: The Quadratic Kernel

Suppose $x, z \in \mathbb{R}^d$, and let's first consider the function $K(\cdot, \cdot)$ defined as:

$$
K(x, z) = (x^T z)^2.
$$

*Implementation details are provided in the accompanying Python examples file.*

**Intuition**: This kernel measures the squared similarity between two points. Points that are similar (high inner product) will have high kernel values.

**The Feature Map**: We can also write this as

$$
\begin{align*}
K(x, z) &= \left( \sum_{i=1}^d x_i z_i \right) \left( \sum_{j=1}^d x_j z_j \right) \\
        &= \sum_{i=1}^d \sum_{j=1}^d x_i x_j z_i z_j \\
        &= \sum_{i,j=1}^d (x_i x_j)(z_i z_j)
\end{align*}
$$

*Implementation details are provided in the accompanying Python examples file.*

**The Key Insight**: This shows that $K(x, z) = \langle \phi(x), \phi(z) \rangle$ is the kernel function that corresponds to the feature mapping $\phi$ given (shown here for the case of $d=3$) by

$$
\phi(x) = \begin{bmatrix}
    x_1 x_1 \\
    x_1 x_2 \\
    x_1 x_3 \\
    x_2 x_1 \\
    x_2 x_2 \\
    x_2 x_3 \\
    x_3 x_1 \\
    x_3 x_2 \\
    x_3 x_3
\end{bmatrix}.
$$

*Implementation details are provided in the accompanying Python examples file.*

**Computational Efficiency**: Revisiting the computational efficiency perspective of kernel, note that whereas calculating the high-dimensional $\phi(x)$ requires $O(d^2)$ time, finding $K(x, z)$ takes only $O(d)$ time—linear in the dimension of the input attributes.

**Why This Matters**: 
- **Explicit features**: Need to compute all pairwise products $x_i x_j$
- **Kernel**: Just compute the inner product once and square it
- **Speedup**: From $O(d^2)$ to $O(d)$ operations

### Example 2: The Quadratic Kernel with Bias

For another related example, also consider $K(\cdot, \cdot)$ defined by

$$
K(x, z) = (x^T z + c)^2
$$

$$
= \sum_{i,j=1}^d (x_i x_j)(z_i z_j) + \sum_{i=1}^d (\sqrt{2c}x_i)(\sqrt{2c}z_i) + c^2.
$$

*Implementation details are provided in the accompanying Python examples file.*

**The Feature Map**: This corresponds to a feature map that includes:
1. All pairwise products $x_i x_j$ (quadratic terms)
2. All linear terms $\sqrt{2c} x_i$ (linear terms)
3. A constant term $c^2$ (bias term)

*Implementation details are provided in the accompanying Python examples file.*

**Generalization**: More broadly, the kernel $K(x, z) = (x^T z + c)^k$ corresponds to a feature mapping to a $\binom{d+k}{k}$ feature space, corresponding to all monomials of the form $x_{i_1} x_{i_2} \ldots x_{i_k}$ that are up to order $k$. However, despite working in this $O(d^k)$-dimensional space, computing $K(x, z)$ still takes only $O(d)$ time, and hence we never need to explicitly represent feature vectors in this very high dimensional feature space.

*Implementation details are provided in the accompanying Python examples file.*

**The Magic**: We can work in a $\binom{d+k}{k}$-dimensional space with only $O(d)$ computation!

## Kernels as Similarity Metrics

### Intuitive Understanding

Now, let's talk about a slightly different view of kernels. Intuitively, (and there are things wrong with this intuition, but nevermind), if $\phi(x)$ and $\phi(z)$ are close together, then we might expect $K(x, z) = \phi(x)^T \phi(z)$ to be large. Conversely, if $\phi(x)$ and $\phi(z)$ are far apart—say nearly orthogonal to each other—then $K(x, z) = \phi(x)^T \phi(z)$ will be small. So, we can think of $K(x, z)$ as some measurement of how similar are $\phi(x)$ and $\phi(z)$, or of how similar are $x$ and $z$.

**The Similarity Interpretation**:
- **High kernel value**: Points are similar in the feature space
- **Low kernel value**: Points are dissimilar in the feature space
- **Zero kernel value**: Points are orthogonal (perpendicular) in the feature space

**Examples**:
- **Linear kernel**: $K(x, z) = \langle x, z \rangle$ - measures cosine similarity
- **RBF kernel**: $K(x, z) = \exp(-\gamma \|x - z\|^2)$ - measures distance-based similarity
- **Polynomial kernel**: $K(x, z) = (\langle x, z \rangle + 1)^d$ - measures polynomial similarity

### The RBF Kernel Example

Given this intuition, suppose that for some learning problem that you're working on, you've come up with some function $K(x, z)$ that you think might be a reasonable measure of how similar $x$ and $z$ are. For instance, perhaps you chose

$$
K(x, z) = \exp\left(-\frac{\|x - z\|^2}{2\sigma^2}\right).
$$

*Implementation details are provided in the accompanying Python examples file.*

**Intuition Behind RBF Kernel**:
- **Distance-based**: Similarity decreases exponentially with distance
- **Local**: Each point has a "sphere of influence" with radius $\sigma$
- **Smooth**: Provides smooth similarity measures
- **Universal**: Can approximate any continuous function

**The Parameter $\sigma$**:
- **Small $\sigma$**: Narrow influence, sharp boundaries
- **Large $\sigma$**: Wide influence, smooth boundaries

## Necessary Conditions for Valid Kernels

### The Kernel Matrix

Suppose for now that $K$ is indeed a valid kernel corresponding to some feature mapping $\phi$, and we will first see what properties it satisfies. Now, consider some finite set of $n$ points (not necessarily the training set) $\{x^{(1)}, \ldots, x^{(n)}\}$, and let a square, $n$-by-$n$ matrix $K$ be defined so that its $(i, j)$-entry is given by $K_{ij} = K(x^{(i)}, x^{(j)})$. This matrix is called the **kernel matrix**. Note that we've overloaded the notation and used $K$ to denote both the kernel function $K(x, z)$ and the kernel matrix $K$, due to their obvious close relationship.

*Implementation details are provided in the accompanying Python examples file.*

**Properties of the Kernel Matrix**:

1. **Symmetry**: $K_{ij} = K_{ji}$ for all $i, j$
   - **Why**: $K(x^{(i)}, x^{(j)}) = \langle \phi(x^{(i)}), \phi(x^{(j)}) \rangle = \langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle = K(x^{(j)}, x^{(i)})$

2. **Positive Semi-definiteness**: For any vector $z \in \mathbb{R}^n$, $z^T K z \geq 0$
   - **Why**: This follows from the fact that $K_{ij} = \langle \phi(x^{(i)}), \phi(x^{(j)}) \rangle$

### Proof of Positive Semi-definiteness

Now, if $K$ is a valid kernel, then $K_{ij} = K(x^{(i)}, x^{(j)}) = \phi(x^{(i)})^T \phi(x^{(j)}) = \phi(x^{(j)})^T \phi(x^{(i)}) = K(x^{(j)}, x^{(i)}) = K_{ji}$, and hence $K$ must be symmetric. Moreover, letting $\phi_k(x)$ denote the $k$-th coordinate of the vector $\phi(x)$, we have

$$
z^T K z = \sum_i \sum_j z_i K_{ij} z_j \\
= \sum_i \sum_j z_i \phi(x^{(i)})^T \phi(x^{(j)}) z_j \\
= \sum_i \sum_j z_i \sum_k \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j \\
= \sum_k \sum_i \sum_j z_i \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j \\
= \sum_k \left( \sum_i z_i \phi_k(x^{(i)}) \right)^2 \\
\geq 0.
$$

*Implementation details are provided in the accompanying Python examples file.*

**Step-by-step explanation**:

1. **Start**: $z^T K z = \sum_i \sum_j z_i K_{ij} z_j$
2. **Substitute**: $K_{ij} = \phi(x^{(i)})^T \phi(x^{(j)})$
3. **Expand**: $\phi(x^{(i)})^T \phi(x^{(j)}) = \sum_k \phi_k(x^{(i)}) \phi_k(x^{(j)})$
4. **Rearrange**: $\sum_k \sum_i \sum_j z_i \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j$
5. **Factor**: $\sum_k \left( \sum_i z_i \phi_k(x^{(i)}) \right)^2$
6. **Result**: Sum of squares is always non-negative

**The Key Insight**: This shows that any valid kernel must give rise to a positive semi-definite kernel matrix.

## Sufficient Conditions for Valid Kernels

### Mercer's Theorem

More generally, the condition above turns out to be not only a necessary, but also a sufficient, condition for $K$ to be a valid kernel (also called a Mercer kernel). The following result is due to Mercer.

**Theorem (Mercer).** Let $K : \mathbb{R}^d \times \mathbb{R}^d \mapsto \mathbb{R}$ be given. Then for $K$ to be a valid (Mercer) kernel, it is necessary and sufficient that for any $\{x^{(1)}, \ldots, x^{(n)}\}$, $(n < \infty)$, the corresponding kernel matrix is symmetric positive semi-definite.

**What This Means**:
- **Necessary**: Every valid kernel must give a positive semi-definite matrix
- **Sufficient**: Every function that gives positive semi-definite matrices is a valid kernel
- **Practical**: We can test if a function is a kernel by checking positive semi-definiteness

**Testing a Function**:
1. Pick any finite set of points $\{x^{(1)}, \ldots, x^{(n)}\}$
2. Compute the kernel matrix $K_{ij} = K(x^{(i)}, x^{(j)})$
3. Check if $K$ is symmetric and positive semi-definite
4. If yes for all finite sets, then $K$ is a valid kernel

### Why This Matters

Given a function $K$, apart from trying to find a feature mapping $\phi$ that corresponds to it, this theorem therefore gives another way of testing if it is a valid kernel. You'll also have a chance to play with these ideas more in exercise set 2.

**The Power of Mercer's Theorem**:
- **Existence**: Guarantees that a feature map exists (even if we can't write it down)
- **Flexibility**: Allows us to design kernels without explicitly constructing feature maps
- **Theoretical Foundation**: Provides the mathematical basis for kernel methods

## Application of Kernel Methods

**Broad Applicability**: We've seen the application of kernels to linear regression. In the next part, we will introduce the support vector machines to which kernels can be directly applied. 

In fact, the idea of kernels has significantly broader applicability than linear regression and SVMs. Specifically, if you have any learning algorithm that you can write in terms of only inner products $\langle x, z \rangle$ between input attribute vectors, then by replacing this with $K(x, z)$ where $K$ is a kernel, you can "magically" allow your algorithm to work efficiently in the high dimensional feature space corresponding to $K$. For instance, this kernel trick can be applied with the perceptron to derive a kernel perceptron algorithm. Many of the algorithms that we'll see later in this class will also be amenable to this method, which has come to be known as the "kernel trick."

*Implementation details are provided in the accompanying Python examples file.*

**Examples of Kernelizable Algorithms**:
- **Perceptron**: $\langle w, x \rangle$ becomes $\sum_i \alpha_i K(x^{(i)}, x)$
- **Principal Component Analysis**: Covariance matrix becomes kernel matrix
- **Fisher's Linear Discriminant**: Can be kernelized for non-linear discriminant analysis
- **K-means clustering**: Can be kernelized for non-linear clustering

**The Kernel Trick in Practice**:
1. **Identify inner products**: Find where $\langle x, z \rangle$ appears in your algorithm
2. **Replace with kernel**: Substitute $K(x, z)$ for $\langle x, z \rangle$
3. **Work in feature space**: Your algorithm now operates in the feature space implicitly
4. **Enjoy non-linearity**: Capture complex patterns without explicit feature engineering

---

*Implementation details are provided in the accompanying Python examples file.*

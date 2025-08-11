# 5.4 Properties of Kernels: The Mathematical Foundation

## Introduction: From Computation to Theory

In the previous section, we started with an explicitly defined feature map $\phi$, which induces the kernel function $K(x, z) \triangleq \langle \phi(x), \phi(z) \rangle$. Then we saw that the kernel function is so intrinsic that as long as the kernel function is defined, the whole training algorithm can be written entirely in the language of the kernel without referring to the feature map $\phi$, so can the prediction of a test example $x$ (equation (5.12)).

**The Key Insight**: The kernel function is more fundamental than the feature map. We can work entirely with kernels without ever explicitly computing features.

**The philosophical shift:** We've moved from thinking about explicit feature transformations to thinking about implicit similarity measures. Instead of asking "What features should I compute?" we ask "How should I measure similarity?"

Therefore, it would be tempting to define other kernel functions $K(\cdot, \cdot)$ and run the algorithm (5.11). Note that the algorithm (5.11) does not need to explicitly access the feature map $\phi$, and therefore we only need to ensure the existence of the feature map $\phi$, but do not necessarily need to be able to explicitly write $\phi$ down.

**The computational freedom:** This is incredibly liberating! We can design similarity functions without worrying about how to compute the underlying features. It's like being able to design a recipe without having to know how to grow the ingredients.

**The Fundamental Question**: What kinds of functions $K(\cdot, \cdot)$ can correspond to some feature map $\phi$? In other words, can we tell if there is some feature mapping $\phi$ so that $K(x, z) = \phi(x)^T \phi(z)$ for all $x, z$?

**The validation challenge:** Not every function that takes two inputs and returns a number is a valid kernel. We need mathematical criteria to distinguish valid kernels from invalid ones.

If we can answer this question by giving a precise characterization of valid kernel functions, then we can completely change the interface of selecting feature maps $\phi$ to the interface of selecting kernel function $K$. Concretely, we can pick a function $K$, verify that it satisfies the characterization (so that there exists a feature map $\phi$ that $K$ corresponds to), and then we can run update rule (5.11). The benefit here is that we don't have to be able to compute $\phi$ or write it down analytically, and we only need to know its existence.

**The design freedom:** Once we have these criteria, we can design kernels based on domain knowledge, intuition, or mathematical principles, without ever having to construct the feature space explicitly.

We will answer this question at the end of this subsection after we go through several concrete examples of kernels.

## From Computational Techniques to Mathematical Rigor: The Bridge to Theory

In the previous section, we explored the **kernel trick** - a powerful computational technique that allows us to work in high-dimensional feature spaces without explicitly computing the features. We saw how kernels can capture complex non-linear patterns efficiently and enable algorithms to operate in infinite-dimensional spaces with finite computation.

**The computational success:** We discovered that we can work with billion-dimensional feature spaces using only thousand-dimensional computations. This is like being able to navigate a vast library by just knowing how similar books are to each other, without having to read every book.

However, having this computational tool raises a fundamental question: **What makes a function a valid kernel?** Not every function $K(x, z)$ corresponds to an inner product in some feature space. We need mathematical criteria to distinguish valid kernels from invalid ones.

**The validation problem:** Just because a function takes two inputs and returns a number doesn't mean it's a valid kernel. For example, $K(x, z) = x_1 + z_1$ is not a valid kernel, even though it's a well-defined function. We need mathematical criteria to tell us which functions work and which don't.

This motivates our exploration of **kernel properties** - the mathematical foundations that tell us which functions can serve as kernels. We'll learn about positive semi-definiteness, Mercer's theorem, and the conditions that guarantee a function corresponds to a valid feature map.

**The theoretical foundation:** Understanding these properties is like learning the grammar of a language. Once you know the rules, you can create new sentences (kernels) that are guaranteed to be valid.

The transition from kernel methods to kernel properties represents the bridge from computational techniques to mathematical rigor - understanding not just how to use kernels, but why they work and how to design new ones.

**The bridge analogy:** We've learned how to drive a car (use kernels), now we need to understand the engineering principles (mathematical properties) that make the car work.

In this section, we'll explore the mathematical properties that make kernels valid and learn how to test whether a given function can serve as a kernel.

**The practical goal:** By the end of this section, you'll be able to look at any function and determine whether it can serve as a kernel, and even design new kernels based on mathematical principles.

## Concrete Examples of Kernels: Learning by Example

### Example 1: The Quadratic Kernel - From Simple to Complex

Suppose $x, z \in \mathbb{R}^d$, and let's first consider the function $K(\cdot, \cdot)$ defined as:

$$
K(x, z) = (x^T z)^2.
$$

**The simple beauty:** This kernel takes the inner product of two vectors and squares it. It's like measuring how similar two objects are, then amplifying that similarity.

*Implementation details are provided in the accompanying Python examples file.*

**Intuition**: This kernel measures the squared similarity between two points. Points that are similar (high inner product) will have high kernel values.

**The amplification effect:** By squaring the inner product, we amplify the differences. Points that are very similar become even more similar, while points that are dissimilar become even more dissimilar. This creates stronger separation between classes.

**The Feature Map**: We can also write this as

$$
\begin{align*}
K(x, z) &= \left( \sum_{i=1}^d x_i z_i \right) \left( \sum_{j=1}^d x_j z_j \right) \\
        &= \sum_{i=1}^d \sum_{j=1}^d x_i x_j z_i z_j \\
        &= \sum_{i,j=1}^d (x_i x_j)(z_i z_j)
\end{align*}
$$

**The expansion magic:** We've taken a simple squared term and expanded it to reveal all the pairwise interactions. This is like discovering that a simple recipe actually involves many complex interactions between ingredients.

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

**The feature space revelation:** The quadratic kernel corresponds to a feature space where each feature is a product of two input dimensions. This captures all pairwise interactions between features.

**The dimensionality explosion:** For $d=3$, we go from 3 dimensions to 9 dimensions. For $d=100$, we'd go from 100 dimensions to 10,000 dimensions! This is why the kernel trick is so powerful.

*Implementation details are provided in the accompanying Python examples file.*

**Computational Efficiency**: Revisiting the computational efficiency perspective of kernel, note that whereas calculating the high-dimensional $\phi(x)$ requires $O(d^2)$ time, finding $K(x, z)$ takes only $O(d)$ time—linear in the dimension of the input attributes.

**The computational miracle:** We can work in a 10,000-dimensional space using only 100-dimensional computations. This is like being able to navigate a vast library by just knowing how similar books are to each other.

**Why This Matters**: 
- **Explicit features**: Need to compute all pairwise products $x_i x_j$
- **Kernel**: Just compute the inner product once and square it
- **Speedup**: From $O(d^2)$ to $O(d)$ operations

**The practical impact:** For $d=1000$, explicit feature computation would require 1,000,000 operations, while kernel computation requires only 1,000 operations. That's a 1000x speedup!

### Example 2: The Quadratic Kernel with Bias - Adding Flexibility

For another related example, also consider $K(\cdot, \cdot)$ defined by

$$
K(x, z) = (x^T z + c)^2
$$

$$
= \sum_{i,j=1}^d (x_i x_j)(z_i z_j) + \sum_{i=1}^d (\sqrt{2c}x_i)(\sqrt{2c}z_i) + c^2.
$$

**The bias parameter:** The constant $c$ adds flexibility to the kernel. It's like adding a baseline similarity that all points share, regardless of how similar they are in the original space.

**The expansion insight:** When we expand $(x^T z + c)^2$, we get three types of terms:
1. Quadratic terms: $x_i x_j z_i z_j$ (capturing interactions)
2. Linear terms: $\sqrt{2c} x_i \sqrt{2c} z_i$ (capturing individual features)
3. Constant term: $c^2$ (capturing bias)

*Implementation details are provided in the accompanying Python examples file.*

**The Feature Map**: This corresponds to a feature map that includes:
1. All pairwise products $x_i x_j$ (quadratic terms)
2. All linear terms $\sqrt{2c} x_i$ (linear terms)
3. A constant term $c^2$ (bias term)

**The rich feature space:** By adding the bias term, we've created a feature space that includes both linear and quadratic relationships. This is like having a recipe that can capture both simple and complex flavor interactions.

**The parameter interpretation:** The constant $c$ controls the balance between linear and quadratic terms. When $c$ is large, the linear terms dominate. When $c$ is small, the quadratic terms dominate.

*Implementation details are provided in the accompanying Python examples file.*

**Generalization**: More broadly, the kernel $K(x, z) = (x^T z + c)^k$ corresponds to a feature mapping to a $\binom{d+k}{k}$ feature space, corresponding to all monomials of the form $x_{i_1} x_{i_2} \ldots x_{i_k}$ that are up to order $k$. However, despite working in this $O(d^k)$-dimensional space, computing $K(x, z)$ still takes only $O(d)$ time, and hence we never need to explicitly represent feature vectors in this very high dimensional feature space.

**The polynomial family:** This gives us a whole family of kernels, from linear ($k=1$) to quadratic ($k=2$) to cubic ($k=3$) and beyond. Each captures different levels of complexity in the data.

**The dimensionality explosion:** For $d=100$ and $k=3$, we'd work in a space with $\binom{103}{3} = 176,851$ dimensions! But we only need $O(d) = O(100)$ computations.

*Implementation details are provided in the accompanying Python examples file.*

**The Magic**: We can work in a $\binom{d+k}{k}$-dimensional space with only $O(d)$ computation!

**The computational miracle:** This is like being able to navigate a library with 176,851 books using only the knowledge of how similar 100 books are to each other. The kernel trick makes the impossible possible.

## Kernels as Similarity Metrics: The Intuitive Foundation

### Intuitive Understanding: Beyond the Mathematics

Now, let's talk about a slightly different view of kernels. Intuitively, (and there are things wrong with this intuition, but nevermind), if $\phi(x)$ and $\phi(z)$ are close together, then we might expect $K(x, z) = \phi(x)^T \phi(z)$ to be large. Conversely, if $\phi(x)$ and $\phi(z)$ are far apart—say nearly orthogonal to each other—then $K(x, z) = \phi(x)^T \phi(z)$ will be small. So, we can think of $K(x, z)$ as some measurement of how similar are $\phi(x)$ and $\phi(z)$, or of how similar are $x$ and $z$.

**The similarity perspective:** Kernels are fundamentally about measuring similarity between objects. They answer the question: "How similar are these two things?" rather than "What features do these things have?"

**The geometric intuition:** In the feature space, similar points are close together, so their inner product (kernel value) is large. Dissimilar points are far apart, so their inner product is small.

**The Similarity Interpretation**:
- **High kernel value**: Points are similar in the feature space
- **Low kernel value**: Points are dissimilar in the feature space
- **Zero kernel value**: Points are orthogonal (perpendicular) in the feature space

**The practical insight:** This means we can design kernels based on our intuition about what makes two objects similar, rather than having to explicitly define features.

**Examples**:
- **Linear kernel**: $K(x, z) = \langle x, z \rangle$ - measures cosine similarity
- **RBF kernel**: $K(x, z) = \exp(-\gamma \|x - z\|^2)$ - measures distance-based similarity
- **Polynomial kernel**: $K(x, z) = (\langle x, z \rangle + 1)^d$ - measures polynomial similarity

**The design philosophy:** Each kernel captures a different notion of similarity:
- **Linear**: Similarity based on direction (cosine)
- **RBF**: Similarity based on distance (closer = more similar)
- **Polynomial**: Similarity based on polynomial relationships

**Real-world analogy:** Think of it like different ways to measure similarity between people:
- **Linear**: How similar are their preferences (cosine similarity)
- **RBF**: How close do they live to each other (distance-based)
- **Polynomial**: How similar are their complex personality traits (polynomial relationships)

### The RBF Kernel Example: The Universal Similarity Measure

Given this intuition, suppose that for some learning problem that you're working on, you've come up with some function $K(x, z)$ that you think might be a reasonable measure of how similar $x$ and $z$ are. For instance, perhaps you chose

$$
K(x, z) = \exp\left(-\frac{\|x - z\|^2}{2\sigma^2}\right).
$$

**The RBF intuition:** This kernel measures similarity based on distance. Points that are close together are very similar (kernel value close to 1), while points that are far apart are very dissimilar (kernel value close to 0).

**The exponential decay:** The similarity decreases exponentially with the squared distance. This creates a smooth, natural measure of similarity that feels intuitive.

*Implementation details are provided in the accompanying Python examples file.*

**Intuition Behind RBF Kernel**:
- **Distance-based**: Similarity decreases exponentially with distance
- **Local**: Each point has a "sphere of influence" with radius $\sigma$
- **Smooth**: Provides smooth similarity measures
- **Universal**: Can approximate any continuous function

**The local influence:** Each training point creates a "bubble" of influence around it. Points inside the bubble are considered similar, points outside are considered dissimilar.

**The smoothness property:** The exponential function provides smooth transitions between similar and dissimilar regions, avoiding sharp boundaries that could lead to overfitting.

**The universal approximation:** The RBF kernel can approximate any continuous function, making it a very flexible choice for many problems.

**The Parameter $\sigma$**:
- **Small $\sigma$**: Narrow influence, sharp boundaries
- **Large $\sigma$**: Wide influence, smooth boundaries

**The bandwidth interpretation:** The parameter $\sigma$ controls the "bandwidth" of the kernel - how far the influence of each point extends.

**Real-world analogy:** Think of $\sigma$ like the range of a radio signal. A small $\sigma$ is like a short-range radio that only reaches nearby receivers. A large $\sigma$ is like a long-range radio that reaches receivers far away.

**The practical choice:** Choosing $\sigma$ is often the most important decision when using RBF kernels. Too small, and the model overfits. Too large, and the model underfits.

## Necessary Conditions for Valid Kernels: The Mathematical Requirements

### The Kernel Matrix: The Bridge Between Theory and Practice

Suppose for now that $K$ is indeed a valid kernel corresponding to some feature mapping $\phi$, and we will first see what properties it satisfies. Now, consider some finite set of $n$ points (not necessarily the training set) $\{x^{(1)}, \ldots, x^{(n)}\}$, and let a square, $n$-by-$n$ matrix $K$ be defined so that its $(i, j)$-entry is given by $K_{ij} = K(x^{(i)}, x^{(j)})$. This matrix is called the **kernel matrix**. Note that we've overloaded the notation and used $K$ to denote both the kernel function $K(x, z)$ and the kernel matrix $K$, due to their obvious close relationship.

**The kernel matrix insight:** The kernel matrix is like a "similarity table" that shows how similar each point is to every other point. It's the bridge between the abstract kernel function and the concrete computations we need to do.

**The matrix interpretation:** Each entry $K_{ij}$ tells us how similar point $i$ is to point $j$ in the feature space. The diagonal entries $K_{ii}$ tell us how similar each point is to itself (which should be the maximum similarity).

*Implementation details are provided in the accompanying Python examples file.*

**Properties of the Kernel Matrix**:

1. **Symmetry**: $K_{ij} = K_{ji}$ for all $i, j$
   - **Why**: $K(x^{(i)}, x^{(j)}) = \langle \phi(x^{(i)}), \phi(x^{(j)}) \rangle = \langle \phi(x^{(j)}), \phi(x^{(i)}) \rangle = K(x^{(j)}, x^{(i)})$

**The symmetry intuition:** Similarity is symmetric - if A is similar to B, then B is similar to A. This is like saying "if Alice is similar to Bob, then Bob is similar to Alice."

2. **Positive Semi-definiteness**: For any vector $z \in \mathbb{R}^n$, $z^T K z \geq 0$
   - **Why**: This follows from the fact that $K_{ij} = \langle \phi(x^{(i)}), \phi(x^{(j)}) \rangle$

**The positive semi-definiteness intuition:** This property ensures that the kernel matrix behaves like a proper similarity matrix. It guarantees that the "similarity energy" is always non-negative, which makes mathematical sense.

**The geometric interpretation:** Positive semi-definiteness means that the kernel matrix can be interpreted as a Gram matrix of some set of vectors in some inner product space. This is the mathematical foundation that makes kernels work.

### Proof of Positive Semi-definiteness

Now, if $K$ is a valid kernel, then $K_{ij} = K(x^{(i)}, x^{(j)}) = \phi(x^{(i)})^T \phi(x^{(j)}) = \phi(x^{(j)})^T \phi(x^{(i)}) = K(x^{(j)}, x^{(i)}) = K_{ji}$, and hence $K$ must be symmetric. Moreover, letting $\phi_k(x)$ denote the $k$-th coordinate of the vector $\phi(x)$, we have

$$
\begin{align}
z^T K z &= \sum_i \sum_j z_i K_{ij} z_j \\
&= \sum_i \sum_j z_i \phi(x^{(i)})^T \phi(x^{(j)}) z_j \\
&= \sum_i \sum_j z_i \sum_k \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j \\
&= \sum_k \sum_i \sum_j z_i \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j \\
&= \sum_k \left( \sum_i z_i \phi_k(x^{(i)}) \right)^2 \\
&\geq 0.
\end{align}
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

## From Kernel Foundations to Support Vector Machines

We've now established the mathematical foundations of kernel methods - understanding what makes a function a valid kernel through positive semi-definiteness and Mercer's theorem. This theoretical framework provides the rigor needed to design and validate kernel functions.

However, kernels are most powerful when applied to specific algorithms that can leverage their computational efficiency. **Support Vector Machines (SVMs)** represent the perfect marriage of kernel methods with a powerful classification algorithm that naturally benefits from the kernel trick.

The key insight that connects kernels to SVMs is the concept of **margins** - the distance between the decision boundary and the closest data points. SVMs seek to maximize this margin, leading to robust classifiers that generalize well. The kernel trick allows SVMs to find large-margin decision boundaries in high-dimensional feature spaces without explicitly computing the features.

The transition from kernel properties to SVM margins represents the bridge from mathematical foundations to practical algorithms - taking the theoretical understanding of kernels and applying it to build powerful, robust classifiers.

In the next section, we'll explore how margins provide both geometric intuition and theoretical guarantees for classification, setting the stage for the optimal margin classifier that will naturally lead to the dual formulation and kernelization.

---

**Previous: [Kernel Methods](01_kernel_methods.md)** - Learn about the kernel trick and computational techniques for non-linear classification.

**Next: [SVM Margins](03_svm_margins.md)** - Understand the geometric intuition and mathematical formulation of margins in support vector machines.

*Implementation details are provided in the accompanying Python examples file.*

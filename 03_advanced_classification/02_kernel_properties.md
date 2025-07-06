# 5.4 Properties of kernels

In the last subsection, we started with an explicitly defined feature map $\phi$, which induces the kernel function $K(x, z) \triangleq \langle \phi(x), \phi(z) \rangle$. Then we saw that the kernel function is so intrinsic so that as long as the kernel function is defined, the whole training algorithm can be written entirely in the language of the kernel without referring to the feature map $\phi$, so can the prediction of a test example $x$ (equation (5.12)).

Therefore, it would be tempted to define other kernel function $K(\cdot, \cdot)$ and run the algorithm (5.11). Note that the algorithm (5.11) does not need to explicitly access the feature map $\phi$, and therefore we only need to ensure the existence of the feature map $\phi$, but do not necessarily need to be able to explicitly write $\phi$ down.

What kinds of functions $K(\cdot, \cdot)$ can correspond to some feature map $\phi$? In other words, can we tell if there is some feature mapping $\phi$ so that $K(x, z) = \phi(x)^T \phi(z)$ for all $x, z$?

If we can answer this question by giving a precise characterization of valid kernel functions, then we can completely change the interface of selecting feature maps $\phi$ to the interface of selecting kernel function $K$. Concretely, we can pick a function $K$, verify that it satisfies the characterization (so that there exists a feature map $\phi$ that $K$ corresponds to), and then we can run update rule (5.11). The benefit here is that we don't have to be able to compute $\phi$ or write it down analytically, and we only need to know its existence. We will answer this question at the end of this subsection after we go through several concrete examples of kernels.

Suppose $x, z \in \mathbb{R}^d$, and let's first consider the function $K(\cdot, \cdot)$ defined as:

$$
K(x, z) = (x^T z)^2.
$$

**Python code (pure and NumPy):**
```python
# Pure Python
x = [1, 2, 3]
z = [4, 5, 6]
def quadratic_kernel(x, z):
    return sum(xi * zi for xi, zi in zip(x, z)) ** 2
print(quadratic_kernel(x, z))

# NumPy
import numpy as np
x = np.array([1, 2, 3])
z = np.array([4, 5, 6])
def quadratic_kernel_np(x, z):
    return np.dot(x, z) ** 2
print(quadratic_kernel_np(x, z))
```

We can also write this as

$$
\begin{align*}
K(x, z) &= \left( \sum_{i=1}^d x_i z_i \right) \left( \sum_{j=1}^d x_j z_j \right) \\
        &= \sum_{i=1}^d \sum_{j=1}^d x_i x_j z_i z_j \\
        &= \sum_{i,j=1}^d (x_i x_j)(z_i z_j)
\end{align*}
$$

**Python code for expanded form:**
```python
# Pure Python expanded form
def expanded_quadratic_kernel(x, z):
    d = len(x)
    return sum(x[i] * x[j] * z[i] * z[j] for i in range(d) for j in range(d))
print(expanded_quadratic_kernel(x, z))
```

Thus, we see that $K(x, z) = \langle \phi(x), \phi(z) \rangle$ is the kernel function that corresponds to the the feature mapping $\phi$ given (shown here for the case of $d=3$) by

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

**Python code for explicit feature mapping ($d=3$):**
```python
def phi_quadratic(x):
    return [x[0]*x[0], x[0]*x[1], x[0]*x[2],
            x[1]*x[0], x[1]*x[1], x[1]*x[2],
            x[2]*x[0], x[2]*x[1], x[2]*x[2]]
print(phi_quadratic([1, 2, 3]))
```

Revisiting the computational efficiency perspective of kernel, note that whereas calculating the high-dimensional $\phi(x)$ requires $O(d^2)$ time, finding $K(x, z)$ takes only $O(d)$ time—linear in the dimension of the input attributes.

For another related example, also consider $K(\cdot, \cdot)$ defined by

$$
K(x, z) = (x^T z + c)^2
$$

$$
= \sum_{i,j=1}^d (x_i x_j)(z_i z_j) + \sum_{i=1}^d (\sqrt{2c}x_i)(\sqrt{2c}z_i) + c^2.
$$

**Python code for $(x^T z + c)^2$ kernel:**
```python
def poly2_kernel(x, z, c=1.0):
    return (np.dot(x, z) + c) ** 2
print(poly2_kernel(x, z, c=2.0))
```

**Python code for explicit feature mapping ($d=3$):**
```python
def phi_poly2(x, c=1.0):
    import math
    return [x[0]*x[0], x[0]*x[1], x[0]*x[2],
            x[1]*x[0], x[1]*x[1], x[1]*x[2],
            x[2]*x[0], x[2]*x[1], x[2]*x[2],
            math.sqrt(2*c)*x[0], math.sqrt(2*c)*x[1], math.sqrt(2*c)*x[2], c]
print(phi_poly2([1, 2, 3], c=2.0))
```

More broadly, the kernel $K(x, z) = (x^T z + c)^k$ corresponds to a feature mapping to a $\binom{d+k}{k}$ feature space, corresponding to all monomials of the form $x_{i_1} x_{i_2} \ldots x_{i_k}$ that are up to order $k$. However, despite working in this $O(d^k)$-dimensional space, computing $K(x, z)$ still takes only $O(d)$ time, and hence we never need to explicitly represent feature vectors in this very high dimensional feature space.

**Python code for general polynomial kernel (NumPy and scikit-learn):**
```python
# NumPy
def poly_kernel(x, z, c=1.0, degree=3):
    return (np.dot(x, z) + c) ** degree
print(poly_kernel(x, z, c=1.0, degree=3))

# scikit-learn
from sklearn.metrics.pairwise import polynomial_kernel
X = np.array([[1, 2, 3], [4, 5, 6]])
print(polynomial_kernel(X, X, degree=3, coef0=1.0))
```

## Kernels as similarity metrics

Now, let's talk about a slightly different view of kernels. Intuitively, (and there are things wrong with this intuition, but nevermind), if $\phi(x)$ and $\phi(z)$ are close together, then we might expect $K(x, z) = \phi(x)^T \phi(z)$ to be large. Conversely, if $\phi(x)$ and $\phi(z)$ are far apart—say nearly orthogonal to each other—then $K(x, z) = \phi(x)^T \phi(z)$ will be small. So, we can think of $K(x, z)$ as some measurement of how similar are $\phi(x)$ and $\phi(z)$, or of how similar are $x$ and $z$.

Given this intuition, suppose that for some learning problem that you're working on, you've come up with some function $K(x, z)$ that you think might be a reasonable measure of how similar $x$ and $z$ are. For instance, perhaps you chose

$$
K(x, z) = \exp\left(-\frac{\|x - z\|^2}{2\sigma^2}\right).
$$

**Python code for Gaussian (RBF) kernel:**
```python
def rbf_kernel(x, z, sigma=1.0):
    diff = np.array(x) - np.array(z)
    return np.exp(-np.dot(diff, diff) / (2 * sigma ** 2))
print(rbf_kernel([1, 2, 3], [4, 5, 6], sigma=2.0))

# scikit-learn
from sklearn.metrics.pairwise import rbf_kernel
X = np.array([[1, 2, 3], [4, 5, 6]])
print(rbf_kernel(X, X, gamma=1/(2*2.0**2)))  # gamma = 1/(2*sigma^2)
```

## Necessary conditions for valid kernels

Suppose for now that $K$ is indeed a valid kernel corresponding to some feature mapping $\phi$, and we will first see what properties it satisfies. Now, consider some finite set of $n$ points (not necessarily the training set) $\{x^{(1)}, \ldots, x^{(n)}\}$, and let a square, $n$-by-$n$ matrix $K$ be defined so that its $(i, j)$-entry is given by $K_{ij} = K(x^{(i)}, x^{(j)})$. This matrix is called the **kernel matrix**. Note that we've overloaded the notation and used $K$ to denote both the kernel function $K(x, z)$ and the kernel matrix $K$, due to their obvious close relationship.

**Python code to compute a kernel matrix:**
```python
def kernel_matrix(X, kernel_func):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(X[i], X[j])
    return K
X = [[1, 0], [0, 1], [1, 1]]
print(kernel_matrix(X, quadratic_kernel))
```

Now, if $K$ is a valid kernel, then $K_{ij} = K(x^{(i)}, x^{(j)}) = \phi(x^{(i)})^T \phi(x^{(j)}) = \phi(x^{(j)})^T \phi(x^{(i)}) = K(x^{(j)}, x^{(i)}) = K_{ji}$, and hence $K$ must be symmetric. Moreover, letting $o(x)$ denote the $k$-th coordinate of the vector $\phi(x)$, we have

$$
z^T K z = \sum_i \sum_j z_i K_{ij} z_j \\
= \sum_i \sum_j z_i \phi(x^{(i)})^T \phi(x^{(j)}) z_j \\
= \sum_i \sum_j z_i \sum_k \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j \\
= \sum_k \sum_i \sum_j z_i \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j \\
= \sum_k \left( \sum_i z_i \phi_k(x^{(i)}) \right)^2 \\
\geq 0.
$$

**Python code to check positive semidefinite (PSD):**
```python
# Check if a matrix is positive semidefinite
K = kernel_matrix(X, quadratic_kernel)
eigvals = np.linalg.eigvalsh(K)
print("Eigenvalues:", eigvals)
print("Is PSD?", np.all(eigvals >= -1e-10))  # Allow for small numerical errors
```

## Sufficient conditions for valid kernels

More generally, the condition above turns out to be not only a necessary, but also a sufficient, condition for $K$ to be a valid kernel (also called a Mercer kernel). The following result is due to Mercer.

**Theorem (Mercer).** Let $K : \mathbb{R}^d \times \mathbb{R}^d \mapsto \mathbb{R}$ be given. Then for $K$ to be a valid (Mercer) kernel, it is necessary and sufficient that for any $\{x^{(1)}, \ldots, x^{(n)}\}$, $(n < \infty)$, the corresponding kernel matrix is symmetric positive semi-definite.

Given a function $K$, apart from trying to find a feature mapping $\phi$ that corresponds to it, this theorem therefore gives another way of testing if it is a valid kernel. You'll also have a chance to play with these ideas more in exercise set 2.

---

**Application of kernel methods:** We've seen the application of kernels to linear regression. In the next part, we will introduce the support vector machines to which kernels can be directly applied. 

In fact, the idea of kernels has significantly broader applicability than linear regression and SVMs. Specifically, if you have any learning algorithm that you can write in terms of only inner products $\langle x, z \rangle$ between input attribute vectors, then by replacing this with $K(x, z)$ where $K$ is a kernel, you can "magically" allow your algorithm to work efficiently in the high dimensional feature space corresponding to $K$. For instance, this kernel trick can be applied with the perceptron to derive a kernel perceptron algorithm. Many of the algorithms that we'll see later in this class will also be amenable to this method, which has come to be known as the "kernel trick."

**Python code: SVM with kernels using scikit-learn**
```python
from sklearn import svm
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
X, y = make_moons(n_samples=100, noise=0.1)
clf = svm.SVC(kernel='rbf', gamma=2.0)
clf.fit(X, y)

# Plot decision boundary
import numpy as np
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('SVM with RBF Kernel')
plt.show()
```

---

**String kernel (conceptual code):**
```python
def substring_kernel(x, z, k=3):
    # x, z are strings
    from collections import Counter
    def k_substrings(s):
        return [s[i:i+k] for i in range(len(s)-k+1)]
    cx = Counter(k_substrings(x))
    cz = Counter(k_substrings(z))
    # Dot product in substring count space
    return sum(cx[sub] * cz[sub] for sub in set(cx) | set(cz))
print(substring_kernel('GATTACA', 'TACAGAT', k=2))
```

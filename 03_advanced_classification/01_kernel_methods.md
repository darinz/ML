# Kernel Methods: A Comprehensive Guide

## 5.1 Feature Maps and the Motivation for Kernels

### 5.1.1 The Linear Model Limitation

Recall that in our discussion about linear regression, we considered the problem of predicting the price of a house (denoted by $y$) from the living area of the house (denoted by $x$), and we fit a linear function of $x$ to the training data. What if the price $y$ can be more accurately represented as a *non-linear* function of $x$? In this case, we need a more expressive family of models than linear models.

**Example: Housing Price Prediction**
Consider a dataset where house prices follow a non-linear pattern:
- Small houses (500-1000 sq ft): Price increases slowly
- Medium houses (1000-2000 sq ft): Price increases rapidly  
- Large houses (2000+ sq ft): Price increases slowly again

A linear model $y = \theta_1 x + \theta_0$ would fail to capture this pattern, leading to poor predictions.

### 5.1.2 Polynomial Feature Maps

We start by considering fitting cubic functions $y = \theta_3 x^3 + \theta_2 x^2 + \theta_1 x + \theta_0$. It turns out that we can view the cubic function as a linear function over a different set of feature variables (defined below). Concretely, let the function $\phi : \mathbb{R} \to \mathbb{R}^4$ be defined as

```math
\phi(x) = \begin{bmatrix} 1 \\ x \\ x^2 \\ x^3 \end{bmatrix} \in \mathbb{R}^4.
```

Let $\theta \in \mathbb{R}^4$ be the vector containing $\theta_0, \theta_1, \theta_2, \theta_3$ as entries. Then we can rewrite the cubic function in $x$ as:

```math
\theta_3 x^3 + \theta_2 x^2 + \theta_1 x + \theta_0 = \theta^T \phi(x)
```

**Key Insight**: A cubic function of the variable $x$ can be viewed as a linear function over the variables $\phi(x)$. This is the fundamental idea behind feature maps.

### 5.1.3 Terminology and Definitions

To distinguish between these two sets of variables, in the context of kernel methods, we will call the "original" input value the input **attributes** of a problem (in this case, $x$, the living area). When the original input is mapped to some new set of quantities $\phi(x)$, we will call those new quantities the **features** variables. We will call $\phi$ a **feature map**, which maps the attributes to the features.

**Formal Definition**: A feature map is a function $\phi : \mathcal{X} \to \mathcal{H}$ where:
- $\mathcal{X}$ is the input space (e.g., $\mathbb{R}^d$)
- $\mathcal{H}$ is the feature space (e.g., $\mathbb{R}^p$ where $p \geq d$)

### 5.1.4 Examples of Feature Maps

#### Polynomial Feature Maps
For degree $k$ polynomials in $d$ dimensions:
```math
\phi(x) = [1, x_1, x_2, \ldots, x_d, x_1^2, x_1x_2, \ldots, x_d^k]^T
```

#### Radial Basis Function (RBF) Feature Maps
```math
\phi(x) = [\exp(-\gamma\|x - c_1\|^2), \exp(-\gamma\|x - c_2\|^2), \ldots]^T
```
where $c_i$ are centers and $\gamma$ is a parameter.

#### Trigonometric Feature Maps
```math
\phi(x) = [1, \sin(x), \cos(x), \sin(2x), \cos(2x), \ldots]^T
```

### 5.1.5 The Curse of Dimensionality

As we increase the degree of polynomial features or the dimensionality of the input, the feature space grows exponentially:

- Degree 2 polynomial in $d$ dimensions: $O(d^2)$ features
- Degree 3 polynomial in $d$ dimensions: $O(d^3)$ features  
- Degree $k$ polynomial in $d$ dimensions: $O(d^k)$ features

This exponential growth makes explicit computation of features computationally prohibitive for high-dimensional data.

## 5.2 LMS (Least Mean Squares) with Features

### 5.2.1 Review of Standard LMS

We will derive the gradient descent algorithm for fitting the model $\theta^T \phi(x)$. First recall that for ordinary least square problem where we were to fit $\theta^T x$, the batch gradient descent update is:

```math
\theta := \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - h_\theta(x^{(i)}) \right) x^{(i)}
```

```math
:= \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - \theta^T x^{(i)} \right) x^{(i)}. \tag{5.2}
```

**Derivation**: The gradient of the loss function $J(\theta) = \frac{1}{2n}\sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2$ with respect to $\theta$ is:
```math
\nabla_\theta J(\theta) = -\frac{1}{n}\sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)}) x^{(i)}
```

### 5.2.2 LMS with Feature Maps

Let $\phi : \mathbb{R}^d \to \mathbb{R}^p$ be a feature map that maps attribute $x$ (in $\mathbb{R}^d$) to the features $\phi(x)$ in $\mathbb{R}^p$. Now our goal is to fit the function $\theta^T \phi(x)$, with $\theta$ being a vector in $\mathbb{R}^p$ instead of $\mathbb{R}^d$.

**Key Insight**: We can replace all the occurrences of $x^{(i)}$ in the algorithm above by $\phi(x^{(i)})$ to obtain the new update:

```math
\theta := \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}). \tag{5.3}
```

Similarly, the corresponding stochastic gradient descent update rule is:

```math
\theta := \theta + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}). \tag{5.4}
```

### 5.2.3 Computational Complexity Analysis

**Standard LMS**: $O(d)$ per update
**LMS with Features**: $O(p)$ per update

When $p \gg d$ (e.g., polynomial features), this becomes computationally expensive.

### 5.2.4 Implementation Example

```python
import numpy as np

def polynomial_feature_map(x, degree=3):
    """Create polynomial features up to given degree."""
    features = [1]  # bias term
    for d in range(1, degree + 1):
        features.append(x ** d)
    return np.array(features)

def lms_with_features(X, y, feature_map, learning_rate=0.01, max_iterations=1000):
    """LMS algorithm with custom feature map."""
    n_samples = X.shape[0]
    
    # Initialize parameters
    # We need to determine the feature dimension
    sample_features = feature_map(X[0])
    theta = np.zeros(len(sample_features))
    
    for iteration in range(max_iterations):
        for i in range(n_samples):
            # Compute features
            phi_x = feature_map(X[i])
            
            # Compute prediction
            prediction = np.dot(theta, phi_x)
            
            # Compute error
            error = y[i] - prediction
            
            # Update parameters
            theta += learning_rate * error * phi_x
    
    return theta

# Example usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 4, 9, 16, 25])  # y = x^2

theta = lms_with_features(X, y, lambda x: polynomial_feature_map(x, degree=2))
print(f"Learned parameters: {theta}")
```

## 5.3 The Kernel Trick: Efficient Computation

### 5.3.1 The Computational Challenge

The gradient descent update becomes computationally expensive when the features $\phi(x)$ are high-dimensional. Consider the direct extension of the feature map to high-dimensional input $x$: suppose $x \in \mathbb{R}^d$, and let $\phi(x)$ be the vector that contains all the monomials of $x$ with degree $\leq 3$:

```math
\phi(x) = \begin{bmatrix}
1 \\
x_1 \\
x_2 \\
\vdots \\
x_1^2 \\
x_1 x_2 \\
x_1 x_3 \\
\vdots \\
x_2 x_1 \\
\vdots \\
x_1^3 \\
x_1^2 x_2 \\
\vdots
\end{bmatrix}.
\tag{5.5}
```

The dimension of the features $\phi(x)$ is on the order of $d^3$. This is prohibitively expensive — when $d = 1000$, each update requires computing and storing a $1000^3 = 10^9$ dimensional vector.

### 5.3.2 The Representer Theorem

**Key Insight**: At any time, $\theta$ can be represented as a linear combination of the vectors $\phi(x^{(1)}), \ldots, \phi(x^{(n)})$.

**Proof by Induction**:
1. **Base Case**: At initialization, $\theta = 0 = \sum_{i=1}^n 0 \cdot \phi(x^{(i)})$
2. **Inductive Step**: Assume at some point, $\theta$ can be represented as:

```math
\theta = \sum_{i=1}^n \beta_i \phi(x^{(i)}) \tag{5.6}
```

   for some $\beta_1, \ldots, \beta_n \in \mathbb{R}$.

3. **Update Step**: After one gradient update:

```math
\begin{align*}
\theta &:= \theta + \alpha \sum_{i=1}^n \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}) \\
&= \sum_{i=1}^n \beta_i \phi(x^{(i)}) + \alpha \sum_{i=1}^n \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \phi(x^{(i)}) \\
&= \sum_{i=1}^n \left( \beta_i + \alpha \left( y^{(i)} - \theta^T \phi(x^{(i)}) \right) \right) \phi(x^{(i)}) \tag{5.7}
\end{align*}
```

This shows that $\theta$ remains a linear combination of the training feature vectors.

### 5.3.3 The Kernel Function

**Definition**: The **Kernel** corresponding to the feature map $\phi$ is a function $K : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ satisfying:
```math
K(x, z) \triangleq \langle \phi(x), \phi(z) \rangle
```

**Key Insight**: We can compute $K(x, z)$ efficiently without explicitly computing $\phi(x)$ and $\phi(z)$.

### 5.3.4 The Polynomial Kernel

For the polynomial feature map $\phi$ defined in (5.5), we can compute the kernel efficiently:

```math
\begin{align*}
\langle \phi(x), \phi(z) \rangle &= 1 + \sum_{i=1}^d x_i z_i + \sum_{i,j \in \{1,\ldots,d\}} x_i x_j z_i z_j + \sum_{i,j,k \in \{1,\ldots,d\}} x_i x_j x_k z_i z_j z_k \\
&= 1 + \sum_{i=1}^d x_i z_i + \left( \sum_{i=1}^d x_i z_i \right)^2 + \left( \sum_{i=1}^d x_i z_i \right)^3 \\
&= 1 + \langle x, z \rangle + \langle x, z \rangle^2 + \langle x, z \rangle^3
\end{align*}
\tag{5.9}
```

**Computational Complexity**:
- Explicit feature computation: $O(d^3)$
- Kernel computation: $O(d)$

### 5.3.5 The Kernelized LMS Algorithm

**Step 1**: Pre-compute the kernel matrix $K$ where $K_{ij} = K(x^{(i)}, x^{(j)})$

**Step 2**: Initialize $\beta = 0$

**Step 3**: Iterative updates:
```math
\beta_i := \beta_i + \alpha \left( y^{(i)} - \sum_{j=1}^n \beta_j K(x^{(i)}, x^{(j)}) \right) \tag{5.11}
```

**Vector notation**:
```math
\beta := \beta + \alpha (\vec{y} - K \beta)
```

**Prediction for new point $x$**:
```math
\theta^T \phi(x) = \sum_{i=1}^n \beta_i \phi(x^{(i)})^T \phi(x) = \sum_{i=1}^n \beta_i K(x^{(i)}, x) \tag{5.12}
```

### 5.3.6 Implementation of Kernelized LMS

```python
import numpy as np

def polynomial_kernel(x, z, degree=3):
    """Compute polynomial kernel K(x, z) = (1 + <x, z>)^degree"""
    inner_product = np.dot(x, z)
    return (1 + inner_product) ** degree

def kernelized_lms(X, y, kernel_func, learning_rate=0.01, max_iterations=1000):
    """Kernelized LMS algorithm."""
    n_samples = X.shape[0]
    
    # Pre-compute kernel matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    
    # Initialize beta
    beta = np.zeros(n_samples)
    
    # Gradient descent
    for iteration in range(max_iterations):
        # Compute predictions
        predictions = K @ beta
        
        # Compute errors
        errors = y - predictions
        
        # Update beta
        beta += learning_rate * errors
    
    return beta, K

def predict_kernelized(X_train, X_test, beta, kernel_func):
    """Make predictions using kernelized model."""
    predictions = []
    for x_test in X_test:
        prediction = 0
        for i, x_train in enumerate(X_train):
            prediction += beta[i] * kernel_func(x_train, x_test)
        predictions.append(prediction)
    return np.array(predictions)

# Example usage
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x^2

beta, K = kernelized_lms(X, y, lambda x, z: polynomial_kernel(x, z, degree=2))
print(f"Learned beta coefficients: {beta}")

# Make predictions
X_test = np.array([[1.5], [2.5], [3.5]])
predictions = predict_kernelized(X, X_test, beta, lambda x, z: polynomial_kernel(x, z, degree=2))
print(f"Predictions: {predictions}")
```

## 5.4 Common Kernel Functions

### 5.4.1 Linear Kernel
```math
K(x, z) = \langle x, z \rangle
```
- Feature map: $\phi(x) = x$
- Use case: Linear models

### 5.4.2 Polynomial Kernel
```math
K(x, z) = (\gamma \langle x, z \rangle + r)^d
```
- Feature map: All monomials up to degree $d$
- Parameters: $\gamma$ (scaling), $r$ (bias), $d$ (degree)
- Use case: Polynomial regression

### 5.4.3 Radial Basis Function (RBF) Kernel
```math
K(x, z) = \exp(-\gamma \|x - z\|^2)
```
- Feature map: Infinite-dimensional (Mercer's theorem)
- Parameter: $\gamma$ (bandwidth)
- Use case: Non-linear classification/regression

### 5.4.4 Sigmoid Kernel
```math
K(x, z) = \tanh(\gamma \langle x, z \rangle + r)
```
- Feature map: Neural network-like
- Parameters: $\gamma$ (scaling), $r$ (bias)
- Use case: Neural network approximation

### 5.4.5 Kernel Selection Guidelines

1. **Linear Kernel**: When data is linearly separable
2. **Polynomial Kernel**: When data has polynomial structure
3. **RBF Kernel**: When data has no obvious structure (default choice)
4. **Sigmoid Kernel**: When data has neural network-like structure

## 5.5 Kernel Properties and Mercer's Theorem

### 5.5.1 Positive Definite Kernels

A kernel function $K$ is **positive definite** if for any finite set of points $x_1, \ldots, x_n$ and any real numbers $c_1, \ldots, c_n$:
```math
\sum_{i=1}^n \sum_{j=1}^n c_i c_j K(x_i, x_j) \geq 0
```

### 5.5.2 Mercer's Theorem

**Mercer's Theorem**: If $K$ is a positive definite kernel, then there exists a feature map $\phi$ such that:
```math
K(x, z) = \langle \phi(x), \phi(z) \rangle
```

This theorem guarantees that any positive definite kernel corresponds to an inner product in some feature space.

### 5.5.3 Kernel Construction Rules

If $K_1$ and $K_2$ are kernels, then the following are also kernels:

1. **Scalar multiplication**: $aK_1$ where $a > 0$
2. **Addition**: $K_1 + K_2$
3. **Multiplication**: $K_1 \cdot K_2$
4. **Composition**: $K_1(f(x), f(z))$ where $f$ is any function

## 5.6 Practical Considerations

### 5.6.1 Computational Complexity

**Training**: $O(n^2)$ for kernel matrix computation + $O(n^2)$ per iteration
**Prediction**: $O(n)$ per prediction (need to compute kernel with all training points)

### 5.6.2 Memory Requirements

- Kernel matrix: $O(n^2)$ storage
- For large datasets, this becomes prohibitive

### 5.6.3 Scalability Solutions

1. **Random Fourier Features**: Approximate RBF kernels
2. **Nyström Method**: Approximate kernel matrix
3. **Sparse Approximations**: Use subset of training points

### 5.6.4 Hyperparameter Tuning

**Cross-validation**: Essential for kernel parameter selection
**Grid search**: Common approach for parameter optimization

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Example: Tuning RBF kernel parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

svr = SVR(kernel='rbf')
grid_search = GridSearchCV(svr, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
```

## 5.7 Advanced Topics

### 5.7.1 Multiple Kernel Learning

Combine multiple kernels:
```math
K(x, z) = \sum_{i=1}^m \alpha_i K_i(x, z)
```
where $\alpha_i \geq 0$ and $\sum_{i=1}^m \alpha_i = 1$

### 5.7.2 Kernel PCA

Principal Component Analysis in feature space:
```math
K_{centered} = K - \frac{1}{n}1_n K - \frac{1}{n}K 1_n + \frac{1}{n^2}1_n K 1_n
```

### 5.7.3 Kernel Ridge Regression

Ridge regression with kernels:
```math
\beta = (K + \lambda I)^{-1} y
```

## 5.8 Summary and Key Insights

### 5.8.1 The Kernel Trick

1. **Representer Theorem**: $\theta$ can be written as linear combination of training features
2. **Kernel Function**: $K(x, z) = \langle \phi(x), \phi(z) \rangle$ can be computed efficiently
3. **Dual Representation**: Work with $\beta$ coefficients instead of $\theta$

### 5.8.2 Computational Benefits

- **Explicit features**: $O(d^k)$ computation
- **Kernel trick**: $O(d)$ computation
- **Memory**: $O(n^2)$ for kernel matrix vs $O(d^k)$ for explicit features

### 5.8.3 When to Use Kernels

**Use kernels when**:
- Data is non-linear
- Feature space is high-dimensional
- Explicit feature computation is expensive

**Avoid kernels when**:
- Data is linear
- Dataset is very large
- Interpretability is important

The kernel trick is one of the most powerful ideas in machine learning, allowing us to work in high-dimensional feature spaces efficiently by computing only inner products between data points.

[1]: Here, for simplicity, we include all the monomials with repetitions (so that, e.g., $x_1 x_2 x_3$ and $x_2 x_3 x_1$ both appear in $\phi(x)$). Therefore, there are totally $1 + d + d^2 + d^3$ entries in $\phi(x)$.

[2]: Recall that $\mathcal{X}$ is the space of the input $x$. In our running example, $\mathcal{X} = \mathbb{R}^d$

# Principal components analysis

## Introduction and Motivation

In modern data analysis and machine learning, we often encounter datasets with a large number of features (dimensions). For example, an image with 100x100 pixels has 10,000 features, and a dataset of cars might record dozens of attributes for each vehicle. However, not all of these features are equally important or independent—many are correlated or even redundant. This can make data analysis, visualization, and modeling more difficult, a phenomenon sometimes called the "curse of dimensionality."

Principal Components Analysis (PCA) is a powerful technique for reducing the number of dimensions in a dataset while preserving as much of the original information (variance) as possible. By finding new axes (directions) in the data that capture the most variation, PCA allows us to:
- Simplify complex datasets, making them easier to visualize and interpret
- Remove redundancy and noise
- Speed up machine learning algorithms by reducing input size
- Reveal hidden patterns and structure in the data

**Intuition:** Imagine you have a cloud of points in 3D space, but all the points lie close to a flat plane. Instead of describing each point with three coordinates, you could describe it almost as well with just two coordinates on that plane. PCA finds the best-fitting plane (or line, or higher-dimensional subspace) for your data, so you can represent it with fewer numbers without losing much information.

PCA is computationally efficient: it will require only an eigenvector calculation (easily done with the `eig` function in Matlab, or with `numpy.linalg.eig` in Python).

Suppose we are given a dataset $` x^{(i)}; i = 1, \ldots, n `$ of attributes of $` n `$ different types of automobiles, such as their maximum speed, turn radius, and so on. Let $` x^{(i)} \in \mathbb{R}^d `$ for each $` i `$ ($` d \ll n `$). But unknown to us, two different attributes—some $` x_i `$ and $` x_j `$—respectively give a car's maximum speed measured in miles per hour, and the maximum speed measured in kilometers per hour. These two attributes are therefore almost linearly dependent, up to only small differences introduced by rounding off to the nearest mph or kph. Thus, the data really lies approximately on an $` n-1 `$ dimensional subspace. How can we automatically detect, and perhaps remove, this redundancy?

For a less contrived example, consider a dataset resulting from a survey of pilots for radio-controlled helicopters, where $` x_1^{(i)} `$ is a measure of the piloting skill of pilot $` i `$ , and $` x_2^{(i)} `$ captures how much he/she enjoys flying. Because RC helicopters are very difficult to fly, only the most committed students, ones that truly enjoy flying, become good pilots. So, the two attributes $` x_1 `$ and $` x_2 `$ are strongly correlated. Indeed, we might posit that the data actually lies along some diagonal axis (the $` u_1 `$ direction) capturing the intrinsic piloting "karma" of a person, with only a small amount of noise lying off this axis. (See figure.) How can we automatically compute this $` u_1 `$ direction?

<img src="./img/pca_diagonal_axis.png" width="300px"/>

## Data Normalization: Why and How

Before applying PCA, it is standard practice to preprocess the data by normalizing each feature so that it has mean 0 and variance 1. This step is crucial for several reasons:

- **Equal footing for all features:** If one feature (e.g., height in centimeters) has much larger values than another (e.g., number of children), it can dominate the analysis, even if it is not more important. Normalization ensures that all features contribute equally.
- **Removes bias from units:** Features measured in different units (e.g., miles per hour vs. kilometers per hour) are made comparable.
- **Centers the data:** Subtracting the mean ensures that the data is centered at the origin, which is important for the mathematics of PCA.
- **Variance scaling:** Dividing by the standard deviation ensures that each feature has unit variance, so PCA does not favor features with larger natural variability.

### Step-by-step Example
Suppose we have a dataset with two features:

| Example | Height (cm) | Age (years) |
|---------|-------------|-------------|
| 1       | 170         | 30          |
| 2       | 160         | 25          |
| 3       | 180         | 35          |

**Step 1: Compute the mean of each feature**
- Mean height: (170 + 160 + 180) / 3 = 170
- Mean age: (30 + 25 + 35) / 3 = 30

**Step 2: Subtract the mean from each value (centering)**
- Centered heights: 0, -10, +10
- Centered ages: 0, -5, +5

**Step 3: Compute the standard deviation of each feature**
- Std height: sqrt(((0)^2 + (-10)^2 + (10)^2)/3) = sqrt((0 + 100 + 100)/3) = sqrt(200/3) ≈ 8.16
- Std age: sqrt(((0)^2 + (-5)^2 + (5)^2)/3) = sqrt((0 + 25 + 25)/3) = sqrt(50/3) ≈ 4.08

**Step 4: Divide each centered value by the standard deviation (scaling)**
- Normalized heights: 0/8.16 = 0, -10/8.16 ≈ -1.23, 10/8.16 ≈ 1.23
- Normalized ages: 0/4.08 = 0, -5/4.08 ≈ -1.23, 5/4.08 ≈ 1.23

After normalization, both features have mean 0 and variance 1, and are on the same scale. This prepares the data for PCA.

### Python: Data Normalization
```python
import numpy as np

# Example data: rows are samples, columns are features
X = np.array([
    [170, 30],
    [160, 25],
    [180, 35]
], dtype=float)

# Step 1: Compute mean and std for each feature (column)
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

# Step 2: Normalize (center and scale)
X_normalized = (X - mean) / std
print("Normalized data:\n", X_normalized)
```

Now, having normalized our data, how do we compute the "major axis of variation" $` u `$—that is, the direction on which the data approximately lies? One way is to pose this problem as finding the unit vector $` u `$ so that when the data is projected onto the direction corresponding to $` u `$ , the variance of the projected data is maximized. Intuitively, the data starts off with some amount of variance/information in it. We would like to choose a direction $` u `$ so that if we were to approximate the data as lying in the direction/subspace corresponding to $` u `$ , as much as possible of this variance is still retained.

Consider the following dataset, on which we have already carried out the normalization steps:

<img src="./img/pca_projection_example.png" width="300px"/>

Now, suppose we pick $` u `$ to correspond to the direction shown in the figure below. The circles denote the projections of the original data onto this line.

<img src="./img/pca_projection_line1.png" width="300px"/>

We see that the projected data still has a fairly large variance, and the points tend to be far from zero. In contrast, suppose we had instead picked the following direction:

<img src="./img/pca_projection_line2.png" width="300px"/>

Here, the projections have a significantly smaller variance, and are much closer to the origin.

We would like to automatically select the direction $` u `$ corresponding to the first of the two figures shown above. To formalize this, note that given a unit vector $` u `$ and a point $` x `$ , the length of the projection of $` x `$ onto $` u `$ is given by $` x^T u `$ . I.e., if $` x^{(i)} `$ is a point in our dataset (one of the crosses in the plot), then its projection onto $` u `$ (the corresponding circle in the figure) is distance $` x^{(i)T} u `$ from the origin. Hence, to maximize the variance of the projections, we would like to choose a unit-length $` u `$ so as to maximize:

```math
\frac{1}{n} \sum_{i=1}^n (x^{(i)T} u)^2 = \frac{1}{n} \sum_{i=1}^n u^T x^{(i)} x^{(i)T} u = u^T \left( \frac{1}{n} \sum_{i=1}^n x^{(i)} x^{(i)T} \right) u.
```

We easily recognize that the maximizing $` u `$ subject to $` \|u\|_2 = 1 `$ gives the principal eigenvector of $` \Sigma = \frac{1}{n} \sum_{i=1}^n x^{(i)} x^{(i)T} `$ , which is just the empirical covariance matrix of the data (assuming it has zero mean).[^eig]

To summarize, we have found that if we wish to find a 1-dimensional subspace with which to approximate the data, we should choose $` u `$ to be the principal eigenvector of $` \Sigma `$ . More generally, if we wish to project our data into a $` k `$-dimensional subspace ($` k < d `$), we should choose $` u_1, \ldots, u_k `$ to be the top $` k `$ eigenvectors of $` \Sigma `$ . The $` u_i `$'s now form a new, orthogonal basis for the data.[^orth]

Then, to represent $` x^{(i)} `$ in this basis, we need only compute the corresponding vector

```math
    y^{(i)} = \begin{bmatrix}
        u_1^T x^{(i)} \\
        u_2^T x^{(i)} \\
        \vdots \\
        u_k^T x^{(i)}
    \end{bmatrix} \in \mathbb{R}^k.
```

Thus, whereas $` x^{(i)} \in \mathbb{R}^d `$ , the vector $` y^{(i)} `$ now gives a lower, $` k `$-dimensional, approximation/representation for $` x^{(i)} `$ . PCA is therefore also referred to as a **dimensionality reduction** algorithm. The vectors $` u_1, \ldots, u_k `$ are called the first $` k `$ **principal components** of the data.

**Remark.** Although we have shown it formally only for the case of $` k = 1 `$ , using well-known properties of eigenvectors it is straightforward to show that of all possible orthogonal bases $` u_1, \ldots, u_k `$ , the one that we have chosen maximizes $` \sum_i \|y^{(i)}\|_2^2 `$ . Thus, our choice of a basis preserves as much variability as possible in the original data.

PCA can also be derived by picking the basis that minimizes the approximation error arising from projecting the data onto the $` k `$-dimensional subspace spanned by them. (See more in homework.)

[^eig]: If you haven't seen this before, try using the method of Lagrange multipliers to maximize $` u^T \Sigma u `$ subject to $` u^T u = 1 `$ . You should be able to show that $` \Sigma u = \lambda u `$ for some $` \lambda `$ , which implies $` u `$ is an eigenvector of $` \Sigma `$ , with eigenvalue $` \lambda `$ .

[^orth]: Because $` \Sigma `$ is symmetric, the $` u_i `$'s will (or always can be chosen to be) orthogonal to each other.

PCA has many applications; we will close our discussion with a few examples. First, compression—representing $` x^{(i)} `$'s with lower dimension $` y^{(i)} `$'s—is an obvious application. If we reduce high dimensional data to $` k = 2 `$ or 3 dimensions, then we can also plot the $` y^{(i)} `$'s to visualize the data. For instance, if we were to reduce our automobiles data to 2 dimensions, then we can plot it (one point in our plot would correspond to one car type, say) to see what cars are similar to each other and what groups of cars may cluster together.

Another standard application is to preprocess a dataset to reduce its dimension before running a supervised learning algorithm with the $` x^{(i)} `$'s as inputs. Apart from computational benefits, reducing the data's dimension can also reduce the complexity of the hypothesis class considered and help avoid overfitting (e.g., linear classifiers over lower dimensional input spaces will have smaller VC dimension).

Lastly, as in our RC pilot example, we can also view PCA as a noise reduction algorithm. In our example it estimates the intrinsic "piloting karma" from the noisy measures of piloting skill and enjoyment. In class, we also saw the application of this idea to face images, resulting in eigenfaces method. Here, each point $` x^{(i)} \in \mathbb{R}^{100 \times 100} `$ was a 10000 dimensional vector, with each coordinate corresponding to a pixel intensity value in a 100x100 image of a face. Using PCA, we represent each image $` x^{(i)} `$ with a much lower-dimensional $` y^{(i)} `$ . In doing so, we hope that the principal components we found retain the interesting, systematic variations between faces that capture what a person really looks like, but not the "noise" in the images introduced by minor lighting variations, slightly different imaging conditions, and so on. We then measure distances between faces $` i `$ and $` j `$ by working in the reduced dimension, and computing $` \|y^{(i)} - y^{(j)}\|_2 `$ . This resulted in a surprisingly good face-matching and retrieval algorithm.

## Variance Maximization and Geometric Intuition

After normalization, the next step in PCA is to find the direction in which the data varies the most. But what does this mean, and why do we care about variance?

**Intuitive analogy:** Imagine shining a flashlight on a cloud of points (your data) in a dark room. The shadow cast on the wall is the projection of your data onto a direction. If you want the shadow to be as "spread out" as possible, you would rotate the flashlight until the shadow is longest. This direction of maximum spread is what PCA seeks: the direction along which the data has the greatest variance.

**Why maximize variance?**
- The direction of maximum variance captures the most "information" or "structure" in the data. If you had to summarize the data with a single number (its coordinate along a line), you would want to choose the line that preserves as much of the original variation as possible.
- By projecting onto this direction, you lose the least amount of information about how the data points differ from each other.

**Geometric meaning:**
- Projecting a point $` x `$ onto a direction $` u `$ means dropping a perpendicular from $` x `$ to the line defined by $` u `$.
- The length of this projection is $` x^T u `$ (assuming $` u `$ is a unit vector).
- If you project all your data points onto $` u `$ and look at how spread out the projections are (their variance), the direction $` u `$ that gives the largest variance is the first principal component.

**Visual intuition:**
- If your data forms an elongated cloud, the first principal component points along the long axis of the cloud.
- The second principal component (if you want to keep two dimensions) is perpendicular to the first and points in the next most variable direction, and so on.

This geometric approach is at the heart of PCA: it finds new axes for your data that are ordered by how much variance (spread) they capture, allowing you to keep only the most important ones for further analysis.

## Covariance Matrix Calculation

The covariance matrix captures how each pair of features varies together. It is a key step in PCA, as its eigenvectors define the principal components.

### Python: Covariance Matrix
```python
# X_normalized: rows are samples, columns are features
# Compute the empirical covariance matrix
cov_matrix = np.cov(X_normalized, rowvar=False)
print("Covariance matrix:\n", cov_matrix)
```

## Eigen Decomposition for PCA

PCA finds the directions (principal components) by computing the eigenvectors and eigenvalues of the covariance matrix. The eigenvectors are the directions, and the eigenvalues tell us how much variance is captured by each direction.

### Python: Eigen Decomposition
```python
# Compute eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eig(cov_matrix)
print("Eigenvalues:\n", eigvals)
print("Eigenvectors (columns):\n", eigvecs)

# Sort eigenvectors by descending eigenvalue
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]
eigvals = eigvals[idx]

# First principal component
pc1 = eigvecs[:, 0]
print("First principal component:", pc1)
```

## Projecting Data onto Principal Components

To reduce dimensionality, we project the normalized data onto the top $k$ principal components (eigenvectors).

### Python: Project Data
```python
# Project data onto the first k principal components
k = 1  # or 2 for 2D
W = eigvecs[:, :k]  # projection matrix
X_pca = X_normalized @ W
print(f"Data projected onto first {k} principal component(s):\n", X_pca)
```

## Visualizing PCA Results

Visualization helps us understand how PCA separates the data. For 2D projections, we can plot the data along the first two principal components.

### Python: Visualization Example
```python
import matplotlib.pyplot as plt

# Project onto first 2 principal components
W2 = eigvecs[:, :2]
X_pca2 = X_normalized @ W2

plt.scatter(X_pca2[:, 0], X_pca2[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Projection (first 2 components)')
plt.grid(True)
plt.show()
```

## Example: PCA on a Small Dataset

Let's put it all together with a small example:

### Python: Full PCA Example
```python
import numpy as np
import matplotlib.pyplot as plt

# Small dataset
X = np.array([
    [2.5, 2.4],
    [0.5, 0.7],
    [2.2, 2.9],
    [1.9, 2.2],
    [3.1, 3.0],
    [2.3, 2.7],
    [2, 1.6],
    [1, 1.1],
    [1.5, 1.6],
    [1.1, 0.9]
])

# Normalize
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# Covariance
cov = np.cov(X_norm, rowvar=False)

# Eigen decomposition
eigvals, eigvecs = np.linalg.eig(cov)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx]

# Project
X_pca = X_norm @ eigvecs[:, :2]

# Plot
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Example: Small Dataset')
plt.grid(True)
plt.show()
```

## Using scikit-learn for PCA

For real-world datasets, you can use the `PCA` class from scikit-learn, which handles all steps for you.

### Python: scikit-learn PCA
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("PCA components (directions):\n", pca.components_)
```


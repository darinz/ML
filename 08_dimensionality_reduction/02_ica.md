# Independent components analysis

## Introduction and Motivation

Independent Components Analysis (ICA) is a powerful technique for separating a set of mixed signals into their original, independent sources. While Principal Components Analysis (PCA) finds new axes that maximize variance and decorrelate the data, ICA goes further: it tries to find components that are statistically independent, not just uncorrelated. This makes ICA especially useful for problems where the observed data is a mixture of underlying signals, and we want to recover those original signals.

### The Cocktail Party Problem: An Intuitive Example
Imagine you are at a lively cocktail party. There are several people (say, $` d `$ speakers) all talking at once, and you place $` d `$ microphones around the room. Each microphone records a different mixture of all the voices, depending on how close it is to each speaker. The challenge: can you take the recordings from the microphones and separate out each individual speaker's voice?

This is not just a party trick! Similar problems arise in:
- **Brain imaging:** EEG/MEG sensors record mixtures of brain signals.
- **Finance:** Observed prices are mixtures of underlying market factors.
- **Image processing:** Pixels may be mixtures of different sources of light.

ICA provides a mathematical framework to solve these kinds of problems, where the observed data is a mixture of independent sources.

---

As a motivating example, consider the "cocktail party problem." Here, $` d `$ speakers are speaking simultaneously at a party, and any microphone placed in the room records only an overlapping combination of the $` d `$ speakers' voices. But let's say we have $` d `$ different microphones placed in the room, and because each microphone is a different distance from each of the speakers, it records a different combination of the speakers' voices. Using these microphone recordings, can we separate out the original $` d `$ speakers' speech signals?

To formalize this problem, we imagine that there is some data $` s \in \mathbb{R}^d `$ that is generated via $` d `$ independent sources. What we observe is

```math
x = As,
```

where $` A `$ is an unknown square matrix called the mixing matrix. Repeated observations gives us a dataset $` \{x^{(i)}; i = 1, \ldots, n\} `$, and our goal is to recover the sources $` s^{(i)} `$ that had generated our data ($` x^{(i)} = As^{(i)} `$).

In our cocktail party problem, $` s^{(i)} `$ is an $` d `$-dimensional vector, and $` s_j^{(i)} `$ is the sound that speaker $` j `$ was uttering at time $` i `$ . Also, $` x^{(i)} `$ in an $` d `$-dimensional vector, and $` x_j^{(i)} `$ is the acoustic reading recorded by microphone $` j `$ at time $` i `$.

Let $` W = A^{-1} `$ be the **unmixing matrix**. Our goal is to find $` W `$, so that given our microphone recordings $` x^{(i)} `$, we can recover the sources by computing $` s^{(i)} = W x^{(i)} `$ . For notational convenience, we also let $` w_i^T `$ denote the $` i `$-th row of $` W `$, so that

```math
W = \begin{bmatrix}
    w_1^T \\
    \vdots \\
    w_d^T
\end{bmatrix}.
```

Thus, $` w_i \in \mathbb{R}^d `$ , and the $` j `$-th source can be recovered as $` s_j^{(i)} = w_j^T x^{(i)} `$.

---

### Python: Simulating the Cocktail Party Problem
```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# Generate independent sources
s1 = np.sin(2 * time)  # Sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Square signal
s3 = np.random.laplace(size=n_samples)  # Non-Gaussian noise
S = np.c_[s1, s2, s3]
S /= S.std(axis=0)  # Standardize data

# Mixing matrix (random)
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
X = S @ A.T  # Generate observations (mixtures)
```

### Python: Visualizing the Sources and Mixtures
```python
plt.figure(figsize=(9, 6))
models = [S, X]
names = ['True Sources', 'Observed Mixtures (Microphones)']
for i, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(2, 1, i)
    for sig in model.T:
        plt.plot(sig)
    plt.title(name)
plt.tight_layout()
plt.show()
```

---

## ICA Ambiguities

There are some fundamental ambiguities in ICA:
- **Permutation ambiguity:** The order of the recovered sources cannot be determined (any permutation of the sources is equally valid).
- **Scaling ambiguity:** The scale of each recovered source cannot be determined (multiplying a source by a constant and dividing the corresponding column of $` A `$ by the same constant leaves $` x `$ unchanged).
- **Sign ambiguity:** The sign of each source is arbitrary (flipping the sign of a source and the corresponding column of $` A `$ leaves $` x `$ unchanged).

These ambiguities do not matter for most applications, as we are usually interested in the independent sources themselves, not their order or scale.

Further, ICA only works when the sources are **non-Gaussian**. If the sources are Gaussian, the mixing is not identifiable due to rotational symmetry of the Gaussian distribution.

---

## Densities and Linear Transformations

Suppose a random variable $` s `$ is drawn according to some density $` p_s(s) `$ . For simplicity, assume for now that $` s \in \mathbb{R} `$ is a real number. Now, let the random variable $` x `$ be defined according to $` x = As `$ (here, $` x \in \mathbb{R} `$, $` A \in \mathbb{R} `$). Let $` p_x `$ be the density of $` x `$ . What is $` p_x `$(x)?

Let $` W = A^{-1} `$ . To calculate the "probability" of a particular value of $` x `$ , it is tempting to compute $` s = Wx `$ , then then evaluate $` p_s `$ at that point, and conclude that "$` p_x(x) = p_s(Wx) `$. However, *this is incorrect*. For example, let $` s \sim \mathrm{Uniform}[0, 1] `$ , so $` p_s(s) = 1\{0 \leq s \leq 1\} `$ . Now, let $` A = 2 `$ , so $` x = 2s `$ . Clearly, $` x `$ is distributed uniformly in the interval $` [0, 2] `$ . Thus, its density is given by $` p_x(x) = (0.5)1\{0 \leq x \leq 2\} `$ . This does not equal $` p_s(Wx) `$ , where $` W = 0.5 = A^{-1} `$ . Instead, the correct formula is $` p_x(x) = p_s(Wx)|W| `$ .

More generally, if $` s `$ is a vector-valued distribution with density $` p_s `$ , and $` x = As `$ for a square, invertible matrix $` A `$ , then the density of $` x `$ is given by

```math
p_x(x) = p_s(Wx) \cdot |W|,
```

where $` W = A^{-1} `$ .

---

## Whitening/Preprocessing

Before applying ICA, it is common to preprocess the data by whitening (decorrelating and scaling) the mixtures. This step is not strictly necessary for all ICA algorithms, but it often improves convergence and interpretability.

### Python: Whitening the Mixtures (PCA Preprocessing)
```python
X_centered = X - X.mean(axis=0)
cov = np.cov(X_centered, rowvar=False)
D, E = np.linalg.eigh(cov)
D = np.diag(1.0 / np.sqrt(D))
X_white = (X_centered @ E) @ D
print("Whitened data shape:", X_white.shape)
```

---

## ICA Algorithm (Gradient Ascent for W, simplified)

The following is a simplified version of the ICA algorithm using gradient ascent. For practical use, see the FastICA implementation below.

We suppose that the distribution of each source $` s_j `$ is given by a density $` p_s `$ , and that the joint distribution of the sources $` s `$ is given by

```math
p(s) = \prod_{j=1}^d p_s(s_j).
```

Using our formulas from the previous section, this implies the following density on $` x = As = W^{-1}s `$:

```math
p(x) = \prod_{j=1}^d p_s(w_j^T x) \cdot |W|.
```

To specify a density for the $` s_i `$'s, all we need to do is to specify some cdf for it. A cdf has to be a monotonic function that increases from zero to one. A common choice is the sigmoid function $` g(s) = 1/(1 + e^{-s}) `$ . Hence, $` p_s(s) = g'(s) `$ .

The square matrix $` W `$ is the parameter in our model. Given a training set $` \{x^{(i)}; i = 1, \ldots, n\} `$ , the log likelihood is given by

```math
\ell(W) = \sum_{i=1}^n \left( \sum_{j=1}^d \log g'(w_j^T x^{(i)}) + \log |W| \right).
```

We would like to maximize this in terms of $` W `$ . By taking derivatives and using the fact (from the first set of notes) that $` \nabla_W |W| = |W| (W^{-1})^T `$ , we easily derive a stochastic gradient ascent learning rule. For a training example $` x^{(i)} `$ , the update rule is:

```math
W := W + \alpha \left( \begin{bmatrix}
1 - 2g(w_1^T x^{(i)}) \\
1 - 2g(w_2^T x^{(i)}) \\
\vdots \\
1 - 2g(w_d^T x^{(i)})
\end{bmatrix} x^{(i)T} + (W^T)^{-1} \right),
```

where $` \alpha `$ is the learning rate.

After the algorithm converges, we then compute $` s^{(i)} = W x^{(i)} `$ to recover the original sources.

### Python: ICA via Gradient Ascent (Simplified)
```python
def g(x):
    return 1 / (1 + np.exp(-x))  # Sigmoid

def g_prime(x):
    gx = g(x)
    return gx * (1 - gx)

# FastICA-like fixed-point iteration for demonstration
W = np.random.randn(3, 3)
alpha = 0.1
for iteration in range(100):
    WX = X_white @ W.T
    gwx = g(WX)
    g_wx = 1 - 2 * gwx
    dW = (g_wx.T @ X_white) / n_samples + np.linalg.inv(W.T)
    W += alpha * dW
    # Decorrelate rows of W (optional, for stability)
    # from scipy.linalg import sqrtm
    # W = np.linalg.inv(sqrtm(W @ W.T)) @ W

S_est = X_white @ W.T
```

### Python: Visualizing the Recovered Sources
```python
plt.figure(figsize=(9, 6))
plt.subplot(2, 1, 1)
for sig in S.T:
    plt.plot(sig)
plt.title('True Sources')
plt.subplot(2, 1, 2)
for sig in S_est.T:
    plt.plot(sig)
plt.title('Recovered Sources (ICA)')
plt.tight_layout()
plt.show()
```

---

## Using scikit-learn's FastICA

For practical ICA, use the FastICA implementation from scikit-learn, which is robust and efficient.

### Python: ICA using scikit-learn's FastICA
```python
from sklearn.decomposition import FastICA
ica = FastICA(n_components=3, random_state=0)
S_ica = ica.fit_transform(X)  # Reconstruct signals
A_ica = ica.mixing_  # Estimated mixing matrix

plt.figure(figsize=(9, 6))
plt.subplot(2, 1, 1)
for sig in S.T:
    plt.plot(sig)
plt.title('True Sources')
plt.subplot(2, 1, 2)
for sig in S_ica.T:
    plt.plot(sig)
plt.title('Recovered Sources (FastICA)')
plt.tight_layout()
plt.show()
```

---

**Remark.** When writing down the likelihood of the data, we implicitly assumed that the $` x^{(i)} `$'s were independent of each other (for different values of $` i `$; note this issue is different from whether the different coordinates of $` x^{(i)} `$ are independent), so that the likelihood of the training set was given by $` \prod_i p(x^{(i)}; W) `$ . This assumption is clearly incorrect for speech data and other time series where the $` x^{(i)} `$'s are dependent, but it can be shown that having correlated training examples will not hurt the performance of the algorithm if we have sufficient data. However, for problems where successive training examples are correlated, when implementing stochastic gradient ascent, it sometimes helps accelerate convergence if we visit training examples in a randomly permuted order. (I.e., run stochastic gradient ascent on a randomly shuffled copy of the training set.)

# 11.3 General EM algorithms

Suppose we have an estimation problem in which we have a training set $`\{x^{(1)}, \ldots, x^{(n)}\}`$ consisting of $`n`$ independent examples. We have a latent variable model $`p(x, z; \theta)`$ with $`z`$ being the latent variable (which for simplicity is assumed to take a finite number of values). The density for $`x`$ can be obtained by marginalizing over the latent variable $`z`$:

```math
p(x; \theta) = \sum_z p(x, z; \theta)
```

**Intuition:**
- In many real-world problems, we observe data $`x`$ but believe there are hidden or latent factors $`z`$ that influence the data. For example, in clustering, $`z`$ could represent the cluster assignment for each data point.
- The model $`p(x, z; \theta)`$ describes how both the observed data and the hidden variables are generated together.
- To get the probability of the observed data alone, we sum (marginalize) over all possible values of the hidden variables.

We wish to fit the parameters $`\theta`$ by maximizing the log-likelihood of the data, defined by

```math
\ell(\theta) = \sum_{i=1}^n \log p(x^{(i)}; \theta)
```

**Why log-likelihood?**
- The log-likelihood is a standard objective in statistics and machine learning. Maximizing it means finding parameters that make the observed data as probable as possible under the model.

We can rewrite the objective in terms of the joint density $`p(x, z; \theta)`$ by

```math
\ell(\theta) = \sum_{i=1}^n \log p(x^{(i)}; \theta)
           = \sum_{i=1}^n \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta)
```

**Challenge:**
- The sum inside the log makes this a hard, non-convex optimization problem. If we knew the $`z^{(i)}`$'s, the problem would be much easier (as in supervised learning).

In such a setting, the EM algorithm gives an efficient method for maximum likelihood estimation. Maximizing $`\ell(\theta)`$ explicitly might be difficult, and our strategy will be to instead repeatedly construct a lower-bound on $`\ell`$ (E-step), and then optimize that lower-bound (M-step).

**Key idea:**
- The EM algorithm alternates between "guessing" the hidden variables (E-step) and updating the parameters (M-step), always improving a lower bound on the log-likelihood.

## The Evidence Lower Bound (ELBO) and Jensen's Inequality

To make the math tractable, we introduce a distribution $`Q(z)`$ over the possible values of $`z`$. That is, $`\sum_z Q(z) = 1`$, $`Q(z) \geq 0`$.

We can rewrite $`\log p(x; \theta)`$ as:

```math
\log p(x; \theta) = \log \sum_z p(x, z; \theta)
                  = \log \sum_z Q(z) \frac{p(x, z; \theta)}{Q(z)}
                  \geq \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)}
```

**Why is this true?**
- The last step uses Jensen's inequality, which says that the log of an average is greater than or equal to the average of the log (for concave functions like log).
- This gives us a lower bound (the ELBO) on the log-likelihood.

The term

```math
\sum_z Q(z) \left[ \frac{p(x, z; \theta)}{Q(z)} \right]
```

is just the expectation of $`[p(x, z; \theta)/Q(z)]`$ with respect to $`z`$ drawn from $`Q`$.

By Jensen's inequality,

```math
f\left( \mathbb{E}_{z \sim Q} \left[ \frac{p(x, z; \theta)}{Q(z)} \right] \right) \geq \mathbb{E}_{z \sim Q} \left[ f\left( \frac{p(x, z; \theta)}{Q(z)} \right) \right]
```

where $`f(x) = \log x`$ is concave.

**Tightness:**
- The bound is tight (i.e., becomes an equality) when $`Q(z)`$ is chosen to be proportional to $`p(x, z; \theta)`$.
- Normalizing, we get $`Q(z) = p(z|x; \theta)`$ (the posterior over $`z`$ given $`x`$).

## The EM Algorithm: Alternating Maximization

For a dataset $`\{x^{(1)}, \ldots, x^{(n)}\}`$, we introduce $`n`$ distributions $`Q_1, \ldots, Q_n`$, one for each example. For each $`x^{(i)}`$:

```math
\log p(x^{(i)}; \theta) \geq \mathrm{ELBO}(x^{(i)}; Q_i, \theta) = \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
```

Summing over all examples gives a lower bound for the log-likelihood:

```math
\ell(\theta) \geq \sum_i \mathrm{ELBO}(x^{(i)}; Q_i, \theta)
           = \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
```

**Optimal Q:**
- The bound is tight when $`Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)}; \theta)`$.
- The EM algorithm alternates between:
  - **E-step:** Set $`Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)}; \theta)`$ (compute the posterior over $`z`$ for each data point).
  - **M-step:** Maximize the ELBO with respect to $`\theta`$ (update the parameters).

**Algorithm:**

Repeat until convergence {

(E-step) For each $`i`$, set

```math
Q_i(z^{(i)}) := p(z^{(i)}|x^{(i)}; \theta)
```

(M-step) Set

```math
\theta := \arg\max_{\theta} \sum_{i=1}^n \mathrm{ELBO}(x^{(i)}; Q_i, \theta)
```

}

**Convergence:**
- Each iteration of EM increases (or leaves unchanged) the log-likelihood $`\ell(\theta)`$.
- The algorithm is guaranteed to converge to a local optimum (not necessarily the global optimum).
- A practical convergence test is to check if the increase in $`\ell(\theta)`$ between iterations is below a small threshold.

## The Evidence Lower Bound (ELBO): Alternative Forms and Intuition

The ELBO can be written in several equivalent ways. For a single data point $`x`$:

```math
\mathrm{ELBO}(x; Q, \theta) = \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)}
```

This can be rewritten as:

```math
\mathrm{ELBO}(x; Q, \theta) = \mathbb{E}_{z \sim Q}[\log p(x, z; \theta)] - \mathbb{E}_{z \sim Q}[\log Q(z)]
= \mathbb{E}_{z \sim Q}[\log p(x|z; \theta)] - D_{KL}(Q \| p_z)
```

where $`D_{KL}(Q \| p_z)`$ is the KL divergence between $`Q`$ and the marginal distribution $`p_z`$ of $`z`$.

Another useful form is:

```math
\mathrm{ELBO}(x; Q, \theta) = \log p(x) - D_{KL}(Q \| p_{z|x})
```

where $`p_{z|x}`$ is the posterior distribution of $`z`$ given $`x`$.

**Intuition:**
- Maximizing the ELBO with respect to $`Q`$ makes $`Q`$ close to the true posterior $`p_{z|x}`$.
- Maximizing the ELBO with respect to $`\theta`$ improves the model parameters.
- The EM algorithm can be viewed as alternating maximization of the ELBO with respect to $`Q`$ (E-step) and $`\theta`$ (M-step).

## 11.4 Mixture of Gaussians revisited

Let's revisit the mixture of Gaussians example using the general EM framework.

### E-step
- For each data point $`x^{(i)}`$, compute the "soft assignment" (responsibility) $`w_j^{(i)}`$:

```math
w_j^{(i)} = Q_i(z^{(i)} = j) = P(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma)
```

This is the probability that $`x^{(i)}`$ was generated by cluster $`j`$ under the current parameters.

### M-step
- Update the parameters $`\phi, \mu, \Sigma`$ to maximize the expected complete-data log-likelihood (the ELBO):

```math
\sum_{i=1}^n \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \phi, \mu, \Sigma)}{Q_i(z^{(i)})}
```

This expands to:

```math
= \sum_{i=1}^n \sum_{j=1}^k Q_i(z^{(i)} = j) \log \frac{p(x^{(i)}|z^{(i)} = j; \mu, \Sigma)p(z^{(i)} = j; \phi)}{Q_i(z^{(i)} = j)}
```

```math
= \sum_{i=1}^n \sum_{j=1}^k w_j^{(i)} \log \left[ \frac{1}{(2\pi)^{d/2}|\Sigma_j|^{1/2}} \exp\left(-\frac{1}{2}(x^{(i)}-\mu_j)^T\Sigma_j^{-1}(x^{(i)}-\mu_j)\right) \phi_j \right] - \log w_j^{(i)}
```

#### Update for $`\mu_j`$
- Take the derivative with respect to $`\mu_l`$ and set to zero:

```math
\mu_l := \frac{\sum_{i=1}^n w_l^{(i)} x^{(i)}}{\sum_{i=1}^n w_l^{(i)}}
```

This is a weighted mean, where each data point is weighted by its responsibility for cluster $`l`$.

#### Update for $`\phi_j`$
- Maximize

```math
\sum_{i=1}^n \sum_{j=1}^k w_j^{(i)} \log \phi_j
```

subject to $`\sum_j \phi_j = 1`$. Using a Lagrange multiplier, we get:

```math
\phi_j := \frac{1}{n} \sum_{i=1}^n w_j^{(i)}
```

This is the average responsibility for cluster $`j`$ across all data points.

#### Update for $`\Sigma_j`$
- The update for $`\Sigma_j`$ is also a weighted average (see previous notes or standard EM for Gaussians).

**Summary:**
- The EM algorithm for mixture of Gaussians alternates between computing soft assignments (E-step) and updating parameters using weighted averages (M-step).
- This general approach extends to many other models with latent variables.

## Python Implementation: General EM Algorithm and Mixture of Gaussians

Below are Python code implementations for the general EM algorithm (using the ELBO framework) and a concrete example for the Gaussian Mixture Model (GMM). Usage examples are included for both.

### General EM Algorithm Skeleton (using ELBO)
```python
import numpy as np

def general_em(X, init_params, e_step, m_step, elbo_fn, max_iters=100, tol=1e-4, verbose=False):
    """
    General EM algorithm using user-supplied E-step, M-step, and ELBO functions.
    Args:
        X: Data array (n_samples, n_features)
        init_params: Initial parameters (user-defined structure)
        e_step: Function(X, params) -> Q (latent variable distributions)
        m_step: Function(X, Q) -> params (update parameters)
        elbo_fn: Function(X, Q, params) -> float (compute ELBO)
        max_iters: Maximum number of iterations
        tol: Convergence tolerance (on ELBO)
        verbose: Print ELBO at each step
    Returns:
        params: Final parameters
        Q: Final latent variable distributions
        elbos: List of ELBO values
    """
    params = init_params
    elbos = []
    for i in range(max_iters):
        Q = e_step(X, params)
        params = m_step(X, Q)
        elbo = elbo_fn(X, Q, params)
        elbos.append(elbo)
        if verbose:
            print(f"Iteration {i+1}, ELBO: {elbo:.4f}")
        if i > 0 and abs(elbos[-1] - elbos[-2]) < tol:
            break
    return params, Q, elbos
```

### EM for Gaussian Mixture Model (GMM)
```python
from scipy.stats import multivariate_normal

def gmm_initialize(X, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n_samples, n_features = X.shape
    phi = np.ones(k) / k
    mu = X[np.random.choice(n_samples, k, replace=False)]
    Sigma = np.array([np.cov(X, rowvar=False) for _ in range(k)])
    return {'phi': phi, 'mu': mu, 'Sigma': Sigma}

def gmm_e_step(X, params):
    phi, mu, Sigma = params['phi'], params['mu'], params['Sigma']
    n_samples, n_features = X.shape
    k = phi.shape[0]
    w = np.zeros((n_samples, k))
    for j in range(k):
        rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
        w[:, j] = phi[j] * rv.pdf(X)
    w_sum = w.sum(axis=1, keepdims=True)
    w = w / w_sum
    return w

def gmm_m_step(X, w):
    n_samples, n_features = X.shape
    k = w.shape[1]
    phi = w.sum(axis=0) / n_samples
    mu = (w.T @ X) / w.sum(axis=0)[:, None]
    Sigma = np.zeros((k, n_features, n_features))
    for j in range(k):
        X_centered = X - mu[j]
        weighted = w[:, j][:, None] * X_centered
        Sigma[j] = (weighted.T @ X_centered) / w[:, j].sum()
    return {'phi': phi, 'mu': mu, 'Sigma': Sigma}

def gmm_elbo(X, w, params):
    phi, mu, Sigma = params['phi'], params['mu'], params['Sigma']
    n_samples, n_features = X.shape
    k = phi.shape[0]
    elbo = 0.0
    for i in range(n_samples):
        for j in range(k):
            if w[i, j] > 0:
                rv = multivariate_normal(mean=mu[j], cov=Sigma[j], allow_singular=True)
                log_p = np.log(phi[j] + 1e-12) + rv.logpdf(X[i])
                elbo += w[i, j] * (log_p - np.log(w[i, j] + 1e-12))
    return elbo
```

### Usage Example: GMM with EM
```python
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # Generate synthetic data
    X, y_true = make_blobs(n_samples=400, centers=3, cluster_std=0.7, random_state=42)
    k = 3

    # Initialize parameters
    init_params = gmm_initialize(X, k, seed=42)

    # Run general EM algorithm for GMM
    params, w, elbos = general_em(
        X,
        init_params,
        gmm_e_step,
        gmm_m_step,
        gmm_elbo,
        max_iters=100,
        tol=1e-4,
        verbose=True
    )

    print("Estimated mixing proportions (phi):", params['phi'])
    print("Estimated means (mu):\n", params['mu'])
    print("Estimated covariances (Sigma):\n", params['Sigma'])

    # Assign each point to the most likely cluster
    labels = np.argmax(w, axis=1)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.6)
    plt.scatter(params['mu'][:, 0], params['mu'][:, 1], c='red', s=200, marker='X', label='Estimated Means')
    plt.title('EM for Gaussian Mixture Model')
    plt.legend()
    plt.show()

    # Plot ELBO
    plt.figure()
    plt.plot(elbos, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.title('EM ELBO Convergence')
    plt.show()
```


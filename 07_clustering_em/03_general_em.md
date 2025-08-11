# General EM Algorithms: Latent Variable Models and the ELBO

## Motivation: Why General EM? The Power of Hidden Variables

Imagine you're a detective trying to solve a complex case. You have evidence (observed data) but many crucial details are missing (hidden variables). For example, you have security camera footage showing people entering and leaving a building, but you don't know their identities, motives, or relationships. The challenge is to piece together the hidden story from the visible evidence.

This is exactly what **latent variable models** do in machine learning. They help us explain observed data $`x`$ in terms of hidden variables $`z`$ that we can't directly observe but that influence what we see.

### Real-World Examples of Latent Variables

**Movie Recommendation Systems:**
- **Observed:** User ratings for movies
- **Latent:** User taste profiles, movie characteristics
- **Goal:** Predict ratings for unseen movies

**Image Recognition:**
- **Observed:** Pixel values in images
- **Latent:** Object identity, pose, lighting conditions
- **Goal:** Identify objects in new images

**Topic Modeling:**
- **Observed:** Words in documents
- **Latent:** Topics, document-topic distributions
- **Goal:** Discover themes in text collections

**Medical Diagnosis:**
- **Observed:** Symptoms, test results
- **Latent:** Underlying diseases, patient conditions
- **Goal:** Diagnose patients accurately

### Why Latent Variables Matter
- **Explanatory power:** They help us understand the underlying structure of data
- **Generalization:** Models with latent variables often generalize better
- **Uncertainty:** They naturally handle uncertainty in our observations
- **Flexibility:** They can model complex, high-dimensional relationships

## From Specific Models to General Framework: The Evolution

We've now explored **Gaussian Mixture Models** and the **Expectation-Maximization algorithm** - a powerful approach to probabilistic clustering that provides soft assignments and can model clusters of different shapes and sizes. We've seen how EM alternates between estimating cluster probabilities (E-step) and updating parameters (M-step), using the ELBO to maximize a lower bound on the likelihood.

However, while GMM provides a specific application of EM, the **Expectation-Maximization algorithm** is actually a general framework that can be applied to any model with latent variables. The principles we've learned - alternating between expectation and maximization steps, using the ELBO, and handling hidden variables - extend far beyond mixture models.

This motivates our exploration of the **general EM framework** - a flexible approach for learning in any latent variable model. We'll see how the ELBO provides a unified framework for variational inference, how to apply EM to different types of models, and how this general approach enables us to tackle a wide range of unsupervised learning problems.

The transition from GMM to general EM represents the bridge from specific application to universal framework - taking our understanding of how EM works with mixture models and extending it to any model with hidden variables.

In this section, we'll explore the mathematical foundations of the general EM framework, understand the ELBO in its most general form, and see how this framework applies to various latent variable models.

## The General Latent Variable Model: A Unified Framework

### The Mathematical Setup
Suppose we have data $`x^{(1)}, ..., x^{(n)}`$ and a model $`p(x, z; \theta)`$ (joint probability of data and latent variable, parameterized by $`\theta`$). The probability of the observed data is:

```math
p(x; \theta) = \sum_z p(x, z; \theta)
```

**What this means:**
- $`p(x, z; \theta)`$ is the joint probability of observing data $`x`$ and latent variable $`z`$
- We sum over all possible values of $`z`$ to get the marginal probability of $`x`$ alone
- This is called **marginalization** - we "integrate out" the hidden variables

### The Generative Process: How Data is Created
Think of a latent variable model as a two-stage process:

1. **Sample the latent variable:** $`z \sim p(z; \theta_z)`$
2. **Generate the data:** $`x \sim p(x|z; \theta_x)`$

**Example - Topic Modeling:**
1. Choose a topic distribution for the document: $`\theta_d \sim \text{Dirichlet}(\alpha)`$
2. For each word, choose a topic: $`z_i \sim \text{Multinomial}(\theta_d)`$
3. Generate the word: $`w_i \sim \text{Multinomial}(\beta_{z_i})`$

### The Challenge: We Only See the Data
The problem is that we only observe $`x`$, not $`z`$. We need to:
- Learn the parameters $`\theta`$ that best explain our data
- Infer the likely values of the latent variables $`z`$ for each data point

## The Log-Likelihood and Why It's Hard: The Mathematical Challenge

### The Objective: Maximizing Likelihood
We want to find parameters $`\theta`$ that maximize the probability of our observed data:

```math
\ell(\theta) = \sum_{i=1}^n \log p(x^{(i)}; \theta)
```

**Why log-likelihood?**
- **Numerical stability:** Products become sums, avoiding underflow
- **Mathematical convenience:** Derivatives are easier to compute
- **Statistical properties:** Log-likelihood has nice asymptotic properties

### The Problem: The Log-Sum Structure
The challenge is that:

```math
\log p(x; \theta) = \log \sum_z p(x, z; \theta)
```

**Why is this hard?**
- **Non-convex:** The log-sum function is not convex in general
- **No closed form:** We can't solve for the optimal parameters directly
- **Computational complexity:** Summing over all possible $`z`$ can be exponential

### The "Oracle" Scenario: If We Knew the Latent Variables
If we magically knew the values of $`z^{(i)}`$ for each data point, the problem would be much easier:

```math
\ell(\theta) = \sum_{i=1}^n \log p(x^{(i)}, z^{(i)}; \theta)
```

**Why this is easier:**
- No sums inside logs
- Each term can be optimized independently
- Often leads to closed-form solutions
- Like supervised learning where we have "labels" $`z^{(i)}`$

**Example - GMM with known assignments:**
- We can directly compute means, covariances, and mixing proportions
- Each cluster's parameters are estimated independently

## The EM Algorithm: Maximizing a Lower Bound with Elegance

### The Key Insight: Jensen's Inequality
The EM algorithm solves the log-sum problem by introducing a distribution $`Q(z)`$ over the latent variables and using **Jensen's inequality**:

```math
\log p(x; \theta) = \log \sum_z Q(z) \frac{p(x, z; \theta)}{Q(z)} \geq \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)}
```

**What is Jensen's inequality?**
For a convex function $`f`$ and any distribution $`Q`$:
```math
f(\mathbb{E}_{z \sim Q}[X]) \leq \mathbb{E}_{z \sim Q}[f(X)]
```

Since $`\log`$ is concave, we get the opposite inequality, allowing us to move the log inside the sum.

### The Evidence Lower Bound (ELBO)
The right-hand side is called the **Evidence Lower Bound (ELBO)**:

```math
\mathcal{L}(Q, \theta) = \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)}
```

**Properties of the ELBO:**
- It's a lower bound on the log-likelihood
- It's a function of both $`Q`$ and $`\theta`$
- When $`Q(z) = p(z|x; \theta)`$, the bound is tight (equality)

### Intuition: What is Q(z)?
- $`Q(z)`$ is our "guess" for the distribution of the latent variable $`z`$ given $`x``
- It represents our uncertainty about the hidden variables
- If $`Q(z)`$ matches the true posterior $`p(z|x; \theta)`$, the bound is tight

**Analogy:** Think of $`Q(z)`$ as our "working hypothesis" about what the hidden variables might be, given what we observe.

## The EM Algorithm: Step-by-Step

### The Algorithm Structure
**Initialize:** Choose starting parameters $`\theta^{(0)}`$

**Repeat until convergence:**

#### E-step (Expectation): Estimate the Posterior
For each data point $`x^{(i)}`$, compute:
```math
Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)}; \theta^{(t)})
```

**What this does:**
- Computes the posterior distribution over latent variables
- Updates our "guess" about the hidden variables
- Makes the ELBO as tight as possible for the current parameters

#### M-step (Maximization): Update Parameters
Maximize the ELBO with respect to $`\theta`$:
```math
\theta^{(t+1)} = \arg\max_\theta \sum_{i=1}^n \sum_z Q_i(z) \log p(x^{(i)}, z; \theta)
```

**What this does:**
- Updates model parameters to better explain the data
- Assumes our current posterior estimates are correct
- Improves the model fit

### Why EM Works: Convergence Guarantees
**Monotonicity:** Each iteration increases (or maintains) the log-likelihood:
```math
\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})
```

**Convergence:** The algorithm converges to a local maximum of the likelihood.

**Proof sketch:**
1. E-step: Sets $`Q`$ to make the bound tight
2. M-step: Increases the ELBO
3. Since the ELBO is a lower bound, the likelihood must also increase

## The ELBO in Different Forms: Multiple Perspectives

### Form 1: The Original Form
```math
\mathcal{L}(Q, \theta) = \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)}
```

### Form 2: Expected Log-Likelihood Minus Entropy
```math
\mathcal{L}(Q, \theta) = \mathbb{E}_{z \sim Q}[\log p(x, z; \theta)] - \mathbb{E}_{z \sim Q}[\log Q(z)]
```

**Breaking this down:**
- **First term:** Expected log-likelihood under our posterior guess
- **Second term:** Entropy of our posterior guess (encourages uncertainty)

### Form 3: Log-Likelihood Minus KL Divergence
```math
\mathcal{L}(Q, \theta) = \log p(x; \theta) - D_{KL}(Q \| p_{z|x})
```

**This is the most insightful form:**
- **First term:** The true log-likelihood (what we want to maximize)
- **Second term:** How far our guess $`Q`$ is from the true posterior $`p_{z|x}`$

**Implications:**
- Maximizing the ELBO with respect to $`Q`$ makes $`Q`$ close to the true posterior
- Maximizing with respect to $`\theta`$ improves the model parameters
- The gap between ELBO and log-likelihood is the KL divergence

### KL Divergence: Measuring Approximation Quality
The KL divergence $`D_{KL}(Q \| p_{z|x})`$ measures how close our approximation $`Q`$ is to the true posterior:

```math
D_{KL}(Q \| p_{z|x}) = \sum_z Q(z) \log \frac{Q(z)}{p(z|x; \theta)}
```

**Properties:**
- Always non-negative: $`D_{KL}(Q \| p) \geq 0`$
- Zero only when $`Q = p`$ (almost everywhere)
- Not symmetric: $`D_{KL}(Q \| p) \neq D_{KL}(p \| Q)`$

**Intuition:** KL divergence measures the "information loss" when we approximate $`p`$ with $`Q`$.

## Detailed Worked Example: Mixture of Gaussians Revisited

### The Model Setup
Recall the mixture of Gaussians model:
- **Latent variable:** $`z^{(i)}`$ (cluster assignment)
- **Observed data:** $`x^{(i)}`$ (data points)
- **Parameters:** $`\theta = \{\phi, \mu, \Sigma\}`$ (mixing proportions, means, covariances)

### The Joint Distribution
```math
p(x^{(i)}, z^{(i)}; \theta) = p(x^{(i)}|z^{(i)}; \mu, \Sigma) \cdot p(z^{(i)}; \phi)
```

Where:
- $`p(z^{(i)} = j; \phi) = \phi_j`$ (prior probability of cluster $`j`$)
- $`p(x^{(i)}|z^{(i)} = j; \mu, \Sigma) = \mathcal{N}(x^{(i)}; \mu_j, \Sigma_j)`$ (Gaussian likelihood)

### E-step: Computing the Posterior
For each data point $`i`$ and cluster $`j`$, compute:

```math
Q_i(z^{(i)} = j) = p(z^{(i)} = j | x^{(i)}; \theta) = \frac{p(x^{(i)}|z^{(i)} = j; \mu, \Sigma) \cdot p(z^{(i)} = j; \phi)}{\sum_{l=1}^k p(x^{(i)}|z^{(i)} = l; \mu, \Sigma) \cdot p(z^{(i)} = l; \phi)}
```

**This is the responsibility $`w_j^{(i)}`$ from our GMM discussion.**

### M-step: Updating Parameters
Maximize the ELBO with respect to each parameter:

**Mixing proportions:**
```math
\phi_j = \frac{1}{n} \sum_{i=1}^n Q_i(z^{(i)} = j)
```

**Means:**
```math
\mu_j = \frac{\sum_{i=1}^n Q_i(z^{(i)} = j) x^{(i)}}{\sum_{i=1}^n Q_i(z^{(i)} = j)}
```

**Covariances:**
```math
\Sigma_j = \frac{\sum_{i=1}^n Q_i(z^{(i)} = j) (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^n Q_i(z^{(i)} = j)}
```

### Numerical Example
Suppose we have 3 points: $`x^{(1)} = 1`$, $`x^{(2)} = 2``, $`x^{(3)} = 10``, and $`k=2`` clusters.

**Initialization:**
- $`\phi = [0.5, 0.5]`$, $`\mu_1 = 1`$, $`\mu_2 = 10``, $`\Sigma_1 = \Sigma_2 = 1`$

**E-step (for point 1):**
- $`p(x^{(1)}|z^{(1)} = 1) = \mathcal{N}(1; 1, 1) = 0.398`$
- $`p(x^{(1)}|z^{(1)} = 2) = \mathcal{N}(1; 10, 1) = 1.27 \times 10^{-20}`$
- $`Q_1(z^{(1)} = 1) = \frac{0.398 \times 0.5}{0.398 \times 0.5 + 1.27 \times 10^{-20} \times 0.5} = 0.999`$
- $`Q_1(z^{(1)} = 2) = 0.001`$

**M-step:**
- $`\phi_1 = \frac{0.999 + 0.999 + 0.001}{3} = 0.666`$
- $`\mu_1 = \frac{0.999 \times 1 + 0.999 \times 2 + 0.001 \times 10}{0.999 + 0.999 + 0.001} = 1.5`$

## Applications of General EM: Beyond Mixture Models

### 1. Hidden Markov Models (HMMs)
**Model:** Sequential data with hidden states
- **Observed:** Output sequence $`y_1, ..., y_T`$
- **Latent:** Hidden state sequence $`s_1, ..., s_T`$
- **Parameters:** Transition probabilities, emission probabilities, initial state distribution

**E-step:** Forward-backward algorithm to compute state posteriors
**M-step:** Update transition and emission parameters

### 2. Latent Dirichlet Allocation (LDA)
**Model:** Topic modeling for documents
- **Observed:** Words in documents
- **Latent:** Topic assignments for each word, document-topic distributions
- **Parameters:** Topic-word distributions, Dirichlet priors

**E-step:** Gibbs sampling or variational inference
**M-step:** Update topic-word distributions

### 3. Factor Analysis
**Model:** Dimensionality reduction with latent factors
- **Observed:** High-dimensional data $`x \in \mathbb{R}^d`$
- **Latent:** Low-dimensional factors $`z \in \mathbb{R}^k`$
- **Parameters:** Factor loading matrix, noise covariance

**E-step:** Compute posterior over factors
**M-step:** Update factor loadings and noise parameters

### 4. Probabilistic Principal Component Analysis (PPCA)
**Model:** Probabilistic version of PCA
- **Observed:** Data points $`x \in \mathbb{R}^d`$
- **Latent:** Principal components $`z \in \mathbb{R}^k`$
- **Parameters:** Principal component directions, noise variance

**E-step:** Compute posterior over principal components
**M-step:** Update principal component directions

## Practical Tips and Pitfalls: Real-World Considerations

### Initialization Strategies
**Random initialization:**
- Start with random parameters
- Run multiple times and pick the best result
- Simple but can be slow to converge

**Domain-specific initialization:**
- Use prior knowledge about the problem
- Initialize parameters based on data characteristics
- Often leads to faster convergence

**Greedy initialization:**
- Start with simple models and gradually increase complexity
- Useful for hierarchical models

### Convergence Monitoring
**ELBO tracking:**
- Monitor the ELBO at each iteration
- Should increase monotonically
- Plateaus indicate convergence

**Parameter stability:**
- Check if parameters change significantly
- Small changes suggest convergence

**Posterior stability:**
- Monitor changes in posterior distributions
- Stable posteriors indicate convergence

### Numerical Stability Issues
**Underflow/overflow:**
- Work in log-space when possible
- Use log-sum-exp trick for numerical stability

**Singularities:**
- Add regularization to prevent parameter collapse
- Use proper priors to constrain parameters

**Convergence to poor local optima:**
- Multiple random initializations
- Use more sophisticated optimization methods

### Model Selection
**Information criteria:**
- **BIC:** $`\text{BIC} = \log p(x|\hat{\theta}) - \frac{d}{2} \log n`$
- **AIC:** $`\text{AIC} = \log p(x|\hat{\theta}) - d`$
- Balance model fit and complexity

**Cross-validation:**
- Split data into training and validation sets
- Choose model with best validation performance

**Domain knowledge:**
- Use understanding of the problem to guide model choice
- Consider interpretability and computational cost

## Advanced Topics: Variational Inference and Beyond

### When Exact E-step is Impossible
In many modern applications, computing the exact posterior $`p(z|x; \theta)`$ is intractable. This leads to **variational inference**:

**Approximate E-step:** Instead of exact posterior, use an approximation:
```math
Q(z) \approx p(z|x; \theta)
```

**Methods:**
- **Mean-field approximation:** $`Q(z) = \prod_i Q_i(z_i)`$
- **Structured approximations:** Maintain some dependencies
- **Neural approximations:** Use neural networks to approximate the posterior

### Stochastic EM
For large datasets, we can use stochastic approximations:

**Stochastic E-step:** Use a subset of data to estimate posteriors
**Stochastic M-step:** Use stochastic gradients for parameter updates

**Benefits:**
- Scales to large datasets
- Can escape local optima
- Faster convergence in some cases

### Online EM
For streaming data, we can update parameters incrementally:

**Online E-step:** Update posteriors for new data points
**Online M-step:** Incrementally update parameters

**Applications:**
- Real-time learning
- Streaming data analysis
- Adaptive models

## Frequently Asked Questions (FAQ)

**Q: What is the difference between EM and variational inference?**
A: EM is a special case of variational inference where the E-step can be computed exactly. In general variational inference, we approximate the posterior distribution.

**Q: Why do we use the log-likelihood?**
A: The log-likelihood is easier to optimize (turns products into sums), has better numerical properties, and has nice statistical properties for model comparison.

**Q: What if the latent variable is continuous?**
A: The sums become integrals, and we may need to use approximations like variational inference or Monte Carlo methods.

**Q: Is EM guaranteed to find the global optimum?**
A: No, it can get stuck in local optima. Multiple runs with different initializations help, but there's no guarantee of finding the global optimum.

**Q: How do I choose the number of latent variables?**
A: Use model selection techniques like information criteria, cross-validation, or domain knowledge about the problem structure.

**Q: What's the computational complexity of EM?**
A: Depends on the model. For GMM: O(nkd²) per iteration. For complex models, the E-step can be the bottleneck.

**Q: Can EM handle missing data?**
A: Yes! EM naturally handles missing data by treating missing values as additional latent variables.

**Q: What's the relationship between EM and gradient descent?**
A: EM can be viewed as a form of coordinate ascent on the ELBO. Gradient-based methods can also be used, but EM often converges faster for latent variable models.

**Q: How do I know if my EM implementation is correct?**
A: Check that the ELBO increases monotonically, verify convergence, and compare results with known solutions or multiple initializations.

## Summary: The Big Picture

The general EM framework provides a powerful and flexible approach for learning in any latent variable model. Here's what we've learned:

### Key Concepts:
- **Latent variable models:** Explain observed data in terms of hidden variables
- **EM algorithm:** Alternating optimization for latent variable models
- **ELBO:** Evidence Lower Bound that makes optimization tractable
- **Variational inference:** Approximate inference when exact computation is impossible

### The EM Algorithm:
1. **E-step:** Compute posterior over latent variables
2. **M-step:** Update model parameters
3. **Repeat** until convergence

### Advantages:
- **Generality:** Applies to any latent variable model
- **Theoretical foundation:** Well-understood convergence properties
- **Flexibility:** Can handle discrete and continuous latent variables
- **Interpretability:** Provides uncertainty estimates

### Applications:
- **Clustering:** Mixture models, topic modeling
- **Dimensionality reduction:** Factor analysis, PPCA
- **Sequence modeling:** Hidden Markov models
- **Deep learning:** Variational autoencoders, generative models

### Best Practices:
- Use multiple initializations to avoid local optima
- Monitor convergence carefully
- Choose appropriate model complexity
- Handle numerical stability issues

### Limitations:
- Can get stuck in local optima
- Requires choosing model structure
- E-step can be computationally expensive
- Assumes model is correctly specified

## From Theoretical Framework to Deep Generative Models: The Next Step

We've now explored the **general EM framework** - a powerful and flexible approach for learning in any latent variable model. We've seen how the ELBO provides a unified framework for variational inference, how to apply EM to different types of models, and how this general approach enables us to tackle a wide range of unsupervised learning problems.

However, while the general EM framework provides the theoretical foundation, modern machine learning often requires dealing with complex, high-dimensional data where the posterior distributions become intractable. Traditional EM methods struggle with these scenarios because they require exact computation of the posterior, which becomes impossible with complex models like neural networks.

This motivates our exploration of **Variational Auto-Encoders (VAEs)** - deep generative models that extend the EM framework to handle complex, high-dimensional data through approximate inference. We'll see how VAEs use neural networks to approximate the posterior distribution, how the reparameterization trick enables efficient training, and how this approach enables us to build powerful generative models for images, text, and other complex data types.

The transition from general EM to VAEs represents the bridge from theoretical framework to practical deep learning - taking our understanding of latent variable models and variational inference and applying it to modern neural network architectures.

In the next section, we'll explore how VAEs work, how to implement them with neural networks, and how this approach enables powerful generative modeling and representation learning.

---

**Previous: [EM and Mixture of Gaussians](02_em_mixture_of_gaussians.md)** - Learn probabilistic clustering using the Expectation-Maximization algorithm.

**Next: [Variational Auto-Encoders](04_variational_auto-encoder.md)** - Explore deep generative models using variational inference.


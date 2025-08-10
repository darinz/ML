# General EM Algorithms: Latent Variable Models and the ELBO

## Motivation: Why General EM?

Many real-world problems involve hidden or unobserved factors. For example, imagine you have a dataset of movie ratings, but you don't know each user's taste profile (latent variable). Or you have images of handwritten digits, but you don't know which digit each image represents. Latent variable models help us explain observed data $`x`$ in terms of hidden variables $`z`$.

- **Latent variable:** A variable that is not directly observed but influences the data.
- **Goal:** Learn both the parameters of the model and the likely values of the latent variables.

## From Specific Models to General Framework

We've now explored **Gaussian Mixture Models** and the **Expectation-Maximization algorithm** - a powerful approach to probabilistic clustering that provides soft assignments and can model clusters of different shapes and sizes. We've seen how EM alternates between estimating cluster probabilities (E-step) and updating parameters (M-step), using the ELBO to maximize a lower bound on the likelihood.

However, while GMM provides a specific application of EM, the **Expectation-Maximization algorithm** is actually a general framework that can be applied to any model with latent variables. The principles we've learned - alternating between expectation and maximization steps, using the ELBO, and handling hidden variables - extend far beyond mixture models.

This motivates our exploration of the **general EM framework** - a flexible approach for learning in any latent variable model. We'll see how the ELBO provides a unified framework for variational inference, how to apply EM to different types of models, and how this general approach enables us to tackle a wide range of unsupervised learning problems.

The transition from GMM to general EM represents the bridge from specific application to universal framework - taking our understanding of how EM works with mixture models and extending it to any model with hidden variables.

In this section, we'll explore the mathematical foundations of the general EM framework, understand the ELBO in its most general form, and see how this framework applies to various latent variable models.

## The General Latent Variable Model

Suppose we have data $`x^{(1)}, ..., x^{(n)}`$ and a model $`p(x, z; \theta)`$ (joint probability of data and latent variable, parameterized by $`\theta`$). The probability of the observed data is:

```math
p(x; \theta) = \sum_z p(x, z; \theta)
```

- **Marginalization:** We sum over all possible values of $`z`$ to get the probability of $`x`$ alone.

## The Log-Likelihood and Why It's Hard

We want to maximize the log-likelihood:

```math
\ell(\theta) = \sum_{i=1}^n \log p(x^{(i)}; \theta)
```

But:

```math
\log p(x; \theta) = \log \sum_z p(x, z; \theta)
```

- The sum inside the log makes optimization hard (non-convex, no closed form).
- If we knew the $`z`$'s, the problem would be much easier (like supervised learning).

## The EM Algorithm: Maximizing a Lower Bound

The EM algorithm solves this by maximizing a **lower bound** on the log-likelihood, called the **Evidence Lower Bound (ELBO)**. The trick is to introduce a distribution $`Q(z)`$ over the latent variables and use **Jensen's inequality**:

```math
\log p(x; \theta) = \log \sum_z Q(z) \frac{p(x, z; \theta)}{Q(z)} \geq \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)}
```

- The right-hand side is the ELBO: a function of both $`Q`$ and $`\theta`$.
- **Jensen's inequality** allows us to move the log inside the sum, making the math tractable.

### Intuition: What is Q(z)?
- $`Q(z)`$ is our "guess" for the distribution of the latent variable $`z`$ given $`x`$.
- If $`Q(z)`$ matches the true posterior $`p(z|x; \theta)`$, the bound is tight (equality).

### Step-by-Step: The EM Algorithm

1. **E-step:** For each data point, set $`Q_i(z^{(i)}) = p(z^{(i)}|x^{(i)}; \theta)`$ (compute the posterior over $`z`$ given $`x`$ and current parameters).
2. **M-step:** Maximize the ELBO with respect to $`\theta`$ (update the parameters).
3. Repeat until convergence.

- Each iteration increases (or leaves unchanged) the log-likelihood.
- The algorithm is guaranteed to converge to a local optimum.

### The ELBO in Different Forms

For a single data point $`x`$:

```math
\mathrm{ELBO}(x; Q, \theta) = \sum_z Q(z) \log \frac{p(x, z; \theta)}{Q(z)}
```

This can be rewritten as:

```math
= \mathbb{E}_{z \sim Q}[\log p(x, z; \theta)] - \mathbb{E}_{z \sim Q}[\log Q(z)]
= \mathbb{E}_{z \sim Q}[\log p(x|z; \theta)] - D_{KL}(Q \| p_z)
```

- $`D_{KL}(Q \| p_z)`$ is the KL divergence between $`Q`$ and the prior $`p_z`$.
- Another form:

```math
\mathrm{ELBO}(x; Q, \theta) = \log p(x) - D_{KL}(Q \| p_{z|x})
```

- Maximizing the ELBO with respect to $`Q`$ makes $`Q`$ close to the true posterior $`p_{z|x}`$.
- Maximizing with respect to $`\theta`$ improves the model parameters.

### Worked Example: Mixture of Gaussians

Recall the mixture of Gaussians model:
- $`z^{(i)}`$ is the cluster assignment (latent variable).
- $`x^{(i)}`$ is the observed data.

**E-step:** Compute the "soft assignment" (responsibility) $`w_j^{(i)}`$:

```math
w_j^{(i)} = Q_i(z^{(i)} = j) = P(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma)
```

**M-step:** Update parameters using the weighted averages:
- $`\mu_j := \frac{\sum_{i=1}^n w_j^{(i)} x^{(i)}}{\sum_{i=1}^n w_j^{(i)}}`$
- $`\phi_j := \frac{1}{n} \sum_{i=1}^n w_j^{(i)}`$
- $`\Sigma_j := \frac{\sum_{i=1}^n w_j^{(i)} (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i=1}^n w_j^{(i)}}`$

### KL Divergence: What Does It Mean?

- The KL divergence $`D_{KL}(Q \| p_{z|x})`$ measures how close our guess $`Q`$ is to the true posterior.
- If $`Q = p_{z|x}`$, the KL divergence is zero and the ELBO equals the log-likelihood.
- In practice, we choose $`Q`$ to make the bound as tight as possible.

## Practical Tips and Pitfalls

- **Initialization matters:** EM can get stuck in local optima. Try multiple runs with different starting points.
- **Choice of Q:** In some models, $`Q`$ is restricted to a family of distributions for tractability (see variational inference).
- **Convergence:** Monitor the increase in the ELBO or log-likelihood to check for convergence.
- **Interpretation:** The ELBO provides a lower bound on the true log-likelihood; higher ELBO means a better model.

## Frequently Asked Questions (FAQ)

**Q: What is the difference between EM and variational inference?**
- EM is a special case of variational inference where the E-step can be computed exactly. In general variational inference, we approximate $`Q`$.

**Q: Why do we use the log-likelihood?**
- The log-likelihood is easier to optimize (turns products into sums) and has nice statistical properties.

**Q: What if the latent variable is continuous?**
- The sums become integrals, and we may need to use approximations (see variational autoencoders).

**Q: Is EM guaranteed to find the global optimum?**
- No, it can get stuck in local optima. Multiple runs help.

## Summary

- General EM is a powerful framework for learning in models with hidden variables.
- It alternates between estimating the distribution over hidden variables (E-step) and updating parameters (M-step).
- The ELBO provides a tractable lower bound on the log-likelihood, and maximizing it leads to better models.
- Understanding the math and intuition behind EM and the ELBO helps you apply these ideas to a wide range of problems.

## From Theoretical Framework to Deep Generative Models

We've now explored the **general EM framework** - a powerful and flexible approach for learning in any latent variable model. We've seen how the ELBO provides a unified framework for variational inference, how to apply EM to different types of models, and how this general approach enables us to tackle a wide range of unsupervised learning problems.

However, while the general EM framework provides the theoretical foundation, modern machine learning often requires dealing with complex, high-dimensional data where the posterior distributions become intractable. Traditional EM methods struggle with these scenarios because they require exact computation of the posterior, which becomes impossible with complex models like neural networks.

This motivates our exploration of **Variational Auto-Encoders (VAEs)** - deep generative models that extend the EM framework to handle complex, high-dimensional data through approximate inference. We'll see how VAEs use neural networks to approximate the posterior distribution, how the reparameterization trick enables efficient training, and how this approach enables us to build powerful generative models for images, text, and other complex data types.

The transition from general EM to VAEs represents the bridge from theoretical framework to practical deep learning - taking our understanding of latent variable models and variational inference and applying it to modern neural network architectures.

In the next section, we'll explore how VAEs work, how to implement them with neural networks, and how this approach enables powerful generative modeling and representation learning.

---

**Previous: [EM and Mixture of Gaussians](02_em_mixture_of_gaussians.md)** - Learn probabilistic clustering using the Expectation-Maximization algorithm.

**Next: [Variational Auto-Encoders](04_variational_auto-encoder.md)** - Explore deep generative models using variational inference.


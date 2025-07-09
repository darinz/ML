# Variational Inference and Variational Auto-Encoders (VAEs)

## Motivation: Why Variational Inference and VAEs?

Suppose you want to generate new images of handwritten digits, but you only have a dataset of real images. You suspect that each image is generated from some hidden (latent) factors—like the digit identity, slant, thickness, etc.—but you don't observe these directly. You want to learn a model that can both explain the data and generate new samples. This is the motivation for **variational inference** and **variational auto-encoders (VAEs)**.

- **Latent variable:** An unobserved variable that explains the structure in the data (e.g., the "style" of a digit).
- **Goal:** Learn both the generative process (how data is made) and how to infer the latent variables from data.

## The Generative Model (Decoder)

We assume data is generated as follows:
- Sample a latent variable $`z \in \mathbb{R}^k`$ from a standard normal distribution:

```math
z \sim \mathcal{N}(0, I_{k \times k})
```

- Pass $`z`$ through a neural network $`g(z; \theta)`$ (the decoder) to get the mean of the data distribution.
- Sample $`x`$ from a Gaussian centered at $`g(z; \theta)`$:

```math
x|z \sim \mathcal{N}(g(z; \theta), \sigma^2 I_{d \times d})
```

- $`\theta`$ are the parameters (weights) of the neural network.

**Intuition:**
- $`z`$ encodes the "essence" of the data (e.g., digit identity, style).
- $`g(z; \theta)`$ maps $`z`$ to the data space (e.g., an image).
- Adding Gaussian noise allows for variability in the data.

## The Challenge of Posterior Inference

Given a data point $`x`$, we want to know which $`z`$ could have generated it. This is the **posterior** $`p(z|x; \theta)`$.
- For simple models (like mixtures of Gaussians), we can compute this exactly.
- For complex models (like neural networks), the posterior is intractable (no closed form).

**Why is this hard?**
- The neural network $`g(z; \theta)`$ makes the math complicated.
- We need to approximate the posterior with a simpler distribution $`Q(z)`$.

## Variational Inference and the ELBO

We introduce a family of distributions $`\mathcal{Q}`$ and try to find the $`Q \in \mathcal{Q}`$ that is closest to the true posterior. We do this by maximizing the **Evidence Lower Bound (ELBO)**:

```math
\mathrm{ELBO}(Q, \theta) = \sum_{i=1}^n \mathbb{E}_{z^{(i)} \sim Q_i} \left[ \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} \right]
```

- The ELBO is a lower bound on the log-likelihood of the data.
- Maximizing the ELBO with respect to $`Q`$ and $`\theta`$ allows us to learn both the generative model and the approximate posterior.

**Intuition:**
- The ELBO balances two terms: how well the model reconstructs the data, and how close $`Q`$ is to the prior.

## The Mean Field Assumption

To make optimization tractable, we often assume $`Q(z)`$ is a Gaussian with independent coordinates (mean field):

```math
Q_i = \mathcal{N}(q(x^{(i)}; \phi), \text{diag}(v(x^{(i)}; \psi))^2)
```

- $`q(x^{(i)}; \phi)`$ and $`v(x^{(i)}; \psi)`$ are neural networks (the encoder) that map $`x^{(i)}`$ to the mean and variance of $`Q_i`$.
- This is called the **encoder** in VAEs.

**Why mean field?**
- It makes the math and optimization much simpler.
- Each latent variable is independent given the data.

## Encoder/Decoder Architecture

- **Encoder:** Maps data $`x`$ to parameters of $`Q(z|x)`$ (mean and variance).
- **Decoder:** Maps latent $`z`$ to data space (mean of $`p(x|z)`$).
- Both are neural networks, trained jointly.

## The ELBO for Continuous Latent Variables

For VAEs, the ELBO for a single data point is:

```math
\mathrm{ELBO}(x; Q, \theta) = \mathbb{E}_{z \sim Q(z|x)} [\log p(x|z; \theta)] - D_{KL}(Q(z|x) \| p(z))
```

- The first term is the expected log-likelihood (reconstruction quality).
- The second term is the KL divergence between the approximate posterior and the prior (regularization).

**Intuition:**
- The model tries to reconstruct the data well, but also keep $`Q(z|x)`$ close to the prior $`p(z)`$.

## The Reparameterization Trick

To train VAEs with gradient descent, we need to backpropagate through samples from $`Q(z|x)`$. The **reparameterization trick** allows this:

- Instead of sampling $`z \sim Q(z|x)`$, we sample $`\xi \sim \mathcal{N}(0, I)`$ and set:

```math
z = \mu(x) + \sigma(x) \odot \xi
```

- This makes $`z`$ a deterministic function of $`x`$, $`\mu(x)`$, $`\sigma(x)`$, and $`\xi`$, so gradients can flow through.

**Why is this important?**
- It allows us to use standard gradient-based optimization to train the model.

## Practical Tips and Pitfalls

- **Initialization matters:** VAEs can get stuck in poor local optima. Try different initializations.
- **KL annealing:** Gradually increase the weight of the KL term during training to avoid posterior collapse.
- **Posterior collapse:** Sometimes the model ignores $`z`$ and just uses the decoder. Monitor the KL term to detect this.
- **Choice of prior:** The standard normal prior is common, but other choices are possible.

## Frequently Asked Questions (FAQ)

**Q: What is the difference between a VAE and a regular autoencoder?**
- A VAE is probabilistic and learns a distribution over the latent space; a regular autoencoder learns a deterministic mapping.

**Q: Why do we use the KL divergence?**
- It measures how close the approximate posterior is to the prior, encouraging the latent space to be well-behaved.

**Q: Can VAEs generate new data?**
- Yes! Sample $`z \sim \mathcal{N}(0, I)`$ and pass it through the decoder.

**Q: What if the latent variable is discrete?**
- Special tricks (like the Gumbel-Softmax) are needed for discrete latent variables.

## Summary

- VAEs are a powerful framework for generative modeling with latent variables.
- They use variational inference to approximate the intractable posterior.
- The encoder and decoder are neural networks trained jointly to maximize the ELBO.
- The reparameterization trick enables efficient training with gradient descent.
- Understanding the math and intuition behind VAEs helps you use them effectively for generative modeling and representation learning.


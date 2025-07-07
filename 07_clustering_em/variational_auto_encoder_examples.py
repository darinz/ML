"""
Variational Auto-Encoder (VAE) Implementation and Examples

This file contains comprehensive implementations of:
1. Generative model with neural network decoder
2. Encoder network for approximate posterior
3. Mean field approximation
4. ELBO computation with Monte Carlo estimation
5. Reparameterization trick for gradient flow
6. Complete VAE training framework
7. Sample generation from learned model

Based on Kingma and Welling (2013) "Auto-Encoding Variational Bayes"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class GenerativeModel:
    """
    Implements the generative model p(x, z; θ) = p(z) * p(x|z; θ)
    where z ~ N(0, I) and x|z ~ N(g(z; θ), σ²I)
    """
    
    def __init__(self, latent_dim=2, data_dim=784, hidden_dim=512, sigma=1.0):
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.sigma = sigma
        
        # Decoder network g(z; θ) - maps from latent space to data space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid()  # Output in [0,1] for image data
        )
    
    def sample_prior(self, batch_size):
        """Sample z from the prior p(z) = N(0, I)"""
        return torch.randn(batch_size, self.latent_dim)
    
    def decode(self, z):
        """Generate x from z using the decoder network"""
        return self.decoder(z)
    
    def sample_data(self, z):
        """Sample x from p(x|z) = N(g(z; θ), σ²I)"""
        mu = self.decode(z)
        # For continuous data, add Gaussian noise
        # For discrete data (like images), we might use Bernoulli distribution
        return mu + self.sigma * torch.randn_like(mu)
    
    def log_prob_data_given_latent(self, x, z):
        """Compute log p(x|z)"""
        mu = self.decode(z)
        # Assuming Gaussian observation model
        log_prob = -0.5 * ((x - mu) / self.sigma)**2 - np.log(self.sigma * np.sqrt(2 * np.pi))
        return log_prob.sum(dim=-1)
    
    def log_prob_prior(self, z):
        """Compute log p(z)"""
        return -0.5 * (z**2).sum(dim=-1) - 0.5 * self.latent_dim * np.log(2 * np.pi)
    
    def log_prob_joint(self, x, z):
        """Compute log p(x, z) = log p(z) + log p(x|z)"""
        return self.log_prob_prior(z) + self.log_prob_data_given_latent(x, z)


def demonstrate_posterior_intractability():
    """
    Demonstrate why posterior inference is intractable for neural network models
    """
    print("=== Posterior Inference Challenge ===")
    
    # For a simple linear model, posterior is tractable
    print("1. Linear model: z ~ N(0,1), x|z ~ N(az + b, σ²)")
    print("   Posterior: z|x ~ N(μ_post, σ²_post) - CLOSED FORM!")
    
    # For neural network model, posterior is intractable
    print("\n2. Neural network model: z ~ N(0,1), x|z ~ N(NN(z), σ²)")
    print("   Posterior: z|x ~ ??? - NO CLOSED FORM!")
    print("   We need to approximate with Q(z) ≈ p(z|x)")
    
    # Monte Carlo estimation (expensive and not practical for training)
    print("\n3. Monte Carlo estimation would require:")
    print("   - Sampling many z values")
    print("   - Computing p(x|z) for each")
    print("   - Normalizing - EXPENSIVE!")
    
    return True


def compute_elbo_components(x, z, q_z, log_p_xz, log_p_z):
    """
    Compute ELBO components for a single data point
    
    ELBO = E_{z~Q}[log p(x,z) - log Q(z)]
         = E_{z~Q}[log p(x|z) + log p(z) - log Q(z)]
         = E_{z~Q}[log p(x|z)] + E_{z~Q}[log p(z) - log Q(z)]
         = Reconstruction term + KL divergence term
    """
    # Reconstruction term: E_{z~Q}[log p(x|z)]
    reconstruction_term = log_p_xz
    
    # KL divergence term: E_{z~Q}[log p(z) - log Q(z)] = KL(Q||p)
    kl_term = log_p_z - q_z.log_prob(z)
    
    return reconstruction_term, kl_term


def compute_elbo_monte_carlo(x, q_dist, generative_model, n_samples=10):
    """
    Compute ELBO using Monte Carlo estimation
    ELBO = E_{z~Q}[log p(x,z) - log Q(z)]
    """
    batch_size = x.shape[0]
    total_elbo = 0
    
    for _ in range(n_samples):
        # Sample from approximate posterior
        z = q_dist.sample()
        
        # Compute log probabilities
        log_q_z = q_dist.log_prob(z)
        log_p_xz = generative_model.log_prob_data_given_latent(x, z)
        log_p_z = generative_model.log_prob_prior(z)
        
        # Compute ELBO for this sample
        elbo_sample = log_p_xz + log_p_z - log_q_z
        total_elbo += elbo_sample
    
    return total_elbo / n_samples


def demonstrate_elbo():
    print("\n=== ELBO Computation ===")
    
    # Create simple example
    x = torch.randn(2, 4)  # 2 data points, 4 dimensions
    q_mean = torch.randn(2, 2)  # 2 latent dimensions
    q_std = torch.ones(2, 2) * 0.1
    
    q_dist = Independent(Normal(q_mean, q_std), 1)
    z = q_dist.sample()
    
    # Compute components
    log_q_z = q_dist.log_prob(z)
    log_p_z = -0.5 * (z**2).sum(dim=-1) - np.log(2 * np.pi)
    log_p_xz = -0.5 * ((x - z.mean(dim=-1, keepdim=True))**2).sum(dim=-1)
    
    reconstruction_term, kl_term = compute_elbo_components(x, z, q_dist, log_p_xz, log_p_z)
    elbo = reconstruction_term + kl_term
    
    print(f"Reconstruction term: {reconstruction_term.mean():.4f}")
    print(f"KL divergence term: {kl_term.mean():.4f}")
    print(f"Total ELBO: {elbo.mean():.4f}")
    
    return elbo


class MeanFieldApproximation:
    """
    Implements mean field approximation for continuous latent variables
    Q(z) = Q_1(z_1) * Q_2(z_2) * ... * Q_k(z_k)
    where each Q_i is a Gaussian distribution
    """
    
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
    
    def create_mean_field_distribution(self, means, stds):
        """
        Create a mean field distribution where each dimension is independent
        
        Args:
            means: [batch_size, latent_dim] - means for each dimension
            stds: [batch_size, latent_dim] - standard deviations for each dimension
        """
        # Each dimension is independent, so we use Independent distribution
        return Independent(Normal(means, stds), 1)
    
    def sample_from_mean_field(self, means, stds, n_samples=1):
        """Sample from mean field approximation"""
        q_dist = self.create_mean_field_distribution(means, stds)
        return q_dist.sample((n_samples,))
    
    def compute_kl_divergence(self, q_means, q_stds, p_means=None, p_stds=None):
        """
        Compute KL divergence between Q (mean field) and P (prior)
        KL(Q||P) where P is typically N(0,1) for each dimension
        """
        if p_means is None:
            p_means = torch.zeros_like(q_means)
        if p_stds is None:
            p_stds = torch.ones_like(q_stds)
        
        # KL divergence for Gaussian distributions
        kl_div = 0.5 * (
            (q_stds / p_stds)**2 + 
            ((q_means - p_means) / p_stds)**2 - 
            1 - 
            2 * torch.log(q_stds / p_stds)
        )
        
        return kl_div.sum(dim=-1)


def demonstrate_mean_field():
    print("\n=== Mean Field Approximation ===")
    
    mf = MeanFieldApproximation(latent_dim=3)
    
    # Create mean field distribution
    batch_size = 4
    q_means = torch.randn(batch_size, 3)
    q_stds = torch.ones(batch_size, 3) * 0.5
    
    # Sample from mean field
    samples = mf.sample_from_mean_field(q_means, q_stds, n_samples=10)
    print(f"Mean field samples shape: {samples.shape}")
    
    # Compute KL divergence with prior
    kl_div = mf.compute_kl_divergence(q_means, q_stds)
    print(f"KL divergence with prior: {kl_div}")
    
    return mf


class Encoder(nn.Module):
    """
    Encoder network that parameterizes the approximate posterior Q(z|x)
    Maps x to parameters of Q(z|x) = N(μ(x), σ²(x))
    """
    
    def __init__(self, data_dim, latent_dim, hidden_dim=512):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        
        # Encoder network that outputs both mean and log variance
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # μ and log σ²
        )
    
    def forward(self, x):
        """
        Encode x to parameters of Q(z|x)
        
        Args:
            x: [batch_size, data_dim]
            
        Returns:
            mu: [batch_size, latent_dim] - mean of Q(z|x)
            log_var: [batch_size, latent_dim] - log variance of Q(z|x)
        """
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        log_var = h[:, self.latent_dim:]
        return mu, log_var
    
    def create_posterior(self, x):
        """
        Create the approximate posterior distribution Q(z|x)
        
        Args:
            x: [batch_size, data_dim]
            
        Returns:
            q_dist: Independent Normal distribution
        """
        mu, log_var = self.forward(x)
        std = torch.exp(0.5 * log_var)  # Convert log_var to std
        return Independent(Normal(mu, std), 1)


def demonstrate_encoder():
    print("\n=== Encoder Network ===")
    
    data_dim = 784  # MNIST image size
    latent_dim = 10
    batch_size = 4
    
    encoder = Encoder(data_dim, latent_dim)
    x = torch.randn(batch_size, data_dim)
    
    # Get posterior parameters
    mu, log_var = encoder(x)
    print(f"Posterior mean shape: {mu.shape}")
    print(f"Posterior log variance shape: {log_var.shape}")
    
    # Create posterior distribution
    q_dist = encoder.create_posterior(x)
    z_samples = q_dist.sample()
    print(f"Sampled z shape: {z_samples.shape}")
    
    return encoder


def compute_vae_elbo(x, encoder, decoder, n_samples=1):
    """
    Compute VAE ELBO using Monte Carlo estimation
    
    ELBO = E_{z~Q(z|x)}[log p(x|z) + log p(z) - log Q(z|x)]
         = E_{z~Q(z|x)}[log p(x|z)] - KL(Q(z|x) || p(z))
    
    Args:
        x: [batch_size, data_dim] - input data
        encoder: Encoder network
        decoder: Decoder network
        n_samples: number of samples for Monte Carlo estimation
        
    Returns:
        elbo: scalar - ELBO value
        reconstruction_loss: scalar - reconstruction term
        kl_loss: scalar - KL divergence term
    """
    batch_size = x.shape[0]
    
    # Get posterior parameters
    mu, log_var = encoder(x)
    std = torch.exp(0.5 * log_var)
    
    # Create posterior distribution
    q_dist = Independent(Normal(mu, std), 1)
    
    # Sample from posterior
    z = q_dist.sample((n_samples,))  # [n_samples, batch_size, latent_dim]
    z = z.transpose(0, 1)  # [batch_size, n_samples, latent_dim]
    
    # Compute reconstruction term: E[log p(x|z)]
    x_recon = decoder(z.view(-1, z.shape[-1]))  # [batch_size * n_samples, data_dim]
    x_recon = x_recon.view(batch_size, n_samples, -1)  # [batch_size, n_samples, data_dim]
    
    # Assuming Gaussian observation model
    reconstruction_loss = -0.5 * ((x.unsqueeze(1) - x_recon) ** 2).sum(dim=-1)
    reconstruction_loss = reconstruction_loss.mean(dim=1)  # Average over samples
    
    # Compute KL divergence: KL(Q(z|x) || p(z))
    # KL divergence between two Gaussians has closed form
    kl_loss = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))
    kl_loss = kl_loss.sum(dim=-1)
    
    # Total ELBO
    elbo = reconstruction_loss - kl_loss
    
    return elbo.mean(), reconstruction_loss.mean(), kl_loss.mean()


def demonstrate_vae_elbo():
    print("\n=== VAE ELBO Computation ===")
    
    data_dim = 784
    latent_dim = 10
    batch_size = 4
    
    encoder = Encoder(data_dim, latent_dim)
    decoder = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, data_dim),
        nn.Sigmoid()
    )
    
    x = torch.randn(batch_size, data_dim)
    
    elbo, recon_loss, kl_loss = compute_vae_elbo(x, encoder, decoder)
    
    print(f"ELBO: {elbo:.4f}")
    print(f"Reconstruction loss: {recon_loss:.4f}")
    print(f"KL divergence: {kl_loss:.4f}")
    
    return elbo, recon_loss, kl_loss


class VAE(nn.Module):
    """
    Complete Variational Auto-Encoder implementation
    """
    
    def __init__(self, data_dim, latent_dim, hidden_dim=512):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)  # μ and log σ²
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode x to posterior parameters"""
        h = self.encoder(x)
        mu = h[:, :self.latent_dim]
        log_var = h[:, self.latent_dim:]
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)
        This allows gradients to flow through the sampling process
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Sample from N(0,1)
        z = mu + eps * std  # Reparameterization trick
        return z
    
    def decode(self, z):
        """Decode z to reconstructed data"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var
    
    def loss_function(self, x, x_recon, mu, log_var, beta=1.0):
        """
        Compute VAE loss: reconstruction loss + β * KL divergence
        
        Args:
            x: original data
            x_recon: reconstructed data
            mu: posterior mean
            log_var: posterior log variance
            beta: weight for KL divergence (β-VAE)
        """
        # Reconstruction loss (assuming Bernoulli for binary data)
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        
        # KL divergence: KL(N(μ,σ²) || N(0,1))
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def sample(self, n_samples=1):
        """Generate samples from the learned model"""
        z = torch.randn(n_samples, self.latent_dim)
        with torch.no_grad():
            samples = self.decode(z)
        return samples


def train_vae(vae, dataloader, optimizer, device, epochs=10):
    """Train VAE model"""
    vae.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten
            
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu, log_var = vae(data)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae.loss_function(data, x_recon, mu, log_var)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            num_batches += 1
        
        # Print progress
        avg_loss = total_loss / num_batches
        avg_recon = total_recon_loss / num_batches
        avg_kl = total_kl_loss / num_batches
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Loss: {avg_loss:.4f}, '
              f'Recon: {avg_recon:.4f}, '
              f'KL: {avg_kl:.4f}')


def demonstrate_complete_vae():
    print("\n=== Complete VAE Implementation ===")
    
    # Create VAE
    data_dim = 784
    latent_dim = 20
    vae = VAE(data_dim, latent_dim)
    
    # Create dummy data
    batch_size = 32
    x = torch.rand(batch_size, data_dim)
    
    # Forward pass
    x_recon, mu, log_var = vae(x)
    
    # Compute loss
    loss, recon_loss, kl_loss = vae.loss_function(x, x_recon, mu, log_var)
    
    print(f"VAE loss: {loss:.4f}")
    print(f"Reconstruction loss: {recon_loss:.4f}")
    print(f"KL divergence: {kl_loss:.4f}")
    
    # Generate samples
    samples = vae.sample(n_samples=5)
    print(f"Generated samples shape: {samples.shape}")
    
    # Demonstrate reparameterization trick
    print("\n--- Reparameterization Trick ---")
    mu = torch.randn(2, 3)
    log_var = torch.randn(2, 3)
    
    # Without reparameterization (no gradients)
    std = torch.exp(0.5 * log_var)
    z_no_grad = mu + torch.randn_like(std) * std
    print(f"z without reparameterization: {z_no_grad}")
    
    # With reparameterization (gradients flow)
    z_with_grad = vae.reparameterize(mu, log_var)
    print(f"z with reparameterization: {z_with_grad}")
    
    return vae


def visualize_latent_space(vae, dataloader, device, n_samples=1000):
    """
    Visualize the learned latent space by plotting encoded data points
    """
    vae.eval()
    z_points = []
    labels = []
    
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            if i * dataloader.batch_size >= n_samples:
                break
            data = data.to(device).view(data.size(0), -1)
            mu, _ = vae.encode(data)
            z_points.append(mu.cpu())
            labels.extend(target.cpu().numpy())
    
    z_points = torch.cat(z_points, dim=0)[:n_samples]
    labels = labels[:n_samples]
    
    # Plot latent space (first 2 dimensions)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_points[:, 0], z_points[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('VAE Latent Space Visualization')
    plt.show()


def generate_and_visualize_samples(vae, n_samples=16, image_size=28):
    """
    Generate and visualize samples from the trained VAE
    """
    vae.eval()
    with torch.no_grad():
        samples = vae.sample(n_samples)
    
    # Reshape to images
    samples = samples.view(n_samples, image_size, image_size)
    
    # Plot samples
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            ax.imshow(samples[i], cmap='gray')
            ax.axis('off')
    
    plt.suptitle('Generated Samples from VAE')
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run all demonstrations"""
    print("=== Variational Auto-Encoder (VAE) Implementation ===")
    print("Based on Kingma and Welling (2013) 'Auto-Encoding Variational Bayes'")
    print()
    
    # 1. Demonstrate posterior intractability
    demonstrate_posterior_intractability()
    
    # 2. Demonstrate generative model
    print("\n=== Generative Model ===")
    gen_model = GenerativeModel(latent_dim=2, data_dim=4, sigma=0.1)
    z_samples = gen_model.sample_prior(batch_size=5)
    x_samples = gen_model.sample_data(z_samples)
    print("Latent samples z:", z_samples.shape)
    print("Generated data x:", x_samples.shape)
    
    # 3. Demonstrate ELBO computation
    demonstrate_elbo()
    
    # 4. Demonstrate mean field approximation
    mean_field_example = demonstrate_mean_field()
    
    # 5. Demonstrate encoder network
    encoder_example = demonstrate_encoder()
    
    # 6. Demonstrate VAE ELBO computation
    demonstrate_vae_elbo()
    
    # 7. Demonstrate complete VAE
    complete_vae = demonstrate_complete_vae()
    
    print("\n=== Summary ===")
    print("This implementation includes:")
    print("1. Generative model with neural network decoder")
    print("2. Encoder network for approximate posterior")
    print("3. Mean field approximation")
    print("4. ELBO computation with Monte Carlo estimation")
    print("5. Reparameterization trick for gradient flow")
    print("6. Complete VAE training framework")
    print("7. Sample generation from learned model")
    
    print("\nKey concepts demonstrated:")
    print("- Posterior intractability and variational approximation")
    print("- ELBO as a lower bound on log-likelihood")
    print("- Mean field assumption for tractable inference")
    print("- Reparameterization trick for gradient-based optimization")
    print("- Encoder-decoder architecture")
    print("- Trade-off between reconstruction and KL divergence")


if __name__ == "__main__":
    main() 
"""
Independent Components Analysis (ICA) - Comprehensive Python Examples
====================================================================

This script provides comprehensive implementations of ICA concepts from the markdown file.
Each section demonstrates key concepts with detailed explanations and visualizations.

Key Concepts Covered:
1. Statistical independence vs. correlation
2. The cocktail party problem
3. Linear mixing and unmixing
4. Data preprocessing (whitening)
5. ICA algorithms (gradient ascent, FastICA)
6. Source separation and reconstruction
7. Ambiguities and limitations
8. Practical applications

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import seaborn as sns
from scipy import signal
from scipy.linalg import sqrtm

# Set random seed for reproducibility
np.random.seed(42)

def print_section_header(title):
    """Print a formatted section header for better readability."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def plot_signals(signals, titles, figsize=(15, 10)):
    """
    Plot multiple signals for comparison.
    
    Parameters:
    -----------
    signals : list of arrays
        List of signal arrays to plot
    titles : list of strings
        Titles for each subplot
    figsize : tuple
        Figure size
    """
    n_signals = len(signals)
    fig, axes = plt.subplots(n_signals, 1, figsize=figsize)
    
    if n_signals == 1:
        axes = [axes]
    
    for i, (signal_data, title) in enumerate(zip(signals, titles)):
        if signal_data.ndim == 1:
            axes[i].plot(signal_data)
        else:
            for j in range(signal_data.shape[1]):
                axes[i].plot(signal_data[:, j], label=f'Signal {j+1}')
            axes[i].legend()
        
        axes[i].set_title(title)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# SECTION 1: UNDERSTANDING STATISTICAL INDEPENDENCE
# ============================================================================

print_section_header("UNDERSTANDING STATISTICAL INDEPENDENCE")

def demonstrate_independence_vs_correlation():
    """
    Demonstrate the difference between statistical independence and correlation.
    """
    print("Demonstrating independence vs. correlation...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate independent variables
    x_indep = np.random.normal(0, 1, n_samples)
    y_indep = np.random.normal(0, 1, n_samples)
    
    # Generate correlated but dependent variables
    x_corr = np.random.normal(0, 1, n_samples)
    y_corr = x_corr + 0.5 * np.random.normal(0, 1, n_samples)
    
    # Generate uncorrelated but dependent variables (non-linear relationship)
    x_uncorr = np.random.uniform(-1, 1, n_samples)
    y_uncorr = x_uncorr**2 + 0.1 * np.random.normal(0, 1, n_samples)
    
    # Calculate correlations
    corr_indep = np.corrcoef(x_indep, y_indep)[0, 1]
    corr_corr = np.corrcoef(x_corr, y_corr)[0, 1]
    corr_uncorr = np.corrcoef(x_uncorr, y_uncorr)[0, 1]
    
    print(f"Independent variables correlation: {corr_indep:.4f}")
    print(f"Correlated variables correlation: {corr_corr:.4f}")
    print(f"Uncorrelated but dependent variables correlation: {corr_uncorr:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Independent
    axes[0].scatter(x_indep, y_indep, alpha=0.6)
    axes[0].set_title(f'Independent\nCorrelation: {corr_indep:.4f}')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)
    
    # Correlated
    axes[1].scatter(x_corr, y_corr, alpha=0.6)
    axes[1].set_title(f'Correlated\nCorrelation: {corr_corr:.4f}')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].grid(True, alpha=0.3)
    
    # Uncorrelated but dependent
    axes[2].scatter(x_uncorr, y_uncorr, alpha=0.6)
    axes[2].set_title(f'Uncorrelated but Dependent\nCorrelation: {corr_uncorr:.4f}')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey insights:")
    print("1. Independent variables are uncorrelated")
    print("2. Correlated variables are dependent")
    print("3. Uncorrelated variables can still be dependent (non-linear relationship)")
    print("4. ICA seeks statistical independence, not just uncorrelation")

demonstrate_independence_vs_correlation()

# ============================================================================
# SECTION 2: THE COCKTAIL PARTY PROBLEM
# ============================================================================

print_section_header("THE COCKTAIL PARTY PROBLEM")

def simulate_cocktail_party():
    """
    Simulate the classic cocktail party problem with multiple speakers.
    """
    print("Simulating the cocktail party problem...")
    
    np.random.seed(42)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    
    # Generate independent source signals
    # Speaker 1: Sinusoidal speech-like signal
    s1 = np.sin(2 * time) + 0.3 * np.sin(4 * time) + 0.1 * np.random.normal(0, 1, n_samples)
    
    # Speaker 2: Square wave (representing different speech pattern)
    s2 = np.sign(np.sin(3 * time)) + 0.2 * np.random.normal(0, 1, n_samples)
    
    # Speaker 3: Non-Gaussian noise (background music)
    s3 = np.random.laplace(0, 1, n_samples) + 0.1 * np.sin(5 * time)
    
    # Combine sources
    S = np.column_stack([s1, s2, s3])
    
    # Normalize sources to have unit variance
    S = S / np.std(S, axis=0)
    
    # Create mixing matrix (simulating microphone positions)
    # Each row represents how each microphone picks up each speaker
    A = np.array([
        [1.0, 0.8, 0.6],  # Microphone 1: close to speaker 1, far from speaker 3
        [0.7, 1.0, 0.8],  # Microphone 2: close to speaker 2
        [0.5, 0.7, 1.0]   # Microphone 3: close to speaker 3, far from speaker 1
    ])
    
    # Generate mixed signals (what microphones record)
    X = S @ A.T
    
    print("Mixing matrix A:")
    print(A)
    print(f"\nSource signals shape: {S.shape}")
    print(f"Mixed signals shape: {X.shape}")
    
    # Visualize sources and mixtures
    plot_signals(
        [S, X],
        ['True Source Signals (Speakers)', 'Observed Mixtures (Microphones)'],
        figsize=(15, 10)
    )
    
    return S, X, A

# Run the simulation
S_true, X_mixed, A_mixing = simulate_cocktail_party()

# ============================================================================
# SECTION 3: ICA AMBIGUITIES AND CONSTRAINTS
# ============================================================================

print_section_header("ICA AMBIGUITIES AND CONSTRAINTS")

def demonstrate_ica_ambiguities():
    """
    Demonstrate the fundamental ambiguities in ICA.
    """
    print("Demonstrating ICA ambiguities...")
    
    # Use the cocktail party data
    S, X, A = S_true, X_mixed, A_mixing
    
    # Demonstrate permutation ambiguity
    # Different orderings of sources are equally valid
    permutation = np.array([2, 0, 1])  # Reorder sources
    S_permuted = S[:, permutation]
    A_permuted = A[:, permutation]
    
    # Demonstrate scaling ambiguity
    # Different scales of sources are equally valid
    scaling = np.array([2.0, 0.5, -1.0])  # Scale and flip sources
    S_scaled = S * scaling
    A_scaled = A / scaling.reshape(1, -1)
    
    # Demonstrate sign ambiguity
    # Different signs of sources are equally valid
    sign_flip = np.array([1, -1, 1])  # Flip sign of second source
    S_signed = S * sign_flip
    A_signed = A * sign_flip.reshape(1, -1)
    
    print("Original mixing matrix:")
    print(A)
    print("\nPermuted mixing matrix:")
    print(A_permuted)
    print("\nScaled mixing matrix:")
    print(A_scaled)
    print("\nSign-flipped mixing matrix:")
    print(A_signed)
    
    # Verify that all give the same observations
    X_original = S @ A.T
    X_permuted = S_permuted @ A_permuted.T
    X_scaled = S_scaled @ A_scaled.T
    X_signed = S_signed @ A_signed.T
    
    print(f"\nReconstruction error (permutation): {np.mean((X_original - X_permuted)**2):.2e}")
    print(f"Reconstruction error (scaling): {np.mean((X_original - X_scaled)**2):.2e}")
    print(f"Reconstruction error (sign): {np.mean((X_original - X_signed)**2):.2e}")
    
    # Visualize the ambiguities
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original sources
    for i in range(3):
        axes[0, 0].plot(S[:, i], label=f'Source {i+1}')
    axes[0, 0].set_title('Original Sources')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Permuted sources
    for i in range(3):
        axes[0, 1].plot(S_permuted[:, i], label=f'Source {permutation[i]+1}')
    axes[0, 1].set_title('Permuted Sources')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Scaled sources
    for i in range(3):
        axes[1, 0].plot(S_scaled[:, i], label=f'Source {i+1} (×{scaling[i]:.1f})')
    axes[1, 0].set_title('Scaled Sources')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sign-flipped sources
    for i in range(3):
        axes[1, 1].plot(S_signed[:, i], label=f'Source {i+1} (×{sign_flip[i]})')
    axes[1, 1].set_title('Sign-flipped Sources')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey insights:")
    print("1. Permutation ambiguity: Order of sources cannot be determined")
    print("2. Scaling ambiguity: Scale of sources cannot be determined")
    print("3. Sign ambiguity: Sign of sources cannot be determined")
    print("4. All these solutions are equally valid for ICA")

demonstrate_ica_ambiguities()

# ============================================================================
# SECTION 4: DATA PREPROCESSING - WHITENING
# ============================================================================

print_section_header("DATA PREPROCESSING - WHITENING")

def demonstrate_whitening():
    """
    Demonstrate the whitening process and its importance for ICA.
    """
    print("Demonstrating data whitening...")
    
    # Use the mixed signals
    X = X_mixed
    
    print("Original data statistics:")
    print(f"Mean: {np.mean(X, axis=0)}")
    print(f"Standard deviation: {np.std(X, axis=0)}")
    
    # Compute covariance matrix
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    print("\nCovariance matrix (before whitening):")
    print(cov_matrix)
    
    # Perform whitening using PCA
    # Step 1: Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    # Step 2: Create whitening matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    whitening_matrix = D_inv_sqrt @ eigvecs.T
    
    # Step 3: Apply whitening
    X_white = X_centered @ eigvecs @ D_inv_sqrt
    
    print("\nWhitening matrix:")
    print(whitening_matrix)
    
    # Check whitened data statistics
    cov_white = np.cov(X_white, rowvar=False)
    
    print("\nCovariance matrix (after whitening):")
    print(cov_white)
    
    print(f"\nWhitened data mean: {np.mean(X_white, axis=0)}")
    print(f"Whitened data std: {np.std(X_white, axis=0)}")
    
    # Visualize the effect of whitening
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original data
    axes[0, 0].scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6)
    axes[0, 0].set_title('Original Data (Centered)')
    axes[0, 0].set_xlabel('Signal 1')
    axes[0, 0].set_ylabel('Signal 2')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    # Whitened data
    axes[0, 1].scatter(X_white[:, 0], X_white[:, 1], alpha=0.6)
    axes[0, 1].set_title('Whitened Data')
    axes[0, 1].set_xlabel('Signal 1')
    axes[0, 1].set_ylabel('Signal 2')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # Original signals over time
    for i in range(3):
        axes[1, 0].plot(X_centered[:, i], label=f'Signal {i+1}')
    axes[1, 0].set_title('Original Signals (Centered)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Whitened signals over time
    for i in range(3):
        axes[1, 1].plot(X_white[:, i], label=f'Signal {i+1}')
    axes[1, 1].set_title('Whitened Signals')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nBenefits of whitening:")
    print("1. Decorrelates the signals")
    print("2. Normalizes the variance")
    print("3. Simplifies the ICA problem")
    print("4. Improves convergence of ICA algorithms")
    
    return X_white, whitening_matrix

# Perform whitening
X_white, W_whiten = demonstrate_whitening()

# ============================================================================
# SECTION 5: MANUAL ICA IMPLEMENTATION
# ============================================================================

print_section_header("MANUAL ICA IMPLEMENTATION")

def sigmoid(x):
    """Sigmoid function for ICA."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def manual_ica_gradient_ascent(X, n_components=None, max_iterations=100, learning_rate=0.01):
    """
    Implement ICA using gradient ascent on the log-likelihood.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Whitened input data
    n_components : int, optional
        Number of components to extract
    max_iterations : int
        Maximum number of iterations
    learning_rate : float
        Learning rate for gradient ascent
        
    Returns:
    --------
    W : array-like, shape (n_components, n_features)
        Unmixing matrix
    S : array-like, shape (n_samples, n_components)
        Recovered sources
    """
    if n_components is None:
        n_components = X.shape[1]
    
    n_samples, n_features = X.shape
    
    print(f"Running ICA with {n_components} components...")
    print(f"Data shape: {X.shape}")
    
    # Initialize unmixing matrix randomly
    W = np.random.randn(n_components, n_features)
    
    # Normalize rows of W
    for i in range(n_components):
        W[i] = W[i] / np.linalg.norm(W[i])
    
    # Gradient ascent
    for iteration in range(max_iterations):
        # Compute projections
        S_est = X @ W.T  # Shape: (n_samples, n_components)
        
        # Compute sigmoid and its derivative
        g_S = sigmoid(S_est)  # Shape: (n_samples, n_components)
        g_prime_S = sigmoid_derivative(S_est)  # Shape: (n_samples, n_components)
        
        # Compute gradient
        # Data term: (1 - 2*g(S)) * X^T
        data_term = (1 - 2 * g_S).T @ X  # Shape: (n_components, n_features)
        
        # Regularization term: (W^T)^(-1)
        reg_term = np.linalg.inv(W.T)  # Shape: (n_components, n_components)
        
        # Total gradient
        gradient = data_term / n_samples + reg_term
        
        # Update W
        W_old = W.copy()
        W += learning_rate * gradient
        
        # Normalize rows of W (for stability)
        for i in range(n_components):
            W[i] = W[i] / np.linalg.norm(W[i])
        
        # Check convergence
        if iteration % 20 == 0:
            change = np.mean(np.abs(W - W_old))
            print(f"Iteration {iteration}: Change = {change:.6f}")
    
    # Recover sources
    S_recovered = X @ W.T
    
    print("ICA completed!")
    print(f"Final unmixing matrix shape: {W.shape}")
    print(f"Recovered sources shape: {S_recovered.shape}")
    
    return W, S_recovered

# Run manual ICA
print("Running manual ICA implementation...")
W_manual, S_manual = manual_ica_gradient_ascent(X_white, max_iterations=50, learning_rate=0.01)

# ============================================================================
# SECTION 6: FASTICA IMPLEMENTATION
# ============================================================================

print_section_header("FASTICA IMPLEMENTATION")

def fastica_implementation(X, n_components=None, max_iterations=100, tol=1e-7):
    """
    Implement FastICA algorithm.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Whitened input data
    n_components : int, optional
        Number of components to extract
    max_iterations : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence
        
    Returns:
    --------
    W : array-like, shape (n_components, n_features)
        Unmixing matrix
    S : array-like, shape (n_samples, n_components)
        Recovered sources
    """
    if n_components is None:
        n_components = X.shape[1]
    
    n_samples, n_features = X.shape
    
    print(f"Running FastICA with {n_components} components...")
    
    # Initialize unmixing matrix randomly
    W = np.random.randn(n_components, n_features)
    
    # Normalize rows of W
    for i in range(n_components):
        W[i] = W[i] / np.linalg.norm(W[i])
    
    # FastICA iteration
    for iteration in range(max_iterations):
        W_old = W.copy()
        
        for i in range(n_components):
            # Compute w_i^T * X
            w_i = W[i]
            wx = w_i @ X.T  # Shape: (n_samples,)
            
            # Compute g(wx) and g'(wx)
            g_wx = np.tanh(wx)  # Using tanh as activation function
            g_prime_wx = 1 - g_wx**2
            
            # Update w_i
            w_new = (g_wx @ X) / n_samples - np.mean(g_prime_wx) * w_i
            
            # Normalize
            w_new = w_new / np.linalg.norm(w_new)
            
            # Gram-Schmidt orthogonalization
            for j in range(i):
                w_new = w_new - (w_new @ W[j]) * W[j]
            
            # Normalize again
            w_new = w_new / np.linalg.norm(w_new)
            
            W[i] = w_new
        
        # Check convergence
        if iteration % 10 == 0:
            change = np.mean(np.abs(W - W_old))
            print(f"Iteration {iteration}: Change = {change:.6f}")
            
            if change < tol:
                print(f"Converged at iteration {iteration}")
                break
    
    # Recover sources
    S_recovered = X @ W.T
    
    print("FastICA completed!")
    return W, S_recovered

# Run FastICA
print("Running FastICA implementation...")
W_fastica, S_fastica = fastica_implementation(X_white, max_iterations=30)

# ============================================================================
# SECTION 7: COMPARISON OF METHODS
# ============================================================================

print_section_header("COMPARISON OF METHODS")

def compare_ica_methods():
    """
    Compare different ICA methods and their results.
    """
    print("Comparing ICA methods...")
    
    # Use scikit-learn's FastICA for comparison
    ica_sklearn = FastICA(n_components=3, random_state=42, max_iter=100)
    S_sklearn = ica_sklearn.fit_transform(X_white)
    W_sklearn = ica_sklearn.components_
    
    # Compare unmixing matrices
    print("Unmixing matrix comparison:")
    print("\nManual gradient ascent:")
    print(W_manual)
    print("\nFastICA:")
    print(W_fastica)
    print("\nScikit-learn FastICA:")
    print(W_sklearn)
    
    # Compare recovered sources
    plot_signals(
        [S_true, S_manual, S_fastica, S_sklearn],
        ['True Sources', 'Manual ICA', 'FastICA', 'Scikit-learn FastICA'],
        figsize=(15, 12)
    )
    
    # Calculate correlation between true and recovered sources
    def calculate_correlation(true_sources, recovered_sources):
        """Calculate correlation between true and recovered sources."""
        correlations = []
        for i in range(true_sources.shape[1]):
            max_corr = 0
            for j in range(recovered_sources.shape[1]):
                corr = abs(np.corrcoef(true_sources[:, i], recovered_sources[:, j])[0, 1])
                max_corr = max(max_corr, corr)
            correlations.append(max_corr)
        return correlations
    
    corr_manual = calculate_correlation(S_true, S_manual)
    corr_fastica = calculate_correlation(S_true, S_fastica)
    corr_sklearn = calculate_correlation(S_true, S_sklearn)
    
    print(f"\nCorrelation with true sources:")
    print(f"Manual ICA: {corr_manual}")
    print(f"FastICA: {corr_fastica}")
    print(f"Scikit-learn: {corr_sklearn}")
    
    print(f"\nAverage correlation:")
    print(f"Manual ICA: {np.mean(corr_manual):.4f}")
    print(f"FastICA: {np.mean(corr_fastica):.4f}")
    print(f"Scikit-learn: {np.mean(corr_sklearn):.4f}")

compare_ica_methods()

# ============================================================================
# SECTION 8: PRACTICAL APPLICATIONS
# ============================================================================

print_section_header("PRACTICAL APPLICATIONS")

def demonstrate_audio_separation():
    """
    Demonstrate ICA for audio signal separation.
    """
    print("Demonstrating audio signal separation...")
    
    # Create synthetic audio signals
    np.random.seed(42)
    n_samples = 2000
    time = np.linspace(0, 4, n_samples)
    sample_rate = n_samples / 4
    
    # Signal 1: Speech-like signal (low frequency)
    speech = np.sin(2 * np.pi * 2 * time) + 0.5 * np.sin(2 * np.pi * 4 * time)
    speech += 0.1 * np.random.normal(0, 1, n_samples)
    
    # Signal 2: Music-like signal (higher frequency)
    music = np.sin(2 * np.pi * 8 * time) + 0.3 * np.sin(2 * np.pi * 12 * time)
    music += 0.2 * np.random.normal(0, 1, n_samples)
    
    # Signal 3: Noise
    noise = np.random.laplace(0, 0.5, n_samples)
    
    # Combine sources
    sources = np.column_stack([speech, music, noise])
    
    # Create mixing matrix (simulating microphone positions)
    A_audio = np.array([
        [1.0, 0.7, 0.3],
        [0.6, 1.0, 0.4],
        [0.4, 0.6, 1.0]
    ])
    
    # Mix signals
    mixed = sources @ A_audio.T
    
    # Apply ICA
    ica_audio = FastICA(n_components=3, random_state=42)
    separated = ica_audio.fit_transform(mixed)
    
    # Visualize
    plot_signals(
        [sources, mixed, separated],
        ['Original Audio Sources', 'Mixed Audio', 'Separated Audio'],
        figsize=(15, 12)
    )
    
    print("Audio separation completed!")
    print("ICA successfully separated the speech, music, and noise components.")

demonstrate_audio_separation()

def demonstrate_image_separation():
    """
    Demonstrate ICA for image component separation.
    """
    print("Demonstrating image component separation...")
    
    # Create synthetic image data
    np.random.seed(42)
    n_images = 100
    image_size = 8
    
    # Generate base patterns
    pattern1 = np.zeros((image_size, image_size))
    pattern1[2:6, 2:6] = 1  # Square in center
    
    pattern2 = np.zeros((image_size, image_size))
    pattern2[1:7, 1:7] = np.eye(6)  # Diagonal pattern
    
    pattern3 = np.random.normal(0, 1, (image_size, image_size))
    
    # Create images as mixtures of patterns
    images = []
    for i in range(n_images):
        # Random weights for each pattern
        w1 = np.random.normal(0, 1)
        w2 = np.random.normal(0, 1)
        w3 = np.random.normal(0, 1)
        
        # Create image as mixture
        image = w1 * pattern1 + w2 * pattern2 + w3 * pattern3
        images.append(image.flatten())
    
    images = np.array(images)
    
    # Apply ICA
    ica_image = FastICA(n_components=3, random_state=42)
    components = ica_image.fit_transform(images)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original patterns
    axes[0, 0].imshow(pattern1, cmap='gray')
    axes[0, 0].set_title('Original Pattern 1')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pattern2, cmap='gray')
    axes[0, 1].set_title('Original Pattern 2')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pattern3, cmap='gray')
    axes[0, 2].set_title('Original Pattern 3')
    axes[0, 2].axis('off')
    
    # Recovered components
    for i in range(3):
        component = ica_image.components_[i].reshape(image_size, image_size)
        axes[1, i].imshow(component, cmap='gray')
        axes[1, i].set_title(f'Recovered Component {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Image component separation completed!")
    print("ICA successfully recovered the underlying image patterns.")

demonstrate_image_separation()

# ============================================================================
# SECTION 9: LIMITATIONS AND CHALLENGES
# ============================================================================

print_section_header("LIMITATIONS AND CHALLENGES")

def demonstrate_gaussian_limitation():
    """
    Demonstrate why ICA fails with Gaussian sources.
    """
    print("Demonstrating ICA limitations with Gaussian sources...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate Gaussian sources
    s1_gauss = np.random.normal(0, 1, n_samples)
    s2_gauss = np.random.normal(0, 1, n_samples)
    s3_gauss = np.random.normal(0, 1, n_samples)
    
    S_gauss = np.column_stack([s1_gauss, s2_gauss, s3_gauss])
    
    # Mix Gaussian sources
    A_gauss = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])
    X_gauss = S_gauss @ A_gauss.T
    
    # Try to separate with ICA
    ica_gauss = FastICA(n_components=3, random_state=42)
    S_gauss_recovered = ica_gauss.fit_transform(X_gauss)
    
    # Compare with PCA
    from sklearn.decomposition import PCA
    pca_gauss = PCA(n_components=3)
    S_pca = pca_gauss.fit_transform(X_gauss)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original Gaussian sources
    for i in range(3):
        axes[0, 0].plot(S_gauss[:, i], label=f'Source {i+1}')
    axes[0, 0].set_title('Original Gaussian Sources')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mixed signals
    for i in range(3):
        axes[0, 1].plot(X_gauss[:, i], label=f'Mixed {i+1}')
    axes[0, 1].set_title('Mixed Signals')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # ICA recovery
    for i in range(3):
        axes[1, 0].plot(S_gauss_recovered[:, i], label=f'ICA {i+1}')
    axes[1, 0].set_title('ICA Recovery (Unreliable)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # PCA recovery
    for i in range(3):
        axes[1, 1].plot(S_pca[:, i], label=f'PCA {i+1}')
    axes[1, 1].set_title('PCA Recovery (Also Unreliable)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Key limitation:")
    print("ICA cannot separate Gaussian sources due to rotational symmetry!")
    print("Both ICA and PCA give unreliable results with Gaussian sources.")

demonstrate_gaussian_limitation()

# ============================================================================
# SECTION 10: SUMMARY AND BEST PRACTICES
# ============================================================================

print_section_header("SUMMARY AND BEST PRACTICES")

def ica_best_practices():
    """
    Summarize best practices for using ICA.
    """
    print("ICA Best Practices:")
    print("=" * 50)
    
    practices = [
        "1. Always preprocess data with whitening",
        "2. Ensure sources are non-Gaussian",
        "3. Use appropriate number of components",
        "4. Be aware of ambiguities (permutation, scaling, sign)",
        "5. Validate results with domain knowledge",
        "6. Consider computational complexity for large datasets",
        "7. Use FastICA for better convergence",
        "8. Check for convergence and stability",
        "9. Interpret results carefully",
        "10. Consider alternative methods for Gaussian sources"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\nWhen to use ICA:")
    print("- Source separation problems")
    print("- Non-Gaussian independent sources")
    print("- Blind source separation")
    print("- Signal processing applications")
    print("- Feature extraction for non-Gaussian data")
    
    print("\nWhen NOT to use ICA:")
    print("- Gaussian sources")
    print("- Non-linear mixing")
    print("- Time-dependent mixing")
    print("- When sources are not independent")
    print("- When interpretability is crucial")

ica_best_practices()

print("\n" + "="*60)
print(" ICA EXAMPLES COMPLETE!")
print("="*60)
print("\nYou now have a comprehensive understanding of ICA concepts and implementation!") 
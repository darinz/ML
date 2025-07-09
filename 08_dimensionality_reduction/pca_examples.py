"""
Principal Components Analysis (PCA) - Comprehensive Python Examples
==================================================================

This script provides comprehensive implementations of PCA concepts from the markdown file.
Each section demonstrates key concepts with detailed explanations and visualizations.

Key Concepts Covered:
1. Data preprocessing and normalization
2. Covariance matrix calculation
3. Eigenvalue decomposition
4. Principal component extraction
5. Data projection and reconstruction
6. Explained variance analysis
7. Visualization techniques
8. Practical applications

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SKPCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, load_iris
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

def print_section_header(title):
    """Print a formatted section header for better readability."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def plot_2d_data_with_pca(X, title="Data with Principal Components"):
    """
    Plot 2D data with principal components overlaid.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        The 2D data to plot
    title : str
        Title for the plot
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Plot data points
    plt.figure(figsize=(10, 8))
    plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6, label='Data Points')
    
    # Plot principal components
    mean = np.mean(X, axis=0)
    for i, (eigval, eigvec) in enumerate(zip(eigvals, eigvecs.T)):
        # Scale the eigenvector by eigenvalue for visualization
        scaled_vec = eigvec * np.sqrt(eigval) * 2
        plt.arrow(0, 0, scaled_vec[0], scaled_vec[1], 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', 
                 label=f'PC{i+1} (λ={eigval:.2f})')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

# ============================================================================
# SECTION 1: UNDERSTANDING THE CURSE OF DIMENSIONALITY
# ============================================================================

print_section_header("UNDERSTANDING THE CURSE OF DIMENSIONALITY")

def demonstrate_curse_of_dimensionality():
    """
    Demonstrate the curse of dimensionality with a simple example.
    Shows how data becomes sparse in high dimensions.
    """
    print("Demonstrating the curse of dimensionality...")
    
    # Generate data in different dimensions
    dimensions = [2, 10, 50, 100]
    n_samples = 1000
    
    plt.figure(figsize=(12, 8))
    
    for i, dim in enumerate(dimensions):
        # Generate random data in this dimension
        X = np.random.randn(n_samples, dim)
        
        # Compute pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(X[:100, :])  # Use subset for speed
        distances = distances[np.triu_indices_from(distances, k=1)]  # Upper triangle
        
        plt.subplot(2, 2, i+1)
        plt.hist(distances, bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'{dim} Dimensions')
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Frequency')
        
        # Calculate statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        plt.text(0.7, 0.9, f'Mean: {mean_dist:.2f}\nStd: {std_dist:.2f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    plt.tight_layout()
    plt.suptitle('Distribution of Pairwise Distances in Different Dimensions', y=1.02)
    plt.show()
    
    print("Notice how distances become more uniform as dimension increases!")
    print("This makes distance-based algorithms less effective in high dimensions.")

demonstrate_curse_of_dimensionality()

# ============================================================================
# SECTION 2: DATA PREPROCESSING - NORMALIZATION
# ============================================================================

print_section_header("DATA PREPROCESSING - NORMALIZATION")

def demonstrate_normalization():
    """
    Demonstrate why normalization is crucial for PCA.
    Shows how features with different scales can dominate the analysis.
    """
    print("Demonstrating the importance of normalization...")
    
    # Create a dataset with features on very different scales
    np.random.seed(42)
    n_samples = 100
    
    # Feature 1: Height in centimeters (large values)
    height_cm = np.random.normal(170, 10, n_samples)
    
    # Feature 2: Number of children (small values)
    children = np.random.poisson(2, n_samples)
    
    # Feature 3: Age in years (medium values)
    age = np.random.normal(35, 8, n_samples)
    
    X = np.column_stack([height_cm, children, age])
    
    print("Original data statistics:")
    print(f"Height (cm): mean={np.mean(height_cm):.2f}, std={np.std(height_cm):.2f}")
    print(f"Children: mean={np.mean(children):.2f}, std={np.std(children):.2f}")
    print(f"Age: mean={np.mean(age):.2f}, std={np.std(age):.2f}")
    
    # Compute covariance matrix without normalization
    cov_unnormalized = np.cov(X, rowvar=False)
    print("\nCovariance matrix (unnormalized):")
    print(cov_unnormalized)
    
    # Normalize the data
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    # Compute covariance matrix after normalization
    cov_normalized = np.cov(X_normalized, rowvar=False)
    print("\nCovariance matrix (normalized):")
    print(cov_normalized)
    
    # Compare the first principal component
    eigvals_unnorm, eigvecs_unnorm = np.linalg.eigh(cov_unnormalized)
    eigvals_norm, eigvecs_norm = np.linalg.eigh(cov_normalized)
    
    # Sort by eigenvalues
    idx_unnorm = np.argsort(eigvals_unnorm)[::-1]
    idx_norm = np.argsort(eigvals_norm)[::-1]
    
    pc1_unnorm = eigvecs_unnorm[:, idx_unnorm[0]]
    pc1_norm = eigvecs_norm[:, idx_norm[0]]
    
    print(f"\nFirst principal component (unnormalized): {pc1_unnorm}")
    print(f"First principal component (normalized): {pc1_norm}")
    
    print("\nKey insight: Without normalization, height dominates the analysis!")
    print("After normalization, all features contribute equally.")

demonstrate_normalization()

# ============================================================================
# SECTION 3: STEP-BY-STEP PCA IMPLEMENTATION
# ============================================================================

print_section_header("STEP-BY-STEP PCA IMPLEMENTATION")

def manual_pca_implementation(X, n_components=2):
    """
    Implement PCA manually step by step.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    n_components : int
        Number of principal components to return
        
    Returns:
    --------
    X_pca : array-like, shape (n_samples, n_components)
        Projected data
    components : array-like, shape (n_components, n_features)
        Principal components (eigenvectors)
    explained_variance : array-like, shape (n_components,)
        Explained variance for each component
    """
    print("Step 1: Center the data")
    X_centered = X - np.mean(X, axis=0)
    print(f"Data shape: {X.shape}")
    print(f"Mean of each feature: {np.mean(X, axis=0)}")
    
    print("\nStep 2: Compute covariance matrix")
    cov_matrix = np.cov(X_centered, rowvar=False)
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print("Covariance matrix:")
    print(cov_matrix)
    
    print("\nStep 3: Compute eigenvalue decomposition")
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    print(f"Eigenvalues: {eigvals}")
    print("Eigenvectors (columns):")
    print(eigvecs)
    
    print("\nStep 4: Select top principal components")
    components = eigvecs[:, :n_components]
    explained_variance = eigvals[:n_components]
    
    print(f"Selected {n_components} principal components")
    print(f"Explained variance: {explained_variance}")
    
    print("\nStep 5: Project data onto principal components")
    X_pca = X_centered @ components
    print(f"Projected data shape: {X_pca.shape}")
    
    return X_pca, components, explained_variance

# Create a synthetic dataset for demonstration
np.random.seed(42)
n_samples, n_features = 100, 4

# Generate correlated data
X_synthetic = np.random.multivariate_normal(
    mean=[0, 0, 0, 0],
    cov=[[1, 0.8, 0.6, 0.4],
         [0.8, 1, 0.7, 0.5],
         [0.6, 0.7, 1, 0.3],
         [0.4, 0.5, 0.3, 1]],
    size=n_samples
)

print("Synthetic dataset with correlated features:")
print(f"Shape: {X_synthetic.shape}")
print("Sample data:")
print(X_synthetic[:5])

# Apply manual PCA
X_pca_manual, components_manual, explained_variance_manual = manual_pca_implementation(
    X_synthetic, n_components=2
)

print(f"\nManual PCA Results:")
print(f"Explained variance ratio: {explained_variance_manual / np.sum(explained_variance_manual)}")
print(f"Cumulative explained variance: {np.sum(explained_variance_manual) / np.sum(explained_variance_manual):.3f}")

# ============================================================================
# SECTION 4: GEOMETRIC INTUITION AND VISUALIZATION
# ============================================================================

print_section_header("GEOMETRIC INTUITION AND VISUALIZATION")

def demonstrate_geometric_intuition():
    """
    Demonstrate the geometric intuition behind PCA with 2D examples.
    """
    print("Demonstrating geometric intuition with 2D examples...")
    
    # Create an elongated cloud of points
    np.random.seed(42)
    n_points = 200
    
    # Generate data along a line with some noise
    t = np.random.uniform(-3, 3, n_points)
    x1 = t + np.random.normal(0, 0.3, n_points)
    x2 = 0.5 * t + np.random.normal(0, 0.3, n_points)
    
    X_elongated = np.column_stack([x1, x2])
    
    # Plot the data with principal components
    plot_2d_data_with_pca(X_elongated, "Elongated Data Cloud with Principal Components")
    
    # Create a more complex example with multiple clusters
    X_clusters, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                              cluster_std=0.5, random_state=42)
    
    plot_2d_data_with_pca(X_clusters, "Clustered Data with Principal Components")
    
    print("Key observations:")
    print("1. First principal component points along the direction of maximum variance")
    print("2. Second principal component is orthogonal to the first")
    print("3. Principal components capture the main directions of variation in the data")

demonstrate_geometric_intuition()

# ============================================================================
# SECTION 5: EXPLAINED VARIANCE AND DIMENSIONALITY REDUCTION
# ============================================================================

print_section_header("EXPLAINED VARIANCE AND DIMENSIONALITY REDUCTION")

def analyze_explained_variance(X, max_components=None):
    """
    Analyze explained variance to determine optimal number of components.
    
    Parameters:
    -----------
    X : array-like
        Input data
    max_components : int, optional
        Maximum number of components to analyze
    """
    if max_components is None:
        max_components = min(X.shape)
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues
    eigvals = np.linalg.eigvals(cov_matrix)
    eigvals = np.real(eigvals)  # Remove imaginary parts if any
    eigvals = np.sort(eigvals)[::-1]  # Sort in descending order
    
    # Calculate explained variance ratios
    explained_variance_ratio = eigvals / np.sum(eigvals)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    
    # Plot explained variance
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Individual explained variance
    plt.subplot(1, 2, 1)
    plt.bar(range(1, max_components + 1), explained_variance_ratio[:max_components])
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, max_components + 1), cumulative_explained_variance[:max_components], 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.axhline(y=0.99, color='g', linestyle='--', label='99% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print recommendations
    n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_explained_variance >= 0.99) + 1
    
    print(f"Explained variance analysis:")
    print(f"Components needed for 95% variance: {n_components_95}")
    print(f"Components needed for 99% variance: {n_components_99}")
    print(f"Total variance explained by first 2 components: {cumulative_explained_variance[1]:.3f}")

# Use the iris dataset for a realistic example
iris = load_iris()
X_iris = iris.data

print("Analyzing explained variance for Iris dataset:")
analyze_explained_variance(X_iris, max_components=4)

# ============================================================================
# SECTION 6: RECONSTRUCTION AND INFORMATION LOSS
# ============================================================================

print_section_header("RECONSTRUCTION AND INFORMATION LOSS")

def demonstrate_reconstruction(X, n_components_list=[1, 2, 3, 4]):
    """
    Demonstrate how reconstruction quality changes with number of components.
    
    Parameters:
    -----------
    X : array-like
        Input data
    n_components_list : list
        List of numbers of components to test
    """
    print("Demonstrating reconstruction quality...")
    
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    mean = np.mean(X, axis=0)
    
    # Compute full PCA
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    plt.figure(figsize=(15, 10))
    
    for i, n_comp in enumerate(n_components_list):
        # Project data
        components = eigvecs[:, :n_comp]
        X_projected = X_centered @ components
        
        # Reconstruct data
        X_reconstructed = X_projected @ components.T + mean
        
        # Calculate reconstruction error
        mse = np.mean((X - X_reconstructed) ** 2)
        explained_variance = np.sum(eigvals[:n_comp]) / np.sum(eigvals)
        
        # Plot original vs reconstructed (first two features)
        plt.subplot(2, 2, i+1)
        plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label='Original', color='blue')
        plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], 
                   alpha=0.6, label='Reconstructed', color='red')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(f'{n_comp} Components\nMSE: {mse:.4f}, Var: {explained_variance:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Key insights:")
    print("1. More components = better reconstruction")
    print("2. Reconstruction error decreases with more components")
    print("3. Trade-off between compression and information preservation")

# Use a subset of iris data for visualization
X_iris_subset = iris.data[:, :2]  # Use only first two features for 2D visualization
demonstrate_reconstruction(X_iris_subset)

# ============================================================================
# SECTION 7: PRACTICAL APPLICATIONS
# ============================================================================

print_section_header("PRACTICAL APPLICATIONS")

def demonstrate_face_recognition_example():
    """
    Demonstrate PCA for face recognition (eigenfaces) with synthetic data.
    """
    print("Demonstrating PCA for face recognition (eigenfaces)...")
    
    # Create synthetic "face" data (simplified)
    np.random.seed(42)
    n_faces = 50
    face_size = 8  # 8x8 "faces" for simplicity
    
    # Generate synthetic faces with some common patterns
    faces = []
    for i in range(n_faces):
        # Base face pattern
        face = np.random.normal(0, 1, (face_size, face_size))
        
        # Add some common features (eyes, nose, mouth)
        face[2:4, 2:4] += 2  # Left eye
        face[2:4, 4:6] += 2  # Right eye
        face[4:6, 3:5] += 1.5  # Nose
        face[6:7, 2:6] += 1  # Mouth
        
        # Add some individual variation
        face += np.random.normal(0, 0.5, (face_size, face_size))
        
        faces.append(face.flatten())
    
    faces = np.array(faces)
    
    # Apply PCA
    pca = SKPCA(n_components=10)
    faces_pca = pca.fit_transform(faces)
    
    # Visualize eigenfaces
    plt.figure(figsize=(15, 8))
    
    # Plot first few eigenfaces
    for i in range(6):
        plt.subplot(2, 3, i+1)
        eigenface = pca.components_[i].reshape(face_size, face_size)
        plt.imshow(eigenface, cmap='gray')
        plt.title(f'Eigenface {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate reconstruction
    plt.figure(figsize=(15, 8))
    
    # Original face
    plt.subplot(2, 3, 1)
    original_face = faces[0].reshape(face_size, face_size)
    plt.imshow(original_face, cmap='gray')
    plt.title('Original Face')
    plt.axis('off')
    
    # Reconstructions with different numbers of components
    for i, n_comp in enumerate([1, 2, 5, 10, 20]):
        pca_temp = SKPCA(n_components=n_comp)
        faces_pca_temp = pca_temp.fit_transform(faces)
        reconstructed = pca_temp.inverse_transform(faces_pca_temp)
        
        plt.subplot(2, 3, i+2)
        recon_face = reconstructed[0].reshape(face_size, face_size)
        plt.imshow(recon_face, cmap='gray')
        plt.title(f'{n_comp} Components')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Eigenfaces capture the main patterns of variation in face images!")
    print("Key patterns (eyes, nose, mouth) are preserved even with few components.")

demonstrate_face_recognition_example()

# ============================================================================
# SECTION 8: COMPARISON WITH SCIKIT-LEARN IMPLEMENTATION
# ============================================================================

print_section_header("COMPARISON WITH SCIKIT-LEARN IMPLEMENTATION")

def compare_implementations(X):
    """
    Compare manual PCA implementation with scikit-learn.
    
    Parameters:
    -----------
    X : array-like
        Input data
    """
    print("Comparing manual PCA with scikit-learn implementation...")
    
    # Manual implementation
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigvals_manual, eigvecs_manual = np.linalg.eigh(cov_matrix)
    
    # Sort
    idx = np.argsort(eigvals_manual)[::-1]
    eigvals_manual = eigvals_manual[idx]
    eigvecs_manual = eigvecs_manual[:, idx]
    
    # Scikit-learn implementation
    pca_sklearn = SKPCA()
    pca_sklearn.fit(X)
    
    # Compare results
    print("Manual PCA eigenvalues:", eigvals_manual[:5])
    print("Sklearn PCA eigenvalues:", pca_sklearn.explained_variance_[:5])
    
    print("\nManual PCA first component:", eigvecs_manual[:, 0][:5])
    print("Sklearn PCA first component:", pca_sklearn.components_[0][:5])
    
    # Check if they're equivalent (up to sign)
    manual_pc1 = eigvecs_manual[:, 0]
    sklearn_pc1 = pca_sklearn.components_[0]
    
    # Check if they're the same or opposite
    correlation = np.corrcoef(manual_pc1, sklearn_pc1)[0, 1]
    print(f"\nCorrelation between first components: {correlation:.6f}")
    
    if abs(correlation) > 0.99:
        print("✓ Manual and sklearn implementations are equivalent!")
    else:
        print("✗ Manual and sklearn implementations differ!")

# Use iris data for comparison
compare_implementations(X_iris)

# ============================================================================
# SECTION 9: ADVANCED TOPICS - KERNEL PCA PREVIEW
# ============================================================================

print_section_header("ADVANCED TOPICS - KERNEL PCA PREVIEW")

def demonstrate_linear_vs_nonlinear():
    """
    Demonstrate when linear PCA fails and when kernel PCA would help.
    """
    print("Demonstrating limitations of linear PCA...")
    
    # Create a non-linear dataset (swiss roll-like)
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data in a spiral pattern
    t = np.linspace(0, 4*np.pi, n_samples)
    r = 1 + 0.3*np.random.randn(n_samples)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = t + 0.3*np.random.randn(n_samples)
    
    X_spiral = np.column_stack([x, y, z])
    
    # Apply linear PCA
    pca_linear = SKPCA(n_components=2)
    X_pca_linear = pca_linear.fit_transform(X_spiral)
    
    # Visualize
    fig = plt.figure(figsize=(15, 5))
    
    # Original 3D data
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X_spiral[:, 0], X_spiral[:, 1], X_spiral[:, 2], alpha=0.6)
    ax1.set_title('Original 3D Data (Spiral)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Linear PCA projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_pca_linear[:, 0], X_pca_linear[:, 1], alpha=0.6)
    ax2.set_title('Linear PCA Projection')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.grid(True, alpha=0.3)
    
    # Show the problem
    ax3 = fig.add_subplot(133)
    ax3.scatter(X_spiral[:, 0], X_spiral[:, 1], alpha=0.6)
    ax3.set_title('Top View (X-Y plane)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Problem with linear PCA:")
    print("1. Linear PCA cannot capture non-linear structure")
    print("2. The spiral structure is lost in the projection")
    print("3. Kernel PCA would be needed to preserve the spiral structure")
    print("4. This is why we need non-linear dimensionality reduction methods")

demonstrate_linear_vs_nonlinear()

# ============================================================================
# SECTION 10: SUMMARY AND BEST PRACTICES
# ============================================================================

print_section_header("SUMMARY AND BEST PRACTICES")

def pca_best_practices():
    """
    Summarize best practices for using PCA.
    """
    print("PCA Best Practices:")
    print("=" * 50)
    
    practices = [
        "1. Always normalize/standardize your data before PCA",
        "2. Check explained variance to choose number of components",
        "3. Consider the trade-off between compression and information loss",
        "4. Use PCA for dimensionality reduction, not feature selection",
        "5. Be aware that PCA assumes linear relationships",
        "6. For non-linear data, consider Kernel PCA or other methods",
        "7. Validate PCA results on held-out data",
        "8. Interpret principal components carefully - they may not be meaningful",
        "9. Use PCA for visualization, compression, and noise reduction",
        "10. Consider computational complexity for large datasets"
    ]
    
    for practice in practices:
        print(practice)
    
    print("\nWhen to use PCA:")
    print("- High-dimensional data with correlated features")
    print("- Need for dimensionality reduction")
    print("- Data visualization in 2D/3D")
    print("- Noise reduction")
    print("- Compression applications")
    
    print("\nWhen NOT to use PCA:")
    print("- Non-linear relationships in data")
    print("- Need to preserve feature interpretability")
    print("- Data already has low dimensionality")
    print("- Need for feature selection (use other methods)")

pca_best_practices()

print("\n" + "="*60)
print(" PCA EXAMPLES COMPLETE!")
print("="*60)
print("\nYou now have a comprehensive understanding of PCA concepts and implementation!") 
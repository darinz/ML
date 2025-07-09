"""
Kernel Properties and Mercer's Theorem: Implementation Examples
============================================================

This file implements the key concepts from the kernel properties document:
- Positive definite kernels and their properties
- Mercer's theorem and its implications
- Kernel construction rules and validation
- Multiple kernel learning and advanced applications

The implementations demonstrate both theoretical concepts and practical
validation methods with detailed mathematical explanations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_circles, make_moons, make_classification
import time


# ============================================================================
# Section 5.5: Kernel Properties and Mercer's Theorem
# ============================================================================

def is_positive_definite(K, tolerance=1e-10):
    """
    Check if a kernel matrix is positive definite.
    
    A kernel matrix K is positive definite if all eigenvalues are non-negative.
    This is a necessary condition for a valid kernel function.
    
    Args:
        K: Kernel matrix (n x n)
        tolerance: Numerical tolerance for eigenvalue check
        
    Returns:
        bool: True if positive definite, False otherwise
        
    Mathematical foundation:
        K is positive definite if for any vector z: z^T K z ≥ 0
        This is equivalent to all eigenvalues being non-negative.
    """
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(K)
    
    # Check if all eigenvalues are non-negative (within tolerance)
    is_pd = np.all(eigenvalues.real >= -tolerance)
    
    return is_pd, eigenvalues


def test_kernel_positive_definiteness():
    """
    Test various kernels for positive definiteness.
    
    This demonstrates how to validate kernel functions using
    the positive definiteness property.
    """
    print("=== Testing Kernel Positive Definiteness ===")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(50, 3)  # 50 samples, 3 features
    
    # Test different kernels
    kernels = [
        ("Linear", lambda X: linear_kernel(X)),
        ("Polynomial (degree=2)", lambda X: polynomial_kernel(X, degree=2)),
        ("RBF (γ=1)", lambda X: rbf_kernel(X, gamma=1.0)),
        ("RBF (γ=0.1)", lambda X: rbf_kernel(X, gamma=0.1))
    ]
    
    results = {}
    
    for kernel_name, kernel_func in kernels:
        print(f"\n--- {kernel_name} ---")
        
        # Compute kernel matrix
        K = kernel_func(X)
        
        # Check positive definiteness
        is_pd, eigenvalues = is_positive_definite(K)
        
        results[kernel_name] = {
            'is_positive_definite': is_pd,
            'eigenvalues': eigenvalues,
            'min_eigenvalue': np.min(eigenvalues.real),
            'max_eigenvalue': np.max(eigenvalues.real)
        }
        
        print(f"Positive definite: {is_pd}")
        print(f"Eigenvalue range: [{np.min(eigenvalues.real):.6f}, {np.max(eigenvalues.real):.6f}]")
        print(f"Condition number: {np.max(eigenvalues.real) / (np.min(eigenvalues.real) + 1e-10):.2f}")
    
    return results


def demonstrate_mercer_theorem():
    """
    Demonstrate Mercer's theorem in practice.
    
    Mercer's theorem states that any positive definite kernel corresponds
    to an inner product in some feature space. This example shows how
    to construct an approximate feature map from a kernel matrix.
    """
    print("=== Mercer's Theorem Demonstration ===")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(30, 2)
    
    # Compute RBF kernel matrix
    K = rbf_kernel(X, gamma=1.0)
    
    # Eigenvalue decomposition (spectral decomposition)
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    
    # Sort in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print(f"Kernel matrix shape: {K.shape}")
    print(f"Number of positive eigenvalues: {np.sum(eigenvalues > 1e-10)}")
    print(f"Eigenvalue spectrum: {eigenvalues[:10]}")  # Show first 10
    
    # Construct approximate feature map
    # φ(x_i) ≈ √λ_j * v_ij where v_ij is the j-th component of eigenvector i
    positive_eigenvalues = eigenvalues[eigenvalues > 1e-10]
    positive_eigenvectors = eigenvectors[:, eigenvalues > 1e-10]
    
    # Feature map: φ(x_i) = [√λ_1 * v_i1, √λ_2 * v_i2, ...]
    feature_map = np.sqrt(positive_eigenvalues) * positive_eigenvectors.T
    
    print(f"Feature map dimension: {feature_map.shape[0]}")
    print(f"Number of features: {feature_map.shape[1]}")
    
    # Verify that K ≈ φ^T φ
    K_reconstructed = feature_map.T @ feature_map
    reconstruction_error = np.mean((K - K_reconstructed) ** 2)
    
    print(f"Reconstruction error: {reconstruction_error:.2e}")
    print("This demonstrates that the kernel matrix can be reconstructed")
    print("from the feature map, confirming Mercer's theorem.")
    
    return feature_map, eigenvalues, eigenvectors


def kernel_construction_rules():
    """
    Demonstrate kernel construction rules.
    
    If K1 and K2 are valid kernels, then the following are also valid kernels:
    1. a*K1 where a > 0 (scalar multiplication)
    2. K1 + K2 (addition)
    3. K1 * K2 (multiplication)
    4. K1(f(x), f(z)) where f is any function (composition)
    """
    print("=== Kernel Construction Rules ===")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(20, 2)
    
    # Base kernels
    K1 = rbf_kernel(X, gamma=1.0)  # RBF kernel
    K2 = polynomial_kernel(X, degree=2)  # Polynomial kernel
    
    # Test construction rules
    kernels = [
        ("K1 (RBF)", K1),
        ("K2 (Polynomial)", K2),
        ("2*K1 (Scalar multiplication)", 2 * K1),
        ("K1 + K2 (Addition)", K1 + K2),
        ("K1 * K2 (Multiplication)", K1 * K2),
        ("K1 + 0.5*K2 (Combination)", K1 + 0.5 * K2)
    ]
    
    print("Testing kernel construction rules:")
    print("-" * 50)
    
    for kernel_name, K in kernels:
        is_pd, eigenvalues = is_positive_definite(K)
        min_eigenval = np.min(eigenvalues.real)
        
        print(f"{kernel_name:25s} | Positive definite: {is_pd:5s} | Min eigenvalue: {min_eigenval:8.6f}")
    
    print("\nAll constructed kernels should be positive definite!")
    print()


def multiple_kernel_learning(X, y, kernel_list, alpha_weights=None):
    """
    Implement multiple kernel learning.
    
    This combines multiple kernels with learned weights to capture
    different aspects of the data.
    
    Args:
        X: Training data
        y: Target labels
        kernel_list: List of kernel functions
        alpha_weights: Initial weights for kernels (optional)
        
    Returns:
        optimal_weights: Learned kernel weights
        combined_kernel: Combined kernel matrix
    """
    print("=== Multiple Kernel Learning ===")
    
    n_kernels = len(kernel_list)
    
    # Initialize weights uniformly if not provided
    if alpha_weights is None:
        alpha_weights = np.ones(n_kernels) / n_kernels
    
    # Compute individual kernel matrices
    kernel_matrices = []
    for kernel_func in kernel_list:
        K = kernel_func(X)
        kernel_matrices.append(K)
    
    # Simple optimization: try different weight combinations
    best_score = -1
    best_weights = alpha_weights.copy()
    
    # Grid search over weight combinations
    weight_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for w1 in weight_values:
        for w2 in weight_values:
            if w1 + w2 <= 1.0:  # Constraint: sum of weights ≤ 1
                w3 = 1.0 - w1 - w2
                weights = np.array([w1, w2, w3])
                
                # Combine kernels
                combined_kernel = sum(w * K for w, K in zip(weights, kernel_matrices))
                
                # Evaluate using cross-validation
                try:
                    clf = SVC(kernel='precomputed')
                    scores = cross_val_score(clf, combined_kernel, y, cv=3)
                    avg_score = np.mean(scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_weights = weights.copy()
                except:
                    continue
    
    # Final combined kernel with optimal weights
    final_kernel = sum(w * K for w, K in zip(best_weights, kernel_matrices))
    
    print(f"Optimal kernel weights: {best_weights}")
    print(f"Best cross-validation score: {best_score:.4f}")
    
    return best_weights, final_kernel


def example_multiple_kernel_learning():
    """
    Example of multiple kernel learning on synthetic data.
    """
    print("=== Multiple Kernel Learning Example ===")
    
    # Create synthetic data with multiple patterns
    np.random.seed(42)
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.5)
    
    # Define different kernels
    kernels = [
        ("Linear", lambda X: linear_kernel(X)),
        ("RBF", lambda X: rbf_kernel(X, gamma=1.0)),
        ("Polynomial", lambda X: polynomial_kernel(X, degree=2))
    ]
    
    # Test individual kernels
    print("Individual kernel performance:")
    for kernel_name, kernel_func in kernels:
        K = kernel_func(X)
        clf = SVC(kernel='precomputed')
        scores = cross_val_score(clf, K, y, cv=3)
        print(f"{kernel_name:12s}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Multiple kernel learning
    optimal_weights, combined_kernel = multiple_kernel_learning(X, y, [k[1] for k in kernels])
    
    # Evaluate combined kernel
    clf = SVC(kernel='precomputed')
    scores = cross_val_score(clf, combined_kernel, y, cv=3)
    print(f"Combined kernel: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    return optimal_weights, combined_kernel


def kernel_pca_example():
    """
    Demonstrate Kernel PCA for dimensionality reduction.
    
    Kernel PCA performs Principal Component Analysis in the feature space
    without explicitly computing the features.
    """
    print("=== Kernel PCA Example ===")
    
    # Create non-linear data (swiss roll-like)
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 200)
    X = np.column_stack([
        t * np.cos(t) + np.random.normal(0, 0.1, 200),
        t * np.sin(t) + np.random.normal(0, 0.1, 200)
    ])
    
    # Apply Kernel PCA
    kpca = KernelPCA(kernel='rbf', gamma=1.0, n_components=2)
    X_kpca = kpca.fit_transform(X)
    
    # Compare with linear PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=t, cmap='viridis')
    plt.title('Original Data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    
    plt.subplot(1, 3, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=t, cmap='viridis')
    plt.title('Linear PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=t, cmap='viridis')
    plt.title('Kernel PCA (RBF)')
    plt.xlabel('KPC1')
    plt.ylabel('KPC2')
    
    plt.tight_layout()
    plt.show()
    
    print("Kernel PCA successfully captures the non-linear structure!")
    print()


def validate_custom_kernel():
    """
    Demonstrate how to validate a custom kernel function.
    
    This shows the process of checking if a proposed function
    is a valid kernel using Mercer's theorem.
    """
    print("=== Custom Kernel Validation ===")
    
    # Define a custom kernel function
    def custom_kernel(x, z, sigma=1.0):
        """
        Custom kernel: K(x, z) = exp(-||x - z||^2 / (2*sigma^2))
        This is actually the RBF kernel with γ = 1/(2*sigma^2)
        """
        diff = x - z
        return np.exp(-np.dot(diff, diff) / (2 * sigma**2))
    
    # Generate test data
    np.random.seed(42)
    X = np.random.randn(20, 3)
    
    # Compute kernel matrix
    K = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            K[i, j] = custom_kernel(X[i], X[j], sigma=1.0)
    
    # Validate the kernel
    is_pd, eigenvalues = is_positive_definite(K)
    
    print(f"Custom kernel validation:")
    print(f"Positive definite: {is_pd}")
    print(f"Eigenvalue range: [{np.min(eigenvalues.real):.6f}, {np.max(eigenvalues.real):.6f}]")
    print(f"Condition number: {np.max(eigenvalues.real) / (np.min(eigenvalues.real) + 1e-10):.2f}")
    
    if is_pd:
        print("✓ Custom kernel is valid!")
    else:
        print("✗ Custom kernel is not valid!")
    
    return is_pd, eigenvalues


def main():
    """
    Main function to run all kernel property examples.
    """
    print("Kernel Properties and Mercer's Theorem: Examples")
    print("=" * 60)
    
    # Run all examples
    test_kernel_positive_definiteness()
    demonstrate_mercer_theorem()
    kernel_construction_rules()
    example_multiple_kernel_learning()
    kernel_pca_example()
    validate_custom_kernel()
    
    print("\nAll kernel property examples completed!")


if __name__ == "__main__":
    main() 
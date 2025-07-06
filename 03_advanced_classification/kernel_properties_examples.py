"""
Kernel Properties Examples
==========================

This file contains all the Python code examples from the kernel properties section,
demonstrating various kernel functions, their implementations, and applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import polynomial_kernel, rbf_kernel
from sklearn import svm
from sklearn.datasets import make_moons
from collections import Counter


def quadratic_kernel_examples():
    """Examples of quadratic kernel K(x, z) = (x^T z)^2"""
    print("=== Quadratic Kernel Examples ===")
    
    # Pure Python implementation
    x = [1, 2, 3]
    z = [4, 5, 6]
    
    def quadratic_kernel(x, z):
        return sum(xi * zi for xi, zi in zip(x, z)) ** 2
    
    print(f"Pure Python quadratic kernel: {quadratic_kernel(x, z)}")
    
    # NumPy implementation
    x_np = np.array([1, 2, 3])
    z_np = np.array([4, 5, 6])
    
    def quadratic_kernel_np(x, z):
        return np.dot(x, z) ** 2
    
    print(f"NumPy quadratic kernel: {quadratic_kernel_np(x_np, z_np)}")
    
    # Expanded form implementation
    def expanded_quadratic_kernel(x, z):
        d = len(x)
        return sum(x[i] * x[j] * z[i] * z[j] for i in range(d) for j in range(d))
    
    print(f"Expanded form quadratic kernel: {expanded_quadratic_kernel(x, z)}")
    
    # Explicit feature mapping for d=3
    def phi_quadratic(x):
        return [x[0]*x[0], x[0]*x[1], x[0]*x[2],
                x[1]*x[0], x[1]*x[1], x[1]*x[2],
                x[2]*x[0], x[2]*x[1], x[2]*x[2]]
    
    print(f"Explicit feature mapping φ(x): {phi_quadratic([1, 2, 3])}")
    print()


def polynomial_kernel_examples():
    """Examples of polynomial kernel K(x, z) = (x^T z + c)^k"""
    print("=== Polynomial Kernel Examples ===")
    
    x = np.array([1, 2, 3])
    z = np.array([4, 5, 6])
    
    # (x^T z + c)^2 kernel
    def poly2_kernel(x, z, c=1.0):
        return (np.dot(x, z) + c) ** 2
    
    print(f"Polynomial kernel (c=2.0): {poly2_kernel(x, z, c=2.0)}")
    
    # Explicit feature mapping for (x^T z + c)^2 (d=3)
    def phi_poly2(x, c=1.0):
        import math
        return [x[0]*x[0], x[0]*x[1], x[0]*x[2],
                x[1]*x[0], x[1]*x[1], x[1]*x[2],
                x[2]*x[0], x[2]*x[1], x[2]*x[2],
                math.sqrt(2*c)*x[0], math.sqrt(2*c)*x[1], math.sqrt(2*c)*x[2], c]
    
    print(f"Explicit feature mapping φ(x) for (x^T z + c)^2: {phi_poly2([1, 2, 3], c=2.0)}")
    
    # General polynomial kernel
    def poly_kernel(x, z, c=1.0, degree=3):
        return (np.dot(x, z) + c) ** degree
    
    print(f"General polynomial kernel (degree=3): {poly_kernel(x, z, c=1.0, degree=3)}")
    
    # Using scikit-learn
    X = np.array([[1, 2, 3], [4, 5, 6]])
    sk_poly = polynomial_kernel(X, X, degree=3, coef0=1.0)
    print(f"scikit-learn polynomial kernel:\n{sk_poly}")
    print()


def rbf_kernel_examples():
    """Examples of Gaussian (RBF) kernel K(x, z) = exp(-||x-z||^2/(2σ^2))"""
    print("=== RBF Kernel Examples ===")
    
    x = [1, 2, 3]
    z = [4, 5, 6]
    
    def rbf_kernel_custom(x, z, sigma=1.0):
        diff = np.array(x) - np.array(z)
        return np.exp(-np.dot(diff, diff) / (2 * sigma ** 2))
    
    print(f"Custom RBF kernel (σ=2.0): {rbf_kernel_custom(x, z, sigma=2.0)}")
    
    # Using scikit-learn
    X = np.array([[1, 2, 3], [4, 5, 6]])
    sk_rbf = rbf_kernel(X, X, gamma=1/(2*2.0**2))  # gamma = 1/(2*sigma^2)
    print(f"scikit-learn RBF kernel:\n{sk_rbf}")
    print()


def kernel_matrix_examples():
    """Examples of computing kernel matrices and checking properties"""
    print("=== Kernel Matrix Examples ===")
    
    def kernel_matrix(X, kernel_func):
        n = len(X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel_func(X[i], X[j])
        return K
    
    X = [[1, 0], [0, 1], [1, 1]]
    
    def quadratic_kernel(x, z):
        return sum(xi * zi for xi, zi in zip(x, z)) ** 2
    
    K = kernel_matrix(X, quadratic_kernel)
    print(f"Kernel matrix:\n{K}")
    
    # Check if matrix is symmetric
    is_symmetric = np.allclose(K, K.T)
    print(f"Is symmetric: {is_symmetric}")
    
    # Check if matrix is positive semidefinite
    eigvals = np.linalg.eigvalsh(K)
    print(f"Eigenvalues: {eigvals}")
    is_psd = np.all(eigvals >= -1e-10)  # Allow for small numerical errors
    print(f"Is positive semidefinite: {is_psd}")
    print()


def svm_with_kernels_example():
    """Example of SVM with RBF kernel using scikit-learn"""
    print("=== SVM with Kernels Example ===")
    
    # Generate moon dataset
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    
    # Create SVM with RBF kernel
    clf = svm.SVC(kernel='rbf', gamma=2.0)
    clf.fit(X, y)
    
    print(f"SVM trained on {len(X)} samples")
    print(f"Number of support vectors: {clf.n_support_}")
    
    # Create mesh for plotting decision boundary
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
    
    # Get decision function values
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred')
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    plt.title('SVM with RBF Kernel')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('svm_rbf_kernel.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("SVM with RBF kernel visualization saved as 'svm_rbf_kernel.png'")
    print()


def string_kernel_example():
    """Example of substring kernel for string similarity"""
    print("=== String Kernel Example ===")
    
    def substring_kernel(x, z, k=3):
        """
        Compute substring kernel between two strings x and z
        K(x, z) = sum over all k-length substrings of their co-occurrence counts
        """
        def k_substrings(s):
            return [s[i:i+k] for i in range(len(s)-k+1)]
        
        cx = Counter(k_substrings(x))
        cz = Counter(k_substrings(z))
        
        # Dot product in substring count space
        return sum(cx[sub] * cz[sub] for sub in set(cx) | set(cz))
    
    # Example with DNA sequences
    seq1 = 'GATTACA'
    seq2 = 'TACAGAT'
    
    print(f"String 1: {seq1}")
    print(f"String 2: {seq2}")
    
    # k=2 substrings
    k2_similarity = substring_kernel(seq1, seq2, k=2)
    print(f"Substring kernel (k=2): {k2_similarity}")
    
    # k=3 substrings
    k3_similarity = substring_kernel(seq1, seq2, k=3)
    print(f"Substring kernel (k=3): {k3_similarity}")
    
    # Show substrings for k=2
    def show_substrings(s, k):
        substrings = [s[i:i+k] for i in range(len(s)-k+1)]
        return substrings
    
    print(f"k=2 substrings of '{seq1}': {show_substrings(seq1, 2)}")
    print(f"k=2 substrings of '{seq2}': {show_substrings(seq2, 2)}")
    print()


def kernel_properties_demo():
    """Demonstrate various kernel properties and validations"""
    print("=== Kernel Properties Demonstration ===")
    
    # Test different kernel functions
    x = np.array([1, 2, 3])
    z = np.array([4, 5, 6])
    
    kernels = {
        'Linear': lambda x, z: np.dot(x, z),
        'Quadratic': lambda x, z: np.dot(x, z) ** 2,
        'Polynomial (degree=3)': lambda x, z: (np.dot(x, z) + 1) ** 3,
        'RBF (σ=1)': lambda x, z: np.exp(-np.linalg.norm(x-z)**2 / 2),
        'RBF (σ=2)': lambda x, z: np.exp(-np.linalg.norm(x-z)**2 / 8)
    }
    
    print("Kernel function values:")
    for name, kernel_func in kernels.items():
        value = kernel_func(x, z)
        print(f"{name}: {value:.4f}")
    
    # Test kernel matrix properties
    X_test = np.array([[1, 0], [0, 1], [1, 1], [2, 0]])
    
    print(f"\nTesting kernel matrix properties with {len(X_test)} points:")
    
    for name, kernel_func in kernels.items():
        K = kernel_matrix(X_test, kernel_func)
        
        # Check symmetry
        is_sym = np.allclose(K, K.T)
        
        # Check positive semidefinite
        eigvals = np.linalg.eigvalsh(K)
        is_psd = np.all(eigvals >= -1e-10)
        
        print(f"{name}: Symmetric={is_sym}, PSD={is_psd}")
    
    print()


def main():
    """Run all kernel examples"""
    print("Kernel Properties Examples")
    print("=" * 50)
    print()
    
    # Run all examples
    quadratic_kernel_examples()
    polynomial_kernel_examples()
    rbf_kernel_examples()
    kernel_matrix_examples()
    string_kernel_example()
    kernel_properties_demo()
    
    # Uncomment to run SVM visualization
    # svm_with_kernels_example()
    
    print("All examples completed!")


if __name__ == "__main__":
    main() 
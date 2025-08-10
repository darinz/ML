"""
Kernel Methods: Comprehensive Implementation Examples
==================================================

This file implements all the key concepts from the kernel methods document:
- Feature maps and the motivation for kernels
- The kernel trick and computational efficiency  
- Common kernel functions and their properties
- Kernelized algorithms and practical applications

The implementations demonstrate both the theoretical concepts and practical usage
with detailed annotations explaining the mathematical foundations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import time


# ============================================================================
# Section 5.1: Feature Maps and the Motivation for Kernels
# ============================================================================

def polynomial_feature_map(x, degree=3):
    """
    Create polynomial features up to given degree.
    
    This demonstrates the explicit feature mapping approach that leads to
    the curse of dimensionality. For degree k in d dimensions, we get O(d^k) features.
    
    Args:
        x: Input value (scalar or array)
        degree: Maximum polynomial degree
        
    Returns:
        Array of polynomial features [1, x, x^2, ..., x^degree]
        
    Example:
        For degree=2: [1, x, x^2]
        For degree=3: [1, x, x^2, x^3]
    """
    if np.isscalar(x):
        features = [1]  # bias term
        for d in range(1, degree + 1):
            features.append(x ** d)
        return np.array(features)
    else:
        # Handle array input
        features = [np.ones_like(x)]  # bias term
        for d in range(1, degree + 1):
            features.append(x ** d)
        return np.column_stack(features)


def demonstrate_curse_of_dimensionality():
    """
    Demonstrate the exponential growth of polynomial features.
    
    This shows why explicit feature computation becomes prohibitive
    for high-dimensional data, motivating the need for the kernel trick.
    """
    print("=== Curse of Dimensionality Demonstration ===")
    
    dimensions = [1, 2, 5, 10, 20]
    degrees = [2, 3, 4]
    
    print("Number of polynomial features for different dimensions and degrees:")
    print("Dimensions | Degree=2 | Degree=3 | Degree=4")
    print("-" * 45)
    
    for d in dimensions:
        row = f"{d:10d} |"
        for k in degrees:
            # Calculate number of features: sum_{i=0}^k C(d+i-1, i)
            n_features = 1
            for i in range(1, k + 1):
                n_features += np.math.comb(d + i - 1, i)
            row += f" {n_features:8d} |"
        print(row)
    
    print("\nThis exponential growth makes explicit feature computation")
    print("computationally prohibitive for high-dimensional data!")
    print()


def lms_with_features(X, y, feature_map, learning_rate=0.01, max_iterations=1000):
    """
    LMS algorithm with custom feature map.
    
    This implements the standard LMS algorithm in the feature space.
    The computational cost is O(p) per update where p is the feature dimension.
    
    Args:
        X: Training data (n_samples, n_features)
        y: Target values
        feature_map: Function that maps input to features
        learning_rate: Learning rate for gradient descent
        max_iterations: Maximum number of iterations
        
    Returns:
        theta: Learned parameters in feature space
    """
    n_samples = X.shape[0]
    
    # Initialize parameters in feature space
    sample_features = feature_map(X[0])
    theta = np.zeros(len(sample_features))
    
    print(f"Training LMS with {len(sample_features)} features...")
    
    for iteration in range(max_iterations):
        total_error = 0
        for i in range(n_samples):
            # Compute features: O(p) operation
            phi_x = feature_map(X[i])
            
            # Compute prediction: O(p) operation
            prediction = np.dot(theta, phi_x)
            
            # Compute error
            error = y[i] - prediction
            total_error += abs(error)
            
            # Update parameters: O(p) operation
            theta += learning_rate * error * phi_x
        
        # Print progress every 100 iterations
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Average error = {total_error/n_samples:.4f}")
    
    return theta


def example_lms_with_features():
    """
    Example demonstrating LMS with polynomial features.
    
    This shows how polynomial features can capture non-linear patterns,
    but at the cost of increased computational complexity.
    """
    print("=== LMS with Polynomial Features Example ===")
    
    # Create quadratic data: y = x^2 + noise
    np.random.seed(42)
    X = np.linspace(0, 5, 20).reshape(-1, 1)
    y = X.flatten() ** 2 + np.random.normal(0, 0.5, 20)
    
    print("Data: y = x^2 + noise")
    print(f"Training on {len(X)} samples...")
    
    # Learn with polynomial features
    start_time = time.time()
    theta = lms_with_features(X, y, lambda x: polynomial_feature_map(x, degree=2))
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.4f} seconds")
    print(f"Learned parameters: {theta}")
    
    # Make predictions
    X_test = np.array([1.5, 2.5, 3.5]).reshape(-1, 1)
    predictions = []
    for x in X_test:
        phi_x = polynomial_feature_map(x, degree=2)
        pred = np.dot(theta, phi_x)
        predictions.append(pred)
    
    print(f"\nTest predictions: {predictions}")
    print(f"True values: {X_test.flatten() ** 2}")
    print(f"Mean squared error: {mean_squared_error(X_test.flatten() ** 2, predictions):.4f}")
    print()


# ============================================================================
# Section 5.3: The Kernel Trick: Efficient Computation
# ============================================================================

def polynomial_kernel(x, z, degree=3, gamma=1.0, r=1.0):
    """
    Compute polynomial kernel K(x, z) = (gamma * <x, z> + r)^degree
    
    This kernel corresponds to polynomial features of degree 'degree'.
    Instead of computing O(d^degree) features explicitly, we compute
    the kernel in O(d) time using the kernel trick.
    
    Args:
        x, z: Input vectors
        degree: Polynomial degree
        gamma: Scaling parameter
        r: Bias parameter
        
    Returns:
        Kernel value
        
    Mathematical foundation:
        K(x, z) = (γ⟨x, z⟩ + r)^d = Σ_{i=0}^d C(d,i) (γ⟨x, z⟩)^i r^(d-i)
        This corresponds to all monomials up to degree d.
    """
    inner_product = np.dot(x, z)
    return (gamma * inner_product + r) ** degree


def rbf_kernel(x, z, gamma=1.0):
    """
    Compute RBF (Gaussian) kernel K(x, z) = exp(-gamma * ||x - z||^2)
    
    This is one of the most popular kernels because:
    1. It can approximate any continuous function (universal kernel)
    2. It provides smooth, local similarity measures
    3. It works well for most problems (default choice)
    
    Args:
        x, z: Input vectors
        gamma: Bandwidth parameter (controls the "reach" of each point)
        
    Returns:
        Kernel value
        
    Intuition:
        - Points close together have high similarity (≈ 1)
        - Points far apart have low similarity (≈ 0)
        - γ controls how quickly similarity decays with distance
    """
    diff = x - z
    return np.exp(-gamma * np.dot(diff, diff))


def linear_kernel(x, z):
    """
    Compute linear kernel K(x, z) = <x, z>
    
    This is the simplest kernel, corresponding to no feature transformation.
    It's equivalent to working in the original input space.
    
    Args:
        x, z: Input vectors
        
    Returns:
        Kernel value (inner product)
    """
    return np.dot(x, z)


def sigmoid_kernel(x, z, gamma=1.0, r=0.0):
    """
    Compute sigmoid kernel K(x, z) = tanh(gamma * <x, z> + r)
    
    This kernel is inspired by neural network activation functions.
    Note: It's not always positive definite, so it may not work with all algorithms.
    
    Args:
        x, z: Input vectors
        gamma: Scaling parameter
        r: Bias parameter
        
    Returns:
        Kernel value
    """
    inner_product = np.dot(x, z)
    return np.tanh(gamma * inner_product + r)


def kernelized_lms(X, y, kernel_func, learning_rate=0.01, max_iterations=1000):
    """
    Kernelized LMS algorithm.
    
    This implements the kernelized version of LMS using the representer theorem.
    Instead of working with parameters θ in feature space, we work with
    coefficients β in the dual space.
    
    Args:
        X: Training data (n_samples, n_features)
        y: Target values
        kernel_func: Kernel function
        learning_rate: Learning rate for gradient descent
        max_iterations: Maximum number of iterations
        
    Returns:
        beta: Dual coefficients
        K: Kernel matrix
        
    Mathematical foundation:
        The representer theorem states that θ = Σ_i β_i φ(x_i)
        This allows us to work entirely with kernel evaluations.
    """
    n_samples = X.shape[0]
    
    print(f"Computing kernel matrix for {n_samples} samples...")
    start_time = time.time()
    
    # Pre-compute kernel matrix: O(n²d) operation
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    
    kernel_time = time.time() - start_time
    print(f"Kernel matrix computed in {kernel_time:.4f} seconds")
    
    # Initialize beta (dual coefficients)
    beta = np.zeros(n_samples)
    
    print("Training kernelized LMS...")
    
    # Gradient descent in dual space
    for iteration in range(max_iterations):
        # Compute predictions: O(n²) operation
        predictions = K @ beta
        
        # Compute errors
        errors = y - predictions
        
        # Update beta: O(n) operation
        beta += learning_rate * errors
        
        # Print progress
        if iteration % 100 == 0:
            mse = np.mean(errors ** 2)
            print(f"Iteration {iteration}: MSE = {mse:.4f}")
    
    return beta, K


def predict_kernelized(X_train, X_test, beta, kernel_func):
    """
    Make predictions using kernelized model.
    
    Args:
        X_train: Training data
        X_test: Test data
        beta: Dual coefficients from training
        kernel_func: Kernel function
        
    Returns:
        Predictions for test data
        
    Formula: f(x) = Σ_i β_i K(x_i, x) + b
    """
    predictions = []
    for x_test in X_test:
        # Compute kernel with all training points
        kernel_values = [kernel_func(x_train, x_test) for x_train in X_train]
        prediction = np.dot(beta, kernel_values)
        predictions.append(prediction)
    
    return np.array(predictions)


def example_kernelized_lms():
    """
    Example demonstrating kernelized LMS.
    
    This shows how the kernel trick allows us to work in high-dimensional
    feature spaces efficiently without explicitly computing features.
    """
    print("=== Kernelized LMS Example ===")
    
    # Create non-linear data
    np.random.seed(42)
    X = np.linspace(0, 5, 50).reshape(-1, 1)
    y = np.sin(X.flatten()) + np.random.normal(0, 0.1, 50)
    
    print("Data: y = sin(x) + noise")
    print(f"Training on {len(X)} samples...")
    
    # Compare different kernels
    kernels = [
        ("Linear", lambda x, z: linear_kernel(x, z)),
        ("Polynomial (degree=2)", lambda x, z: polynomial_kernel(x, z, degree=2)),
        ("RBF (γ=1)", lambda x, z: rbf_kernel(x, z, gamma=1.0)),
        ("RBF (γ=5)", lambda x, z: rbf_kernel(x, z, gamma=5.0))
    ]
    
    results = {}
    
    for kernel_name, kernel_func in kernels:
        print(f"\n--- {kernel_name} Kernel ---")
        
        # Train kernelized LMS
        start_time = time.time()
        beta, K = kernelized_lms(X, y, kernel_func, learning_rate=0.01, max_iterations=500)
        training_time = time.time() - start_time
        
        # Make predictions
        X_test = np.linspace(0, 5, 100).reshape(-1, 1)
        predictions = predict_kernelized(X, X_test, beta, kernel_func)
        
        # Compute error
        y_test_true = np.sin(X_test.flatten())
        mse = mean_squared_error(y_test_true, predictions)
        
        results[kernel_name] = {
            'training_time': training_time,
            'mse': mse,
            'predictions': predictions,
            'beta': beta
        }
        
        print(f"Training time: {training_time:.4f} seconds")
        print(f"Test MSE: {mse:.4f}")
        print(f"Number of non-zero coefficients: {np.sum(np.abs(beta) > 1e-6)}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    for i, (kernel_name, result) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        plt.scatter(X, y, alpha=0.6, label='Training data')
        plt.plot(X_test, result['predictions'], 'r-', linewidth=2, label='Predictions')
        plt.plot(X_test, np.sin(X_test.flatten()), 'g--', linewidth=2, label='True function')
        plt.title(f'{kernel_name}\nMSE: {result["mse"]:.4f}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


def compare_explicit_vs_kernel():
    """
    Compare explicit feature computation vs kernel trick.
    
    This demonstrates the computational advantage of the kernel trick
    for high-dimensional feature spaces.
    """
    print("=== Explicit vs Kernel Comparison ===")
    
    # Create data
    np.random.seed(42)
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    y = np.random.randn(100)
    
    degree = 3
    
    # Method 1: Explicit polynomial features
    print("Method 1: Explicit polynomial features")
    start_time = time.time()
    
    # Compute explicit features (this would be very expensive for high dimensions)
    X_poly = np.ones((X.shape[0], 1))  # bias term
    for d in range(1, degree + 1):
        # This is a simplified version - full polynomial features would be much larger
        X_poly = np.column_stack([X_poly, X ** d])
    
    explicit_time = time.time() - start_time
    print(f"Explicit feature computation time: {explicit_time:.4f} seconds")
    print(f"Feature dimension: {X_poly.shape[1]}")
    
    # Method 2: Kernel trick
    print("\nMethod 2: Kernel trick")
    start_time = time.time()
    
    # Compute kernel matrix
    K = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            K[i, j] = polynomial_kernel(X[i], X[j], degree=degree)
    
    kernel_time = time.time() - start_time
    print(f"Kernel computation time: {kernel_time:.4f} seconds")
    print(f"Kernel matrix size: {K.shape}")
    
    # Compare computational complexity
    print(f"\nSpeedup: {explicit_time/kernel_time:.2f}x")
    print("Note: For higher dimensions, the speedup would be much more dramatic!")
    print()


def main():
    """
    Main function to run all examples.
    """
    print("Kernel Methods: Comprehensive Examples")
    print("=" * 50)
    
    # Run all examples
    demonstrate_curse_of_dimensionality()
    example_lms_with_features()
    example_kernelized_lms()
    compare_explicit_vs_kernel()
    
    print("All examples completed!")


if __name__ == "__main__":
    main() 
"""
Kernel Methods: Implementation Examples
=====================================

This file contains all the Python code examples from the kernel methods document,
organized by sections with clear implementations and examples.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Section 5.2: LMS (Least Mean Squares) with Features
# ============================================================================

def polynomial_feature_map(x, degree=3):
    """
    Create polynomial features up to given degree.
    
    Args:
        x: Input value (scalar or array)
        degree: Maximum polynomial degree
        
    Returns:
        Array of polynomial features
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


def lms_with_features(X, y, feature_map, learning_rate=0.01, max_iterations=1000):
    """
    LMS algorithm with custom feature map.
    
    Args:
        X: Training data (n_samples, n_features)
        y: Target values
        feature_map: Function that maps input to features
        learning_rate: Learning rate for gradient descent
        max_iterations: Maximum number of iterations
        
    Returns:
        theta: Learned parameters
    """
    n_samples = X.shape[0]
    
    # Initialize parameters
    # We need to determine the feature dimension
    sample_features = feature_map(X[0])
    theta = np.zeros(len(sample_features))
    
    for iteration in range(max_iterations):
        for i in range(n_samples):
            # Compute features
            phi_x = feature_map(X[i])
            
            # Compute prediction
            prediction = np.dot(theta, phi_x)
            
            # Compute error
            error = y[i] - prediction
            
            # Update parameters
            theta += learning_rate * error * phi_x
    
    return theta


def example_lms_with_features():
    """Example demonstrating LMS with polynomial features."""
    print("=== LMS with Polynomial Features Example ===")
    
    # Create quadratic data: y = x^2
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([1, 4, 9, 16, 25])
    
    # Learn with polynomial features
    theta = lms_with_features(X, y, lambda x: polynomial_feature_map(x, degree=2))
    print(f"Learned parameters: {theta}")
    
    # Make predictions
    X_test = np.array([1.5, 2.5, 3.5]).reshape(-1, 1)
    predictions = []
    for x in X_test:
        phi_x = polynomial_feature_map(x, degree=2)
        pred = np.dot(theta, phi_x)
        predictions.append(pred)
    
    print(f"Test predictions: {predictions}")
    print(f"True values: {X_test.flatten() ** 2}")
    print()


# ============================================================================
# Section 5.3: The Kernel Trick: Efficient Computation
# ============================================================================

def polynomial_kernel(x, z, degree=3, gamma=1.0, r=1.0):
    """
    Compute polynomial kernel K(x, z) = (gamma * <x, z> + r)^degree
    
    Args:
        x, z: Input vectors
        degree: Polynomial degree
        gamma: Scaling parameter
        r: Bias parameter
        
    Returns:
        Kernel value
    """
    inner_product = np.dot(x, z)
    return (gamma * inner_product + r) ** degree


def rbf_kernel(x, z, gamma=1.0):
    """
    Compute RBF (Gaussian) kernel K(x, z) = exp(-gamma * ||x - z||^2)
    
    Args:
        x, z: Input vectors
        gamma: Bandwidth parameter
        
    Returns:
        Kernel value
    """
    diff = x - z
    return np.exp(-gamma * np.dot(diff, diff))


def linear_kernel(x, z):
    """Compute linear kernel K(x, z) = <x, z>"""
    return np.dot(x, z)


def sigmoid_kernel(x, z, gamma=1.0, r=0.0):
    """
    Compute sigmoid kernel K(x, z) = tanh(gamma * <x, z> + r)
    
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
    
    Args:
        X: Training data (n_samples, n_features)
        y: Target values
        kernel_func: Kernel function
        learning_rate: Learning rate for gradient descent
        max_iterations: Maximum number of iterations
        
    Returns:
        beta: Dual coefficients
        K: Kernel matrix
    """
    n_samples = X.shape[0]
    
    # Pre-compute kernel matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    
    # Initialize beta
    beta = np.zeros(n_samples)
    
    # Gradient descent
    for iteration in range(max_iterations):
        # Compute predictions
        predictions = K @ beta
        
        # Compute errors
        errors = y - predictions
        
        # Update beta
        beta += learning_rate * errors
    
    return beta, K


def predict_kernelized(X_train, X_test, beta, kernel_func):
    """
    Make predictions using kernelized model.
    
    Args:
        X_train: Training data
        X_test: Test data
        beta: Dual coefficients
        kernel_func: Kernel function
        
    Returns:
        Predictions for test data
    """
    predictions = []
    for x_test in X_test:
        prediction = 0
        for i, x_train in enumerate(X_train):
            prediction += beta[i] * kernel_func(x_train, x_test)
        predictions.append(prediction)
    return np.array(predictions)


def example_kernelized_lms():
    """Example demonstrating kernelized LMS."""
    print("=== Kernelized LMS Example ===")
    
    # Create quadratic data: y = x^2
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([1, 4, 9, 16, 25])
    
    # Learn with polynomial kernel
    beta, K = kernelized_lms(X, y, lambda x, z: polynomial_kernel(x, z, degree=2))
    print(f"Learned beta coefficients: {beta}")
    
    # Make predictions
    X_test = np.array([[1.5], [2.5], [3.5]])
    predictions = predict_kernelized(X, X_test, beta, lambda x, z: polynomial_kernel(x, z, degree=2))
    print(f"Test predictions: {predictions}")
    print(f"True values: {X_test.flatten() ** 2}")
    print()


# ============================================================================
# Section 5.6: Practical Considerations - Hyperparameter Tuning
# ============================================================================

def example_hyperparameter_tuning():
    """Example demonstrating hyperparameter tuning for kernel methods."""
    print("=== Hyperparameter Tuning Example ===")
    
    # Generate synthetic data
    X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define parameter grid for RBF kernel
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1]
    }
    
    # Perform grid search
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_scaled, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {-grid_search.best_score_:.4f}")
    print()


# ============================================================================
# Section 5.7: Advanced Topics
# ============================================================================

def kernel_ridge_regression(X, y, kernel_func, lambda_reg=1.0):
    """
    Kernel Ridge Regression.
    
    Args:
        X: Training data
        y: Target values
        kernel_func: Kernel function
        lambda_reg: Regularization parameter
        
    Returns:
        beta: Dual coefficients
        K: Kernel matrix
    """
    n_samples = X.shape[0]
    
    # Compute kernel matrix
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    
    # Solve ridge regression: beta = (K + lambda*I)^(-1) * y
    I = np.eye(n_samples)
    beta = np.linalg.solve(K + lambda_reg * I, y)
    
    return beta, K


def multiple_kernel_learning(X, y, kernel_list, alpha_weights=None):
    """
    Multiple Kernel Learning example.
    
    Args:
        X: Training data
        y: Target values
        kernel_list: List of kernel functions
        alpha_weights: Weights for each kernel (optional)
        
    Returns:
        Combined kernel matrix
    """
    n_samples = X.shape[0]
    n_kernels = len(kernel_list)
    
    if alpha_weights is None:
        alpha_weights = np.ones(n_kernels) / n_kernels
    
    # Compute combined kernel matrix
    K_combined = np.zeros((n_samples, n_samples))
    for i, kernel_func in enumerate(kernel_list):
        K_kernel = np.zeros((n_samples, n_samples))
        for j in range(n_samples):
            for k in range(n_samples):
                K_kernel[j, k] = kernel_func(X[j], X[k])
        K_combined += alpha_weights[i] * K_kernel
    
    return K_combined


def example_advanced_topics():
    """Example demonstrating advanced kernel topics."""
    print("=== Advanced Topics Example ===")
    
    # Generate synthetic data
    X = np.random.randn(50, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(50)
    
    # Kernel Ridge Regression
    beta_krr, K_krr = kernel_ridge_regression(X, y, lambda x, z: rbf_kernel(x, z, gamma=1.0), lambda_reg=0.1)
    print(f"Kernel Ridge Regression - beta shape: {beta_krr.shape}")
    
    # Multiple Kernel Learning
    kernel_list = [
        lambda x, z: linear_kernel(x, z),
        lambda x, z: rbf_kernel(x, z, gamma=1.0),
        lambda x, z: polynomial_kernel(x, z, degree=2)
    ]
    K_mkl = multiple_kernel_learning(X, y, kernel_list)
    print(f"Multiple Kernel Learning - combined kernel shape: {K_mkl.shape}")
    print()


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_kernel_comparison():
    """Plot comparison of different kernel functions."""
    print("=== Kernel Function Comparison ===")
    
    # Create data points
    x = np.linspace(-3, 3, 100)
    z = np.array([0.0])  # Fixed reference point
    
    # Compute different kernels
    linear_vals = [linear_kernel([xi], z) for xi in x]
    poly_vals = [polynomial_kernel([xi], z, degree=2) for xi in x]
    rbf_vals = [rbf_kernel([xi], z, gamma=1.0) for xi in x]
    sigmoid_vals = [sigmoid_kernel([xi], z, gamma=1.0) for xi in x]
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, linear_vals)
    plt.title('Linear Kernel')
    plt.xlabel('x')
    plt.ylabel('K(x, 0)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, poly_vals)
    plt.title('Polynomial Kernel (degree=2)')
    plt.xlabel('x')
    plt.ylabel('K(x, 0)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x, rbf_vals)
    plt.title('RBF Kernel (γ=1.0)')
    plt.xlabel('x')
    plt.ylabel('K(x, 0)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, sigmoid_vals)
    plt.title('Sigmoid Kernel (γ=1.0)')
    plt.xlabel('x')
    plt.ylabel('K(x, 0)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('03_advanced_classification/kernel_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Kernel comparison plot saved as 'kernel_comparison.png'")
    print()


def plot_kernel_matrix_heatmap():
    """Plot heatmap of kernel matrix."""
    print("=== Kernel Matrix Visualization ===")
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(20, 2)
    
    # Compute RBF kernel matrix
    K = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            K[i, j] = rbf_kernel(X[i], X[j], gamma=1.0)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(K, cmap='viridis', aspect='auto')
    plt.colorbar(label='Kernel Value')
    plt.title('RBF Kernel Matrix Heatmap')
    plt.xlabel('Training Sample Index')
    plt.ylabel('Training Sample Index')
    plt.savefig('03_advanced_classification/kernel_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Kernel matrix heatmap saved as 'kernel_matrix_heatmap.png'")
    print()


# ============================================================================
# Performance Comparison Functions
# ============================================================================

def compare_explicit_vs_kernel():
    """Compare explicit feature computation vs kernel trick."""
    print("=== Explicit Features vs Kernel Trick Comparison ===")
    
    # Generate data
    n_samples = 100
    d = 10  # Input dimension
    degree = 3  # Polynomial degree
    
    X = np.random.randn(n_samples, d)
    y = np.random.randn(n_samples)
    
    # Time explicit feature computation
    import time
    
    start_time = time.time()
    # Explicit polynomial features would be O(d^degree)
    # For demonstration, we'll use a simplified version
    explicit_features = np.column_stack([X, X**2, X**3])
    explicit_time = time.time() - start_time
    
    # Time kernel computation
    start_time = time.time()
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = polynomial_kernel(X[i], X[j], degree=degree)
    kernel_time = time.time() - start_time
    
    print(f"Explicit feature computation time: {explicit_time:.4f} seconds")
    print(f"Kernel computation time: {kernel_time:.4f} seconds")
    print(f"Speedup: {explicit_time/kernel_time:.2f}x")
    print()


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run all examples and demonstrations."""
    print("Kernel Methods Implementation Examples")
    print("=" * 50)
    print()
    
    # Run all examples
    example_lms_with_features()
    example_kernelized_lms()
    example_hyperparameter_tuning()
    example_advanced_topics()
    
    # Generate visualizations
    try:
        plot_kernel_comparison()
        plot_kernel_matrix_heatmap()
    except Exception as e:
        print(f"Visualization failed (matplotlib not available): {e}")
    
    # Performance comparison
    compare_explicit_vs_kernel()
    
    print("All examples completed successfully!")


if __name__ == "__main__":
    main() 
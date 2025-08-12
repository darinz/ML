"""
Double Descent Phenomenon Demonstration

This module demonstrates the double descent phenomenon using polynomial regression,
showing how model complexity affects generalization beyond the classical U-shaped curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures


def demonstrate_double_descent():
    """Demonstrate double descent with polynomial regression"""
    
    # Generate data
    np.random.seed(42)
    n_samples = 50
    X = np.random.uniform(-2, 2, n_samples).reshape(-1, 1)
    true_function = 0.5 * X**2 + 0.3 * X + 1.0
    noise = 0.1 * np.random.randn(n_samples, 1)
    y = true_function + noise
    
    # Test different polynomial degrees
    degrees = list(range(1, 21))  # 1 to 20
    train_errors = []
    test_errors = []
    test_errors_ridge = []
    
    # Generate test data
    X_test = np.random.uniform(-2, 2, 100).reshape(-1, 1)
    y_test = 0.5 * X_test**2 + 0.3 * X_test + 1.0 + 0.1 * np.random.randn(100, 1)
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_test_poly = poly.transform(X_test)
        
        # Unregularized (can show double descent)
        model = LinearRegression()
        model.fit(X_poly, y)
        
        train_error = np.mean((model.predict(X_poly) - y)**2)
        test_error = np.mean((model.predict(X_test_poly) - y_test)**2)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
        
        # Regularized (smoother curve)
        ridge = Ridge(alpha=0.1)
        ridge.fit(X_poly, y)
        test_error_ridge = np.mean((ridge.predict(X_test_poly) - y_test)**2)
        test_errors_ridge.append(test_error_ridge)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(degrees, train_errors, 'b-', label='Training Error', linewidth=2)
    plt.plot(degrees, test_errors, 'r-', label='Test Error', linewidth=2)
    plt.axvline(x=n_samples, color='g', linestyle='--', alpha=0.7, label='Interpolation Threshold')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Double Descent Phenomenon')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(degrees, test_errors, 'r-', label='Unregularized', linewidth=2)
    plt.plot(degrees, test_errors_ridge, 'g-', label='Ridge (α=0.1)', linewidth=2)
    plt.axvline(x=n_samples, color='g', linestyle='--', alpha=0.7, label='Interpolation Threshold')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Test Error')
    plt.title('Effect of Regularization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(degrees, np.array(test_errors) - np.array(train_errors), 'purple', linewidth=2)
    plt.axvline(x=n_samples, color='g', linestyle='--', alpha=0.7, label='Interpolation Threshold')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Generalization Gap')
    plt.title('Overfitting Measure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return degrees, train_errors, test_errors, test_errors_ridge


def demonstrate_sample_wise_double_descent():
    """Demonstrate sample-wise double descent by varying the number of training samples"""
    
    # Generate a large dataset
    np.random.seed(42)
    n_total = 200
    X_total = np.random.uniform(-2, 2, n_total).reshape(-1, 1)
    true_function = 0.5 * X_total**2 + 0.3 * X_total + 1.0
    noise = 0.1 * np.random.randn(n_total, 1)
    y_total = true_function + noise
    
    # Test different sample sizes
    sample_sizes = list(range(10, 101, 5))  # 10 to 100 samples
    test_errors = []
    test_errors_ridge = []
    
    # Fixed model complexity (polynomial degree)
    degree = 20  # Overparameterized model
    
    # Generate test data
    X_test = np.random.uniform(-2, 2, 100).reshape(-1, 1)
    y_test = 0.5 * X_test**2 + 0.3 * X_test + 1.0 + 0.1 * np.random.randn(100, 1)
    
    for n_samples in sample_sizes:
        # Sample training data
        indices = np.random.choice(n_total, n_samples, replace=False)
        X_train = X_total[indices]
        y_train = y_total[indices]
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Unregularized
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        test_error = np.mean((model.predict(X_test_poly) - y_test)**2)
        test_errors.append(test_error)
        
        # Regularized
        ridge = Ridge(alpha=0.1)
        ridge.fit(X_train_poly, y_train)
        test_error_ridge = np.mean((ridge.predict(X_test_poly) - y_test)**2)
        test_errors_ridge.append(test_error_ridge)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(sample_sizes, test_errors, 'r-', label='Unregularized', linewidth=2)
    plt.axvline(x=degree+1, color='g', linestyle='--', alpha=0.7, label='Interpolation Threshold')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Test Error')
    plt.title('Sample-Wise Double Descent')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(sample_sizes, test_errors, 'r-', label='Unregularized', linewidth=2)
    plt.plot(sample_sizes, test_errors_ridge, 'g-', label='Ridge (α=0.1)', linewidth=2)
    plt.axvline(x=degree+1, color='g', linestyle='--', alpha=0.7, label='Interpolation Threshold')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Test Error')
    plt.title('Effect of Regularization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(sample_sizes, np.array(test_errors_ridge) - np.array(test_errors), 'purple', linewidth=2)
    plt.axvline(x=degree+1, color='g', linestyle='--', alpha=0.7, label='Interpolation Threshold')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Regularization Effect')
    plt.title('Benefit of Regularization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return sample_sizes, test_errors, test_errors_ridge


def demonstrate_minimum_norm_solution():
    """Demonstrate the minimum norm solution property of gradient descent"""
    
    # Generate data
    np.random.seed(42)
    n_samples = 30
    n_features = 50  # Overparameterized regime
    
    X = np.random.randn(n_samples, n_features)
    true_beta = np.random.randn(n_features, 1) * 0.1  # Sparse true solution
    y = X @ true_beta + 0.1 * np.random.randn(n_samples, 1)
    
    # Find minimum norm solution (what gradient descent converges to)
    beta_min_norm = np.linalg.pinv(X) @ y
    
    # Find other solutions that fit the data
    # Add random vectors in the null space of X
    U, S, Vt = np.linalg.svd(X, full_matrices=True)
    null_space = Vt[n_samples:, :].T  # Vectors in null space
    
    # Create alternative solutions
    beta_alt1 = beta_min_norm + 0.1 * null_space[:, 0:1]
    beta_alt2 = beta_min_norm + 0.5 * null_space[:, 0:1]
    
    # Test on new data
    X_test = np.random.randn(100, n_features)
    y_test = X_test @ true_beta + 0.1 * np.random.randn(100, 1)
    
    # Calculate test errors
    error_min_norm = np.mean((X_test @ beta_min_norm - y_test)**2)
    error_alt1 = np.mean((X_test @ beta_alt1 - y_test)**2)
    error_alt2 = np.mean((X_test @ beta_alt2 - y_test)**2)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Compare solutions
    plt.subplot(1, 3, 1)
    solutions = ['Min Norm', 'Alt 1', 'Alt 2']
    test_errors = [error_min_norm, error_alt1, error_alt2]
    colors = ['blue', 'red', 'green']
    
    bars = plt.bar(solutions, test_errors, color=colors, alpha=0.7)
    plt.ylabel('Test Error')
    plt.title('Generalization of Different Solutions')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, error in zip(bars, test_errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{error:.4f}', ha='center', va='bottom')
    
    # Compare norms
    plt.subplot(1, 3, 2)
    norms = [np.linalg.norm(beta_min_norm), np.linalg.norm(beta_alt1), np.linalg.norm(beta_alt2)]
    
    bars = plt.bar(solutions, norms, color=colors, alpha=0.7)
    plt.ylabel('L2 Norm')
    plt.title('Solution Norms')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, norm in zip(bars, norms):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{norm:.3f}', ha='center', va='bottom')
    
    # Training error (should be zero for all)
    plt.subplot(1, 3, 3)
    train_errors = [np.mean((X @ beta_min_norm - y)**2), 
                   np.mean((X @ beta_alt1 - y)**2), 
                   np.mean((X @ beta_alt2 - y)**2)]
    
    bars = plt.bar(solutions, train_errors, color=colors, alpha=0.7)
    plt.ylabel('Training Error')
    plt.title('Training Error (All Zero)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, error in zip(bars, train_errors):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{error:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Minimum Norm Solution Analysis:")
    print("=" * 50)
    print(f"Minimum Norm Solution:")
    print(f"  Test Error: {error_min_norm:.6f}")
    print(f"  L2 Norm: {np.linalg.norm(beta_min_norm):.6f}")
    print(f"  Training Error: {np.mean((X @ beta_min_norm - y)**2):.6f}")
    print()
    print(f"Alternative Solution 1:")
    print(f"  Test Error: {error_alt1:.6f}")
    print(f"  L2 Norm: {np.linalg.norm(beta_alt1):.6f}")
    print(f"  Training Error: {np.mean((X @ beta_alt1 - y)**2):.6f}")
    print()
    print(f"Alternative Solution 2:")
    print(f"  Test Error: {error_alt2:.6f}")
    print(f"  L2 Norm: {np.linalg.norm(beta_alt2):.6f}")
    print(f"  Training Error: {np.mean((X @ beta_alt2 - y)**2):.6f}")
    
    return beta_min_norm, beta_alt1, beta_alt2, test_errors


def demonstrate_complexity_measures():
    """Demonstrate how different complexity measures affect double descent"""
    
    # Generate data
    np.random.seed(42)
    n_samples = 50
    X = np.random.uniform(-2, 2, n_samples).reshape(-1, 1)
    true_function = 0.5 * X**2 + 0.3 * X + 1.0
    noise = 0.1 * np.random.randn(n_samples, 1)
    y = true_function + noise
    
    # Test different polynomial degrees
    degrees = list(range(1, 21))
    parameter_counts = []
    model_norms = []
    test_errors = []
    
    # Generate test data
    X_test = np.random.uniform(-2, 2, 100).reshape(-1, 1)
    y_test = 0.5 * X_test**2 + 0.3 * X_test + 1.0 + 0.1 * np.random.randn(100, 1)
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_test_poly = poly.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # Calculate complexity measures
        param_count = X_poly.shape[1]  # Number of parameters
        model_norm = np.linalg.norm(model.coef_)  # L2 norm of coefficients
        
        # Calculate test error
        test_error = np.mean((model.predict(X_test_poly) - y_test)**2)
        
        parameter_counts.append(param_count)
        model_norms.append(model_norm)
        test_errors.append(test_error)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Double descent vs parameter count
    plt.subplot(1, 3, 1)
    plt.plot(parameter_counts, test_errors, 'r-', linewidth=2)
    plt.axvline(x=n_samples, color='g', linestyle='--', alpha=0.7, label='Interpolation Threshold')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Error')
    plt.title('Double Descent vs Parameter Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Model norm vs parameter count
    plt.subplot(1, 3, 2)
    plt.plot(parameter_counts, model_norms, 'b-', linewidth=2)
    plt.axvline(x=n_samples, color='g', linestyle='--', alpha=0.7, label='Interpolation Threshold')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Model Norm')
    plt.title('Model Norm vs Parameter Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test error vs model norm
    plt.subplot(1, 3, 3)
    plt.plot(model_norms, test_errors, 'purple', linewidth=2)
    plt.xlabel('Model Norm')
    plt.ylabel('Test Error')
    plt.title('Test Error vs Model Norm')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print analysis
    print("Complexity Measures Analysis:")
    print("=" * 50)
    print(f"Interpolation threshold (n=d): {n_samples} parameters")
    print(f"Peak test error at: {parameter_counts[np.argmax(test_errors)]} parameters")
    print(f"Peak model norm at: {parameter_counts[np.argmax(model_norms)]} parameters")
    print(f"Best test error at: {parameter_counts[np.argmin(test_errors)]} parameters")
    
    return parameter_counts, model_norms, test_errors


if __name__ == "__main__":
    # Run all demonstrations
    print("Running Double Descent Demonstrations...")
    print("=" * 60)
    
    print("\n1. Model-Wise Double Descent:")
    degrees, train_errors, test_errors, test_errors_ridge = demonstrate_double_descent()
    
    print("\n2. Sample-Wise Double Descent:")
    sample_sizes, test_errors_sample, test_errors_ridge_sample = demonstrate_sample_wise_double_descent()
    
    print("\n3. Minimum Norm Solution:")
    beta_min_norm, beta_alt1, beta_alt2, test_errors_norm = demonstrate_minimum_norm_solution()
    
    print("\n4. Complexity Measures:")
    param_counts, model_norms, test_errors_complexity = demonstrate_complexity_measures()
    
    print("\nAll demonstrations completed!")

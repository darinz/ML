"""
Bias-Variance Tradeoff Demonstration

This module demonstrates the bias-variance tradeoff using polynomial regression
with different degrees, showing how model complexity affects training and test error.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def simulate_bias_variance_tradeoff():
    """Demonstrate bias-variance tradeoff with polynomial regression"""
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    true_function = 0.5 * X**2 + 0.3 * X + 1.0
    noise = 0.1 * np.random.randn(100, 1)
    y = true_function + noise
    
    # Test different polynomial degrees
    degrees = [1, 2, 3, 5, 10]
    train_errors = []
    test_errors = []
    
    for degree in degrees:
        # Create polynomial model
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Train on subset of data
        train_idx = np.random.choice(100, 50, replace=False)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[~train_idx], y[~train_idx]
        
        model.fit(X_train, y_train)
        
        # Calculate errors
        train_error = np.mean((model.predict(X_train) - y_train)**2)
        test_error = np.mean((model.predict(X_test) - y_test)**2)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(degrees, train_errors, 'bo-', label='Training Error')
    plt.plot(degrees, test_errors, 'ro-', label='Test Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(degrees, np.array(test_errors) - np.array(train_errors), 'go-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Generalization Gap')
    plt.title('Overfitting Measure')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return degrees, train_errors, test_errors


def demonstrate_bias_variance_components():
    """Demonstrate the individual components of bias and variance"""
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    true_function = 0.5 * X**2 + 0.3 * X + 1.0
    noise_std = 0.1
    
    # Test different polynomial degrees
    degrees = [1, 2, 3, 5, 10]
    bias_squared = []
    variance = []
    irreducible_error = noise_std**2
    
    # Generate multiple datasets to estimate bias and variance
    n_datasets = 50
    X_test = np.array([0.5]).reshape(-1, 1)  # Test point
    
    for degree in degrees:
        predictions = []
        
        for _ in range(n_datasets):
            # Generate noisy data
            noise = noise_std * np.random.randn(100, 1)
            y = true_function + noise
            
            # Create and train model
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
            
            # Train on subset of data
            train_idx = np.random.choice(100, 50, replace=False)
            X_train, y_train = X[train_idx], y[train_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate bias and variance
        avg_prediction = np.mean(predictions)
        true_value = 0.5 * X_test[0, 0]**2 + 0.3 * X_test[0, 0] + 1.0
        
        bias_sq = (true_value - avg_prediction)**2
        var = np.var(predictions)
        
        bias_squared.append(bias_sq)
        variance.append(var)
    
    # Plot bias-variance decomposition
    plt.figure(figsize=(15, 5))
    
    # Individual components
    plt.subplot(1, 3, 1)
    plt.plot(degrees, bias_squared, 'bo-', label='Bias²', linewidth=2, markersize=8)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Bias²')
    plt.title('Bias Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(degrees, variance, 'ro-', label='Variance', linewidth=2, markersize=8)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Variance')
    plt.title('Variance Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Total error decomposition
    plt.subplot(1, 3, 3)
    total_error = np.array(bias_squared) + np.array(variance) + irreducible_error
    plt.plot(degrees, bias_squared, 'bo-', label='Bias²', linewidth=2, markersize=8)
    plt.plot(degrees, variance, 'ro-', label='Variance', linewidth=2, markersize=8)
    plt.plot(degrees, [irreducible_error] * len(degrees), 'go-', label='Irreducible Error', linewidth=2, markersize=8)
    plt.plot(degrees, total_error, 'ko-', label='Total Error', linewidth=3, markersize=10)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error Components')
    plt.title('Bias-Variance Decomposition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Bias-Variance Tradeoff Results:")
    print("=" * 50)
    for i, degree in enumerate(degrees):
        print(f"Degree {degree}:")
        print(f"  Bias²: {bias_squared[i]:.6f}")
        print(f"  Variance: {variance[i]:.6f}")
        print(f"  Irreducible Error: {irreducible_error:.6f}")
        print(f"  Total Error: {total_error[i]:.6f}")
        print()
    
    return degrees, bias_squared, variance, total_error


def demonstrate_overfitting_vs_underfitting():
    """Demonstrate overfitting and underfitting with concrete examples"""
    
    # Generate data
    np.random.seed(42)
    X = np.linspace(-2, 2, 100).reshape(-1, 1)
    true_function = 0.5 * X**2 + 0.3 * X + 1.0
    noise = 0.1 * np.random.randn(100, 1)
    y = true_function + noise
    
    # Create models with different complexity
    models = {
        'Linear (Underfitting)': Pipeline([
            ('poly', PolynomialFeatures(degree=1)),
            ('linear', LinearRegression())
        ]),
        'Quadratic (Good Fit)': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ]),
        '10th Degree (Overfitting)': Pipeline([
            ('poly', PolynomialFeatures(degree=10)),
            ('linear', LinearRegression())
        ])
    }
    
    # Train and evaluate models
    results = {}
    train_idx = np.random.choice(100, 50, replace=False)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[~train_idx], y[~train_idx]
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items()):
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate errors
        train_error = np.mean((y_train_pred - y_train)**2)
        test_error = np.mean((y_test_pred - y_test)**2)
        
        results[name] = {
            'train_error': train_error,
            'test_error': test_error,
            'predictions': model.predict(X)
        }
        
        # Plot
        plt.subplot(1, 3, i+1)
        plt.scatter(X_train, y_train, alpha=0.6, label='Training Data', s=20)
        plt.scatter(X_test, y_test, alpha=0.6, label='Test Data', s=20, marker='s')
        plt.plot(X, true_function, 'k-', label='True Function', linewidth=2)
        plt.plot(X, results[name]['predictions'], 'r-', label='Model Prediction', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'{name}\nTrain Error: {train_error:.4f}\nTest Error: {test_error:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Overfitting vs Underfitting Analysis:")
    print("=" * 50)
    for name, result in results.items():
        gap = result['test_error'] - result['train_error']
        print(f"{name}:")
        print(f"  Training Error: {result['train_error']:.4f}")
        print(f"  Test Error: {result['test_error']:.4f}")
        print(f"  Generalization Gap: {gap:.4f}")
        if gap > 0.01:
            print(f"  Diagnosis: {'Overfitting' if result['train_error'] < 0.01 else 'Underfitting'}")
        else:
            print(f"  Diagnosis: Good Generalization")
        print()
    
    return results


if __name__ == "__main__":
    # Run all demonstrations
    print("Running Bias-Variance Tradeoff Demonstrations...")
    print("=" * 60)
    
    print("\n1. Basic Bias-Variance Tradeoff:")
    degrees, train_errors, test_errors = simulate_bias_variance_tradeoff()
    
    print("\n2. Bias-Variance Decomposition:")
    degrees, bias_squared, variance, total_error = demonstrate_bias_variance_components()
    
    print("\n3. Overfitting vs Underfitting Examples:")
    results = demonstrate_overfitting_vs_underfitting()
    
    print("All demonstrations completed!")

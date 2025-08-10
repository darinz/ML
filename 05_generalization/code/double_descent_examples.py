"""
Double Descent Phenomenon Examples
==================================

This module demonstrates the double descent phenomenon, which challenges the classical
bias-variance tradeoff by showing that test error can decrease again after the
classical U-shaped curve in the overparameterized regime.

The double descent phenomenon occurs in two forms:
1. Model-wise double descent: Varying model complexity (number of parameters)
2. Sample-wise double descent: Varying number of training examples

Key Concepts:
- Classical regime: Follows traditional bias-variance tradeoff
- Interpolation threshold: Where n ≈ d (number of samples ≈ number of parameters)
- Overparameterized regime: Where d > n, models can fit training data perfectly
- Implicit regularization: Modern optimizers find "simple" solutions even in overparameterized regime

This corresponds to the concepts discussed in Section 8.2 of the markdown file.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from typing import List, Tuple, Optional, Callable

# Set random seed for reproducibility
np.random.seed(42)

def generate_cubic_data(n_train: int, n_test: int, noise_std: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data from a cubic function with noise.
    
    This creates a more complex true function than the quadratic example
    to better demonstrate the double descent phenomenon.
    
    Args:
        n_train: Number of training points
        n_test: Number of test points
        noise_std: Standard deviation of noise
        
    Returns:
        x_train, y_train, x_test, y_test
    """
    # True cubic function
    def f(x): 
        return 1.5 * x**3 - 0.5 * x**2 + 0.2 * x + 1
    
    # Generate training data
    x_train = np.random.uniform(-1, 1, n_train)
    y_train = f(x_train) + np.random.normal(0, noise_std, n_train)
    
    # Generate test data
    x_test = np.linspace(-1, 1, n_test)
    y_test = f(x_test)
    
    return x_train, y_train, x_test, y_test

def simulate_modelwise_double_descent(
    n_train: int = 100,
    n_test: int = 1000,
    max_degree: int = 30,
    noise_std: float = 0.5,
    random_seed: int = 42
) -> Tuple[List[int], List[float]]:
    """
    Simulate model-wise double descent by varying polynomial degree.
    
    This demonstrates how test error changes as model complexity increases:
    1. First descent: Error decreases as model becomes more expressive
    2. First ascent: Error increases due to overfitting (classical regime)
    3. Second descent: Error decreases again in overparameterized regime
    
    Args:
        n_train: Number of training points
        n_test: Number of test points
        max_degree: Maximum polynomial degree to test
        noise_std: Standard deviation of noise
        random_seed: Random seed for reproducibility
        
    Returns:
        degrees, test_errors
    """
    np.random.seed(random_seed)
    
    # Generate data
    x_train, y_train, x_test, y_test = generate_cubic_data(n_train, n_test, noise_std)
    
    # Test different polynomial degrees
    degrees = list(range(1, max_degree + 1))
    test_errors = []
    
    print(f"Testing polynomial degrees 1 to {max_degree}")
    print(f"Training set size: {n_train}, Test set size: {n_test}")
    print(f"Interpolation threshold: degree ≈ {n_train} (when parameters ≈ samples)")
    
    for i, degree in enumerate(degrees):
        # Fit polynomial model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x_train.reshape(-1, 1), y_train)
        
        # Evaluate on test set
        y_pred = model.predict(x_test.reshape(-1, 1))
        test_error = np.mean((y_pred - y_test) ** 2)
        test_errors.append(test_error)
        
        # Print progress for key points
        if degree in [1, 2, 5, 10, 15, 20, 25, 30]:
            print(f"  Degree {degree}: Test MSE = {test_error:.4f}")
    
    return degrees, test_errors

def plot_modelwise_double_descent(degrees: List[int], test_errors: List[float]) -> None:
    """
    Plot the model-wise double descent curve.
    
    This visualization shows the three regimes:
    1. Underparameterized: Error decreases with complexity
    2. Interpolation threshold: Error peaks around n ≈ d
    3. Overparameterized: Error decreases again
    
    Args:
        degrees: List of polynomial degrees
        test_errors: Corresponding test errors
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the double descent curve
    plt.plot(degrees, test_errors, 'b-o', linewidth=2, markersize=4, label='Test Error')
    
    # Add vertical line at interpolation threshold
    n_train = 100  # This should match the training set size used
    plt.axvline(x=n_train, color='red', linestyle='--', alpha=0.7, 
                label=f'Interpolation Threshold (n ≈ d = {n_train})')
    
    # Add annotations for the three regimes
    plt.annotate('Underparameterized\n(Classical Regime)', 
                xy=(5, test_errors[4]), xytext=(10, max(test_errors) * 0.8),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.annotate('Interpolation\nThreshold', 
                xy=(n_train, test_errors[n_train-1]), xytext=(n_train+5, max(test_errors) * 0.9),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.annotate('Overparameterized\n(Modern Regime)', 
                xy=(25, test_errors[24]), xytext=(20, max(test_errors) * 0.6),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    plt.ylabel('Test Error (MSE)', fontsize=12)
    plt.title('Model-wise Double Descent: Test Error vs. Model Complexity', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def simulate_samplewise_double_descent(
    d: int = 50,
    n_min: int = 10,
    n_max: int = 100,
    step: int = 2,
    noise_std: float = 0.5,
    reg_strength: Optional[float] = None,
    random_seed: int = 42
) -> Tuple[List[int], List[float]]:
    """
    Simulate sample-wise double descent by varying the number of training samples.
    
    This demonstrates how test error changes as the number of training examples increases:
    1. First descent: Error decreases with more data
    2. Peak: Error increases around n ≈ d (interpolation threshold)
    3. Second descent: Error decreases again with even more data
    
    Args:
        d: Number of features (model parameters)
        n_min: Minimum number of training samples
        n_max: Maximum number of training samples
        step: Step size for sample sizes
        noise_std: Standard deviation of noise
        reg_strength: Regularization strength (None for no regularization)
        random_seed: Random seed for reproducibility
        
    Returns:
        sample_sizes, test_errors
    """
    np.random.seed(random_seed)
    
    # Generate true parameter vector
    beta = np.random.randn(d)
    beta /= np.linalg.norm(beta)  # Normalize for consistent scale
    
    # Generate test data
    n_test = 1000
    X_test = np.random.randn(n_test, d)
    y_test = X_test @ beta + np.random.normal(0, noise_std, n_test)
    
    # Test different sample sizes
    sample_sizes = list(range(n_min, n_max + 1, step))
    test_errors = []
    
    print(f"Testing sample sizes {n_min} to {n_max} (step {step})")
    print(f"Number of features: {d}")
    print(f"Interpolation threshold: n ≈ {d} (when samples ≈ parameters)")
    
    for i, n in enumerate(sample_sizes):
        # Generate training data
        X_train = np.random.randn(n, d)
        y_train = X_train @ beta + np.random.normal(0, noise_std, n)
        
        # Fit model with or without regularization
        if reg_strength is None:
            model = LinearRegression()
        else:
            model = Ridge(alpha=reg_strength)
        
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        test_error = np.mean((y_pred - y_test) ** 2)
        test_errors.append(test_error)
        
        # Print progress for key points
        if n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            reg_label = f" (λ={reg_strength})" if reg_strength is not None else " (no reg)"
            print(f"  n={n}: Test MSE = {test_error:.4f}{reg_label}")
    
    return sample_sizes, test_errors

def plot_samplewise_double_descent(sample_sizes: List[int], test_errors: List[float], d: int = 50) -> None:
    """
    Plot the sample-wise double descent curve.
    
    Args:
        sample_sizes: List of training sample sizes
        test_errors: Corresponding test errors
        d: Number of features (for interpolation threshold line)
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the double descent curve
    plt.plot(sample_sizes, test_errors, 'b-o', linewidth=2, markersize=4, label='Test Error')
    
    # Add vertical line at interpolation threshold
    plt.axvline(x=d, color='red', linestyle='--', alpha=0.7, 
                label=f'Interpolation Threshold (n ≈ d = {d})')
    
    # Add annotations
    plt.annotate('Underparameterized\n(n < d)', 
                xy=(20, test_errors[5]), xytext=(15, max(test_errors) * 0.8),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.annotate('Interpolation\nThreshold (n ≈ d)', 
                xy=(d, test_errors[sample_sizes.index(d) if d in sample_sizes else len(sample_sizes)//2]), 
                xytext=(d+10, max(test_errors) * 0.9),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.annotate('Overparameterized\n(n > d)', 
                xy=(80, test_errors[-10]), xytext=(70, max(test_errors) * 0.6),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7),
                fontsize=10, ha='center')
    
    plt.xlabel('Number of Training Samples (n)', fontsize=12)
    plt.ylabel('Test Error (MSE)', fontsize=12)
    plt.title('Sample-wise Double Descent: Test Error vs. Training Set Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def demonstrate_regularization_effect() -> None:
    """
    Demonstrate how regularization affects the double descent phenomenon.
    
    This shows that proper regularization can:
    1. Reduce the peak at the interpolation threshold
    2. Smooth out the double descent curve
    3. Improve overall generalization
    """
    print("=" * 60)
    print("DEMONSTRATION: EFFECT OF REGULARIZATION ON DOUBLE DESCENT")
    print("=" * 60)
    
    d = 50
    n_min, n_max = 10, 100
    reg_strengths = [None, 1e-4, 1e-2, 1e-1, 1, 10]
    
    plt.figure(figsize=(12, 8))
    
    for reg in reg_strengths:
        print(f"\nTesting regularization strength: {reg}")
        
        sample_sizes, test_errors = simulate_samplewise_double_descent(
            d=d, n_min=n_min, n_max=n_max, reg_strength=reg
        )
        
        label = f'No Regularization' if reg is None else f'λ = {reg}'
        plt.plot(sample_sizes, test_errors, marker='.', label=label, linewidth=2)
    
    # Add vertical line at interpolation threshold
    plt.axvline(x=d, color='red', linestyle='--', alpha=0.7, 
                label=f'Interpolation Threshold (n = d = {d})')
    
    plt.xlabel('Number of Training Samples (n)', fontsize=12)
    plt.ylabel('Test Error (MSE)', fontsize=12)
    plt.title('Effect of Regularization on Sample-wise Double Descent', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nKey observations:")
    print("• No regularization shows the strongest double descent effect")
    print("• Stronger regularization reduces the peak at n ≈ d")
    print("• Optimal regularization can eliminate the peak entirely")
    print("• Too much regularization can hurt performance in all regimes")

def demonstrate_implicit_regularization() -> None:
    """
    Demonstrate implicit regularization in modern optimizers.
    
    This shows how different optimization algorithms can lead to different
    generalization behavior even when they achieve the same training error.
    """
    print("=" * 60)
    print("DEMONSTRATION: IMPLICIT REGULARIZATION")
    print("=" * 60)
    
    # Generate data
    d = 30
    n = 20  # Underparameterized case
    np.random.seed(42)
    
    X = np.random.randn(n, d)
    beta_true = np.random.randn(d)
    beta_true /= np.linalg.norm(beta_true)
    y = X @ beta_true + np.random.normal(0, 0.1, n)
    
    # Test different optimization approaches
    methods = {
        'LinearRegression (sklearn)': LinearRegression(),
        'Ridge (λ=0.01)': Ridge(alpha=0.01),
        'Ridge (λ=0.1)': Ridge(alpha=0.1),
        'Ridge (λ=1.0)': Ridge(alpha=1.0)
    }
    
    print(f"Training set: {n} samples, {d} features")
    print(f"Interpolation threshold: n ≈ {d} (we are underparameterized)")
    print()
    
    results = {}
    for name, model in methods.items():
        model.fit(X, y)
        
        # Compute training error
        y_train_pred = model.predict(X)
        train_error = np.mean((y_train_pred - y) ** 2)
        
        # Compute parameter norm (measure of complexity)
        if hasattr(model, 'coef_'):
            param_norm = np.linalg.norm(model.coef_)
        else:
            param_norm = np.linalg.norm(model.coef_)
        
        results[name] = {
            'train_error': train_error,
            'param_norm': param_norm
        }
        
        print(f"{name}:")
        print(f"  Training MSE: {train_error:.6f}")
        print(f"  Parameter norm: {param_norm:.4f}")
        print()
    
    # Plot parameter norms
    names = list(results.keys())
    param_norms = [results[name]['param_norm'] for name in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, param_norms, color=['blue', 'green', 'orange', 'red'])
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Parameter Norm (||β||₂)', fontsize=12)
    plt.title('Implicit Regularization: Parameter Norms of Different Methods', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("Key insights:")
    print("• All methods achieve similar training error")
    print("• Ridge regression finds solutions with smaller parameter norms")
    print("• This implicit regularization can improve generalization")
    print("• The 'simplest' solution (smallest norm) often generalizes best")

def main():
    """
    Main function to run all double descent demonstrations.
    """
    print("DOUBLE DESCENT PHENOMENON EXAMPLES")
    print("=" * 50)
    print("This demonstrates the double descent phenomenon that challenges")
    print("the classical bias-variance tradeoff in machine learning.")
    print()
    
    # Demonstration 1: Model-wise Double Descent
    print("1. MODEL-WISE DOUBLE DESCENT")
    print("-" * 30)
    print("Varying model complexity (polynomial degree) while keeping data fixed.")
    print()
    
    degrees, test_errors = simulate_modelwise_double_descent()
    plot_modelwise_double_descent(degrees, test_errors)
    
    # Demonstration 2: Sample-wise Double Descent
    print("\n2. SAMPLE-WISE DOUBLE DESCENT")
    print("-" * 30)
    print("Varying number of training samples while keeping model complexity fixed.")
    print()
    
    sample_sizes, test_errors = simulate_samplewise_double_descent()
    plot_samplewise_double_descent(sample_sizes, test_errors)
    
    # Demonstration 3: Effect of Regularization
    print("\n3. EFFECT OF REGULARIZATION")
    print("-" * 30)
    print("How regularization affects the double descent phenomenon.")
    print()
    
    demonstrate_regularization_effect()
    
    # Demonstration 4: Implicit Regularization
    print("\n4. IMPLICIT REGULARIZATION")
    print("-" * 30)
    print("How different optimization methods provide implicit regularization.")
    print()
    
    demonstrate_implicit_regularization()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: KEY INSIGHTS FROM DOUBLE DESCENT")
    print("=" * 60)
    print("1. The classical U-shaped bias-variance curve is incomplete")
    print("2. Test error can decrease again in the overparameterized regime")
    print("3. The peak occurs around the interpolation threshold (n ≈ d)")
    print("4. Regularization can mitigate the peak and improve generalization")
    print("5. Modern optimizers provide implicit regularization")
    print("6. Overparameterized models can generalize well despite fitting training data perfectly")
    print("7. The relationship between complexity and generalization is more nuanced than previously thought")

if __name__ == "__main__":
    main() 
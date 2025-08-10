"""
Bias-Variance Decomposition Examples
===================================

This module demonstrates the bias-variance tradeoff, a fundamental concept in machine learning
that explains the decomposition of prediction error into three components:
1. Bias: Systematic error due to model assumptions
2. Variance: Error due to sensitivity to training data
3. Irreducible error: Error due to noise in the data

The bias-variance decomposition is formalized as:
    MSE = Bias² + Variance + Irreducible Error

This corresponds to equation (8.7) in the markdown file:
    MSE(x) = σ² + (h*(x) - h_avg(x))² + var(h_S(x))

Key Concepts:
- High bias: Model is too simple to capture the true relationship (underfitting)
- High variance: Model is too complex and fits noise in training data (overfitting)
- Optimal complexity: Balances bias and variance for best generalization

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from typing import List, Tuple, Callable

# Set random seed for reproducibility
np.random.seed(42)

def h_star(x: np.ndarray) -> np.ndarray:
    """
    True underlying function (quadratic).
    
    This represents the "ground truth" that we're trying to learn.
    In practice, we never know this function - it's what we're trying to approximate.
    
    Args:
        x: Input values
        
    Returns:
        True function values: 2x² + 0.5
    """
    return 2 * x**2 + 0.5

def generate_data(n_train: int, n_test: int, sigma: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training and test data with noise.
    
    This simulates the data generation process described in the markdown file:
    y = h*(x) + ξ where ξ ~ N(0, σ²)
    
    Args:
        n_train: Number of training points
        n_test: Number of test points  
        sigma: Standard deviation of noise
        
    Returns:
        x_train, y_train, x_test, y_test_true
    """
    # Generate training data with noise
    x_train = np.random.rand(n_train)
    y_train = h_star(x_train) + np.random.normal(0, sigma, n_train)
    
    # Generate test data (no noise for true function evaluation)
    x_test = np.linspace(0, 1, n_test)
    y_test_true = h_star(x_test)
    
    return x_train, y_train, x_test, y_test_true

def fit_polynomial_model(x_train: np.ndarray, y_train: np.ndarray, degree: int) -> Callable:
    """
    Fit a polynomial model of given degree.
    
    This demonstrates different model complexities:
    - degree=1: Linear model (high bias, low variance)
    - degree=2: Quadratic model (optimal for our true function)
    - degree=5: High-degree polynomial (low bias, high variance)
    
    Args:
        x_train: Training inputs
        y_train: Training outputs
        degree: Polynomial degree
        
    Returns:
        Fitted model function
    """
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(x_train.reshape(-1, 1), y_train)
    return model

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error.
    
    This corresponds to equation (8.2) in the markdown file:
    MSE = E[(y - h(x))²]
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)

def estimate_bias_variance_decomposition(
    model_factory: Callable,
    x0: float = 0.5,
    n_repeats: int = 500,
    n_train: int = 8,
    sigma: float = 0.2
) -> Tuple[float, float, float, float]:
    """
    Estimate bias, variance, and noise components at a specific point.
    
    This implements the bias-variance decomposition from equations (8.3)-(8.7):
    1. MSE(x) = E[(y - h_S(x))²]
    2. = σ² + E[(h*(x) - h_S(x))²]  (separate noise)
    3. = σ² + (h*(x) - h_avg(x))² + var(h_S(x))  (bias-variance decomposition)
    
    Args:
        model_factory: Function that creates and fits a model
        x0: Point at which to evaluate the decomposition
        n_repeats: Number of training sets to generate
        n_train: Number of training points per set
        sigma: Noise standard deviation
        
    Returns:
        bias_squared, variance, noise, total_mse
    """
    predictions = []
    
    # Generate predictions from many different training sets
    for _ in range(n_repeats):
        # Generate new training data
        x_train = np.random.rand(n_train)
        y_train = h_star(x_train) + np.random.normal(0, sigma, n_train)
        
        # Fit model and make prediction
        model = model_factory(x_train, y_train)
        pred = model.predict(np.array([[x0]]))[0]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Compute components
    true_value = h_star(x0)
    average_prediction = np.mean(predictions)
    
    bias_squared = (true_value - average_prediction) ** 2
    variance = np.var(predictions)
    noise = sigma ** 2
    total_mse = bias_squared + variance + noise
    
    return bias_squared, variance, noise, total_mse

def plot_bias_variance_tradeoff(
    degrees: List[int],
    x0: float = 0.5,
    n_repeats: int = 200,
    n_train: int = 8,
    sigma: float = 0.2
) -> None:
    """
    Plot the bias-variance tradeoff as a function of model complexity.
    
    This demonstrates the classic U-shaped curve where:
    - Simple models have high bias, low variance
    - Complex models have low bias, high variance
    - Optimal complexity balances both
    
    Args:
        degrees: List of polynomial degrees to test
        x0: Point at which to evaluate
        n_repeats: Number of training sets per degree
        n_train: Number of training points per set
        sigma: Noise standard deviation
    """
    bias_squared_list = []
    variance_list = []
    mse_list = []
    
    print(f"Computing bias-variance decomposition for {len(degrees)} model complexities...")
    
    for i, degree in enumerate(degrees):
        print(f"  Processing degree {degree} ({i+1}/{len(degrees)})")
        
        def model_factory(x_train, y_train):
            return fit_polynomial_model(x_train, y_train, degree)
        
        bias_squared, variance, noise, total_mse = estimate_bias_variance_decomposition(
            model_factory, x0, n_repeats, n_train, sigma
        )
        
        bias_squared_list.append(bias_squared)
        variance_list.append(variance)
        mse_list.append(total_mse)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, bias_squared_list, 'b-o', label='Bias²', linewidth=2, markersize=6)
    plt.plot(degrees, variance_list, 'r-s', label='Variance', linewidth=2, markersize=6)
    plt.plot(degrees, mse_list, 'g-^', label='Total MSE (Bias² + Variance + Noise)', linewidth=2, markersize=6)
    
    # Add horizontal line for irreducible error
    plt.axhline(y=sigma**2, color='gray', linestyle='--', alpha=0.7, label=f'Irreducible Error (σ² = {sigma**2:.3f})')
    
    plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=12)
    plt.ylabel(f'Error at x = {x0}', fontsize=12)
    plt.title('Bias-Variance Tradeoff: Error Components vs. Model Complexity', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print optimal degree
    optimal_degree = degrees[np.argmin(mse_list)]
    print(f"\nOptimal polynomial degree: {optimal_degree}")
    print(f"Minimum total MSE: {min(mse_list):.4f}")

def demonstrate_underfitting_vs_overfitting() -> None:
    """
    Demonstrate underfitting and overfitting with concrete examples.
    
    This shows the three cases discussed in the markdown file:
    1. Linear model (underfitting): high bias, low variance
    2. Quadratic model (optimal): balanced bias and variance  
    3. 5th-degree polynomial (overfitting): low bias, high variance
    """
    print("=" * 60)
    print("DEMONSTRATION: UNDERFITTING vs OVERFITTING")
    print("=" * 60)
    
    # Generate data
    n_train, n_test = 8, 1000
    sigma = 0.2
    x_train, y_train, x_test, y_test_true = generate_data(n_train, n_test, sigma)
    
    # Fit models of different complexities
    models = {
        'Linear (Underfitting)': fit_polynomial_model(x_train, y_train, 1),
        'Quadratic (Optimal)': fit_polynomial_model(x_train, y_train, 2),
        '5th-degree (Overfitting)': fit_polynomial_model(x_train, y_train, 5)
    }
    
    # Evaluate each model
    print("\nModel Performance Comparison:")
    print("-" * 50)
    
    for name, model in models.items():
        y_pred = model.predict(x_test.reshape(-1, 1))
        mse = compute_mse(y_test_true, y_pred)
        train_mse = compute_mse(y_train, model.predict(x_train.reshape(-1, 1)))
        
        print(f"{name}:")
        print(f"  Training MSE: {train_mse:.4f}")
        print(f"  Test MSE:     {mse:.4f}")
        print(f"  Generalization gap: {abs(train_mse - mse):.4f}")
        print()
    
    # Plot the results
    plt.figure(figsize=(15, 5))
    
    for i, (name, model) in enumerate(models.items()):
        plt.subplot(1, 3, i+1)
        
        # Plot training data
        plt.scatter(x_train, y_train, color='red', s=50, alpha=0.7, label='Training Data')
        
        # Plot true function
        plt.plot(x_test, y_test_true, 'k-', linewidth=2, label='True Function h*(x)')
        
        # Plot model predictions
        y_pred = model.predict(x_test.reshape(-1, 1))
        plt.plot(x_test, y_pred, 'b--', linewidth=2, label='Model Prediction')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(name)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_bias_variance_decomposition() -> None:
    """
    Demonstrate the bias-variance decomposition at a single point.
    
    This shows how we can decompose the total error into its components
    by training many models on different datasets and analyzing the
    distribution of predictions.
    """
    print("=" * 60)
    print("DEMONSTRATION: BIAS-VARIANCE DECOMPOSITION")
    print("=" * 60)
    
    x0 = 0.5  # Point at which to evaluate
    n_repeats = 500
    n_train = 8
    sigma = 0.2
    
    print(f"\nEvaluating bias-variance decomposition at x = {x0}")
    print(f"Using {n_repeats} different training sets with {n_train} points each")
    print(f"Noise standard deviation: σ = {sigma}")
    
    # Test different model complexities
    complexities = [1, 2, 5]
    
    for degree in complexities:
        print(f"\n--- Polynomial Degree {degree} ---")
        
        def model_factory(x_train, y_train):
            return fit_polynomial_model(x_train, y_train, degree)
        
        bias_squared, variance, noise, total_mse = estimate_bias_variance_decomposition(
            model_factory, x0, n_repeats, n_train, sigma
        )
        
        print(f"Bias²:     {bias_squared:.4f}")
        print(f"Variance:  {variance:.4f}")
        print(f"Noise:     {noise:.4f}")
        print(f"Total MSE: {total_mse:.4f}")
        print(f"Sum check: {bias_squared + variance + noise:.4f}")
        
        # Interpretation
        if degree == 1:
            print("  → High bias (underfitting): model too simple")
        elif degree == 2:
            print("  → Balanced: optimal complexity for this problem")
        else:
            print("  → High variance (overfitting): model too complex")

def main():
    """
    Main function to run all demonstrations.
    """
    print("BIAS-VARIANCE DECOMPOSITION EXAMPLES")
    print("=" * 50)
    print("This demonstrates the fundamental bias-variance tradeoff in machine learning.")
    print("The total prediction error can be decomposed into three components:")
    print("1. Bias²: Systematic error due to model assumptions")
    print("2. Variance: Error due to sensitivity to training data")
    print("3. Irreducible Error: Error due to noise in the data")
    print()
    
    # Demonstration 1: Underfitting vs Overfitting
    demonstrate_underfitting_vs_overfitting()
    
    # Demonstration 2: Bias-Variance Decomposition
    demonstrate_bias_variance_decomposition()
    
    # Demonstration 3: Bias-Variance Tradeoff Plot
    print("\n" + "=" * 60)
    print("PLOTTING BIAS-VARIANCE TRADEOFF")
    print("=" * 60)
    print("This will take a moment to compute...")
    
    degrees = list(range(1, 11))  # Test degrees 1-10
    plot_bias_variance_tradeoff(degrees)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Key insights from the bias-variance tradeoff:")
    print("• Simple models (low degree) have high bias but low variance")
    print("• Complex models (high degree) have low bias but high variance")
    print("• The optimal model complexity balances both components")
    print("• The irreducible error (noise) cannot be reduced by any model")
    print("• Understanding this tradeoff helps in model selection and regularization")

if __name__ == "__main__":
    main() 
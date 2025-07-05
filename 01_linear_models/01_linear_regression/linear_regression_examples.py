"""
Linear Regression Code Examples with Inline Annotations

This file contains all the code examples from the linear regression notes,
with detailed inline annotations explaining each concept and implementation.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Example 1: Basic Data Visualization
# ============================================================================

def plot_housing_data():
    """
    Example: Visualizing house price data to understand the relationship
    between living area and price.
    """
    print("=== Example 1: Housing Data Visualization ===")
    
    # Example data (living area in ft^2, price in $1000s)
    # This represents the training set from the Portland housing example
    living_area = np.array([2104, 1600, 2400, 1416, 3000])
    price = np.array([400, 330, 369, 232, 540])
    
    # Create scatter plot to visualize the relationship
    plt.figure(figsize=(8, 6))
    plt.scatter(living_area, price, color='blue', alpha=0.7, s=100)
    plt.xlabel('Living area (ft²)')
    plt.ylabel('Price (1000$s)')
    plt.title('House Prices vs. Living Area')
    plt.grid(True, alpha=0.3)
    
    # Add some example predictions to show the concept
    # This line represents a simple linear relationship (not optimized)
    x_line = np.linspace(1400, 3100, 100)
    y_line = 0.1 * x_line + 100  # Simple linear approximation
    plt.plot(x_line, y_line, 'r--', alpha=0.7, label='Simple linear fit')
    
    plt.legend()
    plt.show()
    
    print(f"Data points: {len(living_area)} houses")
    print(f"Living areas range: {living_area.min()} to {living_area.max()} ft²")
    print(f"Prices range: ${price.min()}k to ${price.max()}k")
    print()

# ============================================================================
# Example 2: Hypothesis Function Implementation
# ============================================================================

def hypothesis_function_example():
    """
    Example: Implementing the hypothesis function h_θ(x) = θ^T x
    This function represents our model's prediction for a given input.
    """
    print("=== Example 2: Hypothesis Function ===")
    
    def h_theta(x, theta):
        """
        Hypothesis function: h_θ(x) = θ^T x
        
        Parameters:
        x: numpy array of shape (n_features,) - input features including x0=1
        theta: numpy array of shape (n_features,) - model parameters
        
        Returns:
        float: predicted value
        """
        return np.dot(theta, x)  # θ^T x = θ₀x₀ + θ₁x₁ + θ₂x₂ + ...
    
    # Example usage with two features (living area, bedrooms)
    # Note: x[0] = 1 is the intercept term (x₀)
    x = np.array([1, 2104, 3])  # [x₀=1, living_area, bedrooms]
    theta = np.array([50, 0.1, 20])  # [θ₀, θ₁, θ₂] - example parameters
    
    prediction = h_theta(x, theta)
    
    print(f"Input features: x = {x}")
    print(f"  - x₀ (intercept): {x[0]}")
    print(f"  - x₁ (living area): {x[1]} ft²")
    print(f"  - x₂ (bedrooms): {x[2]}")
    print(f"Model parameters: θ = {theta}")
    print(f"  - θ₀ (bias): {theta[0]}")
    print(f"  - θ₁ (living area weight): {theta[1]}")
    print(f"  - θ₂ (bedroom weight): {theta[2]}")
    print(f"Prediction: h_θ(x) = {prediction:.2f} (thousands of dollars)")
    print(f"Predicted price: ${prediction:.0f},000")
    print()

# ============================================================================
# Example 3: Cost Function Implementation
# ============================================================================

def cost_function_examples():
    """
    Example: Implementing the cost function J(θ) = (1/2) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    This measures how well our model fits the training data.
    """
    print("=== Example 3: Cost Function ===")
    
    # Create sample training data
    # X: feature matrix (n_samples, n_features) including intercept term
    # y: target vector (n_samples,)
    X = np.array([
        [1, 2104, 3],  # [x₀=1, living_area, bedrooms]
        [1, 1600, 3],
        [1, 2400, 3],
        [1, 1416, 2],
        [1, 3000, 4]
    ])
    y = np.array([400, 330, 369, 232, 540])  # prices in $1000s
    
    print(f"Training data shape: X = {X.shape}, y = {y.shape}")
    print(f"Number of training examples: {len(y)}")
    print(f"Number of features (including intercept): {X.shape[1]}")
    print()
    
    # Method 1: Non-vectorized cost function (for understanding)
    def compute_cost_non_vectorized(X, y, theta):
        """
        Non-vectorized implementation of the cost function.
        This is easier to understand but less efficient.
        
        J(θ) = (1/2) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
        """
        n = len(y)  # number of training examples
        total_cost = 0.0
        
        for i in range(n):
            # Compute prediction for i-th example
            prediction = np.dot(theta, X[i])
            # Compute squared error for this example
            squared_error = (prediction - y[i]) ** 2
            total_cost += squared_error
        
        return 0.5 * total_cost
    
    # Method 2: Vectorized cost function (more efficient)
    def compute_cost_vectorized(X, y, theta):
        """
        Vectorized implementation of the cost function.
        This is more efficient and uses matrix operations.
        
        J(θ) = (1/2) * ||Xθ - y||²
        """
        # Compute predictions for all examples at once: Xθ
        predictions = X @ theta  # matrix multiplication
        
        # Compute residuals (errors): Xθ - y
        residuals = predictions - y
        
        # Compute squared norm: ||Xθ - y||²
        squared_norm = np.dot(residuals, residuals)
        
        return 0.5 * squared_norm
    
    # Method 3: Mean Squared Error (MSE)
    def mean_squared_error(X, y, theta):
        """
        Mean Squared Error: average of squared errors.
        
        MSE(θ) = (1/n) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
        """
        n = len(y)
        residuals = X @ theta - y
        return np.dot(residuals, residuals) / n
    
    # Test with example parameters
    theta_example = np.array([50, 0.1, 20])
    
    cost_non_vec = compute_cost_non_vectorized(X, y, theta_example)
    cost_vec = compute_cost_vectorized(X, y, theta_example)
    mse = mean_squared_error(X, y, theta_example)
    
    print(f"Example parameters: θ = {theta_example}")
    print(f"Cost (non-vectorized): J(θ) = {cost_non_vec:.2f}")
    print(f"Cost (vectorized): J(θ) = {cost_vec:.2f}")
    print(f"Mean Squared Error: MSE(θ) = {mse:.2f}")
    print(f"Verification: MSE * n/2 = {mse * len(y) / 2:.2f}")
    print()
    
    # Show the relationship between cost and MSE
    print("Relationship between cost functions:")
    print(f"  J(θ) = (1/2) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)² = {cost_vec:.2f}")
    print(f"  MSE(θ) = (1/n) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)² = {mse:.2f}")
    print(f"  J(θ) = (n/2) * MSE(θ) = {len(y)/2 * mse:.2f}")
    print()

# ============================================================================
# Example 4: Cost Function Visualization
# ============================================================================

def visualize_cost_function():
    """
    Example: Visualizing how the cost function changes with different parameters.
    This helps understand the optimization landscape.
    """
    print("=== Example 4: Cost Function Visualization ===")
    
    # Create simple 1D example for visualization
    # Single feature: living area (excluding intercept for simplicity)
    X_simple = np.array([[2104], [1600], [2400], [1416], [3000]])
    y_simple = np.array([400, 330, 369, 232, 540])
    
    # Add intercept term
    X_with_intercept = np.column_stack([np.ones(len(y_simple)), X_simple])
    
    def compute_cost_2d(X, y, theta0, theta1):
        """Compute cost for 2D parameter space (θ₀, θ₁)"""
        theta = np.array([theta0, theta1])
        predictions = X @ theta
        residuals = predictions - y
        return 0.5 * np.dot(residuals, residuals)
    
    # Create parameter grid for visualization
    theta0_range = np.linspace(-100, 200, 50)
    theta1_range = np.linspace(-0.1, 0.3, 50)
    
    # Compute cost for each parameter combination
    cost_grid = np.zeros((len(theta0_range), len(theta1_range)))
    for i, theta0 in enumerate(theta0_range):
        for j, theta1 in enumerate(theta1_range):
            cost_grid[i, j] = compute_cost_2d(X_with_intercept, y_simple, theta0, theta1)
    
    # Create contour plot
    plt.figure(figsize=(10, 8))
    
    # Contour plot of cost function
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_range, theta1_range)
    plt.contour(theta0_mesh, theta1_mesh, cost_grid.T, levels=20, alpha=0.7)
    plt.colorbar(label='Cost J(θ)')
    
    # Find minimum cost point (approximate)
    min_idx = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    min_theta0 = theta0_range[min_idx[0]]
    min_theta1 = theta1_range[min_idx[1]]
    min_cost = cost_grid[min_idx]
    
    plt.plot(min_theta0, min_theta1, 'r*', markersize=15, label=f'Minimum\nθ₀={min_theta0:.1f}\nθ₁={min_theta1:.3f}')
    
    plt.xlabel('θ₀ (intercept/bias)')
    plt.ylabel('θ₁ (living area weight)')
    plt.title('Cost Function J(θ) Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"Minimum cost found at:")
    print(f"  θ₀ (intercept) = {min_theta0:.1f}")
    print(f"  θ₁ (living area weight) = {min_theta1:.3f}")
    print(f"  Minimum cost = {min_cost:.2f}")
    print()
    
    # Show the best fit line
    plt.figure(figsize=(8, 6))
    plt.scatter(X_simple.flatten(), y_simple, color='blue', alpha=0.7, s=100)
    
    # Plot the best fit line
    x_line = np.linspace(1400, 3100, 100)
    y_line = min_theta0 + min_theta1 * x_line
    plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Best fit: y = {min_theta0:.1f} + {min_theta1:.3f}x')
    
    plt.xlabel('Living area (ft²)')
    plt.ylabel('Price (1000$s)')
    plt.title('Best Fit Line from Cost Minimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================================================
# Example 5: Multiple Features Example
# ============================================================================

def multiple_features_example():
    """
    Example: Working with multiple features (living area + bedrooms).
    This demonstrates how linear regression scales to higher dimensions.
    """
    print("=== Example 5: Multiple Features ===")
    
    # Extended dataset with multiple features
    # Features: [intercept, living_area, bedrooms]
    X_multi = np.array([
        [1, 2104, 3],  # [x₀=1, living_area, bedrooms]
        [1, 1600, 3],
        [1, 2400, 3],
        [1, 1416, 2],
        [1, 3000, 4],
        [1, 1985, 4],
        [1, 1534, 3],
        [1, 1427, 3],
        [1, 1380, 3],
        [1, 1494, 3]
    ])
    y_multi = np.array([400, 330, 369, 232, 540, 400, 330, 369, 232, 540])
    
    print(f"Multiple features dataset:")
    print(f"  Shape: X = {X_multi.shape}, y = {y_multi.shape}")
    print(f"  Features: intercept, living area (ft²), bedrooms")
    print(f"  Target: price (1000$s)")
    print()
    
    # Example parameters for multiple features
    theta_multi = np.array([50, 0.1, 20])  # [θ₀, θ₁, θ₂]
    
    def predict_multiple_features(X, theta):
        """Make predictions for multiple features"""
        return X @ theta
    
    predictions = predict_multiple_features(X_multi, theta_multi)
    
    print("Predictions with multiple features:")
    print("House | Living Area | Bedrooms | Actual Price | Predicted Price | Error")
    print("-" * 70)
    for i in range(len(y_multi)):
        actual = y_multi[i]
        predicted = predictions[i]
        error = predicted - actual
        print(f"{i+1:5d} | {X_multi[i,1]:11.0f} | {X_multi[i,2]:8.0f} | {actual:12.0f} | {predicted:14.1f} | {error:+6.1f}")
    
    # Compute cost for multiple features
    cost_multi = 0.5 * np.sum((predictions - y_multi) ** 2)
    mse_multi = np.mean((predictions - y_multi) ** 2)
    
    print(f"\nCost with multiple features: J(θ) = {cost_multi:.2f}")
    print(f"MSE with multiple features: MSE(θ) = {mse_multi:.2f}")
    print()

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("Linear Regression Code Examples with Inline Annotations")
    print("=" * 60)
    print()
    
    # Run all examples
    plot_housing_data()
    hypothesis_function_example()
    cost_function_examples()
    multiple_features_example()
    
    # Ask user if they want to see the cost function visualization
    print("Would you like to see the cost function visualization? (y/n): ", end="")
    try:
        response = input().lower().strip()
        if response in ['y', 'yes']:
            visualize_cost_function()
    except:
        print("Skipping visualization (no input available)")
    
    print("\nAll examples completed!")
    print("\nKey concepts demonstrated:")
    print("1. Data visualization and understanding relationships")
    print("2. Hypothesis function implementation (h_θ(x) = θ^T x)")
    print("3. Cost function implementation (J(θ) = (1/2) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²)")
    print("4. Vectorized vs non-vectorized implementations")
    print("5. Multiple features handling")
    print("6. Cost function visualization and optimization landscape") 
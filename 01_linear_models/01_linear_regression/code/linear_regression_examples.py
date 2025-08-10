"""
Linear Regression Code Examples with Comprehensive Annotations

This file contains comprehensive code examples that implement the concepts from the 
linear regression notes, with detailed inline annotations explaining each concept 
and implementation. The examples demonstrate:

1. Basic linear regression concepts and hypothesis function
2. Cost function implementation (vectorized and non-vectorized)
3. Data visualization and understanding relationships
4. Multiple features handling
5. Cost function visualization and optimization landscape
6. Real-world applications and practical considerations

Key Concepts Demonstrated:
- Hypothesis function: h_θ(x) = θ^T x
- Cost function: J(θ) = (1/2) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
- Vectorized vs non-vectorized implementations
- Design matrix formulation
- Multiple features and dimensionality
- Geometric interpretation of cost function
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Example 1: Basic Data Visualization and Understanding
# ============================================================================

def plot_housing_data():
    """
    Example: Visualizing house price data to understand the relationship
    between living area and price. This demonstrates the fundamental
    concept of supervised learning - finding patterns in data.
    
    Key Learning Points:
    - Data visualization helps understand relationships
    - Linear relationships can be approximated with straight lines
    - Real data often has noise and scatter around the trend
    """
    print("=== Example 1: Housing Data Visualization ===")
    print("Understanding the relationship between features and targets")
    print()
    
    # Example data (living area in ft^2, price in $1000s)
    # This represents the training set from the Portland housing example
    # Each row is a training example: (x⁽ⁱ⁾, y⁽ⁱ⁾)
    living_area = np.array([2104, 1600, 2400, 1416, 3000])
    price = np.array([400, 330, 369, 232, 540])
    
    print(f"Training set: {len(living_area)} houses")
    print(f"Features (x): living area in ft²")
    print(f"Targets (y): price in thousands of dollars")
    print()
    
    # Create scatter plot to visualize the relationship
    plt.figure(figsize=(10, 8))
    plt.scatter(living_area, price, color='blue', alpha=0.7, s=100, label='Training examples')
    plt.xlabel('Living area (ft²)')
    plt.ylabel('Price (1000$s)')
    plt.title('House Prices vs. Living Area\nSupervised Learning Dataset')
    plt.grid(True, alpha=0.3)
    
    # Add some example predictions to show the concept
    # This line represents a simple linear relationship (not optimized)
    x_line = np.linspace(1400, 3100, 100)
    y_line = 0.1 * x_line + 100  # Simple linear approximation: y = θ₀ + θ₁x
    plt.plot(x_line, y_line, 'r--', alpha=0.7, linewidth=2, 
             label='Simple linear fit: y = 100 + 0.1x')
    
    # Add annotations to explain the concept
    plt.annotate('Training example\n(x⁽ⁱ⁾, y⁽ⁱ⁾)', 
                xy=(2104, 400), xytext=(2300, 450),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.7))
    
    plt.legend()
    plt.show()
    
    # Statistical summary
    print("Data Summary:")
    print(f"  Number of training examples: n = {len(living_area)}")
    print(f"  Living areas range: {living_area.min()} to {living_area.max()} ft²")
    print(f"  Prices range: ${price.min()}k to ${price.max()}k")
    print(f"  Average living area: {living_area.mean():.0f} ft²")
    print(f"  Average price: ${price.mean():.0f}k")
    print()
    
    # Show the relationship
    correlation = np.corrcoef(living_area, price)[0, 1]
    print(f"Correlation between living area and price: {correlation:.3f}")
    print("  (Positive correlation suggests larger houses cost more)")
    print()

# ============================================================================
# Example 2: Hypothesis Function Implementation
# ============================================================================

def hypothesis_function_example():
    """
    Example: Implementing the hypothesis function h_θ(x) = θ^T x
    This function represents our model's prediction for a given input.
    
    Key Learning Points:
    - Hypothesis function is the core of our model
    - Vectorized implementation is efficient
    - Intercept term (x₀ = 1) allows the line to not pass through origin
    - Parameters θ determine the relationship strength
    
    Mathematical Formulation:
    h_θ(x) = θ₀x₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ = θ^T x
    where x₀ = 1 (intercept term)
    """
    print("=== Example 2: Hypothesis Function Implementation ===")
    print("Understanding h_θ(x) = θ^T x")
    print()
    
    def h_theta(x, theta):
        """
        Hypothesis function: h_θ(x) = θ^T x
        
        This is the core prediction function of linear regression.
        It computes a linear combination of features weighted by parameters.
        
        Parameters:
        x: numpy array of shape (n_features,) - input features including x₀=1
        theta: numpy array of shape (n_features,) - model parameters
        
        Returns:
        float: predicted value
        
        Mathematical Form:
        h_θ(x) = θ₀x₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
        where x₀ = 1 (intercept term)
        """
        return np.dot(theta, x)  # θ^T x = θ₀x₀ + θ₁x₁ + θ₂x₂ + ...
    
    # Example usage with two features (living area, bedrooms)
    # Note: x[0] = 1 is the intercept term (x₀)
    x = np.array([1, 2104, 3])  # [x₀=1, living_area, bedrooms]
    theta = np.array([50, 0.1, 20])  # [θ₀, θ₁, θ₂] - example parameters
    
    prediction = h_theta(x, theta)
    
    print("Single Example Prediction:")
    print(f"Input features: x = {x}")
    print(f"  - x₀ (intercept): {x[0]} (always 1)")
    print(f"  - x₁ (living area): {x[1]} ft²")
    print(f"  - x₂ (bedrooms): {x[2]}")
    print()
    print(f"Model parameters: θ = {theta}")
    print(f"  - θ₀ (bias/intercept): {theta[0]} (base price)")
    print(f"  - θ₁ (living area weight): {theta[1]} (price per ft²)")
    print(f"  - θ₂ (bedroom weight): {theta[2]} (price per bedroom)")
    print()
    
    # Show the calculation step by step
    print("Prediction Calculation:")
    print(f"h_θ(x) = θ₀x₀ + θ₁x₁ + θ₂x₂")
    print(f"       = {theta[0]} × {x[0]} + {theta[1]} × {x[1]} + {theta[2]} × {x[2]}")
    print(f"       = {theta[0] * x[0]} + {theta[1] * x[1]:.1f} + {theta[2] * x[2]}")
    print(f"       = {theta[0] * x[0] + theta[1] * x[1] + theta[2] * x[2]:.1f}")
    print()
    print(f"Prediction: h_θ(x) = {prediction:.2f} (thousands of dollars)")
    print(f"Predicted price: ${prediction:.0f},000")
    print()
    
    # Demonstrate with multiple examples
    print("Multiple Examples:")
    X_examples = np.array([
        [1, 1600, 3],  # Small house, 3 bedrooms
        [1, 2400, 3],  # Medium house, 3 bedrooms
        [1, 3000, 4]   # Large house, 4 bedrooms
    ])
    
    print("House | Living Area | Bedrooms | Predicted Price")
    print("-" * 50)
    for i, x_example in enumerate(X_examples):
        pred = h_theta(x_example, theta)
        print(f"{i+1:5d} | {x_example[1]:11.0f} | {x_example[2]:8.0f} | ${pred:12.0f}k")
    print()

# ============================================================================
# Example 3: Cost Function Implementation
# ============================================================================

def cost_function_examples():
    """
    Example: Implementing the cost function J(θ) = (1/2) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    This measures how well our model fits the training data.
    
    Key Learning Points:
    - Cost function quantifies prediction error
    - Squared error penalizes large errors more heavily
    - Vectorized implementation is much more efficient
    - Cost function is the objective we want to minimize
    
    Mathematical Formulation:
    J(θ) = (1/2) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    J(θ) = (1/2) * ||Xθ - y||² (vectorized form)
    """
    print("=== Example 3: Cost Function Implementation ===")
    print("Understanding J(θ) = (1/2) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²")
    print()
    
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
    
    print(f"Training data:")
    print(f"  X shape: {X.shape} (n_samples={X.shape[0]}, n_features={X.shape[1]})")
    print(f"  y shape: {y.shape} (n_samples={y.shape[0]})")
    print(f"  Number of training examples: n = {len(y)}")
    print(f"  Number of features (including intercept): d+1 = {X.shape[1]}")
    print()
    
    # Method 1: Non-vectorized cost function (for understanding)
    def compute_cost_non_vectorized(X, y, theta):
        """
        Non-vectorized implementation of the cost function.
        This is easier to understand but less efficient.
        
        J(θ) = (1/2) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
        
        This implementation shows the mathematical formula directly:
        1. For each training example, compute prediction h_θ(x⁽ⁱ⁾)
        2. Compute squared error (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
        3. Sum all squared errors
        4. Multiply by 1/2
        """
        n = len(y)  # number of training examples
        total_cost = 0.0
        
        print("Non-vectorized calculation:")
        for i in range(n):
            # Compute prediction for i-th example: h_θ(x⁽ⁱ⁾) = θ^T x⁽ⁱ⁾
            prediction = np.dot(theta, X[i])
            # Compute squared error for this example: (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
            squared_error = (prediction - y[i]) ** 2
            total_cost += squared_error
            
            print(f"  Example {i+1}: h_θ(x⁽ⁱ⁾) = {prediction:.2f}, y⁽ⁱ⁾ = {y[i]}")
            print(f"    Error = {prediction - y[i]:+.2f}, Squared error = {squared_error:.2f}")
        
        cost = 0.5 * total_cost
        print(f"  Total squared error: {total_cost:.2f}")
        print(f"  Cost J(θ) = (1/2) × {total_cost:.2f} = {cost:.2f}")
        print()
        
        return cost
    
    # Method 2: Vectorized cost function (more efficient)
    def compute_cost_vectorized(X, y, theta):
        """
        Vectorized implementation of the cost function.
        This is more efficient and uses matrix operations.
        
        J(θ) = (1/2) * ||Xθ - y||²
        
        This implementation uses matrix operations:
        1. Xθ computes all predictions at once
        2. Xθ - y computes all residuals at once
        3. ||Xθ - y||² computes squared norm efficiently
        """
        # Compute predictions for all examples at once: Xθ
        predictions = X @ theta  # matrix multiplication
        
        # Compute residuals (errors): Xθ - y
        residuals = predictions - y
        
        # Compute squared norm: ||Xθ - y||²
        squared_norm = np.dot(residuals, residuals)
        
        cost = 0.5 * squared_norm
        
        print("Vectorized calculation:")
        print(f"  Predictions Xθ: {predictions}")
        print(f"  Residuals Xθ - y: {residuals}")
        print(f"  Squared norm ||Xθ - y||²: {squared_norm:.2f}")
        print(f"  Cost J(θ) = (1/2) × {squared_norm:.2f} = {cost:.2f}")
        print()
        
        return cost
    
    # Method 3: Mean Squared Error (MSE)
    def mean_squared_error(X, y, theta):
        """
        Mean Squared Error: average of squared errors.
        
        MSE(θ) = (1/n) * Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
        
        This is related to the cost function:
        J(θ) = (n/2) * MSE(θ)
        """
        n = len(y)
        residuals = X @ theta - y
        mse = np.dot(residuals, residuals) / n
        
        print(f"Mean Squared Error:")
        print(f"  MSE(θ) = (1/n) × Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)² = {mse:.2f}")
        print()
        
        return mse
    
    # Test with example parameters
    theta_example = np.array([50, 0.1, 20])
    
    print(f"Testing with parameters: θ = {theta_example}")
    print("=" * 50)
    
    cost_non_vec = compute_cost_non_vectorized(X, y, theta_example)
    cost_vec = compute_cost_vectorized(X, y, theta_example)
    mse = mean_squared_error(X, y, theta_example)
    
    print("Results Summary:")
    print(f"  Cost (non-vectorized): J(θ) = {cost_non_vec:.2f}")
    print(f"  Cost (vectorized): J(θ) = {cost_vec:.2f}")
    print(f"  Mean Squared Error: MSE(θ) = {mse:.2f}")
    print(f"  Verification: MSE × n/2 = {mse * len(y) / 2:.2f}")
    print(f"  Methods agree: {abs(cost_non_vec - cost_vec) < 1e-10}")
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
    This helps understand the optimization landscape and why we need optimization.
    
    Key Learning Points:
    - Cost function creates a landscape in parameter space
    - We want to find the minimum of this landscape
    - Different parameters give different costs
    - The minimum corresponds to the best fit line
    """
    print("=== Example 4: Cost Function Visualization ===")
    print("Understanding the optimization landscape")
    print()
    
    # Create simple 1D example for visualization
    # Single feature: living area (excluding intercept for simplicity)
    X_simple = np.array([[2104], [1600], [2400], [1416], [3000]])
    y_simple = np.array([400, 330, 369, 232, 540])
    
    # Add intercept term
    X_with_intercept = np.column_stack([np.ones(len(y_simple)), X_simple])
    
    def compute_cost_2d(X, y, theta0, theta1):
        """
        Compute cost for 2D parameter space (θ₀, θ₁)
        This allows us to visualize the cost function as a surface
        """
        theta = np.array([theta0, theta1])
        predictions = X @ theta
        residuals = predictions - y
        return 0.5 * np.dot(residuals, residuals)
    
    # Create parameter grid for visualization
    theta0_range = np.linspace(-100, 200, 50)  # intercept range
    theta1_range = np.linspace(-0.1, 0.3, 50)  # slope range
    
    # Compute cost for each parameter combination
    cost_grid = np.zeros((len(theta0_range), len(theta1_range)))
    for i, theta0 in enumerate(theta0_range):
        for j, theta1 in enumerate(theta1_range):
            cost_grid[i, j] = compute_cost_2d(X_with_intercept, y_simple, theta0, theta1)
    
    # Create contour plot
    plt.figure(figsize=(12, 10))
    
    # Contour plot of cost function
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_range, theta1_range)
    contours = plt.contour(theta0_mesh, theta1_mesh, cost_grid.T, levels=20, alpha=0.7)
    plt.colorbar(label='Cost J(θ)', shrink=0.8)
    
    # Find minimum cost point (approximate)
    min_idx = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
    min_theta0 = theta0_range[min_idx[0]]
    min_theta1 = theta1_range[min_idx[1]]
    min_cost = cost_grid[min_idx]
    
    plt.plot(min_theta0, min_theta1, 'r*', markersize=15, 
             label=f'Global Minimum\nθ₀={min_theta0:.1f}\nθ₁={min_theta1:.3f}\nJ(θ)={min_cost:.1f}')
    
    # Add some example points to show cost values
    example_points = [
        (50, 0.1, 'blue', 'Example 1'),
        (100, 0.05, 'green', 'Example 2'),
        (-50, 0.2, 'purple', 'Example 3')
    ]
    
    for theta0, theta1, color, label in example_points:
        cost = compute_cost_2d(X_with_intercept, y_simple, theta0, theta1)
        plt.plot(theta0, theta1, 'o', color=color, markersize=8, label=f'{label}\nJ(θ)={cost:.1f}')
    
    plt.xlabel('θ₀ (intercept/bias)')
    plt.ylabel('θ₁ (living area weight)')
    plt.title('Cost Function J(θ) Landscape\nContour Plot of Parameter Space')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Cost function landscape analysis:")
    print(f"  Global minimum found at:")
    print(f"    θ₀ (intercept) = {min_theta0:.1f}")
    print(f"    θ₁ (living area weight) = {min_theta1:.3f}")
    print(f"    Minimum cost = {min_cost:.2f}")
    print()
    
    # Show the best fit line
    plt.figure(figsize=(10, 8))
    plt.scatter(X_simple.flatten(), y_simple, color='blue', alpha=0.7, s=100, label='Training data')
    
    # Plot the best fit line
    x_line = np.linspace(1400, 3100, 100)
    y_line = min_theta0 + min_theta1 * x_line
    plt.plot(x_line, y_line, 'r-', linewidth=3, 
             label=f'Best fit: y = {min_theta0:.1f} + {min_theta1:.3f}x')
    
    # Add some example lines
    for theta0, theta1, color, label in example_points:
        y_example = theta0 + theta1 * x_line
        plt.plot(x_line, y_example, color=color, linestyle='--', alpha=0.7, linewidth=2, label=label)
    
    plt.xlabel('Living area (ft²)')
    plt.ylabel('Price (1000$s)')
    plt.title('Best Fit Line from Cost Minimization\nComparison with Other Parameter Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Interpretation:")
    print("  - The contour plot shows how cost varies with parameters")
    print("  - Lower cost (darker colors) = better fit")
    print("  - The red star shows the optimal parameters")
    print("  - Other points show suboptimal parameter choices")
    print("  - The best fit line minimizes prediction errors")
    print()

# ============================================================================
# Example 5: Multiple Features Example
# ============================================================================

def multiple_features_example():
    """
    Example: Working with multiple features (living area + bedrooms).
    This demonstrates how linear regression scales to higher dimensions.
    
    Key Learning Points:
    - Linear regression works with any number of features
    - Each feature has its own weight parameter
    - Vectorized operations scale efficiently
    - Interpretation becomes more complex with more features
    
    Mathematical Formulation:
    h_θ(x) = θ₀x₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
    where x₀ = 1 (intercept term)
    """
    print("=== Example 5: Multiple Features ===")
    print("Scaling linear regression to higher dimensions")
    print()
    
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
    print(f"  Features: intercept (x₀=1), living area (ft²), bedrooms")
    print(f"  Target: price (1000$s)")
    print(f"  Number of training examples: n = {X_multi.shape[0]}")
    print(f"  Number of features (including intercept): d+1 = {X_multi.shape[1]}")
    print()
    
    # Example parameters for multiple features
    theta_multi = np.array([50, 0.1, 20])  # [θ₀, θ₁, θ₂]
    
    def predict_multiple_features(X, theta):
        """
        Make predictions for multiple features using vectorized operations
        
        Parameters:
        X: design matrix (n_samples, n_features)
        theta: parameter vector (n_features,)
        
        Returns:
        predictions: vector of predictions (n_samples,)
        """
        return X @ theta  # Matrix multiplication: Xθ
    
    predictions = predict_multiple_features(X_multi, theta_multi)
    
    print("Predictions with multiple features:")
    print("House | Living Area | Bedrooms | Actual Price | Predicted Price | Error")
    print("-" * 75)
    for i in range(len(y_multi)):
        actual = y_multi[i]
        predicted = predictions[i]
        error = predicted - actual
        print(f"{i+1:5d} | {X_multi[i,1]:11.0f} | {X_multi[i,2]:8.0f} | {actual:12.0f} | {predicted:14.1f} | {error:+6.1f}")
    
    print()
    
    # Compute cost for multiple features
    cost_multi = 0.5 * np.sum((predictions - y_multi) ** 2)
    mse_multi = np.mean((predictions - y_multi) ** 2)
    
    print(f"Performance metrics:")
    print(f"  Cost with multiple features: J(θ) = {cost_multi:.2f}")
    print(f"  MSE with multiple features: MSE(θ) = {mse_multi:.2f}")
    print(f"  Average absolute error: {np.mean(np.abs(predictions - y_multi)):.1f}k")
    print()
    
    # Show parameter interpretation
    print("Parameter interpretation:")
    print(f"  θ₀ = {theta_multi[0]:.1f}: Base price (price when living_area=0, bedrooms=0)")
    print(f"  θ₁ = {theta_multi[1]:.3f}: Price increase per square foot")
    print(f"  θ₂ = {theta_multi[2]:.1f}: Price increase per bedroom")
    print()
    print("Prediction formula:")
    print(f"  price = {theta_multi[0]:.1f} + {theta_multi[1]:.3f} × living_area + {theta_multi[2]:.1f} × bedrooms")
    print()
    
    # Demonstrate feature importance
    print("Feature importance analysis:")
    living_area_range = np.array([1000, 2000, 3000])
    bedrooms_range = np.array([2, 3, 4])
    
    print("Price contributions by feature:")
    print("Living Area | Bedrooms | Base Price | Living Area Contrib. | Bedroom Contrib. | Total")
    print("-" * 90)
    for area in living_area_range:
        for beds in bedrooms_range:
            base_price = theta_multi[0]
            area_contrib = theta_multi[1] * area
            bed_contrib = theta_multi[2] * beds
            total = base_price + area_contrib + bed_contrib
            print(f"{area:11.0f} | {beds:8.0f} | {base_price:10.1f} | {area_contrib:19.1f} | {bed_contrib:14.1f} | {total:5.1f}")
    print()

# ============================================================================
# Example 6: Real-world Application - Housing Price Prediction
# ============================================================================

def housing_price_prediction_example():
    """
    Example: Complete housing price prediction pipeline.
    This demonstrates a real-world application of linear regression.
    
    Key Learning Points:
    - End-to-end machine learning pipeline
    - Data preprocessing and feature engineering
    - Model training and evaluation
    - Making predictions on new data
    - Practical considerations and limitations
    """
    print("=== Example 6: Real-world Housing Price Prediction ===")
    print("Complete machine learning pipeline demonstration")
    print()
    
    # Simulated housing dataset with more realistic features
    np.random.seed(42)
    n_houses = 100
    
    # Generate realistic housing data
    living_areas = np.random.uniform(800, 4000, n_houses)
    bedrooms = np.random.randint(1, 6, n_houses)
    bathrooms = np.random.randint(1, 4, n_houses)
    ages = np.random.randint(0, 50, n_houses)
    
    # Create features matrix
    X_housing = np.column_stack([
        np.ones(n_houses),  # intercept term
        living_areas,
        bedrooms,
        bathrooms,
        ages
    ])
    
    # Generate realistic prices with some noise
    # Price = base_price + area_factor + bedroom_factor + bathroom_factor - age_factor + noise
    true_theta = np.array([50, 0.08, 15, 25, -0.5])  # [base, area, bedroom, bathroom, age]
    prices = X_housing @ true_theta + np.random.normal(0, 20, n_houses)
    
    print(f"Generated housing dataset:")
    print(f"  Number of houses: {n_houses}")
    print(f"  Features: intercept, living area (ft²), bedrooms, bathrooms, age (years)")
    print(f"  Price range: ${prices.min():.0f}k - ${prices.max():.0f}k")
    print(f"  True parameters: θ = {true_theta}")
    print()
    
    # Split data into training and test sets
    train_size = int(0.8 * n_houses)
    X_train = X_housing[:train_size]
    y_train = prices[:train_size]
    X_test = X_housing[train_size:]
    y_test = prices[train_size:]
    
    print(f"Data split:")
    print(f"  Training set: {train_size} houses")
    print(f"  Test set: {n_houses - train_size} houses")
    print()
    
    # Train model using normal equations
    XTX = X_train.T @ X_train
    XTy = X_train.T @ y_train
    theta_learned = np.linalg.inv(XTX) @ XTy
    
    print("Model training results:")
    print(f"{'Parameter':<12} {'True Value':<12} {'Learned Value':<12} {'Difference':<12}")
    print("-" * 50)
    param_names = ['Base Price', 'Area Factor', 'Bedroom Factor', 'Bathroom Factor', 'Age Factor']
    for i, name in enumerate(param_names):
        true_val = true_theta[i]
        learned_val = theta_learned[i]
        diff = learned_val - true_val
        print(f"{name:<12} {true_val:<12.3f} {learned_val:<12.3f} {diff:<+12.3f}")
    print()
    
    # Make predictions
    y_train_pred = X_train @ theta_learned
    y_test_pred = X_test @ theta_learned
    
    # Evaluate model
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    
    print("Model performance:")
    print(f"  Training MSE: {train_mse:.2f}")
    print(f"  Test MSE: {test_mse:.2f}")
    print(f"  Training MAE: {train_mae:.2f}k")
    print(f"  Test MAE: {test_mae:.2f}k")
    print()
    
    # Make predictions on new houses
    new_houses = np.array([
        [1, 2000, 3, 2, 10],  # 2000 ft², 3 bed, 2 bath, 10 years old
        [1, 1500, 2, 1, 5],   # 1500 ft², 2 bed, 1 bath, 5 years old
        [1, 3000, 4, 3, 0],   # 3000 ft², 4 bed, 3 bath, new construction
    ])
    
    new_predictions = new_houses @ theta_learned
    
    print("Predictions for new houses:")
    print("House | Living Area | Bedrooms | Bathrooms | Age | Predicted Price")
    print("-" * 70)
    for i, (house, pred) in enumerate(zip(new_houses, new_predictions)):
        print(f"{i+1:5d} | {house[1]:11.0f} | {house[2]:8.0f} | {house[3]:9.0f} | {house[4]:3.0f} | ${pred:12.0f}k")
    print()
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Training vs Test performance
    plt.subplot(1, 3, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, label='Training', color='blue')
    plt.scatter(y_test, y_test_pred, alpha=0.6, label='Test', color='red')
    plt.plot([prices.min(), prices.max()], [prices.min(), prices.max()], 'k--', alpha=0.5)
    plt.xlabel('Actual Price (1000$s)')
    plt.ylabel('Predicted Price (1000$s)')
    plt.title('Predicted vs Actual Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(1, 3, 2)
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    plt.scatter(y_train_pred, train_residuals, alpha=0.6, label='Training', color='blue')
    plt.scatter(y_test_pred, test_residuals, alpha=0.6, label='Test', color='red')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Predicted Price (1000$s)')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Feature importance
    plt.subplot(1, 3, 3)
    feature_names = ['Base', 'Area', 'Bedrooms', 'Bathrooms', 'Age']
    feature_importance = np.abs(theta_learned[1:])  # Exclude intercept
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum']
    plt.bar(feature_names, feature_importance, color=colors, alpha=0.7)
    plt.xlabel('Features')
    plt.ylabel('Absolute Parameter Value')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Key insights:")
    print("  - Linear regression can capture complex relationships with multiple features")
    print("  - Model performance on test set indicates generalization ability")
    print("  - Residual plot helps identify model assumptions and potential improvements")
    print("  - Feature importance shows which factors most influence price")
    print("  - Predictions provide actionable insights for real estate decisions")
    print()

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("Linear Regression Code Examples with Comprehensive Annotations")
    print("=" * 70)
    print("This file demonstrates the core concepts of linear regression")
    print("with detailed explanations and practical examples.")
    print()
    
    # Run all examples
    plot_housing_data()
    hypothesis_function_example()
    cost_function_examples()
    multiple_features_example()
    housing_price_prediction_example()
    
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
    print("5. Multiple features handling and interpretation")
    print("6. Cost function visualization and optimization landscape")
    print("7. Real-world application with complete ML pipeline")
    print("\nNext steps:")
    print("- Try different parameter values and observe cost changes")
    print("- Experiment with different datasets and features")
    print("- Implement gradient descent to find optimal parameters")
    print("- Explore regularization techniques for better generalization") 
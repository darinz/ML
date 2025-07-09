"""
Locally Weighted Linear Regression (LWR) Examples with Comprehensive Annotations

This file implements locally weighted linear regression as described in the notes.
LWR is a non-parametric method that fits a local linear model for each query point,
allowing it to capture complex non-linear relationships in the data.

Key Concepts Demonstrated:
- Non-parametric vs parametric learning
- Local vs global modeling approaches
- Weight function design and bandwidth selection
- Bias-variance trade-off in local methods
- Computational complexity and scalability
- Kernel functions and their properties
- Curse of dimensionality

Mathematical Foundations:
- Weight function: w^(i) = exp(-(x^(i) - x)^2 / (2τ^2))
- Weighted least squares: θ = (X^T W X)^(-1) X^T W y
- Local prediction: y = [1, x] @ θ
- Bandwidth parameter τ controls locality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ============================================================================
# Core LWR Implementation with Detailed Explanations
# ============================================================================

def gaussian_kernel(x1, x2, tau):
    """
    Compute Gaussian kernel weight between two points.
    
    This is the core weight function that determines how much
    each training example influences the local model at a query point.
    
    Key Learning Points:
    - Gaussian kernel provides smooth, distance-based weighting
    - Bandwidth τ controls the "locality" of the model
    - Smaller τ = more local, potentially overfitting
    - Larger τ = more global, potentially underfitting
    - Weights decay exponentially with squared distance
    
    Mathematical Form:
    w(x1, x2) = exp(-||x1 - x2||² / (2τ²))
    
    Parameters:
    x1: first point (can be vector)
    x2: second point (can be vector)
    tau: bandwidth parameter
    
    Returns:
    weight: kernel weight between x1 and x2
    """
    # Compute squared Euclidean distance
    squared_distance = np.sum((x1 - x2) ** 2)
    
    # Gaussian kernel formula
    weight = np.exp(-squared_distance / (2 * tau ** 2))
    
    return weight

def locally_weighted_linear_regression(x_query, X, y, tau):
    """
    Perform locally weighted linear regression.
    
    This function fits a local linear model for each query point
    by weighting training examples based on their distance to the query.
    
    Key Learning Points:
    - Each query point gets its own local model
    - Weights are computed using the Gaussian kernel
    - Weighted least squares is solved for each query
    - No global model parameters are learned
    - Computational cost scales with number of query points
    
    Algorithm:
    1. For each query point x:
       a. Compute weights w^(i) = exp(-(x^(i) - x)^2 / (2τ^2))
       b. Solve weighted least squares: θ = (X^T W X)^(-1) X^T W y
       c. Make prediction: y = [1, x] @ θ
    
    Parameters:
    x_query: scalar or array, the point(s) to predict
    X: (n_samples, n_features) array of training inputs
    y: (n_samples,) array of training targets
    tau: bandwidth parameter controlling the locality
    
    Returns:
    predicted y at x_query
    """
    # Ensure X is 2D and x_query is 1D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    x_query = np.atleast_1d(x_query)
    
    # Add intercept term to design matrix
    X_design = np.column_stack([np.ones(X.shape[0]), X])
    
    y_pred = []
    
    for x0 in x_query:
        # Step 1: Compute weights for this query point
        # w^(i) = exp(-(x^(i) - x0)^2 / (2τ^2))
        distances = np.sum((X - x0) ** 2, axis=1)
        weights = np.exp(-distances / (2 * tau ** 2))
        
        # Step 2: Create weight matrix W = diag(w)
        W = np.diag(weights)
        
        # Step 3: Solve weighted least squares
        # θ = (X^T W X)^(-1) X^T W y
        XTWX = X_design.T @ W @ X_design
        XTWy = X_design.T @ W @ y
        
        # Use pseudo-inverse for numerical stability
        theta = np.linalg.pinv(XTWX) @ XTWy
        
        # Step 4: Make prediction
        # y = [1, x0] @ θ
        x0_design = np.array([1, x0])
        y0 = x0_design @ theta
        y_pred.append(y0)
    
    return np.array(y_pred)

def demonstrate_weight_function():
    """
    Demonstrate how the weight function changes with distance and bandwidth.
    This helps understand the locality principle of LWR.
    """
    print("=== Weight Function Demonstration ===")
    print("Understanding how Gaussian kernel weights depend on distance and bandwidth")
    print()
    
    # Create example data
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    query_point = 5.5
    
    print(f"Training points: {X}")
    print(f"Query point: {query_point}")
    print()
    
    # Show weights for different bandwidths
    bandwidths = [0.5, 1.0, 2.0, 5.0]
    
    plt.figure(figsize=(12, 8))
    
    for i, tau in enumerate(bandwidths):
        # Calculate weights
        distances = (X - query_point) ** 2
        weights = np.exp(-distances / (2 * tau ** 2))
        
        plt.subplot(2, 2, i+1)
        plt.bar(X, weights, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.axvline(x=query_point, color='red', linestyle='--', linewidth=2, label=f'Query point: {query_point}')
        plt.xlabel('Training data points')
        plt.ylabel('Weight')
        plt.title(f'Weights for τ = {tau}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        print(f"τ = {tau}:")
        print(f"  Max weight: {weights.max():.4f}")
        print(f"  Min weight: {weights.min():.4f}")
        print(f"  Weight at query point: {weights[np.argmin(np.abs(X - query_point))]:.4f}")
        print(f"  Effective neighbors: {np.sum(weights > 0.1):.0f} points")
        print()
    
    plt.tight_layout()
    plt.show()
    
    # Show weight decay with distance
    print("Weight decay with distance:")
    distances = np.linspace(0, 10, 100)
    
    plt.figure(figsize=(10, 6))
    for tau in bandwidths:
        weights = np.exp(-distances ** 2 / (2 * tau ** 2))
        plt.plot(distances, weights, linewidth=2, label=f'τ = {tau}')
    
    plt.xlabel('Distance from query point')
    plt.ylabel('Weight')
    plt.title('Weight Decay with Distance for Different Bandwidths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("Key insights:")
    print("- Smaller τ: weights decay rapidly, very local model")
    print("- Larger τ: weights decay slowly, more global model")
    print("- Weights are always positive and sum to less than n")
    print("- Bandwidth selection is crucial for performance")
    print()

# ============================================================================
# Comparison with Global Linear Regression
# ============================================================================

def compare_lwr_with_global_linear():
    """
    Compare locally weighted linear regression with global linear regression.
    This demonstrates how LWR can capture local patterns that global linear regression misses.
    """
    print("=== Locally Weighted Linear Regression vs Global Linear Regression ===")
    print("Demonstrating the advantages of local modeling for non-linear data")
    print()
    
    # Generate synthetic data with non-linear pattern
    np.random.seed(42)
    X = np.linspace(0, 10, 30)
    y = np.sin(X) + 0.3 * np.random.randn(30)
    
    print(f"Generated {len(X)} data points with non-linear pattern")
    print(f"X range: [{X.min():.1f}, {X.max():.1f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    print()
    
    # Test points for prediction
    X_test = np.linspace(0, 10, 200)
    
    # 1. Global linear regression (ordinary least squares)
    X_design = np.column_stack([np.ones_like(X), X])
    theta_global = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    y_pred_global = np.array([np.array([1, x]) @ theta_global for x in X_test])
    
    # 2. Locally weighted linear regression with different bandwidths
    tau_small = 0.5   # Small bandwidth = more local, potentially overfitting
    tau_medium = 1.0  # Medium bandwidth = balanced
    tau_large = 2.0   # Large bandwidth = more global, potentially underfitting
    
    y_pred_lwr_small = locally_weighted_linear_regression(X_test, X, y, tau_small)
    y_pred_lwr_medium = locally_weighted_linear_regression(X_test, X, y, tau_medium)
    y_pred_lwr_large = locally_weighted_linear_regression(X_test, X, y, tau_large)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Global linear regression
    plt.subplot(2, 2, 1)
    plt.scatter(X, y, color='blue', alpha=0.7, s=50, label='Training data')
    plt.plot(X_test, y_pred_global, 'r-', linewidth=2, label='Global linear regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Global Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: LWR with small bandwidth
    plt.subplot(2, 2, 2)
    plt.scatter(X, y, color='blue', alpha=0.7, s=50, label='Training data')
    plt.plot(X_test, y_pred_lwr_small, 'g-', linewidth=2, label=f'LWR (τ={tau_small})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('LWR with Small Bandwidth (τ=0.5)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: LWR with medium bandwidth
    plt.subplot(2, 2, 3)
    plt.scatter(X, y, color='blue', alpha=0.7, s=50, label='Training data')
    plt.plot(X_test, y_pred_lwr_medium, 'orange', linewidth=2, label=f'LWR (τ={tau_medium})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('LWR with Medium Bandwidth (τ=1.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: LWR with large bandwidth
    plt.subplot(2, 2, 4)
    plt.scatter(X, y, color='blue', alpha=0.7, s=50, label='Training data')
    plt.plot(X_test, y_pred_lwr_large, 'purple', linewidth=2, label=f'LWR (τ={tau_large})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('LWR with Large Bandwidth (τ=2.0)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate and display errors
    def calculate_mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    # For comparison, we'll use the training data as test data
    y_pred_global_train = np.array([np.array([1, x]) @ theta_global for x in X])
    y_pred_lwr_small_train = locally_weighted_linear_regression(X, X, y, tau_small)
    y_pred_lwr_medium_train = locally_weighted_linear_regression(X, X, y, tau_medium)
    y_pred_lwr_large_train = locally_weighted_linear_regression(X, X, y, tau_large)
    
    print("Mean Squared Error on Training Data:")
    print(f"  Global linear regression: {calculate_mse(y, y_pred_global_train):.4f}")
    print(f"  LWR (τ={tau_small}): {calculate_mse(y, y_pred_lwr_small_train):.4f}")
    print(f"  LWR (τ={tau_medium}): {calculate_mse(y, y_pred_lwr_medium_train):.4f}")
    print(f"  LWR (τ={tau_large}): {calculate_mse(y, y_pred_lwr_large_train):.4f}")
    print()
    
    print("Key insights:")
    print("- Global linear regression fails to capture non-linear patterns")
    print("- LWR can adapt to local structure in the data")
    print("- Small τ captures fine details but may overfit")
    print("- Large τ is more smooth but may miss local patterns")
    print("- Optimal τ balances bias and variance")
    print()

# ============================================================================
# Bandwidth Selection and Cross-Validation
# ============================================================================

def bandwidth_selection_cross_validation():
    """
    Demonstrate bandwidth selection using cross-validation.
    This shows how to choose the optimal bandwidth parameter.
    """
    print("=== Bandwidth Selection with Cross-Validation ===")
    print("Finding the optimal bandwidth parameter for LWR")
    print()
    
    # Generate data with known structure
    np.random.seed(42)
    X = np.linspace(0, 10, 50)
    y = np.sin(X) + 0.2 * np.random.randn(50)
    
    print(f"Generated {len(X)} data points")
    print(f"True function: y = sin(x) + noise")
    print()
    
    # Test different bandwidths
    bandwidths = np.logspace(-1, 1, 20)  # 0.1 to 10
    cv_scores = []
    
    # Leave-one-out cross-validation
    for tau in bandwidths:
        cv_errors = []
        
        for i in range(len(X)):
            # Leave out point i
            X_train = np.delete(X, i)
            y_train = np.delete(y, i)
            X_test = X[i]
            y_test = y[i]
            
            # Predict using LWR
            y_pred = locally_weighted_linear_regression(X_test, X_train, y_train, tau)
            cv_errors.append((y_test - y_pred[0]) ** 2)
        
        # Average cross-validation error
        cv_score = np.mean(cv_errors)
        cv_scores.append(cv_score)
    
    # Find optimal bandwidth
    optimal_idx = np.argmin(cv_scores)
    optimal_tau = bandwidths[optimal_idx]
    
    print("Cross-validation results:")
    print(f"  Optimal bandwidth: τ = {optimal_tau:.3f}")
    print(f"  Minimum CV error: {cv_scores[optimal_idx]:.4f}")
    print()
    
    # Visualize cross-validation results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: CV error vs bandwidth
    plt.subplot(1, 2, 1)
    plt.semilogx(bandwidths, cv_scores, 'bo-', linewidth=2, markersize=6)
    plt.axvline(x=optimal_tau, color='red', linestyle='--', linewidth=2, label=f'Optimal τ = {optimal_tau:.3f}')
    plt.xlabel('Bandwidth τ')
    plt.ylabel('Cross-validation MSE')
    plt.title('Bandwidth Selection via Cross-validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: LWR with optimal bandwidth
    plt.subplot(1, 2, 2)
    X_test = np.linspace(0, 10, 200)
    y_pred_optimal = locally_weighted_linear_regression(X_test, X, y, optimal_tau)
    
    plt.scatter(X, y, color='blue', alpha=0.7, s=50, label='Training data')
    plt.plot(X_test, y_pred_optimal, 'r-', linewidth=2, label=f'LWR (τ={optimal_tau:.3f})')
    
    # Plot true function
    y_true = np.sin(X_test)
    plt.plot(X_test, y_true, 'g--', linewidth=2, label='True function: sin(x)')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('LWR with Optimal Bandwidth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Compare with other bandwidths
    print("Comparison with other bandwidths:")
    test_bandwidths = [0.1, optimal_tau, 2.0, 5.0]
    
    for tau in test_bandwidths:
        y_pred = locally_weighted_linear_regression(X, X, y, tau)
        mse = np.mean((y - y_pred) ** 2)
        print(f"  τ = {tau:.3f}: MSE = {mse:.4f}")
    
    print()
    print("Key insights:")
    print("- Cross-validation helps select optimal bandwidth")
    print("- Too small τ: high variance, overfitting")
    print("- Too large τ: high bias, underfitting")
    print("- Optimal τ balances bias and variance")
    print("- Leave-one-out CV is computationally expensive")
    print()

# ============================================================================
# Real-world Application: Housing Price Prediction
# ============================================================================

def housing_price_lwr():
    """
    Apply LWR to the housing price prediction example.
    This shows how LWR can capture local trends in real estate data.
    """
    print("=== Housing Price Prediction with LWR ===")
    print("Real-world application demonstrating local modeling advantages")
    print()
    
    # Housing data: [living_area, price]
    living_area = np.array([2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494])
    price = np.array([400, 330, 369, 232, 540, 400, 330, 369, 232, 540])
    
    print(f"Housing dataset: {len(living_area)} houses")
    print(f"Living areas: {living_area.min():.0f} to {living_area.max():.0f} ft²")
    print(f"Prices: ${price.min():.0f}k to ${price.max():.0f}k")
    print()
    
    # Test points for prediction
    X_test = np.linspace(1400, 3100, 100)
    
    # Compare different methods
    # 1. Global linear regression
    X_design = np.column_stack([np.ones_like(living_area), living_area])
    theta_global = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ price
    y_pred_global = np.array([np.array([1, x]) @ theta_global for x in X_test])
    
    # 2. LWR with different bandwidths
    bandwidths = [200, 500, 1000]
    y_pred_lwr = {}
    
    for tau in bandwidths:
        y_pred_lwr[tau] = locally_weighted_linear_regression(X_test, living_area, price, tau)
    
    # Visualize results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Global linear regression
    plt.subplot(2, 2, 1)
    plt.scatter(living_area, price, color='blue', alpha=0.7, s=100, label='Training data')
    plt.plot(X_test, y_pred_global, 'r-', linewidth=2, label='Global linear regression')
    plt.xlabel('Living area (ft²)')
    plt.ylabel('Price (1000$s)')
    plt.title('Global Linear Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot LWR with different bandwidths
    for i, tau in enumerate(bandwidths):
        plt.subplot(2, 2, i+2)
        plt.scatter(living_area, price, color='blue', alpha=0.7, s=100, label='Training data')
        plt.plot(X_test, y_pred_lwr[tau], 'g-', linewidth=2, label=f'LWR (τ={tau})')
        plt.xlabel('Living area (ft²)')
        plt.ylabel('Price (1000$s)')
        plt.title(f'LWR with τ = {tau}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Make predictions for specific houses
    test_houses = np.array([1800, 2200, 2800])
    
    print("Predictions for specific houses:")
    print("Living Area | Global LR | LWR (τ=200) | LWR (τ=500) | LWR (τ=1000)")
    print("-" * 80)
    
    for area in test_houses:
        # Global linear regression
        pred_global = np.array([1, area]) @ theta_global
        
        # LWR predictions
        pred_lwr_200 = locally_weighted_linear_regression(area, living_area, price, 200)[0]
        pred_lwr_500 = locally_weighted_linear_regression(area, living_area, price, 500)[0]
        pred_lwr_1000 = locally_weighted_linear_regression(area, living_area, price, 1000)[0]
        
        print(f"{area:11.0f} | {pred_global:9.0f} | {pred_lwr_200:11.0f} | {pred_lwr_500:11.0f} | {pred_lwr_1000:12.0f}")
    
    print()
    
    # Analyze local behavior
    print("Local behavior analysis:")
    print("  - τ = 200: Very local, adapts to nearby points")
    print("  - τ = 500: Moderate locality, smooth predictions")
    print("  - τ = 1000: More global, similar to linear regression")
    print("  - Global LR: Single linear relationship for all areas")
    print()
    
    print("Key insights:")
    print("- LWR can capture non-linear price trends")
    print("- Different bandwidths provide different levels of locality")
    print("- Local models may be more accurate for specific areas")
    print("- Bandwidth selection affects prediction quality")
    print("- LWR is computationally more expensive than global methods")
    print()

# ============================================================================
# Advanced Topics: Computational Complexity and Scalability
# ============================================================================

def computational_complexity_analysis():
    """
    Analyze the computational complexity of LWR and compare with global methods.
    This demonstrates the trade-offs between accuracy and computational cost.
    """
    print("=== Computational Complexity Analysis ===")
    print("Understanding the computational costs of LWR vs global methods")
    print()
    
    import time
    
    # Generate datasets of different sizes
    dataset_sizes = [50, 100, 200, 500, 1000]
    results = {}
    
    for n in dataset_sizes:
        # Generate data
        np.random.seed(42)
        X = np.random.uniform(0, 10, n)
        y = np.sin(X) + 0.2 * np.random.randn(n)
        
        # Test points
        X_test = np.linspace(0, 10, 100)
        
        # Time global linear regression
        start_time = time.time()
        X_design = np.column_stack([np.ones_like(X), X])
        theta_global = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
        y_pred_global = np.array([np.array([1, x]) @ theta_global for x in X_test])
        global_time = time.time() - start_time
        
        # Time LWR
        start_time = time.time()
        y_pred_lwr = locally_weighted_linear_regression(X_test, X, y, tau=1.0)
        lwr_time = time.time() - start_time
        
        results[n] = {
            'global_time': global_time,
            'lwr_time': lwr_time,
            'speedup': global_time / lwr_time
        }
    
    # Display results
    print("Computational time comparison:")
    print(f"{'Dataset Size':<15} {'Global LR (s)':<15} {'LWR (s)':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for n in dataset_sizes:
        result = results[n]
        print(f"{n:<15} {result['global_time']:<15.4f} {result['lwr_time']:<15.4f} {result['speedup']:<10.1f}x")
    
    print()
    
    # Visualize complexity
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Time vs dataset size
    plt.subplot(1, 2, 1)
    sizes = list(results.keys())
    global_times = [results[n]['global_time'] for n in sizes]
    lwr_times = [results[n]['lwr_time'] for n in sizes]
    
    plt.loglog(sizes, global_times, 'bo-', linewidth=2, markersize=8, label='Global LR')
    plt.loglog(sizes, lwr_times, 'ro-', linewidth=2, markersize=8, label='LWR')
    plt.xlabel('Dataset Size')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computational Complexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs dataset size
    plt.subplot(1, 2, 2)
    speedups = [results[n]['speedup'] for n in sizes]
    plt.semilogx(sizes, speedups, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Dataset Size')
    plt.ylabel('Speedup (Global LR / LWR)')
    plt.title('Computational Speedup')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Theoretical complexity analysis
    print("Theoretical complexity analysis:")
    print("  Global Linear Regression:")
    print("    - Training: O(n²) for matrix operations")
    print("    - Prediction: O(1) per query point")
    print("    - Total for m queries: O(n² + m)")
    print()
    print("  Locally Weighted Regression:")
    print("    - No training phase")
    print("    - Prediction: O(n²) per query point")
    print("    - Total for m queries: O(m × n²)")
    print()
    print("Key insights:")
    print("- Global LR scales better with number of queries")
    print("- LWR scales poorly with dataset size")
    print("- LWR is suitable for small datasets or few queries")
    print("- Approximate methods can improve LWR scalability")
    print("- Memory usage is also higher for LWR")
    print()

# ============================================================================
# Curse of Dimensionality and High-dimensional Data
# ============================================================================

def curse_of_dimensionality_demo():
    """
    Demonstrate the curse of dimensionality in LWR.
    This shows how LWR performance degrades in high dimensions.
    """
    print("=== Curse of Dimensionality Demonstration ===")
    print("Understanding how LWR performance degrades with dimensionality")
    print()
    
    np.random.seed(42)
    n_samples = 100
    n_queries = 20
    
    # Test different dimensions
    dimensions = [1, 2, 5, 10, 20]
    results = {}
    
    for d in dimensions:
        # Generate data in d dimensions
        X = np.random.randn(n_samples, d)
        # Simple linear function with noise
        y = np.sum(X, axis=1) + 0.1 * np.random.randn(n_samples)
        
        # Generate query points
        X_query = np.random.randn(n_queries, d)
        
        # Time LWR prediction
        start_time = time.time()
        y_pred = locally_weighted_linear_regression(X_query, X, y, tau=1.0)
        lwr_time = time.time() - start_time
        
        # Calculate prediction error
        y_true = np.sum(X_query, axis=1)
        mse = np.mean((y_pred - y_true) ** 2)
        
        results[d] = {
            'time': lwr_time,
            'mse': mse,
            'avg_weight': 0  # Will calculate below
        }
        
        # Calculate average weight (measure of locality)
        total_weight = 0
        for i in range(n_queries):
            distances = np.sum((X - X_query[i]) ** 2, axis=1)
            weights = np.exp(-distances / (2 * 1.0 ** 2))
            total_weight += np.mean(weights)
        
        results[d]['avg_weight'] = total_weight / n_queries
    
    # Display results
    print("Performance vs dimensionality:")
    print(f"{'Dimensions':<12} {'Time (s)':<10} {'MSE':<10} {'Avg Weight':<12}")
    print("-" * 50)
    
    for d in dimensions:
        result = results[d]
        print(f"{d:<12} {result['time']:<10.4f} {result['mse']:<10.4f} {result['avg_weight']:<12.4f}")
    
    print()
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Time vs dimensions
    plt.subplot(1, 2, 1)
    dims = list(results.keys())
    times = [results[d]['time'] for d in dims]
    plt.semilogy(dims, times, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Computation Time (seconds)')
    plt.title('Computational Cost vs Dimensionality')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Average weight vs dimensions
    plt.subplot(1, 2, 2)
    avg_weights = [results[d]['avg_weight'] for d in dims]
    plt.semilogy(dims, avg_weights, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Average Weight')
    plt.title('Locality vs Dimensionality')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Curse of dimensionality analysis:")
    print("- Computation time increases with dimensionality")
    print("- Average weights decrease exponentially with dimension")
    print("- In high dimensions, all points become equally distant")
    print("- LWR loses its local nature in high dimensions")
    print("- Dimensionality reduction may be necessary")
    print()

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("Locally Weighted Linear Regression Examples with Comprehensive Annotations")
    print("=" * 70)
    print("This file demonstrates locally weighted linear regression")
    print("with detailed explanations and practical applications.")
    print()
    
    # Run demonstrations
    demonstrate_weight_function()
    compare_lwr_with_global_linear()
    bandwidth_selection_cross_validation()
    housing_price_lwr()
    computational_complexity_analysis()
    curse_of_dimensionality_demo()
    
    print("All examples completed!")
    print("\nKey concepts demonstrated:")
    print("1. Non-parametric vs parametric learning approaches")
    print("2. Local vs global modeling strategies")
    print("3. Gaussian kernel weighting and bandwidth selection")
    print("4. Bias-variance trade-off in local methods")
    print("5. Cross-validation for hyperparameter tuning")
    print("6. Computational complexity and scalability")
    print("7. Curse of dimensionality in high-dimensional spaces")
    print("\nNext steps:")
    print("- Explore other kernel functions (Epanechnikov, triangular)")
    print("- Implement approximate LWR methods (k-d trees, ball trees)")
    print("- Study kernel regression and Nadaraya-Watson estimator")
    print("- Investigate adaptive bandwidth selection")
    print("- Apply to other non-parametric methods (LOESS, splines)") 
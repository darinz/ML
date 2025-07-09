"""
LMS Algorithm and Gradient Descent Examples with Comprehensive Annotations

This file implements the Least Mean Squares (LMS) algorithm and various gradient descent
variants as described in the notes. Includes batch, stochastic, and mini-batch methods
with detailed explanations and practical demonstrations.

Key Concepts Demonstrated:
- LMS update rule: θ := θ + α(y - h_θ(x))x
- Gradient descent optimization
- Batch vs Stochastic vs Mini-batch gradient descent
- Learning rate selection and convergence
- Cost function minimization
- Practical implementation considerations

Mathematical Foundations:
- Gradient computation: ∇J(θ) = (1/n)X^T(y - Xθ)
- Update rule: θ := θ - α∇J(θ)
- Cost function: J(θ) = (1/2)Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Core Functions with Detailed Annotations
# ============================================================================

def h_theta(theta, x):
    """
    Compute the linear hypothesis h_theta(x) = theta^T x.
    
    This is the core prediction function that we want to optimize.
    
    Parameters:
    theta: parameter vector (n_features,)
    x: feature vector (n_features,) including intercept term
    
    Returns:
    prediction: scalar prediction value
    """
    return np.dot(theta, x)

def cost_single(theta, x, y):
    """
    Compute the cost for a single example: J(θ) = (1/2)(h_θ(x) - y)².
    
    This is the squared error for one training example.
    Used in stochastic gradient descent for individual updates.
    
    Parameters:
    theta: parameter vector
    x: single feature vector
    y: single target value
    
    Returns:
    cost: squared error for this example
    """
    return 0.5 * (h_theta(theta, x) - y) ** 2

def gradient_single(theta, x, y):
    """
    Compute the gradient of the cost for a single example: ∇J(θ) = (h_θ(x) - y)x.
    
    This is the gradient with respect to θ for one training example.
    The gradient points in the direction of steepest increase in cost.
    
    Mathematical derivation:
    J(θ) = (1/2)(h_θ(x) - y)² = (1/2)(θ^T x - y)²
    ∇J(θ) = (θ^T x - y) * ∇(θ^T x) = (h_θ(x) - y) * x
    
    Parameters:
    theta: parameter vector
    x: single feature vector
    y: single target value
    
    Returns:
    gradient: gradient vector for this example
    """
    prediction = h_theta(theta, x)
    error = prediction - y
    grad = error * x
    return grad

def update_theta_single(theta, x, y, alpha):
    """
    Update theta for a single example using SGD: θ := θ - α∇J(θ).
    
    This implements the LMS update rule for one training example.
    The negative sign ensures we move in the direction of decreasing cost.
    
    Parameters:
    theta: current parameter vector
    x: single feature vector
    y: single target value
    alpha: learning rate
    
    Returns:
    theta_new: updated parameter vector
    """
    grad = gradient_single(theta, x, y)
    return theta - alpha * grad

def cost_batch(theta, X, y):
    """
    Compute the mean cost over a dataset: J(θ) = (1/2n)Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)².
    
    This is the average cost across all training examples.
    Used for monitoring convergence during training.
    
    Parameters:
    theta: parameter vector
    X: design matrix (n_samples, n_features)
    y: target vector (n_samples,)
    
    Returns:
    cost: average cost across all examples
    """
    predictions = X @ theta
    return 0.5 * np.mean((predictions - y) ** 2)

# ============================================================================
# Gradient Descent Implementations with Detailed Explanations
# ============================================================================

def batch_gradient_descent(theta, X, y, alpha, num_iters, verbose=True):
    """
    Perform batch gradient descent for a number of iterations.
    
    Batch gradient descent computes the gradient using ALL training examples
    at each iteration. This provides the most accurate gradient estimate but
    can be computationally expensive for large datasets.
    
    Algorithm:
    1. Compute predictions for all examples: Xθ
    2. Compute gradient: ∇J(θ) = (1/n)X^T(y - Xθ)
    3. Update parameters: θ := θ - α∇J(θ)
    4. Repeat until convergence
    
    Advantages:
    - Most accurate gradient estimate
    - Guaranteed convergence to local minimum
    - Deterministic updates
    
    Disadvantages:
    - Computationally expensive for large datasets
    - Can get stuck in local minima
    - Memory intensive
    
    Parameters:
    theta: initial parameter vector
    X: design matrix (n_samples, n_features)
    y: target vector (n_samples,)
    alpha: learning rate
    num_iters: number of iterations
    verbose: whether to print progress
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values at each iteration
    """
    n = len(y)
    cost_history = []
    
    if verbose:
        print(f"Batch Gradient Descent: {num_iters} iterations, α={alpha}")
        print(f"Dataset size: {n} training examples, {X.shape[1]} features")
        print(f"Initial cost: {cost_batch(theta, X, y):.4f}")
        print()
    
    for i in range(num_iters):
        # Step 1: Compute predictions for all examples
        predictions = X @ theta
        
        # Step 2: Compute gradient using all examples
        # ∇J(θ) = (1/n)X^T(y - Xθ) = (1/n)X^T(y - predictions)
        gradient = (X.T @ (y - predictions)) / n
        
        # Step 3: Update parameters
        # θ := θ - α∇J(θ) (negative because we want to minimize cost)
        theta = theta - alpha * gradient
        
        # Step 4: Record cost for monitoring
        cost = cost_batch(theta, X, y)
        cost_history.append(cost)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"Iteration {i+1}: cost = {cost:.4f}, ||∇J|| = {np.linalg.norm(gradient):.6f}")
    
    if verbose:
        print(f"Final cost: {cost_history[-1]:.4f}")
        print(f"Total iterations: {num_iters}")
        print(f"Final gradient norm: {np.linalg.norm(gradient):.6f}")
        print()
    
    return theta, cost_history

def stochastic_gradient_descent(theta, X, y, alpha, num_epochs, verbose=True):
    """
    Perform stochastic gradient descent for a number of epochs.
    
    Stochastic gradient descent (SGD) uses ONE training example at a time
    to compute the gradient. This makes it much faster but introduces noise
    in the gradient estimates.
    
    Algorithm:
    1. Shuffle training examples
    2. For each example (x⁽ⁱ⁾, y⁽ⁱ⁾):
       a. Compute prediction: h_θ(x⁽ⁱ⁾)
       b. Compute gradient: ∇J(θ) = (h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)x⁽ⁱ⁾
       c. Update parameters: θ := θ - α∇J(θ)
    3. Repeat for specified number of epochs
    
    Advantages:
    - Very fast, especially for large datasets
    - Can escape local minima due to noise
    - Memory efficient
    - Good for online learning
    
    Disadvantages:
    - Noisy gradient estimates
    - May not converge to exact minimum
    - Requires careful learning rate tuning
    
    Parameters:
    theta: initial parameter vector
    X: design matrix (n_samples, n_features)
    y: target vector (n_samples,)
    alpha: learning rate
    num_epochs: number of epochs (passes through the data)
    verbose: whether to print progress
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values at each epoch
    """
    n = len(y)
    cost_history = []
    
    if verbose:
        print(f"Stochastic Gradient Descent: {num_epochs} epochs, α={alpha}")
        print(f"Dataset size: {n} training examples, {X.shape[1]} features")
        print(f"Updates per epoch: {n}")
        print(f"Initial cost: {cost_batch(theta, X, y):.4f}")
        print()
    
    for epoch in range(num_epochs):
        # Step 1: Shuffle the data for each epoch
        # This ensures we don't always see examples in the same order
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Step 2: Update parameters using each example
        for i in range(n):
            xi = X_shuffled[i]
            yi = y_shuffled[i]
            
            # Compute prediction for this example
            prediction = h_theta(theta, xi)
            
            # Compute gradient for this example
            error = prediction - yi
            gradient = error * xi
            
            # Update parameters
            theta = theta - alpha * gradient
        
        # Step 3: Record cost at end of epoch
        cost = cost_batch(theta, X, y)
        cost_history.append(cost)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: cost = {cost:.4f}")
    
    if verbose:
        print(f"Final cost: {cost_history[-1]:.4f}")
        print(f"Total epochs: {num_epochs}")
        print(f"Total updates: {num_epochs * n}")
        print()
    
    return theta, cost_history

def minibatch_gradient_descent(theta, X, y, alpha, batch_size, num_epochs, verbose=True):
    """
    Perform mini-batch gradient descent for a number of epochs.
    
    Mini-batch gradient descent is a compromise between batch and stochastic
    gradient descent. It uses a small batch of examples to compute the gradient,
    providing a balance between accuracy and speed.
    
    Algorithm:
    1. Shuffle training examples
    2. Split data into mini-batches of size batch_size
    3. For each mini-batch:
       a. Compute predictions for batch: X_batch @ θ
       b. Compute gradient: ∇J(θ) = (1/batch_size)X_batch^T(y_batch - X_batch @ θ)
       c. Update parameters: θ := θ - α∇J(θ)
    4. Repeat for specified number of epochs
    
    Advantages:
    - Good balance between accuracy and speed
    - Can leverage vectorized operations
    - More stable than SGD
    - Can escape local minima
    
    Disadvantages:
    - Requires tuning batch size
    - Still some noise in gradient estimates
    - Memory usage scales with batch size
    
    Parameters:
    theta: initial parameter vector
    X: design matrix (n_samples, n_features)
    y: target vector (n_samples,)
    alpha: learning rate
    batch_size: size of mini-batches
    num_epochs: number of epochs
    verbose: whether to print progress
    
    Returns:
    theta: optimized parameters
    cost_history: list of cost values at each epoch
    """
    n = len(y)
    cost_history = []
    
    if verbose:
        print(f"Mini-batch Gradient Descent: {num_epochs} epochs, α={alpha}, batch_size={batch_size}")
        print(f"Dataset size: {n} training examples, {X.shape[1]} features")
        print(f"Batches per epoch: {n // batch_size + (1 if n % batch_size else 0)}")
        print(f"Initial cost: {cost_batch(theta, X, y):.4f}")
        print()
    
    for epoch in range(num_epochs):
        # Step 1: Shuffle the data for each epoch
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Step 2: Process mini-batches
        for start in range(0, n, batch_size):
            end = start + batch_size
            xb = X_shuffled[start:end]
            yb = y_shuffled[start:end]
            
            # Compute predictions for this batch
            predictions = xb @ theta
            
            # Compute gradient for this batch
            # ∇J(θ) = (1/batch_size)X_batch^T(y_batch - X_batch @ θ)
            gradient = (xb.T @ (yb - predictions)) / len(yb)
            
            # Update parameters
            theta = theta - alpha * gradient
        
        # Step 3: Record cost at end of epoch
        cost = cost_batch(theta, X, y)
        cost_history.append(cost)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: cost = {cost:.4f}")
    
    if verbose:
        print(f"Final cost: {cost_history[-1]:.4f}")
        print(f"Total epochs: {num_epochs}")
        print()
    
    return theta, cost_history

# ============================================================================
# Example Demonstrations with Detailed Explanations
# ============================================================================

def demonstrate_single_example_update():
    """
    Demonstrate how a single training example updates the parameters.
    This shows the core LMS update rule in action.
    """
    print("=== Single Example Update Demonstration ===")
    print("Understanding the LMS update rule: θ := θ + α(y - h_θ(x))x")
    print()
    
    # Initial parameters
    theta = np.array([1.0, 0.5])
    print(f"Initial parameters: θ = {theta}")
    
    # Single training example
    x = np.array([1.0, 2.0])  # [intercept, feature]
    y = 3.0  # target value
    
    print(f"Training example: x = {x}, y = {y}")
    print()
    
    # Compute current prediction
    prediction = h_theta(theta, x)
    print(f"Current prediction: h_θ(x) = {prediction:.3f}")
    print(f"Target value: y = {y}")
    print(f"Error: y - h_θ(x) = {y - prediction:.3f}")
    print()
    
    # Compute gradient
    gradient = gradient_single(theta, x, y)
    print(f"Gradient: ∇J(θ) = (h_θ(x) - y)x = {gradient}")
    print()
    
    # Update with different learning rates
    learning_rates = [0.1, 0.5, 1.0]
    
    print("Parameter updates with different learning rates:")
    print("α | Old θ | Gradient | Update | New θ")
    print("-" * 50)
    
    for alpha in learning_rates:
        theta_old = theta.copy()
        update = alpha * gradient
        theta_new = theta_old - update
        
        print(f"{alpha} | {theta_old} | {gradient} | {update} | {theta_new}")
    
    print()
    print("Interpretation:")
    print("- Positive error (y > h_θ(x)): parameters increase")
    print("- Negative error (y < h_θ(x)): parameters decrease")
    print("- Larger learning rate: bigger parameter changes")
    print("- Gradient direction: points toward increasing cost")
    print("- Update direction: opposite to gradient (decreasing cost)")
    print()

def compare_gradient_descent_methods():
    """
    Compare batch, stochastic, and mini-batch gradient descent.
    This demonstrates the trade-offs between different optimization methods.
    """
    print("=== Comparison of Gradient Descent Methods ===")
    print("Understanding the trade-offs between different optimization approaches")
    print()
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # True parameters
    theta_true = np.array([2.0, -1.5, 0.8])
    
    # Generate features and targets
    X = np.random.randn(n_samples, n_features)
    X = np.column_stack([np.ones(n_samples), X])  # Add intercept
    
    # Generate targets with noise
    y = X @ theta_true + 0.5 * np.random.randn(n_samples)
    
    print(f"Generated dataset: {n_samples} samples, {n_features} features")
    print(f"True parameters: θ = {theta_true}")
    print()
    
    # Initial parameters
    theta_init = np.zeros(n_features + 1)
    
    # Test different methods
    methods = {
        'Batch GD': lambda: batch_gradient_descent(theta_init.copy(), X, y, alpha=0.01, num_iters=1000, verbose=False),
        'Stochastic GD': lambda: stochastic_gradient_descent(theta_init.copy(), X, y, alpha=0.01, num_epochs=10, verbose=False),
        'Mini-batch GD': lambda: minibatch_gradient_descent(theta_init.copy(), X, y, alpha=0.01, batch_size=10, num_epochs=10, verbose=False)
    }
    
    results = {}
    
    print("Training different gradient descent methods...")
    print()
    
    for method_name, method_func in methods.items():
        print(f"Training {method_name}...")
        theta_final, cost_history = method_func()
        results[method_name] = {
            'theta': theta_final,
            'cost_history': cost_history,
            'final_cost': cost_history[-1],
            'iterations': len(cost_history)
        }
        print(f"  Final cost: {cost_history[-1]:.4f}")
        print(f"  Iterations: {len(cost_history)}")
        print()
    
    # Compare results
    print("Method Comparison:")
    print(f"{'Method':<15} {'Final Cost':<12} {'Iterations':<12} {'||θ - θ_true||':<15}")
    print("-" * 60)
    
    for method_name, result in results.items():
        theta_error = np.linalg.norm(result['theta'] - theta_true)
        print(f"{method_name:<15} {result['final_cost']:<12.4f} {result['iterations']:<12} {theta_error:<15.4f}")
    
    print()
    
    # Visualize convergence
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Cost convergence
    plt.subplot(1, 3, 1)
    for method_name, result in results.items():
        plt.plot(result['cost_history'], label=method_name, linewidth=2)
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(θ)')
    plt.title('Cost Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to see differences clearly
    
    # Plot 2: Parameter convergence
    plt.subplot(1, 3, 2)
    x_pos = np.arange(len(theta_true))
    width = 0.25
    
    for i, (method_name, result) in enumerate(results.items()):
        plt.bar(x_pos + i*width, result['theta'], width, label=method_name, alpha=0.7)
    
    plt.bar(x_pos + len(results)*width, theta_true, width, label='True θ', alpha=0.7, color='black')
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.title('Final Parameter Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Convergence speed
    plt.subplot(1, 3, 3)
    for method_name, result in results.items():
        # Normalize iterations for fair comparison
        normalized_iterations = np.linspace(0, 1, len(result['cost_history']))
        plt.plot(normalized_iterations, result['cost_history'], label=method_name, linewidth=2)
    
    plt.xlabel('Normalized Progress')
    plt.ylabel('Cost J(θ)')
    plt.title('Convergence Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("Key insights:")
    print("- Batch GD: Most stable but slowest convergence")
    print("- Stochastic GD: Fastest but most noisy")
    print("- Mini-batch GD: Good balance between speed and stability")
    print("- All methods converge to similar solutions")
    print("- Choice depends on dataset size and computational constraints")
    print()

def housing_price_optimization():
    """
    Apply gradient descent to the housing price prediction problem.
    This demonstrates a real-world application of the LMS algorithm.
    """
    print("=== Housing Price Optimization with Gradient Descent ===")
    print("Real-world application of LMS algorithm")
    print()
    
    # Housing data
    living_area = np.array([2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494])
    price = np.array([400, 330, 369, 232, 540, 400, 330, 369, 232, 540])
    
    # Create design matrix
    X = np.column_stack([np.ones(len(living_area)), living_area])
    y = price
    
    print(f"Housing dataset: {len(living_area)} houses")
    print(f"Features: intercept, living area (ft²)")
    print(f"Target: price (1000$s)")
    print()
    
    # Initial parameters
    theta_init = np.array([0.0, 0.0])
    
    # Test different learning rates
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    
    plt.figure(figsize=(15, 10))
    
    for i, alpha in enumerate(learning_rates):
        # Run gradient descent
        theta_final, cost_history = batch_gradient_descent(
            theta_init.copy(), X, y, alpha, num_iters=1000, verbose=False
        )
        
        # Plot convergence
        plt.subplot(2, 2, i+1)
        plt.plot(cost_history, linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Cost J(θ)')
        plt.title(f'Convergence with α = {alpha}')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        print(f"α = {alpha}:")
        print(f"  Final cost: {cost_history[-1]:.4f}")
        print(f"  Final θ: {theta_final}")
        print(f"  Prediction line: y = {theta_final[0]:.3f} + {theta_final[1]:.3f}x")
        print()
    
    plt.tight_layout()
    plt.show()
    
    # Show best fit with optimal learning rate
    best_alpha = 0.01
    theta_optimal, _ = batch_gradient_descent(theta_init.copy(), X, y, best_alpha, num_iters=1000, verbose=False)
    
    # Plot data and best fit line
    plt.figure(figsize=(10, 8))
    plt.scatter(living_area, price, color='blue', alpha=0.7, s=100, label='Training data')
    
    # Plot best fit line
    x_line = np.linspace(1300, 3100, 100)
    y_line = theta_optimal[0] + theta_optimal[1] * x_line
    plt.plot(x_line, y_line, 'r-', linewidth=3, 
             label=f'Best fit: y = {theta_optimal[0]:.1f} + {theta_optimal[1]:.3f}x')
    
    plt.xlabel('Living area (ft²)')
    plt.ylabel('Price (1000$s)')
    plt.title('Housing Price Prediction with Gradient Descent')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Make predictions
    test_areas = np.array([1800, 2200, 2800])
    test_predictions = theta_optimal[0] + theta_optimal[1] * test_areas
    
    print("Predictions for new houses:")
    print("Living Area | Predicted Price")
    print("-" * 30)
    for area, pred in zip(test_areas, test_predictions):
        print(f"{area:11.0f} | ${pred:12.0f}k")
    print()
    
    print("Key insights:")
    print("- Learning rate affects convergence speed and stability")
    print("- Too small α: slow convergence")
    print("- Too large α: may not converge or oscillate")
    print("- Optimal α depends on the specific problem")
    print("- Gradient descent finds the best linear fit to the data")
    print()

# ============================================================================
# Advanced Topics and Practical Considerations
# ============================================================================

def learning_rate_analysis():
    """
    Analyze the impact of learning rate on convergence.
    This demonstrates the importance of proper learning rate selection.
    """
    print("=== Learning Rate Analysis ===")
    print("Understanding the impact of learning rate on optimization")
    print()
    
    # Generate simple data
    np.random.seed(42)
    X = np.random.randn(50, 2)
    X = np.column_stack([np.ones(50), X])
    theta_true = np.array([1.0, 2.0, -1.0])
    y = X @ theta_true + 0.1 * np.random.randn(50)
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
    theta_init = np.zeros(3)
    
    plt.figure(figsize=(15, 10))
    
    for i, alpha in enumerate(learning_rates):
        try:
            theta_final, cost_history = batch_gradient_descent(
                theta_init.copy(), X, y, alpha, num_iters=100, verbose=False
            )
            
            plt.subplot(2, 3, i+1)
            plt.plot(cost_history, linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Cost J(θ)')
            plt.title(f'α = {alpha}')
            plt.grid(True, alpha=0.3)
            
            # Check if converged
            if len(cost_history) > 1:
                cost_change = abs(cost_history[-1] - cost_history[-2])
                if cost_change < 1e-6:
                    plt.title(f'α = {alpha} (Converged)')
                else:
                    plt.title(f'α = {alpha} (Not Converged)')
            
        except:
            plt.subplot(2, 3, i+1)
            plt.text(0.5, 0.5, f'α = {alpha}\nDiverged', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'α = {alpha} (Diverged)')
    
    plt.tight_layout()
    plt.show()
    
    print("Learning rate guidelines:")
    print("- Too small (α < 0.001): Very slow convergence")
    print("- Good range (0.01 - 0.1): Stable and reasonable speed")
    print("- Too large (α > 1.0): May diverge or oscillate")
    print("- Optimal α depends on data scale and problem")
    print("- Adaptive learning rates can help (not shown here)")
    print()

def feature_scaling_impact():
    """
    Demonstrate the impact of feature scaling on gradient descent.
    This shows why normalization is important for optimization.
    """
    print("=== Feature Scaling Impact ===")
    print("Understanding why feature scaling matters for gradient descent")
    print()
    
    # Generate data with different scales
    np.random.seed(42)
    n_samples = 100
    
    # Features with very different scales
    feature1 = np.random.uniform(0, 1, n_samples)  # Small scale
    feature2 = np.random.uniform(1000, 10000, n_samples)  # Large scale
    
    X_unscaled = np.column_stack([np.ones(n_samples), feature1, feature2])
    theta_true = np.array([1.0, 2.0, 0.001])  # Small weight for large feature
    y = X_unscaled @ theta_true + 0.1 * np.random.randn(n_samples)
    
    # Scaled features
    feature1_scaled = (feature1 - feature1.mean()) / feature1.std()
    feature2_scaled = (feature2 - feature2.mean()) / feature2.std()
    X_scaled = np.column_stack([np.ones(n_samples), feature1_scaled, feature2_scaled])
    
    print("Feature statistics:")
    print(f"Feature 1 (unscaled): mean = {feature1.mean():.3f}, std = {feature1.std():.3f}")
    print(f"Feature 2 (unscaled): mean = {feature2.mean():.1f}, std = {feature2.std():.1f}")
    print(f"Feature 1 (scaled): mean = {feature1_scaled.mean():.3f}, std = {feature1_scaled.std():.3f}")
    print(f"Feature 2 (scaled): mean = {feature2_scaled.mean():.3f}, std = {feature2_scaled.std():.3f}")
    print()
    
    # Compare convergence
    theta_init = np.zeros(3)
    alpha = 0.01
    
    # Unscaled data
    theta_unscaled, cost_unscaled = batch_gradient_descent(
        theta_init.copy(), X_unscaled, y, alpha, num_iters=1000, verbose=False
    )
    
    # Scaled data
    theta_scaled, cost_scaled = batch_gradient_descent(
        theta_init.copy(), X_scaled, y, alpha, num_iters=1000, verbose=False
    )
    
    plt.figure(figsize=(12, 5))
    
    # Plot convergence
    plt.subplot(1, 2, 1)
    plt.plot(cost_unscaled, label='Unscaled features', linewidth=2)
    plt.plot(cost_scaled, label='Scaled features', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(θ)')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot final parameters
    plt.subplot(1, 2, 2)
    x_pos = np.arange(len(theta_true))
    width = 0.35
    
    plt.bar(x_pos - width/2, theta_unscaled, width, label='Unscaled', alpha=0.7)
    plt.bar(x_pos + width/2, theta_scaled, width, label='Scaled', alpha=0.7)
    plt.bar(x_pos + 3*width/2, theta_true, width, label='True θ', alpha=0.7, color='black')
    
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.title('Final Parameter Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Results comparison:")
    print(f"{'Metric':<20} {'Unscaled':<15} {'Scaled':<15}")
    print("-" * 50)
    print(f"{'Final cost':<20} {cost_unscaled[-1]:<15.4f} {cost_scaled[-1]:<15.4f}")
    print(f"{'Convergence speed':<20} {'Slow':<15} {'Fast':<15}")
    print(f"{'Parameter accuracy':<20} {'Poor':<15} {'Good':<15}")
    print()
    
    print("Key insights:")
    print("- Unscaled features lead to slow convergence")
    print("- Scaled features converge much faster")
    print("- Feature scaling is crucial for gradient descent")
    print("- Standardization (z-score) is a common scaling method")
    print("- Normalization to [0,1] is another option")
    print()

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("LMS Algorithm and Gradient Descent Examples")
    print("=" * 50)
    print("This file demonstrates various gradient descent optimization methods")
    print("for linear regression with detailed explanations and comparisons.")
    print()
    
    # Run demonstrations
    demonstrate_single_example_update()
    compare_gradient_descent_methods()
    housing_price_optimization()
    learning_rate_analysis()
    feature_scaling_impact()
    
    print("All examples completed!")
    print("\nKey concepts demonstrated:")
    print("1. LMS update rule: θ := θ + α(y - h_θ(x))x")
    print("2. Batch gradient descent for stable optimization")
    print("3. Stochastic gradient descent for large datasets")
    print("4. Mini-batch gradient descent as a compromise")
    print("5. Learning rate selection and its impact")
    print("6. Feature scaling importance")
    print("7. Real-world applications and practical considerations")
    print("\nNext steps:")
    print("- Experiment with different learning rates")
    print("- Try different batch sizes for mini-batch GD")
    print("- Implement adaptive learning rate methods")
    print("- Explore regularization techniques")
    print("- Apply to other optimization problems") 
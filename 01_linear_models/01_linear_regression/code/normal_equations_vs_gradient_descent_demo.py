import numpy as np
import matplotlib.pyplot as plt
import time

def demonstrate_normal_equations_vs_gradient_descent():
    """Demonstrate the trade-offs between normal equations and gradient descent"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create synthetic data
    X = np.random.randn(n_samples, n_features)
    true_theta = np.random.randn(n_features)
    y = X @ true_theta + np.random.normal(0, 0.1, n_samples)
    
    # Add bias term
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    print("Normal Equations vs. Gradient Descent Comparison")
    print("=" * 60)
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print()
    
    # Method 1: Normal Equations
    print("Method 1: Normal Equations (Analytical)")
    print("-" * 40)
    
    start_time = time.time()
    # Normal equations: θ = (X^T X)^(-1) X^T y
    XtX = X_with_bias.T @ X_with_bias
    Xty = X_with_bias.T @ y
    theta_normal = np.linalg.solve(XtX, Xty)  # More stable than inv()
    normal_time = time.time() - start_time
    
    print(f"Time: {normal_time:.4f} seconds")
    print(f"Memory: {XtX.nbytes / 1024**2:.2f} MB (for X^T X)")
    print(f"Parameters: {len(theta_normal)}")
    print(f"Matrix size: {XtX.shape}")
    print()
    
    # Method 2: Gradient Descent
    print("Method 2: Gradient Descent (Iterative)")
    print("-" * 40)
    
    def gradient_descent(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        n_samples, n_features = X.shape
        theta = np.zeros(n_features)
        costs = []
        
        start_time = time.time()
        
        for i in range(max_iterations):
            # Compute predictions
            predictions = X @ theta
            
            # Compute gradient
            gradient = X.T @ (predictions - y) / n_samples
            
            # Update parameters
            theta -= learning_rate * gradient
            
            # Compute cost
            cost = np.mean((predictions - y)**2) / 2
            costs.append(cost)
            
            # Check convergence
            if i > 0 and abs(costs[-1] - costs[-2]) < tolerance:
                break
        
        gradient_time = time.time() - start_time
        return theta, costs, gradient_time
    
    theta_gradient, costs, gradient_time = gradient_descent(X_with_bias, y)
    
    print(f"Time: {gradient_time:.4f} seconds")
    print(f"Memory: {X_with_bias.nbytes / 1024**2:.2f} MB")
    print(f"Iterations: {len(costs)}")
    print(f"Final Cost: {costs[-1]:.6f}")
    print(f"Learning Rate: 0.01")
    print()
    
    # Compare results
    print("Comparison:")
    print("-" * 40)
    print(f"Normal Equations Time: {normal_time:.4f}s")
    print(f"Gradient Descent Time: {gradient_time:.4f}s")
    print(f"Speedup: {gradient_time/normal_time:.1f}x")
    print()
    
    # Check accuracy
    predictions_normal = X_with_bias @ theta_normal
    predictions_gradient = X_with_bias @ theta_gradient
    
    mse_normal = np.mean((predictions_normal - y)**2)
    mse_gradient = np.mean((predictions_gradient - y)**2)
    
    print(f"Normal Equations MSE: {mse_normal:.6f}")
    print(f"Gradient Descent MSE: {mse_gradient:.6f}")
    print(f"Difference: {abs(mse_normal - mse_gradient):.2e}")
    print()
    
    # Check parameter differences
    param_diff = np.linalg.norm(theta_normal - theta_gradient)
    print(f"Parameter Difference: {param_diff:.6f}")
    print(f"Solutions are {'very similar' if param_diff < 1e-3 else 'different'}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Cost convergence for gradient descent
    plt.subplot(1, 3, 1)
    plt.plot(costs, 'b-', linewidth=2)
    plt.axhline(y=mse_normal, color='r', linestyle='--', label='Normal Equations Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Time comparison
    plt.subplot(1, 3, 2)
    methods = ['Normal Equations', 'Gradient Descent']
    times = [normal_time, gradient_time]
    colors = ['red', 'blue']
    
    bars = plt.bar(methods, times, color=colors, alpha=0.7)
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    # Memory comparison
    plt.subplot(1, 3, 3)
    memory_normal = XtX.nbytes / 1024**2
    memory_gradient = X_with_bias.nbytes / 1024**2
    memories = [memory_normal, memory_gradient]
    
    bars = plt.bar(methods, memories, color=colors, alpha=0.7)
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, memory_val in zip(bars, memories):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{memory_val:.1f}MB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Normal equations give exact solution in one step")
    print("2. Gradient descent requires iteration but uses less memory")
    print("3. Both methods give very similar results")
    print("4. Choice depends on dataset size and computational constraints")
    print("5. Normal equations fail when X^T X is not invertible")
    
    return theta_normal, theta_gradient, costs, normal_time, gradient_time

if __name__ == "__main__":
    comparison_demo = demonstrate_normal_equations_vs_gradient_descent()

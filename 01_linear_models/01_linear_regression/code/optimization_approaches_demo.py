import numpy as np
import matplotlib.pyplot as plt
import time

def demonstrate_optimization_approaches():
    """Demonstrate why iterative methods are needed for large-scale optimization"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # Create synthetic data
    X = np.random.randn(n_samples, n_features)
    true_theta = np.random.randn(n_features)
    y = X @ true_theta + np.random.normal(0, 0.1, n_samples)
    
    print("Optimization Approaches Comparison")
    print("=" * 50)
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print()
    
    # Method 1: Normal Equations (Analytical)
    print("Method 1: Normal Equations (Analytical)")
    print("-" * 40)
    
    start_time = time.time()
    # Normal equations: θ = (X^T X)^(-1) X^T y
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    theta_analytical = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    analytical_time = time.time() - start_time
    
    print(f"Time: {analytical_time:.4f} seconds")
    print(f"Memory: {X_with_bias.nbytes / 1024**2:.2f} MB")
    print(f"Parameters: {len(theta_analytical)}")
    print()
    
    # Method 2: Gradient Descent (Iterative)
    print("Method 2: Gradient Descent (Iterative)")
    print("-" * 40)
    
    def gradient_descent(X, y, learning_rate=0.01, max_iterations=1000):
        n_samples, n_features = X.shape
        theta = np.zeros(n_features + 1)  # +1 for bias
        X_with_bias = np.column_stack([np.ones(n_samples), X])
        
        costs = []
        start_time = time.time()
        
        for i in range(max_iterations):
            # Compute predictions
            predictions = X_with_bias @ theta
            
            # Compute gradient
            gradient = X_with_bias.T @ (predictions - y) / n_samples
            
            # Update parameters
            theta -= learning_rate * gradient
            
            # Compute cost
            cost = np.mean((predictions - y)**2) / 2
            costs.append(cost)
            
            # Check convergence
            if i > 0 and abs(costs[-1] - costs[-2]) < 1e-6:
                break
        
        gradient_time = time.time() - start_time
        return theta, costs, gradient_time
    
    theta_gradient, costs, gradient_time = gradient_descent(X, y)
    
    print(f"Time: {gradient_time:.4f} seconds")
    print(f"Memory: {X.nbytes / 1024**2:.2f} MB")
    print(f"Iterations: {len(costs)}")
    print(f"Final Cost: {costs[-1]:.6f}")
    print()
    
    # Compare results
    print("Comparison:")
    print("-" * 40)
    print(f"Analytical Time: {analytical_time:.4f}s")
    print(f"Gradient Time: {gradient_time:.4f}s")
    print(f"Speedup: {analytical_time/gradient_time:.1f}x")
    print()
    
    # Check accuracy
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    predictions_analytical = X_with_bias @ theta_analytical
    predictions_gradient = X_with_bias @ theta_gradient
    
    mse_analytical = np.mean((predictions_analytical - y)**2)
    mse_gradient = np.mean((predictions_gradient - y)**2)
    
    print(f"Analytical MSE: {mse_analytical:.6f}")
    print(f"Gradient MSE: {mse_gradient:.6f}")
    print(f"Difference: {abs(mse_analytical - mse_gradient):.2e}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Cost convergence
    plt.subplot(1, 3, 1)
    plt.plot(costs, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Convergence')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Time comparison
    plt.subplot(1, 3, 2)
    methods = ['Analytical', 'Gradient Descent']
    times = [analytical_time, gradient_time]
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
    memory_analytical = X_with_bias.nbytes / 1024**2
    memory_gradient = X.nbytes / 1024**2
    memories = [memory_analytical, memory_gradient]
    
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
    
    return theta_analytical, theta_gradient, costs, analytical_time, gradient_time

if __name__ == "__main__":
    optimization_demo = demonstrate_optimization_approaches()

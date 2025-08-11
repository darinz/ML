import numpy as np
import matplotlib.pyplot as plt

def demonstrate_learning_rate_effects():
    """Demonstrate the effects of different learning rates"""
    
    # Simple cost function: J(θ) = θ²
    def cost_function(theta):
        return theta**2
    
    def gradient(theta):
        return 2*theta
    
    def gradient_descent_1d(initial_theta, learning_rate, max_iterations=100):
        """Perform gradient descent in 1D"""
        theta = initial_theta
        path = [theta]
        costs = [cost_function(theta)]
        
        for i in range(max_iterations):
            # Compute gradient
            grad = gradient(theta)
            
            # Update parameter
            theta = theta - learning_rate * grad
            
            # Record path and cost
            path.append(theta)
            costs.append(cost_function(theta))
            
            # Check convergence
            if abs(theta) < 1e-6:
                break
        
        return np.array(path), np.array(costs)
    
    # Test different learning rates
    initial_theta = 2.0
    learning_rates = [0.01, 0.1, 0.5, 0.9, 1.1]
    
    print("Learning Rate Effects on Gradient Descent")
    print("=" * 50)
    print(f"Cost function: J(θ) = θ²")
    print(f"Initial θ: {initial_theta}")
    print(f"True minimum: θ = 0")
    print()
    
    # Run gradient descent with different learning rates
    results = {}
    for lr in learning_rates:
        path, costs = gradient_descent_1d(initial_theta, lr)
        results[lr] = {'path': path, 'costs': costs, 'iterations': len(path)}
        print(f"Learning rate {lr}:")
        print(f"  Final θ: {path[-1]:.6f}")
        print(f"  Final cost: {costs[-1]:.6f}")
        print(f"  Iterations: {len(path)}")
        print(f"  Converged: {'Yes' if abs(path[-1]) < 1e-3 else 'No'}")
        print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Parameter convergence
    plt.subplot(1, 3, 1)
    for lr in learning_rates:
        path = results[lr]['path']
        plt.plot(path, linewidth=2, label=f'LR: {lr}')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='True minimum')
    plt.xlabel('Iteration')
    plt.ylabel('θ')
    plt.title('Parameter Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cost convergence
    plt.subplot(1, 3, 2)
    for lr in learning_rates:
        costs = results[lr]['costs']
        plt.plot(costs, linewidth=2, label=f'LR: {lr}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Learning rate vs convergence speed
    plt.subplot(1, 3, 3)
    lr_values = list(results.keys())
    iterations = [results[lr]['iterations'] for lr in lr_values]
    converged = [1 if results[lr]['iterations'] < 100 else 0 for lr in lr_values]
    
    colors = ['green' if c else 'red' for c in converged]
    plt.bar(range(len(lr_values)), iterations, color=colors, alpha=0.7)
    plt.xlabel('Learning Rate')
    plt.ylabel('Iterations to Converge')
    plt.title('Convergence Speed')
    plt.xticks(range(len(lr_values)), lr_values)
    plt.grid(True, alpha=0.3)
    
    # Add labels
    for i, (lr, iters) in enumerate(zip(lr_values, iterations)):
        plt.text(i, iters + 1, f'{iters}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Learning Rate Analysis:")
    print("-" * 30)
    print("Too Small (0.01): Slow but stable convergence")
    print("Good (0.1): Fast and stable convergence")
    print("Optimal (0.5): Very fast convergence")
    print("Large (0.9): Fast but may oscillate")
    print("Too Large (1.1): Diverges (overshoots minimum)")
    print()
    print("Key Insights:")
    print("1. Learning rate controls convergence speed")
    print("2. Too small = slow convergence")
    print("3. Too large = instability or divergence")
    print("4. Optimal rate depends on the problem")
    print("5. Adaptive learning rates can help")
    
    return results

if __name__ == "__main__":
    learning_rate_demo = demonstrate_learning_rate_effects()

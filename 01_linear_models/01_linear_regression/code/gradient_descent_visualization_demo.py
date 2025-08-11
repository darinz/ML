import numpy as np
import matplotlib.pyplot as plt

def demonstrate_gradient_descent_visualization():
    """Demonstrate gradient descent with a simple 2D example"""
    
    # Define a simple cost function: J(θ) = (θ₁ - 2)² + (θ₂ - 3)²
    def cost_function(theta1, theta2):
        return (theta1 - 2)**2 + (theta2 - 3)**2
    
    def gradient(theta1, theta2):
        """Compute gradient of the cost function"""
        return np.array([2*(theta1 - 2), 2*(theta2 - 3)])
    
    def gradient_descent_2d(initial_theta, learning_rate=0.1, max_iterations=50):
        """Perform gradient descent in 2D"""
        theta = np.array(initial_theta)
        path = [theta.copy()]
        costs = [cost_function(theta[0], theta[1])]
        
        for i in range(max_iterations):
            # Compute gradient
            grad = gradient(theta[0], theta[1])
            
            # Update parameters
            theta = theta - learning_rate * grad
            
            # Record path and cost
            path.append(theta.copy())
            costs.append(cost_function(theta[0], theta[1]))
            
            # Check convergence
            if costs[-1] < 1e-6:
                break
        
        return np.array(path), costs
    
    # Run gradient descent from different starting points
    starting_points = [
        [0, 0],    # Bottom-left
        [5, 5],    # Top-right
        [-2, 4],   # Top-left
        [4, -1]    # Bottom-right
    ]
    
    learning_rates = [0.05, 0.1, 0.2, 0.5]
    
    print("Gradient Descent Visualization")
    print("=" * 40)
    print("Cost function: J(θ₁, θ₂) = (θ₁ - 2)² + (θ₂ - 3)²")
    print("Minimum at: θ₁ = 2, θ₂ = 3")
    print()
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Create grid for contour plot
    theta1_range = np.linspace(-3, 7, 100)
    theta2_range = np.linspace(-2, 8, 100)
    theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)
    cost_grid = cost_function(theta1_grid, theta2_grid)
    
    # Plot 1: Contour plot with gradient descent paths
    plt.subplot(2, 3, 1)
    plt.contour(theta1_grid, theta2_grid, cost_grid, levels=20, alpha=0.6)
    plt.colorbar(label='Cost')
    
    # Plot gradient descent paths for different starting points
    colors = ['red', 'blue', 'green', 'purple']
    for i, start_point in enumerate(starting_points):
        path, costs = gradient_descent_2d(start_point, learning_rate=0.1)
        plt.plot(path[:, 0], path[:, 1], 'o-', color=colors[i], 
                label=f'Start: ({start_point[0]}, {start_point[1]})', linewidth=2, markersize=4)
    
    plt.plot(2, 3, 'k*', markersize=15, label='Minimum (2, 3)')
    plt.xlabel('θ₁')
    plt.ylabel('θ₂')
    plt.title('Gradient Descent Paths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Cost convergence for different starting points
    plt.subplot(2, 3, 2)
    for i, start_point in enumerate(starting_points):
        path, costs = gradient_descent_2d(start_point, learning_rate=0.1)
        plt.plot(costs, color=colors[i], linewidth=2, 
                label=f'Start: ({start_point[0]}, {start_point[1]})')
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Effect of learning rate
    plt.subplot(2, 3, 3)
    start_point = [0, 0]
    for i, lr in enumerate(learning_rates):
        path, costs = gradient_descent_2d(start_point, learning_rate=lr)
        plt.plot(costs, linewidth=2, label=f'Learning Rate: {lr}')
    
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Effect of Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 4: Gradient vectors
    plt.subplot(2, 3, 4)
    plt.contour(theta1_grid, theta2_grid, cost_grid, levels=20, alpha=0.6)
    
    # Plot gradient vectors at sample points
    sample_points = np.array([[0, 0], [2, 0], [4, 0], [0, 2], [2, 2], [4, 2]])
    for point in sample_points:
        grad = gradient(point[0], point[1])
        # Normalize gradient for visualization
        grad_norm = grad / np.linalg.norm(grad) * 0.5
        plt.arrow(point[0], point[1], -grad_norm[0], -grad_norm[1], 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
    
    plt.plot(2, 3, 'k*', markersize=15, label='Minimum')
    plt.xlabel('θ₁')
    plt.ylabel('θ₂')
    plt.title('Gradient Vectors\n(Pointing Uphill)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Learning rate comparison - paths
    plt.subplot(2, 3, 5)
    plt.contour(theta1_grid, theta2_grid, cost_grid, levels=20, alpha=0.6)
    
    for i, lr in enumerate(learning_rates):
        path, costs = gradient_descent_2d(start_point, learning_rate=lr)
        plt.plot(path[:, 0], path[:, 1], 'o-', linewidth=2, markersize=3,
                label=f'LR: {lr}')
    
    plt.plot(2, 3, 'k*', markersize=15, label='Minimum')
    plt.xlabel('θ₁')
    plt.ylabel('θ₂')
    plt.title('Learning Rate Effect on Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: 3D surface plot
    ax = plt.subplot(2, 3, 6, projection='3d')
    surf = ax.plot_surface(theta1_grid, theta2_grid, cost_grid, 
                          cmap='viridis', alpha=0.8)
    
    # Plot gradient descent path on surface
    path, costs = gradient_descent_2d([0, 0], learning_rate=0.1)
    path_costs = [cost_function(p[0], p[1]) for p in path]
    ax.plot(path[:, 0], path[:, 1], path_costs, 'r-o', linewidth=3, markersize=4)
    
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_zlabel('Cost')
    ax.set_title('3D Cost Surface')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("1. All paths converge to the minimum regardless of starting point")
    print("2. Learning rate affects convergence speed and stability")
    print("3. Gradient always points uphill (opposite to descent direction)")
    print("4. Path follows the steepest descent direction")
    print("5. Cost decreases monotonically (with appropriate learning rate)")
    
    return starting_points, learning_rates

if __name__ == "__main__":
    gradient_demo = demonstrate_gradient_descent_visualization()

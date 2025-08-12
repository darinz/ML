"""
MSE Calculation Demonstration

This module demonstrates MSE calculation step by step with a simple example
showing how different parameter values affect the loss.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_mse_calculation():
    """Demonstrate MSE calculation step by step"""
    
    # Simple example
    x_data = np.array([1, 2])
    y_true = np.array([2, 4])
    
    # Model: h(x) = 2x
    def model(x, theta):
        return theta * x
    
    # Try different parameters
    thetas = [1.5, 2.0, 2.5]
    
    print("MSE Calculation Example:")
    print("Data: x = [1, 2], y = [2, 4]")
    print("Model: h(x) = θ * x")
    print()
    
    for theta in thetas:
        predictions = model(x_data, theta)
        errors = predictions - y_true
        squared_errors = errors**2
        mse = np.mean(squared_errors)
        
        print(f"θ = {theta}:")
        print(f"  Predictions: h(1) = {theta*1}, h(2) = {theta*2}")
        print(f"  Errors: {errors[0]:.1f}, {errors[1]:.1f}")
        print(f"  Squared Errors: {squared_errors[0]:.2f}, {squared_errors[1]:.2f}")
        print(f"  MSE = {mse:.2f}")
        print()
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    x_plot = np.linspace(0, 3, 100)
    for theta in thetas:
        y_plot = model(x_plot, theta)
        plt.plot(x_plot, y_plot, label=f'θ = {theta}', linewidth=2)
    
    plt.scatter(x_data, y_true, color='red', s=100, zorder=5, label='Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    mse_values = []
    for theta in thetas:
        predictions = model(x_data, theta)
        mse = np.mean((predictions - y_true)**2)
        mse_values.append(mse)
    
    plt.plot(thetas, mse_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('θ')
    plt.ylabel('MSE')
    plt.title('MSE vs θ')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return thetas, mse_values


if __name__ == "__main__":
    theta_demo, mse_demo = demonstrate_mse_calculation()

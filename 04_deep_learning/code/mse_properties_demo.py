"""
Mean Squared Error (MSE) Properties Demonstration

This module demonstrates the properties of MSE loss function and compares it with
other loss functions like Mean Absolute Error (MAE).
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_mse_properties():
    """Demonstrate the properties of MSE loss"""
    
    # Generate sample data
    np.random.seed(42)
    true_values = np.array([10, 20, 30, 40, 50])
    predictions = np.array([12, 18, 32, 35, 55])  # Some predictions are off
    
    # Calculate different loss functions
    mse_loss = np.mean((predictions - true_values)**2)
    mae_loss = np.mean(np.abs(predictions - true_values))
    
    print(f"True values: {true_values}")
    print(f"Predictions: {predictions}")
    print(f"Errors: {predictions - true_values}")
    print(f"MSE Loss: {mse_loss:.2f}")
    print(f"MAE Loss: {mae_loss:.2f}")
    
    # Show how different errors contribute
    errors = predictions - true_values
    squared_errors = errors**2
    abs_errors = np.abs(errors)
    
    print(f"\nError Analysis:")
    print(f"Error\t\tSquared Error\tAbs Error")
    print("-" * 40)
    for i in range(len(errors)):
        print(f"{errors[i]:6.1f}\t\t{squared_errors[i]:8.1f}\t\t{abs_errors[i]:8.1f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(true_values, predictions, alpha=0.7, s=100)
    plt.plot([0, 60], [0, 60], 'r--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(range(len(errors)), errors, alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Data Point')
    plt.ylabel('Error (Prediction - True)')
    plt.title('Individual Errors')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.bar(range(len(errors)), squared_errors, alpha=0.7, color='orange')
    plt.xlabel('Data Point')
    plt.ylabel('Squared Error')
    plt.title('Squared Errors (MSE Components)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return mse_loss, mae_loss


if __name__ == "__main__":
    mse_demo, mae_demo = demonstrate_mse_properties()

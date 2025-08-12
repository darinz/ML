"""
Loss Functions Comparison

This module compares different regression loss functions including MSE, MAE, and Huber loss
on data with outliers to demonstrate their robustness properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor


def demonstrate_loss_functions():
    """Compare different regression loss functions"""
    
    # Generate data with outliers
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y_true = 2 * x + 1 + 0.5 * np.random.randn(100)
    
    # Add some outliers
    y_true[20] += 10  # Outlier 1
    y_true[80] -= 8   # Outlier 2
    
    # Fit models with different loss functions
    # Linear regression (uses MSE)
    lr_mse = LinearRegression()
    lr_mse.fit(x.reshape(-1, 1), y_true)
    y_pred_mse = lr_mse.predict(x.reshape(-1, 1))
    
    # Huber regression
    lr_huber = HuberRegressor(epsilon=1.35)  # Default epsilon
    lr_huber.fit(x.reshape(-1, 1), y_true)
    y_pred_huber = lr_huber.predict(x.reshape(-1, 1))
    
    # Calculate losses
    mse_mse = np.mean((y_pred_mse - y_true)**2)
    mae_mse = np.mean(np.abs(y_pred_mse - y_true))
    mse_huber = np.mean((y_pred_huber - y_true)**2)
    mae_huber = np.mean(np.abs(y_pred_huber - y_true))
    
    print(f"Loss Comparison:")
    print(f"Model\t\tMSE\t\tMAE")
    print("-" * 30)
    print(f"MSE Model\t{mse_mse:.3f}\t\t{mae_mse:.3f}")
    print(f"Huber Model\t{mse_huber:.3f}\t\t{mae_huber:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(x, y_true, alpha=0.6, s=20, label='Data')
    plt.plot(x, y_pred_mse, 'r-', linewidth=2, label='MSE Model')
    plt.plot(x, y_pred_huber, 'g-', linewidth=2, label='Huber Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Model Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    errors_mse = y_pred_mse - y_true
    errors_huber = y_pred_huber - y_true
    plt.hist(errors_mse, bins=20, alpha=0.7, label='MSE Errors', density=True)
    plt.hist(errors_huber, bins=20, alpha=0.7, label='Huber Errors', density=True)
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.title('Error Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(y_true, errors_mse, alpha=0.6, s=20, label='MSE Errors')
    plt.scatter(y_true, errors_huber, alpha=0.6, s=20, label='Huber Errors')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Prediction Errors')
    plt.title('Errors vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return mse_mse, mae_mse, mse_huber, mae_huber


if __name__ == "__main__":
    loss_comparison = demonstrate_loss_functions()

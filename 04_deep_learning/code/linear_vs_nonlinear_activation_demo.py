"""
Linear vs Non-linear Activation Demonstration

This module demonstrates why non-linear activation functions are necessary by comparing
linear and non-linear models on non-linear data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


def demonstrate_linear_vs_nonlinear():
    """Demonstrate why non-linear activation functions are necessary"""
    
    # Generate non-linear data
    np.random.seed(42)
    x = np.linspace(-3, 3, 100)
    y_true = np.sin(x) + 0.3 * np.random.randn(100)
    
    # Try to fit with linear and non-linear models
    
    # Linear model
    linear_model = LinearRegression()
    linear_model.fit(x.reshape(-1, 1), y_true)
    y_linear = linear_model.predict(x.reshape(-1, 1))
    
    # Non-linear model (neural network with ReLU)
    nn_model = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', 
                           random_state=42, max_iter=1000)
    nn_model.fit(x.reshape(-1, 1), y_true)
    y_nn = nn_model.predict(x.reshape(-1, 1))
    
    # Calculate errors
    linear_error = np.mean((y_true - y_linear)**2)
    nn_error = np.mean((y_true - y_nn)**2)
    
    print(f"Fitting Non-linear Data:")
    print(f"Linear Model MSE: {linear_error:.4f}")
    print(f"Neural Network MSE: {nn_error:.4f}")
    print(f"Improvement: {linear_error/nn_error:.1f}x better")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(x, y_true, alpha=0.6, s=20, label='Data')
    plt.plot(x, y_linear, 'r-', linewidth=2, label='Linear Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Model vs Non-linear Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.scatter(x, y_true, alpha=0.6, s=20, label='Data')
    plt.plot(x, y_nn, 'g-', linewidth=2, label='Neural Network')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Neural Network vs Non-linear Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(x, y_true, 'b-', alpha=0.7, label='True Function')
    plt.plot(x, y_linear, 'r--', linewidth=2, label='Linear Model')
    plt.plot(x, y_nn, 'g--', linewidth=2, label='Neural Network')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return linear_error, nn_error


if __name__ == "__main__":
    linear_vs_nonlinear = demonstrate_linear_vs_nonlinear()

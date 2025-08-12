"""
Activation Functions Comparison

This module compares different activation functions (ReLU, Sigmoid, Tanh) and their
derivatives, showing their properties and characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_activation_functions():
    """Compare different activation functions"""
    
    # Generate data
    z = np.linspace(-5, 5, 1000)
    
    # Calculate different activation functions
    relu = np.maximum(0, z)
    sigmoid = 1 / (1 + np.exp(-z))
    tanh = np.tanh(z)
    
    # Calculate derivatives
    relu_deriv = np.where(z > 0, 1, 0)
    sigmoid_deriv = sigmoid * (1 - sigmoid)
    tanh_deriv = 1 - tanh**2
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Activation functions
    plt.subplot(2, 3, 1)
    plt.plot(z, relu, 'b-', linewidth=2, label='ReLU')
    plt.title('ReLU: max(0, z)')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(z, sigmoid, 'r-', linewidth=2, label='Sigmoid')
    plt.title('Sigmoid: 1/(1 + e^(-z))')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.plot(z, tanh, 'g-', linewidth=2, label='Tanh')
    plt.title('Tanh: (e^z - e^(-z))/(e^z + e^(-z))')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Derivatives
    plt.subplot(2, 3, 4)
    plt.plot(z, relu_deriv, 'b-', linewidth=2, label='ReLU Derivative')
    plt.title('ReLU Derivative')
    plt.xlabel('z')
    plt.ylabel('σ\'(z)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.plot(z, sigmoid_deriv, 'r-', linewidth=2, label='Sigmoid Derivative')
    plt.title('Sigmoid Derivative')
    plt.xlabel('z')
    plt.ylabel('σ\'(z)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 6)
    plt.plot(z, tanh_deriv, 'g-', linewidth=2, label='Tanh Derivative')
    plt.title('Tanh Derivative')
    plt.xlabel('z')
    plt.ylabel('σ\'(z)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show key properties
    print("Activation Function Properties:")
    print("ReLU:")
    print("  - Range: [0, ∞)")
    print("  - Pros: Simple, efficient, no vanishing gradient")
    print("  - Cons: Can 'die' (get stuck at 0)")
    print()
    print("Sigmoid:")
    print("  - Range: (0, 1)")
    print("  - Pros: Smooth, interpretable as probability")
    print("  - Cons: Vanishing gradient problem")
    print()
    print("Tanh:")
    print("  - Range: (-1, 1)")
    print("  - Pros: Zero-centered, bounded")
    print("  - Cons: Still has vanishing gradient")
    
    return z, relu, sigmoid, tanh


if __name__ == "__main__":
    activation_demo = demonstrate_activation_functions()

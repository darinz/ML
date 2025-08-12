"""
Advanced Activation Functions Demonstration

This module demonstrates different activation functions including ReLU, Sigmoid, Tanh, and GELU,
showing their properties and derivatives.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_activation_functions():
    """Demonstrate different activation functions"""
    
    # Generate input data
    z = np.linspace(-5, 5, 1000)
    
    # Define activation functions
    def relu(x):
        return np.maximum(0, x)
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def tanh(x):
        return np.tanh(x)
    
    def gelu(x):
        return x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    # Calculate activations
    relu_output = relu(z)
    sigmoid_output = sigmoid(z)
    tanh_output = tanh(z)
    gelu_output = gelu(z)
    
    # Calculate derivatives
    relu_deriv = np.where(z > 0, 1, 0)
    sigmoid_deriv = sigmoid_output * (1 - sigmoid_output)
    tanh_deriv = 1 - tanh_output**2
    gelu_deriv = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3))) + \
                 0.5 * z * (1 - np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3))**2) * \
                 np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * z**2)
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Activation functions
    plt.subplot(2, 4, 1)
    plt.plot(z, relu_output, 'b-', linewidth=2)
    plt.title('ReLU: max(0, z)')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 2)
    plt.plot(z, sigmoid_output, 'r-', linewidth=2)
    plt.title('Sigmoid: 1/(1 + e^(-z))')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 3)
    plt.plot(z, tanh_output, 'g-', linewidth=2)
    plt.title('Tanh: (e^z - e^(-z))/(e^z + e^(-z))')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 4)
    plt.plot(z, gelu_output, 'm-', linewidth=2)
    plt.title('GELU: z * Φ(z)')
    plt.xlabel('z')
    plt.ylabel('σ(z)')
    plt.grid(True, alpha=0.3)
    
    # Derivatives
    plt.subplot(2, 4, 5)
    plt.plot(z, relu_deriv, 'b-', linewidth=2)
    plt.title('ReLU Derivative')
    plt.xlabel('z')
    plt.ylabel('σ\'(z)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 6)
    plt.plot(z, sigmoid_deriv, 'r-', linewidth=2)
    plt.title('Sigmoid Derivative')
    plt.xlabel('z')
    plt.ylabel('σ\'(z)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 7)
    plt.plot(z, tanh_deriv, 'g-', linewidth=2)
    plt.title('Tanh Derivative')
    plt.xlabel('z')
    plt.ylabel('σ\'(z)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 4, 8)
    plt.plot(z, gelu_deriv, 'm-', linewidth=2)
    plt.title('GELU Derivative')
    plt.xlabel('z')
    plt.ylabel('σ\'(z)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show practical example
    print("Activation Function Properties:")
    print("ReLU:")
    print("  - Range: [0, ∞)")
    print("  - Pros: Simple, efficient, no vanishing gradient")
    print("  - Cons: Can 'die' (get stuck at 0)")
    print("  - Use case: Hidden layers in most networks")
    print()
    print("Sigmoid:")
    print("  - Range: (0, 1)")
    print("  - Pros: Smooth, interpretable as probability")
    print("  - Cons: Vanishing gradient problem")
    print("  - Use case: Output layer for binary classification")
    print()
    print("Tanh:")
    print("  - Range: (-1, 1)")
    print("  - Pros: Zero-centered, bounded")
    print("  - Cons: Still has vanishing gradient")
    print("  - Use case: Hidden layers when zero-centered output is desired")
    print()
    print("GELU:")
    print("  - Range: (-∞, ∞)")
    print("  - Pros: Smooth, often performs better than ReLU")
    print("  - Cons: More computationally expensive")
    print("  - Use case: Transformer architectures")
    
    return z, relu_output, sigmoid_output, tanh_output, gelu_output


if __name__ == "__main__":
    activation_demo = demonstrate_activation_functions()

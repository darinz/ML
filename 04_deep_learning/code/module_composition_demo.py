"""
Module Composition Demonstration

This module demonstrates how different modules can be composed to create complex
neural network architectures with different computational graphs.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_module_composition():
    """Demonstrate how modules can be composed"""
    
    # Define simple modules
    def linear_module(x, W, b):
        """Linear transformation module"""
        return np.dot(x, W.T) + b
    
    def relu_module(x):
        """ReLU activation module"""
        return np.maximum(0, x)
    
    def sigmoid_module(x):
        """Sigmoid activation module"""
        return 1 / (1 + np.exp(-x))
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 2)  # 100 samples, 2 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple classification task
    
    # Define different compositions
    def composition_1(x, W1, b1, W2, b2):
        """Simple composition: Linear → ReLU → Linear → Sigmoid"""
        h1 = linear_module(x, W1, b1)      # Module 1: Linear
        h2 = relu_module(h1)               # Module 2: ReLU
        h3 = linear_module(h2, W2, b2)     # Module 3: Linear
        output = sigmoid_module(h3)        # Module 4: Sigmoid
        return output, [h1, h2, h3, output]
    
    def composition_2(x, W1, b1, W2, b2, W3, b3):
        """Deep composition: Linear → ReLU → Linear → ReLU → Linear → Sigmoid"""
        h1 = linear_module(x, W1, b1)      # Module 1: Linear
        h2 = relu_module(h1)               # Module 2: ReLU
        h3 = linear_module(h2, W2, b2)     # Module 3: Linear
        h4 = relu_module(h3)               # Module 4: ReLU
        h5 = linear_module(h4, W3, b3)     # Module 5: Linear
        output = sigmoid_module(h5)        # Module 6: Sigmoid
        return output, [h1, h2, h3, h4, h5, output]
    
    # Initialize weights randomly
    W1 = np.random.randn(5, 2) * 0.1
    b1 = np.zeros(5)
    W2 = np.random.randn(3, 5) * 0.1
    b2 = np.zeros(3)
    W3 = np.random.randn(1, 3) * 0.1
    b3 = np.zeros(1)
    
    # Test compositions
    output1, activations1 = composition_1(X[:5], W1, b1, W3, b3)
    output2, activations2 = composition_2(X[:5], W1, b1, W2, b2, W3, b3)
    
    print("Module Composition Example:")
    print("Input shape:", X[:5].shape)
    print()
    print("Composition 1 (Linear → ReLU → Linear → Sigmoid):")
    for i, (name, activation) in enumerate(zip(['Linear1', 'ReLU1', 'Linear2', 'Sigmoid'], activations1)):
        print(f"  {name}: shape {activation.shape}, range [{activation.min():.3f}, {activation.max():.3f}]")
    print()
    print("Composition 2 (Linear → ReLU → Linear → ReLU → Linear → Sigmoid):")
    for i, (name, activation) in enumerate(zip(['Linear1', 'ReLU1', 'Linear2', 'ReLU2', 'Linear3', 'Sigmoid'], activations2)):
        print(f"  {name}: shape {activation.shape}, range [{activation.min():.3f}, {activation.max():.3f}]")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Show computational graphs
    plt.subplot(1, 3, 1)
    plt.text(0.5, 0.5, 'Composition 1:\nInput → Linear → ReLU → Linear → Sigmoid', 
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Computational Graph 1')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.text(0.5, 0.5, 'Composition 2:\nInput → Linear → ReLU → Linear → ReLU → Linear → Sigmoid', 
             ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
    plt.title('Computational Graph 2')
    plt.axis('off')
    
    # Show activation ranges
    plt.subplot(1, 3, 3)
    names1 = ['Linear1', 'ReLU1', 'Linear2', 'Sigmoid']
    ranges1 = [activations1[i].max() - activations1[i].min() for i in range(len(activations1))]
    
    names2 = ['Linear1', 'ReLU1', 'Linear2', 'ReLU2', 'Linear3', 'Sigmoid']
    ranges2 = [activations2[i].max() - activations2[i].min() for i in range(len(activations2))]
    
    x1 = np.arange(len(names1))
    x2 = np.arange(len(names2))
    
    plt.bar(x1 - 0.2, ranges1, 0.4, alpha=0.7, label='Composition 1')
    plt.bar(x2 + 0.2, ranges2, 0.4, alpha=0.7, label='Composition 2')
    plt.xlabel('Layer')
    plt.ylabel('Activation Range')
    plt.title('Activation Ranges by Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return output1, output2, activations1, activations2


if __name__ == "__main__":
    composition_demo = demonstrate_module_composition()

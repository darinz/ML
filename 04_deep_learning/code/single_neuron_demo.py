"""
Single Neuron Demonstration

This module demonstrates how a single neuron works with different activation functions
and shows the computation process for house price prediction.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_single_neuron():
    """Demonstrate how a single neuron works"""
    
    # Define a simple neuron
    def neuron(x, w, b, activation='relu'):
        # Linear combination
        z = np.dot(w, x) + b
        
        # Non-linear activation
        if activation == 'relu':
            a = np.maximum(0, z)
        elif activation == 'sigmoid':
            a = 1 / (1 + np.exp(-z))
        elif activation == 'tanh':
            a = np.tanh(z)
        else:
            a = z  # Linear
            
        return z, a
    
    # Example: House price prediction
    # Features: [square_feet, bedrooms, age]
    x = np.array([2000, 3, 10])  # 2000 sq ft, 3 bedrooms, 10 years old
    
    # Different weight configurations
    configurations = [
        {'w': np.array([100, 5000, -1000]), 'b': 50000, 'name': 'Price-focused'},
        {'w': np.array([50, 3000, -500]), 'b': 100000, 'name': 'Luxury-focused'},
        {'w': np.array([75, 4000, -750]), 'b': 75000, 'name': 'Balanced'}
    ]
    
    print("Single Neuron: House Price Prediction")
    print("Input features: [square_feet, bedrooms, age]")
    print("Input values:", x)
    print()
    
    for config in configurations:
        z, a = neuron(x, config['w'], config['b'], 'relu')
        
        print(f"{config['name']} Neuron:")
        print(f"  Weights: {config['w']}")
        print(f"  Bias: {config['b']}")
        print(f"  Linear combination (z): {z:.0f}")
        print(f"  Activation (a): {a:.0f}")
        print(f"  Interpretation: ${a:,.0f} predicted price")
        print()
    
    # Visualization of different activation functions
    z_range = np.linspace(-5, 5, 1000)
    
    plt.figure(figsize=(15, 5))
    
    # ReLU
    plt.subplot(1, 3, 1)
    a_relu = np.maximum(0, z_range)
    plt.plot(z_range, a_relu, 'b-', linewidth=2)
    plt.title('ReLU Activation: max(0, z)')
    plt.xlabel('z (pre-activation)')
    plt.ylabel('a (activation)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Sigmoid
    plt.subplot(1, 3, 2)
    a_sigmoid = 1 / (1 + np.exp(-z_range))
    plt.plot(z_range, a_sigmoid, 'r-', linewidth=2)
    plt.title('Sigmoid Activation: 1/(1 + e^(-z))')
    plt.xlabel('z (pre-activation)')
    plt.ylabel('a (activation)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Tanh
    plt.subplot(1, 3, 3)
    a_tanh = np.tanh(z_range)
    plt.plot(z_range, a_tanh, 'g-', linewidth=2)
    plt.title('Tanh Activation: (e^z - e^(-z))/(e^z + e^(-z))')
    plt.xlabel('z (pre-activation)')
    plt.ylabel('a (activation)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=-1, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return configurations


if __name__ == "__main__":
    neuron_demo = demonstrate_single_neuron()

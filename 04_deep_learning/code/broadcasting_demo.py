"""
Broadcasting Demonstration

This module demonstrates broadcasting concepts in deep learning, showing how
automatic shape expansion enables efficient operations between arrays of different shapes.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_broadcasting():
    """Demonstrate broadcasting concepts"""
    
    print("Broadcasting in Deep Learning")
    print("=" * 40)
    
    # Example 1: Bias addition
    print("Example 1: Bias Addition")
    print("-" * 20)
    
    # Create sample data
    Z = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    b = np.array([10, 20, 30])
    
    print("Matrix Z (3x3):")
    print(Z)
    print()
    print("Bias vector b (3,):")
    print(b)
    print()
    
    # Broadcasting addition
    result = Z + b
    print("Result of Z + b (broadcasting):")
    print(result)
    print()
    
    # Show what broadcasting does
    print("What broadcasting does:")
    print("b is automatically expanded to:")
    b_expanded = np.tile(b.reshape(-1, 1), (1, Z.shape[1]))
    print(b_expanded)
    print()
    
    # Example 2: Different shapes
    print("Example 2: Different Shapes")
    print("-" * 20)
    
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([10, 20, 30])
    
    print("Matrix A (2x3):")
    print(A)
    print()
    print("Vector B (3,):")
    print(B)
    print()
    
    result2 = A + B
    print("Result of A + B:")
    print(result2)
    print()
    
    # Example 3: Neural network bias
    print("Example 3: Neural Network Bias")
    print("-" * 20)
    
    # Simulate neural network computation
    batch_size = 4
    hidden_size = 3
    
    # Hidden layer activations
    hidden_activations = np.random.randn(batch_size, hidden_size)
    bias = np.random.randn(hidden_size)
    
    print("Hidden activations (batch_size x hidden_size):")
    print(hidden_activations)
    print()
    print("Bias (hidden_size,):")
    print(bias)
    print()
    
    # Add bias using broadcasting
    output = hidden_activations + bias
    print("Output after adding bias:")
    print(output)
    print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Example 1 visualization
    plt.subplot(1, 3, 1)
    plt.imshow(Z, cmap='viridis', aspect='auto')
    plt.title('Matrix Z')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.colorbar(label='Value')
    
    # Add text annotations
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            plt.text(j, i, f'{Z[i, j]}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    # Bias visualization
    plt.subplot(1, 3, 2)
    plt.bar(range(len(b)), b, color='orange', alpha=0.7)
    plt.title('Bias Vector b')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # Result visualization
    plt.subplot(1, 3, 3)
    plt.imshow(result, cmap='viridis', aspect='auto')
    plt.title('Result: Z + b')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.colorbar(label='Value')
    
    # Add text annotations
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            plt.text(j, i, f'{result[i, j]}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Broadcasting rules explanation
    print("Broadcasting Rules:")
    print("1. Arrays are aligned by their last dimensions")
    print("2. Dimensions with size 1 are expanded to match")
    print("3. Missing dimensions are added with size 1")
    print("4. Operation is performed element-wise")
    print()
    print("Common Broadcasting Patterns:")
    print("- Matrix + Vector: Vector is broadcast across matrix columns")
    print("- Matrix + Scalar: Scalar is broadcast to all elements")
    print("- 3D + 2D: 2D array is broadcast across 3D array")
    
    return Z, b, result, hidden_activations, bias, output


if __name__ == "__main__":
    broadcasting_demo = demonstrate_broadcasting()

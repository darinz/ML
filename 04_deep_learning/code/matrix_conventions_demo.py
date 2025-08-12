"""
Matrix Conventions Demonstration

This module demonstrates the difference between column-major and row-major
conventions in deep learning and how they affect matrix operations.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_matrix_conventions():
    """Demonstrate the difference between column-major and row-major conventions"""
    
    # Generate sample data
    np.random.seed(42)
    m = 5  # number of examples
    d = 3  # input dimension
    h = 2  # hidden dimension
    
    # Create sample data
    data_points = [
        [1, 2, 3],    # Example 1
        [4, 5, 6],    # Example 2
        [7, 8, 9],    # Example 3
        [10, 11, 12], # Example 4
        [13, 14, 15]  # Example 5
    ]
    
    print("Matrix Conventions in Deep Learning")
    print("=" * 50)
    print(f"Data: {m} examples, {d} features each")
    print()
    
    # Column-major convention (mathematical theory)
    X_col_major = np.array(data_points).T  # Transpose to get column-major
    print("Column-Major Convention (Theory):")
    print("Shape:", X_col_major.shape)
    print("Each column is a training example:")
    print(X_col_major)
    print()
    
    # Row-major convention (implementation)
    X_row_major = np.array(data_points)
    print("Row-Major Convention (Implementation):")
    print("Shape:", X_row_major.shape)
    print("Each row is a training example:")
    print(X_row_major)
    print()
    
    # Weight matrix
    W = np.random.randn(h, d)
    b = np.random.randn(h, 1)
    
    print("Weight Matrix W:")
    print("Shape:", W.shape)
    print(W)
    print()
    
    print("Bias Vector b:")
    print("Shape:", b.shape)
    print(b)
    print()
    
    # Matrix multiplication with column-major
    print("Column-Major Matrix Multiplication:")
    print("Z = W @ X + b")
    Z_col = W @ X_col_major + b
    print("Shape:", Z_col.shape)
    print(Z_col)
    print()
    
    # Matrix multiplication with row-major
    print("Row-Major Matrix Multiplication:")
    print("Z = X @ W^T + b")
    Z_row = X_row_major @ W.T + b.T
    print("Shape:", Z_row.shape)
    print(Z_row)
    print()
    
    # Verify results are equivalent
    print("Verification:")
    print("Column-major result (transposed):")
    print(Z_col.T)
    print("Row-major result:")
    print(Z_row)
    print("Results are equivalent:", np.allclose(Z_col.T, Z_row))
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Column-major visualization
    plt.subplot(1, 3, 1)
    plt.imshow(X_col_major, cmap='viridis', aspect='auto')
    plt.title('Column-Major Convention\n(Theory)')
    plt.xlabel('Training Examples')
    plt.ylabel('Features')
    plt.colorbar(label='Feature Value')
    
    # Add text annotations
    for i in range(X_col_major.shape[0]):
        for j in range(X_col_major.shape[1]):
            plt.text(j, i, f'{X_col_major[i, j]:.0f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    # Row-major visualization
    plt.subplot(1, 3, 2)
    plt.imshow(X_row_major, cmap='viridis', aspect='auto')
    plt.title('Row-Major Convention\n(Implementation)')
    plt.xlabel('Features')
    plt.ylabel('Training Examples')
    plt.colorbar(label='Feature Value')
    
    # Add text annotations
    for i in range(X_row_major.shape[0]):
        for j in range(X_row_major.shape[1]):
            plt.text(j, i, f'{X_row_major[i, j]:.0f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    # Weight matrix visualization
    plt.subplot(1, 3, 3)
    plt.imshow(W, cmap='RdBu', aspect='auto')
    plt.title('Weight Matrix W')
    plt.xlabel('Input Features')
    plt.ylabel('Hidden Units')
    plt.colorbar(label='Weight Value')
    
    # Add text annotations
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            plt.text(j, i, f'{W[i, j]:.2f}', 
                    ha='center', va='center', color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return X_col_major, X_row_major, W, b, Z_col, Z_row


if __name__ == "__main__":
    matrix_conventions_demo = demonstrate_matrix_conventions()

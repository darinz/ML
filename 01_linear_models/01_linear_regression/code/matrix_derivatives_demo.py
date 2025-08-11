import numpy as np
import matplotlib.pyplot as plt

def demonstrate_matrix_derivatives():
    """Demonstrate matrix derivative computation"""
    
    # Define a simple matrix function
    def matrix_function(A):
        """f(A) = 3/2 * A[0,0] + 5 * A[0,1]^2 + A[1,0] * A[1,1]"""
        return 1.5 * A[0, 0] + 5 * A[0, 1]**2 + A[1, 0] * A[1, 1]
    
    def matrix_derivative(A):
        """Compute the derivative of f(A) with respect to A"""
        return np.array([
            [1.5, 10 * A[0, 1]],
            [A[1, 1], A[1, 0]]
        ])
    
    # Test with a specific matrix
    A = np.array([[2, 3], [4, 5]])
    
    print("Matrix Derivatives Example")
    print("=" * 40)
    print("Matrix A:")
    print(A)
    print()
    
    print("Function value f(A):")
    f_value = matrix_function(A)
    print(f"f(A) = {f_value}")
    print()
    
    print("Derivative matrix ∇f(A):")
    grad_A = matrix_derivative(A)
    print(grad_A)
    print()
    
    # Verify with finite differences
    print("Verification with Finite Differences:")
    epsilon = 1e-7
    numerical_grad = np.zeros_like(A)
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # Perturb A[i,j]
            A_plus = A.copy()
            A_plus[i, j] += epsilon
            A_minus = A.copy()
            A_minus[i, j] -= epsilon
            
            # Compute finite difference
            f_plus = matrix_function(A_plus)
            f_minus = matrix_function(A_minus)
            numerical_grad[i, j] = (f_plus - f_minus) / (2 * epsilon)
    
    print("Numerical gradient:")
    print(numerical_grad)
    print()
    
    print("Analytical vs Numerical:")
    print("Analytical:")
    print(grad_A)
    print("Numerical:")
    print(numerical_grad)
    print("Difference:")
    print(np.abs(grad_A - numerical_grad))
    print(f"Max difference: {np.max(np.abs(grad_A - numerical_grad)):.2e}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original matrix
    plt.subplot(1, 3, 1)
    plt.imshow(A, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Matrix A')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Add text annotations
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            plt.text(j, i, f'{A[i, j]}', ha='center', va='center', 
                    color='white', fontweight='bold')
    
    # Function value
    plt.subplot(1, 3, 2)
    plt.text(0.5, 0.5, f'f(A) = {f_value:.2f}', ha='center', va='center', 
             fontsize=14, transform=plt.gca().transAxes)
    plt.title('Function Value')
    plt.axis('off')
    
    # Gradient matrix
    plt.subplot(1, 3, 3)
    plt.imshow(grad_A, cmap='RdBu', aspect='auto')
    plt.colorbar(label='Derivative')
    plt.title('Gradient Matrix ∇f(A)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Add text annotations
    for i in range(grad_A.shape[0]):
        for j in range(grad_A.shape[1]):
            plt.text(j, i, f'{grad_A[i, j]:.2f}', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return A, f_value, grad_A, numerical_grad

if __name__ == "__main__":
    matrix_deriv_demo = demonstrate_matrix_derivatives()

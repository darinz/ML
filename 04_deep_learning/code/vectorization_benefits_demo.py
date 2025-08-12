"""
Vectorization Benefits Demonstration

This module demonstrates the dramatic performance benefits of vectorization
by comparing non-vectorized and vectorized operations for element-wise addition
and matrix multiplication.
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def demonstrate_vectorization_benefits():
    """Demonstrate the dramatic performance benefits of vectorization"""
    
    # Generate test data
    np.random.seed(42)
    sizes = [100, 500, 1000, 5000, 10000]
    
    # Test functions
    def non_vectorized_operation(X, Y):
        """Non-vectorized element-wise addition"""
        result = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                result[i, j] = X[i, j] + Y[i, j]
        return result
    
    def vectorized_operation(X, Y):
        """Vectorized element-wise addition"""
        return X + Y
    
    def non_vectorized_matrix_mult(X, Y):
        """Non-vectorized matrix multiplication"""
        result = np.zeros((X.shape[0], Y.shape[1]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                for k in range(X.shape[1]):
                    result[i, j] += X[i, k] * Y[k, j]
        return result
    
    def vectorized_matrix_mult(X, Y):
        """Vectorized matrix multiplication"""
        return X @ Y
    
    # Performance comparison
    non_vectorized_times = []
    vectorized_times = []
    non_vectorized_matmul_times = []
    vectorized_matmul_times = []
    
    print("Vectorization Performance Comparison")
    print("=" * 50)
    
    for size in sizes:
        print(f"\nTesting with size {size}x{size}:")
        
        # Generate random matrices
        X = np.random.randn(size, size)
        Y = np.random.randn(size, size)
        
        # Test element-wise addition
        start_time = time.time()
        result1 = non_vectorized_operation(X, Y)
        non_vectorized_time = time.time() - start_time
        non_vectorized_times.append(non_vectorized_time)
        
        start_time = time.time()
        result2 = vectorized_operation(X, Y)
        vectorized_time = time.time() - start_time
        vectorized_times.append(vectorized_time)
        
        # Test matrix multiplication
        start_time = time.time()
        result3 = non_vectorized_matrix_mult(X, Y)
        non_vectorized_matmul_time = time.time() - start_time
        non_vectorized_matmul_times.append(non_vectorized_matmul_time)
        
        start_time = time.time()
        result4 = vectorized_matrix_mult(X, Y)
        vectorized_matmul_time = time.time() - start_time
        vectorized_matmul_times.append(vectorized_matmul_time)
        
        # Verify results are the same
        assert np.allclose(result1, result2), "Element-wise addition results differ!"
        assert np.allclose(result3, result4), "Matrix multiplication results differ!"
        
        print(f"  Element-wise addition:")
        print(f"    Non-vectorized: {non_vectorized_time:.4f}s")
        print(f"    Vectorized: {vectorized_time:.4f}s")
        print(f"    Speedup: {non_vectorized_time/vectorized_time:.1f}x")
        
        print(f"  Matrix multiplication:")
        print(f"    Non-vectorized: {non_vectorized_matmul_time:.4f}s")
        print(f"    Vectorized: {vectorized_matmul_time:.4f}s")
        print(f"    Speedup: {non_vectorized_matmul_time/vectorized_matmul_time:.1f}x")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Element-wise operation performance
    plt.subplot(1, 3, 1)
    plt.plot(sizes, non_vectorized_times, 'r-o', label='Non-vectorized', linewidth=2, markersize=8)
    plt.plot(sizes, vectorized_times, 'b-s', label='Vectorized', linewidth=2, markersize=8)
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Element-wise Addition Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Matrix multiplication performance
    plt.subplot(1, 3, 2)
    plt.plot(sizes, non_vectorized_matmul_times, 'r-o', label='Non-vectorized', linewidth=2, markersize=8)
    plt.plot(sizes, vectorized_matmul_times, 'b-s', label='Vectorized', linewidth=2, markersize=8)
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Speedup comparison
    plt.subplot(1, 3, 3)
    speedup_element = [n/v for n, v in zip(non_vectorized_times, vectorized_times)]
    speedup_matmul = [n/v for n, v in zip(non_vectorized_matmul_times, vectorized_matmul_times)]
    
    plt.plot(sizes, speedup_element, 'g-o', label='Element-wise', linewidth=2, markersize=8)
    plt.plot(sizes, speedup_matmul, 'm-s', label='Matrix Mult', linewidth=2, markersize=8)
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Factor')
    plt.title('Vectorization Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Memory efficiency comparison
    print(f"\nMemory Efficiency Analysis:")
    print(f"Non-vectorized approach:")
    print(f"  - Creates temporary variables in loops")
    print(f"  - Multiple memory allocations")
    print(f"  - Poor cache utilization")
    print(f"Vectorized approach:")
    print(f"  - Single memory allocation")
    print(f"  - Optimized memory access patterns")
    print(f"  - Better cache utilization")
    
    return sizes, non_vectorized_times, vectorized_times, speedup_element, speedup_matmul


if __name__ == "__main__":
    vectorization_demo = demonstrate_vectorization_benefits()

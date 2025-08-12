"""
Basic Vectorization Demonstration

This module demonstrates the basic concept of vectorization by comparing
non-vectorized and vectorized approaches for neural network forward pass.
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def demonstrate_basic_vectorization():
    """Demonstrate the basic concept of vectorization"""
    
    # Generate test data
    np.random.seed(42)
    m = 1000  # number of examples
    d = 100   # input dimension
    h = 50    # hidden dimension
    
    W = np.random.randn(h, d)  # weight matrix
    X = np.random.randn(d, m)  # input data (column-major)
    b = np.random.randn(h, 1)  # bias vector
    
    print("Basic Vectorization Example")
    print("=" * 40)
    print(f"Processing {m} examples with {d} features each")
    print(f"Output dimension: {h}")
    print()
    
    # Non-vectorized approach
    print("Non-vectorized approach:")
    start_time = time.time()
    Z_non_vectorized = np.zeros((h, m))
    for i in range(m):
        Z_non_vectorized[:, i] = W @ X[:, i] + b.flatten()
    non_vectorized_time = time.time() - start_time
    print(f"  Time: {non_vectorized_time:.4f} seconds")
    print(f"  Operations per second: {m/non_vectorized_time:.0f}")
    
    # Vectorized approach
    print("\nVectorized approach:")
    start_time = time.time()
    Z_vectorized = W @ X + b
    vectorized_time = time.time() - start_time
    print(f"  Time: {vectorized_time:.4f} seconds")
    print(f"  Operations per second: {m/vectorized_time:.0f}")
    
    # Verify results are the same
    assert np.allclose(Z_non_vectorized, Z_vectorized), "Results differ!"
    print(f"\nSpeedup: {non_vectorized_time/vectorized_time:.1f}x")
    
    # Memory usage comparison
    print(f"\nMemory Analysis:")
    print(f"Non-vectorized:")
    print(f"  - Creates temporary arrays in loop")
    print(f"  - Multiple memory allocations")
    print(f"  - Poor cache utilization")
    print(f"Vectorized:")
    print(f"  - Single matrix operation")
    print(f"  - Optimized memory access")
    print(f"  - Better cache utilization")
    
    # Visualization
    plt.figure(figsize=(12, 4))
    
    # Performance comparison
    plt.subplot(1, 2, 1)
    methods = ['Non-vectorized', 'Vectorized']
    times = [non_vectorized_time, vectorized_time]
    colors = ['red', 'blue']
    
    bars = plt.bar(methods, times, color=colors, alpha=0.7)
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    # Operations per second
    plt.subplot(1, 2, 2)
    ops_per_sec = [m/non_vectorized_time, m/vectorized_time]
    
    bars = plt.bar(methods, ops_per_sec, color=colors, alpha=0.7)
    plt.ylabel('Operations per Second')
    plt.title('Throughput Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, ops in zip(bars, ops_per_sec):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{ops:.0f} ops/s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return Z_non_vectorized, Z_vectorized, non_vectorized_time, vectorized_time


if __name__ == "__main__":
    basic_vectorization_demo = demonstrate_basic_vectorization()

"""
Vectorization: The Key to Efficient Deep Learning

This module implements comprehensive examples of vectorization techniques that
are essential for efficient deep learning implementations.

Key Concepts Covered:
1. For-loop vs Vectorized Operations: Performance comparison
2. Broadcasting: Automatic dimension expansion
3. Matrix Operations: Efficient linear algebra
4. Gradient Computation: Vectorized backpropagation
5. Memory Efficiency: Optimizing computational patterns
6. Hardware Acceleration: Leveraging modern computing resources
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class VectorizationExamples:
    """
    A comprehensive class demonstrating vectorization techniques and their benefits.
    
    This class provides practical examples of how vectorization can dramatically
    improve computational efficiency in deep learning applications.
    """
    
    def __init__(self):
        """Initialize the VectorizationExamples class."""
        self.epsilon = 1e-8
        
    def for_loop_vs_vectorized_comparison(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compare for-loop vs vectorized implementation performance.
        
        This example demonstrates the dramatic performance difference between
        explicit loops and vectorized operations, which is crucial for deep learning.
        
        Mathematical Operation:
            Y = XW^T + b
            
        where:
        - X ∈ ℝ^(n×m) is the input matrix
        - W ∈ ℝ^(k×m) is the weight matrix
        - b ∈ ℝ^k is the bias vector
        - Y ∈ ℝ^(n×k) is the output matrix
        
        Args:
            None
            
        Returns:
            Tuple[np.ndarray, np.ndarray, float]: Loop output, vectorized output, and max difference
            
        Example:
            >>> vec = VectorizationExamples()
            >>> loop_output, vec_output, max_diff = vec.for_loop_vs_vectorized_comparison()
            >>> print(f"Maximum difference: {max_diff:.2e}")
        """
        print("=" * 60)
        print("FOR-LOOP vs VECTORIZED IMPLEMENTATION")
        print("=" * 60)
        
        # Generate test data
        np.random.seed(0)
        n, m, k = 1000, 4, 3  # n samples, m features, k outputs
        X = np.random.rand(n, m)
        W = np.random.randn(k, m)
        b = np.random.randn(k)
        
        print(f"Matrix dimensions:")
        print(f"  X: {X.shape} (input matrix)")
        print(f"  W: {W.shape} (weight matrix)")
        print(f"  b: {b.shape} (bias vector)")
        print(f"  Expected output: ({n}, {k})")
        
        # For-loop implementation (slow)
        print("\n--- For-loop Implementation ---")
        start_time = time.time()
        
        outputs_loop = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                outputs_loop[i, j] = np.dot(W[j], X[i]) + b[j]
        
        loop_time = time.time() - start_time
        print(f"Loop time: {loop_time:.4f} seconds")
        
        # Vectorized implementation (fast)
        print("\n--- Vectorized Implementation ---")
        start_time = time.time()
        
        outputs_vec = X @ W.T + b  # shape: (n, k)
        
        vec_time = time.time() - start_time
        print(f"Vectorized time: {vec_time:.4f} seconds")
        
        # Compare results
        max_diff = np.abs(outputs_loop - outputs_vec).max()
        speedup = loop_time / vec_time
        
        print(f"\n--- Results Comparison ---")
        print(f"Maximum difference: {max_diff:.2e}")
        print(f"Speedup: {speedup:.1f}x faster")
        print(f"Results identical: {max_diff < 1e-10}")
        
        if speedup > 10:
            print("🚀 Vectorization provides significant speedup!")
        else:
            print("⚠️  Speedup is modest - consider larger matrices")
        
        return outputs_loop, outputs_vec, max_diff
    
    def broadcasting_examples(self):
        """
        Demonstrate broadcasting and tiling operations.
        
        Broadcasting is a powerful feature that allows operations between
        arrays of different shapes by automatically expanding dimensions.
        
        Mathematical Operations:
        1. Element-wise addition with broadcasting
        2. Matrix-vector operations
        3. Batch operations
        """
        print("\n" + "=" * 60)
        print("BROADCASTING EXAMPLES")
        print("=" * 60)
        
        # Example 1: Basic broadcasting
        print("1. Basic Broadcasting")
        Z = np.random.randn(5, 3)
        b_vec = np.array([1, 2, 3])
        
        print(f"Z shape: {Z.shape}")
        print(f"b_vec shape: {b_vec.shape}")
        print(f"Z:\n{Z}")
        print(f"b_vec: {b_vec}")
        
        # Broadcasting automatically expands b_vec to (5, 3)
        result_broadcast = Z + b_vec
        print(f"Broadcasting result (Z + b_vec):\n{result_broadcast}")
        
        # Manual tiling for comparison
        b_tiled = np.tile(b_vec, (5, 1))
        print(f"Manual tiling result:\n{b_tiled}")
        
        # Verify they're the same
        diff = np.abs(result_broadcast - b_tiled).max()
        print(f"Broadcasting vs tiling difference: {diff:.2e}")
        
        # Example 2: Matrix-vector operations
        print("\n2. Matrix-Vector Operations")
        A = np.random.randn(4, 3)
        v = np.random.randn(3)
        
        print(f"A shape: {A.shape}")
        print(f"v shape: {v.shape}")
        
        # Broadcasting in matrix multiplication
        result_mv = A * v  # Element-wise multiplication with broadcasting
        print(f"A * v (broadcasted):\n{result_mv}")
        
        # Example 3: Batch operations
        print("\n3. Batch Operations")
        batch_size = 3
        features = 4
        X_batch = np.random.randn(batch_size, features)
        W = np.random.randn(features, 2)
        b = np.random.randn(2)
        
        print(f"X_batch shape: {X_batch.shape}")
        print(f"W shape: {W.shape}")
        print(f"b shape: {b.shape}")
        
        # Batch matrix multiplication with broadcasting
        Y_batch = X_batch @ W + b
        print(f"Y_batch shape: {Y_batch.shape}")
        print(f"Y_batch:\n{Y_batch}")
        
        # Verify broadcasting rules
        print(f"\nBroadcasting Rules Summary:")
        print(f"- Arrays are aligned by their last dimensions")
        print(f"- Missing dimensions are added with size 1")
        print(f"- Dimensions with size 1 are expanded to match")
        print(f"- Broadcasting is memory-efficient (no copying)")
    
    def vectorized_gradient_computation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Demonstrate vectorized gradient computation for neural networks.
        
        This example shows how to compute gradients efficiently using
        vectorized operations instead of loops.
        
        Mathematical Operations:
            Forward: Y = XW^T + b
            Loss: L = (1/2n) ||Y - Y_true||²
            Gradients: ∂L/∂W = (1/n) X^T(Y - Y_true)
                      ∂L/∂b = (1/n) Σ(Y - Y_true)
        
        Args:
            None
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Gradients for W and b
            
        Example:
            >>> vec = VectorizationExamples()
            >>> grad_W, grad_b = vec.vectorized_gradient_computation()
            >>> print(f"Gradient shapes: W={grad_W.shape}, b={grad_b.shape}")
        """
        print("\n" + "=" * 60)
        print("VECTORIZED GRADIENT COMPUTATION")
        print("=" * 60)
        
        # Generate synthetic data
        np.random.seed(42)
        n, m, k = 100, 4, 2  # n samples, m features, k outputs
        X = np.random.randn(n, m)
        W_true = np.random.randn(k, m)
        b_true = np.random.randn(k)
        
        # Generate true outputs
        Y_true = X @ W_true.T + b_true + 0.1 * np.random.randn(n, k)
        
        # Initialize parameters
        W = np.random.randn(k, m) * 0.1
        b = np.random.randn(k) * 0.1
        
        print(f"Data dimensions:")
        print(f"  X: {X.shape}")
        print(f"  W: {W.shape}")
        print(f"  b: {b.shape}")
        print(f"  Y_true: {Y_true.shape}")
        
        # Forward pass
        Y_pred = X @ W.T + b
        
        # Compute loss
        loss = np.mean(0.5 * (Y_pred - Y_true) ** 2)
        print(f"Initial loss: {loss:.4f}")
        
        # Vectorized gradient computation
        print("\n--- Vectorized Gradient Computation ---")
        
        # Gradient of loss with respect to predictions
        grad_Y = (Y_pred - Y_true) / n  # (n,)
        
        # Gradient with respect to W: ∂L/∂W = X^T · grad_Y
        grad_W = X.T @ grad_Y  # (m, k)
        
        # Gradient with respect to b: ∂L/∂b = Σ grad_Y
        grad_b = np.sum(grad_Y, axis=0)  # (k,)
        
        print(f"grad_Y shape: {grad_Y.shape}")
        print(f"grad_W shape: {grad_W.shape}")
        print(f"grad_b shape: {grad_b.shape}")
        
        # Verify gradients with finite differences
        print("\n--- Gradient Verification ---")
        epsilon = 1e-6
        
        # Check W gradient
        W_plus = W + epsilon * np.eye(k)[:, 0:1] @ np.ones((1, m))
        Y_plus = X @ W_plus.T + b
        loss_plus = np.mean(0.5 * (Y_plus - Y_true) ** 2)
        
        W_minus = W - epsilon * np.eye(k)[:, 0:1] @ np.ones((1, m))
        Y_minus = X @ W_minus.T + b
        loss_minus = np.mean(0.5 * (Y_minus - Y_true) ** 2)
        
        finite_diff_W = (loss_plus - loss_minus) / (2 * epsilon)
        analytical_W = grad_W[0, 0]
        
        print(f"Finite difference ∂L/∂W[0,0]: {finite_diff_W:.6f}")
        print(f"Analytical ∂L/∂W[0,0]: {analytical_W:.6f}")
        print(f"Difference: {abs(finite_diff_W - analytical_W):.2e}")
        
        return grad_W, grad_b
    
    def two_layer_neural_network_vectorized(self) -> Tuple[np.ndarray, List[float]]:
        """
        Complete vectorized implementation of a two-layer neural network.
        
        This example demonstrates how to implement a complete neural network
        using only vectorized operations, showing the power of modern
        deep learning frameworks.
        
        Architecture:
            Input -> Linear -> ReLU -> Linear -> Output
            (n, m)    (m->k)    (k)     (k->1)    (n, 1)
        
        Args:
            None
            
        Returns:
            Tuple[np.ndarray, List[float]]: Predictions and loss history
            
        Example:
            >>> vec = VectorizationExamples()
            >>> predictions, losses = vec.two_layer_neural_network_vectorized()
            >>> print(f"Final loss: {losses[-1]:.4f}")
        """
        print("\n" + "=" * 60)
        print("TWO-LAYER NEURAL NETWORK (VECTORIZED)")
        print("=" * 60)
        
        # Generate synthetic data
        np.random.seed(42)
        n, m, k = 200, 4, 3  # n samples, m features, k hidden neurons
        
        X = np.random.rand(n, m)
        
        # True underlying relationship
        true_W1 = np.array([[1.2, -0.7, 0.5, 2.0],
                           [0.3, 1.5, -1.0, 0.7],
                           [2.0, 0.1, 0.3, -0.5]])  # (k, m)
        true_b1 = np.array([0.5, -0.2, 0.1])
        
        # Generate hidden layer output
        H = np.maximum(X @ true_W1.T + true_b1, 0)  # ReLU activation
        
        # Generate final output
        true_W2 = np.array([1.0, -2.0, 0.5])
        true_b2 = 0.3
        y = H @ true_W2 + true_b2 + np.random.normal(0, 0.2, size=n)
        
        print(f"Data dimensions:")
        print(f"  X: {X.shape} (input)")
        print(f"  y: {y.shape} (target)")
        print(f"  Network: {m} -> {k} -> 1")
        
        # Initialize parameters
        W1 = np.random.randn(k, m) * 0.1
        b1 = np.zeros(k)
        W2 = np.random.randn(k) * 0.1
        b2 = 0.0
        lr = 0.05
        
        # Training loop
        losses = []
        for epoch in range(600):
            # Forward pass (vectorized)
            Z1 = X @ W1.T + b1  # (n, k)
            A1 = np.maximum(Z1, 0)  # ReLU: (n, k)
            y_pred = A1 @ W2 + b2  # (n,)
            
            # Loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            
            # Backpropagation (vectorized)
            grad_y_pred = 2 * (y_pred - y) / n  # (n,)
            
            # Gradients for second layer
            grad_W2 = A1.T @ grad_y_pred  # (k,)
            grad_b2 = np.sum(grad_y_pred)  # scalar
            
            # Gradients for first layer
            grad_A1 = np.outer(grad_y_pred, W2)  # (n, k)
            grad_Z1 = grad_A1 * (Z1 > 0)  # ReLU derivative: (n, k)
            grad_W1 = grad_Z1.T @ X  # (k, m)
            grad_b1 = np.sum(grad_Z1, axis=0)  # (k,)
            
            # Parameter updates
            W1 -= lr * grad_W1
            b1 -= lr * grad_b1
            W2 -= lr * grad_W2
            b2 -= lr * grad_b2
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:3d}: Loss = {loss:.4f}")
        
        print(f"Final loss: {losses[-1]:.4f}")
        
        # Visualize results
        self._visualize_training_results(y, y_pred, losses)
        
        return y_pred, losses
    
    def _visualize_training_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  losses: List[float]):
        """Create visualization plots for training results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Predictions vs True values
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Two-Layer Neural Network Predictions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss convergence
        ax2.plot(losses)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('Training Loss Convergence')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def memory_efficiency_comparison(self):
        """
        Compare memory efficiency of different computational patterns.
        
        This example demonstrates how different approaches to the same
        computation can have vastly different memory requirements.
        """
        print("\n" + "=" * 60)
        print("MEMORY EFFICIENCY COMPARISON")
        print("=" * 60)
        
        # Large matrix operations
        n, m = 1000, 500
        
        print(f"Matrix dimensions: {n} x {m}")
        
        # Method 1: Inefficient (creates intermediate arrays)
        print("\n1. Inefficient Method (with intermediates)")
        A = np.random.randn(n, m)
        B = np.random.randn(m, n)
        
        # This creates multiple intermediate arrays
        temp1 = A @ B  # (n, n)
        temp2 = temp1 @ A  # (n, m)
        result1 = temp2 + A  # (n, m)
        
        print(f"Memory usage: Creates {n*n + n*m} intermediate elements")
        
        # Method 2: Efficient (minimal intermediates)
        print("\n2. Efficient Method (minimal intermediates)")
        result2 = (A @ B @ A) + A
        
        print(f"Memory usage: Minimal intermediate storage")
        
        # Verify results are the same
        diff = np.abs(result1 - result2).max()
        print(f"Results identical: {diff < 1e-10}")
        
        # Method 3: In-place operations
        print("\n3. In-place Operations")
        C = np.random.randn(n, m)
        D = np.random.randn(n, m)
        
        # In-place addition
        C += D  # Modifies C directly
        
        print(f"In-place operations modify arrays directly")
        print(f"Memory efficient: No new arrays created")
    
    def hardware_acceleration_demo(self):
        """
        Demonstrate concepts related to hardware acceleration.
        
        This example shows how vectorized operations can leverage
        modern hardware features like SIMD instructions and GPUs.
        """
        print("\n" + "=" * 60)
        print("HARDWARE ACCELERATION DEMO")
        print("=" * 60)
        
        # Demonstrate SIMD-friendly operations
        print("1. SIMD-Friendly Operations")
        
        # Vectorized operations that can use SIMD
        n = 1000000
        a = np.random.randn(n)
        b = np.random.randn(n)
        
        # These operations can be parallelized on CPU
        start_time = time.time()
        c_simd = a + b  # Vectorized addition
        simd_time = time.time() - start_time
        
        # Scalar operations (slower)
        start_time = time.time()
        c_scalar = np.zeros(n)
        for i in range(n):
            c_scalar[i] = a[i] + b[i]
        scalar_time = time.time() - start_time
        
        print(f"SIMD time: {simd_time:.4f} seconds")
        print(f"Scalar time: {scalar_time:.4f} seconds")
        print(f"Speedup: {scalar_time/simd_time:.1f}x")
        
        # Demonstrate batch processing
        print("\n2. Batch Processing")
        batch_sizes = [1, 10, 100, 1000]
        
        for batch_size in batch_sizes:
            X_batch = np.random.randn(batch_size, 100)
            W = np.random.randn(100, 50)
            
            start_time = time.time()
            Y_batch = X_batch @ W
            batch_time = time.time() - start_time
            
            print(f"Batch size {batch_size:4d}: {batch_time:.4f}s")
        
        print("\nLarger batches often have better throughput due to:")
        print("- Better cache utilization")
        print("- Reduced overhead per operation")
        print("- Hardware optimization opportunities")
    
    def performance_benchmarks(self):
        """
        Run comprehensive performance benchmarks.
        
        This example provides detailed performance comparisons
        between different computational approaches.
        """
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 60)
        
        # Test different matrix sizes
        sizes = [100, 500, 1000, 2000]
        
        print("Matrix Multiplication Performance:")
        print("Size\t\tLoop (s)\tVectorized (s)\tSpeedup")
        print("-" * 50)
        
        for size in sizes:
            # Generate matrices
            A = np.random.randn(size, size)
            B = np.random.randn(size, size)
            
            # Loop implementation
            start_time = time.time()
            C_loop = np.zeros((size, size))
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        C_loop[i, j] += A[i, k] * B[k, j]
            loop_time = time.time() - start_time
            
            # Vectorized implementation
            start_time = time.time()
            C_vec = A @ B
            vec_time = time.time() - start_time
            
            speedup = loop_time / vec_time
            
            print(f"{size:4d}\t\t{loop_time:.4f}\t\t{vec_time:.4f}\t\t{speedup:.1f}x")
        
        print(f"\nKey Insights:")
        print(f"- Vectorization provides massive speedups")
        print(f"- Speedup increases with matrix size")
        print(f"- Modern hardware is optimized for vectorized operations")


def main():
    """
    Main function demonstrating all vectorization examples.
    
    This function provides comprehensive examples of vectorization
    techniques and their practical applications in deep learning.
    """
    print("=" * 60)
    print("VECTORIZATION: THE KEY TO EFFICIENT DEEP LEARNING")
    print("=" * 60)
    
    # Initialize the examples
    vec = VectorizationExamples()
    
    # Run all demonstrations
    try:
        # 1. For-loop vs vectorized comparison
        loop_output, vec_output, max_diff = vec.for_loop_vs_vectorized_comparison()
        
        # 2. Broadcasting examples
        vec.broadcasting_examples()
        
        # 3. Vectorized gradient computation
        grad_W, grad_b = vec.vectorized_gradient_computation()
        
        # 4. Two-layer neural network
        predictions, losses = vec.two_layer_neural_network_vectorized()
        
        # 5. Memory efficiency comparison
        vec.memory_efficiency_comparison()
        
        # 6. Hardware acceleration demo
        vec.hardware_acceleration_demo()
        
        # 7. Performance benchmarks
        vec.performance_benchmarks()
        
        print("\n" + "=" * 60)
        print("ALL VECTORIZATION EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
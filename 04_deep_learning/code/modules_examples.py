"""
Neural Network Modules: Building Blocks of Modern Deep Learning

This module implements the fundamental building blocks (modules) used in modern
neural networks, demonstrating how complex architectures are constructed from
simple, reusable components.

Key Concepts Covered:
1. Matrix Multiplication Modules: Linear transformations with bias
2. Layer Normalization: Stabilizing training through normalization
3. Convolutional Modules: 1D and 2D convolutions for spatial data
4. Residual Connections: Enabling training of very deep networks
5. Module Composition: Building complex networks from simple modules
6. Scale-Invariant Properties: Mathematical properties of normalization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class NeuralNetworkModules:
    """
    A comprehensive class implementing neural network modules and their properties.
    
    This class demonstrates how modern neural networks are built using modular
    components that can be composed to create complex architectures.
    """
    
    def __init__(self):
        """Initialize the NeuralNetworkModules class."""
        self.epsilon = 1e-8  # Small constant for numerical stability
        
    def matrix_multiplication(self, x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Matrix multiplication module: MM_{W, b}(x) = Wx + b.
        
        This is the most fundamental module in neural networks, performing
        linear transformations with learnable parameters.
        
        Mathematical Definition:
            MM_{W, b}(x) = Wx + b
            
        where:
        - W ∈ ℝ^(n×m) is the weight matrix
        - b ∈ ℝ^n is the bias vector
        - x ∈ ℝ^m is the input vector
        - Output has dimension n
        
        Properties:
        - Linearity: MM(ax₁ + bx₂) = a·MM(x₁) + b·MM(x₂)
        - Parameter count: nm + n (weights + biases)
        - Computational complexity: O(nm)
        
        Args:
            x: Input vector, shape (m,)
            W: Weight matrix, shape (n, m)
            b: Bias vector, shape (n,)
            
        Returns:
            np.ndarray: Output vector, shape (n,)
            
        Example:
            >>> modules = NeuralNetworkModules()
            >>> x = np.array([1.0, 2.0])
            >>> W = np.array([[1.0, 0.5], [0.5, 1.0]])
            >>> b = np.array([0.1, -0.2])
            >>> output = modules.matrix_multiplication(x, W, b)
            >>> print(f"Output: {output}")
        """
        # Validate input shapes
        if x.ndim != 1:
            raise ValueError(f"Input x must be 1D, got shape {x.shape}")
        if W.ndim != 2:
            raise ValueError(f"Weight matrix W must be 2D, got shape {W.shape}")
        if b.ndim != 1:
            raise ValueError(f"Bias vector b must be 1D, got shape {b.shape}")
        if W.shape[1] != x.shape[0]:
            raise ValueError(f"W.shape[1] ({W.shape[1]}) must equal x.shape[0] ({x.shape[0]})")
        if W.shape[0] != b.shape[0]:
            raise ValueError(f"W.shape[0] ({W.shape[0]}) must equal b.shape[0] ({b.shape[0]})")
        
        # Perform matrix multiplication with bias
        return W @ x + b
    
    def layer_normalization_simple(self, z: np.ndarray) -> np.ndarray:
        """
        Layer normalization submodule (LN-S): normalize to mean 0, std 1.
        
        This submodule normalizes the input to have zero mean and unit variance,
        which helps stabilize training by preventing exploding or vanishing gradients.
        
        Mathematical Definition:
            LN-S(z) = (z - μ) / σ
            
        where:
        - μ = (1/m) * Σᵢ zᵢ is the empirical mean
        - σ = √((1/m) * Σᵢ (zᵢ - μ)²) is the empirical standard deviation
        - m is the dimension of z
        
        Properties:
        - Scale invariant: LN-S(αz) = LN-S(z) for any α ≠ 0
        - Output has mean 0 and variance 1
        - Helps with training stability
        
        Args:
            z: Input vector, shape (m,)
            
        Returns:
            np.ndarray: Normalized vector, shape (m,)
            
        Example:
            >>> modules = NeuralNetworkModules()
            >>> z = np.array([1.0, 2.0, 3.0, 4.0])
            >>> normalized = modules.layer_normalization_simple(z)
            >>> print(f"Original: {z}")
            >>> print(f"Normalized: {normalized}")
            >>> print(f"Mean: {np.mean(normalized):.6f}")
            >>> print(f"Std: {np.std(normalized):.6f}")
        """
        # Compute empirical mean and standard deviation
        mu = np.mean(z)
        sigma = np.std(z, ddof=0)  # Population std (divide by m, not m-1)
        
        # Normalize to zero mean and unit variance
        return (z - mu) / (sigma + self.epsilon)
    
    def layer_normalization(self, z: np.ndarray, beta: float = 0.0, gamma: float = 1.0) -> np.ndarray:
        """
        Full layer normalization: affine transform after normalization.
        
        This module applies an affine transformation after normalization, allowing
        the network to learn the optimal scale and shift parameters.
        
        Mathematical Definition:
            LN(z) = β + γ · LN-S(z)
            
        where:
        - β is the learnable shift parameter
        - γ is the learnable scale parameter
        - LN-S(z) is the simple layer normalization
        
        Properties:
        - Scale invariant: LN(αz) = LN(z) for any α ≠ 0
        - Learnable parameters allow optimal normalization
        - Helps with training stability and convergence
        
        Args:
            z: Input vector, shape (m,)
            beta: Shift parameter (default: 0.0)
            gamma: Scale parameter (default: 1.0)
            
        Returns:
            np.ndarray: Normalized and transformed vector, shape (m,)
            
        Example:
            >>> modules = NeuralNetworkModules()
            >>> z = np.array([1.0, 2.0, 3.0, 4.0])
            >>> beta, gamma = 0.5, 2.0
            >>> output = modules.layer_normalization(z, beta, gamma)
            >>> print(f"Output: {output}")
        """
        # Apply simple layer normalization
        normalized = self.layer_normalization_simple(z)
        
        # Apply affine transformation
        return beta + gamma * normalized
    
    def demonstrate_scale_invariance(self):
        """
        Demonstrate the scale-invariant property of layer normalization.
        
        This example shows that layer normalization is invariant to scaling
        of the input, which is a key property that makes it useful in practice.
        """
        print("=" * 60)
        print("LAYER NORMALIZATION SCALE INVARIANCE")
        print("=" * 60)
        
        # Original input
        z = np.array([1.0, 2.0, 3.0, 4.0])
        beta, gamma = 0.5, 2.0
        
        print(f"Original input: {z}")
        print(f"Parameters: β={beta}, γ={gamma}")
        
        # Apply layer normalization to original input
        ln_original = self.layer_normalization(z, beta, gamma)
        print(f"LN(original): {ln_original}")
        
        # Test scale invariance with different scaling factors
        scaling_factors = [0.5, 2.0, 10.0, -1.0]
        
        for alpha in scaling_factors:
            z_scaled = alpha * z
            ln_scaled = self.layer_normalization(z_scaled, beta, gamma)
            
            print(f"\nScaling factor α = {alpha}")
            print(f"Scaled input: {z_scaled}")
            print(f"LN(scaled): {ln_scaled}")
            
            # Check if results are identical (within numerical precision)
            diff = np.max(np.abs(ln_original - ln_scaled))
            print(f"Maximum difference: {diff:.2e}")
            print("✓ Scale invariant!" if diff < 1e-10 else "✗ Not scale invariant!")
    
    def convolution_1d_simple(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        1D convolution (simplified): apply filter w over input z with zero padding.
        
        This module applies a 1D filter over the input sequence, capturing
        local patterns and enabling parameter sharing across positions.
        
        Mathematical Definition:
            Conv1D-S(z)_i = Σⱼ wⱼ z_{i-ℓ+(j-1)}
            
        where:
        - w is the filter (kernel) with k = 2ℓ + 1 elements
        - z is the input vector with zero padding
        - Output has the same dimension as input
        
        Properties:
        - Parameter sharing: Same filter applied at all positions
        - Local connectivity: Each output depends only on a local window
        - Translation invariance: Same pattern detected regardless of position
        - Efficiency: O(km) operations vs O(m²) for full matrix multiplication
        
        Args:
            z: Input vector, shape (m,)
            w: Filter (kernel), shape (k,) where k is odd
            
        Returns:
            np.ndarray: Convolved output, shape (m,)
            
        Example:
            >>> modules = NeuralNetworkModules()
            >>> z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> w = np.array([0.2, 0.5, 0.2])
            >>> output = modules.convolution_1d_simple(z, w)
            >>> print(f"Input: {z}")
            >>> print(f"Filter: {w}")
            >>> print(f"Output: {output}")
        """
        # Validate input
        if len(w) % 2 == 0:
            raise ValueError("Filter length must be odd")
        
        k = len(w)
        l = (k - 1) // 2
        m = len(z)
        
        # Zero padding
        z_padded = np.pad(z, (l, l), mode='constant', constant_values=0)
        
        # Apply convolution
        output = np.zeros(m)
        for i in range(m):
            output[i] = np.dot(w, z_padded[i:i+k])
        
        return output
    
    def convolution_1d_multi_channel(self, z_list: List[np.ndarray], 
                                   w: np.ndarray, C_out: int) -> List[np.ndarray]:
        """
        Multi-channel 1D convolution with multiple input and output channels.
        
        This module extends 1D convolution to handle multiple channels,
        which is common in real-world applications like signal processing
        and natural language processing.
        
        Mathematical Definition:
            Conv1D(z)_i = Σⱼ Conv1D-S_{i,j}(zⱼ)
            
        where:
        - z₁, ..., z_C are input channels
        - Conv1D-S_{i,j} is a separate filter for input channel j and output channel i
        - Output has C' channels
        
        Args:
            z_list: List of C input vectors, each shape (m,)
            w: Weight tensor, shape (C_out, C_in, k)
            C_out: Number of output channels
            
        Returns:
            List[np.ndarray]: List of C_out output vectors, each shape (m,)
            
        Example:
            >>> modules = NeuralNetworkModules()
            >>> z_list = [np.array([1.0, 2.0, 3.0, 4.0]), 
                         np.array([0.5, 1.5, 2.5, 3.5])]
            >>> w = np.random.randn(2, 2, 3)  # (C_out=2, C_in=2, k=3)
            >>> outputs = modules.convolution_1d_multi_channel(z_list, w, 2)
            >>> for i, output in enumerate(outputs):
            >>>     print(f"Output channel {i}: {output}")
        """
        C_in = len(z_list)
        m = len(z_list[0])
        k = w.shape[2]
        
        # Validate input shapes
        if w.shape[0] != C_out:
            raise ValueError(f"w.shape[0] ({w.shape[0]}) must equal C_out ({C_out})")
        if w.shape[1] != C_in:
            raise ValueError(f"w.shape[1] ({w.shape[1]}) must equal C_in ({C_in})")
        
        # Apply convolution for each output channel
        output_list = []
        for i in range(C_out):
            output = np.zeros(m)
            for j in range(C_in):
                # Apply simple convolution for this input-output channel pair
                output += self.convolution_1d_simple(z_list[j], w[i, j])
            output_list.append(output)
        
        return output_list
    
    def convolution_2d_simple(self, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        2D convolution (simplified): apply 2D filter w over input z with zero padding.
        
        This module applies a 2D filter over the input matrix, capturing
        spatial patterns and commonly used in image processing.
        
        Mathematical Definition:
            Conv2D-S(z)_{i,j} = Σₚ Σₚ w_{p,q} z_{i+p-ℓ, j+q-ℓ}
            
        where:
        - w is the 2D filter (kernel) with k × k elements
        - z is the 2D input with zero padding
        - ℓ = (k-1)/2 for odd k
        - Output has the same dimensions as input
        
        Args:
            z: Input matrix, shape (m, m)
            w: 2D filter (kernel), shape (k, k) where k is odd
            
        Returns:
            np.ndarray: Convolved output, shape (m, m)
            
        Example:
            >>> modules = NeuralNetworkModules()
            >>> z = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> w = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            >>> output = modules.convolution_2d_simple(z, w)
            >>> print(f"Input:\n{z}")
            >>> print(f"Filter:\n{w}")
            >>> print(f"Output:\n{output}")
        """
        # Validate input
        if w.shape[0] != w.shape[1]:
            raise ValueError("Filter must be square")
        if w.shape[0] % 2 == 0:
            raise ValueError("Filter size must be odd")
        
        k = w.shape[0]
        l = (k - 1) // 2
        m = z.shape[0]
        
        # Zero padding
        z_padded = np.pad(z, ((l, l), (l, l)), mode='constant', constant_values=0)
        
        # Apply 2D convolution
        output = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                output[i, j] = np.sum(w * z_padded[i:i+k, j:j+k])
        
        return output
    
    def convolution_2d_multi_channel(self, z_list: List[np.ndarray], 
                                   w: np.ndarray, C_out: int) -> List[np.ndarray]:
        """
        Multi-channel 2D convolution with multiple input and output channels.
        
        This module extends 2D convolution to handle multiple channels,
        which is essential for image processing and computer vision.
        
        Args:
            z_list: List of C input matrices, each shape (m, m)
            w: Weight tensor, shape (C_out, C_in, k, k)
            C_out: Number of output channels
            
        Returns:
            List[np.ndarray]: List of C_out output matrices, each shape (m, m)
        """
        C_in = len(z_list)
        m = z_list[0].shape[0]
        k = w.shape[2]
        
        # Validate input shapes
        if w.shape[0] != C_out:
            raise ValueError(f"w.shape[0] ({w.shape[0]}) must equal C_out ({C_out})")
        if w.shape[1] != C_in:
            raise ValueError(f"w.shape[1] ({w.shape[1]}) must equal C_in ({C_in})")
        
        # Apply convolution for each output channel
        output_list = []
        for i in range(C_out):
            output = np.zeros((m, m))
            for j in range(C_in):
                # Apply simple 2D convolution for this input-output channel pair
                output += self.convolution_2d_simple(z_list[j], w[i, j])
            output_list.append(output)
        
        return output_list
    
    def residual_connection(self, z: np.ndarray, f: callable) -> np.ndarray:
        """
        Residual connection: Res(z) = z + f(z).
        
        Residual connections help with training very deep networks by providing
        direct paths for gradient flow and allowing the network to learn
        incremental transformations.
        
        Mathematical Definition:
            Res(z) = z + f(z)
            
        where f is a function (typically a composition of linear and non-linear layers).
        
        Properties:
        - Learn increments: Focus on learning the difference from identity mapping
        - Ease training: Provide direct paths for gradients to flow
        - Prevent degradation: Avoid performance degradation in very deep networks
        
        Args:
            z: Input vector
            f: Function to apply (e.g., a neural network layer)
            
        Returns:
            np.ndarray: Output with residual connection
        """
        return z + f(z)
    
    def demonstrate_residual_properties(self):
        """
        Demonstrate the properties of residual connections.
        """
        print("=" * 60)
        print("RESIDUAL CONNECTION PROPERTIES")
        print("=" * 60)
        
        # Define a simple function (simulating a neural network layer)
        def simple_layer(x):
            return 0.1 * x + 0.05
        
        # Test input
        z = np.array([1.0, 2.0, 3.0, 4.0])
        
        print(f"Input: {z}")
        print(f"Simple layer output: {simple_layer(z)}")
        
        # Apply residual connection
        residual_output = self.residual_connection(z, simple_layer)
        print(f"Residual output: {residual_output}")
        
        # Show that residual connection learns increments
        increment = residual_output - z
        print(f"Increment learned: {increment}")
        
        # Demonstrate gradient flow property
        print(f"\nGradient flow property:")
        print(f"∂Res(z)/∂z = I + ∂f(z)/∂z")
        print(f"This ensures gradients can flow directly through the identity term")
    
    def module_composition(self, x: np.ndarray, modules: List[callable]) -> np.ndarray:
        """
        Compose multiple modules: f_L ∘ f_{L-1} ∘ ... ∘ f_1(x).
        
        This demonstrates how complex neural networks are built by composing
        simple modules in sequence.
        
        Args:
            x: Input vector
            modules: List of functions to compose (applied from left to right)
            
        Returns:
            np.ndarray: Result of module composition
        """
        result = x
        for module in modules:
            result = module(result)
        return result
    
    def demonstrate_module_composition(self):
        """
        Demonstrate how modules can be composed to create complex networks.
        """
        print("=" * 60)
        print("MODULE COMPOSITION")
        print("=" * 60)
        
        # Define simple modules
        def linear_layer(x):
            W = np.array([[1.0, 0.5], [0.5, 1.0]])
            b = np.array([0.1, -0.2])
            return self.matrix_multiplication(x, W, b)
        
        def relu_layer(x):
            return np.maximum(0, x)
        
        def normalization_layer(x):
            return self.layer_normalization(x, beta=0.1, gamma=1.5)
        
        # Test input
        x = np.array([1.0, 2.0])
        print(f"Input: {x}")
        
        # Compose modules
        modules = [linear_layer, relu_layer, normalization_layer]
        output = self.module_composition(x, modules)
        
        print(f"After linear layer: {linear_layer(x)}")
        print(f"After ReLU: {relu_layer(linear_layer(x))}")
        print(f"After normalization: {output}")
        
        print(f"\nThis demonstrates how complex transformations are built")
        print(f"by composing simple, reusable modules.")
    
    def visualize_convolution_effects(self):
        """
        Visualize the effects of different convolution filters.
        """
        print("=" * 60)
        print("CONVOLUTION FILTER EFFECTS")
        print("=" * 60)
        
        # Create a simple 1D signal
        x = np.linspace(0, 4*np.pi, 100)
        signal = np.sin(x) + 0.3 * np.random.randn(100)
        
        # Define different filters
        filters = {
            'Smoothing': np.array([0.25, 0.5, 0.25]),
            'Edge Detection': np.array([-1, 0, 1]),
            'Sharpening': np.array([-0.5, 2, -0.5])
        }
        
        # Apply filters and visualize
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Original signal
        axes[0, 0].plot(x, signal)
        axes[0, 0].set_title('Original Signal')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Apply each filter
        for i, (filter_name, filter_kernel) in enumerate(filters.items()):
            filtered_signal = self.convolution_1d_simple(signal, filter_kernel)
            
            row = (i + 1) // 2
            col = (i + 1) % 2
            axes[row, col].plot(x, filtered_signal)
            axes[row, col].set_title(f'{filter_name} Filter')
            axes[row, col].set_xlabel('Position')
            axes[row, col].set_ylabel('Amplitude')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("Different convolution filters capture different features:")
        print("- Smoothing: Reduces noise and high-frequency components")
        print("- Edge Detection: Highlights rapid changes in the signal")
        print("- Sharpening: Enhances high-frequency components")


def main():
    """
    Main function demonstrating all neural network modules.
    
    This function provides comprehensive examples of neural network modules
    and their practical applications.
    """
    print("=" * 60)
    print("NEURAL NETWORK MODULES: BUILDING BLOCKS OF DEEP LEARNING")
    print("=" * 60)
    
    # Initialize the modules
    modules = NeuralNetworkModules()
    
    # Run all demonstrations
    try:
        # 1. Matrix multiplication
        print("\n1. MATRIX MULTIPLICATION MODULE")
        print("-" * 40)
        x = np.array([1.0, 2.0])
        W = np.array([[1.0, 0.5], [0.5, 1.0]])
        b = np.array([0.1, -0.2])
        output = modules.matrix_multiplication(x, W, b)
        print(f"Input: {x}")
        print(f"Weights: {W}")
        print(f"Bias: {b}")
        print(f"Output: {output}")
        
        # 2. Layer normalization
        print("\n2. LAYER NORMALIZATION")
        print("-" * 40)
        z = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = modules.layer_normalization(z, beta=0.5, gamma=2.0)
        print(f"Input: {z}")
        print(f"Normalized: {normalized}")
        print(f"Mean: {np.mean(normalized):.6f}")
        print(f"Std: {np.std(normalized):.6f}")
        
        # 3. Scale invariance demonstration
        modules.demonstrate_scale_invariance()
        
        # 4. 1D Convolution
        print("\n4. 1D CONVOLUTION")
        print("-" * 40)
        z = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([0.2, 0.5, 0.2])
        conv_output = modules.convolution_1d_simple(z, w)
        print(f"Input: {z}")
        print(f"Filter: {w}")
        print(f"Output: {conv_output}")
        
        # 5. Multi-channel convolution
        print("\n5. MULTI-CHANNEL CONVOLUTION")
        print("-" * 40)
        z_list = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 1.5, 2.5, 3.5])]
        w = np.random.randn(2, 2, 3)  # (C_out=2, C_in=2, k=3)
        outputs = modules.convolution_1d_multi_channel(z_list, w, 2)
        for i, output in enumerate(outputs):
            print(f"Output channel {i}: {output}")
        
        # 6. 2D Convolution
        print("\n6. 2D CONVOLUTION")
        print("-" * 40)
        z_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        w_2d = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        conv_2d_output = modules.convolution_2d_simple(z_2d, w_2d)
        print(f"Input:\n{z_2d}")
        print(f"Filter:\n{w_2d}")
        print(f"Output:\n{conv_2d_output}")
        
        # 7. Residual connections
        modules.demonstrate_residual_properties()
        
        # 8. Module composition
        modules.demonstrate_module_composition()
        
        # 9. Convolution visualization
        modules.visualize_convolution_effects()
        
        print("\n" + "=" * 60)
        print("ALL MODULE EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
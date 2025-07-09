"""
Neural Networks: From Single Neurons to Deep Architectures

This module implements comprehensive examples of neural networks, demonstrating
the evolution from simple single neurons to complex deep architectures.

Key Concepts Covered:
1. Single Neuron Models: Linear and non-linear transformations
2. Multi-Layer Perceptrons: Forward and backward propagation
3. Activation Functions: ReLU, Sigmoid, Tanh, and their properties
4. Vectorization: Efficient matrix operations for neural networks
5. Deep Architectures: Multi-layer networks with backpropagation
6. Feature Learning: Comparison with kernel methods
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles
from scipy.special import erf
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class NeuralNetworkExamples:
    """
    A comprehensive class demonstrating neural network concepts and implementations.
    
    This class provides practical examples of neural networks at different levels
    of complexity, from single neurons to deep architectures.
    """
    
    def __init__(self):
        """Initialize the NeuralNetworkExamples class."""
        self.epsilon = 1e-15
        
    def single_neuron_regression_relu(self, visualize: bool = True) -> Tuple[float, float]:
        """
        Demonstrate single neuron regression with ReLU activation.
        
        This example shows how a single neuron can learn non-linear relationships
        through the use of activation functions like ReLU.
        
        Mathematical Model:
            y = ReLU(w * x + b)
            where ReLU(z) = max(0, z)
        
        Args:
            visualize: Whether to create visualization plots
            
        Returns:
            Tuple[float, float]: Final learned parameters (w, b)
            
        Example:
            >>> nn = NeuralNetworkExamples()
            >>> w, b = nn.single_neuron_regression_relu()
            >>> print(f"Learned parameters: w={w:.4f}, b={b:.4f}")
        """
        print("=" * 50)
        print("SINGLE NEURON REGRESSION WITH ReLU")
        print("=" * 50)
        
        # Generate synthetic data: house sizes and prices
        np.random.seed(0)
        x = np.linspace(500, 3500, 50)  # house sizes
        true_w, true_b = 0.3, 50
        noise = np.random.normal(0, 30, size=x.shape)
        y = np.maximum(true_w * x + true_b + noise, 0)  # true prices, clipped at 0

        # Initialize parameters
        w, b = 0.1, 0.0
        learning_rate = 1e-7

        # Training loop (simple gradient descent)
        for epoch in range(1000):
            y_pred = np.maximum(w * x + b, 0)  # ReLU activation
            error = y_pred - y
            grad_w = np.mean(error * (x * (y_pred > 0)))
            grad_b = np.mean(error * (y_pred > 0))
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

        plt.figure()
        plt.scatter(x, y, label='Data')
        plt.plot(x, np.maximum(w * x + b, 0), color='red', label='Single Neuron Fit')
        plt.xlabel('House Size')
        plt.ylabel('Price')
        plt.legend()
        plt.title('Single Neuron Regression with ReLU')
        plt.show()

        plt.figure()
        plt.scatter(x, y, label='Data')
        plt.plot(x, np.maximum(w * x + b, 0), color='red', label='Single Neuron Fit')
        plt.xlabel('House Size')
        plt.ylabel('Price')
        plt.legend()
        plt.title('Single Neuron Regression with ReLU')
        plt.show()

        return w, b
    
    def two_layer_neural_network(self, visualize: bool = True) -> Tuple[np.ndarray, float, np.ndarray, float]:
        """
        Demonstrate a two-layer neural network with ReLU activation.
        
        This example shows how multiple layers can learn more complex patterns
        by composing non-linear transformations.
        
        Architecture:
            Input -> Linear -> ReLU -> Linear -> Output
        
        Args:
            visualize: Whether to create visualization plots
            
        Returns:
            Tuple containing learned parameters (W1, b1, W2, b2)
        """
        print("\n" + "=" * 50)
        print("TWO-LAYER NEURAL NETWORK")
        print("=" * 50)
        
        # Generate synthetic data
        np.random.seed(1)
        X = np.random.rand(100, 4)  # 100 houses, 4 features each
        
        # True underlying relationship
        true_w1 = np.array([2.0, 1.5, 0.5, 1.0])
        true_b1 = 0.5
        hidden = np.maximum(X @ true_w1 + true_b1, 0)  # First layer (ReLU)
        true_w2 = 3.0
        true_b2 = 2.0
        y = true_w2 * hidden + true_b2 + np.random.normal(0, 0.5, size=hidden.shape)
        
        print(f"Data generated: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"True parameters: W1={true_w1}, b1={true_b1}, W2={true_w2}, b2={true_b2}")
        
        # Initialize parameters
        w1 = np.random.randn(4) * 0.1
        b1 = 0.0
        w2 = 1.0
        b2 = 0.0
        lr = 0.05
        
        print(f"Initial parameters: W1={w1}, b1={b1}, W2={w2}, b2={b2}")
        
        # Training loop
        losses = []
        for epoch in range(500):
            # Forward pass
            z1 = X @ w1 + b1  # First layer pre-activation
            a1 = np.maximum(z1, 0)  # ReLU activation
            y_pred = w2 * a1 + b2  # Output layer
            
            # Compute loss
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            
            # Backward pass (backpropagation)
            grad_y_pred = 2 * (y_pred - y) / len(y)
            
            # Gradients for output layer
            grad_w2 = np.sum(grad_y_pred * a1)
            grad_b2 = np.sum(grad_y_pred)
            
            # Gradients for hidden layer
            grad_a1 = grad_y_pred * w2
            grad_z1 = grad_a1 * (z1 > 0)  # ReLU derivative
            grad_w1 = grad_z1.T @ X
            grad_b1 = np.sum(grad_z1)
            
            # Parameter updates
            w1 -= lr * grad_w1
            b1 -= lr * grad_b1
            w2 -= lr * grad_w2
            b2 -= lr * grad_b2
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:3d}: Loss = {loss:.4f}")
        
        print(f"Final loss: {losses[-1]:.4f}")
        
        if visualize:
            self._plot_two_layer_results(y, y_pred, losses)
        
        return w1, b1, w2, b2
    
    def _plot_two_layer_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               losses: List[float]):
        """Create visualization plots for two-layer network."""
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
    
    def fully_connected_network(self, visualize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Demonstrate a fully-connected neural network with multiple hidden neurons.
        
        This example shows how multiple neurons in each layer can learn
        different features and representations of the input data.
        
        Architecture:
            Input -> Linear -> ReLU -> Linear -> Output
            (4)      (4->3)    (3)     (3->1)    (1)
        
        Args:
            visualize: Whether to create visualization plots
            
        Returns:
            Tuple containing learned parameters (W1, b1, W2, b2)
        """
        print("\n" + "=" * 50)
        print("FULLY-CONNECTED NEURAL NETWORK")
        print("=" * 50)
        
        # Generate synthetic data
        np.random.seed(42)
        X = np.random.rand(200, 4)
        
        # True underlying relationship with multiple hidden neurons
        true_W1 = np.array([[1.2, -0.7, 0.5, 2.0],
                           [0.3, 1.5, -1.0, 0.7],
                           [2.0, 0.1, 0.3, -0.5]])  # 3 hidden neurons
        true_b1 = np.array([0.5, -0.2, 0.1])
        H = np.maximum(X @ true_W1.T + true_b1, 0)  # Hidden layer (ReLU)
        true_W2 = np.array([1.0, -2.0, 0.5])
        true_b2 = 0.3
        y = H @ true_W2 + true_b2 + np.random.normal(0, 0.2, size=H.shape[0])
        
        print(f"Data generated: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Network architecture: {X.shape[1]} -> 3 -> 1")
        
        # Initialize parameters
        W1 = np.random.randn(3, 4) * 0.1
        b1 = np.zeros(3)
        W2 = np.random.randn(3) * 0.1
        b2 = 0.0
        lr = 0.05
        
        # Training loop
        losses = []
        for epoch in range(600):
            # Forward pass
            Z1 = X @ W1.T + b1  # First layer pre-activation
            A1 = np.maximum(Z1, 0)  # ReLU activation
            y_pred = A1 @ W2 + b2  # Output layer
            
            # Compute loss
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            
            # Backward pass (backpropagation)
            grad_y_pred = 2 * (y_pred - y) / len(y)
            
            # Gradients for output layer
            grad_W2 = A1.T @ grad_y_pred
            grad_b2 = np.sum(grad_y_pred)
            
            # Gradients for hidden layer
            grad_A1 = np.outer(grad_y_pred, W2)
            grad_Z1 = grad_A1 * (Z1 > 0)  # ReLU derivative
            grad_W1 = grad_Z1.T @ X
            grad_b1 = np.sum(grad_Z1, axis=0)
            
            # Parameter updates
            W1 -= lr * grad_W1
            b1 -= lr * grad_b1
            W2 -= lr * grad_W2
            b2 -= lr * grad_b2
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch:3d}: Loss = {loss:.4f}")
        
        print(f"Final loss: {losses[-1]:.4f}")
        
        if visualize:
            self._plot_fully_connected_results(y, y_pred, losses)
        
        return W1, b1, W2, b2
    
    def _plot_fully_connected_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                    losses: List[float]):
        """Create visualization plots for fully-connected network."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Predictions vs True values
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Fully-Connected Neural Network Predictions')
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
    
    def vectorization_comparison(self) -> float:
        """
        Demonstrate the efficiency of vectorized operations vs for-loops.
        
        This example shows how vectorization can dramatically improve
        computational efficiency in neural networks.
        
        Returns:
            float: Maximum difference between loop and vectorized results
        """
        print("\n" + "=" * 50)
        print("VECTORIZATION COMPARISON")
        print("=" * 50)
        
        # Generate test data
        np.random.seed(0)
        X = np.random.rand(1000, 4)
        W = np.random.randn(3, 4)
        b = np.random.randn(3)
        
        print(f"Matrix dimensions: X={X.shape}, W={W.shape}, b={b.shape}")
        
        # For-loop implementation (slow)
        print("Computing with for-loops...")
        outputs_loop = np.zeros((1000, 3))
        for i in range(1000):
            for j in range(3):
                outputs_loop[i, j] = np.dot(W[j], X[i]) + b[j]
        
        # Vectorized implementation (fast)
        print("Computing with vectorization...")
        outputs_vec = X @ W.T + b  # shape: (1000, 3)
        
        # Compare results
        max_diff = np.abs(outputs_loop - outputs_vec).max()
        print(f"Maximum difference: {max_diff:.2e}")
        print("Vectorized and loop results are identical!" if max_diff < 1e-10 else "Results differ!")
        
        return max_diff
    
    def deep_neural_network(self, visualize: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Demonstrate a deep neural network with multiple hidden layers.
        
        This example shows how deep networks can learn hierarchical features
        and complex non-linear relationships in data.
        
        Architecture:
            Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
            (4)      (4->5)    (5)     (5->3)    (3)     (3->1)    (1)
        
        Args:
            visualize: Whether to create visualization plots
            
        Returns:
            Tuple containing learned parameters (weights, biases)
        """
        print("\n" + "=" * 50)
        print("DEEP NEURAL NETWORK")
        print("=" * 50)
        
        # Generate synthetic data
        np.random.seed(123)
        X = np.random.rand(300, 4)
        
        # True underlying relationship with multiple layers
        true_W1 = np.random.randn(5, 4)
        true_b1 = np.random.randn(5)
        true_W2 = np.random.randn(3, 5)
        true_b2 = np.random.randn(3)
        true_W3 = np.random.randn(3)
        true_b3 = 0.5
        
        # Forward pass through true network
        H1 = np.maximum(X @ true_W1.T + true_b1, 0)
        H2 = np.maximum(H1 @ true_W2.T + true_b2, 0)
        y = H2 @ true_W3 + true_b3 + np.random.normal(0, 0.2, size=H2.shape[0])
        
        print(f"Data generated: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Network architecture: {X.shape[1]} -> 5 -> 3 -> 1")
        
        # Initialize parameters
        W1 = np.random.randn(5, 4) * 0.1
        b1 = np.zeros(5)
        W2 = np.random.randn(3, 5) * 0.1
        b2 = np.zeros(3)
        W3 = np.random.randn(3) * 0.1
        b3 = 0.0
        lr = 0.03
        
        # Training loop
        losses = []
        for epoch in range(800):
            # Forward pass
            Z1 = X @ W1.T + b1
            A1 = np.maximum(Z1, 0)
            Z2 = A1 @ W2.T + b2
            A2 = np.maximum(Z2, 0)
            y_pred = A2 @ W3 + b3
            
            # Compute loss
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            
            # Backward pass (backpropagation)
            grad_y_pred = 2 * (y_pred - y) / len(y)
            
            # Gradients for output layer
            grad_W3 = A2.T @ grad_y_pred
            grad_b3 = np.sum(grad_y_pred)
            
            # Gradients for second hidden layer
            grad_A2 = np.outer(grad_y_pred, W3)
            grad_Z2 = grad_A2 * (Z2 > 0)
            grad_W2 = grad_Z2.T @ A1
            grad_b2 = np.sum(grad_Z2, axis=0)
            
            # Gradients for first hidden layer
            grad_A1 = grad_Z2 @ W2
            grad_Z1 = grad_A1 * (Z1 > 0)
            grad_W1 = grad_Z1.T @ X
            grad_b1 = np.sum(grad_Z1, axis=0)
            
            # Parameter updates
            W1 -= lr * grad_W1
            b1 -= lr * grad_b1
            W2 -= lr * grad_W2
            b2 -= lr * grad_b2
            W3 -= lr * grad_W3
            b3 -= lr * grad_b3
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch:3d}: Loss = {loss:.4f}")
        
        print(f"Final loss: {losses[-1]:.4f}")
        
        if visualize:
            self._plot_deep_network_results(y, y_pred, losses)
        
        weights = [W1, W2, W3]
        biases = [b1, b2, b3]
        return weights, biases
    
    def _plot_deep_network_results(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 losses: List[float]):
        """Create visualization plots for deep network."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Predictions vs True values
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect prediction')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Deep Neural Network Predictions')
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
    
    def activation_functions_comparison(self):
        """
        Compare different activation functions and their properties.
        
        This visualization helps understand the characteristics of different
        activation functions used in neural networks.
        """
        print("\n" + "=" * 50)
        print("ACTIVATION FUNCTIONS COMPARISON")
        print("=" * 50)
        
        z = np.linspace(-4, 4, 200)
        
        # Define activation functions
        relu = np.maximum(0, z)
        sigmoid = 1 / (1 + np.exp(-z))
        tanh = np.tanh(z)
        leaky_relu = np.where(z > 0, z, 0.1 * z)
        gelu = 0.5 * z * (1 + erf(z / np.sqrt(2)))
        softplus = np.log(1 + np.exp(z))
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot activation functions
        plt.subplot(2, 1, 1)
        plt.plot(z, relu, label='ReLU', linewidth=2)
        plt.plot(z, sigmoid, label='Sigmoid', linewidth=2)
        plt.plot(z, tanh, label='Tanh', linewidth=2)
        plt.plot(z, leaky_relu, label='Leaky ReLU', linewidth=2)
        plt.plot(z, gelu, label='GELU', linewidth=2)
        plt.plot(z, softplus, label='Softplus', linewidth=2)
        plt.legend()
        plt.title('Common Activation Functions')
        plt.xlabel('Input z')
        plt.ylabel('Activation')
        plt.grid(True, alpha=0.3)
        
        # Plot derivatives
        plt.subplot(2, 1, 2)
        plt.plot(z, np.where(z > 0, 1, 0), label='ReLU\'', linewidth=2)
        plt.plot(z, sigmoid * (1 - sigmoid), label='Sigmoid\'', linewidth=2)
        plt.plot(z, 1 - tanh**2, label='Tanh\'', linewidth=2)
        plt.plot(z, np.where(z > 0, 1, 0.1), label='Leaky ReLU\'', linewidth=2)
        plt.plot(z, 0.5 * (1 + erf(z / np.sqrt(2))) + 0.5 * z * np.exp(-z**2/2) / np.sqrt(2*np.pi), 
                label='GELU\'', linewidth=2)
        plt.plot(z, sigmoid, label='Softplus\'', linewidth=2)
        plt.legend()
        plt.title('Derivatives of Activation Functions')
        plt.xlabel('Input z')
        plt.ylabel('Derivative')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print properties
        print("Activation Function Properties:")
        print("- ReLU: Range [0,∞), computationally efficient, helps with vanishing gradients")
        print("- Sigmoid: Range (0,1), smooth, can suffer from vanishing gradients")
        print("- Tanh: Range (-1,1), zero-centered, often preferred over sigmoid")
        print("- Leaky ReLU: Range (-∞,∞), prevents dying ReLU problem")
        print("- GELU: Range (-∞,∞), smooth approximation of ReLU, used in transformers")
        print("- Softplus: Range (0,∞), smooth approximation of ReLU")
    
    def kernel_vs_neural_network(self):
        """
        Compare kernel methods (SVM) with neural networks for feature learning.
        
        This example demonstrates the difference between fixed feature maps
        (kernels) and learned feature maps (neural networks).
        """
        print("\n" + "=" * 50)
        print("KERNEL vs NEURAL NETWORK FEATURE LEARNING")
        print("=" * 50)
        
        # Generate non-linear dataset (circles)
        X, y = make_circles(n_samples=300, factor=0.5, noise=0.1, random_state=0)
        
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print("Non-linear classification problem (concentric circles)")
        
        # Train SVM with RBF kernel (fixed feature map)
        print("\nTraining SVM with RBF kernel...")
        svm = SVC(kernel='rbf', gamma=2)
        svm.fit(X, y)
        svm_score = svm.score(X, y)
        print(f"SVM accuracy: {svm_score:.4f}")
        
        # Train neural network (learned feature map)
        print("\nTraining neural network...")
        mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', 
                           max_iter=2000, random_state=0)
        mlp.fit(X, y)
        mlp_score = mlp.score(X, y)
        print(f"Neural network accuracy: {mlp_score:.4f}")
        
        # Create decision boundary visualization
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
        
        # SVM predictions
        Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # Neural network predictions
        Z_mlp = mlp.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # SVM decision boundary
        ax1.contourf(xx, yy, Z_svm, alpha=0.3, cmap='RdYlBu')
        ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='k', s=50)
        ax1.set_title(f'SVM with RBF Kernel\n(Fixed Feature Map)\nAccuracy: {svm_score:.4f}')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        
        # Neural network decision boundary
        ax2.contourf(xx, yy, Z_mlp, alpha=0.3, cmap='RdYlBu')
        ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolor='k', s=50)
        ax2.set_title(f'Neural Network\n(Learned Feature Map)\nAccuracy: {mlp_score:.4f}')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        
        plt.tight_layout()
        plt.show()
        
        print("\nKey Differences:")
        print("- SVM: Uses fixed RBF kernel, feature map is predetermined")
        print("- Neural Network: Learns optimal feature representations")
        print("- Both can solve non-linear problems, but with different approaches")


def main():
    """
    Main function demonstrating all neural network examples.
    
    This function runs comprehensive examples of neural networks at different
    levels of complexity, from single neurons to deep architectures.
    """
    print("=" * 60)
    print("NEURAL NETWORKS: FROM SINGLE NEURONS TO DEEP ARCHITECTURES")
    print("=" * 60)
    
    # Initialize the examples
    nn_examples = NeuralNetworkExamples()
    
    # Run all examples
    try:
        # 1. Single neuron regression
        w, b = nn_examples.single_neuron_regression_relu()
        
        # 2. Two-layer network
        w1, b1, w2, b2 = nn_examples.two_layer_neural_network()
        
        # 3. Fully-connected network
        W1, b1, W2, b2 = nn_examples.fully_connected_network()
        
        # 4. Vectorization comparison
        max_diff = nn_examples.vectorization_comparison()
        
        # 5. Deep neural network
        weights, biases = nn_examples.deep_neural_network()
        
        # 6. Activation functions comparison
        nn_examples.activation_functions_comparison()
        
        # 7. Kernel vs neural network
        nn_examples.kernel_vs_neural_network()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
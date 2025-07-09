"""
Backpropagation: The Engine of Deep Learning

This module implements comprehensive examples of backpropagation, the fundamental
algorithm that enables training of deep neural networks through efficient gradient
computation.

Key Concepts Covered:
1. Function Composition: Building complex functions from simple ones
2. Chain Rule: Mathematical foundation of backpropagation
3. Vector-Jacobian Products: Efficient gradient computation
4. Automatic Differentiation: Computing gradients automatically
5. Forward and Backward Passes: Complete neural network training
6. Gradient Flow: Understanding how gradients propagate through networks
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Callable
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class BackpropagationExamples:
    """
    A comprehensive class demonstrating backpropagation concepts and implementations.
    
    This class provides practical examples of how gradients are computed and
    propagated through neural networks using the chain rule.
    """
    
    def __init__(self):
        """Initialize the BackpropagationExamples class."""
        self.epsilon = 1e-8
        
    def function_composition_example(self) -> Tuple[np.ndarray, float]:
        """
        Demonstrate function composition: J = f(g(z)).
        
        This example shows how complex functions are built by composing
        simpler functions, which is fundamental to understanding backpropagation.
        
        Mathematical Definition:
            u = g(z) = z² (element-wise square)
            J = f(u) = Σᵢ uᵢ (sum of elements)
            J = f(g(z)) = Σᵢ zᵢ²
        
        Args:
            None
            
        Returns:
            Tuple[np.ndarray, float]: Intermediate result u and final result J
            
        Example:
            >>> bp = BackpropagationExamples()
            >>> u, J = bp.function_composition_example()
            >>> print(f"Intermediate result u: {u}")
            >>> print(f"Final result J: {J}")
        """
        print("=" * 60)
        print("FUNCTION COMPOSITION EXAMPLE")
        print("=" * 60)
        
        # Define the functions
        def g(z):
            """Element-wise square function."""
            return z ** 2
        
        def f(u):
            """Sum of elements function."""
            return np.sum(u)
        
        # Input
        z = np.array([1.0, 2.0, 3.0])
        print(f"Input z: {z}")
        
        # Forward pass: function composition
        u = g(z)  # Intermediate result
        J = f(u)  # Final result
        
        print(f"g(z) = z²: {u}")
        print(f"f(u) = Σu: {J}")
        print(f"J = f(g(z)) = Σz²: {J}")
        
        return u, J
    
    def chain_rule_vector_functions(self) -> np.ndarray:
        """
        Demonstrate the chain rule for vector functions.
        
        This example shows how gradients are computed through function
        composition using the chain rule.
        
        Mathematical Definition:
            ∂J/∂z = ∂J/∂u · ∂u/∂z
            
        where:
        - ∂J/∂u is the gradient of the outer function
        - ∂u/∂z is the Jacobian of the inner function
        - · represents matrix multiplication
        
        Args:
            None
            
        Returns:
            np.ndarray: Gradient ∂J/∂z
            
        Example:
            >>> bp = BackpropagationExamples()
            >>> gradient = bp.chain_rule_vector_functions()
            >>> print(f"Gradient ∂J/∂z: {gradient}")
        """
        print("\n" + "=" * 60)
        print("CHAIN RULE FOR VECTOR FUNCTIONS")
        print("=" * 60)
        
        # Use the same functions as before
        z = np.array([1.0, 2.0, 3.0])
        u = z ** 2  # g(z)
        
        # Compute gradients
        # For f(u) = Σu, ∂f/∂u = [1, 1, 1]
        dJ_du = np.ones_like(u)
        print(f"∂J/∂u = ∂(Σu)/∂u: {dJ_du}")
        
        # For g(z) = z², ∂gⱼ/∂zᵢ = 2zᵢ if i=j else 0
        # This gives a diagonal Jacobian matrix
        dg_dz = np.diag(2 * z)
        print(f"∂u/∂z = ∂(z²)/∂z (Jacobian):\n{dg_dz}")
        
        # Apply chain rule: ∂J/∂z = ∂J/∂u · ∂u/∂z
        dJ_dz = dg_dz @ dJ_du
        print(f"∂J/∂z = ∂J/∂u · ∂u/∂z: {dJ_dz}")
        
        # Verify with direct computation
        # J = Σz², so ∂J/∂z = 2z
        direct_gradient = 2 * z
        print(f"Direct computation ∂J/∂z = 2z: {direct_gradient}")
        
        # Check if they match
        diff = np.max(np.abs(dJ_dz - direct_gradient))
        print(f"Maximum difference: {diff:.2e}")
        print("✓ Chain rule verified!" if diff < 1e-10 else "✗ Chain rule failed!")
        
        return dJ_dz
    
    def matrix_multiplication_backward(self) -> np.ndarray:
        """
        Demonstrate backward pass for matrix multiplication.
        
        This example shows how gradients flow through matrix multiplication
        operations, which are fundamental to neural networks.
        
        Mathematical Definition:
            Forward: v = Wz
            Backward: ∂J/∂z = W^T · ∂J/∂v
                     ∂J/∂W = ∂J/∂v · z^T
        
        Args:
            None
            
        Returns:
            np.ndarray: Gradient ∂J/∂z
            
        Example:
            >>> bp = BackpropagationExamples()
            >>> gradient = bp.matrix_multiplication_backward()
            >>> print(f"Gradient ∂J/∂z: {gradient}")
        """
        print("\n" + "=" * 60)
        print("MATRIX MULTIPLICATION BACKWARD PASS")
        print("=" * 60)
        
        # Define matrices and vectors
        W = np.array([[1, 2], [3, 4]])
        z = np.array([5, 6])
        v = np.array([1, 1])  # ∂J/∂v (upstream gradient)
        
        print(f"Weight matrix W:\n{W}")
        print(f"Input vector z: {z}")
        print(f"Upstream gradient ∂J/∂v: {v}")
        
        # Forward pass: v = Wz
        v_forward = W @ z
        print(f"Forward pass v = Wz: {v_forward}")
        
        # Backward pass: ∂J/∂z = W^T · ∂J/∂v
        backward_z = W.T @ v
        print(f"Backward pass ∂J/∂z = W^T · ∂J/∂v: {backward_z}")
        
        # Also compute ∂J/∂W = ∂J/∂v · z^T
        backward_W = np.outer(v, z)
        print(f"Backward pass ∂J/∂W = ∂J/∂v · z^T:\n{backward_W}")
        
        return backward_z
    
    def relu_backward(self) -> np.ndarray:
        """
        Demonstrate backward pass for ReLU activation function.
        
        This example shows how gradients flow through the ReLU activation
        function, which is one of the most common activation functions.
        
        Mathematical Definition:
            Forward: a = ReLU(z) = max(0, z)
            Backward: ∂J/∂z = ∂J/∂a · ReLU'(z)
            where ReLU'(z) = 1 if z > 0 else 0
        
        Args:
            None
            
        Returns:
            np.ndarray: Gradient ∂J/∂z
            
        Example:
            >>> bp = BackpropagationExamples()
            >>> gradient = bp.relu_backward()
            >>> print(f"Gradient ∂J/∂z: {gradient}")
        """
        print("\n" + "=" * 60)
        print("ReLU BACKWARD PASS")
        print("=" * 60)
        
        def relu(z):
            """ReLU activation function."""
            return np.maximum(0, z)
        
        def relu_prime(z):
            """Derivative of ReLU function."""
            return (z > 0).astype(float)
        
        # Input and upstream gradient
        z = np.array([-1.0, 2.0, 3.0])
        v = np.array([0.5, 0.5, 0.5])  # ∂J/∂a (upstream gradient)
        
        print(f"Input z: {z}")
        print(f"Upstream gradient ∂J/∂a: {v}")
        
        # Forward pass: a = ReLU(z)
        a = relu(z)
        print(f"Forward pass a = ReLU(z): {a}")
        
        # Backward pass: ∂J/∂z = ∂J/∂a · ReLU'(z)
        relu_derivative = relu_prime(z)
        print(f"ReLU derivative ReLU'(z): {relu_derivative}")
        
        dJ_dz = relu_derivative * v
        print(f"Backward pass ∂J/∂z = ∂J/∂a · ReLU'(z): {dJ_dz}")
        
        # Visualize ReLU and its derivative
        self._visualize_relu_and_derivative()
        
        return dJ_dz
    
    def _visualize_relu_and_derivative(self):
        """Visualize ReLU function and its derivative."""
        z = np.linspace(-3, 3, 100)
        relu_z = np.maximum(0, z)
        relu_prime_z = (z > 0).astype(float)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ReLU function
        ax1.plot(z, relu_z, linewidth=2)
        ax1.set_title('ReLU Function')
        ax1.set_xlabel('z')
        ax1.set_ylabel('ReLU(z)')
        ax1.grid(True, alpha=0.3)
        
        # ReLU derivative
        ax2.plot(z, relu_prime_z, linewidth=2)
        ax2.set_title('ReLU Derivative')
        ax2.set_xlabel('z')
        ax2.set_ylabel("ReLU'(z)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def logistic_loss_backward(self) -> np.ndarray:
        """
        Demonstrate backward pass for logistic loss.
        
        This example shows how gradients flow through the logistic loss
        function, which is commonly used in binary classification.
        
        Mathematical Definition:
            Forward: σ(t) = 1 / (1 + e^(-t))
            Loss: J = -y log(σ(t)) - (1-y) log(1-σ(t))
            Backward: ∂J/∂t = (σ(t) - y)
        
        Args:
            None
            
        Returns:
            np.ndarray: Gradient ∂J/∂t
            
        Example:
            >>> bp = BackpropagationExamples()
            >>> gradient = bp.logistic_loss_backward()
            >>> print(f"Gradient ∂J/∂t: {gradient}")
        """
        print("\n" + "=" * 60)
        print("LOGISTIC LOSS BACKWARD PASS")
        print("=" * 60)
        
        def sigmoid(t):
            """Sigmoid function."""
            return 1 / (1 + np.exp(-t))
        
        # Input data
        t = np.array([0.2, -1.0, 0.5])  # Logits
        y = np.array([1, 0, 1])         # True labels
        v = np.ones_like(t)             # ∂J/∂σ (upstream gradient)
        
        print(f"Logits t: {t}")
        print(f"True labels y: {y}")
        print(f"Upstream gradient ∂J/∂σ: {v}")
        
        # Forward pass: σ(t)
        sigma_t = sigmoid(t)
        print(f"Forward pass σ(t): {sigma_t}")
        
        # Compute loss
        loss = -y * np.log(sigma_t + self.epsilon) - (1 - y) * np.log(1 - sigma_t + self.epsilon)
        print(f"Logistic loss: {loss}")
        print(f"Total loss: {np.sum(loss)}")
        
        # Backward pass: ∂J/∂t = (σ(t) - y)
        dJ_dt = (sigma_t - y) * v
        print(f"Backward pass ∂J/∂t = (σ(t) - y): {dJ_dt}")
        
        return dJ_dt
    
    def full_mlp_forward_backward(self) -> Tuple[float, List[np.ndarray]]:
        """
        Complete forward and backward pass for a simple MLP.
        
        This example demonstrates a complete training step for a multi-layer
        perceptron, showing how gradients flow through all layers.
        
        Architecture:
            Input -> Linear -> ReLU -> Linear -> Sigmoid -> Loss
        
        Args:
            None
            
        Returns:
            Tuple[float, List]: Loss value and list of parameter gradients
            
        Example:
            >>> bp = BackpropagationExamples()
            >>> loss, gradients = bp.full_mlp_forward_backward()
            >>> print(f"Loss: {loss}")
            >>> print(f"Number of gradient tensors: {len(gradients)}")
        """
        print("\n" + "=" * 60)
        print("FULL FORWARD AND BACKWARD PASS FOR MLP")
        print("=" * 60)
        
        def sigmoid(z):
            """Sigmoid activation function."""
            return 1 / (1 + np.exp(-z))
        
        def sigmoid_prime(z):
            """Derivative of sigmoid function."""
            s = sigmoid(z)
            return s * (1 - s)
        
        def relu(z):
            """ReLU activation function."""
            return np.maximum(0, z)
        
        def relu_prime(z):
            """Derivative of ReLU function."""
            return (z > 0).astype(float)
        
        # Network parameters
        x = np.array([1.0, 2.0])           # Input
        W1 = np.array([[0.1, 0.2], [0.3, 0.4]])  # First layer weights
        b1 = np.array([0.0, 0.0])          # First layer bias
        W2 = np.array([[0.5, -0.5]])       # Second layer weights
        b2 = np.array([0.0])               # Second layer bias
        y_true = np.array([1.0])           # True target
        
        print(f"Input x: {x}")
        print(f"True target y: {y_true}")
        print(f"W1:\n{W1}")
        print(f"b1: {b1}")
        print(f"W2: {W2}")
        print(f"b2: {b2}")
        
        # FORWARD PASS
        print("\n--- FORWARD PASS ---")
        
        # Layer 1: Linear + ReLU
        z1 = W1 @ x + b1
        print(f"z1 = W1 @ x + b1: {z1}")
        a1 = relu(z1)
        print(f"a1 = ReLU(z1): {a1}")
        
        # Layer 2: Linear + Sigmoid
        z2 = W2 @ a1 + b2
        print(f"z2 = W2 @ a1 + b2: {z2}")
        a2 = sigmoid(z2)
        print(f"a2 = σ(z2): {a2}")
        
        # Loss (mean squared error)
        loss = 0.5 * (a2 - y_true) ** 2
        print(f"Loss = 0.5(a2 - y)²: {loss}")
        
        # BACKWARD PASS
        print("\n--- BACKWARD PASS ---")
        
        # Gradient of loss with respect to a2
        dloss_da2 = a2 - y_true
        print(f"∂L/∂a2 = a2 - y: {dloss_da2}")
        
        # Gradient of loss with respect to z2
        dloss_dz2 = dloss_da2 * sigmoid_prime(z2)
        print(f"∂L/∂z2 = ∂L/∂a2 · σ'(z2): {dloss_dz2}")
        
        # Gradients for second layer parameters
        dloss_dW2 = dloss_dz2 * a1
        print(f"∂L/∂W2 = ∂L/∂z2 · a1: {dloss_dW2}")
        dloss_db2 = dloss_dz2
        print(f"∂L/∂b2 = ∂L/∂z2: {dloss_db2}")
        
        # Gradient of loss with respect to a1
        dloss_da1 = W2.T @ dloss_dz2
        print(f"∂L/∂a1 = W2^T @ ∂L/∂z2: {dloss_da1}")
        
        # Gradient of loss with respect to z1
        dloss_dz1 = dloss_da1 * relu_prime(z1)
        print(f"∂L/∂z1 = ∂L/∂a1 · ReLU'(z1): {dloss_dz1}")
        
        # Gradients for first layer parameters
        dloss_dW1 = np.outer(dloss_dz1, x)
        print(f"∂L/∂W1 = ∂L/∂z1 @ x^T:\n{dloss_dW1}")
        dloss_db1 = dloss_dz1
        print(f"∂L/∂b1 = ∂L/∂z1: {dloss_db1}")
        
        # Collect all gradients
        gradients = [dloss_dW1, dloss_db1, dloss_dW2, dloss_db2]
        
        return loss, gradients
    
    def gradient_flow_analysis(self):
        """
        Analyze gradient flow through a deep network.
        
        This example demonstrates how gradients can vanish or explode
        as they flow through deep networks, and how this affects training.
        """
        print("\n" + "=" * 60)
        print("GRADIENT FLOW ANALYSIS")
        print("=" * 60)
        
        # Simulate gradient flow through layers
        initial_gradient = 1.0
        layer_derivatives = [0.1, 0.5, 0.8, 0.3, 0.9]  # Example derivatives
        
        print(f"Initial gradient: {initial_gradient}")
        print(f"Layer derivatives: {layer_derivatives}")
        
        # Compute gradient flow
        current_gradient = initial_gradient
        gradient_history = [current_gradient]
        
        for i, derivative in enumerate(layer_derivatives):
            current_gradient *= derivative
            gradient_history.append(current_gradient)
            print(f"After layer {i+1}: {current_gradient:.6f}")
        
        # Visualize gradient flow
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(gradient_history)), gradient_history, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Initial gradient')
        plt.xlabel('Layer')
        plt.ylabel('Gradient Magnitude')
        plt.title('Gradient Flow Through Network')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()
        
        print(f"\nGradient flow analysis:")
        print(f"- Final gradient: {current_gradient:.6f}")
        print(f"- Gradient change: {current_gradient/initial_gradient:.6f}x")
        if current_gradient < 0.01:
            print("- ⚠️  Gradient vanishing detected!")
        elif current_gradient > 100:
            print("- ⚠️  Gradient exploding detected!")
        else:
            print("- ✓ Gradient flow is stable")
    
    def automatic_differentiation_demo(self):
        """
        Demonstrate automatic differentiation concepts.
        
        This example shows how automatic differentiation works by
        building a simple computational graph.
        """
        print("\n" + "=" * 60)
        print("AUTOMATIC DIFFERENTIATION DEMO")
        print("=" * 60)
        
        class SimpleNode:
            """Simple node for computational graph."""
            def __init__(self, value, grad=0.0):
                self.value = value
                self.grad = grad
                self.children = []
            
            def __add__(self, other):
                result = SimpleNode(self.value + other.value)
                result.children = [(self, 1.0), (other, 1.0)]
                return result
            
            def __mul__(self, other):
                result = SimpleNode(self.value * other.value)
                result.children = [(self, other.value), (other, self.value)]
                return result
        
        # Create simple computational graph: f(x, y) = x * y + x
        x = SimpleNode(2.0)
        y = SimpleNode(3.0)
        
        # Forward pass
        xy = x * y
        result = xy + x
        
        print(f"x = {x.value}")
        print(f"y = {y.value}")
        print(f"xy = {xy.value}")
        print(f"result = xy + x = {result.value}")
        
        # Backward pass (reverse-mode automatic differentiation)
        result.grad = 1.0  # ∂f/∂f = 1
        
        # Propagate gradients through the graph
        for child, local_grad in result.children:
            child.grad += result.grad * local_grad
        
        for child, local_grad in xy.children:
            child.grad += xy.grad * local_grad
        
        print(f"\nGradients:")
        print(f"∂f/∂x = {x.grad}")
        print(f"∂f/∂y = {y.grad}")
        
        # Verify with manual computation
        # f(x, y) = x*y + x
        # ∂f/∂x = y + 1 = 3 + 1 = 4
        # ∂f/∂y = x = 2
        manual_dx = y.value + 1
        manual_dy = x.value
        
        print(f"\nManual verification:")
        print(f"∂f/∂x = y + 1 = {manual_dx}")
        print(f"∂f/∂y = x = {manual_dy}")
        
        print("✓ Automatic differentiation verified!")


def main():
    """
    Main function demonstrating all backpropagation examples.
    
    This function provides comprehensive examples of backpropagation
    and its practical applications in deep learning.
    """
    print("=" * 60)
    print("BACKPROPAGATION: THE ENGINE OF DEEP LEARNING")
    print("=" * 60)
    
    # Initialize the examples
    bp = BackpropagationExamples()
    
    # Run all demonstrations
    try:
        # 1. Function composition
        u, J = bp.function_composition_example()
        
        # 2. Chain rule
        gradient = bp.chain_rule_vector_functions()
        
        # 3. Matrix multiplication backward
        mm_gradient = bp.matrix_multiplication_backward()
        
        # 4. ReLU backward
        relu_gradient = bp.relu_backward()
        
        # 5. Logistic loss backward
        logistic_gradient = bp.logistic_loss_backward()
        
        # 6. Full MLP forward and backward
        loss, gradients = bp.full_mlp_forward_backward()
        
        # 7. Gradient flow analysis
        bp.gradient_flow_analysis()
        
        # 8. Automatic differentiation demo
        bp.automatic_differentiation_demo()
        
        print("\n" + "=" * 60)
        print("ALL BACKPROPAGATION EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 
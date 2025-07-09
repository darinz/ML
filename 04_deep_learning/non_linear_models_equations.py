"""
Non-Linear Models: Mathematical Foundations and Implementations

This module implements the core mathematical concepts and algorithms for non-linear models
in deep learning, including loss functions, activation functions, and optimization techniques.

Key Concepts Covered:
1. Loss Functions: MSE, Binary Cross-Entropy, Categorical Cross-Entropy
2. Activation Functions: Sigmoid, Softmax, ReLU, Tanh
3. Optimization: Gradient Descent, Stochastic Gradient Descent, Mini-batch SGD
4. Non-linear Transformations: Function composition and chain rule applications
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class NonLinearModels:
    """
    A comprehensive class implementing non-linear models and their mathematical foundations.
    
    This class demonstrates how non-linear transformations enable neural networks to
    learn complex patterns and relationships in data.
    """
    
    def __init__(self):
        """Initialize the NonLinearModels class with default parameters."""
        self.epsilon = 1e-15  # Small constant to prevent numerical issues
        
    def mean_squared_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE) loss.
        
        MSE measures the average squared difference between predicted and true values.
        It's commonly used for regression problems and provides a smooth, differentiable
        loss function that penalizes large errors more heavily than small ones.
        
        Mathematical Formula:
            MSE = (1/n) * Σ(y_pred - y_true)²
        
        Args:
            y_true: True target values, shape (n_samples,) or (n_samples, 1)
            y_pred: Predicted values, shape (n_samples,) or (n_samples, 1)
            
        Returns:
            float: Mean squared error value
            
        Example:
            >>> model = NonLinearModels()
            >>> y_true = np.array([2, 4, 6])
            >>> y_pred = np.array([2.1, 3.9, 6.2])
            >>> mse = model.mean_squared_error(y_true, y_pred)
            >>> print(f"MSE: {mse:.4f}")
        """
        # Ensure inputs are numpy arrays and have the same shape
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")
        
        # Compute MSE: average of squared differences
        mse = np.mean(0.5 * (y_pred - y_true) ** 2)
        return float(mse)
    
    def mse_gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of MSE loss with respect to predictions.
        
        The gradient of MSE is used in backpropagation to update model parameters.
        
        Mathematical Formula:
            ∂MSE/∂y_pred = (y_pred - y_true)
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            np.ndarray: Gradient of MSE with respect to predictions
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        return y_pred - y_true
    
    def sigmoid(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the sigmoid (logistic) activation function.
        
        The sigmoid function maps any real number to the range (0, 1), making it
        useful for binary classification problems and as a smooth approximation
        of a step function.
        
        Mathematical Formula:
            σ(z) = 1 / (1 + e^(-z))
        
        Properties:
        - Range: (0, 1)
        - Symmetric around (0, 0.5)
        - Smooth and differentiable everywhere
        - Outputs can be interpreted as probabilities
        
        Args:
            z: Input value(s), can be scalar or array
            
        Returns:
            Sigmoid activation value(s)
            
        Example:
            >>> model = NonLinearModels()
            >>> z = np.array([-2, 0, 2])
            >>> sigmoid_values = model.sigmoid(z)
            >>> print(f"Sigmoid values: {sigmoid_values}")
        """
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the derivative of the sigmoid function.
        
        The derivative is used in backpropagation to compute gradients.
        
        Mathematical Formula:
            σ'(z) = σ(z) * (1 - σ(z))
        
        Args:
            z: Input value(s) or sigmoid output(s)
            
        Returns:
            Derivative value(s)
        """
        sigmoid_z = self.sigmoid(z)
        return sigmoid_z * (1 - sigmoid_z)
    
    def relu(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the Rectified Linear Unit (ReLU) activation function.
        
        ReLU is one of the most popular activation functions in deep learning
        due to its computational efficiency and ability to mitigate vanishing gradients.
        
        Mathematical Formula:
            ReLU(z) = max(0, z)
        
        Properties:
        - Range: [0, ∞)
        - Computationally efficient
        - Helps with vanishing gradient problem
        - Introduces sparsity (many neurons output 0)
        
        Args:
            z: Input value(s)
            
        Returns:
            ReLU activation value(s)
        """
        return np.maximum(0, z)
    
    def relu_derivative(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the derivative of the ReLU function.
        
        Mathematical Formula:
            ReLU'(z) = 1 if z > 0, 0 otherwise
        
        Args:
            z: Input value(s)
            
        Returns:
            Derivative value(s)
        """
        return np.where(z > 0, 1, 0)
    
    def tanh(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the hyperbolic tangent (tanh) activation function.
        
        Tanh is similar to sigmoid but maps to the range (-1, 1), making it
        zero-centered and often preferred for hidden layers.
        
        Mathematical Formula:
            tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
        
        Properties:
        - Range: (-1, 1)
        - Zero-centered (outputs are centered around 0)
        - Smooth and differentiable
        - Often preferred over sigmoid for hidden layers
        
        Args:
            z: Input value(s)
            
        Returns:
            Tanh activation value(s)
        """
        return np.tanh(z)
    
    def tanh_derivative(self, z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute the derivative of the tanh function.
        
        Mathematical Formula:
            tanh'(z) = 1 - tanh²(z)
        
        Args:
            z: Input value(s)
            
        Returns:
            Derivative value(s)
        """
        return 1 - np.tanh(z) ** 2
    
    def softmax(self, logits: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Compute the softmax function for multi-class classification.
        
        Softmax converts a vector of arbitrary real numbers into a probability
        distribution, making it ideal for multi-class classification problems.
        
        Mathematical Formula:
            softmax(z_i) = e^(z_i) / Σ(e^(z_j))
        
        Properties:
        - Outputs sum to 1 (probability distribution)
        - Preserves relative order of inputs
        - Numerically stable implementation
        - Used in final layer of classification networks
        
        Args:
            logits: Input logits, shape (n_samples, n_classes) or (n_classes,)
            axis: Axis along which to apply softmax
            
        Returns:
            Softmax probabilities
            
        Example:
            >>> model = NonLinearModels()
            >>> logits = np.array([2.0, 1.0, 0.1])
            >>> probs = model.softmax(logits)
            >>> print(f"Probabilities: {probs}")
            >>> print(f"Sum: {np.sum(probs):.6f}")  # Should be 1.0
        """
        # Numerical stability: subtract max before exponentiating
        logits_shifted = logits - np.max(logits, axis=axis, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        return exp_logits / np.sum(exp_logits, axis=axis, keepdims=True)
    
    def binary_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy (log loss) for binary classification.
        
        Binary cross-entropy measures the performance of a binary classification
        model by quantifying the difference between predicted probabilities and
        true binary labels.
        
        Mathematical Formula:
            BCE = -Σ[y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)]
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted probabilities (between 0 and 1)
            
        Returns:
            float: Binary cross-entropy loss
            
        Example:
            >>> model = NonLinearModels()
            >>> y_true = np.array([1, 0, 1, 0])
            >>> y_pred = np.array([0.9, 0.1, 0.8, 0.2])
            >>> bce = model.binary_cross_entropy(y_true, y_pred)
            >>> print(f"Binary Cross-Entropy: {bce:.4f}")
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute binary cross-entropy
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return float(bce)
    
    def categorical_cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute categorical cross-entropy for multi-class classification.
        
        Categorical cross-entropy is used when the target variable has more than
        two classes. It measures the difference between predicted class probabilities
        and true class labels.
        
        Mathematical Formula:
            CCE = -Σ y_true * log(y_pred)
        
        Args:
            y_true: True class labels (one-hot encoded)
            y_pred: Predicted class probabilities
            
        Returns:
            float: Categorical cross-entropy loss
        """
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        # Compute categorical cross-entropy
        cce = -np.sum(y_true * np.log(y_pred))
        return float(cce)
    
    def gradient_descent_update(self, theta: np.ndarray, gradient: np.ndarray, 
                               learning_rate: float) -> np.ndarray:
        """
        Perform a single gradient descent parameter update.
        
        Gradient descent is the most fundamental optimization algorithm in deep learning.
        It updates parameters in the direction opposite to the gradient to minimize
        the loss function.
        
        Mathematical Formula:
            θ_new = θ_old - α * ∇J(θ)
        
        where:
        - θ: parameters
        - α: learning rate
        - ∇J(θ): gradient of loss function with respect to parameters
        
        Args:
            theta: Current parameter values
            gradient: Gradient of loss with respect to parameters
            learning_rate: Step size for the update
            
        Returns:
            np.ndarray: Updated parameter values
            
        Example:
            >>> model = NonLinearModels()
            >>> theta = np.array([1.0, 2.0])
            >>> gradient = np.array([0.1, -0.2])
            >>> lr = 0.01
            >>> new_theta = model.gradient_descent_update(theta, gradient, lr)
            >>> print(f"Updated parameters: {new_theta}")
        """
        return theta - learning_rate * gradient
    
    def stochastic_gradient_descent(self, theta: np.ndarray, gradients: List[np.ndarray], 
                                   learning_rate: float) -> np.ndarray:
        """
        Perform stochastic gradient descent update using a batch of gradients.
        
        SGD is a variant of gradient descent that uses a subset (mini-batch) of
        the training data to estimate the gradient, making it more efficient for
        large datasets.
        
        Args:
            theta: Current parameter values
            gradients: List of gradients for each example in the batch
            learning_rate: Step size for the update
            
        Returns:
            np.ndarray: Updated parameter values
        """
        # Average gradients across the batch
        avg_gradient = np.mean(gradients, axis=0)
        return self.gradient_descent_update(theta, avg_gradient, learning_rate)
    
    def function_composition(self, x: float, functions: List[callable]) -> float:
        """
        Demonstrate function composition: f(g(h(x))).
        
        Function composition is fundamental to neural networks, where each layer
        applies a transformation to the output of the previous layer.
        
        Args:
            x: Input value
            functions: List of functions to compose (applied from right to left)
            
        Returns:
            float: Result of function composition
        """
        result = x
        for func in reversed(functions):
            result = func(result)
        return result
    
    def visualize_activation_functions(self):
        """
        Create visualizations of different activation functions and their derivatives.
        
        This helps understand the behavior and properties of different activation
        functions used in neural networks.
        """
        x = np.linspace(-5, 5, 1000)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Activation Functions and Their Derivatives', fontsize=16)
        
        # Sigmoid
        axes[0, 0].plot(x, self.sigmoid(x), label='Sigmoid', linewidth=2)
        axes[0, 0].plot(x, self.sigmoid_derivative(x), label='Derivative', linestyle='--')
        axes[0, 0].set_title('Sigmoid Function')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ReLU
        axes[0, 1].plot(x, self.relu(x), label='ReLU', linewidth=2)
        axes[0, 1].plot(x, self.relu_derivative(x), label='Derivative', linestyle='--')
        axes[0, 1].set_title('ReLU Function')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tanh
        axes[1, 0].plot(x, self.tanh(x), label='Tanh', linewidth=2)
        axes[1, 0].plot(x, self.tanh_derivative(x), label='Derivative', linestyle='--')
        axes[1, 0].set_title('Tanh Function')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Softmax (for a 3-class example)
        logits = np.array([x, np.zeros_like(x), -x])
        softmax_probs = self.softmax(logits, axis=0)
        axes[1, 1].plot(x, softmax_probs[0], label='Class 1', linewidth=2)
        axes[1, 1].plot(x, softmax_probs[1], label='Class 2', linewidth=2)
        axes[1, 1].plot(x, softmax_probs[2], label='Class 3', linewidth=2)
        axes[1, 1].set_title('Softmax Probabilities')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_gradient_descent(self, initial_theta: float = 2.0, 
                                   learning_rate: float = 0.1, 
                                   num_iterations: int = 20):
        """
        Demonstrate gradient descent optimization on a simple quadratic function.
        
        This example shows how gradient descent finds the minimum of a loss function
        by iteratively updating parameters in the direction of steepest descent.
        
        Args:
            initial_theta: Starting parameter value
            learning_rate: Learning rate for gradient descent
            num_iterations: Number of optimization iterations
        """
        # Define a simple quadratic loss function: J(θ) = θ²
        def loss_function(theta):
            return theta ** 2
        
        def loss_gradient(theta):
            return 2 * theta
        
        # Track optimization progress
        thetas = [initial_theta]
        losses = [loss_function(initial_theta)]
        
        # Perform gradient descent
        theta = initial_theta
        for i in range(num_iterations):
            gradient = loss_gradient(theta)
            theta = self.gradient_descent_update(theta, gradient, learning_rate)
            thetas.append(theta)
            losses.append(loss_function(theta))
        
        # Visualize optimization
        plt.figure(figsize=(12, 4))
        
        # Plot loss function and optimization path
        theta_range = np.linspace(-3, 3, 100)
        loss_range = [loss_function(t) for t in theta_range]
        
        plt.subplot(1, 2, 1)
        plt.plot(theta_range, loss_range, 'b-', label='Loss Function J(θ) = θ²')
        plt.plot(thetas, losses, 'ro-', label='Optimization Path')
        plt.xlabel('Parameter θ')
        plt.ylabel('Loss J(θ)')
        plt.title('Gradient Descent Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot convergence
        plt.subplot(1, 2, 2)
        plt.plot(losses, 'ro-')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Convergence')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Final parameter value: {theta:.6f}")
        print(f"Final loss: {losses[-1]:.6f}")
        print(f"True minimum: 0.0")


def main():
    """
    Main function demonstrating the usage of NonLinearModels class.
    
    This function provides comprehensive examples of all the implemented
    concepts and their practical applications.
    """
    print("=" * 60)
    print("NON-LONLINEAR MODELS: MATHEMATICAL FOUNDATIONS")
    print("=" * 60)
    
    # Initialize the model
    model = NonLinearModels()
    
    # Example 1: Loss Functions
    print("\n1. LOSS FUNCTIONS")
    print("-" * 30)
    
    # Mean Squared Error
    y_true = np.array([2, 4, 6, 8])
    y_pred = np.array([2.1, 3.9, 6.2, 7.8])
    mse = model.mean_squared_error(y_true, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")
    
    # Binary Cross-Entropy
    y_true_binary = np.array([1, 0, 1, 0])
    y_pred_binary = np.array([0.9, 0.1, 0.8, 0.2])
    bce = model.binary_cross_entropy(y_true_binary, y_pred_binary)
    print(f"Binary Cross-Entropy: {bce:.4f}")
    
    # Categorical Cross-Entropy
    y_true_cat = np.array([1, 0, 0])  # One-hot encoded
    y_pred_cat = np.array([0.7, 0.2, 0.1])
    cce = model.categorical_cross_entropy(y_true_cat, y_pred_cat)
    print(f"Categorical Cross-Entropy: {cce:.4f}")
    
    # Example 2: Activation Functions
    print("\n2. ACTIVATION FUNCTIONS")
    print("-" * 30)
    
    x = np.array([-2, -1, 0, 1, 2])
    print(f"Input values: {x}")
    print(f"Sigmoid: {model.sigmoid(x)}")
    print(f"ReLU: {model.relu(x)}")
    print(f"Tanh: {model.tanh(x)}")
    
    # Softmax example
    logits = np.array([2.0, 1.0, 0.1])
    probs = model.softmax(logits)
    print(f"Logits: {logits}")
    print(f"Softmax probabilities: {probs}")
    print(f"Sum of probabilities: {np.sum(probs):.6f}")
    
    # Example 3: Gradient Descent
    print("\n3. GRADIENT DESCENT OPTIMIZATION")
    print("-" * 30)
    
    theta = np.array([1.0, 2.0])
    gradient = np.array([0.1, -0.2])
    learning_rate = 0.01
    
    print(f"Initial parameters: {theta}")
    print(f"Gradient: {gradient}")
    print(f"Learning rate: {learning_rate}")
    
    new_theta = model.gradient_descent_update(theta, gradient, learning_rate)
    print(f"Updated parameters: {new_theta}")
    
    # Example 4: Function Composition
    print("\n4. FUNCTION COMPOSITION")
    print("-" * 30)
    
    # Define simple functions
    def square(x): return x ** 2
    def add_one(x): return x + 1
    def multiply_by_two(x): return x * 2
    
    x = 3
    functions = [square, add_one, multiply_by_two]
    result = model.function_composition(x, functions)
    
    print(f"Input: {x}")
    print(f"Functions: multiply_by_two -> add_one -> square")
    print(f"Result: {result}")
    print(f"Verification: {square(add_one(multiply_by_two(x)))}")
    
    # Example 5: Stochastic Gradient Descent
    print("\n5. STOCHASTIC GRADIENT DESCENT")
    print("-" * 30)
    
    theta = np.array([1.0, 2.0])
    gradients = [
        np.array([0.1, -0.2]),
        np.array([0.05, -0.1]),
        np.array([0.15, -0.3])
    ]
    learning_rate = 0.01
    
    print(f"Initial parameters: {theta}")
    print(f"Batch gradients: {gradients}")
    
    new_theta = model.stochastic_gradient_descent(theta, gradients, learning_rate)
    print(f"Updated parameters: {new_theta}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    # Uncomment the following lines to see visualizations
    # print("\nGenerating visualizations...")
    # model.visualize_activation_functions()
    # model.demonstrate_gradient_descent()


if __name__ == "__main__":
    main() 
"""
Backpropagation Necessity Demonstration

This module demonstrates why backpropagation is essential for training neural networks
by comparing random search (without gradients) with gradient descent (with backpropagation).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


def demonstrate_backpropagation_necessity():
    """Demonstrate why backpropagation is essential for training neural networks"""
    
    # Generate complex data
    np.random.seed(42)
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define a simple neural network
    def simple_nn(x, W1, b1, W2, b2):
        """Simple 2-layer neural network"""
        h1 = np.maximum(0, np.dot(x, W1.T) + b1)  # ReLU activation
        output = 1 / (1 + np.exp(-(np.dot(h1, W2.T) + b2)))  # Sigmoid activation
        return output
    
    def loss_function(predictions, targets):
        """Binary cross-entropy loss"""
        epsilon = 1e-15  # Prevent log(0)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    # Method 1: Random search (without backpropagation)
    def random_search_training(X, y, iterations=1000):
        """Train using random search - very inefficient without gradients"""
        best_loss = float('inf')
        best_params = None
        
        # Initialize parameters
        W1 = np.random.randn(10, 2) * 0.1
        b1 = np.zeros(10)
        W2 = np.random.randn(1, 10) * 0.1
        b2 = np.zeros(1)
        
        losses = []
        
        for i in range(iterations):
            # Random perturbation
            dW1 = np.random.randn(*W1.shape) * 0.01
            db1 = np.random.randn(*b1.shape) * 0.01
            dW2 = np.random.randn(*W2.shape) * 0.01
            db2 = np.random.randn(*b2.shape) * 0.01
            
            # Try perturbed parameters
            predictions = simple_nn(X, W1 + dW1, b1 + db1, W2 + dW2, b2 + db2)
            loss = loss_function(predictions, y)
            
            # Keep if better
            if loss < best_loss:
                best_loss = loss
                W1 += dW1
                b1 += db1
                W2 += dW2
                b2 += db2
            
            losses.append(best_loss)
            
            if i % 100 == 0:
                print(f"Random Search Iteration {i}: Loss = {best_loss:.4f}")
        
        return W1, b1, W2, b2, losses
    
    # Method 2: Gradient descent (with backpropagation)
    def gradient_descent_training(X, y, iterations=1000, learning_rate=0.1):
        """Train using gradient descent - efficient with gradients"""
        # Initialize parameters
        W1 = np.random.randn(10, 2) * 0.1
        b1 = np.zeros(10)
        W2 = np.random.randn(1, 10) * 0.1
        b2 = np.zeros(1)
        
        losses = []
        
        for i in range(iterations):
            # Forward pass
            h1 = np.maximum(0, np.dot(X, W1.T) + b1)
            predictions = 1 / (1 + np.exp(-(np.dot(h1, W2.T) + b2)))
            
            # Compute loss
            loss = loss_function(predictions, y)
            losses.append(loss)
            
            # Backward pass (backpropagation)
            # Gradient of loss with respect to predictions
            d_predictions = (predictions - y) / (predictions * (1 - predictions))
            
            # Gradient with respect to W2 and b2
            dW2 = np.dot(d_predictions.T, h1)
            db2 = np.sum(d_predictions, axis=0)
            
            # Gradient with respect to h1
            dh1 = np.dot(d_predictions, W2)
            
            # Gradient with respect to W1 and b1
            dh1_relu = dh1 * (h1 > 0)  # Gradient of ReLU
            dW1 = np.dot(dh1_relu.T, X)
            db1 = np.sum(dh1_relu, axis=0)
            
            # Update parameters
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            
            if i % 100 == 0:
                print(f"Gradient Descent Iteration {i}: Loss = {loss:.4f}")
        
        return W1, b1, W2, b2, losses
    
    # Train using both methods
    print("Training with Random Search (without backpropagation):")
    W1_rs, b1_rs, W2_rs, b2_rs, losses_rs = random_search_training(X_train, y_train, iterations=500)
    
    print("\nTraining with Gradient Descent (with backpropagation):")
    W1_gd, b1_gd, b2_gd, W2_gd, losses_gd = gradient_descent_training(X_train, y_train, iterations=500)
    
    # Evaluate both methods
    def evaluate_model(X, y, W1, b1, W2, b2):
        predictions = simple_nn(X, W1, b1, W2, b2)
        predictions_binary = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions_binary == y)
        return accuracy
    
    accuracy_rs = evaluate_model(X_test, y_test, W1_rs, b1_rs, W2_rs, b2_rs)
    accuracy_gd = evaluate_model(X_test, y_test, W1_gd, b1_gd, W2_gd, b2_gd)
    
    print(f"\nResults:")
    print(f"Random Search Accuracy: {accuracy_rs:.3f}")
    print(f"Gradient Descent Accuracy: {accuracy_gd:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Loss comparison
    plt.subplot(1, 3, 1)
    plt.plot(losses_rs, 'r-', label='Random Search', alpha=0.7)
    plt.plot(losses_gd, 'b-', label='Gradient Descent', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Decision boundaries
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Random search decision boundary
    plt.subplot(1, 3, 2)
    Z_rs = simple_nn(np.c_[xx.ravel(), yy.ravel()], W1_rs, b1_rs, W2_rs, b2_rs)
    Z_rs = Z_rs.reshape(xx.shape)
    plt.contourf(xx, yy, Z_rs, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title(f'Random Search\nAccuracy: {accuracy_rs:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Gradient descent decision boundary
    plt.subplot(1, 3, 3)
    Z_gd = simple_nn(np.c_[xx.ravel(), yy.ravel()], W1_gd, b1_gd, W2_gd, b2_gd)
    Z_gd = Z_gd.reshape(xx.shape)
    plt.contourf(xx, yy, Z_gd, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title(f'Gradient Descent\nAccuracy: {accuracy_gd:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return accuracy_rs, accuracy_gd, losses_rs, losses_gd


if __name__ == "__main__":
    backprop_demo = demonstrate_backpropagation_necessity()

"""
Modular Design Demonstration

This module demonstrates the power of modular design by comparing simple linear models
with modular neural networks on non-linear data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def demonstrate_modular_design():
    """Demonstrate the power of modular design"""
    
    # Generate data
    np.random.seed(42)
    X, y = make_circles(n_samples=1000, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define simple modules
    def linear_module(x, W, b):
        """Linear transformation module"""
        return np.dot(x, W.T) + b
    
    def relu_module(x):
        """ReLU activation module"""
        return np.maximum(0, x)
    
    def sigmoid_module(x):
        """Sigmoid activation module"""
        return 1 / (1 + np.exp(-x))
    
    # Compose modules to create different architectures
    def simple_mlp(x, W1, b1, W2, b2):
        """Simple MLP: Linear → ReLU → Linear → Sigmoid"""
        h1 = linear_module(x, W1, b1)  # Module 1: Linear
        h2 = relu_module(h1)           # Module 2: ReLU
        h3 = linear_module(h2, W2, b2) # Module 3: Linear
        output = sigmoid_module(h3)    # Module 4: Sigmoid
        return output
    
    def deep_mlp(x, weights, biases):
        """Deep MLP with multiple layers"""
        h = x
        for i in range(len(weights) - 1):
            h = linear_module(h, weights[i], biases[i])  # Linear module
            h = relu_module(h)                          # ReLU module
        # Final layer
        output = linear_module(h, weights[-1], biases[-1])  # Linear module
        output = sigmoid_module(output)                     # Sigmoid module
        return output
    
    # Train simple model
    simple_model = LogisticRegression(random_state=42)
    simple_model.fit(X_train, y_train)
    simple_score = simple_model.score(X_test, y_test)
    
    # Train neural network (modular approach)
    nn_model = MLPClassifier(hidden_layer_sizes=(10, 5), random_state=42, max_iter=1000)
    nn_model.fit(X_train, y_train)
    nn_score = nn_model.score(X_test, y_test)
    
    print(f"Modular Design Results:")
    print(f"Simple Linear Model: {simple_score:.3f}")
    print(f"Modular Neural Network: {nn_score:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Simple model decision boundary
    plt.subplot(1, 3, 2)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z_simple = simple_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_simple = Z_simple.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z_simple, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title(f'Simple Model\nAccuracy: {simple_score:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Modular model decision boundary
    plt.subplot(1, 3, 3)
    Z_nn = nn_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_nn = Z_nn.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z_nn, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title(f'Modular Neural Network\nAccuracy: {nn_score:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return simple_score, nn_score


if __name__ == "__main__":
    modular_demo = demonstrate_modular_design()

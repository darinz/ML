"""
Neural Network Basics Demonstration

This module demonstrates basic neural network concepts by comparing different
architectures (single layer, one hidden layer, two hidden layers) on non-linear data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def demonstrate_neural_network_basics():
    """Demonstrate basic neural network concepts"""
    
    # Generate non-linear data
    np.random.seed(42)
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train different neural network architectures
    
    # Single layer (no hidden layers)
    nn_single = MLPClassifier(hidden_layer_sizes=(), random_state=42, max_iter=1000)
    nn_single.fit(X_train, y_train)
    single_score = nn_single.score(X_test, y_test)
    
    # One hidden layer
    nn_one_layer = MLPClassifier(hidden_layer_sizes=(10,), random_state=42, max_iter=1000)
    nn_one_layer.fit(X_train, y_train)
    one_layer_score = nn_one_layer.score(X_test, y_test)
    
    # Two hidden layers
    nn_two_layers = MLPClassifier(hidden_layer_sizes=(10, 5), random_state=42, max_iter=1000)
    nn_two_layers.fit(X_train, y_train)
    two_layer_score = nn_two_layers.score(X_test, y_test)
    
    print(f"Neural Network Performance:")
    print(f"Single Layer (Linear): {single_score:.3f}")
    print(f"One Hidden Layer: {one_layer_score:.3f}")
    print(f"Two Hidden Layers: {two_layer_score:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title('Original Data (Non-linear Pattern)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Single layer decision boundary
    plt.subplot(1, 3, 2)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z_single = nn_single.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_single = Z_single.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z_single, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title(f'Single Layer (Linear)\nAccuracy: {single_score:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Two layer decision boundary
    plt.subplot(1, 3, 3)
    Z_two = nn_two_layers.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_two = Z_two.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z_two, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title(f'Two Hidden Layers\nAccuracy: {two_layer_score:.3f}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return single_score, one_layer_score, two_layer_score


if __name__ == "__main__":
    nn_demo = demonstrate_neural_network_basics()

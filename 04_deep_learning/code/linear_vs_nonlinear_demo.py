"""
Linear vs Non-linear Models Demonstration

This module demonstrates why non-linear models are necessary for certain types of data
by comparing linear and non-linear classifiers on a non-linear dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def demonstrate_linear_vs_nonlinear():
    """Demonstrate why non-linear models are necessary"""
    
    # Generate non-linear data (XOR-like problem)
    np.random.seed(42)
    X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train linear model
    linear_model = LogisticRegression(random_state=42)
    linear_model.fit(X_train, y_train)
    linear_score = linear_model.score(X_test, y_test)
    
    # Train non-linear model (neural network)
    non_linear_model = MLPClassifier(hidden_layer_sizes=(10, 5), random_state=42, max_iter=1000)
    non_linear_model.fit(X_train, y_train)
    non_linear_score = non_linear_model.score(X_test, y_test)
    
    print(f"Linear Model Accuracy: {linear_score:.3f}")
    print(f"Non-linear Model Accuracy: {non_linear_score:.3f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title('Original Data (Non-linear Pattern)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Linear decision boundary
    plt.subplot(1, 3, 2)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z_linear = linear_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_linear = Z_linear.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z_linear, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title(f'Linear Model (Accuracy: {linear_score:.3f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    # Non-linear decision boundary
    plt.subplot(1, 3, 3)
    Z_nonlinear = non_linear_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_nonlinear = Z_nonlinear.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z_nonlinear, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.6, s=20)
    plt.title(f'Non-linear Model (Accuracy: {non_linear_score:.3f})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return linear_score, non_linear_score


if __name__ == "__main__":
    linear_acc, nonlinear_acc = demonstrate_linear_vs_nonlinear()

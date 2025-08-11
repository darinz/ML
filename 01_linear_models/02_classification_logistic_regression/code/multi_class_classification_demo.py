import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def demonstrate_multi_class_classification():
    """Demonstrate multi-class classification with a simple example"""
    
    # Generate multi-class data
    np.random.seed(42)
    n_samples = 300
    n_classes = 4
    n_features = 2
    
    # Create synthetic data with 4 classes
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    print("Multi-Class Classification Demonstration")
    print("=" * 50)
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"Class distribution: {np.bincount(y)}")
    print()
    
    # Train multi-class logistic regression
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    
    # Evaluate performance
    accuracy = accuracy_score(y, y_pred)
    
    print("Model Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    print()
    
    print("Classification Report:")
    print(classification_report(y, y_pred))
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Data points colored by class
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(n_classes):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                   label=f'Class {i}', alpha=0.7, s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Multi-Class Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Decision boundaries
    plt.subplot(1, 3, 2)
    
    # Create mesh grid for decision boundaries
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    
    # Predict on mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.4, colors=colors)
    
    # Plot data points
    for i in range(n_classes):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                   label=f'Class {i}', alpha=0.7, s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundaries')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Probability distributions
    plt.subplot(1, 3, 3)
    
    # Show probability distributions for a few points
    sample_indices = np.random.choice(len(X), 10, replace=False)
    
    for idx in sample_indices:
        x_point = X[idx]
        y_true = y[idx]
        y_prob = y_proba[idx]
        
        # Plot point
        plt.scatter(x_point[0], x_point[1], c=colors[y_true], 
                   s=100, marker='o', edgecolors='black', linewidth=2)
        
        # Add probability text
        max_prob_idx = np.argmax(y_prob)
        plt.annotate(f'P({max_prob_idx})={y_prob[max_prob_idx]:.2f}', 
                    (x_point[0], x_point[1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', alpha=0.7))
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Probability Distributions')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Multi-class problems have more complex decision boundaries")
    print("2. Each class has its own probability distribution")
    print("3. Decision boundaries separate pairs of classes")
    print("4. Model provides confidence scores for each class")
    print("5. Performance metrics are more complex than binary classification")
    
    return model, X, y, y_pred, y_proba

if __name__ == "__main__":
    multi_class_demo = demonstrate_multi_class_classification()

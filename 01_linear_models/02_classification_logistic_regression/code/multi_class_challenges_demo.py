import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def demonstrate_multi_class_challenges():
    """Demonstrate the challenges of multi-class classification"""
    
    # Generate imbalanced multi-class data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Create imbalanced class distribution
    class_weights = [0.5, 0.25, 0.15, 0.07, 0.03]  # Imbalanced
    n_samples_per_class = [int(w * n_samples) for w in class_weights]
    
    X_list = []
    y_list = []
    
    for i in range(n_classes):
        n_class_samples = n_samples_per_class[i]
        # Create class-specific data
        X_class = np.random.randn(n_class_samples, 2) + i * 2
        y_class = np.full(n_class_samples, i)
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    print("Multi-Class Classification Challenges")
    print("=" * 50)
    print(f"Total samples: {len(X)}")
    print(f"Number of classes: {n_classes}")
    print()
    
    print("Class Distribution (Imbalanced):")
    for i in range(n_classes):
        count = np.sum(y == i)
        percentage = count / len(y) * 100
        print(f"Class {i}: {count} samples ({percentage:.1f}%)")
    print()
    
    # Train model on imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        stratify=y, random_state=42)
    
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Model Performance on Imbalanced Data:")
    print(f"Overall Accuracy: {accuracy:.3f}")
    print()
    
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Imbalanced data distribution
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i in range(n_classes):
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                   label=f'Class {i} ({np.sum(mask)})', alpha=0.7, s=30)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Imbalanced Multi-Class Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Confusion matrix
    plt.subplot(1, 3, 2)
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add text annotations
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center')
    
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.xticks(range(n_classes))
    plt.yticks(range(n_classes))
    
    # Plot 3: Per-class accuracy
    plt.subplot(1, 3, 3)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    class_counts = np.bincount(y_test)
    
    bars = plt.bar(range(n_classes), per_class_accuracy, 
                   color=colors[:n_classes], alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(n_classes))
    plt.ylim(0, 1)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, per_class_accuracy)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Challenges Identified:")
    print("-" * 30)
    print("1. Class Imbalance: Rare classes are harder to learn")
    print("2. Decision Boundaries: More complex with more classes")
    print("3. Evaluation Complexity: Need per-class metrics")
    print("4. Computational Cost: Scales with number of classes")
    print("5. Data Requirements: Need sufficient examples per class")
    
    return model, X, y, X_test, y_test, y_pred

if __name__ == "__main__":
    challenges_demo = demonstrate_multi_class_challenges()

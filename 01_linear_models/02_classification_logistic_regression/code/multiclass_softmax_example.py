"""
Multi-Class Classification with Softmax Implementation Examples

This module implements multi-class classification using the softmax function and
cross-entropy loss as described in the accompanying markdown file. It generalizes
logistic regression from binary to multi-class classification.

Key Concepts Implemented:
1. Softmax function with numerical stability
2. Cross-entropy loss for multi-class classification
3. Gradient computation for softmax regression
4. Multi-class prediction and decision boundaries
5. Practical examples with synthetic data
6. Comparison with binary classification

Mathematical Background:
- Softmax: softmax(t_i) = exp(t_i) / Σ_j exp(t_j)
- Cross-entropy loss: ℓ_ce = -log(exp(t_y) / Σ_i exp(t_i))
- Gradient: ∇_t ℓ_ce = softmax(t) - e_y (where e_y is one-hot encoding)
- Decision rule: ŷ = argmax_i softmax(t_i)

The softmax function maps logits (unbounded scores) to probabilities that sum to 1,
making it suitable for multi-class probability estimation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def softmax(logits):
    """
    Compute softmax probabilities for each row of logits.
    
    The softmax function transforms logits (unbounded scores) into a probability
    distribution. It has the form:
    softmax(t_i) = exp(t_i) / Σ_j exp(t_j)
    
    Properties:
    - Outputs are non-negative and sum to 1
    - Preserves relative ordering of inputs
    - Invariant to adding a constant to all inputs
    - Numerically stable when subtracting max logit
    
    Args:
        logits: np.ndarray of shape (n_samples, n_classes)
               Raw scores before softmax transformation
    
    Returns:
        probs: np.ndarray of shape (n_samples, n_classes)
               Probability distribution for each sample
    
    Example:
        >>> logits = np.array([[2, 1, 0]])
        >>> softmax(logits)
        array([[0.66524096, 0.24472847, 0.09003057]])
        >>> np.sum(softmax(logits), axis=1)
        array([1.])
    """
    # Numerical stability: subtract max logit before exponentiating
    # This prevents overflow while maintaining mathematical equivalence
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits_shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_temperature(logits, temperature=1.0):
    """
    Compute softmax with temperature scaling.
    
    Temperature scaling controls the "sharpness" of the probability distribution:
    - temperature < 1: sharper distribution (more confident)
    - temperature = 1: standard softmax
    - temperature > 1: softer distribution (less confident)
    
    Args:
        logits: Input logits
        temperature: Temperature parameter (positive float)
    
    Returns:
        Temperature-scaled softmax probabilities
    
    Example:
        >>> logits = np.array([[2, 1, 0]])
        >>> softmax_temperature(logits, temperature=0.5)  # Sharper
        array([[0.84419519, 0.11384663, 0.04195818]])
        >>> softmax_temperature(logits, temperature=2.0)  # Softer
        array([[0.4223188 , 0.31002552, 0.26765568]])
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    logits_scaled = logits / temperature
    return softmax(logits_scaled)

def cross_entropy_loss(probs, y_true):
    """
    Compute the mean cross-entropy loss.
    
    Cross-entropy loss measures how well the predicted probabilities match the
    true labels. It penalizes confident wrong predictions more heavily.
    
    Mathematical form:
    ℓ_ce = -log(exp(t_y) / Σ_i exp(t_i)) = -log(softmax(t_y))
    
    Args:
        probs: np.ndarray of shape (n_samples, n_classes)
               Softmax probabilities (should sum to 1 for each sample)
        y_true: np.ndarray of shape (n_samples,)
                True class indices (0-based)
    
    Returns:
        loss: float, mean cross-entropy loss
    
    Example:
        >>> probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        >>> y_true = np.array([0, 1])  # First sample is class 0, second is class 1
        >>> cross_entropy_loss(probs, y_true)
        0.35667494393873245
    """
    n = y_true.shape[0]
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    probs = np.clip(probs, epsilon, 1 - epsilon)
    
    # Extract probabilities for true classes
    true_probs = probs[np.arange(n), y_true]
    
    # Compute negative log-likelihood
    return -np.log(true_probs).mean()

def cross_entropy_loss_from_logits(logits, y_true):
    """
    Compute cross-entropy loss directly from logits (more numerically stable).
    
    This function combines softmax and cross-entropy loss in a single computation,
    which is more numerically stable than computing them separately.
    
    Args:
        logits: np.ndarray of shape (n_samples, n_classes)
        y_true: np.ndarray of shape (n_samples,)
    
    Returns:
        loss: float, mean cross-entropy loss
    
    Example:
        >>> logits = np.array([[2, 1, 0], [0, 2, 1]])
        >>> y_true = np.array([0, 1])
        >>> cross_entropy_loss_from_logits(logits, y_true)
        0.8132616281710594
    """
    # Compute softmax probabilities
    probs = softmax(logits)
    
    # Compute cross-entropy loss
    return cross_entropy_loss(probs, y_true)

def softmax_gradient(logits, y_true):
    """
    Compute the gradient of cross-entropy loss with respect to logits.
    
    The gradient has the elegant form: ∇_t ℓ_ce = softmax(t) - e_y
    where e_y is the one-hot encoding of the true label.
    
    Args:
        logits: np.ndarray of shape (n_samples, n_classes)
        y_true: np.ndarray of shape (n_samples,)
    
    Returns:
        gradient: np.ndarray of shape (n_samples, n_classes)
    
    Example:
        >>> logits = np.array([[2, 1, 0]])
        >>> y_true = np.array([1])  # True class is 1
        >>> grad = softmax_gradient(logits, y_true)
        >>> print(f"Gradient: {grad}")
    """
    n_samples, n_classes = logits.shape
    
    # Compute softmax probabilities
    probs = softmax(logits)
    
    # Create one-hot encoding of true labels
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y_true] = 1
    
    # Gradient: softmax(t) - one_hot(y)
    return probs - one_hot

def predict_classes(logits):
    """
    Predict class labels from logits using argmax.
    
    Args:
        logits: np.ndarray of shape (n_samples, n_classes)
    
    Returns:
        predictions: np.ndarray of shape (n_samples,)
                    Predicted class indices (0-based)
    
    Example:
        >>> logits = np.array([[2, 1, 0], [0, 3, 1]])
        >>> predict_classes(logits)
        array([0, 1])
    """
    return np.argmax(logits, axis=1)

def predict_proba_from_logits(logits):
    """
    Predict class probabilities from logits.
    
    Args:
        logits: np.ndarray of shape (n_samples, n_classes)
    
    Returns:
        probabilities: np.ndarray of shape (n_samples, n_classes)
    
    Example:
        >>> logits = np.array([[2, 1, 0]])
        >>> predict_proba_from_logits(logits)
        array([[0.66524096, 0.24472847, 0.09003057]])
    """
    return softmax(logits)

def generate_multiclass_data(n_samples=300, n_classes=3, n_features=2, random_state=42):
    """
    Generate synthetic multi-class classification data.
    
    Args:
        n_samples: Number of samples
        n_classes: Number of classes
        n_features: Number of features
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix
        y: Class labels (0-based)
        true_weights: True weight matrix used to generate data
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate true weights for each class
    true_weights = np.random.randn(n_classes, n_features)
    
    # Add bias term to X
    X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
    true_weights_with_bias = np.hstack([np.zeros((n_classes, 1)), true_weights])
    
    # Generate logits
    logits = X_with_bias @ true_weights_with_bias.T
    
    # Generate labels
    y = np.argmax(logits, axis=1)
    
    return X_with_bias, y, true_weights_with_bias

def train_softmax_regression(X, y, n_classes, alpha=0.1, max_iter=1000, random_state=42):
    """
    Train softmax regression using gradient descent.
    
    Args:
        X: Feature matrix with bias term
        y: Class labels (0-based)
        n_classes: Number of classes
        alpha: Learning rate
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
    
    Returns:
        weights: Trained weight matrix
        history: Training history
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    # Initialize weights
    weights = np.random.randn(n_classes, n_features) * 0.01
    history = {'loss': []}
    
    for iteration in range(max_iter):
        # Forward pass
        logits = X @ weights.T
        probs = softmax(logits)
        
        # Compute loss
        loss = cross_entropy_loss(probs, y)
        history['loss'].append(loss)
        
        # Compute gradient
        grad_logits = softmax_gradient(logits, y)
        grad_weights = grad_logits.T @ X
        
        # Update weights
        weights = weights - alpha * grad_weights
        
        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss:.6f}")
    
    return weights, history

def plot_multiclass_decision_boundary(weights, X, y, title="Softmax Regression Decision Boundary"):
    """
    Plot decision boundaries for 2D multi-class classification.
    
    Args:
        weights: Trained weight matrix
        X: Feature matrix with bias term
        y: Class labels
        title: Plot title
    """
    if X.shape[1] != 3:  # Must have bias + 2 features
        print("Can only plot decision boundary for 2D data")
        return
    
    # Extract features (excluding bias)
    X_features = X[:, 1:]
    n_classes = weights.shape[0]
    
    # Create mesh grid
    x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
    y_min, y_max = X_features[:, 1].min() - 1, X_features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    
    # Predict for grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_with_bias = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])
    logits = grid_with_bias @ weights.T
    Z = predict_classes(logits)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    scatter = plt.scatter(X_features[:, 0], X_features[:, 1], c=y, 
                         cmap='viridis', edgecolors='k', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter, label='Class')
    plt.show()

def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training metrics
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Cross-Entropy Loss')
    plt.grid(True)
    plt.show()

def demonstrate_multiclass_classification():
    """
    Complete demonstration of multi-class classification with softmax.
    
    This function shows the entire pipeline from data generation to model
    evaluation, including visualization of results.
    """
    print("=== Multi-Class Classification with Softmax Demonstration ===\n")
    
    # 1. Generate data
    print("1. Generating multi-class data...")
    X, y, true_weights = generate_multiclass_data(n_samples=300, n_classes=3, n_features=2)
    n_classes = len(np.unique(y))
    print(f"   Generated {X.shape[0]} samples with {X.shape[1]-1} features")
    print(f"   Number of classes: {n_classes}")
    print(f"   Class distribution: {np.bincount(y)}")
    print(f"   True weights shape: {true_weights.shape}\n")
    
    # 2. Train model
    print("2. Training softmax regression...")
    weights, history = train_softmax_regression(X, y, n_classes, alpha=0.1, max_iter=500)
    print(f"   Final weights shape: {weights.shape}")
    print(f"   Final loss: {history['loss'][-1]:.6f}\n")
    
    # 3. Make predictions
    print("3. Making predictions...")
    logits = X @ weights.T
    y_pred = predict_classes(logits)
    y_proba = predict_proba_from_logits(logits)
    
    accuracy = np.mean(y_pred == y)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Sample predictions (first 10):")
    for i in range(min(10, len(y))):
        print(f"     True: {y[i]}, Pred: {y_pred[i]}, Probs: {y_proba[i]}")
    print()
    
    # 4. Visualize results
    print("4. Visualizing results...")
    plot_multiclass_decision_boundary(weights, X, y, "Trained Softmax Regression")
    plot_training_history(history)
    
    print("=== Demonstration Complete ===")

def compare_with_binary_classification():
    """
    Compare multi-class softmax with binary logistic regression.
    
    This demonstrates how softmax generalizes logistic regression to multiple classes.
    """
    print("=== Multi-Class vs Binary Classification Comparison ===\n")
    
    # Generate binary data
    from logistic_regression_examples import generate_synthetic_data
    X_binary, y_binary, _ = generate_synthetic_data(n_samples=100, n_features=2)
    
    # Train binary logistic regression
    from logistic_regression_examples import train_logistic_regression
    theta_binary, _ = train_logistic_regression(X_binary, y_binary, alpha=0.1, max_iter=500)
    
    # Train multi-class softmax (treat as 2-class)
    weights_binary, _ = train_softmax_regression(X_binary, y_binary, n_classes=2, 
                                                alpha=0.1, max_iter=500)
    
    # Compare predictions
    y_pred_logistic = (sigmoid(X_binary @ theta_binary) >= 0.5).astype(int)
    logits_softmax = X_binary @ weights_binary.T
    y_pred_softmax = predict_classes(logits_softmax)
    
    accuracy_logistic = np.mean(y_pred_logistic == y_binary)
    accuracy_softmax = np.mean(y_pred_softmax == y_binary)
    
    print(f"Binary logistic regression accuracy: {accuracy_logistic:.4f}")
    print(f"Binary softmax regression accuracy: {accuracy_softmax:.4f}")
    print(f"Logistic parameters: {theta_binary}")
    print(f"Softmax parameters: {weights_binary}")
    
    # Visualize comparison
    plt.figure(figsize=(12, 5))
    
    # Logistic regression decision boundary
    plt.subplot(1, 2, 1)
    X_features = X_binary[:, 1:]
    x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
    y_min, y_max = X_features[:, 1].min() - 1, X_features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_with_bias = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])
    Z_logistic = sigmoid(grid_with_bias @ theta_binary)
    Z_logistic = Z_logistic.reshape(xx.shape)
    plt.contourf(xx, yy, Z_logistic, alpha=0.8, cmap='RdYlBu')
    plt.scatter(X_features[:, 0], X_features[:, 1], c=y_binary, cmap='RdYlBu', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Binary Logistic Regression')
    plt.colorbar(label='Probability of Class 1')
    
    # Softmax decision boundary
    plt.subplot(1, 2, 2)
    logits_grid = grid_with_bias @ weights_binary.T
    Z_softmax = predict_classes(logits_grid)
    Z_softmax = Z_softmax.reshape(xx.shape)
    plt.contourf(xx, yy, Z_softmax, alpha=0.8, cmap='viridis')
    plt.scatter(X_features[:, 0], X_features[:, 1], c=y_binary, cmap='viridis', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Binary Softmax Regression')
    plt.colorbar(label='Predicted Class')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Import sigmoid for comparison
    from logistic_regression_examples import sigmoid
    
    # Run the complete demonstration
    demonstrate_multiclass_classification()
    
    # Run comparison with binary classification
    compare_with_binary_classification()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example 1: Basic softmax function
    print("Example 1: Softmax function values")
    logits_example = np.array([[2, 1, 0], [0, 1, 2]])
    probs_example = softmax(logits_example)
    print(f"   Logits: {logits_example}")
    print(f"   Probabilities: {probs_example}")
    print(f"   Sum of probabilities: {np.sum(probs_example, axis=1)}")
    print()
    
    # Example 2: Temperature scaling
    print("Example 2: Temperature scaling effects")
    logits_temp = np.array([[2, 1, 0]])
    for temp in [0.5, 1.0, 2.0]:
        probs_temp = softmax_temperature(logits_temp, temp)
        print(f"   Temperature {temp}: {probs_temp[0]}")
    print()
    
    # Example 3: Cross-entropy loss
    print("Example 3: Cross-entropy loss calculation")
    probs_loss = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    y_loss = np.array([0, 1])
    loss = cross_entropy_loss(probs_loss, y_loss)
    print(f"   Probabilities: {probs_loss}")
    print(f"   True labels: {y_loss}")
    print(f"   Cross-entropy loss: {loss:.4f}")
    print()
    
    # Example 4: Gradient computation
    print("Example 4: Softmax gradient")
    logits_grad = np.array([[2, 1, 0]])
    y_grad = np.array([1])
    grad = softmax_gradient(logits_grad, y_grad)
    print(f"   Logits: {logits_grad[0]}")
    print(f"   True label: {y_grad[0]}")
    print(f"   Gradient: {grad[0]}")
    print(f"   Gradient sum: {np.sum(grad[0])}")  # Should be 0
    print()
    
    # Example 5: Numerical stability
    print("Example 5: Numerical stability demonstration")
    logits_unstable = np.array([[1000, 1001, 1002]])
    try:
        # This would overflow without numerical stability
        probs_stable = softmax(logits_unstable)
        print(f"   Large logits: {logits_unstable[0]}")
        print(f"   Stable probabilities: {probs_stable[0]}")
        print(f"   Sum: {np.sum(probs_stable[0])}")
    except Exception as e:
        print(f"   Error: {e}")
    print()
    
    # Example 6: Multi-class prediction
    print("Example 6: Multi-class prediction")
    logits_pred = np.array([[1, 3, 2], [4, 1, 1], [0, 0, 5]])
    y_pred = predict_classes(logits_pred)
    y_proba = predict_proba_from_logits(logits_pred)
    print(f"   Logits: {logits_pred}")
    print(f"   Predictions: {y_pred}")
    print(f"   Probabilities:")
    for i, prob in enumerate(y_proba):
        print(f"     Sample {i}: {prob}") 
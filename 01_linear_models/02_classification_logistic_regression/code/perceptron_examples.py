"""
Perceptron Learning Algorithm Implementation Examples

This module implements the perceptron learning algorithm as described in the 
accompanying markdown file. The perceptron is a foundational algorithm in machine
learning that learns linear decision boundaries for binary classification.

Key Concepts Implemented:
1. Perceptron threshold function (Heaviside step function)
2. Perceptron prediction and decision boundary
3. Perceptron learning rule and weight updates
4. Convergence analysis and limitations
5. Practical examples with synthetic data
6. Visualization of learning process

Mathematical Background:
- Threshold function: g(z) = 1 if z ≥ 0, else 0
- Hypothesis: h_θ(x) = g(θ^T x)
- Update rule: θ_j := θ_j + α(y^(i) - h_θ(x^(i)))x_j^(i)
- Decision boundary: θ^T x = 0

Historical Context:
The perceptron was introduced by Frank Rosenblatt in 1958 as one of the earliest
computational models of a biological neuron. While limited to linearly separable
data, it laid the foundation for modern neural networks and deep learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def perceptron_threshold(z):
    """
    Compute the perceptron threshold function (Heaviside step function).
    
    The threshold function is the key difference between perceptron and logistic
    regression. Instead of a smooth sigmoid, it uses a hard threshold that outputs
    exactly 0 or 1.
    
    Mathematical form:
    g(z) = 1 if z ≥ 0, else 0
    
    Properties:
    - Outputs exactly 0 or 1 (no probabilities)
    - Not differentiable at z = 0
    - Creates sharp decision boundaries
    
    Args:
        z: Input value (can be any real number)
    
    Returns:
        Binary output (0 or 1)
    
    Example:
        >>> perceptron_threshold(-2)
        0
        >>> perceptron_threshold(0)
        1
        >>> perceptron_threshold(3)
        1
    """
    return 1 if z >= 0 else 0

def perceptron_threshold_vec(z):
    """
    Vectorized version of the perceptron threshold function.
    
    This function applies the threshold function to each element of a numpy array,
    making it efficient for batch processing.
    
    Args:
        z: Input array of any shape
    
    Returns:
        Array of same shape with binary outputs
    
    Example:
        >>> z = np.array([-1, 0, 1, 2])
        >>> perceptron_threshold_vec(z)
        array([0, 1, 1, 1])
    """
    return np.vectorize(perceptron_threshold)(z)

def predict(theta, x):
    """
    Compute perceptron prediction for input x and weights theta.
    
    The prediction combines the linear transformation θ^T x with the threshold
    function to produce a binary classification.
    
    Args:
        theta: Parameter vector [θ₀, θ₁, ..., θₙ] (including bias term)
        x: Feature vector [1, x₁, ..., xₙ] (first element should be 1 for bias)
    
    Returns:
        Binary prediction (0 or 1)
    
    Example:
        >>> theta = [0.5, 1.0, -0.5]
        >>> x = [1, 2, 3]  # Note: first element is bias term
        >>> predict(theta, x)
        1
    """
    z = np.dot(theta, x)
    return perceptron_threshold(z)

def perceptron_update(theta, x, y, alpha):
    """
    Update theta using the perceptron learning rule.
    
    The perceptron learning rule is the core of the algorithm. It updates weights
    only when a misclassification occurs, moving the decision boundary to better
    separate the classes.
    
    Update rule: θ_j := θ_j + α(y^(i) - h_θ(x^(i)))x_j^(i)
    
    Intuition:
    - If prediction is correct (y = h_θ(x)): no update needed
    - If false positive (h_θ(x) = 1, y = 0): decrease weights
    - If false negative (h_θ(x) = 0, y = 1): increase weights
    
    Args:
        theta: Current parameter vector (numpy array)
        x: Input feature vector (numpy array, including bias term)
        y: True label (0 or 1)
        alpha: Learning rate (float)
    
    Returns:
        Updated theta (numpy array)
    
    Example:
        >>> theta = np.array([0.0, 0.0, 0.0])
        >>> x = np.array([1, 2, 3])
        >>> y = 1
        >>> alpha = 0.1
        >>> theta_new = perceptron_update(theta, x, y, alpha)
        >>> print(f"Updated theta: {theta_new}")
    """
    prediction = predict(theta, x)
    # Only update if prediction is wrong
    if prediction != y:
        theta = theta + alpha * (y - prediction) * x
    return theta

def train_perceptron(X, y, alpha=0.1, max_iter=1000, random_state=42):
    """
    Train a perceptron classifier using the perceptron learning algorithm.
    
    This function implements the complete training loop for the perceptron,
    including convergence checking and early stopping.
    
    Important Notes:
    - The perceptron only converges if the data is linearly separable
    - If not linearly separable, it will never converge
    - The algorithm stops when all examples are correctly classified
    
    Args:
        X: Feature matrix (n_samples, n_features) with bias term
        y: Binary labels (0 or 1)
        alpha: Learning rate
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility
    
    Returns:
        Trained parameter vector and training history
    
    Example:
        >>> X, y = generate_linearly_separable_data()
        >>> theta, history = train_perceptron(X, y)
        >>> print(f"Final theta: {theta}")
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    history = {'errors': [], 'theta_norm': []}
    
    for iteration in range(max_iter):
        # Count misclassifications
        errors = 0
        for i in range(n_samples):
            prediction = predict(theta, X[i])
            if prediction != y[i]:
                errors += 1
                # Update weights for misclassified example
                theta = perceptron_update(theta, X[i], y[i], alpha)
        
        # Store history
        history['errors'].append(errors)
        history['theta_norm'].append(np.linalg.norm(theta))
        
        # Check convergence (no errors)
        if errors == 0:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        # Print progress
        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}, Errors: {errors}")
    
    if errors > 0:
        print(f"Warning: Did not converge after {max_iter} iterations")
        print("Data may not be linearly separable")
    
    return theta, history

def predict_perceptron(theta, X):
    """
    Make predictions using a trained perceptron.
    
    Args:
        theta: Trained parameter vector
        X: Feature matrix with bias term
    
    Returns:
        Binary predictions (0 or 1) for each sample
    """
    predictions = []
    for i in range(X.shape[0]):
        pred = predict(theta, X[i])
        predictions.append(pred)
    return np.array(predictions)

def generate_linearly_separable_data(n_samples=100, n_features=2, random_state=42):
    """
    Generate synthetic linearly separable data for perceptron demonstration.
    
    This function creates data that is guaranteed to be linearly separable,
    ensuring the perceptron will converge.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features (excluding bias)
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix with bias term
        y: Binary labels
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create a simple linear decision boundary
    # Class 0: x1 + x2 < 0, Class 1: x1 + x2 >= 0
    y = (X[:, 0] + X[:, 1] >= 0).astype(int)
    
    # Add bias term to X
    X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
    
    return X_with_bias, y

def generate_non_separable_data(n_samples=100, random_state=42):
    """
    Generate synthetic non-linearly separable data (XOR-like).
    
    This function creates data that is not linearly separable, demonstrating
    the perceptron's limitations.
    
    Args:
        n_samples: Number of samples
        random_state: Random seed for reproducibility
    
    Returns:
        X: Feature matrix with bias term
        y: Binary labels
    """
    np.random.seed(random_state)
    
    # Generate XOR-like pattern
    X = np.random.randn(n_samples, 2)
    
    # XOR pattern: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    # We'll use a threshold to create this pattern
    y = ((X[:, 0] > 0) != (X[:, 1] > 0)).astype(int)
    
    # Add bias term to X
    X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
    
    return X_with_bias, y

def plot_decision_boundary(theta, X, y, title="Perceptron Decision Boundary"):
    """
    Plot the decision boundary for 2D perceptron.
    
    Args:
        theta: Trained parameter vector [bias, θ₁, θ₂]
        X: Feature matrix with bias term
        y: Binary labels
        title: Plot title
    """
    if X.shape[1] != 3:  # Must have bias + 2 features
        print("Can only plot decision boundary for 2D data")
        return
    
    # Extract features (excluding bias)
    X_features = X[:, 1:]
    
    # Create mesh grid
    x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
    y_min, y_max = X_features[:, 1].min() - 1, X_features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    
    # Predict for grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_with_bias = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])
    Z = predict_perceptron(theta, grid_with_bias)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
    plt.scatter(X_features[:, 0], X_features[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(label='Predicted Class')
    plt.show()

def plot_training_history(history):
    """
    Plot the training history of the perceptron.
    
    Args:
        history: Dictionary containing 'errors' and 'theta_norm' lists
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['errors'])
    plt.title('Training Errors')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Errors')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['theta_norm'])
    plt.title('Parameter Norm')
    plt.xlabel('Iteration')
    plt.ylabel('||θ||₂')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def demonstrate_perceptron():
    """
    Complete demonstration of perceptron training and evaluation.
    
    This function shows the entire pipeline from data generation to model
    evaluation, including visualization of results and convergence analysis.
    """
    print("=== Perceptron Learning Algorithm Demonstration ===\n")
    
    # 1. Linearly separable data demonstration
    print("1. Training on linearly separable data...")
    X_sep, y_sep = generate_linearly_separable_data(n_samples=200)
    print(f"   Generated {X_sep.shape[0]} samples with {X_sep.shape[1]-1} features")
    print(f"   Class distribution: {np.bincount(y_sep)}\n")
    
    theta_sep, history_sep = train_perceptron(X_sep, y_sep, alpha=0.1, max_iter=1000)
    print(f"   Final parameters: {theta_sep}")
    print(f"   Final errors: {history_sep['errors'][-1]}\n")
    
    # 2. Non-separable data demonstration
    print("2. Training on non-linearly separable data...")
    X_non, y_non = generate_non_separable_data(n_samples=200)
    print(f"   Generated {X_non.shape[0]} samples")
    print(f"   Class distribution: {np.bincount(y_non)}\n")
    
    theta_non, history_non = train_perceptron(X_non, y_non, alpha=0.1, max_iter=1000)
    print(f"   Final parameters: {theta_non}")
    print(f"   Final errors: {history_non['errors'][-1]}\n")
    
    # 3. Evaluate performance
    print("3. Evaluating performance...")
    y_pred_sep = predict_perceptron(theta_sep, X_sep)
    accuracy_sep = np.mean(y_pred_sep == y_sep)
    print(f"   Linearly separable data accuracy: {accuracy_sep:.4f}")
    
    y_pred_non = predict_perceptron(theta_non, X_non)
    accuracy_non = np.mean(y_pred_non == y_non)
    print(f"   Non-separable data accuracy: {accuracy_non:.4f}\n")
    
    # 4. Visualize results
    print("4. Visualizing results...")
    
    # Plot linearly separable results
    plot_decision_boundary(theta_sep, X_sep, y_sep, "Perceptron - Linearly Separable Data")
    plot_training_history(history_sep)
    
    # Plot non-separable results
    plot_decision_boundary(theta_non, X_non, y_non, "Perceptron - Non-Separable Data")
    plot_training_history(history_non)
    
    print("=== Demonstration Complete ===")

def compare_with_logistic_regression():
    """
    Compare perceptron with logistic regression on the same data.
    
    This demonstrates the key differences between the two algorithms:
    - Perceptron: Hard threshold, no probabilities
    - Logistic regression: Soft sigmoid, probabilistic outputs
    """
    print("=== Perceptron vs Logistic Regression Comparison ===\n")
    
    # Generate data
    X, y = generate_linearly_separable_data(n_samples=100)
    
    # Train perceptron
    theta_perceptron, _ = train_perceptron(X, y, alpha=0.1, max_iter=1000)
    
    # Train logistic regression (simplified)
    from logistic_regression_examples import train_logistic_regression
    theta_logistic, _ = train_logistic_regression(X, y, alpha=0.1, max_iter=1000)
    
    # Compare predictions
    y_pred_perceptron = predict_perceptron(theta_perceptron, X)
    y_pred_logistic = (sigmoid(X @ theta_logistic) >= 0.5).astype(int)
    
    accuracy_perceptron = np.mean(y_pred_perceptron == y)
    accuracy_logistic = np.mean(y_pred_logistic == y)
    
    print(f"Perceptron accuracy: {accuracy_perceptron:.4f}")
    print(f"Logistic regression accuracy: {accuracy_logistic:.4f}")
    print(f"Perceptron parameters: {theta_perceptron}")
    print(f"Logistic parameters: {theta_logistic}")
    
    # Visualize decision boundaries
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_decision_boundary(theta_perceptron, X, y, "Perceptron Decision Boundary")
    
    plt.subplot(1, 2, 2)
    # Plot logistic regression decision boundary
    X_features = X[:, 1:]
    x_min, x_max = X_features[:, 0].min() - 1, X_features[:, 0].max() + 1
    y_min, y_max = X_features[:, 1].min() - 1, X_features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_with_bias = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])
    Z = sigmoid(grid_with_bias @ theta_logistic)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
    plt.scatter(X_features[:, 0], X_features[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.colorbar(label='Probability of Class 1')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Import sigmoid for comparison
    from logistic_regression_examples import sigmoid
    
    # Run the complete demonstration
    demonstrate_perceptron()
    
    # Run comparison with logistic regression
    compare_with_logistic_regression()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example 1: Basic threshold function
    print("Example 1: Perceptron threshold function")
    z_values = [-3, -1, 0, 1, 3]
    for z in z_values:
        print(f"   threshold({z}) = {perceptron_threshold(z)}")
    print()
    
    # Example 2: Single update step
    print("Example 2: Single perceptron update")
    theta = np.array([0.0, 0.0, 0.0])
    x = np.array([1, 2, 3])
    y = 1
    alpha = 0.1
    theta_new = perceptron_update(theta, x, y, alpha)
    print(f"   Before: {theta}")
    print(f"   After:  {theta_new}")
    print()
    
    # Example 3: Prediction example
    print("Example 3: Perceptron prediction")
    theta = np.array([-0.5, 1.0, -0.5])
    test_points = [
        [1, 0, 0],  # Should be class 0
        [1, 1, 1],  # Should be class 0
        [1, 2, 0],  # Should be class 1
    ]
    for i, x in enumerate(test_points):
        pred = predict(theta, x)
        print(f"   Point {i+1}: {x[1:]} -> Class {pred}")
    print()
    
    # Example 4: Convergence demonstration
    print("Example 4: Convergence on simple data")
    X_simple = np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])
    y_simple = np.array([0, 0, 1, 1])  # Simple AND function
    theta_simple, history_simple = train_perceptron(X_simple, y_simple, alpha=0.1)
    print(f"   Final theta: {theta_simple}")
    print(f"   Converged in {len(history_simple['errors'])} iterations") 
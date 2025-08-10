"""
Logistic Regression Implementation Examples

This module implements the key concepts from logistic regression as described in the 
accompanying markdown file. It includes the sigmoid function, hypothesis computation,
gradient calculations, and optimization algorithms.

Key Concepts Implemented:
1. Sigmoid (logistic) function and its derivative
2. Logistic regression hypothesis function
3. Log-likelihood and cross-entropy loss
4. Gradient computation for optimization
5. Gradient ascent algorithm
6. Practical examples with synthetic data

Mathematical Background:
- Hypothesis: h_θ(x) = g(θ^T x) = 1/(1 + e^(-θ^T x))
- Log-likelihood: ℓ(θ) = Σ[y^(i) log(h_θ(x^(i))) + (1-y^(i)) log(1-h_θ(x^(i)))]
- Gradient: ∇_θ ℓ(θ) = X^T (y - h_θ(X))
- Update rule: θ := θ + α ∇_θ ℓ(θ)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    """
    Compute the sigmoid (logistic) function.
    
    The sigmoid function maps any real number to the interval (0,1), making it
    suitable for modeling probabilities. It has the form:
    g(z) = 1 / (1 + e^(-z))
    
    Properties:
    - g(0) = 0.5 (uncertainty)
    - g(z) → 1 as z → ∞ (high confidence in class 1)
    - g(z) → 0 as z → -∞ (high confidence in class 0)
    - g'(z) = g(z)(1 - g(z)) (elegant derivative)
    
    Args:
        z: Input value or array (can be any real number)
    
    Returns:
        Sigmoid output in range (0,1)
    
    Example:
        >>> sigmoid(0)
        0.5
        >>> sigmoid(2)
        0.8807970779778823
        >>> sigmoid(-2)
        0.11920292202211755
    """
    # Numerical stability: clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Compute the derivative of the sigmoid function.
    
    The derivative has the elegant form: g'(z) = g(z)(1 - g(z))
    This property greatly simplifies gradient computation in neural networks.
    
    Args:
        z: Input value or array
    
    Returns:
        Derivative of sigmoid function
    
    Example:
        >>> sigmoid_derivative(0)
        0.25
        >>> sigmoid_derivative(2)
        0.10499358540350662
    """
    s = sigmoid(z)
    return s * (1 - s)

def hypothesis(theta, x):
    """
    Compute the logistic regression hypothesis h_θ(x).
    
    The hypothesis function combines the linear transformation θ^T x with the
    sigmoid activation to produce a probability between 0 and 1.
    
    Args:
        theta: Parameter vector [θ₀, θ₁, ..., θₙ] (including bias term)
        x: Feature vector [1, x₁, ..., xₙ] (first element should be 1 for bias)
    
    Returns:
        Probability that y = 1 given x and θ
    
    Example:
        >>> theta = [0.5, 1.0, -0.5]
        >>> x = [1, 2, 3]  # Note: first element is bias term
        >>> hypothesis(theta, x)
        0.6224593312018546
    """
    return sigmoid(np.dot(theta, x))

def log_likelihood(theta, X, y):
    """
    Compute the log-likelihood for logistic regression.
    
    The log-likelihood measures how well our model explains the observed data.
    We maximize this function to find the best parameters θ.
    
    Mathematical form:
    ℓ(θ) = Σ[y^(i) log(h_θ(x^(i))) + (1-y^(i)) log(1-h_θ(x^(i)))]
    
    Args:
        theta: Parameter vector
        X: Feature matrix (n_samples, n_features) with bias term
        y: Binary labels (0 or 1)
    
    Returns:
        Log-likelihood value (higher is better)
    
    Example:
        >>> X = np.array([[1, 2], [1, 3], [1, 4]])
        >>> y = np.array([0, 1, 1])
        >>> theta = np.array([0.1, 0.2])
        >>> ll = log_likelihood(theta, X, y)
        >>> print(f"Log-likelihood: {ll:.4f}")
    """
    h = sigmoid(X @ theta)
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    return np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient(theta, X, y):
    """
    Compute the gradient of the log-likelihood.
    
    The gradient points in the direction of steepest ascent for the log-likelihood.
    For logistic regression, it has the simple form: X^T (y - h_θ(X))
    
    Args:
        theta: Parameter vector
        X: Feature matrix (n_samples, n_features) with bias term
        y: Binary labels (0 or 1)
    
    Returns:
        Gradient vector with same shape as theta
    
    Example:
        >>> X = np.array([[1, 2], [1, 3], [1, 4]])
        >>> y = np.array([0, 1, 1])
        >>> theta = np.array([0.1, 0.2])
        >>> grad = gradient(theta, X, y)
        >>> print(f"Gradient: {grad}")
    """
    h = sigmoid(X @ theta)
    return X.T @ (y - h)

def logistic_loss(t, y):
    """
    Compute the logistic loss (cross-entropy) for a single example.
    
    The logistic loss penalizes incorrect predictions, especially when the model
    is confident but wrong. It's equivalent to negative log-likelihood.
    
    Mathematical form:
    ℓ_logistic(t, y) = y log(1 + exp(-t)) + (1-y) log(1 + exp(t))
    
    Args:
        t: Logit (θ^T x) - the raw score before sigmoid
        y: True label (0 or 1)
    
    Returns:
        Logistic loss value
    
    Example:
        >>> t = 2.0  # logit
        >>> y = 1    # true label
        >>> loss = logistic_loss(t, y)
        >>> print(f"Logistic loss: {loss:.4f}")
    """
    return y * np.log(1 + np.exp(-t)) + (1 - y) * np.log(1 + np.exp(t))

def gradient_ascent_update(theta, X, y, alpha=0.01):
    """
    Perform one step of gradient ascent for logistic regression.
    
    Gradient ascent moves in the direction of the gradient to maximize the
    log-likelihood. The update rule is: θ := θ + α ∇_θ ℓ(θ)
    
    Args:
        theta: Current parameter vector
        X: Feature matrix with bias term
        y: Binary labels
        alpha: Learning rate (step size)
    
    Returns:
        Updated parameter vector
    
    Example:
        >>> X = np.array([[1, 2], [1, 3], [1, 4]])
        >>> y = np.array([0, 1, 1])
        >>> theta = np.zeros(2)
        >>> theta_new = gradient_ascent_update(theta, X, y, alpha=0.1)
        >>> print(f"Updated theta: {theta_new}")
    """
    grad = gradient(theta, X, y)
    return theta + alpha * grad

def train_logistic_regression(X, y, alpha=0.01, max_iter=1000, tol=1e-6):
    """
    Train logistic regression using gradient ascent.
    
    This function implements the complete training loop for logistic regression,
    including convergence checking and learning rate scheduling.
    
    Args:
        X: Feature matrix with bias term
        y: Binary labels
        alpha: Learning rate
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence
    
    Returns:
        Trained parameter vector and training history
    
    Example:
        >>> X, y = generate_synthetic_data()
        >>> theta, history = train_logistic_regression(X, y)
        >>> print(f"Final theta: {theta}")
    """
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    history = {'loss': [], 'theta_norm': []}
    
    for i in range(max_iter):
        # Store current state
        current_loss = -log_likelihood(theta, X, y)  # Negative for minimization
        history['loss'].append(current_loss)
        history['theta_norm'].append(np.linalg.norm(theta))
        
        # Update parameters
        theta_new = gradient_ascent_update(theta, X, y, alpha)
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < tol:
            print(f"Converged after {i+1} iterations")
            break
            
        theta = theta_new
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}, Loss: {current_loss:.6f}")
    
    return theta, history

def predict_proba(theta, X):
    """
    Predict class probabilities using trained logistic regression.
    
    Args:
        theta: Trained parameter vector
        X: Feature matrix with bias term
    
    Returns:
        Probability of class 1 for each sample
    """
    return sigmoid(X @ theta)

def predict(theta, X, threshold=0.5):
    """
    Predict binary class labels using trained logistic regression.
    
    Args:
        theta: Trained parameter vector
        X: Feature matrix with bias term
        threshold: Decision threshold (default: 0.5)
    
    Returns:
        Binary predictions (0 or 1)
    """
    return (predict_proba(theta, X) >= threshold).astype(int)

def generate_synthetic_data(n_samples=100, n_features=2, random_state=42):
    """
    Generate synthetic binary classification data for demonstration.
    
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
    
    # Generate true parameters
    true_theta = np.random.randn(n_features + 1)
    
    # Add bias term to X
    X_with_bias = np.hstack([np.ones((n_samples, 1)), X])
    
    # Generate labels using true parameters
    logits = X_with_bias @ true_theta
    y = (sigmoid(logits) > 0.5).astype(float)
    
    return X_with_bias, y, true_theta

def plot_decision_boundary(theta, X, y, title="Logistic Regression Decision Boundary"):
    """
    Plot the decision boundary for 2D logistic regression.
    
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
    
    # Predict probabilities for grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_with_bias = np.hstack([np.ones((grid_points.shape[0], 1)), grid_points])
    Z = predict_proba(theta, grid_with_bias)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
    plt.scatter(X_features[:, 0], X_features[:, 1], c=y, cmap='RdYlBu', edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(label='Probability of Class 1')
    plt.show()

def demonstrate_logistic_regression():
    """
    Complete demonstration of logistic regression training and evaluation.
    
    This function shows the entire pipeline from data generation to model
    evaluation, including visualization of results.
    """
    print("=== Logistic Regression Demonstration ===\n")
    
    # 1. Generate synthetic data
    print("1. Generating synthetic data...")
    X, y, true_theta = generate_synthetic_data(n_samples=200, n_features=2)
    print(f"   Generated {X.shape[0]} samples with {X.shape[1]-1} features")
    print(f"   True parameters: {true_theta}")
    print(f"   Class distribution: {np.bincount(y.astype(int))}\n")
    
    # 2. Train the model
    print("2. Training logistic regression...")
    theta, history = train_logistic_regression(X, y, alpha=0.1, max_iter=500)
    print(f"   Final parameters: {theta}")
    print(f"   Final loss: {history['loss'][-1]:.6f}\n")
    
    # 3. Make predictions
    print("3. Making predictions...")
    y_pred_proba = predict_proba(theta, X)
    y_pred = predict(theta, X)
    accuracy = np.mean(y_pred == y)
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Sample predictions (first 10):")
    for i in range(min(10, len(y))):
        print(f"     True: {y[i]}, Pred: {y_pred[i]}, Prob: {y_pred_proba[i]:.3f}")
    print()
    
    # 4. Visualize results
    print("4. Visualizing decision boundary...")
    plot_decision_boundary(theta, X, y, "Trained Logistic Regression")
    
    # 5. Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Negative Log-Likelihood')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['theta_norm'])
    plt.title('Parameter Norm')
    plt.xlabel('Iteration')
    plt.ylabel('||θ||₂')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("=== Demonstration Complete ===")

if __name__ == "__main__":
    # Run the complete demonstration
    demonstrate_logistic_regression()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example 1: Basic sigmoid function
    print("Example 1: Sigmoid function values")
    z_values = [-5, -2, 0, 2, 5]
    for z in z_values:
        print(f"   sigmoid({z}) = {sigmoid(z):.4f}")
    print()
    
    # Example 2: Gradient computation
    print("Example 2: Gradient computation")
    X_example = np.array([[1, 2], [1, 3], [1, 4]])
    y_example = np.array([0, 1, 1])
    theta_example = np.array([0.1, 0.2])
    grad = gradient(theta_example, X_example, y_example)
    print(f"   Gradient: {grad}")
    print()
    
    # Example 3: Single update step
    print("Example 3: Single gradient ascent update")
    theta_before = np.array([0.0, 0.0])
    theta_after = gradient_ascent_update(theta_before, X_example, y_example, alpha=0.1)
    print(f"   Before: {theta_before}")
    print(f"   After:  {theta_after}")
    print()
    
    # Example 4: Logistic loss
    print("Example 4: Logistic loss for different predictions")
    test_cases = [(2.0, 1), (2.0, 0), (-2.0, 1), (-2.0, 0)]
    for t, y in test_cases:
        loss = logistic_loss(t, y)
        prob = sigmoid(t)
        print(f"   t={t}, y={y}: loss={loss:.4f}, prob={prob:.4f}") 
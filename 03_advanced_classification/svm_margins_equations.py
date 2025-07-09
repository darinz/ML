"""
SVM Margins and Lagrangian Duality: Implementation Examples
=========================================================

This file implements the key concepts from the SVM margins document:
- Functional and geometric margins
- The optimal margin classifier optimization problem
- Lagrangian duality and KKT conditions
- Hard-margin SVM implementation

The implementations demonstrate both theoretical concepts and practical
optimization methods with detailed mathematical foundations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import time


# ============================================================================
# Section 6.2: Notation and Linear Classifiers
# ============================================================================

def linear_classifier_predict(x, w, b):
    """
    Predicts the class label for input x using weights w and bias b.
    
    Args:
        x: Input vector
        w: Weight vector
        b: Bias term
        
    Returns:
        1 if w^T x + b >= 0, else -1
        
    Mathematical foundation:
        f(x) = sign(w^T x + b)
        where sign(z) = 1 if z ≥ 0, -1 otherwise
    """
    decision_value = np.dot(w, x) + b
    return 1 if decision_value >= 0 else -1


def linear_classifier_decision_boundary(w, b, x1_range):
    """
    Compute decision boundary for 2D visualization.
    
    Args:
        w: Weight vector [w1, w2]
        b: Bias term
        x1_range: Range of x1 values
        
    Returns:
        x2_values: Corresponding x2 values for decision boundary
        
    Mathematical foundation:
        Decision boundary: w1*x1 + w2*x2 + b = 0
        Therefore: x2 = -(w1*x1 + b) / w2
    """
    w1, w2 = w
    x2_values = -(w1 * x1_range + b) / w2
    return x2_values


def demonstrate_linear_classifier():
    """
    Demonstrate linear classifier with visualization.
    """
    print("=== Linear Classifier Demonstration ===")
    
    # Create synthetic linearly separable data
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    
    # Convert labels to {-1, 1}
    y = 2 * y - 1
    
    # Train a linear SVM to get weights
    clf = SVC(kernel='linear', C=1000)  # High C for hard margin
    clf.fit(X, y)
    
    w = clf.coef_[0]
    b = clf.intercept_[0]
    
    print(f"Weight vector w: {w}")
    print(f"Bias term b: {b}")
    print(f"Number of support vectors: {clf.n_support_}")
    
    # Visualize
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class +1', alpha=0.7)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', label='Class -1', alpha=0.7)
    
    # Plot decision boundary
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_range = np.linspace(x1_min, x1_max, 100)
    x2_boundary = linear_classifier_decision_boundary(w, b, x1_range)
    
    plt.plot(x1_range, x2_boundary, 'k-', linewidth=2, label='Decision Boundary')
    
    # Plot margin boundaries
    margin = 1 / np.linalg.norm(w)
    x2_margin_plus = x2_boundary + margin * w[1] / np.linalg.norm(w)
    x2_margin_minus = x2_boundary - margin * w[1] / np.linalg.norm(w)
    
    plt.plot(x1_range, x2_margin_plus, 'k--', alpha=0.5, label='Margin Boundary')
    plt.plot(x1_range, x2_margin_minus, 'k--', alpha=0.5, label='Margin Boundary')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Linear Classifier with Margins')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return w, b, X, y


# ============================================================================
# Section 6.3: Functional and Geometric Margins
# ============================================================================

def functional_margin(w, b, x_i, y_i):
    """
    Computes the functional margin for a single training example (x_i, y_i).
    
    Args:
        w: Weight vector
        b: Bias term
        x_i: Training example
        y_i: Label (-1 or 1)
        
    Returns:
        Functional margin: γ̂_i = y_i * (w^T x_i + b)
        
    Mathematical foundation:
        The functional margin measures how confident the classifier is
        about its prediction, but it's not scale-invariant.
    """
    return y_i * (np.dot(w, x_i) + b)


def min_functional_margin(w, b, X, y):
    """
    Computes the minimum functional margin over a dataset.
    
    Args:
        w: Weight vector
        b: Bias term
        X: Training data (n_samples, n_features)
        y: Labels (n_samples,)
        
    Returns:
        Minimum functional margin: γ̂ = min_i γ̂_i
        
    Mathematical foundation:
        γ̂ = min_{i=1,...,m} y_i * (w^T x_i + b)
    """
    margins = [functional_margin(w, b, x_i, y_i) for x_i, y_i in zip(X, y)]
    return min(margins)


def geometric_margin(w, b, x_i, y_i):
    """
    Computes the geometric margin for a single training example (x_i, y_i).
    
    Args:
        w: Weight vector
        b: Bias term
        x_i: Training example
        y_i: Label (-1 or 1)
        
    Returns:
        Geometric margin: γ_i = y_i * (w^T x_i + b) / ||w||
        
    Mathematical foundation:
        The geometric margin is the Euclidean distance from x_i to the
        decision boundary, scaled by the label. It's scale-invariant.
    """
    norm_w = np.linalg.norm(w)
    return y_i * (np.dot(w, x_i) + b) / norm_w


def min_geometric_margin(w, b, X, y):
    """
    Computes the minimum geometric margin over a dataset.
    
    Args:
        w: Weight vector
        b: Bias term
        X: Training data (n_samples, n_features)
        y: Labels (n_samples,)
        
    Returns:
        Minimum geometric margin: γ = min_i γ_i
        
    Mathematical foundation:
        γ = min_{i=1,...,m} y_i * (w^T x_i + b) / ||w||
    """
    margins = [geometric_margin(w, b, x_i, y_i) for x_i, y_i in zip(X, y)]
    return min(margins)


def demonstrate_margins():
    """
    Demonstrate functional and geometric margins.
    """
    print("=== Margin Demonstration ===")
    
    # Get trained classifier
    w, b, X, y = demonstrate_linear_classifier()
    
    # Compute margins
    func_margins = [functional_margin(w, b, x_i, y_i) for x_i, y_i in zip(X, y)]
    geom_margins = [geometric_margin(w, b, x_i, y_i) for x_i, y_i in zip(X, y)]
    
    min_func_margin = min_functional_margin(w, b, X, y)
    min_geom_margin = min_geometric_margin(w, b, X, y)
    
    print(f"Minimum functional margin: {min_func_margin:.4f}")
    print(f"Minimum geometric margin: {min_geom_margin:.4f}")
    print(f"Weight norm ||w||: {np.linalg.norm(w):.4f}")
    print(f"Ratio (functional/geometric): {min_func_margin/min_geom_margin:.4f}")
    
    # Demonstrate scale invariance
    print("\n--- Scale Invariance Test ---")
    scale_factors = [0.5, 1.0, 2.0, 5.0]
    
    for scale in scale_factors:
        w_scaled = scale * w
        b_scaled = scale * b
        
        func_margin_scaled = min_functional_margin(w_scaled, b_scaled, X, y)
        geom_margin_scaled = min_geometric_margin(w_scaled, b_scaled, X, y)
        
        print(f"Scale {scale:4.1f}: Functional margin = {func_margin_scaled:8.4f}, "
              f"Geometric margin = {geom_margin_scaled:8.4f}")
    
    print("\nNote: Geometric margin is scale-invariant, functional margin is not!")
    
    return func_margins, geom_margins


# ============================================================================
# Section 6.4: The Optimal Margin Classifier
# ============================================================================

def svm_primal_objective(w, b, X, y, C=1.0):
    """
    Compute the SVM primal objective function.
    
    Args:
        w: Weight vector
        b: Bias term
        X: Training data
        y: Labels
        C: Regularization parameter
        
    Returns:
        Objective value: (1/2)||w||^2 + C * Σ_i max(0, 1 - y_i(w^T x_i + b))
        
    Mathematical foundation:
        This is the soft-margin SVM objective that balances margin maximization
        with classification error minimization.
    """
    # Regularization term
    reg_term = 0.5 * np.dot(w, w)
    
    # Hinge loss term
    hinge_loss = 0
    for x_i, y_i in zip(X, y):
        margin = y_i * (np.dot(w, x_i) + b)
        hinge_loss += max(0, 1 - margin)
    
    return reg_term + C * hinge_loss


def svm_gradient(w, b, X, y, C=1.0):
    """
    Compute gradients of SVM objective with respect to w and b.
    
    Args:
        w: Weight vector
        b: Bias term
        X: Training data
        y: Labels
        C: Regularization parameter
        
    Returns:
        grad_w: Gradient with respect to w
        grad_b: Gradient with respect to b
        
    Mathematical foundation:
        ∂L/∂w = w - C * Σ_i y_i x_i * I[margin_i < 1]
        ∂L/∂b = -C * Σ_i y_i * I[margin_i < 1]
        where I[condition] is the indicator function
    """
    grad_w = w.copy()
    grad_b = 0
    
    for x_i, y_i in zip(X, y):
        margin = y_i * (np.dot(w, x_i) + b)
        if margin < 1:  # Point violates margin or is misclassified
            grad_w -= C * y_i * x_i
            grad_b -= C * y_i
    
    return grad_w, grad_b


def svm_gradient_descent(X, y, learning_rate=0.01, max_iterations=1000, C=1.0):
    """
    Implement SVM using gradient descent.
    
    Args:
        X: Training data
        y: Labels
        learning_rate: Learning rate
        max_iterations: Maximum iterations
        C: Regularization parameter
        
    Returns:
        w: Learned weight vector
        b: Learned bias term
        history: List of objective values
    """
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0
    history = []
    
    print(f"Training SVM with gradient descent (C={C})...")
    
    for iteration in range(max_iterations):
        # Compute objective
        objective = svm_primal_objective(w, b, X, y, C)
        history.append(objective)
        
        # Compute gradients
        grad_w, grad_b = svm_gradient(w, b, X, y, C)
        
        # Update parameters
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        
        # Print progress
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Objective = {objective:.4f}")
    
    return w, b, history


def demonstrate_svm_optimization():
    """
    Demonstrate SVM optimization with different C values.
    """
    print("=== SVM Optimization Demonstration ===")
    
    # Create synthetic data
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                             n_informative=2, n_clusters_per_class=1, 
                             random_state=42)
    y = 2 * y - 1  # Convert to {-1, 1}
    
    # Test different C values
    C_values = [0.1, 1.0, 10.0, 100.0]
    results = {}
    
    for C in C_values:
        print(f"\n--- Training with C = {C} ---")
        
        # Train using gradient descent
        w, b, history = svm_gradient_descent(X, y, learning_rate=0.01, 
                                           max_iterations=500, C=C)
        
        # Compute margins
        min_func_margin = min_functional_margin(w, b, X, y)
        min_geom_margin = min_geometric_margin(w, b, X, y)
        objective = svm_primal_objective(w, b, X, y, C)
        
        results[C] = {
            'w': w,
            'b': b,
            'min_func_margin': min_func_margin,
            'min_geom_margin': min_geom_margin,
            'objective': objective,
            'history': history
        }
        
        print(f"Final objective: {objective:.4f}")
        print(f"Min functional margin: {min_func_margin:.4f}")
        print(f"Min geometric margin: {min_geom_margin:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    for i, C in enumerate(C_values):
        plt.subplot(1, 4, i+1)
        
        w = results[C]['w']
        b = results[C]['b']
        
        # Plot data
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', alpha=0.7)
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', alpha=0.7)
        
        # Plot decision boundary
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x1_range = np.linspace(x1_min, x1_max, 100)
        x2_boundary = linear_classifier_decision_boundary(w, b, x1_range)
        
        plt.plot(x1_range, x2_boundary, 'k-', linewidth=2)
        plt.title(f'C = {C}\nMargin = {results[C]["min_geom_margin"]:.3f}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results


# ============================================================================
# Section 6.5: Lagrange Duality
# ============================================================================

def lagrangian(w, beta, f, h):
    """
    Computes the Lagrangian for equality constraints.
    
    Args:
        w: Variables
        f: Objective function
        h: List of equality constraint functions h_i(w) = 0
        beta: Lagrange multipliers
        
    Returns:
        Lagrangian: L(w, β) = f(w) + Σ_i β_i h_i(w)
        
    Mathematical foundation:
        The Lagrangian combines the objective function with constraint
        violations weighted by Lagrange multipliers.
    """
    return f(w) + sum(b * h_i(w) for b, h_i in zip(beta, h))


def generalized_lagrangian(w, alpha, beta, f, g, h):
    """
    Computes the generalized Lagrangian with inequality and equality constraints.
    
    Args:
        w: Variables
        f: Objective function
        g: List of inequality constraint functions g_i(w) ≤ 0
        h: List of equality constraint functions h_i(w) = 0
        alpha: Lagrange multipliers for inequalities (α_i ≥ 0)
        beta: Lagrange multipliers for equalities
        
    Returns:
        Generalized Lagrangian: L(w, α, β) = f(w) + Σ_i α_i g_i(w) + Σ_i β_i h_i(w)
        
    Mathematical foundation:
        This is the standard form for constrained optimization problems.
        The KKT conditions provide necessary conditions for optimality.
    """
    return (f(w) +
            sum(a * g_i(w) for a, g_i in zip(alpha, g)) +
            sum(b * h_i(w) for b, h_i in zip(beta, h)))


def demonstrate_lagrangian():
    """
    Demonstrate Lagrangian duality with a simple example.
    """
    print("=== Lagrangian Duality Demonstration ===")
    
    # Simple example: minimize f(x) = x^2 subject to x = 1
    def f(x):
        return x**2
    
    def h(x):
        return x - 1  # constraint: x = 1
    
    # Analytical solution
    x_opt = 1.0
    f_opt = f(x_opt)
    
    print(f"Analytical solution: x* = {x_opt}, f(x*) = {f_opt}")
    
    # Test Lagrangian for different β values
    beta_values = np.linspace(-2, 2, 10)
    
    print("\nLagrangian values for different β:")
    print("β\t\tL(x*, β)")
    print("-" * 30)
    
    for beta in beta_values:
        L = lagrangian(x_opt, [beta], f, [h])
        print(f"{beta:6.2f}\t\t{L:8.4f}")
    
    print("\nNote: The Lagrangian equals the objective at the optimal point")
    print("when the constraint is satisfied (h(x*) = 0).")
    
    return x_opt, f_opt


def main():
    """
    Main function to run all SVM margin examples.
    """
    print("SVM Margins and Lagrangian Duality: Examples")
    print("=" * 50)
    
    # Run all examples
    demonstrate_linear_classifier()
    demonstrate_margins()
    demonstrate_svm_optimization()
    demonstrate_lagrangian()
    
    print("\nAll SVM margin examples completed!")


if __name__ == "__main__":
    main() 
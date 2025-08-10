"""
SVM Optimal Margin Classifier: Dual Formulation Implementation
============================================================

This file implements the key concepts from the SVM optimal margin document:
- Primal SVM optimization problem
- Dual SVM optimization problem and Lagrangian formulation
- Support vector identification and interpretation
- Kernel trick motivation and implementation
- Complete SVM training and prediction pipeline

The implementations demonstrate both theoretical concepts and practical
optimization methods with detailed mathematical foundations.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import time


# ============================================================================
# Section 6.6: Primal SVM Implementation
# ============================================================================

def primal_svm_objective(params, X, y):
    """
    Primal SVM objective function.
    
    Args:
        params: [w, b] where w is the weight vector and b is the bias
        X: Training data (n_samples, n_features)
        y: Labels (-1 or 1)
        
    Returns:
        Objective value: (1/2) * ||w||^2
        
    Mathematical foundation:
        This is the primal SVM objective that minimizes the squared norm
        of the weight vector, which maximizes the geometric margin.
    """
    n_features = X.shape[1]
    w = params[:n_features]
    b = params[n_features]
    
    # Objective: (1/2) * ||w||^2
    objective = 0.5 * np.dot(w, w)
    return objective


def primal_svm_constraints(params, X, y):
    """
    Primal SVM constraints: y[i] * (w^T * x[i] + b) >= 1.
    
    Args:
        params: [w, b] parameters
        X: Training data
        y: Labels
        
    Returns:
        Constraint violations (negative values for violated constraints)
        
    Mathematical foundation:
        These constraints ensure that all training points have functional
        margin at least 1, which corresponds to geometric margin 1/||w||.
    """
    n_features = X.shape[1]
    w = params[:n_features]
    b = params[n_features]
    
    # Constraints: y[i] * (w^T * x[i] + b) >= 1
    constraints = y * (np.dot(X, w) + b) - 1
    return constraints


def solve_primal_svm(X, y):
    """
    Solve the primal SVM optimization problem.
    
    Args:
        X: Training data (n_samples, n_features)
        y: Labels (-1 or 1)
        
    Returns:
        w_opt: Optimal weight vector
        b_opt: Optimal bias term
        
    Mathematical foundation:
        This solves the constrained optimization problem:
        minimize (1/2) * ||w||^2
        subject to y[i] * (w^T * x[i] + b) >= 1 for all i
    """
    n_samples, n_features = X.shape
    
    # Initial guess
    w0 = np.zeros(n_features)
    b0 = 0
    params0 = np.concatenate([w0, [b0]])
    
    # Define constraints
    constraints = {
        'type': 'ineq',
        'fun': lambda params: primal_svm_constraints(params, X, y)
    }
    
    print(f"Solving primal SVM with {n_samples} samples and {n_features} features...")
    
    # Solve optimization problem
    result = minimize(
        primal_svm_objective, 
        params0, 
        args=(X, y),
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 1000}
    )
    
    if result.success:
        w_opt = result.x[:n_features]
        b_opt = result.x[n_features]
        print(f"Primal SVM optimization successful!")
        print(f"Final objective value: {result.fun:.6f}")
        return w_opt, b_opt
    else:
        raise ValueError(f"Optimization failed: {result.message}")


# ============================================================================
# Section 6.7: Lagrangian Formulation
# ============================================================================

def constraint_g(w, b, x_i, y_i):
    """
    Individual constraint function g_i(w, b) = -y_i * (w^T * x_i + b) + 1 <= 0.
    
    Args:
        w: Weight vector
        b: Bias term
        x_i: Training example
        y_i: Label
        
    Returns:
        Constraint value (≤ 0 when satisfied)
        
    Mathematical foundation:
        This is the standard form of the SVM constraint for use in the
        Lagrangian formulation.
    """
    return -y_i * (np.dot(w, x_i) + b) + 1


def all_constraints(w, b, X, y):
    """
    All constraint functions for the dataset.
    
    Args:
        w: Weight vector
        b: Bias term
        X: Training data
        y: Labels
        
    Returns:
        Array of constraint values
    """
    return np.array([constraint_g(w, b, X[i], y[i]) for i in range(len(y))])


def lagrangian(w, b, alpha, X, y):
    """
    Lagrangian function for SVM.
    
    Args:
        w: Weight vector
        b: Bias term
        alpha: Lagrange multipliers
        X: Training data
        y: Labels
        
    Returns:
        Lagrangian value: L(w, b, α) = (1/2) * ||w||^2 - Σ(α_i * [y_i * (w^T * x_i + b) - 1])
        
    Mathematical foundation:
        The Lagrangian combines the objective function with constraint
        violations weighted by Lagrange multipliers α_i ≥ 0.
    """
    # First term: (1/2) * ||w||^2
    first_term = 0.5 * np.dot(w, w)
    
    # Second term: -Σ(α_i * [y_i * (w^T * x_i + b) - 1])
    second_term = 0
    for i in range(len(y)):
        constraint = y[i] * (np.dot(w, X[i]) + b) - 1
        second_term -= alpha[i] * constraint
    
    return first_term + second_term


# ============================================================================
# Section 6.8: Dual Formulation Derivation
# ============================================================================

def compute_w_from_alpha(alpha, X, y):
    """
    Compute w from alpha using the dual relationship.
    
    Args:
        alpha: Lagrange multipliers
        X: Training data
        y: Labels
        
    Returns:
        Weight vector: w = Σ(α_i * y_i * x_i)
        
    Mathematical foundation:
        From the KKT conditions, at optimality we have:
        ∂L/∂w = w - Σ(α_i * y_i * x_i) = 0
        Therefore: w = Σ(α_i * y_i * x_i)
    """
    w = np.zeros(X.shape[1])
    for i in range(len(y)):
        w += alpha[i] * y[i] * X[i]
    return w


def compute_w_from_alpha_vectorized(alpha, X, y):
    """
    Vectorized computation of w from alpha.
    
    Args:
        alpha: Lagrange multipliers
        X: Training data
        y: Labels
        
    Returns:
        Weight vector computed efficiently using vectorization
    """
    return np.sum((alpha * y)[:, None] * X, axis=0)


def derivative_b_lagrangian(alpha, y):
    """
    Derivative of Lagrangian with respect to b.
    
    Args:
        alpha: Lagrange multipliers
        y: Labels
        
    Returns:
        ∂L/∂b = Σ(α_i * y_i) = 0
        
    Mathematical foundation:
        This constraint ensures that the bias term is properly determined
        in the dual formulation.
    """
    return np.sum(alpha * y)


def alpha_constraint(alpha, y):
    """
    Constraint on alpha: Σ(α_i * y_i) = 0.
    
    Args:
        alpha: Lagrange multipliers
        y: Labels
        
    Returns:
        Constraint value (should be 0)
    """
    return np.sum(alpha * y)


# ============================================================================
# Section 6.9: Dual Objective Function
# ============================================================================

def dual_objective(alpha, X, y):
    """
    Dual objective function W(α).
    
    Args:
        alpha: Lagrange multipliers
        X: Training data
        y: Labels
        
    Returns:
        Dual objective: W(α) = Σ(α_i) - (1/2) * Σ(α_i * α_j * y_i * y_j * <x_i, x_j>)
        
    Mathematical foundation:
        This is the dual objective obtained by substituting the optimal w
        back into the Lagrangian. The dual problem is:
        maximize W(α) subject to α_i ≥ 0 and Σ(α_i * y_i) = 0
    """
    n_samples = len(y)
    
    # First term: Σ(α_i)
    first_term = np.sum(alpha)
    
    # Second term: (1/2) * Σ(α_i * α_j * y_i * y_j * <x_i, x_j>)
    second_term = 0
    for i in range(n_samples):
        for j in range(n_samples):
            inner_product = np.dot(X[i], X[j])
            second_term += alpha[i] * alpha[j] * y[i] * y[j] * inner_product
    second_term *= 0.5
    
    return first_term - second_term


def dual_objective_vectorized(alpha, X, y):
    """
    Vectorized computation of dual objective.
    
    Args:
        alpha: Lagrange multipliers
        X: Training data
        y: Labels
        
    Returns:
        Dual objective computed efficiently
    """
    # Compute kernel matrix K[i,j] = <x_i, x_j>
    K = np.dot(X, X.T)
    
    # Compute dual objective
    first_term = np.sum(alpha)
    second_term = 0.5 * np.sum(alpha[:, None] * alpha[None, :] * 
                               y[:, None] * y[None, :] * K)
    
    return first_term - second_term


def solve_dual_svm(X, y):
    """
    Solve the dual SVM optimization problem.
    
    Args:
        X: Training data (n_samples, n_features)
        y: Labels (-1 or 1)
        
    Returns:
        alpha_opt: Optimal Lagrange multipliers
        w_opt: Optimal weight vector
        b_opt: Optimal bias term
        
    Mathematical foundation:
        This solves the dual optimization problem:
        maximize Σ(α_i) - (1/2) * Σ(α_i * α_j * y_i * y_j * <x_i, x_j>)
        subject to α_i ≥ 0 and Σ(α_i * y_i) = 0
    """
    n_samples = len(y)
    
    # Initial guess for alpha
    alpha0 = np.ones(n_samples) / n_samples
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda alpha: alpha_constraint(alpha, y)},
        {'type': 'ineq', 'fun': lambda alpha: alpha}  # α_i ≥ 0
    ]
    
    print(f"Solving dual SVM with {n_samples} samples...")
    
    # Solve optimization problem
    result = minimize(
        lambda alpha: -dual_objective_vectorized(alpha, X, y),  # Minimize negative of dual
        alpha0,
        constraints=constraints,
        method='SLSQP',
        options={'maxiter': 1000}
    )
    
    if result.success:
        alpha_opt = result.x
        w_opt = compute_w_from_alpha_vectorized(alpha_opt, X, y)
        b_opt = compute_b_from_support_vectors(alpha_opt, X, y, w_opt)
        
        print(f"Dual SVM optimization successful!")
        print(f"Final dual objective value: {-result.fun:.6f}")
        print(f"Number of support vectors: {np.sum(alpha_opt > 1e-5)}")
        
        return alpha_opt, w_opt, b_opt
    else:
        raise ValueError(f"Optimization failed: {result.message}")


def compute_b_from_support_vectors(alpha, X, y, w):
    """
    Compute optimal bias term from support vectors.
    
    Args:
        alpha: Lagrange multipliers
        X: Training data
        y: Labels
        w: Weight vector
        
    Returns:
        Optimal bias term
        
    Mathematical foundation:
        For any support vector (α_i > 0), we have:
        y_i * (w^T * x_i + b) = 1
        Therefore: b = y_i - w^T * x_i
        We average this over all support vectors for numerical stability.
    """
    # Find support vectors (α_i > 0)
    sv_indices = np.where(alpha > 1e-5)[0]
    
    if len(sv_indices) == 0:
        return 0.0
    
    # Compute b for each support vector
    b_values = []
    for i in sv_indices:
        b_i = y[i] - np.dot(w, X[i])
        b_values.append(b_i)
    
    # Return average
    return np.mean(b_values)


def get_support_vectors(alpha, X, y, tolerance=1e-5):
    """
    Identify support vectors from Lagrange multipliers.
    
    Args:
        alpha: Lagrange multipliers
        X: Training data
        y: Labels
        tolerance: Threshold for considering α_i > 0
        
    Returns:
        X_sv: Support vector features
        y_sv: Support vector labels
        alpha_sv: Support vector multipliers
        
    Mathematical foundation:
        Support vectors are training points with α_i > 0, meaning they
        lie on or within the margin boundary.
    """
    sv_indices = np.where(alpha > tolerance)[0]
    return X[sv_indices], y[sv_indices], alpha[sv_indices]


# ============================================================================
# Section 6.10: Prediction Functions
# ============================================================================

def svm_decision_function(x, X, y, alpha, b):
    """
    SVM decision function for linear kernel.
    
    Args:
        x: Input point
        X: Training data
        y: Labels
        alpha: Lagrange multipliers
        b: Bias term
        
    Returns:
        Decision value: f(x) = Σ(α_i * y_i * <x_i, x>) + b
        
    Mathematical foundation:
        This is the dual representation of the decision function,
        which only requires inner products between input points.
    """
    decision_value = b
    for i in range(len(y)):
        decision_value += alpha[i] * y[i] * np.dot(X[i], x)
    return decision_value


def svm_predict(x, X, y, alpha, b):
    """
    SVM prediction function.
    
    Args:
        x: Input point
        X: Training data
        y: Labels
        alpha: Lagrange multipliers
        b: Bias term
        
    Returns:
        Predicted label: sign(f(x))
    """
    decision_value = svm_decision_function(x, X, y, alpha, b)
    return 1 if decision_value >= 0 else -1


def svm_decision_function_kernel(x, X, y, alpha, b, kernel_func=None):
    """
    SVM decision function with arbitrary kernel.
    
    Args:
        x: Input point
        X: Training data
        y: Labels
        alpha: Lagrange multipliers
        b: Bias term
        kernel_func: Kernel function (default: linear)
        
    Returns:
        Decision value: f(x) = Σ(α_i * y_i * K(x_i, x)) + b
        
    Mathematical foundation:
        This is the kernelized version of the decision function,
        which allows us to work in high-dimensional feature spaces
        without explicitly computing the features.
    """
    if kernel_func is None:
        kernel_func = lambda x1, x2: np.dot(x1, x2)
    
    decision_value = b
    for i in range(len(y)):
        decision_value += alpha[i] * y[i] * kernel_func(X[i], x)
    return decision_value


def svm_predict_kernel(x, X, y, alpha, b, kernel_func=None):
    """
    SVM prediction function with kernel.
    
    Args:
        x: Input point
        X: Training data
        y: Labels
        alpha: Lagrange multipliers
        b: Bias term
        kernel_func: Kernel function
        
    Returns:
        Predicted label: sign(f(x))
    """
    decision_value = svm_decision_function_kernel(x, X, y, alpha, b, kernel_func)
    return 1 if decision_value >= 0 else -1


def svm_decision_function_support_vectors_only(x, X_sv, y_sv, alpha_sv, b, kernel_func=None):
    """
    SVM decision function using only support vectors.
    
    Args:
        x: Input point
        X_sv: Support vector features
        y_sv: Support vector labels
        alpha_sv: Support vector multipliers
        b: Bias term
        kernel_func: Kernel function
        
    Returns:
        Decision value computed efficiently using only support vectors
        
    Mathematical foundation:
        Since α_i = 0 for non-support vectors, we only need to compute
        kernel evaluations with support vectors, making prediction efficient.
    """
    if kernel_func is None:
        kernel_func = lambda x1, x2: np.dot(x1, x2)
    
    decision_value = b
    for i in range(len(y_sv)):
        decision_value += alpha_sv[i] * y_sv[i] * kernel_func(X_sv[i], x)
    return decision_value


# ============================================================================
# Section 6.11: Kernel Functions
# ============================================================================

def linear_kernel(x1, x2):
    """Linear kernel: K(x1, x2) = <x1, x2>"""
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, degree=2, gamma=1.0, coef0=0):
    """
    Polynomial kernel: K(x1, x2) = (γ<x1, x2> + r)^d
    
    Args:
        x1, x2: Input vectors
        degree: Polynomial degree
        gamma: Scaling parameter
        coef0: Bias parameter
        
    Returns:
        Kernel value
    """
    return (gamma * np.dot(x1, x2) + coef0) ** degree


def rbf_kernel(x1, x2, gamma=1.0):
    """
    RBF (Gaussian) kernel: K(x1, x2) = exp(-γ||x1 - x2||^2)
    
    Args:
        x1, x2: Input vectors
        gamma: Bandwidth parameter
        
    Returns:
        Kernel value
    """
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))


# ============================================================================
# Section 6.12: Complete SVM Implementation
# ============================================================================

def complete_svm_implementation(X, y, kernel_func=None):
    """
    Complete SVM implementation with training and prediction.
    
    Args:
        X: Training data
        y: Labels
        kernel_func: Kernel function (default: linear)
        
    Returns:
        Trained SVM model with predict method
    """
    # Solve dual SVM
    alpha, w, b = solve_dual_svm(X, y)
    
    # Get support vectors
    X_sv, y_sv, alpha_sv = get_support_vectors(alpha, X, y)
    
    def predict(x):
        """Prediction function"""
        return svm_predict_kernel(x, X, y, alpha, b, kernel_func)
    
    def decision_function(x):
        """Decision function"""
        return svm_decision_function_kernel(x, X, y, alpha, b, kernel_func)
    
    # Return model object
    model = {
        'alpha': alpha,
        'w': w,
        'b': b,
        'X_sv': X_sv,
        'y_sv': y_sv,
        'alpha_sv': alpha_sv,
        'predict': predict,
        'decision_function': decision_function,
        'n_support_vectors': len(X_sv)
    }
    
    return model


# ============================================================================
# Section 6.13: Visualization and Examples
# ============================================================================

def plot_svm_decision_boundary(X, y, w, b, support_vectors=None, title="SVM Decision Boundary"):
    """
    Plot SVM decision boundary and support vectors.
    
    Args:
        X: Training data
        y: Labels
        w: Weight vector
        b: Bias term
        support_vectors: Support vector indices
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class +1', alpha=0.7)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', label='Class -1', alpha=0.7)
    
    # Plot decision boundary
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                           np.linspace(x2_min, x2_max, 100))
    
    Z = np.zeros(xx1.shape)
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            x = np.array([xx1[i, j], xx2[i, j]])
            Z[i, j] = np.dot(w, x) + b
    
    plt.contour(xx1, xx2, Z, levels=[0], colors='black', linewidths=2)
    plt.contour(xx1, xx2, Z, levels=[-1, 1], colors='gray', linewidths=1, linestyles='--')
    
    # Highlight support vectors
    if support_vectors is not None:
        plt.scatter(X[support_vectors, 0], X[support_vectors, 1], 
                   s=100, facecolors='none', edgecolors='black', linewidth=2, 
                   label='Support Vectors')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def example_linear_svm():
    """
    Example of linear SVM on linearly separable data.
    """
    print("=== Linear SVM Example ===")
    
    # Create linearly separable data
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1,
                             random_state=42)
    y = 2 * y - 1  # Convert to {-1, 1}
    
    # Train SVM
    model = complete_svm_implementation(X, y, linear_kernel)
    
    print(f"Number of support vectors: {model['n_support_vectors']}")
    print(f"Bias term: {model['b']:.4f}")
    
    # Evaluate accuracy
    predictions = [model['predict'](x) for x in X]
    accuracy = np.mean(np.array(predictions) == y)
    print(f"Training accuracy: {accuracy:.4f}")
    
    # Visualize
    sv_indices = np.where(model['alpha'] > 1e-5)[0]
    plot_svm_decision_boundary(X, y, model['w'], model['b'], 
                              support_vectors=sv_indices,
                              title="Linear SVM Decision Boundary")
    
    return model


def example_kernel_svm():
    """
    Example of kernel SVM on non-linearly separable data.
    """
    print("=== Kernel SVM Example ===")
    
    # Create non-linearly separable data (circles)
    np.random.seed(42)
    X, y = make_circles(n_samples=100, noise=0.1, factor=0.5, random_state=42)
    y = 2 * y - 1  # Convert to {-1, 1}
    
    # Define RBF kernel
    def rbf_kernel_wrapper(x1, x2):
        return rbf_kernel(x1, x2, gamma=1.0)
    
    # Train SVM with RBF kernel
    model = complete_svm_implementation(X, y, rbf_kernel_wrapper)
    
    print(f"Number of support vectors: {model['n_support_vectors']}")
    print(f"Bias term: {model['b']:.4f}")
    
    # Evaluate accuracy
    predictions = [model['predict'](x) for x in X]
    accuracy = np.mean(np.array(predictions) == y)
    print(f"Training accuracy: {accuracy:.4f}")
    
    # Visualize decision boundary
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class +1', alpha=0.7)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', label='Class -1', alpha=0.7)
    
    # Plot decision boundary
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 50),
                           np.linspace(x2_min, x2_max, 50))
    
    Z = np.zeros(xx1.shape)
    for i in range(xx1.shape[0]):
        for j in range(xx1.shape[1]):
            x = np.array([xx1[i, j], xx2[i, j]])
            Z[i, j] = model['decision_function'](x)
    
    plt.contour(xx1, xx2, Z, levels=[0], colors='black', linewidths=2)
    plt.contourf(xx1, xx2, Z, levels=20, alpha=0.3, cmap='RdYlBu')
    
    # Highlight support vectors
    sv_indices = np.where(model['alpha'] > 1e-5)[0]
    plt.scatter(X[sv_indices, 0], X[sv_indices, 1], 
               s=100, facecolors='none', edgecolors='black', linewidth=2, 
               label='Support Vectors')
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Kernel SVM Decision Boundary (RBF)')
    plt.legend()
    plt.colorbar()
    plt.show()
    
    return model


def compare_primal_dual():
    """
    Compare primal and dual SVM formulations.
    """
    print("=== Primal vs Dual SVM Comparison ===")
    
    # Create small dataset for comparison
    np.random.seed(42)
    X, y = make_classification(n_samples=50, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1,
                             random_state=42)
    y = 2 * y - 1
    
    print("Training primal SVM...")
    start_time = time.time()
    w_primal, b_primal = solve_primal_svm(X, y)
    primal_time = time.time() - start_time
    
    print("Training dual SVM...")
    start_time = time.time()
    alpha_dual, w_dual, b_dual = solve_dual_svm(X, y)
    dual_time = time.time() - start_time
    
    print(f"\nComparison Results:")
    print(f"Primal SVM time: {primal_time:.4f} seconds")
    print(f"Dual SVM time: {dual_time:.4f} seconds")
    print(f"Speedup: {primal_time/dual_time:.2f}x")
    
    # Compare solutions
    w_diff = np.linalg.norm(w_primal - w_dual)
    b_diff = abs(b_primal - b_dual)
    
    print(f"Weight vector difference: {w_diff:.6f}")
    print(f"Bias term difference: {b_diff:.6f}")
    
    # Compare predictions
    predictions_primal = [1 if np.dot(w_primal, x) + b_primal >= 0 else -1 for x in X]
    predictions_dual = [svm_predict(x, X, y, alpha_dual, b_dual) for x in X]
    
    agreement = np.mean(np.array(predictions_primal) == np.array(predictions_dual))
    print(f"Prediction agreement: {agreement:.4f}")
    
    return w_primal, b_primal, alpha_dual, w_dual, b_dual


def main():
    """
    Main function to run all SVM optimal margin examples.
    """
    print("SVM Optimal Margin Classifier: Examples")
    print("=" * 50)
    
    # Run all examples
    example_linear_svm()
    example_kernel_svm()
    compare_primal_dual()
    
    print("\nAll SVM optimal margin examples completed!")


if __name__ == "__main__":
    main() 
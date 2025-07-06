"""
SVM Optimal Margin Classifier Implementation
============================================

This file contains complete implementations of Support Vector Machines (SVM)
for optimal margin classification, including both primal and dual formulations.

Key Concepts:
- Primal SVM optimization problem
- Dual SVM optimization problem  
- Lagrangian formulation
- Support vector identification
- Kernel functions
- Prediction functions

Equations implemented:
- (6.8) Primal optimization problem
- (6.9) Lagrangian function
- (6.10) w = sum(α_i * y_i * x_i)
- (6.11) sum(α_i * y_i) = 0
- (6.12) Dual optimization problem
- (6.13) Optimal b computation
- (6.14) Decision function
- (6.15) Kernel decision function
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler


# ============================================================================
# PRIMAL SVM IMPLEMENTATION
# ============================================================================

def primal_svm_objective(params, X, y):
    """
    Primal SVM objective function
    params: [w, b] where w is the weight vector and b is the bias
    """
    n_features = X.shape[1]
    w = params[:n_features]
    b = params[n_features]
    
    # Objective: (1/2) * ||w||^2
    objective = 0.5 * np.dot(w, w)
    return objective


def primal_svm_constraints(params, X, y):
    """
    Primal SVM constraints: y[i] * (w^T * x[i] + b) >= 1
    Returns negative values for violated constraints
    """
    n_features = X.shape[1]
    w = params[:n_features]
    b = params[n_features]
    
    # Constraints: y[i] * (w^T * x[i] + b) >= 1
    constraints = y * (np.dot(X, w) + b) - 1
    return constraints


def solve_primal_svm(X, y):
    """
    Solve the primal SVM optimization problem
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
    
    # Solve optimization problem
    result = minimize(
        primal_svm_objective, 
        params0, 
        args=(X, y),
        constraints=constraints,
        method='SLSQP'
    )
    
    if result.success:
        w_opt = result.x[:n_features]
        b_opt = result.x[n_features]
        return w_opt, b_opt
    else:
        raise ValueError("Optimization failed")


# ============================================================================
# CONSTRAINT FUNCTIONS
# ============================================================================

def constraint_g(w, b, x_i, y_i):
    """
    Individual constraint function g_i(w, b) = -y_i * (w^T * x_i + b) + 1 <= 0
    """
    return -y_i * (np.dot(w, x_i) + b) + 1


def all_constraints(w, b, X, y):
    """
    All constraint functions for the dataset
    """
    return np.array([constraint_g(w, b, X[i], y[i]) for i in range(len(y))])


# ============================================================================
# LAGRANGIAN FORMULATION
# ============================================================================

def lagrangian(w, b, alpha, X, y):
    """
    Lagrangian function for SVM
    L(w, b, α) = (1/2) * ||w||^2 - sum(α_i * [y_i * (w^T * x_i + b) - 1])
    """
    # First term: (1/2) * ||w||^2
    first_term = 0.5 * np.dot(w, w)
    
    # Second term: -sum(α_i * [y_i * (w^T * x_i + b) - 1])
    second_term = 0
    for i in range(len(y)):
        constraint = y[i] * (np.dot(w, X[i]) + b) - 1
        second_term -= alpha[i] * constraint
    
    return first_term + second_term


# ============================================================================
# DUAL FORM DERIVATION
# ============================================================================

def compute_w_from_alpha(alpha, X, y):
    """
    Compute w from alpha using equation (6.10)
    w = sum(α_i * y_i * x_i)
    """
    w = np.zeros(X.shape[1])
    for i in range(len(y)):
        w += alpha[i] * y[i] * X[i]
    return w


def compute_w_from_alpha_vectorized(alpha, X, y):
    """
    Vectorized computation of w from alpha
    """
    return np.sum((alpha * y)[:, None] * X, axis=0)


def derivative_b_lagrangian(alpha, y):
    """
    Derivative of Lagrangian with respect to b
    ∂L/∂b = sum(α_i * y_i) = 0
    """
    return np.sum(alpha * y)


def alpha_constraint(alpha, y):
    """
    Constraint: sum(α_i * y_i) = 0
    """
    return np.sum(alpha * y)


# ============================================================================
# DUAL OBJECTIVE FUNCTIONS
# ============================================================================

def dual_objective(alpha, X, y):
    """
    Dual objective function W(α)
    W(α) = sum(α_i) - (1/2) * sum(sum(α_i * α_j * y_i * y_j * <x_i, x_j>))
    """
    n_samples = len(y)
    
    # First term: sum(α_i)
    first_term = np.sum(alpha)
    
    # Second term: (1/2) * sum(sum(α_i * α_j * y_i * y_j * <x_i, x_j>))
    second_term = 0
    for i in range(n_samples):
        for j in range(n_samples):
            inner_product = np.dot(X[i], X[j])
            second_term += alpha[i] * alpha[j] * y[i] * y[j] * inner_product
    second_term *= 0.5
    
    return first_term - second_term


def dual_objective_vectorized(alpha, X, y):
    """
    Vectorized dual objective function
    """
    # Compute kernel matrix K[i,j] = <x_i, x_j>
    K = np.dot(X, X.T)
    
    # Compute y[i] * y[j] * K[i,j]
    yyK = np.outer(y, y) * K
    
    # Compute objective
    first_term = np.sum(alpha)
    second_term = 0.5 * np.dot(alpha, np.dot(yyK, alpha))
    
    return first_term - second_term


# ============================================================================
# DUAL SVM SOLVER
# ============================================================================

def solve_dual_svm(X, y):
    """
    Solve the dual SVM optimization problem
    """
    n_samples = len(y)
    
    # Initial guess for alpha
    alpha0 = np.zeros(n_samples)
    
    # Define constraints
    # 1. α_i >= 0 for all i
    bounds = [(0, None) for _ in range(n_samples)]
    
    # 2. sum(α_i * y_i) = 0
    constraints = {
        'type': 'eq',
        'fun': lambda alpha: np.sum(alpha * y)
    }
    
    # Solve optimization problem (maximize dual objective)
    result = minimize(
        lambda alpha: -dual_objective_vectorized(alpha, X, y),  # Negative for maximization
        alpha0,
        constraints=constraints,
        bounds=bounds,
        method='SLSQP'
    )
    
    if result.success:
        alpha_opt = result.x
        w_opt = compute_w_from_alpha_vectorized(alpha_opt, X, y)
        b_opt = compute_b_from_support_vectors(alpha_opt, X, y, w_opt)
        return w_opt, b_opt, alpha_opt
    else:
        raise ValueError("Dual optimization failed")


# ============================================================================
# SUPPORT VECTOR OPERATIONS
# ============================================================================

def compute_b_from_support_vectors(alpha, X, y, w):
    """
    Compute optimal b from support vectors using equation (6.13)
    b* = (-max_{i:y_i=-1} w*^T x_i + min_{i:y_i=1} w*^T x_i) / 2
    """
    # Compute w^T * x for all points
    wx = np.dot(X, w)
    
    # Find negative and positive examples
    neg_idx = np.where(y == -1)[0]
    pos_idx = np.where(y == 1)[0]
    
    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise ValueError("Need both positive and negative examples")
    
    # Compute max for negative examples and min for positive examples
    max_neg = np.max(wx[neg_idx])
    min_pos = np.min(wx[pos_idx])
    
    # Compute b
    b = (-max_neg + min_pos) / 2
    return b


def get_support_vectors(alpha, X, y, tolerance=1e-5):
    """
    Identify support vectors (points with α_i > tolerance)
    """
    support_vector_indices = np.where(alpha > tolerance)[0]
    return support_vector_indices, X[support_vector_indices], y[support_vector_indices]


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def svm_decision_function(x, X, y, alpha, b):
    """
    Compute decision function using equation (6.14)
    f(x) = w^T * x + b = sum(α_i * y_i * <x_i, x>) + b
    """
    # Compute inner products between x and all training points
    inner_products = np.dot(X, x)
    
    # Compute decision function
    decision = np.sum(alpha * y * inner_products) + b
    return decision


def svm_predict(x, X, y, alpha, b):
    """
    Make prediction for a new point x
    """
    decision = svm_decision_function(x, X, y, alpha, b)
    return 1 if decision > 0 else -1


def svm_decision_function_kernel(x, X, y, alpha, b, kernel_func=None):
    """
    Compute decision function using kernel trick (equation 6.15)
    f(x) = sum(α_i * y_i * K(x_i, x)) + b
    """
    if kernel_func is None:
        # Linear kernel: K(x_i, x) = <x_i, x>
        kernel_values = np.dot(X, x)
    else:
        # Custom kernel function
        kernel_values = np.array([kernel_func(X[i], x) for i in range(len(X))])
    
    # Compute decision function
    decision = np.sum(alpha * y * kernel_values) + b
    return decision


def svm_predict_kernel(x, X, y, alpha, b, kernel_func=None):
    """
    Make prediction using kernel trick
    """
    decision = svm_decision_function_kernel(x, X, y, alpha, b, kernel_func)
    return 1 if decision > 0 else -1


def svm_decision_function_support_vectors_only(x, X_sv, y_sv, alpha_sv, b, kernel_func=None):
    """
    Compute decision function using only support vectors (more efficient)
    """
    if kernel_func is None:
        # Linear kernel
        kernel_values = np.dot(X_sv, x)
    else:
        # Custom kernel function
        kernel_values = np.array([kernel_func(X_sv[i], x) for i in range(len(X_sv))])
    
    # Compute decision function using only support vectors
    decision = np.sum(alpha_sv * y_sv * kernel_values) + b
    return decision


# ============================================================================
# KERNEL FUNCTIONS
# ============================================================================

def linear_kernel(x1, x2):
    """Linear kernel: K(x1, x2) = <x1, x2>"""
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, degree=2, gamma=1.0, coef0=0):
    """Polynomial kernel: K(x1, x2) = (γ * <x1, x2> + r)^d"""
    return (gamma * np.dot(x1, x2) + coef0) ** degree


def rbf_kernel(x1, x2, gamma=1.0):
    """RBF kernel: K(x1, x2) = exp(-γ * ||x1 - x2||^2)"""
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))


# ============================================================================
# COMPLETE SVM IMPLEMENTATION
# ============================================================================

def complete_svm_implementation(X, y, kernel_func=None):
    """
    Complete SVM implementation from training to prediction
    """
    # Step 1: Solve dual problem
    w_opt, b_opt, alpha_opt = solve_dual_svm(X, y)
    
    # Step 2: Identify support vectors
    sv_indices, X_sv, y_sv = get_support_vectors(alpha_opt, X, y)
    alpha_sv = alpha_opt[sv_indices]
    
    # Step 3: Create prediction function
    def predict(x):
        return svm_decision_function_support_vectors_only(
            x, X_sv, y_sv, alpha_sv, b_opt, kernel_func
        )
    
    return {
        'w': w_opt,
        'b': b_opt,
        'alpha': alpha_opt,
        'support_vectors': X_sv,
        'support_vector_labels': y_sv,
        'support_vector_alphas': alpha_sv,
        'predict': predict
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_svm_decision_boundary(X, y, w, b, support_vectors=None, title="SVM Decision Boundary"):
    """
    Plot SVM decision boundary and support vectors
    """
    plt.figure(figsize=(10, 8))
    
    # Plot data points
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class 1', alpha=0.7)
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', label='Class -1', alpha=0.7)
    
    # Plot support vectors if provided
    if support_vectors is not None:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                   c='green', s=100, marker='o', edgecolors='black', 
                   linewidth=2, label='Support Vectors')
    
    # Create meshgrid for decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Compute decision function for each point
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j]])
            Z[i, j] = np.dot(w, point) + b
    
    # Plot decision boundary
    plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2)
    plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], 
                colors=['blue', 'red'], alpha=0.3)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def example_linear_svm():
    """
    Example: Linear SVM on synthetic data
    """
    print("=== Linear SVM Example ===")
    
    # Generate synthetic data
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.0, random_state=42)
    y = 2 * y - 1  # Convert to {-1, 1}
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount((y + 1) // 2)}")
    
    # Solve using dual formulation
    w_dual, b_dual, alpha_dual = solve_dual_svm(X, y)
    
    # Get support vectors
    sv_indices, X_sv, y_sv = get_support_vectors(alpha_dual, X, y)
    
    print(f"Number of support vectors: {len(sv_indices)}")
    print(f"Weight vector w: {w_dual}")
    print(f"Bias b: {b_dual}")
    
    # Test predictions
    test_point = np.array([0.5, 0.5])
    prediction = svm_predict(test_point, X, y, alpha_dual, b_dual)
    print(f"Prediction for test point {test_point}: {prediction}")
    
    # Plot results
    plot_svm_decision_boundary(X, y, w_dual, b_dual, X_sv, "Linear SVM Decision Boundary")
    
    return w_dual, b_dual, alpha_dual


def example_kernel_svm():
    """
    Example: Kernel SVM on non-linear data
    """
    print("\n=== Kernel SVM Example ===")
    
    # Generate non-linear data (XOR-like pattern)
    np.random.seed(42)
    n_samples = 200
    
    # Create XOR-like pattern
    X = np.random.randn(n_samples, 2) * 0.5
    y = np.sign(X[:, 0] * X[:, 1])  # XOR-like pattern
    
    print(f"Data shape: {X.shape}")
    print(f"Class distribution: {np.bincount((y + 1) // 2)}")
    
    # Use RBF kernel
    def rbf_kernel_wrapper(x1, x2):
        return rbf_kernel(x1, x2, gamma=1.0)
    
    # Solve using dual formulation with kernel
    w_dual, b_dual, alpha_dual = solve_dual_svm(X, y)
    
    # Get support vectors
    sv_indices, X_sv, y_sv = get_support_vectors(alpha_dual, X, y)
    
    print(f"Number of support vectors: {len(sv_indices)}")
    
    # Test predictions with kernel
    test_point = np.array([0.5, 0.5])
    prediction = svm_predict_kernel(test_point, X, y, alpha_dual, b_dual, rbf_kernel_wrapper)
    print(f"Prediction for test point {test_point}: {prediction}")
    
    # Plot results (note: decision boundary visualization is more complex with kernels)
    plot_svm_decision_boundary(X, y, w_dual, b_dual, X_sv, "Kernel SVM Decision Boundary")
    
    return w_dual, b_dual, alpha_dual


def compare_primal_dual():
    """
    Compare primal and dual SVM solutions
    """
    print("\n=== Primal vs Dual Comparison ===")
    
    # Generate simple data
    X, y = make_blobs(n_samples=50, centers=2, cluster_std=0.8, random_state=42)
    y = 2 * y - 1
    
    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Solve using primal formulation
    w_primal, b_primal = solve_primal_svm(X, y)
    
    # Solve using dual formulation
    w_dual, b_dual, alpha_dual = solve_dual_svm(X, y)
    
    print("Primal solution:")
    print(f"  w: {w_primal}")
    print(f"  b: {b_primal}")
    
    print("Dual solution:")
    print(f"  w: {w_dual}")
    print(f"  b: {b_dual}")
    
    # Compare solutions
    w_diff = np.linalg.norm(w_primal - w_dual)
    b_diff = abs(b_primal - b_dual)
    
    print(f"Difference in w: {w_diff:.6f}")
    print(f"Difference in b: {b_diff:.6f}")
    
    return w_primal, b_primal, w_dual, b_dual


if __name__ == "__main__":
    """
    Run examples when script is executed directly
    """
    print("SVM Optimal Margin Classifier Examples")
    print("=" * 50)
    
    try:
        # Example 1: Linear SVM
        w1, b1, alpha1 = example_linear_svm()
        
        # Example 2: Kernel SVM
        w2, b2, alpha2 = example_kernel_svm()
        
        # Example 3: Compare primal and dual
        w_primal, b_primal, w_dual, b_dual = compare_primal_dual()
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install numpy scipy matplotlib scikit-learn") 
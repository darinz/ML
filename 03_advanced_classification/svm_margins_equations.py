import numpy as np
# SVM Margins and Lagrangian Equations - Python Implementations

# 6.2 Notation (option reading)
def linear_classifier_predict(x, w, b):
    """
    Predicts the class label for input x using weights w and bias b.
    Returns 1 if w^T x + b >= 0, else -1.
    """
    return 1 if np.dot(w, x) + b >= 0 else -1

# 6.3 Functional and geometric margins (option reading)
# Functional margin for a single example
def functional_margin(w, b, x_i, y_i):
    """
    Computes the functional margin for a single training example (x_i, y_i).
    """
    return y_i * (np.dot(w, x_i) + b)

# Minimum functional margin over a dataset
def min_functional_margin(w, b, X, y):
    """
    Computes the minimum functional margin over a dataset.
    X: array of shape (n_samples, n_features)
    y: array of shape (n_samples,)
    """
    margins = [functional_margin(w, b, x_i, y_i) for x_i, y_i in zip(X, y)]
    return min(margins)

# Geometric margin for a single example
def geometric_margin(w, b, x_i, y_i):
    """
    Computes the geometric margin for a single training example (x_i, y_i).
    """
    norm_w = np.linalg.norm(w)
    return y_i * (np.dot(w, x_i) + b) / norm_w

# Minimum geometric margin over a dataset
def min_geometric_margin(w, b, X, y):
    """
    Computes the minimum geometric margin over a dataset.
    """
    margins = [geometric_margin(w, b, x_i, y_i) for x_i, y_i in zip(X, y)]
    return min(margins)

# 6.4 The optimal margin classifier (option reading)
# SVM primal optimization (conceptual, for use with a QP solver)
# This is a conceptual representation; actual QP solvers require matrix setup.
# Minimize (1/2) * ||w||^2
# Subject to: y_i * (w^T x_i + b) >= 1 for all i
# Example using cvxopt (not a full implementation, just the setup):
try:
    from cvxopt import matrix, solvers
    def svm_qp_solver(X, y):
        """
        Solves the hard-margin SVM QP problem for linearly separable data.
        X: (n_samples, n_features)
        y: (n_samples,)
        """
        n_samples, n_features = X.shape
        P = matrix(np.eye(n_features), tc='d')
        q = matrix(np.zeros(n_features), tc='d')
        G = matrix(-y[:, None] * X, tc='d')
        h = matrix(-np.ones(n_samples), tc='d')
        # This is a simplified version; a full implementation would include b and slack variables for soft margin.
        sol = solvers.qp(P, q, G, h)
        w = np.array(sol['x']).flatten()
        return w
except ImportError:
    def svm_qp_solver(X, y):
        raise ImportError("cvxopt is required for this function.")

# 6.5 Lagrange duality (optional reading)
# Lagrangian for equality constraints
def lagrangian(w, beta, f, h):
    """
    Computes the Lagrangian for equality constraints.
    f: function f(w)
    h: list of constraint functions h_i(w)
    beta: list/array of Lagrange multipliers
    """
    return f(w) + sum(b * h_i(w) for b, h_i in zip(beta, h))

# Generalized Lagrangian (with inequality and equality constraints)
def generalized_lagrangian(w, alpha, beta, f, g, h):
    """
    Computes the generalized Lagrangian.
    f: function f(w)
    g: list of inequality constraint functions g_i(w)
    h: list of equality constraint functions h_i(w)
    alpha: list/array of Lagrange multipliers for inequalities
    beta: list/array of Lagrange multipliers for equalities
    """
    return (f(w) +
            sum(a * g_i(w) for a, g_i in zip(alpha, g)) +
            sum(b * h_i(w) for b, h_i in zip(beta, h))) 
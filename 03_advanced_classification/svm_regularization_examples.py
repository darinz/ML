"""
SVM Regularization: Slack Variables and SMO Algorithm Implementation
==================================================================

This file implements the key concepts from the SVM regularization document:
- Soft-margin SVM with slack variables
- The regularization parameter C and its interpretation
- Sequential Minimal Optimization (SMO) algorithm
- KKT conditions and convergence criteria
- Practical implementation with different kernels

The implementations demonstrate both theoretical concepts and practical
optimization methods with detailed mathematical foundations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import time


# ============================================================================
# Section 6.14: Soft-Margin SVM with Slack Variables
# ============================================================================

class SVMRegularization:
    """
    Implementation of SVM with regularization (slack variables) for non-separable case.
    
    This class implements the soft-margin SVM that allows some training points
    to violate the margin or be misclassified, controlled by the parameter C.
    
    Mathematical foundation:
        Primal problem:
        minimize (1/2)||w||^2 + C * Σ_i ξ_i
        subject to y_i(w^T x_i + b) ≥ 1 - ξ_i for all i
        and ξ_i ≥ 0 for all i
        
        Dual problem:
        maximize Σ_i α_i - (1/2)Σ_i,j α_i α_j y_i y_j K(x_i, x_j)
        subject to 0 ≤ α_i ≤ C for all i
        and Σ_i α_i y_i = 0
    """
    
    def __init__(self, C=1.0, kernel='linear', tol=1e-3, max_iter=1000):
        """
        Initialize SVM with regularization.
        
        Args:
            C: Regularization parameter (controls trade-off between margin and slack)
            kernel: Kernel function ('linear', 'rbf', 'poly')
            tol: Convergence tolerance
            max_iter: Maximum iterations for SMO
        """
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.max_iter = max_iter
        self.alphas = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.X_train = None
        self.y_train = None
        
    def linear_kernel(self, x1, x2):
        """
        Linear kernel: K(x1, x2) = <x1, x2>
        
        Args:
            x1, x2: Input vectors
            
        Returns:
            Kernel value
        """
        return np.dot(x1, x2)
    
    def rbf_kernel(self, x1, x2, gamma=1.0):
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
    
    def polynomial_kernel(self, x1, x2, degree=2, gamma=1.0, coef0=0):
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
    
    def compute_kernel_matrix(self, X):
        """
        Compute kernel matrix K[i,j] = K(x_i, x_j).
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            Kernel matrix (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == 'linear':
                    K[i, j] = self.linear_kernel(X[i], X[j])
                elif self.kernel == 'rbf':
                    K[i, j] = self.rbf_kernel(X[i], X[j])
                elif self.kernel == 'poly':
                    K[i, j] = self.polynomial_kernel(X[i], X[j])
        
        return K
    
    def objective_function(self, alphas, K, y):
        """
        Dual objective function.
        
        Args:
            alphas: Lagrange multipliers
            K: Kernel matrix
            y: Target labels
            
        Returns:
            Dual objective value: W(α) = Σα_i - (1/2)ΣΣ y_i y_j α_i α_j K(x_i, x_j)
        """
        n_samples = len(alphas)
        term1 = np.sum(alphas)
        term2 = 0.5 * np.sum(alphas[:, np.newaxis] * alphas * 
                            (y[:, np.newaxis] * y) * K)
        return term1 - term2
    
    def kkt_conditions(self, alphas, X, y, b, K):
        """
        Check KKT conditions for convergence.
        
        Args:
            alphas: Lagrange multipliers
            X: Training data
            y: Labels
            b: Bias term
            K: Kernel matrix
            
        Returns:
            List of KKT violations
            
        Mathematical foundation:
            KKT conditions for soft-margin SVM:
            - α_i = 0 → y_i(w^T x_i + b) ≥ 1 (point correctly classified with margin)
            - α_i = C → y_i(w^T x_i + b) ≤ 1 (point violates margin or is misclassified)
            - 0 < α_i < C → y_i(w^T x_i + b) = 1 (point lies on margin boundary)
        """
        n_samples = len(alphas)
        violations = []
        
        # Compute decision function values
        decision_values = np.zeros(n_samples)
        for i in range(n_samples):
            decision_values[i] = np.sum(alphas * y * K[i, :]) + b
        
        for i in range(n_samples):
            margin = y[i] * decision_values[i]
            
            if alphas[i] == 0:
                if margin < 1 - self.tol:
                    violations.append(f"α_{i}=0 but margin={margin:.3f} < 1")
            elif alphas[i] == self.C:
                if margin > 1 + self.tol:
                    violations.append(f"α_{i}=C but margin={margin:.3f} > 1")
            elif 0 < alphas[i] < self.C:
                if abs(margin - 1) > self.tol:
                    violations.append(f"0<α_{i}<C but margin={margin:.3f} ≠ 1")
        
        return violations


# ============================================================================
# Section 6.15: Sequential Minimal Optimization (SMO) Algorithm
# ============================================================================

    def smo_algorithm(self, X, y):
        """
        Sequential Minimal Optimization (SMO) algorithm.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (-1, 1)
            
        Returns:
            alphas: Optimal Lagrange multipliers
            b: Optimal bias term
            
        Mathematical foundation:
            SMO is an efficient algorithm for solving the SVM dual problem.
            It updates two Lagrange multipliers at a time, which allows for
            analytical solution of the two-variable optimization problem.
        """
        n_samples = X.shape[0]
        
        # Initialize alphas and b
        alphas = np.zeros(n_samples)
        b = 0.0
        
        # Compute kernel matrix
        print(f"Computing kernel matrix for {n_samples} samples...")
        K = self.compute_kernel_matrix(X)
        
        # SMO main loop
        iter_count = 0
        alpha_pairs_changed = 0
        
        print(f"Starting SMO algorithm with C={self.C}...")
        
        while iter_count < self.max_iter:
            alpha_pairs_changed = 0
            
            for i in range(n_samples):
                # Compute error for sample i
                Ei = np.sum(alphas * y * K[i, :]) + b - y[i]
                
                # Check KKT conditions
                ri = y[i] * Ei
                
                if ((ri < -self.tol and alphas[i] < self.C) or 
                    (ri > self.tol and alphas[i] > 0)):
                    
                    # Choose second alpha j
                    j = self.select_second_alpha(i, n_samples)
                    Ej = np.sum(alphas * y * K[j, :]) + b - y[j]
                    
                    # Save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    
                    # Compute bounds for alpha_j
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta (second derivative)
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    alphas[j] = alpha_j_old - y[j] * (Ei - Ej) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    
                    if abs(alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])
                    
                    # Update b
                    b1 = b - Ei - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - Ej - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    
                    alpha_pairs_changed += 1
            
            if alpha_pairs_changed == 0:
                iter_count += 1
            else:
                iter_count = 0
            
            # Print progress
            if iter_count % 100 == 0 and iter_count > 0:
                print(f"Iteration {iter_count}: {alpha_pairs_changed} pairs changed")
        
        # Store results
        self.alphas = alphas
        self.b = b
        
        # Find support vectors
        sv_indices = np.where(alphas > self.tol)[0]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alphas = alphas[sv_indices]
        
        print(f"SMO completed! Found {len(sv_indices)} support vectors")
        
        return alphas, b
    
    def select_second_alpha(self, i, n_samples):
        """
        Select second alpha for SMO algorithm.
        
        Args:
            i: Index of first alpha
            n_samples: Total number of samples
            
        Returns:
            Index of second alpha
            
        Mathematical foundation:
            The second alpha is chosen to maximize the step size,
            which is related to the difference in errors |E_i - E_j|.
        """
        # Simple heuristic: choose random alpha different from i
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
    
    def predict(self, X):
        """
        Make predictions for new data.
        
        Args:
            X: Test data (n_samples, n_features)
            
        Returns:
            Predictions (-1 or 1)
        """
        if self.alphas is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        predictions = []
        for x in X:
            decision_value = 0
            for i in range(len(self.X_train)):
                if self.kernel == 'linear':
                    kernel_val = self.linear_kernel(self.X_train[i], x)
                elif self.kernel == 'rbf':
                    kernel_val = self.rbf_kernel(self.X_train[i], x)
                elif self.kernel == 'poly':
                    kernel_val = self.polynomial_kernel(self.X_train[i], x)
                
                decision_value += self.alphas[i] * self.y_train[i] * kernel_val
            
            decision_value += self.b
            predictions.append(1 if decision_value >= 0 else -1)
        
        return np.array(predictions)
    
    def fit(self, X, y):
        """
        Train the SVM model.
        
        Args:
            X: Training data (n_samples, n_features)
            y: Training labels (-1, 1)
            
        Returns:
            self: Trained model
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Train using SMO
        self.smo_algorithm(X, y)
        
        return self


# ============================================================================
# Section 6.16: Coordinate Ascent and Two-Variable Optimization
# ============================================================================

def coordinate_ascent_example():
    """
    Demonstrate coordinate ascent optimization.
    
    This example shows how coordinate ascent works by optimizing
    one variable at a time while keeping others fixed.
    """
    print("=== Coordinate Ascent Example ===")
    
    def objective_function(x, y):
        """
        Example objective function: f(x, y) = x^2 + y^2 - 2xy
        """
        return x**2 + y**2 - 2*x*y
    
    def gradient_x(x, y):
        """Partial derivative with respect to x"""
        return 2*x - 2*y
    
    def gradient_y(x, y):
        """Partial derivative with respect to y"""
        return 2*y - 2*x
    
    # Initial point
    x, y = 1.0, 1.0
    learning_rate = 0.1
    max_iterations = 50
    
    print(f"Initial point: ({x:.4f}, {y:.4f})")
    print(f"Initial objective: {objective_function(x, y):.6f}")
    
    # Coordinate ascent
    for iteration in range(max_iterations):
        # Update x while keeping y fixed
        x_old = x
        x = x - learning_rate * gradient_x(x, y)
        
        # Update y while keeping x fixed
        y_old = y
        y = y - learning_rate * gradient_y(x, y)
        
        # Check convergence
        if abs(x - x_old) < 1e-6 and abs(y - y_old) < 1e-6:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        if iteration % 10 == 0:
            obj_val = objective_function(x, y)
            print(f"Iteration {iteration}: ({x:.4f}, {y:.4f}), obj = {obj_val:.6f}")
    
    print(f"Final point: ({x:.6f}, {y:.6f})")
    print(f"Final objective: {objective_function(x, y):.6f}")
    print("Note: This demonstrates the coordinate ascent principle used in SMO")
    print()


# ============================================================================
# Section 6.17: Slack Variables and Margin Violations
# ============================================================================

def slack_variables_visualization():
    """
    Visualize slack variables and margin violations.
    
    This example demonstrates how slack variables allow the SVM to
    handle non-separable data by permitting margin violations.
    """
    print("=== Slack Variables Visualization ===")
    
    # Create non-separable data
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1,
                             random_state=42)
    y = 2 * y - 1  # Convert to {-1, 1}
    
    # Add some noise to make data non-separable
    X += np.random.normal(0, 0.3, X.shape)
    
    # Test different C values
    C_values = [0.1, 1.0, 10.0, 100.0]
    
    plt.figure(figsize=(15, 10))
    
    for i, C in enumerate(C_values):
        plt.subplot(2, 2, i+1)
        
        # Train SVM with different C
        svm = SVMRegularization(C=C, kernel='linear')
        svm.fit(X, y)
        
        # Plot data points
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='Class +1', alpha=0.7)
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='blue', label='Class -1', alpha=0.7)
        
        # Plot decision boundary
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                               np.linspace(x2_min, x2_max, 100))
        
        Z = np.zeros(xx1.shape)
        for i_grid in range(xx1.shape[0]):
            for j_grid in range(xx1.shape[1]):
                x = np.array([xx1[i_grid, j_grid], xx2[i_grid, j_grid]])
                Z[i_grid, j_grid] = svm.predict([x])[0]
        
        plt.contour(xx1, xx2, Z, levels=[0], colors='black', linewidth=2)
        plt.contour(xx1, xx2, Z, levels=[-1, 1], colors='gray', linewidth=1, linestyles='--')
        
        # Highlight support vectors
        sv_indices = np.where(svm.alphas > 1e-5)[0]
        plt.scatter(X[sv_indices, 0], X[sv_indices, 1], 
                   s=100, facecolors='none', edgecolors='black', linewidth=2, 
                   label='Support Vectors')
        
        plt.title(f'C = {C}\nSupport Vectors: {len(sv_indices)}')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization shows how C controls the trade-off between")
    print("margin size and classification error (slack variables).")
    print("Higher C = smaller margin, fewer errors")
    print("Lower C = larger margin, more errors")
    print()


# ============================================================================
# Section 6.18: KKT Conditions and Convergence
# ============================================================================

def kkt_conditions_demonstration():
    """
    Demonstrate KKT conditions and convergence checking.
    
    This example shows how to verify that the SVM solution
    satisfies the Karush-Kuhn-Tucker conditions.
    """
    print("=== KKT Conditions Demonstration ===")
    
    # Create synthetic data
    np.random.seed(42)
    X, y = make_classification(n_samples=50, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1,
                             random_state=42)
    y = 2 * y - 1
    
    # Train SVM
    svm = SVMRegularization(C=1.0, kernel='linear')
    svm.fit(X, y)
    
    # Check KKT conditions
    K = svm.compute_kernel_matrix(X)
    violations = svm.kkt_conditions(svm.alphas, X, y, svm.b, K)
    
    print(f"Number of KKT violations: {len(violations)}")
    if violations:
        print("KKT violations:")
        for violation in violations[:5]:  # Show first 5
            print(f"  - {violation}")
    else:
        print("All KKT conditions satisfied!")
    
    # Analyze alpha values
    alpha_zero = np.sum(svm.alphas < 1e-5)
    alpha_c = np.sum(np.abs(svm.alphas - svm.C) < 1e-5)
    alpha_between = np.sum((svm.alphas > 1e-5) & (np.abs(svm.alphas - svm.C) > 1e-5))
    
    print(f"\nAlpha value analysis:")
    print(f"  α_i = 0: {alpha_zero} points (correctly classified with margin)")
    print(f"  α_i = C: {alpha_c} points (violate margin or misclassified)")
    print(f"  0 < α_i < C: {alpha_between} points (on margin boundary)")
    
    return violations


# ============================================================================
# Section 6.19: Margin Calculation and Interpretation
# ============================================================================

def margin_calculation():
    """
    Calculate and interpret margins in soft-margin SVM.
    
    This example demonstrates how to compute functional and geometric
    margins, including the effect of slack variables.
    """
    print("=== Margin Calculation ===")
    
    # Create data
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1,
                             random_state=42)
    y = 2 * y - 1
    
    # Train SVM
    svm = SVMRegularization(C=1.0, kernel='linear')
    svm.fit(X, y)
    
    # Compute margins for all points
    margins = []
    for i in range(len(X)):
        # Functional margin: y_i * (w^T x_i + b)
        decision_value = np.sum(svm.alphas * y * svm.compute_kernel_matrix(X)[i, :]) + svm.b
        functional_margin = y[i] * decision_value
        margins.append(functional_margin)
    
    margins = np.array(margins)
    
    # Analyze margins
    print(f"Margin statistics:")
    print(f"  Minimum margin: {np.min(margins):.4f}")
    print(f"  Maximum margin: {np.max(margins):.4f}")
    print(f"  Mean margin: {np.mean(margins):.4f}")
    print(f"  Points with margin < 1: {np.sum(margins < 1):.0f}")
    print(f"  Points with margin = 1: {np.sum(np.abs(margins - 1) < 1e-5):.0f}")
    print(f"  Points with margin > 1: {np.sum(margins > 1):.0f}")
    
    # Plot margin distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(margins, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(x=1, color='red', linestyle='--', label='Margin = 1')
    plt.xlabel('Functional Margin')
    plt.ylabel('Frequency')
    plt.title('Distribution of Functional Margins')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c=margins[y == 1], 
               cmap='viridis', label='Class +1', alpha=0.7)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], c=margins[y == -1], 
               cmap='viridis', label='Class -1', alpha=0.7)
    plt.colorbar(label='Functional Margin')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Functional Margins by Point')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return margins


# ============================================================================
# Section 6.20: Regularization Trade-off Analysis
# ============================================================================

def regularization_tradeoff_analysis():
    """
    Analyze the trade-off between margin size and classification error.
    
    This example demonstrates how the parameter C controls the balance
    between maximizing the margin and minimizing classification errors.
    """
    print("=== Regularization Trade-off Analysis ===")
    
    # Create non-separable data
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                             n_informative=2, n_clusters_per_class=1,
                             random_state=42)
    y = 2 * y - 1
    
    # Add noise to make data non-separable
    X += np.random.normal(0, 0.4, X.shape)
    
    # Test different C values
    C_values = np.logspace(-2, 3, 20)
    results = []
    
    for C in C_values:
        # Train SVM
        svm = SVMRegularization(C=C, kernel='linear')
        svm.fit(X, y)
        
        # Calculate metrics
        predictions = svm.predict(X)
        accuracy = np.mean(predictions == y)
        
        # Count support vectors
        n_support_vectors = np.sum(svm.alphas > 1e-5)
        
        # Calculate margin (approximate)
        w_norm = np.sqrt(np.sum(svm.alphas[:, None] * svm.alphas[None, :] * 
                               (y[:, None] * y[None, :]) * svm.compute_kernel_matrix(X)))
        margin = 1 / w_norm if w_norm > 0 else 0
        
        results.append({
            'C': C,
            'accuracy': accuracy,
            'n_support_vectors': n_support_vectors,
            'margin': margin
        })
    
    # Plot results
    results = np.array(results)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.semilogx(results['C'], results['accuracy'], 'bo-')
    plt.xlabel('Regularization Parameter C')
    plt.ylabel('Training Accuracy')
    plt.title('Accuracy vs C')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.semilogx(results['C'], results['n_support_vectors'], 'ro-')
    plt.xlabel('Regularization Parameter C')
    plt.ylabel('Number of Support Vectors')
    plt.title('Support Vectors vs C')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.semilogx(results['C'], results['margin'], 'go-')
    plt.xlabel('Regularization Parameter C')
    plt.ylabel('Margin Size')
    plt.title('Margin vs C')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Trade-off analysis shows:")
    print("- Higher C: More support vectors, smaller margin, higher accuracy")
    print("- Lower C: Fewer support vectors, larger margin, lower accuracy")
    print("- Optimal C balances generalization and training performance")
    
    return results


def main():
    """
    Main function to run all SVM regularization examples.
    """
    print("SVM Regularization: Slack Variables and SMO Algorithm")
    print("=" * 60)
    
    # Run all examples
    coordinate_ascent_example()
    slack_variables_visualization()
    kkt_conditions_demonstration()
    margin_calculation()
    regularization_tradeoff_analysis()
    
    print("\nAll SVM regularization examples completed!")


if __name__ == "__main__":
    main() 
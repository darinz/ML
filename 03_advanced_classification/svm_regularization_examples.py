import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cvxopt
from cvxopt import matrix, solvers

class SVMRegularization:
    """
    Implementation of SVM with regularization (slack variables) for non-separable case
    """
    
    def __init__(self, C=1.0, kernel='linear', tol=1e-3):
        """
        Initialize SVM with regularization
        
        Parameters:
        C: Regularization parameter (controls trade-off between margin and slack)
        kernel: Kernel function ('linear', 'rbf', etc.)
        tol: Convergence tolerance
        """
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.alphas = None
        self.b = None
        self.support_vectors = None
        self.support_vector_labels = None
        
    def linear_kernel(self, x1, x2):
        """Linear kernel function"""
        return np.dot(x1, x2)
    
    def rbf_kernel(self, x1, x2, gamma=1.0):
        """RBF kernel function"""
        return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
    
    def compute_kernel_matrix(self, X):
        """Compute kernel matrix K[i,j] = K(x_i, x_j)"""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                if self.kernel == 'linear':
                    K[i, j] = self.linear_kernel(X[i], X[j])
                elif self.kernel == 'rbf':
                    K[i, j] = self.rbf_kernel(X[i], X[j])
        
        return K
    
    def objective_function(self, alphas, K, y):
        """
        Dual objective function: W(α) = Σα_i - (1/2)ΣΣ y_i y_j α_i α_j K(x_i, x_j)
        
        Parameters:
        alphas: Lagrange multipliers
        K: Kernel matrix
        y: Target labels
        """
        n_samples = len(alphas)
        term1 = np.sum(alphas)
        term2 = 0.5 * np.sum(alphas[:, np.newaxis] * alphas * 
                            (y[:, np.newaxis] * y) * K)
        return term1 - term2
    
    def kkt_conditions(self, alphas, X, y, b, K):
        """
        Check KKT conditions for convergence
        
        α_i = 0 → y_i(w^T x_i + b) ≥ 1
        α_i = C → y_i(w^T x_i + b) ≤ 1  
        0 < α_i < C → y_i(w^T x_i + b) = 1
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
    
    def smo_algorithm(self, X, y, max_iter=1000):
        """
        Sequential Minimal Optimization (SMO) algorithm
        
        Parameters:
        X: Training features
        y: Training labels (-1, 1)
        max_iter: Maximum iterations
        """
        n_samples = X.shape[0]
        
        # Initialize alphas
        alphas = np.zeros(n_samples)
        b = 0.0
        
        # Compute kernel matrix
        K = self.compute_kernel_matrix(X)
        
        # SMO main loop
        iter_count = 0
        while iter_count < max_iter:
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
                    
                    # Compute bounds
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
        
        # Store results
        self.alphas = alphas
        self.b = b
        
        # Find support vectors
        sv_indices = np.where(alphas > self.tol)[0]
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alphas = alphas[sv_indices]
        
        return alphas, b
    
    def select_second_alpha(self, i, n_samples):
        """Simple heuristic to select second alpha"""
        j = i
        while j == i:
            j = np.random.randint(0, n_samples)
        return j
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        if self.alphas is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        for x in X:
            decision_value = 0
            for i, sv in enumerate(self.support_vectors):
                if self.kernel == 'linear':
                    kernel_val = self.linear_kernel(x, sv)
                elif self.kernel == 'rbf':
                    kernel_val = self.rbf_kernel(x, sv)
                
                decision_value += (self.support_vector_alphas[i] * 
                                 self.support_vector_labels[i] * kernel_val)
            
            decision_value += self.b
            predictions.append(1 if decision_value > 0 else -1)
        
        return np.array(predictions)
    
    def fit(self, X, y):
        """Fit the SVM model"""
        # Convert labels to -1, 1 if needed
        y = np.array(y)
        if set(y) != {-1, 1}:
            y = np.where(y == 0, -1, 1)
        
        self.smo_algorithm(X, y)
        return self

def coordinate_ascent_example():
    """
    Example of coordinate ascent optimization
    """
    def objective_function(x, y):
        """Simple quadratic function for demonstration"""
        return -(x**2 + y**2 - 2*x*y + 2*x + 2*y)
    
    # Initial point
    x, y = 0.0, 0.0
    max_iter = 50
    
    print("Coordinate Ascent Optimization:")
    print("Iteration | x       | y       | f(x,y)")
    print("-" * 40)
    
    for i in range(max_iter):
        # Optimize x while holding y fixed
        x = 1 - y  # Optimal x for fixed y
        
        # Optimize y while holding x fixed  
        y = 1 - x  # Optimal y for fixed x
        
        f_val = objective_function(x, y)
        print(f"{i+1:9d} | {x:7.4f} | {y:7.4f} | {f_val:7.4f}")
        
        if i > 0 and abs(f_val - prev_f_val) < 1e-6:
            break
        prev_f_val = f_val
    
    print(f"\nOptimal solution: x = {x:.4f}, y = {y:.4f}")
    print(f"Optimal value: f(x,y) = {objective_function(x, y):.4f}")

def slack_variables_visualization():
    """
    Visualize how slack variables work in SVM
    """
    # Generate non-separable data
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.5, random_state=42)
    y = np.where(y == 0, -1, 1)
    
    # Add some outliers
    X[0] = [2, 3]  # Outlier in upper region
    X[1] = [-2, -3]  # Outlier in lower region
    
    # Fit SVM with different C values
    C_values = [0.1, 1.0, 10.0]
    models = []
    
    for C in C_values:
        svm = SVMRegularization(C=C)
        svm.fit(X, y)
        models.append(svm)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (C, model) in enumerate(zip(C_values, models)):
        ax = axes[i]
        
        # Plot data points
        ax.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Class -1', alpha=0.6)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class 1', alpha=0.6)
        
        # Plot support vectors
        if model.support_vectors is not None:
            ax.scatter(model.support_vectors[:, 0], model.support_vectors[:, 1], 
                      s=100, facecolors='none', edgecolors='black', linewidth=2, 
                      label='Support Vectors')
        
        # Create decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contour(xx, yy, Z, levels=[0], colors='green', linewidths=2)
        ax.set_title(f'SVM with C = {C}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def kkt_conditions_demonstration():
    """
    Demonstrate KKT conditions checking
    """
    # Generate sample data
    np.random.seed(42)
    X, y = make_blobs(n_samples=50, centers=2, cluster_std=1.0, random_state=42)
    y = np.where(y == 0, -1, 1)
    
    # Fit SVM
    svm = SVMRegularization(C=1.0)
    alphas, b = svm.smo_algorithm(X, y)
    
    # Check KKT conditions
    K = svm.compute_kernel_matrix(X)
    violations = svm.kkt_conditions(alphas, X, y, b, K)
    
    print("KKT Conditions Check:")
    print(f"Total violations: {len(violations)}")
    if violations:
        print("Violations found:")
        for violation in violations[:5]:  # Show first 5 violations
            print(f"  - {violation}")
    else:
        print("All KKT conditions satisfied!")
    
    # Show alpha distribution
    print(f"\nAlpha distribution:")
    print(f"α = 0: {np.sum(alphas == 0)} points")
    print(f"0 < α < C: {np.sum((alphas > 0) & (alphas < svm.C))} points")
    print(f"α = C: {np.sum(alphas == svm.C)} points")
    print(f"Support vectors: {np.sum(alphas > svm.tol)} points")

def margin_calculation():
    """
    Calculate and visualize margins for different C values
    """
    # Generate data
    np.random.seed(42)
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=42)
    y = np.where(y == 0, -1, 1)
    
    C_values = [0.1, 1.0, 10.0]
    margins = []
    
    for C in C_values:
        svm = SVMRegularization(C=C)
        svm.fit(X, y)
        
        # Calculate margin: 1 / ||w||
        if svm.kernel == 'linear':
            w = np.sum(svm.support_vector_alphas[:, np.newaxis] * 
                      svm.support_vector_labels[:, np.newaxis] * 
                      svm.support_vectors, axis=0)
            margin = 1.0 / np.linalg.norm(w)
        else:
            margin = "N/A (non-linear kernel)"
        
        margins.append(margin)
        print(f"C = {C}: Margin = {margin}")
    
    return margins

def regularization_tradeoff_analysis():
    """
    Analyze the trade-off between margin size and classification error
    """
    # Generate data with different noise levels
    np.random.seed(42)
    noise_levels = [0.5, 1.0, 1.5, 2.0]
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    results = []
    
    for noise in noise_levels:
        X, y = make_blobs(n_samples=100, centers=2, cluster_std=noise, random_state=42)
        y = np.where(y == 0, -1, 1)
        
        noise_results = []
        for C in C_values:
            svm = SVMRegularization(C=C)
            svm.fit(X, y)
            
            # Calculate accuracy
            predictions = svm.predict(X)
            accuracy = accuracy_score(y, predictions)
            
            # Count support vectors
            n_support_vectors = len(svm.support_vectors) if svm.support_vectors is not None else 0
            
            noise_results.append({
                'C': C,
                'accuracy': accuracy,
                'n_support_vectors': n_support_vectors
            })
        
        results.append({
            'noise': noise,
            'results': noise_results
        })
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, result in enumerate(results):
        noise = result['noise']
        C_vals = [r['C'] for r in result['results']]
        accuracies = [r['accuracy'] for r in result['results']]
        n_sv = [r['n_support_vectors'] for r in result['results']]
        
        axes[0].semilogx(C_vals, accuracies, 'o-', label=f'Noise={noise}')
        axes[1].semilogx(C_vals, n_sv, 's-', label=f'Noise={noise}')
    
    axes[0].set_xlabel('Regularization Parameter C')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy vs Regularization Parameter')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel('Regularization Parameter C')
    axes[1].set_ylabel('Number of Support Vectors')
    axes[1].set_title('Support Vectors vs Regularization Parameter')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("SVM Regularization Examples")
    print("=" * 50)
    
    # Run examples
    print("\n1. Coordinate Ascent Example:")
    coordinate_ascent_example()
    
    print("\n2. Slack Variables Visualization:")
    slack_variables_visualization()
    
    print("\n3. KKT Conditions Demonstration:")
    kkt_conditions_demonstration()
    
    print("\n4. Margin Calculation:")
    margin_calculation()
    
    print("\n5. Regularization Trade-off Analysis:")
    regularization_tradeoff_analysis() 
"""
Newton's Method Implementation Examples

This module implements Newton's method for optimization as described in the 
accompanying markdown file. Newton's method is a second-order optimization
algorithm that uses both gradient and Hessian information for faster convergence.

Key Concepts Implemented:
1. Newton's method for root finding (1D)
2. Newton's method for function maximization (1D)
3. Newton's method for logistic regression (multidimensional)
4. Hessian computation and numerical stability
5. Convergence analysis and comparison with gradient methods
6. Practical examples with synthetic data

Mathematical Background:
- Root finding: x_{n+1} = x_n - f(x_n) / f'(x_n)
- Maximization: x_{n+1} = x_n - f'(x_n) / f''(x_n)
- Multidimensional: θ_{n+1} = θ_n - H^{-1} ∇f(θ_n)
- Logistic regression Hessian: H = -X^T D X (where D = diag(h(1-h)))

Advantages of Newton's Method:
- Quadratic convergence rate (much faster than gradient descent)
- Fewer iterations needed for convergence
- Automatic step size selection (no learning rate tuning)

Disadvantages:
- Requires computing and inverting Hessian matrix (expensive)
- Memory intensive for high-dimensional problems
- May fail if Hessian is not positive definite
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def newton_1d(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for finding a root of f(x) = 0 in 1D.
    
    Newton's method for root finding uses the iterative formula:
    x_{n+1} = x_n - f(x_n) / f'(x_n)
    
    This method converges quadratically when the initial guess is close to the root
    and the function is well-behaved.
    
    Args:
        f: function f(x) - the function whose root we want to find
        df: derivative f'(x) - the derivative of f
        x0: initial guess
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    
    Returns:
        x: estimated root
    
    Example:
        >>> f = lambda x: x**2 - 2  # Find sqrt(2)
        >>> df = lambda x: 2*x
        >>> root = newton_1d(f, df, x0=1.0)
        >>> print(f"Root: {root}")
        >>> print(f"f(root): {f(root)}")
    """
    x = x0
    history = {'x': [x], 'f': [f(x)]}
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        # Check if derivative is too small (could cause division by zero)
        if abs(dfx) < 1e-12:
            raise ValueError("Derivative too small - Newton's method may fail")
        
        # Newton update
        x_new = x - fx / dfx
        
        # Store history
        history['x'].append(x_new)
        history['f'].append(f(x_new))
        
        # Check convergence
        if abs(x_new - x) < tol:
            print(f"Converged after {i+1} iterations")
            return x_new, history
        
        x = x_new
    
    print(f"Warning: Did not converge after {max_iter} iterations")
    return x, history

def newton_maximize_1d(l, dl, ddl, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for maximizing l(x) in 1D.
    
    For maximization, we find the root of the derivative: dl(x) = 0
    The update formula becomes: x_{n+1} = x_n - dl(x_n) / ddl(x_n)
    
    This method is particularly effective for concave functions where the
    second derivative is negative (ddl < 0).
    
    Args:
        l: function l(x) - the function to maximize
        dl: first derivative l'(x)
        ddl: second derivative l''(x)
        x0: initial guess
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    
    Returns:
        x: estimated maximizer
    
    Example:
        >>> l = lambda x: -(x-2)**2 + 3  # Maximum at x=2
        >>> dl = lambda x: -2*(x-2)
        >>> ddl = lambda x: -2
        >>> max_x = newton_maximize_1d(l, dl, ddl, x0=0.0)
        >>> print(f"Maximum at: {max_x}")
    """
    x = x0
    history = {'x': [x], 'l': [l(x)], 'dl': [dl(x)]}
    
    for i in range(max_iter):
        grad = dl(x)
        hess = ddl(x)
        
        # Check if second derivative is too small
        if abs(hess) < 1e-12:
            raise ValueError("Second derivative too small - Newton's method may fail")
        
        # Newton update for maximization
        x_new = x - grad / hess
        
        # Store history
        history['x'].append(x_new)
        history['l'].append(l(x_new))
        history['dl'].append(dl(x_new))
        
        # Check convergence
        if abs(x_new - x) < tol:
            print(f"Converged after {i+1} iterations")
            return x_new, history
        
        x = x_new
    
    print(f"Warning: Did not converge after {max_iter} iterations")
    return x, history

def sigmoid(z):
    """
    Compute the sigmoid function.
    
    Args:
        z: Input value or array
    
    Returns:
        Sigmoid output in range (0,1)
    """
    # Numerical stability: clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def log_likelihood(theta, X, y):
    """
    Compute the log-likelihood for logistic regression.
    
    Args:
        theta: Parameter vector
        X: Feature matrix with bias term
        y: Binary labels (0 or 1)
    
    Returns:
        Log-likelihood value
    """
    h = sigmoid(X @ theta)
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    return np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient(theta, X, y):
    """
    Compute the gradient of the log-likelihood.
    
    Args:
        theta: Parameter vector
        X: Feature matrix with bias term
        y: Binary labels (0 or 1)
    
    Returns:
        Gradient vector
    """
    h = sigmoid(X @ theta)
    return X.T @ (y - h)

def hessian(theta, X):
    """
    Compute the Hessian matrix of the log-likelihood.
    
    The Hessian for logistic regression has the form:
    H = -X^T D X
    where D = diag(h(1-h)) is a diagonal matrix with h_i(1-h_i) on the diagonal.
    
    Args:
        theta: Parameter vector
        X: Feature matrix with bias term
    
    Returns:
        Hessian matrix
    """
    h = sigmoid(X @ theta)
    # Create diagonal matrix D = diag(h(1-h))
    D = np.diag(h * (1 - h))
    return -X.T @ D @ X

def newton_logistic_regression(X, y, tol=1e-6, max_iter=20):
    """
    Newton's method for logistic regression.
    
    This function implements Newton's method to find the maximum likelihood
    estimate for logistic regression parameters. The update rule is:
    θ_{n+1} = θ_n - H^{-1} ∇ℓ(θ_n)
    
    Advantages over gradient descent:
    - Quadratic convergence rate
    - No learning rate tuning required
    - Fewer iterations needed
    
    Disadvantages:
    - Requires computing and inverting Hessian (O(n³) per iteration)
    - Memory intensive for large datasets
    - May fail if Hessian is singular
    
    Args:
        X: feature matrix (n_samples, n_features) with bias term
        y: labels (n_samples,)
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    
    Returns:
        theta: estimated parameters
        history: training history
    """
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    history = {'theta_norm': [], 'log_likelihood': [], 'gradient_norm': []}
    
    for i in range(max_iter):
        # Compute current state
        current_ll = log_likelihood(theta, X, y)
        current_grad = gradient(theta, X, y)
        current_grad_norm = np.linalg.norm(current_grad)
        
        # Store history
        history['theta_norm'].append(np.linalg.norm(theta))
        history['log_likelihood'].append(current_ll)
        history['gradient_norm'].append(current_grad_norm)
        
        # Compute Hessian
        H = hessian(theta, X)
        
        try:
            # Solve linear system: H * delta = grad
            delta = np.linalg.solve(H, current_grad)
        except np.linalg.LinAlgError:
            # Add regularization if Hessian is singular
            print(f"Warning: Hessian singular at iteration {i+1}, adding regularization")
            H_reg = H - 1e-4 * np.eye(n_features)
            delta = np.linalg.solve(H_reg, current_grad)
        
        # Newton update
        theta_new = theta - delta
        
        # Check convergence
        if np.linalg.norm(theta_new - theta) < tol:
            print(f"Converged after {i+1} iterations")
            return theta_new, history
        
        theta = theta_new
        
        # Print progress
        if (i + 1) % 5 == 0:
            print(f"Iteration {i+1}, Log-likelihood: {current_ll:.6f}, "
                  f"Gradient norm: {current_grad_norm:.6f}")
    
    print(f"Warning: Did not converge after {max_iter} iterations")
    return theta, history

def compare_newton_vs_gradient(X, y, max_iter_newton=20, max_iter_gradient=1000):
    """
    Compare Newton's method with gradient descent for logistic regression.
    
    This function demonstrates the key differences between the two optimization
    methods in terms of convergence speed and computational cost.
    
    Args:
        X: Feature matrix with bias term
        y: Binary labels
        max_iter_newton: Maximum iterations for Newton's method
        max_iter_gradient: Maximum iterations for gradient descent
    
    Returns:
        Comparison results and training histories
    """
    print("=== Newton's Method vs Gradient Descent Comparison ===\n")
    
    # Newton's method
    print("1. Running Newton's method...")
    theta_newton, history_newton = newton_logistic_regression(X, y, max_iter=max_iter_newton)
    final_ll_newton = history_newton['log_likelihood'][-1]
    print(f"   Final log-likelihood: {final_ll_newton:.6f}")
    print(f"   Final gradient norm: {history_newton['gradient_norm'][-1]:.6f}")
    print(f"   Iterations: {len(history_newton['log_likelihood'])}\n")
    
    # Gradient descent
    print("2. Running gradient descent...")
    from logistic_regression_examples import train_logistic_regression
    theta_grad, history_grad = train_logistic_regression(X, y, alpha=0.1, max_iter=max_iter_gradient)
    final_ll_grad = -history_grad['loss'][-1]  # Convert from negative log-likelihood
    print(f"   Final log-likelihood: {final_ll_grad:.6f}")
    print(f"   Iterations: {len(history_grad['loss'])}\n")
    
    # Compare results
    print("3. Comparison:")
    print(f"   Newton's method log-likelihood: {final_ll_newton:.6f}")
    print(f"   Gradient descent log-likelihood: {final_ll_grad:.6f}")
    print(f"   Newton's method iterations: {len(history_newton['log_likelihood'])}")
    print(f"   Gradient descent iterations: {len(history_grad['loss'])}")
    print(f"   Speedup factor: {len(history_grad['loss']) / len(history_newton['log_likelihood']):.1f}x")
    
    return {
        'newton': {'theta': theta_newton, 'history': history_newton},
        'gradient': {'theta': theta_grad, 'history': history_grad}
    }

def plot_convergence_comparison(history_newton, history_gradient):
    """
    Plot convergence comparison between Newton's method and gradient descent.
    
    Args:
        history_newton: Training history from Newton's method
        history_gradient: Training history from gradient descent
    """
    plt.figure(figsize=(15, 5))
    
    # Plot log-likelihood
    plt.subplot(1, 3, 1)
    plt.plot(history_newton['log_likelihood'], 'b-', label="Newton's Method", linewidth=2)
    plt.plot(-np.array(history_gradient['loss']), 'r-', label='Gradient Descent', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.title('Convergence: Log-Likelihood')
    plt.legend()
    plt.grid(True)
    
    # Plot gradient norm
    plt.subplot(1, 3, 2)
    plt.plot(history_newton['gradient_norm'], 'b-', label="Newton's Method", linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.title('Convergence: Gradient Norm')
    plt.legend()
    plt.grid(True)
    
    # Plot parameter norm
    plt.subplot(1, 3, 3)
    plt.plot(history_newton['theta_norm'], 'b-', label="Newton's Method", linewidth=2)
    plt.plot(history_gradient['theta_norm'], 'r-', label='Gradient Descent', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Norm')
    plt.title('Convergence: Parameter Norm')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def demonstrate_newtons_method():
    """
    Complete demonstration of Newton's method for various optimization problems.
    
    This function shows Newton's method applied to different scenarios:
    1. Root finding in 1D
    2. Function maximization in 1D
    3. Logistic regression optimization
    4. Comparison with gradient descent
    """
    print("=== Newton's Method Demonstration ===\n")
    
    # 1. Root finding demonstration
    print("1. Root finding demonstration...")
    f = lambda x: x**2 - 2  # Find sqrt(2)
    df = lambda x: 2*x
    root, history_root = newton_1d(f, df, x0=1.0)
    print(f"   Found root: {root:.6f}")
    print(f"   f(root): {f(root):.2e}")
    print(f"   True sqrt(2): {np.sqrt(2):.6f}")
    print(f"   Error: {abs(root - np.sqrt(2)):.2e}\n")
    
    # 2. Function maximization demonstration
    print("2. Function maximization demonstration...")
    l = lambda x: -(x-2)**2 + 3  # Maximum at x=2
    dl = lambda x: -2*(x-2)
    ddl = lambda x: -2
    max_x, history_max = newton_maximize_1d(l, dl, ddl, x0=0.0)
    print(f"   Found maximum at: {max_x:.6f}")
    print(f"   l(max_x): {l(max_x):.6f}")
    print(f"   True maximum at: 2.0")
    print(f"   Error: {abs(max_x - 2.0):.2e}\n")
    
    # 3. Logistic regression demonstration
    print("3. Logistic regression with Newton's method...")
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 2)
    X = np.hstack([np.ones((100, 1)), X])  # Add intercept
    true_theta = np.array([0.5, 2.0, -1.0])
    logits = X @ true_theta
    y = (sigmoid(logits) > 0.5).astype(float)
    
    theta_est, history_logistic = newton_logistic_regression(X, y)
    print(f"   Estimated theta: {theta_est}")
    print(f"   True theta: {true_theta}")
    print(f"   Final log-likelihood: {history_logistic['log_likelihood'][-1]:.6f}\n")
    
    # 4. Comparison with gradient descent
    print("4. Comparing with gradient descent...")
    comparison = compare_newton_vs_gradient(X, y)
    plot_convergence_comparison(comparison['newton']['history'], 
                               comparison['gradient']['history'])
    
    print("=== Demonstration Complete ===")

def analyze_hessian_properties():
    """
    Analyze the properties of the Hessian matrix in logistic regression.
    
    This function demonstrates important properties of the Hessian:
    - Positive definiteness (for concave log-likelihood)
    - Condition number and numerical stability
    - Eigenvalue distribution
    """
    print("=== Hessian Analysis ===\n")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(50, 2)
    X = np.hstack([np.ones((50, 1)), X])
    true_theta = np.array([0.5, 1.0, -0.5])
    logits = X @ true_theta
    y = (sigmoid(logits) > 0.5).astype(float)
    
    # Compute Hessian at different points
    theta_zero = np.zeros(3)
    theta_true = true_theta
    theta_random = np.random.randn(3)
    
    H_zero = hessian(theta_zero, X)
    H_true = hessian(theta_true, X)
    H_random = hessian(theta_random, X)
    
    # Analyze eigenvalues
    eigenvals_zero = np.linalg.eigvals(H_zero)
    eigenvals_true = np.linalg.eigvals(H_true)
    eigenvals_random = np.linalg.eigvals(H_random)
    
    print("1. Eigenvalue analysis:")
    print(f"   At θ=0: min={eigenvals_zero.min():.4f}, max={eigenvals_zero.max():.4f}")
    print(f"   At θ=true: min={eigenvals_true.min():.4f}, max={eigenvals_true.max():.4f}")
    print(f"   At θ=random: min={eigenvals_random.min():.4f}, max={eigenvals_random.max():.4f}")
    
    print("\n2. Condition number analysis:")
    print(f"   At θ=0: {np.linalg.cond(H_zero):.2e}")
    print(f"   At θ=true: {np.linalg.cond(H_true):.2e}")
    print(f"   At θ=random: {np.linalg.cond(H_random):.2e}")
    
    print("\n3. Positive definiteness:")
    print(f"   At θ=0: {'Positive definite' if eigenvals_zero.min() > 0 else 'Not positive definite'}")
    print(f"   At θ=true: {'Positive definite' if eigenvals_true.min() > 0 else 'Not positive definite'}")
    print(f"   At θ=random: {'Positive definite' if eigenvals_random.min() > 0 else 'Not positive definite'}")

if __name__ == "__main__":
    # Run the complete demonstration
    demonstrate_newtons_method()
    
    # Run Hessian analysis
    analyze_hessian_properties()
    
    # Additional examples
    print("\n=== Additional Examples ===\n")
    
    # Example 1: Root finding with different functions
    print("Example 1: Root finding with different functions")
    
    # Find cube root of 8
    f1 = lambda x: x**3 - 8
    df1 = lambda x: 3*x**2
    root1, _ = newton_1d(f1, df1, x0=1.0)
    print(f"   Cube root of 8: {root1:.6f} (true: 2.0)")
    
    # Find solution to cos(x) = x
    f2 = lambda x: np.cos(x) - x
    df2 = lambda x: -np.sin(x) - 1
    root2, _ = newton_1d(f2, df2, x0=0.5)
    print(f"   Solution to cos(x) = x: {root2:.6f}")
    print()
    
    # Example 2: Function maximization
    print("Example 2: Function maximization")
    
    # Maximize Gaussian function
    l1 = lambda x: np.exp(-(x-3)**2/2)
    dl1 = lambda x: -(x-3) * np.exp(-(x-3)**2/2)
    ddl1 = lambda x: ((x-3)**2 - 1) * np.exp(-(x-3)**2/2)
    max1, _ = newton_maximize_1d(l1, dl1, ddl1, x0=0.0)
    print(f"   Maximum of Gaussian at x=3: found at {max1:.6f}")
    
    # Maximize log function
    l2 = lambda x: np.log(1 + x**2) if x > -1 else -np.inf
    dl2 = lambda x: 2*x / (1 + x**2) if x > -1 else 0
    ddl2 = lambda x: 2*(1 - x**2) / (1 + x**2)**2 if x > -1 else 0
    max2, _ = newton_maximize_1d(l2, dl2, ddl2, x0=1.0)
    print(f"   Maximum of log(1+x²): found at {max2:.6f}")
    print()
    
    # Example 3: Newton's method convergence visualization
    print("Example 3: Convergence visualization")
    
    # Plot convergence for root finding
    f_plot = lambda x: x**2 - 4
    df_plot = lambda x: 2*x
    _, history_plot = newton_1d(f_plot, df_plot, x0=3.0)
    
    plt.figure(figsize=(12, 4))
    
    # Plot function and iterations
    plt.subplot(1, 2, 1)
    x_range = np.linspace(0, 4, 100)
    plt.plot(x_range, f_plot(x_range), 'b-', label='f(x) = x² - 4')
    plt.plot(history_plot['x'], history_plot['f'], 'ro-', label='Newton iterations')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Newton\'s Method: Root Finding')
    plt.legend()
    plt.grid(True)
    
    # Plot convergence of x values
    plt.subplot(1, 2, 2)
    plt.plot(history_plot['x'], 'ro-')
    plt.axhline(y=2, color='k', linestyle='--', alpha=0.5, label='True root')
    plt.xlabel('Iteration')
    plt.ylabel('x')
    plt.title('Convergence of x values')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("   Root finding converged to:", history_plot['x'][-1])
    print("   True root: 2.0")
    print("   Final error:", abs(history_plot['x'][-1] - 2.0)) 
import numpy as np

# --- Newton's Method in 1D (Root Finding) ---
def newton_1d(f, df, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for finding a root of f(x) = 0 in 1D.
    Args:
        f: function f(x)
        df: derivative f'(x)
        x0: initial guess
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    Returns:
        x: estimated root
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ValueError("Derivative too small.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Example usage for 1D root finding:
if __name__ == "__main__":
    # Find root of f(x) = x^2 - 2 (should be sqrt(2))
    f = lambda x: x**2 - 2
    df = lambda x: 2*x
    root = newton_1d(f, df, x0=1.0)
    print(f"Root of x^2 - 2: {root}")

# --- Newton's Method for Maximizing a Function in 1D ---
def newton_maximize_1d(l, dl, ddl, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for maximizing l(x) in 1D.
    Args:
        l: function l(x)
        dl: first derivative l'(x)
        ddl: second derivative l''(x)
        x0: initial guess
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    Returns:
        x: estimated maximizer
    """
    x = x0
    for i in range(max_iter):
        grad = dl(x)
        hess = ddl(x)
        if abs(hess) < 1e-12:
            raise ValueError("Second derivative too small.")
        x_new = x - grad / hess
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x

# Example usage for 1D maximization:
if __name__ == "__main__":
    # Maximize l(x) = - (x-2)^2 + 3 (maximum at x=2)
    l = lambda x: - (x-2)**2 + 3
    dl = lambda x: -2*(x-2)
    ddl = lambda x: -2
    max_x = newton_maximize_1d(l, dl, ddl, x0=0.0)
    print(f"Maximum of l(x) at: {max_x}")

# --- Newton's Method for Logistic Regression (Multidimensional Case) ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(theta, X, y):
    h = sigmoid(X @ theta)
    return np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient(theta, X, y):
    h = sigmoid(X @ theta)
    return X.T @ (y - h)

def hessian(theta, X):
    h = sigmoid(X @ theta)
    D = np.diag(h * (1 - h))
    return -X.T @ D @ X

def newton_logistic_regression(X, y, tol=1e-6, max_iter=20):
    """
    Newton's method for logistic regression.
    Args:
        X: feature matrix (n_samples, n_features)
        y: labels (n_samples,)
        tol: tolerance for convergence
        max_iter: maximum number of iterations
    Returns:
        theta: estimated parameters
    """
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    for i in range(max_iter):
        grad = gradient(theta, X, y)
        H = hessian(theta, X)
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            # Add regularization if Hessian is singular
            H_reg = H - 1e-4 * np.eye(n_features)
            delta = np.linalg.solve(H_reg, grad)
        theta_new = theta - delta
        if np.linalg.norm(theta_new - theta) < tol:
            return theta_new
        theta = theta_new
    return theta

# Example usage for logistic regression:
if __name__ == "__main__":
    # Generate a simple dataset for demonstration
    np.random.seed(0)
    X = np.random.randn(100, 2)
    X = np.hstack([np.ones((100, 1)), X])  # Add intercept
    true_theta = np.array([0.5, 2.0, -1.0])
    logits = X @ true_theta
    y = (sigmoid(logits) > 0.5).astype(float)
    theta_est = newton_logistic_regression(X, y)
    print(f"Estimated theta: {theta_est}") 
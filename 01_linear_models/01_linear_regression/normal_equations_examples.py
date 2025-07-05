import numpy as np

# --- Matrix Derivative Example ---
def matrix_derivative_example():
    """Compute the gradient of f(A) = 1.5*A11 + 5*A12^2 + A21*A22 for a 2x2 matrix A."""
    def f(A):
        return 1.5 * A[0, 0] + 5 * A[0, 1] ** 2 + A[1, 0] * A[1, 1]

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    grad = np.zeros_like(A)
    grad[0, 0] = 1.5
    grad[0, 1] = 10 * A[0, 1]
    grad[1, 0] = A[1, 1]
    grad[1, 1] = A[1, 0]
    print("Gradient of f at A:\n", grad)

# --- Least Squares Prediction and Residuals ---
def least_squares_prediction():
    """Compute predictions and residuals for a simple linear regression example."""
    X = np.array([[1, 2], [1, 3], [1, 4]])  # Example with intercept
    theta = np.array([[0.5], [1.0]])
    y = np.array([[2.5], [3.5], [4.5]])

    predictions = X @ theta  # Matrix multiplication
    residuals = predictions - y
    print("Predictions:\n", predictions)
    print("Residuals:\n", residuals)
    return X, theta, y

# --- Cost Function ---
def cost_function(X, theta, y):
    """Compute the cost function J(theta) for linear regression."""
    residuals = X @ theta - y
    return 0.5 * np.sum(residuals ** 2)

# --- Closed-form Solution for Theta (Normal Equation) ---
def closed_form_theta(X, y):
    """Compute the closed-form solution for theta using the normal equation."""
    XTX = X.T @ X
    XTy = X.T @ y
    theta_closed_form = np.linalg.inv(XTX) @ XTy
    print("Closed-form theta:\n", theta_closed_form)
    return theta_closed_form

if __name__ == "__main__":
    print("--- Matrix Derivative Example ---")
    matrix_derivative_example()
    print("\n--- Least Squares Example ---")
    X, theta, y = least_squares_prediction()
    print("\n--- Cost Function ---")
    J = cost_function(X, theta, y)
    print("Cost J(theta):", J)
    print("\n--- Closed-form Solution for Theta ---")
    closed_form_theta(X, y) 
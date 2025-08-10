"""
Normal Equations Examples with Comprehensive Annotations

This file implements the normal equations approach for linear regression,
providing analytical solutions as described in the notes. The normal equations
give us the exact solution to the linear regression optimization problem.

Key Concepts Demonstrated:
- Matrix calculus and derivatives
- Design matrix formulation
- Normal equations derivation and solution
- Analytical vs iterative optimization
- Computational complexity and numerical stability
- Practical implementation considerations

Mathematical Foundations:
- Cost function: J(θ) = (1/2)||Xθ - y||²
- Gradient: ∇J(θ) = X^T(Xθ - y)
- Normal equations: X^T X θ = X^T y
- Solution: θ = (X^T X)^(-1) X^T y
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# Matrix Derivative Examples with Detailed Explanations
# ============================================================================

def matrix_derivative_example():
    """
    Compute the gradient of f(A) = 1.5*A11 + 5*A12^2 + A21*A22 for a 2x2 matrix A.
    This demonstrates matrix calculus concepts used in deriving the normal equations.
    
    Key Learning Points:
    - Matrix derivatives follow specific rules
    - Partial derivatives with respect to matrix elements
    - Gradient is a matrix of the same shape
    - These concepts extend to vector derivatives
    
    Mathematical Background:
    For a scalar function f(A) of a matrix A, the gradient ∇f(A) is a matrix
    where [∇f(A)]_ij = ∂f/∂A_ij
    """
    print("=== Matrix Derivative Example ===")
    print("Understanding matrix calculus for gradient computation")
    print()
    
    def f(A):
        """
        Function f(A) = 1.5*A11 + 5*A12^2 + A21*A22
        
        This is a simple scalar function of a 2x2 matrix A.
        We'll compute its gradient with respect to A.
        """
        return 1.5 * A[0, 0] + 5 * A[0, 1] ** 2 + A[1, 0] * A[1, 1]

    # Example matrix
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    
    print(f"Matrix A:\n{A}")
    print(f"Function value f(A) = {f(A):.2f}")
    print()
    
    # Compute gradient manually using partial derivatives
    grad = np.zeros_like(A)
    
    # ∂f/∂A11 = 1.5 (derivative of 1.5*A11)
    grad[0, 0] = 1.5
    
    # ∂f/∂A12 = 10*A12 (derivative of 5*A12^2)
    grad[0, 1] = 10 * A[0, 1]
    
    # ∂f/∂A21 = A22 (derivative of A21*A22 with respect to A21)
    grad[1, 0] = A[1, 1]
    
    # ∂f/∂A22 = A21 (derivative of A21*A22 with respect to A22)
    grad[1, 1] = A[1, 0]
    
    print("Gradient computation:")
    print(f"∂f/∂A11 = 1.5 = {grad[0, 0]}")
    print(f"∂f/∂A12 = 10*A12 = 10*{A[0, 1]} = {grad[0, 1]}")
    print(f"∂f/∂A21 = A22 = {A[1, 1]} = {grad[1, 0]}")
    print(f"∂f/∂A22 = A21 = {A[1, 0]} = {grad[1, 1]}")
    print()
    print(f"Gradient ∇f(A):\n{grad}")
    print()
    
    # Verify with numerical approximation
    print("Numerical verification:")
    epsilon = 1e-6
    grad_numerical = np.zeros_like(A)
    
    for i in range(2):
        for j in range(2):
            # Compute f(A + ε*E_ij) - f(A) / ε
            A_plus = A.copy()
            A_plus[i, j] += epsilon
            grad_numerical[i, j] = (f(A_plus) - f(A)) / epsilon
    
    print(f"Numerical gradient:\n{grad_numerical}")
    print(f"Difference: {np.linalg.norm(grad - grad_numerical):.2e}")
    print(f"Analytical and numerical gradients agree: {np.linalg.norm(grad - grad_numerical) < 1e-5}")
    print()
    
    return A, grad

# ============================================================================
# Design Matrix and Vector Operations with Detailed Explanations
# ============================================================================

def design_matrix_example():
    """
    Demonstrate the design matrix formulation and matrix-vector operations.
    This shows how training data is organized for efficient computation.
    
    Key Learning Points:
    - Design matrix X includes intercept term (first column of 1s)
    - Matrix-vector operations enable efficient computation
    - Vectorized operations are much faster than loops
    - Matrix dimensions must be compatible for operations
    
    Mathematical Formulation:
    X: design matrix (n_samples, n_features) where first column is all 1s
    y: target vector (n_samples,)
    θ: parameter vector (n_features,)
    """
    print("=== Design Matrix Example ===")
    print("Understanding matrix formulation of linear regression")
    print()
    
    # Example training data
    # Features: [intercept, feature1, feature2]
    X = np.array([
        [1, 2, 3],  # Training example 1: [x₀=1, x₁=2, x₂=3]
        [1, 4, 1],  # Training example 2: [x₀=1, x₁=4, x₂=1]
        [1, 0, 5],  # Training example 3: [x₀=1, x₁=0, x₂=5]
        [1, 1, 2]   # Training example 4: [x₀=1, x₁=1, x₂=2]
    ])
    
    # Target values
    y = np.array([5.0, 3.0, 7.0, 4.0])
    
    # Example parameters
    theta = np.array([1.0, 0.5, 1.0])
    
    print(f"Design matrix X (shape: {X.shape}):")
    print(f"  Rows: {X.shape[0]} training examples")
    print(f"  Columns: {X.shape[1]} features (including intercept)")
    print(X)
    print()
    
    print(f"Target vector y (shape: {y.shape}):")
    print(y)
    print()
    
    print(f"Parameter vector θ (shape: {theta.shape}):")
    print(f"  θ₀ (intercept): {theta[0]}")
    print(f"  θ₁ (feature 1 weight): {theta[1]}")
    print(f"  θ₂ (feature 2 weight): {theta[2]}")
    print(theta)
    print()
    
    # Matrix-vector operations
    print("Matrix-vector operations:")
    
    # 1. Predictions: Xθ
    predictions = X @ theta  # Matrix multiplication
    print(f"Predictions Xθ (shape: {predictions.shape}):")
    print(f"  h_θ(x⁽¹⁾) = {predictions[0]:.2f}")
    print(f"  h_θ(x⁽²⁾) = {predictions[1]:.2f}")
    print(f"  h_θ(x⁽³⁾) = {predictions[2]:.2f}")
    print(f"  h_θ(x⁽⁴⁾) = {predictions[3]:.2f}")
    print(predictions)
    print()
    
    # 2. Residuals: Xθ - y
    residuals = predictions - y
    print(f"Residuals Xθ - y (shape: {residuals.shape}):")
    print(f"  ε⁽¹⁾ = {residuals[0]:.2f}")
    print(f"  ε⁽²⁾ = {residuals[1]:.2f}")
    print(f"  ε⁽³⁾ = {residuals[2]:.2f}")
    print(f"  ε⁽⁴⁾ = {residuals[3]:.2f}")
    print(residuals)
    print()
    
    # 3. Cost function: (1/2)||Xθ - y||²
    cost = 0.5 * np.dot(residuals, residuals)
    print(f"Cost function J(θ) = (1/2)||Xθ - y||² = {cost:.4f}")
    print()
    
    # 4. Gradient: X^T(Xθ - y)
    gradient = X.T @ residuals
    print(f"Gradient ∇J(θ) = X^T(Xθ - y) (shape: {gradient.shape}):")
    print(f"  ∂J/∂θ₀ = {gradient[0]:.2f}")
    print(f"  ∂J/∂θ₁ = {gradient[1]:.2f}")
    print(f"  ∂J/∂θ₂ = {gradient[2]:.2f}")
    print(gradient)
    print()
    
    return X, y, theta

def cost_function_verification():
    """
    Verify that the matrix form of the cost function matches the sum form.
    This demonstrates the equivalence: J(θ) = (1/2)||Xθ - y||² = (1/2)Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
    
    Key Learning Points:
    - Matrix form and sum form are mathematically equivalent
    - Matrix form is more efficient for computation
    - Vectorized operations avoid explicit loops
    - Both forms give identical results
    """
    print("=== Cost Function Verification ===")
    print("Verifying equivalence of matrix and sum forms")
    print()
    
    X, y, theta = design_matrix_example()
    
    # Method 1: Matrix form
    def cost_matrix_form(X, theta, y):
        """
        J(θ) = (1/2)||Xθ - y||²
        
        This is the vectorized form using matrix operations.
        Computationally efficient and mathematically elegant.
        """
        residuals = X @ theta - y
        return 0.5 * np.dot(residuals, residuals)
    
    # Method 2: Sum form
    def cost_sum_form(X, theta, y):
        """
        J(θ) = (1/2)Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
        
        This is the explicit sum form that shows the mathematical definition.
        Less efficient but more intuitive.
        """
        total = 0.0
        for i in range(len(y)):
            prediction = np.dot(theta, X[i])
            squared_error = (prediction - y[i]) ** 2
            total += squared_error
        return 0.5 * total
    
    cost_matrix = cost_matrix_form(X, theta, y)
    cost_sum = cost_sum_form(X, theta, y)
    
    print("Cost function comparison:")
    print(f"Matrix form: J(θ) = (1/2)||Xθ - y||² = {cost_matrix:.4f}")
    print(f"Sum form: J(θ) = (1/2)Σ(h_θ(x⁽ⁱ⁾) - y⁽ⁱ⁾)² = {cost_sum:.4f}")
    print(f"Difference: {abs(cost_matrix - cost_sum):.10f}")
    print(f"Forms are equivalent: {abs(cost_matrix - cost_sum) < 1e-10}")
    print()
    
    # Show step-by-step computation for sum form
    print("Step-by-step sum form computation:")
    for i in range(len(y)):
        prediction = np.dot(theta, X[i])
        error = prediction - y[i]
        squared_error = error ** 2
        print(f"  Example {i+1}: h_θ(x⁽ⁱ⁾) = {prediction:.2f}, y⁽ⁱ⁾ = {y[i]:.1f}")
        print(f"    Error = {error:.2f}, Squared error = {squared_error:.4f}")
    
    print(f"  Total squared error: {2 * cost_sum:.4f}")
    print(f"  Cost J(θ) = (1/2) × {2 * cost_sum:.4f} = {cost_sum:.4f}")
    print()

# ============================================================================
# Normal Equations Solution with Detailed Explanations
# ============================================================================

def normal_equations_solution(X, y):
    """
    Solve the normal equations: θ = (X^T X)^(-1) X^T y
    This provides the analytical solution for linear regression.
    
    Key Learning Points:
    - Normal equations give the exact solution
    - No iteration required (unlike gradient descent)
    - Solution minimizes the cost function exactly
    - Requires matrix inversion (computational cost)
    
    Mathematical Derivation:
    1. Cost function: J(θ) = (1/2)||Xθ - y||²
    2. Gradient: ∇J(θ) = X^T(Xθ - y)
    3. Set gradient to zero: X^T(Xθ - y) = 0
    4. Solve: X^T X θ = X^T y
    5. Solution: θ = (X^T X)^(-1) X^T y
    """
    print("=== Normal Equations Solution ===")
    print("Computing the analytical solution to linear regression")
    print()
    
    # Step 1: Compute X^T X and X^T y
    XTX = X.T @ X
    XTy = X.T @ y
    
    print(f"X^T X (shape: {XTX.shape}):")
    print(f"  This is a {XTX.shape[0]}×{XTX.shape[1]} symmetric matrix")
    print(XTX)
    print()
    
    print(f"X^T y (shape: {XTy.shape}):")
    print(f"  This is a {XTy.shape[0]}-dimensional vector")
    print(XTy)
    print()
    
    # Step 2: Check if X^T X is invertible
    det = np.linalg.det(XTX)
    print(f"Determinant of X^T X: {det:.6f}")
    print(f"X^T X is invertible: {abs(det) > 1e-10}")
    
    if abs(det) < 1e-10:
        print("Warning: X^T X is nearly singular. Solution may be unstable.")
    print()
    
    # Step 3: Solve normal equations
    # Method 1: Direct matrix inversion
    theta_analytical = np.linalg.inv(XTX) @ XTy
    
    # Method 2: Using solve (more numerically stable)
    theta_solve = np.linalg.solve(XTX, XTy)
    
    print(f"Analytical solution θ = (X^T X)^(-1) X^T y:")
    print(f"  Using inv(): {theta_analytical}")
    print(f"  Using solve(): {theta_solve}")
    print(f"  Difference: {np.linalg.norm(theta_analytical - theta_solve):.2e}")
    print()
    
    # Step 4: Verify the solution
    # Check that gradient is zero at the solution
    predictions = X @ theta_analytical
    residuals = predictions - y
    gradient_at_solution = X.T @ residuals
    
    print("Verification:")
    print(f"Gradient at solution: ∇J(θ) = {gradient_at_solution}")
    print(f"Gradient norm: ||∇J(θ)|| = {np.linalg.norm(gradient_at_solution):.2e}")
    print(f"Solution is optimal: {np.linalg.norm(gradient_at_solution) < 1e-10}")
    print()
    
    # Step 5: Compute cost at solution
    cost_at_solution = 0.5 * np.dot(residuals, residuals)
    print(f"Cost at solution: J(θ) = {cost_at_solution:.6f}")
    print()
    
    return theta_analytical, XTX, XTy

def compare_analytical_vs_iterative():
    """
    Compare the analytical solution (normal equations) with iterative gradient descent.
    This demonstrates when each approach is preferred.
    
    Key Learning Points:
    - Analytical solution: exact, no iteration, but O(n³) complexity
    - Iterative solution: approximate, requires iteration, but O(n²) per iteration
    - Choice depends on dataset size and computational constraints
    - Normal equations preferred for small datasets
    - Gradient descent preferred for large datasets
    """
    print("=== Analytical vs Iterative Solution ===")
    print("Comparing normal equations with gradient descent")
    print()
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 50
    n_features = 3
    
    # True parameters
    theta_true = np.array([2.0, -1.5, 0.8])
    
    # Generate features and targets
    X = np.random.randn(n_samples, n_features)
    X = np.column_stack([np.ones(n_samples), X])  # Add intercept
    
    # Generate targets with noise
    y = X @ theta_true + 0.5 * np.random.randn(n_samples)
    
    print(f"Generated {n_samples} training examples with {n_features} features")
    print(f"True parameters: θ = {theta_true}")
    print()
    
    # 1. Analytical solution (normal equations)
    print("1. Analytical solution (normal equations):")
    start_time = time.time()
    theta_analytical, _, _ = normal_equations_solution(X, y)
    analytical_time = time.time() - start_time
    
    print(f"   Solution: θ = {theta_analytical}")
    print(f"   Computation time: {analytical_time:.6f} seconds")
    print()
    
    # 2. Iterative solution (gradient descent)
    def gradient_descent(X, y, alpha=0.01, num_iters=1000, tol=1e-6):
        """
        Simple gradient descent implementation
        
        Parameters:
        X: design matrix
        y: target vector
        alpha: learning rate
        num_iters: maximum iterations
        tol: convergence tolerance
        
        Returns:
        theta: optimized parameters
        cost_history: list of cost values
        iterations: number of iterations performed
        """
        n_features = X.shape[1]
        theta = np.zeros(n_features)
        cost_history = []
        
        for i in range(num_iters):
            # Compute predictions and residuals
            predictions = X @ theta
            residuals = predictions - y
            
            # Compute gradient
            gradient = (X.T @ residuals) / len(y)
            
            # Update parameters
            theta = theta - alpha * gradient
            
            # Compute cost
            cost = 0.5 * np.dot(residuals, residuals)
            cost_history.append(cost)
            
            # Check convergence
            if i > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
                break
        
        return theta, cost_history, i + 1
    
    print("2. Iterative solution (gradient descent):")
    start_time = time.time()
    theta_iterative, cost_history, iterations = gradient_descent(X, y)
    iterative_time = time.time() - start_time
    
    print(f"   Solution: θ = {theta_iterative}")
    print(f"   Iterations: {iterations}")
    print(f"   Final cost: {cost_history[-1]:.6f}")
    print(f"   Computation time: {iterative_time:.6f} seconds")
    print()
    
    # Compare results
    print("Comparison:")
    print(f"{'Metric':<20} {'Analytical':<15} {'Iterative':<15}")
    print("-" * 50)
    
    # Parameter accuracy
    analytical_error = np.linalg.norm(theta_analytical - theta_true)
    iterative_error = np.linalg.norm(theta_iterative - theta_true)
    print(f"{'Parameter error':<20} {analytical_error:<15.6f} {iterative_error:<15.6f}")
    
    # Solution difference
    solution_diff = np.linalg.norm(theta_analytical - theta_iterative)
    print(f"{'Solution difference':<20} {0:<15.6f} {solution_diff:<15.6f}")
    
    # Computation time
    print(f"{'Computation time':<20} {analytical_time:<15.6f} {iterative_time:<15.6f}")
    
    # Cost at solution
    analytical_cost = 0.5 * np.dot(X @ theta_analytical - y, X @ theta_analytical - y)
    iterative_cost = cost_history[-1]
    print(f"{'Final cost':<20} {analytical_cost:<15.6f} {iterative_cost:<15.6f}")
    print()
    
    # Visualize convergence
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Cost convergence
    plt.subplot(1, 2, 1)
    plt.plot(cost_history, 'b-', linewidth=2, label='Gradient Descent')
    plt.axhline(y=analytical_cost, color='r', linestyle='--', linewidth=2, label='Analytical Solution')
    plt.xlabel('Iteration')
    plt.ylabel('Cost J(θ)')
    plt.title('Cost Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Parameter convergence
    plt.subplot(1, 2, 2)
    x_pos = np.arange(len(theta_true))
    width = 0.35
    
    plt.bar(x_pos - width/2, theta_analytical, width, label='Analytical', alpha=0.7)
    plt.bar(x_pos + width/2, theta_iterative, width, label='Iterative', alpha=0.7)
    plt.bar(x_pos + 3*width/2, theta_true, width, label='True θ', alpha=0.7, color='black')
    
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Key insights:")
    print("- Analytical solution is exact and fast for small datasets")
    print("- Iterative solution converges to the same result")
    print("- Gradient descent requires tuning (learning rate, iterations)")
    print("- Normal equations have O(n³) complexity due to matrix inversion")
    print("- For large datasets, gradient descent is preferred")
    print()

# ============================================================================
# Practical Applications and Considerations
# ============================================================================

def housing_price_normal_equations():
    """
    Apply normal equations to the housing price prediction problem.
    This demonstrates a real-world application of the analytical solution.
    
    Key Learning Points:
    - Normal equations work well for small to medium datasets
    - Solution is exact and interpretable
    - Can handle multiple features efficiently
    - Provides insights into feature importance
    """
    print("=== Housing Price Prediction with Normal Equations ===")
    print("Real-world application of analytical solution")
    print()
    
    # Housing data with multiple features
    # Features: [intercept, living_area, bedrooms, age]
    X_housing = np.array([
        [1, 2104, 3, 15],  # [intercept, living_area, bedrooms, age]
        [1, 1600, 3, 20],
        [1, 2400, 3, 10],
        [1, 1416, 2, 25],
        [1, 3000, 4, 5],
        [1, 1985, 4, 12],
        [1, 1534, 3, 18],
        [1, 1427, 3, 22],
        [1, 1380, 3, 30],
        [1, 1494, 3, 8]
    ])
    
    # Target prices (in thousands of dollars)
    y_housing = np.array([400, 330, 369, 232, 540, 400, 330, 369, 232, 540])
    
    print(f"Housing dataset: {len(y_housing)} houses")
    print(f"Features: intercept, living area (ft²), bedrooms, age (years)")
    print(f"Target: price (1000$s)")
    print()
    
    # Solve using normal equations
    theta_optimal, XTX, XTy = normal_equations_solution(X_housing, y_housing)
    
    print("Optimal parameters:")
    print(f"  θ₀ (intercept): {theta_optimal[0]:.2f}")
    print(f"  θ₁ (living area): {theta_optimal[1]:.4f}")
    print(f"  θ₂ (bedrooms): {theta_optimal[2]:.2f}")
    print(f"  θ₃ (age): {theta_optimal[3]:.2f}")
    print()
    
    # Make predictions
    predictions = X_housing @ theta_optimal
    
    print("Predictions vs Actual:")
    print("House | Living Area | Bedrooms | Age | Actual | Predicted | Error")
    print("-" * 75)
    for i in range(len(y_housing)):
        actual = y_housing[i]
        predicted = predictions[i]
        error = predicted - actual
        print(f"{i+1:5d} | {X_housing[i,1]:11.0f} | {X_housing[i,2]:8.0f} | {X_housing[i,3]:3.0f} | {actual:6.0f} | {predicted:9.1f} | {error:+6.1f}")
    
    print()
    
    # Compute performance metrics
    mse = np.mean((predictions - y_housing) ** 2)
    mae = np.mean(np.abs(predictions - y_housing))
    r_squared = 1 - np.sum((y_housing - predictions) ** 2) / np.sum((y_housing - np.mean(y_housing)) ** 2)
    
    print("Performance metrics:")
    print(f"  Mean Squared Error: {mse:.2f}")
    print(f"  Mean Absolute Error: {mae:.2f}k")
    print(f"  R-squared: {r_squared:.3f}")
    print()
    
    # Feature importance analysis
    print("Feature importance analysis:")
    feature_names = ['Intercept', 'Living Area', 'Bedrooms', 'Age']
    
    # Standardize features for fair comparison (excluding intercept)
    X_std = X_housing.copy()
    for i in range(1, X_housing.shape[1]):
        X_std[:, i] = (X_housing[:, i] - X_housing[:, i].mean()) / X_housing[:, i].std()
    
    # Solve with standardized features
    theta_std, _, _ = normal_equations_solution(X_std, y_housing)
    
    print("Standardized coefficients (comparable feature importance):")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {theta_std[i]:.2f}")
    print()
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Predicted vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_housing, predictions, alpha=0.7, s=100)
    plt.plot([y_housing.min(), y_housing.max()], [y_housing.min(), y_housing.max()], 'r--', alpha=0.7)
    plt.xlabel('Actual Price (1000$s)')
    plt.ylabel('Predicted Price (1000$s)')
    plt.title('Predicted vs Actual Prices')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    plt.subplot(1, 3, 2)
    residuals = predictions - y_housing
    plt.scatter(predictions, residuals, alpha=0.7, s=100)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Predicted Price (1000$s)')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Feature importance
    plt.subplot(1, 3, 3)
    feature_importance = np.abs(theta_std[1:])  # Exclude intercept
    feature_names_no_intercept = feature_names[1:]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    plt.bar(feature_names_no_intercept, feature_importance, color=colors, alpha=0.7)
    plt.xlabel('Features')
    plt.ylabel('Absolute Standardized Coefficient')
    plt.title('Feature Importance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Make predictions for new houses
    new_houses = np.array([
        [1, 2000, 3, 10],  # 2000 ft², 3 bed, 10 years old
        [1, 1500, 2, 5],   # 1500 ft², 2 bed, 5 years old
        [1, 3000, 4, 0],   # 3000 ft², 4 bed, new construction
    ])
    
    new_predictions = new_houses @ theta_optimal
    
    print("Predictions for new houses:")
    print("House | Living Area | Bedrooms | Age | Predicted Price")
    print("-" * 60)
    for i, (house, pred) in enumerate(zip(new_houses, new_predictions)):
        print(f"{i+1:5d} | {house[1]:11.0f} | {house[2]:8.0f} | {house[3]:3.0f} | ${pred:12.0f}k")
    print()
    
    print("Key insights:")
    print("- Normal equations provide exact solution quickly")
    print("- Multiple features can be handled efficiently")
    print("- Feature importance can be analyzed through coefficients")
    print("- Model performance can be assessed with various metrics")
    print("- Predictions are interpretable and actionable")
    print()

# ============================================================================
# Advanced Topics: Numerical Stability and Computational Complexity
# ============================================================================

def numerical_stability_analysis():
    """
    Analyze numerical stability issues with normal equations.
    This demonstrates when normal equations may fail and how to handle it.
    
    Key Learning Points:
    - Normal equations can be numerically unstable
    - Multicollinearity causes problems
    - Regularization can help with stability
    - Alternative methods exist for ill-conditioned problems
    """
    print("=== Numerical Stability Analysis ===")
    print("Understanding when normal equations may fail")
    print()
    
    # Create a well-conditioned problem
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    # Well-conditioned data
    X_well = np.random.randn(n_samples, n_features)
    X_well = np.column_stack([np.ones(n_samples), X_well])
    theta_true = np.array([1.0, 2.0, -1.0, 0.5])
    y_well = X_well @ theta_true + 0.1 * np.random.randn(n_samples)
    
    # Ill-conditioned data (nearly collinear features)
    X_ill = X_well.copy()
    X_ill[:, 2] = X_ill[:, 1] + 0.01 * np.random.randn(n_samples)  # Nearly collinear
    y_ill = X_ill @ theta_true + 0.1 * np.random.randn(n_samples)
    
    print("Problem setup:")
    print(f"  Well-conditioned: features are independent")
    print(f"  Ill-conditioned: features are nearly collinear")
    print()
    
    # Solve both problems
    print("1. Well-conditioned problem:")
    try:
        theta_well, XTX_well, _ = normal_equations_solution(X_well, y_well)
        cond_well = np.linalg.cond(XTX_well)
        print(f"   Condition number: {cond_well:.2e}")
        print(f"   Solution: θ = {theta_well}")
        print(f"   Error from true: {np.linalg.norm(theta_well - theta_true):.6f}")
    except:
        print("   Failed to solve")
    print()
    
    print("2. Ill-conditioned problem:")
    try:
        theta_ill, XTX_ill, _ = normal_equations_solution(X_ill, y_ill)
        cond_ill = np.linalg.cond(XTX_ill)
        print(f"   Condition number: {cond_ill:.2e}")
        print(f"   Solution: θ = {theta_ill}")
        print(f"   Error from true: {np.linalg.norm(theta_ill - theta_true):.6f}")
    except:
        print("   Failed to solve")
    print()
    
    # Compare condition numbers
    print("Condition number comparison:")
    print(f"  Well-conditioned: {cond_well:.2e}")
    print(f"  Ill-conditioned: {cond_ill:.2e}")
    print(f"  Ratio: {cond_ill/cond_well:.2e}")
    print()
    
    # Regularized solution (Ridge regression)
    print("3. Regularized solution (Ridge regression):")
    lambda_reg = 0.1
    XTX_reg = XTX_ill + lambda_reg * np.eye(XTX_ill.shape[0])
    theta_reg = np.linalg.solve(XTX_reg, X_ill.T @ y_ill)
    
    print(f"   Regularization parameter: λ = {lambda_reg}")
    print(f"   Regularized solution: θ = {theta_reg}")
    print(f"   Error from true: {np.linalg.norm(theta_reg - theta_true):.6f}")
    print()
    
    # Visualize the problem
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Feature correlation
    plt.subplot(1, 2, 1)
    correlation_well = np.corrcoef(X_well[:, 1:].T)
    correlation_ill = np.corrcoef(X_ill[:, 1:].T)
    
    plt.subplot(1, 2, 1)
    plt.imshow(correlation_well, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Feature Correlation (Well-conditioned)')
    plt.xticks(range(3), ['Feature 1', 'Feature 2', 'Feature 3'])
    plt.yticks(range(3), ['Feature 1', 'Feature 2', 'Feature 3'])
    
    plt.subplot(1, 2, 2)
    plt.imshow(correlation_ill, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Feature Correlation (Ill-conditioned)')
    plt.xticks(range(3), ['Feature 1', 'Feature 2', 'Feature 3'])
    plt.yticks(range(3), ['Feature 1', 'Feature 2', 'Feature 3'])
    
    plt.tight_layout()
    plt.show()
    
    print("Key insights:")
    print("- Condition number measures numerical stability")
    print("- High condition number indicates instability")
    print("- Multicollinearity causes high condition numbers")
    print("- Regularization can improve stability")
    print("- Alternative methods (SVD, QR) may be preferred")
    print()

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("Normal Equations Examples with Comprehensive Annotations")
    print("=" * 60)
    print("This file demonstrates the analytical solution to linear regression")
    print("using normal equations with detailed explanations and comparisons.")
    print()
    
    # Run demonstrations
    matrix_derivative_example()
    design_matrix_example()
    cost_function_verification()
    compare_analytical_vs_iterative()
    housing_price_normal_equations()
    numerical_stability_analysis()
    
    print("All examples completed!")
    print("\nKey concepts demonstrated:")
    print("1. Matrix calculus and gradient computation")
    print("2. Design matrix formulation and operations")
    print("3. Normal equations derivation and solution")
    print("4. Analytical vs iterative optimization comparison")
    print("5. Real-world applications and interpretation")
    print("6. Numerical stability and computational considerations")
    print("\nNext steps:")
    print("- Explore regularization techniques (Ridge, Lasso)")
    print("- Implement alternative solution methods (SVD, QR)")
    print("- Apply to other linear models (polynomial regression)")
    print("- Study computational complexity and scaling")
    print("- Investigate feature selection and model selection") 
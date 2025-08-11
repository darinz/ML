# Normal Equations: The Analytical Solution to Linear Regression

## From Iterative to Analytical Optimization: The Power of Direct Solutions

In the previous section, we explored gradient descent and the LMS algorithm, which provide iterative methods to find the optimal parameters $\theta$ that minimize our cost function. These methods work by taking small steps in the direction of steepest descent, gradually approaching the minimum.

However, for linear regression, there's a more direct approach available. Since our cost function $J(\theta) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2$ is a quadratic function of $\theta$, we can find its minimum analytically by setting the derivatives to zero and solving the resulting equations.

This analytical approach, known as the **normal equations**, gives us the exact solution in one step, without the need for iteration or learning rate tuning. Understanding both the iterative (gradient descent) and analytical (normal equations) approaches gives us a complete toolkit for solving linear regression problems.

**Real-World Analogy: The GPS vs. Map Problem**
Think of normal equations like using GPS navigation vs. following a map:
- **Gradient Descent**: Like following turn-by-turn directions (iterative, step-by-step)
- **Normal Equations**: Like having the complete map and calculating the optimal route directly (analytical, one-step)
- **GPS Navigation**: Takes time but adapts to traffic and road conditions
- **Direct Route**: Instant calculation but assumes perfect conditions
- **Choice**: Depends on the situation and requirements

**Visual Analogy: The Recipe Problem**
Think of normal equations like having a complete recipe vs. cooking by taste:
- **Gradient Descent**: Like adjusting seasoning while cooking (iterative refinement)
- **Normal Equations**: Like following a precise recipe (exact measurements)
- **Cooking by Taste**: Takes time but adapts to ingredients and preferences
- **Recipe Method**: Instant but assumes exact ingredients and conditions
- **Result**: Both can produce excellent results, but in different ways

**Mathematical Intuition: The Puzzle Problem**
Think of normal equations like solving a puzzle:
- **Gradient Descent**: Like trying pieces one by one until they fit (iterative)
- **Normal Equations**: Like having the solution manual (analytical)
- **Puzzle Solving**: Takes time but builds intuition
- **Solution Manual**: Instant but doesn't teach the process
- **Learning**: Both approaches have their value

## The Normal Equations: The Mathematical Shortcut

Gradient descent gives one way of minimizing $J$. However, gradient descent requires iterative updates and careful tuning of the learning rate. In contrast, the normal equations approach allows us to directly solve for the optimal parameters in one step, provided the problem is well-posed and the necessary matrix inverses exist. This is especially useful for linear regression problems where the cost function is quadratic and differentiable. In this method, we will minimize $J$ by explicitly taking its derivatives with respect to the $\theta_j$'s, and setting them to zero. To enable us to do this without having to write reams of algebra and pages full of matrices of derivatives, let's introduce some notation for doing calculus with matrices.

**Real-World Analogy: The Calculator vs. Abacus Problem**
Think of normal equations like using a calculator vs. an abacus:
- **Gradient Descent**: Like using an abacus - step-by-step calculations
- **Normal Equations**: Like using a calculator - direct computation
- **Abacus**: Teaches the process but is slow
- **Calculator**: Fast but hides the mechanics
- **Understanding**: Both tools have their place in learning

**Visual Analogy: The Elevator vs. Stairs Problem**
Think of normal equations like taking an elevator vs. climbing stairs:
- **Gradient Descent**: Like climbing stairs - step by step to the top
- **Normal Equations**: Like taking an elevator - direct route to the destination
- **Stairs**: Takes time but you see each step
- **Elevator**: Fast but you miss the journey
- **Choice**: Depends on your goals and constraints

## Why Normal Equations? - Understanding the Trade-offs

Before diving into the mathematics, let's understand why we might prefer normal equations over gradient descent:

**Real-World Analogy: The Restaurant Problem**
Think of normal equations vs. gradient descent like different restaurant experiences:
- **Normal Equations**: Like a fine dining restaurant - precise, exact, no surprises
- **Gradient Descent**: Like a buffet - you can keep trying until you're satisfied
- **Fine Dining**: Expensive but guaranteed quality
- **Buffet**: More affordable but quality varies
- **Choice**: Depends on your budget and preferences

**Visual Analogy: The Photography Problem**
Think of normal equations vs. gradient descent like different photography approaches:
- **Normal Equations**: Like using auto-focus - instant, precise, no adjustment needed
- **Gradient Descent**: Like manual focus - takes time but gives you control
- **Auto-focus**: Fast but less control
- **Manual Focus**: Slower but more precise
- **Result**: Both can produce excellent photos

**Advantages of Normal Equations:**
1. **Exact solution**: No need for iterative optimization
2. **No hyperparameters**: No learning rate to tune
3. **Guaranteed convergence**: Always finds the global minimum (if it exists)
4. **Theoretical insight**: Provides understanding of the optimal solution structure

**Disadvantages of Normal Equations:**
1. **Computational cost**: $O(n^3)$ for matrix inversion vs $O(n^2)$ per iteration for gradient descent
2. **Memory usage**: Requires storing $X^T X$ matrix
3. **Numerical instability**: Matrix inversion can be numerically unstable
4. **Non-invertible matrices**: Fails when $X^T X$ is singular

**When to use each method:**
- **Normal equations**: Small to medium datasets (< 10,000 examples), when you need exact solution
- **Gradient descent**: Large datasets, when approximate solution is acceptable, or when $X^T X$ is singular

**Practical Example - Normal Equations vs. Gradient Descent:**
```python
import numpy as np
import matplotlib.pyplot as plt
import time

def demonstrate_normal_equations_vs_gradient_descent():
    """Demonstrate the trade-offs between normal equations and gradient descent"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create synthetic data
    X = np.random.randn(n_samples, n_features)
    true_theta = np.random.randn(n_features)
    y = X @ true_theta + np.random.normal(0, 0.1, n_samples)
    
    # Add bias term
    X_with_bias = np.column_stack([np.ones(n_samples), X])
    
    print("Normal Equations vs. Gradient Descent Comparison")
    print("=" * 60)
    print(f"Dataset: {n_samples} samples, {n_features} features")
    print()
    
    # Method 1: Normal Equations
    print("Method 1: Normal Equations (Analytical)")
    print("-" * 40)
    
    start_time = time.time()
    # Normal equations: θ = (X^T X)^(-1) X^T y
    XtX = X_with_bias.T @ X_with_bias
    Xty = X_with_bias.T @ y
    theta_normal = np.linalg.solve(XtX, Xty)  # More stable than inv()
    normal_time = time.time() - start_time
    
    print(f"Time: {normal_time:.4f} seconds")
    print(f"Memory: {XtX.nbytes / 1024**2:.2f} MB (for X^T X)")
    print(f"Parameters: {len(theta_normal)}")
    print(f"Matrix size: {XtX.shape}")
    print()
    
    # Method 2: Gradient Descent
    print("Method 2: Gradient Descent (Iterative)")
    print("-" * 40)
    
    def gradient_descent(X, y, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        n_samples, n_features = X.shape
        theta = np.zeros(n_features)
        costs = []
        
        start_time = time.time()
        
        for i in range(max_iterations):
            # Compute predictions
            predictions = X @ theta
            
            # Compute gradient
            gradient = X.T @ (predictions - y) / n_samples
            
            # Update parameters
            theta -= learning_rate * gradient
            
            # Compute cost
            cost = np.mean((predictions - y)**2) / 2
            costs.append(cost)
            
            # Check convergence
            if i > 0 and abs(costs[-1] - costs[-2]) < tolerance:
                break
        
        gradient_time = time.time() - start_time
        return theta, costs, gradient_time
    
    theta_gradient, costs, gradient_time = gradient_descent(X_with_bias, y)
    
    print(f"Time: {gradient_time:.4f} seconds")
    print(f"Memory: {X_with_bias.nbytes / 1024**2:.2f} MB")
    print(f"Iterations: {len(costs)}")
    print(f"Final Cost: {costs[-1]:.6f}")
    print(f"Learning Rate: 0.01")
    print()
    
    # Compare results
    print("Comparison:")
    print("-" * 40)
    print(f"Normal Equations Time: {normal_time:.4f}s")
    print(f"Gradient Descent Time: {gradient_time:.4f}s")
    print(f"Speedup: {gradient_time/normal_time:.1f}x")
    print()
    
    # Check accuracy
    predictions_normal = X_with_bias @ theta_normal
    predictions_gradient = X_with_bias @ theta_gradient
    
    mse_normal = np.mean((predictions_normal - y)**2)
    mse_gradient = np.mean((predictions_gradient - y)**2)
    
    print(f"Normal Equations MSE: {mse_normal:.6f}")
    print(f"Gradient Descent MSE: {mse_gradient:.6f}")
    print(f"Difference: {abs(mse_normal - mse_gradient):.2e}")
    print()
    
    # Check parameter differences
    param_diff = np.linalg.norm(theta_normal - theta_gradient)
    print(f"Parameter Difference: {param_diff:.6f}")
    print(f"Solutions are {'very similar' if param_diff < 1e-3 else 'different'}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Cost convergence for gradient descent
    plt.subplot(1, 3, 1)
    plt.plot(costs, 'b-', linewidth=2)
    plt.axhline(y=mse_normal, color='r', linestyle='--', label='Normal Equations Cost')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Gradient Descent Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Time comparison
    plt.subplot(1, 3, 2)
    methods = ['Normal Equations', 'Gradient Descent']
    times = [normal_time, gradient_time]
    colors = ['red', 'blue']
    
    bars = plt.bar(methods, times, color=colors, alpha=0.7)
    plt.ylabel('Time (seconds)')
    plt.title('Computation Time Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time_val:.4f}s', ha='center', va='bottom')
    
    # Memory comparison
    plt.subplot(1, 3, 3)
    memory_normal = XtX.nbytes / 1024**2
    memory_gradient = X_with_bias.nbytes / 1024**2
    memories = [memory_normal, memory_gradient]
    
    bars = plt.bar(methods, memories, color=colors, alpha=0.7)
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, memory_val in zip(bars, memories):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{memory_val:.1f}MB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Normal equations give exact solution in one step")
    print("2. Gradient descent requires iteration but uses less memory")
    print("3. Both methods give very similar results")
    print("4. Choice depends on dataset size and computational constraints")
    print("5. Normal equations fail when X^T X is not invertible")
    
    return theta_normal, theta_gradient, costs, normal_time, gradient_time

comparison_demo = demonstrate_normal_equations_vs_gradient_descent()
```

## Matrix Derivatives: The Language of Analytical Optimization

When working with functions that take matrices as inputs, we generalize the concept of derivatives to matrices. For a function $f : \mathbb{R}^{n \times d} \mapsto \mathbb{R}$ mapping from $n$-by-$d$ matrices to the real numbers, we define the derivative of $f$ with respect to $A$ to be:

$$
\nabla_A f(A) = \begin{bmatrix}
\frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1d}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f}{\partial A_{n1}} & \cdots & \frac{\partial f}{\partial A_{nd}}
\end{bmatrix}
$$

**Real-World Analogy: The Weather Map Problem**
Think of matrix derivatives like a weather map showing temperature gradients:
- **Weather Map**: Shows temperature at each location (matrix of values)
- **Temperature Gradient**: Shows how temperature changes in each direction (matrix of derivatives)
- **Direction**: Gradient points toward higher temperatures
- **Magnitude**: How fast temperature changes
- **Navigation**: To get warmer, follow the gradient direction

**Visual Analogy: The Elevation Map Problem**
Think of matrix derivatives like an elevation map:
- **Elevation Map**: Shows height at each point (matrix of values)
- **Slope Map**: Shows how steep the terrain is in each direction (matrix of derivatives)
- **Steepest Ascent**: Gradient points uphill
- **Steepest Descent**: Negative gradient points downhill
- **Navigation**: To reach the peak, follow the gradient

### Understanding Matrix Derivatives: The Mathematical Foundation

The gradient of a scalar-valued function with respect to a matrix is itself a matrix, where each entry is the partial derivative of the function with respect to the corresponding entry in the input matrix. This allows us to perform calculus operations in a compact, vectorized form, which is essential for efficient computation in machine learning.

**Real-World Analogy: The Recipe Sensitivity Problem**
Think of matrix derivatives like understanding how recipe changes affect taste:
- **Recipe Matrix**: Amount of each ingredient (matrix of values)
- **Taste Sensitivity**: How taste changes with each ingredient (matrix of derivatives)
- **Optimal Recipe**: Find ingredient amounts that maximize taste
- **Gradient**: Points toward better tasting recipes
- **Optimization**: Adjust ingredients in gradient direction

**Key properties:**

1. **Linearity**: $\nabla_A (f(A) + g(A)) = \nabla_A f(A) + \nabla_A g(A)$

2. **Chain rule**: $\nabla_A f(g(A)) = \nabla_{g(A)} f(g(A)) \cdot \nabla_A g(A)$

3. **Transpose rule**: $\nabla_A f(A^T) = (\nabla_{A^T} f(A^T))^T$

**Visual Analogy: The Assembly Line Problem**
Think of matrix derivatives like an assembly line:
- **Input Matrix**: Raw materials for each product (matrix of values)
- **Output Sensitivity**: How output quality changes with each input (matrix of derivatives)
- **Quality Optimization**: Find input amounts that maximize quality
- **Gradient**: Points toward better quality
- **Adjustment**: Change inputs in gradient direction

### Example: Matrix Derivative Computation

For example, suppose 

$$
A = 
\begin{bmatrix} 
A_{11} & A_{12} \\ 
A_{21} & A_{22} 
\end{bmatrix}
$$

is a 2-by-2 matrix, and the function $f : \mathbb{R}^{2 \times 2} \mapsto \mathbb{R}$ is given by

$$
f(A) = \frac{3}{2}A_{11} + 5A_{12}^2 + A_{21}A_{22}.
$$

Here, $A_{ij}$ denotes the $(i, j)$ entry of the matrix $A$. We then have

$$
\nabla_A f(A) = \begin{bmatrix}
\frac{3}{2} & 10A_{12} \\
A_{22} & A_{21}
\end{bmatrix}.
$$

**Real-World Analogy: The Factory Production Problem**
Think of this example like a factory production system:
- **Input Matrix A**: Amount of each raw material for each product
- **Production Function f(A)**: Total production output
- **Sensitivity Matrix**: How production changes with each input
- **Optimization**: Find input amounts that maximize production
- **Gradient**: Points toward higher production

**Step-by-step computation:**
1. $\frac{\partial f}{\partial A_{11}} = \frac{3}{2}$ (derivative of $\frac{3}{2}A_{11}$)
2. $\frac{\partial f}{\partial A_{12}} = 10A_{12}$ (derivative of $5A_{12}^2$)
3. $\frac{\partial f}{\partial A_{21}} = A_{22}$ (derivative of $A_{21}A_{22}$ with respect to $A_{21}$)
4. $\frac{\partial f}{\partial A_{22}} = A_{21}$ (derivative of $A_{21}A_{22}$ with respect to $A_{22}$)

**Practical Example - Matrix Derivatives:**
```python
def demonstrate_matrix_derivatives():
    """Demonstrate matrix derivative computation"""
    
    # Define a simple matrix function
    def matrix_function(A):
        """f(A) = 3/2 * A[0,0] + 5 * A[0,1]^2 + A[1,0] * A[1,1]"""
        return 1.5 * A[0, 0] + 5 * A[0, 1]**2 + A[1, 0] * A[1, 1]
    
    def matrix_derivative(A):
        """Compute the derivative of f(A) with respect to A"""
        return np.array([
            [1.5, 10 * A[0, 1]],
            [A[1, 1], A[1, 0]]
        ])
    
    # Test with a specific matrix
    A = np.array([[2, 3], [4, 5]])
    
    print("Matrix Derivatives Example")
    print("=" * 40)
    print("Matrix A:")
    print(A)
    print()
    
    print("Function value f(A):")
    f_value = matrix_function(A)
    print(f"f(A) = {f_value}")
    print()
    
    print("Derivative matrix ∇f(A):")
    grad_A = matrix_derivative(A)
    print(grad_A)
    print()
    
    # Verify with finite differences
    print("Verification with Finite Differences:")
    epsilon = 1e-7
    numerical_grad = np.zeros_like(A)
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            # Perturb A[i,j]
            A_plus = A.copy()
            A_plus[i, j] += epsilon
            A_minus = A.copy()
            A_minus[i, j] -= epsilon
            
            # Compute finite difference
            f_plus = matrix_function(A_plus)
            f_minus = matrix_function(A_minus)
            numerical_grad[i, j] = (f_plus - f_minus) / (2 * epsilon)
    
    print("Numerical gradient:")
    print(numerical_grad)
    print()
    
    print("Analytical vs Numerical:")
    print("Analytical:")
    print(grad_A)
    print("Numerical:")
    print(numerical_grad)
    print("Difference:")
    print(np.abs(grad_A - numerical_grad))
    print(f"Max difference: {np.max(np.abs(grad_A - numerical_grad)):.2e}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original matrix
    plt.subplot(1, 3, 1)
    plt.imshow(A, cmap='viridis', aspect='auto')
    plt.colorbar(label='Value')
    plt.title('Matrix A')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Add text annotations
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            plt.text(j, i, f'{A[i, j]}', ha='center', va='center', 
                    color='white', fontweight='bold')
    
    # Function value
    plt.subplot(1, 3, 2)
    plt.text(0.5, 0.5, f'f(A) = {f_value:.2f}', ha='center', va='center', 
             fontsize=14, transform=plt.gca().transAxes)
    plt.title('Function Value')
    plt.axis('off')
    
    # Gradient matrix
    plt.subplot(1, 3, 3)
    plt.imshow(grad_A, cmap='RdBu', aspect='auto')
    plt.colorbar(label='Derivative')
    plt.title('Gradient Matrix ∇f(A)')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Add text annotations
    for i in range(grad_A.shape[0]):
        for j in range(grad_A.shape[1]):
            plt.text(j, i, f'{grad_A[i, j]:.2f}', ha='center', va='center', 
                    color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return A, f_value, grad_A, numerical_grad

matrix_deriv_demo = demonstrate_matrix_derivatives()
```

In this example, we see how to compute the gradient of a function with respect to a matrix. Each entry in the resulting gradient matrix is obtained by differentiating the function with respect to the corresponding entry in $A$. This process is analogous to taking partial derivatives with respect to each variable in multivariable calculus, but extended to matrices.

### Important Matrix Calculus Rules

For our derivation of the normal equations, we'll need these key rules:

1. **Linear term**: $\nabla_x (a^T x) = a$
2. **Quadratic term**: $\nabla_x (x^T A x) = 2A x$ (for symmetric matrix $A$)
3. **Transpose property**: $(A^T)^T = A$
4. **Matrix multiplication**: $(AB)^T = B^T A^T$

**Real-World Analogy: The Toolbox Problem**
Think of matrix calculus rules like tools in a toolbox:
- **Linear Rule**: Like a screwdriver - simple, direct application
- **Quadratic Rule**: Like a wrench - handles more complex connections
- **Transpose Rule**: Like a mirror - reflects the structure
- **Multiplication Rule**: Like a chain - connects multiple operations
- **Toolbox**: Complete set of tools for any matrix calculation

These rules will allow us to efficiently compute the gradient of our cost function.

**Key Insights from Matrix Derivatives:**
1. **Matrix derivatives are matrices**: Each entry is a partial derivative
2. **Rules generalize calculus**: Familiar rules work with matrices
3. **Vectorized computation**: Enables efficient optimization
4. **Foundation for optimization**: Essential for analytical solutions
5. **Numerical verification**: Can check analytical results with finite differences

## 1.2.2 Least squares revisited

Armed with the tools of matrix derivatives, let us now proceed to find in closed-form the value of $\theta$ that minimizes $J(\theta)$. We begin by re-writing $J$ in matrix-vectorial notation. The design matrix $X$ is a convenient way to represent all the input features of your training data in a single matrix. Each row corresponds to one training example, and each column corresponds to a feature (including the intercept term if present). This matrix formulation allows us to express the entire dataset and the linear model compactly, making it easier to apply linear algebra techniques.

### The Design Matrix

Given a training set, define the **design matrix** $X$ to be the `n-by-d` matrix (actually `n-by-(d+1)`, if we include the intercept term) that contains the training examples' input values in its rows:

$$
X = \begin{bmatrix}
--- (x^{(1)})^T --- \\
--- (x^{(2)})^T --- \\
\vdots \\
--- (x^{(n)})^T ---
\end{bmatrix}.
$$

**Understanding the design matrix:**
- **Rows**: Each row represents one training example
- **Columns**: Each column represents one feature
- **First column**: Usually all ones (intercept term)
- **Dimensions**: $n \times (d+1)$ where $n$ is number of examples, $d$ is number of features

**Example**: For our housing dataset with living area and bedrooms:
$$X = \begin{bmatrix}
1 & 2104 & 3 \\
1 & 1600 & 3 \\
1 & 2400 & 3 \\
\vdots & \vdots & \vdots
\end{bmatrix}$$

### The Target Vector

Also, let $\vec{y}$ be the $n$-dimensional vector containing all the target values from the training set:

$$
\vec{y} = \begin{bmatrix}
y^{(1)} \\
y^{(2)} \\
\vdots \\
y^{(n)}
\end{bmatrix}.
$$

The vector $\vec{y}$ stacks all the target values (labels) from the training set into a single column vector. This matches the structure of $X$, so that we can perform matrix operations between $X$ and $\vec{y}$.

**Example**: For our housing dataset:
$$\vec{y} = \begin{bmatrix}
400 \\
330 \\
369 \\
\vdots
\end{bmatrix}$$

### Vectorized Predictions and Errors

Now, since $h_\theta(x^{(i)}) = (x^{(i)})^T \theta$, we can easily verify that

$$
X\theta - \vec{y} = \begin{bmatrix}
(x^{(1)})^T \theta \\
\vdots \\
(x^{(n)})^T \theta
\end{bmatrix} - \begin{bmatrix}
y^{(1)} \\
\vdots \\
y^{(n)}
\end{bmatrix} = \begin{bmatrix}
h_\theta(x^{(1)}) - y^{(1)} \\
\vdots \\
h_\theta(x^{(n)}) - y^{(n)}
\end{bmatrix}.
$$

**Understanding this expression:**
- $X\theta$ computes all predictions at once using matrix multiplication
- $X\theta - \vec{y}$ gives the vector of prediction errors (residuals)
- This is much more efficient than computing each prediction individually

The expression $X\theta$ computes the predicted values for all training examples at once, using matrix multiplication. Subtracting $\vec{y}$ gives the vector of residuals (errors) for each example. This vectorized form is much more efficient than computing each prediction and error individually.

### Vectorized Cost Function

Thus, using the fact that for a vector $z$, we have that $z^T z = \sum_i z_i^2$:

$$
\frac{1}{2}(X\theta - \vec{y})^T (X\theta - \vec{y}) = \frac{1}{2} \sum_{i=1}^n (h_\theta(x^{(i)}) - y^{(i)})^2 = J(\theta)
$$

**Understanding the vectorized form:**
- $(X\theta - \vec{y})^T (X\theta - \vec{y})$ computes the dot product of the error vector with itself
- This is equivalent to summing the squared errors: $\sum_i (h_\theta(x^{(i)}) - y^{(i)})^2$
- The factor $\frac{1}{2}$ is included for mathematical convenience

The cost function $J(\theta)$ for linear regression is the mean squared error (up to a factor of $1/2$ for convenience in differentiation). The matrix form $\frac{1}{2}(X\theta - \vec{y})^T (X\theta - \vec{y})$ is equivalent to summing the squared errors for all training examples, but is more compact and enables efficient computation and differentiation.

### Computing the Gradient

Finally, to minimize $J$, let's find its derivatives with respect to $\theta$. Hence,

$$
\begin{align*}
\nabla_\theta J(\theta)
    &= \nabla_\theta \frac{1}{2}(X\theta - \vec{y})^T (X\theta - \vec{y}) \\
    &= \frac{1}{2} \nabla_\theta \left( (X\theta)^T X\theta - (X\theta)^T \vec{y} - \vec{y}^T X\theta + \vec{y}^T \vec{y} \right) \\
    &= \frac{1}{2} \nabla_\theta \left( \theta^T X^T X \theta - \vec{y}^T X\theta - \vec{y}^T X\theta \right) \\
    &= \frac{1}{2} \nabla_\theta \left( \theta^T X^T X \theta - 2 (X^T \vec{y})^T \theta \right) \\
    &= \frac{1}{2} \left( 2 X^T X \theta - 2 X^T \vec{y} \right) \\
    &= X^T X \theta - X^T \vec{y}
\end{align*}
$$

**Step-by-step derivation:**

#### Step 1: Expand the quadratic form
$(X\theta - \vec{y})^T (X\theta - \vec{y}) = (X\theta)^T X\theta - (X\theta)^T \vec{y} - \vec{y}^T X\theta + \vec{y}^T \vec{y}$

#### Step 2: Simplify using matrix properties
- $(X\theta)^T = \theta^T X^T$
- $(X\theta)^T \vec{y} = \vec{y}^T X\theta$ (since both are scalars)
- $\vec{y}^T \vec{y}$ is constant with respect to $\theta$

#### Step 3: Apply matrix calculus rules
- $\nabla_\theta (\theta^T X^T X \theta) = 2 X^T X \theta$ (quadratic term rule)
- $\nabla_\theta ((X^T \vec{y})^T \theta) = X^T \vec{y}$ (linear term rule)

#### Step 4: Simplify
The $\frac{1}{2}$ factor cancels the 2's, giving us the final result.

Here, we use properties of matrix calculus to differentiate the cost function with respect to $\theta$. The key steps involve expanding the quadratic form, applying the rules for differentiating with respect to vectors and matrices, and simplifying. The result is a linear equation in $\theta$.

In the third step, we used the fact that $a^T b = b^T a$, and in the fifth step used the facts $\nabla_x b^T x = b$ and $\nabla_x x^T A x = 2A x$ for symmetric matrix $A$ (for more details, see Section 4.3 of "Linear Algebra Review and Reference"). To minimize $J$, we set its derivatives to zero, and obtain the **normal equations**:

$$
X^T X \theta = X^T \vec{y}
$$

### Understanding the Normal Equations

Setting the gradient to zero gives us the condition for optimality. The resulting equation, called the normal equation, is a system of linear equations that can be solved directly for $\theta$ (provided $X^T X$ is invertible). This is the closed-form solution for linear regression.

**Geometric interpretation:**
- $X^T X$ is the Gram matrix, which measures the correlations between features
- $X^T \vec{y}$ is the correlation between features and target
- The normal equations say: "Find $\theta$ such that the predicted correlations match the actual correlations"

**When does this solution exist?**
- When $X^T X$ is invertible (i.e., when $X$ has full column rank)
- This means no feature is a perfect linear combination of other features
- If $X^T X$ is singular, we need regularization or use gradient descent

### The Closed-Form Solution

Thus, the value of $\theta$ that minimizes $J(\theta)$ is given in closed form by the equation

$$
\theta = (X^T X)^{-1} X^T \vec{y}
$$

**Understanding this formula:**
- $(X^T X)^{-1}$ is the inverse of the Gram matrix
- $X^T \vec{y}$ is the correlation between features and target
- The product gives us the optimal parameters

This formula gives the optimal parameters for linear regression in one step. It is derived from the normal equations and uses the inverse of $X^T X$. In practice, this approach is efficient for small to medium-sized datasets, but for very large datasets or when $X^T X$ is not invertible, iterative methods like gradient descent or regularization techniques are preferred.

### Computational Complexity

**Time complexity:**
- Matrix multiplication $X^T X$: $O(nd^2)$
- Matrix inversion $(X^T X)^{-1}$: $O(d^3)$
- Final multiplication: $O(d^2)$
- **Total**: $O(nd^2 + d^3)$

**Space complexity:**
- Storing $X^T X$: $O(d^2)$
- Storing $(X^T X)^{-1}$: $O(d^2)$
- **Total**: $O(d^2)$

**Comparison with gradient descent:**
- **Normal equations**: $O(nd^2 + d^3)$ one-time cost
- **Gradient descent**: $O(nd)$ per iteration, but many iterations needed

### Numerical Stability Considerations

**Potential issues:**
1. **Singular matrices**: $X^T X$ may not be invertible
2. **Ill-conditioned matrices**: Small changes in data cause large changes in solution
3. **Numerical precision**: Matrix inversion can amplify roundoff errors

**Solutions:**
1. **Regularization**: Add $\lambda I$ to $X^T X$ before inverting
2. **QR decomposition**: More numerically stable than direct inversion
3. **SVD decomposition**: Handles singular matrices gracefully

### Practical Implementation

**Direct implementation:**
```python
theta = np.linalg.inv(X.T @ X) @ X.T @ y
```

**More stable implementation:**
```python
theta = np.linalg.solve(X.T @ X, X.T @ y)
```

**With regularization:**
```python
lambda_reg = 0.01
theta = np.linalg.solve(X.T @ X + lambda_reg * np.eye(X.shape[1]), X.T @ y)
```

### Summary

The normal equations provide a beautiful closed-form solution to linear regression. They show us that:

1. **The optimal solution exists** when the features are linearly independent
2. **The solution can be computed directly** without iteration
3. **The solution has a clear geometric interpretation** in terms of correlations
4. **The method is efficient** for small to medium datasets

However, they also have limitations that make gradient descent preferable in many practical scenarios, especially with large datasets or when numerical stability is a concern.

## From Optimization to Probabilistic Foundations

We've now explored two powerful approaches to solving linear regression: the iterative gradient descent method and the analytical normal equations. Both methods give us ways to find the parameters $\theta$ that minimize our cost function, but they approach the problem from different perspectives.

However, there's a deeper question we haven't addressed yet: **Why does the least squares cost function make sense in the first place?** What justifies using the sum of squared errors as our measure of model quality?

The answer lies in **probabilistic thinking**. By making certain assumptions about how our data is generated, we can show that the least squares approach is not just a heuristic, but the **optimal solution** under those assumptions. This probabilistic interpretation connects our optimization methods to fundamental principles in statistics and provides a foundation for understanding more sophisticated regression techniques.

In the next section, we'll explore the probabilistic assumptions that justify least squares regression and see how maximum likelihood estimation naturally leads to our familiar cost function.

---

**Previous: [LMS Algorithm](02_lms_algorithm.md)** - Learn about gradient descent and the LMS algorithm for optimizing the cost function.

**Next: [Probabilistic Interpretation](04_probabilistic_interpretation.md)** - Understand the probabilistic foundations of linear regression and maximum likelihood estimation.
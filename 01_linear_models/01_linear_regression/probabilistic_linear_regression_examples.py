"""
Probabilistic Linear Regression Examples with Comprehensive Annotations

This file implements the probabilistic interpretation of linear regression,
demonstrating the connection between least squares and maximum likelihood estimation.
The probabilistic framework provides a deeper understanding of linear regression
and connects it to statistical inference.

Key Concepts Demonstrated:
- Probabilistic model: y = Xθ + ε, where ε ~ N(0, σ²)
- Gaussian likelihood and log-likelihood
- Maximum likelihood estimation
- Connection to least squares
- Noise impact on parameter estimation
- Confidence intervals and uncertainty quantification
- Model assumptions and their implications

Mathematical Foundations:
- Likelihood: L(θ) = ∏ p(y_i | x_i; θ)
- Log-likelihood: log L(θ) = -(n/2)log(2πσ²) - (1/2σ²)Σ(y_i - θ^T x_i)²
- MLE: θ_MLE = argmax L(θ) = argmin Σ(y_i - θ^T x_i)²
- Connection: MLE = Least Squares when ε ~ N(0, σ²)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# ============================================================================
# Data Generation and Linear Model with Detailed Explanations
# ============================================================================

def generate_linear_data(n_samples=100, n_features=2, noise_std=1.0, seed=42):
    """
    Generate synthetic data according to the linear model:
    y = Xθ + ε, where ε ~ N(0, σ²)
    
    This function creates data that follows the probabilistic assumptions
    of linear regression, allowing us to study the properties of our
    estimation methods.
    
    Key Learning Points:
    - Data is generated from a true underlying model
    - Noise is Gaussian with known variance
    - We can compare estimated parameters to true parameters
    - Different noise levels affect estimation quality
    
    Parameters:
    n_samples: number of data points
    n_features: number of features (excluding intercept)
    noise_std: standard deviation of Gaussian noise
    seed: random seed for reproducibility
    
    Returns:
    X: design matrix with intercept
    y: target values
    theta_true: true parameters used to generate data
    """
    np.random.seed(seed)
    
    # True parameters that we want to recover
    theta_true = np.array([2.0, -1.5, 0.8])  # [intercept, feature1, feature2]
    
    # Generate features from standard normal distribution
    X_features = np.random.randn(n_samples, n_features)
    X = np.column_stack([np.ones(n_samples), X_features])  # Add intercept
    
    # Generate Gaussian noise
    epsilon = np.random.normal(0, noise_std, size=n_samples)
    
    # Generate targets according to the linear model
    y = X @ theta_true + epsilon
    
    print(f"Generated {n_samples} data points with {n_features} features")
    print(f"True parameters: θ = {theta_true}")
    print(f"Noise standard deviation: σ = {noise_std}")
    print(f"Signal-to-noise ratio: {np.linalg.norm(X @ theta_true) / np.linalg.norm(epsilon):.2f}")
    print()
    
    return X, y, theta_true

# ============================================================================
# Gaussian Likelihood Functions with Detailed Explanations
# ============================================================================

def gaussian_likelihood(y_i, x_i, theta, sigma):
    """
    Compute the likelihood of a single data point: p(y_i | x_i; θ)
    
    This is the probability density of observing y_i given the features x_i
    and parameters θ, assuming Gaussian noise.
    
    Key Learning Points:
    - Likelihood measures how well parameters explain observed data
    - Gaussian assumption leads to squared error in log-likelihood
    - Higher likelihood = better parameter values
    - Likelihood depends on noise variance σ²
    
    Mathematical Form:
    p(y_i | x_i; θ) = (1/√(2πσ²)) * exp(-(y_i - θ^T x_i)²/(2σ²))
    
    Parameters:
    y_i: target value for i-th example
    x_i: feature vector for i-th example
    theta: parameter vector
    sigma: noise standard deviation
    
    Returns:
    likelihood: p(y_i | x_i; θ)
    """
    # Compute mean: μ = θ^T x_i
    mu = np.dot(theta, x_i)
    
    # Gaussian likelihood formula
    coeff = 1.0 / np.sqrt(2 * np.pi * sigma**2)
    exponent = -((y_i - mu) ** 2) / (2 * sigma**2)
    likelihood = coeff * np.exp(exponent)
    
    return likelihood

def log_likelihood(y, X, theta, sigma):
    """
    Compute the log-likelihood for the entire dataset:
    log L(θ) = Σ log p(y_i | x_i; θ)
    
    The log-likelihood is often preferred over likelihood because:
    1. It converts products to sums (numerically stable)
    2. It's easier to maximize (monotonic transformation)
    3. It connects directly to squared error minimization
    
    Key Learning Points:
    - Log-likelihood is sum of individual log-likelihoods
    - Maximizing log-likelihood = minimizing squared error
    - This connects MLE to least squares
    - Log-likelihood depends on noise variance
    
    Mathematical Form:
    log L(θ) = -(n/2)log(2πσ²) - (1/2σ²)Σ(y_i - θ^T x_i)²
    
    Parameters:
    y: target vector
    X: design matrix
    theta: parameter vector
    sigma: noise standard deviation
    
    Returns:
    log_likelihood: log L(θ)
    """
    n = len(y)
    mu = X @ theta
    
    # Log-likelihood formula: log L(θ) = -(n/2)log(2πσ²) - (1/2σ²)Σ(y_i - μ_i)²
    ll = -0.5 * n * np.log(2 * np.pi * sigma**2)
    ll -= np.sum((y - mu) ** 2) / (2 * sigma**2)
    
    return ll

def demonstrate_likelihood_calculation():
    """
    Demonstrate likelihood calculations for individual data points.
    This shows how likelihood measures parameter quality and how
    it connects to the squared error.
    """
    print("=== Likelihood Calculation Demonstration ===")
    print("Understanding how likelihood measures parameter quality")
    print()
    
    # Generate data
    X, y, theta_true = generate_linear_data(n_samples=10, n_features=2, noise_std=1.0)
    
    # Test different parameter values
    theta_test1 = theta_true  # True parameters
    theta_test2 = theta_true + 0.5  # Perturbed parameters
    theta_test3 = np.zeros_like(theta_true)  # Zero parameters
    
    print("Likelihood comparison for first data point:")
    print(f"Data point: x = {X[0]}, y = {y[0]:.4f}")
    print()
    
    for i, theta in enumerate([theta_test1, theta_test2, theta_test3], 1):
        likelihood = gaussian_likelihood(y[0], X[0], theta, sigma=1.0)
        log_likelihood_val = np.log(likelihood)
        
        # Compute prediction and error
        prediction = np.dot(theta, X[0])
        error = y[0] - prediction
        squared_error = error ** 2
        
        print(f"θ_{i} = {theta}")
        print(f"  Prediction: h_θ(x) = {prediction:.4f}")
        print(f"  Error: y - h_θ(x) = {error:.4f}")
        print(f"  Squared error: (y - h_θ(x))² = {squared_error:.4f}")
        print(f"  Likelihood p(y|x;θ) = {likelihood:.6f}")
        print(f"  Log-likelihood = {log_likelihood_val:.6f}")
        print()
    
    # Show that maximum likelihood corresponds to least squares
    print("Log-likelihood for entire dataset:")
    print(f"{'Parameter Set':<15} {'Log-likelihood':<15} {'Squared Error':<15}")
    print("-" * 50)
    
    for i, theta in enumerate([theta_test1, theta_test2, theta_test3], 1):
        ll = log_likelihood(y, X, theta, sigma=1.0)
        predictions = X @ theta
        squared_error = np.sum((y - predictions) ** 2)
        
        print(f"θ_{i:<12} {ll:<15.4f} {squared_error:<15.4f}")
    
    print()
    print("Key insights:")
    print("- Higher likelihood = better parameter values")
    print("- Log-likelihood is inversely related to squared error")
    print("- True parameters give highest likelihood")
    print("- MLE minimizes squared error (when noise is Gaussian)")
    print()

# ============================================================================
# Maximum Likelihood Estimation with Detailed Explanations
# ============================================================================

def maximum_likelihood_estimation(X, y, sigma=1.0):
    """
    Find the maximum likelihood estimate of θ.
    This is equivalent to the least squares solution.
    
    Key Learning Points:
    - MLE finds parameters that maximize likelihood
    - For Gaussian noise, MLE = least squares solution
    - This provides statistical justification for least squares
    - MLE is asymptotically unbiased and efficient
    
    Mathematical Derivation:
    1. Log-likelihood: log L(θ) = -(n/2)log(2πσ²) - (1/2σ²)Σ(y_i - θ^T x_i)²
    2. To maximize log L(θ), minimize Σ(y_i - θ^T x_i)²
    3. This is exactly the least squares objective
    4. Solution: θ = (X^T X)^(-1) X^T y
    
    Parameters:
    X: design matrix
    y: target vector
    sigma: noise standard deviation (doesn't affect θ estimate)
    
    Returns:
    theta_mle: maximum likelihood estimate of θ
    """
    # MLE solution is the same as least squares: θ = (X^T X)^(-1) X^T y
    XTX = X.T @ X
    XTy = X.T @ y
    theta_mle = np.linalg.inv(XTX) @ XTy
    
    return theta_mle

def compare_mle_with_least_squares():
    """
    Demonstrate that MLE equals least squares solution.
    This shows the fundamental connection between probabilistic
    and geometric interpretations of linear regression.
    """
    print("=== Maximum Likelihood vs Least Squares ===")
    print("Demonstrating the equivalence of MLE and least squares")
    print()
    
    # Generate data
    X, y, theta_true = generate_linear_data(n_samples=50, n_features=2, noise_std=1.0)
    
    # 1. Maximum likelihood estimation
    theta_mle = maximum_likelihood_estimation(X, y, sigma=1.0)
    
    # 2. Least squares solution (normal equations)
    XTX = X.T @ X
    XTy = X.T @ y
    theta_ls = np.linalg.inv(XTX) @ XTy
    
    print("Parameter comparison:")
    print(f"{'Method':<20} {'θ₀':<10} {'θ₁':<10} {'θ₂':<10}")
    print("-" * 50)
    print(f"{'True parameters':<20} {theta_true[0]:<10.4f} {theta_true[1]:<10.4f} {theta_true[2]:<10.4f}")
    print(f"{'Maximum likelihood':<20} {theta_mle[0]:<10.4f} {theta_mle[1]:<10.4f} {theta_mle[2]:<10.4f}")
    print(f"{'Least squares':<20} {theta_ls[0]:<10.4f} {theta_ls[1]:<10.4f} {theta_ls[2]:<10.4f}")
    print()
    
    # Verify they are identical
    difference = np.linalg.norm(theta_mle - theta_ls)
    print(f"Difference between MLE and LS: {difference:.10f}")
    print(f"Methods are identical: {difference < 1e-10}")
    print()
    
    # Show likelihood and cost function values
    print("Objective function comparison:")
    print(f"{'Method':<20} {'Log-likelihood':<15} {'Squared Error':<15}")
    print("-" * 55)
    
    # MLE objective (log-likelihood)
    ll_mle = log_likelihood(y, X, theta_mle, sigma=1.0)
    
    # Least squares objective (squared error)
    predictions_ls = X @ theta_ls
    squared_error_ls = np.sum((y - predictions_ls) ** 2)
    
    print(f"{'MLE':<20} {ll_mle:<15.4f} {'N/A':<15}")
    print(f"{'Least squares':<20} {'N/A':<15} {squared_error_ls:<15.4f}")
    print()
    
    # Show the mathematical connection
    print("Mathematical connection:")
    print(f"Log-likelihood = -(n/2)log(2πσ²) - (1/2σ²) × Squared Error")
    print(f"              = {ll_mle:.4f}")
    print(f"              = -{len(y)/2:.1f} × log(2π) - {squared_error_ls/(2*1.0**2):.4f}")
    print()
    
    return theta_mle, theta_ls

# ============================================================================
# Likelihood Surface Visualization with Detailed Explanations
# ============================================================================

def visualize_likelihood_surface():
    """
    Visualize the likelihood surface to understand the optimization landscape.
    This shows how likelihood varies with parameters and why MLE works.
    """
    print("=== Likelihood Surface Visualization ===")
    print("Understanding the optimization landscape of likelihood")
    print()
    
    # Generate simple 2D data for visualization
    np.random.seed(42)
    n_samples = 20
    theta_true = np.array([1.0, 2.0])  # [intercept, slope]
    
    # Generate data
    X_features = np.random.randn(n_samples, 1)
    X = np.column_stack([np.ones(n_samples), X_features])
    y = X @ theta_true + 0.5 * np.random.randn(n_samples)
    
    print(f"Generated {n_samples} data points")
    print(f"True parameters: θ = {theta_true}")
    print()
    
    # Create parameter grid for visualization
    theta0_range = np.linspace(-2, 4, 50)
    theta1_range = np.linspace(-1, 5, 50)
    
    # Compute likelihood for each parameter combination
    likelihood_grid = np.zeros((len(theta0_range), len(theta1_range)))
    log_likelihood_grid = np.zeros((len(theta0_range), len(theta1_range)))
    
    for i, theta0 in enumerate(theta0_range):
        for j, theta1 in enumerate(theta1_range):
            theta = np.array([theta0, theta1])
            log_likelihood_grid[i, j] = log_likelihood(y, X, theta, sigma=0.5)
            likelihood_grid[i, j] = np.exp(log_likelihood_grid[i, j])
    
    # Find MLE
    theta_mle = maximum_likelihood_estimation(X, y, sigma=0.5)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Likelihood surface
    plt.subplot(1, 3, 1)
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_range, theta1_range)
    plt.contour(theta0_mesh, theta1_mesh, likelihood_grid.T, levels=20, alpha=0.7)
    plt.colorbar(label='Likelihood L(θ)', shrink=0.8)
    
    # Mark true parameters and MLE
    plt.plot(theta_true[0], theta_true[1], 'r*', markersize=15, label='True θ')
    plt.plot(theta_mle[0], theta_mle[1], 'go', markersize=10, label='MLE θ')
    
    plt.xlabel('θ₀ (intercept)')
    plt.ylabel('θ₁ (slope)')
    plt.title('Likelihood Surface L(θ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood surface
    plt.subplot(1, 3, 2)
    plt.contour(theta0_mesh, theta1_mesh, log_likelihood_grid.T, levels=20, alpha=0.7)
    plt.colorbar(label='Log-likelihood log L(θ)', shrink=0.8)
    
    # Mark true parameters and MLE
    plt.plot(theta_true[0], theta_true[1], 'r*', markersize=15, label='True θ')
    plt.plot(theta_mle[0], theta_mle[1], 'go', markersize=10, label='MLE θ')
    
    plt.xlabel('θ₀ (intercept)')
    plt.ylabel('θ₁ (slope)')
    plt.title('Log-likelihood Surface log L(θ)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Data and fitted line
    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 1], y, alpha=0.7, s=50, label='Data')
    
    # Plot true line
    x_line = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    y_true = theta_true[0] + theta_true[1] * x_line
    plt.plot(x_line, y_true, 'r-', linewidth=2, label=f'True: y = {theta_true[0]:.1f} + {theta_true[1]:.1f}x')
    
    # Plot MLE line
    y_mle = theta_mle[0] + theta_mle[1] * x_line
    plt.plot(x_line, y_mle, 'g--', linewidth=2, label=f'MLE: y = {theta_mle[0]:.1f} + {theta_mle[1]:.1f}x')
    
    plt.xlabel('Feature x')
    plt.ylabel('Target y')
    plt.title('Data and Fitted Lines')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Likelihood surface analysis:")
    print(f"  True parameters: θ = {theta_true}")
    print(f"  MLE parameters: θ = {theta_mle}")
    print(f"  Parameter error: {np.linalg.norm(theta_mle - theta_true):.4f}")
    print()
    print("Key insights:")
    print("- Likelihood surface shows how well parameters explain data")
    print("- MLE finds the peak of the likelihood surface")
    print("- Log-likelihood surface is easier to optimize (more linear)")
    print("- True parameters are close to MLE (good estimation)")
    print("- Likelihood decreases as we move away from optimal parameters")
    print()

# ============================================================================
# Noise Impact Analysis with Detailed Explanations
# ============================================================================

def analyze_noise_impact():
    """
    Analyze how noise level affects parameter estimation quality.
    This demonstrates the impact of the Gaussian noise assumption
    and how it affects our confidence in parameter estimates.
    """
    print("=== Noise Impact Analysis ===")
    print("Understanding how noise affects parameter estimation")
    print()
    
    # Generate data with different noise levels
    np.random.seed(42)
    n_samples = 100
    theta_true = np.array([2.0, -1.5, 0.8])
    
    noise_levels = [0.1, 0.5, 1.0, 2.0]
    results = {}
    
    for noise_std in noise_levels:
        # Generate data with this noise level
        X_features = np.random.randn(n_samples, 2)
        X = np.column_stack([np.ones(n_samples), X_features])
        y = X @ theta_true + noise_std * np.random.randn(n_samples)
        
        # Compute MLE
        theta_mle = maximum_likelihood_estimation(X, y, sigma=noise_std)
        
        # Compute estimation error
        error = np.linalg.norm(theta_mle - theta_true)
        
        # Compute log-likelihood
        ll = log_likelihood(y, X, theta_mle, sigma=noise_std)
        
        results[noise_std] = {
            'theta_mle': theta_mle,
            'error': error,
            'log_likelihood': ll,
            'signal_to_noise': np.linalg.norm(X @ theta_true) / np.linalg.norm(y - X @ theta_true)
        }
    
    # Display results
    print("Noise impact on parameter estimation:")
    print(f"{'Noise σ':<10} {'Parameter Error':<15} {'Log-likelihood':<15} {'S/N Ratio':<10}")
    print("-" * 60)
    
    for noise_std in noise_levels:
        result = results[noise_std]
        print(f"{noise_std:<10.1f} {result['error']:<15.4f} {result['log_likelihood']:<15.4f} {result['signal_to_noise']:<10.2f}")
    
    print()
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Parameter error vs noise
    plt.subplot(1, 3, 1)
    noise_levels_list = list(results.keys())
    errors = [results[noise]['error'] for noise in noise_levels_list]
    plt.plot(noise_levels_list, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Noise Standard Deviation σ')
    plt.ylabel('Parameter Error ||θ_MLE - θ_true||')
    plt.title('Parameter Error vs Noise Level')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Log-likelihood vs noise
    plt.subplot(1, 3, 2)
    log_likelihoods = [results[noise]['log_likelihood'] for noise in noise_levels_list]
    plt.plot(noise_levels_list, log_likelihoods, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Noise Standard Deviation σ')
    plt.ylabel('Log-likelihood log L(θ_MLE)')
    plt.title('Log-likelihood vs Noise Level')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Parameter estimates for different noise levels
    plt.subplot(1, 3, 3)
    x_pos = np.arange(len(theta_true))
    width = 0.2
    
    for i, noise_std in enumerate(noise_levels):
        theta_est = results[noise_std]['theta_mle']
        plt.bar(x_pos + i*width, theta_est, width, label=f'σ = {noise_std}', alpha=0.7)
    
    plt.bar(x_pos + len(noise_levels)*width, theta_true, width, label='True θ', alpha=0.7, color='black')
    
    plt.xlabel('Parameter Index')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Estimates vs Noise Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Key insights:")
    print("- Higher noise leads to larger parameter estimation errors")
    print("- Log-likelihood decreases with increasing noise")
    print("- Signal-to-noise ratio affects estimation quality")
    print("- MLE remains unbiased but variance increases with noise")
    print("- Gaussian noise assumption is crucial for MLE = least squares")
    print()

# ============================================================================
# Real-world Application: Housing Price Prediction
# ============================================================================

def housing_price_probabilistic():
    """
    Apply probabilistic linear regression to housing price prediction.
    This demonstrates how the probabilistic framework provides
    additional insights beyond point predictions.
    """
    print("=== Housing Price Prediction with Probabilistic Framework ===")
    print("Real-world application with uncertainty quantification")
    print()
    
    # Housing data with realistic features
    np.random.seed(42)
    n_houses = 50
    
    # Generate realistic housing data
    living_areas = np.random.uniform(800, 4000, n_houses)
    bedrooms = np.random.randint(1, 6, n_houses)
    ages = np.random.randint(0, 50, n_houses)
    
    # Create design matrix
    X_housing = np.column_stack([
        np.ones(n_houses),  # intercept
        living_areas,
        bedrooms,
        ages
    ])
    
    # True parameters (realistic values)
    theta_true = np.array([50, 0.08, 15, -0.5])  # [base, area, bedroom, age]
    
    # Generate prices with realistic noise
    noise_std = 20  # $20k standard deviation
    prices = X_housing @ theta_true + noise_std * np.random.randn(n_houses)
    
    print(f"Housing dataset: {n_houses} houses")
    print(f"Features: intercept, living area (ft²), bedrooms, age (years)")
    print(f"True parameters: θ = {theta_true}")
    print(f"Noise standard deviation: σ = {noise_std}k")
    print()
    
    # Fit probabilistic model
    theta_mle = maximum_likelihood_estimation(X_housing, prices, sigma=noise_std)
    
    print("Model fitting results:")
    print(f"{'Parameter':<12} {'True Value':<12} {'MLE Estimate':<12} {'Difference':<12}")
    print("-" * 50)
    param_names = ['Base Price', 'Area Factor', 'Bedroom Factor', 'Age Factor']
    for i, name in enumerate(param_names):
        true_val = theta_true[i]
        mle_val = theta_mle[i]
        diff = mle_val - true_val
        print(f"{name:<12} {true_val:<12.3f} {mle_val:<12.3f} {diff:<+12.3f}")
    print()
    
    # Compute log-likelihood
    ll = log_likelihood(prices, X_housing, theta_mle, sigma=noise_std)
    print(f"Log-likelihood at MLE: {ll:.2f}")
    print()
    
    # Make predictions with uncertainty
    test_houses = np.array([
        [1, 2000, 3, 10],  # 2000 ft², 3 bed, 10 years old
        [1, 1500, 2, 5],   # 1500 ft², 2 bed, 5 years old
        [1, 3000, 4, 0],   # 3000 ft², 4 bed, new construction
    ])
    
    predictions = test_houses @ theta_mle
    
    print("Predictions with uncertainty:")
    print("House | Living Area | Bedrooms | Age | Predicted Price | 95% CI")
    print("-" * 70)
    
    for i, (house, pred) in enumerate(zip(test_houses, predictions)):
        # 95% confidence interval (assuming Gaussian noise)
        ci_width = 1.96 * noise_std  # 1.96 is the 95% quantile of standard normal
        lower_ci = pred - ci_width
        upper_ci = pred + ci_width
        
        print(f"{i+1:5d} | {house[1]:11.0f} | {house[2]:8.0f} | {house[3]:3.0f} | ${pred:12.0f}k | [${lower_ci:.0f}k, ${upper_ci:.0f}k]")
    
    print()
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Predicted vs Actual with uncertainty
    plt.subplot(1, 3, 1)
    train_predictions = X_housing @ theta_mle
    plt.scatter(prices, train_predictions, alpha=0.7, s=50, label='Training data')
    
    # Add uncertainty bands
    ci_width = 1.96 * noise_std
    plt.fill_between([prices.min(), prices.max()], 
                    [prices.min() - ci_width, prices.max() - ci_width],
                    [prices.min() + ci_width, prices.max() + ci_width],
                    alpha=0.2, color='gray', label='95% CI')
    
    plt.plot([prices.min(), prices.max()], [prices.min(), prices.max()], 'r--', alpha=0.7)
    plt.xlabel('Actual Price (1000$s)')
    plt.ylabel('Predicted Price (1000$s)')
    plt.title('Predictions with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Residuals with normality check
    plt.subplot(1, 3, 2)
    residuals = train_predictions - prices
    plt.scatter(train_predictions, residuals, alpha=0.7, s=50)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Predicted Price (1000$s)')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Residual histogram vs normal
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=15, alpha=0.7, density=True, label='Residuals')
    
    # Overlay normal distribution
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    y_norm = norm.pdf(x_norm, 0, noise_std)
    plt.plot(x_norm, y_norm, 'r-', linewidth=2, label=f'N(0, {noise_std}²)')
    
    plt.xlabel('Residual Value')
    plt.ylabel('Density')
    plt.title('Residual Distribution vs Normal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Model diagnostics
    print("Model diagnostics:")
    print(f"  Mean residual: {np.mean(residuals):.2f}k")
    print(f"  Residual standard deviation: {np.std(residuals):.2f}k")
    print(f"  Theoretical noise std: {noise_std}k")
    print(f"  Residual normality (Shapiro-Wilk p-value): {norm.cdf(np.mean(residuals), 0, np.std(residuals)):.3f}")
    print()
    
    print("Key insights:")
    print("- Probabilistic framework provides uncertainty quantification")
    print("- Confidence intervals reflect our uncertainty in predictions")
    print("- Residual analysis helps validate model assumptions")
    print("- Gaussian noise assumption enables statistical inference")
    print("- MLE provides optimal parameter estimates under these assumptions")
    print()

# ============================================================================
# Advanced Topics: Model Assumptions and Violations
# ============================================================================

def analyze_model_assumptions():
    """
    Analyze what happens when model assumptions are violated.
    This demonstrates the importance of checking assumptions
    and understanding their impact on inference.
    """
    print("=== Model Assumptions Analysis ===")
    print("Understanding the impact of assumption violations")
    print()
    
    np.random.seed(42)
    n_samples = 100
    theta_true = np.array([2.0, -1.5])
    
    # Generate features
    X_features = np.random.randn(n_samples, 1)
    X = np.column_stack([np.ones(n_samples), X_features])
    
    # Case 1: Gaussian noise (assumption satisfied)
    y_gaussian = X @ theta_true + 1.0 * np.random.randn(n_samples)
    
    # Case 2: Non-Gaussian noise (assumption violated)
    y_non_gaussian = X @ theta_true + 1.0 * np.random.laplace(0, 1, n_samples)
    
    # Case 3: Heteroscedastic noise (variance depends on x)
    y_heteroscedastic = X @ theta_true + 0.5 * np.abs(X[:, 1]) * np.random.randn(n_samples)
    
    # Fit models
    theta_gaussian = maximum_likelihood_estimation(X, y_gaussian, sigma=1.0)
    theta_non_gaussian = maximum_likelihood_estimation(X, y_non_gaussian, sigma=1.0)
    theta_heteroscedastic = maximum_likelihood_estimation(X, y_heteroscedastic, sigma=1.0)
    
    print("Parameter estimation under different noise assumptions:")
    print(f"{'Noise Type':<15} {'θ₀':<10} {'θ₁':<10} {'Error':<10}")
    print("-" * 50)
    
    error_gaussian = np.linalg.norm(theta_gaussian - theta_true)
    error_non_gaussian = np.linalg.norm(theta_non_gaussian - theta_true)
    error_heteroscedastic = np.linalg.norm(theta_heteroscedastic - theta_true)
    
    print(f"{'Gaussian':<15} {theta_gaussian[0]:<10.4f} {theta_gaussian[1]:<10.4f} {error_gaussian:<10.4f}")
    print(f"{'Non-Gaussian':<15} {theta_non_gaussian[0]:<10.4f} {theta_non_gaussian[1]:<10.4f} {error_non_gaussian:<10.4f}")
    print(f"{'Heteroscedastic':<15} {theta_heteroscedastic[0]:<10.4f} {theta_heteroscedastic[1]:<10.4f} {error_heteroscedastic:<10.4f}")
    print()
    
    # Visualize residuals
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Gaussian residuals
    plt.subplot(1, 3, 1)
    residuals_gaussian = X @ theta_gaussian - y_gaussian
    plt.scatter(X[:, 1], residuals_gaussian, alpha=0.7, s=50)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Feature x')
    plt.ylabel('Residual')
    plt.title('Gaussian Noise (Assumption Satisfied)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Non-Gaussian residuals
    plt.subplot(1, 3, 2)
    residuals_non_gaussian = X @ theta_non_gaussian - y_non_gaussian
    plt.scatter(X[:, 1], residuals_non_gaussian, alpha=0.7, s=50)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Feature x')
    plt.ylabel('Residual')
    plt.title('Non-Gaussian Noise (Assumption Violated)')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Heteroscedastic residuals
    plt.subplot(1, 3, 3)
    residuals_heteroscedastic = X @ theta_heteroscedastic - y_heteroscedastic
    plt.scatter(X[:, 1], residuals_heteroscedastic, alpha=0.7, s=50)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Feature x')
    plt.ylabel('Residual')
    plt.title('Heteroscedastic Noise (Assumption Violated)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Assumption violation analysis:")
    print("- Gaussian noise: MLE is optimal and efficient")
    print("- Non-Gaussian noise: MLE may be suboptimal")
    print("- Heteroscedastic noise: constant variance assumption violated")
    print("- Residual plots help detect assumption violations")
    print("- Robust methods may be needed when assumptions fail")
    print()

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("Probabilistic Linear Regression Examples with Comprehensive Annotations")
    print("=" * 70)
    print("This file demonstrates the probabilistic interpretation of linear regression")
    print("with detailed explanations and practical applications.")
    print()
    
    # Run demonstrations
    demonstrate_likelihood_calculation()
    compare_mle_with_least_squares()
    visualize_likelihood_surface()
    analyze_noise_impact()
    housing_price_probabilistic()
    analyze_model_assumptions()
    
    print("All examples completed!")
    print("\nKey concepts demonstrated:")
    print("1. Probabilistic model: y = Xθ + ε, ε ~ N(0, σ²)")
    print("2. Gaussian likelihood and log-likelihood computation")
    print("3. Maximum likelihood estimation and its properties")
    print("4. Connection between MLE and least squares")
    print("5. Noise impact on parameter estimation")
    print("6. Uncertainty quantification and confidence intervals")
    print("7. Model assumptions and their validation")
    print("\nNext steps:")
    print("- Explore Bayesian linear regression")
    print("- Implement robust regression methods")
    print("- Study generalized linear models")
    print("- Investigate model selection and validation")
    print("- Apply to other probabilistic models") 
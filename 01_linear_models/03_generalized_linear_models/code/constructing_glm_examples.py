"""
Generalized Linear Models (GLMs): Comprehensive Implementation and Examples

This module provides a complete implementation of GLM construction concepts
as discussed in the GLM construction theory document. It includes:

1. Systematic GLM construction framework
2. Detailed implementations of Linear Regression and Logistic Regression as GLMs
3. Parameter estimation methods (MLE, IRLS, Gradient Descent)
4. Model diagnostics and validation
5. Real-world examples and applications

Key Concepts Implemented:
- Three fundamental GLM assumptions
- Exponential family response distributions
- Canonical link functions
- Maximum likelihood estimation
- Model diagnostics and validation
- Practical applications and case studies

Author: Machine Learning Course
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, bernoulli, poisson
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================================
# GENERIC GLM FRAMEWORK
# ============================================================================

class GLMFramework:
    """
    Generic GLM framework implementing the three fundamental assumptions.
    
    This class provides the foundation for constructing GLMs:
    1. Exponential family response distribution
    2. Prediction goal: h(x) = E[y|x]
    3. Linear relationship: η = θ^T x
    
    The framework enables systematic construction of GLMs for various
    response distributions and link functions.
    """
    
    def __init__(self, response_distribution, link_function, canonical_link=True):
        """
        Initialize GLM framework.
        
        Parameters:
        -----------
        response_distribution : class
            Exponential family distribution class
        link_function : callable
            Link function g(η) that maps natural parameter to mean
        canonical_link : bool
            Whether using canonical link function
        """
        self.response_distribution = response_distribution
        self.link_function = link_function
        self.canonical_link = canonical_link
        self.theta = None  # Model parameters
        
    def linear_predictor(self, X, theta):
        """
        Compute linear predictor: η = θ^T x
        
        This implements Assumption 3: η = θ^T x
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        theta : array-like, shape (n_features,)
            Model parameters
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Linear predictor values
        """
        return np.dot(X, theta)
    
    def response_function(self, eta):
        """
        Compute response function: μ = g(η)
        
        This maps the linear predictor to the mean parameter.
        
        Parameters:
        -----------
        eta : array-like
            Linear predictor values
            
        Returns:
        --------
        array-like
            Mean parameter values
        """
        return self.link_function(eta)
    
    def hypothesis_function(self, X, theta):
        """
        Compute hypothesis function: h(x) = E[y|x]
        
        This implements Assumption 2: h(x) = E[y|x]
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        theta : array-like, shape (n_features,)
            Model parameters
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted mean values
        """
        eta = self.linear_predictor(X, theta)
        return self.response_function(eta)
    
    def log_likelihood(self, X, y, theta):
        """
        Compute log-likelihood function.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Observed responses
        theta : array-like, shape (n_features,)
            Model parameters
            
        Returns:
        --------
        float
            Log-likelihood value
        """
        eta = self.linear_predictor(X, theta)
        mu = self.response_function(eta)
        
        # Compute log-likelihood using response distribution
        log_likelihood = 0
        for i in range(len(y)):
            log_likelihood += self.response_distribution.log_pdf(y[i], mu[i])
        
        return log_likelihood
    
    def fit_mle(self, X, y, initial_theta=None, method='L-BFGS-B'):
        """
        Fit GLM using maximum likelihood estimation.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Observed responses
        initial_theta : array-like, optional
            Initial parameter values
        method : str
            Optimization method
            
        Returns:
        --------
        self
            Fitted model
        """
        if initial_theta is None:
            initial_theta = np.zeros(X.shape[1])
        
        # Define negative log-likelihood for minimization
        def neg_log_likelihood(theta):
            return -self.log_likelihood(X, y, theta)
        
        # Optimize
        result = minimize(neg_log_likelihood, initial_theta, method=method)
        
        if result.success:
            self.theta = result.x
            print(f"GLM fitted successfully. Final log-likelihood: {-result.fun:.4f}")
        else:
            print("Warning: Optimization failed!")
            self.theta = result.x
        
        return self
    
    def predict(self, X):
        """
        Make predictions using fitted model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted values
        """
        if self.theta is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.hypothesis_function(X, self.theta)

# ============================================================================
# LINEAR REGRESSION AS GLM
# ============================================================================

class GaussianResponseDistribution:
    """
    Gaussian response distribution for GLMs.
    
    This implements the Gaussian distribution in exponential family form
    for use in GLM construction. The canonical link is the identity function.
    """
    
    @staticmethod
    def log_pdf(y, mu):
        """
        Compute log probability density function.
        
        Parameters:
        -----------
        y : float
            Observed value
        mu : float
            Mean parameter
            
        Returns:
        --------
        float
            Log PDF value
        """
        # Log PDF of Gaussian: -0.5 * log(2π) - 0.5 * (y-μ)²
        return -0.5 * np.log(2 * np.pi) - 0.5 * (y - mu)**2
    
    @staticmethod
    def canonical_link(eta):
        """
        Canonical link function for Gaussian: identity function.
        
        Parameters:
        -----------
        eta : float
            Natural parameter
            
        Returns:
        --------
        float
            Mean parameter
        """
        return eta

class LinearRegressionGLM(GLMFramework):
    """
    Linear Regression implemented as a GLM.
    
    This demonstrates how ordinary least squares (OLS) is a special case
    of the GLM framework with:
    - Response distribution: Gaussian
    - Canonical link: Identity function
    - Linear predictor: η = θ^T x
    - Hypothesis function: h(x) = θ^T x
    """
    
    def __init__(self):
        """Initialize Linear Regression GLM."""
        super().__init__(
            response_distribution=GaussianResponseDistribution,
            link_function=GaussianResponseDistribution.canonical_link,
            canonical_link=True
        )
    
    def fit_ols(self, X, y):
        """
        Fit using ordinary least squares (analytical solution).
        
        This provides the analytical solution for linear regression,
        which is equivalent to MLE under Gaussian assumptions.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Observed responses
            
        Returns:
        --------
        self
            Fitted model
        """
        # Add intercept if not present
        if X.shape[1] == 0 or not np.allclose(X[:, 0], 1):
            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        else:
            X_with_intercept = X
        
        # Analytical solution: θ = (X^T X)^(-1) X^T y
        self.theta = np.linalg.solve(
            np.dot(X_with_intercept.T, X_with_intercept),
            np.dot(X_with_intercept.T, y)
        )
        
        print(f"OLS fitted. Parameters: {self.theta}")
        return self
    
    def compute_r_squared(self, X, y):
        """
        Compute R-squared (coefficient of determination).
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            Observed responses
            
        Returns:
        --------
        float
            R-squared value
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)

# ============================================================================
# LOGISTIC REGRESSION AS GLM
# ============================================================================

class BernoulliResponseDistribution:
    """
    Bernoulli response distribution for GLMs.
    
    This implements the Bernoulli distribution in exponential family form
    for use in GLM construction. The canonical link is the logit function.
    """
    
    @staticmethod
    def log_pdf(y, mu):
        """
        Compute log probability mass function.
        
        Parameters:
        -----------
        y : float
            Observed value (0 or 1)
        mu : float
            Probability parameter
            
        Returns:
        --------
        float
            Log PMF value
        """
        # Log PMF of Bernoulli: y*log(μ) + (1-y)*log(1-μ)
        return y * np.log(mu) + (1 - y) * np.log(1 - mu)
    
    @staticmethod
    def canonical_link(eta):
        """
        Canonical link function for Bernoulli: sigmoid function.
        
        Parameters:
        -----------
        eta : float
            Natural parameter (log-odds)
            
        Returns:
        --------
        float
            Probability parameter
        """
        return 1 / (1 + np.exp(-eta))

class LogisticRegressionGLM(GLMFramework):
    """
    Logistic Regression implemented as a GLM.
    
    This demonstrates how logistic regression is a special case
    of the GLM framework with:
    - Response distribution: Bernoulli
    - Canonical link: Sigmoid function
    - Linear predictor: η = θ^T x
    - Hypothesis function: h(x) = 1/(1 + e^(-θ^T x))
    """
    
    def __init__(self):
        """Initialize Logistic Regression GLM."""
        super().__init__(
            response_distribution=BernoulliResponseDistribution,
            link_function=BernoulliResponseDistribution.canonical_link,
            canonical_link=True
        )
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted probabilities
        """
        return self.predict(X)
    
    def predict_classes(self, X, threshold=0.5):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        threshold : float
            Classification threshold
            
        Returns:
        --------
        array-like, shape (n_samples,)
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def compute_accuracy(self, X, y):
        """
        Compute classification accuracy.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input features
        y : array-like, shape (n_samples,)
            True class labels
            
        Returns:
        --------
        float
            Classification accuracy
        """
        y_pred = self.predict_classes(X)
        return accuracy_score(y, y_pred)

# ============================================================================
# PARAMETER ESTIMATION METHODS
# ============================================================================

def iterative_reweighted_least_squares(X, y, response_distribution, max_iter=100, tol=1e-6):
    """
    Implement Iteratively Reweighted Least Squares (IRLS) for GLMs.
    
    IRLS is an efficient algorithm for fitting GLMs with canonical links.
    It iteratively solves weighted least squares problems.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    y : array-like, shape (n_samples,)
        Observed responses
    response_distribution : class
        Response distribution class
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    array-like, shape (n_features,)
        Estimated parameters
    """
    # Add intercept if not present
    if X.shape[1] == 0 or not np.allclose(X[:, 0], 1):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_with_intercept = X
    
    n_features = X_with_intercept.shape[1]
    theta = np.zeros(n_features)  # Initialize parameters
    
    for iteration in range(max_iter):
        theta_old = theta.copy()
        
        # Compute linear predictor and mean
        eta = np.dot(X_with_intercept, theta)
        mu = response_distribution.canonical_link(eta)
        
        # Compute working responses and weights
        # z = η + (y - μ) / g'(μ)
        # w = 1 / (g'(μ)² * Var(y))
        
        # For canonical links, g'(μ) = 1/Var(y), so:
        # z = η + (y - μ) * Var(y)
        # w = Var(y)
        
        # For Bernoulli: Var(y) = μ(1-μ)
        variance = mu * (1 - mu)
        working_response = eta + (y - mu) / variance
        weights = variance
        
        # Solve weighted least squares
        W = np.diag(weights)
        theta = np.linalg.solve(
            np.dot(X_with_intercept.T, np.dot(W, X_with_intercept)),
            np.dot(X_with_intercept.T, np.dot(W, working_response))
        )
        
        # Check convergence
        if np.linalg.norm(theta - theta_old) < tol:
            print(f"IRLS converged in {iteration + 1} iterations")
            break
    
    return theta

def gradient_descent_glm(X, y, response_distribution, learning_rate=0.01, 
                        max_iter=1000, tol=1e-6):
    """
    Implement gradient descent for GLM parameter estimation.
    
    This is an alternative to IRLS that works for non-canonical links
    and can be more robust for some problems.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input features
    y : array-like, shape (n_samples,)
        Observed responses
    response_distribution : class
        Response distribution class
    learning_rate : float
        Learning rate for gradient descent
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    array-like, shape (n_features,)
        Estimated parameters
    """
    # Add intercept if not present
    if X.shape[1] == 0 or not np.allclose(X[:, 0], 1):
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
    else:
        X_with_intercept = X
    
    n_features = X_with_intercept.shape[1]
    theta = np.zeros(n_features)  # Initialize parameters
    
    for iteration in range(max_iter):
        theta_old = theta.copy()
        
        # Compute predictions
        eta = np.dot(X_with_intercept, theta)
        mu = response_distribution.canonical_link(eta)
        
        # Compute gradient (simplified for canonical links)
        # For Bernoulli with canonical link:
        # ∇ℓ = X^T (y - μ)
        gradient = np.dot(X_with_intercept.T, (y - mu))
        
        # Update parameters
        theta = theta + learning_rate * gradient
        
        # Check convergence
        if np.linalg.norm(theta - theta_old) < tol:
            print(f"Gradient descent converged in {iteration + 1} iterations")
            break
    
    return theta

# ============================================================================
# MODEL DIAGNOSTICS AND VALIDATION
# ============================================================================

def compute_deviance_residuals(y, mu, response_distribution):
    """
    Compute deviance residuals for GLM diagnostics.
    
    Deviance residuals measure the contribution of each observation
    to the overall model fit.
    
    Parameters:
    -----------
    y : array-like, shape (n_samples,)
        Observed responses
    mu : array-like, shape (n_samples,)
        Predicted means
    response_distribution : class
        Response distribution class
        
    Returns:
    --------
    array-like, shape (n_samples,)
        Deviance residuals
    """
    residuals = []
    
    for i in range(len(y)):
        # Compute log-likelihood for observed value
        ll_observed = response_distribution.log_pdf(y[i], y[i])
        
        # Compute log-likelihood for predicted value
        ll_predicted = response_distribution.log_pdf(y[i], mu[i])
        
        # Deviance residual: sign(y - μ) * sqrt(2 * (ll_observed - ll_predicted))
        sign = np.sign(y[i] - mu[i])
        deviance = sign * np.sqrt(2 * (ll_observed - ll_predicted))
        residuals.append(deviance)
    
    return np.array(residuals)

def compute_pearson_residuals(y, mu, variance_function):
    """
    Compute Pearson residuals for GLM diagnostics.
    
    Pearson residuals are standardized residuals that should be
    approximately normal under the correct model.
    
    Parameters:
    -----------
    y : array-like, shape (n_samples,)
        Observed responses
    mu : array-like, shape (n_samples,)
        Predicted means
    variance_function : callable
        Function that computes variance as function of mean
        
    Returns:
    --------
    array-like, shape (n_samples,)
        Pearson residuals
    """
    variance = variance_function(mu)
    return (y - mu) / np.sqrt(variance)

def plot_glm_diagnostics(y, y_pred, residuals, title="GLM Diagnostics"):
    """
    Create diagnostic plots for GLM validation.
    
    Parameters:
    -----------
    y : array-like
        Observed responses
    y_pred : array-like
        Predicted responses
    residuals : array-like
        Model residuals
    title : str
        Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Observed vs Predicted
    axes[0, 0].scatter(y_pred, y, alpha=0.6)
    axes[0, 0].plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Observed')
    axes[0, 0].set_title('Observed vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals histogram
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    from scipy.stats import probplot
    probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normal)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# REAL-WORLD EXAMPLES AND APPLICATIONS
# ============================================================================

def housing_price_example():
    """
    Demonstrate Linear Regression GLM with housing price data.
    
    This example shows how linear regression can be constructed as a GLM
    and how to interpret the results.
    """
    print("=" * 60)
    print("HOUSING PRICE PREDICTION: LINEAR REGRESSION AS GLM")
    print("=" * 60)
    
    # Generate synthetic housing data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: square footage, number of bedrooms, age
    square_footage = np.random.normal(2000, 500, n_samples)
    bedrooms = np.random.poisson(3, n_samples)
    age = np.random.exponential(10, n_samples)
    
    # True parameters
    true_theta = np.array([100, 50, 25, -1000])  # intercept, sqft, bedrooms, age
    
    # Generate prices with noise
    X = np.column_stack([square_footage, bedrooms, age])
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # True prices
    true_prices = np.dot(X_with_intercept, true_theta)
    
    # Add noise (Gaussian with σ = 5000)
    prices = true_prices + np.random.normal(0, 5000, n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, prices, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset: {n_samples} houses")
    print(f"Features: Square footage, Bedrooms, Age")
    print(f"Target: Price")
    
    # Fit using our GLM implementation
    print("\n1. Fitting Linear Regression GLM...")
    lr_glm = LinearRegressionGLM()
    lr_glm.fit_ols(X_train, y_train)
    
    print(f"Estimated parameters: {lr_glm.theta}")
    print(f"True parameters: {true_theta}")
    
    # Compare with sklearn
    print("\n2. Comparing with sklearn LinearRegression...")
    lr_sklearn = LinearRegression()
    lr_sklearn.fit(X_train, y_train)
    
    print(f"sklearn intercept: {lr_sklearn.intercept_:.2f}")
    print(f"sklearn coefficients: {lr_sklearn.coef_}")
    
    # Evaluate performance
    print("\n3. Model Performance:")
    y_pred_glm = lr_glm.predict(X_test)
    y_pred_sklearn = lr_sklearn.predict(X_test)
    
    mse_glm = mean_squared_error(y_test, y_pred_glm)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_glm = lr_glm.compute_r_squared(X_test, y_test)
    r2_sklearn = lr_sklearn.score(X_test, y_test)
    
    print(f"GLM MSE: {mse_glm:.2f}")
    print(f"sklearn MSE: {mse_sklearn:.2f}")
    print(f"GLM R²: {r2_glm:.4f}")
    print(f"sklearn R²: {r2_sklearn:.4f}")
    
    # Model diagnostics
    print("\n4. Model Diagnostics...")
    residuals = y_test - y_pred_glm
    plot_glm_diagnostics(y_test, y_pred_glm, residuals, 
                        "Linear Regression GLM Diagnostics")
    
    # Feature importance
    print("\n5. Feature Importance:")
    feature_names = ['Intercept', 'Square Footage', 'Bedrooms', 'Age']
    for name, coef in zip(feature_names, lr_glm.theta):
        print(f"{name}: {coef:.2f}")
    
    print("\nInterpretation:")
    print("- Square footage: Each additional sq ft adds $50 to price")
    print("- Bedrooms: Each additional bedroom adds $25 to price")
    print("- Age: Each year reduces price by $1000")

def medical_diagnosis_example():
    """
    Demonstrate Logistic Regression GLM with medical diagnosis data.
    
    This example shows how logistic regression can be constructed as a GLM
    and how to interpret the results in terms of odds ratios.
    """
    print("\n" + "=" * 60)
    print("MEDICAL DIAGNOSIS: LOGISTIC REGRESSION AS GLM")
    print("=" * 60)
    
    # Generate synthetic medical data
    np.random.seed(42)
    n_samples = 1000
    
    # Features: age, blood pressure, cholesterol, smoking status
    age = np.random.normal(55, 15, n_samples)
    blood_pressure = np.random.normal(140, 20, n_samples)
    cholesterol = np.random.normal(200, 40, n_samples)
    smoking = np.random.binomial(1, 0.3, n_samples)
    
    # True parameters (log-odds)
    true_theta = np.array([-8, 0.05, 0.02, 0.01, 1.5])  # intercept, age, bp, chol, smoking
    
    # Generate disease probability
    X = np.column_stack([age, blood_pressure, cholesterol, smoking])
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # True log-odds
    true_log_odds = np.dot(X_with_intercept, true_theta)
    
    # True probabilities
    true_probabilities = 1 / (1 + np.exp(-true_log_odds))
    
    # Generate disease status
    disease_status = np.random.binomial(1, true_probabilities)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, disease_status, test_size=0.2, random_state=42, stratify=disease_status
    )
    
    print(f"\nDataset: {n_samples} patients")
    print(f"Features: Age, Blood Pressure, Cholesterol, Smoking Status")
    print(f"Target: Disease Status (0=Healthy, 1=Disease)")
    print(f"Disease prevalence: {np.mean(disease_status):.1%}")
    
    # Fit using our GLM implementation
    print("\n1. Fitting Logistic Regression GLM...")
    lr_glm = LogisticRegressionGLM()
    lr_glm.fit_mle(X_train, y_train)
    
    print(f"Estimated parameters: {lr_glm.theta}")
    print(f"True parameters: {true_theta}")
    
    # Compare with sklearn
    print("\n2. Comparing with sklearn LogisticRegression...")
    lr_sklearn = LogisticRegression(random_state=42)
    lr_sklearn.fit(X_train, y_train)
    
    print(f"sklearn intercept: {lr_sklearn.intercept_[0]:.4f}")
    print(f"sklearn coefficients: {lr_sklearn.coef_[0]}")
    
    # Evaluate performance
    print("\n3. Model Performance:")
    y_pred_glm = lr_glm.predict_classes(X_test)
    y_pred_sklearn = lr_sklearn.predict(X_test)
    
    accuracy_glm = lr_glm.compute_accuracy(X_test, y_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    
    print(f"GLM Accuracy: {accuracy_glm:.4f}")
    print(f"sklearn Accuracy: {accuracy_sklearn:.4f}")
    
    # Detailed classification report
    print("\n4. Classification Report:")
    print(classification_report(y_test, y_pred_glm, 
                               target_names=['Healthy', 'Disease']))
    
    # Odds ratios interpretation
    print("\n5. Odds Ratios (Feature Effects):")
    feature_names = ['Intercept', 'Age', 'Blood Pressure', 'Cholesterol', 'Smoking']
    for name, coef in zip(feature_names, lr_glm.theta):
        if name != 'Intercept':
            odds_ratio = np.exp(coef)
            print(f"{name}: OR = {odds_ratio:.3f}")
            if odds_ratio > 1:
                print(f"  → {name} increases disease risk by {(odds_ratio-1)*100:.1f}%")
            else:
                print(f"  → {name} decreases disease risk by {(1-odds_ratio)*100:.1f}%")
    
    # Model diagnostics
    print("\n6. Model Diagnostics...")
    y_pred_proba = lr_glm.predict_proba(X_test)
    
    # Deviance residuals
    deviance_residuals = compute_deviance_residuals(
        y_test, y_pred_proba, BernoulliResponseDistribution
    )
    
    plot_glm_diagnostics(y_test, y_pred_proba, deviance_residuals,
                        "Logistic Regression GLM Diagnostics")
    
    print("\nInterpretation:")
    print("- Age: Each year increases disease risk by 5%")
    print("- Blood Pressure: Each unit increases disease risk by 2%")
    print("- Cholesterol: Each unit increases disease risk by 1%")
    print("- Smoking: Smokers have 4.5x higher disease risk")

# ============================================================================
# COMPARISON AND BENCHMARKING
# ============================================================================

def compare_estimation_methods():
    """
    Compare different parameter estimation methods for GLMs.
    
    This demonstrates the equivalence and differences between:
    1. Maximum Likelihood Estimation (MLE)
    2. Iteratively Reweighted Least Squares (IRLS)
    3. Gradient Descent
    4. Analytical solutions (when available)
    """
    print("=" * 60)
    print("COMPARING GLM ESTIMATION METHODS")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    
    # Features
    X = np.random.normal(0, 1, (n_samples, 2))
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # True parameters
    true_theta = np.array([0.5, 1.0, -0.5])
    
    # Generate responses (logistic regression)
    log_odds = np.dot(X_with_intercept, true_theta)
    probabilities = 1 / (1 + np.exp(-log_odds))
    y = np.random.binomial(1, probabilities)
    
    print(f"Dataset: {n_samples} samples, 2 features")
    print(f"True parameters: {true_theta}")
    
    # Method 1: Our GLM implementation (MLE)
    print("\n1. GLM Framework (MLE):")
    lr_glm = LogisticRegressionGLM()
    lr_glm.fit_mle(X, y)
    print(f"Parameters: {lr_glm.theta}")
    print(f"Log-likelihood: {lr_glm.log_likelihood(X, y, lr_glm.theta):.4f}")
    
    # Method 2: IRLS
    print("\n2. Iteratively Reweighted Least Squares (IRLS):")
    theta_irls = iterative_reweighted_least_squares(X, y, BernoulliResponseDistribution)
    print(f"Parameters: {theta_irls}")
    
    # Method 3: Gradient Descent
    print("\n3. Gradient Descent:")
    theta_gd = gradient_descent_glm(X, y, BernoulliResponseDistribution)
    print(f"Parameters: {theta_gd}")
    
    # Method 4: sklearn
    print("\n4. sklearn LogisticRegression:")
    lr_sklearn = LogisticRegression(random_state=42)
    lr_sklearn.fit(X, y)
    sklearn_theta = np.concatenate([lr_sklearn.intercept_, lr_sklearn.coef_[0]])
    print(f"Parameters: {sklearn_theta}")
    
    # Compare results
    print("\n5. Comparison:")
    methods = ['GLM MLE', 'IRLS', 'Gradient Descent', 'sklearn']
    thetas = [lr_glm.theta, theta_irls, theta_gd, sklearn_theta]
    
    for method, theta in zip(methods, thetas):
        mse = np.mean((theta - true_theta)**2)
        print(f"{method}: MSE = {mse:.6f}")

# ============================================================================
# MAIN EXECUTION AND DEMONSTRATIONS
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block demonstrating GLM construction concepts.
    
    This section runs comprehensive demonstrations of:
    1. Linear Regression as GLM
    2. Logistic Regression as GLM
    3. Parameter estimation methods
    4. Model diagnostics and validation
    5. Real-world applications
    """
    
    print("GENERALIZED LINEAR MODELS: COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstration shows how to systematically construct GLMs")
    print("and how familiar models like linear regression and logistic")
    print("regression are special cases of the GLM framework.")
    
    # Run real-world examples
    housing_price_example()
    medical_diagnosis_example()
    
    # Compare estimation methods
    compare_estimation_methods()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. GLMs provide a unified framework for diverse prediction problems")
    print("2. Linear regression and logistic regression are special cases")
    print("3. The three GLM assumptions enable systematic model construction")
    print("4. Canonical links provide optimal statistical properties")
    print("5. Multiple estimation methods are available and equivalent")
    print("6. Model diagnostics ensure appropriate fit and interpretation")
    
    print("\nNext Steps:")
    print("- Explore other exponential family distributions")
    print("- Implement non-canonical link functions")
    print("- Apply GLMs to your own prediction problems")
    print("- Study advanced topics like regularization and mixed models")
    print("- Explore advanced materials in exponential_family/ directory")
    
    print("\nAdditional Resources:")
    print("- exponential_family/ directory contains comprehensive reference materials")
    print("- MIT, Princeton, Berkeley, Columbia, and Purdue materials available")
    print("- Study advanced theoretical treatments and practical applications") 
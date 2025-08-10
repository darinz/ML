"""
Gaussian Discriminant Analysis (GDA) Implementation

This module implements Gaussian Discriminant Analysis, a generative learning algorithm
that models class-conditional distributions as multivariate normal distributions.

Key Concepts:
- Models p(x|y) as multivariate normal distributions
- Assumes shared covariance matrix across classes (leads to linear decision boundaries)
- Uses Bayes' rule to compute posterior probabilities
- Maximum likelihood estimation for parameter learning

Mathematical Foundation:
- y ~ Bernoulli(φ)  (class prior)
- x|y=0 ~ N(μ₀, Σ)  (class 0 distribution)
- x|y=1 ~ N(μ₁, Σ)  (class 1 distribution)
- p(y|x) = p(x|y)p(y)/p(x)  (Bayes' rule)

Author: Machine Learning Course Materials
"""

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def bayes_posterior(px_y, py, px):
    """
    Compute posterior probability using Bayes' rule.
    
    This is the fundamental equation that converts generative models (p(x|y), p(y))
    into discriminative predictions (p(y|x)).
    
    Parameters:
    -----------
    px_y : array-like
        Likelihood p(x|y) for each class [p(x|y=0), p(x|y=1)]
    py : array-like  
        Prior p(y) for each class [p(y=0), p(y=1)]
    px : float
        Evidence p(x) = Σ_y p(x|y)p(y) (normalization constant)
    
    Returns:
    --------
    array-like
        Posterior p(y|x) for each class [p(y=0|x), p(y=1|x)]
    
    Example:
    --------
    >>> px_y = np.array([0.2, 0.6])  # p(x|y=0), p(x|y=1)
    >>> py = np.array([0.7, 0.3])    # p(y=0), p(y=1)
    >>> px = np.sum(px_y * py)       # p(x) = 0.2*0.7 + 0.6*0.3 = 0.32
    >>> posterior = bayes_posterior(px_y, py, px)
    >>> print(f"Posterior: {posterior}")
    """
    return (px_y * py) / px

def multivariate_normal_density(x, mu, Sigma):
    """
    Compute multivariate normal density at point x.
    
    The multivariate normal density function is:
    p(x; μ, Σ) = (1/(2π)^(d/2) |Σ|^(1/2)) * exp(-0.5 * (x-μ)^T Σ^(-1) (x-μ))
    
    Parameters:
    -----------
    x : array-like
        Data point (d-dimensional)
    mu : array-like
        Mean vector (d-dimensional)
    Sigma : array-like
        Covariance matrix (d x d)
    
    Returns:
    --------
    float
        Density value at x
    
    Example:
    --------
    >>> x = np.array([1.0, 2.0])
    >>> mu = np.array([0.0, 0.0])
    >>> Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity matrix
    >>> density = multivariate_normal_density(x, mu, Sigma)
    >>> print(f"Density at x: {density:.6f}")
    """
    return multivariate_normal.pdf(x, mean=mu, cov=Sigma)

def gda_fit(X, y):
    """
    Fit Gaussian Discriminant Analysis model using maximum likelihood estimation.
    
    This function estimates the parameters φ, μ₀, μ₁, and Σ from training data.
    The MLE estimates are:
    - φ = (1/n) Σᵢ 1{y⁽ⁱ⁾ = 1}  (fraction of class 1 examples)
    - μ₀ = (Σᵢ 1{y⁽ⁱ⁾ = 0} x⁽ⁱ⁾) / (Σᵢ 1{y⁽ⁱ⁾ = 0})  (mean of class 0)
    - μ₁ = (Σᵢ 1{y⁽ⁱ⁾ = 1} x⁽ⁱ⁾) / (Σᵢ 1{y⁽ⁱ⁾ = 1})  (mean of class 1)
    - Σ = (1/n) Σᵢ (x⁽ⁱ⁾ - μ_{y⁽ⁱ⁾})(x⁽ⁱ⁾ - μ_{y⁽ⁱ⁾})^T  (shared covariance)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training data matrix
    y : array-like, shape (n_samples,)
        Binary labels (0 or 1)
    
    Returns:
    --------
    tuple
        (φ, μ₀, μ₁, Σ) - estimated parameters
    
    Example:
    --------
    >>> X = np.array([[1, 2], [1.2, 1.9], [0.8, 2.2],  # class 0
    ...               [3, 3.5], [3.2, 3], [2.8, 3.2]]) # class 1
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> phi, mu0, mu1, Sigma = gda_fit(X, y)
    >>> print(f"Class prior φ: {phi:.3f}")
    >>> print(f"Class 0 mean μ₀: {mu0}")
    >>> print(f"Class 1 mean μ₁: {mu1}")
    >>> print(f"Shared covariance Σ:\n{Sigma}")
    """
    n = X.shape[0]
    
    # Estimate class prior φ
    phi = np.mean(y == 1)
    
    # Estimate class means μ₀ and μ₁
    mu0 = X[y == 0].mean(axis=0)
    mu1 = X[y == 1].mean(axis=0)
    
    # Estimate shared covariance matrix Σ
    Sigma = np.zeros((X.shape[1], X.shape[1]))
    for i in range(n):
        # Use appropriate mean based on class
        mu_yi = mu1 if y[i] == 1 else mu0
        diff = (X[i] - mu_yi).reshape(-1, 1)
        Sigma += diff @ diff.T
    Sigma /= n
    
    return phi, mu0, mu1, Sigma

def gda_predict(X, phi, mu0, mu1, Sigma):
    """
    Make predictions using fitted GDA model.
    
    For each data point, compute posterior probabilities using Bayes' rule:
    p(y=1|x) = p(x|y=1)p(y=1) / [p(x|y=1)p(y=1) + p(x|y=0)p(y=0)]
    
    Then classify based on MAP decision rule: argmax_y p(y|x)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to predict on
    phi : float
        Prior probability of class 1
    mu0 : array-like
        Mean vector for class 0
    mu1 : array-like
        Mean vector for class 1
    Sigma : array-like
        Shared covariance matrix
    
    Returns:
    --------
    array-like, shape (n_samples,)
        Predicted class labels (0 or 1)
    
    Example:
    --------
    >>> # After fitting the model
    >>> phi, mu0, mu1, Sigma = gda_fit(X_train, y_train)
    >>> predictions = gda_predict(X_test, phi, mu0, mu1, Sigma)
    >>> print(f"Predictions: {predictions}")
    """
    # Compute likelihoods p(x|y) for each class
    p0 = multivariate_normal.pdf(X, mean=mu0, cov=Sigma)  # p(x|y=0)
    p1 = multivariate_normal.pdf(X, mean=mu1, cov=Sigma)  # p(x|y=1)
    
    # Compute unnormalized posteriors
    post0 = p0 * (1 - phi)  # p(x|y=0)p(y=0)
    post1 = p1 * phi        # p(x|y=1)p(y=1)
    
    # MAP decision rule: choose class with higher posterior
    return (post1 > post0).astype(int)

def gda_predict_proba(X, phi, mu0, mu1, Sigma):
    """
    Compute posterior probabilities using fitted GDA model.
    
    Returns the posterior probability p(y=1|x) for each data point.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Data to predict on
    phi : float
        Prior probability of class 1
    mu0 : array-like
        Mean vector for class 0
    mu1 : array-like
        Mean vector for class 1
    Sigma : array-like
        Shared covariance matrix
    
    Returns:
    --------
    array-like, shape (n_samples,)
        Posterior probabilities p(y=1|x)
    
    Example:
    --------
    >>> phi, mu0, mu1, Sigma = gda_fit(X_train, y_train)
    >>> probas = gda_predict_proba(X_test, phi, mu0, mu1, Sigma)
    >>> print(f"Posterior probabilities: {probas}")
    """
    # Compute likelihoods
    p0 = multivariate_normal.pdf(X, mean=mu0, cov=Sigma)
    p1 = multivariate_normal.pdf(X, mean=mu1, cov=Sigma)
    
    # Compute posteriors using Bayes' rule
    post0 = p0 * (1 - phi)
    post1 = p1 * phi
    
    # Normalize to get p(y=1|x)
    total = post0 + post1
    return post1 / total

def sigmoid(z):
    """
    Sigmoid function: σ(z) = 1/(1 + e^(-z))
    
    This is used in logistic regression and appears in the relationship
    between GDA and logistic regression.
    
    Parameters:
    -----------
    z : array-like
        Input values
    
    Returns:
    --------
    array-like
        Sigmoid values between 0 and 1
    """
    return 1 / (1 + np.exp(-z))

def logistic_regression_predict(X, theta):
    """
    Logistic regression prediction for comparison with GDA.
    
    This shows how logistic regression directly models p(y=1|x) = σ(θ^T x),
    while GDA models p(x|y) and p(y) separately.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data
    theta : array-like, shape (n_features,)
        Logistic regression parameters
    
    Returns:
    --------
    array-like, shape (n_samples,)
        Predicted probabilities p(y=1|x)
    
    Example:
    --------
    >>> theta = np.array([0.5, -0.25])
    >>> probs = logistic_regression_predict(X, theta)
    >>> print(f"Logistic regression probabilities: {probs}")
    """
    return sigmoid(X @ theta)

def plot_gda_decision_boundary(X, y, phi, mu0, mu1, Sigma, title="GDA Decision Boundary"):
    """
    Visualize GDA decision boundary and class distributions.
    
    This function creates a plot showing:
    - Training data points colored by class
    - Fitted Gaussian contours for each class
    - Linear decision boundary (due to shared covariance)
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        2D training data
    y : array-like, shape (n_samples,)
        Class labels
    phi, mu0, mu1, Sigma : fitted GDA parameters
    title : str
        Plot title
    """
    # Create grid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute predictions on grid
    predictions = gda_predict(grid_points, phi, mu0, mu1, Sigma)
    predictions = predictions.reshape(xx.shape)
    
    # Plot decision boundary and data
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class 1', alpha=0.7)
    plt.scatter(mu0[0], mu0[1], c='red', s=200, marker='x', linewidths=3, label='μ₀')
    plt.scatter(mu1[0], mu1[1], c='blue', s=200, marker='x', linewidths=3, label='μ₁')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# =====================
# Example Usage & Tests
# =====================

if __name__ == "__main__":
    print("=" * 60)
    print("Gaussian Discriminant Analysis (GDA) Examples")
    print("=" * 60)
    
    # Example 1: Bayes' Rule
    print("\n1. Bayes' Rule Example:")
    print("-" * 30)
    px_y = np.array([0.2, 0.6])  # p(x|y=0), p(x|y=1)
    py = np.array([0.7, 0.3])    # p(y=0), p(y=1)
    px = np.sum(px_y * py)       # p(x) = evidence
    posterior = bayes_posterior(px_y, py, px)
    print(f"Likelihoods p(x|y): {px_y}")
    print(f"Priors p(y): {py}")
    print(f"Evidence p(x): {px:.3f}")
    print(f"Posterior p(y|x): {posterior}")
    print(f"Sum of posteriors: {np.sum(posterior):.3f}")

    # Example 2: Multivariate Normal Density
    print("\n2. Multivariate Normal Density Example:")
    print("-" * 40)
    x = np.array([1.0, 2.0])
    mu = np.array([0.0, 0.0])
    Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity matrix
    density = multivariate_normal_density(x, mu, Sigma)
    print(f"Point x: {x}")
    print(f"Mean μ: {mu}")
    print(f"Covariance Σ:\n{Sigma}")
    print(f"Density at x: {density:.6f}")

    # Example 3: GDA on Synthetic Data
    print("\n3. GDA Parameter Estimation Example:")
    print("-" * 35)
    
    # Create synthetic dataset with two well-separated classes
    np.random.seed(42)
    n_per_class = 50
    
    # Class 0: centered around (0, 0)
    X0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_per_class)
    
    # Class 1: centered around (3, 3)
    X1 = np.random.multivariate_normal([3, 3], [[1, 0.5], [0.5, 1]], n_per_class)
    
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Fit GDA model
    phi, mu0, mu1, Sigma = gda_fit(X, y)
    print(f"\nFitted Parameters:")
    print(f"Class prior φ: {phi:.3f}")
    print(f"Class 0 mean μ₀: [{mu0[0]:.3f}, {mu0[1]:.3f}]")
    print(f"Class 1 mean μ₁: [{mu1[0]:.3f}, {mu1[1]:.3f}]")
    print(f"Shared covariance Σ:\n{Sigma}")
    
    # Make predictions
    predictions = gda_predict(X, phi, mu0, mu1, Sigma)
    accuracy = np.mean(predictions == y)
    print(f"\nTraining accuracy: {accuracy:.3f}")
    
    # Compute posterior probabilities for a few examples
    probas = gda_predict_proba(X[:5], phi, mu0, mu1, Sigma)
    print(f"\nPosterior probabilities for first 5 examples:")
    for i, (true_label, prob) in enumerate(zip(y[:5], probas)):
        print(f"  Example {i+1}: true={int(true_label)}, p(y=1|x)={prob:.3f}")

    # Example 4: Comparison with Logistic Regression
    print("\n4. GDA vs Logistic Regression Comparison:")
    print("-" * 40)
    
    # For logistic regression, we need to fit parameters (using a simple approach)
    # In practice, you'd use sklearn or optimize the likelihood
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    
    gda_probas = gda_predict_proba(X, phi, mu0, mu1, Sigma)
    lr_probas = lr.predict_proba(X)[:, 1]
    
    print(f"GDA vs Logistic Regression probabilities (first 5 examples):")
    for i in range(5):
        print(f"  Example {i+1}: GDA={gda_probas[i]:.3f}, LR={lr_probas[i]:.3f}")
    
    # Example 5: Visualization
    print("\n5. Generating visualization...")
    try:
        plot_gda_decision_boundary(X, y, phi, mu0, mu1, Sigma, 
                                 "GDA Decision Boundary with Synthetic Data")
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("(This requires matplotlib to be installed)")

    print("\n" + "=" * 60)
    print("GDA Examples Completed!")
    print("=" * 60) 
"""
Model Selection, Cross-Validation, and Bayesian Methods Examples
================================================================

This file demonstrates all concepts covered in 02_model_selection.md including
model selection, cross-validation techniques, and Bayesian statistical methods.

Key Concepts Covered:
- Model selection problem and bias-variance tradeoff
- Hold-out, k-fold, and leave-one-out cross-validation
- Maximum Likelihood Estimation (MLE)
- Maximum A Posteriori (MAP) estimation
- Full Bayesian inference
- Practical guidelines and best practices
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, BayesianRidge
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.datasets import make_regression, make_classification
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("MODEL SELECTION, CROSS-VALIDATION, AND BAYESIAN METHODS")
print("=" * 80)

# ============================================================================
# SECTION 1: MODEL SELECTION PROBLEM
# ============================================================================

print("\n" + "="*60)
print("SECTION 1: MODEL SELECTION PROBLEM")
print("="*60)

def demonstrate_model_selection_problem():
    """
    Demonstrates the fundamental challenge of model selection:
    finding the right balance between bias and variance.
    """
    print("\n1.1 The Bias-Variance Tradeoff")
    print("-" * 50)
    
    # Generate synthetic data with known true function
    def true_function(x):
        """True underlying function: sin(2πx) + noise"""
        return np.sin(2 * np.pi * x)
    
    # Generate data
    np.random.seed(42)
    x = np.sort(np.random.rand(100))
    y = true_function(x) + np.random.randn(100) * 0.1
    
    print(f"Generated dataset: {len(x)} points")
    print(f"True function: sin(2πx)")
    print(f"Noise level: 0.1")
    
    # Split data
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)
    
    print("\n1.2 Polynomial Model Selection")
    print("-" * 50)
    
    # Try different polynomial degrees
    degrees = range(1, 11)
    train_errors = []
    val_errors = []
    
    print("Degree\tTrain MSE\tVal MSE\t\tBias\t\tVariance")
    print("-" * 60)
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(x_train[:, None])
        X_val_poly = poly.transform(x_val[:, None])
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_poly)
        y_val_pred = model.predict(X_val_poly)
        
        # Calculate errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)
        
        # Estimate bias and variance (simplified)
        bias_est = train_mse  # Simplified bias estimate
        variance_est = val_mse - train_mse  # Simplified variance estimate
        
        train_errors.append(train_mse)
        val_errors.append(val_mse)
        
        print(f"{degree}\t{train_mse:.4f}\t\t{val_mse:.4f}\t\t{bias_est:.4f}\t\t{variance_est:.4f}")
    
    # Find optimal degree
    optimal_degree = degrees[np.argmin(val_errors)]
    print(f"\nOptimal polynomial degree: {optimal_degree}")
    print(f"Best validation MSE: {min(val_errors):.4f}")
    
    # Visualize the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(degrees, train_errors, 'b-o', label='Training Error')
    plt.plot(degrees, val_errors, 'r-o', label='Validation Error')
    plt.axvline(optimal_degree, color='green', linestyle='--', label=f'Optimal Degree: {optimal_degree}')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Selection: Training vs Validation Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    # Plot the best model
    poly_optimal = PolynomialFeatures(optimal_degree)
    X_train_optimal = poly_optimal.fit_transform(x_train[:, None])
    X_val_optimal = poly_optimal.transform(x_val[:, None])
    
    model_optimal = LinearRegression()
    model_optimal.fit(X_train_optimal, y_train)
    
    # Plot predictions
    x_plot = np.linspace(0, 1, 100)
    X_plot = poly_optimal.transform(x_plot[:, None])
    y_plot = model_optimal.predict(X_plot)
    
    plt.scatter(x_train, y_train, alpha=0.6, label='Training Data')
    plt.scatter(x_val, y_val, alpha=0.6, label='Validation Data')
    plt.plot(x_plot, true_function(x_plot), 'g-', label='True Function', linewidth=2)
    plt.plot(x_plot, y_plot, 'r--', label=f'Polynomial (degree={optimal_degree})', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Best Model Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Bias-variance decomposition
    bias_est = np.array(train_errors)
    variance_est = np.array(val_errors) - np.array(train_errors)
    
    plt.plot(degrees, bias_est, 'b-o', label='Bias (Estimated)')
    plt.plot(degrees, variance_est, 'r-o', label='Variance (Estimated)')
    plt.plot(degrees, bias_est + variance_est, 'g-o', label='Total Error')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Error')
    plt.title('Bias-Variance Decomposition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Show overfitting and underfitting examples
    # Underfitting: degree 1
    poly_under = PolynomialFeatures(1)
    X_train_under = poly_under.fit_transform(x_train[:, None])
    model_under = LinearRegression()
    model_under.fit(X_train_under, y_train)
    y_under = model_under.predict(poly_under.transform(x_plot[:, None]))
    
    # Overfitting: degree 9
    poly_over = PolynomialFeatures(9)
    X_train_over = poly_over.fit_transform(x_train[:, None])
    model_over = LinearRegression()
    model_over.fit(X_train_over, y_train)
    y_over = model_over.predict(poly_over.transform(x_plot[:, None]))
    
    plt.scatter(x_train, y_train, alpha=0.6, label='Training Data')
    plt.plot(x_plot, true_function(x_plot), 'g-', label='True Function', linewidth=2)
    plt.plot(x_plot, y_under, 'b--', label='Underfitting (degree=1)', linewidth=2)
    plt.plot(x_plot, y_over, 'r--', label='Overfitting (degree=9)', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Underfitting vs Overfitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_selection_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# SECTION 2: CROSS-VALIDATION METHODS
# ============================================================================

print("\n" + "="*60)
print("SECTION 2: CROSS-VALIDATION METHODS")
print("="*60)

def demonstrate_hold_out_validation():
    """
    Demonstrates hold-out cross-validation (simple train/validation split).
    """
    print("\n2.1 Hold-out Cross Validation")
    print("-" * 50)
    
    # Generate data
    X, y, true_coef = make_regression(n_samples=1000, n_features=10, n_informative=5, 
                                     noise=0.5, random_state=42, coef=True)
    
    print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Different split ratios
    split_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("\nHold-out Validation with Different Split Ratios")
    print("Train %\tVal %\tTrain Size\tVal Size\tVal MSE")
    print("-" * 60)
    
    for train_ratio in split_ratios:
        val_ratio = 1 - train_ratio
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=val_ratio, random_state=42
        )
        
        # Fit Ridge regression
        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = ridge.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        
        print(f"{train_ratio*100:.0f}%\t{val_ratio*100:.0f}%\t{len(X_train)}\t\t{len(X_val)}\t\t{val_mse:.4f}")
    
    print("\nKey Observations:")
    print("- Larger validation sets give more reliable estimates")
    print("- But smaller training sets may hurt model performance")
    print("- 70-30 split is often a good compromise")

def demonstrate_k_fold_cross_validation():
    """
    Demonstrates k-fold cross-validation with different k values.
    """
    print("\n2.2 k-Fold Cross Validation")
    print("-" * 50)
    
    # Generate data
    X, y, true_coef = make_regression(n_samples=200, n_features=10, n_informative=5, 
                                     noise=0.5, random_state=42, coef=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different k values
    k_values = [2, 3, 5, 10, 20]
    
    print("k\tFold Size\tCV MSE\t\tCV Std\t\tTime (relative)")
    print("-" * 60)
    
    for k in k_values:
        # Perform k-fold CV
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        ridge = Ridge(alpha=1.0, random_state=42)
        
        scores = cross_val_score(ridge, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
        cv_mse = -np.mean(scores)
        cv_std = np.std(scores)
        
        # Relative time (simplified)
        relative_time = k  # More folds = more time
        
        print(f"{k}\t{len(X_scaled)//k}\t\t{cv_mse:.4f}\t\t{cv_std:.4f}\t\t{relative_time:.1f}x")
    
    print("\nKey Observations:")
    print("- Larger k: less bias, more variance in estimate")
    print("- Smaller k: more bias, less variance in estimate")
    print("- k=10 is often a good choice")
    print("- Computational cost increases with k")

def demonstrate_leave_one_out_cv():
    """
    Demonstrates leave-one-out cross-validation (k = n).
    """
    print("\n2.3 Leave-One-Out Cross Validation")
    print("-" * 50)
    
    # Use smaller dataset for LOOCV (computationally expensive)
    X, y, true_coef = make_regression(n_samples=50, n_features=5, n_informative=3, 
                                     noise=0.5, random_state=42, coef=True)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset size: {X.shape[0]} samples (small for LOOCV)")
    
    # Compare different CV methods
    cv_methods = [
        ('Hold-out (70-30)', None),
        ('5-Fold', 5),
        ('10-Fold', 10),
        ('Leave-One-Out', len(X_scaled))
    ]
    
    print("\nCross-Validation Method Comparison")
    print("Method\t\t\tCV MSE\t\tCV Std\t\tBias\t\tVariance")
    print("-" * 70)
    
    for method_name, k in cv_methods:
        if k is None:
            # Hold-out
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_train, y_train)
            y_val_pred = ridge.predict(X_val)
            cv_mse = mean_squared_error(y_val, y_val_pred)
            cv_std = 0.0  # Single estimate
        else:
            # k-fold CV
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            ridge = Ridge(alpha=1.0, random_state=42)
            scores = cross_val_score(ridge, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
            cv_mse = -np.mean(scores)
            cv_std = np.std(scores)
        
        # Simplified bias/variance estimates
        bias_est = cv_mse  # Simplified
        variance_est = cv_std**2  # Simplified
        
        print(f"{method_name}\t{cv_mse:.4f}\t\t{cv_std:.4f}\t\t{bias_est:.4f}\t\t{variance_est:.4f}")
    
    print("\nKey Observations:")
    print("- LOOCV has the least bias but highest variance")
    print("- Hold-out is fast but has high bias")
    print("- k-fold CV provides a good balance")

# ============================================================================
# SECTION 3: BAYESIAN METHODS
# ============================================================================

print("\n" + "="*60)
print("SECTION 3: BAYESIAN METHODS")
print("="*60)

def demonstrate_mle():
    """
    Demonstrates Maximum Likelihood Estimation (MLE).
    """
    print("\n3.1 Maximum Likelihood Estimation (MLE)")
    print("-" * 50)
    
    # Generate data for logistic regression
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, 
                              n_redundant=1, random_state=42)
    
    print(f"Generated classification dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Fit MLE (no regularization)
    mle_model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000, random_state=42)
    mle_model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = mle_model.predict(X_train)
    y_test_pred = mle_model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"\nMLE Results:")
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Number of parameters: {len(mle_model.coef_[0])}")
    print(f"Parameter L2 norm: {np.sqrt(np.sum(mle_model.coef_[0]**2)):.4f}")
    
    print(f"\nMLE coefficients:")
    for i, coef in enumerate(mle_model.coef_[0]):
        print(f"θ_{i}: {coef:.4f}")
    
    print("\nKey Observations:")
    print("- MLE finds parameters that maximize the likelihood")
    print("- No regularization means no overfitting control")
    print("- Can lead to large parameter values")

def demonstrate_map_estimation():
    """
    Demonstrates Maximum A Posteriori (MAP) estimation.
    """
    print("\n3.2 Maximum A Posteriori (MAP) Estimation")
    print("-" * 50)
    
    # Use the same data as MLE
    X, y = make_classification(n_samples=200, n_features=5, n_informative=3, 
                              n_redundant=1, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Try different regularization strengths (MAP with different priors)
    C_values = [0.1, 1.0, 10.0, 100.0]  # C = 1/λ
    
    print("C (1/λ)\tλ\t\tTrain Acc\tTest Acc\tL2 Norm\t\tL1 Norm")
    print("-" * 70)
    
    for C in C_values:
        lambda_val = 1.0 / C
        
        # MAP with L2 regularization (Gaussian prior)
        map_model = LogisticRegression(penalty='l2', C=C, random_state=42)
        map_model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = map_model.predict(X_train)
        y_test_pred = map_model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # Parameter norms
        l2_norm = np.sqrt(np.sum(map_model.coef_[0]**2))
        l1_norm = np.sum(np.abs(map_model.coef_[0]))
        
        print(f"{C}\t\t{lambda_val:.3f}\t\t{train_acc:.4f}\t\t{test_acc:.4f}\t\t{l2_norm:.4f}\t\t{l1_norm:.4f}")
    
    print("\nKey Observations:")
    print("- MAP adds prior knowledge through regularization")
    print("- Smaller C (larger λ) = stronger regularization")
    print("- MAP often generalizes better than MLE")

def demonstrate_bayesian_inference():
    """
    Demonstrates full Bayesian inference with uncertainty quantification.
    """
    print("\n3.3 Full Bayesian Inference")
    print("-" * 50)
    
    # Generate regression data
    X, y, true_coef = make_regression(n_samples=100, n_features=3, n_informative=2, 
                                     noise=0.5, random_state=42, coef=True)
    
    print(f"Generated regression dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True coefficients: {true_coef}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Bayesian Ridge Regression
    bayesian_model = BayesianRidge(compute_score=True)
    bayesian_model.fit(X_train, y_train)
    
    # Predictions with uncertainty
    y_pred, y_std = bayesian_model.predict(X_test, return_std=True)
    
    print(f"\nBayesian Ridge Results:")
    print(f"Test MSE: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"Average prediction uncertainty: {np.mean(y_std):.4f}")
    
    print(f"\nPosterior mean coefficients:")
    for i, coef in enumerate(bayesian_model.coef_):
        print(f"θ_{i}: {coef:.4f}")
    
    print(f"\nPosterior coefficient uncertainties:")
    for i, std in enumerate(np.sqrt(np.diag(bayesian_model.sigma_))):
        print(f"σ_{i}: {std:.4f}")
    
    # Compare with true coefficients
    print(f"\nComparison with true coefficients:")
    print("Feature\tTrue θ\tPosterior θ\tUncertainty\tCovered?")
    print("-" * 60)
    
    for i in range(len(true_coef)):
        true_val = true_coef[i]
        post_mean = bayesian_model.coef_[i]
        post_std = np.sqrt(bayesian_model.sigma_[i, i])
        
        # Check if true value is within 2 standard deviations
        covered = abs(true_val - post_mean) <= 2 * post_std
        
        print(f"{i}\t{true_val:.4f}\t{post_mean:.4f}\t\t{post_std:.4f}\t\t{covered}")
    
    # Visualize predictions with uncertainty
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Bayesian Predictions vs True Values')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.errorbar(range(len(y_test)), y_pred, yerr=y_std, fmt='o', alpha=0.6)
    plt.plot(y_test, 'r-', label='True Values', linewidth=2)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Predicted Value')
    plt.title('Predictions with Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Coefficient comparison
    x_pos = np.arange(len(true_coef))
    width = 0.35
    
    plt.bar(x_pos - width/2, true_coef, width, label='True Coefficients', alpha=0.7)
    plt.bar(x_pos + width/2, bayesian_model.coef_, width, label='Posterior Mean', alpha=0.7)
    plt.errorbar(x_pos + width/2, bayesian_model.coef_, 
                yerr=np.sqrt(np.diag(bayesian_model.sigma_)), fmt='none', color='black', capsize=5)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('True vs Posterior Coefficients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Uncertainty distribution
    plt.hist(y_std, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Uncertainties')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_inference_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nKey Observations:")
    print("- Bayesian inference provides uncertainty quantification")
    print("- Posterior means are similar to MAP estimates")
    print("- Uncertainty estimates help assess prediction reliability")
    print("- True coefficients are typically within uncertainty bounds")

# ============================================================================
# SECTION 4: PRACTICAL GUIDELINES
# ============================================================================

print("\n" + "="*60)
print("SECTION 4: PRACTICAL GUIDELINES")
print("="*60)

def demonstrate_practical_guidelines():
    """
    Demonstrates practical guidelines for model selection and validation.
    """
    print("\n4.1 Model Selection Best Practices")
    print("-" * 50)
    
    # Generate data
    X, y, true_coef = make_regression(n_samples=500, n_features=20, n_informative=10, 
                                     noise=0.5, random_state=42, coef=True)
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Informative features: {np.sum(true_coef != 0)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Compare different approaches
    approaches = [
        ('Linear Regression (No Reg)', LinearRegression()),
        ('Ridge Regression (L2)', Ridge(alpha=1.0, random_state=42)),
        ('LASSO (L1)', LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42)),
        ('Elastic Net', LogisticRegression(penalty='elasticnet', solver='saga', C=1.0, l1_ratio=0.5, random_state=42))
    ]
    
    print("\nModel Comparison with Cross-Validation")
    print("Model\t\t\tCV MSE\t\tCV Std\t\tParams\t\tSparsity")
    print("-" * 70)
    
    for name, model in approaches:
        # Perform 5-fold CV
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        cv_mse = -np.mean(scores)
        cv_std = np.std(scores)
        
        # Fit on full data to get parameters
        model.fit(X_scaled, y)
        
        if hasattr(model, 'coef_'):
            n_params = len(model.coef_[0]) if model.coef_.ndim > 1 else len(model.coef_)
            sparsity = np.sum(model.coef_[0] == 0) / len(model.coef_[0]) * 100 if model.coef_.ndim > 1 else np.sum(model.coef_ == 0) / len(model.coef_) * 100
        else:
            n_params = len(model.coef_)
            sparsity = np.sum(model.coef_ == 0) / len(model.coef_) * 100
        
        print(f"{name}\t{cv_mse:.4f}\t\t{cv_std:.4f}\t\t{n_params}\t\t{sparsity:.1f}%")
    
    print("\n4.2 Hyperparameter Tuning")
    print("-" * 50)
    
    # Grid search for Ridge regression
    ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge_grid = GridSearchCV(Ridge(random_state=42), ridge_params, cv=5, scoring='neg_mean_squared_error')
    ridge_grid.fit(X_scaled, y)
    
    print(f"Best Ridge α: {ridge_grid.best_params_['alpha']}")
    print(f"Best CV score: {-ridge_grid.best_score_:.4f}")
    
    # Grid search for LASSO
    lasso_params = {'C': [0.1, 1.0, 10.0, 100.0]}
    lasso_grid = GridSearchCV(LogisticRegression(penalty='l1', solver='liblinear', random_state=42), 
                             lasso_params, cv=5, scoring='neg_mean_squared_error')
    lasso_grid.fit(X_scaled, y)
    
    print(f"Best LASSO C: {lasso_grid.best_params_['C']}")
    print(f"Best CV score: {-lasso_grid.best_score_:.4f}")
    
    print("\nKey Guidelines:")
    print("1. Always use cross-validation for model selection")
    print("2. Never use the test set for model selection")
    print("3. Choose validation strategy based on dataset size")
    print("4. Use grid search for hyperparameter tuning")
    print("5. Consider both performance and interpretability")
    print("6. Document your choices for reproducibility")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_model_selection_problem()
    demonstrate_hold_out_validation()
    demonstrate_k_fold_cross_validation()
    demonstrate_leave_one_out_cv()
    demonstrate_mle()
    demonstrate_map_estimation()
    demonstrate_bayesian_inference()
    demonstrate_practical_guidelines()
    
    print("\n" + "="*80)
    print("MODEL SELECTION AND BAYESIAN METHODS COMPLETED")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Model selection requires balancing bias and variance")
    print("2. Cross-validation provides reliable performance estimates")
    print("3. Different CV methods have different trade-offs")
    print("4. MLE finds best fit, MAP adds regularization")
    print("5. Bayesian methods provide uncertainty quantification")
    print("6. Always use validation sets, never test sets for selection")
    print("7. Document your choices and assumptions") 
"""
Regularization Examples and Implementations
===========================================

This file demonstrates all regularization concepts covered in 01_regularization.md
including L1, L2, Elastic Net regularization, parameter selection, and practical applications.

Key Concepts Covered:
- Mathematical framework of regularization
- L2 regularization (Ridge regression)
- L1 regularization (LASSO)
- Elastic Net regularization
- Parameter selection via cross-validation
- Scaling and preprocessing considerations
- Practical guidelines and best practices
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression, make_classification
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("REGULARIZATION EXAMPLES AND IMPLEMENTATIONS")
print("=" * 80)

# ============================================================================
# SECTION 1: MATHEMATICAL FOUNDATION
# ============================================================================

print("\n" + "="*60)
print("SECTION 1: MATHEMATICAL FOUNDATION")
print("="*60)

def demonstrate_regularization_framework():
    """
    Demonstrates the core regularization equation: J_λ(θ) = J(θ) + λR(θ)
    
    This function shows how regularization works mathematically and how
    different regularizers affect the parameter values.
    """
    print("\n1.1 Regularization Framework Demonstration")
    print("-" * 50)
    
    # Example parameter vector θ
    theta = np.array([1.0, -2.0, 0.0, 3.0, -1.5])
    print(f"Original parameter vector θ: {theta}")
    
    # Example original loss J(θ)
    J_theta = 2.5
    print(f"Original loss J(θ): {J_theta}")
    
    # Different regularization parameters λ
    lambda_values = [0.0, 0.1, 1.0, 10.0]
    
    print("\nRegularization Effects:")
    print("λ\tL2 Regularizer\tL1 Regularizer\tRegularized Loss (L2)\tRegularized Loss (L1)")
    print("-" * 90)
    
    for lambda_val in lambda_values:
        # L2 regularization: R(θ) = 0.5 * ||θ||₂²
        R_theta_l2 = 0.5 * np.sum(theta ** 2)
        
        # L1 regularization: R(θ) = ||θ||₁
        R_theta_l1 = np.sum(np.abs(theta))
        
        # Regularized losses
        J_lambda_l2 = J_theta + lambda_val * R_theta_l2
        J_lambda_l1 = J_theta + lambda_val * R_theta_l1
        
        print(f"{lambda_val}\t{R_theta_l2:.3f}\t\t{R_theta_l1:.3f}\t\t{J_lambda_l2:.3f}\t\t\t{J_lambda_l1:.3f}")
    
    print("\nKey Observations:")
    print("- When λ = 0: No regularization (original loss)")
    print("- As λ increases: Regularization penalty becomes more important")
    print("- L2 penalizes large values quadratically")
    print("- L1 penalizes large values linearly")

# ============================================================================
# SECTION 2: L2 REGULARIZATION (RIDGE REGRESSION)
# ============================================================================

print("\n" + "="*60)
print("SECTION 2: L2 REGULARIZATION (RIDGE REGRESSION)")
print("="*60)

def demonstrate_l2_regularization():
    """
    Demonstrates L2 regularization (Ridge regression) with practical examples.
    
    L2 regularization adds a penalty term λ||θ||₂² to the loss function,
    encouraging all parameters to be small but not necessarily zero.
    """
    print("\n2.1 L2 Regularization Mathematical Properties")
    print("-" * 50)
    
    # Generate synthetic data
    X, y, true_coef = make_regression(n_samples=100, n_features=10, n_informative=5, 
                                     noise=0.5, random_state=42, coef=True)
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True coefficients: {true_coef}")
    
    # Standardize features (IMPORTANT for regularization!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    print("\n2.2 L2 Regularization Effects on Parameters")
    print("-" * 50)
    
    # Try different alpha values (λ in our notation)
    alpha_values = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    print("α (lambda)\t||θ||₂²\t\tMax |θ|\t\tMin |θ|\t\tTest MSE")
    print("-" * 80)
    
    for alpha in alpha_values:
        # Fit Ridge regression
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_train, y_train)
        
        # Calculate metrics
        l2_norm_squared = np.sum(ridge.coef_**2)
        max_coef = np.max(np.abs(ridge.coef_))
        min_coef = np.min(np.abs(ridge.coef_))
        test_mse = mean_squared_error(y_test, ridge.predict(X_test))
        
        print(f"{alpha}\t\t{l2_norm_squared:.4f}\t\t{max_coef:.4f}\t\t{min_coef:.4f}\t\t{test_mse:.4f}")
    
    print("\nKey Observations:")
    print("- As α increases, ||θ||₂² decreases (parameters become smaller)")
    print("- All parameters shrink towards zero, but none become exactly zero")
    print("- There's an optimal α that balances bias and variance")

def visualize_l2_regularization():
    """
    Visualizes the effect of L2 regularization on parameter values.
    """
    print("\n2.3 L2 Regularization Visualization")
    print("-" * 50)
    
    # Generate data
    X, y, true_coef = make_regression(n_samples=100, n_features=5, noise=0.5, 
                                     random_state=42, coef=True)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try many alpha values
    alpha_values = np.logspace(-3, 3, 50)
    coef_paths = []
    
    for alpha in alpha_values:
        ridge = Ridge(alpha=alpha, random_state=42)
        ridge.fit(X_scaled, y)
        coef_paths.append(ridge.coef_)
    
    coef_paths = np.array(coef_paths)
    
    # Plot coefficient paths
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for i in range(coef_paths.shape[1]):
        plt.semilogx(alpha_values, coef_paths[:, i], label=f'θ_{i}')
    plt.xlabel('α (Regularization Parameter)')
    plt.ylabel('Coefficient Value')
    plt.title('L2 Regularization: Coefficient Paths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    l2_norms = np.sqrt(np.sum(coef_paths**2, axis=1))
    plt.semilogx(alpha_values, l2_norms)
    plt.xlabel('α (Regularization Parameter)')
    plt.ylabel('L2 Norm of Coefficients')
    plt.title('L2 Regularization: Parameter Norm')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Cross-validation to find optimal alpha
    cv_scores = []
    for alpha in alpha_values:
        ridge = Ridge(alpha=alpha, random_state=42)
        scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-np.mean(scores))
    
    plt.semilogx(alpha_values, cv_scores)
    plt.xlabel('α (Regularization Parameter)')
    plt.ylabel('Cross-Validation MSE')
    plt.title('L2 Regularization: CV Performance')
    plt.grid(True, alpha=0.3)
    
    # Mark optimal alpha
    optimal_alpha = alpha_values[np.argmin(cv_scores)]
    plt.axvline(optimal_alpha, color='red', linestyle='--', 
                label=f'Optimal α = {optimal_alpha:.3f}')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Compare with true coefficients
    ridge_optimal = Ridge(alpha=optimal_alpha, random_state=42)
    ridge_optimal.fit(X_scaled, y)
    
    x_pos = np.arange(len(true_coef))
    width = 0.35
    
    plt.bar(x_pos - width/2, true_coef, width, label='True Coefficients', alpha=0.7)
    plt.bar(x_pos + width/2, ridge_optimal.coef_, width, label='L2 Regularized', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('True vs L2 Regularized Coefficients')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('l2_regularization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Optimal α found via cross-validation: {optimal_alpha:.3f}")

# ============================================================================
# SECTION 3: L1 REGULARIZATION (LASSO)
# ============================================================================

print("\n" + "="*60)
print("SECTION 3: L1 REGULARIZATION (LASSO)")
print("="*60)

def demonstrate_l1_regularization():
    """
    Demonstrates L1 regularization (LASSO) with practical examples.
    
    L1 regularization adds a penalty term λ||θ||₁ to the loss function,
    encouraging sparsity by setting some parameters exactly to zero.
    """
    print("\n3.1 L1 Regularization and Sparsity")
    print("-" * 50)
    
    # Generate data with some irrelevant features
    X, y, true_coef = make_regression(n_samples=100, n_features=10, n_informative=3, 
                                     noise=0.5, random_state=42, coef=True)
    
    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True coefficients (only first 3 are informative): {true_coef}")
    print(f"Number of non-zero true coefficients: {np.sum(true_coef != 0)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    print("\n3.2 L1 Regularization Effects on Sparsity")
    print("-" * 50)
    
    # Try different alpha values
    alpha_values = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    print("α (lambda)\t||θ||₁\t\tNon-zero θ\tSparsity %\tTest MSE")
    print("-" * 70)
    
    for alpha in alpha_values:
        # Fit LASSO
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        lasso.fit(X_train, y_train)
        
        # Calculate metrics
        l1_norm = np.sum(np.abs(lasso.coef_))
        non_zero_coef = np.sum(lasso.coef_ != 0)
        sparsity = (len(lasso.coef_) - non_zero_coef) / len(lasso.coef_) * 100
        test_mse = mean_squared_error(y_test, lasso.predict(X_test))
        
        print(f"{alpha}\t\t{l1_norm:.4f}\t\t{non_zero_coef}\t\t{sparsity:.1f}%\t\t{test_mse:.4f}")
    
    print("\nKey Observations:")
    print("- As α increases, more parameters become exactly zero")
    print("- L1 regularization performs automatic feature selection")
    print("- Sparsity increases with stronger regularization")

def visualize_l1_regularization():
    """
    Visualizes the effect of L1 regularization on parameter values and sparsity.
    """
    print("\n3.3 L1 Regularization Visualization")
    print("-" * 50)
    
    # Generate data
    X, y, true_coef = make_regression(n_samples=100, n_features=5, noise=0.5, 
                                     random_state=42, coef=True)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try many alpha values
    alpha_values = np.logspace(-3, 2, 50)
    coef_paths = []
    
    for alpha in alpha_values:
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        coef_paths.append(lasso.coef_)
    
    coef_paths = np.array(coef_paths)
    
    # Plot coefficient paths
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    for i in range(coef_paths.shape[1]):
        plt.semilogx(alpha_values, coef_paths[:, i], label=f'θ_{i}')
    plt.xlabel('α (Regularization Parameter)')
    plt.ylabel('Coefficient Value')
    plt.title('L1 Regularization: Coefficient Paths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    l1_norms = np.sum(np.abs(coef_paths), axis=1)
    plt.semilogx(alpha_values, l1_norms)
    plt.xlabel('α (Regularization Parameter)')
    plt.ylabel('L1 Norm of Coefficients')
    plt.title('L1 Regularization: Parameter Norm')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    # Count non-zero coefficients
    non_zero_counts = np.sum(coef_paths != 0, axis=1)
    plt.semilogx(alpha_values, non_zero_counts)
    plt.xlabel('α (Regularization Parameter)')
    plt.ylabel('Number of Non-zero Coefficients')
    plt.title('L1 Regularization: Sparsity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Cross-validation to find optimal alpha
    cv_scores = []
    for alpha in alpha_values:
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=2000)
        scores = cross_val_score(lasso, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        cv_scores.append(-np.mean(scores))
    
    plt.semilogx(alpha_values, cv_scores)
    plt.xlabel('α (Regularization Parameter)')
    plt.ylabel('Cross-Validation MSE')
    plt.title('L1 Regularization: CV Performance')
    plt.grid(True, alpha=0.3)
    
    # Mark optimal alpha
    optimal_alpha = alpha_values[np.argmin(cv_scores)]
    plt.axvline(optimal_alpha, color='red', linestyle='--', 
                label=f'Optimal α = {optimal_alpha:.3f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('l1_regularization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Optimal α found via cross-validation: {optimal_alpha:.3f}")

# ============================================================================
# SECTION 4: ELASTIC NET REGULARIZATION
# ============================================================================

print("\n" + "="*60)
print("SECTION 4: ELASTIC NET REGULARIZATION")
print("="*60)

def demonstrate_elastic_net():
    """
    Demonstrates Elastic Net regularization which combines L1 and L2 penalties.
    
    Elastic Net: R(θ) = α * ρ * ||θ||₁ + α * (1-ρ) * 0.5 * ||θ||₂²
    where α is the regularization strength and ρ controls the L1/L2 balance.
    """
    print("\n4.1 Elastic Net Regularization")
    print("-" * 50)
    
    # Generate data with correlated features
    X, y, true_coef = make_regression(n_samples=100, n_features=10, n_informative=5, 
                                     noise=0.5, random_state=42, coef=True)
    
    # Add correlation between features
    X[:, 1] = X[:, 0] + 0.1 * np.random.randn(X.shape[0])
    X[:, 2] = X[:, 0] - 0.1 * np.random.randn(X.shape[0])
    
    print(f"Generated dataset with correlated features: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    print("\n4.2 Elastic Net Parameter Effects")
    print("-" * 50)
    
    # Try different l1_ratio values (ρ in our notation)
    l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    alpha = 0.1
    
    print("ρ (l1_ratio)\tL1 Component\tL2 Component\tNon-zero θ\tTest MSE")
    print("-" * 70)
    
    for l1_ratio in l1_ratios:
        # Fit Elastic Net
        elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
        elastic.fit(X_train, y_train)
        
        # Calculate metrics
        l1_component = alpha * l1_ratio * np.sum(np.abs(elastic.coef_))
        l2_component = alpha * (1 - l1_ratio) * 0.5 * np.sum(elastic.coef_**2)
        non_zero_coef = np.sum(elastic.coef_ != 0)
        test_mse = mean_squared_error(y_test, elastic.predict(X_test))
        
        print(f"{l1_ratio}\t\t{l1_component:.4f}\t\t{l2_component:.4f}\t\t{non_zero_coef}\t\t{test_mse:.4f}")
    
    print("\nKey Observations:")
    print("- ρ = 0: Pure L2 regularization (Ridge)")
    print("- ρ = 1: Pure L1 regularization (LASSO)")
    print("- 0 < ρ < 1: Combination of both")
    print("- Elastic Net handles correlated features better than LASSO")

# ============================================================================
# SECTION 5: PRACTICAL CONSIDERATIONS
# ============================================================================

print("\n" + "="*60)
print("SECTION 5: PRACTICAL CONSIDERATIONS")
print("="*60)

def demonstrate_scaling_importance():
    """
    Demonstrates why feature scaling is crucial for regularization.
    """
    print("\n5.1 Importance of Feature Scaling")
    print("-" * 50)
    
    # Generate data with different scales
    np.random.seed(42)
    n_samples, n_features = 100, 3
    
    # Features with different scales
    X = np.column_stack([
        np.random.randn(n_samples) * 1.0,      # Scale ~1
        np.random.randn(n_samples) * 100.0,    # Scale ~100
        np.random.randn(n_samples) * 0.01      # Scale ~0.01
    ])
    
    # True coefficients
    true_coef = np.array([1.0, 1.0, 1.0])
    y = X @ true_coef + np.random.randn(n_samples) * 0.1
    
    print("Feature scales before standardization:")
    print(f"Feature 1: mean={X[:, 0].mean():.3f}, std={X[:, 0].std():.3f}")
    print(f"Feature 2: mean={X[:, 1].mean():.3f}, std={X[:, 1].std():.3f}")
    print(f"Feature 3: mean={X[:, 2].mean():.3f}, std={X[:, 2].std():.3f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("\n5.2 Regularization Without Scaling")
    print("-" * 50)
    
    # Try regularization without scaling
    alpha = 1.0
    ridge_unscaled = Ridge(alpha=alpha, random_state=42)
    ridge_unscaled.fit(X_train, y_train)
    
    print("Coefficients without scaling:")
    for i, coef in enumerate(ridge_unscaled.coef_):
        print(f"θ_{i}: {coef:.6f}")
    
    print("\n5.3 Regularization With Scaling")
    print("-" * 50)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge_scaled = Ridge(alpha=alpha, random_state=42)
    ridge_scaled.fit(X_train_scaled, y_train)
    
    print("Coefficients with scaling:")
    for i, coef in enumerate(ridge_scaled.coef_):
        print(f"θ_{i}: {coef:.6f}")
    
    print("\nKey Insight:")
    print("- Without scaling, regularization penalizes features with larger scales more heavily")
    print("- With scaling, all features are penalized equally")
    print("- Always standardize features before applying regularization!")

def demonstrate_parameter_selection():
    """
    Demonstrates systematic parameter selection using cross-validation.
    """
    print("\n5.4 Systematic Parameter Selection")
    print("-" * 50)
    
    # Generate data
    X, y, true_coef = make_regression(n_samples=200, n_features=10, n_informative=5, 
                                     noise=0.5, random_state=42, coef=True)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Grid Search for Optimal Regularization Parameters")
    print("-" * 60)
    
    # Grid search for Ridge regression
    print("\nRidge Regression (L2):")
    ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge_grid = GridSearchCV(Ridge(random_state=42), ridge_params, cv=5, scoring='neg_mean_squared_error')
    ridge_grid.fit(X_scaled, y)
    
    print(f"Best α: {ridge_grid.best_params_['alpha']}")
    print(f"Best CV score: {-ridge_grid.best_score_:.4f}")
    
    # Grid search for LASSO
    print("\nLASSO (L1):")
    lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
    lasso_grid = GridSearchCV(Lasso(random_state=42, max_iter=2000), lasso_params, 
                             cv=5, scoring='neg_mean_squared_error')
    lasso_grid.fit(X_scaled, y)
    
    print(f"Best α: {lasso_grid.best_params_['alpha']}")
    print(f"Best CV score: {-lasso_grid.best_score_:.4f}")
    
    # Grid search for Elastic Net
    print("\nElastic Net:")
    elastic_params = {
        'alpha': [0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
    elastic_grid = GridSearchCV(ElasticNet(random_state=42, max_iter=2000), elastic_params, 
                               cv=5, scoring='neg_mean_squared_error')
    elastic_grid.fit(X_scaled, y)
    
    print(f"Best α: {elastic_grid.best_params_['alpha']}")
    print(f"Best l1_ratio: {elastic_grid.best_params_['l1_ratio']}")
    print(f"Best CV score: {-elastic_grid.best_score_:.4f}")
    
    # Compare final models
    print("\nModel Comparison:")
    print("-" * 30)
    
    # Fit best models
    best_ridge = ridge_grid.best_estimator_
    best_lasso = lasso_grid.best_estimator_
    best_elastic = elastic_grid.best_estimator_
    
    # Evaluate on test set
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    best_ridge.fit(X_train_final, y_train_final)
    best_lasso.fit(X_train_final, y_train_final)
    best_elastic.fit(X_train_final, y_train_final)
    
    ridge_mse = mean_squared_error(y_test_final, best_ridge.predict(X_test_final))
    lasso_mse = mean_squared_error(y_test_final, best_lasso.predict(X_test_final))
    elastic_mse = mean_squared_error(y_test_final, best_elastic.predict(X_test_final))
    
    print(f"Ridge Test MSE: {ridge_mse:.4f}")
    print(f"LASSO Test MSE: {lasso_mse:.4f}")
    print(f"Elastic Net Test MSE: {elastic_mse:.4f}")
    
    # Compare sparsity
    ridge_sparsity = np.sum(best_ridge.coef_ == 0) / len(best_ridge.coef_) * 100
    lasso_sparsity = np.sum(best_lasso.coef_ == 0) / len(best_lasso.coef_) * 100
    elastic_sparsity = np.sum(best_elastic.coef_ == 0) / len(best_elastic.coef_) * 100
    
    print(f"\nSparsity (% zero coefficients):")
    print(f"Ridge: {ridge_sparsity:.1f}%")
    print(f"LASSO: {lasso_sparsity:.1f}%")
    print(f"Elastic Net: {elastic_sparsity:.1f}%")

# ============================================================================
# SECTION 6: IMPLICIT REGULARIZATION
# ============================================================================

print("\n" + "="*60)
print("SECTION 6: IMPLICIT REGULARIZATION")
print("="*60)

def demonstrate_implicit_regularization():
    """
    Demonstrates how optimization choices can act as implicit regularization.
    """
    print("\n6.1 Implicit Regularization Effects")
    print("-" * 50)
    
    # Generate data
    X, y, true_coef = make_regression(n_samples=100, n_features=5, noise=0.5, 
                                     random_state=42, coef=True)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Effect of Different Learning Rates on Parameter Norms")
    print("-" * 60)
    
    # Simulate different optimization trajectories
    from sklearn.linear_model import SGDRegressor
    
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    
    print("Learning Rate\tL2 Norm\t\tL1 Norm\t\tNon-zero θ")
    print("-" * 50)
    
    for lr in learning_rates:
        sgd = SGDRegressor(learning_rate='constant', eta0=lr, max_iter=1000, 
                          random_state=42, tol=1e-6)
        sgd.fit(X_scaled, y)
        
        l2_norm = np.sqrt(np.sum(sgd.coef_**2))
        l1_norm = np.sum(np.abs(sgd.coef_))
        non_zero = np.sum(sgd.coef_ != 0)
        
        print(f"{lr}\t\t{l2_norm:.4f}\t\t{l1_norm:.4f}\t\t{non_zero}")
    
    print("\nKey Insight:")
    print("- Different optimization settings can lead to different solutions")
    print("- This is especially important in deep learning")
    print("- Understanding implicit regularization helps in model design")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_regularization_framework()
    demonstrate_l2_regularization()
    visualize_l2_regularization()
    demonstrate_l1_regularization()
    visualize_l1_regularization()
    demonstrate_elastic_net()
    demonstrate_scaling_importance()
    demonstrate_parameter_selection()
    demonstrate_implicit_regularization()
    
    print("\n" + "="*80)
    print("REGULARIZATION EXAMPLES COMPLETED")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Regularization helps prevent overfitting by penalizing model complexity")
    print("2. L2 regularization encourages small but non-zero parameters")
    print("3. L1 regularization encourages sparsity and feature selection")
    print("4. Elastic Net combines the benefits of both L1 and L2")
    print("5. Always standardize features before applying regularization")
    print("6. Use cross-validation to select optimal regularization parameters")
    print("7. Consider both explicit and implicit regularization effects") 
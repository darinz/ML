"""
Model Selection Demonstration

This module demonstrates various model selection techniques including cross-validation,
Bayesian methods, and parameter selection strategies in machine learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, StratifiedKFold, TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
from scipy.optimize import minimize
from scipy.stats import beta, multivariate_normal
import torch
import torch.nn as nn


def demonstrate_naive_model_selection():
    """Demonstrate why naive model selection fails"""
    
    np.random.seed(42)
    n_samples = 100
    
    # Generate data with some noise
    X = np.random.uniform(-3, 3, n_samples).reshape(-1, 1)
    true_function = 0.5 * X**2 + 0.3 * X + 1.0
    noise = 0.2 * np.random.randn(n_samples, 1)
    y = true_function + noise
    
    # Try different polynomial degrees
    degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    train_errors = []
    test_errors = []
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Calculate errors
        train_pred = model.predict(X_train_poly)
        test_pred = model.predict(X_test_poly)
        
        train_error = mean_squared_error(y_train, train_pred)
        test_error = mean_squared_error(y_test, test_pred)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot errors
    plt.subplot(1, 3, 1)
    plt.plot(degrees, train_errors, 'bo-', label='Training Error', linewidth=2)
    plt.plot(degrees, test_errors, 'ro-', label='Test Error', linewidth=2)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('Naive Model Selection: Training vs Test Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show the problem
    plt.subplot(1, 3, 2)
    best_train_degree = degrees[np.argmin(train_errors)]
    best_test_degree = degrees[np.argmin(test_errors)]
    
    plt.bar(['Best Training', 'Best Test'], 
            [best_train_degree, best_test_degree], 
            color=['blue', 'red'], alpha=0.7)
    plt.ylabel('Polynomial Degree')
    plt.title('Best Model by Different Criteria')
    plt.grid(True, alpha=0.3)
    
    # Show overfitting
    plt.subplot(1, 3, 3)
    plt.plot(degrees, np.array(test_errors) - np.array(train_errors), 'purple', linewidth=2)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Generalization Gap')
    plt.title('Overfitting Measure')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Naive Model Selection Analysis:")
    print("Degree\tTrain Error\tTest Error\tGap")
    print("-" * 40)
    for i, degree in enumerate(degrees):
        gap = test_errors[i] - train_errors[i]
        print(f"{degree:6d}\t{train_errors[i]:10.4f}\t{test_errors[i]:10.4f}\t{gap:8.4f}")
    
    print(f"\nBest by training error: degree {best_train_degree}")
    print(f"Best by test error: degree {best_test_degree}")
    print(f"Overfitting starts at degree: {degrees[np.argmax(np.array(test_errors) - np.array(train_errors))]}")
    
    return degrees, train_errors, test_errors


def demonstrate_holdout_validation():
    """Demonstrate hold-out cross-validation"""
    
    np.random.seed(42)
    n_samples = 200
    
    # Generate data
    X = np.random.randn(n_samples, 3)
    true_weights = np.array([1.5, -0.8, 0.3])
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Different models to compare
    models = {
        'Linear': LinearRegression(),
        'Ridge (α=0.1)': Ridge(alpha=0.1),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Lasso (α=1.0)': Lasso(alpha=1.0)
    }
    
    # Hold-out validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    
    results = {}
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_error = mean_squared_error(y_train, train_pred)
        val_error = mean_squared_error(y_val, val_pred)
        
        results[name] = {
            'train_error': train_error,
            'val_error': val_error,
            'model': model
        }
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot errors
    plt.subplot(1, 3, 1)
    names = list(results.keys())
    train_errors = [results[name]['train_error'] for name in names]
    val_errors = [results[name]['val_error'] for name in names]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    plt.bar(x_pos - width/2, train_errors, width, label='Training Error', alpha=0.7)
    plt.bar(x_pos + width/2, val_errors, width, label='Validation Error', alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('Mean Squared Error')
    plt.title('Hold-out Validation Results')
    plt.xticks(x_pos, names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show best model
    plt.subplot(1, 3, 2)
    best_model_name = min(results.keys(), key=lambda x: results[x]['val_error'])
    best_val_error = results[best_model_name]['val_error']
    
    plt.bar(names, val_errors, alpha=0.7)
    plt.bar(best_model_name, best_val_error, color='red', alpha=0.7, label='Best Model')
    plt.xlabel('Model')
    plt.ylabel('Validation Error')
    plt.title('Best Model Selection')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show generalization gap
    plt.subplot(1, 3, 3)
    gaps = [results[name]['val_error'] - results[name]['train_error'] for name in names]
    plt.bar(names, gaps, alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('Generalization Gap')
    plt.title('Overfitting Measure')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Hold-out Validation Results:")
    print("Model\t\t\tTrain Error\tVal Error\tGap")
    print("-" * 60)
    for name in names:
        gap = results[name]['val_error'] - results[name]['train_error']
        print(f"{name:<20}\t{results[name]['train_error']:.4f}\t\t{results[name]['val_error']:.4f}\t\t{gap:.4f}")
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best validation error: {best_val_error:.4f}")
    
    return results, best_model_name


def demonstrate_kfold_cross_validation():
    """Demonstrate k-fold cross-validation"""
    
    np.random.seed(42)
    n_samples = 150
    
    # Generate data
    X = np.random.randn(n_samples, 4)
    true_weights = np.array([1.0, -0.5, 0.3, 0.0])  # Some zero coefficients
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Different regularization strengths
    alphas = [0.001, 0.01, 0.1, 1, 10, 100]
    
    # Compare different k values
    k_values = [3, 5, 10]
    results = {}
    
    for k in k_values:
        cv_scores = []
        for alpha in alphas:
            model = Ridge(alpha=alpha)
            scores = cross_val_score(model, X, y, cv=k, scoring='neg_mean_squared_error')
            mean_score = -scores.mean()  # Convert back to MSE
            cv_scores.append(mean_score)
        
        results[f'k={k}'] = cv_scores
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot CV scores for different k
    plt.subplot(1, 3, 1)
    for k, scores in results.items():
        plt.semilogx(alphas, scores, 'o-', label=k, linewidth=2)
    plt.xlabel('Regularization Parameter (α)')
    plt.ylabel('Cross-Validation MSE')
    plt.title('k-Fold Cross-Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show best alpha for each k
    plt.subplot(1, 3, 2)
    best_alphas = []
    best_scores = []
    for k, scores in results.items():
        best_idx = np.argmin(scores)
        best_alphas.append(alphas[best_idx])
        best_scores.append(scores[best_idx])
    
    k_labels = list(results.keys())
    plt.bar(k_labels, best_scores, alpha=0.7)
    plt.ylabel('Best CV MSE')
    plt.title('Best Performance by k')
    plt.grid(True, alpha=0.3)
    
    # Show computational cost
    plt.subplot(1, 3, 3)
    computational_cost = [k * len(alphas) for k in k_values]
    plt.bar(k_labels, computational_cost, alpha=0.7)
    plt.ylabel('Number of Model Fits')
    plt.title('Computational Cost')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("k-Fold Cross-Validation Analysis:")
    print("α\t\tk=3\t\tk=5\t\tk=10")
    print("-" * 50)
    for i, alpha in enumerate(alphas):
        scores = [results[f'k={k}'][i] for k in k_values]
        print(f"{alpha:8.3f}\t{scores[0]:8.4f}\t{scores[1]:8.4f}\t{scores[2]:8.4f}")
    
    print(f"\nBest α by k:")
    for k, alpha, score in zip(k_labels, best_alphas, best_scores):
        print(f"{k}: α={alpha}, MSE={score:.4f}")
    
    return results, best_alphas


def demonstrate_leave_one_out_cv():
    """Demonstrate leave-one-out cross-validation"""
    
    np.random.seed(42)
    n_samples = 30  # Small dataset for LOOCV
    
    # Generate data
    X = np.random.randn(n_samples, 2)
    true_weights = np.array([1.0, -0.5])
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Different polynomial degrees
    degrees = [1, 2, 3, 4, 5]
    
    # Compare LOOCV with k-fold
    loo_scores = []
    kfold_scores = []
    
    for degree in degrees:
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # LOOCV
        loo = LeaveOneOut()
        loo_cv_scores = cross_val_score(LinearRegression(), X_poly, y, cv=loo, scoring='neg_mean_squared_error')
        loo_scores.append(-loo_cv_scores.mean())
        
        # 5-fold CV
        kfold_cv_scores = cross_val_score(LinearRegression(), X_poly, y, cv=5, scoring='neg_mean_squared_error')
        kfold_scores.append(-kfold_cv_scores.mean())
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Compare LOOCV vs k-fold
    plt.subplot(1, 3, 1)
    plt.plot(degrees, loo_scores, 'bo-', label='LOOCV', linewidth=2)
    plt.plot(degrees, kfold_scores, 'ro-', label='5-fold CV', linewidth=2)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Cross-Validation MSE')
    plt.title('LOOCV vs k-fold Cross-Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show computational cost
    plt.subplot(1, 3, 2)
    loo_cost = [n_samples * len(degrees)]  # n fits for each degree
    kfold_cost = [5 * len(degrees)]  # 5 fits for each degree
    methods = ['LOOCV', '5-fold CV']
    costs = [loo_cost[0], kfold_cost[0]]
    
    plt.bar(methods, costs, alpha=0.7)
    plt.ylabel('Total Model Fits')
    plt.title('Computational Cost')
    plt.grid(True, alpha=0.3)
    
    # Show variance in estimates
    plt.subplot(1, 3, 3)
    # Simulate variance by running multiple times
    loo_variances = []
    kfold_variances = []
    
    for degree in degrees:
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        
        # Multiple LOOCV runs
        loo_runs = []
        kfold_runs = []
        for _ in range(10):
            # LOOCV
            loo_cv_scores = cross_val_score(LinearRegression(), X_poly, y, cv=LeaveOneOut(), scoring='neg_mean_squared_error')
            loo_runs.append(-loo_cv_scores.mean())
            
            # k-fold
            kfold_cv_scores = cross_val_score(LinearRegression(), X_poly, y, cv=5, scoring='neg_mean_squared_error')
            kfold_runs.append(-kfold_cv_scores.mean())
        
        loo_variances.append(np.var(loo_runs))
        kfold_variances.append(np.var(kfold_runs))
    
    plt.plot(degrees, loo_variances, 'bo-', label='LOOCV Variance', linewidth=2)
    plt.plot(degrees, kfold_variances, 'ro-', label='5-fold CV Variance', linewidth=2)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Variance of CV Estimate')
    plt.title('Estimate Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Leave-One-Out Cross-Validation Analysis:")
    print("Degree\tLOOCV MSE\t5-fold MSE")
    print("-" * 35)
    for i, degree in enumerate(degrees):
        print(f"{degree:6d}\t{loo_scores[i]:10.4f}\t{kfold_scores[i]:10.4f}")
    
    print(f"\nComputational cost:")
    print(f"LOOCV: {loo_cost[0]} model fits")
    print(f"5-fold CV: {kfold_cost[0]} model fits")
    
    return loo_scores, kfold_scores, degrees


def demonstrate_bayesian_coin_flipping():
    """Demonstrate Bayesian inference with coin flipping"""
    
    np.random.seed(42)
    
    # Prior: Beta(2, 2) - slightly favoring fair coin
    prior_alpha, prior_beta = 2, 2
    
    # Different datasets
    datasets = [
        (3, 10),   # 3 heads out of 10
        (7, 10),   # 7 heads out of 10
        (70, 100), # 70 heads out of 100
        (700, 1000) # 700 heads out of 1000
    ]
    
    results = {}
    
    for heads, total in datasets:
        # Posterior: Beta(prior_alpha + heads, prior_beta + tails)
        posterior_alpha = prior_alpha + heads
        posterior_beta = prior_beta + (total - heads)
        
        # Point estimates
        prior_mean = prior_alpha / (prior_alpha + prior_beta)
        posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
        posterior_std = np.sqrt((posterior_alpha * posterior_beta) / 
                               ((posterior_alpha + posterior_beta)**2 * (posterior_alpha + posterior_beta + 1)))
        
        # Credible intervals
        ci_lower = beta.ppf(0.025, posterior_alpha, posterior_beta)
        ci_upper = beta.ppf(0.975, posterior_alpha, posterior_beta)
        
        results[f'{heads}/{total}'] = {
            'posterior_alpha': posterior_alpha,
            'posterior_beta': posterior_beta,
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot posteriors
    theta = np.linspace(0, 1, 100)
    
    for i, (dataset, result) in enumerate(results.items()):
        plt.subplot(2, 2, i+1)
        
        # Prior
        prior = beta.pdf(theta, prior_alpha, prior_beta)
        plt.plot(theta, prior, 'b-', label='Prior: Beta(2, 2)', linewidth=2, alpha=0.7)
        
        # Posterior
        posterior = beta.pdf(theta, result['posterior_alpha'], result['posterior_beta'])
        plt.plot(theta, posterior, 'r-', label=f'Posterior: Beta({result["posterior_alpha"]}, {result["posterior_beta"]})', linewidth=2)
        
        # Credible interval
        plt.axvspan(result['ci_lower'], result['ci_upper'], alpha=0.3, color='red', label='95% Credible Interval')
        
        # Data point
        heads, total = dataset.split('/')
        plt.axvline(int(heads)/int(total), color='g', linestyle='--', alpha=0.7, label=f'Data: {dataset}')
        
        plt.xlabel('Probability of Heads')
        plt.ylabel('Density')
        plt.title(f'Dataset: {dataset} heads')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Bayesian Coin Flipping Analysis:")
    print("Dataset\tPosterior Mean\tStd\t95% Credible Interval")
    print("-" * 60)
    for dataset, result in results.items():
        print(f"{dataset:10}\t{result['posterior_mean']:.3f}\t\t{result['posterior_std']:.3f}\t"
              f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    
    return results


def demonstrate_map_estimation():
    """Demonstrate MAP estimation and its connection to regularization"""
    
    np.random.seed(42)
    n_samples, n_features = 100, 5
    
    # Generate data
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.0, -0.5, 0.3, 0.0, 0.0])  # Some zero coefficients
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    def map_estimation_with_gaussian_prior(X, y, prior_std=1.0, noise_std=1.0):
        """MAP estimation with Gaussian prior (equivalent to Ridge regression)"""
        def objective(theta):
            # Log-likelihood (assuming Gaussian noise)
            predictions = X @ theta
            residuals = y - predictions
            log_likelihood = -0.5 * np.sum(residuals**2) / (noise_std**2)
            
            # Log-prior (Gaussian)
            log_prior = -0.5 * np.sum(theta**2) / (prior_std**2)
            
            return -(log_likelihood + log_prior)  # Negative because we minimize
        
        # Optimize
        initial_theta = np.zeros(X.shape[1])
        result = minimize(objective, initial_theta)
        return result.x
    
    # Compare different prior strengths
    prior_stds = [0.1, 0.5, 1.0, 2.0, 10.0]
    map_estimates = []
    ridge_estimates = []
    
    for prior_std in prior_stds:
        # MAP estimation
        map_theta = map_estimation_with_gaussian_prior(X, y, prior_std=prior_std)
        map_estimates.append(map_theta)
        
        # Ridge regression (should be equivalent)
        ridge_alpha = 1.0 / (2 * prior_std**2)  # alpha = 1/(2*prior_std^2)
        ridge = Ridge(alpha=ridge_alpha)
        ridge.fit(X, y)
        ridge_estimates.append(ridge.coef_)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Compare MAP vs Ridge
    plt.subplot(1, 3, 1)
    for i, prior_std in enumerate(prior_stds):
        plt.plot(range(n_features), map_estimates[i], 'o-', label=f'MAP (σ={prior_std})', alpha=0.7)
    plt.plot(range(n_features), true_weights, 'k*', markersize=15, label='True Weights')
    plt.xlabel('Feature Index')
    plt.ylabel('Weight Value')
    plt.title('MAP Estimates vs True Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show MAP vs Ridge equivalence
    plt.subplot(1, 3, 2)
    differences = []
    for i, prior_std in enumerate(prior_stds):
        diff = np.linalg.norm(map_estimates[i] - ridge_estimates[i])
        differences.append(diff)
    
    plt.semilogx(prior_stds, differences, 'bo-', linewidth=2)
    plt.xlabel('Prior Standard Deviation')
    plt.ylabel('||MAP - Ridge||')
    plt.title('MAP vs Ridge Equivalence')
    plt.grid(True, alpha=0.3)
    
    # Show regularization effect
    plt.subplot(1, 3, 3)
    weight_norms = [np.linalg.norm(est) for est in map_estimates]
    plt.semilogx(prior_stds, weight_norms, 'ro-', linewidth=2)
    plt.xlabel('Prior Standard Deviation')
    plt.ylabel('L2 Norm of Weights')
    plt.title('Regularization Effect')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("MAP Estimation Analysis:")
    print("Prior σ\tMAP-Ridge Diff\tWeight Norm")
    print("-" * 40)
    for i, prior_std in enumerate(prior_stds):
        diff = np.linalg.norm(map_estimates[i] - ridge_estimates[i])
        norm = np.linalg.norm(map_estimates[i])
        print(f"{prior_std:8.1f}\t{diff:12.6f}\t{norm:10.4f}")
    
    print(f"\nTrue weights: {true_weights}")
    print(f"MAP estimates (σ=1.0): {map_estimates[2]}")
    print(f"Ridge estimates (α=0.5): {ridge_estimates[2]}")
    
    return map_estimates, ridge_estimates, prior_stds


def demonstrate_bayesian_logistic_regression():
    """Demonstrate Bayesian logistic regression"""
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    np.random.seed(42)
    n_samples, n_features = 200, 3
    
    # Generate data
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.5, -0.8, 0.3])
    logits = X @ true_weights
    probs = sigmoid(logits)
    y = (np.random.rand(n_samples) < probs).astype(int)
    
    # Split data
    train_idx = np.random.choice(n_samples, 150, replace=False)
    test_idx = np.setdiff1d(np.arange(n_samples), train_idx)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    def bayesian_logistic_regression(X_train, y_train, X_test, prior_std=1.0):
        """Bayesian logistic regression with Gaussian prior"""
        def objective(theta):
            # Log-likelihood
            logits = X_train @ theta
            probs = sigmoid(logits)
            log_likelihood = np.sum(y_train * np.log(probs + 1e-15) + 
                                   (1 - y_train) * np.log(1 - probs + 1e-15))
            
            # Log-prior
            log_prior = -0.5 * np.sum(theta**2) / (prior_std**2)
            
            return -(log_likelihood + log_prior)
        
        # MAP estimation
        initial_theta = np.zeros(X_train.shape[1])
        result = minimize(objective, initial_theta, method='L-BFGS-B')
        map_theta = result.x
        
        # Predictions
        logits_test = X_test @ map_theta
        probs_test = sigmoid(logits_test)
        predictions = (probs_test > 0.5).astype(int)
        
        return map_theta, probs_test, predictions
    
    # Compare different prior strengths
    prior_stds = [0.1, 0.5, 1.0, 2.0]
    results = {}
    
    for prior_std in prior_stds:
        map_theta, probs_test, predictions = bayesian_logistic_regression(
            X_train, y_train, X_test, prior_std=prior_std
        )
        
        accuracy = accuracy_score(y_test, predictions)
        log_loss_val = log_loss(y_test, probs_test)
        
        results[prior_std] = {
            'theta': map_theta,
            'accuracy': accuracy,
            'log_loss': log_loss_val,
            'probs': probs_test
        }
    
    # Compare with sklearn
    sklearn_lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42)
    sklearn_lr.fit(X_train, y_train)
    sklearn_predictions = sklearn_lr.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Compare accuracies
    plt.subplot(1, 3, 1)
    prior_stds_list = list(results.keys())
    accuracies = [results[std]['accuracy'] for std in prior_stds_list]
    
    plt.plot(prior_stds_list, accuracies, 'bo-', linewidth=2, label='Bayesian')
    plt.axhline(sklearn_accuracy, color='red', linestyle='--', label='Sklearn')
    plt.xlabel('Prior Standard Deviation')
    plt.ylabel('Test Accuracy')
    plt.title('Bayesian vs Sklearn Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare weights
    plt.subplot(1, 3, 2)
    for i, prior_std in enumerate(prior_stds_list):
        plt.plot(range(n_features), results[prior_std]['theta'], 
                'o-', label=f'σ={prior_std}', alpha=0.7)
    plt.plot(range(n_features), true_weights, 'k*', markersize=15, label='True Weights')
    plt.xlabel('Feature Index')
    plt.ylabel('Weight Value')
    plt.title('MAP Estimates vs True Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show predictions
    plt.subplot(1, 3, 3)
    best_prior_std = prior_stds_list[np.argmax(accuracies)]
    best_probs = results[best_prior_std]['probs']
    
    plt.scatter(X_test[:, 0], y_test, alpha=0.6, label='True labels')
    plt.scatter(X_test[:, 0], best_probs, color='red', alpha=0.6, label='Predicted probabilities')
    plt.xlabel('Feature 1')
    plt.ylabel('Probability')
    plt.title(f'Bayesian Predictions (σ={best_prior_std})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Bayesian Logistic Regression Analysis:")
    print("Prior σ\tAccuracy\tLog Loss")
    print("-" * 30)
    for prior_std in prior_stds_list:
        print(f"{prior_std:8.1f}\t{results[prior_std]['accuracy']:.3f}\t\t{results[prior_std]['log_loss']:.4f}")
    
    print(f"\nSklearn accuracy: {sklearn_accuracy:.3f}")
    print(f"Best Bayesian accuracy: {max(accuracies):.3f} (σ={best_prior_std})")
    print(f"True weights: {true_weights}")
    print(f"Best MAP estimates: {results[best_prior_std]['theta']}")
    
    return results, sklearn_accuracy


if __name__ == "__main__":
    # Run all demonstrations
    print("Running Model Selection Demonstrations...")
    print("=" * 60)
    
    print("\n1. Naive Model Selection (Why it fails):")
    degrees, train_errors, test_errors = demonstrate_naive_model_selection()
    
    print("\n2. Hold-out Cross-Validation:")
    holdout_results, best_model = demonstrate_holdout_validation()
    
    print("\n3. k-Fold Cross-Validation:")
    kfold_results, best_alphas = demonstrate_kfold_cross_validation()
    
    print("\n4. Leave-One-Out Cross-Validation:")
    loo_scores, kfold_scores, degrees = demonstrate_leave_one_out_cv()
    
    print("\n5. Bayesian Coin Flipping:")
    coin_results = demonstrate_bayesian_coin_flipping()
    
    print("\n6. MAP Estimation:")
    map_estimates, ridge_estimates, prior_stds = demonstrate_map_estimation()
    
    print("\n7. Bayesian Logistic Regression:")
    bayes_results, sklearn_accuracy = demonstrate_bayesian_logistic_regression()
    
    print("\nAll demonstrations completed!")

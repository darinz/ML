"""
Regularization Demonstration

This module demonstrates various regularization techniques including L1, L2, 
Elastic Net, and implicit regularization effects in machine learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn


def demonstrate_l2_regularization():
    """Demonstrate L2 regularization (Ridge) effects"""
    
    np.random.seed(42)
    n_samples, n_features = 100, 20
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features) * 0.5
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Different lambda values
    lambda_values = [0, 0.01, 0.1, 1, 10, 100]
    weights_history = []
    train_scores = []
    test_scores = []
    
    # Split data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    for lambda_val in lambda_values:
        # Fit Ridge regression
        ridge = Ridge(alpha=lambda_val)
        ridge.fit(X_train, y_train)
        
        # Store results
        weights_history.append(ridge.coef_)
        train_scores.append(ridge.score(X_train, y_train))
        test_scores.append(ridge.score(X_test, y_test))
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot coefficient magnitudes
    plt.subplot(1, 3, 1)
    weights_array = np.array(weights_history)
    for i, lambda_val in enumerate(lambda_values):
        plt.plot(range(n_features), np.abs(weights_array[i]), 
                marker='o', label=f'λ={lambda_val}')
    plt.xlabel('Feature Index')
    plt.ylabel('|Coefficient|')
    plt.title('L2 Regularization: Coefficient Shrinkage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot performance
    plt.subplot(1, 3, 2)
    plt.semilogx(lambda_values, train_scores, 'bo-', label='Training Score')
    plt.semilogx(lambda_values, test_scores, 'ro-', label='Test Score')
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('R² Score')
    plt.title('L2 Regularization: Performance vs λ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot coefficient norm
    plt.subplot(1, 3, 3)
    norms = [np.linalg.norm(w) for w in weights_history]
    plt.semilogx(lambda_values, norms, 'go-')
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('L2 Norm of Coefficients')
    plt.title('L2 Regularization: Coefficient Norm')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("L2 Regularization Analysis:")
    print("λ\t\tTrain Score\tTest Score\tCoeff Norm")
    print("-" * 50)
    for i, lambda_val in enumerate(lambda_values):
        print(f"{lambda_val:8.2f}\t{train_scores[i]:10.4f}\t{test_scores[i]:10.4f}\t{norms[i]:10.4f}")
    
    return lambda_values, weights_history, train_scores, test_scores


def demonstrate_l1_regularization():
    """Demonstrate L1 regularization (LASSO) and feature selection"""
    
    np.random.seed(42)
    n_samples, n_features = 100, 50
    
    # Generate data with only 5 relevant features
    X = np.random.randn(n_samples, n_features)
    true_weights = np.zeros(n_features)
    true_weights[:5] = [1, 2, 3, 4, 5]  # Only first 5 features matter
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Different lambda values
    lambda_values = [0.001, 0.01, 0.1, 1, 10]
    sparsity_levels = []
    selected_features = []
    
    # Split data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    for lambda_val in lambda_values:
        # Fit LASSO
        lasso = Lasso(alpha=lambda_val)
        lasso.fit(X_train, y_train)
        
        # Calculate sparsity
        sparsity = np.sum(lasso.coef_ == 0) / n_features
        sparsity_levels.append(sparsity)
        
        # Get selected features
        selected = np.where(lasso.coef_ != 0)[0]
        selected_features.append(selected)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot sparsity vs lambda
    plt.subplot(1, 3, 1)
    plt.semilogx(lambda_values, sparsity_levels, 'bo-')
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('Sparsity Level')
    plt.title('L1 Regularization: Sparsity vs λ')
    plt.grid(True, alpha=0.3)
    
    # Plot coefficient paths
    plt.subplot(1, 3, 2)
    for lambda_val in lambda_values:
        lasso = Lasso(alpha=lambda_val)
        lasso.fit(X_train, y_train)
        plt.plot(range(n_features), lasso.coef_, 
                marker='o', label=f'λ={lambda_val}')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('L1 Regularization: Coefficient Paths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot feature selection accuracy
    plt.subplot(1, 3, 3)
    selection_accuracy = []
    for selected in selected_features:
        # Check how many of the first 5 features were selected
        correct_selections = len(set(selected) & set(range(5)))
        accuracy = correct_selections / 5
        selection_accuracy.append(accuracy)
    
    plt.semilogx(lambda_values, selection_accuracy, 'ro-')
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('Feature Selection Accuracy')
    plt.title('L1 Regularization: Selection Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("L1 Regularization Analysis:")
    print("λ\t\tSparsity\tSelected Features\tSelection Accuracy")
    print("-" * 60)
    for i, lambda_val in enumerate(lambda_values):
        print(f"{lambda_val:8.3f}\t{sparsity_levels[i]:8.3f}\t{len(selected_features[i]):8d}\t\t{selection_accuracy[i]:8.3f}")
    
    return lambda_values, sparsity_levels, selected_features


def demonstrate_elastic_net():
    """Demonstrate Elastic Net regularization"""
    
    np.random.seed(42)
    n_samples, n_features = 100, 20
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    # Make some features correlated
    X[:, 5] = X[:, 0] + 0.1 * np.random.randn(n_samples)
    X[:, 6] = X[:, 1] + 0.1 * np.random.randn(n_samples)
    
    true_weights = np.zeros(n_features)
    true_weights[:8] = [1, 2, 0, 0, 3, 0, 0, 4]  # Some zero, some correlated
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Different alpha values (L1 ratio)
    alpha_values = [0, 0.25, 0.5, 0.75, 1.0]
    lambda_val = 0.1
    
    # Split data
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Compare different methods
    methods = {
        'Ridge (L2)': Ridge(alpha=lambda_val),
        'Lasso (L1)': Lasso(alpha=lambda_val),
        'Elastic Net (α=0.5)': ElasticNet(alpha=lambda_val, l1_ratio=0.5)
    }
    
    results = {}
    for name, model in methods.items():
        model.fit(X_train, y_train)
        results[name] = {
            'coef': model.coef_,
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test),
            'sparsity': np.sum(model.coef_ == 0) / n_features
        }
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Compare coefficients
    plt.subplot(1, 3, 1)
    x_pos = np.arange(n_features)
    width = 0.25
    
    for i, (name, result) in enumerate(results.items()):
        plt.bar(x_pos + i*width, result['coef'], width, label=name, alpha=0.7)
    
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Elastic Net: Coefficient Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare performance
    plt.subplot(1, 3, 2)
    names = list(results.keys())
    train_scores = [results[name]['train_score'] for name in names]
    test_scores = [results[name]['test_score'] for name in names]
    
    x_pos = np.arange(len(names))
    plt.bar(x_pos - 0.2, train_scores, 0.4, label='Training Score', alpha=0.7)
    plt.bar(x_pos + 0.2, test_scores, 0.4, label='Test Score', alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('R² Score')
    plt.title('Elastic Net: Performance Comparison')
    plt.xticks(x_pos, names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare sparsity
    plt.subplot(1, 3, 3)
    sparsities = [results[name]['sparsity'] for name in names]
    plt.bar(names, sparsities, alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('Sparsity Level')
    plt.title('Elastic Net: Sparsity Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Elastic Net Comparison:")
    print("Method\t\t\tTrain Score\tTest Score\tSparsity")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<20}\t{result['train_score']:.4f}\t\t{result['test_score']:.4f}\t\t{result['sparsity']:.3f}")
    
    return results


def demonstrate_regularization_parameter_selection():
    """Demonstrate how to choose regularization parameters using cross-validation"""
    
    np.random.seed(42)
    n_samples, n_features = 200, 30
    
    # Generate data
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features) * 0.5
    y = X @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Define lambda values to try
    lambda_values = [0.001, 0.01, 0.1, 1, 10, 100]
    cv_scores = []
    
    # Use cross-validation to find best lambda
    for lambda_val in lambda_values:
        model = Ridge(alpha=lambda_val)
        scores = cross_val_score(model, X, y, cv=5)
        cv_scores.append(np.mean(scores))
    
    best_lambda = lambda_values[np.argmax(cv_scores)]
    best_score = max(cv_scores)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(lambda_values, cv_scores, 'bo-', linewidth=2)
    plt.axvline(best_lambda, color='red', linestyle='--', label=f'Best λ={best_lambda}')
    plt.xlabel('Regularization Parameter (λ)')
    plt.ylabel('Cross-Validation Score')
    plt.title('Regularization Parameter Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show the effect of the best lambda
    plt.subplot(1, 2, 2)
    # Compare unregularized vs regularized
    unreg_model = Ridge(alpha=0)
    reg_model = Ridge(alpha=best_lambda)
    
    unreg_model.fit(X, y)
    reg_model.fit(X, y)
    
    plt.plot(range(n_features), unreg_model.coef_, 'bo-', label='Unregularized', alpha=0.7)
    plt.plot(range(n_features), reg_model.coef_, 'ro-', label=f'Regularized (λ={best_lambda})', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Effect of Best Regularization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Regularization Parameter Selection:")
    print("λ\t\tCV Score")
    print("-" * 20)
    for lambda_val, score in zip(lambda_values, cv_scores):
        marker = " *" if lambda_val == best_lambda else ""
        print(f"{lambda_val:8.3f}\t{score:.4f}{marker}")
    
    print(f"\nBest lambda: {best_lambda}")
    print(f"Best CV score: {best_score:.4f}")
    
    return lambda_values, cv_scores, best_lambda


def demonstrate_feature_scaling_importance():
    """Demonstrate why feature scaling is crucial for regularization"""
    
    np.random.seed(42)
    n_samples = 100
    
    # Generate features with different scales
    X_unscaled = np.column_stack([
        np.random.uniform(0, 1, n_samples),      # Feature 1: 0-1
        np.random.uniform(0, 1000, n_samples),   # Feature 2: 0-1000
        np.random.uniform(-0.1, 0.1, n_samples)  # Feature 3: -0.1 to 0.1
    ])
    
    # True relationship
    true_weights = np.array([1.0, 0.001, 10.0])  # Different scales needed
    y = X_unscaled @ true_weights + 0.1 * np.random.randn(n_samples)
    
    # Split data
    X_train, X_test = X_unscaled[:80], X_unscaled[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Compare unscaled vs scaled
    # Unscaled
    ridge_unscaled = Ridge(alpha=1.0)
    ridge_unscaled.fit(X_train, y_train)
    
    # Scaled
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ridge_scaled = Ridge(alpha=1.0)
    ridge_scaled.fit(X_train_scaled, y_train)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Compare coefficients
    plt.subplot(1, 3, 1)
    x_pos = np.arange(3)
    width = 0.35
    
    plt.bar(x_pos - width/2, ridge_unscaled.coef_, width, label='Unscaled', alpha=0.7)
    plt.bar(x_pos + width/2, ridge_scaled.coef_, width, label='Scaled', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Scaling: Coefficient Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Compare performance
    plt.subplot(1, 3, 2)
    methods = ['Unscaled', 'Scaled']
    train_scores = [ridge_unscaled.score(X_train, y_train), 
                   ridge_scaled.score(X_train_scaled, y_train)]
    test_scores = [ridge_unscaled.score(X_test, y_test), 
                  ridge_scaled.score(X_test_scaled, y_test)]
    
    x_pos = np.arange(len(methods))
    plt.bar(x_pos - 0.2, train_scores, 0.4, label='Training Score', alpha=0.7)
    plt.bar(x_pos + 0.2, test_scores, 0.4, label='Test Score', alpha=0.7)
    plt.xlabel('Method')
    plt.ylabel('R² Score')
    plt.title('Feature Scaling: Performance Comparison')
    plt.xticks(x_pos, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show feature scales
    plt.subplot(1, 3, 3)
    feature_scales = [X_train[:, i].std() for i in range(3)]
    plt.bar(range(3), feature_scales, alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Standard Deviation')
    plt.title('Feature Scaling: Original Feature Scales')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Feature Scaling Analysis:")
    print("Method\t\tTrain Score\tTest Score")
    print("-" * 40)
    print(f"Unscaled\t{train_scores[0]:.4f}\t\t{test_scores[0]:.4f}")
    print(f"Scaled\t\t{train_scores[1]:.4f}\t\t{test_scores[1]:.4f}")
    
    print(f"\nFeature scales (std): {feature_scales}")
    print(f"Unscaled coefficients: {ridge_unscaled.coef_}")
    print(f"Scaled coefficients: {ridge_scaled.coef_}")
    
    return ridge_unscaled, ridge_scaled, feature_scales


def demonstrate_implicit_regularization():
    """Demonstrate implicit regularization effects of different optimizers"""
    
    # Set up simple neural network
    class SimpleNet(nn.Module):
        def __init__(self, input_size=10, hidden_size=20, output_size=1):
            super().__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.layer2 = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.layer2(x)
            return x
    
    # Generate data
    np.random.seed(42)
    n_samples = 100
    X = torch.randn(n_samples, 10)
    y = torch.randn(n_samples, 1)
    
    # Different optimizers
    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop
    }
    
    results = {}
    
    for opt_name, opt_class in optimizers.items():
        # Create model
        model = SimpleNet()
        
        # Choose optimizer
        if opt_name == 'SGD':
            optimizer = opt_class(model.parameters(), lr=0.01, momentum=0.9)
        elif opt_name == 'Adam':
            optimizer = opt_class(model.parameters(), lr=0.001)
        else:  # RMSprop
            optimizer = opt_class(model.parameters(), lr=0.01)
        
        # Training loop
        criterion = nn.MSELoss()
        losses = []
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Store results
        results[opt_name] = {
            'final_loss': losses[-1],
            'loss_history': losses,
            'weight_norms': [torch.norm(p).item() for p in model.parameters()]
        }
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 3, 1)
    for opt_name, result in results.items():
        plt.plot(result['loss_history'], label=opt_name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Implicit Regularization: Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot final losses
    plt.subplot(1, 3, 2)
    opt_names = list(results.keys())
    final_losses = [results[name]['final_loss'] for name in opt_names]
    plt.bar(opt_names, final_losses, alpha=0.7)
    plt.ylabel('Final Training Loss')
    plt.title('Implicit Regularization: Final Loss Comparison')
    plt.grid(True, alpha=0.3)
    
    # Plot weight norms
    plt.subplot(1, 3, 3)
    weight_norms = [results[name]['weight_norms'] for name in opt_names]
    x_pos = np.arange(len(weight_norms[0]))
    width = 0.25
    
    for i, opt_name in enumerate(opt_names):
        plt.bar(x_pos + i*width, weight_norms[i], width, label=opt_name, alpha=0.7)
    
    plt.xlabel('Layer Index')
    plt.ylabel('Weight Norm')
    plt.title('Implicit Regularization: Weight Norms')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("Implicit Regularization Analysis:")
    print("Optimizer\tFinal Loss\tAvg Weight Norm")
    print("-" * 45)
    for opt_name, result in results.items():
        avg_norm = np.mean(result['weight_norms'])
        print(f"{opt_name:<12}\t{result['final_loss']:.6f}\t{avg_norm:.4f}")
    
    return results


if __name__ == "__main__":
    # Run all demonstrations
    print("Running Regularization Demonstrations...")
    print("=" * 60)
    
    print("\n1. L2 Regularization (Ridge):")
    lambda_values, weights_history, train_scores, test_scores = demonstrate_l2_regularization()
    
    print("\n2. L1 Regularization (LASSO):")
    lambda_values_l1, sparsity_levels, selected_features = demonstrate_l1_regularization()
    
    print("\n3. Elastic Net:")
    elastic_net_results = demonstrate_elastic_net()
    
    print("\n4. Regularization Parameter Selection:")
    lambda_values_cv, cv_scores, best_lambda = demonstrate_regularization_parameter_selection()
    
    print("\n5. Feature Scaling Importance:")
    ridge_unscaled, ridge_scaled, feature_scales = demonstrate_feature_scaling_importance()
    
    print("\n6. Implicit Regularization:")
    implicit_results = demonstrate_implicit_regularization()
    
    print("\nAll demonstrations completed!")

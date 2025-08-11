import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def demonstrate_linear_assumptions():
    """Demonstrate when linear models work and when they don't"""
    
    # Generate data with different relationship types
    np.random.seed(42)
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    
    # Linear relationship
    y_linear = 2 + 1.5 * x + np.random.normal(0, 0.5, n_samples)
    
    # Non-linear relationship (quadratic)
    y_quadratic = 2 + 0.5 * x**2 + np.random.normal(0, 0.5, n_samples)
    
    # Multiplicative noise
    y_multiplicative = 2 + 1.5 * x + np.random.normal(0, 0.1 * x, n_samples)
    
    print("Linear Model Assumptions")
    print("=" * 40)
    print("When linear models work well:")
    print("1. Linear relationship between features and target")
    print("2. Additive noise (not multiplicative)")
    print("3. Constant variance across all x values")
    print("4. Independent errors")
    print()
    
    # Fit linear models
    X = x.reshape(-1, 1)
    
    lr_linear = LinearRegression()
    lr_linear.fit(X, y_linear)
    y_pred_linear = lr_linear.predict(X)
    
    lr_quadratic = LinearRegression()
    lr_quadratic.fit(X, y_quadratic)
    y_pred_quadratic = lr_quadratic.predict(X)
    
    lr_multiplicative = LinearRegression()
    lr_multiplicative.fit(X, y_multiplicative)
    y_pred_multiplicative = lr_multiplicative.predict(X)
    
    # Calculate R-squared
    r2_linear = r2_score(y_linear, y_pred_linear)
    r2_quadratic = r2_score(y_quadratic, y_pred_quadratic)
    r2_multiplicative = r2_score(y_multiplicative, y_pred_multiplicative)
    
    print("Model Performance (R² scores):")
    print(f"Linear relationship: {r2_linear:.3f} (Excellent)")
    print(f"Quadratic relationship: {r2_quadratic:.3f} (Poor)")
    print(f"Multiplicative noise: {r2_multiplicative:.3f} (Poor)")
    print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Linear relationship
    plt.subplot(1, 3, 1)
    plt.scatter(x, y_linear, alpha=0.6, label='Data')
    plt.plot(x, y_pred_linear, 'r-', linewidth=2, label='Linear Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Linear Relationship\nR² = {r2_linear:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Quadratic relationship
    plt.subplot(1, 3, 2)
    plt.scatter(x, y_quadratic, alpha=0.6, label='Data')
    plt.plot(x, y_pred_quadratic, 'r-', linewidth=2, label='Linear Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Quadratic Relationship\nR² = {r2_quadratic:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Multiplicative noise
    plt.subplot(1, 3, 3)
    plt.scatter(x, y_multiplicative, alpha=0.6, label='Data')
    plt.plot(x, y_pred_multiplicative, 'r-', linewidth=2, label='Linear Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Multiplicative Noise\nR² = {r2_multiplicative:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Linear models work perfectly for linear relationships")
    print("2. They fail for non-linear relationships")
    print("3. They assume additive noise")
    print("4. They assume constant variance")
    print("5. Check assumptions before using linear models")
    
    return r2_linear, r2_quadratic, r2_multiplicative

if __name__ == "__main__":
    linear_assumptions_demo = demonstrate_linear_assumptions()

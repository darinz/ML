import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def demonstrate_linear_vs_nonlinear():
    """Demonstrate when linear models work well and when they don't"""
    
    # Generate data with different relationships
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    
    # Linear relationship
    y_linear = 2 * x + 1 + np.random.normal(0, 0.5, 100)
    
    # Non-linear relationship (quadratic)
    y_quadratic = 0.5 * x**2 - 2 * x + 3 + np.random.normal(0, 0.5, 100)
    
    # Non-linear relationship (exponential)
    y_exponential = 2 * np.exp(0.3 * x) + np.random.normal(0, 1, 100)
    
    print("Linear vs. Non-linear Relationships")
    print("=" * 40)
    print("Linear models work well when:")
    print("1. The true relationship is approximately linear")
    print("2. We only need a rough approximation")
    print("3. We want interpretable coefficients")
    print("4. We have limited data")
    print()
    
    # Fit linear models to all three datasets
    # Linear relationship
    X_linear = x.reshape(-1, 1)
    lr_linear = LinearRegression()
    lr_linear.fit(X_linear, y_linear)
    y_pred_linear = lr_linear.predict(X_linear)
    
    # Quadratic relationship
    lr_quadratic = LinearRegression()
    lr_quadratic.fit(X_linear, y_quadratic)
    y_pred_quadratic = lr_quadratic.predict(X_linear)
    
    # Exponential relationship
    lr_exponential = LinearRegression()
    lr_exponential.fit(X_linear, y_exponential)
    y_pred_exponential = lr_exponential.predict(X_linear)
    
    # Calculate R-squared scores
    r2_linear = r2_score(y_linear, y_pred_linear)
    r2_quadratic = r2_score(y_quadratic, y_pred_quadratic)
    r2_exponential = r2_score(y_exponential, y_pred_exponential)
    
    print("Model Performance (R² scores):")
    print(f"Linear relationship: {r2_linear:.3f} (Excellent fit)")
    print(f"Quadratic relationship: {r2_quadratic:.3f} (Good approximation)")
    print(f"Exponential relationship: {r2_exponential:.3f} (Poor fit)")
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
    
    # Exponential relationship
    plt.subplot(1, 3, 3)
    plt.scatter(x, y_exponential, alpha=0.6, label='Data')
    plt.plot(x, y_pred_exponential, 'r-', linewidth=2, label='Linear Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Exponential Relationship\nR² = {r2_exponential:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Linear models work perfectly for linear relationships")
    print("2. They provide good approximations for mildly non-linear relationships")
    print("3. They fail for highly non-linear relationships")
    print("4. R² score tells us how well the linear model fits")
    print("5. Sometimes a simple approximation is better than a complex model")
    
    return r2_linear, r2_quadratic, r2_exponential

if __name__ == "__main__":
    linear_demo = demonstrate_linear_vs_nonlinear()

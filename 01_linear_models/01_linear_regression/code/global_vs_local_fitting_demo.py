import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def demonstrate_global_vs_local_fitting():
    """Demonstrate the difference between global and local fitting"""
    
    # Generate non-linear data
    np.random.seed(42)
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    
    # True relationship: sine wave with noise
    y_true = 2 * np.sin(x) + 0.5 * x
    y = y_true + np.random.normal(0, 0.3, n_samples)
    
    print("Global vs. Local Fitting Comparison")
    print("=" * 50)
    print("Challenge: Fit a model to non-linear data")
    print("Global approach: One model for all data")
    print("Local approach: Different models for different regions")
    print()
    
    # Global linear fit
    X_linear = x.reshape(-1, 1)
    lr_global = LinearRegression()
    lr_global.fit(X_linear, y)
    y_pred_global = lr_global.predict(X_linear)
    
    # Global polynomial fit
    poly_global = Pipeline([
        ('poly', PolynomialFeatures(degree=5)),
        ('linear', LinearRegression())
    ])
    poly_global.fit(X_linear, y)
    y_pred_poly = poly_global.predict(X_linear)
    
    # Local weighted fit (simplified version)
    def local_weighted_regression(x_query, x_data, y_data, tau=1.0):
        """Simple implementation of locally weighted regression"""
        weights = np.exp(-0.5 * ((x_data - x_query) / tau)**2)
        weighted_X = X_linear * weights.reshape(-1, 1)
        weighted_y = y * weights
        
        # Fit weighted linear regression
        theta = np.linalg.lstsq(weighted_X, weighted_y, rcond=None)[0]
        return theta[0] + theta[1] * x_query
    
    # Make predictions using local weighted regression
    y_pred_local = np.array([local_weighted_regression(xi, x, y) for xi in x])
    
    # Calculate errors
    from sklearn.metrics import mean_squared_error
    mse_global = mean_squared_error(y, y_pred_global)
    mse_poly = mean_squared_error(y, y_pred_poly)
    mse_local = mean_squared_error(y, y_pred_local)
    
    print("Model Performance (MSE):")
    print(f"Global Linear: {mse_global:.3f}")
    print(f"Global Polynomial: {mse_poly:.3f}")
    print(f"Local Weighted: {mse_local:.3f}")
    print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Global linear fit
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, alpha=0.6, label='Data')
    plt.plot(x, y_true, 'g-', linewidth=2, label='True Relationship')
    plt.plot(x, y_pred_global, 'r-', linewidth=2, label='Global Linear Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Global Linear Fit\nMSE: {mse_global:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Global polynomial fit
    plt.subplot(1, 3, 2)
    plt.scatter(x, y, alpha=0.6, label='Data')
    plt.plot(x, y_true, 'g-', linewidth=2, label='True Relationship')
    plt.plot(x, y_pred_poly, 'r-', linewidth=2, label='Global Polynomial Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Global Polynomial Fit\nMSE: {mse_poly:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Local weighted fit
    plt.subplot(1, 3, 3)
    plt.scatter(x, y, alpha=0.6, label='Data')
    plt.plot(x, y_true, 'g-', linewidth=2, label='True Relationship')
    plt.plot(x, y_pred_local, 'r-', linewidth=2, label='Local Weighted Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Local Weighted Fit\nMSE: {mse_local:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Global linear model fails for non-linear data")
    print("2. Global polynomial model overfits and is unstable")
    print("3. Local weighted model adapts to local structure")
    print("4. Local models capture complex patterns with simple functions")
    print("5. Trade-off: Better fit vs. computational cost")
    
    return mse_global, mse_poly, mse_local

if __name__ == "__main__":
    fitting_demo = demonstrate_global_vs_local_fitting()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def demonstrate_probabilistic_thinking():
    """Demonstrate why probabilistic interpretation matters"""
    
    # Generate data with different noise distributions
    np.random.seed(42)
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    
    # True relationship
    true_theta = [2, 1.5]  # intercept, slope
    y_true = true_theta[0] + true_theta[1] * x
    
    # Different noise distributions
    noise_gaussian = np.random.normal(0, 1, n_samples)
    noise_laplace = np.random.laplace(0, 1, n_samples)  # heavy tails
    noise_outliers = np.random.normal(0, 1, n_samples)
    noise_outliers[np.random.choice(n_samples, 5, replace=False)] += 10  # add outliers
    
    y_gaussian = y_true + noise_gaussian
    y_laplace = y_true + noise_laplace
    y_outliers = y_true + noise_outliers
    
    print("Probabilistic Interpretation: Why It Matters")
    print("=" * 60)
    print("Different noise distributions lead to different optimal methods")
    print()
    
    # Fit models using different approaches
    X = x.reshape(-1, 1)
    
    # Least squares (optimal for Gaussian noise)
    lr_gaussian = LinearRegression()
    lr_gaussian.fit(X, y_gaussian)
    y_pred_gaussian = lr_gaussian.predict(X)
    
    lr_laplace = LinearRegression()
    lr_laplace.fit(X, y_laplace)
    y_pred_laplace = lr_laplace.predict(X)
    
    lr_outliers = LinearRegression()
    lr_outliers.fit(X, y_outliers)
    y_pred_outliers = lr_outliers.predict(X)
    
    # Calculate errors
    mse_gaussian = mean_squared_error(y_gaussian, y_pred_gaussian)
    mse_laplace = mean_squared_error(y_laplace, y_pred_laplace)
    mse_outliers = mean_squared_error(y_outliers, y_pred_outliers)
    
    mae_gaussian = mean_absolute_error(y_gaussian, y_pred_gaussian)
    mae_laplace = mean_absolute_error(y_laplace, y_pred_laplace)
    mae_outliers = mean_absolute_error(y_outliers, y_pred_outliers)
    
    print("Model Performance Comparison:")
    print("-" * 40)
    print("Gaussian Noise (Least Squares Optimal):")
    print(f"  MSE: {mse_gaussian:.3f}")
    print(f"  MAE: {mae_gaussian:.3f}")
    print()
    print("Laplace Noise (MAE Optimal):")
    print(f"  MSE: {mse_laplace:.3f}")
    print(f"  MAE: {mae_laplace:.3f}")
    print()
    print("Outliers (Robust Methods Better):")
    print(f"  MSE: {mse_outliers:.3f}")
    print(f"  MAE: {mae_outliers:.3f}")
    print()
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    # Gaussian noise
    plt.subplot(2, 3, 1)
    plt.scatter(x, y_gaussian, alpha=0.6, label='Data')
    plt.plot(x, y_true, 'r-', linewidth=2, label='True Relationship')
    plt.plot(x, y_pred_gaussian, 'g--', linewidth=2, label='Least Squares Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Gaussian Noise\n(Least Squares Optimal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Laplace noise
    plt.subplot(2, 3, 2)
    plt.scatter(x, y_laplace, alpha=0.6, label='Data')
    plt.plot(x, y_true, 'r-', linewidth=2, label='True Relationship')
    plt.plot(x, y_pred_laplace, 'g--', linewidth=2, label='Least Squares Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Laplace Noise\n(MAE Optimal)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Outliers
    plt.subplot(2, 3, 3)
    plt.scatter(x, y_outliers, alpha=0.6, label='Data')
    plt.plot(x, y_true, 'r-', linewidth=2, label='True Relationship')
    plt.plot(x, y_pred_outliers, 'g--', linewidth=2, label='Least Squares Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Outliers\n(Robust Methods Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Noise distributions
    plt.subplot(2, 3, 4)
    noise_range = np.linspace(-4, 4, 1000)
    plt.plot(noise_range, norm.pdf(noise_range, 0, 1), 'b-', linewidth=2, label='Gaussian')
    plt.plot(noise_range, laplace.pdf(noise_range, 0, 1), 'r-', linewidth=2, label='Laplace')
    plt.xlabel('Noise Value')
    plt.ylabel('Probability Density')
    plt.title('Noise Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Residuals comparison
    plt.subplot(2, 3, 5)
    residuals_gaussian = y_gaussian - y_pred_gaussian
    residuals_laplace = y_laplace - y_pred_laplace
    residuals_outliers = y_outliers - y_pred_outliers
    
    plt.hist(residuals_gaussian, bins=20, alpha=0.7, label='Gaussian', density=True)
    plt.hist(residuals_laplace, bins=20, alpha=0.7, label='Laplace', density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residual Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error comparison
    plt.subplot(2, 3, 6)
    datasets = ['Gaussian', 'Laplace', 'Outliers']
    mse_values = [mse_gaussian, mse_laplace, mse_outliers]
    mae_values = [mae_gaussian, mae_laplace, mae_outliers]
    
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    plt.bar(x_pos - width/2, mse_values, width, label='MSE', alpha=0.7)
    plt.bar(x_pos + width/2, mae_values, width, label='MAE', alpha=0.7)
    plt.xlabel('Noise Type')
    plt.ylabel('Error')
    plt.title('Error Comparison')
    plt.xticks(x_pos, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Gaussian noise: Least squares is optimal")
    print("2. Laplace noise: MAE is more appropriate")
    print("3. Outliers: Robust methods work better")
    print("4. Assumptions matter for method choice")
    print("5. Probabilistic thinking guides method selection")
    
    return mse_gaussian, mse_laplace, mse_outliers

if __name__ == "__main__":
    prob_demo = demonstrate_probabilistic_thinking()

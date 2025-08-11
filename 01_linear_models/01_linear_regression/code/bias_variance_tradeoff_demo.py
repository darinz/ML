import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def demonstrate_bias_variance_tradeoff():
    """Demonstrate the bias-variance trade-off in model complexity"""
    
    # Generate data with true relationship
    np.random.seed(42)
    n_samples = 50
    x = np.linspace(0, 10, n_samples)
    y_true = 2 * np.sin(x) + 0.5 * x
    y = y_true + np.random.normal(0, 0.5, n_samples)
    
    print("Bias-Variance Trade-off in Model Complexity")
    print("=" * 50)
    print("Challenge: Find the right level of model complexity")
    print("Too simple: High bias, low variance (underfitting)")
    print("Too complex: Low bias, high variance (overfitting)")
    print("Just right: Balanced bias and variance")
    print()
    
    # Different polynomial degrees
    degrees = [1, 2, 5, 10, 15]
    models = []
    predictions = []
    
    X = x.reshape(-1, 1)
    
    for degree in degrees:
        if degree == 1:
            model = LinearRegression()
        else:
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree)),
                ('linear', LinearRegression())
            ])
        
        model.fit(X, y)
        y_pred = model.predict(X)
        
        models.append(model)
        predictions.append(y_pred)
    
    # Calculate bias and variance (simplified)
    from sklearn.metrics import mean_squared_error
    
    print("Model Performance:")
    print("-" * 30)
    for i, degree in enumerate(degrees):
        mse = mean_squared_error(y, predictions[i])
        bias_squared = np.mean((y_true - predictions[i])**2)
        variance = np.var(predictions[i])
        
        print(f"Degree {degree}:")
        print(f"  MSE: {mse:.3f}")
        print(f"  Bias²: {bias_squared:.3f}")
        print(f"  Variance: {variance:.3f}")
        print()
    
    # Visualization
    plt.figure(figsize=(15, 10))
    
    for i, degree in enumerate(degrees):
        plt.subplot(2, 3, i+1)
        plt.scatter(x, y, alpha=0.6, label='Data')
        plt.plot(x, y_true, 'g-', linewidth=2, label='True Relationship')
        plt.plot(x, predictions[i], 'r-', linewidth=2, label=f'Degree {degree}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Polynomial Degree {degree}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Degree 1: Too simple, high bias (underfitting)")
    print("2. Degree 2: Better balance, captures main pattern")
    print("3. Degree 5: Good fit, captures true relationship")
    print("4. Degree 10: Starting to overfit, fits noise")
    print("5. Degree 15: Severe overfitting, high variance")
    print("6. Sweet spot: Degree 2-5 for this data")
    
    return degrees, predictions

if __name__ == "__main__":
    bias_variance_demo = demonstrate_bias_variance_tradeoff()

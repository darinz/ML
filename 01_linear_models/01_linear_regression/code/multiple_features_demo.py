import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def demonstrate_multiple_features():
    """Demonstrate linear regression with multiple features"""
    
    # Generate sample data with multiple features
    np.random.seed(42)
    n_samples = 100
    
    # Features: living area, bedrooms, age
    living_area = np.random.uniform(1000, 3000, n_samples)
    bedrooms = np.random.randint(1, 6, n_samples)
    age = np.random.uniform(0, 50, n_samples)
    
    # Target: house price (with some feature interactions)
    base_price = 100
    area_effect = 0.15 * living_area
    bedroom_effect = 25 * bedrooms
    age_effect = -2 * age  # older houses are cheaper
    noise = np.random.normal(0, 20, n_samples)
    
    price = base_price + area_effect + bedroom_effect + age_effect + noise
    
    print("Multiple Features Linear Regression")
    print("=" * 50)
    print("Features: Living Area, Bedrooms, Age")
    print("Target: House Price")
    print()
    
    # Create feature matrix
    X = np.column_stack([living_area, bedrooms, age])
    
    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X, price)
    
    # Get coefficients
    coefficients = lr.coef_
    intercept = lr.intercept_
    
    print("Model Coefficients:")
    print(f"Intercept (θ₀): ${intercept:.2f}")
    print(f"Living Area (θ₁): ${coefficients[0]:.3f} per sq ft")
    print(f"Bedrooms (θ₂): ${coefficients[1]:.2f} per bedroom")
    print(f"Age (θ₃): ${coefficients[2]:.2f} per year")
    print()
    
    # Make predictions
    predictions = lr.predict(X)
    mse = mean_squared_error(price, predictions)
    r2 = lr.score(X, price)
    
    print("Model Performance:")
    print(f"Mean Squared Error: ${mse:.2f}")
    print(f"R-squared: {r2:.3f}")
    print()
    
    # Example predictions
    print("Example Predictions:")
    examples = [
        [2000, 3, 10],  # 2000 sq ft, 3 bedrooms, 10 years old
        [1500, 2, 5],   # 1500 sq ft, 2 bedrooms, 5 years old
        [3000, 4, 20]   # 3000 sq ft, 4 bedrooms, 20 years old
    ]
    
    for i, example in enumerate(examples):
        pred = lr.predict([example])[0]
        print(f"House {i+1}: {example[0]} sq ft, {example[1]} bedrooms, {example[2]} years old")
        print(f"  Predicted Price: ${pred:.0f}")
        print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Feature vs. Price plots
    features = ['Living Area', 'Bedrooms', 'Age']
    colors = ['blue', 'green', 'red']
    
    for i, (feature, color) in enumerate(zip([living_area, bedrooms, age], colors)):
        plt.subplot(1, 3, i+1)
        plt.scatter(feature, price, alpha=0.6, c=color)
        plt.xlabel(features[i])
        plt.ylabel('Price ($1000s)')
        plt.title(f'{features[i]} vs. Price')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Each feature contributes independently to the price")
    print("2. Coefficients show the marginal effect of each feature")
    print("3. Intercept represents the base price")
    print("4. Model assumes no interactions between features")
    print("5. R-squared shows how well the model fits the data")
    
    return coefficients, intercept, mse, r2

if __name__ == "__main__":
    multi_feature_demo = demonstrate_multiple_features()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

def demonstrate_regression_vs_classification():
    """Demonstrate the difference between regression and classification"""
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    
    # Regression data: house size vs. price
    house_sizes = np.random.uniform(1000, 3000, n_samples)
    house_prices = 100 + 0.15 * house_sizes + np.random.normal(0, 20, n_samples)
    
    # Classification data: house size vs. expensive/cheap
    price_threshold = np.median(house_prices)
    house_categories = (house_prices > price_threshold).astype(int)  # 0=cheap, 1=expensive
    
    print("Regression vs. Classification Comparison")
    print("=" * 50)
    print("Regression: Predicting continuous values")
    print("Classification: Predicting discrete categories")
    print()
    
    # Fit regression model
    lr_regression = LinearRegression()
    lr_regression.fit(house_sizes.reshape(-1, 1), house_prices)
    price_predictions = lr_regression.predict(house_sizes.reshape(-1, 1))
    
    # Fit classification model
    lr_classification = LogisticRegression()
    lr_classification.fit(house_sizes.reshape(-1, 1), house_categories)
    category_predictions = lr_classification.predict(house_sizes.reshape(-1, 1))
    category_probabilities = lr_classification.predict_proba(house_sizes.reshape(-1, 1))[:, 1]
    
    # Calculate metrics
    mse = mean_squared_error(house_prices, price_predictions)
    accuracy = accuracy_score(house_categories, category_predictions)
    
    print("Model Performance:")
    print(f"Regression MSE: {mse:.2f}")
    print(f"Classification Accuracy: {accuracy:.3f}")
    print()
    
    print("Prediction Examples:")
    print("House Size: 2000 sq ft")
    print(f"  Regression: ${price_predictions[50]:.0f} (continuous)")
    print(f"  Classification: {'Expensive' if category_predictions[50] else 'Cheap'} (discrete)")
    print(f"  Classification Probability: {category_probabilities[50]:.3f}")
    print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Regression plot
    plt.subplot(1, 3, 1)
    plt.scatter(house_sizes, house_prices, alpha=0.6, c='blue', label='Data')
    plt.plot(house_sizes, price_predictions, 'r-', linewidth=2, label='Regression Line')
    plt.xlabel('House Size (sq ft)')
    plt.ylabel('Price ($1000s)')
    plt.title('Regression: Continuous Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Classification plot
    plt.subplot(1, 3, 2)
    colors = ['red' if cat == 0 else 'blue' for cat in house_categories]
    plt.scatter(house_sizes, house_categories, c=colors, alpha=0.6, label='Data')
    plt.plot(house_sizes, category_probabilities, 'g-', linewidth=2, label='Probability')
    plt.xlabel('House Size (sq ft)')
    plt.ylabel('Category (0=Cheap, 1=Expensive)')
    plt.title('Classification: Discrete Prediction')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparison
    plt.subplot(1, 3, 3)
    plt.hist(house_prices, bins=20, alpha=0.7, label='Price Distribution')
    plt.axvline(price_threshold, color='red', linestyle='--', label='Classification Threshold')
    plt.xlabel('Price ($1000s)')
    plt.ylabel('Frequency')
    plt.title('Price Distribution & Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Differences:")
    print("-" * 20)
    print("Regression:")
    print("  - Output: Continuous values")
    print("  - Loss: Mean squared error")
    print("  - Interpretation: 'How much?'")
    print("  - Example: Price = $342,500")
    print()
    print("Classification:")
    print("  - Output: Discrete categories")
    print("  - Loss: Cross-entropy")
    print("  - Interpretation: 'Which category?'")
    print("  - Example: Expensive (with 85% confidence)")
    
    return mse, accuracy

if __name__ == "__main__":
    reg_class_demo = demonstrate_regression_vs_classification()

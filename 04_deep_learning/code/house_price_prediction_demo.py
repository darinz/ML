"""
House Price Prediction with Single Neuron

This module demonstrates how a single neuron can be used for house price prediction,
showing the complete process from data generation to model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def demonstrate_house_price_prediction():
    """Demonstrate single neuron for house price prediction"""
    
    # Generate house price data
    np.random.seed(42)
    house_sizes = np.random.uniform(500, 3000, 100)  # Square feet
    base_price = 50000  # Base price
    price_per_sqft = 150  # Price per square foot
    noise = np.random.normal(0, 10000, 100)  # Some noise
    
    house_prices = base_price + price_per_sqft * house_sizes + noise
    house_prices = np.maximum(house_prices, 0)  # Ensure non-negative
    
    # Train a single neuron (linear regression with ReLU)
    
    # Linear model
    linear_model = LinearRegression()
    linear_model.fit(house_sizes.reshape(-1, 1), house_prices)
    
    # Predictions
    predictions = linear_model.predict(house_sizes.reshape(-1, 1))
    predictions = np.maximum(predictions, 0)  # Apply ReLU
    
    # Calculate performance
    mse = np.mean((house_prices - predictions)**2)
    mae = np.mean(np.abs(house_prices - predictions))
    
    print(f"House Price Prediction Results:")
    print(f"Learned price per sq ft: ${linear_model.coef_[0]:.2f}")
    print(f"Learned base price: ${linear_model.intercept_:.2f}")
    print(f"MSE: ${mse:,.0f}")
    print(f"MAE: ${mae:,.0f}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(house_sizes, house_prices, alpha=0.6, s=30)
    plt.plot(house_sizes, predictions, 'r-', linewidth=2, label='Neuron Prediction')
    plt.xlabel('House Size (sq ft)')
    plt.ylabel('House Price ($)')
    plt.title('House Price vs Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    errors = house_prices - predictions
    plt.scatter(house_sizes, errors, alpha=0.6, s=30)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('House Size (sq ft)')
    plt.ylabel('Prediction Error ($)')
    plt.title('Prediction Errors')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.hist(errors, bins=20, alpha=0.7)
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show the neuron's decision process
    print(f"\nNeuron Decision Process:")
    print(f"For a 2000 sq ft house:")
    x_example = 2000
    z = linear_model.coef_[0] * x_example + linear_model.intercept_
    a = max(0, z)
    print(f"  Input (size): {x_example} sq ft")
    print(f"  Weight (price/sq ft): ${linear_model.coef_[0]:.2f}")
    print(f"  Bias (base price): ${linear_model.intercept_:.2f}")
    print(f"  Linear combination: ${z:,.2f}")
    print(f"  ReLU activation: ${a:,.2f}")
    print(f"  Final prediction: ${a:,.2f}")
    
    return linear_model, mse, mae


if __name__ == "__main__":
    house_demo = demonstrate_house_price_prediction()

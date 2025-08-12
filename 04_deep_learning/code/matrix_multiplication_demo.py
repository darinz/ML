"""
Matrix Multiplication Module Demonstration

This module demonstrates the matrix multiplication module, which is the fundamental
building block for linear transformations in neural networks.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_matrix_multiplication():
    """Demonstrate matrix multiplication module"""
    
    # Example: House price prediction
    # Features: [square_feet, bedrooms, age, location_score]
    houses = np.array([
        [2000, 3, 10, 8.5],  # House 1
        [1500, 2, 5, 7.0],   # House 2
        [3000, 4, 15, 9.0],  # House 3
        [1200, 1, 3, 6.5]    # House 4
    ])
    
    # Weight matrix: [price_per_sqft, price_per_bedroom, age_penalty, location_premium]
    W = np.array([
        [100, 5000, -1000, 2000],  # Model 1: Price-focused
        [75, 3000, -500, 1500],    # Model 2: Balanced
        [50, 2000, -200, 1000]     # Model 3: Budget-focused
    ])
    
    # Bias: Base price for each model
    b = np.array([50000, 75000, 100000])
    
    # Apply matrix multiplication module
    predictions = np.dot(houses, W.T) + b
    
    print("Matrix Multiplication Module: House Price Prediction")
    print("Input Features: [square_feet, bedrooms, age, location_score]")
    print()
    print("Houses:")
    for i, house in enumerate(houses):
        print(f"  House {i+1}: {house}")
    print()
    print("Weight Matrix (price per unit):")
    print(f"  Model 1 (Price-focused): {W[0]}")
    print(f"  Model 2 (Balanced): {W[1]}")
    print(f"  Model 3 (Budget-focused): {W[2]}")
    print()
    print("Bias (base price):", b)
    print()
    print("Predictions:")
    for i, house in enumerate(houses):
        print(f"  House {i+1}:")
        for j, model_name in enumerate(['Price-focused', 'Balanced', 'Budget-focused']):
            print(f"    {model_name}: ${predictions[i, j]:,.0f}")
        print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Show how different features contribute
    plt.subplot(1, 3, 1)
    feature_names = ['Square Feet', 'Bedrooms', 'Age', 'Location']
    model_names = ['Price-focused', 'Balanced', 'Budget-focused']
    
    for i, model_name in enumerate(model_names):
        plt.bar(feature_names, W[i], alpha=0.7, label=model_name)
    
    plt.title('Feature Weights by Model')
    plt.ylabel('Weight')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Show predictions for each house
    plt.subplot(1, 3, 2)
    x_pos = np.arange(len(houses))
    width = 0.25
    
    for i, model_name in enumerate(model_names):
        plt.bar(x_pos + i*width, predictions[:, i], width, alpha=0.7, label=model_name)
    
    plt.title('Price Predictions by Model')
    plt.xlabel('House')
    plt.ylabel('Predicted Price ($)')
    plt.xticks(x_pos + width, [f'House {i+1}' for i in range(len(houses))])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show linearity property
    plt.subplot(1, 3, 3)
    # Test linearity: double the features, double the output
    doubled_houses = houses * 2
    doubled_predictions = np.dot(doubled_houses, W.T) + b
    original_predictions = np.dot(houses, W.T) + b
    
    plt.scatter(original_predictions.flatten(), doubled_predictions.flatten(), alpha=0.7)
    plt.plot([0, max(original_predictions.flatten())], [0, max(doubled_predictions.flatten())], 'r--', label='Perfect Linearity')
    plt.xlabel('Original Predictions')
    plt.ylabel('Doubled Input Predictions')
    plt.title('Linearity Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return houses, W, b, predictions


if __name__ == "__main__":
    matrix_demo = demonstrate_matrix_multiplication()

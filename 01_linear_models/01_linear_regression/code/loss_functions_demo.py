import numpy as np
import matplotlib.pyplot as plt

def demonstrate_loss_functions():
    """Demonstrate different loss functions and their properties"""
    
    # Generate sample data
    np.random.seed(42)
    x = np.linspace(-3, 3, 100)
    y_true = 2 * x + 1 + np.random.normal(0, 0.5, 100)
    
    # Different predictions (errors)
    errors = np.linspace(-2, 2, 100)
    
    # Calculate different loss functions
    squared_error = errors**2
    absolute_error = np.abs(errors)
    huber_loss = np.where(np.abs(errors) <= 1, 0.5 * errors**2, np.abs(errors) - 0.5)
    
    print("Loss Function Comparison")
    print("=" * 40)
    print("Different ways to measure prediction errors:")
    print("1. Squared Error: Penalizes large errors heavily")
    print("2. Absolute Error: Treats all errors equally")
    print("3. Huber Loss: Combines benefits of both")
    print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Loss functions
    plt.subplot(1, 3, 1)
    plt.plot(errors, squared_error, 'b-', linewidth=2, label='Squared Error')
    plt.plot(errors, absolute_error, 'r-', linewidth=2, label='Absolute Error')
    plt.plot(errors, huber_loss, 'g-', linewidth=2, label='Huber Loss')
    plt.xlabel('Error')
    plt.ylabel('Loss')
    plt.title('Loss Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Derivatives
    plt.subplot(1, 3, 2)
    plt.plot(errors, 2 * errors, 'b-', linewidth=2, label='Squared Error Derivative')
    plt.plot(errors, np.sign(errors), 'r-', linewidth=2, label='Absolute Error Derivative')
    plt.plot(errors, np.where(np.abs(errors) <= 1, errors, np.sign(errors)), 'g-', linewidth=2, label='Huber Loss Derivative')
    plt.xlabel('Error')
    plt.ylabel('Derivative')
    plt.title('Loss Function Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Penalty comparison
    plt.subplot(1, 3, 3)
    error_sizes = [0.1, 0.5, 1.0, 2.0]
    se_penalties = [e**2 for e in error_sizes]
    ae_penalties = [abs(e) for e in error_sizes]
    
    x_pos = np.arange(len(error_sizes))
    width = 0.35
    
    plt.bar(x_pos - width/2, se_penalties, width, label='Squared Error', alpha=0.7)
    plt.bar(x_pos + width/2, ae_penalties, width, label='Absolute Error', alpha=0.7)
    plt.xlabel('Error Size')
    plt.ylabel('Penalty')
    plt.title('Penalty Comparison')
    plt.xticks(x_pos, error_sizes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Properties:")
    print("-" * 20)
    print("Squared Error:")
    print("  - Differentiable everywhere")
    print("  - Heavily penalizes large errors")
    print("  - Leads to closed-form solutions")
    print("  - Corresponds to Gaussian noise")
    print()
    print("Absolute Error:")
    print("  - Not differentiable at zero")
    print("  - Treats all errors equally")
    print("  - More robust to outliers")
    print("  - Corresponds to Laplace noise")
    print()
    print("Huber Loss:")
    print("  - Combines benefits of both")
    print("  - Robust to outliers")
    print("  - Differentiable everywhere")
    print("  - Best of both worlds")
    
    return squared_error, absolute_error, huber_loss

if __name__ == "__main__":
    loss_demo = demonstrate_loss_functions()

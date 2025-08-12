"""
Automatic Differentiation Demonstration

This module demonstrates automatic differentiation concepts by comparing forward mode
and reverse mode (backpropagation) approaches for computing gradients.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_automatic_differentiation():
    """Demonstrate automatic differentiation concepts"""
    
    # Simple example: f(x, y) = x^2 + y^3
    def forward_mode_autodiff(x, y, dx=1, dy=0):
        """Forward mode automatic differentiation"""
        # Forward pass with derivatives
        x_squared = x**2
        dx_squared = 2*x*dx  # Derivative of x^2
        
        y_cubed = y**3
        dy_cubed = 3*y**2*dy  # Derivative of y^3
        
        result = x_squared + y_cubed
        d_result = dx_squared + dy_cubed
        
        return result, d_result
    
    def reverse_mode_autodiff(x, y):
        """Reverse mode automatic differentiation (backpropagation)"""
        # Forward pass (store intermediate values)
        x_squared = x**2
        y_cubed = y**3
        result = x_squared + y_cubed
        
        # Backward pass (compute gradients)
        d_result = 1.0  # Gradient of output with respect to itself
        
        # Gradient with respect to x_squared and y_cubed
        d_x_squared = d_result * 1.0  # ∂result/∂x_squared = 1
        d_y_cubed = d_result * 1.0    # ∂result/∂y_cubed = 1
        
        # Gradient with respect to x and y
        d_x = d_x_squared * 2*x       # ∂x_squared/∂x = 2x
        d_y = d_y_cubed * 3*y**2      # ∂y_cubed/∂y = 3y^2
        
        return result, d_x, d_y
    
    # Test points
    x, y = 2.0, 3.0
    
    print("Automatic Differentiation Example: f(x, y) = x² + y³")
    print(f"Input: x = {x}, y = {y}")
    print()
    
    # Forward mode
    result_forward, d_result_dx = forward_mode_autodiff(x, y, dx=1, dy=0)
    _, d_result_dy = forward_mode_autodiff(x, y, dx=0, dy=1)
    
    print("Forward Mode Autodiff:")
    print(f"f(x, y) = {result_forward}")
    print(f"∂f/∂x = {d_result_dx}")
    print(f"∂f/∂y = {d_result_dy}")
    print()
    
    # Reverse mode
    result_reverse, d_x_reverse, d_y_reverse = reverse_mode_autodiff(x, y)
    
    print("Reverse Mode Autodiff (Backpropagation):")
    print(f"f(x, y) = {result_reverse}")
    print(f"∂f/∂x = {d_x_reverse}")
    print(f"∂f/∂y = {d_y_reverse}")
    print()
    
    # Analytical derivatives
    d_x_analytical = 2*x
    d_y_analytical = 3*y**2
    
    print("Analytical Derivatives:")
    print(f"∂f/∂x = 2x = {d_x_analytical}")
    print(f"∂f/∂y = 3y² = {d_y_analytical}")
    print()
    
    # Compare efficiency
    print("Efficiency Comparison:")
    print("Forward Mode: Requires 2 passes for 2 inputs")
    print("Reverse Mode: Requires 1 pass for all inputs")
    print("For deep learning (many inputs, few outputs):")
    print("  - Forward Mode: O(n) passes for n inputs")
    print("  - Reverse Mode: O(1) pass for all inputs")
    
    # Visualization
    x_range = np.linspace(0, 4, 50)
    y_range = np.linspace(0, 4, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + Y**3
    
    plt.figure(figsize=(15, 5))
    
    # Function surface
    plt.subplot(1, 3, 1)
    contour = plt.contour(X, Y, Z, levels=20)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.scatter(x, y, color='red', s=100, label=f'Point ({x}, {y})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function: f(x, y) = x² + y³')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gradient vectors
    plt.subplot(1, 3, 2)
    plt.contour(X, Y, Z, levels=20, alpha=0.3)
    plt.scatter(x, y, color='red', s=100)
    
    # Plot gradient vector
    plt.arrow(x, y, d_x_reverse, d_y_reverse, head_width=0.1, head_length=0.1, 
              fc='blue', ec='blue', label='Gradient')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Vector')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparison of methods
    plt.subplot(1, 3, 3)
    methods = ['Forward Mode', 'Reverse Mode', 'Analytical']
    dx_values = [d_result_dx, d_x_reverse, d_x_analytical]
    dy_values = [d_result_dy, d_y_reverse, d_y_analytical]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    plt.bar(x_pos - width/2, dx_values, width, label='∂f/∂x', alpha=0.7)
    plt.bar(x_pos + width/2, dy_values, width, label='∂f/∂y', alpha=0.7)
    
    plt.xlabel('Method')
    plt.ylabel('Gradient Value')
    plt.title('Gradient Comparison')
    plt.xticks(x_pos, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return result_forward, d_result_dx, d_result_dy, d_x_reverse, d_y_reverse


if __name__ == "__main__":
    autodiff_demo = demonstrate_automatic_differentiation()

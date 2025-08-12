"""
Computational Graph Demonstration

This module demonstrates computational graph concepts by showing how data flows
through a simple function composition and how gradients are computed backward.
"""

import numpy as np
import matplotlib.pyplot as plt


def demonstrate_computational_graph():
    """Demonstrate computational graph concepts"""
    
    # Define a simple computational graph: J = f(g(h(x)))
    def h(x):
        """First function: h(x) = x^2"""
        return x**2
    
    def g(u):
        """Second function: g(u) = sin(u)"""
        return np.sin(u)
    
    def f(v):
        """Third function: f(v) = exp(v)"""
        return np.exp(v)
    
    # Derivatives
    def h_prime(x):
        return 2*x
    
    def g_prime(u):
        return np.cos(u)
    
    def f_prime(v):
        return np.exp(v)
    
    # Test point
    x = 1.5
    
    # Forward pass
    u = h(x)
    v = g(u)
    J = f(v)
    
    print("Computational Graph Example: J = f(g(h(x)))")
    print(f"Input: x = {x}")
    print(f"Step 1: u = h(x) = {x}^2 = {u}")
    print(f"Step 2: v = g(u) = sin({u}) = {v}")
    print(f"Step 3: J = f(v) = exp({v}) = {J}")
    print()
    
    # Backward pass
    dJ_dv = f_prime(v)
    dJ_du = dJ_dv * g_prime(u)
    dJ_dx = dJ_du * h_prime(x)
    
    print("Backward Pass (Gradient Computation):")
    print(f"Step 3: dJ/dv = f'(v) = exp({v}) = {dJ_dv}")
    print(f"Step 2: dJ/du = dJ/dv × g'(u) = {dJ_dv} × cos({u}) = {dJ_du}")
    print(f"Step 1: dJ/dx = dJ/du × h'(x) = {dJ_du} × 2×{x} = {dJ_dx}")
    print()
    
    # Verify with finite differences
    epsilon = 1e-7
    J_plus = f(g(h(x + epsilon)))
    J_minus = f(g(h(x - epsilon)))
    dJ_dx_finite = (J_plus - J_minus) / (2 * epsilon)
    
    print("Verification with Finite Differences:")
    print(f"Analytical gradient: {dJ_dx}")
    print(f"Finite difference gradient: {dJ_dx_finite}")
    print(f"Relative error: {abs(dJ_dx - dJ_dx_finite) / abs(dJ_dx):.2e}")
    
    # Visualization
    x_range = np.linspace(0, 3, 100)
    u_range = h(x_range)
    v_range = g(u_range)
    J_range = f(v_range)
    
    plt.figure(figsize=(15, 5))
    
    # Function values
    plt.subplot(1, 3, 1)
    plt.plot(x_range, u_range, 'b-', label='u = h(x) = x²', linewidth=2)
    plt.plot(x_range, v_range, 'r-', label='v = g(u) = sin(u)', linewidth=2)
    plt.plot(x_range, J_range, 'g-', label='J = f(v) = exp(v)', linewidth=2)
    plt.axvline(x, color='k', linestyle='--', alpha=0.5, label=f'x = {x}')
    plt.xlabel('x')
    plt.ylabel('Function Values')
    plt.title('Forward Pass: Function Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Derivatives
    plt.subplot(1, 3, 2)
    h_prime_range = h_prime(x_range)
    g_prime_range = g_prime(u_range)
    f_prime_range = f_prime(v_range)
    
    plt.plot(x_range, h_prime_range, 'b-', label="h'(x) = 2x", linewidth=2)
    plt.plot(x_range, g_prime_range, 'r-', label="g'(u) = cos(u)", linewidth=2)
    plt.plot(x_range, f_prime_range, 'g-', label="f'(v) = exp(v)", linewidth=2)
    plt.axvline(x, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('Derivative Values')
    plt.title('Local Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Computational graph visualization
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.8, 'Computational Graph', ha='center', va='center', fontsize=14, fontweight='bold')
    plt.text(0.2, 0.6, f'x = {x}', ha='center', va='center', fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.text(0.5, 0.6, f'u = {u:.3f}', ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    plt.text(0.8, 0.6, f'v = {v:.3f}', ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    plt.text(0.5, 0.3, f'J = {J:.3f}', ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    # Arrows
    plt.arrow(0.3, 0.6, 0.15, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.6, 0.6, 0.15, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')
    plt.arrow(0.5, 0.5, 0, -0.15, head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    # Labels
    plt.text(0.35, 0.65, 'h', ha='center', va='center', fontsize=10)
    plt.text(0.65, 0.65, 'g', ha='center', va='center', fontsize=10)
    plt.text(0.55, 0.45, 'f', ha='center', va='center', fontsize=10)
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Graph Structure')
    
    plt.tight_layout()
    plt.show()
    
    return x, u, v, J, dJ_dx


if __name__ == "__main__":
    graph_demo = demonstrate_computational_graph()

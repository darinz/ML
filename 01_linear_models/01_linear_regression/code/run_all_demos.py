#!/usr/bin/env python3
"""
Main script to run all linear regression demonstrations.

This script imports and runs all the demonstration functions from the linear regression chapter.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run all linear regression demonstrations."""
    
    print("Linear Regression Demonstrations")
    print("=" * 50)
    print()
    
    # Import all demo modules
    try:
        from linear_vs_nonlinear_demo import demonstrate_linear_vs_nonlinear
        from regression_vs_classification_demo import demonstrate_regression_vs_classification
        from multiple_features_demo import demonstrate_multiple_features
        from loss_functions_demo import demonstrate_loss_functions
        from optimization_approaches_demo import demonstrate_optimization_approaches
        from gradient_descent_visualization_demo import demonstrate_gradient_descent_visualization
        from learning_rate_effects_demo import demonstrate_learning_rate_effects
        from normal_equations_vs_gradient_descent_demo import demonstrate_normal_equations_vs_gradient_descent
        from matrix_derivatives_demo import demonstrate_matrix_derivatives
        from probabilistic_thinking_demo import demonstrate_probabilistic_thinking
        from linear_assumptions_demo import demonstrate_linear_assumptions
        from global_vs_local_fitting_demo import demonstrate_global_vs_local_fitting
        from bias_variance_tradeoff_demo import demonstrate_bias_variance_tradeoff
        
        print("All modules imported successfully!")
        print()
        
        # Run demonstrations
        print("1. Linear vs. Non-linear Relationships")
        print("-" * 40)
        demonstrate_linear_vs_nonlinear()
        print("\n" + "="*50 + "\n")
        
        print("2. Regression vs. Classification")
        print("-" * 40)
        demonstrate_regression_vs_classification()
        print("\n" + "="*50 + "\n")
        
        print("3. Multiple Features")
        print("-" * 40)
        demonstrate_multiple_features()
        print("\n" + "="*50 + "\n")
        
        print("4. Loss Functions")
        print("-" * 40)
        demonstrate_loss_functions()
        print("\n" + "="*50 + "\n")
        
        print("5. Optimization Approaches")
        print("-" * 40)
        demonstrate_optimization_approaches()
        print("\n" + "="*50 + "\n")
        
        print("6. Gradient Descent Visualization")
        print("-" * 40)
        demonstrate_gradient_descent_visualization()
        print("\n" + "="*50 + "\n")
        
        print("7. Learning Rate Effects")
        print("-" * 40)
        demonstrate_learning_rate_effects()
        print("\n" + "="*50 + "\n")
        
        print("8. Normal Equations vs. Gradient Descent")
        print("-" * 40)
        demonstrate_normal_equations_vs_gradient_descent()
        print("\n" + "="*50 + "\n")
        
        print("9. Matrix Derivatives")
        print("-" * 40)
        demonstrate_matrix_derivatives()
        print("\n" + "="*50 + "\n")
        
        print("10. Probabilistic Thinking")
        print("-" * 40)
        demonstrate_probabilistic_thinking()
        print("\n" + "="*50 + "\n")
        
        print("11. Linear Assumptions")
        print("-" * 40)
        demonstrate_linear_assumptions()
        print("\n" + "="*50 + "\n")
        
        print("12. Global vs. Local Fitting")
        print("-" * 40)
        demonstrate_global_vs_local_fitting()
        print("\n" + "="*50 + "\n")
        
        print("13. Bias-Variance Trade-off")
        print("-" * 40)
        demonstrate_bias_variance_tradeoff()
        print("\n" + "="*50 + "\n")
        
        print("All demonstrations completed successfully!")
        
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all required packages are installed:")
        print("pip install numpy matplotlib scikit-learn")
        return 1
    except Exception as e:
        print(f"Error running demonstrations: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

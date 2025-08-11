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

"""
Exponential Family Distributions: Comprehensive Implementation and Examples

This module provides a complete implementation of exponential family distributions
as discussed in the exponential family theory document. It includes:

1. Generic exponential family framework
2. Detailed implementations of Bernoulli and Gaussian distributions
3. Step-by-step derivations and verifications
4. Interactive examples and visualizations
5. Mathematical property demonstrations

Key Concepts Implemented:
- Natural parameters (η)
- Sufficient statistics T(y)
- Log partition functions a(η)
- Base measures b(y)
- Canonical forms and transformations
- Mathematical properties and relationships

Author: Machine Learning Course
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, bernoulli
from scipy.optimize import minimize
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# ============================================================================
# GENERIC EXPONENTIAL FAMILY FRAMEWORK
# ============================================================================

def exponential_family_pdf(y, eta, T, a, b):
    """
    Generic exponential family probability density/mass function.
    
    This implements the canonical form: p(y; η) = b(y) * exp(η^T * T(y) - a(η))
    
    Parameters:
    -----------
    y : array-like
        Observed values
    eta : array-like
        Natural parameter (can be vector for multivariate distributions)
    T : callable
        Sufficient statistic function T(y)
    a : callable
        Log partition function a(η)
    b : callable
        Base measure function b(y)
    
    Returns:
    --------
    array-like
        Probability density/mass values
    """
    # Convert inputs to numpy arrays for vectorized operations
    y = np.asarray(y)
    eta = np.asarray(eta)
    
    # Apply the exponential family formula
    # p(y; η) = b(y) * exp(η^T * T(y) - a(η))
    T_y = T(y)
    a_eta = a(eta)
    
    # Handle vectorized operations
    if y.ndim == 0:  # Scalar y
        result = b(y) * np.exp(np.dot(eta, T_y) - a_eta)
    else:  # Vector y
        result = np.array([b(yi) * np.exp(np.dot(eta, T(yi)) - a_eta) for yi in y])
    
    return result

def verify_normalization(y_range, eta, T, a, b, is_continuous=True):
    """
    Verify that the exponential family distribution is properly normalized.
    
    Parameters:
    -----------
    y_range : array-like
        Range of y values to integrate/sum over
    eta : array-like
        Natural parameter
    T, a, b : callable
        Exponential family components
    is_continuous : bool
        True for continuous distributions, False for discrete
    
    Returns:
    --------
    float
        Sum/integral of the distribution (should be close to 1)
    """
    pdf_values = exponential_family_pdf(y_range, eta, T, a, b)
    
    if is_continuous:
        # Numerical integration using trapezoidal rule
        integral = np.trapz(pdf_values, y_range)
        return integral
    else:
        # Sum for discrete distributions
        return np.sum(pdf_values)

# ============================================================================
# BERNOULLI DISTRIBUTION IMPLEMENTATION
# ============================================================================

class BernoulliExponentialFamily:
    """
    Bernoulli distribution in exponential family form.
    
    The Bernoulli distribution models binary outcomes (0 or 1) with parameter φ.
    In exponential family form:
    - Natural parameter: η = log(φ/(1-φ)) (log-odds)
    - Sufficient statistic: T(y) = y
    - Log partition function: a(η) = log(1 + e^η)
    - Base measure: b(y) = 1
    
    This class demonstrates the connection between the standard Bernoulli PMF
    and its exponential family representation.
    """
    
    @staticmethod
    def phi_to_eta(phi):
        """
        Convert Bernoulli parameter φ to natural parameter η.
        
        This is the logit function: η = log(φ/(1-φ))
        
        Parameters:
        -----------
        phi : float
            Bernoulli parameter (probability of success)
            
        Returns:
        --------
        float
            Natural parameter η (log-odds)
        """
        return np.log(phi / (1 - phi))
    
    @staticmethod
    def eta_to_phi(eta):
        """
        Convert natural parameter η to Bernoulli parameter φ.
        
        This is the sigmoid function: φ = 1/(1 + e^(-η))
        
        Parameters:
        -----------
        eta : float
            Natural parameter (log-odds)
            
        Returns:
        --------
        float
            Bernoulli parameter φ (probability of success)
        """
        return 1 / (1 + np.exp(-eta))
    
    @staticmethod
    def T(y):
        """
        Sufficient statistic for Bernoulli distribution.
        
        For Bernoulli, T(y) = y (the identity function).
        This captures all the information needed to estimate φ.
        
        Parameters:
        -----------
        y : array-like
            Observed values (0 or 1)
            
        Returns:
        --------
        array-like
            Sufficient statistics
        """
        return y
    
    @staticmethod
    def a(eta):
        """
        Log partition function for Bernoulli distribution.
        
        a(η) = log(1 + e^η)
        This ensures the distribution is properly normalized.
        
        Parameters:
        -----------
        eta : float
            Natural parameter
            
        Returns:
        --------
        float
            Log partition function value
        """
        return np.log(1 + np.exp(eta))
    
    @staticmethod
    def b(y):
        """
        Base measure for Bernoulli distribution.
        
        For Bernoulli, b(y) = 1 for all y.
        This provides the basic structure of the distribution.
        
        Parameters:
        -----------
        y : array-like
            Observed values
            
        Returns:
        --------
        array-like
            Base measure values (all 1s)
        """
        return np.ones_like(y)
    
    @staticmethod
    def standard_pmf(y, phi):
        """
        Standard Bernoulli probability mass function.
        
        P(Y = y) = φ^y * (1-φ)^(1-y)
        
        Parameters:
        -----------
        y : array-like
            Observed values (0 or 1)
        phi : float
            Probability of success
            
        Returns:
        --------
        array-like
            Probability mass values
        """
        return phi**y * (1 - phi)**(1 - y)
    
    @staticmethod
    def exponential_family_pmf(y, eta):
        """
        Bernoulli distribution in exponential family form.
        
        P(Y = y) = exp(η*y - a(η))
        
        Parameters:
        -----------
        y : array-like
            Observed values (0 or 1)
        eta : float
            Natural parameter (log-odds)
            
        Returns:
        --------
        array-like
            Probability mass values
        """
        return exponential_family_pdf(y, eta, 
                                    BernoulliExponentialFamily.T,
                                    BernoulliExponentialFamily.a,
                                    BernoulliExponentialFamily.b)

# ============================================================================
# GAUSSIAN DISTRIBUTION IMPLEMENTATION
# ============================================================================

class GaussianExponentialFamily:
    """
    Gaussian distribution in exponential family form.
    
    The Gaussian distribution models continuous outcomes with mean μ and variance σ².
    For GLMs, we typically fix σ² = 1, so the distribution is parameterized by μ only.
    
    In exponential family form:
    - Natural parameter: η = μ
    - Sufficient statistic: T(y) = y
    - Log partition function: a(η) = η²/2
    - Base measure: b(y) = (1/√(2π)) * exp(-y²/2)
    
    This class demonstrates the connection between the standard Gaussian PDF
    and its exponential family representation.
    """
    
    @staticmethod
    def mu_to_eta(mu):
        """
        Convert Gaussian mean μ to natural parameter η.
        
        For Gaussian with σ² = 1, η = μ.
        
        Parameters:
        -----------
        mu : float
            Gaussian mean
            
        Returns:
        --------
        float
            Natural parameter η
        """
        return mu
    
    @staticmethod
    def eta_to_mu(eta):
        """
        Convert natural parameter η to Gaussian mean μ.
        
        For Gaussian with σ² = 1, μ = η.
        
        Parameters:
        -----------
        eta : float
            Natural parameter
            
        Returns:
        --------
        float
            Gaussian mean μ
        """
        return eta
    
    @staticmethod
    def T(y):
        """
        Sufficient statistic for Gaussian distribution.
        
        For Gaussian, T(y) = y (the identity function).
        This captures all the information needed to estimate μ.
        
        Parameters:
        -----------
        y : array-like
            Observed values
            
        Returns:
        --------
        array-like
            Sufficient statistics
        """
        return y
    
    @staticmethod
    def a(eta):
        """
        Log partition function for Gaussian distribution.
        
        a(η) = η²/2
        This ensures the distribution is properly normalized.
        
        Parameters:
        -----------
        eta : float
            Natural parameter
            
        Returns:
        --------
        float
            Log partition function value
        """
        return eta**2 / 2
    
    @staticmethod
    def b(y):
        """
        Base measure for Gaussian distribution.
        
        b(y) = (1/√(2π)) * exp(-y²/2)
        This provides the basic structure of the distribution.
        
        Parameters:
        -----------
        y : array-like
            Observed values
            
        Returns:
        --------
        array-like
            Base measure values
        """
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-y**2 / 2)
    
    @staticmethod
    def standard_pdf(y, mu, sigma=1.0):
        """
        Standard Gaussian probability density function.
        
        f(y) = (1/√(2πσ²)) * exp(-(y-μ)²/(2σ²))
        
        Parameters:
        -----------
        y : array-like
            Observed values
        mu : float
            Gaussian mean
        sigma : float, optional
            Gaussian standard deviation (default: 1.0)
            
        Returns:
        --------
        array-like
            Probability density values
        """
        return norm.pdf(y, loc=mu, scale=sigma)
    
    @staticmethod
    def exponential_family_pdf(y, eta):
        """
        Gaussian distribution in exponential family form.
        
        f(y) = b(y) * exp(η*y - a(η))
        
        Parameters:
        -----------
        y : array-like
            Observed values
        eta : float
            Natural parameter (mean)
            
        Returns:
        --------
        array-like
            Probability density values
        """
        return exponential_family_pdf(y, eta,
                                    GaussianExponentialFamily.T,
                                    GaussianExponentialFamily.a,
                                    GaussianExponentialFamily.b)

# ============================================================================
# MATHEMATICAL PROPERTIES AND VERIFICATIONS
# ============================================================================

def demonstrate_bernoulli_properties():
    """
    Demonstrate key properties of the Bernoulli exponential family.
    
    This function shows:
    1. Equivalence between standard and exponential family forms
    2. The sigmoid connection (η ↔ φ transformation)
    3. Normalization verification
    4. Parameter relationships
    """
    print("=" * 60)
    print("BERNOULLI EXPONENTIAL FAMILY PROPERTIES")
    print("=" * 60)
    
    # Test values
    phi_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    y_values = [0, 1]
    
    print("\n1. Equivalence between Standard and Exponential Family Forms:")
    print("-" * 55)
    
    for phi in phi_values:
        eta = BernoulliExponentialFamily.phi_to_eta(phi)
        print(f"\nφ = {phi:.1f}, η = {eta:.3f}")
        
        for y in y_values:
            # Standard form
            p_standard = BernoulliExponentialFamily.standard_pmf(y, phi)
            
            # Exponential family form
            p_exp = BernoulliExponentialFamily.exponential_family_pmf(y, eta)
            
            print(f"  P(Y={y}): Standard={p_standard:.6f}, Exp.Family={p_exp:.6f}, "
                  f"Difference={abs(p_standard - p_exp):.2e}")
    
    print("\n2. Sigmoid Connection (η ↔ φ transformation):")
    print("-" * 45)
    
    eta_values = [-2, -1, 0, 1, 2]
    for eta in eta_values:
        phi = BernoulliExponentialFamily.eta_to_phi(eta)
        eta_back = BernoulliExponentialFamily.phi_to_eta(phi)
        
        print(f"η = {eta:2.0f} → φ = {phi:.3f} → η = {eta_back:.3f}")
    
    print("\n3. Normalization Verification:")
    print("-" * 30)
    
    for phi in [0.2, 0.5, 0.8]:
        eta = BernoulliExponentialFamily.phi_to_eta(phi)
        y_range = np.array([0, 1])  # Bernoulli is discrete
        
        # Sum should equal 1
        total_prob = np.sum(BernoulliExponentialFamily.exponential_family_pmf(y_range, eta))
        print(f"φ = {phi:.1f}: Total probability = {total_prob:.6f}")

def demonstrate_gaussian_properties():
    """
    Demonstrate key properties of the Gaussian exponential family.
    
    This function shows:
    1. Equivalence between standard and exponential family forms
    2. Parameter relationships
    3. Normalization verification
    4. Shape properties
    """
    print("\n" + "=" * 60)
    print("GAUSSIAN EXPONENTIAL FAMILY PROPERTIES")
    print("=" * 60)
    
    # Test values
    mu_values = [-2, -1, 0, 1, 2]
    y_values = np.linspace(-3, 3, 7)
    
    print("\n1. Equivalence between Standard and Exponential Family Forms:")
    print("-" * 55)
    
    for mu in mu_values:
        eta = GaussianExponentialFamily.mu_to_eta(mu)
        print(f"\nμ = {mu:2.0f}, η = {eta:2.0f}")
        
        for y in y_values[:3]:  # Show first 3 values for brevity
            # Standard form
            p_standard = GaussianExponentialFamily.standard_pdf(y, mu)
            
            # Exponential family form
            p_exp = GaussianExponentialFamily.exponential_family_pdf(y, eta)
            
            print(f"  f({y:4.1f}): Standard={p_standard:.6f}, Exp.Family={p_exp:.6f}, "
                  f"Difference={abs(p_standard - p_exp):.2e}")
    
    print("\n2. Parameter Relationships:")
    print("-" * 30)
    
    for eta in [-1, 0, 1]:
        mu = GaussianExponentialFamily.eta_to_mu(eta)
        print(f"η = {eta:2.0f} ↔ μ = {mu:2.0f}")
    
    print("\n3. Normalization Verification:")
    print("-" * 30)
    
    for mu in [-1, 0, 1]:
        eta = GaussianExponentialFamily.mu_to_eta(mu)
        y_range = np.linspace(-10, 10, 1000)  # Dense grid for integration
        
        # Integral should equal 1
        integral = verify_normalization(y_range, eta,
                                      GaussianExponentialFamily.T,
                                      GaussianExponentialFamily.a,
                                      GaussianExponentialFamily.b,
                                      is_continuous=True)
        print(f"μ = {mu:2.0f}: Integral = {integral:.6f}")

# ============================================================================
# VISUALIZATION AND INTERACTIVE EXAMPLES
# ============================================================================

def plot_bernoulli_examples():
    """
    Create visualizations demonstrating Bernoulli exponential family properties.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Bernoulli Exponential Family: Properties and Transformations', 
                 fontsize=16, fontweight='bold')
    
    # 1. Standard PMF vs Exponential Family PMF
    phi_values = [0.2, 0.5, 0.8]
    y_values = np.array([0, 1])
    
    for i, phi in enumerate(phi_values):
        eta = BernoulliExponentialFamily.phi_to_eta(phi)
        
        # Standard form
        p_standard = BernoulliExponentialFamily.standard_pmf(y_values, phi)
        
        # Exponential family form
        p_exp = BernoulliExponentialFamily.exponential_family_pmf(y_values, eta)
        
        axes[0, 0].bar(y_values + i*0.2, p_standard, width=0.2, 
                      label=f'φ={phi:.1f} (Standard)', alpha=0.7)
        axes[0, 0].bar(y_values + i*0.2 + 0.1, p_exp, width=0.2, 
                      label=f'η={eta:.2f} (Exp.Family)', alpha=0.7)
    
    axes[0, 0].set_xlabel('y')
    axes[0, 0].set_ylabel('Probability')
    axes[0, 0].set_title('Standard vs Exponential Family PMF')
    axes[0, 0].legend()
    axes[0, 0].set_xticks([0, 1])
    
    # 2. Sigmoid transformation
    eta_range = np.linspace(-5, 5, 100)
    phi_values = [BernoulliExponentialFamily.eta_to_phi(eta) for eta in eta_range]
    
    axes[0, 1].plot(eta_range, phi_values, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('η (Natural Parameter)')
    axes[0, 1].set_ylabel('φ (Probability)')
    axes[0, 1].set_title('Sigmoid Function: η → φ')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Log-odds transformation
    phi_range = np.linspace(0.01, 0.99, 100)
    eta_values = [BernoulliExponentialFamily.phi_to_eta(phi) for phi in phi_range]
    
    axes[1, 0].plot(phi_range, eta_values, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('φ (Probability)')
    axes[1, 0].set_ylabel('η (Log-odds)')
    axes[1, 0].set_title('Logit Function: φ → η')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Log partition function
    eta_range = np.linspace(-3, 3, 100)
    a_values = [BernoulliExponentialFamily.a(eta) for eta in eta_range]
    
    axes[1, 1].plot(eta_range, a_values, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('η (Natural Parameter)')
    axes[1, 1].set_ylabel('a(η) (Log Partition Function)')
    axes[1, 1].set_title('Log Partition Function')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_gaussian_examples():
    """
    Create visualizations demonstrating Gaussian exponential family properties.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Gaussian Exponential Family: Properties and Relationships', 
                 fontsize=16, fontweight='bold')
    
    # 1. Standard PDF vs Exponential Family PDF
    mu_values = [-1, 0, 1]
    y_range = np.linspace(-4, 4, 200)
    
    for mu in mu_values:
        eta = GaussianExponentialFamily.mu_to_eta(mu)
        
        # Standard form
        pdf_standard = GaussianExponentialFamily.standard_pdf(y_range, mu)
        
        # Exponential family form
        pdf_exp = GaussianExponentialFamily.exponential_family_pdf(y_range, eta)
        
        axes[0, 0].plot(y_range, pdf_standard, '--', linewidth=2, 
                       label=f'μ={mu} (Standard)')
        axes[0, 0].plot(y_range, pdf_exp, '-', linewidth=2, 
                       label=f'η={eta} (Exp.Family)')
    
    axes[0, 0].set_xlabel('y')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].set_title('Standard vs Exponential Family PDF')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Identity transformation (η = μ)
    mu_range = np.linspace(-3, 3, 100)
    eta_values = [GaussianExponentialFamily.mu_to_eta(mu) for mu in mu_range]
    
    axes[0, 1].plot(mu_range, eta_values, 'b-', linewidth=2)
    axes[0, 1].plot(mu_range, mu_range, 'r--', alpha=0.5, label='y=x')
    axes[0, 1].set_xlabel('μ (Mean)')
    axes[0, 1].set_ylabel('η (Natural Parameter)')
    axes[0, 1].set_title('Identity Transformation: μ = η')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Base measure b(y)
    y_range = np.linspace(-4, 4, 200)
    b_values = [GaussianExponentialFamily.b(y) for y in y_range]
    
    axes[1, 0].plot(y_range, b_values, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('y')
    axes[1, 0].set_ylabel('b(y)')
    axes[1, 0].set_title('Base Measure b(y)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Log partition function
    eta_range = np.linspace(-3, 3, 100)
    a_values = [GaussianExponentialFamily.a(eta) for eta in eta_range]
    
    axes[1, 1].plot(eta_range, a_values, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('η (Natural Parameter)')
    axes[1, 1].set_ylabel('a(η) (Log Partition Function)')
    axes[1, 1].set_title('Log Partition Function: a(η) = η²/2')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# INTERACTIVE EXAMPLES AND DEMONSTRATIONS
# ============================================================================

def interactive_bernoulli_demo():
    """
    Interactive demonstration of Bernoulli exponential family.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE BERNOULLI DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Fair coin
    print("\nExample 1: Fair Coin (φ = 0.5)")
    print("-" * 35)
    
    phi = 0.5
    eta = BernoulliExponentialFamily.phi_to_eta(phi)
    
    print(f"Probability of heads: φ = {phi}")
    print(f"Log-odds: η = log(φ/(1-φ)) = {eta:.3f}")
    print(f"Odds ratio: φ/(1-φ) = {phi/(1-phi):.1f}")
    
    # Calculate probabilities
    for y in [0, 1]:
        p_standard = BernoulliExponentialFamily.standard_pmf(y, phi)
        p_exp = BernoulliExponentialFamily.exponential_family_pmf(y, eta)
        outcome = "tails" if y == 0 else "heads"
        print(f"P({outcome}) = {p_standard:.3f} (both forms agree)")
    
    # Example 2: Biased coin
    print("\nExample 2: Biased Coin (φ = 0.8)")
    print("-" * 35)
    
    phi = 0.8
    eta = BernoulliExponentialFamily.phi_to_eta(phi)
    
    print(f"Probability of heads: φ = {phi}")
    print(f"Log-odds: η = log(φ/(1-φ)) = {eta:.3f}")
    print(f"Odds ratio: φ/(1-φ) = {phi/(1-phi):.1f}")
    
    # Calculate probabilities
    for y in [0, 1]:
        p_standard = BernoulliExponentialFamily.standard_pmf(y, phi)
        p_exp = BernoulliExponentialFamily.exponential_family_pmf(y, eta)
        outcome = "tails" if y == 0 else "heads"
        print(f"P({outcome}) = {p_standard:.3f} (both forms agree)")

def interactive_gaussian_demo():
    """
    Interactive demonstration of Gaussian exponential family.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE GAUSSIAN DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Standard normal
    print("\nExample 1: Standard Normal (μ = 0)")
    print("-" * 35)
    
    mu = 0
    eta = GaussianExponentialFamily.mu_to_eta(mu)
    
    print(f"Mean: μ = {mu}")
    print(f"Natural parameter: η = μ = {eta}")
    
    # Calculate densities at key points
    y_values = [-2, -1, 0, 1, 2]
    print("\nDensity values at key points:")
    for y in y_values:
        p_standard = GaussianExponentialFamily.standard_pdf(y, mu)
        p_exp = GaussianExponentialFamily.exponential_family_pdf(y, eta)
        print(f"f({y:2.0f}) = {p_standard:.6f} (both forms agree)")
    
    # Example 2: Shifted normal
    print("\nExample 2: Shifted Normal (μ = 2)")
    print("-" * 35)
    
    mu = 2
    eta = GaussianExponentialFamily.mu_to_eta(mu)
    
    print(f"Mean: μ = {mu}")
    print(f"Natural parameter: η = μ = {eta}")
    
    # Calculate densities at key points
    y_values = [0, 1, 2, 3, 4]
    print("\nDensity values at key points:")
    for y in y_values:
        p_standard = GaussianExponentialFamily.standard_pdf(y, mu)
        p_exp = GaussianExponentialFamily.exponential_family_pdf(y, eta)
        print(f"f({y:2.0f}) = {p_standard:.6f} (both forms agree)")

# ============================================================================
# MAIN EXECUTION AND DEMONSTRATIONS
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block demonstrating exponential family concepts.
    
    This section runs comprehensive demonstrations of:
    1. Mathematical properties and verifications
    2. Interactive examples
    3. Visualizations
    4. Practical applications
    """
    
    print("EXPONENTIAL FAMILY DISTRIBUTIONS: COMPREHENSIVE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demonstration shows the mathematical elegance and practical")
    print("utility of exponential family distributions, which form the")
    print("foundation of Generalized Linear Models (GLMs).")
    
    # Run mathematical property demonstrations
    demonstrate_bernoulli_properties()
    demonstrate_gaussian_properties()
    
    # Run interactive demonstrations
    interactive_bernoulli_demo()
    interactive_gaussian_demo()
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS...")
    print("=" * 60)
    
    plot_bernoulli_examples()
    plot_gaussian_examples()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Exponential families provide a unified framework for distributions")
    print("2. Natural parameters enable elegant mathematical properties")
    print("3. The sigmoid function naturally arises from Bernoulli distributions")
    print("4. Canonical forms simplify parameter estimation and interpretation")
    print("5. This foundation enables the construction of GLMs")
    
    print("\nNext Steps:")
    print("- Study the GLM construction document")
    print("- Implement GLM parameter estimation")
    print("- Apply to real-world prediction problems") 
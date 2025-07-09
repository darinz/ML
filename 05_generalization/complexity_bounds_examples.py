"""
Sample Complexity Bounds and Theoretical Foundations Examples
============================================================

This module demonstrates the theoretical foundations of generalization in machine learning,
including concentration inequalities, sample complexity bounds, and the VC dimension.

Key Concepts:
1. Hoeffding/Chernoff Bounds: Provide probabilistic guarantees for sample mean convergence
2. Union Bound: Bounds probability of multiple rare events
3. Empirical Risk vs. Generalization Error: The gap between training and test performance
4. Sample Complexity: How many samples are needed for good generalization
5. VC Dimension: Measure of model complexity for classification problems

These concepts correspond to the theoretical foundations discussed in Section 8.3 of the markdown file.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from typing import List, Tuple, Callable, Optional
import itertools

# Set random seed for reproducibility
np.random.seed(42)

def demonstrate_hoeffding_bound(
    phi: float = 0.6,
    n: int = 100,
    gamma: float = 0.1,
    n_trials: int = 10000
) -> None:
    """
    Demonstrate the Hoeffding bound for Bernoulli random variables.
    
    The Hoeffding bound states that for independent random variables X₁, ..., Xₙ
    with 0 ≤ Xᵢ ≤ 1, and their sample mean X̄ = (1/n)∑ᵢXᵢ:
    
        P(|X̄ - E[X̄]| > γ) ≤ 2exp(-2γ²n)
    
    This corresponds to equation (8.8) in the markdown file.
    
    Args:
        phi: True probability of success for Bernoulli(φ)
        n: Number of samples
        gamma: Deviation threshold
        n_trials: Number of Monte Carlo trials
    """
    print("=" * 60)
    print("DEMONSTRATION: HOEFFDING BOUND")
    print("=" * 60)
    print(f"Testing P(|X̄ - φ| > {gamma}) for Xᵢ ~ Bernoulli({phi})")
    print(f"Sample size: n = {n}")
    print(f"Number of trials: {n_trials}")
    print()
    
    # Simulate the probability of large deviations
    deviations = []
    for trial in range(n_trials):
        # Generate n Bernoulli samples
        samples = bernoulli.rvs(phi, size=n)
        sample_mean = np.mean(samples)
        
        # Check if deviation exceeds threshold
        deviation = abs(sample_mean - phi) > gamma
        deviations.append(deviation)
    
    # Compute empirical probability
    empirical_prob = np.mean(deviations)
    
    # Compute Hoeffding bound
    hoeffding_bound = 2 * np.exp(-2 * gamma**2 * n)
    
    # Print results
    print(f"Empirical P(|X̄ - φ| > {gamma}): {empirical_prob:.6f}")
    print(f"Hoeffding bound:                    {hoeffding_bound:.6f}")
    print(f"Bound is tight:                     {empirical_prob <= hoeffding_bound}")
    print()
    
    # Demonstrate how the bound changes with n
    print("Hoeffding bound as a function of sample size:")
    ns = [10, 25, 50, 100, 200, 500, 1000]
    for n_test in ns:
        bound = 2 * np.exp(-2 * gamma**2 * n_test)
        print(f"  n = {n_test:4d}: P > {gamma} ≤ {bound:.6f}")
    
    # Plot the relationship
    plt.figure(figsize=(10, 6))
    ns_plot = np.logspace(1, 3, 100)
    bounds = 2 * np.exp(-2 * gamma**2 * ns_plot)
    
    plt.semilogy(ns_plot, bounds, 'b-', linewidth=2, label='Hoeffding Bound')
    plt.axhline(y=empirical_prob, color='red', linestyle='--', alpha=0.7, 
                label=f'Empirical Probability ({empirical_prob:.4f})')
    
    plt.xlabel('Sample Size (n)', fontsize=12)
    plt.ylabel('Probability Bound', fontsize=12)
    plt.title(f'Hoeffding Bound: P(|X̄ - {phi}| > {gamma}) ≤ 2exp(-2({gamma}²)n)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def demonstrate_union_bound(p_single: float = 0.01, k: int = 10) -> None:
    """
    Demonstrate the union bound for multiple events.
    
    The union bound states that for any events A₁, A₂, ..., Aₖ:
        P(A₁ ∪ A₂ ∪ ... ∪ Aₖ) ≤ P(A₁) + P(A₂) + ... + P(Aₖ)
    
    This is a fundamental tool in probability theory used extensively in
    machine learning theory to bound probabilities of multiple rare events.
    
    Args:
        p_single: Probability of a single event
        k: Number of events
    """
    print("=" * 60)
    print("DEMONSTRATION: UNION BOUND")
    print("=" * 60)
    print(f"Testing P(A₁ ∪ A₂ ∪ ... ∪ Aₖ) where P(Aᵢ) = {p_single} for all i")
    print(f"Number of events: k = {k}")
    print()
    
    # Compute true probability using inclusion-exclusion
    # For independent events: P(union) = 1 - (1 - p)^k
    prob_union = 1 - (1 - p_single) ** k
    
    # Union bound
    union_bound = k * p_single
    
    print(f"True probability of at least one event: {prob_union:.6f}")
    print(f"Union bound:                          {union_bound:.6f}")
    print(f"Bound is tight:                       {prob_union <= union_bound}")
    print(f"Gap:                                  {union_bound - prob_union:.6f}")
    print()
    
    # Demonstrate how the gap changes with k
    print("Union bound vs. true probability for different k:")
    ks = [1, 2, 5, 10, 20, 50, 100]
    for k_test in ks:
        true_prob = 1 - (1 - p_single) ** k_test
        bound = k_test * p_single
        gap = bound - true_prob
        print(f"  k = {k_test:3d}: True = {true_prob:.4f}, Bound = {bound:.4f}, Gap = {gap:.4f}")
    
    # Plot the relationship
    plt.figure(figsize=(10, 6))
    ks_plot = np.arange(1, 101)
    true_probs = 1 - (1 - p_single) ** ks_plot
    bounds = ks_plot * p_single
    
    plt.plot(ks_plot, true_probs, 'b-', linewidth=2, label='True Probability')
    plt.plot(ks_plot, bounds, 'r--', linewidth=2, label='Union Bound')
    
    plt.xlabel('Number of Events (k)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'Union Bound: P(∪ᵢAᵢ) ≤ ∑ᵢP(Aᵢ) = {p_single}k', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def demonstrate_empirical_vs_generalization_error(
    phi: float = 0.7,
    n: int = 50,
    n_test: int = 10000
) -> None:
    """
    Demonstrate the difference between empirical risk and generalization error.
    
    This shows how training error (empirical risk) can be misleading as a
    measure of true performance (generalization error), especially with
    limited data or complex models.
    
    Args:
        phi: True probability of positive class
        n: Number of training samples
        n_test: Number of test samples
    """
    print("=" * 60)
    print("DEMONSTRATION: EMPIRICAL RISK vs GENERALIZATION ERROR")
    print("=" * 60)
    print(f"True probability of positive class: φ = {phi}")
    print(f"Training set size: n = {n}")
    print(f"Test set size: {n_test}")
    print()
    
    # Generate training and test data
    y_train = bernoulli.rvs(phi, size=n)
    y_test = bernoulli.rvs(phi, size=n_test)
    
    # Define different classifiers
    classifiers = {
        'Always Predict 1': lambda x: 1,
        'Always Predict 0': lambda x: 0,
        'Random Classifier': lambda x: np.random.choice([0, 1]),
        'Majority Class': lambda x: 1 if phi > 0.5 else 0
    }
    
    print("Classifier Performance Comparison:")
    print("-" * 50)
    
    for name, classifier in classifiers.items():
        # Compute empirical risk (training error)
        train_predictions = [classifier(None) for _ in range(n)]
        train_error = np.mean(np.array(train_predictions) != y_train)
        
        # Compute generalization error (test error)
        test_predictions = [classifier(None) for _ in range(n_test)]
        test_error = np.mean(np.array(test_predictions) != y_test)
        
        # Compute generalization gap
        generalization_gap = abs(train_error - test_error)
        
        print(f"{name}:")
        print(f"  Empirical Risk (Training Error): {train_error:.4f}")
        print(f"  Generalization Error (Test Error): {test_error:.4f}")
        print(f"  Generalization Gap: {generalization_gap:.4f}")
        print()
    
    # Demonstrate the effect of sample size on generalization gap
    print("Effect of training set size on generalization gap:")
    print("-" * 50)
    
    ns = [10, 25, 50, 100, 200, 500]
    gaps = []
    
    for n_train in ns:
        y_train_small = bernoulli.rvs(phi, size=n_train)
        train_predictions = [1 for _ in range(n_train)]  # Always predict 1
        train_error = np.mean(np.array(train_predictions) != y_train_small)
        test_error = 1 - phi  # True error for always predicting 1
        gap = abs(train_error - test_error)
        gaps.append(gap)
        
        print(f"  n = {n_train:3d}: Gap = {gap:.4f}")
    
    # Plot the relationship
    plt.figure(figsize=(10, 6))
    plt.plot(ns, gaps, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Training Set Size (n)', fontsize=12)
    plt.ylabel('Generalization Gap', fontsize=12)
    plt.title('Generalization Gap vs. Training Set Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def demonstrate_sample_complexity_bounds(
    gamma: float = 0.1,
    delta: float = 0.05,
    k_max: int = 1000
) -> None:
    """
    Demonstrate sample complexity bounds for finite hypothesis classes.
    
    For a finite hypothesis class H with |H| = k, the sample complexity bound is:
        n ≥ (1/(2γ²)) * log(2k/δ)
    
    This ensures that with probability at least 1-δ, the empirical risk
    is within γ of the true risk for all hypotheses in H.
    
    Args:
        gamma: Accuracy parameter
        delta: Confidence parameter
        k_max: Maximum number of hypotheses to consider
    """
    print("=" * 60)
    print("DEMONSTRATION: SAMPLE COMPLEXITY BOUNDS")
    print("=" * 60)
    print(f"Accuracy parameter: γ = {gamma}")
    print(f"Confidence parameter: δ = {delta}")
    print(f"Sample complexity bound: n ≥ (1/(2γ²)) * log(2k/δ)")
    print()
    
    # Compute sample complexity for different k values
    ks = np.arange(1, k_max + 1)
    n_bounds = (1 / (2 * gamma**2)) * np.log(2 * ks / delta)
    
    # Print some key values
    key_ks = [1, 10, 100, 1000]
    print("Sample complexity for different hypothesis class sizes:")
    for k in key_ks:
        n_bound = (1 / (2 * gamma**2)) * np.log(2 * k / delta)
        print(f"  |H| = {k:4d}: n ≥ {n_bound:.1f}")
    
    # Plot the relationship
    plt.figure(figsize=(10, 6))
    plt.plot(ks, n_bounds, 'b-', linewidth=2)
    
    # Add horizontal line for reference
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='n = 100')
    
    plt.xlabel('Number of Hypotheses (|H|)', fontsize=12)
    plt.ylabel('Required Sample Size (n)', fontsize=12)
    plt.title(f'Sample Complexity Bound: n ≥ (1/(2{gamma}²)) * log(2k/{delta})', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Demonstrate the effect of γ and δ
    print("\nEffect of accuracy and confidence parameters:")
    print("-" * 50)
    
    gammas = [0.05, 0.1, 0.2]
    deltas = [0.01, 0.05, 0.1]
    k = 100
    
    for gamma_test in gammas:
        for delta_test in deltas:
            n_bound = (1 / (2 * gamma_test**2)) * np.log(2 * k / delta_test)
            print(f"  γ = {gamma_test:.2f}, δ = {delta_test:.2f}: n ≥ {n_bound:.1f}")

def demonstrate_vc_dimension_2d() -> None:
    """
    Demonstrate VC dimension for linear classifiers in 2D.
    
    The VC dimension of linear classifiers in 2D is 3, meaning:
    1. Any 3 points can be shattered (all 2³ = 8 labelings are possible)
    2. No set of 4 points can be shattered (some labelings are impossible)
    
    This demonstrates the concept of VC dimension as a measure of model complexity.
    """
    print("=" * 60)
    print("DEMONSTRATION: VC DIMENSION FOR LINEAR CLASSIFIERS IN 2D")
    print("=" * 60)
    print("VC dimension = 3 for linear classifiers in 2D")
    print("This means any 3 points can be shattered, but no 4 points can be shattered.")
    print()
    
    # Define 3 points that can be shattered
    points = np.array([[0, 0], [1, 1], [2, 0]])
    
    # Generate all possible labelings (2³ = 8)
    all_labelings = list(itertools.product([0, 1], repeat=3))
    
    print("All 8 possible labelings of 3 points:")
    for i, labeling in enumerate(all_labelings):
        print(f"  Labeling {i+1}: {labeling}")
    
    # Visualize all labelings
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (ax, labeling) in enumerate(zip(axes, all_labelings)):
        # Plot points with different colors for different labels
        for j, (point, label) in enumerate(zip(points, labeling)):
            color = 'blue' if label == 1 else 'red'
            marker = 'o' if label == 1 else 'x'
            ax.scatter(point[0], point[1], c=color, s=100, marker=marker, alpha=0.7)
        
        # Add point labels
        for j, point in enumerate(points):
            ax.annotate(f'P{j+1}', (point[0], point[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10)
        
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Labeling {i+1}: {labeling}')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('VC Dimension = 3: All 8 Labelings of 3 Points are Possible', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Demonstrate that 4 points cannot be shattered
    print("\nDemonstrating that 4 points cannot be shattered:")
    print("-" * 50)
    
    # Add a fourth point
    points_4 = np.array([[0, 0], [1, 1], [2, 0], [1, 0]])
    
    # Check if the XOR-like labeling is possible
    xor_labeling = [0, 1, 0, 1]  # This is impossible with a linear classifier
    
    print("4 points with XOR-like labeling [0, 1, 0, 1]:")
    print("This labeling is impossible with a linear classifier!")
    print("Therefore, VC dimension < 4, confirming VC dimension = 3")
    
    # Visualize the impossible case
    plt.figure(figsize=(8, 6))
    for i, (point, label) in enumerate(zip(points_4, xor_labeling)):
        color = 'blue' if label == 1 else 'red'
        marker = 'o' if label == 1 else 'x'
        plt.scatter(point[0], point[1], c=color, s=100, marker=marker, alpha=0.7)
        plt.annotate(f'P{i+1}({label})', (point[0], point[1]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=12)
    
    plt.xlim(-0.5, 2.5)
    plt.ylim(-0.5, 1.5)
    plt.title('Impossible Labeling: XOR Pattern with 4 Points\n(VC Dimension < 4)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def demonstrate_learning_curves() -> None:
    """
    Demonstrate learning curves and the relationship between sample size and generalization.
    
    This shows how both training and test error change as the number of training
    examples increases, illustrating the concepts of underfitting and overfitting.
    """
    print("=" * 60)
    print("DEMONSTRATION: LEARNING CURVES")
    print("=" * 60)
    print("How training and test error change with sample size")
    print()
    
    # Generate data from a true function with noise
    def true_function(x):
        return 2 * x**2 + 0.5
    
    # Parameters
    n_max = 200
    noise_std = 0.2
    n_test = 1000
    
    # Generate test data
    x_test = np.linspace(0, 1, n_test)
    y_test = true_function(x_test)
    
    # Test different training set sizes
    n_train_sizes = [5, 10, 20, 50, 100, 200]
    train_errors = []
    test_errors = []
    
    for n_train in n_train_sizes:
        # Generate training data
        x_train = np.random.rand(n_train)
        y_train = true_function(x_train) + np.random.normal(0, noise_std, n_train)
        
        # Fit polynomial models of different degrees
        degrees = [1, 2, 5]
        degree_errors = {'train': [], 'test': []}
        
        for degree in degrees:
            # Fit model
            model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
            model.fit(x_train.reshape(-1, 1), y_train)
            
            # Compute errors
            y_train_pred = model.predict(x_train.reshape(-1, 1))
            y_test_pred = model.predict(x_test.reshape(-1, 1))
            
            train_error = np.mean((y_train_pred - y_train) ** 2)
            test_error = np.mean((y_test_pred - y_test) ** 2)
            
            degree_errors['train'].append(train_error)
            degree_errors['test'].append(test_error)
        
        # Store average errors across degrees
        train_errors.append(np.mean(degree_errors['train']))
        test_errors.append(np.mean(degree_errors['test']))
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(n_train_sizes, train_errors, 'b-o', linewidth=2, markersize=6, label='Training Error')
    plt.plot(n_train_sizes, test_errors, 'r-s', linewidth=2, markersize=6, label='Test Error')
    plt.xlabel('Training Set Size (n)', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('Learning Curves: Error vs. Sample Size', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    generalization_gaps = [abs(te - tr) for te, tr in zip(test_errors, train_errors)]
    plt.plot(n_train_sizes, generalization_gaps, 'g-^', linewidth=2, markersize=6)
    plt.xlabel('Training Set Size (n)', fontsize=12)
    plt.ylabel('Generalization Gap', fontsize=12)
    plt.title('Generalization Gap vs. Sample Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("Key observations:")
    print("• Training error generally decreases with more data")
    print("• Test error decreases and then stabilizes")
    print("• Generalization gap decreases with more data")
    print("• This illustrates the sample complexity concept")

def main():
    """
    Main function to run all theoretical demonstrations.
    """
    print("SAMPLE COMPLEXITY BOUNDS AND THEORETICAL FOUNDATIONS")
    print("=" * 60)
    print("This demonstrates the theoretical foundations of generalization in machine learning.")
    print("These concepts provide mathematical guarantees for learning algorithms.")
    print()
    
    # Demonstration 1: Hoeffding Bound
    demonstrate_hoeffding_bound()
    
    # Demonstration 2: Union Bound
    demonstrate_union_bound()
    
    # Demonstration 3: Empirical vs Generalization Error
    demonstrate_empirical_vs_generalization_error()
    
    # Demonstration 4: Sample Complexity Bounds
    demonstrate_sample_complexity_bounds()
    
    # Demonstration 5: VC Dimension
    demonstrate_vc_dimension_2d()
    
    # Demonstration 6: Learning Curves
    demonstrate_learning_curves()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: KEY THEORETICAL INSIGHTS")
    print("=" * 60)
    print("1. Concentration inequalities (Hoeffding) provide probabilistic guarantees")
    print("2. Union bound allows us to bound probabilities of multiple events")
    print("3. Empirical risk can be misleading; generalization error is the true measure")
    print("4. Sample complexity bounds tell us how much data we need")
    print("5. VC dimension measures the complexity of hypothesis classes")
    print("6. Learning curves show how performance improves with more data")
    print("7. These theoretical tools help us understand and design better learning algorithms")

if __name__ == "__main__":
    main() 
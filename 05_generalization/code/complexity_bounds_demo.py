"""
Sample Complexity Bounds Demonstration

This module demonstrates the theoretical foundations of generalization through
sample complexity bounds, including union bounds, Hoeffding's inequality,
uniform convergence, and ERM analysis for finite hypothesis classes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def demonstrate_union_bound():
    """Demonstrate the union bound with multiple hypotheses"""
    
    # Simulate multiple hypotheses with different error rates
    np.random.seed(42)
    n_hypotheses = 100
    n_samples = 1000
    
    # True error rates for each hypothesis
    true_errors = np.random.beta(2, 8, n_hypotheses)  # Most hypotheses have low error
    
    # Simulate training errors
    training_errors = np.random.binomial(n_samples, true_errors) / n_samples
    
    # Calculate deviations
    deviations = np.abs(true_errors - training_errors)
    
    # Union bound calculation
    gamma = 0.05  # Desired accuracy
    individual_prob = 2 * np.exp(-2 * gamma**2 * n_samples)  # Hoeffding bound
    union_bound_prob = n_hypotheses * individual_prob
    
    # Actual probability (from simulation)
    actual_prob = np.mean(deviations > gamma)
    
    print(f"Individual failure probability: {individual_prob:.6f}")
    print(f"Union bound probability: {union_bound_prob:.6f}")
    print(f"Actual failure probability: {actual_prob:.6f}")
    print(f"Union bound is conservative by factor: {union_bound_prob/actual_prob:.2f}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(true_errors, training_errors, alpha=0.6)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect agreement')
    plt.fill_between([0, 1], [0, 1-gamma], [0, 1+gamma], alpha=0.2, color='green', label=f'±{gamma} tolerance')
    plt.xlabel('True Error Rate')
    plt.ylabel('Training Error Rate')
    plt.title('Training vs True Error Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(deviations, bins=30, alpha=0.7, label='Actual deviations')
    plt.axvline(gamma, color='red', linestyle='--', label=f'Threshold γ={gamma}')
    plt.xlabel('Deviation |True - Training|')
    plt.ylabel('Frequency')
    plt.title('Distribution of Deviations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return true_errors, training_errors, deviations


def demonstrate_hoeffding():
    """Demonstrate Hoeffding's inequality with error rate estimation"""
    
    np.random.seed(42)
    true_error_rate = 0.15  # 15% true error rate
    gamma = 0.05  # 5% accuracy
    delta = 0.05  # 95% confidence
    
    # Calculate required sample size from Hoeffding
    required_n = int(np.ceil(np.log(2/delta) / (2 * gamma**2)))
    print(f"Required sample size for {gamma*100}% accuracy with {delta*100}% confidence: {required_n}")
    
    # Simulate different sample sizes
    sample_sizes = [10, 50, 100, 500, 1000, 5000]
    empirical_probabilities = []
    theoretical_bounds = []
    
    for n in sample_sizes:
        # Simulate many experiments
        n_experiments = 10000
        large_deviations = 0
        
        for _ in range(n_experiments):
            # Generate n Bernoulli samples
            samples = np.random.binomial(1, true_error_rate, n)
            sample_mean = np.mean(samples)
            
            # Check if deviation is large
            if abs(sample_mean - true_error_rate) > gamma:
                large_deviations += 1
        
        empirical_prob = large_deviations / n_experiments
        theoretical_bound = 2 * np.exp(-2 * gamma**2 * n)
        
        empirical_probabilities.append(empirical_prob)
        theoretical_bounds.append(theoretical_bound)
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogy(sample_sizes, empirical_probabilities, 'bo-', label='Empirical', linewidth=2)
    plt.semilogy(sample_sizes, theoretical_bounds, 'r--', label='Hoeffding Bound', linewidth=2)
    plt.axhline(delta, color='g', linestyle=':', label=f'Target δ={delta}', alpha=0.7)
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Probability of Large Deviation')
    plt.title('Hoeffding Inequality in Practice')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(sample_sizes, empirical_probabilities, 'bo-', label='Empirical', linewidth=2)
    plt.plot(sample_sizes, theoretical_bounds, 'r--', label='Hoeffding Bound', linewidth=2)
    plt.axhline(delta, color='g', linestyle=':', label=f'Target δ={delta}', alpha=0.7)
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Probability of Large Deviation')
    plt.title('Hoeffding Inequality (Linear Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show sample size requirements for different accuracies
    accuracies = [0.01, 0.02, 0.05, 0.10]
    confidences = [0.01, 0.05, 0.10]
    
    print("\nSample Size Requirements:")
    print("Accuracy\tConfidence\tSample Size")
    print("-" * 40)
    for gamma in accuracies:
        for delta in confidences:
            n = int(np.ceil(np.log(2/delta) / (2 * gamma**2)))
            print(f"{gamma*100:6.1f}%\t\t{delta*100:6.1f}%\t\t{n:8d}")
    
    return sample_sizes, empirical_probabilities, theoretical_bounds


def learning_theory_recipe():
    """The basic recipe for proving generalization bounds"""
    
    print("Learning Theory Recipe:")
    print("1. Single Hypothesis: Use Hoeffding to bound |ε(h) - ε̂(h)|")
    print("2. All Hypotheses: Use Union Bound to get uniform convergence")
    print("3. Learned Hypothesis: Combine with ERM to bound ε(ĥ)")
    print("4. Sample Complexity: Solve for required n")
    
    # Example calculation
    k = 1000  # Number of hypotheses
    gamma = 0.05  # Desired accuracy
    delta = 0.05  # Desired confidence
    
    # Step 3: Set equal to delta and solve for n
    n_required = int(np.ceil(np.log(2*k/delta) / (2 * gamma**2)))
    
    print(f"\nExample: k={k}, γ={gamma}, δ={delta}")
    print(f"Required sample size: n ≥ {n_required}")
    print(f"With probability ≥ {1-delta}, |ε(h) - ε̂(h)| ≤ {gamma} for all h")
    
    return n_required


def demonstrate_error_rates():
    """Demonstrate the difference between training and generalization error"""
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate data
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    true_weights = true_weights / np.linalg.norm(true_weights)  # Normalize
    
    # True function: linear classifier with some noise
    logits = X @ true_weights
    true_labels = (logits > 0).astype(int)
    
    # Add some noise to make it realistic
    noise = np.random.binomial(1, 0.1, n_samples)  # 10% noise
    noisy_labels = (true_labels + noise) % 2
    
    # Split into training and test sets
    train_size = 800
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = noisy_labels[:train_size], noisy_labels[train_size:]
    
    # Train a simple linear classifier
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate errors
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)
    
    print(f"Training Error: {train_error:.4f}")
    print(f"Test Error: {test_error:.4f}")
    print(f"Generalization Gap: {test_error - train_error:.4f}")
    
    # Visualize the relationship
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6, s=20)
    plt.title('Training Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.6, s=20)
    plt.title('Test Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    return train_error, test_error


def demonstrate_erm():
    """Demonstrate ERM with different hypothesis classes"""
    
    np.random.seed(42)
    n_samples = 200
    n_features = 2
    
    # Generate data with a non-linear pattern
    X = np.random.uniform(-2, 2, (n_samples, n_features))
    y = ((X[:, 0]**2 + X[:, 1]**2) < 1).astype(int)  # Circle pattern
    
    # Add some noise
    noise = np.random.binomial(1, 0.1, n_samples)
    y = (y + noise) % 2
    
    # Split data
    train_size = 150
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Different hypothesis classes
    models = {
        'Linear': LogisticRegression(random_state=42),
        'Polynomial (degree=2)': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LogisticRegression(random_state=42))
        ]),
        'Polynomial (degree=5)': Pipeline([
            ('poly', PolynomialFeatures(degree=5)),
            ('linear', LogisticRegression(random_state=42))
        ]),
        'Decision Tree (depth=3)': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Decision Tree (depth=10)': DecisionTreeClassifier(max_depth=10, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_error = 1 - model.score(X_train, y_train)
        test_error = 1 - model.score(X_test, y_test)
        results[name] = {'train': train_error, 'test': test_error}
    
    # Display results
    print("ERM Results:")
    print("Model\t\t\tTrain Error\tTest Error\tGap")
    print("-" * 50)
    for name, errors in results.items():
        gap = errors['test'] - errors['train']
        print(f"{name:<20}\t{errors['train']:.4f}\t\t{errors['test']:.4f}\t\t{gap:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, model) in enumerate(models.items()):
        if i >= 5:  # Only show first 5
            break
            
        # Plot decision boundary
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        axes[i].contourf(xx, yy, Z, alpha=0.3)
        axes[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, alpha=0.6, s=20)
        axes[i].set_title(f'{name}\nTrain: {results[name]["train"]:.3f}, Test: {results[name]["test"]:.3f}')
        axes[i].set_xlim(x_min, x_max)
        axes[i].set_ylim(y_min, y_max)
    
    # Hide the last subplot
    axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return results


def demonstrate_single_hypothesis():
    """Demonstrate Hoeffding's inequality for a single hypothesis"""
    
    np.random.seed(42)
    true_error = 0.2  # 20% true error rate
    n_experiments = 1000
    sample_sizes = [10, 50, 100, 500, 1000]
    
    results = []
    for n in sample_sizes:
        # Simulate many experiments
        large_deviations = 0
        deviations = []
        
        for _ in range(n_experiments):
            # Generate n Bernoulli samples
            samples = np.random.binomial(1, true_error, n)
            sample_mean = np.mean(samples)
            deviation = abs(sample_mean - true_error)
            deviations.append(deviation)
            
            # Check if deviation is large
            if deviation > 0.05:  # 5% threshold
                large_deviations += 1
        
        empirical_prob = large_deviations / n_experiments
        theoretical_bound = 2 * np.exp(-2 * 0.05**2 * n)
        
        results.append({
            'n': n,
            'empirical': empirical_prob,
            'theoretical': theoretical_bound,
            'deviations': deviations
        })
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    ns = [r['n'] for r in results]
    empirical_probs = [r['empirical'] for r in results]
    theoretical_bounds = [r['theoretical'] for r in results]
    
    plt.semilogy(ns, empirical_probs, 'bo-', label='Empirical', linewidth=2)
    plt.semilogy(ns, theoretical_bounds, 'r--', label='Hoeffding Bound', linewidth=2)
    plt.xlabel('Sample Size (n)')
    plt.ylabel('Probability of Large Deviation')
    plt.title('Single Hypothesis: Hoeffding Inequality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    # Show distribution of deviations for n=100
    n_100_data = [r for r in results if r['n'] == 100][0]
    plt.hist(n_100_data['deviations'], bins=30, alpha=0.7, label='n=100')
    plt.axvline(0.05, color='red', linestyle='--', label='Threshold γ=0.05')
    plt.xlabel('Deviation |True - Training|')
    plt.ylabel('Frequency')
    plt.title('Distribution of Deviations (n=100)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Show distribution of deviations for n=1000
    n_1000_data = [r for r in results if r['n'] == 1000][0]
    plt.hist(n_1000_data['deviations'], bins=30, alpha=0.7, label='n=1000')
    plt.axvline(0.05, color='red', linestyle='--', label='Threshold γ=0.05')
    plt.xlabel('Deviation |True - Training|')
    plt.ylabel('Frequency')
    plt.title('Distribution of Deviations (n=1000)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print("Single Hypothesis Analysis:")
    print("Sample Size\tEmpirical\tTheoretical\tRatio")
    print("-" * 50)
    for r in results:
        ratio = r['theoretical'] / r['empirical'] if r['empirical'] > 0 else float('inf')
        print(f"{r['n']:10d}\t{r['empirical']:.6f}\t{r['theoretical']:.6f}\t{ratio:.2f}")
    
    return results


def demonstrate_uniform_convergence():
    """Demonstrate uniform convergence for finite hypothesis classes"""
    
    np.random.seed(42)
    k_hypotheses = 50  # Number of hypotheses
    n_samples = 1000   # Sample size
    n_experiments = 1000  # Number of experiments
    
    # Generate true error rates for each hypothesis
    true_errors = np.random.beta(2, 8, k_hypotheses)  # Most hypotheses have low error
    
    # Simulate experiments
    gamma = 0.05  # Desired accuracy
    failures = 0
    
    for _ in range(n_experiments):
        # Generate training errors for all hypotheses
        training_errors = np.random.binomial(n_samples, true_errors) / n_samples
        
        # Check if any hypothesis has large deviation
        deviations = np.abs(true_errors - training_errors)
        if np.any(deviations > gamma):
            failures += 1
    
    empirical_prob = failures / n_experiments
    theoretical_bound = k_hypotheses * 2 * np.exp(-2 * gamma**2 * n_samples)
    
    print("Uniform Convergence Analysis:")
    print(f"Number of hypotheses (k): {k_hypotheses}")
    print(f"Sample size (n): {n_samples}")
    print(f"Desired accuracy (γ): {gamma}")
    print(f"Empirical failure probability: {empirical_prob:.6f}")
    print(f"Theoretical bound: {theoretical_bound:.6f}")
    print(f"Bound is conservative by factor: {theoretical_bound/empirical_prob:.2f}")
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Show one experiment
    training_errors = np.random.binomial(n_samples, true_errors) / n_samples
    deviations = np.abs(true_errors - training_errors)
    
    plt.scatter(true_errors, training_errors, alpha=0.6, s=20)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect agreement')
    plt.fill_between([0, 1], [0, 1-gamma], [0, 1+gamma], alpha=0.2, color='green', label=f'±{gamma} tolerance')
    plt.xlabel('True Error Rate')
    plt.ylabel('Training Error Rate')
    plt.title('Training vs True Error Rates (One Experiment)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(deviations, bins=30, alpha=0.7, label='Deviations')
    plt.axvline(gamma, color='red', linestyle='--', label=f'Threshold γ={gamma}')
    plt.xlabel('Deviation |True - Training|')
    plt.ylabel('Frequency')
    plt.title('Distribution of Deviations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return empirical_prob, theoretical_bound


def demonstrate_erm_analysis():
    """Demonstrate ERM analysis with finite hypothesis classes"""
    
    np.random.seed(42)
    k_hypotheses = 20
    n_samples = 500
    n_experiments = 1000
    
    # Generate true error rates
    true_errors = np.random.beta(2, 8, k_hypotheses)
    best_hypothesis = np.argmin(true_errors)
    best_error = true_errors[best_hypothesis]
    
    print(f"Best hypothesis index: {best_hypothesis}")
    print(f"Best true error: {best_error:.4f}")
    
    # Simulate ERM experiments
    erm_errors = []
    generalization_gaps = []
    erm_hypothesis_list = []
    
    for _ in range(n_experiments):
        # Generate training errors
        training_errors = np.random.binomial(n_samples, true_errors) / n_samples
        
        # ERM: choose hypothesis with minimum training error
        erm_hypothesis = np.argmin(training_errors)
        erm_error = true_errors[erm_hypothesis]
        
        erm_errors.append(erm_error)
        generalization_gaps.append(erm_error - best_error)
        erm_hypothesis_list.append(erm_hypothesis)
    
    # Calculate statistics
    mean_erm_error = np.mean(erm_errors)
    mean_gap = np.mean(generalization_gaps)
    max_gap = np.max(generalization_gaps)
    
    print(f"\nERM Analysis Results:")
    print(f"Mean ERM error: {mean_erm_error:.4f}")
    print(f"Mean gap from best: {mean_gap:.4f}")
    print(f"Maximum gap from best: {max_gap:.4f}")
    print(f"ERM chose best hypothesis: {np.mean(np.array(erm_errors) == best_error):.1%} of the time")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(erm_errors, bins=30, alpha=0.7, label='ERM errors')
    plt.axvline(best_error, color='red', linestyle='--', label=f'Best error: {best_error:.4f}')
    plt.xlabel('Generalization Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of ERM Generalization Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(generalization_gaps, bins=30, alpha=0.7, label='Gaps')
    plt.axvline(0, color='red', linestyle='--', label='No gap')
    plt.xlabel('Gap from Best Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Generalization Gaps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Show how often each hypothesis was chosen
    hypothesis_counts = np.bincount(erm_hypothesis_list, minlength=k_hypotheses)
    plt.bar(range(k_hypotheses), hypothesis_counts, alpha=0.7)
    plt.axvline(best_hypothesis, color='red', linestyle='--', label=f'Best hypothesis: {best_hypothesis}')
    plt.xlabel('Hypothesis Index')
    plt.ylabel('Times Chosen by ERM')
    plt.title('ERM Hypothesis Selection Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return erm_errors, generalization_gaps


def demonstrate_sample_complexity():
    """Demonstrate sample complexity requirements"""
    
    # Parameters
    k_values = [10, 100, 1000, 10000]
    gamma_values = [0.01, 0.02, 0.05, 0.10]
    delta_values = [0.01, 0.05, 0.10]
    
    print("Sample Size Requirements for Finite Hypothesis Classes:")
    print("k\t\tγ\t\tδ\t\tRequired n")
    print("-" * 60)
    
    for k in k_values:
        for gamma in gamma_values:
            for delta in delta_values:
                n_required = int(np.ceil(np.log(2*k/delta) / (2 * gamma**2)))
                print(f"{k:8d}\t\t{gamma:.2f}\t\t{delta:.2f}\t\t{n_required:8d}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    # Fix gamma and delta, vary k
    gamma = 0.05
    delta = 0.05
    k_range = np.logspace(1, 4, 50)
    n_required = np.log(2*k_range/delta) / (2 * gamma**2)
    
    plt.semilogx(k_range, n_required, 'b-', linewidth=2)
    plt.xlabel('Number of Hypotheses (k)')
    plt.ylabel('Required Sample Size (n)')
    plt.title('Sample Size vs Number of Hypotheses')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    # Fix k and delta, vary gamma
    k = 1000
    delta = 0.05
    gamma_range = np.logspace(-2, -1, 50)
    n_required = np.log(2*k/delta) / (2 * gamma_range**2)
    
    plt.semilogx(gamma_range, n_required, 'r-', linewidth=2)
    plt.xlabel('Desired Accuracy (γ)')
    plt.ylabel('Required Sample Size (n)')
    plt.title('Sample Size vs Desired Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Fix k and gamma, vary delta
    k = 1000
    gamma = 0.05
    delta_range = np.logspace(-2, -1, 50)
    n_required = np.log(2*k/delta_range) / (2 * gamma**2)
    
    plt.semilogx(delta_range, n_required, 'g-', linewidth=2)
    plt.xlabel('Desired Confidence (δ)')
    plt.ylabel('Required Sample Size (n)')
    plt.title('Sample Size vs Desired Confidence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show practical implications
    print(f"\nPractical Implications:")
    print(f"For k=1000 hypotheses, γ=0.05 accuracy, δ=0.05 confidence:")
    n_practical = int(np.ceil(np.log(2*1000/0.05) / (2 * 0.05**2)))
    print(f"Required sample size: n ≥ {n_practical}")
    print(f"This means you need about {n_practical} training examples")
    print(f"to be 95% confident that your model's error is within 10% of the best possible")
    
    return k_values, gamma_values, delta_values


if __name__ == "__main__":
    # Run all demonstrations
    print("Running Sample Complexity Bounds Demonstrations...")
    print("=" * 60)
    
    print("\n1. Union Bound Demonstration:")
    true_errors, training_errors, deviations = demonstrate_union_bound()
    
    print("\n2. Hoeffding's Inequality:")
    sample_sizes, empirical_probs, theoretical_bounds = demonstrate_hoeffding()
    
    print("\n3. Learning Theory Recipe:")
    n_required = learning_theory_recipe()
    
    print("\n4. Error Rates Demonstration:")
    train_error, test_error = demonstrate_error_rates()
    
    print("\n5. ERM with Different Hypothesis Classes:")
    erm_results = demonstrate_erm()
    
    print("\n6. Single Hypothesis Analysis:")
    single_hypothesis_results = demonstrate_single_hypothesis()
    
    print("\n7. Uniform Convergence:")
    uniform_conv_results = demonstrate_uniform_convergence()
    
    print("\n8. ERM Analysis:")
    erm_errors, generalization_gaps = demonstrate_erm_analysis()
    
    print("\n9. Sample Complexity Requirements:")
    k_values, gamma_values, delta_values = demonstrate_sample_complexity()
    
    print("\nAll demonstrations completed!")

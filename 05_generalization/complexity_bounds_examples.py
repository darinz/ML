import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# --- 1. Hoeffding/Chernoff Bound Demonstration ---

def hoeffding_demo(phi=0.6, n=100, gamma=0.1, n_trials=10000):
    """
    Simulate the probability that the sample mean of n Bernoulli(phi) deviates from phi by more than gamma.
    Compare with Hoeffding's bound.
    """
    np.random.seed(42)
    deviations = []
    for _ in range(n_trials):
        samples = bernoulli.rvs(phi, size=n)
        phat = np.mean(samples)
        deviations.append(abs(phat - phi) > gamma)
    empirical_prob = np.mean(deviations)
    hoeffding_bound = 2 * np.exp(-2 * gamma**2 * n)
    print(f"Empirical P(|phat - phi| > {gamma}): {empirical_prob:.4f}")
    print(f"Hoeffding bound: {hoeffding_bound:.4f}")

# --- 2. Union Bound Demonstration ---

def union_bound_demo(p_single=0.01, k=10):
    """
    Show that the probability of at least one of k rare events is at most k*p_single (union bound).
    """
    prob_union = 1 - (1 - p_single) ** k
    union_bound = k * p_single
    print(f"True probability of at least one event: {prob_union:.4f}")
    print(f"Union bound: {union_bound:.4f}")

# --- 3. Empirical Risk and Generalization Error Simulation ---

def empirical_vs_generalization_demo(phi=0.7, n=50, n_test=10000):
    """
    Simulate empirical risk (training error) and generalization error for a Bernoulli classifier.
    """
    np.random.seed(0)
    y_train = bernoulli.rvs(phi, size=n)
    y_test = bernoulli.rvs(phi, size=n_test)
    # Suppose our classifier always predicts 1 (minimizes empirical risk if phi > 0.5)
    h = lambda x: 1
    train_error = np.mean(h(y_train) != y_train)
    test_error = np.mean(h(y_test) != y_test)
    print(f"Empirical risk (train error): {train_error:.4f}")
    print(f"Generalization error (test error): {test_error:.4f}")

# --- 4. Sample Complexity Bound Visualization ---

def sample_complexity_plot(gamma=0.1, delta=0.05, k=100):
    """
    Plot the sample complexity bound n >= (1/(2*gamma^2)) * log(2k/delta) as a function of k.
    """
    ks = np.arange(1, 1001)
    n_bound = (1/(2*gamma**2)) * np.log(2*ks/delta)
    plt.figure(figsize=(7,4))
    plt.plot(ks, n_bound)
    plt.xlabel('Number of Hypotheses (k)')
    plt.ylabel('Required n for uniform convergence')
    plt.title('Sample Complexity Bound vs. Number of Hypotheses')
    plt.grid(True)
    plt.show()

# --- 5. VC Dimension Example: Linear Classifiers in 2D ---

def plot_vc_shattering():
    """
    Visualize shattering of 3 points in 2D by linear classifiers.
    """
    points = np.array([[0,0], [1,1], [2,0]])
    fig, axes = plt.subplots(2, 4, figsize=(12,6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        labels = [(i >> j) & 1 for j in range(3)]
        for pt, lbl in zip(points, labels):
            ax.scatter(*pt, c='C0' if lbl else 'C1', s=80, marker='o' if lbl else 'x')
        ax.set_xlim(-0.5,2.5)
        ax.set_ylim(-0.5,1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Labeling {i+1}')
    plt.suptitle('All 8 Labelings of 3 Points (VC Dimension = 3)')
    plt.tight_layout()
    plt.show()

# --- Run all demonstrations ---
if __name__ == '__main__':
    print("--- Hoeffding/Chernoff Bound Demo ---")
    hoeffding_demo()
    print("\n--- Union Bound Demo ---")
    union_bound_demo()
    print("\n--- Empirical Risk vs. Generalization Error Demo ---")
    empirical_vs_generalization_demo()
    print("\n--- Sample Complexity Bound Visualization ---")
    sample_complexity_plot()
    print("\n--- VC Dimension Example (Shattering 3 Points) ---")
    plot_vc_shattering() 
"""
Complete example usage for linear bandits.

This script demonstrates how to use all the linear bandit algorithms together
with a complete example including environment setup, algorithm comparison, and visualization.
"""

import numpy as np
from linucb import LinUCB
from linear_thompson_sampling import LinearThompsonSampling
from oful import OFUL
from linear_bandit_environment import (
    LinearBanditEnvironment, 
    run_linear_bandit_experiment,
    compare_linear_algorithms,
    plot_linear_bandit_results
)

def run_linear_bandit_example():
    """Run a complete linear bandit experiment comparing all algorithms."""
    
    # Set up problem parameters
    d = 5  # Feature dimension
    K = 10  # Number of arms
    T = 1000  # Number of time steps
    n_runs = 50  # Number of independent runs
    
    print(f"Linear Bandit Experiment Setup:")
    print(f"  Feature dimension: {d}")
    print(f"  Number of arms: {K}")
    print(f"  Time steps: {T}")
    print(f"  Independent runs: {n_runs}")
    
    # Generate true parameter vector
    theta_star = np.random.randn(d)
    theta_star = theta_star / np.linalg.norm(theta_star)  # Normalize
    print(f"  True parameter norm: {np.linalg.norm(theta_star):.3f}")
    
    # Generate random arms (feature vectors)
    arms = np.random.randn(K, d)
    arms = arms / np.linalg.norm(arms, axis=1, keepdims=True)  # Normalize
    
    # Create environment
    env = LinearBanditEnvironment(theta_star, arms, noise_std=0.1)
    
    # Calculate optimal reward for reference
    optimal_reward = env.get_optimal_reward()
    print(f"  Optimal expected reward: {optimal_reward:.3f}")
    
    # Define algorithms to compare
    algorithms = {
        'LinUCB (α=1.0)': lambda d: LinUCB(d, alpha=1.0, lambda_reg=1.0),
        'LinUCB (α=2.0)': lambda d: LinUCB(d, alpha=2.0, lambda_reg=1.0),
        'Linear TS (σ=1.0)': lambda d: LinearThompsonSampling(d, sigma=1.0, lambda_reg=1.0),
        'Linear TS (σ=0.5)': lambda d: LinearThompsonSampling(d, sigma=0.5, lambda_reg=1.0),
        'OFUL (δ=0.1)': lambda d: OFUL(d, delta=0.1, lambda_reg=1.0)
    }
    
    # Run comparison
    print("\nRunning algorithm comparison...")
    results = compare_linear_algorithms(env, algorithms, T=T, n_runs=n_runs)
    
    # Print final results
    print("\nFinal cumulative regrets:")
    for name, regret in results.items():
        final_regret = regret[-1]
        print(f"  {name}: {final_regret:.2f}")
    
    # Find best performing algorithm
    best_algorithm = min(results.keys(), key=lambda k: results[k][-1])
    print(f"\nBest performing algorithm: {best_algorithm}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_linear_bandit_results(results, T)
    
    return results, env

def analyze_algorithm_performance(results, T):
    """Analyze and compare algorithm performance."""
    
    print("\nAlgorithm Performance Analysis:")
    print("=" * 50)
    
    # Final regret comparison
    print("\nFinal Cumulative Regret:")
    sorted_algorithms = sorted(results.items(), key=lambda x: x[1][-1])
    for i, (name, regret) in enumerate(sorted_algorithms):
        print(f"  {i+1}. {name}: {regret[-1]:.2f}")
    
    # Regret rate analysis (regret / sqrt(t))
    print("\nRegret Rate Analysis (R(t) / √t) - Final 100 steps average:")
    for name, regret in results.items():
        final_100_regret_rate = np.mean(regret[-100:] / np.sqrt(range(T-99, T+1)))
        print(f"  {name}: {final_100_regret_rate:.3f}")
    
    # Convergence analysis
    print("\nConvergence Analysis (regret in last 100 steps vs first 100 steps):")
    for name, regret in results.items():
        first_100_avg = np.mean(regret[99:199])  # Steps 100-199
        last_100_avg = np.mean(regret[-100:])
        improvement = (first_100_avg - last_100_avg) / first_100_avg * 100
        print(f"  {name}: {improvement:.1f}% improvement")

def run_single_algorithm_demo():
    """Run a demonstration with a single algorithm."""
    
    print("\nSingle Algorithm Demonstration:")
    print("=" * 40)
    
    # Set up smaller problem for demonstration
    d = 3
    K = 5
    T = 500
    
    # Generate problem
    theta_star = np.array([0.5, 0.3, 0.2])
    arms = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.7, 0.7, 0.0],
        [0.5, 0.5, 0.7]
    ])
    
    env = LinearBanditEnvironment(theta_star, arms, noise_std=0.1)
    
    # Run LinUCB
    algorithm = LinUCB(d, alpha=1.0)
    chosen_arms, rewards = run_linear_bandit_experiment(env, algorithm, T)
    
    # Analyze results
    regret = env.get_regret(chosen_arms, rewards)
    final_regret = regret[-1]
    
    print(f"  Final cumulative regret: {final_regret:.2f}")
    print(f"  Average reward: {np.mean(rewards):.3f}")
    print(f"  Optimal reward: {env.get_optimal_reward():.3f}")
    
    # Show arm selection distribution
    arm_counts = np.bincount(chosen_arms, minlength=K)
    print(f"  Arm selection counts: {arm_counts}")
    
    return algorithm, chosen_arms, rewards

if __name__ == "__main__":
    print("Linear Bandits Complete Example")
    print("=" * 50)
    
    # Run main experiment
    results, env = run_linear_bandit_example()
    
    # Analyze performance
    analyze_algorithm_performance(results, 1000)
    
    # Run single algorithm demo
    algorithm, chosen_arms, rewards = run_single_algorithm_demo()
    
    print("\nExperiment completed!")
    print("Check the generated plots to see the regret comparison.")

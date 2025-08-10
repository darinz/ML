"""
Complete example usage for contextual bandits.

This script demonstrates how to use all the contextual bandit algorithms together
with a complete example including environment setup, algorithm comparison, and visualization.
"""

import numpy as np
from contextual_ucb import ContextualUCB
from contextual_thompson_sampling import ContextualThompsonSampling
from neural_contextual_bandit import NeuralContextualBandit
from disjoint_linucb import DisjointLinUCB
from contextual_bandit_environment import (
    ContextualBanditEnvironment,
    run_contextual_bandit_experiment,
    compare_contextual_algorithms,
    plot_contextual_bandit_results
)

def run_contextual_bandit_example():
    """Run a complete contextual bandit experiment comparing all algorithms."""
    
    # Set up problem parameters
    d = 10  # Feature dimension
    n_arms = 10  # Number of arms
    T = 1000  # Number of time steps
    n_runs = 50  # Number of independent runs
    
    print(f"Contextual Bandit Experiment Setup:")
    print(f"  Feature dimension: {d}")
    print(f"  Number of arms: {n_arms}")
    print(f"  Time steps: {T}")
    print(f"  Independent runs: {n_runs}")
    
    # Generate true parameter vector
    theta_star = np.random.randn(d)
    theta_star = theta_star / np.linalg.norm(theta_star)  # Normalize
    print(f"  True parameter norm: {np.linalg.norm(theta_star):.3f}")
    
    # Context generator (stochastic contexts)
    def context_generator():
        return np.random.randn(5)  # 5-dimensional context
    
    # Create environment
    env = ContextualBanditEnvironment(theta_star, context_generator, noise_std=0.1)
    
    # Define algorithms to compare
    algorithms = {
        'Contextual UCB (α=1.0)': lambda d: ContextualUCB(d, alpha=1.0, lambda_reg=1.0),
        'Contextual UCB (α=2.0)': lambda d: ContextualUCB(d, alpha=2.0, lambda_reg=1.0),
        'Contextual TS (σ=1.0)': lambda d: ContextualThompsonSampling(d, sigma=1.0, lambda_reg=1.0),
        'Contextual TS (σ=0.5)': lambda d: ContextualThompsonSampling(d, sigma=0.5, lambda_reg=1.0),
        'Disjoint LinUCB': lambda d: DisjointLinUCB(d, num_arms=n_arms, alpha=1.0),
        'Neural Bandit': lambda d: NeuralContextualBandit(d, num_arms=n_arms)
    }
    
    # Run comparison
    print("\nRunning algorithm comparison...")
    results = compare_contextual_algorithms(env, algorithms, T=T, n_arms=n_arms, n_runs=n_runs)
    
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
    plot_contextual_bandit_results(results, T)
    
    return results, env

def analyze_contextual_performance(results, T):
    """Analyze and compare contextual algorithm performance."""
    
    print("\nContextual Algorithm Performance Analysis:")
    print("=" * 60)
    
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

def run_single_contextual_demo():
    """Run a demonstration with a single contextual algorithm."""
    
    print("\nSingle Contextual Algorithm Demonstration:")
    print("=" * 50)
    
    # Set up smaller problem for demonstration
    d = 5
    n_arms = 5
    T = 500
    
    # Generate problem
    theta_star = np.random.randn(d)
    theta_star = theta_star / np.linalg.norm(theta_star)
    
    # Context generator
    def context_generator():
        return np.random.randn(2)  # 2-dimensional context
    
    env = ContextualBanditEnvironment(theta_star, context_generator, noise_std=0.1)
    
    # Run Contextual UCB
    algorithm = ContextualUCB(d, alpha=1.0)
    chosen_arms, rewards, optimal_rewards = run_contextual_bandit_experiment(
        env, algorithm, T, n_arms
    )
    
    # Analyze results
    regret = env.get_regret(chosen_arms, rewards, optimal_rewards)
    final_regret = regret[-1]
    
    print(f"  Final cumulative regret: {final_regret:.2f}")
    print(f"  Average reward: {np.mean(rewards):.3f}")
    print(f"  Average optimal reward: {np.mean(optimal_rewards):.3f}")
    
    # Show arm selection distribution
    arm_counts = np.bincount(chosen_arms, minlength=n_arms)
    print(f"  Arm selection counts: {arm_counts}")
    
    return algorithm, chosen_arms, rewards, optimal_rewards

def demonstrate_context_adaptation():
    """Demonstrate how algorithms adapt to changing contexts."""
    
    print("\nContext Adaptation Demonstration:")
    print("=" * 40)
    
    # Create environment with changing contexts
    d = 8
    n_arms = 6
    T = 300
    
    theta_star = np.random.randn(d)
    theta_star = theta_star / np.linalg.norm(theta_star)
    
    # Context generator that changes over time
    def changing_context_generator():
        t = len(changing_context_generator.contexts)
        if t < T // 3:
            # First phase: context biased toward first half of features
            context = np.random.randn(3) * 0.5
            context = np.concatenate([context, np.random.randn(2) * 0.1])
        elif t < 2 * T // 3:
            # Second phase: context biased toward second half of features
            context = np.random.randn(3) * 0.1
            context = np.concatenate([context, np.random.randn(2) * 0.5])
        else:
            # Third phase: balanced context
            context = np.random.randn(5) * 0.3
        changing_context_generator.contexts.append(context)
        return context
    
    changing_context_generator.contexts = []
    
    env = ContextualBanditEnvironment(theta_star, changing_context_generator, noise_std=0.1)
    
    # Run algorithm
    algorithm = ContextualThompsonSampling(d, sigma=1.0)
    chosen_arms, rewards, optimal_rewards = run_contextual_bandit_experiment(
        env, algorithm, T, n_arms
    )
    
    # Analyze performance in different phases
    phase1_regret = np.mean(env.get_regret(chosen_arms[:T//3], rewards[:T//3], optimal_rewards[:T//3]))
    phase2_regret = np.mean(env.get_regret(chosen_arms[T//3:2*T//3], rewards[T//3:2*T//3], optimal_rewards[T//3:2*T//3]))
    phase3_regret = np.mean(env.get_regret(chosen_arms[2*T//3:], rewards[2*T//3:], optimal_rewards[2*T//3:]))
    
    print(f"  Phase 1 average regret: {phase1_regret:.3f}")
    print(f"  Phase 2 average regret: {phase2_regret:.3f}")
    print(f"  Phase 3 average regret: {phase3_regret:.3f}")
    print(f"  Adaptation improvement: {(phase1_regret - phase3_regret) / phase1_regret * 100:.1f}%")

if __name__ == "__main__":
    print("Contextual Bandits Complete Example")
    print("=" * 50)
    
    # Run main experiment
    results, env = run_contextual_bandit_example()
    
    # Analyze performance
    analyze_contextual_performance(results, 1000)
    
    # Run single algorithm demo
    algorithm, chosen_arms, rewards, optimal_rewards = run_single_contextual_demo()
    
    # Demonstrate context adaptation
    demonstrate_context_adaptation()
    
    print("\nExperiment completed!")
    print("Check the generated plots to see the regret comparison.")

"""
Complete example usage for Best Arm Identification.

This script demonstrates how to use all the BAI algorithms together
with a complete example including environment setup, algorithm comparison, and visualization.
"""

import numpy as np
from successive_elimination import SuccessiveElimination
from racing_algorithm import RacingAlgorithm
from lucb import LUCB
from sequential_halving import SequentialHalving
from bai_environment import (
    BAIEnvironment,
    run_bai_experiment,
    compare_bai_algorithms,
    plot_bai_results
)

def run_bai_example():
    """Run a complete BAI experiment comparing all algorithms."""
    
    # Set up problem parameters
    arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
    env = BAIEnvironment(arm_means)
    
    print(f"BAI Experiment Setup:")
    print(f"  Number of arms: {env.n_arms}")
    print(f"  Optimal arm: {env.optimal_arm}")
    print(f"  Optimal mean: {env.optimal_mean:.3f}")
    print(f"  Gaps: {env.get_gaps()}")
    
    # Define algorithms to compare
    algorithms = {
        'Successive Elimination': lambda n: SuccessiveElimination(n, delta=0.1),
        'Racing': lambda n: RacingAlgorithm(n, delta=0.1),
        'LUCB': lambda n: LUCB(n, delta=0.1),
        'Sequential Halving': lambda n: SequentialHalving(n, budget=1000)
    }
    
    # Run comparison
    print("\nRunning algorithm comparison...")
    results = compare_bai_algorithms(env, algorithms, n_runs=50, max_pulls=1000)
    
    # Print results
    print("\nResults:")
    for name, result in results.items():
        print(f"  {name}:")
        print(f"    Success Rate: {result['success_rate']:.3f}")
        print(f"    Avg Sample Complexity: {result['avg_sample_complexity']:.1f}")
        print(f"    Std Sample Complexity: {result['std_sample_complexity']:.1f}")
    
    # Find best performing algorithm
    best_algorithm = max(results.keys(), key=lambda k: results[k]['success_rate'])
    print(f"\nBest performing algorithm: {best_algorithm}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_bai_results(results)
    
    return results, env

def analyze_bai_performance(results):
    """Analyze and compare BAI algorithm performance."""
    
    print("\nBAI Algorithm Performance Analysis:")
    print("=" * 50)
    
    # Success rate comparison
    print("\nSuccess Rate Ranking:")
    sorted_algorithms = sorted(results.items(), key=lambda x: x[1]['success_rate'], reverse=True)
    for i, (name, result) in enumerate(sorted_algorithms):
        print(f"  {i+1}. {name}: {result['success_rate']:.3f}")
    
    # Sample complexity comparison
    print("\nSample Complexity Ranking (lower is better):")
    sorted_complexity = sorted(results.items(), key=lambda x: x[1]['avg_sample_complexity'])
    for i, (name, result) in enumerate(sorted_complexity):
        print(f"  {i+1}. {name}: {result['avg_sample_complexity']:.1f} ± {result['std_sample_complexity']:.1f}")
    
    # Efficiency analysis (success rate / sample complexity)
    print("\nEfficiency Analysis (success rate / sample complexity):")
    efficiencies = {}
    for name, result in results.items():
        efficiency = result['success_rate'] / result['avg_sample_complexity']
        efficiencies[name] = efficiency
    
    sorted_efficiency = sorted(efficiencies.items(), key=lambda x: x[1], reverse=True)
    for i, (name, efficiency) in enumerate(sorted_efficiency):
        print(f"  {i+1}. {name}: {efficiency:.6f}")

def run_single_bai_demo():
    """Run a demonstration with a single BAI algorithm."""
    
    print("\nSingle BAI Algorithm Demonstration:")
    print("=" * 40)
    
    # Set up smaller problem for demonstration
    arm_means = [0.1, 0.15, 0.2, 0.25, 0.3]
    env = BAIEnvironment(arm_means)
    
    # Run LUCB algorithm
    algorithm = LUCB(env.n_arms, delta=0.1)
    
    print(f"  Environment: {env.n_arms} arms with means {arm_means}")
    print(f"  Optimal arm: {env.optimal_arm}")
    
    # Run experiment
    success, pulls_used, identified_arm = run_bai_experiment(env, algorithm, max_pulls=500)
    
    print(f"  Success: {success}")
    print(f"  Pulls used: {pulls_used}")
    print(f"  Identified arm: {identified_arm}")
    print(f"  Correct identification: {identified_arm == env.optimal_arm}")
    
    # Show algorithm statistics
    print(f"  Final empirical means: {algorithm.empirical_means}")
    print(f"  Final pulls: {algorithm.pulls}")
    
    return algorithm, success, pulls_used

def demonstrate_confidence_intervals():
    """Demonstrate confidence interval behavior in BAI algorithms."""
    
    print("\nConfidence Interval Demonstration:")
    print("=" * 40)
    
    # Set up environment with clear gaps
    arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
    env = BAIEnvironment(arm_means, noise_std=0.05)
    
    # Run Racing algorithm and track confidence intervals
    algorithm = RacingAlgorithm(env.n_arms, delta=0.1)
    
    print(f"  Environment: {env.n_arms} arms with means {arm_means}")
    print(f"  Optimal arm: {env.optimal_arm}")
    
    # Track confidence intervals over time
    interval_history = []
    for step in range(100):
        arm = algorithm.select_arm()
        reward = env.pull_arm(arm)
        algorithm.update(arm, reward)
        
        if step % 20 == 0:  # Record every 20 steps
            intervals = algorithm.get_confidence_intervals()
            interval_history.append((step, intervals.copy()))
            
            print(f"  Step {step}:")
            for i, (lower, upper) in enumerate(intervals):
                print(f"    Arm {i}: [{lower:.3f}, {upper:.3f}]")
        
        if algorithm.is_complete():
            print(f"  Algorithm completed at step {step}")
            break
    
    return algorithm, interval_history

if __name__ == "__main__":
    print("Best Arm Identification Complete Example")
    print("=" * 50)
    
    # Run main experiment
    results, env = run_bai_example()
    
    # Analyze performance
    analyze_bai_performance(results)
    
    # Run single algorithm demo
    algorithm, success, pulls_used = run_single_bai_demo()
    
    # Demonstrate confidence intervals
    racing_alg, interval_history = demonstrate_confidence_intervals()
    
    print("\nExperiment completed!")
    print("Check the generated plots to see the algorithm comparison.")

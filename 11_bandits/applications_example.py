"""
Complete example usage for bandit applications across different domains.

This script demonstrates how to use all the bandit application classes together
with a complete example including experiment setup, performance comparison, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from bandit_application_framework import (
    BanditApplication,
    compare_applications,
    plot_application_comparison
)

def run_applications_example():
    """Run a complete bandit applications experiment comparing different domains."""
    
    print("Bandit Applications Complete Example")
    print("=" * 50)
    
    # Run comparison across different application domains
    print("\nRunning application comparison...")
    results = compare_applications()
    
    # Print results
    print("\nResults:")
    for app_type, result in results.items():
        print(f"  {app_type}:")
        print(f"    Average Reward: {result['avg_reward']:.3f}")
        print(f"    Cumulative Regret: {result['cumulative_regret']:.3f}")
        print(f"    Final Regret: {result['final_regret']:.3f}")
    
    # Find best performing application
    best_app = max(results.keys(), key=lambda k: results[k]['avg_reward'])
    print(f"\nBest performing application: {best_app}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_application_comparison(results)
    
    return results

def run_single_application_demo():
    """Run a demonstration with a single application type."""
    
    print("\nSingle Application Demonstration:")
    print("=" * 40)
    
    # Set up ad selection application
    app = BanditApplication("ad_selection", n_arms=10, feature_dim=5)
    
    print(f"  Application: Ad Selection")
    print(f"  Number of arms: {app.n_arms}")
    print(f"  Feature dimension: {app.feature_dim}")
    
    # Run experiment
    rewards, regrets = app.run_experiment(n_steps=200)
    
    print(f"  Average reward: {np.mean(rewards):.3f}")
    print(f"  Cumulative regret: {np.sum(regrets):.3f}")
    print(f"  Final regret: {np.mean(regrets[-50:]):.3f}")
    
    # Plot learning curve
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Rewards Over Time')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(regrets))
    plt.title('Cumulative Regret')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Regret')
    
    plt.tight_layout()
    plt.show()
    
    return app, rewards, regrets

def demonstrate_context_generation():
    """Demonstrate context generation for different application types."""
    
    print("\nContext Generation Demonstration:")
    print("=" * 40)
    
    # Test different application types
    app_types = ["ad_selection", "recommendation", "clinical_trial", "dynamic_pricing"]
    
    for app_type in app_types:
        print(f"\n  {app_type.upper()}:")
        app = BanditApplication(app_type, n_arms=5, feature_dim=4)
        
        # Generate contexts
        contexts = []
        for _ in range(3):
            context = app._generate_context()
            contexts.append(context)
            print(f"    Context {_+1}: {context}")
        
        # Show context statistics
        contexts_array = np.array(contexts)
        print(f"    Mean: {np.mean(contexts_array, axis=0)}")
        print(f"    Std: {np.std(contexts_array, axis=0)}")

def analyze_application_performance(results):
    """Analyze and compare application performance."""
    
    print("\nApplication Performance Analysis:")
    print("=" * 50)
    
    # Reward ranking
    print("\nAverage Reward Ranking:")
    sorted_apps = sorted(results.items(), key=lambda x: x[1]['avg_reward'], reverse=True)
    for i, (app_type, result) in enumerate(sorted_apps):
        print(f"  {i+1}. {app_type}: {result['avg_reward']:.3f}")
    
    # Regret ranking (lower is better)
    print("\nCumulative Regret Ranking (lower is better):")
    sorted_regret = sorted(results.items(), key=lambda x: x[1]['cumulative_regret'])
    for i, (app_type, result) in enumerate(sorted_regret):
        print(f"  {i+1}. {app_type}: {result['cumulative_regret']:.3f}")
    
    # Efficiency analysis (reward / regret)
    print("\nEfficiency Analysis (reward / cumulative_regret):")
    efficiencies = {}
    for app_type, result in results.items():
        if result['cumulative_regret'] > 0:
            efficiency = result['avg_reward'] / result['cumulative_regret']
            efficiencies[app_type] = efficiency
    
    sorted_efficiency = sorted(efficiencies.items(), key=lambda x: x[1], reverse=True)
    for i, (app_type, efficiency) in enumerate(sorted_efficiency):
        print(f"  {i+1}. {app_type}: {efficiency:.6f}")

def run_cross_domain_comparison():
    """Run a more detailed cross-domain comparison."""
    
    print("\nCross-Domain Comparison:")
    print("=" * 40)
    
    # Define different scenarios
    scenarios = [
        ("ad_selection", 8, 6, "Online Advertising"),
        ("recommendation", 15, 10, "Content Recommendation"),
        ("clinical_trial", 4, 8, "Clinical Trials"),
        ("dynamic_pricing", 12, 5, "Dynamic Pricing")
    ]
    
    detailed_results = {}
    
    for app_type, n_arms, feature_dim, description in scenarios:
        print(f"\n  Running {description}...")
        
        app = BanditApplication(app_type, n_arms, feature_dim)
        rewards, regrets = app.run_experiment(n_steps=300)
        
        # Calculate detailed metrics
        detailed_results[description] = {
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'cumulative_regret': np.sum(regrets),
            'final_regret': np.mean(regrets[-50:]),
            'learning_rate': np.polyfit(range(len(regrets)), regrets, 1)[0],  # Slope of regret
            'convergence_step': np.argmin(np.abs(np.diff(regrets[-50:]))) + len(regrets) - 50
        }
    
    # Print detailed results
    print("\nDetailed Results:")
    for description, result in detailed_results.items():
        print(f"\n  {description}:")
        print(f"    Average Reward: {result['avg_reward']:.3f} ± {result['std_reward']:.3f}")
        print(f"    Cumulative Regret: {result['cumulative_regret']:.3f}")
        print(f"    Final Regret: {result['final_regret']:.3f}")
        print(f"    Learning Rate: {result['learning_rate']:.6f}")
        print(f"    Convergence Step: {result['convergence_step']}")
    
    return detailed_results

if __name__ == "__main__":
    # Run main experiment
    results = run_applications_example()
    
    # Analyze performance
    analyze_application_performance(results)
    
    # Run single application demo
    app, rewards, regrets = run_single_application_demo()
    
    # Demonstrate context generation
    demonstrate_context_generation()
    
    # Run cross-domain comparison
    detailed_results = run_cross_domain_comparison()
    
    print("\nExperiment completed!")
    print("Check the generated plots to see the application comparisons.")

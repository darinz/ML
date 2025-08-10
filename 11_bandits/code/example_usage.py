"""
Example usage of classical multi-armed bandit algorithms.

This script demonstrates how to use the epsilon-greedy, UCB, and Thompson sampling
algorithms with the bandit environment.
"""

import numpy as np
from epsilon_greedy import epsilon_greedy
from ucb import ucb
from thompson_sampling import thompson_sampling
from bandit_environment import BanditEnvironment, compare_algorithms, plot_regret_comparison

def run_bandit_experiment():
    """Run a complete bandit experiment comparing all algorithms."""
    
    # Define arm means (true reward probabilities)
    arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"Arm means: {arm_means}")
    print(f"Optimal arm: {np.argmax(arm_means)} (mean: {np.max(arm_means):.2f})")
    
    # Create bandit environment
    env = BanditEnvironment(arm_means)
    
    # Define algorithms to compare
    def epsilon_greedy_wrapper(env, T):
        """Wrapper for epsilon-greedy algorithm."""
        chosen_arms = []
        rewards = []
        empirical_means = epsilon_greedy(env.arms, epsilon=0.1, T=T)
        
        # Simulate the algorithm to get arm choices and rewards
        for t in range(T):
            if np.random.random() < 0.1:
                action = np.random.randint(0, len(env.arms))
            else:
                action = np.argmax(empirical_means)
            
            reward = env.arms[action].pull()
            chosen_arms.append(action)
            rewards.append(reward)
        
        return chosen_arms, rewards
    
    def ucb_wrapper(env, T):
        """Wrapper for UCB algorithm."""
        chosen_arms = []
        rewards = []
        empirical_means = ucb(env.arms, T)
        
        # Simulate the algorithm to get arm choices and rewards
        n_arms = len(env.arms)
        pulls = [0] * n_arms
        means = [0] * n_arms
        
        # Pull each arm once initially
        for i in range(n_arms):
            reward = env.arms[i].pull()
            means[i] = reward
            pulls[i] = 1
            chosen_arms.append(i)
            rewards.append(reward)
        
        # Main UCB loop
        for t in range(n_arms, T):
            ucb_values = []
            for i in range(n_arms):
                exploration_bonus = np.sqrt(2 * np.log(t) / pulls[i])
                ucb_values.append(means[i] + exploration_bonus)
            
            action = np.argmax(ucb_values)
            reward = env.arms[action].pull()
            
            pulls[action] += 1
            means[action] = ((means[action] * (pulls[action] - 1) + reward) / pulls[action])
            
            chosen_arms.append(action)
            rewards.append(reward)
        
        return chosen_arms, rewards
    
    def thompson_wrapper(env, T):
        """Wrapper for Thompson sampling algorithm."""
        chosen_arms = []
        rewards = []
        empirical_means = thompson_sampling(env.arms, T)
        
        # Simulate the algorithm to get arm choices and rewards
        n_arms = len(env.arms)
        alpha = [1] * n_arms
        beta = [1] * n_arms
        
        for t in range(T):
            samples = [np.random.beta(alpha[i], beta[i]) for i in range(n_arms)]
            action = np.argmax(samples)
            
            reward = env.arms[action].pull()
            
            if reward == 1:
                alpha[action] += 1
            else:
                beta[action] += 1
            
            chosen_arms.append(action)
            rewards.append(reward)
        
        return chosen_arms, rewards
    
    algorithms = {
        'Epsilon-Greedy (ε=0.1)': epsilon_greedy_wrapper,
        'UCB': ucb_wrapper,
        'Thompson Sampling': thompson_wrapper
    }
    
    # Run comparison
    print("\nRunning algorithm comparison...")
    T = 1000  # Number of time steps
    n_runs = 50  # Number of independent runs
    
    results = compare_algorithms(env, algorithms, T=T, n_runs=n_runs)
    
    # Print final results
    print("\nFinal cumulative regrets:")
    for name, regret in results.items():
        final_regret = regret[-1]
        print(f"{name}: {final_regret:.2f}")
    
    # Plot results
    print("\nGenerating plot...")
    plot_regret_comparison(results, T)
    
    return results

if __name__ == "__main__":
    print("Classical Multi-Armed Bandits Example")
    print("=" * 40)
    
    results = run_bandit_experiment()
    
    print("\nExperiment completed!")
    print("Check the generated plot to see the regret comparison.")

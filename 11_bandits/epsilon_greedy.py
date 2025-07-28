"""
Epsilon-Greedy Algorithm Implementation

This module implements the epsilon-greedy algorithm for multi-armed bandits.
The algorithm balances exploration (random selection) and exploitation (greedy selection)
based on a fixed epsilon parameter.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple


class EpsilonGreedy:
    """
    Epsilon-Greedy algorithm for multi-armed bandits.
    
    Attributes:
        n_arms (int): Number of arms
        epsilon (float): Exploration probability
        empirical_means (np.ndarray): Empirical means for each arm
        pulls (np.ndarray): Number of pulls for each arm
        total_pulls (int): Total number of pulls
    """
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        """
        Initialize epsilon-greedy algorithm.
        
        Args:
            n_arms (int): Number of arms
            epsilon (float): Exploration probability (0 <= epsilon <= 1)
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.total_pulls = 0
        
        # History for analysis
        self.rewards_history = []
        self.actions_history = []
        
    def select_arm(self) -> int:
        """
        Select an arm using epsilon-greedy strategy.
        
        Returns:
            int: Index of selected arm
        """
        if np.random.random() < self.epsilon:
            # Exploration: choose random arm
            action = np.random.randint(0, self.n_arms)
        else:
            # Exploitation: choose arm with highest empirical mean
            action = np.argmax(self.empirical_means)
        
        return action
    
    def update(self, arm: int, reward: float):
        """
        Update algorithm with observed reward.
        
        Args:
            arm (int): Index of pulled arm
            reward (float): Observed reward
        """
        # Update empirical mean
        self.pulls[arm] += 1
        self.empirical_means[arm] = (
            (self.empirical_means[arm] * (self.pulls[arm] - 1) + reward) 
            / self.pulls[arm]
        )
        
        # Update total pulls
        self.total_pulls += 1
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm)
    
    def get_empirical_means(self) -> np.ndarray:
        """Get current empirical means."""
        return self.empirical_means.copy()
    
    def get_pulls(self) -> np.ndarray:
        """Get number of pulls for each arm."""
        return self.pulls.copy()
    
    def get_best_arm(self) -> int:
        """Get the arm with highest empirical mean."""
        return np.argmax(self.empirical_means)


class DecayingEpsilonGreedy(EpsilonGreedy):
    """
    Epsilon-Greedy with decaying exploration rate.
    
    The epsilon parameter decreases over time to focus more on exploitation
    as the algorithm learns more about the arms.
    """
    
    def __init__(self, n_arms: int, initial_epsilon: float = 1.0, 
                 decay_rate: float = 0.99):
        """
        Initialize decaying epsilon-greedy algorithm.
        
        Args:
            n_arms (int): Number of arms
            initial_epsilon (float): Initial exploration probability
            decay_rate (float): Rate at which epsilon decays
        """
        super().__init__(n_arms, initial_epsilon)
        self.initial_epsilon = initial_epsilon
        self.decay_rate = decay_rate
    
    def select_arm(self) -> int:
        """
        Select an arm using decaying epsilon-greedy strategy.
        
        Returns:
            int: Index of selected arm
        """
        # Decay epsilon
        self.epsilon = self.initial_epsilon * (self.decay_rate ** self.total_pulls)
        
        return super().select_arm()


def run_epsilon_greedy_experiment(arm_means: List[float], 
                                n_steps: int = 1000,
                                epsilon: float = 0.1,
                                n_runs: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run epsilon-greedy experiment.
    
    Args:
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
        epsilon (float): Exploration probability
        n_runs (int): Number of independent runs
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Average rewards and regrets
    """
    n_arms = len(arm_means)
    optimal_arm = np.argmax(arm_means)
    optimal_reward = arm_means[optimal_arm]
    
    all_rewards = []
    all_regrets = []
    
    for run in range(n_runs):
        # Initialize algorithm
        bandit = EpsilonGreedy(n_arms, epsilon)
        
        run_rewards = []
        run_regrets = []
        
        for step in range(n_steps):
            # Select arm
            arm = bandit.select_arm()
            
            # Get reward
            reward = np.random.normal(arm_means[arm], 0.1)
            reward = np.clip(reward, 0, 1)  # Clip to [0, 1]
            
            # Update algorithm
            bandit.update(arm, reward)
            
            # Calculate regret
            regret = optimal_reward - arm_means[arm]
            
            run_rewards.append(reward)
            run_regrets.append(regret)
        
        all_rewards.append(run_rewards)
        all_regrets.append(run_regrets)
    
    # Average across runs
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_regrets = np.mean(all_regrets, axis=0)
    
    return avg_rewards, avg_regrets


def compare_epsilon_values(arm_means: List[float], 
                         epsilon_values: List[float] = [0.01, 0.1, 0.3],
                         n_steps: int = 1000,
                         n_runs: int = 50) -> dict:
    """
    Compare different epsilon values.
    
    Args:
        arm_means (List[float]): True means of arms
        epsilon_values (List[float]): Epsilon values to compare
        n_steps (int): Number of time steps
        n_runs (int): Number of independent runs
        
    Returns:
        dict: Results for each epsilon value
    """
    results = {}
    
    for epsilon in epsilon_values:
        print(f"Running epsilon-greedy with epsilon = {epsilon}")
        rewards, regrets = run_epsilon_greedy_experiment(
            arm_means, n_steps, epsilon, n_runs
        )
        
        results[epsilon] = {
            'rewards': rewards,
            'regrets': regrets,
            'cumulative_regret': np.sum(regrets),
            'final_regret': np.mean(regrets[-100:])
        }
    
    return results


def plot_epsilon_comparison(results: dict, n_steps: int):
    """
    Plot comparison of different epsilon values.
    
    Args:
        results (dict): Results from compare_epsilon_values
        n_steps (int): Number of time steps
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cumulative rewards
    for epsilon, data in results.items():
        cumulative_rewards = np.cumsum(data['rewards'])
        ax1.plot(range(n_steps), cumulative_rewards, 
                label=f'ε = {epsilon}', linewidth=2)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Reward Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative regrets
    for epsilon, data in results.items():
        cumulative_regrets = np.cumsum(data['regrets'])
        ax2.plot(range(n_steps), cumulative_regrets, 
                label=f'ε = {epsilon}', linewidth=2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Cumulative Regret Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_arm_selection(arm_means: List[float], 
                         n_steps: int = 1000,
                         epsilon: float = 0.1) -> dict:
    """
    Analyze arm selection behavior.
    
    Args:
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
        epsilon (float): Exploration probability
        
    Returns:
        dict: Analysis results
    """
    n_arms = len(arm_means)
    optimal_arm = np.argmax(arm_means)
    
    # Run single experiment for detailed analysis
    bandit = EpsilonGreedy(n_arms, epsilon)
    
    arm_counts = np.zeros(n_arms)
    arm_rewards = [[] for _ in range(n_arms)]
    
    for step in range(n_steps):
        arm = bandit.select_arm()
        arm_counts[arm] += 1
        
        reward = np.random.normal(arm_means[arm], 0.1)
        reward = np.clip(reward, 0, 1)
        arm_rewards[arm].append(reward)
        
        bandit.update(arm, reward)
    
    # Calculate statistics
    arm_avg_rewards = [np.mean(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    arm_std_rewards = [np.std(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    
    return {
        'arm_counts': arm_counts,
        'arm_avg_rewards': arm_avg_rewards,
        'arm_std_rewards': arm_std_rewards,
        'optimal_arm': optimal_arm,
        'empirical_means': bandit.get_empirical_means(),
        'total_pulls': bandit.total_pulls
    }


if __name__ == "__main__":
    # Example usage
    print("Running Epsilon-Greedy Example")
    
    # Define arm means
    arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Compare different epsilon values
    results = compare_epsilon_values(arm_means, n_steps=1000, n_runs=50)
    
    # Plot results
    plot_epsilon_comparison(results, 1000)
    
    # Analyze arm selection
    analysis = analyze_arm_selection(arm_means)
    
    print("\nArm Selection Analysis:")
    print(f"Optimal arm: {analysis['optimal_arm']}")
    print(f"Arm counts: {analysis['arm_counts']}")
    print(f"Empirical means: {analysis['empirical_means']}")
    print(f"Total pulls: {analysis['total_pulls']}") 
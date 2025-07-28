"""
Upper Confidence Bound (UCB) Algorithm Implementation

This module implements the UCB algorithm for multi-armed bandits.
The algorithm uses confidence intervals to balance exploration and exploitation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from scipy.stats import norm


class UCB:
    """
    Upper Confidence Bound algorithm for multi-armed bandits.
    
    Attributes:
        n_arms (int): Number of arms
        alpha (float): Exploration parameter
        empirical_means (np.ndarray): Empirical means for each arm
        pulls (np.ndarray): Number of pulls for each arm
        total_pulls (int): Total number of pulls
    """
    
    def __init__(self, n_arms: int, alpha: float = 2.0):
        """
        Initialize UCB algorithm.
        
        Args:
            n_arms (int): Number of arms
            alpha (float): Exploration parameter (typically 2.0)
        """
        self.n_arms = n_arms
        self.alpha = alpha
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.total_pulls = 0
        
        # History for analysis
        self.rewards_history = []
        self.actions_history = []
        self.ucb_values_history = []
        
    def select_arm(self) -> int:
        """
        Select an arm using UCB strategy.
        
        Returns:
            int: Index of selected arm
        """
        # If any arm hasn't been pulled, pull it
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                return arm
        
        # Calculate UCB values
        ucb_values = []
        for arm in range(self.n_arms):
            # Exploitation term
            exploitation = self.empirical_means[arm]
            
            # Exploration term
            exploration = self.alpha * np.sqrt(np.log(self.total_pulls) / self.pulls[arm])
            
            ucb_value = exploitation + exploration
            ucb_values.append(ucb_value)
        
        # Store UCB values for analysis
        self.ucb_values_history.append(ucb_values)
        
        return np.argmax(ucb_values)
    
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
    
    def get_confidence_intervals(self) -> List[Tuple[float, float]]:
        """
        Get confidence intervals for all arms.
        
        Returns:
            List[Tuple[float, float]]: (lower, upper) confidence intervals
        """
        intervals = []
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                intervals.append((0.0, float('inf')))
            else:
                radius = self.alpha * np.sqrt(np.log(self.total_pulls) / self.pulls[arm])
                lower = self.empirical_means[arm] - radius
                upper = self.empirical_means[arm] + radius
                intervals.append((lower, upper))
        return intervals


class UCB1(UCB):
    """
    UCB1 algorithm (original UCB with alpha=2).
    """
    
    def __init__(self, n_arms: int):
        super().__init__(n_arms, alpha=2.0)


class UCB2(UCB):
    """
    UCB2 algorithm with improved exploration schedule.
    """
    
    def __init__(self, n_arms: int, alpha: float = 2.0):
        super().__init__(n_arms, alpha)
        self.phase_lengths = self._calculate_phase_lengths()
        self.current_phase = 0
        self.pulls_in_phase = 0
    
    def _calculate_phase_lengths(self) -> List[int]:
        """Calculate phase lengths for UCB2."""
        phases = []
        for arm in range(self.n_arms):
            arm_phases = []
            for phase in range(100):  # Limit phases
                length = int(np.ceil((1 + self.alpha) ** phase))
                arm_phases.append(length)
            phases.append(arm_phases)
        return phases
    
    def select_arm(self) -> int:
        """Select arm using UCB2 strategy."""
        # If any arm hasn't been pulled, pull it
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                return arm
        
        # Check if we should switch to next phase
        if self.pulls_in_phase >= self.phase_lengths[self.current_phase][self.current_phase]:
            self.current_phase += 1
            self.pulls_in_phase = 0
        
        # Calculate UCB values
        ucb_values = []
        for arm in range(self.n_arms):
            exploitation = self.empirical_means[arm]
            exploration = self.alpha * np.sqrt(
                np.log(self.total_pulls) / self.pulls[arm]
            )
            ucb_value = exploitation + exploration
            ucb_values.append(ucb_value)
        
        self.ucb_values_history.append(ucb_values)
        selected_arm = np.argmax(ucb_values)
        
        # Update phase pulls
        self.pulls_in_phase += 1
        
        return selected_arm


def run_ucb_experiment(arm_means: List[float], 
                      n_steps: int = 1000,
                      alpha: float = 2.0,
                      n_runs: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run UCB experiment.
    
    Args:
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
        alpha (float): Exploration parameter
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
        bandit = UCB(n_arms, alpha)
        
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


def compare_ucb_variants(arm_means: List[float], 
                        n_steps: int = 1000,
                        n_runs: int = 50) -> dict:
    """
    Compare different UCB variants.
    
    Args:
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
        n_runs (int): Number of independent runs
        
    Returns:
        dict: Results for each UCB variant
    """
    results = {}
    
    # Test different alpha values
    alpha_values = [1.0, 2.0, 4.0]
    
    for alpha in alpha_values:
        print(f"Running UCB with alpha = {alpha}")
        rewards, regrets = run_ucb_experiment(
            arm_means, n_steps, alpha, n_runs
        )
        
        results[f'UCB(α={alpha})'] = {
            'rewards': rewards,
            'regrets': regrets,
            'cumulative_regret': np.sum(regrets),
            'final_regret': np.mean(regrets[-100:])
        }
    
    # Test UCB2
    print("Running UCB2")
    n_arms = len(arm_means)
    optimal_arm = np.argmax(arm_means)
    optimal_reward = arm_means[optimal_arm]
    
    all_rewards = []
    all_regrets = []
    
    for run in range(n_runs):
        bandit = UCB2(n_arms)
        
        run_rewards = []
        run_regrets = []
        
        for step in range(n_steps):
            arm = bandit.select_arm()
            reward = np.random.normal(arm_means[arm], 0.1)
            reward = np.clip(reward, 0, 1)
            bandit.update(arm, reward)
            
            regret = optimal_reward - arm_means[arm]
            run_rewards.append(reward)
            run_regrets.append(regret)
        
        all_rewards.append(run_rewards)
        all_regrets.append(run_regrets)
    
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_regrets = np.mean(all_regrets, axis=0)
    
    results['UCB2'] = {
        'rewards': avg_rewards,
        'regrets': avg_regrets,
        'cumulative_regret': np.sum(avg_regrets),
        'final_regret': np.mean(avg_regrets[-100:])
    }
    
    return results


def plot_ucb_comparison(results: dict, n_steps: int):
    """
    Plot comparison of different UCB variants.
    
    Args:
        results (dict): Results from compare_ucb_variants
        n_steps (int): Number of time steps
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cumulative rewards
    for variant, data in results.items():
        cumulative_rewards = np.cumsum(data['rewards'])
        ax1.plot(range(n_steps), cumulative_rewards, 
                label=variant, linewidth=2)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('UCB Variants - Cumulative Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative regrets
    for variant, data in results.items():
        cumulative_regrets = np.cumsum(data['regrets'])
        ax2.plot(range(n_steps), cumulative_regrets, 
                label=variant, linewidth=2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('UCB Variants - Cumulative Regret')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_ucb_behavior(arm_means: List[float], 
                        n_steps: int = 1000,
                        alpha: float = 2.0) -> dict:
    """
    Analyze UCB behavior in detail.
    
    Args:
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
        alpha (float): Exploration parameter
        
    Returns:
        dict: Detailed analysis results
    """
    n_arms = len(arm_means)
    optimal_arm = np.argmax(arm_means)
    
    # Run single experiment for detailed analysis
    bandit = UCB(n_arms, alpha)
    
    arm_counts = np.zeros(n_arms)
    arm_rewards = [[] for _ in range(n_arms)]
    ucb_values_over_time = []
    
    for step in range(n_steps):
        arm = bandit.select_arm()
        arm_counts[arm] += 1
        
        reward = np.random.normal(arm_means[arm], 0.1)
        reward = np.clip(reward, 0, 1)
        arm_rewards[arm].append(reward)
        
        bandit.update(arm, reward)
        
        # Store UCB values
        if step >= n_arms:  # After initial exploration
            ucb_values_over_time.append(bandit.ucb_values_history[-1])
    
    # Calculate statistics
    arm_avg_rewards = [np.mean(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    arm_std_rewards = [np.std(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    
    # Analyze UCB value evolution
    ucb_values_array = np.array(ucb_values_over_time)
    
    return {
        'arm_counts': arm_counts,
        'arm_avg_rewards': arm_avg_rewards,
        'arm_std_rewards': arm_std_rewards,
        'optimal_arm': optimal_arm,
        'empirical_means': bandit.get_empirical_means(),
        'total_pulls': bandit.total_pulls,
        'ucb_values_over_time': ucb_values_array,
        'confidence_intervals': bandit.get_confidence_intervals()
    }


def plot_ucb_analysis(analysis: dict, n_steps: int):
    """
    Plot detailed UCB analysis.
    
    Args:
        analysis (dict): Results from analyze_ucb_behavior
        n_steps (int): Number of time steps
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Arm selection counts
    arms = range(len(analysis['arm_counts']))
    ax1.bar(arms, analysis['arm_counts'])
    ax1.set_xlabel('Arm')
    ax1.set_ylabel('Number of Pulls')
    ax1.set_title('Arm Selection Distribution')
    ax1.axvline(analysis['optimal_arm'], color='red', linestyle='--', 
                label=f'Optimal Arm ({analysis["optimal_arm"]})')
    ax1.legend()
    
    # Empirical means vs true means
    true_means = [0.1, 0.2, 0.3, 0.4, 0.5]  # Assuming these are the true means
    empirical_means = analysis['empirical_means']
    ax2.bar([f'Arm {i}' for i in arms], empirical_means, alpha=0.7, label='Empirical')
    ax2.bar([f'Arm {i}' for i in arms], true_means, alpha=0.3, label='True')
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('Empirical vs True Means')
    ax2.legend()
    
    # UCB values over time (for first few arms)
    if len(analysis['ucb_values_over_time']) > 0:
        ucb_values = analysis['ucb_values_over_time']
        for arm in range(min(3, len(arms))):
            ax3.plot(ucb_values[:, arm], label=f'Arm {arm}')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('UCB Value')
        ax3.set_title('UCB Values Over Time')
        ax3.legend()
    
    # Confidence intervals
    intervals = analysis['confidence_intervals']
    centers = [(lower + upper) / 2 for lower, upper in intervals]
    errors = [(upper - lower) / 2 for lower, upper in intervals]
    
    ax4.errorbar(arms, centers, yerr=errors, fmt='o', capsize=5)
    ax4.set_xlabel('Arm')
    ax4.set_ylabel('Mean ± Confidence Interval')
    ax4.set_title('Confidence Intervals')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running UCB Example")
    
    # Define arm means
    arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Compare different UCB variants
    results = compare_ucb_variants(arm_means, n_steps=1000, n_runs=50)
    
    # Plot results
    plot_ucb_comparison(results, 1000)
    
    # Analyze UCB behavior
    analysis = analyze_ucb_behavior(arm_means)
    
    print("\nUCB Analysis:")
    print(f"Optimal arm: {analysis['optimal_arm']}")
    print(f"Arm counts: {analysis['arm_counts']}")
    print(f"Empirical means: {analysis['empirical_means']}")
    print(f"Total pulls: {analysis['total_pulls']}")
    
    # Plot detailed analysis
    plot_ucb_analysis(analysis, 1000) 
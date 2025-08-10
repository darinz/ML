"""
Regret Analysis and Visualization

This module provides tools for analyzing and visualizing regret in multi-armed bandits.
It includes functions for computing different types of regret and creating informative plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from scipy import stats


class RegretAnalyzer:
    """
    Regret analysis for multi-armed bandit algorithms.
    
    This class provides methods to compute and analyze different types of regret
    and create visualizations for algorithm comparison.
    """
    
    def __init__(self, arm_means: List[float]):
        """
        Initialize regret analyzer.
        
        Args:
            arm_means (List[float]): True means of all arms
        """
        self.arm_means = np.array(arm_means)
        self.optimal_arm = np.argmax(arm_means)
        self.optimal_reward = arm_means[self.optimal_arm]
        self.n_arms = len(arm_means)
        
    def compute_instantaneous_regret(self, actions: List[int], 
                                   rewards: List[float]) -> np.ndarray:
        """
        Compute instantaneous regret for each time step.
        
        Args:
            actions (List[int]): Sequence of chosen arms
            rewards (List[float]): Sequence of observed rewards
            
        Returns:
            np.ndarray: Instantaneous regret at each time step
        """
        regrets = []
        for action, reward in zip(actions, rewards):
            # Instantaneous regret = optimal reward - reward of chosen arm
            regret = self.optimal_reward - self.arm_means[action]
            regrets.append(regret)
        return np.array(regrets)
    
    def compute_cumulative_regret(self, actions: List[int], 
                                rewards: List[float]) -> np.ndarray:
        """
        Compute cumulative regret over time.
        
        Args:
            actions (List[int]): Sequence of chosen arms
            rewards (List[float]): Sequence of observed rewards
            
        Returns:
            np.ndarray: Cumulative regret at each time step
        """
        instantaneous_regret = self.compute_instantaneous_regret(actions, rewards)
        return np.cumsum(instantaneous_regret)
    
    def compute_pseudo_regret(self, actions: List[int]) -> np.ndarray:
        """
        Compute pseudo-regret (expected regret without noise).
        
        Args:
            actions (List[int]): Sequence of chosen arms
            
        Returns:
            np.ndarray: Pseudo-regret at each time step
        """
        regrets = []
        for action in actions:
            regret = self.optimal_reward - self.arm_means[action]
            regrets.append(regret)
        return np.array(regrets)
    
    def compute_cumulative_pseudo_regret(self, actions: List[int]) -> np.ndarray:
        """
        Compute cumulative pseudo-regret.
        
        Args:
            actions (List[int]): Sequence of chosen arms
            
        Returns:
            np.ndarray: Cumulative pseudo-regret at each time step
        """
        pseudo_regret = self.compute_pseudo_regret(actions)
        return np.cumsum(pseudo_regret)
    
    def compute_arm_pulls(self, actions: List[int]) -> np.ndarray:
        """
        Compute number of pulls for each arm.
        
        Args:
            actions (List[int]): Sequence of chosen arms
            
        Returns:
            np.ndarray: Number of pulls for each arm
        """
        pulls = np.zeros(self.n_arms)
        for action in actions:
            pulls[action] += 1
        return pulls
    
    def compute_empirical_means(self, actions: List[int], 
                              rewards: List[float]) -> np.ndarray:
        """
        Compute empirical means for each arm.
        
        Args:
            actions (List[int]): Sequence of chosen arms
            rewards (List[float]): Sequence of observed rewards
            
        Returns:
            np.ndarray: Empirical means for each arm
        """
        empirical_means = np.zeros(self.n_arms)
        pulls = np.zeros(self.n_arms)
        
        for action, reward in zip(actions, rewards):
            pulls[action] += 1
            empirical_means[action] = (
                (empirical_means[action] * (pulls[action] - 1) + reward) 
                / pulls[action]
            )
        
        return empirical_means


def compare_algorithms(algorithm_results: Dict[str, Dict], 
                     arm_means: List[float],
                     n_steps: int) -> Dict[str, np.ndarray]:
    """
    Compare multiple algorithms using regret analysis.
    
    Args:
        algorithm_results (Dict[str, Dict]): Results from different algorithms
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
        
    Returns:
        Dict[str, np.ndarray]: Cumulative regrets for each algorithm
    """
    analyzer = RegretAnalyzer(arm_means)
    cumulative_regrets = {}
    
    for algorithm_name, results in algorithm_results.items():
        actions = results['actions']
        rewards = results['rewards']
        
        cumulative_regret = analyzer.compute_cumulative_regret(actions, rewards)
        cumulative_regrets[algorithm_name] = cumulative_regret
    
    return cumulative_regrets


def plot_regret_comparison(cumulative_regrets: Dict[str, np.ndarray], 
                          n_steps: int,
                          title: str = "Algorithm Regret Comparison"):
    """
    Plot cumulative regret comparison for multiple algorithms.
    
    Args:
        cumulative_regrets (Dict[str, np.ndarray]): Cumulative regrets for each algorithm
        n_steps (int): Number of time steps
        title (str): Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative regrets
    for algorithm_name, regret in cumulative_regrets.items():
        plt.plot(range(1, n_steps + 1), regret, label=algorithm_name, linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_regret_rate(cumulative_regrets: Dict[str, np.ndarray], 
                    n_steps: int,
                    title: str = "Regret Rate Comparison"):
    """
    Plot regret rate (regret / sqrt(t)) for algorithm comparison.
    
    Args:
        cumulative_regrets (Dict[str, np.ndarray]): Cumulative regrets for each algorithm
        n_steps (int): Number of time steps
        title (str): Plot title
    """
    plt.figure(figsize=(12, 8))
    
    time_steps = np.arange(1, n_steps + 1)
    
    for algorithm_name, regret in cumulative_regrets.items():
        regret_rate = regret / np.sqrt(time_steps)
        plt.plot(time_steps, regret_rate, label=algorithm_name, linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Regret Rate (R(t) / √t)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_arm_selection_analysis(algorithm_results: Dict[str, Dict], 
                              arm_means: List[float]):
    """
    Plot arm selection analysis for multiple algorithms.
    
    Args:
        algorithm_results (Dict[str, Dict]): Results from different algorithms
        arm_means (List[float]): True means of arms
    """
    n_algorithms = len(algorithm_results)
    n_arms = len(arm_means)
    
    fig, axes = plt.subplots(2, n_algorithms, figsize=(4 * n_algorithms, 8))
    if n_algorithms == 1:
        axes = axes.reshape(2, 1)
    
    for i, (algorithm_name, results) in enumerate(algorithm_results.items()):
        actions = results['actions']
        
        # Compute arm pulls
        pulls = np.zeros(n_arms)
        for action in actions:
            pulls[action] += 1
        
        # Arm selection distribution
        axes[0, i].bar(range(n_arms), pulls)
        axes[0, i].set_xlabel('Arm')
        axes[0, i].set_ylabel('Number of Pulls')
        axes[0, i].set_title(f'{algorithm_name} - Arm Selection')
        axes[0, i].axvline(np.argmax(arm_means), color='red', linestyle='--', 
                          label='Optimal Arm')
        axes[0, i].legend()
        
        # Empirical vs true means
        analyzer = RegretAnalyzer(arm_means)
        empirical_means = analyzer.compute_empirical_means(actions, results['rewards'])
        
        x = np.arange(n_arms)
        width = 0.35
        
        axes[1, i].bar(x - width/2, arm_means, width, label='True Means', alpha=0.7)
        axes[1, i].bar(x + width/2, empirical_means, width, label='Empirical Means', alpha=0.7)
        axes[1, i].set_xlabel('Arm')
        axes[1, i].set_ylabel('Mean Reward')
        axes[1, i].set_title(f'{algorithm_name} - Means Comparison')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.show()


def plot_regret_evolution(algorithm_results: Dict[str, Dict], 
                        arm_means: List[float],
                        window_size: int = 100):
    """
    Plot regret evolution over time with moving average.
    
    Args:
        algorithm_results (Dict[str, Dict]): Results from different algorithms
        arm_means (List[float]): True means of arms
        window_size (int): Window size for moving average
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    for algorithm_name, results in algorithm_results.items():
        actions = results['actions']
        rewards = results['rewards']
        
        analyzer = RegretAnalyzer(arm_means)
        instantaneous_regret = analyzer.compute_instantaneous_regret(actions, rewards)
        
        # Moving average of instantaneous regret
        moving_avg = np.convolve(instantaneous_regret, 
                                np.ones(window_size) / window_size, 
                                mode='valid')
        
        time_steps = range(len(moving_avg))
        ax1.plot(time_steps, moving_avg, label=algorithm_name, linewidth=2)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel(f'Moving Average Regret (window={window_size})')
    ax1.set_title('Instantaneous Regret Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot cumulative regret
    cumulative_regrets = compare_algorithms(algorithm_results, arm_means, 
                                          len(list(algorithm_results.values())[0]['actions']))
    
    for algorithm_name, regret in cumulative_regrets.items():
        ax2.plot(range(1, len(regret) + 1), regret, label=algorithm_name, linewidth=2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Cumulative Regret Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compute_regret_statistics(cumulative_regrets: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    Compute statistical summary of regret performance.
    
    Args:
        cumulative_regrets (Dict[str, np.ndarray]): Cumulative regrets for each algorithm
        
    Returns:
        Dict[str, Dict]: Statistical summary for each algorithm
    """
    statistics = {}
    
    for algorithm_name, regret in cumulative_regrets.items():
        final_regret = regret[-1]
        avg_regret_rate = np.mean(regret / np.sqrt(np.arange(1, len(regret) + 1)))
        
        # Compute regret growth rate (slope of log-log plot)
        log_t = np.log(np.arange(1, len(regret) + 1))
        log_regret = np.log(regret + 1e-10)  # Add small constant to avoid log(0)
        slope, _, _, _, _ = stats.linregress(log_t, log_regret)
        
        statistics[algorithm_name] = {
            'final_regret': final_regret,
            'avg_regret_rate': avg_regret_rate,
            'regret_growth_rate': slope,
            'total_regret': np.sum(regret),
            'regret_std': np.std(regret)
        }
    
    return statistics


def plot_regret_statistics(statistics: Dict[str, Dict]):
    """
    Plot regret statistics comparison.
    
    Args:
        statistics (Dict[str, Dict]): Statistical summary for each algorithm
    """
    algorithms = list(statistics.keys())
    metrics = ['final_regret', 'avg_regret_rate', 'regret_growth_rate']
    metric_names = ['Final Regret', 'Avg Regret Rate', 'Regret Growth Rate']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [statistics[alg][metric] for alg in algorithms]
        
        axes[i].bar(algorithms, values)
        axes[i].set_ylabel(metric_name)
        axes[i].set_title(f'{metric_name} Comparison')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def create_regret_report(algorithm_results: Dict[str, Dict], 
                       arm_means: List[float]) -> str:
    """
    Create a comprehensive regret analysis report.
    
    Args:
        algorithm_results (Dict[str, Dict]): Results from different algorithms
        arm_means (List[float]): True means of arms
        
    Returns:
        str: Formatted report
    """
    analyzer = RegretAnalyzer(arm_means)
    cumulative_regrets = compare_algorithms(algorithm_results, arm_means, 
                                          len(list(algorithm_results.values())[0]['actions']))
    statistics = compute_regret_statistics(cumulative_regrets)
    
    report = "=" * 60 + "\n"
    report += "MULTI-ARMED BANDIT REGRET ANALYSIS REPORT\n"
    report += "=" * 60 + "\n\n"
    
    # Problem setup
    report += "PROBLEM SETUP:\n"
    report += f"Number of arms: {len(arm_means)}\n"
    report += f"Optimal arm: {np.argmax(arm_means)} (mean: {np.max(arm_means):.3f})\n"
    report += f"Arm means: {[f'{m:.3f}' for m in arm_means]}\n\n"
    
    # Algorithm comparison
    report += "ALGORITHM COMPARISON:\n"
    report += "-" * 30 + "\n"
    
    for algorithm_name, stats in statistics.items():
        report += f"\n{algorithm_name}:\n"
        report += f"  Final regret: {stats['final_regret']:.3f}\n"
        report += f"  Average regret rate: {stats['avg_regret_rate']:.3f}\n"
        report += f"  Regret growth rate: {stats['regret_growth_rate']:.3f}\n"
        report += f"  Total regret: {stats['total_regret']:.3f}\n"
        report += f"  Regret std: {stats['regret_std']:.3f}\n"
    
    # Best performing algorithm
    best_algorithm = min(statistics.keys(), 
                        key=lambda x: statistics[x]['final_regret'])
    report += f"\nBEST PERFORMING ALGORITHM: {best_algorithm}\n"
    report += f"Final regret: {statistics[best_algorithm]['final_regret']:.3f}\n"
    
    return report


def run_comprehensive_analysis(algorithm_results: Dict[str, Dict], 
                             arm_means: List[float],
                             n_steps: int):
    """
    Run comprehensive regret analysis with all visualizations.
    
    Args:
        algorithm_results (Dict[str, Dict]): Results from different algorithms
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
    """
    print("Running Comprehensive Regret Analysis...")
    
    # Compute cumulative regrets
    cumulative_regrets = compare_algorithms(algorithm_results, arm_means, n_steps)
    
    # Create visualizations
    plot_regret_comparison(cumulative_regrets, n_steps)
    plot_regret_rate(cumulative_regrets, n_steps)
    plot_arm_selection_analysis(algorithm_results, arm_means)
    plot_regret_evolution(algorithm_results, arm_means)
    
    # Compute and plot statistics
    statistics = compute_regret_statistics(cumulative_regrets)
    plot_regret_statistics(statistics)
    
    # Generate report
    report = create_regret_report(algorithm_results, arm_means)
    print(report)


if __name__ == "__main__":
    # Example usage
    print("Running Regret Analysis Example")
    
    # Define arm means
    arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_steps = 1000
    
    # Simulate results from different algorithms
    np.random.seed(42)
    
    # Simulate epsilon-greedy results
    epsilon_actions = []
    epsilon_rewards = []
    for step in range(n_steps):
        if np.random.random() < 0.1:  # epsilon = 0.1
            action = np.random.randint(0, 5)
        else:
            action = 4  # Assume it learned the best arm
        epsilon_actions.append(action)
        reward = np.random.normal(arm_means[action], 0.1)
        epsilon_rewards.append(reward)
    
    # Simulate UCB results
    ucb_actions = []
    ucb_rewards = []
    for step in range(n_steps):
        if step < 5:  # Initial exploration
            action = step
        else:
            action = 4  # Assume UCB converges to best arm
        ucb_actions.append(action)
        reward = np.random.normal(arm_means[action], 0.1)
        ucb_rewards.append(reward)
    
    # Create results dictionary
    algorithm_results = {
        'Epsilon-Greedy': {
            'actions': epsilon_actions,
            'rewards': epsilon_rewards
        },
        'UCB': {
            'actions': ucb_actions,
            'rewards': ucb_rewards
        }
    }
    
    # Run comprehensive analysis
    run_comprehensive_analysis(algorithm_results, arm_means, n_steps) 
"""
Linear UCB (LinUCB) Algorithm Implementation

This module implements the Linear UCB algorithm for linear bandits.
The algorithm extends UCB to handle linear reward functions with feature vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from scipy.linalg import solve
from scipy.stats import multivariate_normal


class LinUCB:
    """
    Linear UCB algorithm for linear bandits.
    
    The algorithm assumes rewards are linear functions of arm features:
    r_t = <theta*, x_{a_t}> + eta_t
    
    Attributes:
        d (int): Feature dimension
        alpha (float): Exploration parameter
        lambda_reg (float): Regularization parameter
        A (np.ndarray): Design matrix
        b (np.ndarray): Cumulative rewards
        theta_hat (np.ndarray): Parameter estimate
    """
    
    def __init__(self, d: int, alpha: float = 1.0, lambda_reg: float = 1.0):
        """
        Initialize LinUCB algorithm.
        
        Args:
            d (int): Feature dimension
            alpha (float): Exploration parameter
            lambda_reg (float): Regularization parameter
        """
        self.d = d
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Initialize design matrix and parameter estimate
        self.A = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.theta_hat = np.zeros(d)
        
        # History for analysis
        self.rewards_history = []
        self.actions_history = []
        self.ucb_values_history = []
        
    def select_arm(self, arms: List[np.ndarray]) -> int:
        """
        Select arm using LinUCB algorithm.
        
        Args:
            arms (List[np.ndarray]): List of feature vectors for each arm
            
        Returns:
            int: Index of selected arm
        """
        # Update parameter estimate
        self.theta_hat = solve(self.A, self.b)
        
        # Calculate UCB values for all arms
        ucb_values = []
        for x in arms:
            # Exploitation term
            exploitation = np.dot(self.theta_hat, x)
            
            # Exploration term
            exploration = self.alpha * np.sqrt(np.dot(x, solve(self.A, x)))
            
            ucb_value = exploitation + exploration
            ucb_values.append(ucb_value)
        
        # Store UCB values for analysis
        self.ucb_values_history.append(ucb_values)
        
        return np.argmax(ucb_values)
    
    def update(self, arm_idx: int, reward: float, arms: List[np.ndarray]):
        """
        Update algorithm with observed reward.
        
        Args:
            arm_idx (int): Index of pulled arm
            reward (float): Observed reward
            arms (List[np.ndarray]): List of feature vectors for each arm
        """
        x = arms[arm_idx]
        
        # Update design matrix and cumulative rewards
        self.A += np.outer(x, x)
        self.b += reward * x
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm_idx)
    
    def get_parameter_estimate(self) -> np.ndarray:
        """Get current parameter estimate."""
        return self.theta_hat.copy()
    
    def get_confidence_ellipsoid(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence ellipsoid parameters.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Center and covariance matrix
        """
        center = self.theta_hat
        covariance = solve(self.A, np.eye(self.d))
        return center, covariance
    
    def get_ucb_values(self, arms: List[np.ndarray]) -> List[float]:
        """
        Get UCB values for all arms.
        
        Args:
            arms (List[np.ndarray]): List of feature vectors for each arm
            
        Returns:
            List[float]: UCB values for each arm
        """
        self.theta_hat = solve(self.A, self.b)
        
        ucb_values = []
        for x in arms:
            exploitation = np.dot(self.theta_hat, x)
            exploration = self.alpha * np.sqrt(np.dot(x, solve(self.A, x)))
            ucb_values.append(exploitation + exploration)
        
        return ucb_values


class LinUCBDisjoint(LinUCB):
    """
    LinUCB with disjoint models for each arm.
    
    This variant maintains separate linear models for each arm,
    allowing for more flexible arm-specific reward functions.
    """
    
    def __init__(self, d: int, n_arms: int, alpha: float = 1.0, lambda_reg: float = 1.0):
        """
        Initialize LinUCB with disjoint models.
        
        Args:
            d (int): Feature dimension
            n_arms (int): Number of arms
            alpha (float): Exploration parameter
            lambda_reg (float): Regularization parameter
        """
        self.d = d
        self.n_arms = n_arms
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Separate models for each arm
        self.A = [lambda_reg * np.eye(d) for _ in range(n_arms)]
        self.b = [np.zeros(d) for _ in range(n_arms)]
        self.theta_hat = [np.zeros(d) for _ in range(n_arms)]
        
        # History for analysis
        self.rewards_history = []
        self.actions_history = []
        self.ucb_values_history = []
    
    def select_arm(self, arms: List[np.ndarray]) -> int:
        """
        Select arm using disjoint LinUCB.
        
        Args:
            arms (List[np.ndarray]): List of feature vectors for each arm
            
        Returns:
            int: Index of selected arm
        """
        ucb_values = []
        
        for i in range(self.n_arms):
            # Update parameter estimate for arm i
            self.theta_hat[i] = solve(self.A[i], self.b[i])
            
            # Calculate UCB value for arm i
            x = arms[i]
            exploitation = np.dot(self.theta_hat[i], x)
            exploration = self.alpha * np.sqrt(np.dot(x, solve(self.A[i], x)))
            ucb_value = exploitation + exploration
            ucb_values.append(ucb_value)
        
        # Store UCB values for analysis
        self.ucb_values_history.append(ucb_values)
        
        return np.argmax(ucb_values)
    
    def update(self, arm_idx: int, reward: float, arms: List[np.ndarray]):
        """
        Update algorithm with observed reward.
        
        Args:
            arm_idx (int): Index of pulled arm
            reward (float): Observed reward
            arms (List[np.ndarray]): List of feature vectors for each arm
        """
        x = arms[arm_idx]
        
        # Update design matrix and cumulative rewards for the chosen arm
        self.A[arm_idx] += np.outer(x, x)
        self.b[arm_idx] += reward * x
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm_idx)
    
    def get_parameter_estimate(self) -> List[np.ndarray]:
        """Get current parameter estimates for all arms."""
        return [theta.copy() for theta in self.theta_hat]


def run_linucb_experiment(theta_star: np.ndarray, 
                         arm_features: List[np.ndarray],
                         n_steps: int = 1000,
                         alpha: float = 1.0,
                         n_runs: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run LinUCB experiment.
    
    Args:
        theta_star (np.ndarray): True parameter vector
        arm_features (List[np.ndarray]): Feature vectors for each arm
        n_steps (int): Number of time steps
        alpha (float): Exploration parameter
        n_runs (int): Number of independent runs
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Average rewards and regrets
    """
    d = len(theta_star)
    n_arms = len(arm_features)
    
    # Calculate optimal rewards for each arm
    optimal_rewards = [np.dot(theta_star, x) for x in arm_features]
    optimal_arm = np.argmax(optimal_rewards)
    optimal_reward = optimal_rewards[optimal_arm]
    
    all_rewards = []
    all_regrets = []
    
    for run in range(n_runs):
        # Initialize algorithm
        bandit = LinUCB(d, alpha)
        
        run_rewards = []
        run_regrets = []
        
        for step in range(n_steps):
            # Select arm
            arm_idx = bandit.select_arm(arm_features)
            
            # Get reward
            expected_reward = np.dot(theta_star, arm_features[arm_idx])
            noise = np.random.normal(0, 0.1)
            reward = expected_reward + noise
            
            # Update algorithm
            bandit.update(arm_idx, reward, arm_features)
            
            # Calculate regret
            regret = optimal_reward - expected_reward
            
            run_rewards.append(reward)
            run_regrets.append(regret)
        
        all_rewards.append(run_rewards)
        all_regrets.append(run_regrets)
    
    # Average across runs
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_regrets = np.mean(all_regrets, axis=0)
    
    return avg_rewards, avg_regrets


def compare_linucb_variants(theta_star: np.ndarray, 
                           arm_features: List[np.ndarray],
                           n_steps: int = 1000,
                           n_runs: int = 50) -> dict:
    """
    Compare different LinUCB variants.
    
    Args:
        theta_star (np.ndarray): True parameter vector
        arm_features (List[np.ndarray]): Feature vectors for each arm
        n_steps (int): Number of time steps
        n_runs (int): Number of independent runs
        
    Returns:
        dict: Results for each LinUCB variant
    """
    results = {}
    
    # Test different alpha values
    alpha_values = [0.5, 1.0, 2.0]
    
    for alpha in alpha_values:
        print(f"Running LinUCB with alpha = {alpha}")
        rewards, regrets = run_linucb_experiment(
            theta_star, arm_features, n_steps, alpha, n_runs
        )
        
        results[f'LinUCB(α={alpha})'] = {
            'rewards': rewards,
            'regrets': regrets,
            'cumulative_regret': np.sum(regrets),
            'final_regret': np.mean(regrets[-100:])
        }
    
    # Test disjoint LinUCB
    print("Running Disjoint LinUCB")
    d = len(theta_star)
    n_arms = len(arm_features)
    
    all_rewards = []
    all_regrets = []
    
    for run in range(n_runs):
        bandit = LinUCBDisjoint(d, n_arms, alpha=1.0)
        
        run_rewards = []
        run_regrets = []
        
        for step in range(n_steps):
            arm_idx = bandit.select_arm(arm_features)
            expected_reward = np.dot(theta_star, arm_features[arm_idx])
            noise = np.random.normal(0, 0.1)
            reward = expected_reward + noise
            bandit.update(arm_idx, reward, arm_features)
            
            optimal_reward = max([np.dot(theta_star, x) for x in arm_features])
            regret = optimal_reward - expected_reward
            
            run_rewards.append(reward)
            run_regrets.append(regret)
        
        all_rewards.append(run_rewards)
        all_regrets.append(run_regrets)
    
    avg_rewards = np.mean(all_rewards, axis=0)
    avg_regrets = np.mean(all_regrets, axis=0)
    
    results['Disjoint LinUCB'] = {
        'rewards': avg_rewards,
        'regrets': avg_regrets,
        'cumulative_regret': np.sum(avg_regrets),
        'final_regret': np.mean(avg_regrets[-100:])
    }
    
    return results


def plot_linucb_comparison(results: dict, n_steps: int):
    """
    Plot comparison of different LinUCB variants.
    
    Args:
        results (dict): Results from compare_linucb_variants
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
    ax1.set_title('LinUCB Variants - Cumulative Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative regrets
    for variant, data in results.items():
        cumulative_regrets = np.cumsum(data['regrets'])
        ax2.plot(range(n_steps), cumulative_regrets, 
                label=variant, linewidth=2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('LinUCB Variants - Cumulative Regret')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_linucb_behavior(theta_star: np.ndarray, 
                           arm_features: List[np.ndarray],
                           n_steps: int = 1000,
                           alpha: float = 1.0) -> dict:
    """
    Analyze LinUCB behavior in detail.
    
    Args:
        theta_star (np.ndarray): True parameter vector
        arm_features (List[np.ndarray]): Feature vectors for each arm
        n_steps (int): Number of time steps
        alpha (float): Exploration parameter
        
    Returns:
        dict: Detailed analysis results
    """
    d = len(theta_star)
    n_arms = len(arm_features)
    
    # Run single experiment for detailed analysis
    bandit = LinUCB(d, alpha)
    
    arm_counts = np.zeros(n_arms)
    arm_rewards = [[] for _ in range(n_arms)]
    parameter_estimates = []
    
    for step in range(n_steps):
        arm_idx = bandit.select_arm(arm_features)
        arm_counts[arm_idx] += 1
        
        expected_reward = np.dot(theta_star, arm_features[arm_idx])
        noise = np.random.normal(0, 0.1)
        reward = expected_reward + noise
        arm_rewards[arm_idx].append(reward)
        
        bandit.update(arm_idx, reward, arm_features)
        
        # Store parameter estimates
        parameter_estimates.append(bandit.get_parameter_estimate())
    
    # Calculate statistics
    arm_avg_rewards = [np.mean(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    arm_std_rewards = [np.std(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    
    # Analyze parameter estimation
    parameter_estimates_array = np.array(parameter_estimates)
    
    return {
        'arm_counts': arm_counts,
        'arm_avg_rewards': arm_avg_rewards,
        'arm_std_rewards': arm_std_rewards,
        'optimal_arm': np.argmax([np.dot(theta_star, x) for x in arm_features]),
        'final_parameter_estimate': bandit.get_parameter_estimate(),
        'true_parameter': theta_star,
        'parameter_estimates_over_time': parameter_estimates_array,
        'ucb_values_history': bandit.ucb_values_history,
        'confidence_ellipsoid': bandit.get_confidence_ellipsoid()
    }


def plot_linucb_analysis(analysis: dict, n_steps: int):
    """
    Plot detailed LinUCB analysis.
    
    Args:
        analysis (dict): Results from analyze_linucb_behavior
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
    
    # Parameter estimation error
    true_param = analysis['true_parameter']
    final_estimate = analysis['final_parameter_estimate']
    param_error = np.linalg.norm(true_param - final_estimate)
    
    ax2.plot(true_param, 'o-', label='True Parameter', linewidth=2)
    ax2.plot(final_estimate, 's-', label='Estimated Parameter', linewidth=2)
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title(f'Parameter Estimation (Error: {param_error:.3f})')
    ax2.legend()
    
    # Parameter estimation over time
    if len(analysis['parameter_estimates_over_time']) > 0:
        param_estimates = analysis['parameter_estimates_over_time']
        for i in range(min(3, len(true_param))):
            ax3.plot(param_estimates[:, i], label=f'Parameter {i}')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Parameter Value')
        ax3.set_title('Parameter Estimation Over Time')
        ax3.legend()
    
    # UCB values over time (for first few arms)
    if len(analysis['ucb_values_history']) > 0:
        ucb_values = analysis['ucb_values_history']
        ucb_array = np.array(ucb_values)
        for arm in range(min(3, len(arms))):
            ax4.plot(ucb_array[:, arm], label=f'Arm {arm}')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('UCB Value')
        ax4.set_title('UCB Values Over Time')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running LinUCB Example")
    
    # Define problem parameters
    d = 5  # Feature dimension
    n_arms = 10  # Number of arms
    
    # Generate random true parameter
    np.random.seed(42)
    theta_star = np.random.randn(d)
    theta_star = theta_star / np.linalg.norm(theta_star)  # Normalize
    
    # Generate random arm features
    arm_features = []
    for i in range(n_arms):
        features = np.random.randn(d)
        features = features / np.linalg.norm(features)  # Normalize
        arm_features.append(features)
    
    # Compare different LinUCB variants
    results = compare_linucb_variants(theta_star, arm_features, n_steps=1000, n_runs=50)
    
    # Plot results
    plot_linucb_comparison(results, 1000)
    
    # Analyze LinUCB behavior
    analysis = analyze_linucb_behavior(theta_star, arm_features)
    
    print("\nLinUCB Analysis:")
    print(f"Optimal arm: {analysis['optimal_arm']}")
    print(f"Arm counts: {analysis['arm_counts']}")
    print(f"Parameter estimation error: {np.linalg.norm(analysis['true_parameter'] - analysis['final_parameter_estimate']):.3f}")
    
    # Plot detailed analysis
    plot_linucb_analysis(analysis, 1000) 
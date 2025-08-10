"""
Linear Thompson Sampling Algorithm Implementation

This module implements Linear Thompson Sampling for linear bandits.
The algorithm extends Thompson sampling to handle linear reward functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from scipy.linalg import solve
from scipy.stats import multivariate_normal


class LinearThompsonSampling:
    """
    Linear Thompson Sampling algorithm for linear bandits.
    
    The algorithm maintains a Gaussian posterior over the parameter vector
    and samples from this posterior to select actions.
    
    Attributes:
        d (int): Feature dimension
        nu (float): Noise parameter
        lambda_reg (float): Regularization parameter
        A (np.ndarray): Design matrix
        b (np.ndarray): Cumulative rewards
        mu (np.ndarray): Posterior mean
        Sigma (np.ndarray): Posterior covariance
    """
    
    def __init__(self, d: int, nu: float = 1.0, lambda_reg: float = 1.0):
        """
        Initialize Linear Thompson Sampling algorithm.
        
        Args:
            d (int): Feature dimension
            nu (float): Noise parameter
            lambda_reg (float): Regularization parameter
        """
        self.d = d
        self.nu = nu
        self.lambda_reg = lambda_reg
        
        # Initialize posterior parameters
        self.A = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.mu = np.zeros(d)
        self.Sigma = lambda_reg * np.eye(d)
        
        # History for analysis
        self.rewards_history = []
        self.actions_history = []
        self.samples_history = []
        
    def select_arm(self, arms: List[np.ndarray]) -> int:
        """
        Select arm using Linear Thompson Sampling.
        
        Args:
            arms (List[np.ndarray]): List of feature vectors for each arm
            
        Returns:
            int: Index of selected arm
        """
        # Sample from posterior
        theta_sample = np.random.multivariate_normal(self.mu, self.Sigma)
        
        # Store sample for analysis
        self.samples_history.append(theta_sample)
        
        # Calculate expected rewards for all arms
        expected_rewards = []
        for x in arms:
            reward = np.dot(theta_sample, x)
            expected_rewards.append(reward)
        
        return np.argmax(expected_rewards)
    
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
        
        # Update posterior parameters
        self.Sigma = solve(self.A, np.eye(self.d))
        self.mu = solve(self.A, self.b)
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm_idx)
    
    def get_posterior_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current posterior parameters.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and covariance
        """
        return self.mu.copy(), self.Sigma.copy()
    
    def get_credible_intervals(self, confidence: float = 0.95) -> List[Tuple[float, float]]:
        """
        Get credible intervals for each parameter.
        
        Args:
            confidence (float): Confidence level
            
        Returns:
            List[Tuple[float, float]]: Credible intervals for each parameter
        """
        alpha_level = (1 - confidence) / 2
        
        intervals = []
        for i in range(self.d):
            mean = self.mu[i]
            std = np.sqrt(self.Sigma[i, i])
            
            lower = mean + norm.ppf(alpha_level) * std
            upper = mean + norm.ppf(1 - alpha_level) * std
            intervals.append((lower, upper))
        
        return intervals


def run_linear_thompson_experiment(theta_star: np.ndarray, 
                                 arm_features: List[np.ndarray],
                                 n_steps: int = 1000,
                                 nu: float = 1.0,
                                 n_runs: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Linear Thompson Sampling experiment.
    
    Args:
        theta_star (np.ndarray): True parameter vector
        arm_features (List[np.ndarray]): Feature vectors for each arm
        n_steps (int): Number of time steps
        nu (float): Noise parameter
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
        bandit = LinearThompsonSampling(d, nu)
        
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


def compare_linear_thompson_variants(theta_star: np.ndarray, 
                                   arm_features: List[np.ndarray],
                                   n_steps: int = 1000,
                                   n_runs: int = 50) -> dict:
    """
    Compare different Linear Thompson Sampling variants.
    
    Args:
        theta_star (np.ndarray): True parameter vector
        arm_features (List[np.ndarray]): Feature vectors for each arm
        n_steps (int): Number of time steps
        n_runs (int): Number of independent runs
        
    Returns:
        dict: Results for each Linear Thompson Sampling variant
    """
    results = {}
    
    # Test different nu values
    nu_values = [0.5, 1.0, 2.0]
    
    for nu in nu_values:
        print(f"Running Linear Thompson Sampling with nu = {nu}")
        rewards, regrets = run_linear_thompson_experiment(
            theta_star, arm_features, n_steps, nu, n_runs
        )
        
        results[f'Linear TS(ν={nu})'] = {
            'rewards': rewards,
            'regrets': regrets,
            'cumulative_regret': np.sum(regrets),
            'final_regret': np.mean(regrets[-100:])
        }
    
    return results


def plot_linear_thompson_comparison(results: dict, n_steps: int):
    """
    Plot comparison of different Linear Thompson Sampling variants.
    
    Args:
        results (dict): Results from compare_linear_thompson_variants
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
    ax1.set_title('Linear Thompson Sampling - Cumulative Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative regrets
    for variant, data in results.items():
        cumulative_regrets = np.cumsum(data['regrets'])
        ax2.plot(range(n_steps), cumulative_regrets, 
                label=variant, linewidth=2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Linear Thompson Sampling - Cumulative Regret')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_linear_thompson_behavior(theta_star: np.ndarray, 
                                   arm_features: List[np.ndarray],
                                   n_steps: int = 1000,
                                   nu: float = 1.0) -> dict:
    """
    Analyze Linear Thompson Sampling behavior in detail.
    
    Args:
        theta_star (np.ndarray): True parameter vector
        arm_features (List[np.ndarray]): Feature vectors for each arm
        n_steps (int): Number of time steps
        nu (float): Noise parameter
        
    Returns:
        dict: Detailed analysis results
    """
    d = len(theta_star)
    n_arms = len(arm_features)
    
    # Run single experiment for detailed analysis
    bandit = LinearThompsonSampling(d, nu)
    
    arm_counts = np.zeros(n_arms)
    arm_rewards = [[] for _ in range(n_arms)]
    posterior_means = []
    posterior_covariances = []
    
    for step in range(n_steps):
        arm_idx = bandit.select_arm(arm_features)
        arm_counts[arm_idx] += 1
        
        expected_reward = np.dot(theta_star, arm_features[arm_idx])
        noise = np.random.normal(0, 0.1)
        reward = expected_reward + noise
        arm_rewards[arm_idx].append(reward)
        
        bandit.update(arm_idx, reward, arm_features)
        
        # Store posterior parameters
        mu, Sigma = bandit.get_posterior_parameters()
        posterior_means.append(mu)
        posterior_covariances.append(Sigma)
    
    # Calculate statistics
    arm_avg_rewards = [np.mean(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    arm_std_rewards = [np.std(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    
    # Analyze posterior evolution
    posterior_means_array = np.array(posterior_means)
    posterior_covariances_array = np.array(posterior_covariances)
    
    return {
        'arm_counts': arm_counts,
        'arm_avg_rewards': arm_avg_rewards,
        'arm_std_rewards': arm_std_rewards,
        'optimal_arm': np.argmax([np.dot(theta_star, x) for x in arm_features]),
        'final_posterior_mean': bandit.get_posterior_parameters()[0],
        'true_parameter': theta_star,
        'posterior_means_over_time': posterior_means_array,
        'posterior_covariances_over_time': posterior_covariances_array,
        'samples_history': bandit.samples_history,
        'credible_intervals': bandit.get_credible_intervals()
    }


def plot_linear_thompson_analysis(analysis: dict, n_steps: int):
    """
    Plot detailed Linear Thompson Sampling analysis.
    
    Args:
        analysis (dict): Results from analyze_linear_thompson_behavior
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
    final_estimate = analysis['final_posterior_mean']
    param_error = np.linalg.norm(true_param - final_estimate)
    
    ax2.plot(true_param, 'o-', label='True Parameter', linewidth=2)
    ax2.plot(final_estimate, 's-', label='Posterior Mean', linewidth=2)
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title(f'Parameter Estimation (Error: {param_error:.3f})')
    ax2.legend()
    
    # Posterior means over time
    if len(analysis['posterior_means_over_time']) > 0:
        posterior_means = analysis['posterior_means_over_time']
        for i in range(min(3, len(true_param))):
            ax3.plot(posterior_means[:, i], label=f'Parameter {i}')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Posterior Mean')
        ax3.set_title('Posterior Means Over Time')
        ax3.legend()
    
    # Sample distribution (for last few time steps)
    if len(analysis['samples_history']) > 0:
        recent_samples = analysis['samples_history'][-100:]  # Last 100 samples
        samples_array = np.array(recent_samples)
        
        for i in range(min(3, len(true_param))):
            ax4.hist(samples_array[:, i], alpha=0.7, label=f'Parameter {i}', bins=20)
        ax4.set_xlabel('Sampled Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Recent Sample Distribution')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running Linear Thompson Sampling Example")
    
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
    
    # Compare different Linear Thompson Sampling variants
    results = compare_linear_thompson_variants(theta_star, arm_features, n_steps=1000, n_runs=50)
    
    # Plot results
    plot_linear_thompson_comparison(results, 1000)
    
    # Analyze Linear Thompson Sampling behavior
    analysis = analyze_linear_thompson_behavior(theta_star, arm_features)
    
    print("\nLinear Thompson Sampling Analysis:")
    print(f"Optimal arm: {analysis['optimal_arm']}")
    print(f"Arm counts: {analysis['arm_counts']}")
    print(f"Parameter estimation error: {np.linalg.norm(analysis['true_parameter'] - analysis['final_posterior_mean']):.3f}")
    
    # Plot detailed analysis
    plot_linear_thompson_analysis(analysis, 1000) 
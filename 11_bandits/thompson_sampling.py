"""
Thompson Sampling Algorithm Implementation

This module implements Thompson sampling for multi-armed bandits.
Thompson sampling is a Bayesian approach that maintains posterior distributions
over arm rewards and samples from these posteriors to select actions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from scipy.stats import beta, norm


class ThompsonSampling:
    """
    Thompson Sampling algorithm for multi-armed bandits.
    
    This implementation assumes Bernoulli rewards (0 or 1).
    For other reward distributions, different conjugate priors are needed.
    
    Attributes:
        n_arms (int): Number of arms
        alpha (np.ndarray): Beta distribution parameters (successes + 1)
        beta (np.ndarray): Beta distribution parameters (failures + 1)
        total_pulls (int): Total number of pulls
    """
    
    def __init__(self, n_arms: int):
        """
        Initialize Thompson sampling algorithm.
        
        Args:
            n_arms (int): Number of arms
        """
        self.n_arms = n_arms
        
        # Initialize Beta distribution parameters
        # Prior: Beta(1, 1) = Uniform(0, 1)
        self.alpha = np.ones(n_arms)  # successes + 1
        self.beta = np.ones(n_arms)   # failures + 1
        
        # History for analysis
        self.rewards_history = []
        self.actions_history = []
        self.samples_history = []
        
    def select_arm(self) -> int:
        """
        Select an arm using Thompson sampling.
        
        Returns:
            int: Index of selected arm
        """
        # Sample from posterior distributions
        samples = []
        for arm in range(self.n_arms):
            sample = np.random.beta(self.alpha[arm], self.beta[arm])
            samples.append(sample)
        
        # Store samples for analysis
        self.samples_history.append(samples)
        
        # Choose arm with highest sampled value
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        """
        Update algorithm with observed reward.
        
        Args:
            arm (int): Index of pulled arm
            reward (float): Observed reward (should be 0 or 1 for Bernoulli)
        """
        # Update posterior parameters
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm)
    
    def get_posterior_means(self) -> np.ndarray:
        """
        Get posterior means for all arms.
        
        Returns:
            np.ndarray: Posterior means
        """
        return self.alpha / (self.alpha + self.beta)
    
    def get_posterior_vars(self) -> np.ndarray:
        """
        Get posterior variances for all arms.
        
        Returns:
            np.ndarray: Posterior variances
        """
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total ** 2 * (total + 1))
    
    def get_best_arm(self) -> int:
        """Get the arm with highest posterior mean."""
        return np.argmax(self.get_posterior_means())
    
    def get_credible_intervals(self, confidence: float = 0.95) -> List[Tuple[float, float]]:
        """
        Get credible intervals for all arms.
        
        Args:
            confidence (float): Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            List[Tuple[float, float]]: (lower, upper) credible intervals
        """
        intervals = []
        alpha_level = (1 - confidence) / 2
        
        for arm in range(self.n_arms):
            lower = beta.ppf(alpha_level, self.alpha[arm], self.beta[arm])
            upper = beta.ppf(1 - alpha_level, self.alpha[arm], self.beta[arm])
            intervals.append((lower, upper))
        
        return intervals


class GaussianThompsonSampling:
    """
    Thompson Sampling for Gaussian rewards.
    
    Assumes rewards are normally distributed with unknown mean and known variance.
    Uses Gaussian-Gaussian conjugate prior.
    """
    
    def __init__(self, n_arms: int, prior_mean: float = 0.0, 
                 prior_var: float = 1.0, reward_var: float = 1.0):
        """
        Initialize Gaussian Thompson sampling.
        
        Args:
            n_arms (int): Number of arms
            prior_mean (float): Prior mean for all arms
            prior_var (float): Prior variance
            reward_var (float): Known reward variance
        """
        self.n_arms = n_arms
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.reward_var = reward_var
        
        # Initialize posterior parameters
        self.posterior_means = np.full(n_arms, prior_mean)
        self.posterior_vars = np.full(n_arms, prior_var)
        self.pulls = np.zeros(n_arms, dtype=int)
        
        # History for analysis
        self.rewards_history = []
        self.actions_history = []
        self.samples_history = []
    
    def select_arm(self) -> int:
        """
        Select an arm using Gaussian Thompson sampling.
        
        Returns:
            int: Index of selected arm
        """
        # Sample from posterior distributions
        samples = []
        for arm in range(self.n_arms):
            sample = np.random.normal(self.posterior_means[arm], 
                                    np.sqrt(self.posterior_vars[arm]))
            samples.append(sample)
        
        # Store samples for analysis
        self.samples_history.append(samples)
        
        # Choose arm with highest sampled value
        return np.argmax(samples)
    
    def update(self, arm: int, reward: float):
        """
        Update algorithm with observed reward.
        
        Args:
            arm (int): Index of pulled arm
            reward (float): Observed reward
        """
        # Update posterior parameters
        self.pulls[arm] += 1
        n = self.pulls[arm]
        
        # Posterior update for Gaussian-Gaussian
        # New posterior mean = (prior_mean/prior_var + sum_rewards/reward_var) / (1/prior_var + n/reward_var)
        # New posterior variance = 1 / (1/prior_var + n/reward_var)
        
        # For simplicity, we'll use a simpler update rule
        old_mean = self.posterior_means[arm]
        old_var = self.posterior_vars[arm]
        
        # Update posterior mean
        self.posterior_means[arm] = (old_mean * (n - 1) + reward) / n
        
        # Update posterior variance (simplified)
        self.posterior_vars[arm] = self.reward_var / n
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm)
    
    def get_best_arm(self) -> int:
        """Get the arm with highest posterior mean."""
        return np.argmax(self.posterior_means)


def run_thompson_experiment(arm_means: List[float], 
                           n_steps: int = 1000,
                           n_runs: int = 100,
                           reward_type: str = 'bernoulli') -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Thompson sampling experiment.
    
    Args:
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
        n_runs (int): Number of independent runs
        reward_type (str): 'bernoulli' or 'gaussian'
        
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
        if reward_type == 'bernoulli':
            bandit = ThompsonSampling(n_arms)
        else:
            bandit = GaussianThompsonSampling(n_arms)
        
        run_rewards = []
        run_regrets = []
        
        for step in range(n_steps):
            # Select arm
            arm = bandit.select_arm()
            
            # Get reward
            if reward_type == 'bernoulli':
                # Bernoulli reward
                reward = np.random.binomial(1, arm_means[arm])
            else:
                # Gaussian reward
                reward = np.random.normal(arm_means[arm], 0.1)
                reward = np.clip(reward, 0, 1)
            
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


def compare_thompson_variants(arm_means: List[float], 
                             n_steps: int = 1000,
                             n_runs: int = 50) -> dict:
    """
    Compare different Thompson sampling variants.
    
    Args:
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
        n_runs (int): Number of independent runs
        
    Returns:
        dict: Results for each Thompson sampling variant
    """
    results = {}
    
    # Test Bernoulli Thompson sampling
    print("Running Bernoulli Thompson Sampling")
    rewards, regrets = run_thompson_experiment(
        arm_means, n_steps, n_runs, 'bernoulli'
    )
    
    results['Bernoulli TS'] = {
        'rewards': rewards,
        'regrets': regrets,
        'cumulative_regret': np.sum(regrets),
        'final_regret': np.mean(regrets[-100:])
    }
    
    # Test Gaussian Thompson sampling
    print("Running Gaussian Thompson Sampling")
    rewards, regrets = run_thompson_experiment(
        arm_means, n_steps, n_runs, 'gaussian'
    )
    
    results['Gaussian TS'] = {
        'rewards': rewards,
        'regrets': regrets,
        'cumulative_regret': np.sum(regrets),
        'final_regret': np.mean(regrets[-100:])
    }
    
    return results


def plot_thompson_comparison(results: dict, n_steps: int):
    """
    Plot comparison of different Thompson sampling variants.
    
    Args:
        results (dict): Results from compare_thompson_variants
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
    ax1.set_title('Thompson Sampling Variants - Cumulative Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative regrets
    for variant, data in results.items():
        cumulative_regrets = np.cumsum(data['regrets'])
        ax2.plot(range(n_steps), cumulative_regrets, 
                label=variant, linewidth=2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Thompson Sampling Variants - Cumulative Regret')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_thompson_behavior(arm_means: List[float], 
                            n_steps: int = 1000,
                            reward_type: str = 'bernoulli') -> dict:
    """
    Analyze Thompson sampling behavior in detail.
    
    Args:
        arm_means (List[float]): True means of arms
        n_steps (int): Number of time steps
        reward_type (str): 'bernoulli' or 'gaussian'
        
    Returns:
        dict: Detailed analysis results
    """
    n_arms = len(arm_means)
    optimal_arm = np.argmax(arm_means)
    
    # Run single experiment for detailed analysis
    if reward_type == 'bernoulli':
        bandit = ThompsonSampling(n_arms)
    else:
        bandit = GaussianThompsonSampling(n_arms)
    
    arm_counts = np.zeros(n_arms)
    arm_rewards = [[] for _ in range(n_arms)]
    posterior_means_over_time = []
    
    for step in range(n_steps):
        arm = bandit.select_arm()
        arm_counts[arm] += 1
        
        # Get reward
        if reward_type == 'bernoulli':
            reward = np.random.binomial(1, arm_means[arm])
        else:
            reward = np.random.normal(arm_means[arm], 0.1)
            reward = np.clip(reward, 0, 1)
        
        arm_rewards[arm].append(reward)
        bandit.update(arm, reward)
        
        # Store posterior means
        if reward_type == 'bernoulli':
            posterior_means_over_time.append(bandit.get_posterior_means())
        else:
            posterior_means_over_time.append(bandit.posterior_means)
    
    # Calculate statistics
    arm_avg_rewards = [np.mean(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    arm_std_rewards = [np.std(rewards) if rewards else 0 
                       for rewards in arm_rewards]
    
    # Analyze posterior evolution
    posterior_means_array = np.array(posterior_means_over_time)
    
    return {
        'arm_counts': arm_counts,
        'arm_avg_rewards': arm_avg_rewards,
        'arm_std_rewards': arm_std_rewards,
        'optimal_arm': optimal_arm,
        'posterior_means': bandit.get_posterior_means() if reward_type == 'bernoulli' else bandit.posterior_means,
        'total_pulls': len(bandit.rewards_history),
        'posterior_means_over_time': posterior_means_array,
        'samples_history': bandit.samples_history,
        'credible_intervals': bandit.get_credible_intervals() if reward_type == 'bernoulli' else None
    }


def plot_thompson_analysis(analysis: dict, n_steps: int):
    """
    Plot detailed Thompson sampling analysis.
    
    Args:
        analysis (dict): Results from analyze_thompson_behavior
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
    
    # Posterior means vs true means
    true_means = [0.1, 0.2, 0.3, 0.4, 0.5]  # Assuming these are the true means
    posterior_means = analysis['posterior_means']
    ax2.bar([f'Arm {i}' for i in arms], posterior_means, alpha=0.7, label='Posterior')
    ax2.bar([f'Arm {i}' for i in arms], true_means, alpha=0.3, label='True')
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('Posterior vs True Means')
    ax2.legend()
    
    # Posterior means over time (for first few arms)
    if len(analysis['posterior_means_over_time']) > 0:
        posterior_means_over_time = analysis['posterior_means_over_time']
        for arm in range(min(3, len(arms))):
            ax3.plot(posterior_means_over_time[:, arm], label=f'Arm {arm}')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Posterior Mean')
        ax3.set_title('Posterior Means Over Time')
        ax3.legend()
    
    # Sample distribution (for last few time steps)
    if len(analysis['samples_history']) > 0:
        recent_samples = analysis['samples_history'][-100:]  # Last 100 samples
        samples_array = np.array(recent_samples)
        
        for arm in range(min(3, len(arms))):
            ax4.hist(samples_array[:, arm], alpha=0.7, label=f'Arm {arm}', bins=20)
        ax4.set_xlabel('Sampled Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Recent Sample Distribution')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running Thompson Sampling Example")
    
    # Define arm means
    arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Compare different Thompson sampling variants
    results = compare_thompson_variants(arm_means, n_steps=1000, n_runs=50)
    
    # Plot results
    plot_thompson_comparison(results, 1000)
    
    # Analyze Thompson sampling behavior
    analysis = analyze_thompson_behavior(arm_means, reward_type='bernoulli')
    
    print("\nThompson Sampling Analysis:")
    print(f"Optimal arm: {analysis['optimal_arm']}")
    print(f"Arm counts: {analysis['arm_counts']}")
    print(f"Posterior means: {analysis['posterior_means']}")
    print(f"Total pulls: {analysis['total_pulls']}")
    
    # Plot detailed analysis
    plot_thompson_analysis(analysis, 1000) 
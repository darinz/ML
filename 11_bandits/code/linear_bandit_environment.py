import numpy as np
import matplotlib.pyplot as plt

class LinearBanditEnvironment:
    """
    Linear bandit environment for simulating linear bandit problems.
    """
    
    def __init__(self, theta_star, arms, noise_std=0.1):
        """
        Initialize linear bandit environment.
        
        Args:
            theta_star: True parameter vector
            arms: List of feature vectors for each arm
            noise_std: Standard deviation of noise
        """
        self.theta_star = theta_star
        self.arms = arms
        self.noise_std = noise_std
        self.d = len(theta_star)
        
    def pull_arm(self, arm_idx):
        """
        Pull arm and return reward.
        
        Args:
            arm_idx: Index of arm to pull
            
        Returns:
            float: Observed reward
        """
        x = self.arms[arm_idx]
        expected_reward = np.dot(self.theta_star, x)
        noise = np.random.normal(0, self.noise_std)
        return expected_reward + noise
    
    def get_optimal_reward(self):
        """
        Get optimal expected reward.
        
        Returns:
            float: Optimal expected reward
        """
        rewards = [np.dot(self.theta_star, x) for x in self.arms]
        return max(rewards)
    
    def get_regret(self, chosen_arms, rewards):
        """
        Calculate cumulative regret.
        
        Args:
            chosen_arms: List of chosen arm indices
            rewards: List of observed rewards
            
        Returns:
            np.ndarray: Cumulative regret
        """
        optimal_reward = self.get_optimal_reward()
        cumulative_optimal = np.cumsum([optimal_reward] * len(rewards))
        cumulative_rewards = np.cumsum(rewards)
        return cumulative_optimal - cumulative_rewards

def run_linear_bandit_experiment(env, algorithm, T=1000):
    """
    Run linear bandit experiment.
    
    Args:
        env: LinearBanditEnvironment instance
        algorithm: Bandit algorithm instance
        T: Number of time steps
        
    Returns:
        tuple: (chosen_arms, rewards)
    """
    chosen_arms = []
    rewards = []
    
    for t in range(T):
        # Select arm
        arm_idx = algorithm.select_arm(env.arms)
        
        # Pull arm and observe reward
        reward = env.pull_arm(arm_idx)
        
        # Update algorithm
        algorithm.update(arm_idx, reward, env.arms)
        
        chosen_arms.append(arm_idx)
        rewards.append(reward)
    
    return chosen_arms, rewards

def compare_linear_algorithms(env, algorithms, T=1000, n_runs=50):
    """
    Compare different linear bandit algorithms.
    
    Args:
        env: LinearBanditEnvironment instance
        algorithms: Dictionary of algorithm classes
        T: Number of time steps
        n_runs: Number of independent runs
        
    Returns:
        dict: Results for each algorithm
    """
    results = {}
    
    for name, algorithm_class in algorithms.items():
        regrets = []
        for run in range(n_runs):
            # Create fresh algorithm instance
            algorithm = algorithm_class(env.d)
            
            # Run experiment
            chosen_arms, rewards = run_linear_bandit_experiment(env, algorithm, T)
            
            # Calculate regret
            regret = env.get_regret(chosen_arms, rewards)
            regrets.append(regret)
        
        results[name] = np.mean(regrets, axis=0)
    
    return results

def plot_linear_bandit_results(results, T):
    """
    Plot regret comparison for linear bandit algorithms.
    
    Args:
        results: Dictionary of regret results
        T: Number of time steps
    """
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative regret
    plt.subplot(2, 1, 1)
    for name, regret in results.items():
        plt.plot(range(1, T+1), regret, label=name, linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.title('Linear Bandit Algorithm Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot regret rate (regret / sqrt(t))
    plt.subplot(2, 1, 2)
    for name, regret in results.items():
        regret_rate = regret / np.sqrt(range(1, T+1))
        plt.plot(range(1, T+1), regret_rate, label=name, linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Regret Rate (R(t) / √t)')
    plt.title('Regret Rate Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

class ContextualBanditEnvironment:
    """
    Contextual bandit environment for simulating contextual bandit problems.
    """
    
    def __init__(self, theta_star, context_generator, noise_std=0.1):
        """
        Initialize contextual bandit environment.
        
        Args:
            theta_star: True parameter vector
            context_generator: Function that generates contexts
            noise_std: Standard deviation of noise
        """
        self.theta_star = theta_star
        self.context_generator = context_generator
        self.noise_std = noise_std
        self.d = len(theta_star)
        
    def generate_context(self):
        """
        Generate context for current time step.
        
        Returns:
            np.ndarray: Context vector
        """
        return self.context_generator()
    
    def generate_arm_features(self, context, n_arms):
        """
        Generate arm features based on context.
        
        Args:
            context: Current context vector
            n_arms: Number of arms
            
        Returns:
            list: List of arm feature vectors
        """
        arm_features = []
        for i in range(n_arms):
            # Combine context with arm-specific features
            arm_specific = np.random.randn(self.d - len(context))
            combined = np.concatenate([context, arm_specific])
            arm_features.append(combined)
        return arm_features
    
    def pull_arm(self, arm_idx, context_features):
        """
        Pull arm and return reward.
        
        Args:
            arm_idx: Index of arm to pull
            context_features: List of arm feature vectors
            
        Returns:
            float: Observed reward
        """
        x = context_features[arm_idx]
        expected_reward = np.dot(self.theta_star, x)
        noise = np.random.normal(0, self.noise_std)
        return expected_reward + noise
    
    def get_optimal_reward(self, context_features):
        """
        Get optimal expected reward for current context.
        
        Args:
            context_features: List of arm feature vectors
            
        Returns:
            float: Optimal expected reward
        """
        rewards = [np.dot(self.theta_star, x) for x in context_features]
        return max(rewards)
    
    def get_regret(self, chosen_arms, rewards, optimal_rewards):
        """
        Calculate cumulative regret.
        
        Args:
            chosen_arms: List of chosen arm indices
            rewards: List of observed rewards
            optimal_rewards: List of optimal rewards for each time step
            
        Returns:
            np.ndarray: Cumulative regret
        """
        cumulative_optimal = np.cumsum(optimal_rewards)
        cumulative_rewards = np.cumsum(rewards)
        return cumulative_optimal - cumulative_rewards

def run_contextual_bandit_experiment(env, algorithm, T=1000, n_arms=10):
    """
    Run contextual bandit experiment.
    
    Args:
        env: ContextualBanditEnvironment instance
        algorithm: Bandit algorithm instance
        T: Number of time steps
        n_arms: Number of arms
        
    Returns:
        tuple: (chosen_arms, rewards, optimal_rewards)
    """
    chosen_arms = []
    rewards = []
    optimal_rewards = []
    
    for t in range(T):
        # Generate context
        context = env.generate_context()
        
        # Generate arm features for current context
        context_features = env.generate_arm_features(context, n_arms)
        
        # Select arm
        arm_idx = algorithm.select_arm(context_features)
        
        # Pull arm and observe reward
        reward = env.pull_arm(arm_idx, context_features)
        
        # Get optimal reward for comparison
        optimal_reward = env.get_optimal_reward(context_features)
        
        # Update algorithm
        algorithm.update(arm_idx, reward, context_features)
        
        chosen_arms.append(arm_idx)
        rewards.append(reward)
        optimal_rewards.append(optimal_reward)
    
    return chosen_arms, rewards, optimal_rewards

def compare_contextual_algorithms(env, algorithms, T=1000, n_arms=10, n_runs=50):
    """
    Compare different contextual bandit algorithms.
    
    Args:
        env: ContextualBanditEnvironment instance
        algorithms: Dictionary of algorithm classes
        T: Number of time steps
        n_arms: Number of arms
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
            chosen_arms, rewards, optimal_rewards = run_contextual_bandit_experiment(
                env, algorithm, T, n_arms
            )
            
            # Calculate regret
            regret = env.get_regret(chosen_arms, rewards, optimal_rewards)
            regrets.append(regret)
        
        results[name] = np.mean(regrets, axis=0)
    
    return results

def plot_contextual_bandit_results(results, T):
    """
    Plot regret comparison for contextual bandit algorithms.
    
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
    plt.title('Contextual Bandit Algorithm Comparison')
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

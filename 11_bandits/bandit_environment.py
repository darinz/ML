import numpy as np
import matplotlib.pyplot as plt

class BernoulliArm:
    """Bernoulli arm with success probability p."""
    def __init__(self, p):
        self.p = p
    
    def pull(self):
        return np.random.binomial(1, self.p)

class BanditEnvironment:
    """Multi-armed bandit environment."""
    def __init__(self, arm_means):
        self.arms = [BernoulliArm(p) for p in arm_means]
        self.optimal_arm = np.argmax(arm_means)
        self.optimal_mean = arm_means[self.optimal_arm]
    
    def get_regret(self, chosen_arms, rewards):
        """Calculate cumulative regret."""
        cumulative_optimal = np.cumsum([self.optimal_mean] * len(rewards))
        cumulative_rewards = np.cumsum(rewards)
        return cumulative_optimal - cumulative_rewards

def compare_algorithms(env, algorithms, T=1000, n_runs=100):
    """
    Compare different bandit algorithms.
    
    Args:
        env: BanditEnvironment instance
        algorithms: Dictionary of algorithm functions
        T: Number of time steps
        n_runs: Number of independent runs
    
    Returns:
        results: Dictionary with average regrets for each algorithm
    """
    results = {}
    
    for name, algorithm in algorithms.items():
        regrets = []
        for run in range(n_runs):
            chosen_arms, rewards = algorithm(env, T)
            regret = env.get_regret(chosen_arms, rewards)
            regrets.append(regret)
        
        results[name] = np.mean(regrets, axis=0)
    
    return results

def plot_regret_comparison(results, T):
    """Plot regret comparison of different algorithms."""
    plt.figure(figsize=(10, 6))
    
    for name, regret in results.items():
        plt.plot(range(1, T+1), regret, label=name)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.title('Regret Comparison of Bandit Algorithms')
    plt.legend()
    plt.grid(True)
    plt.show()

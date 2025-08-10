import numpy as np
import matplotlib.pyplot as plt

class BAIEnvironment:
    """
    Best Arm Identification environment for simulating BAI problems.
    """
    
    def __init__(self, arm_means, noise_std=0.1):
        """
        Initialize BAI environment.
        
        Args:
            arm_means: True mean rewards for each arm
            noise_std: Standard deviation of noise
        """
        self.arm_means = arm_means
        self.noise_std = noise_std
        self.n_arms = len(arm_means)
        self.optimal_arm = np.argmax(arm_means)
        self.optimal_mean = arm_means[self.optimal_arm]
        
    def pull_arm(self, arm_idx):
        """
        Pull arm and return reward.
        
        Args:
            arm_idx: Index of arm to pull
            
        Returns:
            float: Observed reward
        """
        expected_reward = self.arm_means[arm_idx]
        noise = np.random.normal(0, self.noise_std)
        return expected_reward + noise
    
    def get_gaps(self):
        """
        Calculate gaps between optimal arm and others.
        
        Returns:
            list: List of gaps
        """
        gaps = []
        for i in range(self.n_arms):
            if i != self.optimal_arm:
                gap = self.optimal_mean - self.arm_means[i]
                gaps.append(gap)
        return gaps

def run_bai_experiment(env, algorithm, max_pulls=1000):
    """
    Run BAI experiment.
    
    Args:
        env: BAIEnvironment instance
        algorithm: BAI algorithm instance
        max_pulls: Maximum number of pulls
        
    Returns:
        tuple: (success, pulls_used, identified_arm)
    """
    pulls_used = 0
    success = False
    
    while pulls_used < max_pulls and not algorithm.is_complete():
        # Select arm
        arm_idx = algorithm.select_arm()
        
        # Pull arm and observe reward
        reward = env.pull_arm(arm_idx)
        
        # Update algorithm
        algorithm.update(arm_idx, reward)
        pulls_used += 1
        
        # Check if correct arm is identified
        if algorithm.is_complete():
            identified_arm = algorithm.get_best_arm()
            success = (identified_arm == env.optimal_arm)
            break
    
    return success, pulls_used, algorithm.get_best_arm()

def compare_bai_algorithms(env, algorithms, n_runs=100, max_pulls=1000):
    """
    Compare different BAI algorithms.
    
    Args:
        env: BAIEnvironment instance
        algorithms: Dictionary of algorithm classes
        n_runs: Number of independent runs
        max_pulls: Maximum number of pulls per run
        
    Returns:
        dict: Results for each algorithm
    """
    results = {}
    
    for name, algorithm_class in algorithms.items():
        success_rates = []
        sample_complexities = []
        
        for run in range(n_runs):
            # Create fresh algorithm instance
            algorithm = algorithm_class(env.n_arms)
            
            # Run experiment
            success, pulls, identified_arm = run_bai_experiment(env, algorithm, max_pulls)
            
            success_rates.append(success)
            sample_complexities.append(pulls)
        
        results[name] = {
            'success_rate': np.mean(success_rates),
            'avg_sample_complexity': np.mean(sample_complexities),
            'std_sample_complexity': np.std(sample_complexities)
        }
    
    return results

def plot_bai_results(results):
    """
    Plot BAI algorithm comparison.
    
    Args:
        results: Dictionary of results from compare_bai_algorithms
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rate comparison
    names = list(results.keys())
    success_rates = [results[name]['success_rate'] for name in names]
    
    ax1.bar(names, success_rates)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('BAI Algorithm Success Rate Comparison')
    ax1.set_ylim(0, 1)
    
    # Sample complexity comparison
    sample_complexities = [results[name]['avg_sample_complexity'] for name in names]
    std_complexities = [results[name]['std_sample_complexity'] for name in names]
    
    ax2.bar(names, sample_complexities, yerr=std_complexities, capsize=5)
    ax2.set_ylabel('Average Sample Complexity')
    ax2.set_title('BAI Algorithm Sample Complexity Comparison')
    
    plt.tight_layout()
    plt.show()

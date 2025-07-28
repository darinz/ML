"""
Multi-Objective Bandit Algorithms

This module implements bandit algorithms for multi-objective problems,
where rewards are vectors and the goal is to find Pareto-optimal solutions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform


class MultiObjectiveBandit:
    """
    Base class for multi-objective bandit algorithms.
    
    This class provides common functionality for algorithms
    that handle vector-valued rewards.
    """
    
    def __init__(self, n_arms: int, n_objectives: int):
        """
        Initialize multi-objective bandit.
        
        Args:
            n_arms (int): Number of arms
            n_objectives (int): Number of objectives
        """
        self.n_arms = n_arms
        self.n_objectives = n_objectives
        
        # Initialize statistics
        self.empirical_means = np.zeros((n_arms, n_objectives))
        self.pulls = np.zeros(n_arms, dtype=int)
        
        # History for analysis
        self.rewards_history = []
        self.actions_history = []
    
    def update(self, arm: int, reward: np.ndarray):
        """
        Update algorithm with observed reward vector.
        
        Args:
            arm (int): Index of pulled arm
            reward (np.ndarray): Observed reward vector
        """
        # Update empirical mean
        self.pulls[arm] += 1
        self.empirical_means[arm] = (
            (self.empirical_means[arm] * (self.pulls[arm] - 1) + reward) 
            / self.pulls[arm]
        )
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm)
    
    def is_pareto_dominated(self, point: np.ndarray, points: List[np.ndarray]) -> bool:
        """
        Check if a point is Pareto dominated by any point in the list.
        
        Args:
            point (np.ndarray): Point to check
            points (List[np.ndarray]): List of points to compare against
            
        Returns:
            bool: True if point is dominated
        """
        for other_point in points:
            if np.all(other_point >= point) and np.any(other_point > point):
                return True
        return False
    
    def get_pareto_frontier(self, points: List[np.ndarray]) -> List[np.ndarray]:
        """
        Get Pareto frontier from a list of points.
        
        Args:
            points (List[np.ndarray]): List of points
            
        Returns:
            List[np.ndarray]: Pareto optimal points
        """
        pareto_points = []
        
        for point in points:
            if not self.is_pareto_dominated(point, points):
                pareto_points.append(point)
        
        return pareto_points


class ScalarizedMultiObjectiveBandit(MultiObjectiveBandit):
    """
    Scalarized multi-objective bandit.
    
    This algorithm uses scalarization functions to convert
    multi-objective rewards into scalar rewards.
    """
    
    def __init__(self, n_arms: int, n_objectives: int, 
                 scalarization_func: callable = None, weights: np.ndarray = None):
        """
        Initialize scalarized multi-objective bandit.
        
        Args:
            n_arms (int): Number of arms
            n_objectives (int): Number of objectives
            scalarization_func (callable): Function to scalarize rewards
            weights (np.ndarray): Weights for linear scalarization
        """
        super().__init__(n_arms, n_objectives)
        
        if scalarization_func is None:
            if weights is None:
                weights = np.ones(n_objectives) / n_objectives
            self.scalarization_func = lambda x: np.dot(x, weights)
        else:
            self.scalarization_func = scalarization_func
    
    def select_arm(self) -> int:
        """
        Select arm using scalarized rewards.
        
        Returns:
            int: Index of selected arm
        """
        # If any arm hasn't been pulled, pull it
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                return arm
        
        # Calculate scalarized values
        scalarized_values = []
        for arm in range(self.n_arms):
            scalarized_value = self.scalarization_func(self.empirical_means[arm])
            scalarized_values.append(scalarized_value)
        
        return np.argmax(scalarized_values)


class ParetoUCB(MultiObjectiveBandit):
    """
    Pareto UCB algorithm for multi-objective bandits.
    
    This algorithm extends UCB to handle multi-objective rewards
    by maintaining confidence intervals for each objective.
    """
    
    def __init__(self, n_arms: int, n_objectives: int, alpha: float = 2.0):
        """
        Initialize Pareto UCB.
        
        Args:
            n_arms (int): Number of arms
            n_objectives (int): Number of objectives
            alpha (float): Exploration parameter
        """
        super().__init__(n_arms, n_objectives)
        self.alpha = alpha
    
    def select_arm(self) -> int:
        """
        Select arm using Pareto UCB.
        
        Returns:
            int: Index of selected arm
        """
        # If any arm hasn't been pulled, pull it
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                return arm
        
        # Calculate UCB values for each objective
        ucb_values = []
        for arm in range(self.n_arms):
            arm_ucb = []
            for obj in range(self.n_objectives):
                exploitation = self.empirical_means[arm, obj]
                exploration = self.alpha * np.sqrt(np.log(np.sum(self.pulls)) / self.pulls[arm])
                ucb_value = exploitation + exploration
                arm_ucb.append(ucb_value)
            ucb_values.append(arm_ucb)
        
        # Find Pareto optimal arms
        pareto_arms = self._get_pareto_optimal_arms(ucb_values)
        
        # Select randomly from Pareto optimal arms
        return np.random.choice(pareto_arms)
    
    def _get_pareto_optimal_arms(self, ucb_values: List[np.ndarray]) -> List[int]:
        """
        Get Pareto optimal arms based on UCB values.
        
        Args:
            ucb_values (List[np.ndarray]): UCB values for each arm
            
        Returns:
            List[int]: Indices of Pareto optimal arms
        """
        pareto_arms = []
        
        for arm in range(self.n_arms):
            if not self.is_pareto_dominated(ucb_values[arm], ucb_values):
                pareto_arms.append(arm)
        
        return pareto_arms


class HypervolumeUCB(MultiObjectiveBandit):
    """
    Hypervolume UCB algorithm for multi-objective bandits.
    
    This algorithm uses hypervolume as a measure of quality
    for multi-objective solutions.
    """
    
    def __init__(self, n_arms: int, n_objectives: int, alpha: float = 2.0, 
                 reference_point: np.ndarray = None):
        """
        Initialize Hypervolume UCB.
        
        Args:
            n_arms (int): Number of arms
            n_objectives (int): Number of objectives
            alpha (float): Exploration parameter
            reference_point (np.ndarray): Reference point for hypervolume calculation
        """
        super().__init__(n_arms, n_objectives)
        self.alpha = alpha
        
        if reference_point is None:
            self.reference_point = np.ones(n_objectives)
        else:
            self.reference_point = reference_point
    
    def select_arm(self) -> int:
        """
        Select arm using Hypervolume UCB.
        
        Returns:
            int: Index of selected arm
        """
        # If any arm hasn't been pulled, pull it
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                return arm
        
        # Calculate UCB values for each objective
        ucb_values = []
        for arm in range(self.n_arms):
            arm_ucb = []
            for obj in range(self.n_objectives):
                exploitation = self.empirical_means[arm, obj]
                exploration = self.alpha * np.sqrt(np.log(np.sum(self.pulls)) / self.pulls[arm])
                ucb_value = exploitation + exploration
                arm_ucb.append(ucb_value)
            ucb_values.append(arm_ucb)
        
        # Calculate hypervolume contributions
        hypervolume_contributions = []
        for arm in range(self.n_arms):
            contribution = self._calculate_hypervolume_contribution(ucb_values, arm)
            hypervolume_contributions.append(contribution)
        
        return np.argmax(hypervolume_contributions)
    
    def _calculate_hypervolume_contribution(self, ucb_values: List[np.ndarray], 
                                          arm: int) -> float:
        """
        Calculate hypervolume contribution of an arm.
        
        Args:
            ucb_values (List[np.ndarray]): UCB values for each arm
            arm (int): Index of arm
            
        Returns:
            float: Hypervolume contribution
        """
        # Simplified hypervolume contribution calculation
        # In practice, this would use a proper hypervolume calculation library
        
        # For simplicity, use the minimum distance to reference point
        distance = np.linalg.norm(self.reference_point - ucb_values[arm])
        return -distance  # Negative because we want to maximize


class MultiObjectiveThompsonSampling(MultiObjectiveBandit):
    """
    Multi-objective Thompson sampling.
    
    This algorithm extends Thompson sampling to handle
    multi-objective rewards using multivariate normal distributions.
    """
    
    def __init__(self, n_arms: int, n_objectives: int, nu: float = 1.0):
        """
        Initialize multi-objective Thompson sampling.
        
        Args:
            n_arms (int): Number of arms
            n_objectives (int): Number of objectives
            nu (float): Noise parameter
        """
        super().__init__(n_arms, n_objectives)
        self.nu = nu
        
        # Initialize posterior parameters
        self.posterior_means = np.zeros((n_arms, n_objectives))
        self.posterior_covariances = [np.eye(n_objectives) for _ in range(n_arms)]
    
    def select_arm(self) -> int:
        """
        Select arm using multi-objective Thompson sampling.
        
        Returns:
            int: Index of selected arm
        """
        # Sample from posterior distributions
        samples = []
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                # If arm hasn't been pulled, use prior
                sample = np.random.multivariate_normal(
                    np.zeros(self.n_objectives), 
                    np.eye(self.n_objectives)
                )
            else:
                # Sample from posterior
                sample = np.random.multivariate_normal(
                    self.posterior_means[arm],
                    self.posterior_covariances[arm]
                )
            samples.append(sample)
        
        # Find Pareto optimal arms
        pareto_arms = self._get_pareto_optimal_arms(samples)
        
        # Select randomly from Pareto optimal arms
        return np.random.choice(pareto_arms)
    
    def update(self, arm: int, reward: np.ndarray):
        """
        Update algorithm with observed reward vector.
        
        Args:
            arm (int): Index of pulled arm
            reward (np.ndarray): Observed reward vector
        """
        super().update(arm, reward)
        
        # Update posterior parameters (simplified)
        if self.pulls[arm] > 1:
            # Update posterior mean
            self.posterior_means[arm] = self.empirical_means[arm]
            
            # Update posterior covariance (simplified)
            self.posterior_covariances[arm] = np.eye(self.n_objectives) / self.pulls[arm]


def run_multi_objective_experiment(algorithm: MultiObjectiveBandit, 
                                 arm_means: List[np.ndarray],
                                 n_steps: int = 1000) -> Dict:
    """
    Run multi-objective bandit experiment.
    
    Args:
        algorithm (MultiObjectiveBandit): Multi-objective bandit algorithm
        arm_means (List[np.ndarray]): True mean vectors for each arm
        n_steps (int): Number of time steps
        
    Returns:
        Dict: Experiment results
    """
    n_arms = len(arm_means)
    n_objectives = len(arm_means[0])
    
    rewards_history = []
    actions_history = []
    pareto_frontiers = []
    
    for step in range(n_steps):
        # Select arm
        arm = algorithm.select_arm()
        
        # Get reward vector
        reward = np.random.multivariate_normal(arm_means[arm], 0.1 * np.eye(n_objectives))
        reward = np.clip(reward, 0, 1)  # Clip to [0, 1]
        
        # Update algorithm
        algorithm.update(arm, reward)
        
        # Store results
        rewards_history.append(reward)
        actions_history.append(arm)
        
        # Calculate Pareto frontier
        pareto_frontier = algorithm.get_pareto_frontier(algorithm.empirical_means)
        pareto_frontiers.append(pareto_frontier)
    
    return {
        'rewards_history': rewards_history,
        'actions_history': actions_history,
        'pareto_frontiers': pareto_frontiers,
        'final_empirical_means': algorithm.empirical_means,
        'final_pareto_frontier': algorithm.get_pareto_frontier(algorithm.empirical_means)
    }


def compare_multi_objective_algorithms(n_arms: int, n_objectives: int, 
                                     n_steps: int = 1000) -> Dict:
    """
    Compare different multi-objective bandit algorithms.
    
    Args:
        n_arms (int): Number of arms
        n_objectives (int): Number of objectives
        n_steps (int): Number of time steps
        
    Returns:
        Dict: Comparison results
    """
    # Generate random arm means
    np.random.seed(42)
    arm_means = []
    for arm in range(n_arms):
        mean_vector = np.random.rand(n_objectives)
        arm_means.append(mean_vector)
    
    # Initialize algorithms
    algorithms = {
        'Scalarized (Equal Weights)': ScalarizedMultiObjectiveBandit(n_arms, n_objectives),
        'Pareto UCB': ParetoUCB(n_arms, n_objectives),
        'Hypervolume UCB': HypervolumeUCB(n_arms, n_objectives),
        'Multi-Objective Thompson Sampling': MultiObjectiveThompsonSampling(n_arms, n_objectives)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"Running {name}")
        result = run_multi_objective_experiment(algorithm, arm_means, n_steps)
        results[name] = result
    
    return results


def plot_multi_objective_comparison(results: Dict, n_objectives: int):
    """
    Plot comparison of multi-objective bandit algorithms.
    
    Args:
        results (Dict): Results from compare_multi_objective_algorithms
        n_objectives (int): Number of objectives
    """
    if n_objectives == 2:
        # 2D plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(results.items()):
            final_means = result['final_empirical_means']
            pareto_frontier = result['final_pareto_frontier']
            
            # Plot all points
            axes[i].scatter(final_means[:, 0], final_means[:, 1], 
                          alpha=0.6, label='All Arms')
            
            # Plot Pareto frontier
            if pareto_frontier:
                pareto_array = np.array(pareto_frontier)
                axes[i].scatter(pareto_array[:, 0], pareto_array[:, 1], 
                              color='red', s=100, label='Pareto Frontier')
            
            axes[i].set_xlabel('Objective 1')
            axes[i].set_ylabel('Objective 2')
            axes[i].set_title(name)
            axes[i].legend()
            axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    else:
        # For higher dimensions, plot hypervolume over time
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for name, result in results.items():
            pareto_frontiers = result['pareto_frontiers']
            
            # Calculate hypervolume over time (simplified)
            hypervolumes = []
            for frontier in pareto_frontiers:
                if frontier:
                    # Simplified hypervolume calculation
                    hv = len(frontier)  # Number of Pareto optimal points
                    hypervolumes.append(hv)
                else:
                    hypervolumes.append(0)
            
            ax.plot(hypervolumes, label=name, linewidth=2)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Hypervolume (Simplified)')
        ax.set_title('Multi-Objective Bandit Comparison')
        ax.legend()
        ax.grid(True)
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running Multi-Objective Bandit Example")
    
    # Compare algorithms
    n_arms = 10
    n_objectives = 2
    n_steps = 1000
    
    results = compare_multi_objective_algorithms(n_arms, n_objectives, n_steps)
    
    # Plot results
    plot_multi_objective_comparison(results, n_objectives)
    
    # Print summary
    print("\nMulti-Objective Bandit Comparison Summary:")
    for name, result in results.items():
        final_pareto = result['final_pareto_frontier']
        print(f"{name}:")
        print(f"  Number of Pareto optimal arms: {len(final_pareto)}")
        print(f"  Average reward per step: {np.mean([np.mean(r) for r in result['rewards_history']]):.3f}")
        print() 
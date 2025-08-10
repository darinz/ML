"""
Best Arm Identification (BAI) Algorithms

This module implements algorithms for best arm identification,
which focuses on identifying the best arm with high confidence
rather than maximizing cumulative reward.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from scipy.stats import norm


class SuccessiveElimination:
    """
    Successive Elimination algorithm for best arm identification.
    
    This algorithm eliminates arms that are clearly suboptimal
    based on confidence intervals.
    """
    
    def __init__(self, n_arms: int, delta: float = 0.1, n0: int = None):
        """
        Initialize Successive Elimination.
        
        Args:
            n_arms (int): Number of arms
            delta (float): Confidence parameter
            n0 (int): Initial number of pulls per arm
        """
        self.n_arms = n_arms
        self.delta = delta
        self.n0 = n0 if n0 else max(1, int(np.log(n_arms / delta)))
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.active_arms = set(range(n_arms))
        
        # History for analysis
        self.elimination_history = []
        self.confidence_intervals_history = []
    
    def select_arm(self) -> int:
        """
        Select an arm to pull.
        
        Returns:
            int: Index of selected arm
        """
        if not self.active_arms:
            return -1  # No active arms
        
        # Pull each active arm equally
        arm = min(self.active_arms, key=lambda a: self.pulls[a])
        return arm
    
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
        
        # Check for elimination after initial exploration
        if min(self.pulls) >= self.n0:
            self._eliminate_arms()
    
    def _eliminate_arms(self):
        """Eliminate arms that are clearly suboptimal."""
        if len(self.active_arms) <= 1:
            return
        
        # Calculate confidence intervals
        confidence_intervals = self._get_confidence_intervals()
        
        # Find best arm among active arms
        best_arm = max(self.active_arms, key=lambda a: self.empirical_means[a])
        
        # Eliminate arms
        arms_to_eliminate = set()
        for arm in self.active_arms:
            if arm != best_arm:
                lower_best, upper_best = confidence_intervals[best_arm]
                lower_arm, upper_arm = confidence_intervals[arm]
                
                # Eliminate if upper bound of arm < lower bound of best arm
                if upper_arm < lower_best:
                    arms_to_eliminate.add(arm)
        
        # Update active arms
        self.active_arms -= arms_to_eliminate
        
        # Store history
        self.elimination_history.append(len(arms_to_eliminate))
        self.confidence_intervals_history.append(confidence_intervals)
    
    def _get_confidence_intervals(self) -> List[Tuple[float, float]]:
        """
        Get confidence intervals for all arms.
        
        Returns:
            List[Tuple[float, float]]: Confidence intervals
        """
        intervals = []
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                intervals.append((0.0, 1.0))
            else:
                # Hoeffding confidence interval
                radius = np.sqrt(np.log(2 * self.n_arms / self.delta) / (2 * self.pulls[arm]))
                lower = self.empirical_means[arm] - radius
                upper = self.empirical_means[arm] + radius
                intervals.append((lower, upper))
        return intervals
    
    def get_best_arm(self) -> int:
        """Get the current best arm estimate."""
        if not self.active_arms:
            return np.argmax(self.empirical_means)
        return max(self.active_arms, key=lambda a: self.empirical_means[a])
    
    def is_complete(self) -> bool:
        """Check if the algorithm has identified the best arm."""
        return len(self.active_arms) == 1


class RacingAlgorithm:
    """
    Racing algorithm for best arm identification.
    
    This algorithm maintains confidence intervals and stops
    when one arm is clearly the best.
    """
    
    def __init__(self, n_arms: int, delta: float = 0.1):
        """
        Initialize Racing algorithm.
        
        Args:
            n_arms (int): Number of arms
            delta (float): Confidence parameter
        """
        self.n_arms = n_arms
        self.delta = delta
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.active_arms = set(range(n_arms))
        
        # Racing parameters
        self.phase = 0
        self.phase_length = 1
        
        # History for analysis
        self.phase_history = []
        self.active_arms_history = []
    
    def select_arm(self) -> int:
        """
        Select an arm to pull.
        
        Returns:
            int: Index of selected arm
        """
        if not self.active_arms:
            return -1  # No active arms
        
        # Pull each active arm equally in current phase
        arm = min(self.active_arms, key=lambda a: self.pulls[a])
        return arm
    
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
        
        # Check if phase is complete
        if all(self.pulls[arm] >= self.phase_length for arm in self.active_arms):
            self._advance_phase()
    
    def _advance_phase(self):
        """Advance to next phase and eliminate arms."""
        # Calculate confidence intervals
        confidence_intervals = self._get_confidence_intervals()
        
        # Find best arm among active arms
        best_arm = max(self.active_arms, key=lambda a: self.empirical_means[a])
        
        # Eliminate arms
        arms_to_eliminate = set()
        for arm in self.active_arms:
            if arm != best_arm:
                lower_best, upper_best = confidence_intervals[best_arm]
                lower_arm, upper_arm = confidence_intervals[arm]
                
                # Eliminate if upper bound of arm < lower bound of best arm
                if upper_arm < lower_best:
                    arms_to_eliminate.add(arm)
        
        # Update active arms
        self.active_arms -= arms_to_eliminate
        
        # Advance phase
        self.phase += 1
        self.phase_length = 2 ** self.phase
        
        # Store history
        self.phase_history.append(self.phase)
        self.active_arms_history.append(len(self.active_arms))
    
    def _get_confidence_intervals(self) -> List[Tuple[float, float]]:
        """
        Get confidence intervals for all arms.
        
        Returns:
            List[Tuple[float, float]]: Confidence intervals
        """
        intervals = []
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                intervals.append((0.0, 1.0))
            else:
                # Hoeffding confidence interval
                radius = np.sqrt(np.log(2 * self.n_arms / self.delta) / (2 * self.pulls[arm]))
                lower = self.empirical_means[arm] - radius
                upper = self.empirical_means[arm] + radius
                intervals.append((lower, upper))
        return intervals
    
    def get_best_arm(self) -> int:
        """Get the current best arm estimate."""
        if not self.active_arms:
            return np.argmax(self.empirical_means)
        return max(self.active_arms, key=lambda a: self.empirical_means[a])
    
    def is_complete(self) -> bool:
        """Check if the algorithm has identified the best arm."""
        return len(self.active_arms) == 1


class LUCB:
    """
    Lower-Upper Confidence Bound (LUCB) algorithm.
    
    This algorithm maintains confidence intervals and pulls
    the arms with highest upper bound and highest lower bound.
    """
    
    def __init__(self, n_arms: int, delta: float = 0.1):
        """
        Initialize LUCB algorithm.
        
        Args:
            n_arms (int): Number of arms
            delta (float): Confidence parameter
        """
        self.n_arms = n_arms
        self.delta = delta
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        
        # LUCB parameters
        self.alpha = 1.0  # Exploration parameter
        
        # History for analysis
        self.ucb_history = []
        self.lcb_history = []
    
    def select_arm(self) -> int:
        """
        Select an arm to pull.
        
        Returns:
            int: Index of selected arm
        """
        # If any arm hasn't been pulled, pull it
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                return arm
        
        # Calculate confidence intervals
        confidence_intervals = self._get_confidence_intervals()
        
        # Find arm with highest upper bound
        ucb_values = [upper for _, upper in confidence_intervals]
        best_ucb_arm = np.argmax(ucb_values)
        
        # Find arm with highest lower bound (excluding best UCB arm)
        lcb_values = [lower for lower, _ in confidence_intervals]
        lcb_values[best_ucb_arm] = -float('inf')  # Exclude best UCB arm
        best_lcb_arm = np.argmax(lcb_values)
        
        # Pull the arm with highest UCB or highest LCB
        if ucb_values[best_ucb_arm] > lcb_values[best_lcb_arm]:
            return best_ucb_arm
        else:
            return best_lcb_arm
    
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
        
        # Store history
        confidence_intervals = self._get_confidence_intervals()
        self.ucb_history.append([upper for _, upper in confidence_intervals])
        self.lcb_history.append([lower for lower, _ in confidence_intervals])
    
    def _get_confidence_intervals(self) -> List[Tuple[float, float]]:
        """
        Get confidence intervals for all arms.
        
        Returns:
            List[Tuple[float, float]]: Confidence intervals
        """
        intervals = []
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                intervals.append((0.0, 1.0))
            else:
                # Hoeffding confidence interval
                radius = self.alpha * np.sqrt(np.log(2 * self.n_arms / self.delta) / (2 * self.pulls[arm]))
                lower = self.empirical_means[arm] - radius
                upper = self.empirical_means[arm] + radius
                intervals.append((lower, upper))
        return intervals
    
    def get_best_arm(self) -> int:
        """Get the current best arm estimate."""
        return np.argmax(self.empirical_means)
    
    def is_complete(self) -> bool:
        """Check if the algorithm has identified the best arm."""
        confidence_intervals = self._get_confidence_intervals()
        
        # Find best arm
        best_arm = np.argmax(self.empirical_means)
        best_lower, best_upper = confidence_intervals[best_arm]
        
        # Check if best arm is clearly better than all others
        for arm in range(self.n_arms):
            if arm != best_arm:
                lower, upper = confidence_intervals[arm]
                if upper >= best_lower:
                    return False
        
        return True


class SequentialHalving:
    """
    Sequential Halving algorithm for best arm identification.
    
    This algorithm divides the budget into phases and eliminates
    half of the remaining arms in each phase.
    """
    
    def __init__(self, n_arms: int, budget: int):
        """
        Initialize Sequential Halving.
        
        Args:
            n_arms (int): Number of arms
            budget (int): Total budget (number of pulls)
        """
        self.n_arms = n_arms
        self.budget = budget
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.active_arms = set(range(n_arms))
        
        # Sequential halving parameters
        self.phase = 0
        self.pulls_per_phase = self._calculate_pulls_per_phase()
        
        # History for analysis
        self.phase_history = []
        self.active_arms_history = []
    
    def _calculate_pulls_per_phase(self) -> int:
        """Calculate number of pulls per arm in current phase."""
        n_active = len(self.active_arms)
        if n_active == 1:
            return 0
        
        # Distribute remaining budget equally among active arms
        remaining_budget = self.budget - np.sum(self.pulls)
        return max(1, remaining_budget // n_active)
    
    def select_arm(self) -> int:
        """
        Select an arm to pull.
        
        Returns:
            int: Index of selected arm
        """
        if not self.active_arms:
            return -1  # No active arms
        
        # Pull each active arm equally in current phase
        arm = min(self.active_arms, key=lambda a: self.pulls[a])
        return arm
    
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
        
        # Check if phase is complete
        if all(self.pulls[arm] >= self.pulls_per_phase for arm in self.active_arms):
            self._advance_phase()
    
    def _advance_phase(self):
        """Advance to next phase and eliminate arms."""
        if len(self.active_arms) <= 1:
            return
        
        # Eliminate worst half of arms
        n_eliminate = len(self.active_arms) // 2
        worst_arms = sorted(self.active_arms, key=lambda a: self.empirical_means[a])[:n_eliminate]
        
        # Update active arms
        self.active_arms -= set(worst_arms)
        
        # Advance phase
        self.phase += 1
        self.pulls_per_phase = self._calculate_pulls_per_phase()
        
        # Store history
        self.phase_history.append(self.phase)
        self.active_arms_history.append(len(self.active_arms))
    
    def get_best_arm(self) -> int:
        """Get the current best arm estimate."""
        if not self.active_arms:
            return np.argmax(self.empirical_means)
        return max(self.active_arms, key=lambda a: self.empirical_means[a])
    
    def is_complete(self) -> bool:
        """Check if the algorithm has identified the best arm."""
        return len(self.active_arms) == 1


def run_bai_experiment(algorithm, arm_means: List[float], 
                      max_pulls: int = 10000) -> Dict:
    """
    Run best arm identification experiment.
    
    Args:
        algorithm: BAI algorithm
        arm_means (List[float]): True means of arms
        max_pulls (int): Maximum number of pulls
        
    Returns:
        Dict: Experiment results
    """
    n_arms = len(arm_means)
    optimal_arm = np.argmax(arm_means)
    
    pulls = 0
    rewards = []
    actions = []
    best_arm_estimates = []
    
    while pulls < max_pulls and not algorithm.is_complete():
        # Select arm
        arm = algorithm.select_arm()
        if arm == -1:
            break
        
        # Get reward
        reward = np.random.normal(arm_means[arm], 0.1)
        reward = np.clip(reward, 0, 1)
        
        # Update algorithm
        algorithm.update(arm, reward)
        
        # Store results
        rewards.append(reward)
        actions.append(arm)
        best_arm_estimates.append(algorithm.get_best_arm())
        
        pulls += 1
    
    # Check if correct arm was identified
    final_best_arm = algorithm.get_best_arm()
    correct_identification = (final_best_arm == optimal_arm)
    
    return {
        'total_pulls': pulls,
        'rewards': rewards,
        'actions': actions,
        'best_arm_estimates': best_arm_estimates,
        'final_best_arm': final_best_arm,
        'optimal_arm': optimal_arm,
        'correct_identification': correct_identification,
        'is_complete': algorithm.is_complete()
    }


def compare_bai_algorithms(arm_means: List[float], 
                          max_pulls: int = 10000,
                          n_runs: int = 50) -> Dict:
    """
    Compare different BAI algorithms.
    
    Args:
        arm_means (List[float]): True means of arms
        max_pulls (int): Maximum number of pulls
        n_runs (int): Number of independent runs
        
    Returns:
        Dict: Comparison results
    """
    n_arms = len(arm_means)
    optimal_arm = np.argmax(arm_means)
    
    algorithms = {
        'Successive Elimination': SuccessiveElimination(n_arms),
        'Racing': RacingAlgorithm(n_arms),
        'LUCB': LUCB(n_arms),
        'Sequential Halving': SequentialHalving(n_arms, max_pulls)
    }
    
    results = {}
    
    for name, algorithm_class in algorithms.items():
        print(f"Running {name}")
        
        total_pulls_list = []
        correct_identifications = 0
        
        for run in range(n_runs):
            # Initialize algorithm
            if name == 'Sequential Halving':
                algorithm = algorithm_class
            else:
                algorithm = algorithm_class.__class__(**algorithm_class.__dict__)
            
            # Run experiment
            result = run_bai_experiment(algorithm, arm_means, max_pulls)
            
            total_pulls_list.append(result['total_pulls'])
            if result['correct_identification']:
                correct_identifications += 1
        
        results[name] = {
            'avg_pulls': np.mean(total_pulls_list),
            'std_pulls': np.std(total_pulls_list),
            'success_rate': correct_identifications / n_runs,
            'total_pulls_list': total_pulls_list
        }
    
    return results


def plot_bai_comparison(results: Dict):
    """
    Plot comparison of BAI algorithms.
    
    Args:
        results (Dict): Results from compare_bai_algorithms
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot average pulls
    algorithms = list(results.keys())
    avg_pulls = [results[alg]['avg_pulls'] for alg in algorithms]
    std_pulls = [results[alg]['std_pulls'] for alg in algorithms]
    
    ax1.bar(algorithms, avg_pulls, yerr=std_pulls, capsize=5)
    ax1.set_ylabel('Average Number of Pulls')
    ax1.set_title('BAI Algorithm Comparison - Sample Complexity')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot success rate
    success_rates = [results[alg]['success_rate'] for alg in algorithms]
    
    ax2.bar(algorithms, success_rates)
    ax2.set_ylabel('Success Rate')
    ax2.set_title('BAI Algorithm Comparison - Success Rate')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running Best Arm Identification Example")
    
    # Define arm means
    arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_pulls = 10000
    n_runs = 50
    
    # Compare algorithms
    results = compare_bai_algorithms(arm_means, max_pulls, n_runs)
    
    # Plot results
    plot_bai_comparison(results)
    
    # Print summary
    print("\nBest Arm Identification Comparison Summary:")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Average Pulls: {result['avg_pulls']:.1f} ± {result['std_pulls']:.1f}")
        print(f"  Success Rate: {result['success_rate']:.3f}")
        print() 
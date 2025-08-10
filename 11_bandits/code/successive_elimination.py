import numpy as np
from scipy.stats import norm

class SuccessiveElimination:
    """
    Successive Elimination algorithm for Best Arm Identification.
    
    This algorithm eliminates arms progressively based on empirical comparisons.
    """
    
    def __init__(self, n_arms, delta=0.1, n0=None):
        """
        Initialize Successive Elimination algorithm.
        
        Args:
            n_arms: Number of arms
            delta: Failure probability
            n0: Initial sample size per arm
        """
        self.n_arms = n_arms
        self.delta = delta
        self.n0 = n0 if n0 else max(1, int(np.log(n_arms / delta)))
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.active_arms = set(range(n_arms))
        
    def select_arm(self):
        """
        Select arm to pull.
        
        Returns:
            int: Index of selected arm
        """
        if len(self.active_arms) == 1:
            return list(self.active_arms)[0]
        
        # Pull arms that haven't been pulled n0 times
        for arm in self.active_arms:
            if self.pulls[arm] < self.n0:
                return arm
        
        # All arms have been pulled n0 times, eliminate worst arm
        worst_arm = min(self.active_arms, key=lambda i: self.empirical_means[i])
        self.active_arms.remove(worst_arm)
        
        # Continue pulling remaining arms
        return list(self.active_arms)[0]
    
    def update(self, arm, reward):
        """
        Update algorithm with observed reward.
        
        Args:
            arm: Index of pulled arm
            reward: Observed reward
        """
        self.pulls[arm] += 1
        self.empirical_means[arm] = ((self.empirical_means[arm] * (self.pulls[arm] - 1) + reward) / self.pulls[arm])
    
    def get_best_arm(self):
        """
        Return the identified best arm.
        
        Returns:
            int: Index of best arm
        """
        if len(self.active_arms) == 1:
            return list(self.active_arms)[0]
        else:
            return np.argmax(self.empirical_means)
    
    def is_complete(self):
        """
        Check if algorithm has completed.
        
        Returns:
            bool: True if algorithm has identified best arm
        """
        return len(self.active_arms) == 1

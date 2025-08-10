import numpy as np

class RacingAlgorithm:
    """
    Racing algorithm for Best Arm Identification.
    
    This algorithm maintains confidence intervals for all arms and stops
    when one arm is clearly the best.
    """
    
    def __init__(self, n_arms, delta=0.1, n0=None):
        """
        Initialize Racing algorithm.
        
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
        self.total_pulls = 0
        
    def get_confidence_radius(self, arm):
        """
        Calculate confidence radius for arm.
        
        Args:
            arm: Arm index
            
        Returns:
            float: Confidence radius
        """
        if self.pulls[arm] == 0:
            return float('inf')
        
        # Hoeffding-based confidence interval
        beta = np.sqrt(np.log(2 * self.n_arms / self.delta) / (2 * self.pulls[arm]))
        return beta
    
    def get_confidence_intervals(self):
        """
        Get confidence intervals for all arms.
        
        Returns:
            list: List of (lower, upper) confidence intervals
        """
        intervals = []
        for arm in range(self.n_arms):
            radius = self.get_confidence_radius(arm)
            lower = self.empirical_means[arm] - radius
            upper = self.empirical_means[arm] + radius
            intervals.append((lower, upper))
        return intervals
    
    def select_arm(self):
        """
        Select arm to pull.
        
        Returns:
            int: Index of selected arm
        """
        # Initial phase: pull each arm n0 times
        for arm in range(self.n_arms):
            if self.pulls[arm] < self.n0:
                return arm
        
        # Racing phase: pull arm with highest uncertainty
        intervals = self.get_confidence_intervals()
        
        # Find arm with highest upper bound
        best_arm = max(range(self.n_arms), key=lambda i: intervals[i][1])
        
        # Find arm with highest lower bound (different from best)
        other_arms = [i for i in range(self.n_arms) if i != best_arm]
        if other_arms:
            challenger = max(other_arms, key=lambda i: intervals[i][0])
            
            # Pull the arm with highest uncertainty
            if intervals[best_arm][1] - intervals[best_arm][0] > intervals[challenger][1] - intervals[challenger][0]:
                return best_arm
            else:
                return challenger
        
        return best_arm
    
    def update(self, arm, reward):
        """
        Update algorithm with observed reward.
        
        Args:
            arm: Index of pulled arm
            reward: Observed reward
        """
        self.pulls[arm] += 1
        self.total_pulls += 1
        self.empirical_means[arm] = ((self.empirical_means[arm] * (self.pulls[arm] - 1) + reward) / self.pulls[arm])
    
    def get_best_arm(self):
        """
        Return the identified best arm.
        
        Returns:
            int: Index of best arm
        """
        return np.argmax(self.empirical_means)
    
    def is_complete(self):
        """
        Check if algorithm has completed.
        
        Returns:
            bool: True if algorithm has identified best arm
        """
        intervals = self.get_confidence_intervals()
        
        # Check if one arm's lower bound exceeds all others' upper bounds
        for i in range(self.n_arms):
            lower_i = intervals[i][0]
            all_others_upper = [intervals[j][1] for j in range(self.n_arms) if j != i]
            if all(lower_i > upper_j for upper_j in all_others_upper):
                return True
        
        return False

import numpy as np

class LUCB:
    """
    LUCB (Lower-Upper Confidence Bound) algorithm for Best Arm Identification.
    
    This algorithm pulls the arm with highest upper bound and the arm with
    highest lower bound among the remaining arms.
    """
    
    def __init__(self, n_arms, delta=0.1, n0=None):
        """
        Initialize LUCB algorithm.
        
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
        
        # LUCB-specific confidence interval
        beta = np.sqrt(np.log(4 * self.n_arms * self.total_pulls**2 / self.delta) / (2 * self.pulls[arm]))
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
        Select arm to pull using LUCB.
        
        Returns:
            int: Index of selected arm
        """
        # Initial phase: pull each arm n0 times
        for arm in range(self.n_arms):
            if self.pulls[arm] < self.n0:
                return arm
        
        # LUCB phase: pull arms with highest upper and lower bounds
        intervals = self.get_confidence_intervals()
        
        # Find arm with highest upper bound
        h = max(range(self.n_arms), key=lambda i: intervals[i][1])
        
        # Find arm with highest lower bound among others
        others = [i for i in range(self.n_arms) if i != h]
        l = max(others, key=lambda i: intervals[i][0])
        
        # Pull the arm with higher uncertainty
        if intervals[h][1] - intervals[h][0] > intervals[l][1] - intervals[l][0]:
            return h
        else:
            return l
    
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
        
        # Find arm with highest upper bound
        h = max(range(self.n_arms), key=lambda i: intervals[i][1])
        
        # Find arm with highest lower bound among others
        others = [i for i in range(self.n_arms) if i != h]
        l = max(others, key=lambda i: intervals[i][0])
        
        # Check if intervals separate
        return intervals[h][0] > intervals[l][1]

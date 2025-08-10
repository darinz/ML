import numpy as np

class SequentialHalving:
    """
    Sequential Halving algorithm for Best Arm Identification.
    
    This algorithm eliminates half of the remaining arms in each round.
    """
    
    def __init__(self, n_arms, budget):
        """
        Initialize Sequential Halving algorithm.
        
        Args:
            n_arms: Number of arms
            budget: Total budget (number of pulls)
        """
        self.n_arms = n_arms
        self.budget = budget
        
        # Calculate number of rounds
        self.n_rounds = int(np.log2(n_arms))
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.active_arms = list(range(n_arms))
        self.current_round = 0
        self.pulls_per_arm = 0
        
    def select_arm(self):
        """
        Select arm to pull.
        
        Returns:
            int: Index of selected arm
        """
        if len(self.active_arms) == 1:
            return self.active_arms[0]
        
        # Calculate pulls per arm for current round
        if self.current_round == 0:
            self.pulls_per_arm = self.budget // (len(self.active_arms) * self.n_rounds)
        else:
            self.pulls_per_arm = self.budget // (len(self.active_arms) * (self.n_rounds - self.current_round))
        
        # Find arm that needs more pulls
        for arm in self.active_arms:
            if self.pulls[arm] < self.pulls_per_arm:
                return arm
        
        # All arms have been pulled enough, eliminate bottom half
        self._eliminate_bottom_half()
        return self.active_arms[0]  # Return first remaining arm
    
    def _eliminate_bottom_half(self):
        """Eliminate bottom half of active arms."""
        # Sort arms by empirical means
        sorted_arms = sorted(self.active_arms, key=lambda i: self.empirical_means[i], reverse=True)
        
        # Keep top half
        keep_count = len(self.active_arms) // 2
        self.active_arms = sorted_arms[:keep_count]
        self.current_round += 1
    
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
            return self.active_arms[0]
        else:
            return max(self.active_arms, key=lambda i: self.empirical_means[i])
    
    def is_complete(self):
        """
        Check if algorithm has completed.
        
        Returns:
            bool: True if algorithm has identified best arm
        """
        return len(self.active_arms) == 1

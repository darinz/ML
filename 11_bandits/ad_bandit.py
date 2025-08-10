import numpy as np

class AdBandit:
    """Multi-armed bandit for online advertising."""
    
    def __init__(self, n_ads):
        self.n_ads = n_ads
        self.empirical_ctr = [0] * n_ads
        self.pulls = [0] * n_ads
        self.total_pulls = 0
        
    def select_ad(self, user_context):
        """
        Select an ad using UCB strategy.
        
        Args:
            user_context: User context (not used in basic implementation)
        
        Returns:
            int: Index of selected ad
        """
        # UCB selection
        ucb_values = []
        for i in range(self.n_ads):
            if self.pulls[i] == 0:
                ucb_values.append(float('inf'))
            else:
                exploration = np.sqrt(2 * np.log(self.total_pulls) / self.pulls[i])
                ucb_values.append(self.empirical_ctr[i] + exploration)
        
        return np.argmax(ucb_values)
    
    def update(self, ad_id, click):
        """
        Update the bandit with observed click.
        
        Args:
            ad_id: Index of the ad that was shown
            click: Binary reward (1 for click, 0 for no click)
        """
        self.pulls[ad_id] += 1
        self.total_pulls += 1
        self.empirical_ctr[ad_id] = ((self.empirical_ctr[ad_id] * (self.pulls[ad_id] - 1) + click) / self.pulls[ad_id])

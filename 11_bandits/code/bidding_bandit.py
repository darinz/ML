import numpy as np
from contextual_thompson_sampling import ContextualThompsonSampling

class BiddingBandit:
    """
    Bidding Bandit for real-time bidding optimization.
    
    This class implements a contextual bandit for learning optimal bid amounts
    in real-time bidding auctions to maximize profit.
    """
    
    def __init__(self, n_bid_levels, user_feature_dim):
        """
        Initialize Bidding Bandit.
        
        Args:
            n_bid_levels: Number of bid levels to choose from
            user_feature_dim: Dimension of user feature vectors
        """
        self.bandit = ContextualThompsonSampling(user_feature_dim)
        self.bid_levels = np.linspace(0.1, 10.0, n_bid_levels)
        
    def select_bid(self, user_context, reserve_price):
        """
        Select bid amount based on user context.
        
        Args:
            user_context: User feature vector
            reserve_price: Reserve price for the auction
            
        Returns:
            float: Selected bid amount
        """
        # Create contextual features for each bid level
        contextual_features = []
        for bid in self.bid_levels:
            # Combine user context with bid information
            bid_features = [bid, reserve_price, bid/reserve_price]
            combined = np.concatenate([user_context, bid_features])
            contextual_features.append(combined)
        
        bid_idx = self.bandit.select_arm(contextual_features)
        return self.bid_levels[bid_idx]
    
    def update(self, bid_idx, user_context, reserve_price, won, revenue, cost):
        """
        Update with auction outcome.
        
        Args:
            bid_idx: Index of selected bid level
            user_context: User feature vector
            reserve_price: Reserve price for the auction
            won: Whether the auction was won (True/False)
            revenue: Revenue if won, 0 otherwise
            cost: Cost if won, 0 otherwise
        """
        reward = revenue - cost if won else 0
        
        contextual_features = []
        for bid in self.bid_levels:
            bid_features = [bid, reserve_price, bid/reserve_price]
            combined = np.concatenate([user_context, bid_features])
            contextual_features.append(combined)
        
        self.bandit.update(bid_idx, reward, contextual_features)

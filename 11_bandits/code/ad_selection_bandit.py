import numpy as np
from contextual_ucb import ContextualUCB

class AdSelectionBandit:
    """
    Ad Selection Bandit for online advertising.
    
    This class implements a contextual bandit for selecting the best ad creative
    based on user context to maximize click-through rate (CTR).
    """
    
    def __init__(self, n_ads, user_feature_dim):
        """
        Initialize Ad Selection Bandit.
        
        Args:
            n_ads: Number of available ads
            user_feature_dim: Dimension of user feature vectors
        """
        self.bandit = ContextualUCB(user_feature_dim)
        self.ad_features = self._extract_ad_features(n_ads)
        
    def select_ad(self, user_context):
        """
        Select ad based on user context.
        
        Args:
            user_context: User feature vector
            
        Returns:
            int: Index of selected ad
        """
        contextual_features = self._combine_features(user_context, self.ad_features)
        ad_idx = self.bandit.select_arm(contextual_features)
        return ad_idx
    
    def update(self, ad_idx, user_context, click):
        """
        Update with click feedback.
        
        Args:
            ad_idx: Index of shown ad
            user_context: User feature vector
            click: Binary click indicator (1 for click, 0 for no click)
        """
        contextual_features = self._combine_features(user_context, self.ad_features)
        self.bandit.update(ad_idx, click, contextual_features)
    
    def _extract_ad_features(self, n_ads):
        """
        Extract features for each ad.
        
        Args:
            n_ads: Number of ads
            
        Returns:
            np.ndarray: Array of ad features
        """
        features = []
        for i in range(n_ads):
            # Ad-specific features: category, color, text length, etc.
            ad_features = [
                i % 5,  # Category
                (i // 5) % 3,  # Color scheme
                np.random.randint(10, 100),  # Text length
                np.random.choice([0, 1]),  # Has image
                np.random.choice([0, 1])   # Has video
            ]
            features.append(ad_features)
        return np.array(features)
    
    def _combine_features(self, user_context, ad_features):
        """
        Combine user and ad features.
        
        Args:
            user_context: User feature vector
            ad_features: Array of ad feature vectors
            
        Returns:
            list: List of combined feature vectors
        """
        combined_features = []
        for ad_feat in ad_features:
            # Simple concatenation
            combined = np.concatenate([user_context, ad_feat])
            combined_features.append(combined)
        return combined_features

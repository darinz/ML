import numpy as np
from neural_contextual_bandit import NeuralContextualBandit

class ContentRecommender:
    """
    Content Recommender using neural contextual bandits.
    
    This class implements a neural contextual bandit for recommending content
    (articles, videos, music) based on user preferences.
    """
    
    def __init__(self, n_items, user_feature_dim):
        """
        Initialize Content Recommender.
        
        Args:
            n_items: Number of content items
            user_feature_dim: Dimension of user feature vectors
        """
        self.bandit = NeuralContextualBandit(user_feature_dim, num_arms=n_items)
        self.item_features = self._extract_item_features(n_items)
        
    def recommend(self, user_context, n_recommendations=5):
        """
        Recommend content items.
        
        Args:
            user_context: User feature vector
            n_recommendations: Number of recommendations to return
            
        Returns:
            list: List of recommended item indices
        """
        contextual_features = self._create_contextual_features(user_context)
        
        # Get top-k recommendations
        recommendations = []
        for _ in range(n_recommendations):
            item_idx = self.bandit.select_arm(contextual_features)
            recommendations.append(item_idx)
        
        return recommendations
    
    def update(self, item_idx, user_context, engagement):
        """
        Update with user engagement.
        
        Args:
            item_idx: Index of recommended item
            user_context: User feature vector
            engagement: User engagement metric (view time, likes, shares)
        """
        contextual_features = self._create_contextual_features(user_context)
        self.bandit.update(item_idx, engagement, contextual_features)
    
    def _extract_item_features(self, n_items):
        """
        Extract features for each content item.
        
        Args:
            n_items: Number of items
            
        Returns:
            np.ndarray: Array of item features
        """
        features = []
        for i in range(n_items):
            item_features = [
                i % 10,  # Category
                np.random.randint(100, 10000),  # Length
                np.random.choice([0, 1]),  # Has image
                np.random.choice([0, 1]),  # Has video
                np.random.uniform(0, 5),  # Average rating
                np.random.randint(0, 1000)  # Popularity
            ]
            features.append(item_features)
        return np.array(features)
    
    def _create_contextual_features(self, user_context):
        """
        Create contextual features for all items.
        
        Args:
            user_context: User feature vector
            
        Returns:
            list: List of contextual feature vectors
        """
        contextual_features = []
        for item_feat in self.item_features:
            # Combine user context with item features
            combined = np.concatenate([user_context, item_feat])
            contextual_features.append(combined)
        return contextual_features

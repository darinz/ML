import numpy as np
from linucb import LinUCB

class ProductRecommender:
    """
    Product Recommender using linear contextual bandits.
    
    This class implements a linear contextual bandit for recommending products
    in e-commerce based on user behavior.
    """
    
    def __init__(self, n_products, user_feature_dim):
        """
        Initialize Product Recommender.
        
        Args:
            n_products: Number of products
            user_feature_dim: Dimension of user feature vectors
        """
        self.bandit = LinUCB(user_feature_dim)
        self.product_features = self._extract_product_features(n_products)
        
    def recommend_products(self, user_context, n_recommendations=10):
        """
        Recommend products based on user context.
        
        Args:
            user_context: User feature vector
            n_recommendations: Number of recommendations to return
            
        Returns:
            list: List of recommended product indices
        """
        contextual_features = self._create_contextual_features(user_context)
        
        # Get recommendations using bandit
        recommendations = []
        for _ in range(n_recommendations):
            product_idx = self.bandit.select_arm(contextual_features)
            recommendations.append(product_idx)
        
        return recommendations
    
    def update(self, product_idx, user_context, purchased):
        """
        Update with purchase feedback.
        
        Args:
            product_idx: Index of recommended product
            user_context: User feature vector
            purchased: Binary purchase indicator (1 for purchase, 0 for no purchase)
        """
        contextual_features = self._create_contextual_features(user_context)
        self.bandit.update(product_idx, purchased, contextual_features)
    
    def _extract_product_features(self, n_products):
        """
        Extract features for each product.
        
        Args:
            n_products: Number of products
            
        Returns:
            np.ndarray: Array of product features
        """
        features = []
        for i in range(n_products):
            product_features = [
                i % 20,  # Category
                np.random.uniform(10, 1000),  # Price
                np.random.uniform(0, 5),  # Rating
                np.random.randint(0, 1000),  # Sales count
                np.random.choice([0, 1]),  # In stock
                np.random.choice([0, 1])   # On sale
            ]
            features.append(product_features)
        return np.array(features)
    
    def _create_contextual_features(self, user_context):
        """
        Create contextual features for all products.
        
        Args:
            user_context: User feature vector
            
        Returns:
            list: List of contextual feature vectors
        """
        contextual_features = []
        for product_feat in self.product_features:
            # Combine user context with product features
            combined = np.concatenate([user_context, product_feat])
            contextual_features.append(combined)
        return contextual_features

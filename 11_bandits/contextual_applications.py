"""
Application examples for contextual bandits.

This module contains example implementations for various real-world applications
of contextual bandits including online advertising, recommendation systems, clinical trials, and dynamic pricing.
"""

import numpy as np
from contextual_ucb import ContextualUCB
from contextual_thompson_sampling import ContextualThompsonSampling
from neural_contextual_bandit import NeuralContextualBandit

class ContextualAdSelector:
    """
    Online advertising system using contextual bandits.
    """
    
    def __init__(self, n_ads, user_feature_dim):
        """
        Initialize contextual ad selector.
        
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
            user_context: User context feature vector
            
        Returns:
            int: Index of selected ad
        """
        # Combine user context with ad features
        contextual_features = self._combine_user_ad_features(user_context, self.ad_features)
        
        # Select ad using contextual bandit
        ad_idx = self.bandit.select_arm(contextual_features)
        return ad_idx
    
    def update(self, ad_idx, user_context, click):
        """
        Update model with click feedback.
        
        Args:
            ad_idx: Index of shown ad
            user_context: User context feature vector
            click: Binary click indicator
        """
        contextual_features = self._combine_user_ad_features(user_context, self.ad_features)
        self.bandit.update(ad_idx, click, contextual_features)
    
    def _extract_ad_features(self, n_ads):
        """Extract features for ads (placeholder implementation)."""
        # In practice, this would extract real ad features
        return np.random.randn(n_ads, self.bandit.d)
    
    def _combine_user_ad_features(self, user_context, ad_features):
        """Combine user and ad features for contextual learning."""
        combined_features = []
        for ad_feat in ad_features:
            # Simple concatenation
            combined = np.concatenate([user_context, ad_feat])
            combined_features.append(combined)
        return combined_features

class PersonalizedRecommender:
    """
    Personalized content recommendation system using contextual bandits.
    """
    
    def __init__(self, n_items, user_feature_dim):
        """
        Initialize personalized recommender.
        
        Args:
            n_items: Number of items to recommend
            user_feature_dim: Dimension of user feature vectors
        """
        self.bandit = ContextualThompsonSampling(user_feature_dim)
        self.item_features = self._extract_item_features(n_items)
    
    def recommend(self, user_context):
        """
        Recommend content based on user context.
        
        Args:
            user_context: User context feature vector
            
        Returns:
            int: Index of recommended item
        """
        # Combine user context with item features
        contextual_features = self._combine_user_item_features(user_context, self.item_features)
        
        # Select item using contextual bandit
        item_idx = self.bandit.select_arm(contextual_features)
        return item_idx
    
    def update(self, item_idx, user_context, engagement):
        """
        Update model with user engagement.
        
        Args:
            item_idx: Index of recommended item
            user_context: User context feature vector
            engagement: User engagement metric
        """
        contextual_features = self._combine_user_item_features(user_context, self.item_features)
        self.bandit.update(item_idx, engagement, contextual_features)
    
    def _extract_item_features(self, n_items):
        """Extract features for items (placeholder implementation)."""
        # In practice, this would extract real item features
        return np.random.randn(n_items, self.bandit.d)
    
    def _combine_user_item_features(self, user_context, item_features):
        """Combine user and item features for contextual learning."""
        combined_features = []
        for item_feat in item_features:
            # Simple concatenation
            combined = np.concatenate([user_context, item_feat])
            combined_features.append(combined)
        return combined_features

class AdaptiveClinicalTrial:
    """
    Adaptive clinical trial system using neural contextual bandits.
    """
    
    def __init__(self, n_treatments, patient_feature_dim):
        """
        Initialize adaptive clinical trial.
        
        Args:
            n_treatments: Number of available treatments
            patient_feature_dim: Dimension of patient feature vectors
        """
        self.bandit = NeuralContextualBandit(patient_feature_dim, num_arms=n_treatments)
    
    def assign_treatment(self, patient_features):
        """
        Assign treatment based on patient features.
        
        Args:
            patient_features: Patient feature vector
            
        Returns:
            int: Index of assigned treatment
        """
        # Convert patient features to contextual features
        contextual_features = self._create_contextual_features(patient_features)
        
        # Select treatment using neural contextual bandit
        treatment_idx = self.bandit.select_arm(contextual_features)
        return treatment_idx
    
    def update(self, treatment_idx, patient_features, outcome):
        """
        Update model with treatment outcome.
        
        Args:
            treatment_idx: Index of assigned treatment
            patient_features: Patient feature vector
            outcome: Treatment outcome (e.g., recovery, side effects)
        """
        contextual_features = self._create_contextual_features(patient_features)
        self.bandit.update(treatment_idx, outcome, contextual_features)
    
    def _create_contextual_features(self, patient_features):
        """Create contextual features for all treatments."""
        # For neural bandits, we need features for each treatment
        contextual_features = []
        for treatment in range(self.bandit.num_arms):
            # Combine patient features with treatment encoding
            treatment_features = np.zeros(self.bandit.num_arms)
            treatment_features[treatment] = 1.0
            combined = np.concatenate([patient_features, treatment_features])
            contextual_features.append(combined)
        return contextual_features

class DynamicPricer:
    """
    Dynamic pricing system using contextual bandits.
    """
    
    def __init__(self, n_price_levels, customer_feature_dim):
        """
        Initialize dynamic pricer.
        
        Args:
            n_price_levels: Number of available price levels
            customer_feature_dim: Dimension of customer feature vectors
        """
        self.bandit = ContextualUCB(customer_feature_dim)
        self.price_levels = np.linspace(10, 100, n_price_levels)
    
    def set_price(self, customer_features):
        """
        Set price based on customer features.
        
        Args:
            customer_features: Customer feature vector
            
        Returns:
            float: Selected price
        """
        # Create contextual features for each price level
        contextual_features = self._create_price_features(customer_features)
        
        # Select price using contextual bandit
        price_idx = self.bandit.select_arm(contextual_features)
        return self.price_levels[price_idx]
    
    def update(self, price_idx, customer_features, purchase):
        """
        Update model with purchase decision.
        
        Args:
            price_idx: Index of selected price
            customer_features: Customer feature vector
            purchase: Binary purchase indicator
        """
        contextual_features = self._create_price_features(customer_features)
        self.bandit.update(price_idx, purchase, contextual_features)
    
    def _create_price_features(self, customer_features):
        """Create contextual features for each price level."""
        contextual_features = []
        for price in self.price_levels:
            # Combine customer features with price
            price_feature = price / 100.0  # Normalize price
            combined = np.concatenate([customer_features, [price_feature]])
            contextual_features.append(combined)
        return contextual_features

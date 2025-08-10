import numpy as np
from contextual_ucb import ContextualUCB
from neural_contextual_bandit import NeuralContextualBandit

class DynamicPricer:
    """
    Dynamic Pricer using contextual bandits.
    
    This class implements a contextual bandit for setting optimal prices
    to maximize revenue.
    """
    
    def __init__(self, n_price_levels, customer_feature_dim):
        """
        Initialize Dynamic Pricer.
        
        Args:
            n_price_levels: Number of price levels
            customer_feature_dim: Dimension of customer feature vectors
        """
        self.bandit = ContextualUCB(customer_feature_dim)
        self.price_levels = np.linspace(10, 100, n_price_levels)
        
    def set_price(self, customer_features, market_conditions):
        """
        Set price based on customer and market.
        
        Args:
            customer_features: Customer feature vector
            market_conditions: Market condition features
            
        Returns:
            float: Selected price
        """
        contextual_features = self._create_contextual_features(customer_features, market_conditions)
        price_idx = self.bandit.select_arm(contextual_features)
        return self.price_levels[price_idx]
    
    def update(self, price_idx, customer_features, market_conditions, demand, revenue):
        """
        Update with demand and revenue data.
        
        Args:
            price_idx: Index of selected price level
            customer_features: Customer feature vector
            market_conditions: Market condition features
            demand: Observed demand
            revenue: Observed revenue
        """
        contextual_features = self._create_contextual_features(customer_features, market_conditions)
        self.bandit.update(price_idx, revenue, contextual_features)
    
    def _create_contextual_features(self, customer_features, market_conditions):
        """
        Create contextual features for each price level.
        
        Args:
            customer_features: Customer feature vector
            market_conditions: Market condition features
            
        Returns:
            list: List of contextual feature vectors
        """
        contextual_features = []
        for price in self.price_levels:
            # Combine customer, market, and price features
            price_features = [price, price/50.0, np.log(price)]  # Normalize price
            combined = np.concatenate([customer_features, market_conditions, price_features])
            contextual_features.append(combined)
        return contextual_features

class RevenueManager:
    """
    Revenue Manager using neural contextual bandits.
    
    This class implements a neural contextual bandit for optimizing pricing
    strategies for perishable inventory (hotels, airlines).
    """
    
    def __init__(self, n_pricing_strategies, market_feature_dim):
        """
        Initialize Revenue Manager.
        
        Args:
            n_pricing_strategies: Number of pricing strategies
            market_feature_dim: Dimension of market feature vectors
        """
        self.bandit = NeuralContextualBandit(market_feature_dim, num_arms=n_pricing_strategies)
        
    def set_pricing_strategy(self, market_features):
        """
        Select pricing strategy based on market conditions.
        
        Args:
            market_features: Market feature vector
            
        Returns:
            int: Index of selected pricing strategy
        """
        contextual_features = self._create_contextual_features(market_features)
        strategy_idx = self.bandit.select_arm(contextual_features)
        return strategy_idx
    
    def update(self, strategy_idx, market_features, profit_margin):
        """
        Update with profit margin data.
        
        Args:
            strategy_idx: Index of selected pricing strategy
            market_features: Market feature vector
            profit_margin: Observed profit margin
        """
        contextual_features = self._create_contextual_features(market_features)
        self.bandit.update(strategy_idx, profit_margin, contextual_features)
    
    def _create_contextual_features(self, market_features):
        """
        Create contextual features for each pricing strategy.
        
        Args:
            market_features: Market feature vector
            
        Returns:
            list: List of contextual feature vectors
        """
        contextual_features = []
        for strategy in range(self.bandit.num_arms):
            # Combine market features with strategy encoding
            strategy_features = np.zeros(self.bandit.num_arms)
            strategy_features[strategy] = 1.0
            combined = np.concatenate([market_features, strategy_features])
            contextual_features.append(combined)
        return contextual_features

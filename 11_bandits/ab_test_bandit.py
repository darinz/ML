import numpy as np
from successive_elimination import SuccessiveElimination

class ABTestBandit:
    """
    A/B Test Bandit for website optimization.
    
    This class implements a best arm identification bandit for testing
    different website designs to maximize conversion rate.
    """
    
    def __init__(self, n_variants, user_feature_dim):
        """
        Initialize A/B Test Bandit.
        
        Args:
            n_variants: Number of website variants
            user_feature_dim: Dimension of user feature vectors
        """
        self.bandit = SuccessiveElimination(n_variants, delta=0.05)
        self.variant_features = self._extract_variant_features(n_variants)
        
    def select_variant(self, user_context):
        """
        Select website variant to show.
        
        Args:
            user_context: User feature vector
            
        Returns:
            int: Index of selected variant
        """
        # For BAI, we focus on identification rather than cumulative reward
        variant_idx = self.bandit.select_arm()
        return variant_idx
    
    def update(self, variant_idx, user_context, converted):
        """
        Update with conversion data.
        
        Args:
            variant_idx: Index of shown variant
            user_context: User feature vector
            converted: Binary conversion indicator (1 for conversion, 0 for no conversion)
        """
        self.bandit.update(variant_idx, converted)
    
    def get_best_variant(self):
        """
        Get the identified best variant.
        
        Returns:
            int: Index of best variant
        """
        return self.bandit.get_best_arm()
    
    def is_complete(self):
        """
        Check if A/B test is complete.
        
        Returns:
            bool: True if test has identified best variant
        """
        return self.bandit.is_complete()
    
    def _extract_variant_features(self, n_variants):
        """
        Extract features for each website variant.
        
        Args:
            n_variants: Number of variants
            
        Returns:
            np.ndarray: Array of variant features
        """
        features = []
        for i in range(n_variants):
            variant_features = [
                i % 3,  # Layout type
                (i // 3) % 2,  # Color scheme
                (i // 6) % 2,  # Button style
                np.random.randint(1, 10),  # Content length
                np.random.choice([0, 1])   # Has video
            ]
            features.append(variant_features)
        return np.array(features)

class AlgorithmSelector:
    """
    Algorithm Selector for machine learning model selection.
    
    This class implements a best arm identification bandit for choosing
    the best algorithm for a specific dataset.
    """
    
    def __init__(self, algorithms, dataset_feature_dim):
        """
        Initialize Algorithm Selector.
        
        Args:
            algorithms: List of ML algorithms to evaluate
            dataset_feature_dim: Dimension of dataset feature vectors
        """
        from racing_algorithm import RacingAlgorithm
        self.algorithms = algorithms
        self.bandit = RacingAlgorithm(len(algorithms), delta=0.1)
        
    def select_algorithm(self, dataset_features):
        """
        Select algorithm to evaluate.
        
        Args:
            dataset_features: Dataset feature vector
            
        Returns:
            object: Selected algorithm
        """
        algorithm_idx = self.bandit.select_arm()
        return self.algorithms[algorithm_idx]
    
    def update(self, algorithm_idx, dataset_features, performance):
        """
        Update with algorithm performance.
        
        Args:
            algorithm_idx: Index of evaluated algorithm
            dataset_features: Dataset feature vector
            performance: Performance metric (accuracy, F1-score)
        """
        self.bandit.update(algorithm_idx, performance)
    
    def get_best_algorithm(self):
        """
        Get the identified best algorithm.
        
        Returns:
            object: Best algorithm
        """
        return self.algorithms[self.bandit.get_best_arm()]
    
    def is_complete(self):
        """
        Check if algorithm selection is complete.
        
        Returns:
            bool: True if selection has identified best algorithm
        """
        return self.bandit.is_complete()

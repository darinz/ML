"""
Feature Engineering for Bandit Algorithms

This module provides tools for feature extraction and preprocessing
for contextual bandit algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression


class FeatureEngineer:
    """
    Feature engineering utilities for bandit algorithms.
    
    This class provides methods for feature extraction, preprocessing,
    and dimensionality reduction for contextual bandits.
    """
    
    def __init__(self, feature_dim: int = None):
        """
        Initialize feature engineer.
        
        Args:
            feature_dim (int): Target feature dimension
        """
        self.feature_dim = feature_dim
        self.scaler = None
        self.pca = None
        self.feature_selector = None
        
    def normalize_features(self, features: np.ndarray, method: str = 'standard') -> np.ndarray:
        """
        Normalize features using different methods.
        
        Args:
            features (np.ndarray): Input features
            method (str): Normalization method ('standard', 'minmax', 'l2')
            
        Returns:
            np.ndarray: Normalized features
        """
        if method == 'standard':
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(features)
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            return self.scaler.fit_transform(features)
        elif method == 'l2':
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return features / norms
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def apply_pca(self, features: np.ndarray, n_components: int = None) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            features (np.ndarray): Input features
            n_components (int): Number of components to keep
            
        Returns:
            np.ndarray: Reduced features
        """
        if n_components is None:
            n_components = min(features.shape[1], self.feature_dim or features.shape[1])
        
        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(features)
    
    def select_features(self, features: np.ndarray, targets: np.ndarray, 
                       k: int = None) -> np.ndarray:
        """
        Select top-k features based on correlation with targets.
        
        Args:
            features (np.ndarray): Input features
            targets (np.ndarray): Target values
            k (int): Number of features to select
            
        Returns:
            np.ndarray: Selected features
        """
        if k is None:
            k = min(features.shape[1], self.feature_dim or features.shape[1])
        
        self.feature_selector = SelectKBest(score_func=f_regression, k=k)
        return self.feature_selector.fit_transform(features, targets)
    
    def create_polynomial_features(self, features: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Create polynomial features.
        
        Args:
            features (np.ndarray): Input features
            degree (int): Polynomial degree
            
        Returns:
            np.ndarray: Polynomial features
        """
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(features)
    
    def create_interaction_features(self, features: np.ndarray) -> np.ndarray:
        """
        Create interaction features between pairs of features.
        
        Args:
            features (np.ndarray): Input features
            
        Returns:
            np.ndarray: Features with interactions
        """
        n_features = features.shape[1]
        interactions = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction = features[:, i] * features[:, j]
                interactions.append(interaction)
        
        if interactions:
            interaction_features = np.column_stack(interactions)
            return np.hstack([features, interaction_features])
        else:
            return features
    
    def create_time_features(self, n_samples: int, include_trend: bool = True) -> np.ndarray:
        """
        Create time-based features.
        
        Args:
            n_samples (int): Number of samples
            include_trend (bool): Whether to include trend features
            
        Returns:
            np.ndarray: Time features
        """
        time_features = []
        
        # Basic time features
        time_features.append(np.arange(n_samples))  # Time step
        time_features.append(np.arange(n_samples) % 24)  # Hour of day
        time_features.append(np.arange(n_samples) % 7)   # Day of week
        
        if include_trend:
            time_features.append(np.arange(n_samples) ** 2)  # Quadratic trend
            time_features.append(np.sin(2 * np.pi * np.arange(n_samples) / 24))  # Daily cycle
            time_features.append(np.cos(2 * np.pi * np.arange(n_samples) / 24))  # Daily cycle
        
        return np.column_stack(time_features)
    
    def create_user_features(self, user_ids: List[int], n_features: int = 5) -> np.ndarray:
        """
        Create user-specific features.
        
        Args:
            user_ids (List[int]): List of user IDs
            n_features (int): Number of features per user
            
        Returns:
            np.ndarray: User features
        """
        unique_users = list(set(user_ids))
        user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        
        features = []
        for user_id in user_ids:
            user_idx = user_to_idx[user_id]
            # Create deterministic features based on user ID
            np.random.seed(user_id)
            user_feat = np.random.randn(n_features)
            features.append(user_feat)
        
        return np.array(features)
    
    def create_contextual_features(self, base_features: np.ndarray, 
                                 context_features: np.ndarray) -> np.ndarray:
        """
        Create contextual features by combining base and context features.
        
        Args:
            base_features (np.ndarray): Base features (e.g., arm features)
            context_features (np.ndarray): Context features (e.g., user features)
            
        Returns:
            np.ndarray: Combined contextual features
        """
        # Simple concatenation
        combined = np.hstack([base_features, context_features])
        
        # Add interaction terms
        interactions = []
        for i in range(base_features.shape[1]):
            for j in range(context_features.shape[1]):
                interaction = base_features[:, i:i+1] * context_features[:, j:j+1]
                interactions.append(interaction)
        
        if interactions:
            interaction_features = np.hstack(interactions)
            return np.hstack([combined, interaction_features])
        else:
            return combined


class ArmFeatureExtractor:
    """
    Extract features for arms in bandit problems.
    """
    
    @staticmethod
    def extract_numeric_features(arm_data: List[Dict]) -> np.ndarray:
        """
        Extract numeric features from arm data.
        
        Args:
            arm_data (List[Dict]): List of dictionaries containing arm information
            
        Returns:
            np.ndarray: Numeric features
        """
        features = []
        for arm in arm_data:
            # Extract numeric values
            numeric_values = []
            for key, value in arm.items():
                if isinstance(value, (int, float)):
                    numeric_values.append(value)
            features.append(numeric_values)
        
        return np.array(features)
    
    @staticmethod
    def extract_categorical_features(arm_data: List[Dict], 
                                  categorical_keys: List[str]) -> np.ndarray:
        """
        Extract categorical features using one-hot encoding.
        
        Args:
            arm_data (List[Dict]): List of dictionaries containing arm information
            categorical_keys (List[str]): Keys for categorical features
            
        Returns:
            np.ndarray: One-hot encoded features
        """
        # Collect all unique values for each categorical feature
        unique_values = {}
        for key in categorical_keys:
            unique_values[key] = set()
            for arm in arm_data:
                if key in arm:
                    unique_values[key].add(arm[key])
        
        # Create one-hot encoding
        features = []
        for arm in arm_data:
            arm_features = []
            for key in categorical_keys:
                if key in arm:
                    for value in sorted(unique_values[key]):
                        arm_features.append(1 if arm[key] == value else 0)
                else:
                    # Handle missing values
                    arm_features.extend([0] * len(unique_values[key]))
            features.append(arm_features)
        
        return np.array(features)
    
    @staticmethod
    def extract_text_features(arm_data: List[Dict], 
                            text_keys: List[str], 
                            max_features: int = 100) -> np.ndarray:
        """
        Extract text features using simple bag-of-words.
        
        Args:
            arm_data (List[Dict]): List of dictionaries containing arm information
            text_keys (List[str]): Keys for text features
            max_features (int): Maximum number of features
            
        Returns:
            np.ndarray: Text features
        """
        from collections import Counter
        
        # Collect all words
        all_words = []
        for arm in arm_data:
            for key in text_keys:
                if key in arm and isinstance(arm[key], str):
                    words = arm[key].lower().split()
                    all_words.extend(words)
        
        # Get most common words
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(max_features)]
        
        # Create feature vectors
        features = []
        for arm in arm_data:
            arm_text = ""
            for key in text_keys:
                if key in arm and isinstance(arm[key], str):
                    arm_text += " " + arm[key].lower()
            
            arm_words = arm_text.split()
            word_counts = Counter(arm_words)
            
            arm_features = []
            for word in common_words:
                arm_features.append(word_counts.get(word, 0))
            features.append(arm_features)
        
        return np.array(features)


class ContextFeatureExtractor:
    """
    Extract context features for contextual bandits.
    """
    
    @staticmethod
    def extract_user_features(user_data: Dict) -> np.ndarray:
        """
        Extract user features from user data.
        
        Args:
            user_data (Dict): User information dictionary
            
        Returns:
            np.ndarray: User features
        """
        features = []
        
        # Demographics
        if 'age' in user_data:
            features.append(user_data['age'])
        if 'gender' in user_data:
            features.append(1 if user_data['gender'] == 'M' else 0)
        
        # Location
        if 'location' in user_data:
            # Simple location encoding (could be more sophisticated)
            location_hash = hash(user_data['location']) % 100
            features.append(location_hash)
        
        # Device
        if 'device' in user_data:
            device_mapping = {'mobile': 0, 'desktop': 1, 'tablet': 2}
            features.append(device_mapping.get(user_data['device'], 0))
        
        # Time features
        if 'timestamp' in user_data:
            import datetime
            dt = datetime.datetime.fromtimestamp(user_data['timestamp'])
            features.extend([
                dt.hour,
                dt.weekday(),
                dt.month
            ])
        
        return np.array(features)
    
    @staticmethod
    def extract_session_features(session_data: Dict) -> np.ndarray:
        """
        Extract session-based features.
        
        Args:
            session_data (Dict): Session information dictionary
            
        Returns:
            np.ndarray: Session features
        """
        features = []
        
        # Session length
        if 'session_length' in session_data:
            features.append(session_data['session_length'])
        
        # Previous interactions
        if 'previous_clicks' in session_data:
            features.append(session_data['previous_clicks'])
        if 'previous_purchases' in session_data:
            features.append(session_data['previous_purchases'])
        
        # Session time
        if 'session_start_time' in session_data:
            features.append(session_data['session_start_time'])
        
        return np.array(features)
    
    @staticmethod
    def extract_market_features(market_data: Dict) -> np.ndarray:
        """
        Extract market/context features.
        
        Args:
            market_data (Dict): Market information dictionary
            
        Returns:
            np.ndarray: Market features
        """
        features = []
        
        # Market conditions
        if 'demand_level' in market_data:
            features.append(market_data['demand_level'])
        if 'competition_level' in market_data:
            features.append(market_data['competition_level'])
        
        # Seasonal factors
        if 'season' in market_data:
            season_mapping = {'spring': 0, 'summer': 1, 'fall': 2, 'winter': 3}
            features.append(season_mapping.get(market_data['season'], 0))
        
        # Economic indicators
        if 'economic_index' in market_data:
            features.append(market_data['economic_index'])
        
        return np.array(features)


def create_synthetic_features(n_samples: int, n_features: int, 
                            feature_type: str = 'random') -> np.ndarray:
    """
    Create synthetic features for testing.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        feature_type (str): Type of features ('random', 'correlated', 'sparse')
        
    Returns:
        np.ndarray: Synthetic features
    """
    if feature_type == 'random':
        return np.random.randn(n_samples, n_features)
    
    elif feature_type == 'correlated':
        # Create correlated features
        base_features = np.random.randn(n_samples, n_features // 2)
        correlated_features = base_features + 0.5 * np.random.randn(n_samples, n_features // 2)
        return np.hstack([base_features, correlated_features])
    
    elif feature_type == 'sparse':
        # Create sparse features
        features = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            # Randomly activate some features
            active_features = np.random.choice(n_features, size=n_features // 4, replace=False)
            features[i, active_features] = np.random.randn(len(active_features))
        return features
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def analyze_feature_importance(features: np.ndarray, targets: np.ndarray) -> Dict:
    """
    Analyze feature importance for bandit algorithms.
    
    Args:
        features (np.ndarray): Feature matrix
        targets (np.ndarray): Target values
        
    Returns:
        Dict: Feature importance analysis
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    
    # Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(features, targets)
    rf_importance = rf.feature_importances_
    
    # Linear regression coefficients
    lr = LinearRegression()
    lr.fit(features, targets)
    lr_coefficients = np.abs(lr.coef_)
    
    # Correlation with target
    correlations = []
    for i in range(features.shape[1]):
        corr = np.corrcoef(features[:, i], targets)[0, 1]
        correlations.append(abs(corr) if not np.isnan(corr) else 0)
    
    return {
        'random_forest_importance': rf_importance,
        'linear_regression_coefficients': lr_coefficients,
        'correlations': correlations,
        'feature_ranking': np.argsort(rf_importance)[::-1]
    }


def plot_feature_analysis(features: np.ndarray, targets: np.ndarray):
    """
    Plot feature analysis for bandit algorithms.
    
    Args:
        features (np.ndarray): Feature matrix
        targets (np.ndarray): Target values
    """
    analysis = analyze_feature_importance(features, targets)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Feature importance (Random Forest)
    n_features = features.shape[1]
    ax1.bar(range(n_features), analysis['random_forest_importance'])
    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('Importance')
    ax1.set_title('Random Forest Feature Importance')
    
    # Linear regression coefficients
    ax2.bar(range(n_features), analysis['linear_regression_coefficients'])
    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('|Coefficient|')
    ax2.set_title('Linear Regression Coefficients')
    
    # Correlations with target
    ax3.bar(range(n_features), analysis['correlations'])
    ax3.set_xlabel('Feature Index')
    ax3.set_ylabel('|Correlation|')
    ax3.set_title('Feature-Target Correlations')
    
    # Feature distribution
    ax4.hist(features.flatten(), bins=50, alpha=0.7)
    ax4.set_xlabel('Feature Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Feature Distribution')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running Feature Engineering Example")
    
    # Create synthetic data
    n_samples = 1000
    n_features = 10
    
    # Generate features
    features = create_synthetic_features(n_samples, n_features, 'correlated')
    targets = np.random.randn(n_samples)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(feature_dim=5)
    
    # Apply feature engineering
    normalized_features = engineer.normalize_features(features, 'standard')
    reduced_features = engineer.apply_pca(normalized_features, n_components=5)
    
    print(f"Original features shape: {features.shape}")
    print(f"Normalized features shape: {normalized_features.shape}")
    print(f"Reduced features shape: {reduced_features.shape}")
    
    # Analyze features
    plot_feature_analysis(features, targets) 
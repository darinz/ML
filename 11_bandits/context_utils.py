import numpy as np

def extract_user_features(user_data):
    """
    Extract user features for contextual bandit.
    
    Args:
        user_data: Dictionary containing user information
        
    Returns:
        np.ndarray: User feature vector
    """
    features = []
    
    # Demographics
    features.extend([
        user_data.get('age', 0) / 100.0,  # Normalize age
        user_data.get('gender', 0),  # Binary or categorical
        user_data.get('location', 0)  # Location encoding
    ])
    
    # Behavioral features
    features.extend([
        user_data.get('session_duration', 0) / 3600.0,  # Hours
        user_data.get('page_views', 0) / 100.0,  # Normalize
        user_data.get('purchase_history', 0)  # Purchase count
    ])
    
    # Contextual features
    features.extend([
        user_data.get('time_of_day', 0) / 24.0,  # Hour of day
        user_data.get('day_of_week', 0) / 7.0,  # Day of week
        user_data.get('device_type', 0)  # Device encoding
    ])
    
    return np.array(features)

def extract_item_features(item_data):
    """
    Extract item features for contextual bandit.
    
    Args:
        item_data: Dictionary containing item information
        
    Returns:
        np.ndarray: Item feature vector
    """
    features = []
    
    # Item characteristics
    features.extend([
        item_data.get('category', 0),  # Category encoding
        item_data.get('price', 0) / 1000.0,  # Normalize price
        item_data.get('rating', 0) / 5.0,  # Normalize rating
        item_data.get('popularity', 0) / 1000.0  # Normalize popularity
    ])
    
    # Content features
    features.extend([
        item_data.get('content_length', 0) / 1000.0,
        item_data.get('has_image', 0),  # Binary
        item_data.get('has_video', 0)  # Binary
    ])
    
    return np.array(features)

def combine_context_arm_features(context, arm_features):
    """
    Combine context and arm features.
    
    Args:
        context: Context feature vector
        arm_features: Arm feature vector
        
    Returns:
        np.ndarray: Combined feature vector
    """
    # Simple concatenation
    combined = np.concatenate([context, arm_features])
    
    # Or use interaction features
    interactions = np.outer(context, arm_features).flatten()
    combined = np.concatenate([context, arm_features, interactions])
    
    return combined

def create_contextual_features(context, all_arms):
    """
    Create contextual features for all arms.
    
    Args:
        context: Context feature vector
        all_arms: List of arm feature vectors
        
    Returns:
        list: List of contextual feature vectors for each arm
    """
    contextual_features = []
    
    for arm_features in all_arms:
        combined = combine_context_arm_features(context, arm_features)
        contextual_features.append(combined)
    
    return contextual_features

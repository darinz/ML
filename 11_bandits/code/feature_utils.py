import numpy as np

def normalize_features(features):
    """
    Normalize feature vectors to unit norm.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        
    Returns:
        np.ndarray: Normalized features
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms

def standardize_features(features):
    """
    Standardize features to zero mean and unit variance.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        
    Returns:
        np.ndarray: Standardized features
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / std

def stable_solve(A, b):
    """
    Stable matrix solve with regularization.
    
    Args:
        A: Design matrix
        b: Target vector
        
    Returns:
        np.ndarray: Solution vector
    """
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Add small regularization if matrix is singular
        A_reg = A + 1e-6 * np.eye(A.shape[0])
        return np.linalg.solve(A_reg, b)

def check_conditioning(A):
    """
    Check condition number of design matrix.
    
    Args:
        A: Design matrix
        
    Returns:
        float: Condition number
    """
    eigenvals = np.linalg.eigvals(A)
    condition_number = np.max(eigenvals) / np.min(eigenvals)
    return condition_number

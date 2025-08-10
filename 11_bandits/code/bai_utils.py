import numpy as np

def stable_confidence_radius(pulls, delta, method='hoeffding'):
    """
    Calculate stable confidence radius.
    
    Args:
        pulls: Number of pulls for the arm
        delta: Failure probability
        method: Method for confidence interval ('hoeffding' or 'chernoff')
        
    Returns:
        float: Confidence radius
    """
    if pulls == 0:
        return float('inf')
    
    if method == 'hoeffding':
        return np.sqrt(np.log(2 / delta) / (2 * pulls))
    elif method == 'chernoff':
        # For Bernoulli rewards
        return np.sqrt(np.log(1 / delta) / pulls)
    else:
        raise ValueError(f"Unknown method: {method}")

def stable_mean_update(old_mean, old_count, new_value):
    """
    Stable incremental mean update.
    
    Args:
        old_mean: Previous mean
        old_count: Previous count
        new_value: New value to add
        
    Returns:
        float: Updated mean
    """
    return (old_mean * old_count + new_value) / (old_count + 1)

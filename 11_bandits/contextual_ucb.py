import numpy as np
from scipy.linalg import solve

class ContextualUCB:
    """
    Contextual Upper Confidence Bound (CUCB) algorithm for contextual bandits.
    
    This algorithm extends the UCB principle to handle changing contexts by maintaining
    context-dependent confidence intervals.
    """
    
    def __init__(self, d, alpha=1.0, lambda_reg=1.0):
        """
        Initialize Contextual UCB algorithm.
        
        Args:
            d: Dimension of feature vectors
            alpha: Exploration parameter (typically sqrt(d * log T))
            lambda_reg: Regularization parameter for ridge regression
        """
        self.d = d
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Initialize design matrix and parameter estimate
        self.A = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.theta_hat = np.zeros(d)
        
    def select_arm(self, context_features):
        """
        Select arm using Contextual UCB algorithm.
        
        Args:
            context_features: List of feature vectors for available arms in current context
            
        Returns:
            int: Index of selected arm
        """
        # Update parameter estimate
        self.theta_hat = solve(self.A, self.b)
        
        # Calculate UCB values for all arms in current context
        ucb_values = []
        for x in context_features:
            # Exploitation term
            exploitation = np.dot(self.theta_hat, x)
            
            # Exploration term
            exploration = self.alpha * np.sqrt(np.dot(x, solve(self.A, x)))
            
            ucb_values.append(exploitation + exploration)
        
        return np.argmax(ucb_values)
    
    def update(self, arm_idx, reward, context_features):
        """
        Update algorithm with observed reward.
        
        Args:
            arm_idx: Index of the arm that was pulled
            reward: Observed reward
            context_features: List of feature vectors for all arms in the context
        """
        x = context_features[arm_idx]
        
        # Update design matrix and cumulative rewards
        self.A += np.outer(x, x)
        self.b += reward * x

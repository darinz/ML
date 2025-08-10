import numpy as np

class OFUL:
    """
    Optimism in the Face of Uncertainty for Linear Bandits (OFUL).
    
    This algorithm constructs confidence ellipsoids and uses optimism
    to guide exploration in linear bandits.
    """
    
    def __init__(self, d, delta=0.1, lambda_reg=1.0):
        """
        Initialize OFUL algorithm.
        
        Args:
            d: Dimension of feature vectors
            delta: Confidence parameter (failure probability)
            lambda_reg: Regularization parameter
        """
        self.d = d
        self.delta = delta
        self.lambda_reg = lambda_reg
        
        self.A = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.theta_hat = np.zeros(d)
        
    def select_arm(self, arms):
        """
        Select arm using OFUL algorithm.
        
        Args:
            arms: List of feature vectors for available arms
            
        Returns:
            int: Index of selected arm
        """
        # Update parameter estimate
        self.theta_hat = np.linalg.solve(self.A, self.b)
        
        # Calculate confidence radius
        beta = self._calculate_beta()
        
        # Find optimistic parameter within confidence ellipsoid
        theta_opt = self._find_optimistic_parameter(arms, beta)
        
        # Choose arm with highest optimistic reward
        predicted_rewards = [np.dot(theta_opt, x) for x in arms]
        return np.argmax(predicted_rewards)
    
    def _calculate_beta(self):
        """
        Calculate confidence radius.
        
        Returns:
            float: Confidence radius beta
        """
        t = np.trace(self.A) - self.d * self.lambda_reg
        return np.sqrt(2 * np.log(1 / self.delta) + self.d * np.log(1 + t / (self.d * self.lambda_reg)))
    
    def _find_optimistic_parameter(self, arms, beta):
        """
        Find optimistic parameter within confidence ellipsoid.
        
        Args:
            arms: List of feature vectors
            beta: Confidence radius
            
        Returns:
            np.ndarray: Optimistic parameter vector
        """
        # This is a simplified version - in practice, this requires solving an optimization problem
        # For simplicity, we'll use the UCB approach
        return self.theta_hat + beta * np.linalg.solve(self.A, np.random.randn(self.d))
    
    def update(self, arm_idx, reward, arm_features):
        """
        Update algorithm with observed reward.
        
        Args:
            arm_idx: Index of the arm that was pulled
            reward: Observed reward
            arm_features: List of feature vectors for all arms
        """
        x = arm_features[arm_idx]
        
        # Update design matrix and cumulative rewards
        self.A += np.outer(x, x)
        self.b += reward * x

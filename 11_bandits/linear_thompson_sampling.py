import numpy as np
from scipy.stats import multivariate_normal

class LinearThompsonSampling:
    """
    Linear Thompson Sampling algorithm for linear bandits.
    
    This algorithm maintains a Gaussian posterior over the parameter vector
    and samples from this posterior to select actions.
    """
    
    def __init__(self, d, sigma=1.0, lambda_reg=1.0):
        """
        Initialize Linear Thompson Sampling algorithm.
        
        Args:
            d: Dimension of feature vectors
            sigma: Noise standard deviation
            lambda_reg: Regularization parameter for prior
        """
        self.d = d
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        
        # Initialize posterior parameters
        self.A = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.theta_hat = np.zeros(d)
        
    def select_arm(self, arms):
        """
        Select arm using Linear Thompson Sampling.
        
        Args:
            arms: List of feature vectors for available arms
            
        Returns:
            int: Index of selected arm
        """
        # Update parameter estimate
        self.theta_hat = np.linalg.solve(self.A, self.b)
        
        # Sample from posterior
        posterior_cov = self.sigma**2 * np.linalg.inv(self.A)
        theta_sample = multivariate_normal.rvs(
            mean=self.theta_hat, 
            cov=posterior_cov
        )
        
        # Choose arm with highest sampled reward
        predicted_rewards = [np.dot(theta_sample, x) for x in arms]
        return np.argmax(predicted_rewards)
    
    def update(self, arm_idx, reward, arm_features):
        """
        Update algorithm with observed reward.
        
        Args:
            arm_idx: Index of the arm that was pulled
            reward: Observed reward
            arm_features: List of feature vectors for all arms
        """
        x = arm_features[arm_idx]
        
        # Update posterior parameters
        self.A += np.outer(x, x) / (self.sigma**2)
        self.b += reward * x / (self.sigma**2)

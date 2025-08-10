import numpy as np

class DisjointLinUCB:
    """
    LinUCB with disjoint models for contextual bandits.
    
    This algorithm maintains separate linear models for each arm,
    allowing for more flexible context-arm interactions.
    """
    
    def __init__(self, d, num_arms, alpha=1.0, lambda_reg=1.0):
        """
        Initialize Disjoint LinUCB algorithm.
        
        Args:
            d: Dimension of context features
            num_arms: Number of available arms
            alpha: Exploration parameter
            lambda_reg: Regularization parameter
        """
        self.d = d
        self.num_arms = num_arms
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Separate models for each arm
        self.A = [lambda_reg * np.eye(d) for _ in range(num_arms)]
        self.b = [np.zeros(d) for _ in range(num_arms)]
        self.theta_hat = [np.zeros(d) for _ in range(num_arms)]
        
    def select_arm(self, context):
        """
        Select arm using disjoint LinUCB.
        
        Args:
            context: Context feature vector
            
        Returns:
            int: Index of selected arm
        """
        ucb_values = []
        
        for i in range(self.num_arms):
            # Update parameter estimate for arm i
            self.theta_hat[i] = np.linalg.solve(self.A[i], self.b[i])
            
            # Calculate UCB value for arm i
            exploitation = np.dot(self.theta_hat[i], context)
            exploration = self.alpha * np.sqrt(np.dot(context, np.linalg.solve(self.A[i], context)))
            ucb_values.append(exploitation + exploration)
        
        return np.argmax(ucb_values)
    
    def update(self, arm_idx, reward, context):
        """
        Update model for specific arm.
        
        Args:
            arm_idx: Index of the arm that was pulled
            reward: Observed reward
            context: Context feature vector
        """
        # Update design matrix and cumulative rewards for the chosen arm
        self.A[arm_idx] += np.outer(context, context)
        self.b[arm_idx] += reward * context

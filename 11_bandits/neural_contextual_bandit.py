import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class NeuralContextualBandit:
    """
    Neural contextual bandit using deep neural networks to model complex,
    non-linear reward functions that depend on context.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_arms=10, dropout_rate=0.1):
        """
        Initialize Neural Contextual Bandit.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            num_arms: Number of available arms
            dropout_rate: Dropout rate for regularization
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_arms = num_arms
        self.dropout_rate = dropout_rate
        
        # Neural network model
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_arms)
        )
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def select_arm(self, context_features):
        """
        Select arm using neural contextual bandit.
        
        Args:
            context_features: List of feature vectors for available arms in current context
            
        Returns:
            int: Index of selected arm
        """
        # Convert context features to tensor
        context_tensor = torch.FloatTensor(context_features).unsqueeze(0)
        
        # Get predictions from neural network
        with torch.no_grad():
            predictions = self.model(context_tensor)
            
        # Add exploration noise (Thompson sampling approximation)
        noise = torch.randn_like(predictions) * 0.1
        predictions += noise
        
        return torch.argmax(predictions).item()
    
    def update(self, arm_idx, reward, context_features):
        """
        Update neural network with observed reward.
        
        Args:
            arm_idx: Index of the arm that was pulled
            reward: Observed reward
            context_features: List of feature vectors for all arms in the context
        """
        context_tensor = torch.FloatTensor(context_features).unsqueeze(0)
        
        # Create target vector
        target = torch.zeros(self.num_arms)
        target[arm_idx] = reward
        target = target.unsqueeze(0)
        
        # Forward pass
        predictions = self.model(context_tensor)
        
        # Compute loss and update
        loss = self.criterion(predictions, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

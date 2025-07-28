"""
Neural Bandits Implementation

This module implements neural network-based bandit algorithms.
It includes deep learning approaches for both classical and contextual bandits.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class NeuralBandit(nn.Module):
    """
    Neural network-based bandit algorithm.
    
    This implementation uses a neural network to model the reward function
    and provides uncertainty estimates for exploration.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 n_arms: int = 1, dropout_rate: float = 0.1):
        """
        Initialize neural bandit.
        
        Args:
            input_dim (int): Input dimension (context + arm features)
            hidden_dim (int): Hidden layer dimension
            n_arms (int): Number of arms
            dropout_rate (float): Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_arms = n_arms
        self.dropout_rate = dropout_rate
        
        # Neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, n_arms)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Training parameters
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Experience buffer
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
        # Uncertainty estimation
        self.ensemble_size = 5
        self.ensemble_networks = []
        self._create_ensemble()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_ensemble(self):
        """Create ensemble of networks for uncertainty estimation."""
        for _ in range(self.ensemble_size):
            network = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.n_arms)
            )
            self._initialize_weights_network(network)
            self.ensemble_networks.append(network)
    
    def _initialize_weights_network(self, network):
        """Initialize weights for a specific network."""
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        return self.network(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict rewards with uncertainty estimates.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predictions and uncertainties
        """
        predictions = []
        
        # Get predictions from ensemble
        for network in self.ensemble_networks:
            with torch.no_grad():
                pred = network(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate mean and variance
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0)
        
        return mean_pred, uncertainty
    
    def select_arm(self, context: np.ndarray, arm_features: List[np.ndarray] = None) -> int:
        """
        Select arm using neural network predictions.
        
        Args:
            context (np.ndarray): Context vector
            arm_features (List[np.ndarray]): Features for each arm
            
        Returns:
            int: Index of selected arm
        """
        if arm_features is None:
            # Simple case: just context
            x = torch.FloatTensor(context).unsqueeze(0)
        else:
            # Combine context with arm features
            combined_features = []
            for arm_feat in arm_features:
                combined = np.concatenate([context, arm_feat])
                combined_features.append(combined)
            x = torch.FloatTensor(combined_features)
        
        # Get predictions with uncertainty
        mean_pred, uncertainty = self.predict_with_uncertainty(x)
        
        # UCB-style selection: mean + uncertainty
        ucb_values = mean_pred + 0.1 * torch.sqrt(uncertainty)
        
        return torch.argmax(ucb_values).item()
    
    def update(self, arm: int, reward: float, context: np.ndarray, 
               arm_features: np.ndarray = None):
        """
        Update the network with observed reward.
        
        Args:
            arm (int): Index of pulled arm
            reward (float): Observed reward
            context (np.ndarray): Context vector
            arm_features (np.ndarray): Features of the pulled arm
        """
        # Store experience
        if arm_features is not None:
            combined = np.concatenate([context, arm_features])
        else:
            combined = context
        
        self.experience_buffer.append((combined, arm, reward))
        
        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # Train network if we have enough data
        if len(self.experience_buffer) >= 32:
            self._train_network()
    
    def _train_network(self):
        """Train the neural network."""
        # Sample batch
        batch_size = min(64, len(self.experience_buffer))
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        features_batch = []
        arms_batch = []
        rewards_batch = []
        
        for idx in batch_indices:
            features, arm, reward = self.experience_buffer[idx]
            features_batch.append(features)
            arms_batch.append(arm)
            rewards_batch.append(reward)
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features_batch)
        arms_tensor = torch.LongTensor(arms_batch)
        rewards_tensor = torch.FloatTensor(rewards_batch)
        
        # Forward pass
        predictions = self.forward(features_tensor)
        
        # Create target values
        targets = torch.zeros_like(predictions)
        targets[torch.arange(batch_size), arms_tensor] = rewards_tensor
        
        # Backward pass
        self.optimizer.zero_grad()
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()
        
        # Update ensemble networks
        self._update_ensemble(features_tensor, targets)
    
    def _update_ensemble(self, features: torch.Tensor, targets: torch.Tensor):
        """Update ensemble networks."""
        for network in self.ensemble_networks:
            optimizer = optim.Adam(network.parameters(), lr=0.001)
            
            predictions = network(features)
            loss = self.criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


class DeepContextualBandit(nn.Module):
    """
    Deep contextual bandit using neural networks.
    
    This implementation uses a deep neural network to model
    the reward function for contextual bandits with uncertainty estimation.
    """
    
    def __init__(self, context_dim: int, n_arms: int, hidden_dims: List[int] = [128, 64],
                 dropout_rate: float = 0.1):
        """
        Initialize deep contextual bandit.
        
        Args:
            context_dim (int): Context dimension
            n_arms (int): Number of arms
            hidden_dims (List[int]): Hidden layer dimensions
            dropout_rate (float): Dropout rate
        """
        super().__init__()
        
        self.context_dim = context_dim
        self.n_arms = n_arms
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build network architecture
        layers = []
        input_dim = context_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, n_arms))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Training parameters
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # Experience buffer
        self.experience_buffer = []
        self.max_buffer_size = 10000
        
        # Uncertainty estimation
        self.ensemble_size = 3
        self.ensemble_networks = []
        self._create_ensemble()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _create_ensemble(self):
        """Create ensemble of networks."""
        for _ in range(self.ensemble_size):
            layers = []
            input_dim = self.context_dim
            
            for hidden_dim in self.hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate)
                ])
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, self.n_arms))
            network = nn.Sequential(*layers)
            
            # Initialize weights
            for module in network.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            
            self.ensemble_networks.append(network)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            context (torch.Tensor): Context tensor
            
        Returns:
            torch.Tensor: Reward predictions for all arms
        """
        return self.network(context)
    
    def predict_with_uncertainty(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict rewards with uncertainty estimates.
        
        Args:
            context (torch.Tensor): Context tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predictions and uncertainties
        """
        predictions = []
        
        # Get predictions from ensemble
        for network in self.ensemble_networks:
            with torch.no_grad():
                pred = network(context)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate mean and variance
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.var(predictions, dim=0)
        
        return mean_pred, uncertainty
    
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select arm using deep contextual bandit.
        
        Args:
            context (np.ndarray): Context vector
            
        Returns:
            int: Index of selected arm
        """
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        
        # Get predictions with uncertainty
        mean_pred, uncertainty = self.predict_with_uncertainty(context_tensor)
        
        # UCB-style selection
        ucb_values = mean_pred + 0.1 * torch.sqrt(uncertainty)
        
        return torch.argmax(ucb_values).item()
    
    def update(self, arm: int, reward: float, context: np.ndarray):
        """
        Update the network with observed reward.
        
        Args:
            arm (int): Index of pulled arm
            reward (float): Observed reward
            context (np.ndarray): Context vector
        """
        # Store experience
        self.experience_buffer.append((context, arm, reward))
        
        # Limit buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        # Train network if we have enough data
        if len(self.experience_buffer) >= 32:
            self._train_network()
    
    def _train_network(self):
        """Train the neural network."""
        # Sample batch
        batch_size = min(64, len(self.experience_buffer))
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        contexts_batch = []
        arms_batch = []
        rewards_batch = []
        
        for idx in batch_indices:
            context, arm, reward = self.experience_buffer[idx]
            contexts_batch.append(context)
            arms_batch.append(arm)
            rewards_batch.append(reward)
        
        # Convert to tensors
        contexts_tensor = torch.FloatTensor(contexts_batch)
        arms_tensor = torch.LongTensor(arms_batch)
        rewards_tensor = torch.FloatTensor(rewards_batch)
        
        # Forward pass
        predictions = self.forward(contexts_tensor)
        
        # Create target values
        targets = torch.zeros_like(predictions)
        targets[torch.arange(batch_size), arms_tensor] = rewards_tensor
        
        # Backward pass
        self.optimizer.zero_grad()
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()
        
        # Update ensemble networks
        self._update_ensemble(contexts_tensor, targets)
    
    def _update_ensemble(self, contexts: torch.Tensor, targets: torch.Tensor):
        """Update ensemble networks."""
        for network in self.ensemble_networks:
            optimizer = optim.Adam(network.parameters(), lr=0.001)
            
            predictions = network(contexts)
            loss = self.criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_neural_bandit_experiment(bandit: NeuralBandit, 
                               context_generator: callable,
                               reward_function: callable,
                               n_steps: int = 1000) -> Dict:
    """
    Run neural bandit experiment.
    
    Args:
        bandit (NeuralBandit): Neural bandit algorithm
        context_generator (callable): Function to generate contexts
        reward_function (callable): Function to generate rewards
        n_steps (int): Number of time steps
        
    Returns:
        Dict: Experiment results
    """
    rewards = []
    regrets = []
    actions = []
    contexts = []
    
    for step in range(n_steps):
        # Generate context
        context = context_generator()
        
        # Select arm
        arm = bandit.select_arm(context)
        
        # Get reward
        reward = reward_function(arm, context)
        
        # Update bandit
        bandit.update(arm, reward, context)
        
        # Store results
        rewards.append(reward)
        actions.append(arm)
        contexts.append(context)
        
        # Calculate regret
        optimal_arm = np.argmax([reward_function(i, context) for i in range(bandit.n_arms)])
        optimal_reward = reward_function(optimal_arm, context)
        regret = optimal_reward - reward
        regrets.append(regret)
    
    return {
        'rewards': rewards,
        'regrets': regrets,
        'actions': actions,
        'contexts': contexts,
        'cumulative_reward': np.sum(rewards),
        'cumulative_regret': np.sum(regrets)
    }


def compare_neural_algorithms(context_dim: int, n_arms: int, 
                            n_steps: int = 1000) -> Dict:
    """
    Compare different neural bandit algorithms.
    
    Args:
        context_dim (int): Context dimension
        n_arms (int): Number of arms
        n_steps (int): Number of time steps
        
    Returns:
        Dict: Comparison results
    """
    # Define context and reward generators
    def context_generator():
        return np.random.randn(context_dim)
    
    def reward_function(arm: int, context: np.ndarray):
        # Non-linear reward function
        arm_features = np.random.randn(context_dim)
        expected_reward = np.tanh(np.dot(context, arm_features))
        noise = np.random.normal(0, 0.1)
        return expected_reward + noise
    
    # Initialize algorithms
    algorithms = {
        'Neural Bandit': NeuralBandit(context_dim, n_arms=n_arms),
        'Deep Contextual Bandit': DeepContextualBandit(context_dim, n_arms)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"Running {name}")
        result = run_neural_bandit_experiment(algorithm, context_generator, reward_function, n_steps)
        results[name] = result
    
    return results


def plot_neural_comparison(results: Dict, n_steps: int):
    """
    Plot comparison of neural bandit algorithms.
    
    Args:
        results (Dict): Results from compare_neural_algorithms
        n_steps (int): Number of time steps
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cumulative rewards
    for name, result in results.items():
        cumulative_rewards = np.cumsum(result['rewards'])
        ax1.plot(range(n_steps), cumulative_rewards, label=name, linewidth=2)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Neural Bandit Comparison - Cumulative Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative regrets
    for name, result in results.items():
        cumulative_regrets = np.cumsum(result['regrets'])
        ax2.plot(range(n_steps), cumulative_regrets, label=name, linewidth=2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Neural Bandit Comparison - Cumulative Regret')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running Neural Bandit Example")
    
    # Compare algorithms
    context_dim = 10
    n_arms = 5
    n_steps = 1000
    
    results = compare_neural_algorithms(context_dim, n_arms, n_steps)
    
    # Plot results
    plot_neural_comparison(results, n_steps)
    
    # Print summary
    print("\nNeural Bandit Comparison Summary:")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Cumulative Reward: {result['cumulative_reward']:.3f}")
        print(f"  Cumulative Regret: {result['cumulative_regret']:.3f}")
        print(f"  Average Reward: {np.mean(result['rewards']):.3f}")
        print() 
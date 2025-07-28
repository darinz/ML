"""
Contextual Bandit Framework

This module provides a framework for implementing contextual bandit algorithms.
It includes base classes and utilities for contextual bandit problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
from abc import ABC, abstractmethod


class ContextualBandit(ABC):
    """
    Abstract base class for contextual bandit algorithms.
    
    This class defines the interface that all contextual bandit algorithms
    should implement.
    """
    
    def __init__(self, n_arms: int, context_dim: int):
        """
        Initialize contextual bandit algorithm.
        
        Args:
            n_arms (int): Number of arms
            context_dim (int): Dimension of context vectors
        """
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.total_pulls = 0
        
        # History for analysis
        self.rewards_history = []
        self.actions_history = []
        self.contexts_history = []
    
    @abstractmethod
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select an arm based on the current context.
        
        Args:
            context (np.ndarray): Current context vector
            
        Returns:
            int: Index of selected arm
        """
        pass
    
    @abstractmethod
    def update(self, arm: int, reward: float, context: np.ndarray):
        """
        Update the algorithm with observed reward.
        
        Args:
            arm (int): Index of pulled arm
            reward (float): Observed reward
            context (np.ndarray): Context vector
        """
        pass
    
    def get_empirical_means(self, context: np.ndarray) -> np.ndarray:
        """
        Get empirical means for all arms given context.
        
        Args:
            context (np.ndarray): Context vector
            
        Returns:
            np.ndarray: Empirical means for each arm
        """
        # Default implementation - can be overridden
        return np.zeros(self.n_arms)
    
    def get_confidence_intervals(self, context: np.ndarray) -> List[Tuple[float, float]]:
        """
        Get confidence intervals for all arms given context.
        
        Args:
            context (np.ndarray): Context vector
            
        Returns:
            List[Tuple[float, float]]: Confidence intervals for each arm
        """
        # Default implementation - can be overridden
        return [(0.0, 1.0)] * self.n_arms


class ContextualUCB(ContextualBandit):
    """
    Contextual UCB algorithm.
    
    Extends UCB to handle contextual information by maintaining
    separate confidence intervals for each context.
    """
    
    def __init__(self, n_arms: int, context_dim: int, alpha: float = 2.0):
        """
        Initialize Contextual UCB.
        
        Args:
            n_arms (int): Number of arms
            context_dim (int): Dimension of context vectors
            alpha (float): Exploration parameter
        """
        super().__init__(n_arms, context_dim)
        self.alpha = alpha
        
        # Initialize statistics for each arm
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.context_sums = np.zeros((n_arms, context_dim))
        self.context_counts = np.zeros(n_arms, dtype=int)
    
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select arm using Contextual UCB.
        
        Args:
            context (np.ndarray): Current context vector
            
        Returns:
            int: Index of selected arm
        """
        ucb_values = []
        
        for arm in range(self.n_arms):
            if self.pulls[arm] == 0:
                # If arm hasn't been pulled, pull it
                ucb_values.append(float('inf'))
            else:
                # Calculate UCB value
                exploitation = self.empirical_means[arm]
                exploration = self.alpha * np.sqrt(np.log(self.total_pulls) / self.pulls[arm])
                
                # Add context-dependent term
                context_similarity = np.dot(context, self.context_sums[arm]) / max(self.context_counts[arm], 1)
                ucb_value = exploitation + exploration + context_similarity
                ucb_values.append(ucb_value)
        
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float, context: np.ndarray):
        """
        Update algorithm with observed reward.
        
        Args:
            arm (int): Index of pulled arm
            reward (float): Observed reward
            context (np.ndarray): Context vector
        """
        # Update empirical mean
        self.pulls[arm] += 1
        self.empirical_means[arm] = (
            (self.empirical_means[arm] * (self.pulls[arm] - 1) + reward) 
            / self.pulls[arm]
        )
        
        # Update context statistics
        self.context_sums[arm] += context
        self.context_counts[arm] += 1
        
        # Update total pulls
        self.total_pulls += 1
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm)
        self.contexts_history.append(context.copy())


class ContextualThompsonSampling(ContextualBandit):
    """
    Contextual Thompson Sampling algorithm.
    
    Extends Thompson sampling to handle contextual information
    by maintaining context-dependent posterior distributions.
    """
    
    def __init__(self, n_arms: int, context_dim: int, nu: float = 1.0):
        """
        Initialize Contextual Thompson Sampling.
        
        Args:
            n_arms (int): Number of arms
            context_dim (int): Dimension of context vectors
            nu (float): Noise parameter
        """
        super().__init__(n_arms, context_dim)
        self.nu = nu
        
        # Initialize posterior parameters for each arm
        self.A = [np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]
        self.mu = [np.zeros(context_dim) for _ in range(n_arms)]
        self.Sigma = [np.eye(context_dim) for _ in range(n_arms)]
    
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select arm using Contextual Thompson Sampling.
        
        Args:
            context (np.ndarray): Current context vector
            
        Returns:
            int: Index of selected arm
        """
        expected_rewards = []
        
        for arm in range(self.n_arms):
            # Sample from posterior for this arm
            theta_sample = np.random.multivariate_normal(self.mu[arm], self.Sigma[arm])
            
            # Calculate expected reward
            expected_reward = np.dot(theta_sample, context)
            expected_rewards.append(expected_reward)
        
        return np.argmax(expected_rewards)
    
    def update(self, arm: int, reward: float, context: np.ndarray):
        """
        Update algorithm with observed reward.
        
        Args:
            arm (int): Index of pulled arm
            reward (float): Observed reward
            context (np.ndarray): Context vector
        """
        # Update posterior parameters for the chosen arm
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Update posterior mean and covariance
        self.Sigma[arm] = np.linalg.inv(self.A[arm])
        self.mu[arm] = self.Sigma[arm] @ self.b[arm]
        
        # Update total pulls
        self.total_pulls += 1
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm)
        self.contexts_history.append(context.copy())


class NeuralContextualBandit(ContextualBandit):
    """
    Neural Contextual Bandit using neural networks.
    
    This implementation uses a simple neural network to model
    the reward function for contextual bandits.
    """
    
    def __init__(self, n_arms: int, context_dim: int, hidden_dim: int = 64):
        """
        Initialize Neural Contextual Bandit.
        
        Args:
            n_arms (int): Number of arms
            context_dim (int): Dimension of context vectors
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__(n_arms, context_dim)
        self.hidden_dim = hidden_dim
        
        # Initialize neural network parameters
        self.W1 = np.random.randn(context_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_arms) * 0.01
        self.b2 = np.zeros(n_arms)
        
        # Training parameters
        self.learning_rate = 0.01
        self.batch_size = 32
        
        # Experience replay buffer
        self.experience_buffer = []
        self.max_buffer_size = 1000
    
    def _forward(self, context: np.ndarray) -> np.ndarray:
        """
        Forward pass through the neural network.
        
        Args:
            context (np.ndarray): Input context
            
        Returns:
            np.ndarray: Output predictions
        """
        # Hidden layer
        hidden = np.tanh(context @ self.W1 + self.b1)
        
        # Output layer
        output = hidden @ self.W2 + self.b2
        
        return output
    
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select arm using neural network predictions.
        
        Args:
            context (np.ndarray): Current context vector
            
        Returns:
            int: Index of selected arm
        """
        predictions = self._forward(context)
        return np.argmax(predictions)
    
    def update(self, arm: int, reward: float, context: np.ndarray):
        """
        Update algorithm with observed reward.
        
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
        if len(self.experience_buffer) >= self.batch_size:
            self._train_network()
        
        # Update total pulls
        self.total_pulls += 1
        
        # Store history
        self.rewards_history.append(reward)
        self.actions_history.append(arm)
        self.contexts_history.append(context.copy())
    
    def _train_network(self):
        """Train the neural network using experience replay."""
        # Sample batch
        batch_size = min(self.batch_size, len(self.experience_buffer))
        batch_indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        
        contexts_batch = []
        arms_batch = []
        rewards_batch = []
        
        for idx in batch_indices:
            context, arm, reward = self.experience_buffer[idx]
            contexts_batch.append(context)
            arms_batch.append(arm)
            rewards_batch.append(reward)
        
        contexts_batch = np.array(contexts_batch)
        arms_batch = np.array(arms_batch)
        rewards_batch = np.array(rewards_batch)
        
        # Forward pass
        hidden = np.tanh(contexts_batch @ self.W1 + self.b1)
        predictions = hidden @ self.W2 + self.b2
        
        # Create target values
        targets = np.zeros_like(predictions)
        for i, arm in enumerate(arms_batch):
            targets[i, arm] = rewards_batch[i]
        
        # Backward pass (simplified gradient descent)
        error = predictions - targets
        
        # Update weights
        self.W2 -= self.learning_rate * hidden.T @ error
        self.b2 -= self.learning_rate * np.sum(error, axis=0)
        
        # Gradient for hidden layer
        hidden_error = error @ self.W2.T * (1 - np.tanh(contexts_batch @ self.W1 + self.b1) ** 2)
        
        self.W1 -= self.learning_rate * contexts_batch.T @ hidden_error
        self.b1 -= self.learning_rate * np.sum(hidden_error, axis=0)


def run_contextual_experiment(bandit: ContextualBandit, 
                            context_generator: Callable,
                            reward_function: Callable,
                            n_steps: int = 1000) -> Dict:
    """
    Run contextual bandit experiment.
    
    Args:
        bandit (ContextualBandit): Bandit algorithm
        context_generator (Callable): Function to generate contexts
        reward_function (Callable): Function to generate rewards
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
        
        # Calculate regret (assuming we know the optimal arm)
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


def compare_contextual_algorithms(n_arms: int, context_dim: int, 
                                n_steps: int = 1000) -> Dict:
    """
    Compare different contextual bandit algorithms.
    
    Args:
        n_arms (int): Number of arms
        context_dim (int): Context dimension
        n_steps (int): Number of time steps
        
    Returns:
        Dict: Comparison results
    """
    # Define context and reward generators
    def context_generator():
        return np.random.randn(context_dim)
    
    def reward_function(arm: int, context: np.ndarray):
        # Simple linear reward function
        arm_features = np.random.randn(context_dim)
        expected_reward = np.dot(context, arm_features)
        noise = np.random.normal(0, 0.1)
        return expected_reward + noise
    
    # Initialize algorithms
    algorithms = {
        'Contextual UCB': ContextualUCB(n_arms, context_dim),
        'Contextual Thompson Sampling': ContextualThompsonSampling(n_arms, context_dim),
        'Neural Contextual Bandit': NeuralContextualBandit(n_arms, context_dim)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"Running {name}")
        result = run_contextual_experiment(algorithm, context_generator, reward_function, n_steps)
        results[name] = result
    
    return results


def plot_contextual_comparison(results: Dict, n_steps: int):
    """
    Plot comparison of contextual bandit algorithms.
    
    Args:
        results (Dict): Results from compare_contextual_algorithms
        n_steps (int): Number of time steps
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cumulative rewards
    for name, result in results.items():
        cumulative_rewards = np.cumsum(result['rewards'])
        ax1.plot(range(n_steps), cumulative_rewards, label=name, linewidth=2)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Contextual Bandit Comparison - Cumulative Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative regrets
    for name, result in results.items():
        cumulative_regrets = np.cumsum(result['regrets'])
        ax2.plot(range(n_steps), cumulative_regrets, label=name, linewidth=2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Contextual Bandit Comparison - Cumulative Regret')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running Contextual Bandit Example")
    
    # Compare algorithms
    n_arms = 5
    context_dim = 10
    n_steps = 1000
    
    results = compare_contextual_algorithms(n_arms, context_dim, n_steps)
    
    # Plot results
    plot_contextual_comparison(results, n_steps)
    
    # Print summary
    print("\nContextual Bandit Comparison Summary:")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Cumulative Reward: {result['cumulative_reward']:.3f}")
        print(f"  Cumulative Regret: {result['cumulative_regret']:.3f}")
        print(f"  Average Reward: {np.mean(result['rewards']):.3f}")
        print() 
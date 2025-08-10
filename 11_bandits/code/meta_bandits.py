"""
Meta-Learning Bandit Algorithms

This module implements meta-learning approaches for bandit algorithms.
These algorithms learn to adapt quickly to new bandit problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim


class MetaBandit(nn.Module):
    """
    Meta-learning bandit algorithm.
    
    This implementation uses Model-Agnostic Meta-Learning (MAML)
    to learn initialization parameters that enable fast adaptation
    to new bandit problems.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 n_arms: int = 1, meta_lr: float = 0.01, task_lr: float = 0.001):
        """
        Initialize meta bandit.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            n_arms (int): Number of arms
            meta_lr (float): Meta-learning rate
            task_lr (float): Task-specific learning rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_arms = n_arms
        self.meta_lr = meta_lr
        self.task_lr = task_lr
        
        # Neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_arms)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.parameters(), lr=meta_lr)
        self.criterion = nn.MSELoss()
        
        # Task-specific parameters
        self.task_params = None
        self.task_optimizer = None
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
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
    
    def adapt_to_task(self, task_data: List[Tuple], n_adaptation_steps: int = 5):
        """
        Adapt to a new task using MAML.
        
        Args:
            task_data (List[Tuple]): Task-specific data (context, arm, reward)
            n_adaptation_steps (int): Number of adaptation steps
        """
        # Initialize task-specific parameters
        self.task_params = {name: param.clone() for name, param in self.named_parameters()}
        self.task_optimizer = optim.SGD(self.task_params.values(), lr=self.task_lr)
        
        # Adaptation steps
        for _ in range(n_adaptation_steps):
            # Sample batch from task data
            if len(task_data) > 0:
                batch = np.random.choice(task_data, min(8, len(task_data)), replace=False)
                
                contexts = []
                arms = []
                rewards = []
                
                for context, arm, reward in batch:
                    contexts.append(context)
                    arms.append(arm)
                    rewards.append(reward)
                
                # Convert to tensors
                contexts_tensor = torch.FloatTensor(contexts)
                arms_tensor = torch.LongTensor(arms)
                rewards_tensor = torch.FloatTensor(rewards)
                
                # Forward pass
                predictions = self.forward(contexts_tensor)
                
                # Create targets
                targets = torch.zeros_like(predictions)
                targets[torch.arange(len(arms)), arms_tensor] = rewards_tensor
                
                # Compute loss and update task parameters
                loss = self.criterion(predictions, targets)
                
                self.task_optimizer.zero_grad()
                loss.backward()
                self.task_optimizer.step()
    
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select arm using adapted parameters.
        
        Args:
            context (np.ndarray): Context vector
            
        Returns:
            int: Index of selected arm
        """
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        
        # Use task-specific parameters if available
        if self.task_params is not None:
            # Temporarily set parameters
            original_params = {}
            for name, param in self.named_parameters():
                original_params[name] = param.data.clone()
                param.data = self.task_params[name].data.clone()
            
            # Forward pass
            predictions = self.forward(context_tensor)
            
            # Restore original parameters
            for name, param in self.named_parameters():
                param.data = original_params[name]
        else:
            # Use original parameters
            predictions = self.forward(context_tensor)
        
        return torch.argmax(predictions).item()
    
    def meta_update(self, tasks: List[List[Tuple]]):
        """
        Perform meta-update using multiple tasks.
        
        Args:
            tasks (List[List[Tuple]]): List of tasks, each containing (context, arm, reward) tuples
        """
        meta_loss = 0
        
        for task_data in tasks:
            # Adapt to task
            self.adapt_to_task(task_data)
            
            # Evaluate on task
            if len(task_data) > 0:
                batch = np.random.choice(task_data, min(8, len(task_data)), replace=False)
                
                contexts = []
                arms = []
                rewards = []
                
                for context, arm, reward in batch:
                    contexts.append(context)
                    arms.append(arm)
                    rewards.append(reward)
                
                # Convert to tensors
                contexts_tensor = torch.FloatTensor(contexts)
                arms_tensor = torch.LongTensor(arms)
                rewards_tensor = torch.FloatTensor(rewards)
                
                # Forward pass with task parameters
                original_params = {}
                for name, param in self.named_parameters():
                    original_params[name] = param.data.clone()
                    param.data = self.task_params[name].data.clone()
                
                predictions = self.forward(contexts_tensor)
                
                # Restore original parameters
                for name, param in self.named_parameters():
                    param.data = original_params[name]
                
                # Create targets
                targets = torch.zeros_like(predictions)
                targets[torch.arange(len(arms)), arms_tensor] = rewards_tensor
                
                # Compute loss
                loss = self.criterion(predictions, targets)
                meta_loss += loss
        
        # Meta-update
        if len(tasks) > 0:
            meta_loss /= len(tasks)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()


class FastAdaptiveBandit(nn.Module):
    """
    Fast adaptive bandit using gradient-based meta-learning.
    
    This implementation uses Reptile algorithm for fast adaptation
    to new bandit problems.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 n_arms: int = 1, meta_lr: float = 0.01, task_lr: float = 0.001):
        """
        Initialize fast adaptive bandit.
        
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            n_arms (int): Number of arms
            meta_lr (float): Meta-learning rate
            task_lr (float): Task-specific learning rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_arms = n_arms
        self.meta_lr = meta_lr
        self.task_lr = task_lr
        
        # Neural network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_arms)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(self.parameters(), lr=meta_lr)
        self.criterion = nn.MSELoss()
        
        # Task-specific parameters
        self.task_params = None
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
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
    
    def adapt_to_task(self, task_data: List[Tuple], n_adaptation_steps: int = 3):
        """
        Adapt to a new task using Reptile.
        
        Args:
            task_data (List[Tuple]): Task-specific data
            n_adaptation_steps (int): Number of adaptation steps
        """
        # Initialize task-specific parameters
        self.task_params = {name: param.clone() for name, param in self.named_parameters()}
        
        # Adaptation steps
        for _ in range(n_adaptation_steps):
            if len(task_data) > 0:
                batch = np.random.choice(task_data, min(8, len(task_data)), replace=False)
                
                contexts = []
                arms = []
                rewards = []
                
                for context, arm, reward in batch:
                    contexts.append(context)
                    arms.append(arm)
                    rewards.append(reward)
                
                # Convert to tensors
                contexts_tensor = torch.FloatTensor(contexts)
                arms_tensor = torch.LongTensor(arms)
                rewards_tensor = torch.FloatTensor(rewards)
                
                # Forward pass
                predictions = self.forward(contexts_tensor)
                
                # Create targets
                targets = torch.zeros_like(predictions)
                targets[torch.arange(len(arms)), arms_tensor] = rewards_tensor
                
                # Compute loss
                loss = self.criterion(predictions, targets)
                
                # Update task parameters
                gradients = torch.autograd.grad(loss, self.parameters())
                
                for param, grad in zip(self.task_params.values(), gradients):
                    param.data -= self.task_lr * grad.data
    
    def select_arm(self, context: np.ndarray) -> int:
        """
        Select arm using adapted parameters.
        
        Args:
            context (np.ndarray): Context vector
            
        Returns:
            int: Index of selected arm
        """
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        
        # Use task-specific parameters if available
        if self.task_params is not None:
            # Temporarily set parameters
            original_params = {}
            for name, param in self.named_parameters():
                original_params[name] = param.data.clone()
                param.data = self.task_params[name].data.clone()
            
            # Forward pass
            predictions = self.forward(context_tensor)
            
            # Restore original parameters
            for name, param in self.named_parameters():
                param.data = original_params[name]
        else:
            # Use original parameters
            predictions = self.forward(context_tensor)
        
        return torch.argmax(predictions).item()
    
    def reptile_update(self, tasks: List[List[Tuple]]):
        """
        Perform Reptile meta-update.
        
        Args:
            tasks (List[List[Tuple]]): List of tasks
        """
        for task_data in tasks:
            # Adapt to task
            self.adapt_to_task(task_data)
            
            # Reptile update: move towards task parameters
            for param, task_param in zip(self.parameters(), self.task_params.values()):
                param.data += self.meta_lr * (task_param.data - param.data)


def generate_meta_tasks(n_tasks: int, n_samples_per_task: int, 
                       context_dim: int, n_arms: int) -> List[List[Tuple]]:
    """
    Generate meta-learning tasks.
    
    Args:
        n_tasks (int): Number of tasks
        n_samples_per_task (int): Number of samples per task
        context_dim (int): Context dimension
        n_arms (int): Number of arms
        
    Returns:
        List[List[Tuple]]: List of tasks
    """
    tasks = []
    
    for task_idx in range(n_tasks):
        task_data = []
        
        # Generate task-specific reward function
        task_weights = np.random.randn(n_arms, context_dim)
        
        for _ in range(n_samples_per_task):
            context = np.random.randn(context_dim)
            arm = np.random.randint(0, n_arms)
            
            # Generate reward based on task-specific weights
            expected_reward = np.tanh(np.dot(task_weights[arm], context))
            noise = np.random.normal(0, 0.1)
            reward = expected_reward + noise
            
            task_data.append((context, arm, reward))
        
        tasks.append(task_data)
    
    return tasks


def run_meta_bandit_experiment(bandit: MetaBandit, 
                             task_data: List[Tuple],
                             n_steps: int = 100) -> Dict:
    """
    Run meta bandit experiment.
    
    Args:
        bandit (MetaBandit): Meta bandit algorithm
        task_data (List[Tuple]): Task-specific data
        n_steps (int): Number of time steps
        
    Returns:
        Dict: Experiment results
    """
    # Adapt to task
    bandit.adapt_to_task(task_data)
    
    rewards = []
    regrets = []
    actions = []
    contexts = []
    
    for step in range(n_steps):
        # Generate context
        context = np.random.randn(bandit.input_dim)
        
        # Select arm
        arm = bandit.select_arm(context)
        
        # Generate reward (using task-specific weights)
        task_weights = np.random.randn(bandit.n_arms, bandit.input_dim)
        expected_reward = np.tanh(np.dot(task_weights[arm], context))
        noise = np.random.normal(0, 0.1)
        reward = expected_reward + noise
        
        # Store results
        rewards.append(reward)
        actions.append(arm)
        contexts.append(context)
        
        # Calculate regret
        optimal_arm = np.argmax([np.tanh(np.dot(task_weights[i], context)) for i in range(bandit.n_arms)])
        optimal_reward = np.tanh(np.dot(task_weights[optimal_arm], context))
        regret = optimal_reward - expected_reward
        regrets.append(regret)
    
    return {
        'rewards': rewards,
        'regrets': regrets,
        'actions': actions,
        'contexts': contexts,
        'cumulative_reward': np.sum(rewards),
        'cumulative_regret': np.sum(regrets)
    }


def compare_meta_algorithms(context_dim: int, n_arms: int, 
                          n_tasks: int = 10, n_steps: int = 100) -> Dict:
    """
    Compare different meta-learning bandit algorithms.
    
    Args:
        context_dim (int): Context dimension
        n_arms (int): Number of arms
        n_tasks (int): Number of tasks
        n_steps (int): Number of steps per task
        
    Returns:
        Dict: Comparison results
    """
    # Generate meta-learning tasks
    tasks = generate_meta_tasks(n_tasks, 50, context_dim, n_arms)
    
    # Initialize algorithms
    algorithms = {
        'Meta Bandit (MAML)': MetaBandit(context_dim, n_arms=n_arms),
        'Fast Adaptive Bandit (Reptile)': FastAdaptiveBandit(context_dim, n_arms=n_arms)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"Running {name}")
        
        # Meta-training
        if name == 'Meta Bandit (MAML)':
            algorithm.meta_update(tasks)
        else:
            algorithm.reptile_update(tasks)
        
        # Test on new task
        test_task = generate_meta_tasks(1, 50, context_dim, n_arms)[0]
        result = run_meta_bandit_experiment(algorithm, test_task, n_steps)
        results[name] = result
    
    return results


def plot_meta_comparison(results: Dict, n_steps: int):
    """
    Plot comparison of meta-learning bandit algorithms.
    
    Args:
        results (Dict): Results from compare_meta_algorithms
        n_steps (int): Number of time steps
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot cumulative rewards
    for name, result in results.items():
        cumulative_rewards = np.cumsum(result['rewards'])
        ax1.plot(range(n_steps), cumulative_rewards, label=name, linewidth=2)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Meta-Learning Bandit Comparison - Cumulative Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Plot cumulative regrets
    for name, result in results.items():
        cumulative_regrets = np.cumsum(result['regrets'])
        ax2.plot(range(n_steps), cumulative_regrets, label=name, linewidth=2)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Meta-Learning Bandit Comparison - Cumulative Regret')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Running Meta-Learning Bandit Example")
    
    # Compare algorithms
    context_dim = 10
    n_arms = 5
    n_tasks = 10
    n_steps = 100
    
    results = compare_meta_algorithms(context_dim, n_arms, n_tasks, n_steps)
    
    # Plot results
    plot_meta_comparison(results, n_steps)
    
    # Print summary
    print("\nMeta-Learning Bandit Comparison Summary:")
    for name, result in results.items():
        print(f"{name}:")
        print(f"  Cumulative Reward: {result['cumulative_reward']:.3f}")
        print(f"  Cumulative Regret: {result['cumulative_regret']:.3f}")
        print(f"  Average Reward: {np.mean(result['rewards']):.3f}")
        print() 
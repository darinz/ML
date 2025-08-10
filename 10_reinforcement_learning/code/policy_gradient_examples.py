"""
Policy Gradient Methods Implementation and Examples

This file implements the core concepts from 04_policy_gradient.md:

1. REINFORCE Algorithm: Monte Carlo policy gradient for finite-horizon MDPs
2. Policy Gradient Theorem: ∇_θ J(θ) = E[∇_θ log π_θ(a|s) Q^π(s,a)]
3. Baseline Methods: Reducing variance with state-dependent baselines
4. Actor-Critic Methods: Combining policy gradients with value function approximation
5. Natural Policy Gradients: Using Fisher information matrix for better updates
6. Trust Region Methods: Constraining policy updates for stability

Key Concepts Demonstrated:
- Policy gradient theorem: ∇_θ J(θ) = E[∇_θ log π_θ(a|s) A^π(s,a)]
- REINFORCE: θ ← θ + α ∇_θ log π_θ(a|s) G_t
- Baseline reduction: θ ← θ + α ∇_θ log π_θ(a|s) (G_t - b(s_t))
- Actor-critic: θ ← θ + α ∇_θ log π_θ(a|s) δ_t
- Natural gradients: θ ← θ + α F^(-1) ∇_θ J(θ)

"""

import numpy as np
from collections import namedtuple, defaultdict
from typing import Callable, List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass

# For baseline regression
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
except ImportError:
    LinearRegression = None
    PolynomialFeatures = None

# Define data structures
Trajectory = namedtuple('Trajectory', ['states', 'actions', 'rewards', 'next_states', 'dones'])

@dataclass
class PolicyGradientResult:
    """Result of policy gradient training."""
    policy_params: np.ndarray
    training_returns: List[float]
    final_policy: object
    converged: bool

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities.
    
    This implements the softmax function:
    softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
    
    Args:
        x: Input array
        
    Returns:
        Probability distribution
    """
    # Numerical stability: subtract max before exponentiating
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class CategoricalPolicy:
    """
    Categorical (discrete) policy using softmax parameterization.
    
    This implements π_θ(a|s) = softmax(θ_s)_a where θ_s are the logits
    for state s. The policy is parameterized by a matrix θ ∈ ℝ^{|S| × |A|}.
    
    Key properties:
    - ∇_θ log π_θ(a|s) = e_a - π_θ(a|s) (one-hot minus probabilities)
    - Supports exploration through stochastic action selection
    - Easy to compute gradients and sample actions
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1):
        """
        Initialize categorical policy.
        
        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: Learning rate for gradient updates
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        
        # Initialize policy parameters (logits)
        self.theta = np.zeros((n_states, n_actions))
        
        # Track training statistics
        self.training_history = []
    
    def action_probs(self, s: int) -> np.ndarray:
        """
        Get action probabilities for state s.
        
        Args:
            s: State index
            
        Returns:
            Action probabilities π_θ(a|s)
        """
        logits = self.theta[s]
        return softmax(logits)
    
    def sample_action(self, s: int) -> int:
        """
        Sample action from policy for state s.
        
        Args:
            s: State index
            
        Returns:
            Sampled action index
        """
        probs = self.action_probs(s)
        return np.random.choice(self.n_actions, p=probs)
    
    def log_prob(self, s: int, a: int) -> float:
        """
        Compute log probability of action a in state s.
        
        Args:
            s: State index
            a: Action index
            
        Returns:
            log π_θ(a|s)
        """
        probs = self.action_probs(s)
        return np.log(probs[a] + 1e-8)  # Add small epsilon for numerical stability
    
    def grad_log_prob(self, s: int, a: int) -> np.ndarray:
        """
        Compute gradient of log probability with respect to θ.
        
        This implements the policy gradient:
        ∇_θ log π_θ(a|s) = e_a - π_θ(a|s)
        
        Args:
            s: State index
            a: Action index
            
        Returns:
            Gradient vector of shape (n_actions,)
        """
        probs = self.action_probs(s)
        grad = -probs.copy()  # Start with negative probabilities
        grad[a] += 1.0        # Add 1 for the taken action
        return grad
    
    def update(self, s: int, a: int, advantage: float):
        """
        Update policy parameters using policy gradient.
        
        Args:
            s: State index
            a: Action index
            advantage: Advantage estimate A^π(s,a)
        """
        grad = self.grad_log_prob(s, a)
        self.theta[s] += self.learning_rate * advantage * grad
    
    def get_entropy(self, s: int) -> float:
        """
        Compute policy entropy for state s.
        
        Args:
            s: State index
            
        Returns:
            Entropy H(π_θ(·|s))
        """
        probs = self.action_probs(s)
        return -np.sum(probs * np.log(probs + 1e-8))

class GaussianPolicy:
    """
    Gaussian (continuous) policy for continuous action spaces.
    
    This implements π_θ(a|s) = N(a; μ_θ(s), σ_θ(s)) where the mean and
    standard deviation are parameterized functions of the state.
    
    For simplicity, we use a linear parameterization:
    μ_θ(s) = θ_μ^T φ(s)
    σ_θ(s) = exp(θ_σ^T φ(s)) (ensures positivity)
    """
    
    def __init__(self, state_dim: int, action_dim: int, feature_fn: Callable,
                 learning_rate: float = 0.01):
        """
        Initialize Gaussian policy.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            feature_fn: Feature function φ(s)
            learning_rate: Learning rate
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_fn = feature_fn
        self.learning_rate = learning_rate
        
        # Feature dimension
        self.feature_dim = len(feature_fn(np.zeros(state_dim)))
        
        # Policy parameters
        self.theta_mu = np.zeros((self.feature_dim, action_dim))  # Mean parameters
        self.theta_sigma = np.zeros((self.feature_dim, action_dim))  # Log-std parameters
        
        # Initialize log-std to small values
        self.theta_sigma.fill(-1.0)
    
    def get_params(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mean and standard deviation for state s.
        
        Args:
            s: State vector
            
        Returns:
            (mean, std) parameters
        """
        phi = self.feature_fn(s)
        mu = self.theta_mu.T @ phi
        sigma = np.exp(self.theta_sigma.T @ phi)
        return mu, sigma
    
    def sample_action(self, s: np.ndarray) -> np.ndarray:
        """
        Sample action from policy.
        
        Args:
            s: State vector
            
        Returns:
            Sampled action
        """
        mu, sigma = self.get_params(s)
        return np.random.normal(mu, sigma)
    
    def log_prob(self, s: np.ndarray, a: np.ndarray) -> float:
        """
        Compute log probability of action.
        
        Args:
            s: State vector
            a: Action vector
            
        Returns:
            log π_θ(a|s)
        """
        mu, sigma = self.get_params(s)
        log_prob = -0.5 * np.sum(((a - mu) / sigma) ** 2) - np.sum(np.log(sigma)) - 0.5 * len(a) * np.log(2 * np.pi)
        return log_prob
    
    def grad_log_prob(self, s: np.ndarray, a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradients of log probability.
        
        Args:
            s: State vector
            a: Action vector
            
        Returns:
            (grad_mu, grad_sigma) gradients
        """
        phi = self.feature_fn(s)
        mu, sigma = self.get_params(s)
        
        # Gradients
        grad_mu = phi.reshape(-1, 1) * ((a - mu) / sigma ** 2).reshape(1, -1)
        grad_sigma = phi.reshape(-1, 1) * (((a - mu) ** 2 / sigma ** 2) - 1).reshape(1, -1)
        
        return grad_mu, grad_sigma
    
    def update(self, s: np.ndarray, a: np.ndarray, advantage: float):
        """
        Update policy parameters.
        
        Args:
            s: State vector
            a: Action vector
            advantage: Advantage estimate
        """
        grad_mu, grad_sigma = self.grad_log_prob(s, a)
        
        self.theta_mu += self.learning_rate * advantage * grad_mu
        self.theta_sigma += self.learning_rate * advantage * grad_sigma

# ----------------------
# REINFORCE Algorithm
# ----------------------

def compute_returns(rewards: List[float], gamma: float = 1.0) -> List[float]:
    """
    Compute discounted returns for a trajectory.
    
    This implements: G_t = Σ_{k=t}^{T-1} γ^{k-t} r_k
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        List of returns G_t for each timestep
    """
    returns = []
    G = 0.0
    
    # Compute returns backwards
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    return returns

def reinforce(env, policy: CategoricalPolicy, gamma: float = 1.0, 
             n_episodes: int = 100, baseline: Optional[Callable] = None) -> PolicyGradientResult:
    """
    REINFORCE algorithm for finite-horizon MDPs.
    
    This implements the vanilla policy gradient algorithm:
    1. Collect trajectory τ = (s_0, a_0, r_0, ..., s_T, a_T, r_T)
    2. Compute returns G_t for each timestep
    3. Update policy: θ ← θ + α Σ_t ∇_θ log π_θ(a_t|s_t) G_t
    
    With baseline (optional):
    θ ← θ + α Σ_t ∇_θ log π_θ(a_t|s_t) (G_t - b(s_t))
    
    Args:
        env: Environment with reset() and step(a) methods
        policy: CategoricalPolicy instance
        gamma: Discount factor
        n_episodes: Number of training episodes
        baseline: Optional baseline function b(s)
        
    Returns:
        PolicyGradientResult with training history
    """
    training_returns = []
    
    for episode in range(n_episodes):
        # Collect trajectory
        s = env.reset()
        states, actions, rewards = [], [], []
        done = False
        
        while not done:
            a = policy.sample_action(s)
            s_next, r, done, _ = env.step(a)
            
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_next
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        episode_return = sum(rewards)
        training_returns.append(episode_return)
        
        # Policy update
        for t in range(len(states)):
            s_t, a_t = states[t], actions[t]
            G_t = returns[t]
            
            # Apply baseline if provided
            if baseline is not None:
                advantage = G_t - baseline(s_t)
            else:
                advantage = G_t
            
            # Update policy
            policy.update(s_t, a_t, advantage)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_return = np.mean(training_returns[-10:])
            print(f"Episode {episode + 1}: Average return = {avg_return:.3f}")
    
    return PolicyGradientResult(
        policy_params=policy.theta.copy(),
        training_returns=training_returns,
        final_policy=policy,
        converged=True
    )

def reinforce_with_baseline(env, policy: CategoricalPolicy, gamma: float = 1.0,
                          n_episodes: int = 100, baseline_type: str = 'mean') -> PolicyGradientResult:
    """
    REINFORCE with baseline for variance reduction.
    
    Baseline options:
    - 'mean': Use mean return as baseline
    - 'state': Fit state-dependent baseline using regression
    - 'value': Use learned value function as baseline
    
    Args:
        env: Environment
        policy: Policy to train
        gamma: Discount factor
        n_episodes: Number of episodes
        baseline_type: Type of baseline to use
        
    Returns:
        PolicyGradientResult
    """
    training_returns = []
    baseline_fn = None
    
    if baseline_type == 'state' and LinearRegression is not None:
        # Fit state-dependent baseline
        baseline_model = LinearRegression()
        state_features = []
        return_targets = []
    
    for episode in range(n_episodes):
        # Collect trajectory
        s = env.reset()
        states, actions, rewards = [], [], []
        done = False
        
        while not done:
            a = policy.sample_action(s)
            s_next, r, done, _ = env.step(a)
            
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_next
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        episode_return = sum(rewards)
        training_returns.append(episode_return)
        
        # Update baseline
        if baseline_type == 'mean':
            baseline_fn = lambda s: np.mean(training_returns)
        elif baseline_type == 'state' and LinearRegression is not None:
            # Add to training data
            for s_t, G_t in zip(states, returns):
                state_features.append([s_t])  # Simple 1D state
                return_targets.append(G_t)
            
            # Fit baseline every 10 episodes
            if episode % 10 == 0 and len(state_features) > 0:
                X = np.array(state_features)
                y = np.array(return_targets)
                baseline_model.fit(X, y)
                baseline_fn = lambda s: baseline_model.predict([[s]])[0]
        
        # Policy update
        for t in range(len(states)):
            s_t, a_t = states[t], actions[t]
            G_t = returns[t]
            
            # Compute advantage
            if baseline_fn is not None:
                advantage = G_t - baseline_fn(s_t)
            else:
                advantage = G_t
            
            # Update policy
            policy.update(s_t, a_t, advantage)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_return = np.mean(training_returns[-10:])
            print(f"Episode {episode + 1}: Average return = {avg_return:.3f}")
    
    return PolicyGradientResult(
        policy_params=policy.theta.copy(),
        training_returns=training_returns,
        final_policy=policy,
        converged=True
    )

# ----------------------
# Actor-Critic Methods
# ----------------------

class ValueFunction:
    """
    Simple value function approximator.
    
    This implements V_θ(s) ≈ V^π(s) using linear function approximation.
    """
    
    def __init__(self, n_states: int, learning_rate: float = 0.1):
        """
        Initialize value function.
        
        Args:
            n_states: Number of states
            learning_rate: Learning rate
        """
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.theta = np.zeros(n_states)
    
    def __call__(self, s: int) -> float:
        """Get value estimate for state s."""
        return self.theta[s]
    
    def update(self, s: int, target: float):
        """
        Update value function using TD learning.
        
        Args:
            s: State index
            target: Target value
        """
        current_value = self.theta[s]
        td_error = target - current_value
        self.theta[s] += self.learning_rate * td_error
        return td_error

def actor_critic(env, policy: CategoricalPolicy, value_fn: ValueFunction,
                gamma: float = 0.99, n_episodes: int = 100) -> PolicyGradientResult:
    """
    Actor-critic algorithm.
    
    This combines policy gradients with value function approximation:
    1. Use value function to estimate advantages: A^π(s,a) ≈ r + γV(s') - V(s)
    2. Update policy using advantage estimates
    3. Update value function using TD learning
    
    Args:
        env: Environment
        policy: Policy to train
        value_fn: Value function approximator
        gamma: Discount factor
        n_episodes: Number of episodes
        
    Returns:
        PolicyGradientResult
    """
    training_returns = []
    
    for episode in range(n_episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        next_states = []
        done = False
        
        # Collect trajectory
        while not done:
            a = policy.sample_action(s)
            s_next, r, done, _ = env.step(a)
            
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_next)
            s = s_next
        
        episode_return = sum(rewards)
        training_returns.append(episode_return)
        
        # Update policy and value function
        for t in range(len(states)):
            s_t, a_t, r_t = states[t], actions[t], rewards[t]
            
            if t == len(states) - 1:
                # Terminal state
                s_next_t = s_t  # No next state
                target = r_t
            else:
                s_next_t = next_states[t]
                target = r_t + gamma * value_fn(s_next_t)
            
            # Compute advantage
            advantage = target - value_fn(s_t)
            
            # Update policy (actor)
            policy.update(s_t, a_t, advantage)
            
            # Update value function (critic)
            value_fn.update(s_t, target)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_return = np.mean(training_returns[-10:])
            print(f"Episode {episode + 1}: Average return = {avg_return:.3f}")
    
    return PolicyGradientResult(
        policy_params=policy.theta.copy(),
        training_returns=training_returns,
        final_policy=policy,
        converged=True
    )

# ----------------------
# Natural Policy Gradients
# ----------------------

def compute_fisher_information(policy: CategoricalPolicy, states: List[int]) -> np.ndarray:
    """
    Compute Fisher information matrix for policy.
    
    The Fisher information matrix is:
    F = E[∇_θ log π_θ(a|s) ∇_θ log π_θ(a|s)^T]
    
    Args:
        policy: Policy
        states: List of states
        
    Returns:
        Fisher information matrix
    """
    n_params = policy.n_states * policy.n_actions
    F = np.zeros((n_params, n_params))
    
    for s in states:
        probs = policy.action_probs(s)
        for a in range(policy.n_actions):
            grad = policy.grad_log_prob(s, a)
            grad_flat = grad.reshape(-1)
            F += probs[a] * np.outer(grad_flat, grad_flat)
    
    return F

def natural_policy_gradient(policy: CategoricalPolicy, states: List[int],
                          actions: List[int], advantages: List[float],
                          reg: float = 1e-3) -> np.ndarray:
    """
    Compute natural policy gradient.
    
    The natural gradient is: F^(-1) ∇_θ J(θ)
    where F is the Fisher information matrix.
    
    Args:
        policy: Policy
        states: List of states
        actions: List of actions
        advantages: List of advantages
        reg: Regularization parameter
        
    Returns:
        Natural gradient update
    """
    # Compute Fisher information matrix
    F = compute_fisher_information(policy, states)
    
    # Add regularization
    F += reg * np.eye(F.shape[0])
    
    # Compute vanilla gradient
    vanilla_grad = np.zeros_like(policy.theta)
    for s, a, adv in zip(states, actions, advantages):
        grad = policy.grad_log_prob(s, a)
        vanilla_grad[s] += adv * grad
    
    # Compute natural gradient
    vanilla_grad_flat = vanilla_grad.reshape(-1)
    natural_grad_flat = np.linalg.solve(F, vanilla_grad_flat)
    natural_grad = natural_grad_flat.reshape(policy.theta.shape)
    
    return natural_grad

# ----------------------
# Example Environments
# ----------------------

class SimpleMDP:
    """
    Simple MDP environment for demonstration.
    
    States: 0, 1
    Actions: 0, 1
    Transitions: Deterministic
    Rewards: State-dependent
    """
    
    def __init__(self, max_steps: int = 10):
        self.n_states = 2
        self.n_actions = 2
        self.max_steps = max_steps
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.s = 0
        self.steps = 0
        return self.s
    
    def step(self, a):
        """
        Take action a in current state.
        
        Returns:
            (next_state, reward, done, info)
        """
        if self.s == 0:
            if a == 0:
                s_next, r = 0, 0
            else:
                s_next, r = 1, 1
        else:  # s == 1
            if a == 0:
                s_next, r = 0, 1
            else:
                s_next, r = 1, 0
        
        self.s = s_next
        self.steps += 1
        done = self.steps >= self.max_steps
        
        return self.s, r, done, {}

class ContinuousCartPole:
    """
    Continuous version of CartPole for demonstration.
    
    State: [x, x_dot, theta, theta_dot]
    Action: Continuous force in [-1, 1]
    """
    
    def __init__(self, max_steps: int = 200):
        self.state_dim = 4
        self.action_dim = 1
        self.max_steps = max_steps
        self.dt = 0.02
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.reset()
    
    def reset(self):
        """Reset to initial state."""
        self.state = np.array([0.0, 0.0, 0.0, 0.0])  # Start upright
        self.steps = 0
        return self.state.copy()
    
    def step(self, action):
        """
        Take action in environment.
        
        Args:
            action: Continuous action in [-1, 1]
            
        Returns:
            (next_state, reward, done, info)
        """
        # Clip action
        action = np.clip(action, -1.0, 1.0)
        
        # Extract state variables
        x, x_dot, theta, theta_dot = self.state
        
        # Apply force
        force = action * 10.0  # Scale action to reasonable force
        
        # Compute acceleration
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Update state
        x_dot += xacc * self.dt
        x += x_dot * self.dt
        theta_dot += thetaacc * self.dt
        theta += theta_dot * self.dt
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        # Compute reward
        reward = 1.0  # Reward for staying alive
        
        # Check termination
        done = False
        if abs(x) > 2.4 or abs(theta) > 0.2 or self.steps >= self.max_steps:
            done = True
            reward = 0.0
        
        return self.state.copy(), reward, done, {}

# ----------------------
# Demonstrations
# ----------------------

def demonstrate_reinforce():
    """Demonstrate REINFORCE algorithm."""
    print("=" * 60)
    print("REINFORCE ALGORITHM DEMONSTRATION")
    print("=" * 60)
    
    # Create environment and policy
    env = SimpleMDP(max_steps=10)
    policy = CategoricalPolicy(n_states=2, n_actions=2, learning_rate=0.1)
    
    # Train with REINFORCE
    result = reinforce(env, policy, gamma=0.99, n_episodes=200)
    
    print("Training Results:")
    print("-" * 40)
    print(f"Final average return: {np.mean(result.training_returns[-20:]):.3f}")
    print(f"Best return: {np.max(result.training_returns):.3f}")
    
    # Show learned policy
    print("\nLearned Policy:")
    print("-" * 40)
    for s in range(2):
        probs = policy.action_probs(s)
        print(f"State {s}: π(a=0)={probs[0]:.3f}, π(a=1)={probs[1]:.3f}")
    
    # Plot training curve
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(result.training_returns, alpha=0.6, label='Episode Returns')
        
        # Plot moving average
        window = 20
        moving_avg = np.convolve(result.training_returns, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(result.training_returns)), moving_avg, 
                'r-', linewidth=2, label=f'{window}-Episode Moving Average')
        
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('REINFORCE Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot")

def demonstrate_baseline_comparison():
    """Compare REINFORCE with and without baseline."""
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON DEMONSTRATION")
    print("=" * 60)
    
    env = SimpleMDP(max_steps=10)
    
    # Train without baseline
    policy_no_baseline = CategoricalPolicy(n_states=2, n_actions=2, learning_rate=0.1)
    result_no_baseline = reinforce(env, policy_no_baseline, gamma=0.99, n_episodes=100)
    
    # Train with baseline
    policy_with_baseline = CategoricalPolicy(n_states=2, n_actions=2, learning_rate=0.1)
    result_with_baseline = reinforce_with_baseline(env, policy_with_baseline, 
                                                 gamma=0.99, n_episodes=100, 
                                                 baseline_type='mean')
    
    print("Comparison Results:")
    print("-" * 40)
    print(f"Without baseline - Final avg return: {np.mean(result_no_baseline.training_returns[-20:]):.3f}")
    print(f"With baseline - Final avg return: {np.mean(result_with_baseline.training_returns[-20:]):.3f}")
    
    # Plot comparison
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot individual returns
        plt.subplot(2, 1, 1)
        plt.plot(result_no_baseline.training_returns, alpha=0.6, label='No Baseline')
        plt.plot(result_with_baseline.training_returns, alpha=0.6, label='With Baseline')
        plt.ylabel('Episode Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot moving averages
        plt.subplot(2, 1, 2)
        window = 10
        avg_no_baseline = np.convolve(result_no_baseline.training_returns, 
                                     np.ones(window)/window, mode='valid')
        avg_with_baseline = np.convolve(result_with_baseline.training_returns, 
                                       np.ones(window)/window, mode='valid')
        
        plt.plot(range(window-1, len(result_no_baseline.training_returns)), 
                avg_no_baseline, 'b-', linewidth=2, label='No Baseline')
        plt.plot(range(window-1, len(result_with_baseline.training_returns)), 
                avg_with_baseline, 'r-', linewidth=2, label='With Baseline')
        plt.xlabel('Episode')
        plt.ylabel(f'{window}-Episode Average Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot")

def demonstrate_actor_critic():
    """Demonstrate actor-critic algorithm."""
    print("\n" + "=" * 60)
    print("ACTOR-CRITIC ALGORITHM DEMONSTRATION")
    print("=" * 60)
    
    env = SimpleMDP(max_steps=10)
    policy = CategoricalPolicy(n_states=2, n_actions=2, learning_rate=0.1)
    value_fn = ValueFunction(n_states=2, learning_rate=0.1)
    
    # Train with actor-critic
    result = actor_critic(env, policy, value_fn, gamma=0.99, n_episodes=100)
    
    print("Actor-Critic Results:")
    print("-" * 40)
    print(f"Final average return: {np.mean(result.training_returns[-20:]):.3f}")
    
    # Show learned value function
    print("\nLearned Value Function:")
    print("-" * 40)
    for s in range(2):
        print(f"V({s}) = {value_fn(s):.3f}")
    
    # Show learned policy
    print("\nLearned Policy:")
    print("-" * 40)
    for s in range(2):
        probs = policy.action_probs(s)
        print(f"State {s}: π(a=0)={probs[0]:.3f}, π(a=1)={probs[1]:.3f}")

def demonstrate_continuous_policy():
    """Demonstrate continuous policy on CartPole."""
    print("\n" + "=" * 60)
    print("CONTINUOUS POLICY DEMONSTRATION")
    print("=" * 60)
    
    env = ContinuousCartPole(max_steps=200)
    
    # Simple feature function
    def feature_fn(s):
        return np.array([1.0, s[0], s[1], s[2], s[3], 
                        s[0]**2, s[1]**2, s[2]**2, s[3]**2])
    
    policy = GaussianPolicy(state_dim=4, action_dim=1, feature_fn=feature_fn, learning_rate=0.01)
    
    # Simple training loop
    n_episodes = 50
    training_returns = []
    
    for episode in range(n_episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        done = False
        
        while not done:
            a = policy.sample_action(s)
            s_next, r, done, _ = env.step(a)
            
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s_next
        
        episode_return = sum(rewards)
        training_returns.append(episode_return)
        
        # Simple policy update (no baseline for simplicity)
        returns = compute_returns(rewards, gamma=0.99)
        for t in range(len(states)):
            advantage = returns[t]  # Simple advantage estimate
            policy.update(states[t], actions[t], advantage)
        
        if (episode + 1) % 10 == 0:
            avg_return = np.mean(training_returns[-10:])
            print(f"Episode {episode + 1}: Average return = {avg_return:.3f}")
    
    print(f"\nFinal average return: {np.mean(training_returns[-10:]):.3f}")

if __name__ == "__main__":
    # Run comprehensive demonstrations
    demonstrate_reinforce()
    demonstrate_baseline_comparison()
    demonstrate_actor_critic()
    demonstrate_continuous_policy()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("This demonstration shows:")
    print("1. REINFORCE algorithm for policy gradients")
    print("2. Baseline methods for variance reduction")
    print("3. Actor-critic methods combining policy and value learning")
    print("4. Continuous policies for continuous action spaces")
    print("\nKey insights:")
    print("- Policy gradients provide direct optimization of expected return")
    print("- Baselines reduce variance without changing the gradient")
    print("- Actor-critic methods enable online learning")
    print("- Continuous policies handle continuous action spaces naturally")
    print("- Natural gradients can improve convergence") 
"""
Continuous State Markov Decision Processes (MDPs) Implementation and Examples

This file implements the core concepts from 02_continuous_state_mdp.md:

1. Discretization: Converting continuous state spaces to discrete grids
2. Value Function Approximation: Using linear regression and feature mappings
3. Fitted Value Iteration: Solving continuous MDPs with function approximation
4. Feature Engineering: Creating effective representations for continuous states
5. Sample-based Methods: Learning from experience in continuous spaces

Key Concepts Demonstrated:
- State space discretization: s ∈ ℝ^d → grid cell indices
- Linear value function approximation: V(s) ≈ θ^T φ(s)
- Fitted value iteration: V_{k+1}(s) = max_a [R(s,a) + γ E[V_k(s')]]
- Feature engineering: Polynomial, radial basis, and neural network features
- Sample complexity and approximation error

"""

import numpy as np
from collections import defaultdict
from typing import Callable, Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# ----------------------
# Discretization Utilities
# ----------------------

def discretize_state(state: np.ndarray, grid_bounds: List[Tuple[float, float]], 
                    grid_bins: List[int]) -> Tuple[int, ...]:
    """
    Discretize a continuous state into a grid cell index.
    
    This implements the discretization approach discussed in the markdown:
    - Maps continuous state s ∈ ℝ^d to discrete grid cell indices
    - Handles boundary conditions (clipping to grid bounds)
    - Returns tuple of indices for multi-dimensional states
    
    Args:
        state: np.array of shape (d,) representing continuous state
        grid_bounds: List of (min, max) bounds for each dimension
        grid_bins: List of number of bins for each dimension
        
    Returns:
        Tuple of grid indices (i1, i2, ..., id)
        
    Example:
        >>> state = np.array([0.3, 1.5])
        >>> bounds = [(0.0, 1.0), (0.0, 2.0)]
        >>> bins = [4, 5]
        >>> discretize_state(state, bounds, bins)
        (1, 3)
    """
    if len(state) != len(grid_bounds) or len(state) != len(grid_bins):
        raise ValueError("State dimension must match bounds and bins")
    
    idx = []
    for i, (val, (low, high), bins) in enumerate(zip(state, grid_bounds, grid_bins)):
        # Handle boundary conditions
        if val <= low:
            idx.append(0)
        elif val >= high:
            idx.append(bins - 1)
        else:
            # Linear mapping from [low, high] to [0, bins-1]
            normalized = (val - low) / (high - low)
            idx.append(int(normalized * bins))
    
    return tuple(idx)

def create_discrete_mdp_from_continuous(continuous_mdp, grid_bounds: List[Tuple[float, float]], 
                                      grid_bins: List[int]):
    """
    Create a discrete MDP by discretizing a continuous MDP.
    
    This demonstrates the discretization approach for solving continuous MDPs:
    1. Discretize the state space into a grid
    2. Approximate continuous transitions by sampling
    3. Create a discrete MDP that can be solved with standard algorithms
    
    Args:
        continuous_mdp: Object with reward_fn(s,a) and transition_fn(s,a) methods
        grid_bounds: Bounds for each state dimension
        grid_bins: Number of bins for each dimension
        
    Returns:
        Discrete MDP with states, actions, transitions, and rewards
    """
    # Create discrete states
    discrete_states = []
    for i in range(grid_bins[0]):
        for j in range(grid_bins[1]):
            discrete_states.append((i, j))
    
    # Create discrete actions (assuming same as continuous)
    discrete_actions = [0, 1]  # Example actions
    
    # Build transition probabilities by sampling
    P = {}
    R = {}
    
    for s_disc in discrete_states:
        P[s_disc] = {}
        R[s_disc] = {}
        
        # Convert discrete state back to continuous for sampling
        s_cont = np.array([
            grid_bounds[0][0] + (s_disc[0] + 0.5) * (grid_bounds[0][1] - grid_bounds[0][0]) / grid_bins[0],
            grid_bounds[1][0] + (s_disc[1] + 0.5) * (grid_bounds[1][1] - grid_bounds[1][0]) / grid_bins[1]
        ])
        
        for a in discrete_actions:
            P[s_disc][a] = defaultdict(float)
            R[s_disc][a] = 0.0
            
            # Sample transitions to estimate probabilities
            n_samples = 100
            for _ in range(n_samples):
                s_next_cont = continuous_mdp.transition_fn(s_cont, a)
                s_next_disc = discretize_state(s_next_cont, grid_bounds, grid_bins)
                P[s_disc][a][s_next_disc] += 1.0 / n_samples
                
                r = continuous_mdp.reward_fn(s_cont, a)
                R[s_disc][a] += r / n_samples
    
    return discrete_states, discrete_actions, P, R

# ----------------------
# Value Function Approximation
# ----------------------

class LinearValueFunction:
    """
    Linear value function approximator: V(s) ≈ θ^T φ(s)
    
    This implements the linear approximation approach discussed in the markdown:
    - Uses feature function φ(s) to map states to feature vectors
    - Learns parameters θ via least squares regression
    - Provides efficient value estimation for continuous states
    
    The approximation error depends on:
    1. Feature function expressiveness
    2. Number of training samples
    3. State space complexity
    """
    
    def __init__(self, feature_fn: Callable[[np.ndarray], np.ndarray], d_features: int):
        """
        Initialize linear value function approximator.
        
        Args:
            feature_fn: Function that maps state to feature vector φ(s)
            d_features: Dimension of feature vector
        """
        self.feature_fn = feature_fn
        self.theta = np.zeros(d_features)
        self.d_features = d_features
    
    def __call__(self, s: np.ndarray) -> float:
        """Compute value estimate: V(s) = θ^T φ(s)"""
        phi = self.feature_fn(s)
        return np.dot(self.theta, phi)
    
    def fit(self, states: List[np.ndarray], targets: np.ndarray):
        """
        Fit parameters θ using least squares regression.
        
        This solves: min_θ ||Φθ - y||^2
        where Φ is the feature matrix and y is the target vector.
        
        Args:
            states: List of states
            targets: Target values for each state
        """
        # Build feature matrix Φ
        Phi = np.stack([self.feature_fn(s) for s in states])
        
        # Solve least squares: θ = (Φ^T Φ)^(-1) Φ^T y
        self.theta = np.linalg.lstsq(Phi, targets, rcond=None)[0]
    
    def get_gradient(self, s: np.ndarray) -> np.ndarray:
        """Get gradient of value function with respect to θ: ∇_θ V(s) = φ(s)"""
        return self.feature_fn(s)

class NeuralValueFunction:
    """
    Neural network value function approximator.
    
    This demonstrates a more sophisticated approximation approach:
    - Uses a neural network to learn complex value functions
    - Can capture non-linear relationships in the state space
    - Requires gradient-based optimization for training
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [64, 32]):
        """
        Initialize neural value function.
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
        """
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims
        
        # Simple implementation using numpy (in practice, use PyTorch/TensorFlow)
        self.weights = []
        self.biases = []
        
        # Build network architecture
        layer_dims = [state_dim] + hidden_dims + [1]
        for i in range(len(layer_dims) - 1):
            self.weights.append(np.random.randn(layer_dims[i], layer_dims[i+1]) * 0.1)
            self.biases.append(np.random.randn(layer_dims[i+1]) * 0.1)
    
    def forward(self, s: np.ndarray) -> float:
        """Forward pass through the network."""
        x = s.reshape(-1)
        for i in range(len(self.weights) - 1):
            x = np.tanh(x @ self.weights[i] + self.biases[i])
        x = x @ self.weights[-1] + self.biases[-1]
        return x[0]
    
    def __call__(self, s: np.ndarray) -> float:
        return self.forward(s)

# ----------------------
# Feature Engineering
# ----------------------

def polynomial_features(degree: int = 2) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create polynomial feature function.
    
    For 1D state s, creates features [1, s, s^2, ..., s^degree]
    For 2D state (s1, s2), creates features [1, s1, s2, s1^2, s1*s2, s2^2, ...]
    
    Args:
        degree: Maximum polynomial degree
        
    Returns:
        Feature function φ(s)
    """
    def feature_fn(s: np.ndarray) -> np.ndarray:
        s = s.flatten()
        features = [1.0]  # Bias term
        
        # Add polynomial terms up to specified degree
        for d in range(1, degree + 1):
            if len(s) == 1:
                # 1D case: add s^d
                features.append(s[0] ** d)
            else:
                # Multi-dimensional case: add all combinations
                from itertools import combinations_with_replacement
                for combo in combinations_with_replacement(range(len(s)), d):
                    term = 1.0
                    for idx in combo:
                        term *= s[idx]
                    features.append(term)
        
        return np.array(features)
    
    return feature_fn

def radial_basis_features(centers: np.ndarray, sigma: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create radial basis function (RBF) features.
    
    Features are φ_i(s) = exp(-||s - c_i||^2 / (2σ^2))
    where c_i are the centers and σ is the bandwidth.
    
    Args:
        centers: Array of center points
        sigma: Bandwidth parameter
        
    Returns:
        Feature function φ(s)
    """
    def feature_fn(s: np.ndarray) -> np.ndarray:
        s = s.flatten()
        features = [1.0]  # Bias term
        
        # Add RBF features
        for center in centers:
            dist_sq = np.sum((s - center) ** 2)
            features.append(np.exp(-dist_sq / (2 * sigma ** 2)))
        
        return np.array(features)
    
    return feature_fn

def fourier_features(freqs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create Fourier basis features.
    
    Features are φ_i(s) = cos(ω_i^T s) and φ_{i+1}(s) = sin(ω_i^T s)
    where ω_i are frequency vectors.
    
    Args:
        freqs: Array of frequency vectors
        
    Returns:
        Feature function φ(s)
    """
    def feature_fn(s: np.ndarray) -> np.ndarray:
        s = s.flatten()
        features = [1.0]  # Bias term
        
        # Add Fourier features
        for freq in freqs:
            features.append(np.cos(np.dot(freq, s)))
            features.append(np.sin(np.dot(freq, s)))
        
        return np.array(features)
    
    return feature_fn

# ----------------------
# Fitted Value Iteration
# ----------------------

def fitted_value_iteration(
    sample_states: List[np.ndarray],
    actions: List[int],
    reward_fn: Callable[[np.ndarray, int], float],
    transition_fn: Callable[[np.ndarray, int], np.ndarray],
    feature_fn: Callable[[np.ndarray], np.ndarray],
    gamma: float = 0.99,
    n_iter: int = 20,
    n_samples_next: int = 10,
) -> LinearValueFunction:
    """
    Fitted Value Iteration for continuous state MDPs.
    
    This implements the algorithm discussed in the markdown:
    1. Initialize value function V_0(s) = 0
    2. For k = 0, 1, 2, ...:
       a. Compute targets: y_i = max_a [R(s_i,a) + γ E[V_k(s')]]
       b. Fit V_{k+1} by regression: min_θ ||Φθ - y||^2
    
    The algorithm addresses the curse of dimensionality by:
    - Using function approximation instead of tabular representation
    - Sampling states and transitions to estimate expectations
    - Using regression to fit value functions
    
    Args:
        sample_states: List of states to use for approximation
        actions: List of possible actions
        reward_fn: Function (state, action) -> reward
        transition_fn: Function (state, action) -> next_state
        feature_fn: Function (state) -> feature vector
        gamma: Discount factor
        n_iter: Number of value iteration steps
        n_samples_next: Number of samples to estimate expectation over next state
        
    Returns:
        Trained value function approximator
    """
    # Initialize value function
    vf = LinearValueFunction(feature_fn, len(feature_fn(sample_states[0])))
    
    print(f"Starting fitted value iteration with {len(sample_states)} states, "
          f"{len(actions)} actions, {n_iter} iterations")
    
    for iteration in range(n_iter):
        targets = []
        
        for s in sample_states:
            # Compute Q-values for all actions
            q_vals = []
            for a in actions:
                # Sample next states to estimate expectation
                next_vs = []
                for _ in range(n_samples_next):
                    s_next = transition_fn(s, a)
                    next_vs.append(vf(s_next))
                
                expected_v = np.mean(next_vs)
                q = reward_fn(s, a) + gamma * expected_v
                q_vals.append(q)
            
            # Target is maximum Q-value (Bellman optimality)
            targets.append(np.max(q_vals))
        
        # Fit value function to targets
        vf.fit(sample_states, np.array(targets))
        
        if (iteration + 1) % 5 == 0:
            print(f"Iteration {iteration + 1}/{n_iter} completed")
    
    return vf

def fitted_q_iteration(
    sample_states: List[np.ndarray],
    actions: List[int],
    reward_fn: Callable[[np.ndarray, int], float],
    transition_fn: Callable[[np.ndarray, int], np.ndarray],
    feature_fn: Callable[[np.ndarray], np.ndarray],
    gamma: float = 0.99,
    n_iter: int = 20,
    n_samples_next: int = 10,
) -> LinearValueFunction:
    """
    Fitted Q-Iteration for continuous state-action spaces.
    
    This learns Q(s,a) directly instead of V(s):
    1. Initialize Q_0(s,a) = 0
    2. For k = 0, 1, 2, ...:
       a. Compute targets: y_i = R(s_i,a_i) + γ max_a' Q_k(s',a')
       b. Fit Q_{k+1} by regression
    
    Args:
        sample_states: List of states
        actions: List of actions
        reward_fn: Function (state, action) -> reward
        transition_fn: Function (state, action) -> next_state
        feature_fn: Function (state, action) -> feature vector
        gamma: Discount factor
        n_iter: Number of iterations
        n_samples_next: Number of samples for expectation
        
    Returns:
        Trained Q-function approximator
    """
    # Create state-action features
    def sa_feature_fn(sa_pair):
        s, a = sa_pair
        state_features = feature_fn(s)
        # Add action features (one-hot encoding)
        action_features = np.zeros(len(actions))
        action_features[a] = 1.0
        return np.concatenate([state_features, action_features])
    
    # Initialize Q-function
    qf = LinearValueFunction(sa_feature_fn, len(sa_feature_fn((sample_states[0], actions[0]))))
    
    # Create state-action pairs
    sa_pairs = [(s, a) for s in sample_states for a in actions]
    
    for iteration in range(n_iter):
        targets = []
        
        for s, a in sa_pairs:
            # Sample next state
            s_next = transition_fn(s, a)
            
            # Compute max Q-value over next actions
            next_q_vals = []
            for a_next in actions:
                next_q_vals.append(qf((s_next, a_next)))
            
            max_next_q = np.max(next_q_vals)
            target = reward_fn(s, a) + gamma * max_next_q
            targets.append(target)
        
        # Fit Q-function to targets
        qf.fit(sa_pairs, np.array(targets))
        
        if (iteration + 1) % 5 == 0:
            print(f"Q-iteration {iteration + 1}/{n_iter} completed")
    
    return qf

# ----------------------
# Example Usage and Demonstrations
# ----------------------

class ContinuousMDP:
    """Example continuous MDP for demonstration."""
    
    def __init__(self, state_dim: int = 2):
        self.state_dim = state_dim
    
    def reward_fn(self, s: np.ndarray, a: int) -> float:
        """Reward function: peak at origin for action 1, otherwise 0."""
        if a == 1:
            # Gaussian reward centered at origin
            return np.exp(-np.sum(s ** 2))
        else:
            return 0.0
    
    def transition_fn(self, s: np.ndarray, a: int) -> np.ndarray:
        """Transition function: noisy movement."""
        s_next = s.copy()
        
        if a == 0:
            # Move towards negative direction
            s_next -= 0.1
        else:
            # Move towards positive direction
            s_next += 0.1
        
        # Add noise
        s_next += 0.01 * np.random.randn(self.state_dim)
        
        # Clip to bounds
        s_next = np.clip(s_next, -1.0, 1.0)
        
        return s_next

def demonstrate_discretization():
    """Demonstrate state space discretization."""
    print("=" * 60)
    print("STATE SPACE DISCRETIZATION DEMONSTRATION")
    print("=" * 60)
    
    # 1D discretization example
    print("\n1. 1D State Space Discretization:")
    print("-" * 40)
    
    grid_bounds_1d = [(0.0, 1.0)]
    grid_bins_1d = [10]
    
    test_states_1d = [0.0, 0.25, 0.5, 0.75, 1.0, -0.1, 1.1]
    
    for x in test_states_1d:
        state = np.array([x])
        idx = discretize_state(state, grid_bounds_1d, grid_bins_1d)
        print(f"State {x:.2f} -> grid cell {idx}")
    
    # 2D discretization example
    print("\n2. 2D State Space Discretization:")
    print("-" * 40)
    
    grid_bounds_2d = [(0.0, 1.0), (0.0, 2.0)]
    grid_bins_2d = [4, 5]
    
    test_states_2d = [
        np.array([0.1, 0.2]),
        np.array([0.5, 1.0]),
        np.array([0.9, 1.8]),
        np.array([0.0, 0.0]),
        np.array([1.0, 2.0])
    ]
    
    for s in test_states_2d:
        idx = discretize_state(s, grid_bounds_2d, grid_bins_2d)
        print(f"State {s} -> grid cell {idx}")

def demonstrate_value_approximation():
    """Demonstrate value function approximation."""
    print("\n" + "=" * 60)
    print("VALUE FUNCTION APPROXIMATION DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    xs = np.linspace(-1, 1, 50)
    states = [np.array([x]) for x in xs]
    
    # True value function: V(s) = 2 + 3s - s^2
    def true_vf(s):
        return 2 + 3 * s[0] - s[0] ** 2
    
    targets = np.array([true_vf(s) for s in states])
    
    # Test different feature functions
    feature_functions = {
        "Polynomial (degree 2)": polynomial_features(degree=2),
        "Polynomial (degree 3)": polynomial_features(degree=3),
        "RBF (5 centers)": radial_basis_features(
            centers=np.array([[-0.8], [-0.4], [0.0], [0.4], [0.8]]),
            sigma=0.3
        ),
        "Fourier (3 freqs)": fourier_features(
            freqs=np.array([[1.0], [2.0], [3.0]])
        )
    }
    
    print("\nApproximation Results:")
    print("-" * 40)
    
    for name, feature_fn in feature_functions.items():
        # Create and fit value function
        vf = LinearValueFunction(feature_fn, len(feature_fn(states[0])))
        vf.fit(states, targets)
        
        # Evaluate on test points
        test_xs = np.linspace(-1, 1, 21)
        test_states = [np.array([x]) for x in test_xs]
        
        mse = 0.0
        for s in test_states:
            pred = vf(s)
            true_val = true_vf(s)
            mse += (pred - true_val) ** 2
        mse /= len(test_states)
        
        print(f"{name}: MSE = {mse:.6f}")
    
    # Visualize best approximation
    best_vf = LinearValueFunction(polynomial_features(degree=3), 4)
    best_vf.fit(states, targets)
    
    try:
        plt.figure(figsize=(10, 6))
        
        # Plot true function
        plt.plot(xs, [true_vf(np.array([x])) for x in xs], 'b-', label='True V(s)', linewidth=2)
        
        # Plot approximation
        preds = [best_vf(np.array([x])) for x in xs]
        plt.plot(xs, preds, 'r--', label='Approximated V(s)', linewidth=2)
        
        # Plot training points
        plt.scatter(xs, targets, c='green', alpha=0.6, label='Training Data')
        
        plt.xlabel('State s')
        plt.ylabel('Value V(s)')
        plt.title('Value Function Approximation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot")

def demonstrate_fitted_value_iteration():
    """Demonstrate fitted value iteration."""
    print("\n" + "=" * 60)
    print("FITTED VALUE ITERATION DEMONSTRATION")
    print("=" * 60)
    
    # Create continuous MDP
    mdp = ContinuousMDP(state_dim=1)
    
    # Sample states uniformly
    sample_states = [np.array([x]) for x in np.linspace(-1, 1, 20)]
    actions = [0, 1]
    
    # Create feature function
    feature_fn = polynomial_features(degree=2)
    
    # Run fitted value iteration
    vf = fitted_value_iteration(
        sample_states=sample_states,
        actions=actions,
        reward_fn=mdp.reward_fn,
        transition_fn=mdp.transition_fn,
        feature_fn=feature_fn,
        gamma=0.95,
        n_iter=30,
        n_samples_next=5
    )
    
    # Evaluate learned value function
    print("\nLearned Value Function:")
    print("-" * 40)
    print("State\tValue")
    for x in np.linspace(-1, 1, 11):
        v = vf(np.array([x]))
        print(f"{x:.2f}\t{v:.3f}")
    
    # Compare with discretization approach
    print("\nComparison with Discretization:")
    print("-" * 40)
    
    # Create discrete MDP
    grid_bounds = [(-1.0, 1.0)]
    grid_bins = [10]
    
    discrete_states, discrete_actions, P, R = create_discrete_mdp_from_continuous(
        mdp, grid_bounds, grid_bins
    )
    
    # Solve discrete MDP (simplified)
    print(f"Discrete MDP has {len(discrete_states)} states")
    print("Discrete approach requires tabular representation")
    print("Fitted VI uses function approximation (more scalable)")

def demonstrate_sample_complexity():
    """Demonstrate sample complexity considerations."""
    print("\n" + "=" * 60)
    print("SAMPLE COMPLEXITY DEMONSTRATION")
    print("=" * 60)
    
    # Test different numbers of sample states
    sample_sizes = [10, 20, 50, 100]
    mdp = ContinuousMDP(state_dim=1)
    actions = [0, 1]
    feature_fn = polynomial_features(degree=2)
    
    print("\nEffect of Sample Size on Approximation Quality:")
    print("-" * 50)
    
    for n_samples in sample_sizes:
        # Sample states
        sample_states = [np.array([x]) for x in np.linspace(-1, 1, n_samples)]
        
        # Run fitted value iteration
        vf = fitted_value_iteration(
            sample_states=sample_states,
            actions=actions,
            reward_fn=mdp.reward_fn,
            transition_fn=mdp.transition_fn,
            feature_fn=feature_fn,
            gamma=0.95,
            n_iter=20,
            n_samples_next=5
        )
        
        # Evaluate on test set
        test_states = [np.array([x]) for x in np.linspace(-1, 1, 50)]
        test_values = [vf(s) for s in test_states]
        
        # Compute statistics
        mean_value = np.mean(test_values)
        std_value = np.std(test_values)
        
        print(f"n_samples = {n_samples:3d}: mean = {mean_value:.3f}, std = {std_value:.3f}")

if __name__ == "__main__":
    # Run comprehensive demonstrations
    demonstrate_discretization()
    demonstrate_value_approximation()
    demonstrate_fitted_value_iteration()
    demonstrate_sample_complexity()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("This demonstration shows:")
    print("1. State space discretization for continuous MDPs")
    print("2. Value function approximation with different feature functions")
    print("3. Fitted value iteration algorithm")
    print("4. Sample complexity considerations")
    print("\nKey insights:")
    print("- Discretization trades accuracy for computational tractability")
    print("- Function approximation enables handling continuous state spaces")
    print("- Feature engineering is crucial for approximation quality")
    print("- Sample complexity grows with state space dimension")
    print("- Fitted VI provides a principled approach to continuous MDPs") 
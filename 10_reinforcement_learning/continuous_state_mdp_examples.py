"""
continuous_state_mdp_examples.py

Python implementation of key concepts from continuous-state MDPs:
- Discretization of continuous state spaces
- Value function approximation (linear regression, feature mappings)
- Fitted value iteration for continuous state MDPs
- Example usage

This code is designed to be educational and easy to follow, with clear comments and docstrings.
"""
import numpy as np
from collections import defaultdict
from typing import Callable, Tuple

# ----------------------
# Discretization Utilities
# ----------------------
def discretize_state(state, grid_bounds, grid_bins):
    """
    Discretize a continuous state into a grid cell index.
    state: np.array of shape (d,)
    grid_bounds: list of (min, max) for each dimension
    grid_bins: list of number of bins for each dimension
    Returns: tuple of grid indices
    """
    idx = []
    for i, (val, (low, high), bins) in enumerate(zip(state, grid_bounds, grid_bins)):
        if val <= low:
            idx.append(0)
        elif val >= high:
            idx.append(bins - 1)
        else:
            idx.append(int((val - low) / (high - low) * bins))
    return tuple(idx)

# ----------------------
# Value Function Approximation (Linear)
# ----------------------
class LinearValueFunction:
    def __init__(self, feature_fn: Callable[[np.ndarray], np.ndarray], d_features: int):
        self.feature_fn = feature_fn
        self.theta = np.zeros(d_features)

    def __call__(self, s):
        phi = self.feature_fn(s)
        return np.dot(self.theta, phi)

    def fit(self, states, targets):
        # Fit theta by least squares: min_theta ||Phi*theta - targets||^2
        Phi = np.stack([self.feature_fn(s) for s in states])
        self.theta = np.linalg.lstsq(Phi, targets, rcond=None)[0]

# ----------------------
# Fitted Value Iteration
# ----------------------
def fitted_value_iteration(
    sample_states,
    actions,
    reward_fn: Callable[[np.ndarray, int], float],
    transition_fn: Callable[[np.ndarray, int], np.ndarray],
    feature_fn: Callable[[np.ndarray], np.ndarray],
    gamma=0.99,
    n_iter=20,
    n_samples_next=10,
):
    """
    Fitted Value Iteration for continuous state MDPs.
    sample_states: list of np.ndarray, sampled states
    actions: list of action indices
    reward_fn: function (state, action) -> reward
    transition_fn: function (state, action) -> next_state (or samples next_state)
    feature_fn: function (state) -> feature vector
    gamma: discount factor
    n_iter: number of fitted value iteration steps
    n_samples_next: number of samples to estimate expectation over next state
    Returns: value function approximator (LinearValueFunction)
    """
    vf = LinearValueFunction(feature_fn, len(feature_fn(sample_states[0])))
    vf.theta = np.zeros_like(vf.theta)
    for it in range(n_iter):
        targets = []
        for s in sample_states:
            q_vals = []
            for a in actions:
                # Sample next states and average
                next_vs = []
                for _ in range(n_samples_next):
                    s_next = transition_fn(s, a)
                    next_vs.append(vf(s_next))
                expected_v = np.mean(next_vs)
                q = reward_fn(s, a) + gamma * expected_v
                q_vals.append(q)
            targets.append(np.max(q_vals))
        vf.fit(sample_states, np.array(targets))
    return vf

# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    # Example: 1D continuous state, 2 actions
    np.random.seed(42)
    state_dim = 1
    n_actions = 2
    actions = [0, 1]
    gamma = 0.95
    # State space: x in [0, 1]
    grid_bounds = [(0.0, 1.0)]
    grid_bins = [10]
    # Feature: polynomial (1, x, x^2)
    def feature_fn(s):
        x = s[0]
        return np.array([1.0, x, x**2])
    # Reward: peak at x=0.8 for action 1, otherwise 0
    def reward_fn(s, a):
        if a == 1:
            return np.exp(-10 * (s[0] - 0.8) ** 2)
        else:
            return 0.0
    # Transition: noisy move left or right
    def transition_fn(s, a):
        x = s[0]
        if a == 0:
            x_next = x - 0.05 + 0.01 * np.random.randn()
        else:
            x_next = x + 0.05 + 0.01 * np.random.randn()
        x_next = np.clip(x_next, 0.0, 1.0)
        return np.array([x_next])
    # Sample states uniformly
    sample_states = [np.array([x]) for x in np.linspace(0, 1, 20)]
    # Fitted value iteration
    vf = fitted_value_iteration(
        sample_states,
        actions,
        reward_fn,
        transition_fn,
        feature_fn,
        gamma=gamma,
        n_iter=30,
        n_samples_next=5,
    )
    # Print value estimates
    print("State\tValue")
    for x in np.linspace(0, 1, 11):
        v = vf(np.array([x]))
        print(f"{x:.2f}\t{v:.3f}")

    # Discretization example (1D)
    print("\nDiscretization example (1D):")
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = discretize_state(np.array([x]), grid_bounds, grid_bins)
        print(f"State {x:.2f} -> grid cell {idx}")

    # Discretization example (2D)
    print("\nDiscretization example (2D):")
    grid_bounds_2d = [(0.0, 1.0), (0.0, 2.0)]
    grid_bins_2d = [4, 5]
    example_states_2d = [np.array([0.1, 0.2]), np.array([0.5, 1.0]), np.array([0.9, 1.8]), np.array([0.0, 0.0]), np.array([1.0, 2.0])]
    for s in example_states_2d:
        idx = discretize_state(s, grid_bounds_2d, grid_bins_2d)
        print(f"State {s} -> grid cell {idx}")

    # Value function approximation (regression) example
    print("\nValue function approximation (regression) example:")
    # Fit a quadratic function y = 2 + 3x - x^2
    def true_vf(x):
        return 2 + 3 * x - x ** 2
    xs = np.linspace(0, 1, 20)
    states = [np.array([x]) for x in xs]
    targets = np.array([true_vf(x) for x in xs])
    lin_vf = LinearValueFunction(feature_fn, 3)
    lin_vf.fit(states, targets)
    print("x\tTrue V(x)\tPredicted V(x)")
    for x in np.linspace(0, 1, 11):
        v_true = true_vf(x)
        v_pred = lin_vf(np.array([x]))
        print(f"{x:.2f}\t{v_true:.3f}\t\t{v_pred:.3f}") 
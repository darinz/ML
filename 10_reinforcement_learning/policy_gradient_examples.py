"""
Python implementations and demonstrations for policy gradient methods (REINFORCE) with and without baselines, as described in 04_policy_gradient.md.

- REINFORCE algorithm for finite-horizon MDPs
- Policy gradient with baseline
- Baseline fitting via regression
- Example usage with a simple environment

Each section is self-contained and includes example usage and comments.
"""
import numpy as np
from collections import namedtuple
from typing import Callable, List

# For baseline regression
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None

# Define a trajectory data structure
Trajectory = namedtuple('Trajectory', ['states', 'actions', 'rewards'])

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class CategoricalPolicy:
    """
    Simple softmax policy for discrete action spaces.
    theta: parameter matrix of shape (n_states, n_actions)
    """
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.theta = np.zeros((n_states, n_actions))
    def action_probs(self, s):
        return softmax(self.theta[s])
    def sample_action(self, s):
        return np.random.choice(self.n_actions, p=self.action_probs(s))
    def log_prob(self, s, a):
        probs = self.action_probs(s)
        return np.log(probs[a] + 1e-8)
    def grad_log_prob(self, s, a):
        probs = self.action_probs(s)
        grad = -probs
        grad[a] += 1
        return grad  # shape: (n_actions,)

# --- 1. REINFORCE algorithm (no baseline) ---
def reinforce(env, policy, gamma=1.0, n_episodes=100, alpha=0.1):
    """
    REINFORCE algorithm for finite-horizon MDPs (no baseline).
    env: environment with reset() and step(a) methods
    policy: CategoricalPolicy
    gamma: discount factor
    n_episodes: number of episodes
    alpha: learning rate
    Returns: list of episode returns
    """
    returns = []
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
        # Compute returns
        G = 0
        returns_episode = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns_episode.insert(0, G)
        returns.append(sum(rewards))
        # Policy update
        for t in range(len(states)):
            s_t, a_t = states[t], actions[t]
            grad = policy.grad_log_prob(s_t, a_t)
            policy.theta[s_t] += alpha * grad * (returns_episode[t])
    return returns

# --- 2. Policy gradient with baseline ---
def reinforce_with_baseline(env, policy, gamma=1.0, n_episodes=100, alpha=0.1, fit_baseline=True):
    """
    REINFORCE with baseline. Baseline is fit by regression to returns.
    env: environment with reset() and step(a) methods
    policy: CategoricalPolicy
    gamma: discount factor
    n_episodes: number of episodes
    alpha: learning rate
    fit_baseline: if True, fit baseline by regression; else use mean return
    Returns: list of episode returns
    """
    n_states = policy.n_states
    baseline = np.zeros(n_states)
    returns = []
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
        # Compute returns-to-go
        G = 0
        returns_to_go = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns_to_go.insert(0, G)
        returns.append(sum(rewards))
        # Fit baseline (regression)
        if fit_baseline and LinearRegression is not None:
            X = np.array(states).reshape(-1, 1)
            y = np.array(returns_to_go)
            reg = LinearRegression().fit(X, y)
            baseline_pred = reg.predict(np.arange(n_states).reshape(-1, 1))
            baseline = baseline_pred
        else:
            # Use mean return as baseline
            baseline = np.mean(returns_to_go)
        # Policy update
        for t in range(len(states)):
            s_t, a_t = states[t], actions[t]
            advantage = returns_to_go[t] - baseline[s_t] if fit_baseline and LinearRegression is not None else returns_to_go[t] - baseline
            grad = policy.grad_log_prob(s_t, a_t)
            policy.theta[s_t] += alpha * grad * advantage
    return returns

# --- 3. Example usage: Simple MDP environment ---
class SimpleMDP:
    """
    A toy MDP with 2 states and 2 actions.
    State 0: action 0 -> reward 0, action 1 -> reward 1, both go to state 1
    State 1: action 0 -> reward 1, action 1 -> reward 0, both go to state 0
    Episode ends after 4 steps.
    """
    def __init__(self):
        self.n_states = 2
        self.n_actions = 2
        self.max_steps = 4
        self.reset()
    def reset(self):
        self.s = 0
        self.steps = 0
        return self.s
    def step(self, a):
        if self.s == 0:
            r = 1 if a == 1 else 0
            s_next = 1
        else:
            r = 1 if a == 0 else 0
            s_next = 0
        self.s = s_next
        self.steps += 1
        done = self.steps >= self.max_steps
        return self.s, r, done, {}

# --- 4. Demonstration ---
if __name__ == "__main__":
    print("--- REINFORCE on SimpleMDP (no baseline) ---")
    env = SimpleMDP()
    policy = CategoricalPolicy(n_states=2, n_actions=2)
    returns = reinforce(env, policy, gamma=0.99, n_episodes=100, alpha=0.1)
    print("Final policy parameters (no baseline):\n", policy.theta)
    print("Average return (last 10 episodes):", np.mean(returns[-10:]))

    print("\n--- REINFORCE with baseline on SimpleMDP ---")
    policy2 = CategoricalPolicy(n_states=2, n_actions=2)
    returns2 = reinforce_with_baseline(env, policy2, gamma=0.99, n_episodes=100, alpha=0.1, fit_baseline=True)
    print("Final policy parameters (with baseline):\n", policy2.theta)
    print("Average return (last 10 episodes):", np.mean(returns2[-10:])) 
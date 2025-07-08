"""
markov_decision_processes_examples.py

Python implementation of Markov Decision Processes (MDPs), including:
- MDP class
- Value Iteration
- Policy Iteration
- Model learning from experience
- Example usage

This code is designed to be educational and easy to follow, with clear comments and docstrings.
"""
import numpy as np
from collections import defaultdict, Counter
import random

class MDP:
    def __init__(self, states, actions, transition_probs, rewards, gamma):
        """
        states: list of all states
        actions: list of all actions
        transition_probs: dict of dicts, transition_probs[s][a][s'] = P(s'|s,a)
        rewards: dict, rewards[s] or rewards[(s,a)]
        gamma: discount factor (0 <= gamma < 1)
        """
        self.states = states
        self.actions = actions
        self.P = transition_probs
        self.R = rewards
        self.gamma = gamma

    def get_possible_actions(self, s):
        return self.actions

    def get_transition_probs(self, s, a):
        return self.P[s][a]

    def get_reward(self, s, a=None):
        if (s, a) in self.R:
            return self.R[(s, a)]
        return self.R.get(s, 0)

# ----------------------
# Value Iteration
# ----------------------
def value_iteration(mdp, theta=1e-6, max_iterations=1000):
    """
    Performs value iteration for the given MDP.
    Returns the optimal value function V and the optimal policy pi.
    """
    V = {s: 0 for s in mdp.states}
    for i in range(max_iterations):
        delta = 0
        for s in mdp.states:
            v = V[s]
            action_values = []
            for a in mdp.get_possible_actions(s):
                trans_probs = mdp.get_transition_probs(s, a)
                expected = sum(p * (mdp.get_reward(s, a) + mdp.gamma * V[s_]) for s_, p in trans_probs.items())
                action_values.append(expected)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # Derive policy
    pi = {}
    for s in mdp.states:
        best_a = None
        best_val = float('-inf')
        for a in mdp.get_possible_actions(s):
            trans_probs = mdp.get_transition_probs(s, a)
            expected = sum(p * (mdp.get_reward(s, a) + mdp.gamma * V[s_]) for s_, p in trans_probs.items())
            if expected > best_val:
                best_val = expected
                best_a = a
        pi[s] = best_a
    return V, pi

# ----------------------
# Policy Iteration
# ----------------------
def policy_iteration(mdp, max_iterations=1000):
    """
    Performs policy iteration for the given MDP.
    Returns the optimal value function V and the optimal policy pi.
    """
    # Initialize random policy
    pi = {s: random.choice(mdp.get_possible_actions(s)) for s in mdp.states}
    V = {s: 0 for s in mdp.states}
    for i in range(max_iterations):
        # Policy evaluation (solve for V^pi)
        for _ in range(100):  # simple iterative policy evaluation
            for s in mdp.states:
                a = pi[s]
                trans_probs = mdp.get_transition_probs(s, a)
                V[s] = sum(p * (mdp.get_reward(s, a) + mdp.gamma * V[s_]) for s_, p in trans_probs.items())
        # Policy improvement
        policy_stable = True
        for s in mdp.states:
            old_a = pi[s]
            best_a = old_a
            best_val = float('-inf')
            for a in mdp.get_possible_actions(s):
                trans_probs = mdp.get_transition_probs(s, a)
                expected = sum(p * (mdp.get_reward(s, a) + mdp.gamma * V[s_]) for s_, p in trans_probs.items())
                if expected > best_val:
                    best_val = expected
                    best_a = a
            pi[s] = best_a
            if best_a != old_a:
                policy_stable = False
        if policy_stable:
            break
    return V, pi

# ----------------------
# Model Learning from Experience
# ----------------------
class ModelLearner:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.transition_counts = {s: {a: Counter() for a in actions} for s in states}
        self.action_counts = {s: Counter() for s in states}
        self.reward_sums = {s: {a: 0.0 for a in actions} for s in states}
        self.reward_counts = {s: {a: 0 for a in actions} for s in states}

    def record(self, s, a, s_next, r):
        self.transition_counts[s][a][s_next] += 1
        self.action_counts[s][a] += 1
        self.reward_sums[s][a] += r
        self.reward_counts[s][a] += 1

    def get_transition_probs(self):
        P = {s: {a: {} for a in self.actions} for s in self.states}
        for s in self.states:
            for a in self.actions:
                total = self.action_counts[s][a]
                if total == 0:
                    # Uniform if never seen
                    for s_ in self.states:
                        P[s][a][s_] = 1.0 / len(self.states)
                else:
                    for s_ in self.states:
                        P[s][a][s_] = self.transition_counts[s][a][s_] / total
        return P

    def get_rewards(self):
        R = {}
        for s in self.states:
            for a in self.actions:
                count = self.reward_counts[s][a]
                if count == 0:
                    R[(s, a)] = 0.0
                else:
                    R[(s, a)] = self.reward_sums[s][a] / count
        return R

# ----------------------
# Example Usage
# ----------------------
if __name__ == "__main__":
    # Example: Simple 3-state MDP
    states = ['A', 'B', 'C']
    actions = ['left', 'right']
    gamma = 0.9
    # Transition probabilities: P[s][a][s']
    P = {
        'A': {
            'left': {'A': 0.8, 'B': 0.2, 'C': 0.0},
            'right': {'A': 0.0, 'B': 0.9, 'C': 0.1},
        },
        'B': {
            'left': {'A': 0.1, 'B': 0.7, 'C': 0.2},
            'right': {'A': 0.0, 'B': 0.2, 'C': 0.8},
        },
        'C': {
            'left': {'A': 0.0, 'B': 0.0, 'C': 1.0},
            'right': {'A': 0.0, 'B': 0.0, 'C': 1.0},
        },
    }
    # Rewards: R[(s, a)]
    R = {('A', 'left'): 0, ('A', 'right'): 5, ('B', 'left'): 1, ('B', 'right'): 2, ('C', 'left'): 0, ('C', 'right'): 0}
    mdp = MDP(states, actions, P, R, gamma)

    print("Value Iteration:")
    V_vi, pi_vi = value_iteration(mdp)
    print("Optimal Values:", V_vi)
    print("Optimal Policy:", pi_vi)

    print("\nPolicy Iteration:")
    V_pi, pi_pi = policy_iteration(mdp)
    print("Optimal Values:", V_pi)
    print("Optimal Policy:", pi_pi)

    # Example of model learning from experience
    print("\nModel Learning Example:")
    learner = ModelLearner(states, actions)
    # Simulate some experience
    for _ in range(100):
        s = random.choice(states)
        a = random.choice(actions)
        # Simulate next state and reward using true model
        next_states = list(P[s][a].keys())
        probs = list(P[s][a].values())
        s_next = np.random.choice(next_states, p=probs)
        r = R[(s, a)]
        learner.record(s, a, s_next, r)
    learned_P = learner.get_transition_probs()
    learned_R = learner.get_rewards()
    print("Learned Transition Probabilities:", learned_P)
    print("Learned Rewards:", learned_R)
    # Solve MDP with learned model
    learned_mdp = MDP(states, actions, learned_P, learned_R, gamma)
    V_learned, pi_learned = value_iteration(learned_mdp)
    print("\nValue Iteration on Learned Model:")
    print("Optimal Values:", V_learned)
    print("Optimal Policy:", pi_learned) 
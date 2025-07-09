"""
Markov Decision Processes (MDPs) Implementation and Examples

This file implements the core concepts from 01_markov_decision_processes.md:

1. MDP Framework: States, actions, transitions, rewards, and discount factor
2. Value Iteration: Dynamic programming algorithm for finding optimal value functions
3. Policy Iteration: Alternative algorithm combining policy evaluation and improvement
4. Model Learning: Learning transition probabilities and rewards from experience
5. Bellman Equations: The fundamental equations of dynamic programming

Key Concepts Demonstrated:
- Bellman optimality equation: V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]
- Value iteration convergence: V_{k+1}(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V_k(s')]
- Policy evaluation: V^π(s) = Σ π(a|s) [R(s,a) + γ Σ P(s'|s,a) V^π(s')]
- Policy improvement: π'(s) = argmax_a [R(s,a) + γ Σ P(s'|s,a) V^π(s')]

"""

import numpy as np
from collections import defaultdict, Counter
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

class MDP:
    """
    Markov Decision Process (MDP) class implementing the core framework.
    
    An MDP is defined by the tuple (S, A, P, R, γ) where:
    - S: Set of states
    - A: Set of actions  
    - P: Transition probability function P(s'|s,a)
    - R: Reward function R(s,a) or R(s)
    - γ: Discount factor (0 ≤ γ < 1)
    
    This class provides methods to:
    1. Query the MDP structure
    2. Compute optimal policies via value/policy iteration
    3. Simulate trajectories
    """
    
    def __init__(self, states: List, actions: List, transition_probs: Dict, 
                 rewards: Dict, gamma: float):
        """
        Initialize an MDP.
        
        Args:
            states: List of all possible states
            actions: List of all possible actions
            transition_probs: Dict[state][action][next_state] = probability
            rewards: Dict[(state, action)] = reward or Dict[state] = reward
            gamma: Discount factor (0 ≤ gamma < 1)
        """
        self.states = states
        self.actions = actions
        self.P = transition_probs  # P[s][a][s'] = P(s'|s,a)
        self.R = rewards          # R[(s,a)] or R[s]
        self.gamma = gamma
        
        # Validate MDP structure
        self._validate_mdp()
    
    def _validate_mdp(self):
        """Validate that the MDP structure is well-formed."""
        for s in self.states:
            for a in self.actions:
                # Check transition probabilities sum to 1
                prob_sum = sum(self.P[s][a].values())
                if not np.isclose(prob_sum, 1.0, atol=1e-6):
                    raise ValueError(f"Transition probabilities for (s={s}, a={a}) "
                                   f"sum to {prob_sum}, not 1.0")
    
    def get_possible_actions(self, s: Any) -> List:
        """Get all possible actions in state s."""
        return self.actions
    
    def get_transition_probs(self, s: Any, a: Any) -> Dict:
        """Get transition probabilities P(s'|s,a) for given state and action."""
        return self.P[s][a]
    
    def get_reward(self, s: Any, a: Any = None) -> float:
        """
        Get reward for state s and action a.
        
        Args:
            s: Current state
            a: Action taken (optional)
            
        Returns:
            Reward value R(s,a) or R(s)
        """
        if (s, a) in self.R:
            return self.R[(s, a)]
        return self.R.get(s, 0.0)
    
    def simulate_trajectory(self, policy: Dict, max_steps: int = 100) -> List[Tuple]:
        """
        Simulate a trajectory following a given policy.
        
        Args:
            policy: Dict[state] = action
            max_steps: Maximum number of steps
            
        Returns:
            List of (state, action, reward, next_state) tuples
        """
        trajectory = []
        s = random.choice(self.states)  # Random initial state
        
        for step in range(max_steps):
            a = policy[s]
            r = self.get_reward(s, a)
            
            # Sample next state according to transition probabilities
            next_states = list(self.P[s][a].keys())
            probs = list(self.P[s][a].values())
            s_next = np.random.choice(next_states, p=probs)
            
            trajectory.append((s, a, r, s_next))
            s = s_next
            
        return trajectory

def value_iteration(mdp: MDP, theta: float = 1e-6, max_iterations: int = 1000) -> Tuple[Dict, Dict]:
    """
    Value Iteration Algorithm for solving MDPs.
    
    This implements the Bellman optimality equation:
    V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]
    
    The algorithm iteratively updates the value function:
    V_{k+1}(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V_k(s')]
    
    Args:
        mdp: MDP instance
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
        
    Returns:
        V: Optimal value function Dict[state] = value
        pi: Optimal policy Dict[state] = action
    """
    # Initialize value function to zero
    V = {s: 0.0 for s in mdp.states}
    
    # Value iteration loop
    for iteration in range(max_iterations):
        delta = 0.0  # Track maximum change in value function
        
        for s in mdp.states:
            v_old = V[s]
            
            # Compute value for each action using Bellman equation
            action_values = []
            for a in mdp.get_possible_actions(s):
                # Get transition probabilities
                trans_probs = mdp.get_transition_probs(s, a)
                
                # Compute expected value: R(s,a) + γ Σ P(s'|s,a) V(s')
                expected_value = 0.0
                for s_next, prob in trans_probs.items():
                    expected_value += prob * (mdp.get_reward(s, a) + mdp.gamma * V[s_next])
                
                action_values.append(expected_value)
            
            # Take maximum over all actions (Bellman optimality)
            V[s] = max(action_values)
            delta = max(delta, abs(v_old - V[s]))
        
        # Check for convergence
        if delta < theta:
            print(f"Value iteration converged after {iteration + 1} iterations")
            break
    
    # Extract optimal policy from value function
    pi = {}
    for s in mdp.states:
        best_action = None
        best_value = float('-inf')
        
        for a in mdp.get_possible_actions(s):
            trans_probs = mdp.get_transition_probs(s, a)
            expected_value = 0.0
            for s_next, prob in trans_probs.items():
                expected_value += prob * (mdp.get_reward(s, a) + mdp.gamma * V[s_next])
            
            if expected_value > best_value:
                best_value = expected_value
                best_action = a
        
        pi[s] = best_action
    
    return V, pi

def policy_iteration(mdp: MDP, max_iterations: int = 1000) -> Tuple[Dict, Dict]:
    """
    Policy Iteration Algorithm for solving MDPs.
    
    Policy iteration alternates between:
    1. Policy Evaluation: Solve V^π(s) = Σ π(a|s) [R(s,a) + γ Σ P(s'|s,a) V^π(s')]
    2. Policy Improvement: π'(s) = argmax_a [R(s,a) + γ Σ P(s'|s,a) V^π(s')]
    
    Args:
        mdp: MDP instance
        max_iterations: Maximum number of iterations
        
    Returns:
        V: Optimal value function
        pi: Optimal policy
    """
    # Initialize random policy
    pi = {s: random.choice(mdp.get_possible_actions(s)) for s in mdp.states}
    V = {s: 0.0 for s in mdp.states}
    
    for iteration in range(max_iterations):
        # Policy Evaluation: Solve for V^π
        for eval_iter in range(100):  # Simple iterative policy evaluation
            for s in mdp.states:
                a = pi[s]
                trans_probs = mdp.get_transition_probs(s, a)
                
                # Bellman equation for current policy
                V[s] = sum(prob * (mdp.get_reward(s, a) + mdp.gamma * V[s_next]) 
                          for s_next, prob in trans_probs.items())
        
        # Policy Improvement
        policy_stable = True
        for s in mdp.states:
            old_action = pi[s]
            best_action = old_action
            best_value = float('-inf')
            
            for a in mdp.get_possible_actions(s):
                trans_probs = mdp.get_transition_probs(s, a)
                expected_value = sum(prob * (mdp.get_reward(s, a) + mdp.gamma * V[s_next]) 
                                   for s_next, prob in trans_probs.items())
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = a
            
            pi[s] = best_action
            if best_action != old_action:
                policy_stable = False
        
        if policy_stable:
            print(f"Policy iteration converged after {iteration + 1} iterations")
            break
    
    return V, pi

class ModelLearner:
    """
    Model-based learning from experience.
    
    This class learns the transition probabilities P(s'|s,a) and rewards R(s,a)
    from observed trajectories. It maintains counts of transitions and rewards
    and estimates probabilities using maximum likelihood estimation.
    """
    
    def __init__(self, states: List, actions: List):
        """
        Initialize model learner.
        
        Args:
            states: List of all possible states
            actions: List of all possible actions
        """
        self.states = states
        self.actions = actions
        
        # Count transition occurrences: (s,a) -> s'
        self.transition_counts = {s: {a: Counter() for a in actions} for s in states}
        
        # Count action occurrences: (s,a)
        self.action_counts = {s: Counter() for s in states}
        
        # Accumulate rewards: (s,a) -> sum of rewards
        self.reward_sums = {s: {a: 0.0 for a in actions} for s in states}
        
        # Count reward observations: (s,a) -> number of observations
        self.reward_counts = {s: {a: 0 for a in actions} for s in states}
    
    def record(self, s: Any, a: Any, s_next: Any, r: float):
        """
        Record an experience tuple (s, a, s', r).
        
        Args:
            s: Current state
            a: Action taken
            s_next: Next state
            r: Reward received
        """
        self.transition_counts[s][a][s_next] += 1
        self.action_counts[s][a] += 1
        self.reward_sums[s][a] += r
        self.reward_counts[s][a] += 1
    
    def get_transition_probs(self) -> Dict:
        """
        Estimate transition probabilities from observed data.
        
        Returns:
            Dict[state][action][next_state] = estimated probability
        """
        P = {s: {a: {} for a in self.actions} for s in self.states}
        
        for s in self.states:
            for a in self.actions:
                total_visits = self.action_counts[s][a]
                
                if total_visits == 0:
                    # If never visited, assume uniform distribution
                    for s_next in self.states:
                        P[s][a][s_next] = 1.0 / len(self.states)
                else:
                    # Maximum likelihood estimate: P(s'|s,a) = count(s,a,s') / count(s,a)
                    for s_next in self.states:
                        P[s][a][s_next] = self.transition_counts[s][a][s_next] / total_visits
        
        return P
    
    def get_rewards(self) -> Dict:
        """
        Estimate rewards from observed data.
        
        Returns:
            Dict[(state, action)] = estimated reward
        """
        R = {}
        
        for s in self.states:
            for a in self.actions:
                count = self.reward_counts[s][a]
                
                if count == 0:
                    R[(s, a)] = 0.0  # Default reward if never observed
                else:
                    # Average reward: R(s,a) = sum(r) / count
                    R[(s, a)] = self.reward_sums[s][a] / count
        
        return R

def create_grid_world_mdp(size: int = 4, goal_reward: float = 1.0, 
                         step_cost: float = -0.01) -> MDP:
    """
    Create a grid world MDP for demonstration.
    
    This creates a size x size grid where:
    - Agent starts at (0,0)
    - Goal is at (size-1, size-1)
    - Actions: up, down, left, right
    - Deterministic transitions with boundary walls
    - Goal gives positive reward, other steps give small negative reward
    
    Args:
        size: Grid size (size x size)
        goal_reward: Reward for reaching the goal
        step_cost: Cost for each step
        
    Returns:
        MDP instance representing the grid world
    """
    states = [(i, j) for i in range(size) for j in range(size)]
    actions = ['up', 'down', 'left', 'right']
    
    # Define transitions
    P = {}
    for s in states:
        P[s] = {}
        for a in actions:
            P[s][a] = {}
            
            # Compute next state
            i, j = s
            if a == 'up':
                i_next = max(0, i - 1)
                j_next = j
            elif a == 'down':
                i_next = min(size - 1, i + 1)
                j_next = j
            elif a == 'left':
                i_next = i
                j_next = max(0, j - 1)
            else:  # right
                i_next = i
                j_next = min(size - 1, j + 1)
            
            s_next = (i_next, j_next)
            P[s][a][s_next] = 1.0  # Deterministic transitions
    
    # Define rewards
    R = {}
    goal_state = (size - 1, size - 1)
    
    for s in states:
        for a in actions:
            if s == goal_state:
                R[(s, a)] = 0.0  # No reward for actions in goal state
            else:
                R[(s, a)] = step_cost
    
    # Add goal reward
    for a in actions:
        R[(goal_state, a)] = goal_reward
    
    return MDP(states, actions, P, R, gamma=0.9)

def plot_value_function(V: Dict, size: int, title: str = "Value Function"):
    """Plot value function for grid world."""
    grid = np.zeros((size, size))
    for (i, j), value in V.items():
        grid[i, j] = value
    
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Add value annotations
    for i in range(size):
        for j in range(size):
            plt.text(j, i, f'{grid[i, j]:.2f}', 
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def demonstrate_mdp_concepts():
    """Demonstrate key MDP concepts with examples."""
    
    print("=" * 60)
    print("MARKOV DECISION PROCESSES DEMONSTRATION")
    print("=" * 60)
    
    # Example 1: Simple 3-state MDP
    print("\n1. SIMPLE 3-STATE MDP EXAMPLE")
    print("-" * 40)
    
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
    R = {
        ('A', 'left'): 0, ('A', 'right'): 5,
        ('B', 'left'): 1, ('B', 'right'): 2,
        ('C', 'left'): 0, ('C', 'right'): 0
    }
    
    mdp = MDP(states, actions, P, R, gamma)
    
    print("MDP Structure:")
    print(f"States: {states}")
    print(f"Actions: {actions}")
    print(f"Discount factor: {gamma}")
    print(f"Transition probabilities: {P}")
    print(f"Rewards: {R}")
    
    # Value Iteration
    print("\nValue Iteration Results:")
    V_vi, pi_vi = value_iteration(mdp)
    print("Optimal Values:", {s: f"{v:.3f}" for s, v in V_vi.items()})
    print("Optimal Policy:", pi_vi)
    
    # Policy Iteration
    print("\nPolicy Iteration Results:")
    V_pi, pi_pi = policy_iteration(mdp)
    print("Optimal Values:", {s: f"{v:.3f}" for s, v in V_pi.items()})
    print("Optimal Policy:", pi_pi)
    
    # Verify both methods give same results
    print("\nVerification:")
    print("Value functions match:", all(abs(V_vi[s] - V_pi[s]) < 1e-6 for s in states))
    print("Policies match:", pi_vi == pi_pi)
    
    # Example 2: Model Learning
    print("\n2. MODEL LEARNING FROM EXPERIENCE")
    print("-" * 40)
    
    learner = ModelLearner(states, actions)
    
    # Simulate experience using the true MDP
    print("Collecting experience...")
    for episode in range(100):
        s = random.choice(states)
        for step in range(10):
            a = random.choice(actions)
            
            # Use true MDP to get next state and reward
            next_states = list(P[s][a].keys())
            probs = list(P[s][a].values())
            s_next = np.random.choice(next_states, p=probs)
            r = R[(s, a)]
            
            learner.record(s, a, s_next, r)
            s = s_next
    
    # Learn model from experience
    learned_P = learner.get_transition_probs()
    learned_R = learner.get_rewards()
    
    print("Learned Transition Probabilities:")
    for s in states:
        for a in actions:
            print(f"P(s'|{s},{a}): {learned_P[s][a]}")
    
    print("\nLearned Rewards:")
    for (s, a), r in learned_R.items():
        print(f"R({s},{a}): {r:.3f}")
    
    # Solve MDP with learned model
    learned_mdp = MDP(states, actions, learned_P, learned_R, gamma)
    V_learned, pi_learned = value_iteration(learned_mdp)
    
    print("\nValue Iteration on Learned Model:")
    print("Optimal Values:", {s: f"{v:.3f}" for s, v in V_learned.items()})
    print("Optimal Policy:", pi_learned)
    
    # Compare with true model
    print("\nModel Learning Accuracy:")
    for s in states:
        print(f"State {s}: True V={V_vi[s]:.3f}, Learned V={V_learned[s]:.3f}, "
              f"Error={abs(V_vi[s] - V_learned[s]):.3f}")
    
    # Example 3: Grid World
    print("\n3. GRID WORLD MDP EXAMPLE")
    print("-" * 40)
    
    grid_mdp = create_grid_world_mdp(size=4)
    V_grid, pi_grid = value_iteration(grid_mdp)
    
    print("Grid World Optimal Values:")
    for i in range(4):
        for j in range(4):
            state = (i, j)
            print(f"V({state}) = {V_grid[state]:.3f}, π({state}) = {pi_grid[state]}")
    
    # Plot value function
    try:
        plot_value_function(V_grid, 4, "Grid World Value Function")
    except ImportError:
        print("Matplotlib not available, skipping plot")

def demonstrate_bellman_equations():
    """Demonstrate Bellman equations with concrete examples."""
    
    print("\n" + "=" * 60)
    print("BELLMAN EQUATIONS DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple 2-state MDP
    states = ['S1', 'S2']
    actions = ['A1', 'A2']
    gamma = 0.9
    
    P = {
        'S1': {
            'A1': {'S1': 0.8, 'S2': 0.2},
            'A2': {'S1': 0.3, 'S2': 0.7}
        },
        'S2': {
            'A1': {'S1': 0.1, 'S2': 0.9},
            'A2': {'S1': 0.6, 'S2': 0.4}
        }
    }
    
    R = {
        ('S1', 'A1'): 1, ('S1', 'A2'): 2,
        ('S2', 'A1'): 3, ('S2', 'A2'): 4
    }
    
    mdp = MDP(states, actions, P, R, gamma)
    V, pi = value_iteration(mdp)
    
    print("MDP Setup:")
    print(f"States: {states}")
    print(f"Actions: {actions}")
    print(f"Discount factor: γ = {gamma}")
    
    print("\nBellman Optimality Equation Verification:")
    print("V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]")
    
    for s in states:
        print(f"\nState {s}:")
        print(f"  V*({s}) = {V[s]:.3f}")
        print(f"  π*({s}) = {pi[s]}")
        
        # Verify Bellman equation
        for a in actions:
            trans_probs = P[s][a]
            expected_value = sum(prob * (R[(s, a)] + gamma * V[s_next]) 
                               for s_next, prob in trans_probs.items())
            print(f"  Q({s},{a}) = {expected_value:.3f}")
        
        # Check that optimal action gives maximum Q-value
        best_q = max(sum(prob * (R[(s, a)] + gamma * V[s_next]) 
                        for s_next, prob in P[s][a].items()) 
                    for a in actions)
        print(f"  max_a Q({s},a) = {best_q:.3f}")
        print(f"  V*({s}) = max_a Q({s},a): {abs(V[s] - best_q) < 1e-6}")

if __name__ == "__main__":
    # Run comprehensive demonstrations
    demonstrate_mdp_concepts()
    demonstrate_bellman_equations()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("This demonstration shows:")
    print("1. MDP framework with states, actions, transitions, and rewards")
    print("2. Value iteration algorithm for finding optimal policies")
    print("3. Policy iteration as an alternative approach")
    print("4. Model learning from experience")
    print("5. Bellman equations in action")
    print("6. Grid world example with visualization")
    print("\nKey insights:")
    print("- Value iteration converges to optimal value function")
    print("- Policy iteration often converges faster")
    print("- Model learning enables planning without known dynamics")
    print("- Bellman equations provide the foundation for all MDP algorithms") 
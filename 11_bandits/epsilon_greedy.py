"""
Epsilon-Greedy Algorithm Implementation

This module implements the epsilon-greedy algorithm for multi-armed bandits.
The algorithm balances exploration (random selection) and exploitation (greedy selection)
based on a fixed epsilon parameter.
"""

import random
import numpy as np

def epsilon_greedy(arms, epsilon, T):
    """
    Epsilon-greedy algorithm for multi-armed bandits.
    
    Args:
        arms: List of arm objects with pull() method
        epsilon: Exploration probability
        T: Number of time steps
    
    Returns:
        empirical_means: List of empirical means for each arm
    """
    n_arms = len(arms)
    empirical_means = [0] * n_arms
    pulls = [0] * n_arms
    
    for t in range(T):
        if random.random() < epsilon:
            # Exploration: choose random arm
            action = random.randint(0, n_arms - 1)
        else:
            # Exploitation: choose best empirical arm
            action = np.argmax(empirical_means)
        
        # Pull arm and observe reward
        reward = arms[action].pull()
        
        # Update empirical mean
        pulls[action] += 1
        empirical_means[action] = ((empirical_means[action] * (pulls[action] - 1) + reward) / pulls[action])
    
    return empirical_means 
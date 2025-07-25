import numpy as np

def qlearning(episodes):
    """
    Q-learning placeholder for Mountain Car.
    Args:
        episodes (int): Number of episodes to run.
    Returns:
        q (np.ndarray): Q-value table.
        steps_per_episode (np.ndarray): Steps per episode.
    """
    alpha = 0.05
    gamma = 0.99
    num_states = 100
    num_actions = 2
    actions = np.array([-1, 1])
    q = np.zeros((num_states, num_actions))
    # TODO: Implement Q-learning main loop
    raise NotImplementedError('Q-learning algorithm not implemented.') 
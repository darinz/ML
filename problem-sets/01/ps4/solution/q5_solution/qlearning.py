import numpy as np
from mountain_car import mountain_car

def qlearning(episodes):
    alpha = 0.05
    gamma = 0.99
    num_states = 100
    num_actions = 2
    actions = np.array([-1, 1])
    q = np.zeros((num_states, num_actions))
    steps_per_episode = np.zeros(episodes, dtype=int)
    for i in range(episodes):
        x = np.array([0.0, -np.pi/6])
        x, s, absorb = mountain_car(x, 0)
        a = np.argmax(q[s-1, :])
        if q[s-1, 0] == q[s-1, 1]:
            a = np.random.randint(num_actions)
        steps = 0
        while not absorb:
            x, sn, absorb = mountain_car(x, actions[a])
            reward = -int(absorb == 0)
            an = np.argmax(q[sn-1, :])
            if q[sn-1, 0] == q[sn-1, 1]:
                an = np.random.randint(num_actions)
            q[s-1, a] = (1 - alpha) * q[s-1, a] + alpha * (reward + gamma * q[sn-1, an])
            a = an
            s = sn
            steps += 1
        steps_per_episode[i] = steps
    return q, steps_per_episode 
import numpy as np

def mountain_car(x, a):
    x_next = np.zeros(2)
    x_next[1] = x[1] + 0.001 * a - 0.0025 * np.cos(3 * x[0])
    x_next[0] = x[0] + x_next[1]
    absorb = 0
    if x_next[0] < -1.2:
        x_next[1] = 0
    if x_next[0] > 0.5:
        absorb = 1
    x_next[0] = np.clip(x_next[0], -1.2, 0.5)
    x_next[1] = np.clip(x_next[1], -0.07, 0.07)
    s_idx = (10 * np.floor(10 * (x_next[0] + 1.2) / (1.7 + 1e-10)) +
             np.floor(10 * (x_next[1] + 0.07) / (0.14 + 1e-10)) + 1).astype(int)
    return x_next, s_idx, absorb 
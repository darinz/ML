import numpy as np
import matplotlib.pyplot as plt
from qlearning import qlearning

all_ep_steps = []
for i in range(10):
    q, ep_steps = qlearning(10000)
    all_ep_steps.append(ep_steps)
all_ep_steps = np.array(all_ep_steps)
mean_steps = np.mean(all_ep_steps, axis=0)
mean_steps_reshaped = np.mean(mean_steps.reshape(20, 500), axis=1)
plt.plot(mean_steps_reshaped)
plt.xlabel('Episode block')
plt.ylabel('Average steps per episode')
plt.title('Learning Curve')
plt.show() 
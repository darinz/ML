import numpy as np
import matplotlib.pyplot as plt
from lwlr import lwlr

def plot_lwlr(X, y, tau, resolution):
    pred = np.zeros((resolution, resolution))
    for i in range(resolution):
        for j in range(resolution):
            x1 = 2 * i / (resolution - 1) - 1
            x2 = 2 * j / (resolution - 1) - 1
            x = np.array([x1, x2])
            pred[j, i] = lwlr(X, y, x, tau)
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(pred, vmin=-0.4, vmax=1.3, origin='lower')
    plt.scatter((resolution/2)*(1+X[y==0,0])+0.5, (resolution/2)*(1+X[y==0,1])+0.5, c='b', marker='o', label='y=0')
    plt.scatter((resolution/2)*(1+X[y==1,0])+0.5, (resolution/2)*(1+X[y==1,1])+0.5, c='r', marker='x', label='y=1')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'tau = {tau}', fontsize=18)
    plt.show() 
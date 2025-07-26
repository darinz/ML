import numpy as np
import matplotlib.pyplot as plt

def plot_mountain_car(x):
    plt.clf()
    plt.plot(np.arange(-1.2, 0.5+0.1, 0.1), 0.3*np.sin(3*np.arange(-1.2, 0.5+0.1, 0.1)), 'k-')
    theta = np.arctan2(3*0.3*np.cos(3*x[0]), 1.0)
    y = 0.3*np.sin(3*x[0])
    car = np.array([[-0.05, 0.05, 0.05, -0.05, -0.05],
                   [0.05, 0.05, 0.01, 0.01, 0.05]])
    t = np.arange(0, 2*np.pi+0.5, 0.5)
    fwheel = np.array([0.035 + 0.01*np.cos(t), 0.01 + 0.01*np.sin(t)])
    rwheel = np.array([-0.035 + 0.01*np.cos(t), 0.01 + 0.01*np.sin(t)])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    car = R @ car + np.array([[x[0]], [y]])
    fwheel = R @ fwheel + np.array([[x[0]], [y]])
    rwheel = R @ rwheel + np.array([[x[0]], [y]])
    plt.plot(car[0], car[1])
    plt.plot(fwheel[0], fwheel[1])
    plt.plot(rwheel[0], rwheel[1])
    plt.axis([-1.3, 0.6, -0.4, 0.4])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show() 
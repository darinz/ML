import numpy as np

def load_data():
    X = np.loadtxt('x.dat')
    y = np.loadtxt('y.dat')
    theta_true = np.loadtxt('theta.dat')
    return X, y, theta_true 
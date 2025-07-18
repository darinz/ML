import numpy as np

def load_data():
    X = np.loadtxt('data/x.dat')
    y = np.loadtxt('data/y.dat')
    return X, y 
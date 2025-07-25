import numpy as np

def pca(X):
    U, S, Vt = np.linalg.svd(X @ X.T)
    return U 
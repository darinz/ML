import numpy as np

def ica(X):
    n, m = X.shape
    chunk = 100
    alpha = 0.0005
    W = np.eye(n)
    for iter in range(10):
        print(iter + 1)
        X = X[:, np.random.permutation(m)]
        for i in range(m // chunk):
            Xc = X[:, i*chunk:(i+1)*chunk]
            WXc = W @ Xc
            dW = (1 - 2 / (1 + np.exp(-WXc))) @ Xc.T + chunk * np.linalg.inv(W.T)
            W = W + alpha * dW
    return W 
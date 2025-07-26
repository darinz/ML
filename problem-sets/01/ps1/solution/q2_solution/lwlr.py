import numpy as np
from scipy.special import expit as sigmoid

def lwlr(X_train, y_train, x, tau):
    m, n = X_train.shape
    theta = np.zeros(n)
    # Compute weights
    w = np.exp(-np.sum((X_train - x) ** 2, axis=1) / (2 * tau))
    reg_lambda = 1e-4
    g = np.ones(n)
    while np.linalg.norm(g) > 1e-6:
        h = sigmoid(X_train @ theta)
        g = X_train.T @ (w * (y_train - h)) - reg_lambda * theta
        D = w * h * (1 - h)
        H = -X_train.T @ (D[:, None] * X_train) - reg_lambda * np.eye(n)
        theta = theta - np.linalg.solve(H, g)
    y_pred = sigmoid(x @ theta)
    return int(y_pred > 0.5) 
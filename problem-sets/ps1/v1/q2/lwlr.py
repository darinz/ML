import numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import minimize

def lwlr(X_train, y_train, x, tau):
    # Compute weights for each training example
    diff = X_train - x
    w = np.exp(-np.sum(diff**2, axis=1) / (2 * tau**2))
    
    def weighted_logistic_loss(theta):
        h = sigmoid(X_train @ theta)
        reg = 0.0001 * np.sum(theta**2) / 2
        loss = -np.sum(w * (y_train * np.log(h + 1e-12) + (1 - y_train) * np.log(1 - h + 1e-12)))
        return loss + reg
    
    def weighted_logistic_grad(theta):
        h = sigmoid(X_train @ theta)
        z = w * (y_train - h)
        grad = -X_train.T @ z - 0.0001 * theta
        return grad
    
    theta0 = np.zeros(X_train.shape[1])
    res = minimize(weighted_logistic_loss, theta0, jac=weighted_logistic_grad, method='BFGS')
    theta = res.x
    y_pred = sigmoid(x @ theta)
    return int(y_pred > 0.5) 
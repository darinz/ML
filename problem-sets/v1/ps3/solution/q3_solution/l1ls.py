import numpy as np

def l1ls(X, y, lambda_):
    m, n = X.shape
    theta = np.zeros(n)
    old_theta = np.ones(n)

    while np.linalg.norm(theta - old_theta) > 1e-5:
        old_theta = theta.copy()
        for i in range(n):
            # compute possible values for theta_i
            theta[i] = 0
            X_col = X[:, i]
            residual = X @ theta - y
            denom = X_col @ X_col
            if denom == 0:
                continue  # avoid division by zero
            theta_i1 = max((-X_col @ residual - lambda_) / denom, 0)
            theta_i2 = min((-X_col @ residual + lambda_) / denom, 0)

            # get objective value for both possible thetas
            theta[i] = theta_i1
            obj_theta1 = 0.5 * np.linalg.norm(X @ theta - y) ** 2 + lambda_ * np.linalg.norm(theta, 1)
            theta[i] = theta_i2
            obj_theta2 = 0.5 * np.linalg.norm(X @ theta - y) ** 2 + lambda_ * np.linalg.norm(theta, 1)

            # pick the theta which minimizes the objective
            if obj_theta1 < obj_theta2:
                theta[i] = theta_i1
            else:
                theta[i] = theta_i2
    return theta 
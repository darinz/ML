from sklearn.linear_model import Lasso
import numpy as np

def l1ls(X, y, lambda_):
    """
    L1-regularized least squares (Lasso) solver.
    Args:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        lambda_: Regularization parameter (float)
    Returns:
        theta: Coefficient vector (numpy array)
    """
    model = Lasso(alpha=lambda_, fit_intercept=False, max_iter=10000)
    model.fit(X, y)
    theta = model.coef_
    return theta 
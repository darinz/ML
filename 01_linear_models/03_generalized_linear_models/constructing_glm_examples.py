# Examples for GLM Construction: OLS and Logistic Regression

# --- Ordinary Least Squares (OLS) Example ---
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([2, 3, 4])
model = LinearRegression().fit(X, y)
print('OLS coefficients:', model.coef_)
print('OLS intercept:', model.intercept_)

# --- Logistic Regression Example ---
from sklearn.linear_model import LogisticRegression

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([0, 1, 1])
model = LogisticRegression().fit(X, y)
print('Logistic Regression coefficients:', model.coef_)
print('Logistic Regression intercept:', model.intercept_)
print('Predicted probabilities:', model.predict_proba(X)) 
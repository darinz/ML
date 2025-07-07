# Model Selection and Bayesian Methods Examples
# This file contains Python code for model selection, cross-validation, MLE, MAP, and Bayesian inference

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import make_classification

# ---
# Polynomial Model Selection Example
print("\n--- Polynomial Model Selection Example ---")
def true_func(x):
    return np.sin(2 * np.pi * x)

np.random.seed(0)
x = np.sort(np.random.rand(100))
y = true_func(x) + np.random.randn(100) * 0.1

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)
train_errors = []
val_errors = []
for degree in range(1, 11):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(x_train[:, None])
    X_val_poly = poly.transform(x_val[:, None])
    model = LinearRegression().fit(X_train_poly, y_train)
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))
print("Train errors:", train_errors)
print("Validation errors:", val_errors)

# ---
# Hold-out Cross Validation Example
print("\n--- Hold-out Cross Validation Example ---")
X, y = np.random.randn(100, 5), np.random.randn(100)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
model = Ridge(alpha=1.0).fit(X_train, y_train)
y_pred = model.predict(X_val)
val_error = mean_squared_error(y_val, y_pred)
print("Validation error:", val_error)

# ---
# k-Fold Cross Validation Example
print("\n--- k-Fold Cross Validation Example ---")
kf = KFold(n_splits=10)
model = Ridge(alpha=1.0)
scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
print("Mean CV error:", -np.mean(scores))

# ---
# Maximum Likelihood Estimation (MLE) Example
print("\n--- Maximum Likelihood Estimation (MLE) Example ---")
X_mle, y_mle = make_classification(n_samples=200, n_features=5, random_state=42)
mle_model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
mle_model.fit(X_mle, y_mle)
print("MLE coefficients:", mle_model.coef_)

# ---
# Bayesian Linear Regression (Posterior Predictive) Example
print("\n--- Bayesian Linear Regression (Posterior Predictive) Example ---")
np.random.seed(0)
X_blr = np.random.randn(100, 1)
y_blr = 3 * X_blr[:, 0] + np.random.randn(100) * 0.5
blr_model = BayesianRidge()
blr_model.fit(X_blr, y_blr)
x_new = np.array([[1.5]])
y_mean, y_std = blr_model.predict(x_new, return_std=True)
print("Posterior mean:", y_mean, "Stddev:", y_std)

# ---
# MAP Estimation (Ridge Regression) Example
print("\n--- MAP Estimation (Ridge Regression) Example ---")
ridge = Ridge(alpha=1.0)
ridge.fit(X_blr, y_blr)
print("MAP coefficients:", ridge.coef_)

# ---
# Bayesian Logistic Regression (MAP) Example
print("\n--- Bayesian Logistic Regression (MAP) Example ---")
logreg = LogisticRegression(penalty='l2', C=1.0)
logreg.fit(X_blr, y_blr > np.median(y_blr))
print("MAP coefficients (logistic regression):", logreg.coef_) 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# --- 1. Simulate Data Generation ---

def h_star(x):
    """True function (quadratic)"""
    return 2 * x**2 + 0.5

sigma = 0.2  # Standard deviation of noise
n_train = 8  # Number of training points
n_test = 1000  # Number of test points
np.random.seed(42)

x_train = np.random.rand(n_train)
y_train = h_star(x_train) + np.random.normal(0, sigma, n_train)
x_test = np.linspace(0, 1, n_test)
y_test_true = h_star(x_test)

# --- 2. Fit Models and Compute Predictions ---

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Linear model
linear_model = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
y_pred_linear = linear_model.predict(x_test.reshape(-1, 1))

# 5th-degree polynomial model
poly5_model = make_pipeline(PolynomialFeatures(5), LinearRegression())
poly5_model.fit(x_train.reshape(-1, 1), y_train)
y_pred_poly5 = poly5_model.predict(x_test.reshape(-1, 1))

# Quadratic model (degree 2)
poly2_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly2_model.fit(x_train.reshape(-1, 1), y_train)
y_pred_poly2 = poly2_model.predict(x_test.reshape(-1, 1))

# --- 3. Mean Squared Error (MSE) ---
mse_linear = mse(y_test_true, y_pred_linear)
mse_poly2 = mse(y_test_true, y_pred_poly2)
mse_poly5 = mse(y_test_true, y_pred_poly5)
print(f"MSE (Linear): {mse_linear:.4f}")
print(f"MSE (Quadratic): {mse_poly2:.4f}")
print(f"MSE (5th-degree): {mse_poly5:.4f}")

# --- 4. Bias, Variance, and Noise Decomposition ---
n_repeats = 500
x0 = 0.5  # Test at a single point

preds_poly5 = []
for _ in range(n_repeats):
    x_train = np.random.rand(n_train)
    y_train = h_star(x_train) + np.random.normal(0, sigma, n_train)
    model = make_pipeline(PolynomialFeatures(5), LinearRegression())
    model.fit(x_train.reshape(-1, 1), y_train)
    preds_poly5.append(model.predict(np.array([[x0]]))[0])

preds_poly5 = np.array(preds_poly5)
true_val = h_star(x0)
avg_pred = np.mean(preds_poly5)
bias2 = (true_val - avg_pred) ** 2
variance = np.var(preds_poly5)
noise = sigma ** 2
expected_mse = bias2 + variance + noise

print(f"\nAt x = {x0}")
print(f"Bias^2: {bias2:.4f}")
print(f"Variance: {variance:.4f}")
print(f"Noise: {noise:.4f}")
print(f"Expected MSE: {expected_mse:.4f}")

# --- 5. Visualizing Bias-Variance Tradeoff ---
degrees = range(1, 10)
bias2_list = []
variance_list = []
mse_list = []

for deg in degrees:
    preds = []
    for _ in range(n_repeats):
        x_train = np.random.rand(n_train)
        y_train = h_star(x_train) + np.random.normal(0, sigma, n_train)
        model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        model.fit(x_train.reshape(-1, 1), y_train)
        preds.append(model.predict(np.array([[x0]]))[0])
    preds = np.array(preds)
    avg_pred = np.mean(preds)
    bias2 = (true_val - avg_pred) ** 2
    variance = np.var(preds)
    bias2_list.append(bias2)
    variance_list.append(variance)
    mse_list.append(bias2 + variance + noise)

plt.plot(degrees, bias2_list, label="Bias$^2$")
plt.plot(degrees, variance_list, label="Variance")
plt.plot(degrees, mse_list, label="Total Error (MSE)")
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel(f"Error at x = {x0}")
plt.legend()
plt.title("Bias-Variance Tradeoff")
plt.show() 
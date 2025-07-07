import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# --- 1. Model-wise Double Descent (Varying Model Complexity) ---

def simulate_modelwise_double_descent(n_train=100, n_test=1000, max_degree=30, noise_std=0.5, random_seed=42):
    np.random.seed(random_seed)
    x_train = np.random.uniform(-1, 1, n_train)
    x_test = np.linspace(-1, 1, n_test)
    # True function: cubic
    def f(x): return 1.5 * x**3 - 0.5 * x**2 + 0.2 * x + 1
    y_train = f(x_train) + np.random.normal(0, noise_std, n_train)
    y_test = f(x_test)
    test_errors = []
    degrees = range(1, max_degree + 1)
    for deg in degrees:
        model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
        model.fit(x_train.reshape(-1, 1), y_train)
        y_pred = model.predict(x_test.reshape(-1, 1))
        test_errors.append(np.mean((y_pred - y_test) ** 2))
    plt.figure(figsize=(7, 4))
    plt.plot(degrees, test_errors, marker='o')
    plt.xlabel('Model Complexity (Polynomial Degree)')
    plt.ylabel('Test Error (MSE)')
    plt.title('Model-wise Double Descent (Polynomial Regression)')
    plt.grid(True)
    plt.show()

# --- 2. Sample-wise Double Descent (Varying Number of Samples) ---

def simulate_samplewise_double_descent(d=50, n_min=10, n_max=100, step=2, noise_std=0.5, reg_strength=None, random_seed=42):
    np.random.seed(random_seed)
    # True beta
    beta = np.random.randn(d)
    beta /= np.linalg.norm(beta)
    test_errors = []
    ns = np.arange(n_min, n_max + 1, step)
    n_test = 1000
    X_test = np.random.randn(n_test, d)
    y_test = X_test @ beta + np.random.normal(0, noise_std, n_test)
    for n in ns:
        X_train = np.random.randn(n, d)
        y_train = X_train @ beta + np.random.normal(0, noise_std, n)
        if reg_strength is None:
            model = LinearRegression()
        else:
            model = Ridge(alpha=reg_strength)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_errors.append(np.mean((y_pred - y_test) ** 2))
    plt.figure(figsize=(7, 4))
    plt.plot(ns, test_errors, marker='o', label=f'Reg={reg_strength}')
    plt.xlabel('Number of Training Samples (n)')
    plt.ylabel('Test Error (MSE)')
    plt.title('Sample-wise Double Descent (Linear Regression)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 3. Effect of Regularization ---

def plot_samplewise_with_regularization():
    d = 50
    n_min, n_max = 10, 100
    regs = [None, 1e-4, 1e-2, 1e-1, 1, 10]
    plt.figure(figsize=(8, 5))
    for reg in regs:
        np.random.seed(42)
        beta = np.random.randn(d)
        beta /= np.linalg.norm(beta)
        test_errors = []
        ns = np.arange(n_min, n_max + 1, 2)
        n_test = 1000
        X_test = np.random.randn(n_test, d)
        y_test = X_test @ beta + np.random.normal(0, 0.5, n_test)
        for n in ns:
            X_train = np.random.randn(n, d)
            y_train = X_train @ beta + np.random.normal(0, 0.5, n)
            if reg is None:
                model = LinearRegression()
            else:
                model = Ridge(alpha=reg)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_errors.append(np.mean((y_pred - y_test) ** 2))
        label = f'Reg={reg}' if reg is not None else 'No Reg'
        plt.plot(ns, test_errors, marker='.', label=label)
    plt.xlabel('Number of Training Samples (n)')
    plt.ylabel('Test Error (MSE)')
    plt.title('Sample-wise Double Descent with Regularization')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Run all demonstrations ---
if __name__ == '__main__':
    print("Model-wise double descent (polynomial regression):")
    simulate_modelwise_double_descent()
    print("Sample-wise double descent (linear regression):")
    simulate_samplewise_double_descent()
    print("Sample-wise double descent with regularization:")
    plot_samplewise_with_regularization() 
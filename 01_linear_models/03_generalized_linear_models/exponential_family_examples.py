import numpy as np
from scipy.stats import norm

# --- Generic Exponential Family Function ---
def exponential_family_p(y, eta, T, a, b):
    """
    Generic exponential family probability calculation.
    y: observed value
    eta: natural parameter (can be vector)
    T: sufficient statistic function
    a: log partition function
    b: base measure function
    """
    return b(y) * np.exp(np.dot(eta, T(y)) - a(eta))

# --- Bernoulli Distribution in Exponential Family Form ---
def bernoulli_p(y, phi):
    """Bernoulli PMF."""
    return phi**y * (1 - phi)**(1 - y)

def bernoulli_exp_family(y, phi):
    """Bernoulli in exponential family form."""
    eta = np.log(phi / (1 - phi))
    a = np.log(1 + np.exp(eta))
    return np.exp(eta * y - a)

def T_bernoulli(y):
    return y

def a_bernoulli(eta):
    return np.log(1 + np.exp(eta))

def b_bernoulli(y):
    return 1

# Example usage for Bernoulli:
if __name__ == "__main__":
    y = 1
    phi = 0.7
    print('Bernoulli PMF:', bernoulli_p(y, phi))
    print('Exponential family form (Bernoulli):', bernoulli_exp_family(y, phi))

# --- Gaussian Distribution in Exponential Family Form ---
def gaussian_p(y, mu):
    """Gaussian PDF with sigma^2 = 1."""
    return norm.pdf(y, loc=mu, scale=1)

def gaussian_exp_family(y, mu):
    """Gaussian in exponential family form."""
    eta = mu
    a = mu**2 / 2
    b = (1 / np.sqrt(2 * np.pi)) * np.exp(-y**2 / 2)
    return b * np.exp(eta * y - a)

def eta_gaussian(mu):
    return mu

def T_gaussian(y):
    return y

def a_gaussian(eta):
    return eta**2 / 2

def b_gaussian(y):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-y**2 / 2)

# Example usage for Gaussian:
if __name__ == "__main__":
    y = 0.5
    mu = 1.0
    print('Gaussian PDF:', gaussian_p(y, mu))
    print('Exponential family form (Gaussian):', gaussian_exp_family(y, mu)) 
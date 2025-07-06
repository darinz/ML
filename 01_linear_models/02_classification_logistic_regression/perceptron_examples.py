import numpy as np

def perceptron_threshold(z):
    """Threshold function: returns 1 if z >= 0, else 0."""
    return 1 if z >= 0 else 0

# Vectorized version for arrays
def perceptron_threshold_vec(z):
    """Vectorized threshold function for numpy arrays."""
    return np.vectorize(perceptron_threshold)(z)

def predict(theta, x):
    """Compute perceptron prediction for input x and weights theta."""
    z = np.dot(theta, x)
    return perceptron_threshold(z)

def perceptron_update(theta, x, y, alpha):
    """Update theta using the perceptron learning rule.
    Args:
        theta: parameter vector (numpy array)
        x: input feature vector (numpy array, including bias term)
        y: true label (0 or 1)
        alpha: learning rate (float)
    Returns:
        Updated theta (numpy array)
    """
    prediction = predict(theta, x)
    theta = theta + alpha * (y - prediction) * x
    return theta

if __name__ == "__main__":
    # Example usage:
    alpha = 0.1
    theta = np.zeros(3)  # For 2 features + bias
    x = np.array([1, 2, 3])  # Example input (with bias term as x[0]=1)
    y = 1  # True label

    # Perform one update
    theta = perceptron_update(theta, x, y, alpha)
    print("Updated theta:", theta) 
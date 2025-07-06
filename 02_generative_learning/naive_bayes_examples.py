import numpy as np
from collections import Counter

# --- Bernoulli Naive Bayes Parameter Estimation ---
def estimate_naive_bayes_params(X, y):
    """
    Estimate parameters for Bernoulli Naive Bayes.
    X: shape (n_samples, n_features), binary (0/1)
    y: shape (n_samples,), binary (0/1)
    Returns: phi_j_y1, phi_j_y0, phi_y
    """
    n_samples, n_features = X.shape
    phi_y = np.mean(y)  # fraction of samples with y=1
    phi_j_y1 = np.sum(X[y == 1], axis=0) / np.sum(y == 1)
    phi_j_y0 = np.sum(X[y == 0], axis=0) / np.sum(y == 0)
    return phi_j_y1, phi_j_y0, phi_y

# Example usage (Bernoulli):
X = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 1],
])
y = np.array([1, 1, 0, 0])
phi_j_y1, phi_j_y0, phi_y = estimate_naive_bayes_params(X, y)
print("phi_j|y=1:", phi_j_y1)
print("phi_j|y=0:", phi_j_y0)
print("phi_y:", phi_y)

# --- Prediction (Posterior Calculation) ---
def predict_naive_bayes(x, phi_j_y1, phi_j_y0, phi_y):
    """
    Predict posterior probability p(y=1|x) for Bernoulli Naive Bayes.
    x: binary feature vector (1D)
    phi_j_y1, phi_j_y0: estimated probabilities for each feature
    phi_y: prior probability of y=1
    """
    p_x_given_y1 = np.prod(phi_j_y1 ** x * (1 - phi_j_y1) ** (1 - x))
    p_x_given_y0 = np.prod(phi_j_y0 ** x * (1 - phi_j_y0) ** (1 - x))
    num = p_x_given_y1 * phi_y
    denom = num + p_x_given_y0 * (1 - phi_y)
    return num / denom if denom > 0 else 0.5  # avoid division by zero

# Example prediction:
x_new = np.array([1, 0, 1])
p_y1_given_x = predict_naive_bayes(x_new, phi_j_y1, phi_j_y0, phi_y)
print("P(y=1|x_new):", p_y1_given_x)

# --- Laplace Smoothing (Bernoulli) ---
def laplace_smoothing_bernoulli(X, y):
    """
    Laplace smoothing for Bernoulli Naive Bayes.
    Adds 1 to numerator, 2 to denominator.
    """
    n_samples, n_features = X.shape
    phi_j_y1 = (1 + np.sum(X[y == 1], axis=0)) / (2 + np.sum(y == 1))
    phi_j_y0 = (1 + np.sum(X[y == 0], axis=0)) / (2 + np.sum(y == 0))
    return phi_j_y1, phi_j_y0

# Example usage (Laplace smoothing):
phi_j_y1_smooth, phi_j_y0_smooth = laplace_smoothing_bernoulli(X, y)
print("Laplace-smoothed phi_j|y=1:", phi_j_y1_smooth)
print("Laplace-smoothed phi_j|y=0:", phi_j_y0_smooth)

# --- Multinomial Naive Bayes Parameter Estimation and Laplace Smoothing ---
def multinomial_naive_bayes_params(X_multi, y_multi, vocab_size):
    """
    Estimate parameters for Multinomial Naive Bayes with Laplace smoothing.
    X_multi: list of arrays, each array is the sequence of word indices for a document
    y_multi: array of labels (0/1)
    vocab_size: number of unique words (|V|)
    Returns: phi_k_y1, phi_k_y0
    """
    word_counts_y1 = np.zeros(vocab_size)
    word_counts_y0 = np.zeros(vocab_size)
    total_words_y1 = 0
    total_words_y0 = 0
    for doc, label in zip(X_multi, y_multi):
        counts = Counter(doc)
        if label == 1:
            for idx, cnt in counts.items():
                word_counts_y1[idx] += cnt
            total_words_y1 += len(doc)
        else:
            for idx, cnt in counts.items():
                word_counts_y0[idx] += cnt
            total_words_y0 += len(doc)
    phi_k_y1 = (1 + word_counts_y1) / (vocab_size + total_words_y1)
    phi_k_y0 = (1 + word_counts_y0) / (vocab_size + total_words_y0)
    return phi_k_y1, phi_k_y0

# Example usage (Multinomial):
vocab_size = 3
X_multi = [np.array([0,1,0]), np.array([2,0]), np.array([1,1]), np.array([2])]
y_multi = np.array([1,1,0,0])
phi_k_y1, phi_k_y0 = multinomial_naive_bayes_params(X_multi, y_multi, vocab_size)
print("Multinomial phi_k|y=1:", phi_k_y1)
print("Multinomial phi_k|y=0:", phi_k_y0) 
import numpy as np

# Softmax function with numerical stability

def softmax(logits):
    """
    Compute softmax probabilities for each row of logits.
    Args:
        logits: np.ndarray of shape (n_samples, n_classes)
    Returns:
        probs: np.ndarray of shape (n_samples, n_classes)
    """
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Cross-entropy loss function

def cross_entropy_loss(probs, y_true):
    """
    Compute the mean cross-entropy loss.
    Args:
        probs: np.ndarray of shape (n_samples, n_classes), softmax probabilities
        y_true: np.ndarray of shape (n_samples,), true class indices (0-based)
    Returns:
        loss: float, mean cross-entropy loss
    """
    n = y_true.shape[0]
    return -np.log(probs[np.arange(n), y_true]).mean()

# Example usage
if __name__ == "__main__":
    # Example logits for a single sample with 3 classes
    logits = np.array([[2, 1, 0]])
    probs = softmax(logits)
    y_true = np.array([1])  # class index 1 (second class)
    loss = cross_entropy_loss(probs, y_true)
    print('Probabilities:', probs)
    print('Loss:', loss) 
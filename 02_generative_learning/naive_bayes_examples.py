"""
Naive Bayes Classification Implementation

This module implements Naive Bayes classifiers for both Bernoulli and Multinomial
event models, commonly used for text classification and high-dimensional discrete data.

Key Concepts:
- Models p(x|y) using conditional independence assumption: p(x₁,...,xₙ|y) = ∏ᵢ p(xᵢ|y)
- Bernoulli model: Binary features (word present/absent)
- Multinomial model: Count features (word frequencies)
- Laplace smoothing prevents zero probability estimates
- Uses Bayes' rule for prediction: p(y|x) = p(x|y)p(y)/p(x)

Mathematical Foundation:
- Bernoulli: p(x|y) = ∏ⱼ φⱼ|ᵧˣʲ(1-φⱼ|ᵧ)¹⁻ˣʲ
- Multinomial: p(x|y) = ∏ⱼ φₖ|ᵧˣʲ (where xⱼ is word index)
- Laplace smoothing: φⱼ = (1 + countⱼ) / (2 + total_count)

Author: Machine Learning Course Materials
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def estimate_naive_bayes_params(X, y):
    """
    Estimate parameters for Bernoulli Naive Bayes using maximum likelihood.
    
    This function computes:
    - φᵧ = p(y=1): prior probability of class 1
    - φⱼ|ᵧ₌₁ = p(xⱼ=1|y=1): probability of feature j being 1 in class 1
    - φⱼ|ᵧ₌₀ = p(xⱼ=1|y=0): probability of feature j being 1 in class 0
    
    The MLE estimates are:
    - φᵧ = (1/n) Σᵢ 1{y⁽ⁱ⁾ = 1}
    - φⱼ|ᵧ₌₁ = Σᵢ 1{xⱼ⁽ⁱ⁾ = 1 ∧ y⁽ⁱ⁾ = 1} / Σᵢ 1{y⁽ⁱ⁾ = 1}
    - φⱼ|ᵧ₌₀ = Σᵢ 1{xⱼ⁽ⁱ⁾ = 1 ∧ y⁽ⁱ⁾ = 0} / Σᵢ 1{y⁽ⁱ⁾ = 0}
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Binary feature matrix (0/1 values)
    y : array-like, shape (n_samples,)
        Binary labels (0 or 1)
    
    Returns:
    --------
    tuple
        (φⱼ|ᵧ₌₁, φⱼ|ᵧ₌₀, φᵧ) - estimated parameters
    
    Example:
    --------
    >>> X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]])
    >>> y = np.array([1, 1, 0, 0])
    >>> phi_j_y1, phi_j_y0, phi_y = estimate_naive_bayes_params(X, y)
    >>> print(f"φⱼ|ᵧ₌₁: {phi_j_y1}")
    >>> print(f"φⱼ|ᵧ₌₀: {phi_j_y0}")
    >>> print(f"φᵧ: {phi_y:.3f}")
    """
    n_samples, n_features = X.shape
    
    # Estimate class prior φᵧ
    phi_y = np.mean(y)
    
    # Estimate feature probabilities for each class
    phi_j_y1 = np.sum(X[y == 1], axis=0) / np.sum(y == 1)
    phi_j_y0 = np.sum(X[y == 0], axis=0) / np.sum(y == 0)
    
    return phi_j_y1, phi_j_y0, phi_y

def predict_naive_bayes(x, phi_j_y1, phi_j_y0, phi_y):
    """
    Predict posterior probability p(y=1|x) for Bernoulli Naive Bayes.
    
    Uses Bayes' rule: p(y=1|x) = p(x|y=1)p(y=1) / p(x)
    where p(x|y) = ∏ⱼ φⱼ|ᵧˣʲ(1-φⱼ|ᵧ)¹⁻ˣʲ (independence assumption)
    
    Parameters:
    -----------
    x : array-like, shape (n_features,)
        Binary feature vector (0/1 values)
    phi_j_y1 : array-like, shape (n_features,)
        φⱼ|ᵧ₌₁ for each feature
    phi_j_y0 : array-like, shape (n_features,)
        φⱼ|ᵧ₌₀ for each feature
    phi_y : float
        Prior probability φᵧ = p(y=1)
    
    Returns:
    --------
    float
        Posterior probability p(y=1|x)
    
    Example:
    --------
    >>> x_new = np.array([1, 0, 1])  # features: [buy, cheap, now]
    >>> prob = predict_naive_bayes(x_new, phi_j_y1, phi_j_y0, phi_y)
    >>> print(f"p(y=1|x) = {prob:.3f}")
    """
    # Compute likelihoods using independence assumption
    # p(x|y=1) = ∏ⱼ φⱼ|ᵧ₌₁ˣʲ(1-φⱼ|ᵧ₌₁)¹⁻ˣʲ
    p_x_given_y1 = np.prod(phi_j_y1 ** x * (1 - phi_j_y1) ** (1 - x))
    
    # p(x|y=0) = ∏ⱼ φⱼ|ᵧ₌₀ˣʲ(1-φⱼ|ᵧ₌₀)¹⁻ˣʲ
    p_x_given_y0 = np.prod(phi_j_y0 ** x * (1 - phi_j_y0) ** (1 - x))
    
    # Compute unnormalized posteriors
    num = p_x_given_y1 * phi_y        # p(x|y=1)p(y=1)
    denom = num + p_x_given_y0 * (1 - phi_y)  # p(x|y=1)p(y=1) + p(x|y=0)p(y=0)
    
    # Return normalized posterior (avoid division by zero)
    return num / denom if denom > 0 else 0.5

def laplace_smoothing_bernoulli(X, y):
    """
    Apply Laplace smoothing to Bernoulli Naive Bayes parameter estimation.
    
    Laplace smoothing adds pseudocounts to prevent zero probability estimates:
    φⱼ|ᵧ₌₁ = (1 + Σᵢ 1{xⱼ⁽ⁱ⁾ = 1 ∧ y⁽ⁱ⁾ = 1}) / (2 + Σᵢ 1{y⁽ⁱ⁾ = 1})
    φⱼ|ᵧ₌₀ = (1 + Σᵢ 1{xⱼ⁽ⁱ⁾ = 1 ∧ y⁽ⁱ⁾ = 0}) / (2 + Σᵢ 1{y⁽ⁱ⁾ = 0})
    
    The "1" in numerator and "2" in denominator come from:
    - Adding 1 pseudocount for each possible value (0 and 1)
    - This ensures all probabilities are strictly positive
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Binary feature matrix
    y : array-like, shape (n_samples,)
        Binary labels
    
    Returns:
    --------
    tuple
        (φⱼ|ᵧ₌₁, φⱼ|ᵧ₌₀) - smoothed parameters
    
    Example:
    --------
    >>> X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]])
    >>> y = np.array([1, 1, 0, 0])
    >>> phi_j_y1_smooth, phi_j_y0_smooth = laplace_smoothing_bernoulli(X, y)
    >>> print(f"Smoothed φⱼ|ᵧ₌₁: {phi_j_y1_smooth}")
    >>> print(f"Smoothed φⱼ|ᵧ₌₀: {phi_j_y0_smooth}")
    """
    n_samples, n_features = X.shape
    
    # Add 1 to numerator, 2 to denominator for binary features
    phi_j_y1 = (1 + np.sum(X[y == 1], axis=0)) / (2 + np.sum(y == 1))
    phi_j_y0 = (1 + np.sum(X[y == 0], axis=0)) / (2 + np.sum(y == 0))
    
    return phi_j_y1, phi_j_y0

def multinomial_naive_bayes_params(X_multi, y_multi, vocab_size):
    """
    Estimate parameters for Multinomial Naive Bayes with Laplace smoothing.
    
    In the multinomial model:
    - Each document is a sequence of word indices
    - xⱼ is the word at position j (index into vocabulary)
    - p(xⱼ|y) is the probability of word xⱼ appearing in class y
    
    Parameters are estimated as:
    φₖ|ᵧ = (1 + count of word k in class y) / (|V| + total words in class y)
    
    Parameters:
    -----------
    X_multi : list of arrays
        Each array contains word indices for a document
    y_multi : array-like, shape (n_documents,)
        Binary labels for each document
    vocab_size : int
        Size of vocabulary |V|
    
    Returns:
    --------
    tuple
        (φₖ|ᵧ₌₁, φₖ|ᵧ₌₀) - word probabilities for each class
    
    Example:
    --------
    >>> vocab_size = 3  # words: [0="a", 1="buy", 2="now"]
    >>> X_multi = [np.array([0,1,0]), np.array([2,0]), np.array([1,1]), np.array([2])]
    >>> y_multi = np.array([1, 1, 0, 0])
    >>> phi_k_y1, phi_k_y0 = multinomial_naive_bayes_params(X_multi, y_multi, vocab_size)
    >>> print(f"φₖ|ᵧ₌₁: {phi_k_y1}")
    >>> print(f"φₖ|ᵧ₌₀: {phi_k_y0}")
    """
    # Initialize word count arrays
    word_counts_y1 = np.zeros(vocab_size)
    word_counts_y0 = np.zeros(vocab_size)
    total_words_y1 = 0
    total_words_y0 = 0
    
    # Count word occurrences for each class
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
    
    # Apply Laplace smoothing
    # Add 1 to numerator, |V| to denominator for each class
    phi_k_y1 = (1 + word_counts_y1) / (vocab_size + total_words_y1)
    phi_k_y0 = (1 + word_counts_y0) / (vocab_size + total_words_y0)
    
    return phi_k_y1, phi_k_y0

def predict_multinomial_naive_bayes(doc, phi_k_y1, phi_k_y0, phi_y):
    """
    Predict posterior probability for Multinomial Naive Bayes.
    
    For a document with word sequence [w₁, w₂, ..., wₙ]:
    p(y=1|doc) = p(doc|y=1)p(y=1) / p(doc)
    where p(doc|y=1) = ∏ᵢ φ_{wᵢ}|ᵧ₌₁
    
    Parameters:
    -----------
    doc : array-like
        Sequence of word indices
    phi_k_y1 : array-like, shape (vocab_size,)
        φₖ|ᵧ₌₁ for each word in vocabulary
    phi_k_y0 : array-like, shape (vocab_size,)
        φₖ|ᵧ₌₀ for each word in vocabulary
    phi_y : float
        Prior probability p(y=1)
    
    Returns:
    --------
    float
        Posterior probability p(y=1|doc)
    
    Example:
    --------
    >>> doc = np.array([0, 1])  # words: ["a", "buy"]
    >>> prob = predict_multinomial_naive_bayes(doc, phi_k_y1, phi_k_y0, phi_y)
    >>> print(f"p(y=1|doc) = {prob:.3f}")
    """
    # Count word occurrences in document
    word_counts = Counter(doc)
    
    # Compute likelihoods using independence assumption
    # p(doc|y=1) = ∏ₖ φₖ|ᵧ₌₁^countₖ
    p_doc_given_y1 = 1.0
    for word_idx, count in word_counts.items():
        p_doc_given_y1 *= phi_k_y1[word_idx] ** count
    
    # p(doc|y=0) = ∏ₖ φₖ|ᵧ₌₀^countₖ
    p_doc_given_y0 = 1.0
    for word_idx, count in word_counts.items():
        p_doc_given_y0 *= phi_k_y0[word_idx] ** count
    
    # Compute posterior using Bayes' rule
    num = p_doc_given_y1 * phi_y
    denom = num + p_doc_given_y0 * (1 - phi_y)
    
    return num / denom if denom > 0 else 0.5

def create_bag_of_words(texts, vocab=None):
    """
    Convert text documents to bag-of-words representation.
    
    This is a simple implementation that:
    1. Tokenizes text (splits on whitespace)
    2. Creates vocabulary from unique words
    3. Converts each document to binary feature vector
    
    Parameters:
    -----------
    texts : list of str
        List of text documents
    vocab : list of str, optional
        Pre-defined vocabulary. If None, created from texts.
    
    Returns:
    --------
    tuple
        (X, vocab) where X is binary feature matrix
    
    Example:
    --------
    >>> texts = ["buy cheap now", "buy now", "cheap", "now"]
    >>> X, vocab = create_bag_of_words(texts)
    >>> print(f"Vocabulary: {vocab}")
    >>> print(f"Feature matrix:\n{X}")
    """
    if vocab is None:
        # Create vocabulary from all unique words
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
        vocab = sorted(list(all_words))
    
    # Create binary feature matrix
    X = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        words = text.lower().split()
        for word in words:
            if word in vocab:
                X[i, vocab.index(word)] = 1
    
    return X, vocab

def plot_naive_bayes_comparison(X, y, phi_j_y1, phi_j_y0, phi_y, vocab=None):
    """
    Visualize Naive Bayes feature importance and predictions.
    
    Creates a plot showing:
    - Feature importance (φⱼ|ᵧ₌₁ vs φⱼ|ᵧ₌₀)
    - Training data points
    - Decision boundary
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Binary feature matrix
    y : array-like, shape (n_samples,)
        Class labels
    phi_j_y1, phi_j_y0, phi_y : fitted parameters
    vocab : list of str, optional
        Feature names for labeling
    """
    if X.shape[1] != 2:
        print("Visualization only works for 2 features")
        return
    
    # Create grid for decision boundary
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                        np.linspace(y_min, y_max, 50))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute predictions on grid
    predictions = np.zeros(len(grid_points))
    for i, point in enumerate(grid_points):
        predictions[i] = predict_naive_bayes(point, phi_j_y1, phi_j_y0, phi_y)
    predictions = predictions.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Decision boundary
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, predictions, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Class 0', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class 1', alpha=0.7)
    plt.xlabel('Feature 1' if vocab is None else vocab[0])
    plt.ylabel('Feature 2' if vocab is None else vocab[1])
    plt.title('Naive Bayes Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature importance
    plt.subplot(1, 2, 2)
    features = range(len(phi_j_y1))
    plt.bar([x-0.2 for x in features], phi_j_y0, width=0.4, label='Class 0', alpha=0.7)
    plt.bar([x+0.2 for x in features], phi_j_y1, width=0.4, label='Class 1', alpha=0.7)
    plt.xlabel('Feature Index')
    plt.ylabel('Probability φⱼ|ᵧ')
    plt.title('Feature Importance by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =====================
# Example Usage & Tests
# =====================

if __name__ == "__main__":
    print("=" * 60)
    print("Naive Bayes Classification Examples")
    print("=" * 60)
    
    # Example 1: Bernoulli Naive Bayes - Spam Classification
    print("\n1. Bernoulli Naive Bayes - Spam Classification:")
    print("-" * 50)
    
    # Create synthetic email dataset
    # Features: [buy, cheap, now, free, money]
    X = np.array([
        [1, 1, 0, 0, 0],  # "buy cheap" -> spam
        [1, 0, 1, 0, 0],  # "buy now" -> spam
        [0, 1, 0, 1, 0],  # "cheap free" -> spam
        [0, 0, 1, 0, 1],  # "now money" -> spam
        [0, 0, 0, 0, 0],  # no keywords -> not spam
        [0, 0, 1, 0, 0],  # "now" -> not spam
        [0, 1, 0, 0, 0],  # "cheap" -> not spam
        [1, 0, 0, 0, 0],  # "buy" -> not spam
    ])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 1=spam, 0=not spam
    
    vocab = ['buy', 'cheap', 'now', 'free', 'money']
    print(f"Dataset shape: {X.shape}")
    print(f"Vocabulary: {vocab}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Fit Bernoulli Naive Bayes
    phi_j_y1, phi_j_y0, phi_y = estimate_naive_bayes_params(X, y)
    print(f"\nFitted Parameters (without smoothing):")
    print(f"Class prior φᵧ: {phi_y:.3f}")
    print(f"φⱼ|ᵧ₌₁ (spam): {phi_j_y1}")
    print(f"φⱼ|ᵧ₌₀ (not spam): {phi_j_y0}")
    
    # Apply Laplace smoothing
    phi_j_y1_smooth, phi_j_y0_smooth = laplace_smoothing_bernoulli(X, y)
    print(f"\nFitted Parameters (with Laplace smoothing):")
    print(f"φⱼ|ᵧ₌₁ (spam): {phi_j_y1_smooth}")
    print(f"φⱼ|ᵧ₌₀ (not spam): {phi_j_y0_smooth}")
    
    # Make predictions
    test_emails = [
        np.array([1, 1, 0, 0, 0]),  # "buy cheap" -> should be spam
        np.array([0, 0, 0, 0, 0]),  # no keywords -> should be not spam
        np.array([1, 0, 1, 1, 1]),  # "buy now free money" -> should be spam
    ]
    
    print(f"\nPredictions (using smoothed parameters):")
    for i, email in enumerate(test_emails):
        prob = predict_naive_bayes(email, phi_j_y1_smooth, phi_j_y0_smooth, phi_y)
        words = [vocab[j] for j in range(len(vocab)) if email[j] == 1]
        print(f"  Email {i+1} ({', '.join(words) if words else 'no keywords'}): p(spam) = {prob:.3f}")
    
    # Example 2: Multinomial Naive Bayes - Document Classification
    print("\n2. Multinomial Naive Bayes - Document Classification:")
    print("-" * 55)
    
    # Create synthetic document dataset
    # Vocabulary: [0="a", 1="buy", 2="cheap", 3="now", 4="free"]
    vocab_size = 5
    X_multi = [
        np.array([0, 1, 1, 0, 0]),  # "a buy cheap" -> spam
        np.array([1, 0, 0, 1, 0]),  # "a now" -> not spam
        np.array([0, 1, 0, 1, 1]),  # "buy now free" -> spam
        np.array([1, 0, 1, 0, 0]),  # "a cheap" -> not spam
    ]
    y_multi = np.array([1, 0, 1, 0])
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of documents: {len(X_multi)}")
    print(f"Class distribution: {np.bincount(y_multi)}")
    
    # Fit Multinomial Naive Bayes
    phi_k_y1, phi_k_y0 = multinomial_naive_bayes_params(X_multi, y_multi, vocab_size)
    phi_y = np.mean(y_multi)
    
    print(f"\nFitted Parameters:")
    print(f"Class prior φᵧ: {phi_y:.3f}")
    print(f"φₖ|ᵧ₌₁ (spam): {phi_k_y1}")
    print(f"φₖ|ᵧ₌₀ (not spam): {phi_k_y0}")
    
    # Make predictions
    test_docs = [
        np.array([0, 1, 1, 0, 0]),  # "buy cheap" -> should be spam
        np.array([1, 0, 0, 1, 0]),  # "a now" -> should be not spam
    ]
    
    print(f"\nPredictions:")
    for i, doc in enumerate(test_docs):
        prob = predict_multinomial_naive_bayes(doc, phi_k_y1, phi_k_y0, phi_y)
        words = [f"word_{j}" for j in doc if j < vocab_size]
        print(f"  Document {i+1}: p(spam) = {prob:.3f}")
    
    # Example 3: Text Preprocessing and Bag-of-Words
    print("\n3. Text Preprocessing and Bag-of-Words:")
    print("-" * 40)
    
    texts = [
        "buy cheap now",
        "buy now",
        "cheap free",
        "now money",
        "hello world",
        "good morning",
        "nice day",
        "weather today"
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=spam, 0=not spam
    
    X_bow, vocab_bow = create_bag_of_words(texts)
    print(f"Vocabulary: {vocab_bow}")
    print(f"Feature matrix shape: {X_bow.shape}")
    print(f"Feature matrix:\n{X_bow}")
    
    # Fit model on bag-of-words data
    phi_j_y1_bow, phi_j_y0_bow, phi_y_bow = estimate_naive_bayes_params(X_bow, labels)
    print(f"\nBag-of-words parameters:")
    print(f"φⱼ|ᵧ₌₁: {phi_j_y1_bow}")
    print(f"φⱼ|ᵧ₌₀: {phi_j_y0_bow}")
    
    # Example 4: Visualization (if 2D data)
    print("\n4. Generating visualization...")
    try:
        if X.shape[1] == 2:
            plot_naive_bayes_comparison(X, y, phi_j_y1_smooth, phi_j_y0_smooth, phi_y, vocab)
            print("Visualization completed successfully!")
        else:
            print("Skipping visualization (need 2D data)")
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("(This requires matplotlib to be installed)")
    
    # Example 5: Performance Analysis
    print("\n5. Performance Analysis:")
    print("-" * 25)
    
    # Training accuracy
    train_predictions = []
    for i in range(len(X)):
        prob = predict_naive_bayes(X[i], phi_j_y1_smooth, phi_j_y0_smooth, phi_y)
        train_predictions.append(1 if prob > 0.5 else 0)
    
    accuracy = np.mean(np.array(train_predictions) == y)
    print(f"Training accuracy: {accuracy:.3f}")
    
    # Feature importance analysis
    print(f"\nFeature importance analysis:")
    for i, word in enumerate(vocab):
        importance = phi_j_y1_smooth[i] - phi_j_y0_smooth[i]
        print(f"  {word}: {importance:+.3f} ({'spam' if importance > 0 else 'not spam'} indicator)")
    
    print("\n" + "=" * 60)
    print("Naive Bayes Examples Completed!")
    print("=" * 60) 
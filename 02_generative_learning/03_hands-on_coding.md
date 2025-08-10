# Generative Learning: Hands-On Learning Guide

[![Generative Learning](https://img.shields.io/badge/Generative%20Learning-Probabilistic%20Models-blue.svg)](https://en.wikipedia.org/wiki/Generative_model)
[![Gaussian Discriminant Analysis](https://img.shields.io/badge/GDA-Multivariate%20Normal-green.svg)](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
[![Naive Bayes](https://img.shields.io/badge/Naive%20Bayes-Conditional%20Independence-purple.svg)](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Hands-on Learning](https://img.shields.io/badge/Learning-Hands--on%20Experience-green.svg)](https://en.wikipedia.org/wiki/Experiential_learning)

## From Probabilistic Modeling to Practical Implementation

We've explored the elegant framework of **generative learning algorithms**, which model the joint distribution $p(x,y)$ by learning the data generation process $p(x|y)$ and class priors $p(y)$. This approach provides a powerful alternative to discriminative methods, offering the ability to generate new data, incorporate prior knowledge, and work effectively with limited training data.

However, true understanding comes from **hands-on implementation**. This practical guide will help you translate the theoretical concepts into working code, experiment with different generative models, and develop the intuition needed to apply these algorithms to real-world problems.

## From Theoretical Framework to Hands-On Mastery

We've now built a comprehensive theoretical understanding of **generative learning algorithms** - from Gaussian Discriminant Analysis (GDA) for continuous features to Naive Bayes for discrete features. Both approaches model the data generation process $p(x|y)$ and use Bayes' rule to make predictions, but they handle fundamentally different types of data.

GDA assumes multivariate normal distributions for continuous features, leading to linear decision boundaries and a deep connection to logistic regression. Naive Bayes assumes conditional independence for discrete features, making it particularly powerful for high-dimensional data like text classification.

However, true mastery in generative learning comes from **hands-on implementation**. While understanding the mathematical framework is essential, implementing these generative models from scratch, experimenting with different parameter estimation methods, and applying them to real-world problems is where the concepts truly come to life.

The transition from theory to practice is crucial in generative learning. While the mathematical framework provides the foundation, implementing these models helps develop intuition, reveals practical challenges, and builds the skills needed for real-world applications. Coding these algorithms from scratch forces us to confront the details that theory often abstracts away.

In this practical guide, we'll put our theoretical knowledge into practice through hands-on coding exercises. We'll implement both GDA and Naive Bayes from scratch, experiment with different datasets, and develop the practical skills needed to apply these powerful generative models to real-world problems.

This hands-on approach will solidify our understanding and prepare us for the complex challenges that arise when applying generative learning in practice.

## Learning Objectives

By completing this hands-on learning guide, you will:

1. **Master Gaussian Discriminant Analysis** through interactive implementations
2. **Implement Naive Bayes classifiers** for both Bernoulli and Multinomial data
3. **Understand Bayes' rule** and posterior probability computation
4. **Build text classification systems** using bag-of-words representations
5. **Compare generative vs discriminative approaches** empirically
6. **Develop intuition for model selection** and parameter estimation

## Quick Start

### Prerequisites
- Basic Python knowledge (variables, functions, arrays)
- Familiarity with probability concepts (Bayes' rule, conditional probability)
- Understanding of multivariate normal distributions
- Completion of linear models and classification modules (recommended)

### Estimated Time
- **Setup**: 30 minutes
- **Lesson 1**: 3-4 hours
- **Lesson 2**: 3-4 hours
- **Total**: 7-9 hours

---

## Environment Setup

### Option 1: Using Conda (Recommended)

#### Step 1: Install Miniconda
```bash
# Download Miniconda for your OS
# Windows: https://docs.conda.io/en/latest/miniconda.html
# macOS: https://docs.conda.io/en/latest/miniconda.html
# Linux: https://docs.conda.io/en/latest/miniconda.html

# Verify installation
conda --version
```

#### Step 2: Create Environment
```bash
# Navigate to the generative learning directory
cd 02_generative_learning

# Create a new conda environment
conda env create -f code/environment.yaml

# Activate the environment
conda activate generative-learning-lesson

# Verify installation
python -c "import numpy, matplotlib, scipy, sklearn; print('All packages installed successfully!')"
```

### Option 2: Using pip

#### Step 1: Create Virtual Environment
```bash
# Navigate to the generative learning directory
cd 02_generative_learning

# Create virtual environment
python -m venv generative-learning-env

# Activate environment
# On Windows:
generative-learning-env\Scripts\activate
# On macOS/Linux:
source generative-learning-env/bin/activate

# Install requirements
pip install -r code/requirements.txt

# Verify installation
python -c "import numpy, matplotlib, scipy, sklearn; print('All packages installed successfully!')"
```

### Option 3: Using Jupyter Notebooks

#### Step 1: Install Jupyter
```bash
# After setting up environment above
pip install jupyter notebook

# Launch Jupyter
jupyter notebook
```

#### Step 2: Create New Notebook
```python
# In a new notebook cell, import required packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
np.random.seed(42)  # For reproducible results
```

---

## Lesson Structure

### Lesson 1: Gaussian Discriminant Analysis (3-4 hours)
**File**: `code/gda_examples.py`

#### Learning Goals
- Understand multivariate normal distributions and their properties
- Master GDA parameter estimation using maximum likelihood
- Implement Bayes' rule for posterior probability computation
- Visualize decision boundaries and Gaussian contours
- Compare GDA with logistic regression empirically

#### Hands-On Activities

**Activity 1.1: Understanding Bayes' Rule**
```python
# Explore the fundamental equation of generative learning
from code.gda_examples import bayes_posterior

# Bayes' rule: p(y|x) = p(x|y)p(y) / p(x)
# Let's understand each component:
px_y = np.array([0.2, 0.6])  # p(x|y=0), p(x|y=1) - likelihoods
py = np.array([0.7, 0.3])    # p(y=0), p(y=1) - priors
px = np.sum(px_y * py)       # p(x) - evidence (normalization)

# Compute posterior
posterior = bayes_posterior(px_y, py, px)
print(f"Likelihoods p(x|y): {px_y}")
print(f"Priors p(y): {py}")
print(f"Evidence p(x): {px:.3f}")
print(f"Posterior p(y|x): {posterior}")

# Key insight: Posterior combines likelihood and prior
```

**Activity 1.2: Multivariate Normal Density**
```python
# Understand multivariate normal distributions
from code.gda_examples import multivariate_normal_density

# Test with 2D Gaussian
x = np.array([1.0, 2.0])
mu = np.array([0.0, 0.0])
Sigma = np.array([[1.0, 0.5], [0.5, 1.0]])  # Correlated features

# Compute density
density = multivariate_normal_density(x, mu, Sigma)
print(f"Point x: {x}")
print(f"Mean μ: {mu}")
print(f"Covariance Σ:\n{Sigma}")
print(f"Density p(x): {density:.6f}")

# Try different points and observe density changes
x2 = np.array([0.0, 0.0])  # At the mean
density2 = multivariate_normal_density(x2, mu, Sigma)
print(f"Density at mean: {density2:.6f}")
```

**Activity 1.3: GDA Parameter Estimation**
```python
# Learn GDA parameters from data
from code.gda_examples import gda_fit

# Generate synthetic data
np.random.seed(42)
n_samples = 100

# Class 0: centered at (0, 0)
X0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//2)
y0 = np.zeros(n_samples//2)

# Class 1: centered at (2, 2)
X1 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
y1 = np.ones(n_samples//2)

# Combine data
X = np.vstack([X0, X1])
y = np.concatenate([y0, y1])

# Fit GDA model
phi, mu0, mu1, Sigma = gda_fit(X, y)
print(f"Class prior φ: {phi:.3f}")
print(f"Class 0 mean μ₀: {mu0}")
print(f"Class 1 mean μ₁: {mu1}")
print(f"Shared covariance Σ:\n{Sigma}")
```

**Activity 1.4: GDA Prediction and Visualization**
```python
# Make predictions and visualize results
from code.gda_examples import gda_predict, gda_predict_proba, plot_gda_decision_boundary

# Make predictions
y_pred = gda_predict(X, phi, mu0, mu1, Sigma)
y_proba = gda_predict_proba(X, phi, mu0, mu1, Sigma)

# Evaluate performance
accuracy = np.mean(y_pred == y)
print(f"GDA Accuracy: {accuracy:.3f}")

# Visualize decision boundary
plot_gda_decision_boundary(X, y, phi, mu0, mu1, Sigma, "GDA Decision Boundary")

# Key observations:
# - Linear decision boundary (due to shared covariance)
# - Gaussian contours show class distributions
# - Decision boundary is perpendicular to μ₁ - μ₀
```

**Activity 1.5: Comparison with Logistic Regression**
```python
# Compare GDA with logistic regression
from code.gda_examples import logistic_regression_predict
from sklearn.linear_model import LogisticRegression

# Fit logistic regression
lr = LogisticRegression()
lr.fit(X, y)

# Compare predictions
lr_pred = lr.predict(X)
lr_accuracy = np.mean(lr_pred == y)
print(f"Logistic Regression Accuracy: {lr_accuracy:.3f}")
print(f"GDA Accuracy: {accuracy:.3f}")

# Key insight: Both give linear boundaries, but GDA makes stronger assumptions
```

#### Experimentation Tasks
1. **Experiment with different covariance structures**: Try diagonal vs full covariance matrices
2. **Vary class separation**: Generate data with different distances between class means
3. **Test with non-Gaussian data**: See how GDA performs when assumptions are violated
4. **Compare decision boundaries**: Visualize GDA vs logistic regression boundaries

#### Check Your Understanding
- [ ] Can you explain Bayes' rule and its components?
- [ ] Do you understand why GDA gives linear decision boundaries?
- [ ] Can you implement GDA parameter estimation from scratch?
- [ ] Do you see the relationship between GDA and logistic regression?

---

### Lesson 2: Naive Bayes Classification (3-4 hours)
**File**: `code/naive_bayes_examples.py`

#### Learning Goals
- Understand the conditional independence assumption
- Implement Bernoulli and Multinomial Naive Bayes
- Master Laplace smoothing for robust parameter estimation
- Build text classification systems
- Visualize feature importance and decision boundaries

#### Hands-On Activities

**Activity 2.1: Bernoulli Naive Bayes**
```python
# Implement Bernoulli Naive Bayes for binary features
from code.naive_bayes_examples import estimate_naive_bayes_params, predict_naive_bayes

# Create simple binary dataset
X = np.array([
    [1, 1, 0],  # Document 1: [buy, cheap, now]
    [1, 0, 1],  # Document 2: [buy, expensive, later]
    [0, 1, 0],  # Document 3: [sell, cheap, now]
    [0, 0, 1]   # Document 4: [sell, expensive, later]
])
y = np.array([1, 1, 0, 0])  # 1=spam, 0=ham

# Estimate parameters
phi_j_y1, phi_j_y0, phi_y = estimate_naive_bayes_params(X, y)
print(f"Feature probabilities for spam (y=1): {phi_j_y1}")
print(f"Feature probabilities for ham (y=0): {phi_j_y0}")
print(f"Prior probability of spam: {phi_y:.3f}")

# Make prediction for new document
x_new = np.array([1, 0, 1])  # [buy, expensive, later]
prob_spam = predict_naive_bayes(x_new, phi_j_y1, phi_j_y0, phi_y)
print(f"Probability of spam: {prob_spam:.3f}")
```

**Activity 2.2: Laplace Smoothing**
```python
# Understand the importance of Laplace smoothing
from code.naive_bayes_examples import laplace_smoothing_bernoulli

# Test with data that has missing feature combinations
X_small = np.array([[1, 1], [1, 1], [0, 0]])  # Only two combinations
y_small = np.array([1, 1, 0])

# Without smoothing (problematic)
phi_j_y1_raw, phi_j_y0_raw, phi_y_raw = estimate_naive_bayes_params(X_small, y_small)
print("Without smoothing:")
print(f"φⱼ|ᵧ₌₁: {phi_j_y1_raw}")
print(f"φⱼ|ᵧ₌₀: {phi_j_y0_raw}")

# With Laplace smoothing
phi_j_y1_smooth, phi_j_y0_smooth, phi_y_smooth = laplace_smoothing_bernoulli(X_small, y_small)
print("\nWith Laplace smoothing:")
print(f"φⱼ|ᵧ₌₁: {phi_j_y1_smooth}")
print(f"φⱼ|ᵧ₌₀: {phi_j_y0_smooth}")

# Key insight: Smoothing prevents zero probabilities
```

**Activity 2.3: Multinomial Naive Bayes**
```python
# Implement Multinomial Naive Bayes for count data
from code.naive_bayes_examples import multinomial_naive_bayes_params, predict_multinomial_naive_bayes

# Create count-based dataset (word frequencies)
X_multi = np.array([
    [2, 1, 0, 1],  # Document 1: 2x"buy", 1x"cheap", 0x"expensive", 1x"now"
    [1, 0, 2, 0],  # Document 2: 1x"buy", 0x"cheap", 2x"expensive", 0x"now"
    [0, 1, 1, 1],  # Document 3: 0x"buy", 1x"cheap", 1x"expensive", 1x"now"
])
y_multi = np.array([1, 1, 0])  # 1=spam, 0=ham

# Estimate parameters
vocab_size = 4  # [buy, cheap, expensive, now]
phi_k_y1, phi_k_y0, phi_y = multinomial_naive_bayes_params(X_multi, y_multi, vocab_size)
print(f"Word probabilities for spam: {phi_k_y1}")
print(f"Word probabilities for ham: {phi_k_y0}")
print(f"Prior probability of spam: {phi_y:.3f}")

# Predict for new document
doc_new = [1, 0, 1, 0]  # 1x"buy", 0x"cheap", 1x"expensive", 0x"now"
prob_spam = predict_multinomial_naive_bayes(doc_new, phi_k_y1, phi_k_y0, phi_y)
print(f"Probability of spam: {prob_spam:.3f}")
```

**Activity 2.4: Text Classification with Bag-of-Words**
```python
# Build a complete text classification system
from code.naive_bayes_examples import create_bag_of_words

# Sample text data
texts = [
    "buy cheap now",
    "buy expensive later", 
    "sell cheap now",
    "sell expensive later"
]
labels = [1, 1, 0, 0]  # 1=spam, 0=ham

# Create vocabulary and bag-of-words representation
X_bow, vocab = create_bag_of_words(texts)
print(f"Vocabulary: {vocab}")
print(f"Bag-of-words matrix:\n{X_bow}")

# Fit Bernoulli Naive Bayes
phi_j_y1, phi_j_y0, phi_y = estimate_naive_bayes_params(X_bow, labels)

# Test on new text
new_text = ["buy cheap"]
X_new, _ = create_bag_of_words(new_text, vocab)
prob_spam = predict_naive_bayes(X_new[0], phi_j_y1, phi_j_y0, phi_y)
print(f"New text: {new_text[0]}")
print(f"Probability of spam: {prob_spam:.3f}")
```

**Activity 2.5: Visualization and Analysis**
```python
# Visualize Naive Bayes results
from code.naive_bayes_examples import plot_naive_bayes_comparison

# Create 2D dataset for visualization
np.random.seed(42)
X_2d = np.random.randint(0, 2, (100, 2))  # Binary features
y_2d = (X_2d[:, 0] + X_2d[:, 1] > 1).astype(int)  # Simple rule

# Fit and visualize
phi_j_y1, phi_j_y0, phi_y = estimate_naive_bayes_params(X_2d, y_2d)
plot_naive_bayes_comparison(X_2d, y_2d, phi_j_y1, phi_j_y0, phi_y, vocab=['Feature1', 'Feature2'])

# Key observations:
# - Decision boundary reflects feature independence assumption
# - Feature importance shows which features are most discriminative
```

#### Experimentation Tasks
1. **Test with different smoothing parameters**: Try different alpha values in Laplace smoothing
2. **Compare Bernoulli vs Multinomial**: Use same data with different representations
3. **Analyze feature importance**: Identify which words/features are most discriminative
4. **Test with real text data**: Apply to actual spam/ham or sentiment classification

#### Check Your Understanding
- [ ] Can you explain the conditional independence assumption?
- [ ] Do you understand why Laplace smoothing is necessary?
- [ ] Can you implement both Bernoulli and Multinomial Naive Bayes?
- [ ] Do you see the connection between Naive Bayes and text classification?

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Numerical Underflow in Naive Bayes
```python
# Problem: Very small probabilities cause underflow
# Solution: Use log probabilities
def predict_naive_bayes_log(x, phi_j_y1, phi_j_y0, phi_y):
    # Compute log probabilities to avoid underflow
    log_p_y1 = np.log(phi_y)
    log_p_y0 = np.log(1 - phi_y)
    
    for j in range(len(x)):
        if x[j] == 1:
            log_p_y1 += np.log(phi_j_y1[j])
            log_p_y0 += np.log(phi_j_y0[j])
        else:
            log_p_y1 += np.log(1 - phi_j_y1[j])
            log_p_y0 += np.log(1 - phi_j_y0[j])
    
    # Convert back to probability
    log_sum = np.logaddexp(log_p_y1, log_p_y0)
    return np.exp(log_p_y1 - log_sum)
```

#### Issue 2: Singular Covariance Matrix in GDA
```python
# Problem: Covariance matrix is not invertible
# Solution: Add regularization
def gda_fit_regularized(X, y, lambda_reg=1e-6):
    # ... existing code ...
    
    # Add regularization to covariance matrix
    Sigma_reg = Sigma + lambda_reg * np.eye(Sigma.shape[0])
    
    return phi, mu0, mu1, Sigma_reg
```

#### Issue 3: Zero Probabilities in Text Classification
```python
# Problem: Unseen words cause zero probabilities
# Solution: Use proper vocabulary handling
def create_bag_of_words_robust(texts, vocab=None, min_freq=1):
    if vocab is None:
        # Create vocabulary from training data only
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        # Filter by frequency
        word_counts = Counter(all_words)
        vocab = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create feature matrix
    X = np.zeros((len(texts), len(vocab)))
    for i, text in enumerate(texts):
        words = text.lower().split()
        for word in words:
            if word in vocab:
                X[i, vocab.index(word)] = 1
    
    return X, vocab
```

#### Issue 4: Poor Performance with Correlated Features
```python
# Problem: Naive Bayes assumes feature independence
# Solution: Use feature selection or different model
def feature_selection_naive_bayes(X, y, threshold=0.1):
    # Select features with high mutual information
    from sklearn.feature_selection import mutual_info_classif
    
    mi_scores = mutual_info_classif(X, y)
    selected_features = mi_scores > threshold
    
    return X[:, selected_features], selected_features
```

#### Issue 5: Class Imbalance Issues
```python
# Problem: Uneven class distribution affects performance
# Solution: Adjust priors or use balanced sampling
def balanced_naive_bayes(X, y):
    # Use balanced class priors
    n_classes = len(np.unique(y))
    balanced_priors = np.ones(n_classes) / n_classes
    
    # Fit model with balanced priors
    # ... implementation ...
    
    return model_with_balanced_priors
```

---

## Assessment and Progress Tracking

### Self-Assessment Checklist

#### GDA Level
- [ ] I can explain Bayes' rule and its components
- [ ] I understand multivariate normal distributions
- [ ] I can implement GDA parameter estimation
- [ ] I can visualize decision boundaries and Gaussian contours

#### Naive Bayes Level
- [ ] I can explain the conditional independence assumption
- [ ] I understand Laplace smoothing and its importance
- [ ] I can implement Bernoulli and Multinomial Naive Bayes
- [ ] I can build text classification systems

#### Generative Learning Level
- [ ] I understand the difference between generative and discriminative models
- [ ] I can compare GDA with logistic regression
- [ ] I can choose appropriate models for different data types
- [ ] I can interpret model parameters and predictions

#### Advanced Level
- [ ] I can handle numerical stability issues
- [ ] I can implement feature selection for Naive Bayes
- [ ] I can apply generative models to real-world problems
- [ ] I can tune hyperparameters and evaluate performance

### Progress Tracking

#### Week 1: Gaussian Discriminant Analysis
- **Goal**: Complete Lesson 1
- **Deliverable**: Working GDA implementation with visualization
- **Assessment**: Can you implement GDA and compare with logistic regression?

#### Week 2: Naive Bayes Classification
- **Goal**: Complete Lesson 2
- **Deliverable**: Text classification system using Naive Bayes
- **Assessment**: Can you build a spam classifier and interpret feature importance?

---

## Extension Projects

### Project 1: Email Spam Classifier
**Goal**: Build a complete spam detection system

**Tasks**:
1. Collect email dataset (Enron, SpamAssassin)
2. Implement text preprocessing and feature extraction
3. Train Naive Bayes classifier
4. Add feature selection and hyperparameter tuning
5. Deploy as web application

**Skills Developed**:
- Text preprocessing and feature engineering
- Naive Bayes implementation and tuning
- Model evaluation and interpretation
- Web development and deployment

### Project 2: Document Classification System
**Goal**: Build a multi-class document classifier

**Tasks**:
1. Collect document dataset (news articles, academic papers)
2. Implement TF-IDF feature extraction
3. Train Multinomial Naive Bayes
4. Add topic modeling capabilities
5. Create document recommendation system

**Skills Developed**:
- Multi-class classification
- Advanced text processing
- Topic modeling and recommendation
- System architecture design

### Project 3: Generative Data Synthesis
**Goal**: Use generative models to create synthetic data

**Tasks**:
1. Collect real dataset (medical, financial, etc.)
2. Implement GDA for data generation
3. Add data augmentation techniques
4. Evaluate synthetic data quality
5. Create data privacy-preserving system

**Skills Developed**:
- Data generation and synthesis
- Privacy-preserving machine learning
- Data quality assessment
- Ethical AI considerations

---

## Additional Resources

### Books
- **"Pattern Recognition and Machine Learning"** by Christopher Bishop
- **"Machine Learning"** by Tom Mitchell
- **"Introduction to Information Retrieval"** by Manning, Raghavan, and Schütze

### Online Courses
- **Coursera**: Machine Learning by Andrew Ng
- **edX**: Introduction to Machine Learning
- **MIT OpenCourseWare**: Introduction to Machine Learning

### Practice Datasets
- **UCI Machine Learning Repository**: Various classification datasets
- **Kaggle**: Spam detection, text classification datasets
- **NLTK**: Built-in text datasets for practice

### Advanced Topics
- **Bayesian Networks**: Extending beyond Naive Bayes
- **Latent Variable Models**: Mixture models and EM algorithm
- **Deep Generative Models**: VAEs, GANs, and flow-based models
- **Probabilistic Programming**: PyMC, Stan, and Edward

---

## Conclusion: The Power of Generative Modeling

Congratulations on completing this comprehensive journey through generative learning algorithms! We've explored the elegant probabilistic framework that models data generation processes and uses Bayes' rule for classification.

### The Complete Picture

**1. Probabilistic Foundation** - We started with Bayes' rule, understanding how to combine prior knowledge with observed data to make predictions.

**2. Gaussian Discriminant Analysis** - We learned how to model continuous features using multivariate normal distributions, leading to linear decision boundaries.

**3. Naive Bayes Classification** - We explored discrete feature modeling with the conditional independence assumption, building powerful text classification systems.

**4. Practical Implementation** - We implemented both algorithms from scratch, developing skills in parameter estimation, prediction, and model evaluation.

### Key Insights

- **Generative vs Discriminative**: Generative models learn the data generation process, while discriminative models learn decision boundaries directly
- **Bayes' Rule**: The fundamental equation that converts generative models into discriminative predictions
- **Model Assumptions**: GDA assumes Gaussian class-conditional distributions, Naive Bayes assumes feature independence
- **Practical Applications**: Generative models excel at text classification, data generation, and problems with limited training data

### Looking Forward

This generative learning foundation prepares you for advanced topics:
- **Bayesian Networks**: Extending beyond conditional independence
- **Latent Variable Models**: Mixture models and expectation-maximization
- **Deep Generative Models**: Variational autoencoders and generative adversarial networks
- **Probabilistic Programming**: Advanced Bayesian modeling frameworks

The principles we've learned here - probabilistic modeling, Bayes' rule, and generative approaches - will serve you well throughout your machine learning journey.

### Next Steps

1. **Apply generative models** to your own classification problems
2. **Explore advanced topics** like Bayesian networks and latent variable models
3. **Build a portfolio** of generative learning projects
4. **Contribute to open source** probabilistic modeling projects
5. **Continue learning** with more advanced generative techniques

Remember: Generative models offer a powerful perspective on machine learning, focusing on understanding how data is generated rather than just finding decision boundaries. Keep exploring, building, and applying these concepts to new problems!

---

**Previous: [Naive Bayes](02_naive_bayes.md)** - Learn about Naive Bayes classification for discrete features and text classification.

## Environment Files

### requirements.txt
```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0
jupyter>=1.0.0
notebook>=6.4.0
ipykernel>=6.0.0
nb_conda_kernels>=2.3.0
```

### environment.yaml
```yaml
name: generative-learning-lesson
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy>=1.21.0
  - matplotlib>=3.5.0
  - scipy>=1.7.0
  - scikit-learn>=1.0.0
  - pandas>=1.3.0
  - seaborn>=0.11.0
  - jupyter>=1.0.0
  - notebook>=6.4.0
  - pip
  - pip:
    - ipykernel
    - nb_conda_kernels
```

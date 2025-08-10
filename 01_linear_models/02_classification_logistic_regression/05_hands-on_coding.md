# Logistic Regression Classification: Hands-On Learning Guide

[![Classification](https://img.shields.io/badge/Classification-Binary%20%26%20Multiclass-blue.svg)](https://en.wikipedia.org/wiki/Statistical_classification)
[![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-GLM%20Family-green.svg)](https://en.wikipedia.org/wiki/Logistic_regression)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Hands-on Learning](https://img.shields.io/badge/Learning-Hands--on%20Experience-green.svg)](https://en.wikipedia.org/wiki/Experiential_learning)

## From Theoretical Understanding to Practical Mastery

We've now completed a comprehensive theoretical exploration of classification methods, building from the probabilistic foundations of logistic regression through the geometric insights of the perceptron, from multi-class classification with softmax to advanced optimization with Newton's method. These theoretical concepts provide the mathematical foundation for understanding classification algorithms.

However, true mastery in machine learning comes from **hands-on implementation**. The transition from theory to practice is where we confront the real challenges of machine learning: handling messy data, debugging algorithms, tuning hyperparameters, and interpreting results. This hands-on experience is essential for developing the intuition and skills needed for real-world applications.

In this practical guide, we'll implement each algorithm we've studied, experiment with real datasets, and develop the practical skills needed to apply these methods effectively. We'll see how theoretical concepts translate into working code and discover the nuances that make the difference between a working prototype and a robust, production-ready system.

This hands-on approach will solidify our understanding and prepare us for the complex challenges that arise when applying machine learning in practice.

## Learning Objectives

By completing this hands-on learning guide, you will:

1. **Master binary classification** using logistic regression with sigmoid activation
2. **Understand the perceptron algorithm** as the foundation of neural networks
3. **Implement multi-class classification** using softmax regression
4. **Explore advanced optimization** with Newton's method
5. **Build complete classification pipelines** with real-world applications
6. **Develop intuition for decision boundaries** and probability estimation

## Quick Start

### Prerequisites
- Basic Python knowledge (variables, functions, loops)
- Familiarity with linear algebra (matrices, vectors)
- Understanding of probability concepts (0-1 probabilities)
- Completion of linear regression module (recommended)

### Estimated Time
- **Setup**: 30 minutes
- **Lesson 1-2**: 2-3 hours each
- **Lesson 3-4**: 3-4 hours each
- **Total**: 12-18 hours

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
# Navigate to the classification directory
cd 01_linear_models/02_classification_logistic_regression

# Create a new conda environment
conda env create -f code/environment.yaml

# Activate the environment
conda activate classification-lesson

# Verify installation
python -c "import numpy, matplotlib, scipy, sklearn; print('All packages installed successfully!')"
```

### Option 2: Using pip

#### Step 1: Create Virtual Environment
```bash
# Navigate to the classification directory
cd 01_linear_models/02_classification_logistic_regression

# Create virtual environment
python -m venv classification-env

# Activate environment
# On Windows:
classification-env\Scripts\activate
# On macOS/Linux:
source classification-env/bin/activate

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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
np.random.seed(42)  # For reproducible results
```

---

## Lesson Structure

### Lesson 1: Binary Classification with Logistic Regression (2-3 hours)
**File**: `code/logistic_regression_examples.py`

#### Learning Goals
- Understand why linear regression fails for classification
- Master the sigmoid function and its properties
- Implement logistic regression from scratch
- Visualize decision boundaries and probability surfaces
- Compare with linear regression approaches

#### Hands-On Activities

**Activity 1.1: Understanding the Sigmoid Function**
```python
# Explore the sigmoid function properties
from code.logistic_regression_examples import sigmoid, sigmoid_derivative

# Test sigmoid function
z_values = np.array([-5, -2, 0, 2, 5])
probabilities = sigmoid(z_values)
print("Input values:", z_values)
print("Sigmoid outputs:", probabilities)

# Key observations:
# - sigmoid(0) = 0.5 (uncertainty)
# - sigmoid(z) → 1 as z → ∞ (high confidence)
# - sigmoid(z) → 0 as z → -∞ (low confidence)
```

**Activity 1.2: Logistic Regression Hypothesis**
```python
# Implement and test hypothesis function
from code.logistic_regression_examples import hypothesis

# Test with different parameters
theta = np.array([0.5, 1.0, -0.5])  # [bias, feature1, feature2]
x_test = np.array([1, 2, 3])  # [bias_term, feature1, feature2]
prediction = hypothesis(theta, x_test)
print(f"Predicted probability: {prediction:.3f}")

# Compare with linear regression
linear_prediction = np.dot(theta, x_test)
print(f"Linear prediction: {linear_prediction:.3f}")
print(f"Key difference: Linear can be negative, logistic is always 0-1")
```

**Activity 1.3: Log-Likelihood and Loss Functions**
```python
# Understand the connection between likelihood and loss
from code.logistic_regression_examples import log_likelihood, logistic_loss

# Generate sample data
X = np.array([[1, 2], [1, 3], [1, 1]])
y = np.array([1, 1, 0])
theta = np.array([0.1, 0.5])

# Compare different loss formulations
ll = log_likelihood(theta, X, y)
loss = logistic_loss(theta, X, y)
print(f"Log-likelihood: {ll:.3f}")
print(f"Logistic loss: {loss:.3f}")
print(f"Note: They are related but have different signs")
```

**Activity 1.4: Complete Training Pipeline**
```python
# Run the complete logistic regression example
from code.logistic_regression_examples import demonstrate_logistic_regression

# This will:
# 1. Generate synthetic classification data
# 2. Train logistic regression model
# 3. Visualize decision boundary
# 4. Show training convergence
# 5. Make predictions on new data
```

#### Experimentation Tasks
1. **Experiment with different sigmoid inputs**: Try extreme values and observe behavior
2. **Modify the decision threshold**: Change from 0.5 to other values and observe impact
3. **Compare with linear regression**: Apply linear regression to classification data and observe problems
4. **Create your own dataset**: Generate data with different class separations

#### Check Your Understanding
- [ ] Can you explain why linear regression fails for classification?
- [ ] Do you understand the properties of the sigmoid function?
- [ ] Can you implement logistic regression from scratch?
- [ ] Do you see the relationship between likelihood and loss?

---

### Lesson 2: Perceptron Algorithm (2-3 hours)
**File**: `code/perceptron_examples.py`

#### Learning Goals
- Understand the historical significance of perceptron
- Master the threshold function vs sigmoid comparison
- Implement perceptron learning algorithm
- Analyze convergence properties and limitations
- Visualize learning process and decision boundaries

#### Hands-On Activities

**Activity 2.1: Threshold Function vs Sigmoid**
```python
# Compare perceptron threshold with sigmoid
from code.perceptron_examples import perceptron_threshold
from code.logistic_regression_examples import sigmoid

# Test both functions
z_values = np.array([-2, -1, 0, 1, 2])
threshold_outputs = [perceptron_threshold(z) for z in z_values]
sigmoid_outputs = sigmoid(z_values)

print("Input values:", z_values)
print("Threshold outputs:", threshold_outputs)
print("Sigmoid outputs:", sigmoid_outputs)

# Key observation: Threshold is binary, sigmoid is continuous
```

**Activity 2.2: Perceptron Learning Rule**
```python
# Understand the perceptron update rule
from code.perceptron_examples import perceptron_update

# Simulate a single update
theta = np.array([0.1, 0.2, 0.3])
x = np.array([1, 2, 3])  # [bias, feature1, feature2]
y_true = 1
y_pred = 0  # Perceptron prediction
alpha = 0.1

# Update weights
theta_new = perceptron_update(theta, x, y_true, alpha)
print(f"Original weights: {theta}")
print(f"Updated weights: {theta_new}")
print(f"Update: {theta_new - theta}")
```

**Activity 2.3: Complete Perceptron Training**
```python
# Train perceptron on synthetic data
from code.perceptron_examples import demonstrate_perceptron

# This will:
# 1. Generate linearly separable data
# 2. Train perceptron algorithm
# 3. Visualize learning process
# 4. Show convergence behavior
# 5. Plot final decision boundary
```

**Activity 2.4: Convergence Analysis**
```python
# Analyze perceptron convergence
from code.perceptron_examples import generate_linearly_separable_data, train_perceptron

# Generate data with different separations
X_easy, y_easy = generate_linearly_separable_data(n_samples=50, random_state=42)
X_hard, y_hard = generate_linearly_separable_data(n_samples=200, random_state=43)

# Train on both datasets
theta_easy, history_easy = train_perceptron(X_easy, y_easy)
theta_hard, history_hard = train_perceptron(X_hard, y_hard)

print(f"Easy data converged in: {len(history_easy['errors'])} iterations")
print(f"Hard data converged in: {len(history_hard['errors'])} iterations")
```

#### Experimentation Tasks
1. **Test with non-separable data**: Generate XOR-like data and observe failure
2. **Experiment with learning rates**: Try different alpha values and observe convergence
3. **Compare with logistic regression**: Run both algorithms on same data
4. **Analyze convergence speed**: Measure iterations needed for different datasets

#### Check Your Understanding
- [ ] Can you explain the perceptron learning rule?
- [ ] Do you understand why perceptron only works for linearly separable data?
- [ ] Can you implement perceptron from scratch?
- [ ] Do you see the connection to neural networks?

---

### Lesson 3: Multi-Class Classification (3-4 hours)
**File**: `code/multiclass_softmax_example.py`

#### Learning Goals
- Understand softmax function and its properties
- Implement multi-class classification from scratch
- Master cross-entropy loss for multiple classes
- Visualize multi-class decision boundaries
- Handle numerical stability issues

#### Hands-On Activities

**Activity 3.1: Softmax Function Properties**
```python
# Explore softmax function
from code.multiclass_softmax_example import softmax, softmax_temperature

# Test basic softmax
logits = np.array([[2, 1, 0]])
probs = softmax(logits)
print("Logits:", logits)
print("Probabilities:", probs)
print("Sum of probabilities:", np.sum(probs))

# Test temperature scaling
probs_sharp = softmax_temperature(logits, temperature=0.5)
probs_soft = softmax_temperature(logits, temperature=2.0)
print("Sharp probabilities (T=0.5):", probs_sharp)
print("Soft probabilities (T=2.0):", probs_soft)
```

**Activity 3.2: Cross-Entropy Loss**
```python
# Understand multi-class loss function
from code.multiclass_softmax_example import cross_entropy_loss

# Test with different predictions
logits = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
y_true = np.array([0, 1, 2])  # Correct class for each sample

loss = cross_entropy_loss(logits, y_true)
print(f"Cross-entropy loss: {loss:.3f}")

# Compare with binary classification loss
# Key insight: Multi-class generalizes binary case
```

**Activity 3.3: Multi-Class Training**
```python
# Train softmax regression
from code.multiclass_softmax_example import demonstrate_multiclass_classification

# This will:
# 1. Generate 3-class synthetic data
# 2. Train softmax regression model
# 3. Visualize multi-class decision boundaries
# 4. Show training convergence
# 5. Make predictions on new data
```

**Activity 3.4: Numerical Stability**
```python
# Test numerical stability
from code.multiclass_softmax_example import softmax

# Test with large logits (potential overflow)
logits_large = np.array([[1000, 999, 998]])
probs_stable = softmax(logits_large)
print("Large logits probabilities:", probs_stable)
print("Sum:", np.sum(probs_stable))

# Key insight: Subtracting max prevents overflow
```

#### Experimentation Tasks
1. **Experiment with different numbers of classes**: Try 2, 3, 5, 10 classes
2. **Test temperature scaling**: Observe how temperature affects confidence
3. **Compare with one-vs-all**: Implement one-vs-all classification
4. **Analyze decision boundaries**: Visualize boundaries for different class configurations

#### Check Your Understanding
- [ ] Can you explain why softmax outputs sum to 1?
- [ ] Do you understand the connection to binary classification?
- [ ] Can you implement softmax regression from scratch?
- [ ] Do you see why numerical stability is important?

---

### Lesson 4: Newton's Method Optimization (3-4 hours)
**File**: `code/newtons_method_examples.py`

#### Learning Goals
- Understand Newton's method for optimization
- Implement Hessian computation for logistic regression
- Compare Newton's method with gradient ascent
- Analyze convergence properties
- Handle numerical stability issues

#### Hands-On Activities

**Activity 4.1: Newton's Method in 1D**
```python
# Understand Newton's method for root finding
from code.newtons_method_examples import newton_1d

# Find square root of 2
f = lambda x: x**2 - 2
df = lambda x: 2*x
root, history = newton_1d(f, df, x0=1.0)
print(f"Square root of 2: {root:.6f}")
print(f"f(root): {f(root):.2e}")

# Key insight: Quadratic convergence rate
```

**Activity 4.2: Newton's Method for Maximization**
```python
# Apply Newton's method to function maximization
from code.newtons_method_examples import newton_maximize_1d

# Maximize a simple function
l = lambda x: -(x-2)**2 + 1  # Maximum at x=2
dl = lambda x: -2*(x-2)       # First derivative
ddl = lambda x: -2            # Second derivative

max_x, history = newton_maximize_1d(l, dl, ddl, x0=0.0)
print(f"Maximum at x = {max_x:.6f}")
print(f"Maximum value = {l(max_x):.6f}")
```

**Activity 4.3: Logistic Regression with Newton's Method**
```python
# Apply Newton's method to logistic regression
from code.newtons_method_examples import newton_logistic_regression

# Generate synthetic data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=42, n_clusters_per_class=1)

# Train with Newton's method
theta_newton, history_newton = newton_logistic_regression(X, y)
print(f"Newton's method converged in {len(history_newton['likelihood'])} iterations")
```

**Activity 4.4: Comparison with Gradient Ascent**
```python
# Compare Newton's method with gradient ascent
from code.newtons_method_examples import compare_newton_vs_gradient

# This will:
# 1. Train logistic regression with both methods
# 2. Compare convergence rates
# 3. Visualize convergence curves
# 4. Analyze computational complexity
```

#### Experimentation Tasks
1. **Test with different initial conditions**: Try different starting points
2. **Analyze Hessian properties**: Study when Hessian is positive definite
3. **Compare computational cost**: Time both methods on different dataset sizes
4. **Experiment with regularization**: Add L2 regularization to Newton's method

#### Check Your Understanding
- [ ] Can you explain why Newton's method converges quadratically?
- [ ] Do you understand the Hessian matrix and its role?
- [ ] Can you implement Newton's method for logistic regression?
- [ ] Do you see the trade-offs between Newton's method and gradient ascent?

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Sigmoid Function Overflow
```python
# Problem: exp() causes overflow for large negative values
# Solution: Use numerical stability techniques
def sigmoid_stable(z):
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))
```

#### Issue 2: Softmax Numerical Instability
```python
# Problem: exp() causes overflow for large logits
# Solution: Subtract maximum before exponentiating
def softmax_stable(logits):
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits_shifted)
    return exps / np.sum(exps, axis=1, keepdims=True)
```

#### Issue 3: Perceptron Non-Convergence
```python
# Problem: Perceptron doesn't converge
# Solution: Check if data is linearly separable
def check_linear_separability(X, y):
    # Use linear SVM to test separability
    from sklearn.svm import LinearSVC
    svm = LinearSVC()
    try:
        svm.fit(X, y)
        return True
    except:
        return False
```

#### Issue 4: Newton's Method Failure
```python
# Problem: Hessian is not positive definite
# Solution: Add regularization or use pseudo-inverse
def hessian_regularized(theta, X, lambda_reg=1e-6):
    H = hessian(theta, X)
    H_reg = H + lambda_reg * np.eye(H.shape[0])
    return H_reg
```

#### Issue 5: Decision Boundary Visualization Issues
```python
# Problem: Decision boundary plot is unclear
# Solution: Use proper grid resolution and contour levels
def plot_decision_boundary_improved(theta, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Create grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_with_bias = np.column_stack([np.ones(grid_points.shape[0]), grid_points])
    
    # Make predictions
    predictions = predict(theta, grid_points_with_bias)
    
    # Reshape and plot
    predictions = predictions.reshape(xx.shape)
    plt.contourf(xx, yy, predictions, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
```

---

## Assessment and Progress Tracking

### Self-Assessment Checklist

#### Binary Classification Level
- [ ] I can implement the sigmoid function from scratch
- [ ] I understand why linear regression fails for classification
- [ ] I can implement logistic regression training
- [ ] I can visualize decision boundaries effectively

#### Perceptron Level
- [ ] I can implement the perceptron learning rule
- [ ] I understand the threshold function vs sigmoid
- [ ] I can analyze convergence properties
- [ ] I know when perceptron will fail

#### Multi-Class Level
- [ ] I can implement the softmax function
- [ ] I understand cross-entropy loss for multiple classes
- [ ] I can handle numerical stability issues
- [ ] I can visualize multi-class decision boundaries

#### Advanced Optimization Level
- [ ] I can implement Newton's method for 1D problems
- [ ] I understand Hessian computation and properties
- [ ] I can compare Newton's method with gradient methods
- [ ] I can handle numerical stability in optimization

### Progress Tracking

#### Week 1: Binary Classification
- **Goal**: Complete Lesson 1
- **Deliverable**: Working logistic regression implementation
- **Assessment**: Can you classify binary data with reasonable accuracy?

#### Week 2: Perceptron Foundation
- **Goal**: Complete Lesson 2
- **Deliverable**: Perceptron implementation with convergence analysis
- **Assessment**: Can you explain when perceptron converges?

#### Week 3: Multi-Class Classification
- **Goal**: Complete Lesson 3
- **Deliverable**: Softmax regression for 3+ classes
- **Assessment**: Can you handle multiple classes effectively?

#### Week 4: Advanced Optimization
- **Goal**: Complete Lesson 4
- **Deliverable**: Newton's method implementation
- **Assessment**: Can you explain the advantages of Newton's method?

---

## Extension Projects

### Project 1: Email Spam Classifier
**Goal**: Build a complete spam detection system

**Tasks**:
1. Collect email dataset (Enron, SpamAssassin)
2. Implement text preprocessing and feature extraction
3. Train logistic regression classifier
4. Evaluate with precision, recall, F1-score
5. Deploy as web application

**Skills Developed**:
- Text preprocessing and feature engineering
- Model evaluation metrics
- Web development and deployment
- Real-world classification problems

### Project 2: Handwritten Digit Recognition
**Goal**: Build a multi-class classifier for MNIST digits

**Tasks**:
1. Load and preprocess MNIST dataset
2. Implement softmax regression
3. Add feature engineering (HOG, pixel features)
4. Compare with other algorithms (SVM, Random Forest)
5. Create interactive web demo

**Skills Developed**:
- Image preprocessing and feature extraction
- Multi-class classification
- Model comparison and evaluation
- Interactive application development

### Project 3: Medical Diagnosis System
**Goal**: Build a binary classifier for medical diagnosis

**Tasks**:
1. Collect medical dataset (breast cancer, diabetes)
2. Handle imbalanced classes and missing data
3. Implement logistic regression with regularization
4. Analyze feature importance and interpretability
5. Create confidence intervals for predictions

**Skills Developed**:
- Handling real-world data challenges
- Model interpretability and feature importance
- Uncertainty quantification
- Medical/healthcare applications

---

## Additional Resources

### Books
- **"Pattern Recognition and Machine Learning"** by Christopher Bishop
- **"The Elements of Statistical Learning"** by Hastie, Tibshirani, and Friedman
- **"Machine Learning"** by Tom Mitchell

### Online Courses
- **Coursera**: Machine Learning by Andrew Ng
- **edX**: Introduction to Machine Learning
- **MIT OpenCourseWare**: Introduction to Machine Learning

### Practice Datasets
- **UCI Machine Learning Repository**: Various classification datasets
- **Kaggle**: Titanic, Iris, Breast Cancer Wisconsin
- **scikit-learn**: Built-in classification datasets

### Advanced Topics
- **Regularization**: L1 (Lasso), L2 (Ridge), Elastic Net
- **Feature Engineering**: Polynomial features, interactions
- **Model Selection**: Cross-validation, hyperparameter tuning
- **Ensemble Methods**: Bagging, boosting, stacking

---

## Conclusion: A Complete Classification Framework

Congratulations on completing this comprehensive journey through classification and logistic regression! We've built a complete framework that spans from fundamental concepts to advanced optimization techniques. Let's reflect on what we've learned and how these pieces fit together.

### The Complete Picture

**1. Probabilistic Foundations** - We started with logistic regression, establishing the probabilistic framework for binary classification using the sigmoid function and maximum likelihood estimation.

**2. Geometric Insights** - We explored the perceptron algorithm, which provides a deterministic approach to classification and introduces key concepts about linear separability and decision boundaries.

**3. Multi-Class Extension** - We extended our probabilistic framework to handle multiple classes using the softmax function, showing how to generalize binary classification to complex real-world problems.

**4. Advanced Optimization** - We explored Newton's method, demonstrating how second-order optimization can dramatically improve convergence compared to simple gradient methods.

**5. Practical Implementation** - We put theory into practice through hands-on coding exercises, developing the skills needed for real-world applications.

### Key Insights

- **Multiple approaches**: Probabilistic (logistic regression) and deterministic (perceptron) methods each have their strengths
- **Scalability**: From binary to multi-class classification represents a natural progression
- **Optimization matters**: The choice of optimization algorithm can dramatically affect performance
- **Theory guides practice**: Understanding the mathematical foundations enables effective implementation
- **Real-world complexity**: Practical applications require handling messy data, tuning parameters, and interpreting results

### Looking Forward

This classification framework provides the foundation for understanding more advanced machine learning topics:
- **Neural networks** build on the perceptron's geometric insights
- **Support vector machines** extend the concept of linear separability
- **Deep learning** combines probabilistic modeling with sophisticated optimization
- **Ensemble methods** combine multiple classification approaches
- **Reinforcement learning** applies classification to sequential decision problems

The principles we've learned here - probabilistic modeling, geometric interpretation, optimization, and practical implementation - will recur throughout your machine learning journey.

### Next Steps

1. **Apply your skills** to real-world classification problems
2. **Explore advanced topics** like regularization, feature engineering, and model selection
3. **Build a portfolio** of classification projects
4. **Contribute to open source** machine learning projects
5. **Continue learning** with more advanced algorithms and techniques

Remember: The best way to learn machine learning is through hands-on practice. Keep experimenting, building, and applying these concepts to new problems!

---

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
name: classification-lesson
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

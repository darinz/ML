# Generalized Linear Models: Hands-On Learning Guide

[![GLM](https://img.shields.io/badge/GLM-Generalized%20Linear%20Models-blue.svg)](https://en.wikipedia.org/wiki/Generalized_linear_model)
[![Exponential Family](https://img.shields.io/badge/Exponential%20Family-Statistical%20Foundation-green.svg)](https://en.wikipedia.org/wiki/Exponential_family)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Hands-on Learning](https://img.shields.io/badge/Learning-Hands--on%20Experience-green.svg)](https://en.wikipedia.org/wiki/Experiential_learning)

## From Unification Theory to Practical Implementation

We've explored the elegant mathematical framework of Generalized Linear Models (GLMs), which unifies linear regression, logistic regression, and many other models under a single theoretical umbrella. The exponential family provides the mathematical foundation, while the GLM construction principles give us a systematic approach to building models for any type of response variable.

However, true understanding comes from **hands-on implementation**. This practical guide will help you translate the theoretical concepts into working code, experiment with different distributions and link functions, and develop the intuition needed to apply GLMs to real-world problems.

## Learning Objectives

By completing this hands-on learning guide, you will:

1. **Master exponential family distributions** through interactive implementations
2. **Build GLMs from scratch** using the systematic construction framework
3. **Understand canonical link functions** and their mathematical properties
4. **Implement parameter estimation** using maximum likelihood and iterative methods
5. **Apply GLMs to real-world problems** with proper diagnostics and validation
6. **Develop intuition for model selection** and distribution choice

## Quick Start

### Prerequisites
- Basic Python knowledge (variables, functions, classes)
- Familiarity with linear algebra (matrices, vectors)
- Understanding of probability distributions (Bernoulli, Gaussian)
- Completion of linear regression and logistic regression modules (recommended)

### Estimated Time
- **Setup**: 30 minutes
- **Lesson 1**: 3-4 hours
- **Lesson 2**: 4-5 hours
- **Total**: 8-10 hours

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
# Navigate to the GLM directory
cd 01_linear_models/03_generalized_linear_models

# Create a new conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate glm-lesson

# Verify installation
python -c "import numpy, matplotlib, scipy, sklearn, pandas; print('All packages installed successfully!')"
```

### Option 2: Using pip

#### Step 1: Create Virtual Environment
```bash
# Navigate to the GLM directory
cd 01_linear_models/03_generalized_linear_models

# Create virtual environment
python -m venv glm-env

# Activate environment
# On Windows:
glm-env\Scripts\activate
# On macOS/Linux:
source glm-env/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import numpy, matplotlib, scipy, sklearn, pandas; print('All packages installed successfully!')"
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
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)  # For reproducible results
```

---

## Lesson Structure

### Lesson 1: Exponential Family Foundations (3-4 hours)
**File**: `exponential_family_examples.py`

#### Learning Goals
- Understand the canonical form of exponential family distributions
- Master the mathematical components: natural parameters, sufficient statistics, log partition functions
- Implement Bernoulli and Gaussian distributions as exponential families
- Visualize how parameters affect distribution shapes
- Connect exponential family properties to GLM construction

#### Hands-On Activities

**Activity 1.1: Understanding the Canonical Form**
```python
# Explore the exponential family canonical form
from exponential_family_examples import exponential_family_pdf

# The canonical form: p(y; η) = b(y) * exp(η^T * T(y) - a(η))
# Let's understand each component:
# - η: natural parameter
# - T(y): sufficient statistic
# - a(η): log partition function
# - b(y): base measure

# Test with simple example
y_values = np.array([0, 1])
eta = np.array([0.5])  # Natural parameter

# We'll implement the components step by step
print("Understanding exponential family components...")
```

**Activity 1.2: Bernoulli Distribution as Exponential Family**
```python
# Implement Bernoulli distribution in exponential family form
from exponential_family_examples import BernoulliExponentialFamily

# Create Bernoulli exponential family instance
bernoulli_ef = BernoulliExponentialFamily()

# Test parameter transformations
phi = 0.7  # Standard parameter (probability of success)
eta = bernoulli_ef.phi_to_eta(phi)  # Natural parameter
print(f"Standard parameter φ = {phi}")
print(f"Natural parameter η = {eta:.3f}")

# Verify transformation back
phi_back = bernoulli_ef.eta_to_phi(eta)
print(f"Transformed back: φ = {phi_back:.3f}")

# Test sufficient statistic
y = 1
T_y = bernoulli_ef.T(y)
print(f"Sufficient statistic T({y}) = {T_y}")
```

**Activity 1.3: Gaussian Distribution as Exponential Family**
```python
# Implement Gaussian distribution in exponential family form
from exponential_family_examples import GaussianExponentialFamily

# Create Gaussian exponential family instance
gaussian_ef = GaussianExponentialFamily()

# Test parameter transformations
mu = 2.0  # Mean parameter
eta = gaussian_ef.mu_to_eta(mu)  # Natural parameter
print(f"Mean parameter μ = {mu}")
print(f"Natural parameter η = {eta:.3f}")

# Test sufficient statistic
y = 1.5
T_y = gaussian_ef.T(y)
print(f"Sufficient statistic T({y}) = {T_y}")

# Verify log partition function
a_eta = gaussian_ef.a(eta)
print(f"Log partition function a(η) = {a_eta:.3f}")
```

**Activity 1.4: Interactive Demonstrations**
```python
# Run interactive demonstrations
from exponential_family_examples import interactive_bernoulli_demo, interactive_gaussian_demo

# Bernoulli demonstration
print("=== Bernoulli Distribution Demo ===")
interactive_bernoulli_demo()

# Gaussian demonstration
print("=== Gaussian Distribution Demo ===")
interactive_gaussian_demo()

# These will show:
# 1. How parameters affect distribution shapes
# 2. Relationship between standard and natural parameters
# 3. Mathematical properties in action
```

#### Experimentation Tasks
1. **Explore different parameter values**: Try various φ and μ values and observe effects
2. **Verify normalization**: Check that distributions integrate/sum to 1
3. **Compare standard vs exponential family forms**: Verify they give same results
4. **Implement other distributions**: Try Poisson or Gamma distributions

#### Check Your Understanding
- [ ] Can you explain the canonical form p(y; η) = b(y) * exp(η^T * T(y) - a(η))?
- [ ] Do you understand the relationship between standard and natural parameters?
- [ ] Can you implement Bernoulli and Gaussian as exponential families?
- [ ] Do you see why exponential families are mathematically elegant?

---

### Lesson 2: GLM Construction and Implementation (4-5 hours)
**File**: `constructing_glm_examples.py`

#### Learning Goals
- Master the three fundamental GLM assumptions
- Implement systematic GLM construction framework
- Build linear regression and logistic regression as GLMs
- Understand canonical link functions and their properties
- Apply GLMs to real-world problems with proper diagnostics

#### Hands-On Activities

**Activity 2.1: Understanding GLM Framework**
```python
# Explore the generic GLM framework
from constructing_glm_examples import GLMFramework

# The three fundamental assumptions:
# 1. Exponential family response distribution
# 2. Prediction goal: h(x) = E[y|x]
# 3. Linear relationship: η = θ^T x

# Create a GLM framework instance
# We'll use this to build specific GLMs
print("Understanding GLM framework components...")
```

**Activity 2.2: Linear Regression as GLM**
```python
# Implement linear regression using GLM framework
from constructing_glm_examples import LinearRegressionGLM

# Create linear regression GLM
lr_glm = LinearRegressionGLM()

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 2)
true_theta = np.array([1.5, -0.8, 2.1])  # [bias, feature1, feature2]
X_with_bias = np.column_stack([np.ones(100), X])
y = X_with_bias @ true_theta + np.random.normal(0, 0.5, 100)

# Fit using OLS (ordinary least squares)
lr_glm.fit_ols(X_with_bias, y)
print(f"True parameters: {true_theta}")
print(f"Estimated parameters: {lr_glm.theta}")

# Compare with sklearn
from sklearn.linear_model import LinearRegression
sklearn_lr = LinearRegression()
sklearn_lr.fit(X_with_bias, y)
print(f"Sklearn parameters: {np.concatenate([[sklearn_lr.intercept_], sklearn_lr.coef_])}")
```

**Activity 2.3: Logistic Regression as GLM**
```python
# Implement logistic regression using GLM framework
from constructing_glm_examples import LogisticRegressionGLM

# Create logistic regression GLM
logistic_glm = LogisticRegressionGLM()

# Generate binary classification data
np.random.seed(42)
X = np.random.randn(100, 2)
true_theta = np.array([0.5, 1.2, -0.8])  # [bias, feature1, feature2]
X_with_bias = np.column_stack([np.ones(100), X])

# Generate probabilities and binary outcomes
logits = X_with_bias @ true_theta
probabilities = 1 / (1 + np.exp(-logits))
y = np.random.binomial(1, probabilities)

# Fit using maximum likelihood
logistic_glm.fit_mle(X_with_bias, y)
print(f"True parameters: {true_theta}")
print(f"Estimated parameters: {logistic_glm.theta}")

# Make predictions
y_pred_proba = logistic_glm.predict_proba(X_with_bias)
y_pred_classes = logistic_glm.predict_classes(X_with_bias)
accuracy = logistic_glm.compute_accuracy(X_with_bias, y)
print(f"Accuracy: {accuracy:.3f}")
```

**Activity 2.4: Parameter Estimation Methods**
```python
# Compare different estimation methods
from constructing_glm_examples import compare_estimation_methods

# This will compare:
# 1. Maximum likelihood estimation
# 2. Iterative reweighted least squares (IRLS)
# 3. Gradient descent
# 4. Ordinary least squares (for linear regression)

print("=== Comparing Estimation Methods ===")
compare_estimation_methods()

# Key insights:
# - MLE is the gold standard for GLMs
# - IRLS is often more efficient than gradient descent
# - OLS is optimal for linear regression with Gaussian errors
```

**Activity 2.5: Real-World Applications**
```python
# Apply GLMs to real-world problems
from constructing_glm_examples import housing_price_example, medical_diagnosis_example

# Housing price prediction (linear regression GLM)
print("=== Housing Price Prediction ===")
housing_price_example()

# Medical diagnosis (logistic regression GLM)
print("=== Medical Diagnosis ===")
medical_diagnosis_example()

# These examples will show:
# 1. Data preprocessing and feature engineering
# 2. Model fitting and parameter estimation
# 3. Model diagnostics and validation
# 4. Interpretation of results
```

#### Experimentation Tasks
1. **Try different link functions**: Implement non-canonical links and observe effects
2. **Add regularization**: Implement L1/L2 regularization in GLM framework
3. **Model diagnostics**: Implement residual analysis and goodness-of-fit tests
4. **Cross-validation**: Add cross-validation to model selection process

#### Check Your Understanding
- [ ] Can you explain the three fundamental GLM assumptions?
- [ ] Do you understand why linear and logistic regression are GLMs?
- [ ] Can you implement GLMs from scratch using the framework?
- [ ] Do you see the connection between exponential families and GLMs?

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Exponential Family Parameter Transformations
```python
# Problem: Parameter transformations fail for extreme values
# Solution: Add bounds checking and numerical stability
def phi_to_eta_stable(phi, epsilon=1e-10):
    phi = np.clip(phi, epsilon, 1 - epsilon)  # Prevent log(0) or log(1)
    return np.log(phi / (1 - phi))
```

#### Issue 2: GLM Convergence Problems
```python
# Problem: MLE optimization doesn't converge
# Solution: Use better initialization and optimization settings
def fit_mle_robust(self, X, y, initial_theta=None):
    if initial_theta is None:
        # Use OLS solution as starting point
        initial_theta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Use robust optimization method
    result = minimize(
        lambda theta: -self.log_likelihood(X, y, theta),
        initial_theta,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'gtol': 1e-6}
    )
    return result.x
```

#### Issue 3: Numerical Instability in Link Functions
```python
# Problem: Link functions cause overflow/underflow
# Solution: Implement numerically stable versions
def canonical_link_stable(eta, max_val=500):
    eta = np.clip(eta, -max_val, max_val)  # Prevent overflow
    return 1 / (1 + np.exp(-eta))
```

#### Issue 4: Model Diagnostics Issues
```python
# Problem: Residuals don't follow expected patterns
# Solution: Check model assumptions and data quality
def comprehensive_diagnostics(y, y_pred, residuals):
    # Check for outliers
    outlier_threshold = 3 * np.std(residuals)
    outliers = np.abs(residuals) > outlier_threshold
    
    # Check for heteroscedasticity
    # Check for non-linearity
    # Check for normality
    
    return {
        'outliers': outliers,
        'heteroscedasticity': check_heteroscedasticity(y_pred, residuals),
        'non_linearity': check_non_linearity(y_pred, residuals),
        'normality': check_normality(residuals)
    }
```

#### Issue 5: Link Function Choice
```python
# Problem: Choosing appropriate link function
# Solution: Consider data characteristics and interpretability
def choose_link_function(response_type, data_characteristics):
    if response_type == 'binary':
        return 'logit'  # Canonical for Bernoulli
    elif response_type == 'count':
        return 'log'    # Canonical for Poisson
    elif response_type == 'positive_continuous':
        return 'log'    # Common choice
    elif response_type == 'continuous':
        return 'identity'  # Canonical for Gaussian
    else:
        raise ValueError(f"Unknown response type: {response_type}")
```

---

## Assessment and Progress Tracking

### Self-Assessment Checklist

#### Exponential Family Level
- [ ] I can explain the canonical form of exponential families
- [ ] I understand the relationship between standard and natural parameters
- [ ] I can implement Bernoulli and Gaussian as exponential families
- [ ] I can verify exponential family properties mathematically

#### GLM Framework Level
- [ ] I can explain the three fundamental GLM assumptions
- [ ] I understand why linear and logistic regression are GLMs
- [ ] I can implement the systematic GLM construction process
- [ ] I can choose appropriate link functions for different data types

#### Implementation Level
- [ ] I can build GLMs from scratch using the framework
- [ ] I can implement parameter estimation methods (MLE, IRLS)
- [ ] I can perform model diagnostics and validation
- [ ] I can apply GLMs to real-world problems

#### Advanced Level
- [ ] I can implement non-canonical link functions
- [ ] I can add regularization to GLMs
- [ ] I can handle model selection and comparison
- [ ] I can interpret GLM results in context

### Progress Tracking

#### Week 1: Exponential Family Foundations
- **Goal**: Complete Lesson 1
- **Deliverable**: Working exponential family implementations
- **Assessment**: Can you implement Bernoulli and Gaussian as exponential families?

#### Week 2: GLM Construction and Applications
- **Goal**: Complete Lesson 2
- **Deliverable**: Complete GLM framework with real-world applications
- **Assessment**: Can you build GLMs for different types of response variables?

---

## Extension Projects

### Project 1: Insurance Claims Modeling
**Goal**: Build GLMs for insurance claim prediction

**Tasks**:
1. Collect insurance claims dataset
2. Implement Poisson regression for claim counts
3. Implement Gamma regression for claim amounts
4. Compare different GLM specifications
5. Create risk assessment dashboard

**Skills Developed**:
- Count data modeling with Poisson GLMs
- Positive continuous data with Gamma GLMs
- Model comparison and selection
- Risk assessment and interpretation

### Project 2: Marketing Response Modeling
**Goal**: Build GLMs for marketing campaign effectiveness

**Tasks**:
1. Collect marketing campaign data
2. Implement logistic regression for response rates
3. Implement Poisson regression for purchase counts
4. Add customer segmentation features
5. Create ROI analysis and recommendations

**Skills Developed**:
- Binary response modeling
- Count data analysis
- Customer segmentation
- Marketing analytics

### Project 3: Environmental Impact Analysis
**Goal**: Build GLMs for environmental data analysis

**Tasks**:
1. Collect environmental monitoring data
2. Implement various GLMs for different response types
3. Handle spatial and temporal correlations
4. Create predictive models for policy decisions
5. Develop environmental impact assessment tools

**Skills Developed**:
- Multiple response type modeling
- Spatial and temporal analysis
- Policy-relevant modeling
- Environmental statistics

---

## Additional Resources

### Books
- **"Generalized Linear Models"** by McCullagh and Nelder
- **"Applied Regression Analysis and Generalized Linear Models"** by Fox
- **"Statistical Models"** by A.C. Davison

### Online Courses
- **Coursera**: Statistical Learning by Trevor Hastie and Robert Tibshirani
- **edX**: Statistical Learning for Data Science
- **MIT OpenCourseWare**: Introduction to Statistics

### Practice Datasets
- **UCI Machine Learning Repository**: Various datasets suitable for GLMs
- **Kaggle**: Insurance, marketing, and environmental datasets
- **R datasets**: Built-in datasets for GLM practice

### Advanced Topics
- **Mixed Models**: GLMMs for hierarchical data
- **Regularization**: L1/L2 regularization in GLMs
- **Model Selection**: AIC, BIC, and cross-validation
- **Bayesian GLMs**: Bayesian approaches to GLM estimation

---

## Conclusion: The Power of Unification

Congratulations on completing this comprehensive journey through Generalized Linear Models! We've explored the elegant mathematical framework that unifies many different types of regression models under a single theoretical umbrella.

### The Complete Picture

**1. Mathematical Foundation** - We started with exponential family distributions, understanding how they provide a unified mathematical framework for various probability distributions.

**2. Systematic Construction** - We learned the three fundamental assumptions and systematic process for building GLMs for any type of response variable.

**3. Practical Implementation** - We implemented the framework from scratch, building linear regression and logistic regression as GLMs.

**4. Real-World Applications** - We applied GLMs to practical problems, developing skills in model diagnostics and interpretation.

### Key Insights

- **Unification**: GLMs provide a single framework for many different types of regression problems
- **Mathematical Elegance**: Exponential families have beautiful mathematical properties that enable efficient estimation
- **Systematic Approach**: The three assumptions provide a recipe for building models for any response type
- **Practical Power**: GLMs are widely used in industry and research for their interpretability and flexibility

### Looking Forward

This GLM foundation prepares you for advanced topics:
- **Mixed Models**: Extending GLMs to handle hierarchical data
- **Regularization**: Adding penalty terms for better generalization
- **Bayesian GLMs**: Incorporating prior knowledge and uncertainty
- **Non-linear Extensions**: Generalized additive models and beyond
- **Deep Learning**: Understanding how GLMs relate to neural networks

The principles we've learned here - exponential families, systematic model construction, and practical implementation - will serve you well throughout your statistical modeling and machine learning journey.

### Next Steps

1. **Apply GLMs** to your own datasets and problems
2. **Explore advanced topics** like mixed models and regularization
3. **Build a portfolio** of GLM projects across different domains
4. **Contribute to open source** statistical modeling projects
5. **Continue learning** with more advanced statistical methods

Remember: The power of GLMs lies in their ability to unify different modeling approaches while maintaining interpretability and mathematical rigor. Keep exploring, building, and applying these concepts to new problems!

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
name: glm-lesson
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

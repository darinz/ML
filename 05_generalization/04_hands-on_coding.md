# Generalization in Machine Learning: Hands-On Learning Guide

[![Bias-Variance Tradeoff](https://img.shields.io/badge/Bias--Variance-Tradeoff-blue.svg)](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
[![Double Descent](https://img.shields.io/badge/Double%20Descent-Modern%20ML-green.svg)](https://en.wikipedia.org/wiki/Double_descent)
[![Python](https://img.shields.io/badge/Python-Implementation-yellow.svg)](https://python.org)
[![Hands-on Learning](https://img.shields.io/badge/Learning-Hands--on%20Experience-green.svg)](https://en.wikipedia.org/wiki/Experiential_learning)

## From Training Performance to Generalization Understanding

We've explored the elegant framework of **generalization in machine learning**, which addresses the fundamental question of how well our models perform on unseen data. Understanding generalization is crucial because the ultimate goal of machine learning is not to memorize training data, but to learn patterns that generalize to new, unseen examples.

However, true understanding comes from **hands-on implementation**. This practical guide will help you translate the theoretical concepts into working code, experiment with different model complexities, and develop the intuition needed to understand and improve generalization performance.

## Learning Objectives

By completing this hands-on learning guide, you will:

1. **Master bias-variance decomposition** through interactive implementations and visualizations
2. **Understand the double descent phenomenon** and its implications for modern machine learning
3. **Implement complexity bounds** and theoretical guarantees
4. **Analyze model selection strategies** and regularization effects
5. **Develop intuition for generalization** through practical experimentation
6. **Apply theoretical concepts** to real-world model selection problems

## Quick Start

### Prerequisites
- Basic Python knowledge (variables, functions, arrays)
- Familiarity with machine learning concepts (training, testing, overfitting)
- Understanding of statistical concepts (mean, variance, probability)
- Completion of linear models and classification modules (recommended)

### Estimated Time
- **Setup**: 30 minutes
- **Lesson 1**: 3-4 hours
- **Lesson 2**: 3-4 hours
- **Lesson 3**: 3-4 hours
- **Total**: 10-12 hours

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
# Navigate to the generalization directory
cd 05_generalization

# Create a new conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate generalization-lesson

# Verify installation
python -c "import numpy, matplotlib, scipy, sklearn; print('All packages installed successfully!')"
```

### Option 2: Using pip

#### Step 1: Create Virtual Environment
```bash
# Navigate to the generalization directory
cd 05_generalization

# Create virtual environment
python -m venv generalization-env

# Activate environment
# On Windows:
generalization-env\Scripts\activate
# On macOS/Linux:
source generalization-env/bin/activate

# Install requirements
pip install -r requirements.txt

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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
np.random.seed(42)  # For reproducible results
```

---

## Lesson Structure

### Lesson 1: Bias-Variance Decomposition (3-4 hours)
**File**: `bias_variance_decomposition_examples.py`

#### Learning Goals
- Understand the mathematical foundations of bias-variance decomposition
- Master the tradeoff between model complexity and generalization
- Implement Monte Carlo estimation of bias and variance
- Visualize underfitting vs overfitting scenarios
- Analyze model selection strategies

#### Hands-On Activities

**Activity 1.1: Understanding the True Function**
```python
# Explore the true underlying function and data generation process
from bias_variance_decomposition_examples import h_star, generate_data

# Define the true function (quadratic)
x = np.linspace(0, 1, 100)
y_true = h_star(x)

# Generate noisy training data
x_train, y_train, x_test, y_test_true = generate_data(n_train=20, n_test=100, sigma=0.2)

# Visualize true function vs noisy data
plt.figure(figsize=(10, 6))
plt.plot(x, y_true, 'b-', linewidth=2, label='True Function h*(x)')
plt.scatter(x_train, y_train, c='red', alpha=0.7, label='Noisy Training Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('True Function vs Noisy Training Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"True function: h*(x) = 2x² + 0.5")
print(f"Training data: {len(x_train)} points with noise σ = 0.2")

# Key insight: We never know the true function in practice
```

**Activity 1.2: Model Complexity Comparison**
```python
# Compare different model complexities and their behavior
from bias_variance_decomposition_examples import fit_polynomial_model, compute_mse

# Fit models of different complexities
degrees = [1, 2, 5, 10]
models = {}
predictions = {}

for degree in degrees:
    # Fit model
    model = fit_polynomial_model(x_train, y_train, degree)
    models[degree] = model
    
    # Make predictions
    y_pred = model.predict(x_test.reshape(-1, 1))
    predictions[degree] = y_pred
    
    # Compute MSE
    mse = compute_mse(y_test_true, y_pred)
    print(f"Degree {degree}: MSE = {mse:.4f}")

# Visualize all models
plt.figure(figsize=(12, 8))
plt.plot(x, y_true, 'k-', linewidth=3, label='True Function')
plt.scatter(x_train, y_train, c='red', alpha=0.7, label='Training Data')

colors = ['blue', 'green', 'orange', 'purple']
for i, degree in enumerate(degrees):
    plt.plot(x_test, predictions[degree], color=colors[i], 
             linewidth=2, label=f'Degree {degree}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Complexity Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Key insight: Different complexities lead to different bias-variance tradeoffs
```

**Activity 1.3: Bias-Variance Decomposition**
```python
# Implement Monte Carlo estimation of bias and variance
from bias_variance_decomposition_examples import estimate_bias_variance_decomposition

# Estimate bias-variance decomposition for different model complexities
degrees = [1, 2, 5, 10]
results = {}

for degree in degrees:
    def model_factory(x_train, y_train):
        return fit_polynomial_model(x_train, y_train, degree)
    
    mse, bias_squared, variance, irreducible = estimate_bias_variance_decomposition(
        model_factory, x0=0.5, n_repeats=500, n_train=8, sigma=0.2
    )
    
    results[degree] = {
        'mse': mse,
        'bias_squared': bias_squared,
        'variance': variance,
        'irreducible': irreducible
    }
    
    print(f"\nDegree {degree}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  Bias²: {bias_squared:.4f}")
    print(f"  Variance: {variance:.4f}")
    print(f"  Irreducible: {irreducible:.4f}")

# Key insight: MSE = Bias² + Variance + Irreducible Error
```

**Activity 1.4: Bias-Variance Tradeoff Visualization**
```python
# Visualize the complete bias-variance tradeoff
from bias_variance_decomposition_examples import plot_bias_variance_tradeoff

# Plot the tradeoff curve
degrees = list(range(1, 16))
plot_bias_variance_tradeoff(degrees, x0=0.5, n_repeats=200, n_train=8, sigma=0.2)

# Key insight: Optimal complexity balances bias and variance
```

#### Experimentation Tasks
1. **Experiment with different noise levels**: Try σ = 0.1, 0.2, 0.5 and observe effects
2. **Vary training set size**: Test with n_train = 5, 10, 20, 50
3. **Test different true functions**: Try linear, cubic, or sinusoidal true functions
4. **Analyze optimal complexity**: Find the degree that minimizes total error

#### Check Your Understanding
- [ ] Can you explain the bias-variance decomposition formula?
- [ ] Do you understand why high bias leads to underfitting?
- [ ] Can you implement Monte Carlo estimation of bias and variance?
- [ ] Do you see how model complexity affects the tradeoff?

---

### Lesson 2: Double Descent Phenomenon (3-4 hours)
**File**: `double_descent_examples.py`

#### Learning Goals
- Understand the double descent phenomenon and its implications
- Master model-wise and sample-wise double descent
- Implement regularization effects on double descent
- Analyze implicit regularization in modern optimizers
- Compare classical vs modern machine learning regimes

#### Hands-On Activities

**Activity 2.1: Model-Wise Double Descent**
```python
# Explore model-wise double descent by varying polynomial degree
from double_descent_examples import simulate_modelwise_double_descent, plot_modelwise_double_descent

# Simulate model-wise double descent
degrees, test_errors = simulate_modelwise_double_descent(
    n_train=100, n_test=1000, max_degree=30, noise_std=0.5
)

# Plot the double descent curve
plot_modelwise_double_descent(degrees, test_errors)

# Key insight: Test error can decrease again after the classical U-shaped curve
```

**Activity 2.2: Sample-Wise Double Descent**
```python
# Explore sample-wise double descent by varying training set size
from double_descent_examples import simulate_samplewise_double_descent, plot_samplewise_double_descent

# Simulate sample-wise double descent
sample_sizes, test_errors = simulate_samplewise_double_descent(
    d=50, n_min=10, n_max=100, step=2, noise_std=0.5
)

# Plot the sample-wise double descent curve
plot_samplewise_double_descent(sample_sizes, test_errors, d=50)

# Key insight: Test error can decrease as we add more training data in overparameterized regime
```

**Activity 2.3: Regularization Effects**
```python
# Understand how regularization affects double descent
from double_descent_examples import demonstrate_regularization_effect

# Demonstrate regularization effects
demonstrate_regularization_effect()

# Key insight: Regularization can mitigate the double descent peak
```

**Activity 2.4: Implicit Regularization**
```python
# Explore implicit regularization in modern optimizers
from double_descent_examples import demonstrate_implicit_regularization

# Demonstrate implicit regularization
demonstrate_implicit_regularization()

# Key insight: Modern optimizers find "simple" solutions even in overparameterized regime
```

#### Experimentation Tasks
1. **Experiment with different interpolation thresholds**: Vary the ratio of parameters to samples
2. **Test different regularization strengths**: Try various λ values in ridge regression
3. **Analyze different optimizers**: Compare gradient descent, SGD, and Adam
4. **Study the transition point**: Focus on the region where n ≈ d

#### Check Your Understanding
- [ ] Can you explain the difference between classical and modern regimes?
- [ ] Do you understand why double descent occurs?
- [ ] Can you implement model-wise and sample-wise double descent?
- [ ] Do you see how regularization affects the phenomenon?

---

### Lesson 3: Complexity Bounds and Theoretical Foundations (3-4 hours)
**File**: `complexity_bounds_examples.py`

#### Learning Goals
- Understand concentration inequalities and their applications
- Master sample complexity bounds and theoretical guarantees
- Implement VC dimension calculations and analysis
- Analyze learning curves and generalization behavior
- Apply theoretical concepts to practical model selection

#### Hands-On Activities

**Activity 3.1: Hoeffding Bound Demonstration**
```python
# Understand concentration inequalities through Hoeffding bound
from complexity_bounds_examples import demonstrate_hoeffding_bound

# Demonstrate Hoeffding bound
demonstrate_hoeffding_bound(phi=0.6, n=100, gamma=0.1, n_trials=10000)

# Key insight: Hoeffding bound provides probabilistic guarantees for sample mean convergence
```

**Activity 3.2: Union Bound Analysis**
```python
# Understand the union bound and its applications
from complexity_bounds_examples import demonstrate_union_bound

# Demonstrate union bound
demonstrate_union_bound(p_single=0.01, k=10)

# Key insight: Union bound helps bound probabilities of multiple rare events
```

**Activity 3.3: Empirical vs Generalization Error**
```python
# Analyze the gap between training and test performance
from complexity_bounds_examples import demonstrate_empirical_vs_generalization_error

# Demonstrate empirical vs generalization error
demonstrate_empirical_vs_generalization_error(phi=0.7, n=50, n_test=10000)

# Key insight: There's always a gap between training and test performance
```

**Activity 3.4: Sample Complexity Bounds**
```python
# Understand how many samples are needed for good generalization
from complexity_bounds_examples import demonstrate_sample_complexity_bounds

# Demonstrate sample complexity bounds
demonstrate_sample_complexity_bounds(gamma=0.1, delta=0.05, k_max=1000)

# Key insight: Sample complexity depends on desired accuracy and confidence
```

**Activity 3.5: VC Dimension Analysis**
```python
# Understand VC dimension for measuring model complexity
from complexity_bounds_examples import demonstrate_vc_dimension_2d

# Demonstrate VC dimension for 2D linear classifiers
demonstrate_vc_dimension_2d()

# Key insight: VC dimension measures the complexity of hypothesis classes
```

**Activity 3.6: Learning Curves**
```python
# Analyze learning curves and their implications
from complexity_bounds_examples import demonstrate_learning_curves

# Demonstrate learning curves
demonstrate_learning_curves()

# Key insight: Learning curves show how performance improves with more data
```

#### Experimentation Tasks
1. **Experiment with different confidence levels**: Try δ = 0.01, 0.05, 0.1
2. **Test different accuracy requirements**: Vary γ = 0.05, 0.1, 0.2
3. **Analyze different hypothesis classes**: Compare linear vs polynomial models
4. **Study the effect of model complexity**: Observe how VC dimension affects bounds

#### Check Your Understanding
- [ ] Can you explain the Hoeffding bound and its significance?
- [ ] Do you understand how sample complexity bounds work?
- [ ] Can you calculate VC dimension for simple hypothesis classes?
- [ ] Do you see the connection between theory and practice?

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Numerical Instability in Bias-Variance Estimation
```python
# Problem: Monte Carlo estimation produces unstable results
# Solution: Increase number of trials and use stable numerical methods
def stable_bias_variance_estimation(model_factory, x0, n_repeats=1000, n_train=20):
    """Stable bias-variance estimation with more trials and better numerical stability."""
    predictions = []
    
    for _ in range(n_repeats):
        # Generate fresh training data
        x_train = np.random.rand(n_train)
        y_train = h_star(x_train) + np.random.normal(0, 0.2, n_train)
        
        # Fit model and predict
        model = model_factory(x_train, y_train)
        pred = model.predict([[x0]])[0]
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Compute components with numerical stability
    mean_pred = np.mean(predictions)
    true_value = h_star(x0)
    
    bias_squared = (mean_pred - true_value) ** 2
    variance = np.var(predictions)
    irreducible = 0.2 ** 2  # σ²
    
    return bias_squared + variance + irreducible, bias_squared, variance, irreducible
```

#### Issue 2: Overfitting in Double Descent Experiments
```python
# Problem: Models overfit to training data in double descent experiments
# Solution: Use proper regularization and validation
def robust_double_descent_simulation(n_train, max_degree, reg_strength=0.01):
    """Robust double descent simulation with regularization."""
    degrees = list(range(1, max_degree + 1))
    test_errors = []
    
    for degree in degrees:
        # Use ridge regression for regularization
        model = make_pipeline(
            PolynomialFeatures(degree),
            Ridge(alpha=reg_strength)
        )
        
        # Fit and evaluate
        model.fit(x_train.reshape(-1, 1), y_train)
        y_pred = model.predict(x_test.reshape(-1, 1))
        mse = mean_squared_error(y_test, y_pred)
        test_errors.append(mse)
    
    return degrees, test_errors
```

#### Issue 3: Slow Convergence in Monte Carlo Simulations
```python
# Problem: Monte Carlo simulations take too long
# Solution: Use vectorization and parallel processing
def vectorized_monte_carlo(model_factory, x0, n_repeats=1000, n_train=20):
    """Vectorized Monte Carlo simulation for faster computation."""
    # Generate all training data at once
    x_trains = np.random.rand(n_repeats, n_train)
    y_trains = h_star(x_trains) + np.random.normal(0, 0.2, (n_repeats, n_train))
    
    predictions = []
    for i in range(n_repeats):
        model = model_factory(x_trains[i], y_trains[i])
        pred = model.predict([[x0]])[0]
        predictions.append(pred)
    
    return np.array(predictions)
```

#### Issue 4: Memory Issues with Large Experiments
```python
# Problem: Large experiments consume too much memory
# Solution: Use batch processing and memory-efficient storage
def memory_efficient_experiment(model_factory, x0, n_repeats=10000, batch_size=1000):
    """Memory-efficient experiment with batch processing."""
    results = []
    
    for batch_start in range(0, n_repeats, batch_size):
        batch_end = min(batch_start + batch_size, n_repeats)
        batch_size_actual = batch_end - batch_start
        
        # Process batch
        batch_predictions = []
        for _ in range(batch_size_actual):
            x_train = np.random.rand(20)
            y_train = h_star(x_train) + np.random.normal(0, 0.2, 20)
            model = model_factory(x_train, y_train)
            pred = model.predict([[x0]])[0]
            batch_predictions.append(pred)
        
        results.extend(batch_predictions)
    
    return np.array(results)
```

#### Issue 5: Inconsistent Results Across Runs
```python
# Problem: Results vary significantly between runs
# Solution: Use proper random seed management and increase sample sizes
def reproducible_experiment(model_factory, x0, n_repeats=5000, random_seed=42):
    """Reproducible experiment with proper random seed management."""
    np.random.seed(random_seed)
    
    predictions = []
    for i in range(n_repeats):
        # Use different seed for each trial but maintain reproducibility
        np.random.seed(random_seed + i)
        
        x_train = np.random.rand(20)
        y_train = h_star(x_train) + np.random.normal(0, 0.2, 20)
        model = model_factory(x_train, y_train)
        pred = model.predict([[x0]])[0]
        predictions.append(pred)
    
    return np.array(predictions)
```

---

## Assessment and Progress Tracking

### Self-Assessment Checklist

#### Bias-Variance Understanding Level
- [ ] I can explain the bias-variance decomposition formula
- [ ] I understand why high bias leads to underfitting
- [ ] I can implement Monte Carlo estimation of bias and variance
- [ ] I can identify optimal model complexity

#### Double Descent Level
- [ ] I can explain the difference between classical and modern regimes
- [ ] I understand why double descent occurs
- [ ] I can implement model-wise and sample-wise double descent
- [ ] I can analyze regularization effects on double descent

#### Theoretical Foundations Level
- [ ] I can explain concentration inequalities and their significance
- [ ] I understand sample complexity bounds
- [ ] I can calculate VC dimension for simple hypothesis classes
- [ ] I can analyze learning curves and their implications

#### Practical Application Level
- [ ] I can apply bias-variance analysis to model selection
- [ ] I can use theoretical bounds to guide experimental design
- [ ] I can interpret generalization behavior in practice
- [ ] I can design experiments to study generalization

### Progress Tracking

#### Week 1: Bias-Variance Decomposition
- **Goal**: Complete Lesson 1
- **Deliverable**: Working bias-variance decomposition with visualizations
- **Assessment**: Can you implement bias-variance analysis and explain the tradeoff?

#### Week 2: Double Descent Phenomenon
- **Goal**: Complete Lesson 2
- **Deliverable**: Double descent simulations and analysis
- **Assessment**: Can you implement double descent and understand its implications?

#### Week 3: Complexity Bounds and Theory
- **Goal**: Complete Lesson 3
- **Deliverable**: Theoretical bounds implementation and analysis
- **Assessment**: Can you implement complexity bounds and understand their significance?

---

## Extension Projects

### Project 1: Model Selection Framework
**Goal**: Build a comprehensive model selection system

**Tasks**:
1. Implement cross-validation with bias-variance analysis
2. Add regularization path analysis
3. Create automated model selection algorithms
4. Build visualization dashboard for model comparison
5. Add statistical significance testing

**Skills Developed**:
- Model selection and evaluation
- Statistical analysis and testing
- Visualization and dashboard development
- Automated machine learning

### Project 2: Deep Learning Generalization Analysis
**Goal**: Analyze generalization in deep neural networks

**Tasks**:
1. Implement double descent analysis for neural networks
2. Add implicit regularization analysis
3. Study the effect of different optimizers
4. Analyze the role of initialization and architecture
5. Create generalization prediction models

**Skills Developed**:
- Deep learning theory and practice
- Neural network analysis
- Optimization theory
- Experimental design

### Project 3: Theoretical Bounds Validation
**Goal**: Validate theoretical bounds in practice

**Tasks**:
1. Implement comprehensive bound testing framework
2. Add empirical vs theoretical comparison
3. Study bound tightness across different scenarios
4. Create bound improvement algorithms
5. Develop practical guidelines for bound usage

**Skills Developed**:
- Theoretical machine learning
- Statistical analysis
- Algorithm development
- Research methodology

---

## Additional Resources

### Books
- **"Understanding Machine Learning"** by Shai Shalev-Shwartz and Shai Ben-David
- **"The Elements of Statistical Learning"** by Hastie, Tibshirani, and Friedman
- **"Learning Theory from First Principles"** by Francis Bach

### Online Courses
- **Coursera**: Machine Learning by Andrew Ng
- **edX**: Statistical Learning Theory
- **MIT OpenCourseWare**: Introduction to Machine Learning

### Practice Datasets
- **UCI Machine Learning Repository**: Various datasets for generalization analysis
- **scikit-learn**: Built-in datasets for practice
- **Kaggle**: Real-world datasets for practical application

### Advanced Topics
- **PAC Learning**: Probably Approximately Correct learning theory
- **Rademacher Complexity**: Alternative complexity measures
- **Stability Theory**: Algorithmic stability and generalization
- **Information-Theoretic Bounds**: Information-theoretic approaches to generalization

---

## Conclusion: The Foundation of Machine Learning

Congratulations on completing this comprehensive journey through generalization theory and practice! We've explored the fundamental concepts that underlie all of machine learning - from the classical bias-variance tradeoff to the modern double descent phenomenon.

### The Complete Picture

**1. Bias-Variance Decomposition** - We started with the fundamental decomposition of prediction error into bias, variance, and irreducible error, understanding how model complexity affects generalization.

**2. Double Descent Phenomenon** - We explored how modern machine learning challenges classical wisdom, showing that test error can decrease again in overparameterized regimes.

**3. Theoretical Foundations** - We learned about concentration inequalities, sample complexity bounds, and VC dimension, providing theoretical guarantees for generalization.

**4. Practical Applications** - We implemented these concepts in code, developing intuition for model selection and experimental design.

### Key Insights

- **Bias-Variance Tradeoff**: The fundamental tension between model complexity and generalization
- **Double Descent**: Modern machine learning can defy classical wisdom in overparameterized regimes
- **Theoretical Guarantees**: Concentration inequalities and complexity bounds provide mathematical foundations
- **Practical Implications**: Understanding generalization guides model selection and experimental design
- **Modern Challenges**: Deep learning and overparameterized models require new theoretical frameworks

### Looking Forward

This generalization foundation prepares you for advanced topics:
- **Statistical Learning Theory**: Rigorous mathematical foundations
- **Deep Learning Theory**: Understanding generalization in neural networks
- **Optimization Theory**: How optimization affects generalization
- **Robust Learning**: Generalization under distribution shifts
- **Federated Learning**: Generalization in distributed settings

The principles we've learned here - bias-variance tradeoff, theoretical bounds, and practical model selection - will serve you well throughout your machine learning journey.

### Next Steps

1. **Apply generalization analysis** to your own machine learning projects
2. **Explore advanced theoretical topics** like PAC learning and stability theory
3. **Build model selection frameworks** for practical applications
4. **Contribute to research** on generalization in modern machine learning
5. **Continue learning** with more advanced theoretical and practical topics

Remember: Understanding generalization is the foundation of machine learning - it's what separates memorization from true learning. Keep exploring, building, and applying these concepts to new problems!

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
name: generalization-lesson
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

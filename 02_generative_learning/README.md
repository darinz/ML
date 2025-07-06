# Generative Learning Algorithms

This folder contains materials on **generative learning algorithms**, which model the joint distribution $p(x,y)$ by learning $p(x|y)$ and $p(y)$, then use Bayes' rule to compute $p(y|x)$ for classification.

## Overview

**Generative vs. Discriminative Models:**

- **Generative models** (GDA, Naive Bayes) learn how data is generated for each class by modeling $p(x|y)$ and $p(y)$. They can simulate new data points and are useful for unsupervised tasks.
- **Discriminative models** (logistic regression, SVMs) learn the boundary between classes by modeling $p(y|x)$ directly.

**Key Advantage:** Generative models can generate new data and often work well with limited training data.

## Contents

### Theory Documents

- **[01_gda.md](01_gda.md)** - Gaussian Discriminant Analysis
  - Multivariate normal distribution properties
  - GDA model assumptions and parameter estimation
  - Linear decision boundaries
  - Relationship to logistic regression

- **[02_naive_bayes.md](02_naive_bayes.md)** - Naive Bayes Classification
  - Bernoulli and multinomial variants
  - Feature independence assumption
  - Laplace smoothing for robustness
  - Text classification applications

### Implementation Examples

- **[gda_examples.py](gda_examples.py)** - GDA Implementation
  - Parameter estimation (MLE)
  - Multivariate normal density calculation
  - Prediction using Bayes' rule
  - Comparison with logistic regression

- **[naive_bayes_examples.py](naive_bayes_examples.py)** - Naive Bayes Implementation
  - Bernoulli Naive Bayes parameter estimation
  - Multinomial Naive Bayes with Laplace smoothing
  - Posterior probability calculation
  - Text classification examples

### Visualizations

The `img/` folder contains visualizations demonstrating:
- Gaussian distribution contours and shapes
- GDA decision boundaries
- Parameter estimation results

## Key Concepts

### Bayes' Rule
```math
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
```

### GDA Model
```math
y \sim \text{Bernoulli}(\phi)
x|y=0 \sim \mathcal{N}(\mu_0, \Sigma)
x|y=1 \sim \mathcal{N}(\mu_1, \Sigma)
```

### Naive Bayes Assumption
```math
p(x_1, \ldots, x_d|y) = \prod_{j=1}^d p(x_j|y)
```

## Quick Start

### GDA Example
```python
from gda_examples import gda_fit, gda_predict

# Fit GDA model
phi, mu0, mu1, Sigma = gda_fit(X, y)

# Make predictions
predictions = gda_predict(X_new, phi, mu0, mu1, Sigma)
```

### Naive Bayes Example
```python
from naive_bayes_examples import estimate_naive_bayes_params, predict_naive_bayes

# Estimate parameters
phi_j_y1, phi_j_y0, phi_y = estimate_naive_bayes_params(X, y)

# Make prediction
p_y1_given_x = predict_naive_bayes(x_new, phi_j_y1, phi_j_y0, phi_y)
```

## Applications

### GDA
- **When to use:** Continuous features, limited data, when you want interpretable parameters
- **Assumptions:** Features follow multivariate normal distribution with shared covariance
- **Decision boundary:** Linear (when covariance is shared across classes)

### Naive Bayes
- **When to use:** High-dimensional discrete data, text classification, when features are sparse
- **Assumptions:** Features are conditionally independent given the class
- **Variants:** Bernoulli (binary features), Multinomial (count features)

## Advantages and Limitations

### GDA
**Advantages:**
- Interpretable parameters (means, covariance)
- Works well with limited data
- Natural extension to multiple classes

**Limitations:**
- Assumes normal distribution (may not hold for all data)
- Shared covariance assumption may be too restrictive

### Naive Bayes
**Advantages:**
- Simple and fast training/prediction
- Works well with high-dimensional data
- Robust to irrelevant features
- Requires little training data

**Limitations:**
- Independence assumption is often violated
- May not capture complex feature interactions
- Sensitive to feature representation

## Mathematical Foundations

### Parameter Estimation (MLE)

**GDA:**
```math
\phi = \frac{1}{n} \sum_{i=1}^n 1\{y^{(i)} = 1\}
\mu_0 = \frac{\sum_{i=1}^n 1\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 0\}}
\mu_1 = \frac{\sum_{i=1}^n 1\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 1\}}
\Sigma = \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
```

**Naive Bayes:**
```math
\phi_{j|y=1} = \frac{\sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 1\}}{\sum_{i=1}^n 1\{y^{(i)} = 1\}}
\phi_{j|y=0} = \frac{\sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 0\}}{\sum_{i=1}^n 1\{y^{(i)} = 0\}}
\phi_y = \frac{1}{n} \sum_{i=1}^n 1\{y^{(i)} = 1\}
```

### Laplace Smoothing
```math
\phi_j = \frac{1 + \sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = y\}}{2 + \sum_{i=1}^n 1\{y^{(i)} = y\}}
```

## Related Topics

- **Logistic Regression:** Discriminative counterpart to GDA
- **Quadratic Discriminant Analysis (QDA):** GDA with class-specific covariance matrices
- **Support Vector Machines:** Alternative discriminative approach
- **Text Classification:** Primary application domain for Naive Bayes

## References

- CS229 Lecture Notes on Generative Learning Algorithms
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective 
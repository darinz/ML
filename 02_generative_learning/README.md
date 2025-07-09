# Generative Learning Algorithms

This folder contains comprehensive materials on **generative learning algorithms**, which model the joint distribution $p(x,y)$ by learning $p(x|y)$ and $p(y)$, then use Bayes' rule to compute $p(y|x)$ for classification.

## Overview

### The Generative Learning Paradigm

**Generative vs. Discriminative Models:**

- **Generative models** (GDA, Naive Bayes) learn how data is generated for each class by modeling $p(x|y)$ and $p(y)$. They can simulate new data points and are useful for unsupervised tasks.
- **Discriminative models** (logistic regression, SVMs) learn the boundary between classes by modeling $p(y|x)$ directly.

**Key Advantage:** Generative models can generate new data and often work well with limited training data.

**Philosophical Difference:**
- **Generative:** "How was this data generated?" → Learn the data generation process
- **Discriminative:** "What's the best boundary?" → Learn the decision boundary directly

### Why Study Generative Models?

1. **Data Generation:** Can create synthetic data for training, testing, or augmentation
2. **Interpretability:** Model parameters often have clear interpretations (e.g., means, covariances)
3. **Unsupervised Learning:** Can be applied to unlabeled data for clustering or density estimation
4. **Prior Knowledge:** Can incorporate domain knowledge about data generation processes
5. **Small Sample Efficiency:** Often work well with limited training data when assumptions hold

## Contents

### Theory Documents

- **[01_gda.md](01_gda.md)** - Gaussian Discriminant Analysis
  - **Multivariate Normal Distribution:** Deep dive into properties, geometric interpretation, and mathematical foundations
  - **GDA Model:** Assumptions, parameter estimation via maximum likelihood, and decision boundaries
  - **Linear Decision Boundaries:** Why shared covariance leads to linear boundaries
  - **Relationship to Logistic Regression:** Theoretical connections and bias-variance tradeoffs
  - **When to Use GDA:** Practical guidelines for model selection

- **[02_naive_bayes.md](02_naive_bayes.md)** - Naive Bayes Classification
  - **Discrete Feature Modeling:** Handling high-dimensional categorical data
  - **Bernoulli and Multinomial Variants:** Different approaches for different data types
  - **Feature Independence Assumption:** Mathematical formulation and practical implications
  - **Laplace Smoothing:** Robust parameter estimation and regularization
  - **Text Classification Applications:** Bag-of-words, event models, and real-world considerations

### Implementation Examples

- **[gda_examples.py](gda_examples.py)** - GDA Implementation
  - **Comprehensive Docstrings & Annotations:** All functions are documented with detailed explanations of the math, assumptions, and implementation.
  - **Parameter Estimation (MLE):** Computing class priors, means, and covariance matrices
  - **Multivariate Normal Density:** Efficient computation of probability densities
  - **Prediction Using Bayes' Rule:** Posterior probability calculation and classification
  - **Comparison with Logistic Regression:** Empirical comparison of performance and decision boundaries
  - **Visualization:** Includes code to plot decision boundaries and Gaussian contours for intuitive understanding
  - **Practical Performance Analysis:** Example-driven demonstration with accuracy and probability outputs

- **[naive_bayes_examples.py](naive_bayes_examples.py)** - Naive Bayes Implementation
  - **Comprehensive Docstrings & Annotations:** All functions are documented with detailed explanations of the math, assumptions, and implementation.
  - **Bernoulli Naive Bayes:** Binary feature modeling with parameter estimation
  - **Multinomial Naive Bayes:** Count-based features with Laplace smoothing
  - **Posterior Probability Calculation:** Efficient prediction algorithms
  - **Text Classification Examples:** Real-world applications and preprocessing
  - **Bag-of-Words/Text Preprocessing:** Includes code for converting raw text to feature vectors
  - **Visualization:** Feature importance and decision boundary plots for 2D data
  - **Practical Performance Analysis:** Example-driven demonstration with accuracy and feature importance outputs

### Visualizations

The `img/` folder contains visualizations demonstrating:
- **Gaussian Distribution Properties:** Contours, shapes, and parameter effects
- **GDA Decision Boundaries:** Linear boundaries and class separation
- **Parameter Estimation Results:** Model fitting and convergence

## Key Concepts

### Bayes' Rule: The Foundation
```math
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
```

**Components:**
- **Prior $p(y)$:** Initial beliefs about class probabilities
- **Likelihood $p(x|y)$:** How likely the data is under each class
- **Evidence $p(x)$:** Normalization constant (total probability of data)
- **Posterior $p(y|x)$:** Updated beliefs after observing data

**Decision Rule:**
```math
\arg\max_y p(y|x) = \arg\max_y p(x|y)p(y)
```

### GDA Model: Continuous Features
```math
y \sim \text{Bernoulli}(\phi)
x|y=0 \sim \mathcal{N}(\mu_0, \Sigma)
x|y=1 \sim \mathcal{N}(\mu_1, \Sigma)
```

**Key Assumptions:**
1. **Gaussian Class-Conditionals:** Features follow multivariate normal distributions
2. **Shared Covariance:** Both classes have the same covariance matrix
3. **Linear Boundaries:** Shared covariance leads to linear decision boundaries

**Parameter Estimation (MLE):**
```math
\phi = \frac{1}{n} \sum_{i=1}^n 1\{y^{(i)} = 1\}
```
```math
\mu_0 = \frac{\sum_{i=1}^n 1\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 0\}}
```
```math
\mu_1 = \frac{\sum_{i=1}^n 1\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 1\}}
```
```math
\Sigma = \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
```

### Naive Bayes Assumption: Discrete Features
```math
p(x_1, \ldots, x_d|y) = \prod_{j=1}^d p(x_j|y)
```

**Key Assumptions:**
1. **Conditional Independence:** Features are independent given the class
2. **Discrete Features:** Each feature takes values from a finite set
3. **Parameter Efficiency:** Reduces parameters from exponential to linear in features

**Parameter Estimation with Laplace Smoothing:**
```math
\phi_{j|y=1} = \frac{1 + \sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 1\}}{2 + \sum_{i=1}^n 1\{y^{(i)} = 1\}}
```
```math
\phi_{j|y=0} = \frac{1 + \sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 0\}}{2 + \sum_{i=1}^n 1\{y^{(i)} = 0\}}
```

## Quick Start

### GDA Example

The GDA model can be fit to training data to estimate parameters, then used to make predictions on new data.

**Step-by-Step Process:**
1. **Data Preparation:** Ensure features are continuous and reasonably normal
2. **Parameter Estimation:** Compute class priors, means, and shared covariance
3. **Model Validation:** Check Gaussian assumptions and model fit
4. **Prediction:** Use Bayes' rule to compute posterior probabilities

### Naive Bayes Example

The Naive Bayes model estimates parameters from training data and computes posterior probabilities for predictions.

**Step-by-Step Process:**
1. **Feature Engineering:** Convert data to discrete features (binary or categorical)
2. **Parameter Estimation:** Compute feature probabilities with Laplace smoothing
3. **Model Validation:** Check independence assumptions
4. **Prediction:** Apply Bayes' rule for classification

## Applications

### GDA Applications
- **When to use:** Continuous features, limited data, when you want interpretable parameters
- **Assumptions:** Features follow multivariate normal distribution with shared covariance
- **Decision boundary:** Linear (when covariance is shared across classes)
- **Real-world examples:** Medical diagnosis, financial modeling, quality control

**Strengths:**
- Interpretable parameters (means, covariance)
- Works well with limited data
- Natural extension to multiple classes
- Can generate synthetic data

**Limitations:**
- Assumes normal distribution (may not hold for all data)
- Shared covariance assumption may be too restrictive
- Sensitive to outliers and non-linear relationships

### Naive Bayes Applications
- **When to use:** High-dimensional discrete data, text classification, when features are sparse
- **Assumptions:** Features are conditionally independent given the class
- **Variants:** Bernoulli (binary features), Multinomial (count features)
- **Real-world examples:** Spam filtering, sentiment analysis, document classification

**Strengths:**
- Simple and fast training/prediction
- Works well with high-dimensional data
- Robust to irrelevant features
- Requires little training data
- Handles missing values naturally

**Limitations:**
- Independence assumption is often violated
- May not capture complex feature interactions
- Sensitive to feature representation
- Can be biased by feature selection

## Advantages and Limitations

### GDA
**Advantages:**
- **Interpretable Parameters:** Means and covariance matrices have clear geometric interpretations
- **Data Efficiency:** Works well with limited training data when assumptions hold
- **Natural Multi-Class Extension:** Generalizes easily to multiple classes
- **Generative Capability:** Can generate synthetic data from learned distributions
- **Theoretical Foundation:** Well-understood statistical properties

**Limitations:**
- **Distributional Assumptions:** Requires features to be approximately normal
- **Covariance Constraints:** Shared covariance assumption may be too restrictive
- **Outlier Sensitivity:** Can be heavily influenced by extreme values
- **Linear Boundaries:** Limited to linear decision boundaries (with shared covariance)

### Naive Bayes
**Advantages:**
- **Computational Efficiency:** Very fast training and prediction
- **Scalability:** Handles high-dimensional data well
- **Feature Robustness:** Tolerant of irrelevant features
- **Small Sample Performance:** Works with limited training data
- **Missing Data Handling:** Naturally handles missing features
- **Interpretability:** Feature importance is easily understood

**Limitations:**
- **Independence Assumption:** Often violated in practice
- **Feature Interactions:** Cannot capture complex dependencies between features
- **Representation Sensitivity:** Performance depends heavily on feature engineering
- **Probability Calibration:** May not produce well-calibrated probability estimates

## Mathematical Foundations

### Parameter Estimation (MLE)

**GDA Maximum Likelihood:**
```math
\phi = \frac{1}{n} \sum_{i=1}^n 1\{y^{(i)} = 1\}
```
```math
\mu_0 = \frac{\sum_{i=1}^n 1\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 0\}}
```
```math
\mu_1 = \frac{\sum_{i=1}^n 1\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 1\}}
```
```math
\Sigma = \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
```

**Naive Bayes Maximum Likelihood:**
```math
\phi_{j|y=1} = \frac{\sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 1\}}{\sum_{i=1}^n 1\{y^{(i)} = 1\}}
```
```math
\phi_{j|y=0} = \frac{\sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 0\}}{\sum_{i=1}^n 1\{y^{(i)} = 0\}}
```
```math
\phi_y = \frac{1}{n} \sum_{i=1}^n 1\{y^{(i)} = 1\}
```

### Laplace Smoothing
```math
\phi_j = \frac{1 + \sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = y\}}{2 + \sum_{i=1}^n 1\{y^{(i)} = y\}}
```

**Why Laplace Smoothing:**
- Prevents zero probability estimates
- Acts as regularization
- Improves generalization
- Handles unseen features gracefully

### Decision Theory

**Optimal Classification Rule:**
```math
\hat{y} = \arg\max_y p(y|x) = \arg\max_y p(x|y)p(y)
```

**Risk Minimization:**
- **0-1 Loss:** Minimizes classification error
- **Cost-Sensitive:** Can incorporate different costs for different types of errors
- **Reject Option:** Can abstain from prediction when uncertain

## Related Topics

### Discriminative Counterparts
- **Logistic Regression:** Direct modeling of $p(y|x)$ without distributional assumptions
- **Support Vector Machines:** Margin-based classification with kernel methods
- **Neural Networks:** Non-linear discriminative models

### Generative Extensions
- **Quadratic Discriminant Analysis (QDA):** GDA with class-specific covariance matrices
- **Mixture Models:** Combining multiple distributions per class
- **Hidden Markov Models:** Sequential data modeling
- **Variational Autoencoders:** Deep generative models

### Text Classification
- **Word Embeddings:** Continuous representations of words
- **Transformer Models:** Attention-based architectures
- **BERT and GPT:** Pre-trained language models

## Practical Considerations

### Data Preprocessing
- **Feature Scaling:** Important for GDA (affects covariance estimation)
- **Feature Selection:** Critical for Naive Bayes (removes irrelevant features)
- **Discretization:** Converting continuous features for Naive Bayes
- **Text Preprocessing:** Tokenization, stemming, stop-word removal

### Model Selection
- **Cross-Validation:** Essential for comparing generative vs. discriminative approaches
- **Assumption Checking:** Verify Gaussian assumptions for GDA
- **Feature Independence:** Assess independence assumptions for Naive Bayes
- **Hyperparameter Tuning:** Laplace smoothing parameters, feature thresholds

### Evaluation Metrics
- **Accuracy:** Overall classification performance
- **Precision/Recall:** Important for imbalanced datasets
- **F1-Score:** Harmonic mean of precision and recall
- **ROC Curves:** Performance across different thresholds
- **Calibration:** Quality of probability estimates

## References

### Foundational Papers
- CS229 Lecture Notes on Generative Learning Algorithms
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective

### Advanced Topics
- Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers
- McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning

### Practical Guides
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval
- Jurafsky, D., & Martin, J. H. (2009). Speech and Language Processing 
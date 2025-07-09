# Constructing Generalized Linear Models: A Systematic Approach

## Introduction and Motivation

Suppose you would like to build a model to estimate the number $y$ of customers arriving in your store (or number of page-views on your website) in any given hour, based on certain features $x$ such as store promotions, recent advertising, weather, day-of-week, etc. We know that the Poisson distribution usually gives a good model for numbers of visitors. 

**The Challenge**: How can we systematically construct a model for this problem?

**The Solution**: Fortunately, the Poisson is an exponential family distribution, so we can apply a **Generalized Linear Model (GLM)**. This section provides a systematic recipe for constructing GLMs for any prediction problem.

## 3.2 The GLM Construction Framework

### Core Intuition

Generalized Linear Models (GLMs) provide a unified framework for modeling a wide variety of prediction problems, including regression and classification. The key insight is that many common distributions (Gaussian, Bernoulli, Poisson, etc.) belong to the exponential family, which allows us to use a common recipe for building models.

**Why GLMs are Powerful:**
- **Unified Approach**: Same mathematical framework for different data types
- **Interpretable**: Coefficients have clear statistical meaning
- **Flexible**: Can handle various response distributions
- **Theoretically Sound**: Well-understood statistical properties
- **Computationally Efficient**: Fast training and prediction

### The GLM Recipe

The GLM construction process involves four key steps:

1. **Choose Response Distribution**: Select from exponential family (Gaussian, Bernoulli, Poisson, etc.)
2. **Define Linear Predictor**: $\eta = \theta^T x$
3. **Specify Link Function**: Connect $\eta$ to the mean parameter $\mu$
4. **Estimate Parameters**: Use maximum likelihood or other methods

## 3.2.1 The Three Fundamental Assumptions

To derive a GLM for any prediction problem, we make three fundamental assumptions about the conditional distribution of $y$ given $x$:

### Assumption 1: Exponential Family Response
```math
y \mid x; \theta \sim \text{ExponentialFamily}(\eta)
```

**What this means**: Given $x$ and $\theta$, the distribution of $y$ follows some exponential family distribution with natural parameter $\eta$.

**Why this matters**: This ensures we can use the mathematical properties of exponential families, including:
- Simple gradient calculations
- Convex optimization problems
- Well-understood statistical properties

### Assumption 2: Prediction Goal
```math
h(x) = \mathbb{E}[y|x]
```

**What this means**: Our goal is to predict the expected value of $y$ given $x$.

**Verification**: This assumption is satisfied in both logistic regression and linear regression:
- **Logistic regression**: $h_\theta(x) = p(y = 1|x; \theta) = \mathbb{E}[y|x; \theta]$
- **Linear regression**: $h_\theta(x) = \mathbb{E}[y|x; \theta] = \mu$

### Assumption 3: Linear Relationship
```math
\eta = \theta^T x
```

**What this means**: The natural parameter $\eta$ is a linear function of the input features.

**Why this design choice**: This linearity assumption:
- Makes the model interpretable (each feature contributes additively)
- Enables efficient parameter estimation
- Provides a natural extension of linear models

### The Complete Framework

Combining all three assumptions:

```math
\begin{align*}
y \mid x; \theta &\sim \text{ExponentialFamily}(\eta) \\
h(x)\ &=\ \mathbb{E}[y|x] \\
\eta\ &=\ \theta^T x
\end{align*}
```

**Key Insight**: These three assumptions are sufficient to derive the entire GLM framework, including the form of the hypothesis function and the learning algorithm.

## 3.2.2 Ordinary Least Squares as a GLM

### Problem Setup

Consider a regression problem where:
- **Response variable**: $y$ is continuous
- **Goal**: Predict $y$ as a function of $x$
- **Data**: $(x^{(i)}, y^{(i)})$ pairs

### Step 1: Choose Response Distribution

We model the conditional distribution of $y$ given $x$ as Gaussian:
```math
y \mid x; \theta \sim \mathcal{N}(\mu, \sigma^2)
```

**Why Gaussian?**
- Natural choice for continuous data
- Well-understood properties
- Mathematically tractable

### Step 2: Apply GLM Assumptions

From our exponential family derivation, we know that for the Gaussian:
- **Natural parameter**: $\eta = \mu$
- **Canonical link**: Identity function

Applying the GLM assumptions:

```math
\begin{align*}
h_\theta(x) &= \mathbb{E}[y|x; \theta] \quad \text{(Assumption 2)} \\
            &= \mu \quad \text{(Gaussian mean)} \\
            &= \eta \quad \text{(Natural parameter)} \\
            &= \theta^T x \quad \text{(Assumption 3)}
\end{align*}
```

### Step 3: Derive the Hypothesis Function

This gives us the familiar linear regression hypothesis:
```math
h_\theta(x) = \theta^T x
```

### Step 4: Understand the Link Function

**Canonical Link**: Identity function $g(\eta) = \eta$

**Interpretation**: The mean of $y$ is modeled directly as a linear function of $x$.

### Geometric and Statistical Interpretation

#### Geometric Interpretation
OLS finds the line (or hyperplane) that minimizes the sum of squared vertical distances to the data points. This is equivalent to:
- Projecting the data onto the closest point in the subspace defined by the model
- Finding the orthogonal projection of the response vector onto the feature space

#### Statistical Properties
- **Optimality**: OLS is the best linear unbiased estimator (BLUE) under Gaussian assumptions
- **Efficiency**: Maximum likelihood estimator when errors are normally distributed
- **Interpretability**: Coefficients represent the change in $y$ for a unit change in $x$

#### Practical Considerations
- **Assumptions**: Requires normally distributed, homoscedastic errors
- **Robustness**: Sensitive to outliers
- **Extensions**: Basis for ridge regression, lasso, and other regularized methods

## 3.2.3 Logistic Regression as a GLM

### Problem Setup

Consider a binary classification problem where:
- **Response variable**: $y \in \{0, 1\}$
- **Goal**: Predict the probability that $y = 1$
- **Data**: $(x^{(i)}, y^{(i)})$ pairs with binary outcomes

### Step 1: Choose Response Distribution

We model the conditional distribution of $y$ given $x$ as Bernoulli:
```math
y \mid x; \theta \sim \text{Bernoulli}(\phi)
```

**Why Bernoulli?**
- Natural choice for binary outcomes
- Models probability of success
- Mathematically tractable

### Step 2: Apply GLM Assumptions

From our exponential family derivation, we know that for the Bernoulli:
- **Natural parameter**: $\eta = \log\left(\frac{\phi}{1-\phi}\right)$ (log-odds)
- **Canonical response function**: $\phi = \frac{1}{1 + e^{-\eta}}$ (sigmoid)

Applying the GLM assumptions:

```math
\begin{align*}
h_\theta(x) &= \mathbb{E}[y|x; \theta] \quad \text{(Assumption 2)} \\
            &= \phi \quad \text{(Bernoulli mean)} \\
            &= \frac{1}{1 + e^{-\eta}} \quad \text{(Canonical response)} \\
            &= \frac{1}{1 + e^{-\theta^T x}} \quad \text{(Assumption 3)}
\end{align*}
```

### Step 3: Derive the Hypothesis Function

This gives us the familiar logistic regression hypothesis:
```math
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
```

### Step 4: Understand the Link Function

**Canonical Link**: Logit function $g^{-1}(\phi) = \log\left(\frac{\phi}{1-\phi}\right)$

**Canonical Response**: Sigmoid function $g(\eta) = \frac{1}{1 + e^{-\eta}}$

**Interpretation**: The log-odds of the probability of $y=1$ is modeled as a linear function of $x$.

### The Sigmoid Function: Why It's Natural

**Key Insight**: The sigmoid function $1/(1 + e^{-z})$ isn't arbitrary - it's the canonical response function for the Bernoulli distribution!

**Mathematical Derivation**:
1. Start with log-odds: $\eta = \log\left(\frac{\phi}{1-\phi}\right)$
2. Exponentiate: $e^{\eta} = \frac{\phi}{1-\phi}$
3. Solve for $\phi$: $\phi = \frac{e^{\eta}}{1 + e^{\eta}} = \frac{1}{1 + e^{-\eta}}$

This explains why logistic regression uses the sigmoid function - it's mathematically inevitable given the Bernoulli assumption.

### Geometric and Statistical Interpretation

#### Geometric Interpretation
Logistic regression finds the hyperplane that best separates the two classes in terms of probability:
- **Decision boundary**: Where $h_\theta(x) = 0.5$ (i.e., $\theta^T x = 0$)
- **Probability interpretation**: Distance from decision boundary relates to prediction confidence
- **Linear separability**: Works best when classes are linearly separable in log-odds space

#### Statistical Properties
- **Interpretability**: Coefficients represent changes in log-odds
- **Calibration**: Predicted probabilities are well-calibrated
- **Robustness**: Less sensitive to outliers than linear regression
- **Regularization**: Natural extension to L1/L2 regularization

#### Practical Considerations
- **Assumptions**: Requires independent observations, linear relationship in log-odds
- **Extensions**: Basis for multinomial logistic regression, neural networks
- **Interpretation**: Odds ratios provide intuitive interpretation

## 3.2.4 Link Functions and Response Functions

### Terminology and Definitions

**Response Function** $g(\eta)$: Maps the natural parameter $\eta$ to the mean $\mu$
```math
\mu = g(\eta) = \mathbb{E}[y|\eta]
```

**Link Function** $g^{-1}(\mu)$: Maps the mean $\mu$ to the natural parameter $\eta$
```math
\eta = g^{-1}(\mu)
```

**Canonical Link**: The link function that makes $\eta = \theta^T x$ the natural parameter

### Examples of Canonical Links

| Distribution | Canonical Link | Canonical Response | Use Case |
|--------------|----------------|-------------------|----------|
| **Gaussian** | Identity | Identity | Continuous data |
| **Bernoulli** | Logit | Sigmoid | Binary classification |
| **Poisson** | Log | Exponential | Count data |
| **Gamma** | Inverse | Reciprocal | Positive continuous |

### Why Canonical Links Matter

**Mathematical Advantages**:
- Simplest form of the model
- Natural parameter interpretation
- Optimal statistical properties

**Practical Advantages**:
- Easier interpretation
- Better numerical stability
- Standard software implementations

## 3.2.5 Parameter Estimation in GLMs

### Maximum Likelihood Estimation

The standard approach for estimating GLM parameters is maximum likelihood estimation (MLE).

**Likelihood Function**:
```math
L(\theta) = \prod_{i=1}^n p(y^{(i)}|x^{(i)}; \theta)
```

**Log-Likelihood**:
```math
\ell(\theta) = \sum_{i=1}^n \log p(y^{(i)}|x^{(i)}; \theta)
```

### Iteratively Reweighted Least Squares (IRLS)

For canonical links, the MLE can be computed using IRLS:

1. **Initialize**: $\theta^{(0)} = 0$
2. **Iterate**:
   - Compute working responses: $z^{(i)} = \eta^{(i)} + \frac{y^{(i)} - \mu^{(i)}}{g'(\mu^{(i)})}$
   - Compute weights: $w^{(i)} = \frac{1}{g'(\mu^{(i)})^2 \text{Var}(y^{(i)})}$
   - Update: $\theta^{(t+1)} = (X^T W X)^{-1} X^T W z$

### Gradient Descent Alternative

For non-canonical links or large datasets, gradient descent can be used:

```math
\theta^{(t+1)} = \theta^{(t)} - \alpha \nabla_\theta \ell(\theta^{(t)})
```

## 3.2.6 Model Diagnostics and Validation

### Residual Analysis

**Deviance Residuals**: Measure the contribution of each observation to the model fit
```math
d_i = \text{sign}(y_i - \hat{\mu}_i) \sqrt{2[\ell(y_i; y_i) - \ell(y_i; \hat{\mu}_i)]}
```

**Pearson Residuals**: Standardized residuals
```math
r_i = \frac{y_i - \hat{\mu}_i}{\sqrt{\text{Var}(y_i)}}
```

### Goodness of Fit

**Deviance**: Overall measure of model fit
```math
D = 2[\ell_{\text{saturated}} - \ell_{\text{model}}]
```

**AIC/BIC**: Model selection criteria that balance fit and complexity

### Overdispersion

**Detection**: When the variance exceeds what the model predicts
**Solutions**: Quasi-likelihood, negative binomial, or mixed models

## 3.2.7 Extensions and Advanced Topics

### Non-Canonical Links

Sometimes non-canonical links are preferred:
- **Probit link**: $\Phi^{-1}(\mu)$ for binary data
- **Complementary log-log**: $\log(-\log(1-\mu))$ for survival data
- **Power links**: $\mu^\lambda$ for specific applications

### Regularization

**Ridge Regression (L2)**:
```math
\ell_{\text{ridge}}(\theta) = \ell(\theta) - \lambda \sum_{j=1}^p \theta_j^2
```

**Lasso (L1)**:
```math
\ell_{\text{lasso}}(\theta) = \ell(\theta) - \lambda \sum_{j=1}^p |\theta_j|
```

### Mixed Models

**Random Effects**: Account for hierarchical structure
```math
\eta = \theta^T x + b^T z
```
where $b \sim \mathcal{N}(0, \Sigma)$

## Summary

The GLM construction framework provides a systematic approach to building models for diverse prediction problems:

1. **Choose Response Distribution**: Based on data type and scientific question
2. **Apply GLM Assumptions**: Exponential family, prediction goal, linear relationship
3. **Derive Hypothesis Function**: Using canonical response functions
4. **Estimate Parameters**: Using MLE or other methods
5. **Validate Model**: Using diagnostics and goodness-of-fit measures

This framework unifies linear regression, logistic regression, and many other models under a single theoretical umbrella, providing both mathematical elegance and practical utility.

**Key Takeaways**:
- GLMs provide a unified framework for diverse prediction problems
- The exponential family assumption enables systematic model construction
- Canonical links provide optimal statistical properties
- The framework extends naturally to regularization and mixed models
- Model diagnostics ensure appropriate model fit and interpretation

## Further Reading and Advanced Resources

For deeper theoretical understanding and advanced perspectives on GLM construction and exponential families, the `exponential_family/` directory contains comprehensive reference materials from leading institutions:

### **Academic Reference Materials**
- **MIT Lecture Notes** (`the-exponential-family_MIT18_655S16_LecNote7.pdf`): Comprehensive coverage of exponential families with rigorous mathematical treatment and GLM applications
- **Princeton Lectures** (`exponential-families_princeton.pdf`, `lecture11_princeton.pdf`): Clear explanations of exponential families with practical GLM construction examples
- **Berkeley Materials** (`exponential_family_chapter8.pdf`, `the-exponential-family_chapter8_berkeley.pdf`): Advanced probability theory perspective on exponential families and their role in GLMs
- **Columbia Lecture** (`the-exponential-family_lecture12_columbia.pdf`): Focused treatment of exponential family properties and their application to GLM construction
- **Purdue Materials** (`expfamily_purdue.pdf`): Comprehensive treatment with detailed examples of exponential families in GLM contexts

### **Recommended Study Path for GLM Construction**
1. **Foundation**: Master the concepts in this document and practice with `constructing_glm_examples.py`
2. **Theoretical Deepening**: Study `the-exponential-family_lecture12_columbia.pdf` for clear explanations of exponential family foundations
3. **Advanced Applications**: Dive into `the-exponential-family_MIT18_655S16_LecNote7.pdf` for comprehensive coverage of GLM theory and applications
4. **Specialized Topics**: Use institution-specific materials for particular GLM extensions and advanced topics

These resources provide multiple perspectives on GLM construction, from different teaching approaches to advanced theoretical treatments, complementing the practical implementation focus of this course.

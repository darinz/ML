# Model Selection, Cross-Validation, and Bayesian Methods

## Table of Contents
1. [Introduction to Model Selection](#introduction-to-model-selection)
2. [The Model Selection Problem](#the-model-selection-problem)
3. [Cross-Validation Methods](#cross-validation-the-gold-standard)
4. [Bayesian Statistics](#bayesian-statistics-and-regularization)
5. [Frequentist vs Bayesian Approaches](#frequentist-vs-bayesian-approaches)
6. [Practical Guidelines](#practical-guidelines)
7. [Advanced Topics](#advanced-topics)

---

## Introduction to Model Selection: The Art of Choosing the Right Tool

### What is Model Selection? The Quest for the Perfect Fit

Selecting the right model is one of the most important—and challenging—tasks in machine learning. The "model" could mean the type of algorithm (e.g., linear regression, SVM, neural network), or it could mean the specific settings or complexity of a model (e.g., the degree of a polynomial, the number of layers in a neural network, or the regularization strength in ridge regression).

**The Fundamental Challenge:**
- **Too Simple**: Model may not capture the underlying patterns in the data (high bias, underfitting)
- **Too Complex**: Model may fit the noise in the training data rather than the true signal (high variance, overfitting)
- **The Goal**: Find the "sweet spot" between these two extremes

### Why is Model Selection Important? The Consequences of Poor Choices

**Real-World Analogies:**

**1. The Tool Selection Problem:**
Imagine you're choosing a tool for a job. A simple screwdriver might not be powerful enough for complex tasks, while a Swiss Army knife with 50 tools might be overkill and confusing. You need the right tool for the job—not too simple, not too complex.

**2. The Clothing Fitting Problem:**
Think of model selection like choosing clothes:
- **Too small (underfitting)**: Doesn't fit well, restricts movement
- **Too large (overfitting)**: Looks sloppy, doesn't provide proper support
- **Just right**: Fits well, allows movement, looks good

**3. The Recipe Complexity Problem:**
A chef choosing how complex a recipe to use:
- **Too simple**: Bland, missing important flavors
- **Too complex**: Overwhelming, masks the main ingredients
- **Just right**: Balanced flavors that enhance the dish

**Example - Polynomial Fitting:**  
Suppose you're fitting a curve to data points. If you use a straight line (degree 1 polynomial), it might miss important bends in the data. If you use a degree 10 polynomial, it might wiggle wildly to pass through every point, capturing noise rather than the true trend.

**Key Question:**  
How do we choose the best model or the best complexity for our data, *without* peeking at the test set (which would give us an overly optimistic estimate of performance)?

### The Model Selection Landscape: What We're Really Choosing

**Types of Model Selection:**

**1. Algorithm Selection:**
- Linear regression vs. polynomial regression vs. neural networks
- SVM vs. random forest vs. gradient boosting
- Different activation functions, loss functions

**2. Hyperparameter Tuning:**
- Learning rate, batch size, number of epochs
- Regularization strength ($`\lambda`$)
- Number of layers, neurons per layer
- Kernel parameters (for SVM)

**3. Feature Selection:**
- Which features to include/exclude
- Feature engineering choices
- Dimensionality reduction parameters

**4. Architecture Selection:**
- Model complexity (polynomial degree, network depth)
- Model capacity (number of parameters)
- Model family (parametric vs. non-parametric)

## From Regularization Techniques to Model Selection Strategies

We've now explored **regularization** - the fundamental techniques that help prevent overfitting by adding constraints or penalties to the learning process. We've seen how L1, L2, and Elastic Net regularization work, how implicit regularization affects optimization, and how these techniques help us find the sweet spot between underfitting and overfitting.

However, while regularization provides the tools to control model complexity, we still need systematic methods to **choose the right model** and **estimate its performance** reliably. Regularization tells us how to constrain a model, but it doesn't tell us which model to use or how to compare different options.

This motivates our exploration of **model selection** - the systematic process of choosing among different models, model complexities, and hyperparameters. We'll see how cross-validation provides reliable performance estimates, how Bayesian methods incorporate uncertainty and prior knowledge, and how to avoid common pitfalls in model selection.

The transition from regularization to model selection represents the bridge from technique to strategy - taking our knowledge of how to control model complexity and turning it into a systematic approach for building optimal models.

In this section, we'll explore cross-validation techniques, Bayesian approaches, and practical guidelines for selecting the best model for any given problem.

---

## The Model Selection Problem: Formalizing the Challenge

### Formal Definition: The Mathematical Framework

Suppose we are trying to select among several different models for a learning problem. For instance, we might be using a polynomial regression model $`h_\theta(x) = g(\theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_k x^k)`$, and wish to decide if $`k`$ should be $`0, 1, \ldots, 10`$. How can we automatically select a model that represents a good tradeoff between the twin evils of bias and variance?

**The Model Selection Objective:**
```math
M^* = \arg\min_{M \in \mathcal{M}} \mathbb{E}_{(x,y) \sim P} [L(h_M(x), y)]
```

Where:
- $`\mathcal{M}`$ is the set of candidate models
- $`h_M`$ is the hypothesis learned by model $`M`$
- $`L`$ is the loss function
- $`P`$ is the true data distribution

**Other Examples:**
- Choosing the bandwidth parameter $`\tau`$ for locally weighted regression
- Selecting the parameter $`C`$ for $`\ell_1`$-regularized SVM
- Deciding between linear regression, polynomial regression, or neural networks
- Choosing the number of hidden layers in a neural network

### The Model Space: Exploring the Universe of Possibilities

For the sake of concreteness, in these notes we assume we have some finite set of models $`\mathcal{M} = \{M_1, \ldots, M_d\}`$ that we're trying to select among. For instance, in our first example above, the model $`M_i`$ would be an $`i`$-th degree polynomial regression model.

**Visualizing the Model Space:**
```
Model Space:
M₁ (degree 0) → M₂ (degree 1) → M₃ (degree 2) → ... → M₁₀ (degree 9) → M₁₁ (degree 10)
     ↑              ↑              ↑                        ↑              ↑
  Constant      Linear         Quadratic              High-order     Very complex
   (bias)      (simple)         (moderate)              (complex)      (overfit)
```

**Generalization to Infinite Model Spaces:**
If we are trying to choose from an infinite set of models, say corresponding to the possible values of the bandwidth $`\tau \in \mathbb{R}^+``, we may discretize $`\tau`$ and consider only a finite number of possible values for it. More generally, most of the algorithms described here can all be viewed as performing optimization search in the space of models, and we can perform this search over infinite model classes as well.

**Search Strategies:**
- **Grid search**: Try all combinations of hyperparameters
- **Random search**: Sample random combinations
- **Bayesian optimization**: Use probabilistic models to guide search
- **Evolutionary algorithms**: Use genetic algorithms to evolve good models

### The Bias-Variance Tradeoff: The Fundamental Tension

**Understanding the Tradeoff:**
- **Bias**: How much the model's predictions differ from the true values on average
- **Variance**: How much the model's predictions vary when trained on different datasets
- **Total Error**: $`\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}`$

**Visual Analogy: The Dart Throwing Game**
Think of bias and variance like trying to hit a target with darts:
- **High Bias, Low Variance**: Consistently hitting the same wrong spot (like always hitting 2 inches to the left)
- **Low Bias, High Variance**: Hitting all around the target but rarely on it (like scattering darts everywhere)
- **Low Bias, Low Variance**: Consistently hitting the target (the ideal)

**Mathematical Intuition:**
```math
\text{Bias} = \mathbb{E}[\hat{f}(x)] - f(x)
\text{Variance} = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]
```

Where $`\hat{f}(x)`$ is our model's prediction and $`f(x)`$ is the true function.

**The Sweet Spot:**
The optimal model complexity minimizes the sum of squared bias and variance. This is the fundamental principle behind model selection.

### The Model Selection Challenge: Why It's Hard

**The Core Problem:**
We want to minimize the true generalization error, but we can only observe training error. The gap between these two is what makes model selection challenging.

**Why Training Error is Misleading:**
- **Optimistic bias**: Training error underestimates true error
- **Complexity penalty**: More complex models can always fit training data better
- **No generalization guarantee**: Good training performance doesn't guarantee good test performance

**The Information Gap:**
```
What we want:     True generalization error
What we observe:  Training error
The challenge:    Bridge this gap reliably
```

**Real-World Example:**
Imagine you're studying for an exam:
- **Training error**: How well you do on practice problems
- **True error**: How well you'll do on the actual exam
- **The gap**: Practice problems might not represent the exam perfectly

---

## The Pitfall of Naive Model Selection: Why Simple Approaches Fail

### The Wrong Way: Training Error

A tempting but flawed approach is to simply pick the model that fits the training data best (i.e., has the lowest training error). However, this almost always leads to overfitting: the most complex model will always fit the training data best, but may perform poorly on new, unseen data.

**Why This Fails:**
- Training error is an optimistic estimate of true performance
- More complex models can always fit the training data better
- This doesn't tell us how well the model will generalize

**Analogy - The Memorization Problem:**  
Imagine memorizing answers to practice exam questions. You'll ace the practice test, but if the real exam has different questions, you might struggle. Similarly, a model that "memorizes" the training data may not generalize well.

**Mathematical Explanation:**
The training error $`\hat{\varepsilon}_{\text{train}}(h)`$ is a biased estimate of the true error $`\varepsilon(h)`$:
$`\mathbb{E}[\hat{\varepsilon}_{\text{train}}(h)] \leq \varepsilon(h)`$

The inequality becomes more severe as model complexity increases.

**Visual Example:**
```
Model Complexity:  Low  →  Medium  →  High
Training Error:   0.3  →   0.1    →  0.01
True Error:       0.3  →   0.15   →  0.5
                  ↑              ↑
               Good fit      Overfitting!
```

### Other Naive Approaches That Fail

**1. Using Test Set for Selection:**
- **Problem**: Test set should only be used for final evaluation
- **Why it fails**: Gives overly optimistic performance estimates
- **Result**: Model may not generalize to truly unseen data

**2. Using All Data for Training:**
- **Problem**: No way to estimate generalization performance
- **Why it fails**: Can't distinguish between good and bad models
- **Result**: Blind selection with no performance guarantees

**3. Using Fixed Validation Set:**
- **Problem**: May not be representative of true data distribution
- **Why it fails**: Single split can be unlucky
- **Result**: Unreliable performance estimates

**4. Using Model Complexity as Proxy:**
- **Problem**: Simpler models aren't always better
- **Why it fails**: Ignores the actual data and problem structure
- **Result**: May choose suboptimal models

### The Information Leakage Problem

**What is Information Leakage?**
Information leakage occurs when information from the test set (or future data) is used during model training or selection.

**Common Sources of Leakage:**
- Using test set for hyperparameter tuning
- Feature engineering based on test set statistics
- Data preprocessing that uses test set information
- Model selection based on test set performance

**Why It's Dangerous:**
- Gives overly optimistic performance estimates
- Models may not generalize to truly unseen data
- Results are not reproducible in real-world settings

**Example of Leakage:**
```python
# WRONG: Using test set for model selection
for model in models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # LEAKAGE!
    if score > best_score:
        best_model = model

# CORRECT: Using validation set for model selection
for model in models:
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)  # No leakage
    if score > best_score:
        best_model = model
```

### The Need for Proper Validation

**The Solution:**
We need a way to estimate generalization performance without using the test set. This is where cross-validation comes in.

**Key Principles:**
1. **Separate concerns**: Use different data for training, validation, and testing
2. **Unbiased estimation**: Get reliable estimates of generalization performance
3. **Robust selection**: Choose models that generalize well
4. **Avoid leakage**: Never use test set for model selection

**The Validation Strategy:**
```
Data Split:
Training Set (60-80%)  →  Train models
Validation Set (10-20%) →  Select best model
Test Set (10-20%)      →  Final evaluation
```

This approach ensures that:
- We can train multiple models
- We can compare them fairly
- We get reliable performance estimates
- We avoid information leakage

---

## Cross Validation: The Gold Standard

### What is Cross-Validation? The Art of Simulating New Data

Cross validation is a family of techniques that help us estimate how well a model will perform on new data, *without* using the test set. The core idea is to simulate the process of seeing new data by holding out part of the training data, training the model on the rest, and evaluating it on the held-out part.

**The Core Principle:**
- Split the data into training and validation sets
- Train on one part, evaluate on the other
- This gives us an unbiased estimate of generalization performance

**Visual Analogy: The Practice Test Strategy**
Think of cross-validation like taking practice exams:
- **Training set**: Study materials and practice problems
- **Validation set**: Practice exam that simulates the real test
- **Test set**: The actual exam (only used once at the end)

**Why Cross-Validation Works:**
1. **Simulates real-world conditions**: We evaluate on data the model hasn't seen
2. **Provides unbiased estimates**: No information leakage from test set
3. **Enables fair comparison**: All models evaluated on same data
4. **Reduces variance**: Multiple evaluations give more stable estimates

### Hold-out Cross Validation (Simple Cross Validation): The Foundation

**Algorithm:**
1. **Step 1:** Randomly split your data into a training set and a validation (hold-out) set. A common split is 70% training, 30% validation, but this can vary.
2. **Step 2:** Train each candidate model on the training set.
3. **Step 3:** Evaluate each model on the validation set, and pick the one with the lowest validation error.

**Why does this work?**  
The validation set acts as a "proxy" for new, unseen data. By evaluating on data the model hasn't seen, we get a better estimate of its true generalization ability.

**Mathematical Formulation:**
For each model $`M_i`$, we compute:
$`\hat{\varepsilon}_{\text{val}}(h_i) = \frac{1}{|S_{\text{val}}|} \sum_{(x,y) \in S_{\text{val}}} L(h_i(x), y)`$

Then select: $`M^* = \arg\min_i \hat{\varepsilon}_{\text{val}}(h_i)`$

**Practical Implementation:**
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Try different models
models = {
    'linear': LinearRegression(),
    'polynomial_2': PolynomialFeatures(degree=2),
    'polynomial_3': PolynomialFeatures(degree=3),
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=0.1)
}

best_model = None
best_score = float('inf')

for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    score = mean_squared_error(y_val, y_pred)
    
    print(f"{name}: {score:.4f}")
    
    if score < best_score:
        best_score = score
        best_model = name

print(f"\nBest model: {best_model}")
```

**Practical Considerations:**
- **Large datasets**: You can afford to set aside a substantial validation set
- **Small datasets**: You may need more efficient use of data—this is where k-fold cross validation comes in
- **Stratification**: For classification, ensure the validation set has similar class proportions

### Validation Set Size Guidelines: Finding the Right Balance

By testing/validating on a set of examples $`S_{\text{cv}}`$ that the models were not trained on, we obtain a better estimate of each hypothesis $`h_i`$'s true generalization/test error. Thus, this approach is essentially picking the model with the smallest estimated generalization/test error.

**Size Guidelines:**
- **Typical split**: 70% training, 30% validation
- **Large datasets**: Can use smaller validation fractions (e.g., 10-20%)
- **Small datasets**: May need larger validation fractions (e.g., 40-50%)

**Examples:**
- **Small dataset (n=100)**: 70% training, 30% validation
- **Medium dataset (n=10,000)**: 80% training, 20% validation  
- **Large dataset (n=1,000,000)**: 90% training, 10% validation

**ImageNet Example:**
For the ImageNet dataset that has about 1M training images, the validation set is sometimes set to be 50K images, which is only about 5% of the total examples.

**The Trade-off:**
```
Validation Set Size:  Small  →  Medium  →  Large
Training Set Size:   Large  →  Medium  →  Small
Estimate Variance:   High   →  Medium  →  Low
Estimate Bias:       Low    →  Medium  →  High
```

**Guidelines for Different Scenarios:**
- **Very small datasets (n < 100)**: Use k-fold cross-validation
- **Small datasets (100 ≤ n < 1000)**: 70-30 split
- **Medium datasets (1000 ≤ n < 10000)**: 80-20 split
- **Large datasets (n ≥ 10000)**: 90-10 split or smaller validation

### Optional Retraining Step: Using All Available Data

Optionally, step 3 in the algorithm may also be replaced with selecting the model $`M_i`$ according to $`\arg\min_i \hat{\varepsilon}_{S_{\text{cv}}}(h_i)`$, and then retraining $`M_i`$ on the entire training set $`S`$.

**When to retrain:**
- ✅ **Good idea**: Most learning algorithms benefit from more data
- ❌ **Avoid**: For algorithms sensitive to initialization or data perturbations

**Why retrain?**
- The model selected was trained on only 70% of the data
- Retraining on all data typically improves performance
- Final model should use all available training data

**Implementation:**
```python
# After selecting best model
best_model_name = 'ridge'  # From previous selection
best_model = models[best_model_name]

# Retrain on all training data (train + validation)
X_full_train = np.vstack([X_train, X_val])
y_full_train = np.concatenate([y_train, y_val])

best_model.fit(X_full_train, y_full_train)

# Now evaluate on test set
y_test_pred = best_model.predict(X_test)
final_score = mean_squared_error(y_test, y_test_pred)
print(f"Final test score: {final_score:.4f}")
```

**When NOT to retrain:**
- **Unstable algorithms**: Some algorithms are sensitive to data perturbations
- **Computational constraints**: Retraining might be too expensive
- **Reproducibility concerns**: Need exact same model for comparison

### Limitations of Hold-out Validation: The Data Waste Problem

The disadvantage of using hold out cross validation is that it "wastes" about 30% of the data. Even if we were to take the optional step of retraining the model on the entire training set, it's still as if we're trying to find a good model for a learning problem in which we had $`0.7n`$ training examples, rather than $`n`$ training examples, since we're testing models that were trained on only $`0.7n`$ examples each time.

**When this matters:**
- **Small datasets**: Losing 30% of data can significantly hurt performance
- **Expensive data**: When data collection is costly
- **Rare events**: When positive examples are scarce

**Example - Medical Diagnosis:**
If you have only 100 patients with a rare disease, losing 30 patients for validation might mean:
- Only 70 patients for training
- Reduced statistical power
- Less reliable model selection

**Solution**: Use k-fold cross validation for more efficient data usage.

---

### k-Fold Cross Validation: Maximizing Data Efficiency

#### Motivation: The Data Efficiency Problem

Hold-out validation "wastes" some data, since the model never sees the validation set during training. k-fold cross validation addresses this by rotating the validation set.

**Key Insight:**
Every data point gets to be in the validation set exactly once, and in the training set $`k-1`$ times.

**Visual Analogy: The Round-Robin Tournament**
Think of k-fold as a "round-robin tournament" where every data point gets a chance to be in the validation set:
```
Fold 1: [V][T][T][T][T][T][T][T][T][T]  (1 validation, 9 training)
Fold 2: [T][V][T][T][T][T][T][T][T][T]  (1 validation, 9 training)
Fold 3: [T][T][V][T][T][T][T][T][T][T]  (1 validation, 9 training)
...
Fold 10:[T][T][T][T][T][T][T][T][T][V]  (1 validation, 9 training)
```

#### How it Works: The Algorithm in Detail

**Algorithm:**
1. Split the data into $`k`$ equal-sized "folds"
2. For each fold $`i = 1, 2, \ldots, k`$:
   - Use fold $`i`$ as the validation set
   - Use the remaining $`k-1`$ folds as the training set
   - Train the model and evaluate on the validation fold
   - Record the validation error $`\hat{\varepsilon}_i`$
3. Average the validation errors: $`\hat{\varepsilon}_{\text{CV}} = \frac{1}{k} \sum_{i=1}^k \hat{\varepsilon}_i`$

**Mathematical Formulation:**
For each model $`M_j`$ and fold $`i`$:
$`\hat{\varepsilon}_i^{(j)} = \frac{1}{|S_i|} \sum_{(x,y) \in S_i} L(h_j^{(i)}(x), y)`$

Where $`h_j^{(i)}`$ is model $`M_j`$ trained on all folds except $`i`$.

**Final selection:**
$`M^* = \arg\min_j \frac{1}{k} \sum_{i=1}^k \hat{\varepsilon}_i^{(j)}`$

**Practical Implementation:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

# Define models to compare
models = {
    'ridge_0.1': Ridge(alpha=0.1),
    'ridge_1.0': Ridge(alpha=1.0),
    'ridge_10.0': Ridge(alpha=10.0),
    'lasso_0.1': Lasso(alpha=0.1),
    'lasso_1.0': Lasso(alpha=1.0)
}

# Compare models using 5-fold cross-validation
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    mean_score = -scores.mean()  # Convert back to MSE
    std_score = scores.std()
    
    results[name] = {
        'mean': mean_score,
        'std': std_score,
        'scores': scores
    }
    
    print(f"{name}: {mean_score:.4f} ± {std_score:.4f}")

# Select best model
best_model_name = min(results.keys(), key=lambda x: results[x]['mean'])
print(f"\nBest model: {best_model_name}")
```

#### Common Choices and Trade-offs: Finding the Right k

**Common choices:**  
$`k = 10`$ is popular, but for very small datasets, $`k = n`$ (leave-one-out cross validation) is sometimes used.

**Trade-offs:**  
- **Larger $`k`$**: Less bias (since almost all data is used for training each time), but more computational cost (since you train $`k`$ times)
- **Smaller $`k`$**: More bias but faster computation
- **Leave-one-out**: Unbiased but can be very slow for large datasets

**Guidelines:**
- **$`k = 5`**: Quick and dirty, good for initial exploration
- **$`k = 10`**: Standard choice, good balance of bias and variance
- **$`k = n`** (LOOCV): When data is very scarce

**Decision Framework:**
```
Dataset Size:    Very Small  →  Small  →  Medium  →  Large
Recommended k:   n (LOOCV)   →  10     →  10      →  5
Reason:          Max data     →  Balance →  Balance →  Speed
```

**Computational Considerations:**
```
k = 5:   Train each model 5 times
k = 10:  Train each model 10 times  
k = n:   Train each model n times (very expensive!)
```

#### Computational Complexity: The Cost of Accuracy

A typical choice for the number of folds to use here would be $`k = 10``. While the fraction of data held out each time is now $`1/k`$—much smaller than before—this procedure may also be more computationally expensive than hold-out cross validation, since we now need to train each model $`k`$ times.

**Complexity Analysis:**
- **Hold-out**: Train each model once
- **k-fold**: Train each model $`k`$ times
- **Total cost**: $`k \times \text{number of models} \times \text{training time per model}`$

**Example - Polynomial Regression:**
```python
# Compare polynomial degrees using 10-fold CV
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
results = []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    
    # 10-fold cross-validation
    scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
    mean_mse = -scores.mean()
    
    results.append({
        'degree': degree,
        'mse': mean_mse,
        'std': scores.std()
    })
    
    print(f"Degree {degree}: {mean_mse:.4f} ± {scores.std():.4f}")

# Plot results
degrees = [r['degree'] for r in results]
mses = [r['mse'] for r in results]
stds = [r['std'] for r in results]

plt.errorbar(degrees, mses, yerr=stds, marker='o')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Model Selection via Cross-Validation')
plt.grid(True)
plt.show()
```

#### Leave-One-Out Cross Validation (LOOCV): The Ultimate in Data Efficiency

While $`k = 10`$ is a commonly used choice, in problems in which data is really scarce, sometimes we will use the extreme choice of $`k = n`$ in order to leave out as little data as possible each time. In this setting, we would repeatedly train on all but one of the training examples in $`S`$, and test on that held-out example. The resulting $`n`$ errors are then averaged together to obtain our estimate of the generalization error of a model.

**When to use LOOCV:**
- **Very small datasets** (n < 50)
- **Expensive data collection**
- **When you need the most unbiased estimate possible**

**Advantages:**
- Most unbiased estimate of generalization error
- Uses almost all data for training each time
- No randomness in the split

**Disadvantages:**
- Computationally expensive ($`n`$ training runs)
- High variance in the estimate
- May not be practical for large datasets

**Implementation:**
```python
from sklearn.model_selection import LeaveOneOut

# For very small datasets
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
mean_mse = -scores.mean()
print(f"LOOCV MSE: {mean_mse:.4f}")
```

**When LOOCV is Worth It:**
- **Medical studies**: Each patient is expensive to recruit
- **Rare disease diagnosis**: Very few positive examples
- **Expensive experiments**: Each data point costs significant resources
- **Research validation**: Need most unbiased estimate possible

---

### Cross Validation for Model Evaluation: Beyond Selection

Cross validation isn't just for model selection—it's also a powerful tool for evaluating a single model's performance, especially when you want to report results in a paper or compare algorithms fairly.

**Use Cases:**
- **Research papers**: Report cross-validation performance
- **Algorithm comparison**: Fair comparison between methods
- **Performance estimation**: Get confidence intervals for model performance

**Confidence Intervals:**
```python
# Get confidence interval for model performance
scores = cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')
mean_score = -scores.mean()
std_score = scores.std()

# 95% confidence interval
ci_lower = mean_score - 1.96 * std_score / np.sqrt(10)
ci_upper = mean_score + 1.96 * std_score / np.sqrt(10)

print(f"Performance: {mean_score:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
```

**Practical Tips:**
- Always use a validation set or cross-validation for model selection—never the test set!
- For small datasets, prefer k-fold or leave-one-out cross validation
- For large datasets, a simple hold-out set is often sufficient and computationally efficient
- Use stratified sampling for classification problems
- Fix random seeds for reproducibility

**Stratified Cross-Validation:**
```python
from sklearn.model_selection import StratifiedKFold

# For classification problems
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(classifier, X, y, cv=skf, scoring='accuracy')
```

**Time Series Cross-Validation:**
```python
from sklearn.model_selection import TimeSeriesSplit

# For time series data
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
```

---

## Bayesian Statistics and Regularization: Incorporating Uncertainty and Prior Knowledge

### Introduction to Bayesian Methods: A Different Way of Thinking

So far, we've discussed how to select models and estimate their performance. But how do we *fit* the parameters of a model? And how do we avoid overfitting at the parameter level? This is where statistical estimation and regularization come in.

**The Key Question:**
How do we estimate model parameters in a way that prevents overfitting and incorporates our prior knowledge?

**Two Main Approaches:**
1. **Frequentist**: Treat parameters as fixed but unknown
2. **Bayesian**: Treat parameters as random variables with distributions

**Philosophical Difference:**
- **Frequentist**: "What's the probability of the data given the parameters?"
- **Bayesian**: "What's the probability of the parameters given the data?"

**Real-World Analogy: The Weather Forecast Problem**
Think of parameter estimation like weather forecasting:
- **Frequentist approach**: "Given the current weather patterns, what's the probability it will rain tomorrow?"
- **Bayesian approach**: "Given that it rained today and the forecast was sunny, what's the probability the weather model is correct?"

### The Bayesian Mindset: Embracing Uncertainty

**Key Insight:**
In the Bayesian view, we don't just want a single "best" estimate of parameters—we want to understand our uncertainty about those parameters.

**Why This Matters:**
- **Uncertainty quantification**: We can say "we're 95% confident the parameter is between 0.3 and 0.7"
- **Prior knowledge**: We can incorporate what we already know
- **Robust decisions**: We can make decisions that account for uncertainty
- **Model comparison**: We can compare models more systematically

**Visual Analogy: The Detective's Approach**
Think of Bayesian inference like detective work:
- **Prior**: What we know before seeing the evidence
- **Evidence**: The data we observe
- **Posterior**: What we believe after seeing the evidence
- **Uncertainty**: How confident we are in our conclusions

---

## Frequentist vs Bayesian Approaches: Two Philosophies of Statistics

### Frequentist View: Maximum Likelihood Estimation (MLE)

In the frequentist approach, we treat the model parameters $`\theta`$ as fixed but unknown quantities. Our goal is to find the value of $`\theta`$ that makes the observed data most probable.

**The MLE Principle:**
```math
\theta_{\text{MLE}} = \arg\max_{\theta} \prod_{i=1}^n p(y^{(i)}|x^{(i)}; \theta)
```

**Breaking it down:**
- $`p(y^{(i)}|x^{(i)}; \theta)`$ is the likelihood of observing $`y^{(i)}`$ given $`x^{(i)}`$ and parameters $`\theta`$
- We multiply the likelihoods for all data points (assuming independence)
- We pick the $`\theta`$ that maximizes this product

**Intuition:**  
MLE finds the parameters that "explain" the data best, according to our model.

**Example - Linear Regression:**
For linear regression with Gaussian noise, the likelihood is:
$`p(y^{(i)}|x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)`$

Taking the log and maximizing gives us the familiar least squares solution.

**Properties of MLE:**
- ✅ **Consistent**: Converges to true parameters as $`n \to \infty`$
- ✅ **Efficient**: Has the smallest variance among unbiased estimators
- ❌ **No regularization**: Can overfit with small datasets
- ❌ **No uncertainty quantification**: Doesn't tell us how confident we are

**Practical Implementation:**
```python
import numpy as np
from scipy.optimize import minimize

def negative_log_likelihood(theta, X, y, sigma=1.0):
    """Negative log-likelihood for linear regression with Gaussian noise"""
    predictions = X @ theta
    residuals = y - predictions
    log_likelihood = -0.5 * np.sum(residuals**2) / (sigma**2) - len(y) * np.log(sigma * np.sqrt(2*np.pi))
    return -log_likelihood  # Negative because we minimize

# Generate some data
np.random.seed(42)
X = np.random.randn(100, 2)
true_theta = np.array([1.5, -0.8])
y = X @ true_theta + 0.1 * np.random.randn(100)

# MLE estimation
initial_theta = np.zeros(2)
result = minimize(negative_log_likelihood, initial_theta, args=(X, y))
mle_theta = result.x

print(f"True parameters: {true_theta}")
print(f"MLE estimates: {mle_theta}")
print(f"Difference: {np.linalg.norm(true_theta - mle_theta):.4f}")
```

### Bayesian View: Priors, Posteriors, and Prediction

### The Bayesian Framework: A Complete Probabilistic Model

In the Bayesian approach, we treat $`\theta`$ as a *random variable* with its own probability distribution, reflecting our uncertainty about its true value.

**Key Components:**
- **Prior $`p(\theta)`$:** What we believe about $`\theta`$ before seeing any data
- **Likelihood $`p(y|x, \theta)`$:** How likely the data is, given $`\theta`$
- **Posterior $`p(\theta|S)`$:** What we believe about $`\theta`$ after seeing the data $`S`$

**Visual Analogy: The Learning Process**
Think of Bayesian learning like updating your beliefs:
```
Prior Belief → See Data → Update Belief → Posterior Belief
(Initial)     (Evidence)   (Learning)     (Final)
```

**Example - Coin Flipping:**
- **Prior**: "I think the coin is fair (50% heads)"
- **Data**: Flip 10 times, get 7 heads
- **Posterior**: "Given this data, I think the coin is biased toward heads (maybe 60-70%)"

### Bayes' Rule for Parameters: The Fundamental Equation

**The Fundamental Equation:**
```math
p(\theta|S) = \frac{p(S|\theta)p(\theta)}{p(S)} = \frac{\left(\prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta)\right)p(\theta)}{\int_{\theta} \left(\prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta)p(\theta)\right) d\theta}
```

**Understanding Each Term:**
1. **$`p(S|\theta)`$**: The likelihood of the data given the parameters
2. **$`p(\theta)`$**: The prior distribution over parameters
3. **$`p(S)`$**: The marginal likelihood (normalizing constant)
4. **$`p(\theta|S)`$**: The posterior distribution over parameters

**The Denominator:**
The denominator ensures the posterior is a valid probability distribution (integrates to 1). It's often the most challenging part to compute.

**Intuition:**
- **Numerator**: How likely is this parameter value AND how much do we believe in it?
- **Denominator**: Normalize so probabilities sum to 1
- **Result**: Updated beliefs about parameters given the data

**Practical Example - Bayesian Coin Flipping:**
```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# Prior: Beta(2, 2) - slightly favoring fair coin
prior_alpha, prior_beta = 2, 2

# Data: 7 heads out of 10 flips
heads, total = 7, 10

# Posterior: Beta(2+7, 2+3) = Beta(9, 5)
posterior_alpha = prior_alpha + heads
posterior_beta = prior_beta + (total - heads)

# Plot prior and posterior
theta = np.linspace(0, 1, 100)
prior = beta.pdf(theta, prior_alpha, prior_beta)
posterior = beta.pdf(theta, posterior_alpha, posterior_beta)

plt.figure(figsize=(10, 6))
plt.plot(theta, prior, 'b-', label='Prior: Beta(2, 2)', linewidth=2)
plt.plot(theta, posterior, 'r-', label='Posterior: Beta(9, 5)', linewidth=2)
plt.axvline(0.5, color='k', linestyle='--', alpha=0.5, label='Fair coin')
plt.axvline(heads/total, color='g', linestyle='--', alpha=0.5, label=f'Data: {heads}/{total}')
plt.xlabel('Probability of Heads')
plt.ylabel('Density')
plt.title('Bayesian Coin Flipping: Prior vs Posterior')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Point estimates
prior_mean = prior_alpha / (prior_alpha + prior_beta)
posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
posterior_std = np.sqrt((posterior_alpha * posterior_beta) / 
                       ((posterior_alpha + posterior_beta)**2 * (posterior_alpha + posterior_beta + 1)))

print(f"Prior mean: {prior_mean:.3f}")
print(f"Posterior mean: {posterior_mean:.3f} ± {posterior_std:.3f}")
print(f"95% credible interval: [{beta.ppf(0.025, posterior_alpha, posterior_beta):.3f}, "
      f"{beta.ppf(0.975, posterior_alpha, posterior_beta):.3f}]")
```

### Bayesian Prediction: Making Predictions with Uncertainty

**Fully Bayesian Prediction:**  
To predict for a new $`x`$, we average over all possible $`\theta`$ weighted by their posterior probability:

```math
p(y|x, S) = \int_{\theta} p(y|x, \theta)p(\theta|S)d\theta
```

This is called "fully Bayesian" prediction because we're using the entire posterior distribution.

**Expected Value Prediction:**  
If $`y`$ is continuous, we might want the expected value:

```math
\mathbb{E}[y|x, S] = \int_{y} y p(y|x, S) dy
```

**Example - Bayesian Linear Regression:**
For linear regression with Gaussian prior and likelihood, the posterior is also Gaussian, and predictions have both a mean and variance (uncertainty).

**Practical Implementation:**
```python
from scipy.stats import multivariate_normal

def bayesian_linear_regression(X_train, y_train, X_test, prior_mean, prior_cov, noise_var=1.0):
    """
    Bayesian linear regression with Gaussian prior and likelihood
    """
    # Posterior parameters
    prior_precision = np.linalg.inv(prior_cov)
    data_precision = X_train.T @ X_train / noise_var
    
    posterior_precision = prior_precision + data_precision
    posterior_cov = np.linalg.inv(posterior_precision)
    
    posterior_mean = posterior_cov @ (
        prior_precision @ prior_mean + 
        X_train.T @ y_train / noise_var
    )
    
    # Predictions
    predictions = X_test @ posterior_mean
    
    # Prediction uncertainty
    prediction_var = noise_var + np.diag(X_test @ posterior_cov @ X_test.T)
    prediction_std = np.sqrt(prediction_var)
    
    return predictions, prediction_std, posterior_mean, posterior_cov

# Example usage
np.random.seed(42)
X_train = np.random.randn(50, 2)
true_theta = np.array([1.5, -0.8])
y_train = X_train @ true_theta + 0.1 * np.random.randn(50)

X_test = np.random.randn(20, 2)

# Prior: centered at zero with some uncertainty
prior_mean = np.zeros(2)
prior_cov = np.eye(2) * 10.0

# Bayesian predictions
predictions, pred_std, post_mean, post_cov = bayesian_linear_regression(
    X_train, y_train, X_test, prior_mean, prior_cov
)

print(f"True parameters: {true_theta}")
print(f"Posterior mean: {post_mean}")
print(f"Posterior std: {np.sqrt(np.diag(post_cov))}")

# Plot predictions with uncertainty
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], y_train, alpha=0.6, label='Training data')
plt.scatter(X_test[:, 0], predictions, color='red', alpha=0.6, label='Predictions')
plt.fill_between(X_test[:, 0], 
                predictions - 2*pred_std, 
                predictions + 2*pred_std, 
                alpha=0.3, color='red', label='95% confidence')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Bayesian Linear Regression Predictions')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_train[:, 1], y_train, alpha=0.6, label='Training data')
plt.scatter(X_test[:, 1], predictions, color='red', alpha=0.6, label='Predictions')
plt.fill_between(X_test[:, 1], 
                predictions - 2*pred_std, 
                predictions + 2*pred_std, 
                alpha=0.3, color='red', label='95% confidence')
plt.xlabel('Feature 2')
plt.ylabel('Target')
plt.title('Bayesian Linear Regression Predictions')
plt.legend()

plt.tight_layout()
plt.show()
```

### Choosing the Likelihood Function: The Data Model

In the equation above, $`p(y^{(i)}|x^{(i)}, \theta)`$ comes from whatever model you're using for your learning problem. For example, if you are using Bayesian logistic regression, then you might choose:

$`p(y^{(i)}|x^{(i)}, \theta) = h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x^{(i)}))^{(1-y^{(i)})}`$

Where $`h_\theta(x) = \frac{1}{1 + \exp(-\theta^T x)}`$ is the sigmoid function.

**Common Likelihood Functions:**
- **Linear regression**: Gaussian likelihood
- **Logistic regression**: Bernoulli likelihood
- **Poisson regression**: Poisson likelihood
- **Classification**: Categorical likelihood

---

## MAP Estimation and Regularization: The Connection

### The MAP Approximation: A Practical Compromise

Computing the full posterior is often intractable, so we approximate it by its mode—the most probable value, called the **maximum a posteriori (MAP)** estimate:

```math
\theta_{\text{MAP}} = \arg\max_{\theta} \prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta)p(\theta)
```

**Key Insight:**
This is like MLE, but with an extra term for the prior $`p(\theta)`$.

**Visual Analogy: The Mountain Climbing Problem**
Think of MAP estimation like finding the highest peak in a mountain range:
- **MLE**: Find the highest point in the data likelihood "mountain"
- **MAP**: Find the highest point in the posterior "mountain" (likelihood × prior)
- **Prior**: Acts like a "gravity" that pulls us toward certain regions

### Connection to Regularization: The Mathematical Bridge

**Gaussian Prior → L2 Regularization:**
If the prior is Gaussian, $`\theta \sim \mathcal{N}(0, \tau^2 I)`$, MAP estimation is equivalent to adding L2 regularization (ridge regression) in linear models.

**Derivation:**
1. Prior: $`p(\theta) \propto \exp\left(-\frac{\|\theta\|_2^2}{2\tau^2}\right)`$
2. Log-posterior: $`\log p(\theta|S) = \sum_{i=1}^n \log p(y^{(i)}|x^{(i)}, \theta) - \frac{\|\theta\|_2^2}{2\tau^2} + \text{const}`$
3. MAP objective: $`\arg\max_\theta \sum_{i=1}^n \log p(y^{(i)}|x^{(i)}, \theta) - \lambda \|\theta\|_2^2`$

Where $`\lambda = \frac{1}{2\tau^2}`$.

**Laplace Prior → L1 Regularization:**
If the prior is Laplace, $`p(\theta) \propto \exp(-\lambda \|\theta\|_1)`$, MAP estimation gives L1 regularization (LASSO).

**Practical Note:**  
Regularization helps prevent overfitting by discouraging large parameter values, which often correspond to overly complex models.

**Implementation Example:**
```python
from sklearn.linear_model import Ridge, Lasso
from scipy.optimize import minimize

def map_estimation_with_gaussian_prior(X, y, prior_std=1.0, noise_std=1.0):
    """
    MAP estimation with Gaussian prior (equivalent to Ridge regression)
    """
    def objective(theta):
        # Log-likelihood (assuming Gaussian noise)
        predictions = X @ theta
        residuals = y - predictions
        log_likelihood = -0.5 * np.sum(residuals**2) / (noise_std**2)
        
        # Log-prior (Gaussian)
        log_prior = -0.5 * np.sum(theta**2) / (prior_std**2)
        
        return -(log_likelihood + log_prior)  # Negative because we minimize
    
    # Optimize
    initial_theta = np.zeros(X.shape[1])
    result = minimize(objective, initial_theta)
    return result.x

# Compare MAP with Ridge regression
np.random.seed(42)
X = np.random.randn(100, 5)
true_theta = np.array([1.0, -0.5, 0.3, 0.0, 0.0])  # Some zero coefficients
y = X @ true_theta + 0.1 * np.random.randn(100)

# MAP estimation
map_theta = map_estimation_with_gaussian_prior(X, y, prior_std=1.0)

# Ridge regression (should be equivalent)
ridge = Ridge(alpha=1.0/(2*1.0**2))  # alpha = 1/(2*prior_std^2)
ridge.fit(X, y)
ridge_theta = ridge.coef_

print("Comparison of MAP and Ridge:")
print(f"True parameters: {true_theta}")
print(f"MAP estimates:   {map_theta}")
print(f"Ridge estimates: {ridge_theta}")
print(f"MAP vs Ridge difference: {np.linalg.norm(map_theta - ridge_theta):.6f}")
```

---

## Example: Bayesian Logistic Regression: A Complete Case Study

### The Model: Binary Classification with Uncertainty

Suppose you're doing binary classification with logistic regression. In the Bayesian view, you put a prior on the weights $`\theta`$, and your predictions average over all plausible values of $`\theta`$ given the data.

**Likelihood:**
$`p(y^{(i)}|x^{(i)}, \theta) = h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x^{(i)}))^{(1-y^{(i)})}`$

Where $`h_\theta(x) = \frac{1}{1 + \exp(-\theta^T x)}`$ is the sigmoid function.

**Prior:**
$`\theta \sim \mathcal{N}(0, \tau^2 I)`$ (Gaussian prior)

**Posterior:**
$`p(\theta|S) \propto \left(\prod_{i=1}^n h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x^{(i)}))^{(1-y^{(i)})}\right) \exp\left(-\frac{\|\theta\|_2^2}{2\tau^2}\right)`$

### MAP Estimation: The Practical Approach

In practice, you might use the MAP estimate, which is equivalent to regularized logistic regression:

$`\theta_{\text{MAP}} = \arg\max_\theta \sum_{i=1}^n [y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))] - \lambda \|\theta\|_2^2`$

This is exactly the objective function for L2-regularized logistic regression!

**Complete Implementation:**
```python
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bayesian_logistic_regression(X_train, y_train, X_test, prior_std=1.0):
    """
    Bayesian logistic regression with Gaussian prior
    """
    def objective(theta):
        # Log-likelihood
        logits = X_train @ theta
        probs = sigmoid(logits)
        log_likelihood = np.sum(y_train * np.log(probs + 1e-15) + 
                               (1 - y_train) * np.log(1 - probs + 1e-15))
        
        # Log-prior
        log_prior = -0.5 * np.sum(theta**2) / (prior_std**2)
        
        return -(log_likelihood + log_prior)
    
    # MAP estimation
    initial_theta = np.zeros(X_train.shape[1])
    result = minimize(objective, initial_theta, method='L-BFGS-B')
    map_theta = result.x
    
    # Predictions
    logits_test = X_test @ map_theta
    probs_test = sigmoid(logits_test)
    predictions = (probs_test > 0.5).astype(int)
    
    return map_theta, probs_test, predictions

# Generate synthetic data
np.random.seed(42)
n_samples, n_features = 200, 3
X = np.random.randn(n_samples, n_features)
true_theta = np.array([1.5, -0.8, 0.3])
logits = X @ true_theta
probs = sigmoid(logits)
y = (np.random.rand(n_samples) < probs).astype(int)

# Split data
train_idx = np.random.choice(n_samples, 150, replace=False)
test_idx = np.setdiff1d(np.arange(n_samples), train_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Bayesian logistic regression
map_theta, probs_test, predictions = bayesian_logistic_regression(X_train, y_train, X_test)

# Compare with sklearn
sklearn_lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=42)
sklearn_lr.fit(X_train, y_train)
sklearn_predictions = sklearn_lr.predict(X_test)

print("Bayesian Logistic Regression Results:")
print(f"True parameters: {true_theta}")
print(f"MAP estimates: {map_theta}")
print(f"Sklearn estimates: {sklearn_lr.coef_[0]}")
print(f"Bayesian accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"Sklearn accuracy: {accuracy_score(y_test, sklearn_predictions):.3f}")

# Plot predictions with uncertainty
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], y_test, alpha=0.6, label='True labels')
plt.scatter(X_test[:, 0], probs_test, color='red', alpha=0.6, label='Predicted probabilities')
plt.xlabel('Feature 1')
plt.ylabel('Probability')
plt.title('Bayesian Logistic Regression Predictions')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 1], y_test, alpha=0.6, label='True labels')
plt.scatter(X_test[:, 1], probs_test, color='red', alpha=0.6, label='Predicted probabilities')
plt.xlabel('Feature 2')
plt.ylabel('Probability')
plt.title('Bayesian Logistic Regression Predictions')
plt.legend()

plt.tight_layout()
plt.show()
```

---

## Practical Guidelines: When to Use Each Approach

### When to Use Each Approach: A Decision Framework

**Use MLE when:**
- ✅ You have lots of data
- ✅ You want computational efficiency
- ✅ You don't have strong prior beliefs
- ❌ You have small datasets (risk of overfitting)
- ❌ You need uncertainty quantification

**Use MAP when:**
- ✅ You have some prior knowledge
- ✅ You want regularization
- ✅ You want computational efficiency
- ✅ You have small to medium datasets
- ❌ You need full uncertainty quantification

**Use Full Bayesian when:**
- ✅ You need uncertainty quantification
- ✅ You have strong prior knowledge
- ✅ You want the most principled approach
- ❌ You need computational efficiency
- ❌ You have very large datasets

**Decision Tree:**
```
Do you need uncertainty quantification?
├─ Yes → Full Bayesian
└─ No → Do you have prior knowledge?
    ├─ Yes → MAP
    └─ No → Do you have lots of data?
        ├─ Yes → MLE
        └─ No → MAP (for regularization)
```

### Choosing Priors: The Art of Prior Specification

**Conjugate Priors:**
- **Gaussian prior + Gaussian likelihood → Gaussian posterior**
- **Beta prior + Bernoulli likelihood → Beta posterior**
- **Gamma prior + Poisson likelihood → Gamma posterior**

**Non-informative Priors:**
- **Uniform prior**: $`p(\theta) \propto 1`$
- **Jeffreys prior**: $`p(\theta) \propto \sqrt{I(\theta)}`$ where $`I(\theta)`$ is the Fisher information

**Informative Priors:**
- Based on domain knowledge
- Previous studies or experiments
- Expert opinion

**Practical Guidelines:**
```python
# Example: Choosing priors for different scenarios

# 1. Non-informative prior (when you know nothing)
non_informative_prior = lambda theta: 1.0  # Uniform

# 2. Weakly informative prior (when you have some idea)
weakly_informative_prior = lambda theta: np.exp(-0.1 * np.sum(theta**2))  # N(0, 10)

# 3. Informative prior (when you have strong beliefs)
informative_prior = lambda theta: np.exp(-10 * np.sum((theta - [1.0, -0.5])**2))  # N([1, -0.5], 0.1)

# 4. Hierarchical prior (when you have multiple related parameters)
def hierarchical_prior(theta, hyperprior_params):
    # theta ~ N(0, sigma^2), sigma ~ Gamma(a, b)
    sigma = hyperprior_params['sigma']
    return np.exp(-0.5 * np.sum(theta**2) / (sigma**2))
```

### Computational Considerations: The Practical Reality

**MLE:**
- Usually fast and straightforward
- Can use standard optimization methods
- No need to specify priors

**MAP:**
- Similar computational cost to MLE
- Need to choose and justify priors
- Can use the same optimization methods

**Full Bayesian:**
- Often computationally expensive
- May require MCMC or variational methods
- Provides uncertainty quantification

**Computational Complexity Comparison:**
```
Method:           MLE     →  MAP     →  Full Bayesian
Speed:            Fast    →  Fast    →  Slow
Memory:           Low     →  Low     →  High
Uncertainty:      None    →  Point   →  Full distribution
Implementation:   Easy    →  Easy    →  Complex
```

**When Computational Cost Matters:**
- **Real-time applications**: Use MLE or MAP
- **Large datasets**: Use MLE or MAP
- **Research/analysis**: Use Full Bayesian when possible
- **Production systems**: Use MAP for good balance
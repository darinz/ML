# 9.3 Model selection via cross validation

Selecting the right model is one of the most important—and challenging—tasks in machine learning. The "model" could mean the type of algorithm (e.g., linear regression, SVM, neural network), or it could mean the specific settings or complexity of a model (e.g., the degree of a polynomial, the number of layers in a neural network, or the regularization strength in ridge regression).

**Why is model selection important?**
If our model is too simple, it may not capture the underlying patterns in the data (high bias, underfitting). If it's too complex, it may fit the noise in the training data rather than the true signal (high variance, overfitting). The art and science of model selection is about finding the "sweet spot" between these two extremes.

**Example:**  
Suppose you're fitting a curve to data points. If you use a straight line (degree 1 polynomial), it might miss important bends in the data. If you use a degree 10 polynomial, it might wiggle wildly to pass through every point, capturing noise rather than the true trend.

**Key Question:**  
How do we choose the best model or the best complexity for our data, *without* peeking at the test set (which would give us an overly optimistic estimate of performance)?

---

## Python Example: Polynomial Model Selection

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data
def true_func(x):
    return np.sin(2 * np.pi * x)

np.random.seed(0)
x = np.sort(np.random.rand(100))
y = true_func(x) + np.random.randn(100) * 0.1

# Split into train/validation
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)

# Try polynomial degrees 1 to 10
train_errors = []
val_errors = []
for degree in range(1, 11):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(x_train[:, None])
    X_val_poly = poly.transform(x_val[:, None])
    model = LinearRegression().fit(X_train_poly, y_train)
    y_train_pred = model.predict(X_train_poly)
    y_val_pred = model.predict(X_val_poly)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    val_errors.append(mean_squared_error(y_val, y_val_pred))

print("Train errors:", train_errors)
print("Validation errors:", val_errors)
```

---

Suppose we are trying select among several different models for a learning problem. For instance, we might be using a polynomial regression model $`h_\theta(x) = g(\theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_k x^k)`$, and wish to decide if $`k`$ should be $`0, 1, \ldots, 10`$. How can we automatically select a model that represents a good tradeoff between the twin evils of bias and variance?[$`^5`$] Alternatively, suppose we want to automatically choose the bandwidth parameter $`\tau`$ for locally weighted regression, or the parameter $`C`$ for our $`\ell_1`$-regularized SVM. How can we do that?

For the sake of concreteness, in these notes we assume we have some finite set of models $`\mathcal{M} = \{M_1, \ldots, M_d\}`$ that we're trying to select among. For instance, in our first example above, the model $`M_i`$ would be an $`i`$-th degree polynomial regression model. (The generalization to infinite $`\mathcal{M}`$ is not hard.[$`^6`$]) Alternatively, if we are trying to decide between using an SVM, a neural network or logistic regression, then $`\mathcal{M}`$ may contain these models.

[$`^5`$]: Given that we said in the previous set of notes that bias and variance are two very different beasts, some readers may be wondering if we should be calling them "twin" evils here. Perhaps it'd be better to think of them as non-identical twins. The phrase "the fraternal twin evils of bias and variance" doesn't have the same ring to it, though.

[$`^6`$]: If we are trying to choose from an infinite set of models, say corresponding to the possible values of the bandwidth $`\tau \in \mathbb{R}^+`$, we may discretize $`\tau`$ and consider only a finite number of possible values for it. More generally, most of the algorithms described here can all be viewed as performing optimization search in the space of models, and we can perform this search over infinite model classes as well.

---

## The Pitfall of Naive Model Selection

A tempting but flawed approach is to simply pick the model that fits the training data best (i.e., has the lowest training error). However, this almost always leads to overfitting: the most complex model will always fit the training data best, but may perform poorly on new, unseen data.

**Analogy:**  
Imagine memorizing answers to practice exam questions. You'll ace the practice test, but if the real exam has different questions, you might struggle. Similarly, a model that "memorizes" the training data may not generalize well.

---

## Cross Validation: The Gold Standard

Cross validation is a family of techniques that help us estimate how well a model will perform on new data, *without* using the test set. The core idea is to simulate the process of seeing new data by holding out part of the training data, training the model on the rest, and evaluating it on the held-out part.

### Hold-out Cross Validation (Simple Cross Validation)

- **Step 1:** Randomly split your data into a training set and a validation (hold-out) set. A common split is 70% training, 30% validation, but this can vary.
- **Step 2:** Train each candidate model on the training set.
- **Step 3:** Evaluate each model on the validation set, and pick the one with the lowest validation error.

**Why does this work?**  
The validation set acts as a "proxy" for new, unseen data. By evaluating on data the model hasn't seen, we get a better estimate of its true generalization ability.

**Practical Note:**  
If your dataset is very large, you can afford to set aside a substantial validation set. If your dataset is small, you may need more efficient use of data—this is where k-fold cross validation comes in.

---

## Python Example: Hold-out Cross Validation

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Example: Ridge regression with hold-out validation
X, y = np.random.randn(100, 5), np.random.randn(100)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
model = Ridge(alpha=1.0).fit(X_train, y_train)
y_pred = model.predict(X_val)
val_error = mean_squared_error(y_val, y_pred)
print("Validation error:", val_error)
```

---

By testing/validating on a set of examples $`S_{\text{cv}}`$ that the models were not trained on, we obtain a better estimate of each hypothesis $`h_i`$'s true generalization/test error. Thus, this approach is essentially picking the model with the smallest estimated generalization/test error. The size of the validation set depends on the total number of available examples. Usually, somewhere between $`1/4 - 1/3`$ of the data is used in the hold out cross validation set, and 30% is a typical choice. However, when the total dataset is huge, validation set can be a smaller fraction of the total examples as long as the absolute number of validation examples is decent. For example, for the ImageNet dataset that has about 1M training images, the validation set is sometimes set to be 50K images, which is only about 5% of the total examples.

Optionally, step 3 in the algorithm may also be replaced with selecting the model $`M_i`$ according to $`\arg\min_i \hat{\varepsilon}_{S_{\text{cv}}}(h_i)`$, and then retraining $`M_i`$ on the entire training set $`S`$. (This is often a good idea, with one exception being learning algorithms that are be very sensitive to perturbations of the initial conditions and/or data. For these methods, $`M_i`$ doing well on $`S_{\text{train}}`$ does not necessarily mean it will also do well on $`S_{\text{cv}}`$, and it might be better to forgo this retraining step.)

The disadvantage of using hold out cross validation is that it "wastes" about 30% of the data. Even if we were to take the optional step of retraining the model on the entire training set, it's still as if we're trying to find a good model for a learning problem in which we had $`0.7n`$ training examples, rather than $`n`$ training examples, since we're testing models that were trained on only $`0.7n`$ examples each time. While this is fine if data is abundant and/or cheap, in learning problems in which data is scarce (consider a problem with $`n = 20`$, say), we'd like to do something better.

---

### k-Fold Cross Validation

**Motivation:**  
Hold-out validation "wastes" some data, since the model never sees the validation set during training. k-fold cross validation addresses this by rotating the validation set.

**How it works:**
- Split the data into $`k`$ equal-sized "folds."
- For each fold:
  - Use that fold as the validation set, and the remaining $`k-1`$ folds as the training set.
  - Train the model, evaluate on the validation fold, and record the error.
- Average the validation errors across all $`k`$ folds to get a robust estimate of model performance.

**Common choices:**  
$`k = 10`$ is popular, but for very small datasets, $`k = n`$ (leave-one-out cross validation) is sometimes used.

**Trade-offs:**  
- Larger $`k`$ means less bias (since almost all data is used for training each time), but more computational cost (since you train $`k`$ times).
- Leave-one-out is unbiased but can be very slow for large datasets.

**Analogy:**  
Think of k-fold as a "round robin tournament" where every data point gets a chance to be in the validation set.

---

## Python Example: k-Fold Cross Validation

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Ridge

X, y = np.random.randn(100, 5), np.random.randn(100)
kf = KFold(n_splits=10)
model = Ridge(alpha=1.0)
scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
print("Mean CV error:", -np.mean(scores))
```

---

A typical choice for the number of folds to use here would be $`k = 10`$. While the fraction of data held out each time is now $`1/k`$—much smaller than before—this procedure may also be more computationally expensive than hold-out cross validation, since we now need train to each model $`k`$ times.

While $`k = 10`$ is a commonly used choice, in problems in which data is really scarce, sometimes we will use the extreme choice of $`k = m`$ in order to leave out as little data as possible each time. In this setting, we would repeatedly train on all but one of the training examples in $`S`$, and test on that held-out example. The resulting $`m = k`$ errors are then averaged together to obtain our estimate of the generalization error of a model. This method has its own name; since we're holding out one training example at a time, this method is called **leave-one-out cross validation**.

---

### Cross Validation for Model Evaluation

Cross validation isn't just for model selection—it's also a powerful tool for evaluating a single model's performance, especially when you want to report results in a paper or compare algorithms fairly.

**Practical Tips:**
- Always use a validation set or cross-validation for model selection—never the test set!
- For small datasets, prefer k-fold or leave-one-out cross validation.
- For large datasets, a simple hold-out set is often sufficient and computationally efficient.

---

# 9.4 Bayesian statistics and regularization

So far, we've discussed how to select models and estimate their performance. But how do we *fit* the parameters of a model? And how do we avoid overfitting at the parameter level? This is where statistical estimation and regularization come in.

## Frequentist View: Maximum Likelihood Estimation (MLE)

In the frequentist approach, we treat the model parameters $`\theta`$ as fixed but unknown quantities. Our goal is to find the value of $`\theta`$ that makes the observed data most probable.

```math
\theta_{\text{MLE}} = \arg\max_{\theta} \prod_{i=1}^n p(y^{(i)}|x^{(i)}; \theta)
```

- $`p(y^{(i)}|x^{(i)}; \theta)`$ is the likelihood of observing $`y^{(i)}`$ given $`x^{(i)}`$ and parameters $`\theta`$.
- We multiply the likelihoods for all data points (assuming independence) and pick the $`\theta`$ that maximizes this product.

**Intuition:**  
MLE finds the parameters that "explain" the data best, according to our model.

---

## Python Example: Maximum Likelihood Estimation (MLE)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=200, n_features=5, random_state=42)
model = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
model.fit(X, y)
print("MLE coefficients:", model.coef_)
```

---

## Bayesian View: Priors, Posteriors, and Prediction

In the Bayesian approach, we treat $`\theta`$ as a *random variable* with its own probability distribution, reflecting our uncertainty about its true value.

- **Prior $`p(\theta)`$:** What we believe about $`\theta`$ before seeing any data.
- **Likelihood $`p(y|x, \theta)`$:** How likely the data is, given $`\theta`$.
- **Posterior $`p(\theta|S)`$:** What we believe about $`\theta`$ after seeing the data $`S`$.

**Bayes' Rule for Parameters:**

```math
p(\theta|S) = \frac{p(S|\theta)p(\theta)}{p(S)} = \frac{\left(\prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta)\right)p(\theta)}{\int_{\theta} \left(\prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta)p(\theta)\right) d\theta}
```

- The denominator ensures the posterior is a valid probability distribution (integrates to 1).

**Prediction:**  
To predict for a new $`x`$, we average over all possible $`\theta`$ weighted by their posterior probability:

```math
p(y|x, S) = \int_{\theta} p(y|x, \theta)p(\theta|S)d\theta
```

- This is called "fully Bayesian" prediction.

**Expected Value Prediction:**  
If $`y`$ is continuous, we might want the expected value:

```math
\mathbb{E}[y|x, S] = \int_{y} y p(y|x, S) dy
```

In the equation above, $`p(y^{(i)}|x^{(i)}, \theta)`$ comes from whatever model you're using for your learning problem. For example, if you are using Bayesian logistic regression, then you might choose $`p(y^{(i)}|x^{(i)}, \theta) = h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x^{(i)}))^{(1-y^{(i)})}`$, where $`h_\theta(x^{(i)}) = 1/(1 + \exp(-\theta^T x^{(i)}))`$.[^7]

---

## Python Example: Bayesian Linear Regression (Posterior Predictive)

```python
import numpy as np
from sklearn.linear_model import BayesianRidge

# Generate synthetic data
np.random.seed(0)
X = np.random.randn(100, 1)
y = 3 * X[:, 0] + np.random.randn(100) * 0.5

# Fit Bayesian linear regression
model = BayesianRidge()
model.fit(X, y)

# Posterior mean prediction for new x
x_new = np.array([[1.5]])
y_mean, y_std = model.predict(x_new, return_std=True)
print("Posterior mean:", y_mean, "Stddev:", y_std)
```

---

## MAP Estimation and Regularization

Computing the full posterior is often intractable, so we approximate it by its mode—the most probable value, called the **maximum a posteriori (MAP)** estimate:

```math
\theta_{\text{MAP}} = \arg\max_{\theta} \prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta)p(\theta)
```

- This is like MLE, but with an extra term for the prior $`p(\theta)`$.
- If the prior is Gaussian, $`\theta \sim \mathcal{N}(0, \tau^2 I)`$, MAP estimation is equivalent to adding L2 regularization (ridge regression) in linear models.

**Practical Note:**  
Regularization helps prevent overfitting by discouraging large parameter values, which often correspond to overly complex models.

---

## Python Example: MAP Estimation (Ridge Regression)

```python
from sklearn.linear_model import Ridge

# Ridge regression is equivalent to MAP with Gaussian prior
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print("MAP coefficients:", ridge.coef_)
```

---

## Example: Bayesian Logistic Regression

Suppose you're doing binary classification with logistic regression. In the Bayesian view, you put a prior on the weights $`\theta`$, and your predictions average over all plausible values of $`\theta`$ given the data. In practice, you might use the MAP estimate, which is equivalent to regularized logistic regression.

---

## Python Example: Bayesian Logistic Regression (MAP)

```python
from sklearn.linear_model import LogisticRegression

# Logistic regression with L2 regularization (MAP estimate)
logreg = LogisticRegression(penalty='l2', C=1.0)
logreg.fit(X, y > np.median(y))
print("MAP coefficients (logistic regression):", logreg.coef_)
```

---

## Summary Table

| Approach      | Parameters $`\theta`$ | Prior $`p(\theta)`$ | Output         | Overfitting Control |
|---------------|-----------------------|----------------------|----------------|---------------------|
| MLE           | Fixed, unknown        | None                 | Best fit       | None                |
| MAP           | Fixed, unknown        | Yes                  | Best fit + prior | Regularization      |
| Bayesian      | Random variable       | Yes                  | Average over posterior | Regularization + Uncertainty |

---

## Footnotes and Practical Tips
- Always use a validation set or cross-validation for model selection—never the test set!
- For small datasets, prefer k-fold or leave-one-out cross validation.
- Regularization is essential when the number of features is large compared to the number of examples.

[^7]: Since we are now viewing $`\theta`$ as a random variable, it is okay to condition on it value, and write "$`p(y|x, \theta)`$" instead of "$`p(y|x; \theta)`$".

[^8]: The integral below would be replaced by a summation if $`y`$ is discrete-valued.
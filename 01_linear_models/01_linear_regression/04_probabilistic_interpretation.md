# 1.3 Probabilistic interpretation

When faced with a regression problem, why might linear regression, and specifically why might the least-squares cost function $J$, be a reasonable choice? In this section, we will give a set of probabilistic assumptions, under which least-squares regression is derived as a very natural algorithm.

## From Optimization Methods to Probabilistic Justification

So far, we've learned how to solve linear regression problems using gradient descent and normal equations. These methods give us practical ways to find the optimal parameters $\theta$ that minimize our cost function. But we haven't addressed a fundamental question: **Why should we use the least squares cost function in the first place?**

The answer requires us to think probabilistically about how our data is generated. By making specific assumptions about the underlying data-generating process, we can show that the least squares approach isn't just a convenient heuristic—it's the **optimal solution** under those assumptions.

This probabilistic interpretation connects our optimization methods to fundamental principles in statistics and provides a theoretical foundation that justifies our choice of cost function. It also opens the door to more sophisticated approaches like Bayesian regression and generalized linear models.

## Why Probabilistic Interpretation?

Before diving into the mathematics, let's understand why a probabilistic interpretation is valuable:

**Benefits of probabilistic thinking:**
1. **Theoretical foundation**: Provides a principled justification for least squares
2. **Uncertainty quantification**: Allows us to understand prediction uncertainty
3. **Model comparison**: Enables comparison of different models using likelihood
4. **Extensions**: Foundation for more sophisticated methods (Bayesian regression, GLMs)
5. **Interpretability**: Helps understand what assumptions we're making

**Key insight**: Least squares isn't just a heuristic - it's the optimal solution under certain probabilistic assumptions.

## Linear Model Assumption

Let us assume that the target variables and the inputs are related via the equation

$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)},
$$

where $y^{(i)}$ is the observed output for the $i$-th data point, $x^{(i)}$ is the corresponding input feature vector, $\theta$ is the parameter vector we wish to learn, and $\epsilon^{(i)}$ is an error term. This model asserts that the relationship between the inputs and outputs is linear, up to some noise or unmodeled effects.

### Understanding the Linear Model

**Components of the model:**
- **Systematic part**: $\theta^T x^{(i)}$ - the predictable relationship
- **Random part**: $\epsilon^{(i)}$ - the unpredictable noise

**What this assumption means:**
1. **Linearity**: The expected value of $y$ is a linear function of $x$
2. **Additive noise**: The noise is added to the systematic part, not multiplied
3. **Deterministic parameters**: $\theta$ is fixed (not random) but unknown

**Example**: For house prices:
- $\theta^T x^{(i)}$ might be: $50 + 0.1 \times \text{area} + 20 \times \text{bedrooms}$
- $\epsilon^{(i)}$ captures: location effects, market timing, unique features, measurement error

> **Matrix invertibility note:** In the above step, we are implicitly assuming that $X^T X$ is an invertible matrix, where $X$ is the design matrix whose rows are the $x^{(i)}$'s. This can be checked before calculating the inverse. If either the number of linearly independent examples is fewer than the number of features, or if the features are not linearly independent, then $X^T X$ will not be invertible. Even in such cases, it is possible to "fix" the situation with additional techniques, such as regularization (ridge regression), which we skip here for the sake of simplicity.

## Gaussian Noise Model

The error term $\epsilon^{(i)}$ captures either unmodeled effects (such as if there are some features very pertinent to predicting housing price, but that we'd left out of the regression), or random noise. To proceed probabilistically, we further assume that the $\epsilon^{(i)}$ are distributed IID (independently and identically distributed) according to a Gaussian (Normal) distribution with mean zero and some variance $\sigma^2$:

$$
\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)
$$

### Why Gaussian Noise?

This is a common assumption in statistics and machine learning, as the Gaussian distribution arises naturally in many contexts (e.g., by the Central Limit Theorem). It also leads to mathematically convenient properties.

**Theoretical justifications:**
1. **Central Limit Theorem**: Sum of many small, independent effects tends to be Gaussian
2. **Maximum entropy**: Gaussian has maximum entropy among distributions with given mean/variance
3. **Mathematical convenience**: Gaussian distributions have nice properties for optimization

**Practical considerations:**
- **Robustness**: Results are often robust to moderate violations of Gaussianity
- **Outliers**: Gaussian assumption can be sensitive to outliers
- **Heteroscedasticity**: Assumes constant variance across all $x$ values

### Properties of Gaussian Noise

The density of $\epsilon^{(i)}$ is given by:

$$
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(\epsilon^{(i)})^2}{2\sigma^2}\right)
$$

**Key properties:**
1. **Symmetry**: Positive and negative errors are equally likely
2. **Bell-shaped**: Most errors are small, large errors are rare
3. **68-95-99.7 rule**: 68% of errors within $\pm\sigma$, 95% within $\pm2\sigma$, 99.7% within $\pm3\sigma$

This means that most of the time, the error is close to zero, but larger deviations are possible with exponentially decreasing probability.

### Independence and Identical Distribution

**Independence**: $\epsilon^{(i)}$ and $\epsilon^{(j)}$ are independent for $i \neq j$
- **Meaning**: Errors for different data points don't influence each other
- **Violation**: Time series data, spatial data, repeated measurements

**Identical distribution**: All $\epsilon^{(i)}$ have the same distribution $\mathcal{N}(0, \sigma^2)$
- **Meaning**: The noise characteristics don't change across data points
- **Violation**: Heteroscedasticity (variance changes with $x$)

## Conditional Distribution of $y^{(i)}$

Given the linear model and the Gaussian noise, the conditional distribution of $y^{(i)}$ given $x^{(i)}$ and $\theta$ is also Gaussian:

$$
p(y^{(i)}|x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)
$$

### Understanding the Conditional Distribution

This says that, for a given input $x^{(i)}$, the output $y^{(i)}$ is most likely to be near $\theta^T x^{(i)}$, but can deviate from it due to noise.

**Key insights:**
1. **Mean**: $E[y^{(i)}|x^{(i)}] = \theta^T x^{(i)}$ - the systematic part
2. **Variance**: $\text{Var}[y^{(i)}|x^{(i)}] = \sigma^2$ - constant across all $x$
3. **Shape**: Gaussian distribution around the mean

**Example**: For a house with 2000 sq ft and 3 bedrooms:
- Expected price: $\theta_0 + \theta_1 \times 2000 + \theta_2 \times 3$
- Actual price: Gaussian around this expected value with variance $\sigma^2$

The notation $p(y^{(i)}|x^{(i)}; \theta)$ indicates that this is the probability density of $y^{(i)}$ given $x^{(i)}$ and parameterized by $\theta$. Note that $\theta$ is not a random variable here; it is a parameter to be estimated.

### Visualizing the Model

**For a single feature $x$:**
- The model predicts a line: $y = \theta_0 + \theta_1 x$
- At each $x$ value, $y$ follows a Gaussian distribution
- The variance $\sigma^2$ determines how spread out the $y$ values are

**For multiple features:**
- The model predicts a hyperplane: $y = \theta^T x$
- The same Gaussian noise applies at each point in feature space

## Likelihood Function

Given $X$ (the design matrix, which contains all the $x^{(i)}$'s) and $\theta$, what is the probability of observing the data $\vec{y}$? The likelihood function is defined as:

$$
L(\theta; X, \vec{y}) = p(\vec{y}|X; \theta)
$$

### Understanding Likelihood

The likelihood function measures how probable our observed data is, given a particular choice of parameters $\theta$.

**Key points:**
1. **Data is fixed**: We observe $\vec{y}$ and $X$, these don't change
2. **Parameters vary**: We consider different values of $\theta$
3. **Not a probability**: Likelihood is not a probability distribution over $\theta$

**Intuition**: "How likely is it that we would observe this data if the true parameters were $\theta$?"

### Factorization Under Independence

Assuming the data points are independent given $\theta$, the likelihood factorizes as a product:

$$
L(\theta) = \prod_{i=1}^n p(y^{(i)} \mid x^{(i)}; \theta)
= \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)
$$

**Why this factorization works:**
- Independence means: $p(\vec{y}|X; \theta) = \prod_{i=1}^n p(y^{(i)}|x^{(i)}; \theta)$
- Each term is the probability of observing $y^{(i)}$ given $x^{(i)}$ and $\theta$

This function measures how probable the observed data is, as a function of $\theta$.

### Likelihood as a Function of $\theta$

**For different $\theta$ values:**
- **Good $\theta$**: High likelihood (data is probable)
- **Bad $\theta$**: Low likelihood (data is improbable)
- **Best $\theta$**: Maximum likelihood

**Example**: If $\theta$ predicts house prices close to observed prices, likelihood is high.

## Maximum Likelihood Estimation (MLE)

The principle of **maximum likelihood** says that we should choose $\theta$ to maximize the likelihood $L(\theta)$, i.e., to make the observed data as probable as possible under our model. This is a general principle in statistics and machine learning for parameter estimation.

### Why Maximum Likelihood?

**Theoretical justification:**
1. **Consistency**: MLE converges to true parameters as $n \to \infty$
2. **Efficiency**: MLE has smallest variance among unbiased estimators (asymptotically)
3. **Invariance**: If $\hat{\theta}$ is MLE, then $f(\hat{\theta})$ is MLE of $f(\theta)$

**Intuitive justification:**
- "Choose parameters that make our observations most likely"
- "If our model is correct, the true parameters should make our data probable"

### Connection to Other Methods

**MLE vs. Least Squares:**
- **MLE**: Maximize probability of observing data
- **Least Squares**: Minimize sum of squared errors
- **Under Gaussian noise**: These are equivalent!

**MLE vs. MAP (Maximum A Posteriori):**
- **MLE**: $\hat{\theta} = \arg\max_\theta L(\theta)$
- **MAP**: $\hat{\theta} = \arg\max_\theta L(\theta) p(\theta)$ (includes prior)

## Log-Likelihood and Connection to Least Squares

Maximizing the likelihood is equivalent to maximizing any strictly increasing function of it, such as the log-likelihood. Taking the logarithm simplifies the product into a sum, which is easier to work with:

$$
\begin{align*}
\ell(\theta) &= \log L(\theta) \\
&= \log \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right) \\
&= \sum_{i=1}^n \log \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right) \\
&= n \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{1}{2\sigma^2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2
\end{align*}
$$

### Step-by-Step Derivation

**Step 1: Take logarithm of product**
$\log \prod_{i=1}^n f_i = \sum_{i=1}^n \log f_i$

**Step 2: Expand each term**
$\log \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right) = \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}$

**Step 3: Sum over all terms**
The first term becomes $n \log \frac{1}{\sqrt{2\pi\sigma^2}}$ (constant)
The second term becomes $-\frac{1}{2\sigma^2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2$

### The Key Insight

Notice that the first term does not depend on $\theta$, so maximizing $\ell(\theta)$ is equivalent to minimizing the sum of squared errors:

$$
\frac{1}{2} \sum_{i=1}^n (y^{(i)} - \theta^T x^{(i)})^2
$$

This is exactly the least-squares cost function $J(\theta)$ used in linear regression. Thus, under the Gaussian noise assumption, least-squares regression is equivalent to maximum likelihood estimation.

**Why this matters:**
1. **Justification**: Least squares is optimal under Gaussian noise
2. **Unification**: Connects geometric and probabilistic interpretations
3. **Extensions**: Foundation for more sophisticated methods

## Summary and Broader Perspective

To summarize: Under the previous probabilistic assumptions on the data, least-squares regression corresponds to finding the maximum likelihood estimate of $\theta$. This is thus one set of assumptions under which least-squares regression can be justified as a very natural method that's just doing maximum likelihood estimation. 

### Key Takeaways

1. **Gaussian noise + linear model → least squares is optimal**
2. **MLE provides principled justification for least squares**
3. **Log-likelihood connects probability to optimization**
4. **Assumptions matter - check if they hold in practice**

### Alternative Justifications

However, the probabilistic assumptions are by no means **necessary** for least-squares to be a perfectly good and rational procedure, and there may—and indeed there are—other natural assumptions that can also be used to justify it. For example, least-squares can also be motivated from a geometric perspective (minimizing the Euclidean distance between predictions and observations), or as a method of moments estimator.

**Geometric interpretation:**
- Minimize Euclidean distance between predictions and observations
- Project $\vec{y}$ onto the column space of $X$

**Method of moments:**
- Match sample moments to population moments
- First moment: $E[y] = \theta^T E[x]$

> **Practical note:** In real-world data, the Gaussian noise assumption may not always hold. Outliers, heteroscedasticity (non-constant variance), or non-linear relationships can violate the assumptions. In such cases, alternative models or robust regression techniques may be more appropriate.

### When Assumptions Are Violated

**Non-Gaussian noise:**
- **Heavy tails**: Use robust regression (Huber loss, LAD)
- **Skewed**: Consider log transformation or quantile regression
- **Discrete**: Use Poisson regression, logistic regression

**Non-constant variance (heteroscedasticity):**
- **Weighted least squares**: Weight by inverse variance
- **Generalized least squares**: Model the variance structure
- **Transformations**: Log, square root transformations

**Non-linear relationships:**
- **Polynomial regression**: Add polynomial terms
- **Spline regression**: Use piecewise polynomials
- **Kernel methods**: Non-linear feature transformations

## Independence from $\sigma^2$

Note also that, in our previous discussion, our final choice of $\theta$ did not depend on what was $\sigma^2$, and indeed we'd have arrived at the same result even if $\sigma^2$ were unknown. This is because $\sigma^2$ only affects the scaling of the likelihood, not the location of its maximum with respect to $\theta$. We will use this fact again later, when we talk about the exponential family and generalized linear models.

### Understanding This Result

**Why $\sigma^2$ doesn't affect $\hat{\theta}$:**
1. **Scaling factor**: $\sigma^2$ only scales the likelihood function
2. **Location invariance**: Scaling doesn't change where the maximum occurs
3. **Practical implication**: We can estimate $\theta$ without knowing $\sigma^2$

**Estimating $\sigma^2$:**
If we want to estimate $\sigma^2$ as well, we can use:
$$\hat{\sigma}^2 = \frac{1}{n} \sum_{i=1}^n (y^{(i)} - \hat{\theta}^T x^{(i)})^2$$

**Connection to degrees of freedom:**
For unbiased estimation, we often use:
$$\hat{\sigma}^2 = \frac{1}{n-d-1} \sum_{i=1}^n (y^{(i)} - \hat{\theta}^T x^{(i)})^2$$
where $d+1$ is the number of parameters (accounting for degrees of freedom).

### Broader Implications

This independence property is important because:
1. **Robustness**: Our parameter estimates are robust to misspecification of $\sigma^2$
2. **Simplicity**: We can focus on estimating $\theta$ first, then $\sigma^2$ if needed
3. **Generalization**: This property extends to other exponential family distributions

**In practice:**
- We often don't know the true noise variance
- We can still get good parameter estimates
- We can estimate the variance from the residuals if needed

## From Global to Local Models

We've now built a complete theoretical foundation for linear regression, understanding both how to solve the optimization problem (gradient descent and normal equations) and why the least squares approach makes sense (probabilistic interpretation). Our models assume a **global linear relationship** between features and target, which works well when the data truly follows a linear pattern.

However, real-world data is often more complex. The relationship between features and target might be **locally linear** but **globally non-linear**. For example, house prices might follow different patterns in different neighborhoods, or the effect of temperature on energy consumption might vary by season.

This motivates our final topic: **locally weighted linear regression (LWR)**, which adapts the linear model to capture local structure in the data. Instead of fitting one global model, LWR fits a separate linear model for each prediction point, giving more weight to nearby training examples.

This approach bridges the gap between simple parametric models and complex non-linear methods, showing how we can extend linear regression to handle more sophisticated data patterns while maintaining interpretability.

---

**Previous: [Normal Equations](03_normal_equations.md)** - Learn about the closed-form solution to linear regression using normal equations.

**Next: [Locally Weighted Linear Regression](05_locally_weighted_linear_regression.md)** - Explore non-parametric approaches to linear regression that adapt to local data structure.
# Practice Problems 5 Solutions

## 1. One Answer

Imagine you are building a machine learning model to predict the stopping distance of cars based on their speed.

You obtain a large dataset where each data point is a pair of observed speeds and stopping distances, and you decide to use a simple linear regression model to predict stopping distances from speed.

However, in reality, the stopping distance increases quadratically with speed.
As a result, your model consistently underestimates the stopping distance at higher speeds.

Compared to using a model that can model a quadratic relationship between stopping distance and speed, would your model have high or low bias?

a. High bias

b. Low bias

Correct answers: (a)

**Explanation:**

The correct answer is **(a) - High bias**. Here's the detailed explanation:

**Understanding the Problem:**

**True Relationship:**
The stopping distance increases **quadratically** with speed:
$$\text{Stopping Distance} = k \cdot \text{Speed}^2$$

**Model Assumption:**
Your linear regression model assumes a **linear** relationship:
$$\text{Stopping Distance} = w \cdot \text{Speed} + b$$

**Why This Creates High Bias:**

**1. Model Mismatch:**
- **True relationship**: Quadratic (curved)
- **Model assumption**: Linear (straight line)
- The model is fundamentally incapable of capturing the true relationship

**2. Systematic Underestimation:**
- At higher speeds, the quadratic relationship grows much faster than linear
- Your linear model will consistently underestimate stopping distances
- This creates **systematic error** (bias)

**Mathematical Analysis:**

**True Function:**
$$f_{\text{true}}(x) = kx^2$$

**Linear Model:**
$$f_{\text{model}}(x) = wx + b$$

**Bias at Point $x$:**
$$\text{Bias}(x) = \mathbb{E}[f_{\text{model}}(x)] - f_{\text{true}}(x) = wx + b - kx^2$$

**As $x$ increases:**
- $f_{\text{true}}(x) = kx^2$ grows quadratically
- $f_{\text{model}}(x) = wx + b$ grows linearly
- The bias becomes increasingly negative (underestimation)

**Visual Example:**
```
Speed:    10    20    30    40    50
True:     100   400   900   1600  2500
Linear:   50    100   150   200   250
Bias:     -50   -300  -750  -1400 -2250
```

**Why Other Options Are Wrong:**

**Option (b) - Low bias:**
- Low bias would mean the model can capture the true relationship well
- A linear model cannot capture a quadratic relationship
- The systematic underestimation creates high bias

**Bias-Variance Tradeoff Context:**

**High Bias (Current Situation):**
- Model is too simple for the data
- Cannot capture the true underlying relationship
- Results in systematic prediction errors

**Low Variance:**
- Linear models are stable and consistent
- Predictions don't vary much with small changes in training data
- But this stability comes at the cost of high bias

**Solutions to Reduce Bias:**

1. **Use a quadratic model**: $f(x) = w_1x^2 + w_2x + b$
2. **Polynomial regression**: Higher degree polynomials
3. **Non-linear basis expansion**: Transform features to capture curvature
4. **Neural networks**: Can learn non-linear relationships

**Conclusion:**
The linear model has **high bias** because it cannot capture the quadratic relationship between speed and stopping distance, leading to systematic underestimation at higher speeds.

## 2. One Answer

Follow the same car scenario as the above question. Compared to using a model that can model a quadratic relationship between stopping distance and speed, would your model have high or low variance?

a. High variance

b. Low variance

Correct answers: (b)

**Explanation:**

The correct answer is **(b) - Low variance**. Here's the detailed explanation:

**Understanding the Problem:**

Continuing from the previous question, we have:
- **True relationship**: Quadratic (stopping distance = $k \cdot \text{speed}^2$)
- **Model assumption**: Linear (stopping distance = $w \cdot \text{speed} + b$)
- **Result**: High bias due to model mismatch

**Why This Creates Low Variance:**

**1. Model Simplicity:**
- Linear models are **simple and stable**
- They have few parameters to learn
- Predictions are consistent across different training sets

**2. Insensitive to Training Data Changes:**
- Small changes in training data don't dramatically affect the learned line
- The model always learns a straight line relationship
- Predictions remain relatively stable

**Mathematical Analysis:**

**Linear Model:**
$$f(x) = wx + b$$

**Parameter Learning:**
- $w$ and $b$ are learned from training data
- Small changes in training data cause small changes in $w$ and $b$
- The overall linear structure remains the same

**Variance Definition:**
$$\text{Variance} = \mathbb{E}[(f(x) - \mathbb{E}[f(x)])^2]$$

**Why Variance is Low:**
- Linear models are **deterministic** given the parameters
- Small parameter changes lead to small prediction changes
- The model cannot "memorize" complex patterns in the data

**Visual Example:**
```
Training Set 1: w=2, b=10
Training Set 2: w=2.1, b=9.8
Training Set 3: w=1.9, b=10.2

Predictions at speed=20:
Set 1: 2(20) + 10 = 50
Set 2: 2.1(20) + 9.8 = 51.8
Set 3: 1.9(20) + 10.2 = 48.2

Variance is low because predictions are similar across different training sets.
```

**Why Other Options Are Wrong:**

**Option (a) - High variance:**
- High variance would mean the model is very sensitive to training data
- Linear models are inherently stable and consistent
- They don't overfit to noise in the training data

**Bias-Variance Tradeoff Context:**

**Current Situation:**
- **High Bias**: Model cannot capture true quadratic relationship
- **Low Variance**: Model is stable and consistent

**This is a classic underfitting scenario:**
- Model is too simple for the data
- Cannot capture the true relationship (high bias)
- But is very stable across different training sets (low variance)

**Comparison with More Complex Models:**

**Linear Model (Current):**
- **Bias**: High (cannot capture quadratic relationship)
- **Variance**: Low (stable predictions)

**Quadratic Model (Ideal):**
- **Bias**: Low (can capture true relationship)
- **Variance**: Low (still relatively simple)

**Very Complex Model (e.g., high-degree polynomial):**
- **Bias**: Low (can capture complex relationships)
- **Variance**: High (sensitive to training data)

**Practical Implications:**

**Advantages of Low Variance:**
- Predictions are consistent and reliable
- Model generalizes well to similar data
- Less prone to overfitting

**Disadvantages in This Case:**
- The low variance comes at the cost of high bias
- Model consistently makes the wrong type of prediction
- Cannot capture the true underlying relationship

**Conclusion:**
The linear model has **low variance** because it is simple and stable, making consistent predictions across different training sets, even though these predictions are systematically wrong due to the high bias.

## 3. One Answer

Follow the same car scenario as the above question. In reality, stopping distance is also affected by weather conditions, which your our model does not capture.
Which of these components of overall model error captures the error from not including weather conditions as a feature?

a. Bias

b. Variance

c. Irreducible error

Correct answers: (c)

**Explanation:**

The correct answer is **(c) - Irreducible error**. Here's the detailed explanation:

**Understanding the Problem:**

We have a model that predicts stopping distance based on speed, but in reality, stopping distance is also affected by weather conditions. The model doesn't include weather as a feature.

**Components of Model Error:**

The total prediction error can be decomposed into three components:
$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**1. Bias:**
- Error due to model assumptions (e.g., linear vs quadratic relationship)
- In our case: using linear model when true relationship is quadratic

**2. Variance:**
- Error due to model sensitivity to training data
- In our case: how much predictions vary with different training sets

**3. Irreducible Error:**
- Error due to inherent randomness or missing information
- Cannot be reduced by any model, no matter how complex

**Why Weather Conditions Create Irreducible Error:**

**1. Missing Information:**
- Weather conditions affect stopping distance but are not in our model
- This creates **inherent unpredictability** in the data
- No model can perfectly predict stopping distance without weather information

**2. Inherent Randomness:**
- Even with perfect information, there's natural variation in stopping distances
- Weather introduces additional sources of variation
- This randomness cannot be eliminated by any model

**Mathematical Analysis:**

**True Model (with weather):**
$$\text{Stopping Distance} = f(\text{Speed}, \text{Weather}) + \epsilon$$

**Our Model (without weather):**
$$\text{Stopping Distance} = g(\text{Speed}) + \epsilon'$$

**Error Decomposition:**
$$\text{Error} = \underbrace{(g(\text{Speed}) - f(\text{Speed}, \text{Weather}))}_{\text{Irreducible}} + \underbrace{\epsilon}_{\text{Noise}}$$

**Why Other Options Are Wrong:**

**Option (a) - Bias:**
- Bias comes from model assumptions (linear vs quadratic)
- Weather conditions don't affect the model's functional form
- This is about missing features, not wrong assumptions

**Option (b) - Variance:**
- Variance comes from model sensitivity to training data
- Weather conditions don't make the model more or less sensitive
- This is about inherent unpredictability, not model instability

**Examples of Irreducible Error:**

**1. Weather Effects:**
- Rain: Increases stopping distance by 20-30%
- Snow: Increases stopping distance by 50-100%
- Ice: Increases stopping distance by 200-300%

**2. Other Missing Factors:**
- Driver reaction time
- Tire condition
- Road surface quality
- Vehicle weight

**3. Natural Variation:**
- Even with identical conditions, stopping distances vary
- Human factors (driver skill, attention)
- Mechanical factors (brake wear, tire pressure)

**Practical Implications:**

**1. Model Limitations:**
- No model can achieve perfect predictions without all relevant features
- Irreducible error sets a lower bound on achievable performance

**2. Feature Engineering:**
- Adding weather features would reduce irreducible error
- But some randomness will always remain

**3. Realistic Expectations:**
- Understanding irreducible error helps set realistic performance goals
- Don't expect perfect predictions when important factors are missing

**Conclusion:**
The error from not including weather conditions is **irreducible error** because it represents inherent unpredictability that cannot be eliminated by any model that doesn't have access to weather information.

## 4. Select All That Apply

Which of the following will generally help to reduce model variance?

a. Increasing the size of the training data.

b. Increasing the size of the validation data.

c. Increasing the number of model parameters.

d. Increasing the amount of regularization.

Correct answers: (a), (d)

**Explanation:**

The correct answers are **(a) and (d)**. Here's the detailed explanation:

**Understanding Model Variance:**

**Definition of Variance:**
Variance measures how much the model's predictions change when trained on different datasets drawn from the same underlying distribution:
$$\text{Variance} = \mathbb{E}[(f(x) - \mathbb{E}[f(x)])^2]$$

**High Variance = Overfitting:**
- Model is too sensitive to training data
- Learns noise and idiosyncrasies in the training set
- Poor generalization to new data

**Low Variance = Stability:**
- Model makes consistent predictions across different training sets
- Generalizes well to new data
- Less prone to overfitting

**Analysis of Each Option:**

**Option (a) - Increasing the size of the training data: ✓ CORRECT**

**Why This Reduces Variance:**
1. **More Information**: Larger training sets provide more representative samples
2. **Less Noise Sensitivity**: Model is less likely to overfit to noise in smaller datasets
3. **Better Generalization**: More data helps the model learn the true underlying pattern

**Mathematical Intuition:**
- With more data points, the model can better estimate the true relationship
- Law of large numbers: estimates become more stable with more samples
- Reduces the impact of individual noisy data points

**Example:**
```
Small dataset (10 points): Model might fit noise perfectly
Large dataset (1000 points): Model learns the true trend, ignores noise
```

**Option (b) - Increasing the size of the validation data: ✗ INCORRECT**

**Why This Doesn't Reduce Variance:**
1. **Validation data is not used for training**: It doesn't affect what the model learns
2. **Only affects evaluation**: Helps estimate model performance more accurately
3. **No impact on model behavior**: The model itself doesn't change

**What validation data does:**
- Provides unbiased estimate of model performance
- Helps in model selection and hyperparameter tuning
- Does not prevent overfitting during training

**Option (c) - Increasing the number of model parameters: ✗ INCORRECT**

**Why This Increases Variance:**
1. **More Flexibility**: More parameters allow the model to fit training data more closely
2. **Higher Risk of Overfitting**: Model can memorize training data instead of learning patterns
3. **Reduces Bias**: More parameters typically reduce bias but increase variance

**Bias-Variance Tradeoff:**
- More parameters → Lower bias, Higher variance
- Fewer parameters → Higher bias, Lower variance

**Option (d) - Increasing the amount of regularization: ✓ CORRECT**

**Why This Reduces Variance:**
1. **Constrains Model Complexity**: Prevents the model from fitting noise in training data
2. **Promotes Simplicity**: Encourages the model to learn simpler patterns
3. **Improves Generalization**: Model becomes less sensitive to training data variations

**Types of Regularization:**
- **L1 (Lasso)**: $||w||_1$ penalty, promotes sparsity
- **L2 (Ridge)**: $||w||_2^2$ penalty, promotes small weights
- **Dropout**: Randomly deactivates neurons during training
- **Early Stopping**: Stops training before overfitting

**Mathematical Effect:**
$$\text{Loss} = \text{Training Loss} + \lambda \cdot \text{Regularization Term}$$

As $\lambda$ increases:
- Model becomes simpler
- Less sensitive to training data
- Lower variance, but potentially higher bias

**Practical Examples:**

**Training Data Size:**
```
Small dataset: Model variance high (overfits to noise)
Large dataset: Model variance low (learns true pattern)
```

**Regularization:**
```
No regularization: Model can fit training data perfectly (high variance)
With regularization: Model constrained, more stable (lower variance)
```

**Conclusion:**
Options **(a)** and **(d)** reduce model variance by providing more information (larger training set) and constraining model complexity (regularization), respectively.

## 5. One Answer

For machine learning models and datasets in general, as the number of training data points grows, the prediction error of the model on unseen data (data not found in the training set) approaches 0.

a. True

b. False

Correct answers: (b)

**Explanation:**

The correct answer is **(b) - False**. Here's the detailed explanation:

**Understanding the Problem:**

The statement claims that as the number of training data points grows, the prediction error on unseen data approaches 0. This is false because of the fundamental limitations in machine learning.

**Components of Prediction Error:**

The total prediction error can be decomposed into:
$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**1. Bias:**
- Error due to model assumptions
- Can be reduced with more data and better models
- Approaches zero as model complexity increases

**2. Variance:**
- Error due to model sensitivity to training data
- Can be reduced with more data
- Approaches zero as data size increases

**3. Irreducible Error:**
- Error due to inherent randomness or missing information
- **Cannot be reduced by any amount of data**
- Sets a fundamental lower bound on achievable performance

**Why Prediction Error Cannot Approach Zero:**

**1. Irreducible Error:**
- **Noise in the data**: Measurement errors, random fluctuations
- **Missing information**: Important features not captured
- **Inherent randomness**: Natural variation in the phenomenon

**Mathematical Example:**

**True Model with Noise:**
$$y = f(x) + \epsilon$$
where $\epsilon \sim \mathcal{N}(0, \sigma^2)$ is irreducible noise.

**Expected Prediction Error:**
$$\mathbb{E}[(y - \hat{y})^2] = \mathbb{E}[(f(x) - \hat{f}(x))^2] + \sigma^2$$

**As $n \to \infty$:**
- $\mathbb{E}[(f(x) - \hat{f}(x))^2] \to 0$ (bias and variance approach zero)
- But $\sigma^2$ remains (irreducible error)

**Result**: Prediction error approaches $\sigma^2$, not zero.

**Practical Examples:**

**1. Medical Diagnosis:**
- Even with perfect models and infinite data
- Human biology has inherent variability
- Some diseases have random onset patterns
- Prediction error cannot be zero

**2. Stock Price Prediction:**
- Even with all available information
- Market movements have random components
- Unpredictable events affect prices
- Perfect prediction is impossible

**3. Weather Forecasting:**
- Even with perfect models and infinite historical data
- Weather has chaotic, unpredictable elements
- Small changes can lead to large differences
- Perfect prediction is impossible

**Mathematical Verification:**

**Cramér-Rao Lower Bound:**
For any unbiased estimator $\hat{\theta}$:
$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$
where $I(\theta)$ is the Fisher information.

**This sets a fundamental lower bound** on estimation error, regardless of data size.

**Bayesian Perspective:**
Even with infinite data, there's always uncertainty in predictions due to:
- Model uncertainty
- Parameter uncertainty
- Inherent randomness

**Why Other Options Are Wrong:**

**Option (a) - True:**
- **Problem**: Ignores irreducible error
- **Issue**: Assumes perfect predictability
- **Result**: Unrealistic expectation

**Key Insights:**

**1. Data Limitations:**
- More data reduces bias and variance
- But cannot eliminate irreducible error
- There's always a fundamental limit

**2. Model Limitations:**
- Perfect models don't exist
- All models make assumptions
- These assumptions create bias

**3. Reality Limitations:**
- Many phenomena are inherently random
- Perfect prediction is often impossible
- Understanding irreducible error is crucial

**Practical Implications:**

**1. Realistic Expectations:**
- Don't expect perfect predictions
- Focus on reducing bias and variance
- Accept that some error is unavoidable

**2. Model Selection:**
- Choose models appropriate for the problem
- Consider the irreducible error when evaluating performance
- Don't overfit trying to achieve impossible accuracy

**3. Business Decisions:**
- Set realistic performance targets
- Consider the cost of reducing error further
- Focus on actionable improvements

**Conclusion:**
The statement is **False** because prediction error cannot approach zero due to irreducible error, which represents inherent randomness and missing information that cannot be eliminated by any amount of data or any model.

## 6. Select All That Apply

Which of the following statements about (binary) logistic regression is true?
Recall that the sigmoid function is defined as $\sigma(x)=\frac{1}{1+e^{-x}}$ for $x\in\mathbb{R}.$

a. For any finite input $x\in\mathbb{R}$, $\sigma(x)$ is strictly greater than 0 and strictly less than 1. Thus, a binary logistic regression model with finite input and weights can never output a probability of exactly 0 or 1, and can never achieve a training loss of exactly 0.

b. The first derivative of o is monotonically increasing.

c. There exists a constant value $c\in\mathbb{R}$ such that o is convex when restricted to $x<c$ C and concave when restricted to $x\ge c$

d. For binary logistic regression, if the probability of the positive class is $\sigma(x)$ then the probability of the negative class is $\sigma(-x)$

Correct answers: (a), (c), (d)

**Explanation:**

The correct answers are **(a)**, **(c)**, and **(d)**. Here's the detailed explanation:

**Understanding the Sigmoid Function:**

**Sigmoid Function Definition:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Key Properties:**
- **Range**: $(0, 1)$ (strictly bounded)
- **Symmetry**: $\sigma(-x) = 1 - \sigma(x)$
- **Monotonic**: Strictly increasing function
- **Smooth**: Continuous and differentiable everywhere

**Analysis of Each Statement:**

**Option (a) - For any finite input $x \in \mathbb{R}$, $\sigma(x)$ is strictly greater than 0 and strictly less than 1: TRUE** ✅

**Mathematical Proof:**

**Lower Bound:**
For any finite $x \in \mathbb{R}$:
- $e^{-x} > 0$ (exponential function is always positive)
- $1 + e^{-x} > 1$
- $\sigma(x) = \frac{1}{1 + e^{-x}} < 1$

**Upper Bound:**
For any finite $x \in \mathbb{R}$:
- $e^{-x} < \infty$ (finite exponential)
- $1 + e^{-x} < \infty$
- $\sigma(x) = \frac{1}{1 + e^{-x}} > 0$

**Implications for Logistic Regression:**
- **Output probabilities**: Always in $(0, 1)$, never exactly 0 or 1
- **Training loss**: Cannot achieve exactly 0 (perfect classification)
- **Asymptotic behavior**: Can get arbitrarily close to 0 or 1

**Example:**
```
x = 10: σ(10) ≈ 0.99995 (very close to 1, but not exactly 1)
x = -10: σ(-10) ≈ 0.00005 (very close to 0, but not exactly 0)
```

**Option (b) - The first derivative of σ is monotonically increasing: FALSE** ❌

**Mathematical Analysis:**

**First Derivative:**
$$\sigma'(x) = \frac{d}{dx}\sigma(x) = \frac{e^{-x}}{(1 + e^{-x})^2} = \sigma(x)(1 - \sigma(x))$$

**Second Derivative:**
$$\sigma''(x) = \frac{d^2}{dx^2}\sigma(x) = \sigma(x)(1 - \sigma(x))(1 - 2\sigma(x))$$

**Behavior:**
- **At $x = 0$**: $\sigma(0) = 0.5$, $\sigma'(0) = 0.25$ (maximum)
- **As $x \to \infty$**: $\sigma'(x) \to 0$ (decreasing)
- **As $x \to -\infty$**: $\sigma'(x) \to 0$ (decreasing)

**Result**: The derivative is **not monotonically increasing**. It peaks at $x = 0$ and decreases towards the asymptotes.

**Option (c) - There exists a constant value $c \in \mathbb{R}$ such that σ is convex when restricted to $x < c$ and concave when restricted to $x \geq c$: TRUE** ✅

**Mathematical Proof:**

**Second Derivative Analysis:**
$$\sigma''(x) = \sigma(x)(1 - \sigma(x))(1 - 2\sigma(x))$$

**Convexity Conditions:**
- **Convex**: $\sigma''(x) > 0$
- **Concave**: $\sigma''(x) < 0$

**Critical Point:**
When $\sigma(x) = 0.5$:
- $\sigma''(x) = 0.5 \cdot 0.5 \cdot 0 = 0$

**Behavior:**
- **For $x < 0$**: $\sigma(x) < 0.5$, so $1 - 2\sigma(x) > 0$ → $\sigma''(x) > 0$ (convex)
- **For $x > 0$**: $\sigma(x) > 0.5$, so $1 - 2\sigma(x) < 0$ → $\sigma''(x) < 0$ (concave)

**Result**: $c = 0$ satisfies the condition.

**Option (d) - For binary logistic regression, if the probability of the positive class is $\sigma(x)$, then the probability of the negative class is $\sigma(-x)$: TRUE** ✅

**Mathematical Proof:**

**Sigmoid Symmetry Property:**
$$\sigma(-x) = \frac{1}{1 + e^{x}} = \frac{e^{-x}}{e^{-x} + 1} = 1 - \frac{1}{1 + e^{-x}} = 1 - \sigma(x)$$

**For Binary Classification:**
- **Positive class probability**: $P(Y = 1|X) = \sigma(x)$
- **Negative class probability**: $P(Y = 0|X) = 1 - \sigma(x) = \sigma(-x)$

**Verification:**
$$\sigma(x) + \sigma(-x) = \sigma(x) + (1 - \sigma(x)) = 1$$

**Example:**
```
x = 2: σ(2) ≈ 0.88, σ(-2) ≈ 0.12
Sum: 0.88 + 0.12 = 1 ✓
```

**Practical Implications:**

**1. Training Loss:**
- Cannot achieve exactly zero loss
- Can get arbitrarily close to perfect classification
- Important for setting realistic expectations

**2. Model Interpretability:**
- Outputs are always probabilities
- Symmetry property useful for binary classification
- Convexity/concavity affects optimization

**3. Numerical Stability:**
- Bounded outputs prevent numerical issues
- Smooth function enables gradient-based optimization
- Symmetry simplifies implementation

**Conclusion:**
Statements **(a)**, **(c)**, and **(d)** are correct. The sigmoid function has bounded outputs, changes from convex to concave at $x = 0$, and satisfies the symmetry property $\sigma(-x) = 1 - \sigma(x)$.

## 7. Select All That Apply

Consider performing Lasso regression by finding parameters $w\in\mathbb{R}^{d}$ that minimize
$$f(w)=\sum_{i=1}^{n}(y^{(i)}-x^{(i)\top}w)^{2}+\lambda||w||_{1}.$$ 

Which of the following statements are true?

a. Increasing $\lambda$ will generally reduce the $L_{1}$ norm of the parameters $w$.

b. Consider two models $w_{1}$, $w_{2}\in\mathbb{R}^{d}.$ Assume $w_{1}$ is more sparse, i.e., $w_{1}$
has strictly more zero coefficients than $w_{2}$. Then $||w_{1}||_{1}<||w_{2}||_{1}$

c. Increasing $\lambda$ generally increases model bias.

d. Increasing $\lambda$ generally increases model variance.

Correct answers: (a), (c)

**Explanation:**

The correct answers are **(a)** and **(c)**. Here's the detailed explanation:

**Understanding LASSO Regression:**

**LASSO Objective Function:**
$$f(w) = \sum_{i=1}^{n}(y^{(i)} - x^{(i)\top}w)^2 + \lambda||w||_1$$

where:
- First term: Mean squared error (data fitting term)
- Second term: L1 regularization (sparsity-inducing penalty)
- $\lambda$: Regularization strength (hyperparameter)

**Analysis of Each Statement:**

**Option (a) - Increasing $\lambda$ will generally reduce the L1 norm of the parameters $w$: TRUE** ✅

**Mathematical Explanation:**

**L1 Norm Definition:**
$$||w||_1 = \sum_{j=1}^d |w_j|$$

**Effect of Increasing $\lambda$:**
- **Stronger penalty**: The L1 term becomes more important
- **Coefficient shrinkage**: Weights are pushed towards zero
- **Sparsity promotion**: Some coefficients become exactly zero

**Mathematical Intuition:**
$$\min_w \sum_{i=1}^{n}(y^{(i)} - x^{(i)\top}w)^2 + \lambda||w||_1$$

As $\lambda$ increases:
- The regularization term dominates
- The optimal solution has smaller $||w||_1$
- More coefficients become zero

**Example:**
```
λ = 0: w = [2.1, -1.8, 0.9, -0.5], ||w||₁ = 5.3
λ = 1: w = [1.5, -1.2, 0, -0.1], ||w||₁ = 2.8
λ = 5: w = [0.8, 0, 0, 0], ||w||₁ = 0.8
```

**Option (b) - Consider two models $w_1$, $w_2 \in \mathbb{R}^d$. Assume $w_1$ is more sparse, i.e., $w_1$ has strictly more zero coefficients than $w_2$. Then $||w_1||_1 < ||w_2||_1$: FALSE** ❌

**Counterexample:**

**Model 1 (more sparse):**
$w_1 = [0, 0, 0, 10]$
- Number of zeros: 3
- L1 norm: $||w_1||_1 = 10$

**Model 2 (less sparse):**
$w_2 = [1, 1, 1, 1]$
- Number of zeros: 0
- L1 norm: $||w_2||_1 = 4$

**Result**: $w_1$ is more sparse but has larger L1 norm.

**Why This Happens:**
- **Sparsity**: Counts number of non-zero coefficients
- **L1 norm**: Sum of absolute values of coefficients
- **Magnitude matters**: Large non-zero coefficients can dominate

**Option (c) - Increasing $\lambda$ generally increases model bias: TRUE** ✅

**Mathematical Explanation:**

**Bias-Variance Tradeoff:**
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Effect of Increasing $\lambda$:**

**1. Model Simplification:**
- Larger $\lambda$ forces coefficients to be smaller
- Model becomes less flexible
- Cannot fit complex patterns in data

**2. Bias Increase:**
- Model assumptions become more restrictive
- Systematic error increases
- Underfitting occurs

**Mathematical Verification:**
As $\lambda \to \infty$:
- $w \to 0$ (all coefficients approach zero)
- Model becomes constant predictor
- High bias, low variance

**Example:**
```
λ = 0: Complex model, low bias, high variance
λ = 1: Moderate model, moderate bias, moderate variance  
λ = 10: Simple model, high bias, low variance
```

**Option (d) - Increasing $\lambda$ generally increases model variance: FALSE** ❌

**Why This is Incorrect:**

**Effect of Increasing $\lambda$ on Variance:**
- **Reduced flexibility**: Model becomes less sensitive to training data
- **Stability**: Predictions become more consistent
- **Lower variance**: Model is more robust

**Mathematical Intuition:**
- **Small $\lambda$**: Model can fit training data closely (high variance)
- **Large $\lambda$**: Model is constrained, less sensitive to data (low variance)

**Practical Implications:**

**1. Model Selection:**
- **Small $\lambda$**: Good for complex relationships, risk of overfitting
- **Large $\lambda$**: Good for simple relationships, risk of underfitting
- **Optimal $\lambda$**: Balances bias and variance

**2. Feature Selection:**
- **Sparsity**: LASSO automatically performs feature selection
- **Interpretability**: Fewer features make model easier to understand
- **Computational efficiency**: Fewer features mean faster prediction

**3. Regularization Effect:**
- **Prevents overfitting**: Constrains model complexity
- **Improves generalization**: Better performance on unseen data
- **Robustness**: Less sensitive to noise in training data

**Conclusion:**
Options **(a)** and **(c)** are correct. Increasing $\lambda$ in LASSO reduces the L1 norm of parameters (promoting sparsity) and increases model bias (simplifying the model), while decreasing variance (making the model more stable).

## 8. Select All That Apply

Which of the following statements about ridge regression are true?

a. When there are correlated features, ridge regression typically sets the weights of all but one of the correlated features to 0.

b. Compared to unregularized linear regression, the additional computational cost of ridge regression scales with respect to the number of data points in the dataset.

c. Ridge regression reduces variance at the expense of increasing bias.

d. Using ridge and lasso regularization together (e.g., minimizing a training objective of the form $$f(w)=\sum_{i=1}^{n}(y^{(i)}-x^{(i)\top}w)^{2}+\lambda_{1}||w||_{1}+\lambda_{2}||w||_{2}^{2})$$ makes the training loss no longer convex.

Correct answers: (c)

Explanation:
a) False. This statement is more akin to Lasso regression.
Ridge regression is more likely to somewhat equally decrease the weights of correlated features to each be smaller (as opposed to only keeping one large).
See lecture 5 slide 37-38.
b) False. Ridge regression additional computational cost consists of calculating the L2-norm of all weights.
This scales with respect to the number of features, not number of data points.
c) True. Ridge regression biases the model to have smaller weights and with the hope of being less likely to overfit-adding bias to reduce variance.
d) False. The sum of convex functions is also convex.

## 9. Select All That Apply

Consider minimizing a function $f(x):\mathbb{R}\rightarrow\mathbb{R}$.
Recall the following definitions:
- $x\in\mathbb{R}$ is a global minimum for $f$ if $f(x^{\prime})\ge f(x)$ for all $x^{\prime}\in\mathbb{R}$
- $x\in\mathbb{R}$ is a local minimum for $f$ if there exists $\epsilon>0$ such that $f(x^{\prime})\ge f(x)$ for all $x^{\prime}\in\mathbb{R}$ within $\epsilon$ distance of $x$, that is, $|x^{\prime}-x|<\epsilon.$
Which of the following statements are true?

a. All linear functions $f(x)=ax+b$ for some $a, b\in\mathbb{R}$ are both convex and concave.

b. If $f$ is convex, then it can have at most one global minimum. (That is, if $u, v\in\mathbb{R}$
are both global minima for $f$, then that implies $u=v$.)

c. If $f$ is convex, then all local minima are global minima.

d. If $f$ is convex and bounded below (i.e., there exists $c\in\mathbb{R}$ such that $f(x)\ge c$ for all
$x\in\mathbb{R}$) then it must have at least one global minimum.

e. If $f$ is concave, then it must have no global minima.

Correct answers: (a), (c)

Explanation:
a) True. Linear functions are convex. Any of the tests we discussed in class apply, e.g., their second derivative (which is 0) is always greater than or equal to 0. If f is linear, then f is also linear and therefore convex, so f is also concave.
b) False. Consider the constant function $f(x)=0$. Every
$x\in\mathbb{R}$ is a global minimum.
c) True. See class notes from lecture 7.
d) False. For example, $f(x)$ could be monotonically decreasing and asymptotically approaching $0$ as $x$ increases, so it is bounded below by 0 but has no global minimum.
e) False. Consider the same constant function $f(x)=0$.

## 10. One Answer

Let's say we want to standardize our data (i.e., normalizing the data to have zero mean and unit variance in each dimension) for the purposes of training and evaluating a ML model.
Which of the following would be most appropriate?

a. Split the dataset into the train/val/test splits, standardize the data separately for each split using the mean and variance statistics of that split.

b. Split the dataset into the train/val/test splits, standardize the data for the training set, and use the mean and variance statistics of the training data to standardize the validation and test sets.

c. Split the dataset into the train/val/test splits, standardize the training and validation sets separately using the mean and variance statistics of each split, then use the mean and variance statistics of the validation split to normalize the test set.

d. Standardize the entire dataset (i.e., all splits combined) using the combined mean and variance statistics.
Then, split the standardized data into train/val/test sets.

Correct answers: (b)

Explanation: We should do (b) to avoid leaking test set information to the training process.
Other options may lead to overfitting to the validation or test data when picking hyperparameters.

## 11. Select All That Apply

Which of the following statements about gradient descent are true?
Recall that the gradient descent algorithm updates the weight parameter $w$ at iteration $t$ as follows: $$w_{t+1}=w_{t}-\eta\nabla_{w}l(w)|_{w=w_{t}}$$ (with $\eta$ being the step size).
For this question, we say that gradient descent has converged by iteration $T$ if there is some iteration $t<T$ such that $||\nabla_{w}l(w_{t})||_{2}^{2}\le\epsilon$ for some fixed $\epsilon>0$.

a. The gradient $\nabla_{w}l(w)$ points in the direction that maximizes the training loss.

b. Assume $l(w)$ is convex. Then if gradient descent converges by iteration $T$ for some
fixed $\epsilon>0$ and some step size $\eta$, it will converge in at most $T$ iterations if we increase the step size $\eta$.

c. Assume $l(w)$ is convex. Then if gradient descent converges by iteration $T$ for some fixed $\epsilon>0$ and some step size $\eta$, it will also eventually converge for all smaller step sizes $0<\eta^{\prime}<\eta$ given enough iterations.

Correct answers: (a), (c)

Explanation:
a) True. $\nabla_{w}.l(w)$ points in the direction that maximizes the loss.
Don't confuse this with the gradient descent update which steps in the "negative-gradient" direction.
b) False. large step size may cause the model to overshoot the optimum point, thus taking longer to converge.
c) True. With smaller step size, the model is likely to gradually approach the optimal point with less overshooting even if it takes more iterations.

## 12.

Describe one advantage of mini-batch stochastic gradient descent over full-batch gradient descent.

Answer:

Explanation: One advantage is that mini-batch SGD is faster to compute over full-batch GD, while still offering an unbiased estimate of the gradient full-batch GD would compute.
Another advantage is the variance of mini-batch SGD can lead to randomness that might help avoid local minima where full-batch GD might get stuck.

## 13.

Describe one advantage of mini-batch stochastic gradient descent $(1<B<n)$ over stochastic gradient descent with batch size $B=1$ (e.g., updating the parameters at each iteration based only on one randomly sampled training point).

Answer:

Explanation: Possible answer: the update steps of mini-batch SGD will have less variance and might converge in fewer update steps.
More possible answers:
Noise Reduction: Mini-batches average the gradient over multiple samples, reducing the variance and leading to more stable updates.
Faster Convergence: By reducing noise, the algorithm can converge faster to a minimum.
Computational Efficiency: Mini-batches enable efficient use of parallelization on hardware like GPUs.
Better Generalization: Smoother updates can help the model generalize better.
Reduced Frequency of Parameter Updates: Fewer updates per epoch, which can improve training dynamics and efficiency.
parallelizability

## 14. One Answer

In a machine learning course, the distribution of final exam scores is approximately normal.
However, an administrative error provided some students with prior access to practice materials closely resembling the exam, resulting in significant score increases for these students.
Considering only the scores and without labeled information about who had access to the materials, what type of model would be most appropriate to estimate the likelihood that a given student had access to the practice materials?

Answer:

Explanation: This is an unsupervised learning problem because there are no labels indicating which students had access to the materials.
The overall score distribution is a mixture of two Gaussian distributions:
1. Students without access: Their scores follow the original normal distribution.
2. Students with access: Their scores are higher on average, forming a second Gaussian with a higher mean.
A Gaussian Mixture Model (GMM) is the most suitable choice, as it models this bimodal distribution by combining multiple Gaussians.
k-means clustering could also be used but is less effective, as it assumes.
spherical clusters and does not explicitly account for Gaussian distributions.

## 15. Select All That Apply

Assume we are given a fixed dataset $D=\{x^{(1)},x^{(2)},...,x^{(n)}\}$ drawn i.i.d. (independently and identically distributed) from an underlying distribution $P(x)$.
We use the bootstrap to draw bootstrap samples $\tilde{D}=\{\tilde{x}^{(1)},\tilde{x}^{(2)},...\}$ from a bootstrap distribution $Q(x)$.
Which of the following statements are true?

a. The bootstrap samples in $\tilde{D}$ are drawn by sampling with replacement from D.

b. The bootstrap samples in $\tilde{D}$ are drawn by sampling without replacement from D.

c. The distribution of bootstrap samples in $\tilde{D}$ is always identical to the underlying data distribution P.

d. The bootstrap samples in $\tilde{D}$ are independently and identically distributed.

Correct answers: (a), (d)

Explanation:
a) True. The bootstrap distribution is created by sampling with replacement from the fixed dataset.
b) False. Inverse of option (a)
c) False. Bootstrap samples are not guaranteed to be identical to population distribution.
d) True.
By construction of the bootstrap method.

## 16.

You are given a dataset with four data points $x^{(1)}, x^{(2)}, x^{(3)}, x^{(4)}\in\mathbb{R}$. The coordinates of these data points are:
- $x^{(1)}=0$
- $x^{(2)}=1$
- $x^{(3)}=5$
- $x^{(4)}=9$.
You run k-means on this dataset with $k=3$ centroids, initialized at the first 3 data points: 0, 1, and 5. After k-means converges, what will be the new coordinates of these centroids?
Give your answer as a sequence of 3 numbers in ascending order (e.g., "0, 1, 5").

Answer:

Explanation: 0.1,7. In the first iteration, $x^{(1)}$ will be assigned to the first centroid, $x^{(2)}$ to the second centroid, and $x^{(3)}$ and $x^{(4)}$ to the third centroid.
Thus the centroids will be updated to 0, 1, 7 respectively.
The centroid assignments will not change in subsequent assignments, so k-means will converge after one iteration.
Note that this clustering is not optimal (in the sense of $L_{2}$ distance from centroids);
this is an example of how k-means can fail to find the globally optimal clustering.

## 17. Select All That Apply

Which of the following statements are true about k-means?

a. The output of k-means can change depending on the initial centroid positions.

b. Assuming that the number of data points is divisible by k, k-means with k clusters always outputs clusters of equal sizes.

c. If run for long enough, k-means will always find the globally optimal solution (as measured by the average L2 distance between each point and its assigned cluster centroid).

d. K-means will not converge unless all clusters in the underlying data distribution have equal, spherical variance.

Correct answers: (a)

Explanation:
a) True.
b) False. k-means is not guaranteed to produce clusters of equal sizes, it depends on where the distance between the points
c) False.
k-means will converge when the cluster arrangement no longer changes.
This may only be a local optimum, running longer would not help.
d) False.
k-means will converge when the cluster arrangement no longer changes.
This may only be a local optimum, running longer would not help.

## 18.

Should we initialize all the weights of a neural network to be the same small constant value (e.g., 0.001)?
Why or why not?

Answer:

Explanation: No. It is important to break symmetry so that all neurons do not get the same gradient updates.

## 19. Select All That Apply

In a neural network, the number of layers is an important hyperparameter.
Which of these statements are true about adding layers to a neural network (keeping all other aspects of the model and training process the same)?

a. Hyperparameters are independent, i.e., adding more layers will not affect the optimal choice of step size for gradient descent or the amount of regularization needed.

b. We cannot use cross-validation to select hyperparameters that directly affect model architecture, such as the number of layers.

c. Adding more layers generally decreases the training loss.

d. Adding more layers generally increases the ability of the model to overfit the data.

Correct answers: (c), (d)

Explanation:
a) False. Adding layers can affect optimal learning rates and regularization needs.
b) False.
Cross-validation can be used to select architecture-related hyperparameters like the number of layers.
c) True.
More layers improve representational capacity, reducing training loss.
d) True. Deeper networks can overfit without proper regularization.

## 20. Select All That Apply

Which of the following are advantages of Gaussian Mixture Models (GMMs) over K-means for a clustering application?

a GMMs are better suited if clusters have varying sizes and/or shapes.

b GMMs are better equipped to model overlapping clusters.

c GMMs are better suited to reason probabilistically about the data and the clusters.

d On a given dataset, a single iteration of the EM algorithm for fitting a GMM requires less computation than a single iteration of Lloyd’s Algorithm for fitting K-means.

Correct answers: (a), (b), (c)

Explanation:
 a) True. GMMs can model clusters with different sizes and shapes because they use a combination of Gaussian distributions, each with its own mean and covariance matrix.
b) True. GMMs can handle overlapping clusters by assigning probabilities to each data point for belonging to each cluster.
c) True. GMMs provide a probabilistic framework, giving the likelihood of each data point belonging to each cluster, which is useful for probabilistic reasoning.


d) False. The Expectation-Maximization (EM) algorithm used for fitting GMMs is generally more com putationally intensive per iteration compared to Lloyd’s Algorithm for K-means, due to the additional steps of calculating probabilities and updating the covariance matrices.

## 21. One Answer

Kernel methods calculate the inner products of features in a transformed feature space, without explicitly computing the transformed features.

a. True

b. False

Correct answers: (a)
Explanation: A function K : Rd × Rd → R is a kernel for a map φ if K(x, x0) = φ(x) · φ(x0) = hφ(x), φ(x0)i for all x, x0.
φ(x) doesn’t need to be explicitly computed.

## 22. One Answer

Consider a fully connected neural network (MLP) with an input layer, a hidden layer, and an output layer.
The input layer has n units, the hidden layer has h units, and the output layer has m units.
Assume there are no bias units/terms. Which of the following statements about the number of trainable parameters is true?

a. The total number of trainable parameters is $n \cdot h \cdot m$.

b. The total number of trainable parameters is $n \cdot h + h \cdot m$.

c. The total number of trainable parameters is $(n + 1) \cdot h + (h + 1) \cdot m$.

d. The total number of trainable parameters is $n + h + m$.

Correct answers: (b)
 Explanation: Connections between the input and the hidden layer: $n \cdot h$;
connections between the hidden and the output layer: $h \cdot m$.

## 23. Select All That Apply

Consider a matrix $A \in \mathbb{R}^{m \times n}$ with singular value decomposition $A = USV^\top$, where $S$ is an $r \times r$ diagonal matrix and $r = \operatorname{rank}(A) \leq \min(m, n)$.

Which of the following statements are correct?

a. The columns of $U$ are the eigenvectors of $A^\top A$.

b. The columns of $U$ are the eigenvectors of $A A^\top$.

c. The columns of $V$ are the eigenvectors of $A^\top A$.

d. The columns of $V$ are the eigenvectors of $A A^\top$.

e. The singular values in $S$ are the square roots of the nonzero eigenvalues of $A A^\top$.

f. The singular values in $S$ are the square roots of the nonzero eigenvalues of $A^\top A$.

Correct answers: (b), (c), (e), (f)

Explanation: $A A^\top = U S^2 U^\top$, implying that the columns of $U$ are the eigenvectors of $A A^\top$ with correspond ing eigenvalues along the diagonal of $S^2$.

Similarly, $A^\top A = V S^2 V^\top$, implying that the columns of $V$ are the eigenvectors of $A^\top A$ with corresponding eigenvalues along the diagonal of $S^2$.

## 24.

Consider a dataset $X \in \mathbb{R}^{n \times p}$ with $n$ observations and $p$ features, and with corresponding covariance matrix $\Sigma$.

Let $\lambda_{1} \geq \lambda_{2} \geq ... \geq \lambda_{p}$ be the eigenvalues of $\Sigma$ in descending order.

Express the total variance explained by the first $k$ principal components (obtained by performing Principal Component Analysis (PCA) on $X$) as a fraction of the total variance in the original data.

Answer: Fraction of total variance explained $= \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^p \lambda_i}$.

The fraction of total variance explained by the first $k$ principal components in PCA can be expressed as the ratio of the sum of the first $k$ eigenvalues to the sum of all eigenvalues of the covariance matrix $\Sigma$.

## 25.

Consider a dataset $X \in \mathbb{R}^{n \times 2}$ with $n$ observations and 2 features.
Suppose $\Sigma$ is the covariance matrix of the dataset:

$$
\Sigma = \begin{pmatrix} 3 & \sqrt{3} \\ \sqrt{3} & 5 \end{pmatrix}
$$

This covariance matrix has the following unit-norm eigenvectors $u$ and $v$:

$u = \frac{1}{2}\begin{pmatrix} -\sqrt{3} \\ 1 \end{pmatrix}$, $v = \frac{1}{2}\begin{pmatrix} 1 \\ \sqrt{3} \end{pmatrix}$

Write the second principal component as a unit-length vector in vector form (i.e., $[a, b]$).
Second principal component:

Explanation: $u = \frac{1}{2}\begin{pmatrix} -\sqrt{3} \\ 1 \end{pmatrix}$
A vector $x$ and value $\lambda$ are defined to be an eigenvector-eigenvalue pair of $A$ if $A x = \lambda x$.
$\Sigma u = 2u$, so $\lambda_u = 2$.
$\Sigma v = 6v$, so $\lambda_v = 6$.
Eigenvector-eigenvalue pairs of a covariance matrix represent pairs of principal components and the variance explained by that principal component.
$u$'s eigenvalue is less than $v$'s, so it is the second principal component.
$u$ is already unit-length, so it is the final answer.

## 26. Select All That Apply

You are applying PCA to a training dataset of $n = 1024$ grayscale images that are each $16 \times 16$ pixels ($256$ pixels per image).
Consider reshaping each image into a vector $x_i \in \mathbb{R}^{256}$ and then composing a data matrix $X \in \mathbb{R}^{1024 \times 256}$, where the $i$th row is $x_i^\top$.
Let $\hat{x}_{i,k} \in \mathbb{R}^{256}$ be the PCA reconstruction of image $x_i$ using the top $k$ principal component directions in the data.
Let $R(k)$ be the average reconstruction error on the training data using $k$ principal components, $R(k) = \frac{1}{n} \sum_{i=1}^{n} ||x_i - \hat{x}_{i,k}||^2_2$.
Which of the following statements are true?

a. $R(k)$ is monotonically decreasing as $k$ increases, up to $k = 1024$. That is, if $0 < k_1 < k_2 \leq 1024$, then $R(k_1) > R(k_2)$.

b. If $k < \operatorname{rank}(X)$, then $R(k) > 0$.

c. If $k \geq \operatorname{rank}(X)$, then $R(k) = 0$.

d. For $k \geq 1$, let $\delta(k) = R(k-1) - R(k)$ be the decrease in reconstruction error by going from $k-1$ to $k$ principal components.
(When $k = 0$, define the reconstruction of $x_i$ to simply be the mean image $\bar{x}$.) Then, $\delta(k)$ is monotonically non-increasing as $k$ increases.

Correct answers: (b), (c), (d)

Explanation:
a) False. The number of principal components cannot exceed the rank of $X$, which is $\min(n, 256)$.
Since $X$ is $1024 \times 256$, its rank is at most $256$. Thus, $R(k)$ is only guaranteed to monotonically decrease for $k \leq 256$, not $k \leq 1024$.
b) True.
The reconstruction error is non-zero when the number of principal components $k$ is less than the rank of $X$, as there are remaining variations in $X$ not captured by the top $k$ components.
c) True. When $k$ is greater than or equal to the rank of $X$, the PCA reconstruction captures all the variation in $X$, resulting in zero reconstruction error.
d) True. Each additional principal component explains the maximum remaining variance, so the decrease in reconstruction error ($R(k)$) diminishes as $k$ increases, making $R(k)$ monotonically non-increasing.

## 27. Select All That Apply

Which of the following is/are true about the $k$-Nearest Neighbors (k-NN) algorithm?

a. Testing time (i.e., the amount of time it takes to produce an output for a new test point) increases with the number of training samples.

b. The number of hyperparameters increases with the number of training samples.

c. $k$-NN can learn non-linear decision boundaries.

d. $k$-NN clusters unlabeled samples in a $k$-dimensional space based on their similarity.

Correct answers: (a), (c)

## 28. Select All That Apply

Which of the following statements about random forests and decision trees are true?

a Random forests are generally easier for humans to interpret than individual decision trees.

b Random forests reduce variance (compared to individual decision trees) by aggre gating predictions over multiple decision trees.

c When constructing the individual trees in the random forest, we want their predic tions to be as correlated with each other as possible.

d Random forests can give a notion of confidence estimates by examining the distri bution of outputs that each individual tree in the random forest produces.

Correct answers: (b), (d)

Explanation:

a) False. Procedure is similar except random forest utilizes multiple decision trees

b) True.
Aggregating predictions from multiple trees reduces sensitivity compared to a single tree.

c) False.
Having as correlated trees as possible degenerates to a single tree, losing the benefits of a more complex forest.

d) True. Spread of decisions across different trees gives a confidence estimate. 

## 29. Select All That Apply

Which of the following is a correct statement about (mini-batch) Stochastic Gradient Descent (SGD)?

a The variance of the gradient estimates in SGD decreases as the batch size increases.

b Running SGD with batch size 1 for n iterations is generally slower than running full batch gradient descent with batch size n for 1 iteration, because the gradients for each training point in SGD have to be computed sequentially, whereas the gradients in full-batch gradient descent can be computed in parallel.

c SGD is faster than full-batch gradient descent because it only updates a subset of model parameters with each step.

d SGD provides an unbiased estimate of the true (full-batch) gradient of the training loss.

Correct answers: (a), (b), (d)
Explanation:

a) True. In SGD, the gradient is estimated using a subset of the data.
A sampled batch might not represent the entire dataset well.
As the batch size increases, it becomes more representative of the entire dataset, reducing the variance in the gradient estimates.

b) True. In batch gradient descent, all gradients for the entire dataset are computed in one forward backward pass, which can leverage parallel processing (e.g., on GPUs).

c) False. SGD does not update a subset of model parameters.

It updates all parameters based on the gradient computed from a subset of the data.
The faster convergence of SGD compared to full-batch gradient descent is due to the more frequent updates.

d) True.

The gradient computed on a mini-batch is an unbiased estimate of the full gradient because the mini-batch is a random sample of the dataset.
This randomness ensures that, on average, the mini-batch gradient equals the true gradient over the entire dataset.

## 30. One Answer

The probability density function for a gamma distribution with parameters $\theta > 0$, $k > 0$ is
 $f(x; \theta, k) = \frac{1}{\Gamma(k)\theta^k}x^{k-1}e^{- \frac{x}{\theta}}$,
 where
 $\Gamma(x) = (x − 1)!$

Say we have a dataset D of n data points, $\{x^{(1)}, x^{(2)}, . . .$
$, x^{(n)}\}$, where each $x \in R$.
 
Assume that k is given to us and fixed.

We would like to use D to find the maximum likelihood estimator for $\theta$.
What is the maximum likelihood estimator for $\theta$ in terms of k, n, and $x^{(1)}, x^{(2)}, . . .$
$, x^{(n)}$?

Hint: The argmax of the logarithm of a function is the same as the argmax of the function.

a. $\frac{1}{kn}\sum_{i=1}^{n} x^{(i)}$

b. $\frac{n(k-1)!}{\sum_{i=1}^{n} x^{(i)}e^{- \frac{x^{(i)}}{k}}}$

c. $\ln(\frac{1}{n}\sum_{i=1}^{n} x^{(i)}) − n(k − 1)!$

d. $\ln(k)−\frac{(k-1)!}{k}$

Correct answers: (a)

Explanation: To find the maximum likelihood estimator (MLE) for $\theta$, we start with the likelihood function for a dataset D = $\{x^{(1)}, x^{(2)}, .$.
$. , x^{(n)}\}$:
 $L(\theta) = \prod_{i=1}^{n} f(x^{(i)}; \theta, k) = \prod_{i=1}^{n} \frac{1}{\Gamma(k)\theta^k}(x^{(i)})^{k-1}e^{- \frac{x^{(i)}}{\theta}}$.

The log-likelihood function is:
 $\ell(\theta) = \sum_{i=1}^{n} \ln f(x^{(i)}; \theta, k) = -n \ln \Gamma(k) - kn \ln \theta + (k - 1) \sum_{i=1}^{n} \ln x^{(i)} - \frac{1}{\theta} \sum_{i=1}^{n} x^{(i)}$.

To maximize $\ell(\theta)$, we differentiate with respect to $\theta$ and set the derivative to zero:
 $\frac{\partial\ell}{\partial\theta} = -\frac{kn}{\theta} + \frac{1}{\theta^2} \sum_{i=1}^{n} x^{(i)} = 0$.

Multiply through by $\theta^2$to simplify:
 $-kn\theta + \sum_{i=1}^{n} x^{(i)} = 0 \Rightarrow \theta = \frac{1}{kn} \sum_{i=1}^{n} x^{(i)}$.

Thus, the maximum likelihood estimator for $\theta$ is:
 $\hat{\theta}=\frac{1}{kn}\sum_{i=1}^{n} x^{(i)}$.

The correct answer is (a).

## 31. One Answer

Many ML algorithms, like the k-nearest neighbors (k-NN) algorithm, relies on distances between points.

In high-dimensional spaces, distances can behave counterintuitively. This question illustrates one such example.

Consider two d-dimensional hypercubes S and T centered around the origin.
S has side length 2, while T is contained within S and has side length 1:
 $S = \{x \in R^d: ||x||_\infty \le 1\}$
 $T = \{x \in R^d: ||x||_\infty \le \frac{1}{2}\}$.

Alternatively, we can write $S = [-1, 1]^d$, and $T = [-\frac{1}{2},\frac{1}{2}]^d$.

Let P be the uniform distribution of points in S. What is the probability of drawing a point x ∼ P such that x ∈ T, that is, x is contained within T?

Give your answer in terms of d.

Answer:

Explanation: The volume of S is $2^d$, while the volume of T is $1^d$. Since x is uniformly distributed in S, the probability of x $\in$ T is the relative ratio of their volumes, which is $\frac{1}{2^d}$.

## 32. Select All That Apply

Consider the following dataset of four points in R2:

 $x^{(1)} = (0, 0)$ $y^{(1)} = −1$

 $x^{(2)} = (0, 1)$ $y^{(2)} = +1$

 $x^{(3)} = (1, 0)$ $y^{(3)} = +1$

 $x^{(4)} = (1, 1)$ $y^{(4)} = −1$.

This is also known as a XOR problem because the labels y are the result of applying the XOR operation to the two components of x.

For a given data point $x \in R^2$, denote its first dimension as $x_1$ and its second dimension as $x_2$.

For example, $x^{(2)}_1 = 0$ and $x^{(2)}_2 = 1$. Which of the following statements are true?

a. There exists a linear model $w \in R^3$, which predicts +1 if
 $w^T \begin{bmatrix} x_1 \\ x_2 \\ 1 \end{bmatrix} \ge 0$
 and −1 otherwise, that achieves 100% accuracy on this dataset.

b. There exists a linear model $w \in R^6$, which predicts +1 if
 $w^T \begin{bmatrix} x_1 \\ x_2 \\ x_1^2 \\ x_2^2 \\ x_1x_2 \\ 1 \end{bmatrix} \ge 0$
 and −1 otherwise, that achieves 100% accuracy on this dataset.

c. Define a polynomial feature expansion $\phi(x)$ as any function $\phi(x) : R^2 \to R^d$ that can be written as
 $\begin{bmatrix} x_1^{a_1} x_2^{b_1} \\ x_1^{a_2} x_2^{b_2} \\ . \\ . \\ . \\ x_1^{a_d} x_2^{b_d} \end{bmatrix}$
 for some integer d > 0 and integer vectors a, b $\in Z^d$.

Then there does not exist any polynomial feature expansion $\phi(x)$ such that a linear model w which predicts +1 if $w^T\phi(x) \ge 0$, and −1 otherwise, achieves 100% accuracy on this dataset.

Correct answers: (b)

Explanation:
a) There is no way to separate with linear features.

For option b) and example weight vector is
 $\begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \\ -4 \\ -1 \end{bmatrix}$

For option c) a counter example is feature expansion
 $\begin{bmatrix} x_1^0 x_2^0 \\ x_1^0 x_2^1 \\ x_1^1 x_2^0 \\ x_1^1 x_2^1 \end{bmatrix}$
 with weight vector
 $\begin{bmatrix} -1 \\ 1 \\ 1 \\ -2 \end{bmatrix}$

## 33. One Answer

Consider the following transfer learning setting.

We have a large neural network $\phi : \mathbb{R}^d \to \mathbb{R}^p$ pretrained on ImageNet, and we would like to use this to learn a classifier for our own binary classification task for medical images.

We decide to freeze the neural network $\phi$ and train a logistic regression classifier on top.

Formally, we are given $n$ data points from our own medical imaging task $\{(x^{(1)}, y^{(1)}),(x^{(2)}, y^{(2)}), \ldots, (x^{(n)}, y^{(n)})\}$, where $x^{(i)} \in \mathbb{R}^d$, $y^{(i)} \in \{-1, +1\}$.

We train a classifier $\hat{w} \in \mathbb{R}^p$:

$$\hat{w}= \operatorname{argmin}_{w \in \mathbb{R}^p} \sum_{i=1}^{n} \log \left( 1 + \exp \left( -y^{(i)}w^\top\phi(x^{(i)}) \right) \right).$$

Which of the following statements is true?

a. Learning $\hat{w}$ in this way is a convex optimization problem regardless of how complex $\phi$ is.

b. Learning $\hat{w}$ in this way is a convex optimization problem if and only if $\phi$ is a convex function in each dimension.
(Let $\phi = [\phi_1; \phi_2; \ldots ; \phi_p]$; then we say $\phi$ is convex in each dimension if each of $\phi_1, \phi_2, \ldots, \phi_p$ is a convex function).

c. Learning $\hat{w}$ in this way is a convex optimization problem if and only if $\phi$ is a linear function.

d. Learning $\hat{w}$ in this way is a convex optimization problem if and only if $\phi$ is the identity function and $p = d$.

Correct answers: (a)

Explanation: Since we freeze $\phi$ and do not update it, this is equivalent to logistic regression with a fixed basis expansion.
Thus, it is a convex optimization problem regardless of how complex $\phi$ is.

## 34.

Recall that influence functions are used to approximate the effect of leaving out one training point, without actually retraining the model.

Assume that we have a twice-differentiable, strongly convex loss function $\ell(x, y; w)$, and as usual, we train a model $\hat{w}$ to minimize the average training loss:

$$\hat{w}= \operatorname{argmin}_{w} \frac{1}{n} \sum_{i=1}^{n} \ell_i(w),$$

where $\{(x^{(1)}, y^{(1)}),(x^{(2)}, y^{(2)}), \ldots, (x^{(n)}, y^{(n)})\}$ is our training set, and for notational convenience we define $\ell_i(w) = \ell(x^{(i)}, y^{(i)}; w)$.

Let $\Delta_{-i}$ be the change in the parameters $w$ after we remove training point $(x^{(i)}, y^{(i)})$ and retrain the model.

The influence function approximation tells us that

$$\Delta_{-i} = \frac{1}{n} H(\hat{w})^{-1} \nabla_w \ell_i(w) \Big|_{w=\hat{w}}$$

where the Hessian matrix $H(\hat{w})$ is defined as

$$H(\hat{w}) = \frac{1}{n} \sum_{i=1}^{n} \nabla_w^2 \ell_i(w) \Big|_{w=\hat{w}}$$

Consider the following linear regression model $f_w(x) = w^\top x$, where $x, w \in \mathbb{R}^d$.
We train with unregularized least squares regression to obtain $\hat{w}$.

What is $\Delta_{-i}$ for this model, in terms of $\hat{w}$ and the training data points?

Note: The symbols $\ell$ and $H$ should not appear in your answer. Replace them by working out the appropriate loss.

Answer:

Explanation: For least squares regression, we have that $\ell_i(w) = \frac{1}{2}(y^{(i)} - w^\top x^{(i)})^2$. (The $\frac{1}{2}$ is for convenience;
we can leave it out without changing the final answer.) Thus, $\nabla_w\ell_i(w) = -(y^{(i)} - w^\top x^{(i)})x^{(i)}$, and $H(w) = \frac{1}{n}\sum_{i=1}^{n} x^{(i)}x^{(i)\top}$.

Putting this together,

$$\Delta_{-i} = -\frac{1}{n}\left(\frac{1}{n}\sum_{i=1}^{n}x^{(i)}x^{(i)\top}\right)^{-1}(y^{(i)} - \hat{w}^\top x^{(i)}) x^{(i)}$$

$$= -\left(\sum_{i=1}^{n}x^{(i)}x^{(i)\top}\right)^{-1}(y^{(i)} - \hat{w}^\top x^{(i)}) x^{(i)}.$$

We accept both the simplified version (canceling $\frac{1}{n}$) and the unsimplified version.
 
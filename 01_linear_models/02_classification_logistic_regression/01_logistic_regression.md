# Classification and Logistic Regression

## Introduction to Classification

Classification is a fundamental task in machine learning where the goal is to assign input data points to one of several predefined categories or classes. Unlike regression, where the output variable is continuous, classification deals with discrete outputs. In binary classification, there are only two possible classes, often labeled as 0 and 1 (or negative and positive). 

### Why Classification Matters

Classification is everywhere in our digital world:
- **Email systems** decide whether to put messages in your inbox or spam folder
- **Social media platforms** determine what content to show you
- **Medical devices** help doctors diagnose diseases
- **Financial systems** detect fraudulent transactions
- **Autonomous vehicles** identify pedestrians, signs, and other vehicles

The ability to automatically categorize and make decisions based on data is one of the most powerful applications of machine learning.

### Real-World Examples

- **Email spam detection:** Input features represent properties of an email (word frequency, sender information, etc.), output is 1 if spam, 0 if legitimate
- **Medical diagnosis:** Input features are patient symptoms and test results, output is 1 if disease present, 0 if healthy
- **Credit card fraud detection:** Input features are transaction characteristics, output is 1 if fraudulent, 0 if legitimate

The terms **negative class** and **positive class** are used to refer to these two categories, and the output variable is often called the **label** or **target variable**.

### Mathematical Framework
In binary classification, we have:
- **Input space:** $\mathcal{X} \subseteq \mathbb{R}^d$ (feature vectors)
- **Output space:** $\mathcal{Y} = \{0, 1\}$ (binary labels)
- **Training data:** $\{(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \ldots, (x^{(n)}, y^{(n)})\}$
- **Goal:** Learn a function $h: \mathcal{X} \rightarrow \mathcal{Y}$ that accurately predicts the class label

### The Classification Challenge: Finding Decision Boundaries

At its core, classification is about finding **decision boundaries** - lines, curves, or surfaces that separate different classes in the feature space. Think of it like drawing lines on a map to separate different territories.

**Simple Example:** Imagine you're trying to classify fruits based on their weight and sweetness:
- Apples: heavy and sweet
- Oranges: medium weight and sweet  
- Bananas: light and sweet
- Lemons: light and sour

You could draw lines to separate these categories, and then use those lines to classify new fruits.

## 2.1 Logistic Regression

### Why Not Use Linear Regression for Classification?

At first glance, one might consider using linear regression for classification by thresholding its output. However, this approach has several fundamental problems:

#### Problem 1: Unbounded Outputs
Linear regression can produce predictions less than 0 or greater than 1:
- For $h_\theta(x) = \theta^T x$, if $\theta^T x = 2.5$, we get $h_\theta(x) = 2.5$
- If $\theta^T x = -1.2$, we get $h_\theta(x) = -1.2$
- These values don't make sense as probabilities, which must be between 0 and 1

**Intuition:** Think of probability as a percentage - you can't have 250% or -20% probability. Linear regression doesn't respect these natural boundaries.

#### Problem 2: Linear Relationship Assumption
Linear regression assumes a linear relationship between features and the target. However, the relationship between features and class probability is often nonlinear, especially when:
- Classes are not linearly separable
- The decision boundary is curved
- There are complex interactions between features

**Visual Example:** Imagine trying to separate two classes that form concentric circles. A straight line (linear boundary) can never perfectly separate them, but a curved boundary can.

#### Problem 3: Sensitivity to Outliers
Linear regression is sensitive to outliers because it tries to minimize squared error. In classification, outliers can dramatically affect the decision boundary.

**Example:** Consider a dataset with mostly class 0 examples and a few extreme class 1 examples. Linear regression might predict negative values for many class 0 examples to accommodate the outliers.

**Real-world analogy:** If you're trying to set a price threshold for "expensive" vs "cheap" items, a few extremely expensive items could push the threshold so high that most items are classified as "cheap."

### The Logistic (Sigmoid) Function: A Natural Solution

The **logistic function**, also known as the **sigmoid function**, solves these problems by providing a smooth, bounded mapping from real numbers to probabilities:

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

#### Key Properties of the Sigmoid Function

1. **Bounded Output:** $0 < g(z) < 1$ for all $z \in \mathbb{R}$
2. **Symmetric around 0:** $g(-z) = 1 - g(z)$
3. **Monotonic:** $g'(z) > 0$ for all $z$ (always increasing)
4. **Smooth:** Infinitely differentiable
5. **Asymptotic Behavior:** 
   - As $z \to \infty$, $g(z) \to 1$
   - As $z \to -\infty$, $g(z) \to 0$

#### Intuitive Understanding: The "Squeeze" Function

Think of the sigmoid function as a "squeeze" function that takes any real number and squeezes it into the interval [0,1]:

- **Large positive inputs** get squeezed close to 1
- **Large negative inputs** get squeezed close to 0  
- **Inputs near 0** get mapped to around 0.5

**Real-world analogy:** It's like a dimmer switch for a light bulb. As you turn the knob (input), the brightness (output) changes smoothly from off (0) to full brightness (1), with the middle position giving half brightness (0.5).

#### Visual Interpretation: The S-Curve

The S-shaped curve of the sigmoid function allows for:
- **Gradual transitions:** Small changes in input lead to small changes in probability near the center
- **Sharp decisions:** Large changes in input lead to small changes in probability near the extremes
- **Nonlinear modeling:** Can capture complex decision boundaries

**Why the S-shape matters:**
- **Flat regions** (near 0 and 1): Model is confident, small changes don't affect prediction much
- **Steep region** (near 0.5): Model is uncertain, small changes can flip the prediction
- This matches our intuition about decision-making: we're most sensitive to changes when we're uncertain

#### Mathematical Beauty: The Derivative

The derivative of the sigmoid function has an elegant form:

$$
g'(z) = g(z)(1 - g(z))
$$

This is beautiful because:
1. **Self-referential:** The derivative is expressed in terms of the function itself
2. **Computationally efficient:** We can compute the derivative using only the function value
3. **Numerically stable:** Avoids computing exponentials twice

**Why this matters:** When we train our model, we need to compute gradients (derivatives) efficiently. This property makes training much faster and more stable.

### The Logistic Regression Model

In logistic regression, we model the probability that $y = 1$ given $x$ as:

$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
$$

where $\theta = [\theta_0, \theta_1, \ldots, \theta_d]^T$ is the parameter vector.

#### Parameter Interpretation: The "Influence" Weights

- **$\theta_0$ (intercept):** Controls the baseline probability when all features are zero
  - **Positive $\theta_0$:** Higher baseline probability for class 1
  - **Negative $\theta_0$:** Lower baseline probability for class 1
  
- **$\theta_j$ (feature weights):** Controls how much feature $j$ influences the probability
  - **Positive $\theta_j$:** Feature $j$ increases probability of class 1
  - **Negative $\theta_j$:** Feature $j$ decreases probability of class 1
  - **Large $|\theta_j|$:** Feature $j$ has strong influence
  - **Small $|\theta_j|$:** Feature $j$ has weak influence

**Real-world example:** In medical diagnosis:
- $\theta_0 = -2$: Low baseline probability of disease (healthy population)
- $\theta_{\text{fever}} = 1.5$: Having a fever strongly increases disease probability
- $\theta_{\text{age}} = 0.02$: Age has a small positive effect
- $\theta_{\text{exercise}} = -0.8$: Regular exercise decreases disease probability

#### Decision Boundary: Where Uncertainty Meets Certainty

The decision boundary is where $h_\theta(x) = 0.5$, which occurs when $\theta^T x = 0$:

$$
\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_d x_d = 0
$$

This defines a hyperplane in the feature space that separates the two classes.

**Intuition:** The decision boundary is where the model is most uncertain (probability = 0.5). On one side, the model predicts class 1 with higher confidence; on the other side, it predicts class 0 with higher confidence.

**Visual example:** In 2D, this is a straight line. Points on one side of the line are classified as class 1, points on the other side as class 0.

<img src="./img/sigmoid.png" width="300px" />

### Theoretical Justification: Why the Logistic Function?

The choice of the logistic function is not arbitrary. It emerges naturally from the framework of **Generalized Linear Models (GLMs)**:

#### 1. Exponential Family Connection
The logistic function is the **canonical link function** for the Bernoulli distribution. This means it's the natural choice when modeling binary outcomes.

**What this means:** The Bernoulli distribution describes binary random variables (like coin flips). The logistic function is mathematically "natural" for this distribution.

#### 2. Maximum Entropy Principle
The logistic function maximizes entropy subject to the constraint that the expected value matches the data.

**Intuition:** Among all possible functions that could model binary probabilities, the logistic function makes the fewest assumptions about the underlying data distribution.

#### 3. Convexity Guarantee
The resulting optimization problem is convex, ensuring we can find the global optimum.

**Why this matters:** Many optimization algorithms can get stuck in local minima. Convexity guarantees we'll find the best possible solution.

#### Alternative Functions: Why Not Others?

While other functions can map to [0,1], the logistic function has unique advantages:

- **Probit function:** $\Phi(z)$ (cumulative normal distribution) - similar properties but less computationally convenient
- **Tanh function:** $\tanh(z)$ - maps to [-1,1], requires rescaling
- **ReLU-based:** $\max(0, \min(1, z))$ - not smooth, harder to optimize

**The logistic function strikes the perfect balance:** smooth, bounded, computationally efficient, and theoretically well-founded.

### Probabilistic Interpretation: Beyond Just Predictions

Logistic regression provides a probabilistic framework for classification. We interpret the output as the probability that the label is 1 given the input features:

$$
\begin{align*}
P(y = 1 \mid x; \theta) &= h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}} \\
P(y = 0 \mid x; \theta) &= 1 - h_\theta(x) = \frac{e^{-\theta^T x}}{1 + e^{-\theta^T x}}
\end{align*}
$$

#### Advantages of Probabilistic Interpretation

1. **Uncertainty Quantification:** We get not just predictions but confidence levels
2. **Statistical Framework:** Can use maximum likelihood estimation and Bayesian methods
3. **Calibration:** Probabilities can be calibrated to match true frequencies
4. **Decision Making:** Can incorporate costs/benefits of different decisions

#### Example: Medical Diagnosis with Uncertainty

In medical diagnosis, knowing the probability is crucial:

- **High probability (0.9):** Strong evidence for disease, consider treatment
- **Medium probability (0.5):** Uncertain, order more tests
- **Low probability (0.1):** Likely healthy, but monitor

**Real-world impact:** A model that outputs 0.9 vs 0.1 leads to very different medical decisions, even though both might be classified as "disease present" with a 0.5 threshold.

#### Cost-Sensitive Decision Making

With probabilities, we can make optimal decisions based on costs:

- **False positive cost:** Cost of treating a healthy person
- **False negative cost:** Cost of missing a disease

**Optimal threshold:** $\text{threshold} = \frac{\text{cost of false positive}}{\text{cost of false positive} + \text{cost of false negative}}$

**Example:** If missing cancer (false negative) is 10 times more costly than unnecessary biopsy (false positive), the optimal threshold is $\frac{1}{1+10} = 0.09$.

### Likelihood and Log-Likelihood: Measuring Model Quality

#### Likelihood Function: How Well Does Our Model Explain the Data?

Given a dataset of $n$ independent training examples, the likelihood of the parameters $\theta$ is the probability of observing the data given the model:

$$
L(\theta) = \prod_{i=1}^n P(y^{(i)} \mid x^{(i)}; \theta)
$$

For logistic regression, this becomes:

$$
L(\theta) = \prod_{i=1}^n (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1 - y^{(i)}}
$$

#### Intuitive Understanding: The "Plausibility" Measure

The likelihood measures how well our model explains the observed data:
- **High likelihood:** Model assigns high probability to observed outcomes
- **Low likelihood:** Model assigns low probability to observed outcomes
- **Perfect model:** Likelihood = 1 (assigns probability 1 to all observed outcomes)

**Real-world analogy:** Think of likelihood as a "plausibility score." If your model says an event has 0.1 probability but it actually happened, that's implausible (low likelihood). If your model says an event has 0.9 probability and it happened, that's plausible (high likelihood).

#### Example Calculation: Step by Step

Consider a dataset with 3 examples:
- Example 1: $x = [1, 2]$, $y = 1$, model predicts $h_\theta(x) = 0.8$
- Example 2: $x = [1, 0]$, $y = 0$, model predicts $h_\theta(x) = 0.3$
- Example 3: $x = [1, 1]$, $y = 1$, model predicts $h_\theta(x) = 0.6$

Likelihood = $0.8^1 \cdot (1-0.8)^0 \cdot 0.3^0 \cdot (1-0.3)^1 \cdot 0.6^1 \cdot (1-0.6)^0 = 0.8 \cdot 0.7 \cdot 0.6 = 0.336$

**Interpretation:** The model explains this data with 33.6% plausibility.

#### Log-Likelihood: The Computationally Friendly Version

Maximizing the likelihood is equivalent to maximizing the log-likelihood:

$$
\ell(\theta) = \log L(\theta) = \sum_{i=1}^n y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)}))
$$

#### Why Use Log-Likelihood? Three Key Reasons

1. **Numerical Stability:** Products of small numbers can underflow, sums don't
   - **Example:** $0.1^{100} \approx 10^{-100}$ (underflow)
   - **Log version:** $100 \cdot \log(0.1) = -230$ (manageable)

2. **Mathematical Convenience:** Derivatives of sums are easier than derivatives of products
   - **Product rule:** $(f \cdot g)' = f'g + fg'$ (complex)
   - **Sum rule:** $(f + g)' = f' + g'$ (simple)

3. **Connection to Information Theory:** Related to cross-entropy and KL divergence
   - **Cross-entropy:** $-\frac{1}{n}\ell(\theta)$ (average log-likelihood)
   - **KL divergence:** Measures difference between true and predicted distributions

#### Example Calculation: Log-Likelihood

Using the same 3 examples:
- Example 1: $1 \cdot \log(0.8) + 0 \cdot \log(0.2) = -0.223$
- Example 2: $0 \cdot \log(0.3) + 1 \cdot \log(0.7) = -0.357$
- Example 3: $1 \cdot \log(0.6) + 0 \cdot \log(0.4) = -0.511$

Log-likelihood = $-0.223 - 0.357 - 0.511 = -1.091$

**Interpretation:** The log-likelihood is negative (as expected for probabilities < 1), and we want to maximize it (make it less negative).

### Gradient Ascent for Logistic Regression: Learning the Parameters

#### Optimization Objective: Finding the Best Parameters

We want to find the parameters $\theta$ that maximize the log-likelihood:

$$
\theta^* = \arg\max_\theta \ell(\theta)
$$

Since we're maximizing (not minimizing), we use **gradient ascent**:

$$
\theta_j := \theta_j + \alpha \frac{\partial}{\partial \theta_j} \ell(\theta)
$$

where $\alpha$ is the learning rate.

#### Gradient Derivation: The Beautiful Result

For a single training example, the gradient is:

$$
\begin{align*}
\frac{\partial}{\partial \theta_j} \ell(\theta) &= \frac{\partial}{\partial \theta_j} \left[y \log h_\theta(x) + (1 - y) \log(1 - h_\theta(x))\right] \\
&= y \frac{1}{h_\theta(x)} \frac{\partial h_\theta(x)}{\partial \theta_j} + (1 - y) \frac{1}{1 - h_\theta(x)} \frac{\partial (1 - h_\theta(x))}{\partial \theta_j} \\
&= y \frac{1}{h_\theta(x)} h_\theta(x)(1 - h_\theta(x))x_j + (1 - y) \frac{1}{1 - h_\theta(x)} (-h_\theta(x)(1 - h_\theta(x))x_j) \\
&= y(1 - h_\theta(x))x_j - (1 - y)h_\theta(x)x_j \\
&= (y - h_\theta(x))x_j
\end{align*}
$$

#### Intuitive Understanding: The Error Signal

The gradient $(y - h_\theta(x))x_j$ has a clear interpretation:
- **$y - h_\theta(x)$:** Prediction error (how far off our prediction is)
- **$x_j$:** Feature value (how much this feature contributes)
- **Positive gradient:** Increase $\theta_j$ to improve prediction
- **Negative gradient:** Decrease $\theta_j$ to improve prediction

**Real-world analogy:** It's like adjusting the volume on different speakers in a sound system:
- If the bass is too loud, you turn down the bass control (negative adjustment)
- If the treble is too quiet, you turn up the treble control (positive adjustment)
- The amount you adjust depends on how much that speaker contributes to the overall sound

#### Example: Step-by-Step Gradient Update

Consider updating $\theta_1$ for example $x = [1, 2]$, $y = 1$, current $\theta = [0.5, 0.3]$:
1. Current prediction: $h_\theta(x) = \frac{1}{1 + e^{-(0.5 + 0.3 \cdot 2)}} = \frac{1}{1 + e^{-1.1}} \approx 0.75$
2. Error: $y - h_\theta(x) = 1 - 0.75 = 0.25$ (positive error)
3. Feature value: $x_1 = 2$
4. Gradient: $(y - h_\theta(x))x_1 = 0.25 \cdot 2 = 0.5$
5. Update: $\theta_1 := \theta_1 + \alpha \cdot 0.5$

**Interpretation:** Since we predicted 0.75 but the true label is 1, we need to increase our prediction. The positive gradient tells us to increase $\theta_1$.

#### Stochastic Gradient Ascent: Learning from One Example at a Time

For large datasets, we can use stochastic gradient ascent, updating parameters after each example:

$$
\theta_j := \theta_j + \alpha \left(y^{(i)} - h_\theta(x^{(i)})\right)x_j^{(i)}
$$

**Advantages:**
- **Memory efficient:** Only need one example at a time
- **Online learning:** Can update model as new data arrives
- **Escape local minima:** Stochasticity can help escape poor local optima

**Disadvantages:**
- **Noisy updates:** Individual updates can be noisy
- **Slower convergence:** May take more iterations to converge
- **Sensitive to learning rate:** Need to tune learning rate carefully

#### Comparison with Linear Regression: The Surprising Similarity

Interestingly, the update rule looks identical to linear regression's LMS rule, but with different $h_\theta(x)$:
- **Linear regression:** $h_\theta(x) = \theta^T x$ (linear)
- **Logistic regression:** $h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}$ (nonlinear)

This similarity is not coincidental - it emerges from the mathematical structure of both models. Both are using the same optimization principle: adjust parameters in the direction that reduces prediction error.

### The Logistic Loss and Logit: Alternative Perspectives

#### Logistic Loss Function: A Different View

The **logistic loss** (or log-loss) is another way to express the cost function:

$$
\ell_{\text{logistic}}(t, y) = y \log(1 + \exp(-t)) + (1 - y) \log(1 + \exp(t))
$$

where $t = \theta^T x$ is called the **logit**.

#### Properties of Logistic Loss: Why It's Special

1. **Convex:** Has a unique global minimum
2. **Smooth:** Infinitely differentiable
3. **Well-behaved:** No exploding gradients
4. **Calibrated:** Minimizing it leads to well-calibrated probabilities

**Why convexity matters:** Convex functions have no local minima, so gradient descent will always find the global optimum.

#### Logit Interpretation: The "Raw Score"

The logit $t = \theta^T x$ represents the "raw score" before applying the sigmoid:
- **Large positive logit:** High confidence in class 1
- **Large negative logit:** High confidence in class 0
- **Logit near 0:** Uncertainty

**Real-world analogy:** Think of the logit as a "test score" before converting to a letter grade. A score of 95 becomes an A, a score of 65 becomes a D, but the raw score gives you more information about performance.

#### Connection to Log-Likelihood: Mathematical Equivalence

The negative log-likelihood equals the logistic loss:

$$
-\ell(\theta) = \ell_{\text{logistic}}(\theta^T x, y)
$$

This shows that minimizing logistic loss is equivalent to maximizing log-likelihood.

**Why this matters:** We can think about the problem in two equivalent ways:
1. **Maximizing likelihood:** Find parameters that make observed data most probable
2. **Minimizing loss:** Find parameters that minimize prediction error

Both lead to the same solution, but sometimes one perspective is more intuitive than the other.

### Practical Considerations: Making It Work in the Real World

#### Feature Scaling: Why It Matters

Logistic regression is sensitive to feature scales:
- **Large features:** Can dominate the decision boundary
- **Small features:** May be ignored
- **Solution:** Standardize features to have mean 0 and variance 1

**Example:** If one feature ranges from 0-1 and another from 0-1000, the second feature will dominate the decision boundary even if it's less important.

**Standardization formula:** $x' = \frac{x - \mu}{\sigma}$
- $\mu$: mean of the feature
- $\sigma$: standard deviation of the feature

#### Regularization: Preventing Overfitting

To prevent overfitting, add regularization terms:

**L2 regularization (Ridge):** $\lambda \sum_{j=1}^d \theta_j^2$
- **Effect:** Shrinks all parameters toward zero
- **Intuition:** Prefers simpler models with smaller weights
- **Use case:** When you suspect many features are relevant

**L1 regularization (Lasso):** $\lambda \sum_{j=1}^d |\theta_j|$
- **Effect:** Sets some parameters exactly to zero
- **Intuition:** Performs feature selection automatically
- **Use case:** When you suspect many features are irrelevant

**Elastic net:** Combination of L1 and L2
- **Formula:** $\lambda_1 \sum_{j=1}^d |\theta_j| + \lambda_2 \sum_{j=1}^d \theta_j^2$
- **Advantage:** Combines benefits of both approaches

#### Convergence: How Long Does Training Take?

Gradient ascent typically converges in 10-100 iterations:
- **Small learning rate:** Slower but more stable convergence
- **Large learning rate:** Faster but may oscillate
- **Adaptive learning rate:** Best of both worlds

**Convergence criteria:**
- **Tolerance:** Stop when parameter changes are small
- **Maximum iterations:** Stop after fixed number of iterations
- **Validation loss:** Stop when validation performance stops improving

#### Initialization: Where to Start?

- **Zero initialization:** $\theta = 0$ works well
- **Random initialization:** Can help break symmetry in some cases
- **Warm start:** Use previous solution as starting point

**Why zero initialization works:** The sigmoid function is symmetric around 0, so starting at 0 gives equal probability to both classes, which is a reasonable starting point.

### Advanced Topics: Beyond the Basics

#### Multiclass Classification: Extending to Multiple Classes

Logistic regression can be extended to multiple classes using:
1. **One-vs-Rest (OvR):** Train one classifier per class
2. **Multinomial logistic regression:** Direct extension using softmax function

**Softmax function:** $P(y = k \mid x) = \frac{e^{\theta_k^T x}}{\sum_{j=1}^K e^{\theta_j^T x}}$

#### Bayesian Logistic Regression: Incorporating Prior Knowledge

Instead of finding point estimates, we can model parameter uncertainty:

**Prior:** $P(\theta) = \mathcal{N}(0, \sigma^2 I)$
**Posterior:** $P(\theta \mid D) \propto P(D \mid \theta) P(\theta)$

**Advantages:**
- **Uncertainty quantification:** Get confidence intervals for predictions
- **Regularization:** Prior acts as natural regularizer
- **Interpretability:** Can incorporate domain knowledge

#### Online Learning: Learning from Streaming Data

Update the model as new data arrives:

**Online update rule:** $\theta_{t+1} = \theta_t + \alpha_t (y_t - h_{\theta_t}(x_t))x_t$

**Applications:**
- **Recommendation systems:** Update preferences as users interact
- **Fraud detection:** Adapt to new fraud patterns
- **Trading systems:** Adapt to changing market conditions

### Summary and Further Reading

Logistic regression is a foundational algorithm for binary classification that combines ideas from linear models, probability theory, and optimization. Its probabilistic interpretation makes it a natural choice for many applications, and its loss function and gradient have elegant mathematical properties.

#### Key Takeaways

1. **Probabilistic Framework:** Outputs interpretable probabilities
2. **Convex Optimization:** Guaranteed to find global optimum
3. **Efficient Training:** Simple gradient-based updates
4. **Theoretical Foundation:** Well-grounded in statistics and GLMs
5. **Practical Versatility:** Works well in many real-world applications

#### When to Use Logistic Regression

**Good for:**
- Binary classification problems
- When you need probabilistic outputs
- When features are interpretable
- When you want a simple, fast model
- When you have limited training data

**Consider alternatives when:**
- Classes are not linearly separable
- You need to capture complex feature interactions
- You have very high-dimensional data
- You need to model sequential dependencies

#### Advanced Topics

For more advanced topics, see:
- **Regularization:** Preventing overfitting
- **Multiclass classification:** Extending to multiple classes
- **Generalized Linear Models:** Theoretical framework
- **Bayesian logistic regression:** Incorporating prior knowledge
- **Online learning:** Updating models with streaming data

## From Probabilistic Classification to Deterministic Learning

We've now established a solid foundation for binary classification using logistic regression. Our approach has been **probabilistic** - we model the probability of belonging to each class and use the sigmoid function to ensure our outputs are valid probabilities. This gives us interpretable predictions and a principled way to handle uncertainty.

However, there's another approach to classification that's historically significant and conceptually simpler: the **perceptron algorithm**. Instead of modeling probabilities, the perceptron makes **deterministic** predictions using a hard threshold function. While this approach has limitations, it introduces important concepts that form the foundation of neural networks and provides insights into the geometric nature of classification problems.

The perceptron represents a different philosophical approach to classification - rather than asking "what's the probability this belongs to class 1?", it asks "which side of the decision boundary does this point fall on?" This geometric perspective will help us understand the fundamental challenges and opportunities in classification.

In the next section, we'll explore the perceptron algorithm and see how it relates to, and differs from, our probabilistic logistic regression approach.

---

**Next: [Perceptron Algorithm](02_perceptron.md)** - Learn about the perceptron learning algorithm and its relationship to linear classification.
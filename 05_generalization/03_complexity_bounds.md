# Sample Complexity Bounds: Theoretical Foundations of Generalization

## Introduction: The Quest for Theoretical Guarantees - Why Theory Matters

In the previous sections, we explored the bias-variance tradeoff and the double descent phenomenon through empirical observations and intuitive explanations. Now we turn to the **theoretical foundations** that provide rigorous mathematical guarantees about generalization. These theoretical results help us understand:

- **How many training examples do we need** to achieve good generalization?
- **What is the relationship** between training error and generalization error?
- **How does model complexity** affect the required sample size?
- **What are the fundamental limits** of learning?

This section introduces the mathematical tools and theoretical results that form the foundation of statistical learning theory.

### Why Theoretical Guarantees Matter: The Practical Motivation

**The Real-World Problem:**
Imagine you're building a medical diagnosis system:
- **Question**: How many patient records do I need to train a reliable model?
- **Question**: How confident can I be that my model will work on new patients?
- **Question**: Should I use a simple model with less data or a complex model with more data?

**The Theoretical Answer:**
Sample complexity bounds provide mathematical guarantees that help answer these questions:
- **Sample size requirements**: How much data you need for reliable performance
- **Confidence bounds**: How certain you can be about generalization
- **Complexity trade-offs**: When to use simple vs. complex models

**Real-World Analogy: The Bridge Building Problem**
Think of machine learning like building a bridge:
- **Empirical approach**: Build a bridge and hope it doesn't collapse
- **Theoretical approach**: Calculate the required materials and safety margins
- **Theoretical guarantees**: Provide confidence that the bridge will hold

### The Journey from Intuition to Rigor

**What We've Learned So Far:**
- **Bias-variance tradeoff**: Intuitive understanding of overfitting vs. underfitting
- **Double descent**: Modern phenomena that challenge classical wisdom
- **Empirical observations**: What we see in practice

**What We Need Now:**
- **Mathematical foundations**: Rigorous proofs and guarantees
- **Sample complexity**: How much data is enough?
- **Theoretical bounds**: Worst-case guarantees for safety

**The Bridge from Practice to Theory:**
```
Empirical Observations → Intuitive Understanding → Mathematical Analysis → Theoretical Guarantees
     (What works)           (Why it works)          (How to prove it)      (When it works)
```

## From Empirical Observations to Theoretical Foundations: The Mathematical Bridge

We've now explored the **double descent phenomenon** - a modern discovery that challenges classical wisdom about model complexity and generalization. We've seen how the relationship between complexity and generalization is more nuanced than the traditional U-shaped curve, with very complex models often achieving excellent generalization despite being highly overparameterized.

However, while empirical observations and intuitive explanations help us understand these phenomena, we need **theoretical foundations** that provide rigorous mathematical guarantees about generalization. Understanding why these phenomena occur and when we can expect them requires deeper mathematical analysis.

This motivates our exploration of **sample complexity bounds** - the theoretical tools that provide rigorous mathematical guarantees about generalization. We'll see how theoretical results help us understand the fundamental limits of learning, the relationship between training error and generalization error, and how model complexity affects the required sample size.

The transition from empirical phenomena to theoretical foundations represents the bridge from observation to understanding - taking our knowledge of how generalization works in practice and providing the mathematical framework to explain why.

In this section, we'll explore the mathematical tools and theoretical results that form the foundation of statistical learning theory.

---

## Mathematical Preliminaries: Building Blocks for Learning Theory

### The Union Bound: A Fundamental Tool - When Multiple Things Can Go Wrong

**Intuition:** The union bound is a simple but powerful tool that helps us control the probability of multiple "bad events" happening simultaneously. It says that the probability of any one of several events occurring is at most the sum of their individual probabilities.

**Real-World Analogy: The Weather Forecast Problem**
Think of the union bound like weather forecasting:
- **Event A**: It rains tomorrow (P(A) = 0.3)
- **Event B**: It snows tomorrow (P(B) = 0.1)
- **Event C**: It's sunny tomorrow (P(C) = 0.6)
- **Question**: What's the probability of any precipitation (rain OR snow)?
- **Union bound**: P(rain OR snow) ≤ P(rain) + P(snow) = 0.3 + 0.1 = 0.4

**Mathematical Statement:** Let $`A_1, A_2, \ldots, A_k`$ be $`k`$ events (not necessarily independent). Then:
```math
P(A_1 \cup A_2 \cup \cdots \cup A_k) \leq P(A_1) + P(A_2) + \cdots + P(A_k)
```

**Why This Matters for Learning Theory:**
In machine learning, we often need to ensure that multiple conditions hold simultaneously:
- Training error is close to generalization error for hypothesis 1
- Training error is close to generalization error for hypothesis 2
- ... and so on for all hypotheses in our class

The union bound helps us control the probability that **any** of these conditions fail.

**Visual Analogy: The Safety Net Problem**
Think of the union bound like setting up safety nets:
- **Individual nets**: Each catches one type of failure
- **Combined coverage**: Union bound tells us the total failure probability
- **Conservative estimate**: We may be overestimating, but we're safe

**Example - Multiple Hypotheses:**

See the complete implementation in [`code/complexity_bounds_demo.py`](code/complexity_bounds_demo.py) which demonstrates:

- **Union Bound Analysis**: How the union bound provides conservative estimates for multiple hypotheses
- **Training vs True Error Rates**: Visualization of the relationship between empirical and true error rates
- **Deviation Distribution**: Analysis of how deviations between training and true errors are distributed
- **Conservative Nature**: Demonstration of how the union bound is typically conservative in practice

The code shows how the union bound helps control the probability that any hypothesis has training error far from its generalization error, providing the foundation for uniform convergence.

**When the Union Bound is Tight vs. Loose:**
- **Tight**: When events are mutually exclusive (can't happen together)
- **Loose**: When events are highly correlated (often happen together)
- **Learning theory**: Usually loose because hypotheses are correlated

### The Hoeffding Inequality: Concentration of Averages - Why Averages Are Reliable

**Intuition:** The Hoeffding inequality tells us that the average of many independent random variables is very likely to be close to the true mean, as long as we have enough samples. This is the foundation for why we can trust empirical averages as estimates of true expectations.

**Real-World Analogy: The Polling Problem**
Think of Hoeffding's inequality like political polling:
- **Question**: What percentage of voters support Candidate A?
- **Method**: Ask 1000 random voters
- **Result**: 45% say they support Candidate A
- **Hoeffding**: With high probability, the true support is between 42% and 48%

**Mathematical Statement:** Let $`Z_1, Z_2, \ldots, Z_n`$ be $`n`$ independent and identically distributed (iid) random variables drawn from a Bernoulli($`\phi`$) distribution. That is, $`P(Z_i = 1) = \phi`$ and $`P(Z_i = 0) = 1 - \phi`$. Let $`\hat{\phi} = \frac{1}{n} \sum_{i=1}^n Z_i`$ be the sample mean. Then for any $`\gamma > 0`$:
```math
P(|\phi - \hat{\phi}| > \gamma) \leq 2 \exp(-2\gamma^2 n)
```

**Key Insights:**
- The probability of large deviations decreases **exponentially** with the sample size $`n`$
- The bound depends on the **squared** deviation $`\gamma^2`$
- The factor of 2 comes from considering both positive and negative deviations

**Visual Analogy: The Dart Throwing Game**
Think of Hoeffding's inequality like throwing darts at a target:
- **Individual throws**: Each dart has some error
- **Average position**: The center of all darts
- **Concentration**: As you throw more darts, the average gets more accurate
- **Exponential decay**: The probability of large errors decreases exponentially

**Why This Matters for Learning:**
In machine learning, we often estimate probabilities (like error rates) from finite samples:
- **Training error**: Average loss on training examples
- **True error**: Expected loss on all possible examples
- **Hoeffding**: Tells us how reliable our training error estimate is

**Practical Example - Error Rate Estimation:**

See the complete implementation in [`code/complexity_bounds_demo.py`](code/complexity_bounds_demo.py) which demonstrates:

- **Hoeffding's Inequality**: Empirical vs theoretical bounds for different sample sizes
- **Sample Size Requirements**: How much data is needed for different accuracy and confidence levels
- **Exponential Decay**: Visualization of how probability of large deviations decreases exponentially with sample size
- **Practical Guidelines**: Sample size tables for different accuracy and confidence requirements

The code shows how Hoeffding's inequality provides mathematical guarantees about the reliability of empirical averages, with the probability of large deviations decreasing exponentially with sample size.

**Key Properties of Hoeffding's Inequality:**
1. **Exponential decay**: Probability decreases exponentially with sample size
2. **Squared accuracy**: Higher accuracy requires quadratically more samples
3. **Distribution-free**: Works for any bounded random variables
4. **Conservative**: Often provides loose bounds in practice

### Connection to Learning Theory: The Building Blocks Strategy

These two tools—the union bound and Hoeffding's inequality—are the building blocks for proving generalization bounds. The basic strategy is:

1. **Use Hoeffding's inequality** to show that training error is close to generalization error for a single hypothesis
2. **Use the union bound** to extend this to all hypotheses in our class simultaneously
3. **Combine with optimization** to bound the generalization error of the learned hypothesis

**The Learning Theory Recipe:**

See the complete implementation in [`code/complexity_bounds_demo.py`](code/complexity_bounds_demo.py) which demonstrates:

- **Four-Step Process**: The fundamental recipe for proving generalization bounds
- **Mathematical Framework**: How to combine Hoeffding's inequality with the union bound
- **Sample Complexity Calculation**: How to determine required sample sizes for given accuracy and confidence
- **Practical Example**: Concrete calculations showing the relationship between parameters

The code shows the systematic approach to proving generalization bounds: single hypothesis analysis, uniform convergence, ERM connection, and sample complexity determination.

**The Fundamental Insight:**
The combination of concentration (Hoeffding) and union bound allows us to control the probability that **any** hypothesis in our class has training error far from its generalization error. This is the foundation of uniform convergence, which is the key theoretical tool for understanding generalization.

---

## The Learning Framework: Formalizing the Problem - From Intuition to Mathematics

### Binary Classification Setup: The Simplest Case

To simplify our exposition, we focus on **binary classification** where the labels are $`y \in \{0, 1\}`$. Everything we discuss generalizes to other problems (regression, multi-class classification, etc.).

**Data Generation Process:**
- We have a training set $`S = \{(x^{(i)}, y^{(i)})\}_{i=1}^n`$ of size $`n`$
- Training examples are drawn independently and identically from a distribution $`\mathcal{D}`$
- Each example consists of an input $`x^{(i)}`$ and its corresponding label $`y^{(i)}`$

**Key Assumption:** The training and test data come from the **same distribution** $`\mathcal{D}`$. This is sometimes called the **iid assumption** (independent and identically distributed).

**Real-World Analogy: The Medical Diagnosis Problem**
Think of binary classification like medical diagnosis:
- **Input x**: Patient symptoms, test results, demographics
- **Output y**: 0 (healthy) or 1 (disease present)
- **Distribution D**: The population of patients we want to diagnose
- **Training set S**: Historical patient records
- **Goal**: Learn to diagnose new patients accurately

**Visual Analogy: The Coin Flipping Problem**
Think of learning like predicting coin flips:
- **Input x**: Features of the coin (weight, size, material)
- **Output y**: 0 (heads) or 1 (tails)
- **Distribution D**: The process that generates coins and flips
- **Training set S**: Results from flipping many coins
- **Goal**: Predict the outcome of flipping a new coin

### Error Definitions: Training vs. True Performance

**Training Error (Empirical Risk):** For a hypothesis $`h`$, the training error is:
```math
\hat{\varepsilon}(h) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{h(x^{(i)}) \neq y^{(i)}\}
```

This is the fraction of training examples that $`h`$ misclassifies. When we want to emphasize the dependence on the training set $`S``, we write $`\hat{\varepsilon}_S(h)`$.

**Generalization Error (True Risk):** The generalization error is:
```math
\varepsilon(h) = P_{(x, y) \sim \mathcal{D}}(h(x) \neq y)
```

This is the probability that $`h`$ misclassifies a new example drawn from $`\mathcal{D}``.

**Key Insight:** The training error $`\hat{\varepsilon}(h)`$ is a **random variable** (it depends on the random training set), while the generalization error $`\varepsilon(h)`` is a **fixed quantity** for a given hypothesis.

**Visual Analogy: The Weather Prediction Problem**
Think of error rates like weather prediction:
- **Training error**: How well you predicted weather for the past 100 days
- **Generalization error**: How well you'll predict weather for the next 100 days
- **Key difference**: Past performance is known, future performance is uncertain

**Practical Example - Error Rate Calculation:**

See the complete implementation in [`code/complexity_bounds_demo.py`](code/complexity_bounds_demo.py) which demonstrates:

- **Training vs Generalization Error**: The fundamental difference between empirical and true error rates
- **Linear Classification**: Real-world example using logistic regression on synthetic data
- **Generalization Gap**: Visualization of the difference between training and test performance
- **Data Visualization**: Scatter plots showing training and test data distributions

The code shows how training error (empirical risk) can differ from generalization error (true risk), highlighting the importance of theoretical bounds for understanding this relationship.

### The Learning Algorithm: Empirical Risk Minimization

**Empirical Risk Minimization (ERM):** Given a hypothesis class $`\mathcal{H}``, the learning algorithm chooses:
```math
\hat{h} = \arg\min_{h \in \mathcal{H}} \hat{\varepsilon}(h)
```

That is, it picks the hypothesis with the smallest training error.

**Hypothesis Class:** The set $`\mathcal{H}`` contains all the hypotheses that our learning algorithm considers. For example:
- **Linear classifiers:** $`\mathcal{H} = \{h_\theta : h_\theta(x) = \mathbf{1}\{\theta^T x \geq 0\}, \theta \in \mathbb{R}^{d+1}\}``
- **Neural networks:** $`\mathcal{H}`` is the set of all functions representable by a given architecture
- **Decision trees:** $`\mathcal{H}`` is the set of all decision trees with a given maximum depth

**The Challenge:** We want to bound the generalization error $`\varepsilon(\hat{h})`$ of the learned hypothesis, but we only have access to the training error $`\hat{\varepsilon}(\hat{h})``.

**Real-World Analogy: The Restaurant Selection Problem**
Think of ERM like choosing a restaurant:
- **Hypothesis class H**: All restaurants in the city
- **Training set S**: Reviews from your friends
- **Training error ε̂(h)**: Average rating from your friends
- **ERM**: Choose the restaurant with the best average rating
- **Challenge**: Will this restaurant be good for you too?

**Visual Analogy: The Dart Throwing Game Revisited**
Think of ERM like a dart throwing competition:
- **Hypothesis class H**: All possible throwing techniques
- **Training set S**: Practice throws
- **Training error ε̂(h)**: How well each technique worked in practice
- **ERM**: Choose the technique that worked best in practice
- **Challenge**: Will this technique work well in the actual competition?

**Example - ERM with Different Hypothesis Classes:**

See the complete implementation in [`code/complexity_bounds_demo.py`](code/complexity_bounds_demo.py) which demonstrates:

- **Multiple Hypothesis Classes**: Linear, polynomial, and decision tree models with different complexities
- **ERM Performance Comparison**: Training and test errors for each hypothesis class
- **Decision Boundary Visualization**: How different models learn different decision boundaries
- **Generalization Gap Analysis**: The relationship between model complexity and generalization performance

The code shows how ERM chooses the hypothesis with minimum training error from each class, and how this choice affects generalization performance across different model complexities.

**Key Insights from ERM:**
1. **Training error is optimistic**: It underestimates true error
2. **Complexity matters**: More complex models can have lower training error but higher test error
3. **The generalization gap**: The difference between training and test error
4. **Model selection**: Choosing the right complexity is crucial

**The Fundamental Question:**
How can we bound the generalization error $`\varepsilon(\hat{h})`$ in terms of the training error $`\hat{\varepsilon}(\hat{h})`$ and the complexity of the hypothesis class $`\mathcal{H}``?

This is the central question of statistical learning theory, and the answer involves the mathematical tools we've introduced: concentration inequalities and union bounds.

## Finite Hypothesis Classes: The Simplest Case - When We Can Count Our Options

### Setup and Goal: The Finite Case Strategy

We start with the simplest case: a **finite hypothesis class** $`\mathcal{H} = \{h_1, h_2, \ldots, h_k\}`$ with $`k`$ hypotheses.

**Our Goal:** Prove that with high probability, the generalization error of the learned hypothesis $`\hat{h}`$ is close to the best possible generalization error in $`\mathcal{H}`$.

**Real-World Analogy: The Restaurant Menu Problem**
Think of finite hypothesis classes like choosing from a restaurant menu:
- **Hypothesis class H**: The menu with k dishes
- **Training set S**: Reviews from your friends
- **ERM**: Choose the dish with the best average rating
- **Goal**: Ensure the chosen dish is actually good

**Visual Analogy: The Dart Board Problem**
Think of finite hypothesis classes like a dart board with k targets:
- **Hypothesis class H**: k different targets on the board
- **Training set S**: Practice throws at each target
- **ERM**: Choose the target where you performed best in practice
- **Goal**: Ensure you'll hit the chosen target in the real game

### Step 1: Uniform Convergence for a Single Hypothesis - The Foundation

First, let's understand how training error relates to generalization error for a single hypothesis $`h_i`$.

**Key Insight:** For a fixed hypothesis $`h_i`$, the training error $`\hat{\varepsilon}(h_i)`$ is the average of $`n`$ independent Bernoulli random variables, each with mean $`\varepsilon(h_i)`$.

**Application of Hoeffding's Inequality:** For any $`\gamma > 0`$:
```math
P(|\varepsilon(h_i) - \hat{\varepsilon}(h_i)| > \gamma) \leq 2 \exp(-2\gamma^2 n)
```

This tells us that for a single hypothesis, training error is likely to be close to generalization error.

**Visual Analogy: The Coin Flipping Experiment**
Think of this like flipping a biased coin:
- **True probability**: The coin's actual bias (generalization error)
- **Sample average**: What we observe after n flips (training error)
- **Hoeffding**: With high probability, the sample average is close to the true probability

**Practical Example - Single Hypothesis Analysis:**

See the complete implementation in [`code/complexity_bounds_demo.py`](code/complexity_bounds_demo.py) which demonstrates:

- **Hoeffding's Inequality**: Empirical vs theoretical bounds for single hypothesis analysis
- **Sample Size Effects**: How probability of large deviations changes with sample size
- **Deviation Distributions**: Histograms showing the distribution of deviations for different sample sizes
- **Bound Tightness**: Analysis of how conservative Hoeffding's inequality is in practice

The code shows how Hoeffding's inequality provides exponential concentration guarantees for a single hypothesis, with the probability of large deviations decreasing exponentially with sample size.

### Step 2: Uniform Convergence for All Hypotheses - The Union Bound Magic

Now we want to ensure that training error is close to generalization error for **all** hypotheses simultaneously.

**The Challenge:** We need to control the probability that **any** hypothesis has training error far from its generalization error.

**Application of the Union Bound:** Let $`A_i`$ be the event that $`|\varepsilon(h_i) - \hat{\varepsilon}(h_i)| > \gamma`$. Then:
```math
P(\exists h \in \mathcal{H}: |\varepsilon(h) - \hat{\varepsilon}(h)| > \gamma) = P(A_1 \cup A_2 \cup \cdots \cup A_k)
```

By the union bound:
```math
\leq P(A_1) + P(A_2) + \cdots + P(A_k)
\leq k \times 2 \exp(-2\gamma^2 n)
```

**The Uniform Convergence Result:** With probability at least $`1 - 2k \exp(-2\gamma^2 n)`$:
```math
|\varepsilon(h) - \hat{\varepsilon}(h)| \leq \gamma \quad \text{for all } h \in \mathcal{H}
```

This is called **uniform convergence** because it holds uniformly for all hypotheses in the class.

**Visual Analogy: The Safety Net Problem Revisited**
Think of uniform convergence like setting up multiple safety nets:
- **Individual nets**: Each hypothesis has its own safety net (Hoeffding)
- **Combined coverage**: Union bound ensures all hypotheses are covered
- **Failure probability**: Total probability that any net fails

**Real-World Analogy: The Quality Control Problem**
Think of uniform convergence like quality control in manufacturing:
- **Hypotheses**: Different production lines
- **Training error**: Quality measurements on sample products
- **Generalization error**: True quality of all products
- **Uniform convergence**: All production lines meet quality standards

**Practical Example - Multiple Hypotheses:**

See the complete implementation in [`code/complexity_bounds_demo.py`](code/complexity_bounds_demo.py) which demonstrates:

- **Uniform Convergence**: How the union bound extends single hypothesis guarantees to all hypotheses
- **Failure Probability Analysis**: Empirical vs theoretical bounds for uniform convergence
- **Conservative Nature**: Demonstration of how the union bound is typically conservative
- **Deviation Analysis**: Distribution of deviations across multiple hypotheses

The code shows how uniform convergence ensures that training error is close to generalization error for all hypotheses simultaneously, providing the foundation for bounding ERM performance.

### Step 3: Bounding the Generalization Error - The ERM Connection

Now we can bound the generalization error of the learned hypothesis $`\hat{h}`$.

**Key Insight:** Since $`\hat{h}`$ was chosen to minimize training error, we have $`\hat{\varepsilon}(\hat{h}) \leq \hat{\varepsilon}(h)`$ for all $`h \in \mathcal{H}`$.

**The Argument:**
1. By uniform convergence: $`\varepsilon(\hat{h}) \leq \hat{\varepsilon}(\hat{h}) + \gamma`$
2. Since $`\hat{h}`$ minimizes training error: $`\hat{\varepsilon}(\hat{h}) \leq \hat{\varepsilon}(h^*)`$
3. By uniform convergence again: $`\hat{\varepsilon}(h^*) \leq \varepsilon(h^*) + \gamma`$
4. Combining: $`\varepsilon(\hat{h}) \leq \varepsilon(h^*) + 2\gamma`$

**The Final Result:** With probability at least $`1 - 2k \exp(-2\gamma^2 n)`$:
```math
\varepsilon(\hat{h}) \leq \varepsilon(h^*) + 2\gamma
```

where $`h^* = \arg\min_{h \in \mathcal{H}} \varepsilon(h)`$ is the best hypothesis in the class.

**Visual Analogy: The Race Problem**
Think of this like a race with practice heats:
- **Practice heats**: Training error for each hypothesis
- **ERM**: Choose the hypothesis that performed best in practice
- **Uniform convergence**: Practice times are close to real race times
- **Final bound**: The chosen hypothesis will perform close to the best possible

**Real-World Analogy: The Job Interview Problem**
Think of this like a job interview process:
- **Hypotheses**: Different candidates
- **Training error**: Performance in interviews
- **ERM**: Hire the candidate who performed best in interviews
- **Generalization error**: How well they'll actually perform on the job
- **Bound**: The hired candidate will perform close to the best possible candidate

**Practical Example - ERM Analysis:**

See the complete implementation in [`code/complexity_bounds_demo.py`](code/complexity_bounds_demo.py) which demonstrates:

- **ERM Performance Analysis**: How well ERM performs compared to the best possible hypothesis
- **Generalization Gap Distribution**: Analysis of the gap between ERM and optimal performance
- **Hypothesis Selection Frequency**: How often ERM chooses different hypotheses
- **Statistical Analysis**: Mean, maximum, and distribution of ERM generalization errors

The code shows how ERM performs in practice, demonstrating that it typically achieves generalization error close to the best possible hypothesis in the class, validating the theoretical bounds.

### Step 4: Sample Complexity Bounds - How Much Data Do We Need?

We can solve for the required sample size $`n`$ to achieve a desired accuracy and confidence.

**Setting the Parameters:** Let $`\delta = 2k \exp(-2\gamma^2 n)`$ be our desired confidence level. Solving for $`n`$:
```math
n \geq \frac{1}{2\gamma^2} \log \frac{2k}{\delta}
```

**Interpretation:** To achieve generalization error within $`2\gamma`$ of the best possible with confidence $`1 - \delta`$, we need approximately $`O(\frac{1}{\gamma^2} \log \frac{k}{\delta})`$ samples.

**Key Properties:**
- **Logarithmic dependence on $`k`$**: The number of samples grows only logarithmically with the number of hypotheses
- **Quadratic dependence on $`1/\gamma`$**: Higher accuracy requires quadratically more samples
- **Logarithmic dependence on $`1/\delta`$**: Higher confidence requires only logarithmically more samples

**Visual Analogy: The Library Problem**
Think of sample complexity like building a library:
- **Hypotheses**: Different books
- **Accuracy**: How well you can find the right book
- **Confidence**: How sure you are the library is complete
- **Sample size**: How many books you need to check

**Practical Example - Sample Size Requirements:**

See the complete implementation in [`code/complexity_bounds_demo.py`](code/complexity_bounds_demo.py) which demonstrates:

- **Sample Complexity Tables**: Comprehensive tables showing required sample sizes for different parameters
- **Parameter Dependencies**: How sample size requirements depend on number of hypotheses, accuracy, and confidence
- **Visualization of Relationships**: Log-scale plots showing the relationships between parameters
- **Practical Guidelines**: Concrete examples of sample size requirements for real-world scenarios

The code shows how the theoretical bounds translate into practical sample size requirements, demonstrating the logarithmic dependence on hypothesis count and quadratic dependence on accuracy.

### Alternative Formulation: Error Bounds

We can also hold $`n`$ and $`\delta`$ fixed and solve for $`\gamma`$:
```math
|\varepsilon(h) - \hat{\varepsilon}(h)| \leq \sqrt{\frac{1}{2n} \log \frac{2k}{\delta}}
```

This gives us a bound on how close training and generalization error are likely to be.

**Key Insights from Finite Hypothesis Classes:**
1. **Logarithmic scaling**: The number of hypotheses doesn't require much more data
2. **Accuracy is expensive**: Higher accuracy requires quadratically more data
3. **Confidence is cheap**: Higher confidence requires only logarithmically more data
4. **Uniform convergence**: The key theoretical tool for finite classes

**The Foundation for Infinite Classes:**
The finite case provides the foundation for understanding infinite hypothesis classes. The key insight is that we need to replace the simple counting of hypotheses with more sophisticated measures of complexity.



# Problem Set 8 Solutions

## Problem 1: Irreducible Error

**1 points One Answer**

**Question:** Which of the following is the cause/reason for irreducible error?

**Options:**
- a) Stochastic label noise
- b) Very few data points
- c) Nonlinear relationships in the data
- d) Insufficient model complexity

**Correct answers:** (a)

**Explanation:** 
- **A is correct.** Stochastic label noise is what drives irreducible error. See lecture 4 slides. 
- In essence, irreducible error comes from randomness that cannot be modeled since there is no deeper pattern to it. 
- **B and D are wrong** because fewer data points and insufficient model complexity are responsible for reducible error. 
- **C is wrong** because nonlinear relationships in the data don't have anything to do with irreducible error.

## Detailed Solution Explanation

**Understanding Irreducible Error:**

Irreducible error represents the fundamental uncertainty in the data that cannot be eliminated by any model, no matter how sophisticated. It's the "noise" in the system that makes perfect prediction impossible.

**Mathematical Framework:**
In the standard regression setting, we model the relationship as:
$$y = f(x) + \epsilon$$
where:
- $f(x)$ is the true underlying function
- $\epsilon$ is the irreducible error (noise)

The irreducible error $\epsilon$ is typically assumed to follow a distribution (often Gaussian) with mean 0 and some variance $\sigma^2$:
$$\epsilon \sim \mathcal{N}(0, \sigma^2)$$

**Why Stochastic Label Noise Causes Irreducible Error:**
- **Stochastic** means random and unpredictable
- **Label noise** refers to errors in the target variable $y$
- This noise cannot be learned or predicted from the features $x$
- Even with perfect knowledge of $f(x)$, we cannot predict $\epsilon$

**Examples of Irreducible Error:**
- Measurement errors in data collection
- Random fluctuations in biological systems
- Unpredictable external factors affecting outcomes
- Human error in labeling data

**Why Other Options Are Incorrect:**
- **Option B (Very few data points):** This causes high variance (reducible error) because the model cannot learn the true pattern effectively
- **Option C (Nonlinear relationships):** This can be modeled with appropriate algorithms (e.g., neural networks, kernel methods)
- **Option D (Insufficient model complexity):** This causes high bias (reducible error) because the model is too simple to capture the true relationship

## Problem 2: Bias-Variance Analysis

**1 points One Answer**

**Scenario:** Saket unfortunately did not learn from the midterm and still has not attended lecture. He is now given the task of training 3 neural networks with increasing complexity on a regression task:

* Model A: 1 hidden layer with 10 neurons.
* Model B: 2 hidden layers with 50 neurons each.
* Model C: 10 hidden layers with 100 neurons each.

After training and evaluating these models on an appropriately split dataset with train and test splits, you find the following MSEs:

* Model A: train MSE = 2.5, test MSE = 2.6
* Model B: train MSE = 0.1, test MSE = 0.2
* Model C: train MSE = 0.01, test MSE = 1.3

Saket only knows about bias and variance, So based on the model architectures and train/test MSE losses, chose the best relative bias/variance estimates for each of the models.

<img src="./img/q2_problem.png" width="500px">

**Bias/Variance Estimates:**

| Model | Bias      | Variance  |
|-------|-----------|-----------|
|       | Low | High | Low | High |
| A     | $\bigcirc$ | $\text{\textcircled{O}}$ | $\text{\textcircled{O}}$ | $\bigcirc$ |
| B     | $\text{\textcircled{O}}$ | $\bigcirc$ | $\text{\textcircled{O}}$ | $\bigcirc$ |
| C     | $\text{\textcircled{O}}$ | $\bigcirc$ | $\bigcirc$ | $\text{\textcircled{O}}$ |

**Explanation:** 
**Correct answer:** 
- A → high bias, low variance
- B → low bias, low variance  
- C → low bias, high variance

**Reasoning:**
- **Model A:** Due to the simpler architecture and high MSEs, A likely underfits.
- **Model B:** Achieves low but similar train/test MSEs so probably has a good balance.
- **Model C:** Has a low train MSE but a high test MSE so is probably overfitting, which matches the likely overcomplex architecture.

## Detailed Solution Explanation

**Understanding Bias-Variance Tradeoff:**

The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between model complexity and generalization error.

**Mathematical Framework:**
The expected prediction error can be decomposed as:
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

where:
- **Bias:** How far off the model's predictions are on average from the true values
- **Variance:** How much the model's predictions vary for different training sets
- **Irreducible Error:** The fundamental noise in the data

**Analyzing Each Model:**

**Model A (1 hidden layer, 10 neurons):**
- **Architecture:** Simple model with limited capacity
- **Performance:** High train MSE (2.5) and test MSE (2.6)
- **Analysis:** The model cannot capture the underlying pattern in the data
- **Bias:** High (model is too simple to learn the true relationship)
- **Variance:** Low (simple models are stable across different training sets)

**Model B (2 hidden layers, 50 neurons each):**
- **Architecture:** Moderate complexity
- **Performance:** Low train MSE (0.1) and test MSE (0.2)
- **Analysis:** The model captures the pattern well without overfitting
- **Bias:** Low (model can learn the true relationship)
- **Variance:** Low (good generalization, small gap between train and test)

**Model C (10 hidden layers, 100 neurons each):**
- **Architecture:** Very complex model with high capacity
- **Performance:** Very low train MSE (0.01) but high test MSE (1.3)
- **Analysis:** The model memorizes the training data but doesn't generalize
- **Bias:** Low (model can fit the training data perfectly)
- **Variance:** High (model is sensitive to training data, poor generalization)

**Key Insights:**
- **Underfitting:** High bias, low variance (Model A)
- **Good Fit:** Low bias, low variance (Model B) 
- **Overfitting:** Low bias, high variance (Model C)

**Visual Interpretation:**
- **High Bias:** Model predictions are systematically off-target
- **High Variance:** Model predictions are scattered around the target
- **Optimal:** Model predictions cluster tightly around the target

## Problem 3: K-Fold Cross Validation

**2 points**

**Question:** Explain one upside and one downside of using a high K in K-fold cross validation.

<img src="./img/q4_problem.png" width="350px">

**Upside:**

You get a more accurate estimate of your test error, possibly making hyperparameter selection more accurate.

**Downside:**

A higher K means more folds and therefore much more compute/time needed to find the right hyperparameters. A higher K also means each validation set has fewer data points. This will result in higher variability in the results across different folds.

**Explanation:** 
**Possible answer:**

**Upside:** You get a more accurate estimate of your test error, possibly making hyperparameter selection more accurate.

**Downside:** A higher K means more folds and therefore much more compute/time needed to find the right hyperparameters. A higher K also means each validation set has fewer data points. This will result in higher variability in the results across different folds.

## Detailed Solution Explanation

**Understanding K-Fold Cross Validation:**

K-fold cross validation is a resampling technique used to assess how well a model will generalize to new, unseen data.

**Mathematical Framework:**
For a dataset with $n$ samples, K-fold CV:
- Divides data into $K$ equal-sized folds
- Each fold has approximately $\frac{n}{K}$ samples
- Trains on $K-1$ folds, validates on 1 fold
- Repeats $K$ times, using each fold as validation once

**Expected Test Error Estimate:**
$$\text{CV Error} = \frac{1}{K} \sum_{k=1}^{K} \text{Error}_k$$
where $\text{Error}_k$ is the error on the $k$-th validation fold.

**Upside of High K (e.g., K = 10 or leave-one-out):**

1. **More Accurate Error Estimation:**
   - Higher K means more validation sets
   - Reduces bias in error estimation
   - Better approximation of true generalization error

2. **Better Hyperparameter Selection:**
   - More reliable comparison between different hyperparameter settings
   - Reduces risk of selecting suboptimal hyperparameters due to lucky/unlucky data splits

3. **Statistical Efficiency:**
   - Uses more data for training (each training set has $\frac{K-1}{K} \cdot n$ samples)
   - More representative of the true data distribution

**Downside of High K:**

1. **Computational Cost:**
   - **Time Complexity:** $O(K \cdot T)$ where $T$ is training time for one model
   - **Space Complexity:** Need to store $K$ models
   - **Practical Limitation:** May be infeasible for large datasets or complex models

2. **Smaller Validation Sets:**
   - Each validation set has only $\frac{n}{K}$ samples
   - **Higher Variance:** Smaller validation sets lead to more variable error estimates
   - **Less Reliable:** Individual fold errors may not be representative

3. **Statistical Instability:**
   - High variance in cross-validation estimates
   - May lead to inconsistent hyperparameter selection
   - Risk of overfitting to the cross-validation procedure itself

**Optimal K Selection:**
- **Small datasets:** Use higher K (5-10) or leave-one-out
- **Large datasets:** Lower K (3-5) is often sufficient
- **Computational constraints:** Balance accuracy vs. time
- **Rule of thumb:** K = 5 or K = 10 are common choices

**Example Calculation:**
For $n = 1000$ samples:
- **K = 5:** Each fold has 200 samples, train on 800 samples
- **K = 10:** Each fold has 100 samples, train on 900 samples
- **Leave-one-out:** Each fold has 1 sample, train on 999 samples

## Problem 4: Training and Validation Loss

**1 points Select All That Apply**

**Question:** You are training a model and get the following plot for your training and validation loss.

**Plot Description:**
A line plot titled "Training and Validation Loss" shows two curves over "Number of Epochs" (x-axis) versus "Loss" (y-axis). The x-axis ranges from 2 to 20, with major ticks at 2, 5, 7, 10, 12, 15, 17, and 20. The y-axis ranges from 0.0 to 1.0, with major ticks at 0.0, 0.2, 0.4, 0.6, 0.8, and 1.0.

**Legend:**
- "train" is represented by a solid blue line.
- "validation" is represented by a dashed orange line.

**Train Loss Curve (solid blue):** Starts at a loss of 1.0 at epoch 2, rapidly decreases to near 0.0 by epoch 7, and remains very close to 0.0 for the rest of the epochs up to 20.

**Validation Loss Curve (dashed orange):** Starts at a loss of 1.0 at epoch 2, decreases to approximately 0.4 at epoch 7, then begins to increase, reaching about 0.7 by epoch 12, and subsequently plateaus around 0.7 to 0.75 until epoch 20.

**Sub-question:** Which of the following statements are true?

**Options:**
- a) $\bigcirc$ The model has high bias and low variance.
- b) $\bigcirc$ The large gap between training and validation loss indicates underfitting.
- c) $\bigcirc$ Training for more epochs will eventually decrease validation loss.
- d) $\text{\textcircled{O}}$ The model might be too complex for the dataset.
- e) $\text{\textcircled{O}}$ The model is likely memorizing the training data.

**Correct answers:** (d), (e)

**Explanation:** 
This is a classic example of overfitting, which is caused when we have too complex of a model and it ends up memorizing the training set. Overfitting means the model has low bias and high variance. Thus, the only correct options are D and E.

## Detailed Solution Explanation

**Understanding Overfitting from Loss Curves:**

This problem demonstrates a classic overfitting scenario where the model learns the training data too well but fails to generalize to new data.

**Key Observations from the Plot:**

1. **Training Loss:** Decreases rapidly and reaches near-zero by epoch 7
2. **Validation Loss:** Initially decreases but then increases after epoch 7
3. **Gap Between Curves:** Large and growing gap between training and validation loss
4. **Divergence Point:** Around epoch 7, validation loss starts increasing while training loss remains low

**Mathematical Interpretation:**

**Training Loss Behavior:**
$$\text{Train Loss}(t) \rightarrow 0 \text{ as } t \rightarrow \infty$$
This indicates the model has sufficient capacity to fit the training data perfectly.

**Validation Loss Behavior:**
$$\text{Val Loss}(t) = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$
After epoch 7, the variance term dominates, causing validation loss to increase.

**Why Options D and E Are Correct:**

**Option D: "The model might be too complex for the dataset"**
- **Evidence:** Training loss reaches near-zero quickly
- **Implication:** Model has more parameters than needed
- **Result:** Model memorizes training data instead of learning generalizable patterns

**Option E: "The model is likely memorizing the training data"**
- **Evidence:** Training loss ≈ 0, validation loss increasing
- **Implication:** Model learns training-specific noise
- **Result:** Poor generalization to unseen data

**Why Other Options Are Incorrect:**

**Option A: "The model has high bias and low variance"**
- **Contradiction:** High bias would show high training loss
- **Reality:** Low training loss indicates low bias

**Option B: "The large gap indicates underfitting"**
- **Contradiction:** Underfitting shows high training and validation loss
- **Reality:** Low training loss indicates overfitting, not underfitting

**Option C: "Training for more epochs will eventually decrease validation loss"**
- **Contradiction:** Validation loss is already increasing
- **Reality:** More training will likely worsen overfitting

**Solutions to Overfitting:**

1. **Regularization:** Add L1/L2 penalties to reduce model complexity
2. **Early Stopping:** Stop training when validation loss starts increasing
3. **Data Augmentation:** Increase effective dataset size
4. **Model Simplification:** Reduce number of parameters
5. **Dropout:** Randomly disable neurons during training

**Early Stopping Implementation:**
```python
# Monitor validation loss and stop when it increases
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= patience:
        stop_training()
```

**Visual Indicators of Overfitting:**
- Training loss continues to decrease
- Validation loss starts increasing
- Growing gap between training and validation curves
- Model performance on test set is poor

## Problem 5: Maximum Likelihood Estimation

**1 points Select All That Apply**

**Question:** Which of the following models that we studied in class use maximum likelihood estimation?

**Options:**
- a) $\text{\textcircled{O}}$ Linear regression with Gaussian noise model
- b) $\bigcirc$ Principal Components Analysis
- c) $\text{\textcircled{O}}$ Gaussian Mixture Models
- d) $\text{\textcircled{O}}$ Neural Network trained to do classification with softmax cross entropy loss

**Correct answers:** (a), (c), (d)

**Explanation:** 
- **a) True:** Linear regression with Gaussian noise model is true because you maximize the likelihood of the data under a linear model which assumes a Gaussian distribution on errors.
- **c) True:** Gaussian Mixture Models define a probability distribution that is a mixture of Gaussians and then find the parameters by maximizing likelihood under that model.
- **d) True:** NNs with softmax define a probability distribution over the classification labels and try to maximize it with cross entropy.
- **b) False:** PCA does not use MLE because it does not define a probabilistic distribution for the data, it just uses linear algebra to find vectors that explain a lot of variance in the data.

## Detailed Solution Explanation

**Understanding Maximum Likelihood Estimation (MLE):**

Maximum Likelihood Estimation is a method for estimating parameters of a statistical model by finding the parameter values that maximize the likelihood function.

**Mathematical Framework:**
For a dataset $\mathcal{D} = \{x_1, x_2, \ldots, x_n\}$ and parameters $\theta$:

$$\mathcal{L}(\theta) = P(\mathcal{D} | \theta) = \prod_{i=1}^{n} P(x_i | \theta)$$

The MLE estimate is:
$$\hat{\theta}_{MLE} = \arg\max_{\theta} \mathcal{L}(\theta) = \arg\max_{\theta} \log \mathcal{L}(\theta)$$

**Analysis of Each Model:**

**Option A: Linear Regression with Gaussian Noise**

**Model:** $y_i = w^T x_i + \epsilon_i$ where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$

**Likelihood Function:**
$$\mathcal{L}(w, \sigma^2) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - w^T x_i)^2}{2\sigma^2}\right)$$

**Log-Likelihood:**
$$\log \mathcal{L}(w, \sigma^2) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - w^T x_i)^2$$

**MLE Solution:** Maximizing log-likelihood is equivalent to minimizing MSE:
$$\hat{w}_{MLE} = \arg\min_w \sum_{i=1}^{n} (y_i - w^T x_i)^2$$

**Option C: Gaussian Mixture Models (GMM)**

**Model:** $P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$

**Likelihood Function:**
$$\mathcal{L}(\{\pi_k, \mu_k, \Sigma_k\}) = \prod_{i=1}^{n} \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i | \mu_k, \Sigma_k)$$

**MLE Solution:** Typically solved using EM algorithm to find:
$$\{\hat{\pi}_k, \hat{\mu}_k, \hat{\Sigma}_k\}_{MLE} = \arg\max \mathcal{L}(\{\pi_k, \mu_k, \Sigma_k\})$$

**Option D: Neural Networks with Softmax Cross-Entropy**

**Model:** $P(y_i | x_i) = \text{softmax}(f_\theta(x_i))_y$

**Likelihood Function:**
$$\mathcal{L}(\theta) = \prod_{i=1}^{n} P(y_i | x_i, \theta)$$

**Cross-Entropy Loss:** Negative log-likelihood:
$$\mathcal{L}_{CE} = -\sum_{i=1}^{n} \log P(y_i | x_i, \theta)$$

**MLE Solution:** Minimizing cross-entropy maximizes likelihood:
$$\hat{\theta}_{MLE} = \arg\min_\theta \mathcal{L}_{CE}$$

**Option B: Principal Components Analysis (PCA)**

**Why PCA is NOT MLE:**
- **No Probabilistic Model:** PCA doesn't assume any probability distribution
- **Geometric Approach:** Finds directions of maximum variance
- **Optimization Objective:** Maximizes variance, not likelihood
- **Mathematical Formulation:**
  $$w^* = \arg\max_w \text{Var}(w^T X) = \arg\max_w w^T \Sigma w$$
  subject to $||w|| = 1$

**Key Differences:**

| Method | Probabilistic Model | Optimization Objective | Uses MLE |
|--------|-------------------|----------------------|----------|
| Linear Regression | Yes (Gaussian noise) | Minimize MSE | Yes |
| GMM | Yes (Mixture of Gaussians) | Maximize likelihood | Yes |
| Neural Networks | Yes (Categorical) | Minimize cross-entropy | Yes |
| PCA | No | Maximize variance | No |

**Practical Implications:**
- **MLE Methods:** Provide uncertainty estimates, can be extended to Bayesian inference
- **Non-MLE Methods:** Often faster, but lack probabilistic interpretation

## Problem 6: Maximum Likelihood Estimation - Coin Toss

**1 points One Answer**

**Question:** Yann, a strict frequentist statistician, observes 5 flips of a possibly uneven coin. Here are the outcomes: 1. Heads, 2. Tails, 3. Heads, 4. Heads, 5. Tails. Based on these observations, Yann uses using maximum likelihood estimation to determine the most likely outcome of the next coin toss. What does he predict will happen?

**Options:**
- a) Heads
- b) Tails
- c) Both are equally likely
- d) It hits Marco in the head

**Correct answers:** (a)

**Explanation:** 
There were 3 Heads and 2 Tails. Based on these observations, the estimated probability of Heads is $\frac{3}{5} = 0.6$, which is greater than the estimated probability of Tails ($\frac{2}{5} = 0.4$). Therefore, Heads is the most likely outcome.

## Detailed Solution Explanation

**Understanding Maximum Likelihood Estimation for Bernoulli Trials:**

This problem demonstrates how MLE works for estimating the probability parameter of a Bernoulli distribution (coin toss).

**Mathematical Framework:**

**Bernoulli Distribution:**
For a coin with probability $p$ of heads, the probability mass function is:
$$P(X = x) = p^x(1-p)^{1-x}$$
where $x \in \{0, 1\}$ (0 = tails, 1 = heads)

**Likelihood Function:**
For $n$ independent coin tosses with outcomes $x_1, x_2, \ldots, x_n$:
$$\mathcal{L}(p) = \prod_{i=1}^{n} p^{x_i}(1-p)^{1-x_i} = p^{\sum_{i=1}^{n} x_i}(1-p)^{n - \sum_{i=1}^{n} x_i}$$

**Log-Likelihood:**
$$\log \mathcal{L}(p) = \left(\sum_{i=1}^{n} x_i\right) \log p + \left(n - \sum_{i=1}^{n} x_i\right) \log(1-p)$$

**MLE Solution:**
To find the MLE, we set the derivative to zero:
$$\frac{d}{dp} \log \mathcal{L}(p) = \frac{\sum_{i=1}^{n} x_i}{p} - \frac{n - \sum_{i=1}^{n} x_i}{1-p} = 0$$

Solving for $p$:
$$\frac{\sum_{i=1}^{n} x_i}{p} = \frac{n - \sum_{i=1}^{n} x_i}{1-p}$$
$$(1-p)\sum_{i=1}^{n} x_i = p(n - \sum_{i=1}^{n} x_i)$$
$$\sum_{i=1}^{n} x_i - p\sum_{i=1}^{n} x_i = pn - p\sum_{i=1}^{n} x_i$$
$$\sum_{i=1}^{n} x_i = pn$$
$$\hat{p}_{MLE} = \frac{\sum_{i=1}^{n} x_i}{n} = \frac{\text{Number of heads}}{\text{Total tosses}}$$

**Application to the Problem:**

**Data:** 5 coin tosses with outcomes: H, T, H, H, T

**Counts:**
- Number of heads: $\sum_{i=1}^{5} x_i = 3$
- Number of tails: $5 - 3 = 2$
- Total tosses: $n = 5$

**MLE Estimate:**
$$\hat{p}_{MLE} = \frac{3}{5} = 0.6$$

**Prediction:**
Since $\hat{p}_{MLE} = 0.6 > 0.5$, the most likely outcome for the next toss is **Heads**.

**Verification:**
- $P(\text{Heads}) = 0.6$
- $P(\text{Tails}) = 1 - 0.6 = 0.4$
- $0.6 > 0.4$, so Heads is more likely

**Properties of MLE for Bernoulli:**

1. **Unbiased:** $E[\hat{p}_{MLE}] = p$ (for large $n$)
2. **Consistent:** $\hat{p}_{MLE} \rightarrow p$ as $n \rightarrow \infty$
3. **Efficient:** Achieves the Cramér-Rao lower bound
4. **Asymptotically Normal:** $\hat{p}_{MLE} \sim \mathcal{N}(p, \frac{p(1-p)}{n})$

**Confidence Interval:**
For large $n$, a 95% confidence interval is:
$$\hat{p}_{MLE} \pm 1.96 \sqrt{\frac{\hat{p}_{MLE}(1-\hat{p}_{MLE})}{n}}$$

**Example:** For our estimate $\hat{p} = 0.6$ with $n = 5$:
$$0.6 \pm 1.96 \sqrt{\frac{0.6 \times 0.4}{5}} = 0.6 \pm 0.43 = [0.17, 1.03]$$

Note: This interval is wide due to small sample size and extends beyond [0,1], indicating the normal approximation is poor for small $n$.

**Key Insights:**
- MLE provides the most likely parameter value given the data
- For Bernoulli trials, MLE is simply the sample proportion
- The prediction is based on the estimated probability being greater than 0.5
- Small sample sizes lead to uncertain estimates

## Problem 7: Convex Optimization

**Problem Description:** Consider a function $f: \mathbb{R}^d \to \mathbb{R}$ that is differentiable everywhere. Suppose that $f(y) \ge f(x) + \nabla f(x)^T (y-x)$ for all $x, y \in \mathbb{R}^d$. Also, suppose that there exists a unique $x_* \in \mathbb{R}^d$ such that $\nabla_x f(x_*) = 0$.

### Part (a)

**1 points One Answer**

**Question:** $x_*$ is a:

**Options:**
- a) Minimizer of $f$
- b) Maximizer of $f$
- c) Saddle point of $f$
- d) Not enough information to determine any of the above

**Correct answers:** (a)

### Part (b)

**1 points**

**Question:** Suppose we are unable to solve for $x_*$ in closed-form. Briefly outline a procedure for finding $x_*$.

**Answer:** Gradient descent

**Explanation:** 
- **Part (a):** $f$ is convex, so $x_*$ must be a minimizer of $f$.
- **Part (b):** Gradient descent

## Problem 8: Gradient Descent Convergence

**1 points One Answer**

**Question:** Which of the following is true, given the optimal learning rate?

**Clarification:** All options refer to convex loss functions that have a minimum bound / have a minimum value.

**Options:**
- a) For convex loss functions, stochastic gradient descent is guaranteed to eventually converge to the global optimum while gradient descent is not.
- b) For convex loss functions, both stochastic gradient descent and gradient descent will eventually converge to the global optimum.
- c) Stochastic gradient descent is always guaranteed to converge to the global optimum of a loss function.
- d) For convex loss functions, gradient descent with the optimal learning rate is guaranteed to eventually converge to the global optimum point while stochastic gradient descent is not.

**Correct answers:** (d)

**Explanation:** 
Due to the noisy updates of SGD, it is not guaranteed to converge at the minimum but for instance, cycle close to it whereas batch gradient descent alleviates this and is guaranteed to reach the minimum given appropriate step size.

## Problem 9: Stochastic Gradient Descent

**3 points**

**Problem Description:** Imagine you are trying to find an optimal weight $w$ for a simple model. You have a small dataset consisting of two data points, each influencing the overall loss:
* Data point 1: $(x_1, y_1) = (5,4)$
* Data point 2: $(x_2, y_2) = (1,3)$

You are using a squared error loss function for each individual data point, defined as
$$L_i(w) = (y_i - w \cdot x_i)^2$$

Your current weight parameter is $w_0 = 1$. You will perform one iteration of Stochastic Gradient Descent (SGD) using a learning rate $\alpha = 0.1$. You will process one "randomly" chosen data point to compute the gradient and update the weight. For this exercise, you may choose which data point to process.

### Case 1: Student Selects Data Point 1 ($x_1 = 5, y_1 = 4$)

**a) Selected Data Point:** Data point 1

**b) Loss at $w_0 = 1$:**
$L_1(1) = (y_1 - w_0 x_1)^2 = (4 - 1 \cdot 5)^2 = (4 - 5)^2 = (-1)^2 = 1$

**c) Gradient at $w_0 = 1$:**
$\nabla L_1(1) = -2x_1(y_1 - w_0 x_1) = -2(5)(4 - 1 \cdot 5) = -10(4 - 5) = -10(-1) = 10$

**d) Weight $w_1$ after SGD update:**
$w_1 = w_0 - \alpha \nabla L_1(w_0) = 1 - 0.1(10) = 1 - 1 = 0$

### Case 2: Student Selects Data Point 2 ($x_2 = 1, y_2 = 3$)

**a) Selected Data Point:** Data point 2

**b) Loss at $w_0 = 1$:**
$L_2(1) = (y_2 - w_0 x_2)^2 = (3 - 1 \cdot 1)^2 = (3 - 1)^2 = (2)^2 = 4$

**c) Gradient at $w_0 = 1$:**
$\nabla L_2(1) = -2x_2(y_2 - w_0 x_2) = -2(1)(3 - 1 \cdot 1) = -2(3 - 1) = -2(2) = -4$

**d) Weight $w_1$ after SGD update:**
$w_1 = w_0 - \alpha \cdot \nabla L_2(w_0) = 1 - 0.1(-4) = 1 + 0.4 = 1.4$

**Explanation:** 
We are given:
* Data point 1: $(x_1,y_1) = (5,4)$
* Data point 2: $(x_2,y_2) = (1,3)$
* Loss function: $L_i(w) = (y_i - w \cdot x_i)^2$
* Initial weight: $w_0 = 1$
* Learning rate: $\alpha = 0.1$

**General Formulas:**
* The loss function for a selected data point $(x_i, y_i)$ is $L_i(w) = (y_i - w \cdot x_i)^2$.
* The gradient of the loss with respect to $w$ is:
  $$\nabla L_i(w) = \frac{d}{dw}(y_i - w \cdot x_i)^2 = 2(y_i - w \cdot x_i)(-x_i) = -2x_i(y_i - w \cdot x_i)$$
* The SGD update rule is:
  $$w_{new} = w_{old} - \alpha \cdot \nabla L_i(w_{old})$$

## Problem 10: Activation Functions

**1 points One Answer**

**Question:** Which of the following activation functions saturates, i.e. stops giving meaningful gradients for large positive inputs?

**Options:**
- a) ReLU
- b) Sigmoid
- c) Softmax

**Correct answers:** (b)

**Explanation:** 
- The gradient for Sigmoid and Tanh approaches 0 as the magnitude of the input increases.
- Softmax is not an activation function.

## Problem 11: Matrix Operations (Convolution and Max Pooling)

**2 points**

**Question:** Consider the following matrix $M$ and kernel filter $F$.

$$
M = \begin{pmatrix}
9 & 7 & 8 \\
4 & 1 & 3 \\
2 & 6 & 4
\end{pmatrix}
\quad
F = \begin{pmatrix}
1 & 0 \\
1 & 1
\end{pmatrix}
$$

Apply the filter $F$ to matrix $M$ with padding = 0 and stride = 1, then perform a Max Pooling operation on the result with a 2x2 filter and stride 1. Write the resulting matrix below in the grid of the correct size. Only write answers in one matrix, otherwise the problem will be graded as incorrect.

**Answer Grid:**
(The image shows four empty grids of sizes 1x1, 2x2, 3x3, and a partially visible 4x4, indicating where the user should write their answer.)

**Explanation:** 
After applying $F$ to $M$, we get: $\begin{pmatrix} 14 & 11 \\ 12 & 11 \end{pmatrix}$. Applying a Max Pool operation with a 2x2 filter just means taking the max of this matrix, since it's a 2x2, so we get the final answer of 14.

## Problem 12: Spatial Dimensions of Output Image

**2 points**

**Question:** What are the spatial dimensions of the output image if a 2 x 2 filter is convolved with a 3 x 3 image for paddings of 0, 1, and 2, and strides of 1 and 2? Fill in the dimensions below:

**Table:**

| Padding | 0 | 1 | 2 |
|---------|---|---|---|
| Stride 1 | (2×2) | (4×4) | (6×6) |
| Stride 2 | (1×1) | (2×2) | (3×3) |

**Explanation:** 
- **Stride 1:** Padding 0 (2×2), Padding 1 (4×4), Padding 2 (6×6)
- **Stride 2:** Padding 0 (1×1), Padding 1 (2×2), Padding 2 (3×3)

## Problem 13: Ridge vs. Lasso Regression

**1 points One Answer**

**Question:** Compared to Lasso, Ridge regression tends to be more stable in terms of which features are important to the model's predictions in high-dimensional cases because it doesn't drive correlated weights to 0.

**Clarification:** Clarification made during exam: "Should read as 'More stable in terms of which features are important to the model's predictions as you increase the amount of regularization in high-dimensional...'"

**Options:**
- a) True
- b) False

**Correct answers:** (a)

**Explanation:** 
This is true because Ridge "smoothly shrinks" all weights making it more stable to small changes in the data or noise.

## Problem 14: Logistic Regression with Linearly Separable Data

**1 points One Answer**

**Question:** For $X \in \mathbb{R}^{n \times d}$ and $y \in \{-1, 1\}^n$, if our data is linearly separable then the minimization problem $\arg \min_w \sum_{i=1}^n \log(1 + \exp(-y_i w^T x_i))$ does not have a unique solution.

**Options:**
- a) True
- b) False

**Correct answers:** (a)

**Explanation:** 
If our data is linearly separable we can push the magnitude of $w$ to $\infty$ to push the objective to 0 but never actually reach 0, so there is no solution.

## Problem 15: Singular Value Decomposition

**1 points One Answer**

**Question:** Suppose we have a matrix $M \in \mathbb{R}^{n \times m}$ and perform SVD on it to get 3 matrices $U, S, V$. If we take the first $r$ singular vectors of $U, V$ corresponding to the first $r$ singular values in $S$ (ordered highest to lowest), where $r = \min(n, m)$, then we can perfectly reconstruct $M$ without any loss whatsoever.

**Options:**
- a) True
- b) False

**Correct answers:** (a)

**Explanation:** 
$r = \min(n, m) \ge \text{rank}(M)$. If we perform a rank $r$ reconstruction on a matrix whose maximum rank is $r$, we will get a lossless reconstruction.

## Problem 16: Principal Components

**1 points Select All That Apply**

**Question:** Which of the following are equivalent to the principal components of a data matrix $X$? Assume $X$ has already been de-meaned.

**Options:**
- a) Vectors that create a subspace which maximize the variance of $X$ if $X$ is projected onto that subspace.
- b) Vectors that create a subspace which minimize the variance of $X$ if $X$ is projected onto that subspace.
- c) The eigenvectors of $X^T X$.
- d) The right singular vectors of $X$.

**Correct answers:** (a), (c), (d)

**Explanation:** 
- **A is correct** because this is the definition of principal components.
- **B is the opposite**, so it is false.
- The right singular vectors of $X$ are also the eigenvectors of $X^T X$, and both are equal to the principal components of $X$. Therefore, **C and D are correct**.

## Problem 17: PCA Reconstruction Error

**1 points One Answer**

**Question:** In PCA, minimizing the reconstruction error is equivalent to minimizing the projected variance.

**Options:**
- a) True
- b) False

**Correct answers:** (b)

**Explanation:** 
Minimizing the reconstruction error is equivalent to maximizing the variance.

## Problem 18: PCA Component Selection

**1 points Select All That Apply**

**Question:** You apply PCA on a dataset of 100 features and get 100 principal components. Which of the following are good reasons to choose only the top $q$ principal components instead of all 100? Assume $q < 100$.

**Options:**
- a) To remove noise by discarding the highest variance components.
- b) To reduce redundant features in the dataset.
- c) To reduce the computational cost of working with the data.
- d) To make a beautiful plot.

**Correct answers:** (c), (d)

**Explanation:** 
If we chose the top $q$ components, those would be the ones with highest variance, so A is incorrect. B is incorrect as we have 100 features and 100 PCs in this case, so we are not reducing redundant features here; all of them are meaningful features. C is correct, because by only picking the top $q$ PCs, we are reducing the dimensionality of the dataset and thus reducing computational cost. D is correct, as it has been mentioned numerous times in lecture before: it helps us create a beautiful plot.

## Problem 19: Decision Trees Bias-Variance

**1 points One Answer**

**Question:** Generally, decision trees have:

**Clarification:** Clarification made during exam: "It should be 'decision trees' instead of 'tree-based methods.'"

**Options:**
- a) Low bias, low variance
- b) Low bias, high variance
- c) High bias, low variance
- d) High bias, high variance

**Correct answers:** (b)

**Explanation:** 
Tree-based methods usually have low bias and high variance.

## Problem 20: Decision Tree Overfitting

**1 points Select All That Apply**

**Question:** Forrest just trained a decision tree for predicting whether a person will like a song based on features like its genre, key, length, etc. He notices an extremely low training error, but an abnormally large test error. He also notices that a regularized multi-class logistic regression model performs much better than his tree. What could be the cause of his problem?

**Options:**
- a) Learning rate too high
- b) Decision tree is too deep
- c) There is too much training data
- d) Decision tree is overfitting

**Correct answers:** (b), (d)

**Explanation:** 
He is observing overfitting which could be caused by a complex/deep tree.

## Problem 21: Model Selection Matching

**2.5 points**

**Question:** Match each modeling problem with the best machine learning method from the list below. Use each model type once.

**Modeling Problems:**

**A)** Training a model for a medical setting with a small number of categorical input features, where interpretability of decisions is important.

**B)** Having a small dataset (small $n$) with continuous $Y$ labels and many features. The goal is an interpretable model that can be regularized to identify important features.

**C)** Having a large dataset (large $n$) of images.

**D)** Having a lot of data (large $n$) in a small dimensional feature space (small $d$), assuming labels $y$ change smoothly with changes in the feature space.

**E)** Data with a relatively small number of categorical features, with the goal of winning a Kaggle competition.

<img src="./img/q21_problem.png" width="500px">

**Available Machine Learning Methods:**
- k-Nearest Neighbours (kNN)
- Decision Tree (DT)
- Random Forest (RF)
- Convolutional Neural Network (CNN)
- Linear Regression (LR)

**Matching Table:**

| Problem | Machine Learning Method |
|---------|------------------------|
| A | $\text{\textcircled{O}}$ DT (Decision Tree) |
| B | $\text{\textcircled{O}}$ LR (Linear Regression) |
| C | $\text{\textcircled{O}}$ CNN (Convolutional Neural Network) |
| D | $\text{\textcircled{O}}$ kNN (k-Nearest Neighbours) |
| E | $\text{\textcircled{O}}$ RF (Random Forest) |

**Explanation:** 
- **Problem A:** Decision Tree, because they are good for categorical features and are interpretable.
- **Problem B:** Linear Regression, because it works for small datasets and continuous labels.
- **Problem C:** Convolutional Neural Networks.
- **Problem D:** kNN.
- **Problem E:** Random Forests.

## Problem 22: Entropy in Decision Trees

**1 points One Answer**

**Question:** You are training a decision tree to perform classification into labels $Y \in \{0,1\}$. Your tree sorts the labels into the following leaves. What is the entropy $H(X)$ for each of the following sets $X$:

- a) $X = 1, 1, 1, 1:$
- b) $X = 1, 1, 0, 0:$
- c) $X = 0, 0, 0, 0:$

**Explanation:** The entropy $H(X)$ is calculated using the formula $H(X) = - \sum_i p(i) \log_2 p(i)$.

**Answers:**
- a) 0.0 ($H = -(1 \cdot \log_2(1)) = 0$)
- b) 1.0 ($H = -(0.5 \cdot \log_2(0.5) + 0.5 \cdot \log_2(0.5)) = -(0.5 \cdot (-1) + 0.5 \cdot (-1)) = -(-0.5 - 0.5) = -(-1) = 1$)
- c) 0.0 ($H = -(1 \cdot \log_2(1)) = 0$)

## Problem 23: Kernel Method

**1 points Select All That Apply**

**Question:** You are applying the kernel method to $n$ data points, where each data point $x_i \in \mathbb{R}^d$. Which of the following statements are true.

**Options:**
- a) The kernel method performs computations on a high dimensional feature space $\phi(x_i) \in \mathbb{R}^p$, where $p \gg d$.
- b) A function $K$ is a kernel for a feature map $\phi$ if $K(x, x') = \phi(x)^T \phi(x')$.
- c) The kernel trick relies on the fact if $p \gg n$, then the data spans at most a $d$-dimensional subspace of $\mathbb{R}^p$.
- d) Kernel methods can be considered non-parametric because they require retaining the training data for making predictions about new points.

**Correct answers:** (b), (d)

**Explanation:** 
- **a) is not correct** because the kernel method *avoids* actually performing computations in the $p$-dimensional feature space. Instead, it computes the dot product in the original space.
- **b) is correct**, as this is the definition of a kernel function for a feature map $\phi$.
- **c) is incorrect**. The kernel trick relies on the fact that the data spans at most an $n$-dimensional subspace of $\mathbb{R}^p$, not a $d$-dimensional subspace.
- **d) is correct**. Kernel methods, such as Support Vector Machines (SVMs) with non-linear kernels, are often considered non-parametric because their model complexity grows with the number of training data points, and they typically require retaining (or at least referencing) a subset of the training data (e.g., support vectors) to make predictions on new points.

## Problem 24: Kernel Matrix and Diagonal Entry

**Problem Statement:** Consider data matrix $X \in \mathbb{R}^{n \times d}$ and feature mapping $\phi: \mathbb{R}^d \to \mathbb{R}^p$, for some $p$. Let $K$ be the corresponding kernel matrix.

### Part (a)

**1 points**

**Question:** Let $\phi(X)$ denote $X$ with $\phi$ applied to each data point. Write $K$ in terms of $\phi(X)$.

**Answer:** $K = \phi(X)\phi(X)^T$

### Part (b)

**1 points One Answer**

**Question:** The $i^{th}$ entry on the diagonal of $K$ is:

**Options:**
- a) $\bigcirc$ $||\phi(x_i)||_1$
- b) $\bigcirc$ $||\phi(x_i)||_2$
- c) $\text{\textcircled{O}}$ $||\phi(x_i)||_2^2$
- d) $\bigcirc$ None of the above

**Correct answers:** (c)

**Explanation:** 
- **Part (a):** $K = \phi(X)\phi(X)^T$
- **Part (b):** $K_{ii} = \phi(x_i)^T \phi(x_i) = ||\phi(x_i)||_2^2$

## Problem 25: Curse of Dimensionality

**1 points Select All That Apply**

**Question:** Natasha is trying to train a k-Nearest Neighbors model, and she encounters the "curse of dimensionality". This refers to the fact that as the dimensionality of her feature space $d$ increases...

**Options:**
- a) $\text{\textcircled{O}}$ Distances between points become less meaningful, since all points are far apart.
- b) $\bigcirc$ She has too much data making computation too expensive to perform on a single machine.
- c) $\text{\textcircled{O}}$ The amount of data required to cover the space increases exponentially.
- d) $\bigcirc$ Thinking in more than three dimensions is hard so we should use PCA to make a 2D plot.

**Correct answers:** (a), (c)

**Explanation:** 
a-c are all correct statements of the same idea. d is a joke.

## Problem 26: Clustering Algorithms

**1 points Select All That Apply**

**Question:** You want to cluster this data into 2 clusters. Which of the these algorithms would work well?

*(Image description: A scatter plot of data points forming a distinct 'plus' or 'cross' shape, with four arms extending outwards from a central, denser region. The arms are somewhat spread out, and the central region shows an overlap of points from different arms.)*

**Options:**
- a) Spectral clustering
- b) K-means
- c) $\text{\textcircled{O}}$ GMM clustering

**Correct answers:** (c)

**Explanation:** 
- Only GMM takes the Gaussian distributions of the two clusters into account even when they overlap.

## Problem 27: K-means Clustering Properties

**1 points One Answer**

**Question:** Which of the following statements is true about K-means clustering?

**Options:**
- a) K-means clustering works effectively in all data distributions.
- b) $\text{\textcircled{O}}$ K-means is guaranteed to converge.
- c) K-means clustering is a supervised learning algorithm.
- d) The accuracy of K-means clustering is not affected by the initial centroid selections.

**Correct answers:** (b)

**Explanation:** 
- **A is false** since K-means doesn't work well in all distributions, such as non-spherical clusters.
- **B is true**, since K-means will always converge (see lecture notes for proof).
- **C is false**, since K-means is unsupervised.
- **D is false**, since the accuracy of the classifier is influenced by the initial centroid selections.

## Problem 28: Gaussian Mixture Model Parameters

**1 points One Answer**

**Question:** Suppose a Gaussian Mixture Model (GMM) with $k$ components/clusters is used to model a dataset of dimensionality $d$. Which value does the total number of parameters in the GMM primarily scale with respect to?

**Options:**
- a) $O(k \cdot d)$
- b) $O(k \cdot d^2)$
- c) $O(d)$
- d) $O(d^2)$
- e) $O(k)$
- f) $O(n)$
- g) $O(\frac{d}{n})$

**Correct answers:** (b)

**Explanation:** 
The parameters of a GMM are the mixture weights, the means, and the covariance matrices. There are $k$ mixing weights, each $\in \mathbb{R}$. There are $k$ means, each $\in \mathbb{R}^d$. There are $k$ covariance matrices, each $\in \mathbb{R}^{d \times d}$. Since the covariance matrices have the most parameters, the $k$ covariance matrices are the "determining factor". So the answer is $O(k \cdot d^2)$.

## Problem 29: Bootstrap Sampling

**1 points One Answer**

**Question:** Because bootstrap sampling randomly draws data points with replacement, the size of the original dataset does not affect accuracy of the estimated statistics produced by bootstrapping.

**Options:**
- a) True
- b) False

**Correct answers:** (b)

**Explanation:** 
Smaller datasets will not be as representative of the true dataset, yielding less accurate statistics.

## Problem 30: Fairness in Machine Learning

**1 points One Answer**

**Question:** A loan approval model performs worse and is more likely to reject underrepresented minorities due to bias in demographic information. What is the best way to address this bias?

**Options:**
- a) Remove demographic information.
- b) Over-sample underrepresented groups.
- c) Include fairness constraints to balance Type II error (probability of rejecting someone who deserved a loan) across groups.
- d) Collect more historical data for underrepresented groups and retrain the model.

**Correct answers:** (c)

**Explanation:** 
- **Option (a)** is not entirely helpful because demographic information is often correlated with other features.
- **Option (b)** is helpful for balancing data but not addressing the underlying bias issue.
- **Option (c)** is the current state-of-the-art approach.
- **Option (d)** does not necessarily work because historical data might still be biased.

## Problem 31: Feature Importance in Linear Regression

**1 points One Answer**

**Question:** A linear regression model has been trained, and for two features, $i$ and $j$, the weight $w_i$ is greater than $w_j$ ($w_i > w_j$). Can you conclude that feature $i$ is more important than feature $j$?

**Options:**
- a) True
- b) False

**Correct answers:** (b)

**Explanation:** 
This conclusion is false because features could have different scales (e.g., square feet vs. number of bathrooms), which affects the magnitude of their weights without necessarily indicating importance.

## Problem 32: Neural Network Derivatives

**4 points**

**Question:** Consider the following network:

<img src="./img/q32_problem.png" width="350px">

A diagram of a feedforward neural network is shown.
- **Input Layer:** Consists of a bias node (labeled '1', represented by a dotted circle) and three input nodes (labeled '$x_0$', '$x_1$', '$x_2$', represented by solid circles).
- **Hidden Layer:** Consists of a bias node (labeled '1', represented by a dotted circle) and four hidden nodes (all labeled '$z_0$', represented by solid circles).
- **Output Layer:** Consists of a single output node (labeled '$y$', represented by a solid circle).

**Connections:**
- Dotted lines labeled '$b_0$' connect the input layer bias node to all hidden layer nodes.
- Solid lines connect the input nodes ($x_0, x_1, x_2$) to all hidden layer nodes. These connections are associated with weights $W_0$.
- A dotted line labeled '$b_1$' connects the hidden layer bias node to the output node $y$.
- Solid lines connect all hidden layer nodes ($z_0$) to the output node $y$. These connections are associated with weights $W_1$.

**Network Equations:**
The forward pass for the hidden layer is $z = \sigma(W^{(0)}x+b^0)$, where $\sigma$ refers to the sigmoid activation function.
The output layer is $y= W^{(1)}z+b^1$.

**Task:**
Derive the partial derivatives with respect to $W^{(1)} \in \mathbb{R}^{1 \times h}$, $b^{(1)} \in \mathbb{R}$, $W^{(0)} \in \mathbb{R}^{h \times d}$, and $b^{(0)} \in \mathbb{R}^h$, where $d = 3$ and $h = 4$.

**Clarification made during exam:** "Typo: $b^0 = b_0 = b^{(0)}$. They all refer to the same thing."

**Derivatives to find:**
a) $\frac{\partial y}{\partial W^{(1)}}:$
b) $\frac{\partial y}{\partial b^{(1)}}:$
c) $\frac{\partial y}{\partial W^{(0)}}:$
d) $\frac{\partial y}{\partial b^{(0)}}:$

**Explanation:**
- **a) $\frac{\partial y}{\partial W^{(1)}} = z$**
- **b) $\frac{\partial y}{\partial b^{(1)}} = 1$**
- **c) $\frac{\partial y}{\partial W^{(0)}} = \left[W^{(1)T} \odot z \odot (1 - z)\right] x^T$**

This problem is very similar to the question from section 8. First, to make the math simpler, we can compute $\frac{\partial y}{\partial W^{(0)}_i}$, where $W^{(0)}_i$ is the $i$-th row of $W^{(0)}$. Computing the derivatives w.r.t. to $W^{(0)}$ necessitates chain rule; we can rewrite it as $\frac{\partial y}{\partial W^{(0)}_i} = \frac{\partial y}{\partial z_i} * \frac{\partial z_i}{\partial W^{(0)}_i}$. From here, the derivative of $z_i$ w.r.t. $W^{(0)}_i$ can be computed using the derivative of the sigmoid function ($\sigma * (1-\sigma)$). Doing so, we get $z_i * (1-z_i) * x^T$, where the $x^T$ comes from applying chain rule. Putting everything together, we get $\frac{\partial y}{\partial W^{(0)}_i} = W^{(1)}_i * z_i * (1-z_i) * x^T$. Note that this is a column vector, with the derivatives for a single row. To generalize this and get the derivative of $y$ w.r.t. to the entirety of $W^{(0)}$, we repeat the same process for all rows of $W^{(0)}$, which we can denote using the elementwise operator. Thus, we get $\left[W^{(1)T} \odot z \odot (1-z)\right] x^T$. Note we need to transpose $W^{(1)}$ in order multiply it elementwise with $z \odot (1-z)$.

- **d) $\frac{\partial y}{\partial b^{(0)}} = W^{(1)T} \odot z \odot (1 - z)$**

This derivation is very similar to the one above, except we don't have $x^T$ since only the weights matrix is multiplied with the data vector. So we get: $\frac{\partial y}{\partial b^{(0)}_i} = W^{(1)}_i * \frac{\partial z_i}{\partial b^{(0)}_i} = W^{(1)}_i * z_i * (1-z_i) \rightarrow W^{(1)T} \odot z \odot (1-z)$.

## Problem 33: Electric Car Adoption Prediction

**4 points**

**Problem Description:** Transitioning to electric cars can help fight climate change, but electric cars cause such a strain on the electrical grid that if several people on the same block all buy an electric car within a few weeks or months of each other, it can actually cause the grid to go down!

### Part 1: Feature Engineering/Data Preprocessing (2 points)

**Question:** You've been hired by the electric company to build a cool new machine learning model to help predict which houses will start charging electric cars next. You've been handed several messy files of data. The first contains high-level information about $n$ different houses, including whether they have an electric vehicle or not, each house's location, square footage, value, household income, results of the last election in the house's zipcode, public school ratings in the zip code, etc. But, you can also get detailed electricity data for each house, including daily electricity consumption going back at least 3 years. Describe the feature engineering or data preprocessing steps you would take to prepare to use this data to train a machine learning model.

**Answer:** [Student response area]

### Part 2: Machine Learning Model Description and Justification (2 points)

**Question:** Now, you must use the data you prepared to train a machine learning model that can tell you which houses are likely to get an electric car in the next year. Please describe the machine learning model you will use for this problem. You will be graded on how well you can justify why your model is a good choice for this problem, by explaining how the properties of your model suit the problem.

**Answer:** [Student response area]

**Explanation:** 
The criteria for grading this are do they find ways to mention real things about machine learning models they learned in class. Like 'I will use a random forest because it's good for categorical data but has lower variance than a tree'.

**Valid explanations include:**
- Feature engineering steps. Find some way to reduce the daily electricity data into something more manageable. Could use something like PCA, or manually extract features.
- Can mention separating into train, validation, and test.
- Normalizing features to be on the same scale could be mentioned.
- **Mega bonus points** if the discussion includes propagating features about neighbors' recent adoption of electric cars into the feature space for a house.

**Specific Machine Learning Models and Their Suitability:**

- **Neural Network:** Not requiring extensive feature engineering. Suggests throwing daily electricity data for each house into the features, which would result in a "huge $d$" (dimensionality). Mentions running gradient descent to see if it works.

- **Trees (Decision Trees/Random Forests):** Good for categorical data, providing examples like political affiliations and public school ratings. Recommends using a random forest to reduce variance. Notes that it "doesn't work with a ton of features (high $d$)", implying it should only be used in conjunction with feature engineering when dimensionality is high.

- **Logistic Regression:** Might be mentioned as a suitable choice for a classification problem. Highlights its interpretability as a potential reason for its use, allowing a power grid company to inspect the results.

- **kNN (k-Nearest Neighbors):** Could be a good pick if the feature space is reduced sufficiently. Suggests that it's best for figuring out car adoption based on whether similar people adopted a car, possibly in terms of "literal distance." Warns that it's "not a good answer if they use all the daily electricity data, because then the feature space would be too large."

- **CNN (Convolutional Neural Network):** Not a great answer because no images.

## Problem 34: Bonus Question

**4 points**

**Question:** This is a bonus question. You can get extra points for completing it, but you will not lose points if you do not get the right answer.

Let $f,g: \mathbb{R}^d \to \mathbb{R}$ be convex. Use the epigraph definition of convexity to prove that $h(x) = \max\{f(x), g(x)\}$ is convex.

**Hints:**
- **Hint 1:** You may use that for any convex sets $A, B \subset \mathbb{R}^d$, $A \cap B$ is convex.
- **Hint 2:** You may use that for any $a, b, c \in \mathbb{R}$, $c \ge a \wedge c \ge b$ if and only if $c \ge \max\{a,b\}$.

**Explanation:** 
**Proof:**

Denote $\text{epi}(f) := \{(x,t) \in \mathbb{R}^{d+1} : t \ge f(x)\}$, with $\text{epi}(g)$, $\text{epi}(h)$ defined similarly.

By the epigraph definition of convexity, we know that the sets $\text{epi}(f)$ and $\text{epi}(g)$ are convex.

Note that for any $(x,t) \in \mathbb{R}^{d+1}$ we have by hint 2 that $t \ge f(x)$ and $t \ge g(x)$ if and only if $t \ge \max\{f(x), g(x)\} = h(x)$.

Thus we have that $(x,t) \in \text{epi}(f) \cap \text{epi}(g)$ if and only if $(x,t) \in \text{epi}(h)$.

It follows that $\text{epi}(h) = \text{epi}(f) \cap \text{epi}(g)$.

Since $\text{epi}(f)$ and $\text{epi}(g)$ are convex, by hint 1, $\text{epi}(h)$ must be convex.

So by the epigraph definition of convexity, $h$ is convex. $\square$


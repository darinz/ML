# Problem Set 9 Solutions

## Problem 1: Irreducible Error

**1 point**

**Question:** For a given model, irreducible error can be decreased by improving the model's complexity and increasing the amount of training data.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

**Explanation:** 
You can't reduce irreducible error.

## Detailed Solution Explanation

**Understanding Irreducible Error:**

Irreducible error represents the fundamental uncertainty in the data that cannot be eliminated by any model, no matter how sophisticated or complex.

**Mathematical Framework:**

In the standard regression setting, we model the relationship as:
$$y = f(x) + \epsilon$$
where:
- $f(x)$ is the true underlying function
- $\epsilon$ is the irreducible error (noise)

The irreducible error $\epsilon$ is typically assumed to follow a distribution (often Gaussian) with mean 0 and some variance $\sigma^2$:
$$\epsilon \sim \mathcal{N}(0, \sigma^2)$$

**Why Irreducible Error Cannot Be Reduced:**

**Fundamental Nature:**
- **Randomness:** Irreducible error represents truly random noise in the system
- **Unpredictability:** This noise cannot be predicted from the features $x$
- **Inherent Uncertainty:** Even with perfect knowledge of $f(x)$, we cannot predict $\epsilon$

**Mathematical Proof:**
The expected prediction error can be decomposed as:
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

The irreducible error term $\sigma^2$ is independent of the model and cannot be reduced by:
- Increasing model complexity
- Adding more training data
- Improving the algorithm
- Using better features

**Examples of Irreducible Error:**

**Measurement Error:**
- Sensor noise in data collection
- Human error in labeling
- Instrument precision limitations

**System Noise:**
- Random fluctuations in biological systems
- Environmental factors affecting outcomes
- Unpredictable external influences

**Model Limitations:**
- Missing features that affect the outcome
- Features that are fundamentally unobservable
- Random processes in the system

**Why Other Errors Can Be Reduced:**

**Bias Error:**
- Can be reduced by using more complex models
- Can be reduced by using better algorithms
- Can be reduced by feature engineering

**Variance Error:**
- Can be reduced by using simpler models
- Can be reduced by regularization
- Can be reduced by increasing training data

**Practical Implications:**

**Model Selection:**
- No model can achieve perfect prediction
- The goal is to minimize bias and variance
- Irreducible error sets a lower bound on achievable performance

**Performance Expectations:**
- Understanding irreducible error helps set realistic goals
- Prevents over-optimization of models
- Guides resource allocation decisions

**Key Insights:**
- Irreducible error is a fundamental limitation, not a model deficiency
- It cannot be reduced by any modeling technique
- Understanding this concept is crucial for realistic model evaluation
- The goal is to minimize reducible errors (bias and variance)

## Problem 2: Neural Network Overfitting

**1 point**

**Question:** You're training a classifier model using a neural network built from scratch in PyTorch. You are unable to decide on the depth of the neural network, so you decide to make the network as deep as possible. Despite achieving low training loss, your model performs poorly on the XOR test data.

Why? Choose the most accurate explanation.

**Options:**
- a) The neural network is too complex and has too high of a bias squared error.
- b) The neural network is too complex and has too high of a variance error.
- c) The neural network is unable to learn non-linearities since XOR data is not linearly separable.
- d) We need to make the neural network even deeper to capture the complex relationship in the XOR dataset.

**Correct Answer:** b) The neural network is too complex and has too high of a variance error.

**Explanation:** 
The deep neural network is too complex and fits the training data too well (overfitting) resulting in a low bias squared error but fails to generalize as a result of high variance error.

## Detailed Solution Explanation

**Understanding Neural Network Overfitting:**

This problem demonstrates a classic case of overfitting in deep neural networks, where the model learns the training data too well but fails to generalize to new data.

**Mathematical Framework:**

**Bias-Variance Decomposition:**
The expected prediction error can be decomposed as:
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

where:
- **Bias:** How far off the model's predictions are on average from the true values
- **Variance:** How much the model's predictions vary for different training sets
- **Irreducible Error:** The fundamental noise in the data

**Analysis of the Scenario:**

**Given Conditions:**
- Very deep neural network (as deep as possible)
- Low training loss achieved
- Poor performance on XOR test data
- XOR data is not linearly separable

**Why Option B is Correct:**

**Overfitting Mechanism:**
1. **High Model Complexity:** Very deep networks have many parameters
2. **Low Training Loss:** Model can fit training data perfectly
3. **Poor Generalization:** Model fails on unseen data
4. **High Variance:** Model is sensitive to training data specifics

**Mathematical Interpretation:**

**Training Loss Behavior:**
$$\text{Training Loss} \rightarrow 0 \text{ as complexity } \rightarrow \infty$$

**Test Loss Behavior:**
$$\text{Test Loss} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

As complexity increases:
- **Bias decreases** (model can fit training data better)
- **Variance increases** (model becomes sensitive to training data)
- **Overall test error increases** due to high variance

**Why Other Options Are Incorrect:**

**Option A: "Too high bias squared error"**
- **Contradiction:** Low training loss indicates low bias
- **Reality:** The model fits training data well, so bias is low

**Option C: "Unable to learn non-linearities"**
- **Contradiction:** Deep networks can learn non-linearities
- **Reality:** The issue is overfitting, not inability to learn

**Option D: "Need even deeper network"**
- **Contradiction:** Making it deeper would worsen overfitting
- **Reality:** The network is already too complex

**Visual Representation:**

**Training vs Test Performance:**
```
Training Loss:  ████████████████████ (very low)
Test Loss:      ████████████████████████████████ (high)
                ↑
                High variance gap
```

**Bias-Variance Tradeoff:**
```
Complexity:     Low ←→ High
Bias:           High ←→ Low
Variance:       Low ←→ High
```

**Solutions to Overfitting:**

**1. Regularization:**
- **L1/L2 Regularization:** Add penalty terms to loss function
- **Dropout:** Randomly disable neurons during training
- **Early Stopping:** Stop training before overfitting occurs

**2. Model Simplification:**
- **Reduce Depth:** Use fewer layers
- **Reduce Width:** Use fewer neurons per layer
- **Feature Selection:** Use fewer input features

**3. Data Augmentation:**
- **Increase Training Data:** More data reduces overfitting
- **Data Augmentation:** Create synthetic training examples
- **Cross-Validation:** Better model selection

**4. Architecture Changes:**
- **Batch Normalization:** Stabilize training
- **Residual Connections:** Help with very deep networks
- **Attention Mechanisms:** Focus on relevant features

**Key Insights:**
- Overfitting occurs when model complexity exceeds data complexity
- Low training loss with high test loss indicates overfitting
- The solution is to reduce model complexity or increase data
- Understanding bias-variance tradeoff is crucial for model design
- Regularization techniques help balance this tradeoff

## Problem 3: Leave-One-Out Cross-Validation

**2 points**

**Question:** As dataset sizes increase, would you be more or less inclined to use leave-one-out cross-validation (LOOCV)? Provide reasoning to support your answer.

**Answer:** [Student response area]

**Explanation:** 
For larger datasets, leave-one-out cross validation becomes an extremely expensive process.

## Detailed Solution Explanation

**Understanding Leave-One-Out Cross-Validation Scalability:**

This problem explores the computational and practical limitations of leave-one-out cross-validation (LOOCV) as dataset sizes increase.

**Mathematical Framework:**

**Leave-One-Out Cross-Validation:**
For a dataset with $n$ samples, LOOCV:
- Divides data into $n$ folds
- Each fold has exactly 1 sample
- Trains on $n-1$ samples, validates on 1 sample
- Repeats $n$ times, using each sample as validation once

**Expected Test Error Estimate:**
$$\text{LOOCV Error} = \frac{1}{n} \sum_{i=1}^{n} \text{Error}_i$$
where $\text{Error}_i$ is the error on the $i$-th validation sample.

**Computational Complexity Analysis:**

**Time Complexity:**
- **Training Time per Model:** $T(n-1)$ where $T(m)$ is time to train on $m$ samples
- **Total Training Time:** $n \cdot T(n-1)$
- **Overall Complexity:** $O(n \cdot T(n-1))$

**Space Complexity:**
- **Storage per Model:** $S(n-1)$ where $S(m)$ is space for model trained on $m$ samples
- **Total Storage:** $n \cdot S(n-1)$
- **Overall Complexity:** $O(n \cdot S(n-1))$

**Why LOOCV Becomes Expensive for Large Datasets:**

**1. Linear Scaling with Dataset Size:**
- Number of models trained = $n$
- As $n$ increases, computational cost grows linearly
- For large $n$, this becomes prohibitive

**2. Training Time Growth:**
- Most algorithms have super-linear training time
- $T(n-1) = O((n-1)^p)$ where $p > 1$
- Total time becomes $O(n \cdot (n-1)^p) = O(n^{p+1})$

**3. Memory Requirements:**
- Need to store $n$ models simultaneously
- Memory grows linearly with dataset size
- May exceed available RAM for large datasets

**Practical Examples:**

**Small Dataset (n = 100):**
- Models to train: 100
- Training time: $100 \cdot T(99)$
- Feasible for most algorithms

**Medium Dataset (n = 1,000):**
- Models to train: 1,000
- Training time: $1,000 \cdot T(999)$
- May be slow but feasible

**Large Dataset (n = 100,000):**
- Models to train: 100,000
- Training time: $100,000 \cdot T(99,999)$
- Often computationally infeasible

**Comparison with K-Fold CV:**

**K-Fold Cross-Validation:**
- Models to train: $K$ (typically 5 or 10)
- Training time: $K \cdot T(\frac{K-1}{K} \cdot n)$
- Much more scalable

**Example Comparison:**
For $n = 10,000$:
- **LOOCV:** 10,000 models
- **5-Fold CV:** 5 models
- **10-Fold CV:** 10 models

**Alternative Approaches for Large Datasets:**

**1. K-Fold Cross-Validation:**
- Use $K = 5$ or $K = 10$
- Much faster than LOOCV
- Still provides good error estimates

**2. Stratified Sampling:**
- Use LOOCV on a representative subset
- Maintains statistical properties
- Reduces computational cost

**3. Monte Carlo Cross-Validation:**
- Randomly sample validation sets
- More flexible than fixed folds
- Can control computational cost

**4. Holdout Validation:**
- Single train/validation split
- Fastest approach
- Less reliable than cross-validation

**When to Use LOOCV:**

**Advantages:**
- **Unbiased Estimate:** Uses all data for training
- **Low Variance:** Many validation sets
- **No Randomness:** Deterministic results

**Disadvantages:**
- **High Computational Cost:** Scales linearly with $n$
- **High Variance:** Single sample validation sets
- **Memory Intensive:** Stores many models

**Optimal Use Cases:**
- **Small Datasets:** $n < 1000$
- **Expensive Data Collection:** When every sample is valuable
- **Unbiased Estimation Required:** When accuracy is critical

**Key Insights:**
- LOOCV computational cost grows linearly with dataset size
- For large datasets, K-fold CV is more practical
- The choice depends on dataset size and computational resources
- Understanding scalability is crucial for practical model selection
- Alternative approaches provide good approximations with lower cost

## Problem 4: K-Fold Cross-Validation Calculations

**1 point**

**Question:** You are fine-tuning a model with parameters $\alpha$, $\beta$, and $\gamma$, and have decided to perform 7-fold cross-validation to get the best set of hyperparameters. You have 5 candidate values for $\alpha$, 3 candidate values for $\beta$, and 2 candidate values for $\gamma$. How many validation errors will you be calculating in total?

**Options:**
- a) Cannot be determined.
- b) 10
- c) 96
- d) 210
- e) 30

**Correct Answer:** d) 210

**Explanation:** 
$5 \times 3 \times 2 \times 7 = 210$

## Detailed Solution Explanation

**Understanding K-Fold Cross-Validation Computational Requirements:**

This problem demonstrates how to calculate the total number of model evaluations needed for hyperparameter tuning using K-fold cross-validation.

**Mathematical Framework:**

**K-Fold Cross-Validation Process:**
For each hyperparameter combination:
1. Divide data into $K$ folds
2. Train $K$ models (each using $K-1$ folds for training)
3. Evaluate each model on the held-out fold
4. Average the $K$ validation errors

**Total Number of Model Evaluations:**
$$\text{Total Evaluations} = \text{Hyperparameter Combinations} \times K$$

**Step-by-Step Calculation:**

**Given Parameters:**
- **Hyperparameter $\alpha$:** 5 candidate values
- **Hyperparameter $\beta$:** 3 candidate values  
- **Hyperparameter $\gamma$:** 2 candidate values
- **Cross-validation folds:** $K = 7$

**1. Calculate Hyperparameter Combinations:**
$$\text{Combinations} = |\alpha| \times |\beta| \times |\gamma| = 5 \times 3 \times 2 = 30$$

**2. Calculate Total Model Evaluations:**
$$\text{Total Evaluations} = \text{Combinations} \times K = 30 \times 7 = 210$$

**Verification:**
- Each of the 30 hyperparameter combinations requires 7 model trainings
- Total: $30 \times 7 = 210$ model evaluations

**Visual Representation:**

**Hyperparameter Grid:**
```
α₁,β₁,γ₁  α₁,β₁,γ₂  α₁,β₂,γ₁  α₁,β₂,γ₂  α₁,β₃,γ₁  α₁,β₃,γ₂
α₂,β₁,γ₁  α₂,β₁,γ₂  α₂,β₂,γ₁  α₂,β₂,γ₂  α₂,β₃,γ₁  α₂,β₃,γ₂
α₃,β₁,γ₁  α₃,β₁,γ₂  α₃,β₂,γ₁  α₃,β₂,γ₂  α₃,β₃,γ₁  α₃,β₃,γ₂
α₄,β₁,γ₁  α₄,β₁,γ₂  α₄,β₂,γ₁  α₄,β₂,γ₂  α₄,β₃,γ₁  α₄,β₃,γ₂
α₅,β₁,γ₁  α₅,β₁,γ₂  α₅,β₂,γ₁  α₅,β₂,γ₂  α₅,β₃,γ₁  α₅,β₃,γ₂
```

**For Each Combination:**
```
Fold 1: Train on folds 2-7, validate on fold 1
Fold 2: Train on folds 1,3-7, validate on fold 2
Fold 3: Train on folds 1-2,4-7, validate on fold 3
Fold 4: Train on folds 1-3,5-7, validate on fold 4
Fold 5: Train on folds 1-4,6-7, validate on fold 5
Fold 6: Train on folds 1-5,7, validate on fold 6
Fold 7: Train on folds 1-6, validate on fold 7
```

**Computational Complexity Analysis:**

**Time Complexity:**
- **Per Model Training:** $T(n)$ where $n$ is dataset size
- **Per Combination:** $K \times T(n)$
- **Total Time:** $|\alpha| \times |\beta| \times |\gamma| \times K \times T(n)$

**Space Complexity:**
- **Per Model:** $S(n)$ where $S(n)$ is space for model
- **Total Space:** $K \times S(n)$ (need to store $K$ models simultaneously)

**Why Other Options Are Incorrect:**

**Option A: "Cannot be determined"**
- **Contradiction:** We have all necessary information
- **Reality:** The calculation is straightforward

**Option B: 10**
- **Contradiction:** This would be $|\alpha| + |\beta| + |\gamma| + K$
- **Reality:** We need the product, not the sum

**Option C: 96**
- **Contradiction:** This might be $|\alpha| \times |\beta| \times |\gamma| \times (K-1)$
- **Reality:** We need all $K$ folds for each combination

**Option E: 30**
- **Contradiction:** This is just the number of hyperparameter combinations
- **Reality:** We need to multiply by $K$ for cross-validation

**Practical Implications:**

**Computational Cost:**
- 210 model evaluations can be expensive
- May require parallel processing
- Could take hours or days depending on model complexity

**Optimization Strategies:**

**1. Grid Search Optimization:**
- Use fewer hyperparameter values
- Use smaller $K$ (e.g., 5 instead of 7)
- Use early stopping for poor combinations

**2. Random Search:**
- Sample random hyperparameter combinations
- Often more efficient than grid search
- Can find good solutions with fewer evaluations

**3. Bayesian Optimization:**
- Use probabilistic models to guide search
- More efficient than random search
- Requires fewer evaluations

**4. Parallel Processing:**
- Train multiple models simultaneously
- Reduces wall-clock time
- Requires sufficient computational resources

**Key Insights:**
- The total number of evaluations is the product of hyperparameter combinations and K
- This can become computationally expensive quickly
- Understanding this scaling is crucial for practical hyperparameter tuning
- Alternative search strategies can reduce computational cost
- The choice of K affects both accuracy and computational cost

## Problem 5: Maximum Likelihood Estimation - Exponential Distribution

**3 points**

**Question:** You are analyzing the time until failure for a set of lightbulbs. The data represents the number of months each bulb lasted before failing and is given as follows: $x_1, x_2, x_3, x_4$. Assuming these times are modeled as being drawn from an exponential distribution. Derive the maximum likelihood estimate (MLE) of the rate parameter $\lambda$ of this distribution. You must show your work.

**Recall:** The probability density function (PDF) for the exponential distribution is $f(x|\lambda) = \lambda e^{-\lambda x}$ for $x \ge 0$.

**Hint:** You should not have $n$ in your final answer.

**Answer:** $\lambda = \underline{\hspace{2cm}}$

**Explanation:** 
The answer is $\frac{4}{\sum_{i=1}^{4} x_i}$.

**Detailed Solution:**

First, we want to calculate the likelihood function $L(x_1,..., x_n|\lambda)$ below.

$L(x_1,..., x_n|\lambda) = P(x_1|\lambda) \cdot P(x_2|\lambda) \cdot ... \cdot P(x_n|\lambda) = \lambda e^{-\lambda x_1} \cdot \lambda e^{-\lambda x_2} \cdot ... \cdot \lambda e^{-\lambda x_n}$

$= \lambda^n \cdot e^{-\lambda(x_1+x_2+...+x_n)}$

Now, we calculate the log-likelihood function:

$lnL(x_1,..., x_n|\lambda) = ln(\lambda^n \cdot e^{-\lambda(x_1+x_2+...+x_n)}) = n \cdot ln(\lambda) - \lambda(x_1 + x_2 + ... + x_n)$

To find the argmax of $\lambda$ (and thus the MLE) of this log-likelihood expression, we need to take it's derivative with respect to $\lambda$ and set it equal to 0.

$\frac{d}{d\lambda}lnL(x_1,..., x_n|\lambda) = \frac{d}{d\lambda}(n \cdot ln(\lambda) - \lambda(x_1 + x_2 + ... + x_n)) = \frac{n}{\lambda} - (x_1+x_2+ ... + x_n) = 0$

$\implies \lambda = \frac{n}{x_1+x_2+...+x_n} = \frac{n}{\sum_{i=1}^{n} x_i}$

Thus, the MLE here of $\lambda$ is given by $\lambda = \frac{4}{\sum_{i=1}^{4} x_i}$.

## Problem 6: Convex Set Operations

**1 point**

**Question:** Which of the following can be convex?

**Options:**
- a) The intersection of non-convex sets
- b) The intersection of convex sets
- c) The union of non-convex sets
- d) The union of convex sets

**Correct Answer:** All of them (a, b, c, d)

**Explanation:** 
The answer is all of them.

- For the intersection of non-convex sets, the intersection of two five-pointed stars can be convex.
- For the intersection of convex sets, just consider two circles that are on top of each other.
- For the union of non-convex sets, just consider a circle that is split into two non-convex sets.
- For the intersection of convex sets, just consider two circles that are on top of each other.

## Problem 7: Gradient Descent Convergence

**1 point**

**Question:** For convex optimization objectives, taking a gradient step using full-batch GD ensures that your loss shrinks.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

**Explanation:** 
The answer is False.

Even for convex optimization objectives, if the learning rate is too high, there is a real probability of overshooting the global minima.

## Problem 8: Neural Network Activation Functions

**1 point**

**Question:** You are building a multi-class classifier using a deep neural network. You notice that your network is training slowly and that the gradients are diminishing quickly. Which activation function for the hidden layers of your network should you switch to, in order to avoid these issues?

**Options:**
- a) $f(x_i) = \frac{1}{1+e^{-x_i}}$
- b) $f(x_i) = \max(0, x_i)$
- c) $f(x_i) = x_i$
- d) $f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$

**Correct Answer:** b) $f(x_i) = \max(0, x_i)$

**Explanation:** 
Sigmoid ($f(x_i) = \frac{1}{1+e^{-x_i}}$) can cause vanishing gradients and hence can cause slow learning.
ReLU ($f(x_i) = \max(0, x_i)$) avoids saturation.
Having only linear layers reduces the network to a linear one.
Softmax ($f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}$) should be used in the output layer, but not the hidden layers of the network.

## Problem 9: Neural Network Depth and Training Loss

**1 point**

**Question:** If two neural networks differ only in the number of hidden layers, the deeper network will always achieve a lower training loss given the same training data.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

## Problem 10: Reducing Overfitting in Neural Networks

**1 point**

**Question:** Snoopy is training a neural network to classify birds into 'Woodstock' and 'Not Woodstock'. A plot of the training and validation accuracy for the neural network model during the training process is provided.

<img src="./img/q10_problem.png" width="350px">

**Figure 6: Snoopy's Training Plot**
- **Training Accuracy:** Starts around 70% at 2.5 epochs, steadily increases, and reaches approximately 100% accuracy by 15.0 epochs, remaining high thereafter.
- **Validation Accuracy:** Starts around 62% at 2.5 epochs, shows some fluctuations, increases to about 70% by 7.5 epochs, and then continues to slowly increase, reaching approximately 78% by 20.0 epochs.
- **Key Observation:** A significant and increasing gap is observed between the training accuracy and validation accuracy, particularly after about 7.5 epochs, where the training accuracy continues to rise sharply while the validation accuracy plateaus or increases very slowly. This indicates a clear case of overfitting.

Which of the following actions could Snoopy take to help reduce the difference between training and validation accuracy?

**Options:**
- a) Increase the amount of training data
- b) Apply regularization techniques
- c) Reduce the complexity of the model (e.g., use fewer layers or units)
- d) Train for more epochs without making other changes
- e) Decrease the learning rate

**Correct Answer:** a), b), c)

## Problem 11: Feature Selection (LASSO vs. PCA)

**1 point**

**Question:** Although both LASSO and PCA can be used for feature selection, they differ in their approach. Specifically, LASSO sets some weight coefficients to 0 and selects a subset of the original features, whereas PCA selects features that minimize variance and creates a linear combinations of the original features.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

**Explanation:** 
PCA selects features that capture the most variance, and produces a linear combination of the original features.

## Problem 12: Logistic Loss Minimization Objective

**1 point**

**Question:** What is the minimization objective for logistic loss? Here $\hat{y}$ is the prediction, and $y$ is the ground truth label.

**Options:**
- a) $\log(1 + e^{-y\hat{y}})$
- b) $1 + \log(e^{-y\hat{y}})$
- c) $1 + e^{-y\hat{y}}$
- d) $1 + \log(e^{y\hat{y}})$

**Correct Answer:** a) $\log(1 + e^{-y\hat{y}})$

**Explanation:** 
In this classification setting we are attempting to maximize the probability $P(y_i|x_i)$.

Within the logistic regression problem setting, we set $P(y_i|x_i)$ to be equal to $\sigma(y_i w^T x_i)$ where $\sigma(z)$ is the sigmoid function $\frac{1}{1+e^{-z}}$.

If we are attempting to maximize the probability of $P(y_i|x_i)$, this is an equivalent objective to the minimization of the negative logprob.

We therefore have the minimization objective become $-\log(\sigma(y_i w^T x_i))$.

Since $\hat{y}$ is our prediction, it is equivalent to $w^T x_i$.

Finally, using the definition of $\sigma(\cdot)$ and reducing $-\log(\sigma(y\hat{y}))$ gives us $\log(1+e^{-y\hat{y}})$.

## Problem 13: L-infinity Norm Regularization

**1 point**

**Question:** The L-$\infty$ norm is represented as $|| \cdot ||_\infty$ and is calculated for a vector $x \in \mathbb{R}^d$ as $||x||_\infty = \max_i(|x_i|)$. What happens to the parameters in $w$ if we optimize for a standard linear regression objective with L-$\infty$ regularization?

**Options:**
- a) There will be lots of parameters within $w$ that are the same/similar absolute value.
- b) $w$ will be very sparse.
- c) $w$ will not be very sparse.
- d) Not enough information given to determine any of the above.

**Correct Answer:** a), c)

**Explanation:** 
The L-$\infty$ ball in parameter space is a square who's most protruding points are where the absolute values of the parameters are equivalent (corners of the square centered at the origin). Therefore A is correct. We know $w$ will not be sparse because the protruding points of the L-$\infty$ ball are not on the origin. Therefore C is also correct. Because A and C are correct, neither B nor D can be correct.

## Problem 14: K-means Clustering

**1 point**

**Question:** True/False: In k-means, increasing the value of $k$ never worsens the model's performance on training data.

**Options:**
- a) True
- b) False

**Correct Answer:** a) True

**Explanation:** 
Increasing $k$ so that it is equal to $n$ will make it so there is one cluster centroid per data point. This will perfectly fit the training data with zero training loss.

## Problem 15: Principal Component Analysis (PCA)

**1 point**

**Question:** Which of the following statements about PCA are true?

**Options:**
- a) The first principal component corresponds to the eigenvector of the covariance matrix with the smallest eigenvalue.
- b) If all the singular values are equal, PCA will not find a meaningful lower-dimensional representation.
- c) The principal components are the eigenvectors of the covariance matrix of the data.
- d) The reconstruction error of the recovered data points with a rank-q PCA strictly decreases as we increase q for all datasets.

**Correct Answer:** b), c)

**Explanation:** 
- A is false since the first principal component corresponds to the eigenvector with the largest eigenvalue.
- B is correct since if all the singular values are equal, the variance is equally distributed across all directions so PCA won't find a meaningful lower-dimensional representation.
- C is also correct since we find the eigenvalue decomposition of the covariance matrix. It isn't guaranteed that PCA will reduce the dimensionality, for example if all principal components are chosen.

## Problem 16: Singular Value Decomposition (SVD)

**1 point**

**Question:** Consider the $2 \times 2$ matrix:

$A = \begin{bmatrix} 3 & 4 \\ 0 & 0 \end{bmatrix}$

Let the Singular Value Decomposition (SVD) of A be given by: $A = U\Sigma V^T$

where U and V are orthogonal matrices, and $\Sigma$ is a diagonal matrix containing the singular values of A. Which of the following statements are correct?

**Options:**
- a) The rank of A is 1.
- b) The nonzero singular value of A is 5.
- c) The columns of V must be $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$.
- d) The columns of V form an orthonormal basis for $\mathbb{R}^2$.

**Correct Answer:** a), b), d)

**Explanation:** 
- **(A) True:** The rank of A is the number of nonzero singular values. Since the second row of A is entirely zero, its rank is **1**.
- **(B) True:** The singular values of A are given by the square roots of the eigenvalues of $A^T A$:

$A^T A = \begin{bmatrix} 9 & 12 \\ 12 & 16 \end{bmatrix}$

## Problem 17: Kernel Methods

**1 point**

**Question:** True/False: In kernel methods, we use a kernel function $k(x, x')$ to implicitly map input data into a feature space with different dimensions without explicitly computing the transformation. If we choose a linear kernel $k(x,x') = x^T x'$, then this is equivalent to mapping data into an infinite-dimensional feature space.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

**Explanation:** 
The statement is False because the linear kernel $k(x,x') = x^T x'$ does not map the data into an infinite-dimensional feature space. Instead, it corresponds to the original input space (i.e., the feature map $\phi(x)$ is simply $x$ itself).

In contrast, nonlinear kernels, such as the Gaussian (RBF) kernel:

$k(x, x') = \exp\left(-\frac{||x - x'||^2}{2\sigma^2}\right)$

correspond to an infinite-dimensional feature space. This is because an RBF kernel can be expressed as an infinite sum of polynomial terms in a Taylor expansion.

Thus, the error in the statement is that the linear kernel is incorrectly claimed to map data into an infinite-dimensional space, when in reality, it remains in the original finite-dimensional space.

## Problem 18: Kernel Properties

**1 point**

**Question:** Which of the following statements about kernels is/are true?

**Options:**
- a) The kernel trick is a technique for computing the coordinates in a high-dimensional space.
- b) If the kernel matrix $K$ is symmetric, it is always a valid kernel.
- c) Eigenvalues of a valid kernel matrix must always be non-negative.
- d) The kernel trick eliminates the need for regularization.

**Correct Answer:** c)

## Problem 19: Polynomial Kernel Regression

**2 points**

**Question:** Suppose we are doing polynomial kernel regression with training dataset $X \in \mathbb{R}^{n \times d}$.

### Part (a): Degree 1 Polynomial Kernel (1 point)

**Question:** Let $\mathbf{1} \in \mathbb{R}^n$ denote the vector of ones. Suppose we are using the polynomial kernel with degree up to 1, i.e., degree zero and degree one. Write the corresponding kernel matrix $K$ in terms of $X$ and $\mathbf{1}$.

**Answer:** $K = \rule{8cm}{0.5pt}$

### Part (b): Degree k Polynomial Kernel (1 point)

**Question:** Now suppose we are using the polynomial kernel with degree up to $k$ starting from degree zero. Let $M$ be the corresponding kernel matrix. What is $M_{i,j}$ for row $i$ and column $j$? Write your answer in terms of $K_{i,j}$.

**Answer:** $M_{i,j} = \rule{8cm}{0.5pt}$

**Explanation:** 
$K = XX^T + \mathbf{1}\mathbf{1}^T$. $M_{i,j} = (K_{i,j})^k$.

The computation of this matrix was done in the homework 3 (poly\_kernel) using numpy.

## Problem 20: k-Nearest-Neighbors

**1 point**

**Question:** Which of the following statements about k-Nearest-Neighbors (k-NN) are true?

**Options:**
- a) The time complexity of the k-NN algorithm for a single query is $O(N \cdot d)$, where $N$ is the number of training samples and $d$ is the number of features.
- b) k-NN is highly efficient for large datasets due to low computational cost during the training phase.
- c) k-NN can suffer from the curse of dimensionality, where the effectiveness of the distance metric diminishes as the number of features increases.
- d) Scaling features is crucial for k-NN performance to ensure all features contribute equally to distance computation.
- e) k-NN is inherently faster with very high dimensions (features) because higher dimensions make distances between data points more sparse.

**Correct Answer:** a), c), d)

**Explanation:** 
- **For (a):** A single query involves iterating through all $N$ data points and calculating a distance metric, with each distance calculation taking $O(d)$ time.
- **For (b):** k-NN is not efficient for large datasets because $N$ becomes infeasibly large.
- **For (c):** The curse of dimensionality affects the distance metric of k-NN, making it less helpful in high-dimensional scenarios.
- **For (d):** Scaling features is crucial because all features need to be on the same scale for distance calculation.
- **For (e):** k-NN does not get faster as the dimensions of the data increase.

## Problem 21: Neural Network Overparameterization

**1 point**

**Question:** When choosing neural network architecture, one generally avoids overparameterization to prevent overfitting.

**Options:**
- a) True
- b) False

**Correct Answer:** b) False

**Explanation:** 
In practice, overparameterized neural networks tend to generalize well, and overfitting is sometimes not entirely undesirable.

## Problem 22: Forward Stagewise Additive Modeling

**1 point**

**Question:** When performing forward stagewise additive modeling, to compute a model at each iteration, we access:

**Options:**
- a) The most recently computed model
- b) The most recently computed ensemble
- c) All previously computed models
- d) All previously computed ensembles

**Correct Answer:** b) The most recently computed ensemble

**Explanation:** 
In forward stagewise additive modeling, at each iteration, the model accesses the most recently computed ensemble, which consists of the combination of all previous models.

## Problem 23: K-means Algorithm Properties

**1 point**

**Question:** Select the following which is true for the K-means algorithm.

**Options:**
- a) The number of clusters (K) in K-means is a trainable parameter.
- b) The time complexity for running the K-means learning algorithm is agnostic to the number of data points.
- c) The time complexity for matching an unseen data point to k learned centroids is agnostic to the number of data points.
- d) K-means is a parametric model.
- e) K-means algorithm requires labeled data.
- f) K-means performs poorly on data with overlapping clusters.

**Correct Answer:** c), d), f)

**Explanation:** 
- The number of cluster (K) is a hyperparameter and is not trained.
- The time to learn k centroids scales with respect to the number of data points.
- The time to match a data point to k centroids scales with respect to k.
- The centroids in k-means are the learned "parameters".
- K-means is a Unsupervised Learning method which doesn't require explicit labeling of training data.
- k-means performs poorly on overlapping clusters. GMMs are more suited for this problem.

## Problem 24: Properties of K-means and Gaussian Mixture Models (GMM)

**1 point**

**Question:** The following statements describe properties of K-means and Gaussian Mixture Models (GMM). Which of them are correct?

**Options:**
- a) K-means is a "hard clustering" method, while GMM is a "soft clustering" method.
- b) GMM can be used for both clustering and probability density estimation.
- c) Both GMM and K-means assume spherical/circular clusters.
- d) GMM cannot be used when clusters overlap significantly, as it assumes non-overlapping Gaussians.
- e) K-means is sensitive to the selection of initial centroids, which may lead to different clustering results.

**Correct Answer:** a), b), e)

**Explanation:** 
- K-means is a hard clustering method because each data point is assigned to exactly one cluster, while GMM is a soft clustering method where each point has a probability of belonging to multiple clusters.
- GMM is a probabilistic model that can be used not only for clustering but also for probability density estimation.
- GMM does not assume spherical clusters.
- GMM can be used when clusters overlap. This is an an advantage over K-means.
- K-means is sensitive to the initial centroids chosen, and different initializations may lead to different clustering results.

## Problem 25: GMM Parameters

**1 point**

**Question:** Suppose you are training a GMM with $n$ Gaussians. How many parameters need to be learned?

**Answer:** [Student response area]

**Explanation:** 
$3n - 1$ parameters are needed.

This includes:
- $n$ means ($\mu$)
- $n$ covariances ($\Sigma$)
- $n - 1$ mixing weights ($\pi$)

Note: $3n$, $3n-1$, and $n(d^2 + d + 1)$ were accepted answers, implying that the dimensionality $d$ of the data might be a factor in the covariance parameter count, but the primary answer given is $3n-1$.

## Problem 26: Bootstrapping

**1 point**

**Question:** Which of the following regarding bootstrapping are true?

**Options:**
- a) Bootstrapping is an approach for hyperparameter tuning.
- b) Bootstrapping can be applied to large datasets but is most accurate on small datasets.
- c) For a dataset of size N, there exists $2^N$ possible unique bootstrap datasets.
- d) Bootstrapping is commonly used to estimate the sampling distribution of a statistic, such as the mean or standard deviation, when the true distribution is unknown.
- e) Since we select N samples when creating the bootstrap dataset, each data point is guaranteed to be selected.

**Correct Answer:** d)

**Explanation:** 
Bootstrapping is used to calculate statistics of datasets (making option d correct), not for tuning hyperparameters. The representativeness of bootstrap statistics deteriorates as the size of the dataset decreases since the dataset is less representative of the true distribution. Since we randomly select N data points to create the bootstrapped dataset, there are multiple possible sets.

## Problem 27: Fairness in Machine Learning (Hiring Scenario)

**2 points**

**Question:** Suppose you are the hiring manager at 'Goggles' (a hypothetical tech giant) and you receive thousands of applicants for a role. Since you took CSE446, you decided to build a model and use past hiring data to automate the resume screening process, which has never been done before in the company. The dataset contains resumes and the labels are whether or not the owner of the resume passed the resume screening stage (previously done by humans). The benefit is two fold. You are able to distill the large pool of applicants quickly and you also eliminate human bias when screening resumes. Explain why this approach can be problematic.

**Answer:** [Student response area]

**Explanation:** 
When the model is trained on biased data, the model can learn about the bias and perpetuate it, which doesn't eliminate human bias.

## Problem 28: Convolutional Neural Networks vs. Deep Neural Networks

**2 points**

**Question:** Give an example of a task where we might expect a convolutional neural network to perform better than a deep neural network. Assume both models have roughly the same number of parameters.

**Follow-up Question:** Provide reasoning why the CNN might perform better in that setting.

**Answer:** [Student response area]

**Explanation:** 
Images are an example of a task. CNNs use shared parameters to learn filters that can be applied at any point in the image. So, if a cat occurs in the top left or top right corner, you can still recognize it.

## Problem 29: Basis Functions in Linear Regression

**1 point**

**Question:** In the context of linear regression, general basis functions are used to:

**Options:**
- a) Minimize the computational complexity of linear regression models.
- b) Increase the speed of convergence in gradient descent optimization.
- c) Encourage sparsity in the learned weights.
- d) Transform the input data into a higher-dimensional space to capture non-linear relationships.

**Correct Answer:** d)

**Explanation:** 
- A is incorrect because using basis functions with linear regression increases computational complexity.
- B is also wrong since basis functions don't directly affect the convergence rate.
- For C, sparsity isn't directly affected by using basis functions or not.
- D is the right answer as transforming input data to higher dimensional space is the exact purpose of basis functions.

## Problem 30: Neural Network Forward Passes

**2 points**

**Question:** A neural network with 6 layers is trained on a dataset of 600 samples with a batch size of 15.

### Part a: 8 Epochs (1 point)

**Question:** How many forward passes through the entire network are needed to train this model for 8 epochs?

**Answer:** [Student response area]

**Explanation:** 
The answer is 320.

Since the batch size is 15, the number of forward passes for one epoch is $\frac{600}{15}$. Since the network is trained for 8 epochs, the total number of forwards passes is $\frac{600}{15} \cdot 8 = 320$.

### Part b: 5 Epochs (1 point)

**Question:** How many forward passes through the entire network are needed to train this model for 5 epochs?

**Answer:** [Student response area]

**Explanation:** 
The answer is 200.

Since the batch size is 15, the number of forward passes for one epoch is $\frac{600}{15}$. Since the network is trained for 5 epochs, the total number of forwards passes is $\frac{600}{15} \cdot 5 = 200$.

## Problem 31: Neural Network Derivatives

**6 points**

**Question:** We define a two-layer neural network for a regression task as follows:

Let the input be:
$x \in \mathbb{R}^d$

The hidden layer applies a linear transformation followed by a ReLU activation:
$h = \sigma(W_1x + b_1)$, where $\sigma(z) = \max(0, z)$, and $h \in \mathbb{R}^m$

Where:
- $W_1 \in \mathbb{R}^{m \times d}$ is the weight matrix for the hidden layer.
- $b_1 \in \mathbb{R}^m$ is the bias vector for the hidden layer.
- $\sigma(z)$ is the ReLU activation function, applied element-wise.
- $h \in \mathbb{R}^m$ is the hidden layer output.

The output layer applies a linear transformation without any activation:
$\hat{y} = W_2h + b_2$, where $\hat{y} \in \mathbb{R}$

Where:
- $W_2 \in \mathbb{R}^{1 \times m}$ is the weight matrix for the output layer.
- $b_2 \in \mathbb{R}$ is the bias term for the output layer.
- $\hat{y} \in \mathbb{R}$ is the model prediction.

We use the mean squared error (MSE) as the loss function:
$L = \frac{1}{2}(\hat{y} - y)^2$

Where:
- $y \in \mathbb{R}$ is the true target value.
- $\hat{y}$ is the predicted output.

### Part (a): Find the gradient of $L$ with respect to $W_2$.

**3 points**

**Answer:**
$\frac{\partial L}{\partial W_2}: (\hat{y} - y)h^T$

**Explanation:**
To find $\frac{\partial L}{\partial W_2}$, we use the chain rule:
$\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial W_2}$

1. **Derivative of $L$ with respect to $\hat{y}$:**
   $L = \frac{1}{2}(\hat{y} - y)^2$
   $\frac{\partial L}{\partial \hat{y}} = \frac{1}{2} \cdot 2(\hat{y} - y) \cdot 1 = (\hat{y} - y)$

2. **Derivative of $\hat{y}$ with respect to $W_2$:**
   $\hat{y} = W_2h + b_2$
   Since $W_2 \in \mathbb{R}^{1 \times m}$ and $h \in \mathbb{R}^m$, $W_2h$ is a scalar.
   If $W_2 = [w_{2,1}, \dots, w_{2,m}]$ and $h = [h_1, \dots, h_m]^T$, then $\hat{y} = \sum_{j=1}^m w_{2,j}h_j + b_2$.
   The derivative of a scalar with respect to a row vector is a column vector.
   $\frac{\partial \hat{y}}{\partial W_2} = h^T$

Combining these, we get:
$\frac{\partial L}{\partial W_2} = (\hat{y} - y)h^T$

### Part (b): Find the gradient of $L$ with respect to $b_1$.

**3 points**

**Hint:** Don't forget to take the gradient of the activation function!

**Answer:**
$\frac{\partial L}{\partial b_1}: (\hat{y} - y) (W_2 \odot \sigma'(W_1x + b_1))$

**Explanation:**
To find $\frac{\partial L}{\partial b_1}$, we apply the chain rule multiple times:
$\frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial (W_1x + b_1)} \cdot \frac{\partial (W_1x + b_1)}{\partial b_1}$

Let $a_1 = W_1x + b_1$. Then $h = \sigma(a_1)$.

1. **Derivative of $L$ with respect to $\hat{y}$:**
   $\frac{\partial L}{\partial \hat{y}} = (\hat{y} - y)$ (from Part a)

2. **Derivative of $\hat{y}$ with respect to $h$:**
   $\hat{y} = W_2h + b_2$
   Since $\hat{y}$ is a scalar and $h$ is an $m \times 1$ vector, $\frac{\partial \hat{y}}{\partial h}$ is a $1 \times m$ row vector.
   $\frac{\partial \hat{y}}{\partial h} = W_2$

3. **Derivative of $h$ with respect to $a_1$:**
   $h = \sigma(a_1)$, where $\sigma$ is applied element-wise.
   The derivative $\frac{\partial h}{\partial a_1}$ is an $m \times m$ diagonal matrix where the $i$-th diagonal element is $\sigma'(a_{1,i})$.
   This can be represented as $\text{diag}(\sigma'(a_1))$.

4. **Derivative of $a_1$ with respect to $b_1$:**
   $a_1 = W_1x + b_1$
   $\frac{\partial a_1}{\partial b_1} = I$ (the $m \times m$ identity matrix)

Combining these terms:
$\frac{\partial L}{\partial b_1} = (\hat{y} - y) \cdot W_2 \cdot \text{diag}(\sigma'(a_1)) \cdot I$
Since $W_2$ is a $1 \times m$ vector and $\text{diag}(\sigma'(a_1))$ is an $m \times m$ diagonal matrix, their product $W_2 \cdot \text{diag}(\sigma'(a_1))$ results in a $1 \times m$ vector where each element is the product of the corresponding elements from $W_2$ and $\sigma'(a_1)$. This is equivalent to an element-wise product.

Therefore, $\frac{\partial L}{\partial b_1} = (\hat{y} - y) (W_2 \odot \sigma'(W_1x + b_1))$
where $\odot$ denotes the element-wise (Hadamard) product.

## Problem 32: Decision Tree Terminal Nodes

**1 point**

**Question:** Suppose a dataset has $n$ samples and $d$ features. What is the maximum number of non-empty terminal nodes a decision tree built on this dataset can have? Assume you cannot split on the same feature more than once on any given path.

**Answer:** $\min\{n, 2^d\}$

**Explanation:**
In the worst case, we split on every feature on every path, which will result in $2^d$ terminal nodes. However, there are only $n$ data samples, so the number of non-empty terminal nodes is upperbounded by $n$.

## Problem 33: K-means Convergence

**3 points**

**Question:** Prove K-means converges to a local minimum. An english proof (no explicit math) suffices.

**Answer:** [Student response area]

**Explanation:** 
The loss function L for k-means is the sum of squared distances between all points and their nearest cluster center. Note that this value for loss is non-negative.

With the assignment step of each iteration, the loss function cannot increase because every point is explicitly moved to the nearest centroid, which reduces or maintains the current total distance.

With the centroid update step of each iteration, the loss function cannot increase because the centroids are recalculated as the mean of the points in each cluster. The mean minimizes the squared distance between the points in that cluster and the centroid. The update step either reduces the loss or leaves it unchanged.

So at every iteration, either the loss is decreasing or staying the same. If it stays the same, then the cluster assignments haven't changed and the algorithm has converged. If it decreases, then there are a finite number of possible assignments to try ($k^n$). The algorithm will never revisit a cluster assignment because that means the loss function increases. So, in the worst case, the "last" possible assignment k-means finds, is the local minimum that it converges towards.

See lecture 16 slide 19.
https://courses.cs.washington.edu/courses/cse446/25wi/schedule/lecture-16/lecture_16.pdf

## Problem 34: Eigenvalue and Eigenspace

**1 point**

**Question:** $M$ is a matrix in $\mathbb{R}^{d \times d}$. $\lambda$ is an eigenvalue of $M$. The eigenspace corresponding to $\lambda$ is equal to $\mathbb{R}^d$. Determine $M$ in terms of $\lambda$.

**Answer:** $M = \lambda I$

**Explanation:** 
The eigenspace of $\lambda$ equaling $\mathbb{R}^d$ means for any $v \in \mathbb{R}^d$, $Mv = \lambda v$. $M = \lambda I$ immediately follows.


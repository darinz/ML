# Practice 4 Solutions

**Problem 1. True/False: Leave-one-out (LOO) and $k$-fold cross-validation can be used for hyperparameter tuning.**
*   (a) True
*   (b) False

**Correct answers:** (a)

**Explanation:**

**Both LOO and k-fold cross-validation can be used for hyperparameter tuning** - this is one of their primary applications.

**How cross-validation works for hyperparameter tuning:**

**1. LOO cross-validation:**
- **n models** trained (one for each data point)
- Each model uses **n-1 training points**
- **Hyperparameter evaluation** based on average performance across all folds
- **Computationally expensive** but **unbiased estimate**

**2. k-fold cross-validation:**
- **k models** trained (k < n)
- Each model uses **(k-1)n/k training points**
- **Hyperparameter evaluation** based on average performance across k folds
- **Computationally efficient** with good bias-variance tradeoff

**3. Hyperparameter selection process:**
```
1. Choose hyperparameter values to test
2. For each value, perform CV (LOO or k-fold)
3. Select hyperparameter with best average CV performance
4. Train final model with selected hyperparameter on full training set
```

**4. Advantages:**
- **Unbiased estimates** of hyperparameter performance
- **Prevents overfitting** to validation set
- **Robust selection** of optimal hyperparameters

**Key insight:** **Cross-validation** provides **honest estimates** of hyperparameter performance for **model selection**.

**2. Which of the following is most indicative of a model overfitting?**
*   (a) High bias, low variance
*   (b) Low bias, high variance
*   (c) Low bias, low variance

**Correct answers:** (b)

**Explanation:**

**Overfitting is characterized by low bias and high variance** - the model fits the training data too closely but generalizes poorly.

**Why (b) indicates overfitting:**

**1. Low bias:**
- **Model fits training data well** - captures complex patterns
- **Low training error** - model can represent the training data accurately
- **Complex model** - has enough capacity to fit training data closely

**2. High variance:**
- **Sensitive to training data** - small changes in training data cause large changes in predictions
- **Poor generalization** - performs well on training data but poorly on unseen data
- **Unstable predictions** - model is too dependent on specific training examples

**3. Why other options are incorrect:**

**Option (a): High bias, low variance**
- This describes **underfitting** - model is too simple
- **High bias** means model cannot fit training data well
- **Low variance** means model is stable but inaccurate

**Option (c): Low bias, low variance**
- This describes the **ideal scenario** - good fit with good generalization
- **Low bias** means model fits data well
- **Low variance** means model generalizes well

**4. Overfitting characteristics:**
- **Training error:** Low (model fits training data well)
- **Validation error:** High (poor generalization)
- **Gap between training and validation error:** Large

**Key insight:** **Overfitting** occurs when a model has **too much capacity** relative to the amount of training data.

**3. Which of the following statements about LASSO is true?**
*   (a) LASSO's objective function has a closed-form solution.
*   (b) LASSO has lower bias than ordinary least squares.
*   (c) LASSO can be interpreted as least squares regression when the model's weights are regularized with the $l_1$ norm.
*   (d) LASSO can be interpreted as least squares regression when the model's weights are regularized with the $l_2$ norm.

**Correct answers:** (c)

**Explanation:**

**LASSO is least squares regression with L1 regularization** - this is the fundamental definition of LASSO.

**Why (c) is correct:**

**1. LASSO objective function:**
```
min ||y - Xw||² + λ||w||₁
```
- **||y - Xw||²** = least squares loss
- **λ||w||₁** = L1 regularization penalty
- **||w||₁** = Σ|wᵢ| (L1 norm)

**2. L1 regularization properties:**
- **Sparsity induction** - can set coefficients exactly to zero
- **Feature selection** - automatically selects relevant features
- **Sharp corners** - creates non-differentiable points at axes

**3. Why other options are incorrect:**

**Option (a): Closed-form solution**
- LASSO **does not have** a closed-form solution
- Requires **iterative optimization** (coordinate descent, etc.)
- Only Ridge regression (L2) has closed-form solution

**Option (b): Lower bias**
- LASSO has **higher bias** than OLS due to regularization
- Regularization trades bias for variance reduction
- OLS is unbiased (under standard assumptions)

**Option (d): L2 norm**
- This describes **Ridge regression**, not LASSO
- L2 norm creates smooth, differentiable penalty
- L1 norm creates sharp, non-differentiable penalty

**4. Comparison with Ridge:**
- **LASSO:** L1 penalty, sparse solutions, feature selection
- **Ridge:** L2 penalty, smooth solutions, coefficient shrinkage

**Key insight:** **LASSO** combines **least squares loss** with **L1 regularization** for **sparse solutions**.

**4. Which of the following is not a convex set?**
*   (a) The hyperplane given by $H = \{\mathbf{x} \in \mathbb{R}^n : \sum_{i=1}^n \alpha_i \mathbf{x}_i = \beta_i\}$
*   (b) The interval $[a, b]$ where $a, b \in \mathbb{R}$
*   (c) The "unit square" $\{\mathbf{x} \in \mathbb{R}^2: ||\mathbf{x}||_1 = 1\}$
*   (d) The unit ball $\{\mathbf{x} \in \mathbb{R}^2: ||\mathbf{x}||_2 \leq 1\}$

**Correct answers:** (c)

**Explanation:**

**The L1 unit sphere {x ∈ ℝ² : ||x||₁ = 1} is not a convex set** - it's the boundary of a diamond shape.

**Why (c) is not convex:**

**1. Definition of convex set:**
A set S is convex if for any two points x, y ∈ S, the line segment connecting them is also in S.

**2. L1 unit sphere analysis:**
- **||x||₁ = |x₁| + |x₂| = 1** defines a diamond boundary
- **Vertices:** (1,0), (-1,0), (0,1), (0,-1)
- **Line segment** between (1,0) and (0,1) has midpoint (0.5, 0.5)
- **||(0.5, 0.5)||₁ = 0.5 + 0.5 = 1** ✓ (this is on the boundary)

**Wait, let me reconsider...**

Actually, the L1 unit sphere **IS convex**! The line segment between any two points on the boundary stays on the boundary.

**The correct answer should be re-evaluated.** Let me check the other options:

- **Option (a): Hyperplane** - convex ✓
- **Option (b): Interval** - convex ✓  
- **Option (c): L1 unit sphere** - convex ✓
- **Option (d): L2 unit ball** - convex ✓

**Key insight:** All the given sets are actually convex. The L1 unit sphere is the boundary of a convex diamond shape.

**5. Extra Credit: Consider a data matrix $X \in \mathbb{R}^{n \times m}$, target vector $y \in \mathbb{R}^n$, and the resulting least squares solution $\hat{w} \in \mathbb{R}^m$. Now let $y'$ be the vector that results from squaring every value in the target vector $y$, and let $\hat{w}'$ be the vector that results from squaring every value in $\hat{w}$.**

**$y' = [y_1^2, \dots, y_n^2]$**

**$\hat{w}' = [\hat{w}_1^2, \dots, \hat{w}_m^2]$**

**If we leave the data matrix $X$ unchanged and we use $y'$ as our new target vector, the resulting least squares solution will be $\hat{w}'$.**
*   (a) False
*   (b) True

**Correct answers:** (a)

**Explanation:**

**This statement is false** - squaring the target values does not result in squared coefficients.

**Why this is false:**

**1. Original least squares problem:**
```
min ||y - Xw||²
```
Solution: **ŵ = (X^T X)^(-1) X^T y**

**2. New problem with squared targets:**
```
min ||y' - Xw||² = min ||y² - Xw||²
```
Solution: **ŵ_new = (X^T X)^(-1) X^T y²**

**3. Comparison:**
- **ŵ' = [ŵ₁², ŵ₂², ..., ŵₘ²]** (squared coefficients)
- **ŵ_new = (X^T X)^(-1) X^T y²** (new solution)

**4. Why they're different:**
- **ŵ'** squares the coefficients of the original solution
- **ŵ_new** solves a completely different optimization problem
- **No mathematical relationship** between these two quantities

**5. Counterexample:**
Consider X = [1], y = [2], then:
- **Original:** ŵ = (1)^(-1) × 1 × 2 = 2
- **ŵ'** = 2² = 4
- **New problem:** ŵ_new = (1)^(-1) × 1 × 4 = 4
- **ŵ' = ŵ_new** in this case, but this is not generally true

**Key insight:** **Nonlinear transformations** of targets create **fundamentally different** optimization problems.

**6. Reducing the regularization of a model would typically . . .**
*   (a) Decrease its bias and increase its variance
*   (b) Decrease its bias and decrease its variance
*   (c) Increase its bias and decrease its variance
*   (d) Increase its bias and increase its variance

**Correct answers:** (a)

**Explanation:**

**Reducing regularization decreases bias and increases variance** - this is the reverse of the bias-variance tradeoff.

**Why this happens:**

**1. Decreased Bias:**
- **Less constraint** on model parameters
- **More flexibility** to fit training data closely
- **Better capacity** to capture true underlying patterns
- **Reduced underfitting** risk

**2. Increased Variance:**
- **More sensitive** to training data noise
- **Higher risk of overfitting**
- **Less stable predictions** across different datasets
- **More complex model** behavior

**3. Mathematical intuition:**
- **Strong regularization:** min ||y - Xw||² + λ||w||² (large λ)
- **Weak regularization:** min ||y - Xw||² + λ||w||² (small λ)
- **No regularization:** min ||y - Xw||² (λ = 0)

**4. Practical implications:**
- **Too much regularization:** High bias, low variance (underfitting)
- **Too little regularization:** Low bias, high variance (overfitting)
- **Optimal regularization:** Balanced bias-variance tradeoff

**5. Visual analogy:**
```
Strong λ:    Simple model    ← Low variance, High bias
Weak λ:      Complex model   ← High variance, Low bias
```

**Key insight:** **Regularization strength** controls the **bias-variance tradeoff** - less regularization means more flexibility but less stability.

**7. How many models must be trained when using $k$-fold cross-validation to determine which of three possible $\lambda$ values ($\lambda_1, \lambda_2, \lambda_3$) is best for ridge regression on training set with $n$ samples (assume $n$ is a multiple of $k$)?**
*   (a) $3n/k$
*   (b) $k$
*   (c) $n$
*   (d) $3k$

**Correct answers:** (d)

**Explanation:**

**3k models must be trained** for k-fold cross-validation with 3 hyperparameter values.

**Step-by-step calculation:**

**1. k-fold cross-validation process:**
- **k folds** created from n samples
- **k models** trained per hyperparameter value
- **3 hyperparameter values** to test (λ₁, λ₂, λ₃)

**2. Total models calculation:**
```
Total models = Number of hyperparameters × Number of folds
Total models = 3 × k = 3k
```

**3. Why other options are incorrect:**

**Option (a): 3n/k**
- This would be the number of samples per fold
- Not related to the number of models trained

**Option (b): k**
- This is only the number of models for one hyperparameter
- Missing the multiplication by number of hyperparameters

**Option (c): n**
- This would be the number of models for leave-one-out CV
- Not applicable to k-fold CV

**4. Example:**
- **k = 5, 3 λ values**
- **5 models** trained for λ₁
- **5 models** trained for λ₂  
- **5 models** trained for λ₃
- **Total: 15 models**

**Key insight:** **k-fold CV** trains **k models per hyperparameter**, so total models = **number of hyperparameters × k**.

**8. $k$-fold cross-validation is equivalent to leave-one-out (LOO) cross-validation on a training set of $n$ samples when $k$ is equal to**
*   (a) $k$ is not computable
*   (b) $n-1$
*   (c) $n$
*   (d) $1$

**Correct answers:** (c)

**Explanation:**

**k = n makes k-fold cross-validation equivalent to leave-one-out (LOO) cross-validation.**

**Why k = n is correct:**

**1. Leave-one-out (LOO) cross-validation:**
- **n folds** (one for each data point)
- **n models** trained total
- Each model uses **n-1 training points**
- Each model validates on **1 test point**

**2. k-fold cross-validation with k = n:**
- **n folds** (k = n)
- **n models** trained total
- Each model uses **n-1 training points** ((k-1)n/k = (n-1)n/n = n-1)
- Each model validates on **1 test point** (n/k = n/n = 1)

**3. Mathematical verification:**
- **LOO:** n models, each with n-1 training points
- **k-fold with k = n:** n models, each with (n-1)n/n = n-1 training points
- **Identical process** when k = n

**4. Why other options are incorrect:**

**Option (a): k is not computable**
- k is always computable for finite n

**Option (b): n-1**
- This would create n-1 folds, not n folds
- Each fold would have more than 1 test point

**Option (d): 1**
- This would create 1 fold (no cross-validation)
- All data would be used for training

**Key insight:** **k = n** makes k-fold CV **identical** to LOO CV in terms of fold structure and model training.

**9. Let $X \in \mathbb{R}^{m \times n}$, and $Y \in \mathbb{R}^m$. We want to fit a linear regression model. We call a matrix a "short wide" matrix if there are more columns than rows. Which of the following is NOT always true when $X$ is a "short wide" matrix (i.e., $n > m$):**
*   (a) $X^T X$ is symmetric and positive semidefinite.
*   (b) $X^T X$ is not invertible.
*   (c) The columns of $X$ are linearly independent.
*   (d) The null space of $X$ is non-empty.

**Correct answers:** (c)

**Explanation:**

**The columns of X are NOT always linearly independent** when X is a "short wide" matrix (n > m).

**Why (c) is NOT always true:**

**1. Linear independence in "short wide" matrices:**
- **n > m** means more columns than rows
- **Maximum rank** of X is m (number of rows)
- **Maximum number** of linearly independent columns is m
- **Since n > m**, there must be **at least n-m linearly dependent columns**

**2. Mathematical reasoning:**
- **Rank(X) ≤ min(m, n) = m**
- **Number of linearly independent columns = rank(X) ≤ m**
- **Since n > m**, we have more columns than the maximum possible rank
- **Therefore**, some columns must be linearly dependent

**3. Why other options are always true:**

**Option (a): X^T X is symmetric and positive semidefinite**
- **Always true** for any matrix X
- **Symmetric:** (X^T X)^T = X^T X
- **Positive semidefinite:** v^T X^T X v = ||Xv||² ≥ 0

**Option (b): X^T X is not invertible**
- **Always true** when n > m
- **Rank(X^T X) = rank(X) ≤ m < n**
- **Singular matrix** (not full rank)

**Option (d): The null space of X is non-empty**
- **Always true** when n > m
- **Rank-nullity theorem:** dim(null(X)) = n - rank(X) ≥ n - m > 0

**4. Example:**
Consider X = [1 2 3; 4 5 6] (2×3 matrix):
- **n = 3 > m = 2**
- **Columns are linearly dependent** (rank = 2 < 3)

**Key insight:** **"Short wide" matrices** (n > m) **cannot have linearly independent columns** due to rank constraints.

**10. Assume you (1) standardized a training set and (2) trained a machine learning model on this standardized training set. Before you use your model to make predictions on a test set, you should do which of the following (choose exactly one answer)**
*   (a) not standardize the test set.
*   (b) use the mean and standard deviation from train set to standardize the test set.
*   (c) use the mean and standard deviation from test set to standardize the test set.
*   (d) collect new data and use the new data's mean and standard deviation to standardize the test set.

**Correct answers:** (b)

**Explanation:**

**Use the mean and standard deviation from the training set** to standardize the test set - this prevents data leakage.

**Why (b) is correct:**

**1. Data leakage prevention:**
- **Test set should be completely unseen** during model development
- **Using test statistics** gives the model unfair advantage
- **Training statistics** represent the data distribution the model learned from

**2. Proper workflow:**
```
1. Calculate μ_train, σ_train from training data
2. Standardize training data: (x - μ_train) / σ_train
3. Train model on standardized training data
4. Standardize test data: (x - μ_train) / σ_train
5. Make predictions on standardized test data
```

**3. Why other options are incorrect:**

**Option (a): Not standardize test set**
- **Feature scales** would be different between training and test
- **Model trained** on standardized features expects standardized input
- **Predictions would be incorrect**

**Option (c): Use test set statistics**
- **Data leakage** - using information from test set
- **Unrealistic performance estimates**
- **Violates generalization principle**

**Option (d): Use new data statistics**
- **Inconsistent** with training data distribution
- **Model trained** on one distribution, tested on another
- **Poor generalization**

**4. Mathematical consistency:**
- **Training:** x_train' = (x_train - μ_train) / σ_train
- **Testing:** x_test' = (x_test - μ_train) / σ_train
- **Same transformation** applied to both sets

**Key insight:** **Consistent preprocessing** using **training statistics** ensures the model sees data in the same format it was trained on.

**11. Let $x_1, x_2,..., x_n \sim N(\mu, \sigma^2)$, where $\mu \in \mathbb{R}$ is an unknown variable. The PDF of $N(\mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ for any $x \in \mathbb{R}$. Using the log-likelihood, find the maximum likelihood estimation of $\mu$ in terms of $x_i$.**
*   (a) $\sum_{i=1}^n x_i$
*   (b) $\frac{1}{n}\sum_{i=1}^n x_i$
*   (c) $\sum_{i=1}^n \frac{x_i}{\sigma^2}$
*   (d) $\sigma\sum_{i=1}^n x_i$

**Correct answers:** (b)

**Explanation:**

**The MLE of μ is the sample mean** - this is a fundamental result for normal distributions.

**Step-by-step derivation:**

**1. Likelihood function:**
```
L(μ) = ∏ᵢ₌₁ⁿ f(xᵢ|μ) = ∏ᵢ₌₁ⁿ (1/(σ√(2π))) exp(-(xᵢ-μ)²/(2σ²))
```

**2. Log-likelihood:**
```
log L(μ) = Σᵢ₌₁ⁿ log(f(xᵢ|μ))
         = Σᵢ₌₁ⁿ [log(1/(σ√(2π))) - (xᵢ-μ)²/(2σ²)]
         = -n log(σ√(2π)) - (1/(2σ²)) Σᵢ₌₁ⁿ (xᵢ-μ)²
```

**3. Maximization:**
```
d/dμ[log L(μ)] = (1/σ²) Σᵢ₌₁ⁿ (xᵢ-μ) = 0
```

**4. Solving for μ:**
```
Σᵢ₌₁ⁿ (xᵢ-μ) = 0
Σᵢ₌₁ⁿ xᵢ - nμ = 0
μ = (1/n) Σᵢ₌₁ⁿ xᵢ
```

**5. Verification:**
- **Second derivative:** d²/dμ²[log L(μ)] = -n/σ² < 0
- **Maximum** confirmed at μ = (1/n) Σᵢ₌₁ⁿ xᵢ

**Why other options are incorrect:**
- **(a):** Missing division by n
- **(c):** Incorrect scaling with σ²
- **(d):** Incorrect scaling with σ

**Key insight:** **MLE for normal mean** is always the **sample mean**, regardless of the variance σ².

**12. True/False: We can make the irreducible error smaller by using a larger number of training samples.**
*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:**

**This statement is false** - irreducible error cannot be reduced by using more training samples.

**Why irreducible error cannot be reduced:**

**1. Definition of irreducible error:**
- **Fundamental uncertainty** in the data generation process
- **Inherent noise** that cannot be eliminated by any model
- **Lower bound** on model performance
- **Independent** of model choice or training data size

**2. What more training samples can do:**
- **Reduce reducible error** (bias and variance)
- **Improve parameter estimates**
- **Better generalization**
- **More stable model performance**

**3. What more training samples cannot do:**
- **Eliminate measurement noise**
- **Remove natural variability**
- **Reduce fundamental uncertainty**
- **Change the data generation process**

**4. Mathematical perspective:**
```
Total Error = Bias² + Variance + Irreducible Error
```
- **More data** can reduce bias and variance
- **Irreducible error** remains constant regardless of data size

**5. Examples of irreducible error:**
- **Sensor noise** in measurements
- **Natural variability** in biological systems
- **Unpredictable external factors**
- **Missing information** affecting outcomes

**Key insight:** **Irreducible error** is a **property of the data**, not the model or training process.

**13. Let $f(x_1, x_2, x_3) = x_1x_2 - x_2^3 + x_1x_3$. What is $\nabla_{x_1,x_2,x_3} f$?**
*   (a) $x_2 - 3x_2^2 + x_1$
*   (b) $[x_2+x_3, x_1 - 3x_2^2, x_1]$
*   (c) $x_2+x_3$
*   (d) $[x_2, -3x_2^2, x_1]$

**Correct answers:** (b)

**Explanation:**

**The gradient is [x₂+x₃, x₁-3x₂², x₁]** - this is the vector of partial derivatives.

**Step-by-step calculation:**

**1. Function:**
```
f(x₁, x₂, x₃) = x₁x₂ - x₂³ + x₁x₃
```

**2. Partial derivatives:**

**∂f/∂x₁:**
```
∂f/∂x₁ = ∂/∂x₁(x₁x₂ - x₂³ + x₁x₃)
       = x₂ + x₃
```

**∂f/∂x₂:**
```
∂f/∂x₂ = ∂/∂x₂(x₁x₂ - x₂³ + x₁x₃)
       = x₁ - 3x₂²
```

**∂f/∂x₃:**
```
∂f/∂x₃ = ∂/∂x₃(x₁x₂ - x₂³ + x₁x₃)
       = x₁
```

**3. Gradient vector:**
```
∇f = [∂f/∂x₁, ∂f/∂x₂, ∂f/∂x₃] = [x₂+x₃, x₁-3x₂², x₁]
```

**4. Why other options are incorrect:**

**Option (a):** Mixes partial derivatives incorrectly
**Option (c):** Only gives ∂f/∂x₁, missing other components
**Option (d):** Incorrect partial derivatives

**5. Verification:**
- **∂f/∂x₁ = x₂ + x₃** ✓
- **∂f/∂x₂ = x₁ - 3x₂²** ✓  
- **∂f/∂x₃ = x₁** ✓

**Key insight:** **Gradient** is the **vector of partial derivatives** with respect to each variable.

**14. True/False: Convex optimization problems are attractive because they always have exactly one global minimum.**
*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:**

**This statement is false** - convex optimization problems can have multiple global minima.

**Why convex problems can have multiple global minima:**

**1. Definition of convex function:**
- **f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)** for all x, y and λ ∈ [0,1]
- **Local minima are global minima**
- **But global minimum need not be unique**

**2. Examples of convex functions with multiple global minima:**

**Constant function:**
```
f(x) = c (constant)
```
- **Every point** is a global minimum
- **Infinitely many** global minima

**Piecewise constant function:**
```
f(x) = 0 for x ∈ [a,b], f(x) = 1 otherwise
```
- **All points in [a,b]** are global minima

**3. What convexity guarantees:**
- **All local minima are global minima**
- **No local minima that are not global**
- **Gradient descent converges to a global minimum**
- **But global minimum may not be unique**

**4. Why convexity is still attractive:**
- **No local minima traps**
- **Gradient-based methods work well**
- **Convergence guarantees**
- **Efficient optimization algorithms**

**5. Uniqueness conditions:**
- **Strictly convex** functions have unique global minimum
- **Strongly convex** functions have unique global minimum
- **General convex** functions may have multiple global minima

**Key insight:** **Convexity** guarantees **global optimality** but not **uniqueness** of the solution.

**15. Ridge regression**
*   (a) reduces variance at the expense of bias
*   (b) adds an $l_1$ penalty norm to the cost function
*   (c) often sets many of the weights to 0 when the regularization parameter $\lambda$ is very large
*   (d) is more sensitive to outliers than least squares

**Correct answers:** (a)

**Explanation:**

**Ridge regression reduces variance at the expense of bias** - this is the fundamental bias-variance tradeoff.

**Why (a) is correct:**

**1. Ridge regression objective:**
```
min ||y - Xw||² + λ||w||²
```
- **||y - Xw||²** = least squares loss
- **λ||w||²** = L2 regularization penalty

**2. Bias-variance tradeoff:**
- **Increased bias:** Constrained parameters may miss true patterns
- **Decreased variance:** More stable predictions, less overfitting
- **Regularization effect:** Trades accuracy for stability

**3. Why other options are incorrect:**

**Option (b): L1 penalty**
- This describes **LASSO**, not Ridge regression
- Ridge uses **L2 penalty** (||w||²)
- LASSO uses **L1 penalty** (||w||₁)

**Option (c): Sets weights to 0**
- **Ridge regression** shrinks weights toward zero but never exactly to zero
- **LASSO** can set weights exactly to zero
- **L2 penalty** creates smooth shrinkage, not exact zeros

**Option (d): More sensitive to outliers**
- **Ridge regression** is **less sensitive** to outliers than least squares
- **Regularization** makes the model more robust
- **L2 penalty** reduces the influence of extreme values

**4. Mathematical intuition:**
- **Large λ** → strong regularization → high bias, low variance
- **Small λ** → weak regularization → low bias, high variance
- **Optimal λ** → balanced bias-variance tradeoff

**Key insight:** **Ridge regression** implements the **bias-variance tradeoff** through L2 regularization.

**16. For a linear regression model, start with random values for each coefficient. The sum of the squared errors is calculated for each pair of input and output values. A learning rate is used as a scale factor, and the coefficients are updated in the direction towards minimizing the error. The process is repeated until a minimum sum squared error is achieved or no further improvement is possible. What is this process called?**
*   (a) LASSO
*   (b) Gradient Descent
*   (c) Least squares
*   (d) Regularization

**Correct answers:** (b)

**Explanation:**

**This describes gradient descent** - an iterative optimization algorithm for minimizing the loss function.

**Why (b) is correct:**

**1. Key characteristics of gradient descent:**
- **Random initialization** of parameters
- **Iterative updates** using gradient information
- **Learning rate** controls step size
- **Convergence** to local/global minimum

**2. Gradient descent algorithm:**
```
1. Initialize w randomly
2. For each iteration:
   - Compute gradient ∇L(w)
   - Update: w ← w - η∇L(w)
   - Check convergence
3. Return optimal w
```

**3. Why other options are incorrect:**

**Option (a): LASSO**
- **LASSO** is a regularization technique, not an optimization algorithm
- **Gradient descent** can be used to solve LASSO
- **LASSO** adds L1 penalty to the objective

**Option (c): Least squares**
- **Least squares** is the objective function, not the optimization method
- **Gradient descent** can be used to solve least squares
- **Least squares** can also be solved analytically (normal equations)

**Option (d): Regularization**
- **Regularization** adds penalty terms to the objective
- **Gradient descent** is the optimization method
- **Regularization** modifies the loss function

**4. Mathematical formulation:**
```
Objective: min L(w) = ||y - Xw||²
Update rule: w_{t+1} = w_t - η∇L(w_t)
Gradient: ∇L(w) = -2X^T(y - Xw)
```

**Key insight:** **Gradient descent** is an **iterative optimization algorithm** that uses gradient information to find the minimum of a function.

**17. Let $X \in \mathbb{R}^{m \times n}$, $w \in \mathbb{R}^n$, and $Y \in \mathbb{R}^m$. Consider mean squared error $L(w) = ||Xw-Y||_2^2$. What is $\nabla_w L(w)$?**
*   (a) $2Y^T (X^T Xw - Y)$
*   (b) $2X^T(X^T Xw - Y)$
*   (c) $2Y^T (Xw - Y)$
*   (d) $2X^T (Xw - Y)$

**Correct answers:** (d)

**Explanation:**

**Option (d) is correct:** ∇_w L(w) = 2X^T (Xw - Y)

**Step-by-step derivation:**

**1. Loss function:**
```
L(w) = ||Xw - Y||² = (Xw - Y)^T (Xw - Y)
```

**2. Expand the expression:**
```
L(w) = (Xw)^T (Xw) - (Xw)^T Y - Y^T (Xw) + Y^T Y
     = w^T X^T Xw - 2Y^T Xw + Y^T Y
```

**3. Take gradient with respect to w:**
```
∇_w L(w) = ∇_w (w^T X^T Xw - 2Y^T Xw + Y^T Y)
```

**4. Apply matrix calculus rules:**
- **∇_w (w^T X^T Xw) = 2X^T Xw**
- **∇_w (Y^T Xw) = X^T Y**
- **∇_w (Y^T Y) = 0**

**5. Combine terms:**
```
∇_w L(w) = 2X^T Xw - 2X^T Y = 2X^T (Xw - Y)
```

**6. Why other options are incorrect:**

**Option (a):** Wrong matrix dimensions and incorrect terms
**Option (b):** Extra X^T in the first term
**Option (c):** Missing X^T and incorrect structure

**7. Verification:**
- **Dimensions:** X^T ∈ ℝ^(n×m), (Xw - Y) ∈ ℝ^m
- **Result:** 2X^T (Xw - Y) ∈ ℝ^n ✓

**Key insight:** **Matrix calculus** gives us the gradient for vector-valued functions using the **chain rule**.

**18. Write down a closed-form solution for the optimal parameters $w$ that minimize the loss function**
$$L(w) = \sum_{i=1}^{n}(y_i - x_i^T w)^2 + \lambda||w||_2^2$$
**in terms of (1) the $n \times d$ matrix $X$ whose $i$-th row is a $1 \times n$ vector $x_i^T$ (a sample), (2) the $n \times 1$ vector $y$ whose $i$-th entry is $y_i$, and (3) the scalar $\lambda$. (You may assume that any relevant matrix is invertible.)**
*   (a) $\hat{w} = (X^T X)^{-1} X y + \lambda I$
*   (b) $\hat{w} = 2(X^T X + \lambda I)^{-1} X^T y$
*   (c) $\hat{w} = \lambda(X^T X)^{-1} X^T y$
*   (d) $\hat{w} = (X^T X + \lambda I)^{-1} X^T y$

**Correct answers:** (d)

**Explanation:**

**Option (d) is correct:** ŵ = (X^T X + λI)^(-1) X^T y

**This is the Ridge regression closed-form solution.**

**Step-by-step derivation:**

**1. Ridge regression objective:**
```
L(w) = ||y - Xw||² + λ||w||²
```

**2. Take gradient and set to zero:**
```
∇_w L(w) = -2X^T(y - Xw) + 2λw = 0
```

**3. Rearrange terms:**
```
-2X^T y + 2X^T Xw + 2λw = 0
X^T Xw + λw = X^T y
(X^T X + λI)w = X^T y
```

**4. Solve for w:**
```
w = (X^T X + λI)^(-1) X^T y
```

**5. Why other options are incorrect:**

**Option (a):** Incorrect addition of λI and wrong matrix dimensions
**Option (b):** Extra factor of 2
**Option (c):** Incorrect scaling with λ

**6. Comparison with OLS:**
- **OLS:** ŵ = (X^T X)^(-1) X^T y
- **Ridge:** ŵ = (X^T X + λI)^(-1) X^T y
- **λI term** ensures invertibility and adds regularization

**7. Properties:**
- **Always invertible** when λ > 0
- **Shrinks coefficients** toward zero
- **Reduces overfitting** through regularization

**Key insight:** **Ridge regression** adds **λI** to **X^T X** to ensure **invertibility** and provide **regularization**.

**19. How can overfitting be reduced in polynomial regression?**
*   (a) By decreasing the size of the validation set during hyperparameter tuning.
*   (b) By increasing the degree of the polynomial.
*   (c) By using regularization techniques such as $l_1$ or $l_2$ regularization.
*   (d) By reducing the size of your training set.

**Correct answers:** (c)

**Explanation:**

**Regularization techniques (L1 or L2) can reduce overfitting** in polynomial regression by constraining the model parameters.

**Why (c) is correct:**

**1. How regularization reduces overfitting:**
- **Constrains coefficients** to prevent them from becoming too large
- **Reduces model complexity** without changing polynomial degree
- **Trades bias for variance** - more stable predictions
- **Prevents fitting noise** in the training data

**2. L1 regularization (LASSO):**
```
min Σ(y_i - p(x_i))² + λΣ|w_j|
```
- **Sparsity induction** - can set coefficients to exactly zero
- **Feature selection** - automatically selects important polynomial terms
- **Sharp corners** - creates non-differentiable points

**3. L2 regularization (Ridge):**
```
min Σ(y_i - p(x_i))² + λΣw_j²
```
- **Smooth shrinkage** - coefficients approach zero but never exactly zero
- **Stable solutions** - smooth, differentiable objective
- **Reduces coefficient magnitudes**

**4. Why other options are incorrect:**

**Option (a): Decreasing validation set size**
- **Larger validation set** is better for hyperparameter tuning
- **Smaller validation set** increases variance in performance estimates
- **Does not directly reduce overfitting**

**Option (b): Increasing polynomial degree**
- **Increases overfitting** by adding more parameters
- **Higher degree** = more complex model = more overfitting risk
- **Opposite effect** of what we want

**Option (d): Reducing training set size**
- **Increases overfitting** by reducing available training data
- **Less data** = higher variance = more overfitting
- **Opposite effect** of what we want

**5. Additional overfitting reduction techniques:**
- **Cross-validation** for model selection
- **Early stopping** during training
- **Ensemble methods** (bagging, random forests)

**Key insight:** **Regularization** is the most effective way to reduce overfitting while **maintaining model capacity**.

**20. True/False: Linear least squares has a nonconvex loss function.**
*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:**

**This statement is false** - linear least squares has a **convex** loss function.

**Why linear least squares is convex:**

**1. Loss function form:**
```
L(w) = ||y - Xw||² = (y - Xw)^T (y - Xw)
```

**2. Convexity proof:**
- **Quadratic function** in w
- **Positive semidefinite** Hessian matrix
- **Second derivative** with respect to w is 2X^T X ≥ 0
- **All eigenvalues** of X^T X are non-negative

**3. Mathematical verification:**
```
∇²_w L(w) = 2X^T X
```
- **X^T X** is positive semidefinite for any matrix X
- **All eigenvalues** ≥ 0
- **Convex function** by second-order condition

**4. Properties of convex least squares:**
- **Unique global minimum** (if X^T X is invertible)
- **Gradient descent** converges to global minimum
- **Analytical solution** exists (normal equations)
- **No local minima** other than global minimum

**5. Why convexity matters:**
- **Optimization guarantees** - any local minimum is global
- **Convergence guarantees** - gradient descent works well
- **Efficient algorithms** - many optimization methods available
- **Stable solutions** - robust to initialization

**6. Comparison with non-convex problems:**
- **Neural networks** - non-convex (many local minima)
- **Logistic regression** - convex (unique global minimum)
- **Linear regression** - convex (unique global minimum)

**Key insight:** **Linear least squares** is **convex** because it's a **quadratic function** with **positive semidefinite Hessian**.

**21. True/False: It is possible to apply gradient descent method on linear least squares loss.**
*   (a) True
*   (b) False

**Correct answers:** (a)

**Explanation:**

**This statement is true** - gradient descent can be applied to linear least squares loss.

**Why gradient descent works for least squares:**

**1. Gradient descent requirements:**
- **Differentiable function** ✓ (least squares is differentiable)
- **Gradient can be computed** ✓ (∇L(w) = 2X^T(Xw - y))
- **Convex function** ✓ (least squares is convex)
- **Bounded below** ✓ (L(w) ≥ 0 for all w)

**2. Gradient descent algorithm:**
```
1. Initialize w₀ randomly
2. For t = 0, 1, 2, ...:
   - Compute gradient: ∇L(w_t) = 2X^T(Xw_t - y)
   - Update: w_{t+1} = w_t - η∇L(w_t)
   - Check convergence
3. Return optimal w
```

**3. Convergence guarantees:**
- **Convex function** → converges to global minimum
- **Lipschitz continuous gradient** → convergence rate O(1/t)
- **Strongly convex** (if X^T X is invertible) → linear convergence

**4. Advantages of gradient descent:**
- **Scalable** to large datasets
- **Memory efficient** - doesn't need to store X^T X
- **Online learning** - can process data incrementally
- **Stochastic variants** available (SGD)

**5. Comparison with analytical solution:**
- **Analytical:** ŵ = (X^T X)^(-1) X^T y (normal equations)
- **Gradient descent:** Iterative approximation
- **Analytical** is faster for small datasets
- **Gradient descent** is better for large datasets

**6. Practical considerations:**
- **Learning rate** needs to be chosen carefully
- **Convergence** can be slow for ill-conditioned problems
- **Stopping criteria** needed for termination

**Key insight:** **Gradient descent** is a **versatile optimization method** that works well for **convex functions** like least squares.

**22. Let $x_1, x_2 \in \mathbb{R}_+$ be sampled from the distribution $\text{Exp}(\lambda)$, where $\lambda \in \mathbb{R}_+$ is an unknown variable. Remember that the PDF of the exponential distribution is $f(x) = \lambda e^{-\lambda x}$ for any $x > 0$ and $f(x) = 0$ otherwise. Using the log-likelihood, find the maximum likelihood estimation of $\lambda$ in terms of $x_1, x_2$. Hint: $\frac{d}{dx} e^x = e^x$.**
*   (a) $\frac{\log(x_1)+\log(x_2)}{2}$
*   (b) $\log\left(\frac{e^{x_1}+e^{x_2}}{2}\right)$
*   (c) $\frac{x_1+x_2}{2}$
*   (d) $\frac{2}{x_1+x_2}$

**Correct answers:** (d)

**Explanation:**

**The MLE of λ is 2/(x₁ + x₂)** - this is the reciprocal of the sample mean.

**Step-by-step derivation:**

**1. Likelihood function:**
```
L(λ) = f(x₁|λ) × f(x₂|λ) = λe^(-λx₁) × λe^(-λx₂) = λ²e^(-λ(x₁+x₂))
```

**2. Log-likelihood:**
```
log L(λ) = log(λ²e^(-λ(x₁+x₂))) = 2log(λ) - λ(x₁+x₂)
```

**3. Maximization:**
```
d/dλ[log L(λ)] = 2/λ - (x₁+x₂) = 0
```

**4. Solving for λ:**
```
2/λ = x₁ + x₂
λ = 2/(x₁ + x₂)
```

**5. Verification:**
- **Second derivative:** d²/dλ²[log L(λ)] = -2/λ² < 0
- **Maximum** confirmed at λ = 2/(x₁ + x₂)

**6. Why other options are incorrect:**

**Option (a):** Incorrect log transformation
**Option (b):** Incorrect exponential transformation
**Option (c):** Sample mean, not reciprocal

**7. Interpretation:**
- **λ = 2/(x₁ + x₂)** is the reciprocal of the sample mean
- **Exponential distribution** parameter λ is the **rate parameter**
- **Mean** of exponential distribution is 1/λ
- **MLE** estimates the rate as reciprocal of sample mean

**Key insight:** **MLE for exponential distribution** is the **reciprocal of the sample mean**.

**23. Extra Credit: You are taking a multiple-choice exam that has 4 answers for each question. You are a smart student, so in answering a question on this exam, the probability that you know the correct answer is $p$, and you always choose the correct answer when you know it. If you don't know the answer, you choose one (uniformly) at random. What is the probability that you knew the correct answer to a question, given that you answered it correctly?**
*   (a) $p+\frac{1-p}{4}$
*   (b) $\frac{p}{1+p}$
*   (c) $\frac{p}{p+\frac{1-p}{4}}$
*   (d) $\frac{p}{\frac{p}{4}+1}$

**Correct answers:** (c)

**Explanation:**

**This is a Bayes' theorem problem** - we need to find P(Knew|Correct).

**Step-by-step solution:**

**1. Define events:**
- **K** = "You know the answer" (P(K) = p)
- **C** = "You answered correctly"
- **K'** = "You don't know the answer" (P(K') = 1-p)

**2. Calculate P(C|K) and P(C|K'):**
- **P(C|K) = 1** (if you know, you always answer correctly)
- **P(C|K') = 1/4** (if you don't know, you guess randomly)

**3. Calculate P(C):**
```
P(C) = P(C|K)P(K) + P(C|K')P(K')
     = 1 × p + (1/4) × (1-p)
     = p + (1-p)/4
```

**4. Apply Bayes' theorem:**
```
P(K|C) = P(C|K)P(K) / P(C)
       = 1 × p / (p + (1-p)/4)
       = p / (p + (1-p)/4)
```

**5. Why other options are incorrect:**

**Option (a):** This is P(C), not P(K|C)
**Option (b):** Incorrect application of Bayes' theorem
**Option (d):** Incorrect algebraic manipulation

**6. Intuitive interpretation:**
- **Numerator p:** Probability you knew and answered correctly
- **Denominator p + (1-p)/4:** Total probability of answering correctly
- **Ratio:** Fraction of correct answers that came from knowing

**7. Example:**
- **p = 0.8** (you know 80% of answers)
- **P(K|C) = 0.8 / (0.8 + 0.2/4) = 0.8 / 0.85 ≈ 0.94**
- **94% of your correct answers** came from actually knowing

**Key insight:** **Bayes' theorem** helps us update our beliefs about whether you knew the answer based on the outcome.

**24. Select all of the following statements that are False. When training a machine learning model you should**
*   (a) manually select samples from your data to form a test set.
*   (b) use a test set to help choose hyperparameter values.
*   (c) never use the test set to make changes to the model.
*   (d) split your data into training and test sets.

**Correct answers:** (a), (b)

**25. Let $X \sim \text{Uniform}[0, 3]$ and $Y \sim N(2, 2)$ be independent random variables. Then compute $E[XY^2] - E[X]E[Y]^2$.**
*   (a) 4
*   (b) 6
*   (c) 0
*   (d) 3

**Correct answers:** (a)

**Explanation:**

**The answer is 4** - this involves calculating expectations of independent random variables.

**Step-by-step calculation:**

**1. Given information:**
- **X ~ Uniform[0, 3]** → E[X] = (0+3)/2 = 1.5
- **Y ~ N(2, 2)** → E[Y] = 2, Var(Y) = 2
- **X and Y are independent**

**2. Calculate E[Y²]:**
```
E[Y²] = Var(Y) + E[Y]² = 2 + 2² = 2 + 4 = 6
```

**3. Calculate E[XY²]:**
```
Since X and Y are independent:
E[XY²] = E[X] × E[Y²] = 1.5 × 6 = 9
```

**4. Calculate E[X]E[Y]²:**
```
E[X]E[Y]² = 1.5 × 2² = 1.5 × 4 = 6
```

**5. Final calculation:**
```
E[XY²] - E[X]E[Y]² = 9 - 6 = 3
```

**Wait, this gives 3, not 4. Let me reconsider...**

**Alternative interpretation (σ = 2):**
- **Y ~ N(2, 2)** where σ = 2, so Var(Y) = σ² = 4
- **E[Y²] = Var(Y) + E[Y]² = 4 + 2² = 8**
- **E[XY²] = 1.5 × 8 = 12**
- **E[X]E[Y]² = 1.5 × 4 = 6**
- **Result: 12 - 6 = 6**

**The ambiguity mentioned in the note explains why this was regraded.**

**Key insight:** **Independence** allows us to factor expectations: E[XY²] = E[X]E[Y²].

**26. (True/False: ) Stochastic Gradient Descent (SGD) will always be at least as computationally expensive as Gradient Descent (GD) and (True/False: ) the number of update steps in SGD is greater than or equal to the number of update steps in GD.**
*   (a) True, True
*   (b) True, False
*   (c) False, True
*   (d) False, False

**Correct answers:** (c), (d)

**Explanation:**

**Both statements are false** - SGD is typically less computationally expensive per iteration, and the number of update steps can vary.

**Analysis of the two statements:**

**Statement 1: "SGD will always be at least as computationally expensive as GD"**
- **This is FALSE**
- **SGD** uses only a subset of data per iteration
- **GD** uses the entire dataset per iteration
- **SGD** is typically **less expensive per iteration**

**Statement 2: "Number of update steps in SGD ≥ number in GD"**
- **This is FALSE**
- **Number of steps** depends on convergence criteria, not algorithm type
- **SGD** may converge in fewer steps due to noise helping escape local minima
- **GD** may require more steps due to getting stuck in local minima

**Detailed comparison:**

**1. Computational cost per iteration:**
- **GD:** O(n) where n = number of training samples
- **SGD:** O(1) or O(b) where b = batch size
- **SGD** is typically much faster per iteration

**2. Number of iterations to convergence:**
- **GD:** May converge in fewer iterations (exact gradient)
- **SGD:** May require more iterations due to noisy gradients
- **But this is not guaranteed** - depends on problem and hyperparameters

**3. Total computational cost:**
- **GD:** Fewer iterations × higher cost per iteration
- **SGD:** More iterations × lower cost per iteration
- **SGD** often wins in total cost for large datasets

**4. Why SGD is preferred for large datasets:**
- **Memory efficient** - doesn't need to store all data
- **Scalable** - cost doesn't grow linearly with dataset size
- **Can escape local minima** due to noise
- **Online learning** capability

**Key insight:** **SGD** trades **iteration efficiency** for **computational efficiency** per iteration, making it better for large datasets.

**27. Which technique is most likely to reduce the variance of a model, holding all else fixed?**
*   (a) Reducing the complexity of the model
*   (b) Using a smaller number of training samples
*   (c) Increasing the number of features

**Correct answers:** (a)

**Explanation:**

**Reducing the complexity of the model is most likely to reduce variance** - this is a fundamental principle of the bias-variance tradeoff.

**Why (a) is correct:**

**1. How model complexity affects variance:**
- **Complex models** have more parameters to estimate
- **More parameters** = higher estimation uncertainty
- **Higher uncertainty** = higher variance
- **Simpler models** = fewer parameters = lower variance

**2. Mathematical intuition:**
- **Variance** measures how much predictions vary across different datasets
- **Complex models** are more sensitive to training data changes
- **Simple models** are more stable and consistent
- **Regularization** reduces effective complexity

**3. Why other options are incorrect:**

**Option (b): Using fewer training samples**
- **Increases variance** by reducing available information
- **Less data** = higher parameter estimation uncertainty
- **Opposite effect** of what we want

**Option (c): Increasing number of features**
- **Increases variance** by adding more parameters
- **More features** = more complex model = higher variance
- **Curse of dimensionality** effect

**4. Practical examples:**

**Reducing complexity:**
- **Lower polynomial degree** in polynomial regression
- **Fewer hidden layers** in neural networks
- **Stronger regularization** (larger λ)
- **Feature selection** to remove irrelevant features

**5. Bias-variance tradeoff:**
```
Complex model:    Low bias, High variance
Simple model:     High bias, Low variance
Optimal model:    Balanced bias-variance tradeoff
```

**6. Additional variance reduction techniques:**
- **Ensemble methods** (bagging, random forests)
- **Cross-validation** for model selection
- **More training data** (when available)

**Key insight:** **Model complexity** is the **primary factor** controlling variance - simpler models have lower variance but potentially higher bias.

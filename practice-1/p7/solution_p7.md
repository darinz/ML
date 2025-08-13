# Practice 7 Solutions

**Problem 1. In a machine learning classification problem, you have a dataset with two classes: Positive (P) and Negative (N). The probability of a randomly selected sample being Positive is $3/5$. The probability of a correct classification given that the sample is Positive is $4/5$, and the probability of a correct classification given that the sample is Negative is $7/10$. What is the probability that a randomly selected sample is Positive given that it has been classified as Positive? One Answer**

*   (a) $\frac{4}{5}$
*   (b) $\frac{12}{25}$
*   (c) $\frac{3}{5}$
*   (d) $\frac{12}{19}$

**Correct answers:** (a)

**Explanation:**

**This is a Bayes' theorem problem** - finding the probability of being Positive given a Positive classification.

**Step-by-step solution:**

**1. Define events:**
- **P** = "Sample is Positive" 
- **N** = "Sample is Negative"
- **CP** = "Classified as Positive"
- **CN** = "Classified as Negative"

**2. Given information:**
- **$P(P) = 3/5$** (probability of being Positive)
- **$P(N) = 1 - P(P) = 2/5$** (probability of being Negative)
- **$P(CP|P) = 4/5$** (correct classification given Positive)
- **$P(CN|N) = 7/10$** (correct classification given Negative)

**3. Calculate additional probabilities:**
- **$P(CN|P) = 1 - P(CP|P) = 1/5$** (incorrect classification given Positive)
- **$P(CP|N) = 1 - P(CN|N) = 3/10$** (incorrect classification given Negative)

**4. Apply Bayes' theorem:**
$P(P|CP) = \frac{P(CP|P)P(P)}{P(CP)}$

**5. Calculate $P(CP)$ using law of total probability:**
$P(CP) = P(CP|P)P(P) + P(CP|N)P(N)$

$P(CP) = (4/5)(3/5) + (3/10)(2/5)$

$P(CP) = 12/25 + 6/50 = 12/25 + 3/25 = 15/25 = 3/5$

**6. Substitute into Bayes' theorem:**
$P(P|CP) = \frac{(4/5)(3/5)}{3/5} = \frac{12/25}{3/5} = \frac{12/25}{15/25} = \frac{12}{15} = \frac{4}{5}$

**7. Verification:**
- **Numerator:** $(4/5)(3/5) = 12/25$ (true positives)
- **Denominator:** $3/5$ (all positive classifications)
- **Ratio:** $4/5 = 0.8$ (80% of positive classifications are actually positive)

**Key insight:** **High base rate** (60% positive) combined with **good classification accuracy** leads to high precision.

---

**Problem 2. Which of the following statements must be true for a square matrix A to have an inverse matrix $A^{-1}$?**

*   (a) A must be symmetric.
*   (b) The rank of A is less than its number of columns.
*   (c) A must have at least one column of 0s.
*   (d) The determinant of A is not equal to 0.

**Correct answers:** (d)

**Explanation:** A square matrix is invertible if and only if its determinant is non-zero, which is a fundamental theorem in linear algebra. Thus, choice (d) is correct. However, even if we forgot this fundamental theorems, we can use process-of-elimination. The symmetry of A has nothing to do with its inverse: imagine if A was all 0s; it's of course symmetric, but certainly non-invertible. Choices (b) and (c) being true would mean A has linearly dependent rows or columns, which cannot result in an invertible matrix.

**Problem 3. Consider the following system of linear equations:**

$$2x + 3y = 16$$
$$4x + 6y = 32$$

**Which of the following statements is true?**

*   (a) The system has an infinite number of solutions because the two equations are linearly dependent.
*   (b) The system has a unique solution because there are two equations for two unknowns.
*   (c) The system has no solution because the determinant of the coefficient matrix is zero.
*   (d) The system has no solution because the equations represent parallel lines that never intersect.

**Correct answers:** (a)

**Explanation:** The second equation is a multiple of the first, meaning they are linearly dependent and represent the same line. Since they are the same line, they intersect at every point, leading to an infinite number of solutions.

**Problem 4. For any function $f: \mathbb{R}^n \to \mathbb{R}$, the gradient is defined as:**

$$\nabla_w f(w) = \left[ \frac{\partial f(w)}{\partial w_1} \quad \dots \quad \frac{\partial f(w)}{\partial w_n} \right]^T$$

**What is the value of $\nabla_w (w^T Aw + u^T Bw + w^T Bv)$, given that $A, B \in \mathbb{R}^{n \times n}$, $A$ is symmetric, and $u, v \in \mathbb{R}^n$?**

*   (a) $Aw + Bu + B^T v$
*   (b) $2Aw + B^T u + Bv$
*   (c) $Aw + B^T u + Bv$
*   (d) $2Aw + Bu + B^T v$

**Correct answers:** (b)

**Explanation:**

**This is a matrix calculus problem** - computing the gradient of a quadratic form with linear terms.

**Step-by-step solution:**

**1. Break down the function:**
$f(w) = w^T Aw + u^T Bw + w^T Bv$

**2. Apply gradient rules:**

**For $w^T Aw$ (where A is symmetric):**
$\nabla_w(w^T Aw) = 2Aw$

**For $u^T Bw$:**
$\nabla_w(u^T Bw) = B^T u$

**For $w^T Bv$:**
$\nabla_w(w^T Bv) = Bv$

**3. Combine all terms:**
$\nabla_w f(w) = 2Aw + B^T u + Bv$

**4. Why other options are incorrect:**

**Option (a):** Missing factor of 2 for quadratic term
**Option (c):** Wrong transpose for $B^T u$ term
**Option (d):** Wrong transpose for $Bu$ term

**5. Key matrix calculus rules:**
- **$\nabla_w(w^T Aw) = 2Aw$** (when A is symmetric)
- **$\nabla_w(u^T Bw) = B^T u$**
- **$\nabla_w(w^T Bv) = Bv$**

**Key insight:** **Matrix calculus** requires careful attention to **transpose operations** and **symmetry assumptions**.

---

**Problem 5. Which of the following statements is most accurate regarding the principle of Maximum Likelihood Estimation (MLE) in statistical modeling?**

*   (a) MLE identifies model parameters that maximize the probability of the observed data under the model.
*   (b) MLE directly computes the probability of parameters being correct, independent of observed data.
*   (c) MLE is primarily concerned with minimizing the variance of parameter estimates for model stability.
*   (d) MLE identifies model parameters that minimize the squared prediction error over the training data.

**Correct answers:** (a)

**Explanation:** MLE aims to maximize the probability of observing the given data under different model parameter values.

**Problem 6. A machine learning engineer models the number of website requests per hour using a Poisson distribution. Over 6 hours, the observed requests are 4, 5, 6, 7, 8, and 9. Recall that the probability mass function for a Poisson distribution with parameter $\lambda$ is:**

$$P(x|\lambda) = e^{-\lambda} \frac{\lambda^x}{x!}$$

**What is the maximum likelihood estimation of the rate parameter $\lambda$ of this Poisson distribution?**

*   (a) $e^{\frac{39}{6}}$
*   (b) $\sqrt{\frac{39}{6}}$
*   (b) $\sqrt{\frac{39}{6}}$
*   (c) 6
*   (d) $\frac{39}{6}$

**Correct answers:** (d)

**Explanation:** The MLE estimate of $\lambda$ for a Poisson distribution is simply the mean of the observed data. The mean is $\frac{39}{6} = 6.5$.

**Problem 7. Assume a simple linear model $Y = Xw$. For simplicity, no intercept is considered. Given the following dataset:**

$$X = \begin{bmatrix} 1 & 0 \\ 2 & 2 \end{bmatrix}$$
$$Y = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$$

**(a) Compute the least squares estimate of $w$ without any regularization. You may leave your answer as a fraction, if necessary.**

**Hint:** if $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, then $A^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$

**Answer:** $\hat{w} = \begin{bmatrix} 2 \\ -0.5 \end{bmatrix}$

**(b) Predict $\hat{Y}$ for $X = \begin{bmatrix} 6 \\ 7 \end{bmatrix}$.**

**Answer:** $\hat{Y} = 8.5$

**Explanation:** 

**This is a linear regression problem** - computing least squares estimates and making predictions.

**Step-by-step solution:**

**Part (a): Computing $\hat{w}$**

**1. Set up the normal equations:**
$\hat{w} = (X^T X)^{-1} X^T Y$

**2. Compute $X^T X$:**
$X^T X = \begin{bmatrix} 1 & 2 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 2 & 2 \end{bmatrix} = \begin{bmatrix} 5 & 4 \\ 4 & 4 \end{bmatrix}$

**3. Compute $(X^T X)^{-1}$:**
Using the hint: $ad-bc = 5(4) - 4(4) = 20 - 16 = 4$

$(X^T X)^{-1} = \frac{1}{4} \begin{bmatrix} 4 & -4 \\ -4 & 5 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ -1 & 1.25 \end{bmatrix}$

**4. Compute $X^T Y$:**
$X^T Y = \begin{bmatrix} 1 & 2 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 8 \\ 6 \end{bmatrix}$

**5. Compute $\hat{w}$:**
$\hat{w} = \begin{bmatrix} 1 & -1 \\ -1 & 1.25 \end{bmatrix} \begin{bmatrix} 8 \\ 6 \end{bmatrix} = \begin{bmatrix} 8-6 \\ -8+7.5 \end{bmatrix} = \begin{bmatrix} 2 \\ -0.5 \end{bmatrix}$

**Part (b): Making prediction**

**6. For new input $x = \begin{bmatrix} 6 \\ 7 \end{bmatrix}$:**
$\hat{Y} = \hat{w}^T x = \begin{bmatrix} 2 & -0.5 \end{bmatrix} \begin{bmatrix} 6 \\ 7 \end{bmatrix} = 2(6) + (-0.5)(7) = 12 - 3.5 = 8.5$

**Key insight:** **Normal equations** provide the **closed-form solution** for linear regression without regularization.

---

**Problem 8. You have access to data points $\{(x_i, y_i)\}_{i=1}^n$, where $x_i$ are $d$-dimensional vectors ($x_i \in \mathbb{R}^d$) and $y_i$ are scalars ($y_i \in \mathbb{R}$). Additionally, you have weights $\{w_i\}_{i=1}^n$, where $w_i \in \mathbb{R}$ and $w_i > 0$, representing the "importance" of each data point. You want to solve the weighted least squares regression problem:**

$$\hat{\theta} = \arg \min_{\theta \in \mathbb{R}^d} \sum_{i=1}^n w_i (x_i^T \theta - y_i)^2$$

**Let us define the matrices:**

$$X = \begin{bmatrix} x_1^T \\ \vdots \\ x_n^T \end{bmatrix} \in \mathbb{R}^{n \times d}$$
$$Y = \begin{bmatrix} y_1 \\ \vdots \\ y_n \end{bmatrix} \in \mathbb{R}^n$$
$$W = \begin{bmatrix} w_1 & 0 & \dots & 0 \\ 0 & w_2 & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & w_n \end{bmatrix} \in \mathbb{R}^{n \times n}$$

**What is $\hat{\theta}$ in terms of $X$, $Y$, and $W$?**

*   (a) $(X^T X)^{-1} X^T W^{-1} Y$
*   (b) $W(X^T X)^{-1} X^T Y$
*   (c) $(X^T X)^{-1} X^T W Y$
*   (d) $(X^T W X)^{-1} X^T Y$
*   (e) $(X^T W X)^{-1} X^T W Y$

**Correct answers:** (e)

**Explanation:** The solution can be found by taking the gradient of the optimization algorithm, setting it to 0, and solving for $\theta$. We can convert the objective function to matrix notation: $\arg \min_{\theta} (X\theta - Y)^T W (X\theta - Y)$.

**Problem 9. In the context of least squares regression, how does the presence of high noise levels in the data impact the reliability of the model's parameter estimates?**

*   (a) High noise levels predominantly affect the intercept term of the regression model, but leave the slope estimates relatively unaffected.
*   (b) High noise levels can increase the variability of the parameter estimates, potentially leading to a model that captures random noise rather than the true underlying relationship.
*   (c) High noise levels decrease the variance of the estimated parameters, making the model more robust.
*   (d) Noise in the data generally has minimal impact on the least squares estimates since the method inherently separates signal from noise in most scenarios.

**Correct answers:** (b)

**Explanation:** In least squares regression, high noise levels can lead to overfitting, where the model erroneously adjusts its parameters to account for these random fluctuations, resulting in a model that performs well on the training data but poorly on unseen data. This reduces the model's ability to generalize and accurately predict outcomes on new, unseen data.

**Problem 10. In linear regression analysis using the least squares method, how might outliers in the dataset impact the resulting regression line?**

*   (a) Outliers affect only the precision of the prediction intervals, not the regression line itself.
*   (b) Outliers enhance the model's accuracy by providing a wider range of data points.
*   (c) Outliers can significantly skew the regression line, potentially leading to an inaccurate representation of the overall data trend.
*   (d) Outliers have a minimal impact, as the least squares method averages out their effects.

**Correct answers:** (c)

**Explanation:** Least squares aims to minimize the sum of the squared differences between observed and predicted values. Outliers, which are very distant from other data points, can cause the squared differences to become substantially larger, and consequently "pull" the regression line to themselves. This can lead to a skewed line that does not accurately represent the underlying trend of the majority of the data, affecting the model's predictive accuracy.

**Problem 11. How does increasing the complexity of a model typically affect the properties of that model? Select all that apply.**

*   (a) It tends to decrease bias but increase variance, potentially leading to overfitting.
*   (b) It can increase training accuracy.
*   (c) It tends to increase both bias and variance.
*   (d) It tends to decrease variance but increase bias, potentially leading to underfitting.

**Correct answers:** (a), (b)

**Explanation:**

**This question tests understanding of the bias-variance tradeoff** and how model complexity affects model properties.

**Why (a) and (b) are correct:**

**Option (a): Decreases bias, increases variance**
- **Increasing complexity** allows model to capture more patterns
- **Lower bias** = model can fit more complex relationships
- **Higher variance** = model becomes more sensitive to training data
- **Overfitting risk** increases with complexity

**Option (b): Can increase training accuracy**
- **More complex models** can fit training data better
- **Higher training accuracy** is often achieved
- **But this doesn't guarantee** better generalization

**Why other options are incorrect:**

**Option (c): Increases both bias and variance**
- **Contradicts** the bias-variance tradeoff
- **Complexity typically** decreases bias, increases variance
- **Not a typical** relationship

**Option (d): Decreases variance, increases bias**
- **This describes** underfitting scenario
- **Simple models** have high bias, low variance
- **Not what happens** when increasing complexity

**Key insight:** **Model complexity** follows a **bias-variance tradeoff** - decreasing bias while increasing variance.

---

**Problem 12. True/False: A model with high variance tends to perform well on both the training and test data.**

*   (a) False
*   (b) True

**Correct answers:** (a)

**Explanation:**

**This question tests understanding of high variance models** and their performance characteristics.

**Why (a) is correct:**

**High variance models typically:**
- **Perform well on training data** (they fit it closely)
- **Perform poorly on test data** (they don't generalize well)
- **Are overfitted** to training data
- **Have low bias but high variance**

**Mathematical intuition:**
- **High variance** = model predictions vary significantly with different training sets
- **Good training performance** = model fits training data well
- **Poor test performance** = model doesn't generalize to unseen data
- **This is the classic** overfitting scenario

**Why (b) is incorrect:**
- **High variance models** do NOT perform well on test data
- **They overfit** to training data
- **Poor generalization** is the hallmark of high variance

**Key insight:** **High variance** models **memorize training data** but **fail to generalize** to new data.

---

**Problem 13. The plots below show fits (in black) to the data points ("x" symbols in grey), using several different basis functions:**

<img src="./plots.png" width="550px">

**For each plot, please identify the basis function used:**

**Plot (a): _____, Plot (b): _____, Plot (c): _____, Plot (d): _____**

**Basis functions used:**
1. $h_1(x) = [1,x]$
2. $h_2(x) = [1,x,x^2]$
3. $h_3(x) = [1,x,x^2,x^3]$
4. $h_4(x) = [1, \sin(\frac{4\pi}{25}x)]$

**Explanation:** 

**This question tests understanding of basis functions** and their role in capturing non-linear relationships.

**Step-by-step analysis:**

**Plot (a):** Shows a **sinusoidal pattern** - matches $h_4(x) = [1, \sin(\frac{4\pi}{25}x)]$

**Plot (b):** Shows a **quadratic curve** - matches $h_2(x) = [1,x,x^2]$

**Plot (c):** Shows a **linear relationship** - matches $h_1(x) = [1,x]$

**Plot (d):** Shows a **cubic curve** - matches $h_3(x) = [1,x,x^2,x^3]$

**Key insights:**
- **Basis functions** transform input space to capture non-linear patterns
- **Higher degree polynomials** can fit more complex curves
- **Sinusoidal functions** capture periodic patterns
- **Linear basis** captures only linear relationships

**Answer:** Plot (a): $h_4$, Plot (b): $h_2$, Plot (c): $h_1$, Plot (d): $h_3$

---

**Problem 14. What is the purpose of general basis functions in linear regression?**

*   (a) To increase convergence speed in gradient descent.
*   (b) To encourage sparsity in learned weights.
*   (c) To minimize computational complexity.
*   (d) To transform input data into a higher-dimensional space to capture non-linear relationships.

**Correct answers:** (d)

**Explanation:**

**This question tests understanding of basis functions** and their role in linear regression.

**Why (d) is correct:**

**Basis functions serve to:**
- **Transform input data** into higher-dimensional space
- **Capture non-linear relationships** in data
- **Enable linear models** to fit non-linear patterns
- **Expand feature space** without changing the linear nature of the model

**Mathematical intuition:**
- **Original model:** $y = w^T x$ (linear in $x$)
- **With basis functions:** $y = w^T \phi(x)$ (linear in $\phi(x)$, non-linear in $x$)
- **$\phi(x)$** transforms input to capture non-linear patterns

**Why other options are incorrect:**

**Option (a): Convergence speed**
- **Basis functions** don't directly affect convergence
- **Convergence** depends on optimization algorithm
- **Not the primary purpose**

**Option (b): Sparsity**
- **Basis functions** don't encourage sparsity
- **Regularization** (L1/L2) encourages sparsity
- **Different concept**

**Option (c): Computational complexity**
- **Basis functions** often **increase** complexity
- **More features** = more computation
- **Not a benefit**

**Key insight:** **Basis functions** enable **linear models** to capture **non-linear relationships** through **feature transformation**.

---

**Problem 15. What is the best description of 'irreducible error' in a machine learning predictor?**

*   (a) It's due to inherent noise that cannot be eliminated by any model.
*   (b) It's minimized by cross-validation.
*   (c) It can be minimized by increasing training data size.
*   (d) It arises from feature engineering or irrelevant features.

**Correct answers:** (a)

**Explanation:**

**This question tests understanding of irreducible error** - the fundamental limit on model performance.

**Why (a) is correct:**

**Irreducible error is:**
- **Inherent noise** in the data generation process
- **Cannot be eliminated** by any model, no matter how complex
- **Lower bound** on prediction error
- **Independent** of model choice or training data size

**Mathematical intuition:**
- **True function:** $g(x) = 7x^2 + \epsilon$
- **Noise term:** $\epsilon \sim \mathcal{N}(0, 4)$
- **Irreducible error:** $E[\epsilon^2] = \text{Var}(\epsilon) = 4$
- **No model** can predict $\epsilon$ perfectly

**Why other options are incorrect:**

**Option (b): Minimized by cross-validation**
- **Cross-validation** estimates model performance
- **Doesn't reduce** irreducible error
- **Different concept**

**Option (c): Minimized by more data**
- **More data** reduces variance, not irreducible error
- **Irreducible error** is independent of data size
- **Fundamental limit**

**Option (d): Arises from features**
- **Feature engineering** affects model bias/variance
- **Not the source** of irreducible error
- **Different error component**

**Key insight:** **Irreducible error** is the **fundamental noise** in the data that **no model can eliminate**.

---

**Problem 16. A polynomial regression model of degree $d=3$ approximates a quadratic function $g(x) = 7x^2 + \epsilon$, where $\epsilon$ is a Gaussian random variable with mean $\mu=0$ and variance $\sigma^2=4$. What is the irreducible error?**

*   (a) 2
*   (b) 0
*   (c) 4
*   (d) $x^3$

**Correct answers:** (c)

**Explanation:**

**This question tests understanding of irreducible error** in the context of a specific model.

**Why (c) is correct:**

**Given the setup:**
- **True function:** $g(x) = 7x^2 + \epsilon$
- **Noise:** $\epsilon \sim \mathcal{N}(0, 4)$
- **Irreducible error:** $E[\epsilon^2] = \text{Var}(\epsilon) = 4$

**Mathematical reasoning:**
- **No matter how well** the model fits $7x^2$
- **The noise term** $\epsilon$ cannot be predicted
- **Expected squared error** from noise is $E[\epsilon^2] = 4$
- **This is the irreducible error**

**Why other options are incorrect:**

**Option (a): 2**
- **This would be** $\sqrt{\text{Var}(\epsilon)}$
- **Standard deviation** is 2, not irreducible error
- **Irreducible error** is variance, not standard deviation

**Option (b): 0**
- **Would mean** no noise in the system
- **Contradicts** the given $\epsilon \sim \mathcal{N}(0, 4)$
- **Not realistic**

**Option (d): $x^3$**
- **This is a function** of $x$, not a constant
- **Irreducible error** is independent of $x$
- **Nonsensical** answer

**Key insight:** **Irreducible error** equals the **variance of the noise** in the data generation process.

---

**Problem 17. True/False: Increasing the proportion of your dataset allocated to training (as opposed to testing) will guarantee better performance on unseen data.**

*   (a) False
*   (b) True

**Correct answers:** (a)

**Explanation:**

**This question tests understanding of training-test split** and its impact on model evaluation.

**Why (a) is correct:**

**Increasing training data does NOT guarantee better performance because:**

**1. Quality vs Quantity:**
- **More data** doesn't guarantee **better data**
- **Poor quality data** can hurt performance
- **Data distribution** matters more than size

**2. Overfitting risk:**
- **More training data** can lead to overfitting
- **Model complexity** should match data size
- **Validation** is still needed

**3. Data leakage:**
- **Improper splits** can cause data leakage
- **Test data contamination** leads to optimistic estimates
- **Proper separation** is crucial

**4. Model capacity:**
- **Simple models** may not benefit from more data
- **Complex models** need sufficient data
- **Match model complexity** to data size

**Why (b) is incorrect:**
- **More training data** typically helps, but doesn't guarantee improvement
- **Depends on** data quality, model choice, and proper evaluation
- **Not an absolute** guarantee

**Key insight:** **Data quality** and **proper evaluation** are more important than **data quantity** alone.

---

**Problem 18. Which of the following statements best describes a potential issue that can arise if the test dataset is not properly separated from the training dataset?**

*   (a) The model will always underfit, regardless of the algorithm used.
*   (b) The model will always overfit, regardless of the algorithm used.
*   (c) The evaluation metrics will tend to overestimate the prediction error on unseen data.
*   (d) The test data will influence the training process, leading to an overly optimistic estimate of the model's performance on new, unseen data.
*   (e) The model's computational complexity will significantly increase, resulting in longer training times.

**Correct answers:** (d)

**Explanation:**

**This question tests understanding of data leakage** and proper test set separation.

**Why (d) is correct:**

**Data leakage occurs when:**
- **Test data influences** training process
- **Information from test set** leaks into model development
- **Overly optimistic** performance estimates
- **Poor generalization** to truly unseen data

**Common causes of data leakage:**
- **Preprocessing entire dataset** before splitting
- **Feature selection** using all data
- **Hyperparameter tuning** on test set
- **Model selection** using test set

**Why other options are incorrect:**

**Option (a): Always underfit**
- **Data leakage** doesn't guarantee underfitting
- **Can cause** overfitting to leaked information
- **Depends on** the specific leakage

**Option (b): Always overfit**
- **Not always** the case
- **Depends on** nature of leakage
- **Can cause** various issues

**Option (c): Overestimate error**
- **Data leakage** typically **underestimates** error
- **Leads to optimistic** performance estimates
- **Opposite** of what happens

**Option (e): Increase complexity**
- **Data leakage** doesn't affect computational complexity
- **Affects** performance estimates, not training time
- **Unrelated** to complexity

**Key insight:** **Data leakage** leads to **overly optimistic** performance estimates by **contaminating** the training process.

---

**Problem 19. How should data preprocessing be applied when using k-fold cross-validation? Select the most accurate answer.**

*   (a) Preprocess the entire dataset before splitting into folds to maintain consistency.
*   (b) Avoid preprocessing as it can bias the cross-validation results.
*   (c) Only preprocess the test folds and train our model on raw (unprocessed) data.
*   (d) Apply preprocessing separately on each iteration of k-fold validation to avoid data leakage.

**Correct answers:** (d)

**Explanation:**

**This question tests understanding of proper preprocessing** in cross-validation to avoid data leakage.

**Why (d) is correct:**

**Proper preprocessing in k-fold CV:**
- **Apply preprocessing separately** on each fold
- **Use only training fold** to compute statistics (mean, std, etc.)
- **Apply same transformation** to validation fold
- **Prevents data leakage** from validation set

**Why other options are incorrect:**

**Option (a): Preprocess entire dataset**
- **Causes data leakage** - validation data influences preprocessing
- **Overly optimistic** performance estimates
- **Violates** independence principle

**Option (b): Avoid preprocessing**
- **Preprocessing is often necessary** (scaling, normalization)
- **Avoiding it** can hurt model performance
- **Not a solution** to data leakage

**Option (c): Only preprocess test folds**
- **Incorrect approach** - should preprocess training data
- **Test folds** should use training-derived transformations
- **Backwards** logic

**Key insight:** **Preprocessing must be applied separately** on each fold to **prevent data leakage** and maintain **proper evaluation**.

---

**Problem 20. What is the main advantage of using k-fold cross-validation? One Answer**

*   (a) It guarantees improvement in model accuracy on unseen data.
*   (b) It provides an estimate of model performance for given hyperparameters.
*   (c) It significantly reduces the training time of the model by dividing the dataset into smaller parts.
*   (d) It eliminates the need for a separate test dataset.

**Correct answers:** (b)

**Explanation:**

**This question tests understanding of k-fold cross-validation** and its primary benefits.

**Why (b) is correct:**

**k-fold cross-validation provides:**
- **Robust performance estimates** for given hyperparameters
- **Multiple evaluations** on different data splits
- **Better generalization** estimates than single train-test split
- **Statistical confidence** in model performance

**Why other options are incorrect:**

**Option (a): Guarantees improvement**
- **CV doesn't guarantee** better performance
- **It estimates** performance more accurately
- **Model choice** still matters

**Option (c): Reduces training time**
- **CV actually increases** total training time
- **Multiple models** must be trained
- **Computational overhead** is higher

**Option (d): Eliminates test set**
- **CV doesn't eliminate** need for test set
- **CV is for** hyperparameter tuning
- **Final evaluation** still needs held-out test set

**Key insight:** **k-fold CV** provides **reliable performance estimates** for **hyperparameter selection** and **model comparison**.

---

**Problem 21. In Lasso regression, how does the regularization parameter $\lambda$ influence the risk of overfitting? Select all that apply.**

*   (a) Increasing $\lambda$ always increases the risk of overfitting as it leads to higher model complexity.
*   (b) Decreasing $\lambda$ to zero may increase the risk of overfitting.
*   (c) Increasing $\lambda$ typically reduces the risk of overfitting by increasing sparsity.
*   (d) The choice of $\lambda$ in Ridge regression has no impact on the risk of overfitting.

**Correct answers:** (b), (c)

**Explanation:**

**This question tests understanding of Lasso regularization** and its effect on overfitting.

**Why (b) and (c) are correct:**

**Option (b): Decreasing $\lambda$ to zero increases overfitting risk**
- **$\lambda = 0$** reduces to ordinary least squares
- **No regularization** = higher model complexity
- **Increased risk** of overfitting to training data
- **Poor generalization** to unseen data

**Option (c): Increasing $\lambda$ reduces overfitting risk**
- **Higher $\lambda$** = stronger regularization
- **More coefficients** set to exactly zero
- **Reduced model complexity**
- **Better generalization** (up to a point)

**Why other options are incorrect:**

**Option (a): Increasing $\lambda$ always increases overfitting**
- **Contradicts** the purpose of regularization
- **Higher $\lambda$** reduces model complexity
- **Should reduce** overfitting risk

**Option (d): $\lambda$ in Ridge has no impact**
- **Ridge regression** also uses $\lambda$ for regularization
- **$\lambda$ affects** both Lasso and Ridge
- **Different penalty** functions but same concept

**Key insight:** **Lasso regularization** controls overfitting through **sparsity induction** - higher $\lambda$ = less overfitting.

---

**Problem 22. When comparing Lasso regression to Ridge regression, which of the following properties are true about Lasso regression? Select all that apply.**

*   (a) Lasso regression can be used to select the most important features of a dataset.
*   (b) Lasso regression tends to retain all features but with smaller coefficients.
*   (c) Lasso regression is always better suited for handling high-dimensional data with a large number of features.
*   (d) Lasso regression has fewer hyperparameters to tune.

**Correct answers:** (a)

**Explanation:**

**This question tests understanding of Lasso vs Ridge** regression properties.

**Why (a) is correct:**

**Lasso regression can perform feature selection because:**
- **L1 penalty** can set coefficients exactly to zero
- **Sparsity induction** automatically selects relevant features
- **Irrelevant features** get zero weights
- **Automatic feature selection** is built-in

**Why other options are incorrect:**

**Option (b): Retains all features with smaller coefficients**
- **This describes Ridge regression** (L2 penalty)
- **Ridge shrinks** coefficients but rarely to zero
- **Lasso** sets coefficients exactly to zero

**Option (c): Always better for high-dimensional data**
- **Not always** - depends on data characteristics
- **Ridge** can be better in some cases
- **No universal** superiority

**Option (d): Fewer hyperparameters**
- **Both Lasso and Ridge** have same number of hyperparameters
- **Both use** $\lambda$ regularization parameter
- **No difference** in hyperparameter count

**Key insight:** **Lasso's L1 penalty** enables **automatic feature selection** by setting coefficients exactly to zero.

---

**Problem 23. A student is using ridge regression for housing price prediction. They notice that increasing the regularization strength improves validation set performance but worsens training set performance. What does this suggest about the model before adjusting regularization?**

*   (a) The choice of features was inappropriate.
*   (b) The model was underfitting the training data.
*   (c) The regularization strength was too high.
*   (d) The model was overfitting the training data.

**Correct answers:** (d)

**Explanation:** The model was likely overfitting, capturing noise. Increasing regularization helps mitigate overfitting by penalizing large coefficients, leading to better generalization on unseen data (validation set).

**Problem 24. For a twice-differentiable convex function $f: \mathbb{R}^d \to \mathbb{R}$, what are the properties of the Hessian matrix, $\nabla^2 f(x) \in \mathbb{R}^{d \times d}$?**

**Hint:** Consider the $d=1$ case (second derivative and shape).

*   (a) $\nabla^2 f(x)$ is negative semi-definite.
*   (b) $\nabla^2 f(x)$ is negative definite.
*   (c) $\nabla^2 f(x)$ is positive definite.
*   (d) $\nabla^2 f(x)$ is positive semi-definite.

**Correct answers:** (d)

**Explanation:**

**This question tests understanding of convex functions** and their Hessian properties.

**Why (d) is correct:**

**For a convex function $f(x)$:**
- **Second derivative** $f''(x) \geq 0$ (in 1D)
- **Hessian matrix** $\nabla^2 f(x) \succeq 0$ (positive semi-definite)
- **All eigenvalues** are non-negative
- **Curvature** is non-negative everywhere

**Mathematical intuition:**
- **Convex function** curves upward or is flat
- **Second derivative** measures curvature
- **Positive semi-definite** = non-negative curvature
- **Can be flat** (eigenvalue = 0) but never curves down

**Why other options are incorrect:**

**Option (a): Negative semi-definite**
- **This would be** concave function
- **Opposite** of convex
- **Curves downward**

**Option (b): Negative definite**
- **Strictly concave** function
- **All eigenvalues** negative
- **Not convex**

**Option (c): Positive definite**
- **Strictly convex** function
- **All eigenvalues** positive
- **More restrictive** than convex

**Key insight:** **Convex functions** have **non-negative curvature** everywhere, making their Hessian **positive semi-definite**.

---

**Problem 25. True/False: A solution to a convex optimization problem is guaranteed to be a global minimum.**

*   (a) True
*   (b) False

**Correct answers:** (a)

**Explanation:** 

**This question tests understanding of convex optimization** and global optimality.

**Why (a) is correct:**

**For convex optimization problems:**
- **Any local minimum** is also a global minimum
- **No local minima** that aren't global
- **Gradient-based methods** converge to global optimum
- **Convexity guarantees** global optimality

**Mathematical intuition:**
- **Convex function** has no "valleys" or local minima
- **Any point** where gradient is zero is global minimum
- **No risk** of getting stuck in local minimum
- **Convexity** eliminates local vs global distinction

**Why (b) is incorrect:**
- **Convexity** doesn't guarantee uniqueness
- **Multiple solutions** can exist (e.g., flat regions)
- **Global minimum** can be attained at multiple points
- **Uniqueness** requires additional conditions

**Key insight:** **Convexity guarantees global optimality** but **not uniqueness** of the solution.

---

**Problem 26. True/False: A solution to a convex optimization problem is guaranteed to be unique.**

*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:** The solution, while having minimal value, will not necessarily be unique. Consider the case of Least Squares with fewer data points than dimensions. There are an infinite number of solutions with the minimum value.

**Problem 27. True/False: A convex optimization problem is guaranteed to have a closed-form solution.**

*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:**

**This question tests understanding of convex optimization** and solution methods.

**Why (b) is correct:**

**Convex optimization does NOT guarantee closed-form solutions because:**

**1. Many convex problems require iterative methods:**
- **Large-scale problems** (e.g., neural networks)
- **Complex constraints** that can't be solved analytically
- **Non-differentiable** convex functions
- **High-dimensional** optimization problems

**2. Examples requiring iterative methods:**
- **Support Vector Machines** with kernel functions
- **Logistic regression** with large datasets
- **Neural network training**
- **Lasso/Ridge regression** with many features

**3. When closed-form solutions exist:**
- **Linear regression** (normal equations)
- **Simple quadratic** optimization problems
- **Small-scale** problems with simple constraints
- **Special cases** with analytical solutions

**Why (a) is incorrect:**
- **Many convex problems** require iterative optimization
- **Closed-form solutions** are the exception, not the rule
- **Computational complexity** often makes iterative methods necessary

**Key insight:** **Convexity guarantees global optimality** but **not closed-form solutions** - many convex problems require iterative optimization methods.

---

**Problem 28. Briefly explain the main difference between Mini Batch Gradient Descent and Stochastic Gradient Descent. Then, describe one main advantage of using Mini Batch Gradient Descent over SGD.**

**Answer:**

**Explanation:**

**Main Difference:** The main difference is that SGD uses a single training point to estimate the gradient, while Mini Batch chooses a set of $B$ training points (for some chosen constant $B$).

**Main Advantage:** The main advantage of Mini Batch GD is that by using more points in the gradient estimation, we get a less noisy estimate which improves convergence.

**Problem 29. True/False: Stochastic gradient descent provides biased estimates of the true gradient at each step.**

*   (a) True
*   (b) False

**Correct answers:** (b)

**Explanation:**

**This question tests understanding of SGD gradient estimates** and their bias properties.

**Why (b) is correct:**

**SGD provides UNBIASED estimates of the true gradient:**

**1. Unbiased gradient estimates:**
- **$E[\nabla L_i(\theta)] = \nabla L(\theta)$** for random sample $i$
- **Expected value** of SGD gradient equals true gradient
- **No systematic bias** in gradient estimates
- **Random sampling** preserves unbiasedness

**2. Mathematical intuition:**
- **True gradient:** $\nabla L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla L_i(\theta)$
- **SGD gradient:** $\nabla L_i(\theta)$ for random $i$
- **$E[\nabla L_i(\theta)] = \frac{1}{n} \sum_{i=1}^{n} \nabla L_i(\theta) = \nabla L(\theta)$**
- **Unbiased** but high variance

**3. Why SGD works:**
- **Unbiased estimates** ensure convergence to true optimum
- **High variance** is the trade-off for computational efficiency
- **Law of large numbers** ensures convergence over many steps

**Why (a) is incorrect:**
- **SGD gradients** are unbiased, not biased
- **Bias would prevent** convergence to true optimum
- **Unbiasedness** is crucial for SGD convergence

**Key insight:** **SGD provides unbiased gradient estimates** with **high variance** - the trade-off for computational efficiency.

---

**Problem 30. Consider some function $f(x): \mathbb{R}^d \to \mathbb{R}$, and assume that we want to run an iterative algorithm to find the maximizer of $f$. Which update rule should we use to do this (for some $\eta > 0$)?**

*   (a) $x_{t+1} \leftarrow -x_t + \eta \cdot \nabla_x f(x_t)$
*   (b) $x_{t+1} \leftarrow x_t - \eta \cdot \nabla_x f(x_t)$
*   (c) $x_{t+1} \leftarrow -x_t - \eta \cdot \nabla_x f(x_t)$
*   (d) $x_{t+1} \leftarrow x_t + \eta \cdot \nabla_x f(x_t)$

**Correct answers:** (d)

**Explanation:**

**This question tests understanding of optimization** and gradient ascent for maximization.

**Why (d) is correct:**

**For maximizing a function $f(x)$:**

**1. Gradient ascent rule:**
$x_{t+1} = x_t + \eta \cdot \nabla_x f(x_t)$

**2. Mathematical intuition:**
- **Gradient points** in direction of steepest ascent
- **Adding gradient** moves toward maximum
- **Learning rate $\eta$** controls step size
- **Positive sign** for maximization

**3. Why other options are incorrect:**

**Option (a):** $x_{t+1} = -x_t + \eta \cdot \nabla_x f(x_t)$
- **Negative $x_t$** doesn't make sense for optimization
- **Wrong direction** for maximization

**Option (b):** $x_{t+1} = x_t - \eta \cdot \nabla_x f(x_t)$
- **This is gradient descent** (for minimization)
- **Wrong direction** for maximization

**Option (c):** $x_{t+1} = -x_t - \eta \cdot \nabla_x f(x_t)$
- **Both negative signs** are incorrect
- **Neither direction** nor update makes sense

**4. Key principle:**
- **Gradient ascent:** $x_{t+1} = x_t + \eta \cdot \nabla_x f(x_t)$ (maximization)
- **Gradient descent:** $x_{t+1} = x_t - \eta \cdot \nabla_x f(x_t)$ (minimization)

**Key insight:** **Gradient ascent** moves **in the direction** of the gradient to **maximize** the function.

---

**Problem 31. You run a social media platform and are planning to implement a system to combat the spread of misinformation by detecting fake news articles. To keep things simple, the system only needs to identify articles as one of two classes: (1) being fake news, or (2) not being fake news. Of the model types we have learned in class so far, which would be the best choice to implement this system?**

**Answer:** Logistic Regression

**Explanation:** 

**This question tests understanding of model selection** for binary classification tasks.

**Why Logistic Regression is the best choice:**

**1. Binary classification task:**
- **Two classes:** fake news vs not fake news
- **Logistic regression** is designed for binary classification
- **Outputs probabilities** $P(y=1|x)$
- **Natural fit** for this problem

**2. Advantages of Logistic Regression:**
- **Interpretable** - coefficients show feature importance
- **Probabilistic outputs** - confidence scores for predictions
- **Efficient training** - convex optimization problem
- **Good baseline** - often performs well on binary classification

**3. Why other models are less suitable:**
- **Linear regression:** Designed for continuous outputs, not classification
- **Neural networks:** Overkill for simple binary classification
- **Support Vector Machines:** More complex, less interpretable
- **Decision trees:** Can be less stable than logistic regression

**4. Practical considerations:**
- **Text features** can be easily incorporated
- **Feature engineering** (TF-IDF, word embeddings) works well
- **Regularization** (L1/L2) can prevent overfitting
- **Scalable** to large datasets

**Key insight:** **Logistic Regression** is the **most appropriate** model for **binary classification** tasks like fake news detection.

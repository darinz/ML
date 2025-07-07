# 8.2 The double descent phenomenon

**Model-wise double descent.** Recent works have demonstrated that the test error can present a "double descent" phenomenon in a range of machine learning models including linear models and deep neural networks. The conventional wisdom, as discussed in Section 8.1, is that as we increase the model complexity, the test error first decreases and then increases, as illustrated in Figure 8.8. However, in many cases, we empirically observe that the test error can have a second descent—it first decreases, then increases to a peak around when the model size is large enough to fit all the training data very well, and then decreases again in the so-called overparameterized regime, where the number of parameters is larger than the number of data points. See Figure 8.10 for an illustration of the typical curves of test errors against model complexity (measured by the number of parameters). To some extent, the overparameterized regime with the second descent is considered as new to the machine learning community—partly because lightly-regularized, overparameterized models are only extensively used in the deep learning era. A practical implication of the phenomenon is that one should not hold back from scaling into and experimenting with over-parameterized models because the test error may well decrease again to a level even smaller than the previous lowest point. Actually, in many cases, larger overparameterized models always lead to a better test performance (meaning there won't be a second ascent after the second descent).

<img src="./img/double_descent_modelwise.png" width="500px"/>

Figure 8.10: A typical model-wise double descent phenomenon. As the number of parameters increases, the test error first decreases when the number of parameters is smaller than the training data. Then in the overparameterized regime, the test error decreases again.

*The discovery of the phenomenon perhaps dates back to Opper [1995, 2001], and has been recently popularized by Belkin et al. [2020], Hastie et al. [2019], etc.*

**Sample-wise double descent.** A priori, we would expect that more training examples always lead to smaller test errors—more samples give strictly more information for the algorithm to learn from. However, recent work [Nakkiran, 2019] observes that the test error is not monotonically decreasing as we increase the sample size. Instead, as shown in Figure 8.11, the test error decreases, and then increases and peaks around when the number of examples (denoted by $`n`$) is similar to the number of parameters (denoted by $`d`$), and then decreases again. We refer to this as the sample-wise double descent phenomenon. To some extent, sample-wise double descent and model-wise double descent are essentially describing similar phenomena—the test error is peaked when $`n \approx d`$.

**Explanation and mitigation strategy.** The sample-wise double descent, or, in particular, the peak of test error at $`n \approx d`$, suggests that the existing training algorithms evaluated in these experiments are far from optimal when $`n \approx d`$. We will be better off by tossing away some examples and run the algorithms with a smaller sample size to steer clear of the peak. In other words, in principle, there are other algorithms that can achieve smaller test error when $`n \approx d`$, but the algorithms evaluated in these experiments fail to do so. The sub-optimality of the learning procedure appears to be the culprit of the peak in both sample-wise and model-wise double descent.

Indeed, with an optimally-tuned regularization (which will be discussed more in Section 9), the test error in the $`n \approx d`$ regime can be dramatically improved, and the model-wise and sample-wise double descent are both mitigated. See Figure 8.11.

The intuition above only explains the peak in the model-wise and sample-wise double descent, but does not explain the second descent in the model-wise double descent—why overparameterized models are able to generalize so well. The theoretical understanding of overparameterized models is an active research area with many recent advances. A typical explanation is that the commonly-used optimizers such as gradient descent provide an implicit regularization effect (which will be discussed in more detail in Section 9.2). In other words, even in the overparameterized regime and with an unregularized loss function, the model is still implicitly regularized, and thus exhibits a better test performance than an arbitrary solution that fits the data. For example, for linear models, when $`n \ll d`$, the gradient descent optimizer with zero initialization finds the *minimum norm* solution that fits the data (instead of an arbitrary solution that fits the data), and the minimum norm regularizer turns out to be a sufficiently good for the overparameterized regime (but it's not a good regularizer when $`n \approx d`$, resulting in the peak of test error).

<img src="./img/double_descent_samplewise.png" width="600px"/>

Figure 8.11: **Left:** The sample-wise double descent phenomenon for linear models. **Right:** The sample-wise double descent with different regularization strength for linear models. Using the optimal regularization parameter $\lambda$ (optimally tuned for each $n$, shown in green solid curve) mitigates double descent. **Setup:** The data distribution of $(x, y)$ is $x \sim \mathcal{N}(0, I_d)$ and $y \sim x^T \beta + \mathcal{N}(0, \sigma^2)$ where $d = 500$, $\sigma = 0.5$ and $\|\beta\|_2 = 1$.

Finally, we also remark that the double descent phenomenon has been mostly observed when the model complexity is measured by the number of parameters. It is unclear if and when the number of parameters is the best complexity measure of a model. For example, in many situations, the norm of the models is used as a complexity measure. As shown in Figure 8.12 right, for a particular linear case, if we plot the test error against the norm of the learnt model, the double descent phenomenon no longer occurs. This is partly because the norm of the learned model is also peaked around $n \approx d$ (See Figure 8.12 (middle) or Belkin et al. [2019], Mei and Montanari [2022], and discussions in Section 10.8 of James et al. [2021]). For deep neural networks, the correct complexity measure is even more elusive. The study of double descent phenomenon is an active research topic.

---

<img src="./img/double_descent_norm.png" width="100%"/>

Figure 8.12: **Left:** The double descent phenomenon, where the number of parameters is used as the model complexity. **Middle:** The norm of the learned model is peaked around $n \approx d$. **Right:** The test error against the norm of the learned model. The color bar indicate the number of parameters and the arrows indicates the direction of increasing model size. Their relationship are closer to the convention wisdom than to a double descent. **Setup:** We consider a linear regression with a fixed dataset of size $n = 500$. The input $x$ is a random ReLU feature on Fashion-MNIST, and output $y \in \mathbb{R}^{10}$ is the one-hot label. This is the same setting as in Section 5.2 of Nakkiran et al. (2020).

---

## Expanded Explanations: Double Descent Phenomenon

### Model-wise Double Descent

The **model-wise double descent** phenomenon refers to the surprising observation that as we increase the complexity of a model (for example, by increasing the number of parameters in a neural network or the degree of a polynomial in regression), the test error does not always follow the classic U-shaped curve predicted by the bias-variance tradeoff. Instead, the test error can decrease, then increase (as expected), but then decrease again as the model becomes highly overparameterized. This results in a "double descent" shape for the test error curve.

- **Classic regime (bias-variance tradeoff):** When the number of parameters is less than the number of training samples, increasing model complexity reduces bias but increases variance, leading to a U-shaped test error curve.
- **Modern regime (overparameterization):** When the number of parameters exceeds the number of training samples, the model can fit the training data perfectly. Surprisingly, in this regime, the test error can decrease again as model complexity increases further.

**Figure 8.10** illustrates this phenomenon:
- The x-axis is the number of parameters (model complexity).
- The y-axis is the test error.
- The curve first decreases (better fit), then increases (overfitting), then decreases again (overparameterized regime).
- The vertical dashed line marks the transition where the number of parameters is sufficient to fit the data.

**Practical implication:** In modern machine learning, especially deep learning, it is often beneficial to use very large, overparameterized models, as they can generalize well despite fitting the training data perfectly.

### Sample-wise Double Descent

The **sample-wise double descent** phenomenon is similar, but instead of varying the number of parameters, we vary the number of training samples while keeping the model size fixed.

- **Expectation:** More training data should always reduce test error, since the model has more information to learn from.
- **Observation:** In practice, as the number of samples $n$ approaches the number of parameters $d$, the test error can actually increase, peaking when $n \approx d$, and then decrease again as $n$ increases further.

**Why does this happen?**
- When $n < d$, the model is underdetermined and can fit the training data perfectly, but there are many possible solutions. The solution found by the learning algorithm (often the minimum norm solution) may generalize well if regularization is present.
- When $n \approx d$, the model is just determined, and the solution can be very sensitive to noise in the data, leading to high test error (the peak).
- When $n > d$, the model is overdetermined, and the solution is more stable, so test error decreases again.

**Figure 8.11** shows this effect:
- **Left plot:** Test error vs. number of samples. The curve dips, peaks at $n \approx d$, then dips again.
- **Right plot:** Shows the effect of regularization. Stronger regularization (green curve) smooths out the peak, reducing test error at $n \approx d$.
- **Setup:** The data is generated from a linear model with Gaussian features and noise. The optimal regularization parameter $\lambda$ is chosen for each $n$ in the green curve.

**Practical implication:** If you observe a peak in test error as you increase your dataset size, it may be due to this phenomenon. Using regularization can help mitigate the peak.

### The Role of Regularization

**Regularization** is a technique used to prevent overfitting by adding a penalty to the loss function (e.g., L2 norm of the weights). It helps control the complexity of the model and makes the solution less sensitive to noise in the data.

- When $n \approx d$, the model is most sensitive to noise, and regularization is most beneficial.
- In the overparameterized regime ($n \ll d$), implicit regularization from optimizers like gradient descent (which tends to find minimum norm solutions) can help the model generalize well, even without explicit regularization.

### Complexity Measures: Number of Parameters vs. Norm

The **number of parameters** is a common way to measure model complexity, but it is not always the best. In some cases, the **norm of the learned model** (e.g., the L2 norm of the weights) is a better indicator of complexity and generalization ability.

- **Figure 8.12 (Left):** Test error vs. number of parameters shows double descent.
- **Figure 8.12 (Middle):** The norm of the learned model also peaks at $n \approx d$.
- **Figure 8.12 (Right):** Test error vs. norm of the learned model. When plotted this way, the double descent disappears, and the relationship is more monotonic (test error decreases as norm decreases).

**Practical implication:** When analyzing model generalization, consider both the number of parameters and the norm of the learned model. For deep neural networks, the correct complexity measure is still an open research question.

### Summary
- **Double descent** is a modern phenomenon observed in both model complexity and sample size.
- It challenges the classic bias-variance tradeoff view.
- Regularization (explicit or implicit) is key to mitigating the peak in test error.
- The best measure of model complexity is still an open question, especially for deep learning.

---

## Python Code: Double Descent Phenomenon

Below are Python code snippets that demonstrate the double descent phenomenon, both model-wise and sample-wise, and the effect of regularization. You can run these examples to see the curves described above.

### Model-wise Double Descent (Polynomial Regression)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Simulate data
def f(x):
    return 1.5 * x**3 - 0.5 * x**2 + 0.2 * x + 1
np.random.seed(42)
n_train = 100
n_test = 1000
x_train = np.random.uniform(-1, 1, n_train)
y_train = f(x_train) + np.random.normal(0, 0.5, n_train)
x_test = np.linspace(-1, 1, n_test)
y_test = f(x_test)

test_errors = []
degrees = range(1, 30 + 1)
for deg in degrees:
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(x_train.reshape(-1, 1), y_train)
    y_pred = model.predict(x_test.reshape(-1, 1))
    test_errors.append(np.mean((y_pred - y_test) ** 2))
plt.figure(figsize=(7, 4))
plt.plot(degrees, test_errors, marker='o')
plt.xlabel('Model Complexity (Polynomial Degree)')
plt.ylabel('Test Error (MSE)')
plt.title('Model-wise Double Descent (Polynomial Regression)')
plt.grid(True)
plt.show()
```

### Sample-wise Double Descent (Linear Regression)
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

d = 50
n_min, n_max = 10, 100
np.random.seed(42)
beta = np.random.randn(d)
beta /= np.linalg.norm(beta)
ns = np.arange(n_min, n_max + 1, 2)
n_test = 1000
X_test = np.random.randn(n_test, d)
y_test = X_test @ beta + np.random.normal(0, 0.5, n_test)
test_errors = []
for n in ns:
    X_train = np.random.randn(n, d)
    y_train = X_train @ beta + np.random.normal(0, 0.5, n)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_errors.append(np.mean((y_pred - y_test) ** 2))
plt.figure(figsize=(7, 4))
plt.plot(ns, test_errors, marker='o')
plt.xlabel('Number of Training Samples (n)')
plt.ylabel('Test Error (MSE)')
plt.title('Sample-wise Double Descent (Linear Regression)')
plt.grid(True)
plt.show()
```

### Effect of Regularization on Sample-wise Double Descent
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression

d = 50
n_min, n_max = 10, 100
regs = [None, 1e-4, 1e-2, 1e-1, 1, 10]
plt.figure(figsize=(8, 5))
for reg in regs:
    np.random.seed(42)
    beta = np.random.randn(d)
    beta /= np.linalg.norm(beta)
    test_errors = []
    ns = np.arange(n_min, n_max + 1, 2)
    n_test = 1000
    X_test = np.random.randn(n_test, d)
    y_test = X_test @ beta + np.random.normal(0, 0.5, n_test)
    for n in ns:
        X_train = np.random.randn(n, d)
        y_train = X_train @ beta + np.random.normal(0, 0.5, n)
        if reg is None:
            model = LinearRegression()
        else:
            model = Ridge(alpha=reg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_errors.append(np.mean((y_pred - y_test) ** 2))
    label = f'Reg={reg}' if reg is not None else 'No Reg'
    plt.plot(ns, test_errors, marker='.', label=label)
plt.xlabel('Number of Training Samples (n)')
plt.ylabel('Test Error (MSE)')
plt.title('Sample-wise Double Descent with Regularization')
plt.legend()
plt.grid(True)
plt.show()
```

---

These code snippets allow you to:
- Simulate and visualize the double descent phenomenon for both model complexity and sample size.
- See the effect of regularization on mitigating the peak in test error.
- Experiment with different settings to deepen your understanding of double descent.
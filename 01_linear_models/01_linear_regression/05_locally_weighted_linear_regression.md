# 1.4 Locally weighted linear regression

Consider the problem of predicting $y$ from $x \in \mathbb{R}$. The leftmost figure below shows the result of fitting a $y = \theta_0 + \theta_1 x$ to a dataset. We see that the data doesn't really lie on a straight line, and so the fit is not very good.

<img src="./img/lwlr.png" width="500px" />

In many real-world problems, the relationship between $x$ and $y$ is not strictly linear. For example, in predicting house prices, the effect of square footage on price may depend on the neighborhood or other local factors. A single global linear model may miss these local patterns, leading to poor predictions. Locally weighted linear regression (LWR) addresses this by fitting a model that is tailored to the region around each query point. LWR is especially useful when the data shows local trends or nonlinearities, and when you have enough data to reliably fit local models. For instance, predicting house prices in different neighborhoods or modeling temperature as a function of time in different seasons are scenarios where LWR can excel.

Instead, if we had added an extra feature $x^2$, and fit $y = \theta_0 + \theta_1 x + \theta_2 x^2$, then we obtain a slightly better fit to the data (see middle figure). Naively, it might seem that the more features we add, the better. However, there is also a danger in adding too many features: The rightmost figure is the result of fitting a 5-th order polynomial $y = \sum_{j=0}^5 \theta_j x^j$. We see that even though the fitted curve passes through the data perfectly, we would not expect this to be a very good predictor of, say, housing prices ($y$) for different living areas ($x$). Without formally defining what these terms mean, we'll say the figure on the left shows an instance of **underfitting**—in which the data clearly shows structure not captured by the model—and the figure on the right is an example of **overfitting**. (Later in this class, when we talk about learning theory we'll formalize some of these notions, and also define more carefully just what it means for a hypothesis to be good or bad.)

LWR offers a compromise: it fits simple models, but only to local neighborhoods, allowing it to capture local structure without overfitting globally. Try plotting the data with a straight line, a high-degree polynomial, and an LWR fit to see the differences.

As discussed previously, and as shown in the example above, the choice of features is important to ensuring good performance of a learning algorithm. (When we talk about model selection, we'll also see algorithms for automatically choosing a good set of features.) In this section, let us briefly talk about the locally weighted linear regression (LWR) algorithm which, assuming there is sufficient training data, makes the choice of features less critical. This treatment will be brief, since you'll get a chance to explore some of the properties of the LWR algorithm yourself in the homework.

In the original linear regression algorithm, to make a prediction at a query point $x$ (i.e., to evaluate $h(x)$ ), we would:

1. Fit $\theta$ to minimize $\sum_i (y^{(i)} - \theta^T x^{(i)})^2$.
2. Output $\theta^T x$.

In contrast, the locally weighted linear regression algorithm does the following:

1. Assigns a weight to each training example based on its distance to $x$.
2. Fits $\theta$ to minimize $\sum_i w^{(i)} (y^{(i)} - \theta^T x^{(i)})^2$.
3. Outputs $\theta^T x$.

The solution for $\theta$ (if $X$ is the design matrix and $W$ is a diagonal matrix of weights) is:

$$
\theta = (X^T W X)^{-1} X^T W y
$$

Here, the $w^{(i)}$'s are non-negative valued **weights**. Intuitively, if $w^{(i)}$ is large for a particular value of $i$, then in picking $\theta$, we'll try hard to make $(y^{(i)} - \theta^T x^{(i)})^2$ small. If $w^{(i)}$ is small, then the $(y^{(i)} - \theta^T x^{(i)})^2$ error term will be pretty much ignored in the fit. The most common choice for the weights is the Gaussian kernel:

$$
w^{(i)} = \exp\left( -\frac{(x^{(i)} - x)^2}{2\tau^2} \right)
$$

The parameter $\tau$ controls how quickly the weight of a training example falls off with distance of its $x^{(i)}$ from the query point $x$; $\tau$ is called the **bandwidth** parameter. Small $\tau$ means only very close points influence the fit (can lead to high variance), while large $\tau$ means many points influence the fit (can lead to high bias). Use cross-validation to find the value that gives the best predictive performance, and try several values to see the effect.

Note that the weights depend on the particular point $x$ at which we're trying to evaluate $x$. Moreover, if $|x^{(i)} - x|$ is small, then $w^{(i)}$ is close to 1; and if $|x^{(i)} - x|$ is large, then $w^{(i)}$ is small. Hence, $\theta$ is chosen giving a much higher "weight" to the (errors on) training examples close to the query point $x$. (Note also that while the formula for the weights takes a form that is cosmetically similar to the density of a Gaussian distribution, the $w^{(i)}$'s do not directly have anything to do with Gaussians, and in particular the $w^{(i)}$'s are not random variables, normally distributed or otherwise.)

Locally weighted linear regression is the first example we're seeing of a **non-parametric** algorithm. The (unweighted) linear regression algorithm that we saw earlier is known as a **parametric** learning algorithm, because it has a fixed, finite number of parameters (the $\theta_i$'s), which are fit to the data. Once we've fit the $\theta_i$'s and stored them away, we no longer need to keep the training data around to make future predictions. In contrast, to make predictions using locally weighted linear regression, we need to keep the entire training set around. The term "non-parametric" (roughly) refers to the fact that the amount of stuff we need to keep in order to represent the hypothesis $h$ grows linearly with the size of the training set.

To summarize the difference:

|                | Parametric (OLS)         | Non-parametric (LWR)         |
|----------------|-------------------------|------------------------------|
| Model form     | Fixed, global            | Flexible, local              |
| Memory usage   | Low (just $\theta$)      | High (need all data)         |
| Prediction     | Fast                     | Slow (fit per query)         |
| Flexibility    | Limited                  | High (adapts to local data)  |

Once $\theta$ is learned in OLS, you can discard the data. In LWR, you need the data to make predictions. LWR is powerful for capturing local structure, but can be slow and memory-intensive for large datasets. The bandwidth $\tau$ is a key hyperparameter: too small leads to overfitting, too large to underfitting. LWR works best in low dimensions; in high dimensions, distances become less meaningful (curse of dimensionality). Use LWR when you have enough data, expect local patterns, and prediction speed is not critical.

Below is a simple implementation of LWR for 1D data:

```python
import numpy as np
import matplotlib.pyplot as plt

def locally_weighted_linear_regression(x_query, X, y, tau):
    """
    x_query: scalar or array, the point(s) to predict
    X: (n_samples, 1) array of training inputs
    y: (n_samples,) array of training targets
    tau: bandwidth parameter
    Returns: predicted y at x_query
    """
    X = X.reshape(-1, 1)
    x_query = np.atleast_1d(x_query)
    y_pred = []
    for x0 in x_query:
        # Compute weights
        w = np.exp(-((X.flatten() - x0) ** 2) / (2 * tau ** 2))
        W = np.diag(w)
        # Add intercept
        X_design = np.hstack([np.ones_like(X), X])
        # Closed-form weighted least squares
        theta = np.linalg.pinv(X_design.T @ W @ X_design) @ (X_design.T @ W @ y)
        y0 = np.array([1, x0]) @ theta
        y_pred.append(y0)
    return np.array(y_pred)

# Example usage:
np.random.seed(0)
X = np.linspace(0, 10, 30)
y = np.sin(X) + 0.3 * np.random.randn(30)

X_test = np.linspace(0, 10, 200)
y_pred = locally_weighted_linear_regression(X_test, X, y, tau=0.5)

plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X_test, y_pred, color='red', label='LWR prediction')
plt.title('Locally Weighted Linear Regression')
plt.legend()
plt.show()
```
# Linear Regression: Introduction

Linear regression is one of the most fundamental algorithms in supervised learning. It is used to model the relationship between a scalar dependent variable (target) and one or more explanatory variables (features). The goal is to learn a function that maps input features to the target variable, based on observed data. This approach is widely used in fields such as economics, biology, engineering, and social sciences, wherever we want to predict a continuous outcome from one or more input variables.

Supervised learning refers to the class of machine learning problems where we are given a dataset consisting of input-output pairs, and the goal is to learn a mapping from inputs to outputs. In the context of linear regression, the inputs are the features (such as living area, number of bedrooms, etc.), and the output is the value we want to predict (such as house price).

## Example: Predicting House Prices

Let's start by talking about a few examples of supervised learning problems. Suppose we have a dataset giving the living areas and prices of 47 houses from Portland, Oregon. Each data point represents a house, with its living area (in square feet) and its price (in thousands of dollars):

| Living area (ft²) | Price (1000$s) |
|------------------|---------------|
| 2104             | 400           |
| 1600             | 330           |
| 2400             | 369           |
| 1416             | 232           |
| 3000             | 540           |
| ...              | ...           |

This kind of data is typical in real-world applications, where we want to use measurable features to predict an outcome of interest. For example, a real estate agent might use such a model to estimate the value of a house based on its size.

We can plot this data to visualize the relationship between living area and price. The goal of linear regression is to find the best-fitting line through this data, which can then be used to predict the price of a house given its living area.

<img src="./img/housing_prices.png" width="400px" />

```python
import numpy as np
import matplotlib.pyplot as plt

# Example data (living area in ft^2, price in $1000s)
living_area = np.array([2104, 1600, 2400, 1416, 3000])
price = np.array([400, 330, 369, 232, 540])

plt.scatter(living_area, price)
plt.xlabel('Living area (ft²)')
plt.ylabel('Price (1000$s)')
plt.title('House Prices vs. Living Area')
plt.show()
```

## The Supervised Learning Problem

Given data like this, how can we learn to predict the prices of other houses in Portland, as a function of the size of their living areas? This is the essence of supervised learning: using known examples to make predictions about new, unseen cases.

To formalize this, we define:
- **Input features**: The variables we use to make predictions (e.g., living area), denoted as $x^{(i)}$ for the $i$-th example. In general, $x^{(i)}$ can be a vector if there are multiple features.
- **Target variable**: The value we want to predict (e.g., price), denoted as $y^{(i)}$ for the $i$-th example.
- **Training example**: A pair $(x^{(i)}, y^{(i)})$ representing the input and output for the $i$-th data point.
- **Training set**: The collection of all training examples, $\{(x^{(i)}, y^{(i)}) ; i = 1, \ldots, n\}$, where $n$ is the number of examples.

We use $\mathcal{X}$ to denote the space of input values and $\mathcal{Y}$ for the space of output values. In this example, $\mathcal{X} = \mathcal{Y} = \mathbb{R}$, meaning both inputs and outputs are real numbers. In more complex problems, $\mathcal{X}$ could be a higher-dimensional space.

The goal of supervised learning is, given a training set, to learn a function $h : \mathcal{X} \to \mathcal{Y}$ so that $h(x)$ is a good predictor for the corresponding value of $y$. This function $h$ is called a **hypothesis**. The process of learning is to choose $h$ from a set of possible functions (the hypothesis space) so that it best fits the data.

<img src="./img/learning_algorithm.png" width="300px" />

### Notation Summary

- $x^{(i)}$: Input variable (feature) for the $i$-th training example (e.g., living area)
- $y^{(i)}$: Output variable (target) for the $i$-th training example (e.g., price)
- $(x^{(i)}, y^{(i)})$: The $i$-th training example
- $n$: Number of training examples
- Training set: $\{(x^{(i)}, y^{(i)}) ; i = 1, \ldots, n\}$
- $\mathcal{X}$: Space of input values (features)
- $\mathcal{Y}$: Space of output values (targets)
- $h$: Hypothesis function, $h : \mathcal{X} \to \mathcal{Y}$

## Regression vs. Classification

Supervised learning problems can be broadly categorized into two types: regression and classification.

- **Regression**: When the target variable that we're trying to predict is continuous, such as in our housing example, we call the learning problem a **regression** problem. Linear regression is the most common example, but there are many other regression algorithms.
- **Classification**: When $y$ can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment), we call it a **classification** problem. Classification is used in tasks like spam detection, image recognition, and medical diagnosis.

The distinction is important because it determines the type of model and loss function we use. In this document, we focus on regression, but many of the ideas carry over to classification with appropriate modifications.

## Linear Regression with Multiple Features

In many real-world problems, we have more than one feature. To make our housing example more interesting, let's consider a slightly richer dataset in which we also know the number of bedrooms in each house:

| Living area (ft²) | #bedrooms | Price (1000$s) |
|------------------|-----------|---------------|
| 2104             | 3         | 400           |
| 1600             | 3         | 330           |
| 2400             | 3         | 369           |
| 1416             | 2         | 232           |
| 3000             | 4         | 540           |
| ...              | ...       | ...           |

Here, the $x$'s are two-dimensional vectors in $\mathbb{R}^2$. For instance, $x_1^{(i)}$ is the living area of the $i$-th house in the training set, and $x_2^{(i)}$ is its number of bedrooms. In general, $x^{(i)}$ can be a vector of any length, depending on how many features we include. Feature selection—deciding which features to use—is an important part of building a good model, but for now, let's take the features as given.

The idea of linear regression is to approximate the target variable $y$ as a linear function of the input features $x$. This means we assume that the relationship between the features and the target can be captured by a straight line (or hyperplane in higher dimensions).

To perform supervised learning, we must decide how we're going to represent functions/hypotheses $h$ in a computer. As an initial choice, let's say we decide to approximate $y$ as a linear function of $x$:

$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2
$$

Here, the $\theta$'s are the **parameters** (also called **weights**) parameterizing the space of linear functions mapping from $\mathcal{X}$ to $\mathcal{Y}$. Each $\theta_j$ determines how much the corresponding feature $x_j$ influences the prediction. $\theta_0$ is called the **intercept** or **bias** term, and it allows the model to fit data that does not pass through the origin.

When there is no risk of confusion, we will drop the $\theta$ subscript in $h_\theta(x)$, and write it more simply as $h(x)$. To simplify our notation, we also introduce the convention of letting $x_0 = 1$ (this is the **intercept term**), so that

$$
h(x) = \sum_{i=0}^d \theta_i x_i = \theta^T x,
$$

where on the right-hand side above we are viewing $\theta$ and $x$ both as vectors, and here $d$ is the number of input variables (not counting $x_0$). This vectorized notation is very convenient for both mathematical analysis and efficient computation in code.

```python
# Example: Representing a hypothesis function for two features (living area, bedrooms)
def h_theta(x, theta):
    """
    x: numpy array of shape (n_features,)
    theta: numpy array of shape (n_features,)
    Returns the predicted value (scalar)
    """
    return np.dot(theta, x)

# Example usage:
x = np.array([1, 2104, 3])  # x0=1 (intercept), living area, bedrooms
theta = np.array([50, 0.1, 20])  # Example parameters
prediction = h_theta(x, theta)
print(f"Predicted price: {prediction}")
```

## The Cost Function

Now, given a training set, how do we pick, or learn, the parameters $\theta$? One reasonable method is to make $h(x)$ close to $y$, at least for the training examples we have. To formalize this, we define a function that measures, for each value of the $\theta$'s, how close the $h(x^{(i)})$'s are to the corresponding $y^{(i)}$'s. This function is called the **cost function** (or **loss function**), and for linear regression, it is defined as:

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2.
$$

```python
# Cost function for linear regression (non-vectorized)
def compute_cost(X, y, theta):
    """
    X: numpy array of shape (n_samples, n_features)
    y: numpy array of shape (n_samples,)
    theta: numpy array of shape (n_features,)
    Returns the cost J(theta)
    """
    n = len(y)
    total = 0.0
    for i in range(n):
        prediction = np.dot(theta, X[i])
        total += (prediction - y[i]) ** 2
    return 0.5 * total
```

### Intuition and Explanation
- **Squared Error**: The term $\left( h_\theta(x^{(i)}) - y^{(i)} \right)^2$ measures how far off our prediction $h_\theta(x^{(i)})$ is from the true value $y^{(i)}$ for each training example. Squaring ensures that errors of different signs do not cancel out, and it penalizes larger errors more heavily.
- **Sum over all examples**: We sum this squared error over all $n$ training examples to get a measure of the total error for a given choice of parameters $\theta$.
- **Factor of $\frac{1}{2}$**: The $\frac{1}{2}$ is included for mathematical convenience. When we take derivatives of $J(\theta)$ with respect to $\theta$ (as we do in optimization), the 2 from the square cancels the $\frac{1}{2}$, simplifying the gradient expression.

### Geometric Interpretation
Minimizing $J(\theta)$ means we are finding the parameters $\theta$ that make the predictions $h_\theta(x)$ as close as possible to the actual values $y$ for all training examples. Geometrically, this corresponds to finding the line (or hyperplane, in higher dimensions) that best fits the data in the sense of minimizing the sum of squared vertical distances between the data points and the line.

### Vectorized Form and Mean Squared Error
We can also write the cost function in a more compact, vectorized form. Let $X$ be the $n \times (d+1)$ matrix of input features (including the intercept term), $\theta$ the parameter vector, and $y$ the vector of outputs. Then:

$$
J(\theta) = \frac{1}{2} \| X\theta - y \|^2
$$

```python
# Vectorized cost function for linear regression
def compute_cost_vectorized(X, y, theta):
    """
    X: numpy array of shape (n_samples, n_features)
    y: numpy array of shape (n_samples,)
    theta: numpy array of shape (n_features,)
    Returns the cost J(theta)
    """
    residuals = X @ theta - y
    return 0.5 * np.dot(residuals, residuals)
```

where $\| \cdot \|$ denotes the Euclidean (L2) norm. Sometimes, the cost function is averaged over the number of examples, giving the **mean squared error (MSE)**:

$$
\text{MSE}(\theta) = \frac{1}{n} \sum_{i=1}^n \left( h_\theta(x^{(i)}) - y^{(i)} \right)^2
$$

```python
# Mean Squared Error (MSE) for linear regression
def mean_squared_error(X, y, theta):
    """
    X: numpy array of shape (n_samples, n_features)
    y: numpy array of shape (n_samples,)
    theta: numpy array of shape (n_features,)
    Returns the mean squared error
    """
    n = len(y)
    residuals = X @ theta - y
    return np.dot(residuals, residuals) / n
```

Minimizing the cost function $J(\theta)$ (or equivalently, the MSE) leads to the best fit line or hyperplane for the data, according to the least-squares criterion.


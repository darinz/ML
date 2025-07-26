# Problem Set 1 Solutions

## Section 04: Train-Test Splitting, Generalized Least Squares Regression, MAP as Regularization

### 1. Biased Test Error
Is the test error unbiased for these programs? If not, how can we fix the code so it is?

#### 1.1. Program 1
```python
1 # Given dataset of 1000-by-50 feature
2 # matrix X, and 1000-by-1 labels vector
3
4 mu = np.mean(X, axis=0)
5 X = X - mu
6
7 idx = np.random.permutation(1000)
8 TRAIN = idx[0:900]
9 TEST = idx[900::]
10
11 ytrain = y[TRAIN]
12 Xtrain = X[TRAIN,:]
13
14 # solve for argmin_w ||Xtrain*w - ytrain||_2
15 w = np.linalg.solve(np.dot(Xtrain.T, Xtrain), np.dot(Xtrain.T, ytrain))
16
17 b = np.mean(ytrain)
18
19 ytest = y[TEST]
20 Xtest = X[TEST, :]
21
22 train_error = np.dot(np.dot(Xtrain, w)+b - ytrain,
23                      np.dot(Xtrain, w)+b - ytrain) / len(TRAIN)
24 test_error = np.dot(np.dot(Xtest, w)+b - ytest,
25                     np.dot(Xtest, w)+b - ytest) / len(TEST)
26
27 print('Train error = ', train_error)
28 print('Test error = ', test_error)
```

**Solution:**
The error is at the beginning of the program on lines 4 and 5. Notice how $`\mu`$ is a function of both the train and test data. By de-meaning the entire dataset before splitting, we are intertwining the train and test data. The correct procedure is:
*   Split into train and test
*   Compute the mean of the train data, $`\mu_{\text{train}}`$
*   De-mean both the train and test data with $`\mu_{\text{train}}`$

#### 1.2. Program 2
```python
1 # We are given: 1) dataset X with n=1000 samples and 50 features and 2) a vector y of length 1000 with labels.
2
3 # Consider the following code to train a model, using cross validation to perform hyperparameter tuning.
4 def fit(Xin, Yin, _lambda):
5     w = np.linalg.solve(np.dot(Xin.T, Xin) + _lambda * np.eye(Xin.shape[1]), np.dot(Xin.T, Yin))
6     b = np.mean(Yin) - np.dot(w, mu)
7     return w, b
8
9 def predict(w, b, Xin):
10     return np.dot(Xin, w) + b
11
12 idx = np.random.permutation(1000)
13 TRAIN = idx[0:800]
14 VAL = idx[800:900]
15 TEST = idx[900::]
16
17 ytrain = y[TRAIN]
18 Xtrain = X[TRAIN, :]
19 yval = y[VAL]
20 Xval = X[VAL, :]
21
22 # demean data
23 mu = np.mean(Xtrain, axis=0)
24 Xtrain = Xtrain - mu
25 Xval = Xval - mu
26
27 # use validation set to pick the best hyper-parameter to use
28 lambdas = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2]
29 err = np.zeros(len(lambdas))
30
31 for idx, lambda_val in enumerate(lambdas):
32     w, b = fit(Xtrain, ytrain, lambda_val)
33     yval_hat = predict(w, b, Xval)
34     err[idx] = np.mean((yval_hat - yval) ** 2)
35
36 lambda_best = lambdas[np.argmin(err)]
37
38 Xtot = np.concatenate((Xtrain, Xval), axis=0)
39 ytot = np.concatenate((ytrain, yval), axis=0)
40
41 w, b = fit(Xtot, ytot, lambda_best)
42
43 ytest = y[TEST]
44 Xtest = X[TEST, :]
45
46 # demean data
47 Xtest = Xtest - mu
48
49 ytot_hat = predict(w, b, Xtot)
50 train_error = np.mean((ytot_hat - ytot) ** 2)
51 ytest_hat = predict(w, b, Xtest)
52 test_error = np.mean((ytest_hat - ytest) ** 2)
53
54 print('Train error = ', train_error)
55 print('Test error = ', test_error)
```

**Solution:**
We are adding the validation data back into training (creating $`X_{tot}`$), and then retraining the whole model on this data. However, optimal value of $`\lambda`$ **depends** on size of the training dataset, so by adding more data we are using incorrect value in final fit call. In general, models get better the more data you give them, but only add the validation set back in if you are confident the hyperparameter doesn't depend on the number of elements, and that you aren't allowing your model access to the test set.

### 2. Gradient Descent

Like we've seen in lecture, gradient descent is an important algorithm commonly used to train machine learning models, particularly useful for when there is no closed form solution for the minimum of a loss function. Here, we'll go through short introduction to the algorithm.

Consider some function $`f(w)`$, which has some $`w_*`$ for which $`w_* = \text{arg min}_w f(w)`$:

![Gradient Descent](./q2_gradient_descent.png)

Let $`w_0`$ be some initial guess for the minimum of $`f(w)`$. Gradient descent will allow us to improve this solution.

#### (a) For some $`w`$ that is very close to $`w_0`$, give the Taylor series approximation for $`f(w)`$ starting at $`f(w_0)`$.

**Solution:**
For $`w`$ very close to $`w_0`$, we see that $`f(w) \approx f(w_0) + (w - w_0) \left(\frac{df(w)}{dw}\right)_{w=w_0}`$.

![Solution](./q2_solution.png)

#### (b) Now, let us choose some $`\eta > 0`$ that is very small. With this very small $`\eta`$, let's assume that $`w_1 = w_0 - \eta \left(\frac{df(w)}{dw}\right)_{w=w_0}`$. Using your approximation from part (a), give an expression for $`f(w_1)`$.

**Solution:**


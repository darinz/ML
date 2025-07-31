# Problem Set 4

## Conceptual Questions

**A1.** The answers to these questions should be answerable without referring to external materials. Briefly justify your answers with a few words.

a. True or False: Training deep neural networks requires minimizing a convex loss function, and therefore gradient descent will provide the best result.

b. True or False: It is a good practice to initialize all weights to zero when training a deep neural network.

c. True or False: We use non-linear activation functions in a neural network's hidden layers so that the network learns non-linear decision boundaries.

d. True or False: Given a neural network, the time complexity of the backward pass step in the backpropagation algorithm can be prohibitively larger compared to the relatively low time complexity of the forward pass step.

e. True or False: Neural Networks are the most extensible model and therefore the best choice for any circumstance.

## Kernels

**A2.** Suppose that our inputs $x$ are one-dimensional and that our feature map is infinite-dimensional: $\phi(x)$ is a vector whose $i$th component is:
$$\frac{1}{\sqrt{i!}}e^{-x^2/2}x^i$$
for all nonnegative integers $i$. (Thus, $\phi$ is an infinite-dimensional vector.) Show that $K(x,x') = e^{-\frac{(x-x')^2}{2}}$ is a kernel function for this feature map, i.e.,
$$\phi(x) \cdot \phi(x') = e^{-\frac{(x-x')^2}{2}}$$
Hint: Use the Taylor expansion of $z \mapsto e^z$. (This is the one dimensional version of the Gaussian (RBF) kernel).

## Kernel Ridge Regression

**A3.** This problem will get you familiar with kernel ridge regression using the polynomial and RBF kernels.
First, let's generate some data. Let $n = 30$ and $f_*(x) = 6 \sin(\pi x) \cos(4\pi x^2)$. For $i = 1,...,n$ let each $x_i$ be drawn uniformly at random from $[0, 1]$, and let $y_i = f_*(x_i) + \epsilon_i$ where $\epsilon_i \sim N(0, 1)$. For any function $f$, the true error and the train error are respectively defined as:
$$\mathcal{E}_{\text{true}}(f) = \mathbb{E}_{X,Y} \left[(f(X) - Y)^2\right], \quad \hat{\mathcal{E}}_{\text{train}}(f) = \frac{1}{n} \sum_{i=1}^n (f(x_i) - y_i)^2.$$
Now, our goal is, using kernel ridge regression, to construct a predictor:
$$\hat{\alpha} = \operatorname{argmin}_{\alpha} \|K\alpha - y\|_2^2 + \lambda \alpha^T K \alpha, \quad \hat{f}(x) = \sum_{i=1}^n \hat{\alpha}_i k(x_i, x)$$
where $K \in \mathbb{R}^{n \times n}$ is the kernel matrix such that $K_{i,j} = k(x_i, x_j)$, and $\lambda \ge 0$ is the regularization constant.

a. Use leave-one-out cross-validation to find optimal $\lambda$ and hyperparameter settings for the following two kernels:
* Polynomial kernel: $k_{poly}(x, z) = (1+x^Tz)^d$, where $d \in \mathbb{N}$ is a hyperparameter.
* Radial Basis Function (RBF) kernel: $k_{rbf}(x, z) = \exp(-\gamma||x-z||^2_2)$, where $\gamma > 0$ is a hyperparameter.

You may implement either grid search or random search. Do not use `sklearn`. Reasonable ranges for hyperparameters are $\lambda \in [10^{-5}, 10^{-1}]$ and $d \in [5, 25]$. For $\gamma$, you can use the heuristic¹: the inverse of the median of all $\binom{n}{2}$ squared distances $||x_i - x_j||^2_2$ for a dataset $x_1, \dots, x_n \in \mathbb{R}^d$. You do not need to search over $\gamma$ but can sample from a narrow Gaussian distribution centered at this heuristic value if you wish.

Report the values of $d, \lambda$, and $\gamma$ for both kernels.

b. Using the hyperparameters found in part a, plot the learned functions $\hat{f}_{poly}(x)$ and $\hat{f}_{rbf}(x)$. For each function, make a single plot that includes:
* The original data points $\{(x_i, y_i)\}_{i=1}^n$.
* The true function $f(x)$.
* The learned function $\hat{f}(x)$, plotted on a fine grid over the interval $[0, 1]$.

## Introduction to PyTorch

For questions A.4 and A.5, you will use PyTorch. Please refer to "Section materials (Week 6)" for a useful notebook and PyTorch Documentation.

**A4.** PyTorch is a powerful tool for developing neural networks. In this problem, we will explore how PyTorch is built and re-implement some of its core components. Start by reading the `README.md` file in the `intro_pytorch` subfolder, as problem statements may overlap with its content and function comments.

a. Implement components of custom PyTorch modules. These components are located in folders named `layers`, `losses`, and `optimizers`. Each file in these folders should contain at least one problem function with specific directions. Finally, implement functions in the `train.py` file.

b. Perform hyperparameter search using the modules implemented in A4.a. The loss function is also treated as a hyperparameter. Due to different input shapes for cross-entropy and Mean Squared Error (MSE), two separate files are provided: `crossentropy_search.py` and `mean_squared_error_search.py`.

For each file, build and train 6 specific models in the provided order:
1. Linear neural network (single layer, no activation function).
2. Neural network with one hidden layer (2 units) and sigmoid activation function after the hidden layer.
3. Neural network with one hidden layer (2 units) and ReLU activation function after the hidden layer.
4. Neural network with two hidden layers (each with 2 units) and Sigmoid and ReLU activation functions after the first and second hidden layers, respectively.
5. NN with two hidden layer (each with 2 units) and ReLU, Sigmoid activation functions after first and second hidden layers, respectively.
6. NN with two hidden layer (each with 2 units) and ReLu activation functions after first and second hidden layers.

For each loss function, submit a plot of losses from training and validation sets. All models should be on the same plot (12 lines per plot), with two plots total (1 for MSE, 1 for cross-entropy).

c. For each loss function, report the best performing architecture (best performing is defined here as achieving the lowest validation loss at any point during the training), and plot its guesses on test set. You should use function `plot_model_guesses` from `train.py` file. Lastly, report accuracy of that model on a test set.

## The Softmax function

One of the activation functions we ask you to implement is softmax. For a prediction $\hat{y} \in \mathbb{R}^k$ corresponding to single datapoint (in a problem with $k$ classes):
$$\text{softmax}(\hat{y}_i) = \frac{\exp(\hat{y}_i)}{\sum_j \exp(\hat{y}_j)}$$

## Neural Networks for MNIST

**A5.** In Homework 1, we used ridge regression to train a classifier for the MNIST dataset. In Homework 2, we used logistic regression to distinguish between the digits 2 and 7. Now, in this problem, we will use PyTorch to build a simple neural network classifier for MNIST to further improve our accuracy.

We will implement two different architectures: a shallow but wide network, and a narrow but deeper network. For both architectures, we use $d$ to refer to the number of input features (in MNIST, $d = 28^2 = 784$), $h_i$ to refer to the dimension of the $i$-th hidden layer and $k$ for the number of target classes (in MNIST, $k = 10$). For the non-linear activation, use ReLU. Recall from lecture that
$$\text{ReLU}(x) = \begin{cases} x, & x \ge 0 \\ 0, & x < 0. \end{cases}$$

## Weight Initialization

Consider a weight matrix $W \in \mathbb{R}^{n \times m}$ and $b \in \mathbb{R}^n$. Note that here $m$ refers to the input dimension and $n$ to the output dimension of the transformation $x \mapsto Wx+b$. Define $\alpha = \frac{1}{\sqrt{m}}$. Initialize all your weight matrices and biases according to $\text{Unif}(-\alpha, \alpha)$.

## Training

For this assignment, use the Adam optimizer from `torch.optim`. Adam is a more advanced form of gradient descent that combines momentum and learning rate scaling. It often converges faster than regular gradient descent in practice. You can use either Gradient Descent or any form of Stochastic Gradient Descent. Note that you are still using Adam, but might pass either the full data, a single datapoint or a batch of data to it. Use cross entropy for the loss function and ReLU for the non-linearity.

## Implementing the Neural Networks

a. Let $W_0 \in \mathbb{R}^{h \times d}$, $b_0 \in \mathbb{R}^h$, $W_1 \in \mathbb{R}^{k \times h}$, $b_1 \in \mathbb{R}^k$ and $\sigma(z): \mathbb{R} \rightarrow \mathbb{R}$ some non-linear activation function applied element-wise. Given some $x \in \mathbb{R}^d$, the forward pass of the wide, shallow network can be formulated as:

$$F_1(x) := W_1\sigma(W_0x + b_0) + b_1$$

Use $h = 64$ for the number of hidden units and choose an appropriate learning rate. Train the network until it reaches 99% accuracy on the training data and provide a training plot (loss vs. epoch). Finally evaluate the model on the test data and report both the accuracy and the loss.

b. Let $W_0 \in \mathbb{R}^{h_0 \times d}$, $b_0 \in \mathbb{R}^{h_0}$, $W_1 \in \mathbb{R}^{h_1 \times h_0}$, $b_1 \in \mathbb{R}^{h_1}$, $W_2 \in \mathbb{R}^{k \times h_1}$, $b_2 \in \mathbb{R}^k$ and $\sigma(z): \mathbb{R} \rightarrow \mathbb{R}$ some non-linear activation function. Given some $x \in \mathbb{R}^d$, the forward pass of the network can be formulated as:

$$F_2(x) := W_2\sigma(W_1\sigma(W_0x + b_0) + b_1) + b_2$$

Use $h_0 = h_1 = 32$ and perform the same steps as in part a.

c. Compute the total number of parameters of each network and report them. Then compare the number of parameters as well as the test accuracies the networks achieved. Is one of the approaches (wide, shallow vs. narrow, deeper) better than the other? Give an intuition for why or why not.

## Using PyTorch:

For your solution, you may not use any functionality from the `torch.nn` module except for `torch.nn.functional.relu` and `torch.nn.functional.cross_entropy`. You must implement the networks $F_1$ and $F_2$ from scratch. For starter code and a tutorial on PyTorch refer to the sections 6 and 7 material.

# Problem Set 5

## Conceptual Questions

**A1.** The answers to these questions should be answerable without referring to external materials. Briefly justify your answers with a few words.

a. True or False: Given a data matrix $X \in \mathbb{R}^{n \times d}$ where $d$ is much smaller than $n$ and $k = \text{rank}(X)$, if we project our data onto a $k$-dimensional subspace using PCA, our projection will have zero reconstruction error (in other words, we find a perfect representation of our data, with no information loss).

b. True or False: Suppose that an $n \times n$ matrix $X$ has a singular value decomposition of $USV^T$, where $S$ is a diagonal $n \times n$ matrix. Then, the rows of $V$ are equal to the eigenvectors of $X^T X$.

c. True or False: choosing $k$ to minimize the $k$-means objective (see Equation (4) below) is a good way to find meaningful clusters.

d. True or False: The singular value decomposition of a matrix is unique.

e. True or False: The rank of a square matrix equals the number of its unique nonzero eigenvalues.

## Think before you train

**A2.** The first part of this problem (part a) explores how you would apply machine learning theory and techniques to a real-world problem. There is one scenario detailing a setting, a dataset, and a specific result we hope to achieve. Your job is to describe how you would handle the scenario with the tools we've learned in this class. Your response should include:
(1) any pre-processing steps you would take (e.g. any data processing),
(2) the specific machine learning pipeline you would use (i.e., algorithms and techniques learned in this class),
(3) how your setup acknowledges the constraints and achieves the desired result.
You should also aim to leverage some of the theory we have covered in this class. Some things to consider may be: the nature of the data (i.e., *How hard is it to learn? Do we need more data? Are the data sources good?*), the effectiveness of the pipeline (i.e., *How strong is the model when properly trained and tuned?*), and the time needed to effectively perform the pipeline.

a. Scenario: Disease Susceptibility Predictor
* Setting: You are tasked by a research institute to create an algorithm that learns the factors that contribute most to acquiring a specific disease.
* Dataset: A rich dataset of personal demographic information, location information, risk factors, and whether a person has the disease or not.
* Result: The company wants a system that can determine how susceptible someone is to this disease when they enter in their own personal information. The pipeline should take limited amount of personal data from a new user and infer more detailed metrics about the person.

The second part of this problem (parts b, c) focuses on exploring possible shortcomings of machine learning models, and what real-world implications might follow from ignoring these issues.

b. Briefly describe (1) some potential shortcomings of your training process from the disease susceptibility predictor scenario above that may result in your algorithm having different accuracy on different populations, and (2) how you may modify your procedure to address these shortcomings.

c. Recall in Homework 2 we trained models to predict crime rates using various features. It is important to note that datasets describing crime have many shortcomings in describing the **entire landscape of illegal behavior in a city**, and that these shortcomings often fall disproportionately on minority communities. Some of these shortcomings include that crimes are reported at different rates in different neighborhoods, that police respond differently to the same crime reported or observed in different neighborhoods, and that police spend more time patrolling in some neighborhoods than others. What real-world implications might follow from ignoring these issues?

## Image Classification on CIFAR-10

**A3.** In this problem we will explore different deep learning architectures for image classification on the CIFAR-10 dataset. Make sure that you are familiar with `torch.Tensors`, two-dimensional convolutions (`nn.Conv2d`) and fully-connected layers (`nn.Linear`), ReLU non-linearities (`F.relu`), pooling (`nn.MaxPool2d`), and tensor reshaping (`view`).

A few preliminaries:
* Make sure to read the "Tips for HW4" EdStem post for additional tips about training your models.
* Each network $f$ maps an image $x^{\text{in}} \in \mathbb{R}^{32 \times 32 \times 3}$ (3 channels for RGB) to an output $f(x^{\text{in}}) = x^{\text{out}} \in \mathbb{R}^{10}$. The class label is predicted as $\operatorname{arg max}_{i=0,1,...,9} x_i^{\text{out}}$. An error occurs if the predicted label differs from the true label for a given image.
* The network is trained via multiclass cross-entropy loss.
* Create a validation dataset by appropriately partitioning the train dataset. Hint: look at the documentation for `torch.utils.data.random_split`. Make sure to tune hyperparameters like network architecture and step size on the validation dataset. Do **NOT** validate your hyperparameters on the test dataset.
* At the end of each epoch (one pass over the training data), compute and print the training and validation classification accuracy.
* While one could train a network for hundreds of epochs to reach convergence and maximize accuracy, this can be prohibitively time-consuming, so feel free to train for just a dozen or so epochs.

For parts (a) and (b), apply a hyperparameter tuning method (e.g. random search, grid search, etc.) using the validation set, report the hyperparameter configurations you evaluated and the best set of hyperparameters from this set, and plot the training and validation classification accuracy as a function of epochs. Produce a separate line or plot for each hyperparameter configuration evaluated (top 3 configurations is sufficient to keep the plots clean). Finally, evaluate your best set of hyperparameters on the test data and report the test accuracy.

**Note 1:** Please refer to the provided notebook with starter code for this problem, on the course website. That notebook provides a complete end-to-end example of loading data, training a model using a simple network with a fully-connected output and no hidden layers (this is equivalent to logistic regression), and performing evaluation using canonical Pytorch. We recommend using this as a template for your implementations of the models below.

**Note 2:** If you are attempting this problem and do not have access to GPU we highly recommend using Google Colab. The provided notebook includes instructions on how to use GPU in Google Colab.

Here are the network architectures you will construct and compare.

a. Fully-connected output, 1 fully-connected hidden layer: this network has one hidden layer denoted as $x^{\text{hidden}} \in \mathbb{R}^M$ where $M$ will be a hyperparameter you choose ($M$ could be in the hundreds). The nonlinearity applied to the hidden layer will be the relu ($\operatorname{relu}(x) = \operatorname{max}\{0, x\}$). This network can be written as
$$x^{\text{out}} = W_2 \operatorname{relu}(W_1 x^{\text{in}} + b_1) + b_2$$
where $W_1 \in \mathbb{R}^{M \times 3072}$, $b_1 \in \mathbb{R}^M$, $W_2 \in \mathbb{R}^{10 \times M}$, $b_2 \in \mathbb{R}^{10}$. Tune the different hyperparameters and train for a sufficient number of epochs to achieve a validation accuracy of at least 50%. Provide the hyperparameter configuration used to achieve this performance.

b. Convolutional layer with max-pool and fully-connected output: for a convolutional layer $W_1$ with filters of size $k \times k \times 3$, and $M$ filters (reasonable choices are $M = 100$, $k = 5$), we have that $\operatorname{Conv2d}(x^{\text{in}}, W_1) \in \mathbb{R}^{(33-k) \times (33-k) \times M}$.

* Each convolution will have its own offset applied to each of the output pixels of the convolution; we denote this as $\text{Conv2d}(x^{\text{in}}, W) + b_1$ where $b_1$ is parameterized in $\mathbb{R}^M$. Apply a relu activation to the result of the convolutional layer.
* Next, use a max-pool of size $N \times N$ (a reasonable choice is $N = 14$ to pool to $2 \times 2$ with $k = 5$) we have that $\text{MaxPool}(\text{relu}(\text{Conv2d}(x^{\text{in}}, W_1) + b_1)) \in \mathbb{R}^{[\frac{33-k}{N}] \times [\frac{33-k}{N}] \times M}$.
* We will then apply a fully-connected layer to the output to get a final network given as

$$x^{\text{output}} = W_2(\text{MaxPool}(\text{relu}(\text{Conv2d}(x^{\text{input}}, W_1) + b_1))) + b_2$$

where $W_2 \in \mathbb{R}^{10 \times M([\frac{33-k}{N}])^2}$, $b_2 \in \mathbb{R}^{10}$.

The parameters $M, k, N$ (in addition to the step size and momentum) are all hyperparameters, but you can choose a reasonable value. Tune the different hyperparameters (number of convolutional filters, filter sizes, dimensionality of the fully-connected layers, step size, etc.) and train for a sufficient number of epochs to achieve a validation accuracy of at least 65%. Provide the hyperparameter configuration used to achieve this performance.

The number of hyperparameters to tune, combined with the slow training times, will hopefully give you a taste of how difficult it is to construct networks with good generalization performance. State-of-the-art networks can have dozens of layers, each with their own hyperparameters to tune. Additional hyperparameters you are welcome to play with, if you are so inclined, include: changing the activation function, replace max-pool with average-pool, adding more convolutional or fully connected layers, and experimenting with batch normalization or dropout.

## Matrix Completion and Recommendation System

**A4.** Note: Please refer to the provided notebook with starter code for this problem, on the course website. The notebook includes a template and code for data loading. We recommend copying the starter notebook and filling out the template.

In this problem, you will build a personalized movie recommendation system using the 100K MovieLens dataset, available at `https://grouplens.org/datasets/movielens/100k/`. The dataset contains $m = 1682$ movies and $n = 943$ users, with a total of 100,000 ratings. Each user has rated at least 20 movies. The goal is to recommend movies that users haven't seen.

We can think of this as a matrix $R \in \mathbb{R}^{m \times n}$ where the entry $R_{i,j} \in \{1, \dots, 5\}$ represents the $j$-th user's rating on movie $i$. A higher value indicates greater user satisfaction. The historical data is considered as observed entries in this matrix, with many unknown entries that need to be estimated to recommend the "best" movies.

For this problem, ignore user and movie metadata, and only use the ratings from the `u.data` file. Construct a training and test set from this file using the following code:

```python
import csv
import numpy as np
data = []
with open('u.data') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        data.append((int(row[0])-1, int(row[1])-1, int(row[2])))
data = np.array(data)

num_observations = len(data) # num_observations = 100,000
num_users = max(data[:,0])+1 # num_users = 943, indexed 0,...,942
num_items = max(data[:,1])+1 # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]
```

The `train` and `test` arrays contain $R_{train}$ and $R_{test}$ respectively. Each line in these arrays takes the form "j, i, s", where $j$ is the user index, $i$ is the movie index, and $s$ is the user's score.

Your task is to train a model using `train` to predict $\hat{R} \in \mathbb{R}^{m \times n}$, representing how every user would rate every movie. The model will be evaluated based on the average squared error on the test set, defined by the formula:

$$\mathcal{E}_{\text{test}}(\hat{R}) = \frac{1}{|\text{test}|} \sum_{(i,j, R_{i,j}) \in \text{test}} (\hat{R}_{i,j} - R_{i,j})^2.$$

As a baseline method for personalized recommendation, we will use low-rank matrix factorization. This method learns a vector representation $u_i \in \mathbb{R}^d$ for each movie and a vector representation $v_j \in \mathbb{R}^d$ for each user, such that their inner product $\langle u_i, v_j \rangle$ approximates the rating $R_{i,j}$. You will build a simple latent factor model.

You will implement multiple estimators and use the inner product $\langle u_i, v_j \rangle$ to predict if user $j$ likes movie $i$ in the test data. For simplicity, we will put aside best practices and choose hyperparameters by using those that minimize the test error. You may use fundamental operators from numpy or pytorch in this problem (numpy.linalg.lstsq, SVD, autograd, etc.) but not any precooked algorithm from a package like scikit-learn. If there is a question whether some package is not allowed for use in this problem, it probably is not appropriate.

a. Our first estimator pools all users together and, for each movie, outputs as its prediction the average user rating of that movie in train. That is, if $\mu \in \mathbb{R}^m$ is a vector where $\mu_i$ is the average rating of the users that rated the $i$th movie, write this estimator $\hat{R}$ as a rank-one matrix.
Compute the estimate $\hat{R}$. What is $\mathcal{E}_{\text{test}}(\hat{R})$ for this estimate?

b. Allocate a matrix $R_{i,j} \in \mathbb{R}^{m \times n}$ and set its entries equal to the known values in the training set, and 0 otherwise. Let $\hat{R}^{(d)}$ be the best rank-$d$ approximation (in terms of squared error) approximation to $R$. This is equivalent to computing the singular value decomposition (SVD) and using the top $d$ singular values. This learns a lower-dimensional vector representation for users and movies, assuming that each user would give a rating of 0 to any movie they have not reviewed.
* For each $d = 1, 2, 5, 10, 20, 50$, compute the estimator $\hat{R}^{(d)}$. We recommend using an efficient solver such as `scipy.sparse.linalg.svds`.
* Plot the average squared error of predictions on the training set and test set on a single plot, as a function of $d$.

Note that, in most applications, we would not actually allocate a full $m \times n$ matrix. We do so here only because our data is relatively small and it is instructive.

c. Replacing all missing values by a constant may impose strong and potentially incorrect assumptions on the unobserved entries of $R$. A more reasonable choice is to minimize the MSE (mean squared error) only on rated movies. Define a loss function:
$$\mathcal{L}(\{u_i\}_{i=1}^m, \{v_j\}_{j=1}^n) := \sum_{(i,j, R_{i,j}) \in \text{train}} (\langle u_i, v_j \rangle - R_{i,j})^2 + \lambda \sum_{i=1}^m \|u_i\|^2 + \lambda \sum_{j=1}^n \|v_j\|^2 \quad (1)$$
where $\lambda > 0$ is the regularization coefficient. We will implement algorithms to learn vector representations by minimizing (1). Note: we define the loss function here as the sum of squared errors; be careful to calculate and plot the mean squared error for your results.

Since this is a non-convex optimization problem, the initial starting point and hyperparameters may affect the quality of $\hat{R}$. You may need to tune $\lambda$ and $\sigma$ to optimize the loss you see.

Alternating minimization: First, randomly initialize $\{u_i\}$ and $\{v_j\}$. Then, alternate between (1) minimizing the loss function with respect to $\{u_i\}$ by treating $\{v_j\}$ as fixed; and (2) minimizing the loss function with respect to $\{v_j\}$ by treating $\{u_i\}$ as fixed. Repeat (1) and (2) until both $\{u_i\}$ and $\{v_j\}$ converge. Note that when one of $\{u_i\}$ or $\{v_j\}$ is given, minimizing the loss function with respect to the other part has a closed-form solution. Indeed, it can be shown that when minimizing with respect to a single $u_i$ (with $\{v_j\}$ fixed), the gradient is given by:
$$\nabla_{u_i} \mathcal{L}(\{u_i\}_{i=1}^m, \{v_j\}_{j=1}^n) = 2 \left( \sum_{j \in r(i)} v_j v_j^T + \lambda I \right) u_i - 2 \sum_{j \in r(i)} R_{i,j} v_j \quad (2)$$
where here $r(i)$ is a shorthand for the set of users who have reviewed movie $i$ in the training set, or more formally, $r(i) = \{j : (j, i, R_{i,j}) \in \text{train}\}$. Setting the overall gradient to be equal to 0 gives us that

$$\arg \min_{u_i} L(\{u_i\}_{i=1}^m, \{v_j\}_{j=1}^n) = \left( \sum_{j \in r(i)} v_j v_j^T + \lambda I \right)^{-1} \left( \sum_{j \in r(i)} R_{i,j} v_j \right) \quad (3)$$

Note that this update rule is for a single vector $u_i$, whereas you should update all of the $\{u_i\}_{i=1}^m$ in one round. When it comes to the alternate step which involves fixing $\{u_i\}$ and minimizing $\{v_j\}$, an analogous calculation will give you a very similar update rule.

* Try $d \in \{1, 2, 5, 10, 20, 50\}$ and plot the mean squared error of train and test as a function of $d$.

Some hints:
* Common choices for initializing the vectors $\{u_i\}_{i=1}^m, \{v_j\}_{j=1}^n$ include: entries drawn from `np.random.rand()` scaled by some scale factor $\sigma > 0$ ($\sigma$ is an additional hyperparameter), or using one of the solutions from part b or c.
* The only $m \times n$ matrix you need to allocate is probably for $\tilde{R}$.
* It is crucial that the squared error part of the loss is only defined w.r.t. $R_{i,j}$ that actually exist in the training set. Consider implementing some type of data structures that allow you to keep track of $r(i)$ as well as the reverse mapping $r^{-1}(j)$ from movies to relevant users.
* In computing $\mathcal{E}_{\text{test}}(\hat{R})$ from part a, feel free to use the average train rating as the prediction of movies found in the test set but not in the training set.

## k-means clustering

**A5.** Given a dataset $\mathbf{x}_1, \dots, \mathbf{x}_n \in \mathbb{R}^d$ and an integer $1 \le k \le n$, recall the following k-means objective function

$$\min_{\pi_1, \dots, \pi_k} \sum_{i=1}^k \sum_{j \in \pi_i} \|\mathbf{x}_j - \boldsymbol{\mu}_i\|^2, \quad \boldsymbol{\mu}_i = \frac{1}{|\pi_i|} \sum_{j \in \pi_i} \mathbf{x}_j \quad (4)$$

Above, $\{\pi_i\}_{i=1}^k$ is a partition of $\{1,2,..., n\}$. The objective (4) is NP-hard¹ to find a global minimizer of. Nevertheless, Lloyd's algorithm (discussed in lecture) typically works well in practice.²

a. Implement Lloyd's algorithm for solving the k-means objective (4). Do not use any off-the-shelf implementations, such as those found in scikit-learn.

b. Run Lloyd's algorithm on the *training* dataset of MNIST with $k = 10$. Show the image representing the center of each cluster, as a set of $k$ $28 \times 28$ images.

**Note on Time to Run:**
The runtime of a good implementation for this problem should be fairly fast (a few minutes); if you find it taking upwards of one hour, please check your implementation! (Hint: **For loops are costly**. Can you vectorize it or use Numpy operations to make it faster in some ways? If not, is looping through data-points or through centers faster?)
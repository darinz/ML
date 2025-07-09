# Generative Learning Algorithms

## Introduction and Motivation

So far, we've mainly been talking about learning algorithms that model
```math
p(y|x; \theta)
```
the conditional distribution of $y$ given $x$. For instance, logistic regression modeled $p(y|x; \theta)$ as $h_\theta(x) = g(\theta^T x)$ where $g$ is the sigmoid function. In these notes, we'll talk about a fundamentally different approach to classification: **generative learning algorithms**.

### The Fundamental Difference: Discriminative vs. Generative Models

**Discriminative models** (like logistic regression, SVMs, neural networks) learn the boundary between classes by modeling $p(y|x)$ directly or learning a function $f(x)$ that maps inputs to labels. They focus on distinguishing between classes, often leading to better performance when model assumptions are not strictly met.

**Generative models** (like Naive Bayes, GDA) model how the data is generated for each class by learning $p(x|y)$ and $p(y)$. This allows them to simulate or generate new data points, and they can be used for unsupervised tasks (e.g., clustering, anomaly detection) because they model the full data distribution, not just the boundary.

**Key Intuition:** 
- Generative models answer "How likely is this data under each class?" 
- Discriminative models answer "Which class is more likely for this data?"

### A Concrete Example: Animal Classification

Consider a classification problem in which we want to learn to distinguish between elephants ($y = 1$) and dogs ($y = 0$), based on some features of an animal (e.g., weight, height, trunk length, etc.).

**Discriminative Approach (e.g., Logistic Regression):**
Given a training set, a discriminative algorithm tries to find a decision boundary—often a straight line in feature space—that separates the elephants and dogs. Then, to classify a new animal, it checks on which side of the decision boundary it falls and makes its prediction accordingly.

**Generative Approach (e.g., GDA):**
Here's a different approach. First, looking at elephants, we can build a model of what elephants look like by learning the distribution of elephant features. Then, looking at dogs, we can build a separate model of what dogs look like. Finally, to classify a new animal, we can match the new animal against the elephant model, and match it against the dog model, to see whether the new animal looks more like the elephants or more like the dogs we had seen in the training set.

### Mathematical Framework

Algorithms that try to learn $p(y|x)$ directly (such as logistic regression), or algorithms that try to learn mappings directly from the space of inputs $\mathcal{X}$ to the labels $\{0, 1\}$ (such as the perceptron algorithm) are called **discriminative** learning algorithms. 

Here, we'll talk about algorithms that instead try to model $p(x|y)$ (and $p(y)$). These algorithms are called **generative** learning algorithms. For instance, if $y$ indicates whether an example is a dog (0) or an elephant (1), then:
- $p(x|y=0)$ models the distribution of dogs' features
- $p(x|y=1)$ models the distribution of elephants' features
- $p(y)$ models the prior probability of encountering each type of animal

### Bayes' Rule: The Bridge Between Generative and Discriminative

After modeling $p(y)$ (called the **class priors**) and $p(x|y)$, our algorithm can then use Bayes' rule to derive the posterior distribution on $y$ given $x$:

```math
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
```

where $p(x) = \sum_y p(x|y)p(y)$ is the **evidence** or **marginal likelihood**.

**Understanding Bayes' Rule:**
- **Prior $p(y)$:** Our initial belief about the probability of each class before seeing any data
- **Likelihood $p(x|y)$:** How likely we are to observe feature vector $x$ given that the true class is $y$
- **Posterior $p(y|x)$:** Our updated belief about the probability of each class after observing $x$
- **Evidence $p(x)$:** The total probability of observing $x$ regardless of class (normalization constant)

**Bayes' Rule and Decision Theory**

- **Interpretation:** Bayes' rule allows us to update our beliefs about the class label $y$ after observing data $x$, combining prior knowledge ($p(y)$) and the likelihood ($p(x|y)$).
- **Decision Theory:** In practice, we often care about minimizing classification error, so we choose the class with the highest posterior probability (MAP decision rule):

```math
\arg\max_y p(y|x) = \arg\max_y p(x|y)p(y)
```

Note that we can ignore $p(x)$ in the maximization since it doesn't depend on $y$.

**Bayes Rule for Posterior**

The posterior probability can be computed using Bayes' rule, combining the likelihood, prior, and evidence. This is the fundamental equation that allows us to convert our generative models into discriminative predictions.

## 4.1 Gaussian Discriminant Analysis (GDA)

The first generative learning algorithm that we'll look at is Gaussian Discriminant Analysis (GDA). In this model, we'll assume that $p(x|y)$ is distributed according to a multivariate normal distribution. Let's talk briefly about the properties of multivariate normal distributions before moving on to the GDA model itself.

### 4.1.1 The Multivariate Normal Distribution

The multivariate normal distribution in $d$-dimensions, also called the multivariate Gaussian distribution, is parameterized by a **mean vector** $\mu \in \mathbb{R}^d$ and a **covariance matrix** $\Sigma \in \mathbb{R}^{d \times d}$, where $\Sigma \geq 0$ is symmetric and positive semi-definite. Also written "$\mathcal{N}(\mu, \Sigma)$", its density is given by:

```math
p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x-\mu)^T \Sigma^{-1} (x-\mu) \right)
```

**Understanding the Components:**
- **$(2\pi)^{d/2}$:** Normalization constant for $d$-dimensional space
- **$|\Sigma|^{1/2}$:** Square root of the determinant of the covariance matrix (volume scaling factor)
- **$(x-\mu)^T \Sigma^{-1} (x-\mu)$:** Mahalanobis distance squared - a measure of how far $x$ is from $\mu$ in terms of the distribution's scale and orientation
- **$\exp(-\frac{1}{2} \cdot)$:** Exponential decay function that gives the characteristic "bell curve" shape

In the equation above, "$|\Sigma|$" denotes the determinant of the matrix $\Sigma$.

**Key Properties:**

For a random variable $X$ distributed $\mathcal{N}(\mu, \Sigma)$, the mean is (unsurprisingly) given by $\mu$:

```math
\mathbb{E}[X] = \int_x x\, p(x; \mu, \Sigma) dx = \mu
```

The **covariance** of a vector-valued random variable $Z$ is defined as $\mathrm{Cov}(Z) = \mathbb{E}[(Z - \mathbb{E}[Z])(Z - \mathbb{E}[Z])^T]$. This generalizes the notion of the variance of a real-valued random variable. The covariance can also be defined as $\mathrm{Cov}(Z) = \mathbb{E}[ZZ^T] - (\mathbb{E}[Z])(\mathbb{E}[Z])^T$. (You should be able to prove to yourself that these two definitions are equivalent.) If $X \sim \mathcal{N}(\mu, \Sigma)$, then

```math
\mathrm{Cov}(X) = \Sigma.
```

**Geometric Interpretation:**
- **Mean $\mu$:** The center of the distribution - the point around which the data is most concentrated
- **Covariance $\Sigma$:** Determines the shape, orientation, and spread of the distribution
  - The eigenvectors of $\Sigma$ give the principal axes of the ellipsoid
  - The eigenvalues determine the lengths of these axes (the "spread" in each direction)
  - If $\Sigma$ is diagonal, the features are uncorrelated
  - If $\Sigma = \sigma^2 I$, the distribution is spherical (isotropic)

**Why the Multivariate Normal?**
The multivariate normal distribution is chosen for several reasons:
1. **Central Limit Theorem:** Many natural processes tend toward normal distributions
2. **Mathematical tractability:** Closed-form solutions for parameter estimation
3. **Geometric intuition:** Easy to visualize and understand
4. **Maximum entropy:** Among all distributions with given mean and covariance, the normal has maximum entropy

Here are some examples of what the density of a Gaussian distribution looks like:

<img src="./img/gaussian_distribution_1.png" width="700px" />

The left-most figure shows a Gaussian with mean zero (that is, the 2x1 zero-vector) and covariance matrix $\Sigma = I$ (the 2x2 identity matrix). A Gaussian with zero mean and identity covariance is also called the **standard normal distribution**. The middle figure shows the density of a Gaussian with zero mean and $\Sigma = 0.6I$; and the rightmost figure shows one with $\Sigma = 2I$. We see that as $\Sigma$ becomes larger, the Gaussian becomes more "spread-out," and as it becomes smaller, the distribution becomes more "compressed."

Let's look at some more examples.

<img src="./img/gaussian_compressed.png" width="700px" />

The figures above show Gaussians with mean 0, and with covariance matrices respectively

```math
\Sigma = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} ; \quad
\Sigma = \begin{bmatrix} 1 & 0.5 \\ 0.5 & 1 \end{bmatrix} ; \quad
\Sigma = \begin{bmatrix} 1 & 0.8 \\ 0.8 & 1 \end{bmatrix} .
```

The leftmost figure shows the familiar standard normal distribution, and we see that as we increase the off-diagonal entry in $\Sigma$, the density becomes more "compressed" towards the $45^\circ$ line (given by $x_1 = x_2$). This happens because positive off-diagonal elements indicate positive correlation between the features. We can see this more clearly when we look at the contours of the same three densities:

<img src="./img/contours.png" width="700px" />

**Visual Intuition:**
- The shape and orientation of the contours reflect the covariance structure
- When the means are far apart and the covariance is small, the classes are easily separable
- In GDA, the boundary is linear (for shared covariance) or quadratic (for class-specific covariance)
- In logistic regression, the boundary is always linear

Here's one last set of examples generated by varying $\Sigma$:

<img src="./img/contours_sigma.png" width="700px" />

The plots above used, respectively,

```math
\Sigma = \begin{bmatrix} 1 & -0.5 \\ -0.5 & 1 \end{bmatrix} ; \quad
\Sigma = \begin{bmatrix} 1 & -0.8 \\ -0.8 & 1 \end{bmatrix} ; \quad
\Sigma = \begin{bmatrix} 3 & 0.8 \\ 0.8 & 1 \end{bmatrix} .
```

From the leftmost and middle figures, we see that by decreasing the off-diagonal elements of the covariance matrix (making them negative), the density now becomes "compressed" again, but in the opposite direction. Negative correlations create ellipses oriented perpendicular to the positive correlation case. Lastly, as we vary the parameters, more generally the contours will form ellipses (the rightmost figure showing an example with different variances in different directions).

As our last set of examples, fixing $\Sigma = I$, by varying $\mu$, we can also move the mean of the density around.

<img src="./img/contours_mu.png" width="700px" />

The figures above were generated using $\Sigma = I$, and respectively

```math
\mu = \begin{bmatrix} 1 \\ 0 \end{bmatrix} ; \quad
\mu = \begin{bmatrix} -0.5 \\ 0 \end{bmatrix} ; \quad
\mu = \begin{bmatrix} -1 \\ -1.5 \end{bmatrix} .
```

**Multivariate Normal Density**

The multivariate normal density function can be computed using the probability density function with the given mean vector and covariance matrix. The exponential term in the density function creates the characteristic "bell curve" shape, with the maximum at the mean and decreasing as we move away from the mean.

#### 4.1.2 The Gaussian Discriminant Analysis Model

When we have a classification problem in which the input features $x$ are continuous-valued random variables, we can then use the Gaussian Discriminant Analysis (GDA) model, which models $p(x|y)$ using a multivariate normal distribution. The model is:

```math
y \sim \mathrm{Bernoulli}(\phi)
x|y=0 \sim \mathcal{N}(\mu_0, \Sigma)
x|y=1 \sim \mathcal{N}(\mu_1, \Sigma)
```

**Model Assumptions:**
1. **Class Prior:** $y$ follows a Bernoulli distribution with parameter $\phi$ (the probability of class 1)
2. **Class-Conditional Distributions:** Given the class, $x$ follows a multivariate normal distribution
3. **Shared Covariance:** Both classes share the same covariance matrix $\Sigma$ (this is a key assumption that leads to linear decision boundaries)

**Assumptions and Extensions:**
- GDA assumes that the class-conditional densities are Gaussian with the same covariance matrix for all classes. This leads to linear decision boundaries.
- If you allow each class to have its own covariance matrix, you get Quadratic Discriminant Analysis (QDA), which leads to quadratic decision boundaries.
- The shared covariance assumption is often reasonable when the classes have similar "spread" but different "locations"

**Writing out the distributions explicitly:**

**Class Prior:**
```math
p(y) = \phi^y (1-\phi)^{1-y}
```

**Class-Conditional Distributions:**
```math
p(x|y=0) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x-\mu_0)^T \Sigma^{-1} (x-\mu_0) \right)
```
```math
p(x|y=1) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left( -\frac{1}{2} (x-\mu_1)^T \Sigma^{-1} (x-\mu_1) \right)
```

**Model Parameters:**
The parameters of our model are:
- $\phi$: Prior probability of class 1
- $\mu_0$: Mean vector for class 0
- $\mu_1$: Mean vector for class 1  
- $\Sigma$: Shared covariance matrix

Note that while there are two different mean vectors $\mu_0$ and $\mu_1$, this model uses only one covariance matrix $\Sigma$ for both classes.

**Joint Likelihood:**
The log-likelihood of the data is given by

```math
\ell(\phi, \mu_0, \mu_1, \Sigma) = \log \prod_{i=1}^n p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma)
```
```math
= \log \prod_{i=1}^n p(x^{(i)}|y^{(i)}; \mu_0, \mu_1, \Sigma) p(y^{(i)}; \phi)
```

**Parameter Estimation via Maximum Likelihood:**
The MLE for $\phi$, $\mu_0$, $\mu_1$, and $\Sigma$ can be derived by maximizing the log-likelihood:

**Class Prior:**
```math
\phi = \frac{1}{n} \sum_{i=1}^n 1\{y^{(i)} = 1\}
```
This is simply the fraction of training examples that belong to class 1.

**Class Means:**
```math
\mu_0 = \frac{\sum_{i=1}^n 1\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 0\}}
```
```math
\mu_1 = \frac{\sum_{i=1}^n 1\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^n 1\{y^{(i)} = 1\}}
```
These are the sample means of the feature vectors for each class.

**Shared Covariance:**
```math
\Sigma = \frac{1}{n} \sum_{i=1}^n (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
```
This is the weighted average of the sample covariances for each class.

**Intuition Behind the Estimates:**
- **$\phi$:** We estimate the prior probability of class 1 as the proportion of class 1 examples in our training set
- **$\mu_0, \mu_1$:** We estimate the mean of each class as the average of all feature vectors belonging to that class
- **$\Sigma$:** We estimate the covariance as the average squared deviation from the class means, weighted by class membership

Pictorially, what the algorithm is doing can be seen in as follows:

<img src="./img/gda_result.png" width="400px" />

Shown in the figure are the training set, as well as the contours of the two Gaussian distributions that have been fit to the data in each of the two classes. Note that the two Gaussians have contours that are the same shape and orientation, since they share a covariance matrix $\Sigma$, but they have different means $\mu_0$ and $\mu_1$. Also shown in the figure is the straight line giving the decision boundary at which $p(y=1|x) = 0.5$. On one side of the boundary, we'll predict $y=1$ to be the most likely outcome, and on the other side, we'll predict $y=0$.

**GDA Parameter Estimation**

The parameters can be estimated using maximum likelihood estimation, computing the class priors, means, and shared covariance matrix. The MLE estimates have intuitive interpretations as sample statistics.

**GDA Prediction (Posterior and Class)**

Predictions are made by computing the posterior probabilities using Bayes' rule and selecting the class with higher probability. The decision boundary occurs where the posterior probabilities are equal.

### 4.1.3 Discussion: GDA and Logistic Regression

The GDA model has an interesting relationship to logistic regression. If we view the quantity $p(y=1|x; \phi, \mu_0, \mu_1, \Sigma)$ as a function of $x$, we'll find that it can be expressed in the form

```math
p(y=1|x; \phi, \Sigma, \mu_0, \mu_1) = \frac{1}{1 + \exp(-\theta^T x)}
```

where $\theta$ is some appropriate function of $\phi, \Sigma, \mu_0, \mu_1$. This is exactly the form that logistic regression—a discriminative algorithm—used to model $p(y=1|x)$.

**Theoretical Connection and Bias-Variance Tradeoff:**
- When the GDA assumptions hold, the posterior $p(y|x)$ is a logistic function of $x$, but the converse is not true.
- GDA has lower variance but higher bias (if the Gaussian assumption is wrong); logistic regression has higher variance but lower bias.
- GDA is **asymptotically efficient**: in the limit of very large training sets, no algorithm is strictly better at estimating $p(y|x)$ if the model assumptions are correct.
- Logistic regression is more **robust**: it makes fewer assumptions and is less sensitive to model misspecification.

**When to Choose GDA vs. Logistic Regression:**

**Choose GDA when:**
- You have limited training data (GDA is more data-efficient when assumptions hold)
- You have reason to believe the Gaussian assumption is reasonable
- You want interpretable parameters (means, covariance)
- You want to generate synthetic data from the learned distributions

**Choose Logistic Regression when:**
- You have a large amount of training data
- You're unsure about the Gaussian assumption
- You want a more robust model that makes fewer assumptions
- You only care about classification accuracy, not data generation

**The Bias-Variance Tradeoff in Practice:**
- **GDA (High Bias, Low Variance):** Makes strong assumptions about data distribution, leading to lower variance but higher bias if assumptions are wrong
- **Logistic Regression (Low Bias, High Variance):** Makes fewer assumptions, leading to lower bias but higher variance, especially with limited data

When would we prefer one model over another? GDA and logistic regression will, in general, give different decision boundaries when trained on the same dataset. Which is better?

We just argued that if $p(x|y)$ is multivariate gaussian (with shared $\Sigma$), then $p(y|x)$ necessarily follows a logistic function. The converse, however, is not true; i.e., $p(y|x)$ being a logistic function does not imply $p(x|y)$ is multivariate gaussian. This shows that GDA makes **stronger** modeling assumptions about the data than does logistic regression. It turns out that when these modeling assumptions are correct, then GDA will find better fits to the data, and is a better model. Specifically, when $p(x|y)$ is indeed gaussian (with shared $\Sigma$), then GDA is **asymptotically efficient**. Informally, this means that in the limit of very large training sets (large $n$), there is no algorithm that is strictly better than GDA (in terms of, say, how accurately they estimate $p(y|x)$). In particular, it can be shown that in this setting, GDA will be a better algorithm than logistic regression; and more generally, even for small training set sizes, we would generally expect GDA to better.

In contrast, by making significantly weaker assumptions, logistic regression is also more **robust** and less sensitive to incorrect modeling assumptions. There are many different sets of assumptions that would lead to $p(y|x)$ taking the form of a logistic function. For example, if $x|y=0 \sim \mathrm{Poisson}(\lambda_0)$, and $x|y=1 \sim \mathrm{Poisson}(\lambda_1)$, then $p(y|x)$ will be logistic. Logistic regression will also work well on Poisson data like this. But if we were to use GDA on such data—and fit Gaussian distributions to such non-Gaussian data—then the results will be less predictable, and GDA may (or may not) do well.

To summarize: GDA makes stronger modeling assumptions, and is more data efficient (i.e., requires less training data to learn "well") when the modeling assumptions are correct or at least approximately correct. Logistic regression makes weaker assumptions, and is significantly more robust to deviations from modeling assumptions. Specifically, when the data is indeed non-Gaussian, then in the limit of large datasets, logistic regression will almost always do better than GDA. For this reason, in practice logistic regression is used more often than GDA. (Some related considerations about discriminative vs. generative models also apply for the Naive Bayes algorithm that we discuss next, but the Naive Bayes algorithm is still considered a very good, and is certainly also a very popular, classification algorithm.)

**Logistic Regression Form (for comparison)**

The logistic regression model uses the sigmoid function to model the posterior probability directly. The key insight is that under GDA's Gaussian assumptions, the posterior naturally takes the form of a logistic function, showing the deep connection between these two approaches.

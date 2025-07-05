# Machine Learning Notation Guide

## Basic Training Data Notation

### Individual Examples
- **$x^{(i)}$**: Input variable (feature vector) for the $i$-th training example
  - In regression: could be living area, number of bedrooms, etc.
  - In classification: could be pixel values, text features, etc.
  - Often a vector: $x^{(i)} \in \mathbb{R}^d$ where $d$ is the number of features

- **$y^{(i)}$**: Output variable (target) for the $i$-th training example
  - In regression: continuous value (e.g., house price, temperature)
  - In classification: discrete label (e.g., 0/1, class index, one-hot vector)
  - $y^{(i)} \in \mathbb{R}$ for regression, $y^{(i)} \in \{1, 2, \ldots, k\}$ for k-class classification

- **$(x^{(i)}, y^{(i)})$**: The $i$-th training example (input-target pair)
  - Represents one complete observation from your dataset
  - The fundamental unit of supervised learning

### Dataset Notation
- **$n$**: Number of training examples in the dataset
- **$m$**: Alternative notation for number of training examples (used in some sources)
- **Training set**: $\{(x^{(i)}, y^{(i)}) ; i = 1, \ldots, n\}$ or $\{(x^{(i)}, y^{(i)}) ; i = 1, \ldots, m\}$
  - The complete collection of labeled examples used to train the model
  - Sometimes denoted as $\mathcal{D}$ or $\mathcal{T}$

### Feature Space Notation
- **$\mathcal{X}$**: Space of input values (feature space)
  - $\mathcal{X} \subseteq \mathbb{R}^d$ for continuous features
  - $\mathcal{X} \subseteq \{0,1\}^d$ for binary features
  - $\mathcal{X} \subseteq \mathbb{Z}^d$ for discrete features

- **$\mathcal{Y}$**: Space of output values (target space)
  - $\mathcal{Y} \subseteq \mathbb{R}$ for regression
  - $\mathcal{Y} \subseteq \{1, 2, \ldots, k\}$ for classification
  - $\mathcal{Y} \subseteq [0,1]^k$ for multi-class probability outputs

## Model and Hypothesis Notation

### Hypothesis Function
- **$h$**: Hypothesis function, $h : \mathcal{X} \to \mathcal{Y}$
  - Maps input features to predicted outputs
  - Also called the model, predictor, or learned function
  - Sometimes written as $f$ or $\hat{y}$

- **$h_\theta$**: Hypothesis function parameterized by $\theta$
  - Emphasizes that the function depends on learnable parameters
  - $\theta$ represents the model weights/parameters

- **$\hat{y}^{(i)}$**: Predicted output for the $i$-th example
  - $\hat{y}^{(i)} = h(x^{(i)})$ or $\hat{y}^{(i)} = h_\theta(x^{(i)})$
  - The model's prediction for the true target $y^{(i)}$

## Parameter Notation

### Linear Model Parameters
- **$\theta$**: Parameter vector (weights) of the model
  - $\theta \in \mathbb{R}^{d+1}$ for linear models with bias term
  - $\theta = [\theta_0, \theta_1, \ldots, \theta_d]^T$ where $\theta_0$ is the bias/intercept

- **$\theta_j$**: The $j$-th parameter/weight
  - $\theta_0$: Bias term (intercept)
  - $\theta_1, \theta_2, \ldots, \theta_d$: Feature weights

- **$w$**: Alternative notation for weight vector (used in some sources)
  - $w \in \mathbb{R}^d$ (often without bias term)
  - $b$: Bias term when using $w$ notation

### Matrix Notation
- **$X$**: Design matrix containing all training inputs
  - $X \in \mathbb{R}^{n \times d}$ where each row is $x^{(i)}$
  - $X = [x^{(1)}, x^{(2)}, \ldots, x^{(n)}]^T$

- **$y$**: Target vector containing all training outputs
  - $y \in \mathbb{R}^n$ for regression
  - $y \in \{1, 2, \ldots, k\}^n$ for classification

## Loss and Cost Functions

### Individual Loss
- **$\mathcal{L}(h(x^{(i)}), y^{(i)})$**: Loss function for the $i$-th example
  - Measures how well the prediction $h(x^{(i)})$ matches the true target $y^{(i)}$
  - Common examples: squared error, cross-entropy, hinge loss

### Cost Function
- **$J(\theta)$**: Cost function (average loss over all training examples)
  - $J(\theta) = \frac{1}{n}\sum_{i=1}^n \mathcal{L}(h_\theta(x^{(i)}), y^{(i)})$
  - The objective function we want to minimize during training

## Optimization Notation

### Gradient Descent
- **$\alpha$**: Learning rate (step size)
- **$\nabla_\theta J(\theta)$**: Gradient of cost function with respect to parameters
- **Update rule**: $\theta := \theta - \alpha \nabla_\theta J(\theta)$

### Regularization
- **$\lambda$**: Regularization parameter (controls model complexity)
- **$R(\theta)$**: Regularization term (e.g., L1, L2 penalty)
- **Regularized cost**: $J(\theta) = \frac{1}{n}\sum_{i=1}^n \mathcal{L}(h_\theta(x^{(i)}), y^{(i)}) + \lambda R(\theta)$

## Evaluation Metrics

### Regression Metrics
- **MSE**: Mean Squared Error = $\frac{1}{n}\sum_{i=1}^n (y^{(i)} - \hat{y}^{(i)})^2$
- **RMSE**: Root Mean Squared Error = $\sqrt{\text{MSE}}$
- **MAE**: Mean Absolute Error = $\frac{1}{n}\sum_{i=1}^n |y^{(i)} - \hat{y}^{(i)}|$

### Classification Metrics
- **Accuracy**: $\frac{1}{n}\sum_{i=1}^n \mathbb{I}[y^{(i)} = \hat{y}^{(i)}]$
- **Precision, Recall, F1-score**: Standard classification metrics

## Additional Notation

### Subscripts and Superscripts
- **$x_j^{(i)}$**: The $j$-th feature of the $i$-th training example
- **$X_{ij}$**: Element in the $i$-th row and $j$-th column of matrix $X$

### Sets and Indices
- **$\{1, 2, \ldots, n\}$**: Set of training example indices
- **$\{1, 2, \ldots, d\}$**: Set of feature indices
- **$\{1, 2, \ldots, k\}$**: Set of class indices (for classification)

### Probability Notation
- **$P(y|x)$**: Conditional probability of target given input
- **$P(y|x;\theta)$**: Model's predicted probability distribution
- **$\mathbb{E}[\cdot]$**: Expectation operator 
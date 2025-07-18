## 4.2 Naive Bayes

Naive Bayes is one of the simplest and most effective algorithms for classification, especially when dealing with high-dimensional data such as text. It is widely used in applications like spam filtering, sentiment analysis, document categorization, and even medical diagnosis. The core idea is to use probability theory to make predictions, leveraging the assumption that features are conditionally independent given the class label.

### Motivation and Context

In many real-world problems, the data we encounter is naturally discrete or can be discretized. For example, in text classification, each document (such as an email) can be represented by the words it contains. These words are drawn from a finite vocabulary, and the presence or absence of each word can be encoded as a binary feature. This approach is not limited to text: in medical diagnosis, symptoms can be encoded as binary features (present/absent), and in marketing, customer behaviors (clicked/not clicked) can be similarly represented.

**Why Discrete Features Matter:**
Discrete features are especially useful when:
- The data is inherently categorical (e.g., words, symptoms, product categories).
- The number of possible feature values is large, but each instance only activates a small subset (sparse data).
- We want to use simple, interpretable models that scale well to large datasets.
- The relationships between features and classes are not well-modeled by continuous distributions.

**Real-world analogy:**
Imagine you are a librarian trying to decide whether a new book belongs in the "mystery" or "romance" section. You might look for the presence of certain keywords ("detective," "murder," "love," "kiss") in the book's description. Each keyword acts as a feature, and your decision is based on which words appear. Naive Bayes formalizes this intuition using probability theory.

### The Problem with High-Dimensional Discrete Data

In GDA, the feature vectors $x$ were continuous, real-valued vectors. Let's now talk about a different learning algorithm in which the $x_j$'s are discrete-valued.

For our motivating example, consider building an email spam filter using machine learning. Here, we wish to classify messages according to whether they are unsolicited commercial (spam) email, or non-spam email. After learning to do this, we can then have our mail reader automatically filter out the spam messages and perhaps place them in a separate mail folder. Classifying emails is one example of a broader set of problems called **text classification**.

Let's say we have a training set (a set of emails labeled as spam or non-spam). We'll begin our construction of our spam filter by specifying the features $x_j$ used to represent an email.

### Feature Representation: Bag-of-Words and Beyond

We will represent an email via a feature vector whose length is equal to the number of words in the dictionary (the vocabulary). Specifically, if an email contains the $j$-th word of the dictionary, then we will set $x_j = 1$; otherwise, we let $x_j = 0$. This is known as the **bag-of-words** model, where the order of words is ignored and only their presence or absence is recorded.

**Understanding the Bag-of-Words Model:**
- **Vocabulary:** The set of all unique words that appear in our training data
- **Feature Vector:** A binary vector where each position corresponds to a word in the vocabulary
- **Sparsity:** Most documents contain only a small fraction of all possible words, making the feature vectors very sparse

**Impact of Vocabulary Size:**
- The size of the vocabulary $|V|$ can be very large (tens or hundreds of thousands of words), especially in real-world datasets. This leads to high-dimensional but very sparse feature vectors, since most documents contain only a small subset of all possible words.
- Feature selection or dimensionality reduction (e.g., removing rare words, using only the top $k$ most frequent words, or applying techniques like TF-IDF) can help manage this complexity.

**Feature Engineering:**
- Instead of just using single words (unigrams), we can also include bigrams (pairs of consecutive words), trigrams, or other $n$-grams as features. This can capture some local word order information and improve performance for certain tasks.
- Domain knowledge can guide the inclusion of special features, such as the presence of numbers, punctuation, or specific phrases.

**Bag-of-Words vs. N-grams:**
- Bag-of-words is simple and effective for many tasks, but it ignores word order and context.
- N-gram models add some context but increase the dimensionality further.
- Modern approaches (e.g., word embeddings, transformers) go beyond these representations, but Naive Bayes with bag-of-words remains a strong baseline.

**Practical Note:**
- In practice, preprocessing steps such as lowercasing, stemming/lemmatization, and stop-word removal are often applied to the text before feature extraction.

For instance, the vector

```math
x = \begin{bmatrix}
1 \\
0 \\
0 \\
\vdots \\
1 \\
\vdots \\
0
\end{bmatrix}
```

|         |         |
|---------|---------|
| a       | 1       |
| aardvark| 0       |
| aardwolf| 0       |
| ...     | ...     |
| buy     | 1       |
| ...     | ...     |
| zygmurgy| 0       |

is used to represent an email that contains the words "a" and "buy," but not "aardvark," "aardwolf" or "zygmurgy." The set of words encoded into the feature vector is called the **vocabulary**, so the dimension of $x$ is equal to the size of the vocabulary.

### The Curse of Dimensionality Problem

Having chosen our feature vector, we now want to build a generative model. So, we have to model $p(x|y)$. But if we have, say, a vocabulary of 50,000 words, then $x \in \{0,1\}^{50000}$ ($x$ is a 50,000-dimensional vector of 0's and 1's), and if we were to model $x$ explicitly with a multinomial distribution over the $2^{50000}$ possible outcomes, then we'd end up with a $(2^{50000} - 1)$-dimensional parameter vector. This is clearly too many parameters to estimate reliably from any reasonable amount of training data.

**The Problem:**
- With $d$ binary features, there are $2^d$ possible feature combinations
- Each combination needs its own probability parameter
- For $d = 50,000$, we need $2^{50,000} - 1$ parameters
- This is computationally and statistically intractable

### The Naive Bayes Assumption

To model $p(x|y)$, we will therefore make a very strong assumption. We will assume that the $x_i$'s are conditionally independent given $y$. This is known as the **Naive Bayes (NB) assumption**.

**Mathematical Formulation:**
```math
p(x_1, x_2, \ldots, x_d|y) = \prod_{j=1}^d p(x_j|y)
```

**Intuition:**
- Conditional independence means that, given the class label $y$, knowing the value of one feature $x_j$ tells you nothing about the value of another feature $x_k$.
- **Analogy:** Imagine you are a doctor diagnosing a disease ($y$). If you know the patient has the disease, then whether they have symptom $A$ (e.g., fever) is independent of whether they have symptom $B$ (e.g., cough). In reality, symptoms may be correlated, but the Naive Bayes model assumes they are not, once the disease is known.

**Why make this assumption?**
- It dramatically simplifies the model and reduces the number of parameters from exponential in the number of features to linear.
- It allows us to estimate each $p(x_j|y)$ separately, which is feasible even with limited data.
- The resulting model is computationally efficient and scales well to high-dimensional data.

**When does it fail?**
- In practice, features are often not truly independent given the class. For example, in text, the presence of "machine" often increases the chance of "learning" appearing.
- Despite this, Naive Bayes often works surprisingly well, especially when the dependencies are similar across classes or when the signal from independent features dominates.

**Practical Impact:**
- The model is robust to irrelevant features, as their probabilities will be similar across classes and thus cancel out.
- Strongly correlated features can lead to overcounting evidence, but the simplicity and regularization from the independence assumption often outweigh this drawback in high-dimensional settings.

### Deriving the Naive Bayes Model

We now have:

```math
\begin{align*}
p(x_1, \ldots, x_{50000}|y)
&= p(x_1|y)p(x_2|y, x_1)p(x_3|y, x_1, x_2) \cdots p(x_{50000}|y, x_1, \ldots, x_{49999}) \\
&= p(x_1|y)p(x_2|y)p(x_3|y) \cdots p(x_{50000}|y) \\
&= \prod_{j=1}^d p(x_j|y)
\end{align*}
```

The first equality simply follows from the usual properties of probabilities (chain rule), and the second equality used the NB assumption. We note that even though the Naive Bayes assumption is an extremely strong assumption, the resulting algorithm works well on many problems.

**Understanding the Derivation:**
1. **Chain Rule:** The first step applies the chain rule of probability: $p(A,B,C) = p(A)p(B|A)p(C|A,B)$
2. **Independence Assumption:** The second step applies the Naive Bayes assumption that features are conditionally independent given the class
3. **Product Form:** The final result is a product of individual feature probabilities

> **Note:**
> Actually, rather than looking through an English dictionary for the list of all English words, in practice it is more common to look through our training set and encode in our feature vector only the words that occur at least once there. Apart from reducing the number of words modeled and hence reducing our computational and space requirements, this also has the advantage of allowing us to model/include as a feature many words that may appear in your email (such as "cs229") but that you won't find in a dictionary. Sometimes (as in the homework), we also exclude the very high frequency words (which will be words like "the," "of," "and"; these high frequency, "content free" words are called **stop words**) since they occur in so many documents and do little to indicate whether an email is spam or non-spam.

### Parameter Estimation: Learning from Data

Our model is parameterized by $\phi_{j|y=1} = p(x_j = 1|y = 1)$, $\phi_{j|y=0} = p(x_j = 1|y = 0)$, and $\phi_y = p(y = 1)$. As usual, given a training set $\{(x^{(i)}, y^{(i)}); i = 1, \ldots, n\}$, we can write down the joint likelihood of the data:

```math
\mathcal{L}(\phi_y, \phi_{j|y=0}, \phi_{j|y=1}) = \prod_{i=1}^n p(x^{(i)}, y^{(i)})
```

**Understanding the Likelihood:**
- Each training example contributes a term $p(x^{(i)}, y^{(i)})$ to the likelihood
- We want to find parameters that maximize the probability of observing our training data
- The likelihood measures how well our model explains the observed data

**Worked Example:**
Suppose we have a tiny dataset of 4 emails, with a vocabulary of 3 words: "buy", "cheap", "now". The data is:

| Email | buy | cheap | now | Spam ($y$) |
|-------|-----|-------|-----|------------|
| 1     | 1   | 1     | 0   | 1          |
| 2     | 1   | 0     | 1   | 1          |
| 3     | 0   | 1     | 0   | 0          |
| 4     | 0   | 0     | 1   | 0          |

**Parameter Estimation:**
- $\phi_{1|y=1}$ (probability "buy" appears in spam): $= \frac{2}{2} = 1$
- $\phi_{2|y=1}$ (probability "cheap" appears in spam): $= \frac{1}{2} = 0.5$
- $\phi_{3|y=1}$ (probability "now" appears in spam): $= \frac{1}{2} = 0.5$
- $\phi_{1|y=0}$ (probability "buy" appears in non-spam): $= \frac{0}{2} = 0$
- $\phi_{2|y=0}$ (probability "cheap" appears in non-spam): $= \frac{1}{2} = 0.5$
- $\phi_{3|y=0}$ (probability "now" appears in non-spam): $= \frac{1}{2} = 0.5$
- $\phi_y = \frac{2}{4} = 0.5$

**Interpretation:**
- $\phi_{j|y=1}$ is the fraction of spam emails in which word $j$ appears.
- $\phi_{j|y=0}$ is the fraction of non-spam emails in which word $j$ appears.
- $\phi_y$ is the fraction of emails that are spam.

These parameters summarize the training data and are used to make predictions on new emails.

### Making Predictions with Naive Bayes

Having fit all these parameters, to make a prediction on a new example with features $x$, we then simply calculate

```math
p(y=1|x) = \frac{p(x|y=1)p(y=1)}{p(x)}
```

```math
= \frac{\left(\prod_{j=1}^d p(x_j|y=1)\right)p(y=1)}{\left(\prod_{j=1}^d p(x_j|y=1)\right)p(y=1) + \left(\prod_{j=1}^d p(x_j|y=0)\right)p(y=0)}
```

and pick whichever class has the higher posterior probability.

**Understanding the Prediction Process:**
1. **Compute Likelihoods:** Calculate $p(x|y=1)$ and $p(x|y=0)$ using the independence assumption
2. **Apply Bayes' Rule:** Combine likelihoods with priors to get posterior probabilities
3. **Make Decision:** Choose the class with higher posterior probability

**Computational Efficiency:**
- The product form makes computation very efficient
- We can work in log-space to avoid numerical underflow
- The model scales linearly with the number of features

### Extensions to Multi-Class and Multi-Valued Features

Lastly, we note that while we have developed the Naive Bayes algorithm mainly for the case of problems where the features $x_j$ are binary-valued, the generalization to where $x_j$ can take values in $\{1, 2, \ldots, k_j\}$ is straightforward. Here, we would simply model $p(x_j|y)$ as multinomial rather than as Bernoulli. Indeed, even if some original input attribute (say, the living area of a house, as in our earlier example) were continuous valued, it is quite common to **discretize** it—that is, turn it into a small set of discrete values—and apply Naive Bayes. For instance, if we use some feature $x_j$ to represent living area, we might discretize the continuous values as follows:

| Living area (sq. feet) | < 400 | 400-800 | 800-1200 | 1200-1600 | >1600 |
|:----------------------:|:-----:|:-------:|:--------:|:---------:|:-----:|
| $x_i$                  |   1   |    2    |    3     |     4     |   5   |

Thus, for a house with living area 890 square feet, we would set the value of the corresponding feature $x_j$ to 3. We can then apply the Naive Bayes algorithm, and model $p(x_j|y)$ with a multinomial distribution, as described previously. When the original, continuous-valued attributes are not well-modeled by a multivariate normal distribution, discretizing the features and using Naive Bayes (instead of GDA) will often result in a better classifier.

**Why Discretization Works:**
- Converts continuous features to discrete ones
- Allows application of Naive Bayes to mixed data types
- Can capture non-linear relationships that linear models miss
- Provides a simple way to handle outliers

### 4.2.1 Laplace Smoothing

The Naive Bayes algorithm as we have described it will work fairly well for many problems, but there is a simple change that makes it work much better, especially for text classification. Let's briefly discuss a problem with the algorithm in its current form, and then talk about how we can fix it.

#### The Zero Probability Problem

Consider spam/email classification, and let's suppose that, we are in the year of 20xx, after completing CS229 and having done excellent work on the project, you decide around May 20xx to submit work you did to the NeurIPS conference for publication.[^neurips]
Because you end up discussing the conference in your emails, you also start getting messages with the word "neurips" in it. But this is your first NeurIPS paper, and until this time, you had not previously seen any emails containing the word "neurips"; in particular "neurips" did not ever appear in your training set of spam/non-spam emails. Assuming that "neurips" was the 35000th word in the dictionary, your Naive Bayes spam filter therefore had picked its maximum likelihood estimates of the parameters $\phi_{35000|y}$ to be

```math
\phi_{35000|y=1} = \frac{\sum_{i=1}^n 1\{x_{35000}^{(i)} = 1 \wedge y^{(i)} = 1\}}{\sum_{i=1}^n 1\{y^{(i)} = 1\}} = 0
```

```math
\phi_{35000|y=0} = \frac{\sum_{i=1}^n 1\{x_{35000}^{(i)} = 1 \wedge y^{(i)} = 0\}}{\sum_{i=1}^n 1\{y^{(i)} = 0\}} = 0
```

I.e., because it has never seen "neurips" before in either spam or non-spam training examples, it thinks the probability of seeing it in either type of email is zero. Hence, when trying to decide if one of these messages containing "neurips" is spam, it calculates the class posterior probabilities, and obtains

```math
p(y=1|x) = \frac{\prod_{j=1}^d p(x_j|y=1)p(y=1)}{\prod_{j=1}^d p(x_j|y=1)p(y=1) + \prod_{j=1}^d p(x_j|y=0)p(y=0)}
```

```math
= \frac{0}{0}
```

This is because each of the terms "$\prod_{j=1}^d p(x_j|y)$" includes a term $p(x_{35000}|y) = 0$ that is multiplied into it. Hence, our algorithm obtains $0/0$, and doesn't know how to make a prediction.

[^neurips]: NeurIPS is one of the top machine learning conferences. The deadline for submitting a paper is typically in May-June.

#### The Statistical Problem

Stating the problem more broadly, it is statistically a bad idea to estimate the probability of some event to be zero just because you haven't seen it before in your finite training set. Take the problem of estimating the mean of a multinomial random variable $z$ taking values in $\{1, \ldots, k\}$. We can parameterize our multinomial with $\phi_j = p(z = j)$. Given a set of $n$ independent observations $\{z^{(1)}, \ldots, z^{(n)}\}$, the maximum likelihood estimates are given by

```math
\phi_j = \frac{\sum_{i=1}^n 1\{z^{(i)} = j\}}{n}
```

As we saw previously, if we were to use these maximum likelihood estimates, then some of the $\phi_j$'s might end up as zero, which was a problem. To avoid this, we can use **Laplace smoothing**, which replaces the above estimate with

```math
\phi_j = \frac{1 + \sum_{i=1}^n 1\{z^{(i)} = j\}}{k + n}
```

Here, we've added 1 to the numerator, and $k$ to the denominator. Note that $\sum_{j=1}^k \phi_j = 1$ still holds (check this yourself!), which is a desirable property since the $\phi_j$'s are estimates for probabilities that we know must sum to 1. Also, $\phi_j \neq 0$ for all values of $j$, solving our problem of probabilities being estimated as zero. Under certain (arguably quite strong) conditions, it can be shown that the Laplace smoothing actually gives the optimal estimator of the $\phi_j$'s.

#### Applying Laplace Smoothing to Naive Bayes

Returning to our Naive Bayes classifier, with Laplace smoothing, we therefore obtain the following estimates of parameters:

```math
\phi_{j|y=1} = \frac{1 + \sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 1\}}{2 + \sum_{i=1}^n 1\{y^{(i)} = 1\}}
```

```math
\phi_{j|y=0} = \frac{1 + \sum_{i=1}^n 1\{x_j^{(i)} = 1 \wedge y^{(i)} = 0\}}{2 + \sum_{i=1}^n 1\{y^{(i)} = 0\}}
```

(In practice, it usually doesn't matter much whether we apply Laplace smoothing to $\phi_y$ or not, since we will typically have a fair fraction each of spam and non-spam messages, so $\phi_y$ will be a reasonable estimate of $p(y=1)$ and will be quite far from 0 anyway.)

**Understanding Laplace Smoothing:**
- **Pseudocounts:** We add "fake" observations to prevent zero probabilities
- **Regularization:** The smoothing acts as a form of regularization, preventing overfitting
- **Bayesian Interpretation:** Can be viewed as adding a uniform prior to the multinomial distribution
- **Consistency:** Ensures that all probabilities are strictly positive and sum to 1

#### Parameter Estimation

The parameters $\phi_{j|y=1}$, $\phi_{j|y=0}$, and $\phi_y$ can be computed from the training data using maximum likelihood estimation with Laplace smoothing.

#### Making Predictions

The posterior probability $p(y=1|x)$ can be computed for a new sample using Bayes' rule with the estimated parameters.

#### Laplace Smoothing (Bernoulli Naive Bayes)

Laplace smoothing adds pseudocounts to prevent zero probability estimates, making the model more robust.

### 4.2.2 Event Models for Text Classification

To close off our discussion of generative learning algorithms, let's talk about one more model that is specifically for text classification. While Naive Bayes as we've presented it will work well for many classification problems, for text classification, there is a related model that does even better.

#### Bernoulli vs. Multinomial Event Models

In the specific context of text classification, Naive Bayes as presented uses what's called the **Bernoulli event model** (or sometimes multi-variate Bernoulli event model). In this model, we assumed that the way an email is generated is that first it is randomly determined (according to the class priors $p(y)$) whether a spammer or non-spammer will send you your next message. Then, the person sending the email runs through the dictionary, deciding whether to include each word $j$ in that email independently and according to the probabilities $p(x_j = 1|y) = \phi_{j|y}$. Thus, the probability of a message was given by $p(y) \prod_{j=1}^d p(x_j|y)$.

Here's a different model, called the **Multinomial event model**. To describe this model, we will use a different notation and set of features for representing emails. We let $x_j$ denote the identity of the $j$-th word in the email. Thus, $x_j$ is now an integer taking values in $\{1, \ldots, |V|\}$, where $|V|$ is the size of our vocabulary (dictionary). An email of $d$ words is now represented by a vector $(x_1, x_2, \ldots, x_d)$ of length $d$; note that $d$ can vary for different documents. For instance, if an email starts with "A NeurIPS ...", then $x_1 = 1$ ("a" is the first word in the dictionary), and $x_2 = 35000$ (if "neurips" is the 35000th word in the dictionary).

**Key Differences:**
- **Bernoulli Model:** Each word is a binary feature (present/absent)
- **Multinomial Model:** Each position contains a word from the vocabulary
- **Bernoulli:** Ignores word frequency and position
- **Multinomial:** Captures word frequency and can model word order

In the multinomial event model, we assume that the way an email is generated is via a random process in which spam/non-spam is first determined (according to $p(y)$) as before. Then, the sender of the email writes the email by first generating $x_1$ from some multinomial distribution over words ($p(x_1|y)$). Next, the second word $x_2$ is chosen independently of $x_1$ but from the same multinomial distribution, and similarly for $x_3, x_4$, and so on, until all $d$ words of the email have been generated. Thus, the overall probability of a message is given by $p(y) \prod_{j=1}^d p(x_j|y)$. Note that this formula looks like the one we had earlier for the probability of a message under the Bernoulli event model, but that the terms in the formula now mean very different things. In particular $x_j|y$ is now a multinomial, rather than a Bernoulli distribution.

**Model Parameters:**
The parameters for our new model are $\phi_y = p(y)$ as before, $\phi_{k|y=1} = p(x_j = k|y = 1)$ (for any $j$) and $\phi_{k|y=0} = p(x_j = k|y = 0)$. Note that we have assumed that $p(x_j|y)$ is the same for all values of $j$ (i.e., that the distribution according to which a word is generated does not depend on its position $j$ within the email).

**Likelihood Function:**
If we are given a training set $\{(x^{(i)}, y^{(i)}); i = 1, \ldots, n\}$ where $x^{(i)} = (x_1^{(i)}, x_2^{(i)}, \ldots, x_{d_i}^{(i)})$ (here, $d_i$ is the number of words in the $i$-th training example), the likelihood of the data is given by

```math
\mathcal{L}(\phi_y, \phi_{k|y=0}, \phi_{k|y=1}) = \prod_{i=1}^n p(x^{(i)}, y^{(i)})
```

```math
= \prod_{i=1}^n \left( \prod_{j=1}^{d_i} p(x_j^{(i)}|y; \phi_{k|y=0}, \phi_{k|y=1}) \right) p(y^{(i)}; \phi_y)
```

#### Multinomial Naive Bayes Parameter Estimation and Laplace Smoothing

The multinomial event model parameters can be estimated with Laplace smoothing to handle word counts in text classification.

**Parameter Estimation for Multinomial Model:**
- Count the number of times each word appears in each class
- Apply Laplace smoothing to prevent zero probabilities
- Normalize to get probability distributions

**Advantages of Multinomial Model:**
- Better captures word frequency information
- Often performs better on text classification tasks
- More natural for modeling word generation processes

**When to Use Each Model:**
- **Bernoulli:** When word presence/absence is more important than frequency
- **Multinomial:** When word frequency provides important signal
- **Multinomial:** Generally preferred for text classification tasks

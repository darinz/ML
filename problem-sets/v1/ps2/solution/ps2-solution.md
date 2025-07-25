# Problem Set #2: Kernels, SVMs, and Theory

## 1. Kernel ridge regression

In contrast to ordinary least squares which has a cost function
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)})^2,
```
we can also add a term that penalizes large weights in $`\theta`$. In *ridge regression*, our least squares cost is regularized by adding a term $`\lambda||\theta||^2`$, where $`\lambda > 0`$ is a fixed (known) constant (regularization will be discussed at greater length in an upcoming course lecture). The ridge regression cost function is then
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (\theta^T x^{(i)} - y^{(i)})^2 + \frac{\lambda}{2}||\theta||^2.
```

(a) Use the vector notation described in class to find a closed-form expression for the value of $`\theta`$ which minimizes the ridge regression cost function.

**Answer:**  Using the design matrix notation, we can rewrite $`J(\theta)`$ as

```math
J(\theta) = \frac{1}{2} (X\theta - \vec{y})^T (X\theta - \vec{y}) + \frac{\lambda}{2} \theta^T \theta.
```

Then the gradient is

```math
\nabla_{\theta} J(\theta) = X^T X \theta - X^T \vec{y} + \lambda \theta.
```

Setting the gradient to 0 gives us

```math
0 = X^T X \theta - X^T \vec{y} + \lambda \theta
```
```math
\theta = (X^T X + \lambda I)^{-1} X^T \vec{y}.
```

(b) Suppose that we want to use kernels to implicitly represent our feature vectors in a high-dimensional (possibly infinite dimensional) space. Using a feature mapping $`\phi`$, the ridge regression cost function becomes
```math
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (\theta^T \phi(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2}||\theta||^2.
```

Making a prediction on a new input $`x_{new}`$ would now be done by computing $`\theta^T \phi(x_{new})`$. Show how we can use the "kernel trick" to obtain a closed form for the prediction on the new input without ever explicitly computing $`\phi(x_{new})`$. You may assume that the parameter vector $`\theta`$ can be expressed as a linear combination of the input feature vectors; i.e., $`\theta = \sum_{i=1}^{m} \alpha_i \phi(x^{(i)})`$ for some set of parameters $`\alpha_i`$.

[Hint: You may find the following identity useful:
```math
(\lambda I + BA)^{-1}B = B(\lambda I + AB)^{-1}.
```
If you want, you can try to prove this as well, though this is not required for the problem.]

**Answer:**  Let $`\Phi`$ be the design matrix associated with the feature vectors $`\phi(x^{(i)})`$. Then from parts (a) and (b),

```math
\begin{align*}
\theta &= (\Phi^T \Phi + \lambda I)^{-1} \Phi^T \vec{y} \\
       &= \Phi^T (\Phi \Phi^T + \lambda I)^{-1} \vec{y} \\
       &= \Phi^T (K + \lambda I)^{-1} \vec{y}.
\end{align*}
```

where $`K`$ is the kernel matrix for the training set (since $`\Phi_{i,j} = \phi(x^{(i)})^T \phi(x^{(j)}) = K_{ij}`$.)
To predict a new value $`y_{\text{new}}`$, we can compute

```math
\begin{align*}
\vec{y}_{\text{new}} &= \theta^T \phi(x_{\text{new}}) \\
              &= \vec{y}^T (K + \lambda I)^{-1} \Phi \phi(x_{\text{new}}) \\
              &= \sum_{i=1}^m \alpha_i K(x^{(i)}, x_{\text{new}}).
\end{align*}
```

where $`\alpha = (K + \lambda I)^{-1} \vec{y}`$. All these terms can be efficiently computing using the kernel function.
To prove the identity from the hint, we left-multiply by $`\lambda(I + BA)`$ and right-multiply by $`\lambda(I + AB)`$ on both sides. That is,

```math
\begin{align*}
(\lambda I + BA)^{-1} B &= B(\lambda I + AB)^{-1} \\
B &= (\lambda I + BA)B(\lambda I + AB)^{-1} \\
B(\lambda I + AB) &= (\lambda I + BA)B \\
\lambda B + BAB &= \lambda B + BAB.
\end{align*}
```

This last line clearly holds, proving the identity.

## 2. $`\ell_2`$ norm soft margin SVMs

In the notes, we saw that if our data is not linearly separable, then we need to modify our support vector machine algorithm by introducing an error margin that must be minimized. Specifically, the formulation we have looked at is known as the $`\ell_1`$ norm soft margin SVM. In this problem we will consider an alternative method, known as the $`\ell_2`$ norm soft margin SVM. This new algorithm is given by the following optimization problem (notice that the slack penalties are now squared):

```math
\min_{w, b, \xi} \quad \frac{1}{2} \|w\|^2 + \frac{C}{2} \sum_{i=1}^m \xi_i^2 \\
\text{s.t.}\quad y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i,\quad i = 1, \ldots, m.
```

(a) Notice that we have dropped the $`\xi_i \geq 0`$ constraint in the $`\ell_2`$ problem. Show that these non-negativity constraints can be removed. That is, show that the optimal value of the objective will be the same whether or not these constraints are present.

**Answer:**  Consider a potential solution to the above problem with some $`\xi < 0`$. Then the constraint $`y^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i`$ would also be satisfied for $`\xi_i = 0`$, and the objective function would be lower, proving that this could not be an optimal solution.

(b) What is the Lagrangian of the $`\ell_2`$ soft margin SVM optimization problem?

**Answer:**

```math
\mathcal{L}(w, b, \xi, \alpha) = \frac{1}{2} w^T w + \frac{C}{2} \sum_{i=1}^m \xi_i^2 - \sum_{i=1}^m \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1 + \xi_i],
```

where $`\alpha_i \geq 0`$ for $`i = 1, \ldots, m`$.

(c) Minimize the Lagrangian with respect to $`w`$, $`b`$, and $`\xi`$ by taking the following gradients: $`\nabla_w \mathcal{L}`$, $`\frac{\partial}{\partial b} \mathcal{L}`$, and $`\nabla_\xi \mathcal{L}`$, and then setting them equal to 0. Here, $`\xi = [\xi_1, \xi_2, \ldots, \xi_m]^T`$.

**Answer:**  Taking the gradient with respect to $`w`$, we get

```math
0 = \nabla_w \mathcal{L} = w - \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)},
```

which gives us

```math
w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}.
```

Taking the derivative with respect to $`b`$, we get

```math
0 = \frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^m \alpha_i y^{(i)},
```

giving us

```math
0 = \sum_{i=1}^m \alpha_i y^{(i)}.
```

Finally, taking the gradient with respect to $`\xi`$, we have

```math
0 = \nabla_{\xi} \mathcal{L} = C\xi - \alpha,
```

where $`\alpha = [\alpha_1, \alpha_2, \ldots, \alpha_m]^T`$. Thus, for each $`i = 1, \ldots, m`$, we get

```math
0 = C\xi_i - \alpha_i \implies C\xi_i = \alpha_i.
```

(d) What is the dual of the $`\ell_2`$ soft margin SVM optimization problem?

**Answer:**  The objective function for the dual is

```math
\begin{align*}
W(\alpha) &= \min_{w, b, \xi} \mathcal{L}(w, b, \xi, \alpha) \\
&= \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m (\alpha_i y^{(i)} x^{(i)})^T (\alpha_j y^{(j)} x^{(j)}) + \frac{1}{2} \sum_{i=1}^m \frac{\alpha_i}{\xi_i} \xi_i^2 \\
&\quad - \sum_{i=1}^m \alpha_i \left[ y^{(i)} \left( \left( \sum_{j=1}^m \alpha_j y^{(j)} x^{(j)} \right)^T x^{(i)} + b \right) - 1 + \xi_i \right] \\
&= -\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} + \frac{1}{2} \sum_{i=1}^m \alpha_i \xi_i \\
&\quad - \left( \sum_{i=1}^m \alpha_i y^{(i)} \right) b + \sum_{i=1}^m \alpha_i - \sum_{i=1}^m \alpha_i \xi_i \\
&= \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} - \frac{1}{2} \sum_{i=1}^m \alpha_i \xi_i \\
&= \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} - \frac{1}{2} \sum_{i=1}^m \frac{\alpha_i^2}{C}.
\end{align*}
```

Then the dual formulation of our problem is

```math
\begin{align*}
\max_{\alpha} \quad & \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y^{(i)} y^{(j)} (x^{(i)})^T x^{(j)} - \frac{1}{2} \sum_{i=1}^m \frac{\alpha_i^2}{C} \\
s.t. \quad & \alpha_i \geq 0, \quad i = 1, \ldots, m \\
& \sum_{i=1}^m \alpha_i y^{(i)} = 0
\end{align*}
```

## 3. SVM with Gaussian kernel

Consider the task of training a support vector machine using the Gaussian kernel $`K(x, z) = \exp(-\|x - z\|^2 / \tau^2)`$. We will show that as long as there are no two identical points in the training set, we can always find a value for the bandwidth parameter $`\tau`$ such that the SVM achieves zero training error.


(a) Recall from class that the decision function learned by the support vector machine can be written as

```math
f(x) = \sum_{i=1}^m \alpha_i y^{(i)} K(x^{(i)}, x) + b.
```

Assume that the training data $`\{(x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)})\}`$ consists of points which are separated by at least a distance of $`\epsilon`$; that is, $`\|x^{(j)} - x^{(i)}\| \geq \epsilon`$ for any $`i \neq j`$. Find values for the set of parameters $`\{\alpha_1, \ldots, \alpha_m, b\}`$ and Gaussian kernel width $`\tau`$ such that $`x^{(i)}`$ is correctly classified, for all $`i = 1, \ldots, m`$. [Hint: Let $`\alpha_i = 1`$ for all $`i`$ and $`b = 0`$. Now notice that for $`y \in \{-1, +1\}`$ the prediction on $`x^{(i)}`$ will be correct if $`|f(x^{(i)}) - y^{(i)}| < 1`$, so find a value of $`\tau`$ that satisfies this inequality for all $`i`$.]



(b) Suppose we run a SVM with slack variables using the parameter $`\tau`$ you found in part (a). Will the resulting classifier necessarily obtain zero training error? Why or why not? A short explanation (without proof) will suffice.

(c) Suppose we run the SMO algorithm to train an SVM with slack variables, under the conditions stated above, using the value of $`\tau`$ you picked in the previous part, and using some arbitrary value of $`C`$ (which you do not know beforehand). Will this necessarily result in a classifier that achieve zero training error? Why or why not? Again, a short explanation is sufficient.

## 4. Naive Bayes and SVMs for Spam Classification (Python Version)

In this question you’ll look into the Naive Bayes and Support Vector Machine algorithms for a spam classification problem using Python. Instead of implementing the algorithms yourself, you’ll use the scikit-learn machine learning library (or similar Python libraries). You should use the provided dataset in the `@/q4` directory if possible. The files are in ARFF format, which can be loaded in Python using the `liac-arff` package or by converting to CSV. If you encounter issues with the provided dataset, you may use a public spam dataset that works with Python, such as the [UCI Spambase dataset](https://archive.ics.uci.edu/ml/datasets/spambase) or the built-in datasets in scikit-learn.

**Instructions:**

- Install required packages: `scikit-learn`, `pandas`, and `liac-arff` (if using ARFF files).
- Load the dataset(s) and preprocess as needed (e.g., convert ARFF to pandas DataFrame).
- Use the same train/test split as in the provided files (e.g., `spam_test.arff` for testing, various `spam_train_*.arff` for training).
- If using a different dataset, use its standard train/test split or create your own.

(a) Train a Naive Bayes classifier (e.g., `sklearn.naive_bayes.MultinomialNB`) on the dataset and report the resulting error rates. Evaluate the performance of the classifier using each of the different training files (but each time using the same test file, e.g., `spam_test.arff`). Plot the error rate of the classifier versus the number of training examples.

(b) Repeat the previous part, but using a Support Vector Machine classifier (e.g., `sklearn.svm.SVC`). How does the performance of the SVM compare to that of Naive Bayes?

*Note: If you use a different dataset, clearly state which dataset you used and how you split the data for training and testing.*

## 5. Uniform convergence

In class we proved that for any finite set of hypotheses $`\mathcal{H} = \{h_1, \ldots, h_k\}`$, if we pick the hypothesis $`\hat{h}`$ that minimizes the training error on a set of $`m`$ examples, then with probability at least $`(1 - \delta)`$,

```math
\varepsilon(\hat{h}) \leq \left( \min_i \varepsilon(h_i) \right) + 2 \sqrt{\frac{1}{2m} \log \frac{2k}{\delta}},
```

where $`\varepsilon(h)`$ is the generalization error of hypothesis $`h_i`$. Now consider a special case (often called the realizable case) where we know, a priori, that there is some hypothesis in our class $`\mathcal{H}`$ that achieves zero error on the distribution from which the data is drawn. Then we could obviously just use the above bound with $`\min_i \varepsilon(h_i) = 0`$; however, we can prove a better bound than this.


(a) Consider a learning algorithm which, after looking at $`m`$ training examples, chooses some hypothesis $`\hat{h} \in \mathcal{H}`$ that makes zero mistakes on this training data. (By our assumption, there is at least one such hypothesis, possibly more.) Show that with probability $`1 - \delta`$

```math
\varepsilon(\hat{h}) \leq \frac{1}{m} \log \frac{k}{\delta}.
```

Notice that since we do not have a square root here, this bound is much tighter. [Hint: Consider the probability that a hypothesis with generalization error greater than $`\gamma`$ makes no mistakes on the training data. Instead of the Hoeffding bound, you might also find the following inequality useful: $`(1 - \gamma)^m \leq e^{-\gamma m}`$.]

(b) Rewrite the above bound as a sample complexity bound, i.e., in the form: for fixed $`\delta`$ and $`\gamma`$, for $`\varepsilon(\hat{h}) \leq \gamma`$ to hold with probability at least $`(1 - \delta)`$, it suffices that $`m \geq f(k, \gamma, \delta)`$ (i.e., $`f(\cdot)`$ is some function of $`k`$, $`\gamma`$, and $`\delta`$).



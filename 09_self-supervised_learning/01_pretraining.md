# Self-supervised learning and foundation models

## Introduction and Motivation

In traditional supervised learning, neural networks are trained on labeled datasets—collections of examples where each input is paired with a correct output (label). However, collecting large, high-quality labeled datasets is often expensive and time-consuming. For example, labeling millions of medical images or transcribing thousands of hours of speech requires significant human effort.

Recently, AI and machine learning have undergone a paradigm shift with the rise of **foundation models**—large models (like BERT, GPT-3, and others) that are pre-trained on vast amounts of data, often without any labels. These models are then adapted to a wide range of downstream tasks, sometimes with very little labeled data. This approach is inspired by how humans learn: we absorb a huge amount of information from the world (mostly unlabeled), and then use that knowledge to quickly learn new tasks with just a few examples.

**Why does this matter?**
- Foundation models can leverage massive amounts of unlabeled data, which is much easier to collect than labeled data.
- They can be adapted to many different tasks, making them highly versatile.
- They often require much less labeled data for each new task, reducing the cost and effort of building new AI systems.
- Their scale and training paradigm have led to new, emergent capabilities that were not possible with smaller, task-specific models.

This chapter introduces the paradigm of foundation models and the basic concepts of self-supervised learning, which is the key technique behind their success.

## 14.1 Pretraining and adaptation

The foundation models paradigm consists of two main phases: **pretraining** (or simply training) and **adaptation**. This two-phase approach is what enables foundation models to be so powerful and flexible.

### Pretraining: Learning from Unlabeled Data

In the pretraining phase, we train a large model on a massive dataset of *unlabeled* examples. For instance, this could be billions of images from the internet, or huge amounts of text scraped from websites and books. The key idea is that the model learns to extract useful patterns and representations from the data, even though it doesn't have access to explicit labels.

- **Analogy:** Think of pretraining like a person reading thousands of books in a foreign language, gradually picking up grammar, vocabulary, and style, even before ever taking a language test.
- **How does the model learn without labels?** Through *self-supervised learning*—the model creates its own learning tasks from the data. For example, it might try to predict missing words in a sentence, or whether two image patches come from the same photo.
- **Result:** The model develops a rich internal understanding of the data, which can be reused for many different tasks.

### Adaptation: Specializing to a New Task

After pretraining, we want to use the model for a specific downstream task, such as classifying medical images, translating text, or answering questions. This is the adaptation phase.

- **Zero-shot learning:** Sometimes, we use the pretrained model *as is* to make predictions on a new task, even if we have no labeled examples for that task. This is called zero-shot learning.
- **Few-shot learning:** If we have a small number of labeled examples (say, 10 or 50), we can adapt the model using just those few examples. This is called few-shot learning.
- **Many-shot learning:** If we have a large labeled dataset for the new task, we can further train (finetune) the model on this data.

**Why does this work?** The intuition is that the pretrained model has already learned good *representations*—ways of describing the data that capture its essential structure. Adaptation is like giving the model a few examples of what we care about, so it can quickly adjust to the new task.

- **Analogy:** After reading thousands of books, a person can quickly learn to write a new essay or answer questions on a specific topic, even with just a few examples.

We formalize the two phases below, and then discuss concrete methods for each.

[^1]: Sometimes, pretraining can involve large-scale labeled datasets as well (e.g., the ImageNet dataset).

### Pretraining (Mathematical Details & Example)

Suppose we have an unlabeled pretraining dataset $`\{x^{(1)}, x^{(2)}, \ldots, x^{(n)}\}`$ that consists of $`n`$ examples in $`\mathbb{R}^d`$. Let $`\phi_\theta`$ be a model (such as a neural network) with parameters $`\theta`$ that maps the input $`x`$ to a $`m`$-dimensional vector $`\phi_\theta(x)`$ (the *representation* or *embedding* of $`x`$).

We train the model by minimizing a **pretraining loss** over all examples:

```math
L_{\text{pre}}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell_{\text{pre}}(\theta, x^{(i)}).
```

- $`\ell_{\text{pre}}`$ is a *self-supervised loss*—it does not require labels. For example, it could be the loss for predicting a masked word in a sentence, or for making two augmented views of the same image have similar representations.
- $`\phi_\theta(x)`$ is the output of the model for input $`x`$.
- $`\theta`$ are the parameters we optimize.

**Example:** In BERT, $`\ell_{\text{pre}}`$ is the loss for predicting masked words. In SimCLR, it is the contrastive loss that pulls together representations of augmented views of the same image.

After minimizing $`L_{\text{pre}}(\theta)`$, we obtain a pretrained model $`\hat{\theta}`$ that has learned useful features from the data.

### Adaptation (Linear Probe & Finetuning)

Once we have a pretrained model, we want to use it for a new task. There are two main strategies:

#### Linear Probe (Feature Extraction)
- **Idea:** Freeze the pretrained model $`\hat{\theta}`$ and learn a simple linear classifier (or regressor) on top of the representations.
- **Why?** This is fast, requires little data, and tests how good the learned features are.
- **Mathematical Formulation:**

```math
\min_{w \in \mathbb{R}^m} \frac{1}{n_{\text{task}}} \sum_{i=1}^{n_{\text{task}}} \ell_{\text{task}}(y^{(i)}_{\text{task}}, w^\top \phi_{\hat{\theta}}(x^{(i)}_{\text{task}}))
```

- $`w`$ is the linear head (vector of weights) to be learned.
- $`\phi_{\hat{\theta}}(x)`$ is the fixed representation from the pretrained model.
- $`\ell_{\text{task}}`$ is the loss for the downstream task (e.g., cross-entropy for classification, squared error for regression).

**Tip:** Use a linear probe when you have little labeled data or want to quickly evaluate the quality of your pretrained features.

#### Finetuning
- **Idea:** Continue training *all* the parameters of the model (both $`w`$ and $`\theta`$) on the new task.
- **Why?** This allows the model to adapt its features to the specifics of the new task, often leading to better performance when you have more labeled data.
- **Mathematical Formulation:**

```math
\min_{w, \theta} \frac{1}{n_{\text{task}}} \sum_{i=1}^{n_{\text{task}}} \ell_{\text{task}}(y^{(i)}_{\text{task}}, w^\top \phi_\theta(x^{(i)}_{\text{task}}))
```

- Initialize $`w`$ randomly and $`\theta \leftarrow \hat{\theta}`$ (the pretrained weights).

**Tip:** Use finetuning when you have a moderate or large amount of labeled data for the new task, or when you want the best possible performance.

---

## 14.2 Pretraining methods in computer vision

This section introduces two concrete pretraining methods for computer vision: **supervised pretraining** and **contrastive learning**.

### Supervised pretraining

In supervised pretraining, the model is trained on a large *labeled* dataset (such as ImageNet, which contains millions of images labeled with object categories). The model learns to predict the correct label for each image, just like in standard supervised learning.

- **Neural network analogy:** Imagine a deep neural network as a series of layers that transform the input image into more and more abstract features. The last layer is usually a classifier that predicts the label.
- **What does 'removing the last layer' mean?** After training, we discard the final classification layer and keep the rest of the network. The output of the penultimate layer (just before the classifier) is used as the *representation* or *embedding* of the image. This is what we use as the pretrained model for downstream tasks.
- **Why does this work?** The network has learned to extract features that are useful for distinguishing between many different classes. These features are often general enough to be useful for new tasks, even if the new tasks are different from the original labels.

**Example:**
- Train a ResNet on ImageNet to classify images into 1000 categories.
- Remove the final classification layer.
- Use the output of the last hidden layer as the feature vector for each image.
- For a new task (e.g., classifying medical images), use these features as input to a new classifier or regressor.

### Contrastive learning

Contrastive learning is a self-supervised pretraining method that does *not* require any labels. Instead, it relies on clever use of data augmentation and a special loss function to learn useful representations.

- **Step-by-step example:**
  1. Take an image $`x`$ from the dataset.
  2. Apply two different random augmentations (e.g., cropping, color jitter, flipping) to $`x`$ to get $`\hat{x}`$ and $`\tilde{x}`$.
  3. Pass both $`\hat{x}`$ and $`\tilde{x}`$ through the model to get their representations $`\phi_\theta(\hat{x})`$ and $`\phi_\theta(\tilde{x})`$.
  4. Treat $`(\hat{x}, \tilde{x})`$ as a *positive pair* (they should be close in representation space).
  5. For all other images in the batch, treat their augmentations as *negative pairs* (they should be far apart in representation space).

- **Visual analogy:** Imagine a magnetic board where each image is a magnet. Positive pairs (augmentations of the same image) are pulled together, while negative pairs (different images) are pushed apart. The model learns to organize the representation space so that similar images are close and dissimilar images are far apart.

- **Why does this work?** By pulling together representations of different views of the same image, the model learns to focus on the underlying content rather than superficial details. This leads to features that are robust and generalize well to new tasks.

**Example:**
- Use SimCLR to pretrain a model on unlabeled images.
- For each image, generate two augmentations and treat them as a positive pair.
- Use a contrastive loss to train the model so that positive pairs are close and negative pairs are far apart.

---

[^2]: Negative pairs are not guaranteed to be always semantically unrelated, but in practice, with large datasets and random sampling, this is a reasonable assumption.
[^4]: This is a variant and simplification of the original loss that does not change the essence (but may change the efficiency slightly).
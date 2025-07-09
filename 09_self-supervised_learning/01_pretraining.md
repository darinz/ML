# Self-supervised learning and foundation models

## Introduction and Motivation

In traditional supervised learning, neural networks are trained on labeled datasets—collections of examples where each input is paired with a correct output (label). However, collecting large, high-quality labeled datasets is often expensive and time-consuming. For example, labeling millions of medical images or transcribing thousands of hours of speech requires significant human effort and domain expertise.

Recently, AI and machine learning have undergone a paradigm shift with the rise of **foundation models**—large models (like BERT, GPT-3, CLIP, and others) that are pre-trained on vast amounts of data, often without any labels. These models are then adapted to a wide range of downstream tasks, sometimes with very little labeled data. This approach is inspired by how humans learn: we absorb a huge amount of information from the world (mostly unlabeled), and then use that knowledge to quickly learn new tasks with just a few examples.

### The Data Labeling Problem

The fundamental challenge in supervised learning is the **data labeling bottleneck**. Consider these scenarios:

- **Medical Imaging**: A radiologist might take 10-30 minutes to label a single X-ray image with detailed annotations. For a dataset of 100,000 images, this represents 16,000-50,000 hours of expert time.
- **Speech Recognition**: Transcribing audio requires specialized knowledge and careful listening. Even with modern tools, high-quality transcription remains labor-intensive.
- **Natural Language Processing**: Creating datasets for tasks like sentiment analysis, question answering, or translation requires linguistic expertise and cultural understanding.

**Why does this matter?**
- Foundation models can leverage massive amounts of unlabeled data, which is much easier to collect than labeled data.
- They can be adapted to many different tasks, making them highly versatile and cost-effective.
- They often require much less labeled data for each new task, reducing the cost and effort of building new AI systems.
- Their scale and training paradigm have led to new, emergent capabilities that were not possible with smaller, task-specific models.
- They enable rapid deployment of AI systems in domains where labeled data is scarce or expensive.

### The Foundation Model Paradigm

Foundation models represent a fundamental shift from **task-specific models** to **general-purpose models**. Instead of training a separate model for each task, we train one large model that can be adapted to many tasks. This paradigm has several key advantages:

1. **Economies of Scale**: The cost of training one large model is amortized across many downstream applications.
2. **Knowledge Transfer**: Knowledge learned from one domain can be transferred to related domains.
3. **Rapid Adaptation**: New tasks can be tackled with minimal additional training.
4. **Emergent Capabilities**: Large models often develop unexpected abilities not explicitly trained for.

This chapter introduces the paradigm of foundation models and the basic concepts of self-supervised learning, which is the key technique behind their success.

## 14.1 Pretraining and adaptation

The foundation models paradigm consists of two main phases: **pretraining** (or simply training) and **adaptation**. This two-phase approach is what enables foundation models to be so powerful and flexible.

### Pretraining: Learning from Unlabeled Data

In the pretraining phase, we train a large model on a massive dataset of *unlabeled* examples. For instance, this could be billions of images from the internet, or huge amounts of text scraped from websites and books. The key idea is that the model learns to extract useful patterns and representations from the data, even though it doesn't have access to explicit labels.

#### The Self-Supervised Learning Principle

Self-supervised learning works by creating **surrogate tasks** from the data itself. These tasks are designed so that solving them requires the model to learn useful representations. Here are some examples:

**For Text Data:**
- **Masked Language Modeling**: Hide some words in a sentence and ask the model to predict them
- **Next Sentence Prediction**: Given two sentences, predict whether they naturally follow each other
- **Text Infilling**: Fill in missing parts of text based on context

**For Image Data:**
- **Contrastive Learning**: Learn to identify which image patches come from the same original image
- **Rotation Prediction**: Predict the rotation angle applied to an image
- **Jigsaw Puzzles**: Reconstruct an image from shuffled patches

**For Audio Data:**
- **Temporal Prediction**: Predict future audio frames from past ones
- **Speaker Identification**: Identify which audio segments come from the same speaker

#### Why Self-Supervised Learning Works

The key insight is that **structure exists in unlabeled data**. For example:
- In text, words have semantic relationships and grammatical patterns
- In images, objects have consistent visual features across different viewpoints
- In audio, speech has temporal and spectral patterns

By designing tasks that require understanding this structure, we force the model to learn meaningful representations.

- **Analogy:** Think of pretraining like a person reading thousands of books in a foreign language, gradually picking up grammar, vocabulary, and style, even before ever taking a language test.
- **How does the model learn without labels?** Through *self-supervised learning*—the model creates its own learning tasks from the data. For example, it might try to predict missing words in a sentence, or whether two image patches come from the same photo.
- **Result:** The model develops a rich internal understanding of the data, which can be reused for many different tasks.

### Adaptation: Specializing to a New Task

After pretraining, we want to use the model for a specific downstream task, such as classifying medical images, translating text, or answering questions. This is the adaptation phase.

#### The Adaptation Spectrum

Adaptation methods can be categorized along a spectrum based on how much labeled data is available:

**Zero-shot learning:** Sometimes, we use the pretrained model *as is* to make predictions on a new task, even if we have no labeled examples for that task. This is called zero-shot learning.

**Few-shot learning:** If we have a small number of labeled examples (say, 10 or 50), we can adapt the model using just those few examples. This is called few-shot learning.

**Many-shot learning:** If we have a large labeled dataset for the new task, we can further train (finetune) the model on this data.

#### Why Adaptation Works

The key insight is that the pretrained model has learned **general-purpose representations** that capture fundamental patterns in the data. These representations are often transferable across related tasks.

**Why does this work?** The intuition is that the pretrained model has already learned good *representations*—ways of describing the data that capture its essential structure. Adaptation is like giving the model a few examples of what we care about, so it can quickly adjust to the new task.

- **Analogy:** After reading thousands of books, a person can quickly learn to write a new essay or answer questions on a specific topic, even with just a few examples.

We formalize the two phases below, and then discuss concrete methods for each.

[^1]: Sometimes, pretraining can involve large-scale labeled datasets as well (e.g., the ImageNet dataset).

### Pretraining (Mathematical Details & Example)

Suppose we have an unlabeled pretraining dataset $`\{x^{(1)}, x^{(2)}, \ldots, x^{(n)}\}`$ that consists of $`n`$ examples in $`\mathbb{R}^d`$. Let $`\phi_\theta`$ be a model (such as a neural network) with parameters $`\theta`$ that maps the input $`x`$ to a $`m`$-dimensional vector $`\phi_\theta(x)`$ (the *representation* or *embedding* of $`x`$).

#### The Representation Learning Objective

We train the model by minimizing a **pretraining loss** over all examples:

```math
L_{\text{pre}}(\theta) = \frac{1}{n} \sum_{i=1}^n \ell_{\text{pre}}(\theta, x^{(i)}).
```

**Key Components:**
- $`\ell_{\text{pre}}`$ is a *self-supervised loss*—it does not require labels. For example, it could be the loss for predicting a masked word in a sentence, or for making two augmented views of the same image have similar representations.
- $`\phi_\theta(x)`$ is the output of the model for input $`x`$.
- $`\theta`$ are the parameters we optimize.

#### Understanding the Loss Function

The self-supervised loss function $`\ell_{\text{pre}}(\theta, x)`$ is designed to encourage the model to learn useful representations. Different self-supervised learning methods use different loss functions:

**Contrastive Loss (e.g., SimCLR):**
```math
\ell_{\text{pre}}(\theta, x) = -\log \frac{\exp(\text{sim}(\phi_\theta(\text{aug}_1(x)), \phi_\theta(\text{aug}_2(x))) / \tau)}{\sum_{j=1}^{2N} \mathbb{1}_{[j \neq i]} \exp(\text{sim}(\phi_\theta(\text{aug}_1(x)), \phi_\theta(z_j)) / \tau)}
```

Where:
- $`\text{aug}_1(x)`$ and $`\text{aug}_2(x)`$ are two different augmentations of the same image
- $`\text{sim}(u, v)`$ is the cosine similarity between vectors $`u`$ and $`v`$
- $`\tau`$ is a temperature parameter
- The denominator includes negative examples from the batch

**Masked Language Modeling Loss (e.g., BERT):**
```math
\ell_{\text{pre}}(\theta, x) = -\sum_{i \in M} \log p(x_i | x_{\setminus M})
```

Where:
- $`M`$ is the set of masked positions
- $`x_{\setminus M}`$ represents the sequence with masked tokens
- $`p(x_i | x_{\setminus M})`$ is the probability of predicting the correct token at position $`i`$

**Example:** In BERT, $`\ell_{\text{pre}}`$ is the loss for predicting masked words. In SimCLR, it is the contrastive loss that pulls together representations of augmented views of the same image.

After minimizing $`L_{\text{pre}}(\theta)`$, we obtain a pretrained model $`\hat{\theta}`$ that has learned useful features from the data.

### Adaptation (Linear Probe & Finetuning)

Once we have a pretrained model, we want to use it for a new task. There are two main strategies:

#### Linear Probe (Feature Extraction)

**Idea:** Freeze the pretrained model $`\hat{\theta}`$ and learn a simple linear classifier (or regressor) on top of the representations.

**Why?** This is fast, requires little data, and tests how good the learned features are.

**Mathematical Formulation:**

```math
\min_{w \in \mathbb{R}^m} \frac{1}{n_{\text{task}}} \sum_{i=1}^{n_{\text{task}}} \ell_{\text{task}}(y^{(i)}_{\text{task}}, w^\top \phi_{\hat{\theta}}(x^{(i)}_{\text{task}}))
```

**Key Components:**
- $`w`$ is the linear head (vector of weights) to be learned.
- $`\phi_{\hat{\theta}}(x)`$ is the fixed representation from the pretrained model.
- $`\ell_{\text{task}}`$ is the loss for the downstream task (e.g., cross-entropy for classification, squared error for regression).

**Advantages:**
- **Fast**: Only need to train a linear classifier
- **Stable**: No risk of catastrophic forgetting
- **Interpretable**: Linear relationship between features and predictions
- **Efficient**: Requires minimal computational resources

**Disadvantages:**
- **Limited Expressiveness**: Can only learn linear relationships
- **Suboptimal**: May not achieve best possible performance
- **Feature Quality Dependent**: Performance heavily depends on quality of pretrained features

**Tip:** Use a linear probe when you have little labeled data or want to quickly evaluate the quality of your pretrained features.

#### Finetuning

**Idea:** Continue training *all* the parameters of the model (both $`w`$ and $`\theta`$) on the new task.

**Why?** This allows the model to adapt its features to the specifics of the new task, often leading to better performance when you have more labeled data.

**Mathematical Formulation:**

```math
\min_{w, \theta} \frac{1}{n_{\text{task}}} \sum_{i=1}^{n_{\text{task}}} \ell_{\text{task}}(y^{(i)}_{\text{task}}, w^\top \phi_\theta(x^{(i)}_{\text{task}}))
```

**Key Components:**
- Initialize $`w`$ randomly and $`\theta \leftarrow \hat{\theta}`$ (the pretrained weights).
- Both $`w`$ and $`\theta`$ are updated during training.

**Advantages:**
- **Better Performance**: Can achieve higher accuracy than linear probe
- **Task-Specific Features**: Can adapt representations to the specific task
- **Flexibility**: Can learn complex, non-linear relationships

**Disadvantages:**
- **Computationally Expensive**: Requires training the entire model
- **Risk of Catastrophic Forgetting**: May lose general knowledge from pretraining
- **Requires More Data**: Needs sufficient labeled data to avoid overfitting
- **Hyperparameter Sensitivity**: More sensitive to learning rate and other hyperparameters

**Tip:** Use finetuning when you have a moderate or large amount of labeled data for the new task, or when you want the best possible performance.

#### Choosing Between Linear Probe and Finetuning

The choice between linear probe and finetuning depends on several factors:

1. **Amount of Labeled Data**: 
   - Small dataset (< 1000 examples): Use linear probe
   - Large dataset (> 10000 examples): Use finetuning
   - Medium dataset: Try both and compare

2. **Computational Resources**:
   - Limited resources: Use linear probe
   - Abundant resources: Use finetuning

3. **Task Similarity to Pretraining**:
   - Very similar: Linear probe may be sufficient
   - Very different: Finetuning likely necessary

4. **Performance Requirements**:
   - High accuracy required: Use finetuning
   - Moderate accuracy acceptable: Linear probe may suffice

---

## 14.2 Pretraining methods in computer vision

This section introduces two concrete pretraining methods for computer vision: **supervised pretraining** and **contrastive learning**.

### Supervised pretraining

In supervised pretraining, the model is trained on a large *labeled* dataset (such as ImageNet, which contains millions of images labeled with object categories). The model learns to predict the correct label for each image, just like in standard supervised learning.

#### The ImageNet Dataset

ImageNet is a large-scale dataset containing over 14 million images labeled with 21,841 categories. It has been instrumental in advancing computer vision research and serves as the standard benchmark for supervised pretraining.

**Key Characteristics:**
- **Scale**: 14+ million images across 21,841 categories
- **Diversity**: Covers a wide range of object categories, from animals to vehicles to everyday objects
- **Quality**: High-quality annotations by human experts
- **Hierarchy**: Categories are organized in a hierarchical structure (e.g., "dog" is a subclass of "mammal")

#### The Supervised Pretraining Process

**Neural network analogy:** Imagine a deep neural network as a series of layers that transform the input image into more and more abstract features. The last layer is usually a classifier that predicts the label.

**What does 'removing the last layer' mean?** After training, we discard the final classification layer and keep the rest of the network. The output of the penultimate layer (just before the classifier) is used as the *representation* or *embedding* of the image. This is what we use as the pretrained model for downstream tasks.

**Why does this work?** The network has learned to extract features that are useful for distinguishing between many different classes. These features are often general enough to be useful for new tasks, even if the new tasks are different from the original labels.

#### Mathematical Formulation

For supervised pretraining, the loss function is the standard cross-entropy loss:

```math
L_{\text{supervised}}(\theta) = \frac{1}{n} \sum_{i=1}^n \sum_{c=1}^C -y_{ic} \log(\hat{y}_{ic})
```

Where:
- $`y_{ic}`$ is the true label (1 if image $`i`$ belongs to class $`c`$, 0 otherwise)
- $`\hat{y}_{ic}`$ is the predicted probability that image $`i`$ belongs to class $`c`$
- $`C`$ is the number of classes

The model learns to map images to class probabilities through:
```math
\hat{y} = \text{softmax}(W \phi_\theta(x) + b)
```

Where:
- $`\phi_\theta(x)`$ is the feature representation (output of penultimate layer)
- $`W`$ and $`b`$ are the weights and bias of the final classification layer

**Example:**
- Train a ResNet on ImageNet to classify images into 1000 categories.
- Remove the final classification layer.
- Use the output of the last hidden layer as the feature vector for each image.
- For a new task (e.g., classifying medical images), use these features as input to a new classifier or regressor.

#### Advantages and Limitations

**Advantages:**
- **Proven Effectiveness**: Has been the standard approach for many years
- **Well-Understood**: Extensive research and practical experience
- **Good Performance**: Achieves strong results on many downstream tasks
- **Stable Training**: Relatively straightforward to train and debug

**Limitations:**
- **Requires Labels**: Needs large amounts of labeled data
- **Task-Specific**: Features may be biased toward the pretraining task
- **Expensive**: Creating large labeled datasets is costly
- **Limited Transfer**: May not transfer well to very different domains

### Contrastive learning

Contrastive learning is a self-supervised pretraining method that does *not* require any labels. Instead, it relies on clever use of data augmentation and a special loss function to learn useful representations.

#### The Core Idea

The key insight of contrastive learning is that **different views of the same image should have similar representations, while views of different images should have different representations**.

**Step-by-step example:**
1. Take an image $`x`$ from the dataset.
2. Apply two different random augmentations (e.g., cropping, color jitter, flipping) to $`x`$ to get $`\hat{x}`$ and $`\tilde{x}`$.
3. Pass both $`\hat{x}`$ and $`\tilde{x}`$ through the model to get their representations $`\phi_\theta(\hat{x})`$ and $`\phi_\theta(\tilde{x})`$.
4. Treat $`(\hat{x}, \tilde{x})`$ as a *positive pair* (they should be close in representation space).
5. For all other images in the batch, treat their augmentations as *negative pairs* (they should be far apart in representation space).

#### The Contrastive Loss Function

The most common contrastive loss is the **InfoNCE loss** (used in SimCLR):

```math
L_{\text{contrastive}}(\theta) = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}
```

Where:
- $`z_i = \phi_\theta(\hat{x}_i)`$ and $`z_j = \phi_\theta(\tilde{x}_i)`$ are representations of two augmentations of the same image
- $`\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}`$ is the cosine similarity
- $`\tau`$ is a temperature parameter that controls the sharpness of the distribution
- The denominator includes all other representations in the batch as negative examples

#### Data Augmentation Strategies

The choice of augmentations is crucial for contrastive learning. Common augmentations include:

**Geometric Transformations:**
- Random cropping and resizing
- Random horizontal flipping
- Random rotation
- Random translation

**Color Transformations:**
- Random color jittering (brightness, contrast, saturation, hue)
- Random grayscale conversion
- Random Gaussian blur
- Random solarization

**Why Augmentations Matter:**
- **Invariance Learning**: The model learns to be invariant to irrelevant transformations
- **Robustness**: Representations become robust to variations in the input
- **Generalization**: Better generalization to unseen data

#### Visual Analogy and Intuition

**Visual analogy:** Imagine a magnetic board where each image is a magnet. Positive pairs (augmentations of the same image) are pulled together, while negative pairs (different images) are pushed apart. The model learns to organize the representation space so that similar images are close and dissimilar images are far apart.

**Why does this work?** By pulling together representations of different views of the same image, the model learns to focus on the underlying content rather than superficial details. This leads to features that are robust and generalize well to new tasks.

#### The SimCLR Framework

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is one of the most successful contrastive learning methods:

**Architecture:**
1. **Encoder**: A neural network (e.g., ResNet) that maps images to representations
2. **Projection Head**: A small neural network that maps representations to a lower-dimensional space
3. **Loss Function**: InfoNCE loss applied to the projected representations

**Training Process:**
1. Sample a batch of images
2. Create two augmentations of each image
3. Pass through encoder to get representations
4. Pass through projection head to get projected representations
5. Apply contrastive loss
6. Update model parameters

**Example:**
- Use SimCLR to pretrain a model on unlabeled images.
- For each image, generate two augmentations and treat them as a positive pair.
- Use a contrastive loss to train the model so that positive pairs are close and negative pairs are far apart.

#### Advantages and Limitations

**Advantages:**
- **No Labels Required**: Can use massive amounts of unlabeled data
- **Strong Performance**: Often outperforms supervised pretraining
- **Robust Representations**: Learns features that are invariant to irrelevant transformations
- **Scalable**: Can leverage any amount of unlabeled data

**Limitations:**
- **Computationally Expensive**: Requires large batch sizes for effective training
- **Augmentation Dependent**: Performance heavily depends on choice of augmentations
- **Hyperparameter Sensitive**: Temperature and other parameters need careful tuning
- **Memory Intensive**: Requires storing representations for all images in batch

#### Comparison with Supervised Pretraining

| Aspect | Supervised Pretraining | Contrastive Learning |
|--------|----------------------|---------------------|
| **Data Requirements** | Large labeled dataset | Large unlabeled dataset |
| **Computational Cost** | Moderate | High (large batches) |
| **Performance** | Good | Often better |
| **Robustness** | Moderate | High |
| **Transfer Ability** | Task-specific | General |
| **Implementation** | Straightforward | Complex |

---

[^2]: Negative pairs are not guaranteed to be always semantically unrelated, but in practice, with large datasets and random sampling, this is a reasonable assumption.
[^4]: This is a variant and simplification of the original loss that does not change the essence (but may change the efficiency slightly).
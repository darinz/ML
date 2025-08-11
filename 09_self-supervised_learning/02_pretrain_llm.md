# 14.3 Pretrained large language models

## The Big Picture: Why Large Language Models Matter

**The Language Understanding Challenge:**
Imagine trying to teach a computer to understand and generate human language. Language is incredibly complex - it has grammar, context, ambiguity, idioms, cultural references, and infinite possible combinations. Traditional approaches required hand-crafted rules and extensive labeled data for each specific task.

**The Intuitive Analogy:**
Think of the difference between:
- **Traditional NLP**: Like teaching a computer specific rules for each language task (translation, summarization, question answering)
- **Large Language Models**: Like teaching a computer to read and understand language the way humans do - by absorbing massive amounts of text and learning patterns naturally

**Why Large Language Models Matter:**
- **Universal language understanding**: One model can handle many different language tasks
- **Massive knowledge base**: Learns from billions of documents and conversations
- **Emergent capabilities**: Develops abilities not explicitly trained for
- **Human-like reasoning**: Can understand context, follow instructions, and generate coherent text
- **Democratization**: Makes advanced language AI accessible to more applications

### The Key Insight

**From Task-Specific to General Language Intelligence:**
- **Traditional approach**: Train separate models for each language task
- **LLM approach**: Train one massive model that understands language broadly
- **Adaptation**: Use the same model for translation, summarization, coding, reasoning, etc.

**The Foundation Model Advantage:**
- **Scale**: Larger models often perform better and have emergent capabilities
- **Knowledge transfer**: What you learn about language helps with all language tasks
- **Few-shot learning**: Can learn new tasks with just a few examples
- **Zero-shot capabilities**: Can handle tasks it was never explicitly trained for

Natural language processing is another area where pretraining models are particularly successful. In language problems, an example typically corresponds to a document or generally a sequence (or trunk) of words,[^5] denoted by $x = (x_1, \cdots, x_T)$ where $T$ is the length of the document/sequence, $x_i \in \{1, \cdots, V\}$ are words in the document, and $V$ is the vocabulary size.[^6]

**The Language Data Structure:**
- **Sequence nature**: Language is inherently sequential (words come in order)
- **Variable length**: Documents can be short (tweets) or long (books)
- **Vocabulary size**: Large vocabularies (50K-100K+ words) create complexity
- **Context dependency**: Meaning depends on surrounding words

## From General Foundation Models to Language-Specific Applications

We've now explored **self-supervised learning and foundation models** - a paradigm shift that enables learning from vast amounts of unlabeled data. We've seen how contrastive learning works by creating surrogate tasks from the data itself, how foundation models can be adapted to various downstream tasks, and how this approach addresses the fundamental data labeling bottleneck in machine learning.

However, while the general principles of self-supervised learning apply across different data modalities, **natural language processing** presents unique challenges and opportunities that require specialized approaches. Language has its own structure, patterns, and characteristics that make it particularly well-suited for certain types of self-supervised learning.

**The Language-Specific Challenges:**
- **Sequential nature**: Words depend on previous words in complex ways
- **Context sensitivity**: Same word can mean different things in different contexts
- **Long-range dependencies**: Important information can be far apart in text
- **Ambiguity**: Many words and phrases have multiple meanings
- **Cultural knowledge**: Understanding requires world knowledge

**The Language-Specific Opportunities:**
- **Predictive structure**: Language has strong predictive patterns
- **Self-supervised tasks**: Many natural tasks can be created from text itself
- **Massive data**: Text data is abundant and easily accessible
- **Transfer learning**: Language knowledge transfers well across tasks

This motivates our exploration of **large language models (LLMs)** - specialized foundation models for text that leverage the sequential and contextual nature of language. We'll see how language modeling works through the chain rule of probability, how Transformer architectures process text, and how these models can generate coherent text and adapt to new tasks through prompting.

**The LLM Revolution:**
- **Scale**: Models with billions of parameters trained on massive text corpora
- **Architecture**: Transformer architecture enables parallel processing and long-range attention
- **Training**: Self-supervised learning on next-word prediction
- **Capabilities**: Text generation, understanding, reasoning, and task adaptation

The transition from general foundation models to language-specific applications represents the bridge from universal principles to domain expertise - taking our understanding of self-supervised learning and applying it to the rich, structured world of natural language.

In this section, we'll explore how language models work, how they're trained on massive text corpora, and how they can be adapted to various language tasks through finetuning, zero-shot learning, and in-context learning.

## Understanding Language Modeling

### The Big Picture: What is Language Modeling?

**The Core Problem:**
Language modeling is about predicting what comes next in a sequence of words. Given some text, we want to predict the probability of the next word, or the probability of entire sequences.

**The Intuitive Analogy:**
- **Language modeling**: Like predicting what someone will say next in a conversation
- **Probability estimation**: Like guessing how likely different responses are
- **Context understanding**: Like using what was said before to make better predictions

**Why Language Modeling Matters:**
- **Foundation for all NLP**: Most language tasks can be framed as language modeling
- **Natural self-supervised task**: No labels needed - just predict the next word
- **Universal representation**: Learns general language understanding
- **Scalable training**: Can use any text data for training

### Introduction to Language Modeling

Language modeling is one of the most fundamental tasks in natural language processing. At its core, it involves predicting the probability of sequences of words, which requires understanding the complex patterns and relationships in human language.

**The Language Understanding Challenge:**
- **Complexity**: Language has infinite possible combinations
- **Context**: Meaning depends on surrounding words and world knowledge
- **Ambiguity**: Same words can mean different things
- **Structure**: Grammar, syntax, and semantic relationships

**The Learning Approach:**
- **Pattern recognition**: Learn statistical patterns from massive text data
- **Context modeling**: Understand how words relate to each other
- **Knowledge acquisition**: Absorb factual and conceptual knowledge
- **Generalization**: Apply learned patterns to new situations

### The Language Modeling Problem

A language model is a probabilistic model representing the probability of a document, denoted by $p(x_1, \cdots, x_T)$. This probability distribution is very complex because its support size is $V^T$—exponential in the length of the document. Instead of modeling the distribution of a document itself, we can apply the chain rule of conditional probability to decompose it as follows:

$$
p(x_1, \cdots, x_T) = p(x_1) p(x_2|x_1) \cdots p(x_T|x_1, \cdots, x_{T-1}).
$$

**The Mathematical Challenge:**
- **Direct modeling**: Would require estimating $V^T$ parameters
- **Exponential growth**: Parameters grow exponentially with sequence length
- **Intractability**: Impossible to estimate for realistic document lengths
- **Solution**: Chain rule decomposition breaks the problem into manageable pieces

Now the support size of each of the conditional probability $p(x_t|x_1, \cdots, x_{t-1})$ is $V$.

**The Chain Rule Advantage:**
- **Linear growth**: Only $T \times V$ parameters needed
- **Conditional structure**: Each prediction depends only on previous words
- **Tractable**: Much more manageable computational complexity
- **Natural**: Matches how humans process language sequentially

#### Understanding the Chain Rule Decomposition

The chain rule decomposition is a fundamental concept in probability theory that allows us to break down complex joint distributions into simpler conditional distributions. Here's why this is crucial for language modeling:

**The Problem with Direct Modeling:**
- The vocabulary size $V$ is typically 50,000-100,000 words
- Document length $T$ can be hundreds or thousands of words
- Direct modeling would require estimating $V^T$ parameters
- For a vocabulary of 50,000 and document length of 100, this is $50,000^{100}$ parameters!

**The Exponential Explosion Analogy:**
- **Vocabulary**: Like having 50,000 different building blocks
- **Document**: Like building a structure with 100 blocks
- **Direct modeling**: Like trying to memorize every possible structure
- **Chain rule**: Like learning rules for adding one block at a time

**The Solution with Chain Rule:**
- Each conditional probability $p(x_t|x_1, \cdots, x_{t-1})$ only needs $V$ parameters
- Total parameters needed: $T \times V$ (much more manageable)
- Each prediction depends only on the previous words in the sequence

**The Sequential Learning Intuition:**
- **Step 1**: Learn to predict the first word $p(x_1)$
- **Step 2**: Learn to predict the second word given the first $p(x_2|x_1)$
- **Step 3**: Learn to predict the third word given the first two $p(x_3|x_1, x_2)$
- **Continue**: Build up understanding word by word

**Example:**
Consider the sentence "The cat sat on the mat":
- $p(\text{The})$: Probability of starting with "The"
- $p(\text{cat}|\text{The})$: Probability of "cat" given "The"
- $p(\text{sat}|\text{The cat})$: Probability of "sat" given "The cat"
- And so on...

**The Context Building Process:**
- **"The"**: No context, just word frequency
- **"The cat"**: Context helps predict animal-related words
- **"The cat sat"**: Context suggests action words
- **"The cat sat on"**: Context suggests spatial relationships
- **"The cat sat on the"**: Context suggests objects or surfaces

### Parameterizing the Language Model

We will model the conditional probability $p(x_t|x_1, \cdots, x_{t-1})$ as a function of $x_1, \ldots, x_{t-1}$ parameterized by some parameter $\theta$.

**The Parameterization Challenge:**
- **Input**: Sequence of discrete word indices
- **Output**: Probability distribution over vocabulary
- **Function**: Must capture complex language patterns
- **Learning**: Must be differentiable for gradient-based optimization

A parameterized model takes in numerical inputs and therefore we first introduce embeddings or representations for the words. Let $e_i \in \mathbb{R}^d$ be the embedding of the word $i \in \{1, 2, \cdots, V\}$. We call $[e_1, \cdots, e_V] \in \mathbb{R}^{d \times V}$ the embedding matrix.

#### Word Embeddings: From Discrete to Continuous

**The Challenge:**
- Words are discrete symbols (e.g., "cat", "dog", "house")
- Neural networks work with continuous numerical inputs
- We need a way to convert words to numbers

**The Discrete vs. Continuous Problem:**
- **Discrete words**: "cat", "dog", "house" - no natural numerical relationship
- **Neural networks**: Expect continuous, differentiable inputs
- **Solution**: Map each word to a continuous vector (embedding)
- **Learning**: Embeddings are learned to capture semantic relationships

**The Solution:**
- Each word is assigned a unique integer ID (e.g., "cat" = 42, "dog" = 17)
- Each word ID is mapped to a continuous vector (embedding)
- These embeddings are learned during training

**The Word-to-Vector Process:**
1. **Vocabulary creation**: Assign unique IDs to all words
2. **Embedding initialization**: Start with random vectors
3. **Learning**: Update embeddings based on word usage patterns
4. **Semantic capture**: Similar words get similar embeddings

**Properties of Word Embeddings:**
- **Dimensionality**: Typically 256-1024 dimensions
- **Semantic Similarity**: Similar words have similar embeddings
- **Learnable**: Embeddings are updated during training to capture word relationships
- **Context-independent**: Each word has a fixed embedding (in basic models)

**The Semantic Space Analogy:**
- **Words**: Like points in a high-dimensional space
- **Similar words**: Like points that are close together
- **Related concepts**: Like clusters of nearby points
- **Semantic relationships**: Like geometric relationships between points

**Example:**
```
Vocabulary: ["the", "cat", "sat", "on", "mat"]
Word IDs: [1, 2, 3, 4, 5]
Embeddings: 
  e_1 (the) = [0.1, 0.2, 0.3, ...]
  e_2 (cat) = [0.4, 0.1, 0.8, ...]
  e_3 (sat) = [0.2, 0.9, 0.1, ...]
  ...
```

**The Embedding Learning Process:**
- **Initialization**: Random vectors for each word
- **Context learning**: Words that appear in similar contexts get similar embeddings
- **Semantic capture**: "cat" and "dog" become similar because they appear in similar contexts
- **Relationship learning**: "king" - "man" + "woman" ≈ "queen"

## The Transformer Architecture

The most commonly used model is the Transformer [Vaswani et al., 2017]. In this subsection, we will introduce the input-output interface of a Transformer, but treat the intermediate computation in the Transformer as a blackbox. We will cover more of the transformer architecture in [section 12](../12_llm/).

### The Big Picture: Why Transformers Matter

**The Sequential Processing Problem:**
Traditional neural networks (RNNs, LSTMs) process text word by word sequentially. This creates several problems:
- **Slow training**: Can't parallelize across sequence length
- **Memory issues**: Hard to handle long sequences
- **Gradient problems**: Information can get lost over long distances
- **Limited context**: Hard to capture long-range dependencies

**The Transformer Solution:**
- **Parallel processing**: Can process entire sequences at once
- **Attention mechanism**: Can directly attend to any position in the sequence
- **Long-range dependencies**: Can capture relationships across long distances
- **Scalable**: Performance improves with more data and larger models

### Transformer Overview

The Transformer is a neural network architecture that revolutionized natural language processing. Unlike previous models that processed sequences sequentially (like RNNs), Transformers can process entire sequences in parallel, making them much faster to train and more effective at capturing long-range dependencies.

**The Parallel Processing Advantage:**
- **RNNs**: Must process word 1, then word 2, then word 3...
- **Transformers**: Can process all words simultaneously
- **Speed**: Much faster training and inference
- **Efficiency**: Better utilization of modern hardware (GPUs)

**The Attention Revolution:**
- **Traditional models**: Each word only sees previous words
- **Transformers**: Each word can attend to any other word
- **Context**: Full context available for every prediction
- **Relationships**: Can capture complex relationships between distant words

#### Key Components of the Transformer

1. **Input Embeddings**: Convert word tokens to continuous vectors
2. **Positional Encodings**: Add information about word positions in the sequence
3. **Multi-Head Self-Attention**: Allow words to attend to all other words in the sequence
4. **Feed-Forward Networks**: Process each position independently
5. **Layer Normalization**: Stabilize training
6. **Residual Connections**: Help with gradient flow

**The Transformer Architecture Analogy:**
- **Input Embeddings**: Like translating words into a common language
- **Positional Encodings**: Like adding timestamps to know word order
- **Self-Attention**: Like having a spotlight that can focus on any word
- **Feed-Forward**: Like processing each word's meaning independently
- **Layer Normalization**: Like keeping the signal strength consistent
- **Residual Connections**: Like having shortcuts to preserve information

### Input-Output Interface

As shown in Figure 14.1, given a document $(x_1, \cdots, x_T)$, we first translate the sequence of discrete variables into a sequence of corresponding word embeddings ($e_{x_1}, \cdots, e_{x_T}$).

<img src="./img/transformer_output.png" width="400px"/>

**Figure 14.1 (description):** The inputs to the Transformer are the embeddings $e_{x_0}, e_{x_1}, \ldots, e_{x_T}$ corresponding to the tokens $x_0, x_1, \ldots, x_T$. The Transformer $f_\theta(x)$ outputs a sequence of vectors $u_1, u_2, \ldots, u_{T+1}$, each of which is used to predict the next token in the sequence.

**The Input Processing Pipeline:**
1. **Word tokens**: Discrete word IDs (e.g., [1, 42, 17, 8, 23])
2. **Word embeddings**: Convert to continuous vectors
3. **Positional encoding**: Add position information
4. **Transformer processing**: Apply attention and feed-forward layers
5. **Output logits**: Continuous vectors for next-word prediction

We also introduce a fixed special token $x_0 = \perp$ in the vocabulary with corresponding embedding $e_{x_0}$ to mark the beginning of a document. Then, the word embeddings are passed into a Transformer model, which takes in a sequence of vectors ($e_{x_0}, e_{x_1}, \cdots, e_{x_T}$) and outputs a sequence of vectors ($u_1, u_2, \cdots, u_{T+1}$), where $u_t \in \mathbb{R}^V$ will be interpreted as the logits for the probability distribution of the next word.

#### Understanding the Special Token

The special token $\perp$ (often called `<BOS>` for "beginning of sequence") serves several important purposes:

1. **Sequence Start Marker**: Indicates the beginning of a new sequence
2. **Context for First Word**: Provides context for predicting the first actual word
3. **Consistent Input Length**: Ensures all sequences have the same structure

**The Special Token Analogy:**
- **Purpose**: Like a "start" button that tells the model a new sequence is beginning
- **Context**: Like having a blank slate before the first word
- **Consistency**: Like always having the same starting point for every sequence
- **Prediction**: Like having a reference point for predicting the first word

**Why We Need It:**
- **First word prediction**: Without it, the model has no context for predicting the first word
- **Sequence boundaries**: Helps the model know when one sequence ends and another begins
- **Training consistency**: Ensures all sequences have the same structure
- **Generation**: Provides a starting point for text generation

#### The Autoregressive Property

Here we use the autoregressive version of the Transformer, which by design ensures $u_t$ only depends on $x_1, \cdots, x_{t-1}$ (note that this property does not hold in masked language models [Devlin et al., 2019] where the losses are also different).

**What is Autoregressive?**
- Each output depends only on previous inputs
- No information from future tokens is used
- This is crucial for language generation tasks

**The Autoregressive Intuition:**
- **Causal structure**: Like predicting the next word in a sentence
- **No cheating**: Can't use information from words that haven't been written yet
- **Generation**: Perfect for text generation (predict one word at a time)
- **Training**: Each position predicts the next word given previous words

**Example:**
For the sequence "The cat sat":
- $u_1$ depends on $x_0$ (special token) only
- $u_2$ depends on $x_0, x_1$ ("The")
- $u_3$ depends on $x_0, x_1, x_2$ ("The cat")
- $u_4$ depends on $x_0, x_1, x_2, x_3$ ("The cat sat")

**The Prediction Process:**
- **Position 1**: Predict word after start token
- **Position 2**: Predict word after "The"
- **Position 3**: Predict word after "The cat"
- **Position 4**: Predict word after "The cat sat"

We view the whole mapping from $x$'s to $u$'s as a blackbox in this subsection and call it a Transformer, denoted it by $f_\theta$, where $\theta$ include both the parameters in the Transformer and the input embeddings. We write $u_t = f_\theta(x_0, x_1, \ldots, x_{t-1})$ where $f_\theta$ denotes the mapping from the input to the outputs.

**The Blackbox Understanding:**
- **Input**: Sequence of word embeddings with positional information
- **Processing**: Complex attention and feed-forward computations
- **Output**: Logits for next-word prediction at each position
- **Parameters**: All weights in the Transformer and embeddings

---

**Next: [From Logits to Probabilities](02_pretrain_llm.md#from-logits-to-probabilities)** - Learn how to convert model outputs into probability distributions.

## Introduction to Language Modeling

Language modeling is one of the most fundamental tasks in natural language processing. At its core, it involves predicting the probability of sequences of words, which requires understanding the complex patterns and relationships in human language.

### The Language Modeling Problem

A language model is a probabilistic model representing the probability of a document, denoted by $p(x_1, \cdots, x_T)$. This probability distribution is very complex because its support size is $V^T$—exponential in the length of the document. Instead of modeling the distribution of a document itself, we can apply the chain rule of conditional probability to decompose it as follows:

$$
p(x_1, \cdots, x_T) = p(x_1) p(x_2|x_1) \cdots p(x_T|x_1, \cdots, x_{T-1}).
$$

Now the support size of each of the conditional probability $p(x_t|x_1, \cdots, x_{t-1})$ is $V$.

#### Understanding the Chain Rule Decomposition

The chain rule decomposition is a fundamental concept in probability theory that allows us to break down complex joint distributions into simpler conditional distributions. Here's why this is crucial for language modeling:

**The Problem with Direct Modeling:**
- The vocabulary size $V$ is typically 50,000-100,000 words
- Document length $T$ can be hundreds or thousands of words
- Direct modeling would require estimating $V^T$ parameters
- For a vocabulary of 50,000 and document length of 100, this is $50,000^{100}$ parameters!

**The Solution with Chain Rule:**
- Each conditional probability $p(x_t|x_1, \cdots, x_{t-1})$ only needs $V$ parameters
- Total parameters needed: $T \times V$ (much more manageable)
- Each prediction depends only on the previous words in the sequence

**Example:**
Consider the sentence "The cat sat on the mat":
- $p(\text{The})$: Probability of starting with "The"
- $p(\text{cat}|\text{The})$: Probability of "cat" given "The"
- $p(\text{sat}|\text{The cat})$: Probability of "sat" given "The cat"
- And so on...

### Parameterizing the Language Model

We will model the conditional probability $p(x_t|x_1, \cdots, x_{t-1})$ as a function of $x_1, \ldots, x_{t-1}$ parameterized by some parameter $\theta$.

A parameterized model takes in numerical inputs and therefore we first introduce embeddings or representations for the words. Let $e_i \in \mathbb{R}^d$ be the embedding of the word $i \in \{1, 2, \cdots, V\}$. We call $[e_1, \cdots, e_V] \in \mathbb{R}^{d \times V}$ the embedding matrix.

#### Word Embeddings: From Discrete to Continuous

**The Challenge:**
- Words are discrete symbols (e.g., "cat", "dog", "house")
- Neural networks work with continuous numerical inputs
- We need a way to convert words to numbers

**The Solution:**
- Each word is assigned a unique integer ID (e.g., "cat" = 42, "dog" = 17)
- Each word ID is mapped to a continuous vector (embedding)
- These embeddings are learned during training

**Properties of Word Embeddings:**
- **Dimensionality**: Typically 256-1024 dimensions
- **Semantic Similarity**: Similar words have similar embeddings
- **Learnable**: Embeddings are updated during training to capture word relationships

**Example:**
```
Vocabulary: ["the", "cat", "sat", "on", "mat"]
Word IDs: [1, 2, 3, 4, 5]
Embeddings: 
  e_1 (the) = [0.1, 0.2, 0.3, ...]
  e_2 (cat) = [0.4, 0.1, 0.8, ...]
  e_3 (sat) = [0.2, 0.9, 0.1, ...]
  ...
```

## The Transformer Architecture

The most commonly used model is the Transformer [Vaswani et al., 2017]. In this subsection, we will introduce the input-output interface of a Transformer, but treat the intermediate computation in the Transformer as a blackbox. We will cover more of the transformer architecture in [section 12](../12_llm/).

### Transformer Overview

The Transformer is a neural network architecture that revolutionized natural language processing. Unlike previous models that processed sequences sequentially (like RNNs), Transformers can process entire sequences in parallel, making them much faster to train and more effective at capturing long-range dependencies.

#### Key Components of the Transformer

1. **Input Embeddings**: Convert word tokens to continuous vectors
2. **Positional Encodings**: Add information about word positions in the sequence
3. **Multi-Head Self-Attention**: Allow words to attend to all other words in the sequence
4. **Feed-Forward Networks**: Process each position independently
5. **Layer Normalization**: Stabilize training
6. **Residual Connections**: Help with gradient flow

### Input-Output Interface

As shown in Figure 14.1, given a document $(x_1, \cdots, x_T)$, we first translate the sequence of discrete variables into a sequence of corresponding word embeddings ($e_{x_1}, \cdots, e_{x_T}$).

<img src="./img/transformer_output.png" width="400px"/>

**Figure 14.1 (description):** The inputs to the Transformer are the embeddings $e_{x_0}, e_{x_1}, \ldots, e_{x_T}$ corresponding to the tokens $x_0, x_1, \ldots, x_T$. The Transformer $f_\theta(x)$ outputs a sequence of vectors $u_1, u_2, \ldots, u_{T+1}$, each of which is used to predict the next token in the sequence.

We also introduce a fixed special token $x_0 = \perp$ in the vocabulary with corresponding embedding $e_{x_0}$ to mark the beginning of a document. Then, the word embeddings are passed into a Transformer model, which takes in a sequence of vectors ($e_{x_0}, e_{x_1}, \cdots, e_{x_T}$) and outputs a sequence of vectors ($u_1, u_2, \cdots, u_{T+1}$), where $u_t \in \mathbb{R}^V$ will be interpreted as the logits for the probability distribution of the next word.

#### Understanding the Special Token

The special token $\perp$ (often called `<BOS>` for "beginning of sequence") serves several important purposes:

1. **Sequence Start Marker**: Indicates the beginning of a new sequence
2. **Context for First Word**: Provides context for predicting the first actual word
3. **Consistent Input Length**: Ensures all sequences have the same structure

#### The Autoregressive Property

Here we use the autoregressive version of the Transformer, which by design ensures $u_t$ only depends on $x_1, \cdots, x_{t-1}$ (note that this property does not hold in masked language models [Devlin et al., 2019] where the losses are also different).

**What is Autoregressive?**
- Each output depends only on previous inputs
- No information from future tokens is used
- This is crucial for language generation tasks

**Example:**
For the sequence "The cat sat":
- $u_1$ depends on $x_0$ (special token) only
- $u_2$ depends on $x_0, x_1$ ("The")
- $u_3$ depends on $x_0, x_1, x_2$ ("The cat")
- $u_4$ depends on $x_0, x_1, x_2, x_3$ ("The cat sat")

We view the whole mapping from $x$'s to $u$'s as a blackbox in this subsection and call it a Transformer, denoted it by $f_\theta$, where $\theta$ include both the parameters in the Transformer and the input embeddings. We write $u_t = f_\theta(x_0, x_1, \ldots, x_{t-1})$ where $f_\theta$ denotes the mapping from the input to the outputs.


### From Logits to Probabilities

The conditional probability $p(x_t|x_1, \cdots, x_{t-1})$ is the softmax of the logits:

$$
\begin{bmatrix}
p(x_t = 1|x_1 \cdots, x_{t-1}) \\
p(x_t = 2|x_1 \cdots, x_{t-1}) \\
\vdots \\
p(x_t = V|x_1 \cdots, x_{t-1})
\end{bmatrix}
= \mathrm{softmax}(u_t) \in \mathbb{R}^V \tag{14.6}
$$

or equivalently,

$$
= \mathrm{softmax}(f_\theta(x_0, \ldots, x_{t-1})) \tag{14.7}
$$

#### Understanding Softmax

The softmax function converts a vector of logits into a probability distribution:

$$
\text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^V \exp(z_j)}
$$

**The Softmax Intuition:**
- **Raw scores**: Logits can be any real numbers (positive, negative, large, small)
- **Exponentiation**: Makes all values positive and amplifies differences
- **Normalization**: Divides by sum to make probabilities add to 1
- **Probability distribution**: Output is a valid probability distribution

**The Temperature Analogy:**
- **Logits**: Like raw scores on a test (could be negative, any magnitude)
- **Exponentiation**: Like converting to a positive scale
- **Normalization**: Like converting to percentages that sum to 100%
- **Probability**: Like the chance of each outcome

**Properties:**
- All outputs are positive (probabilities)
- Sum to 1 (valid probability distribution)
- Preserves relative ordering of logits
- Amplifies differences between large and small values

**The Amplification Effect:**
- **Large logits**: Get much larger after exponentiation
- **Small logits**: Stay small after exponentiation
- **Result**: Sharpens the distribution, making confident predictions more confident
- **Purpose**: Helps the model make clear, confident predictions

**Example:**
If $u_t = [2.0, 1.0, 0.5]$ for a 3-word vocabulary:
- $\exp(u_t) = [7.39, 2.72, 1.65]$
- $\sum \exp(u_t) = 11.76$
- $\text{softmax}(u_t) = [0.63, 0.23, 0.14]$

**The Numerical Process:**
1. **Input logits**: [2.0, 1.0, 0.5]
2. **Exponentiate**: [e^2.0, e^1.0, e^0.5] = [7.39, 2.72, 1.65]
3. **Sum**: 7.39 + 2.72 + 1.65 = 11.76
4. **Normalize**: [7.39/11.76, 2.72/11.76, 1.65/11.76] = [0.63, 0.23, 0.14]

## Training the Language Model

We train the Transformer parameter $\theta$ by minimizing the negative log-likelihood of seeing the data under the probabilistic model defined by $\theta$.

which is the cross-entropy loss on the logits.

$$
\text{loss}(\theta) = \frac{1}{T} \sum_{t=1}^T -\log(p_\theta(x_t|x_1, \ldots, x_{t-1})) \tag{14.8}
$$

$$
= \frac{1}{T} \sum_{t=1}^T \ell_{ce}(f_\theta(x_0, x_1, \cdots, x_{t-1}), x_t)
$$

$$
= \frac{1}{T} \sum_{t=1}^T -\log(\mathrm{softmax}(f_\theta(x_0, x_1, \cdots, x_{t-1}))_{x_t})
$$

### The Big Picture: What is Language Model Training?

**The Training Goal:**
Language model training is about teaching the model to predict the next word accurately. We want the model to assign high probability to the correct next word and low probability to incorrect words.

**The Learning Process:**
- **Input**: Sequences of words from a large text corpus
- **Target**: Predict the next word at each position
- **Loss**: Measure how well the model predicts the correct words
- **Optimization**: Update model parameters to improve predictions

**The Training Intuition:**
- **Teacher**: The training data shows what words actually come next
- **Student**: The model tries to predict what comes next
- **Feedback**: Loss function tells the model how wrong its predictions are
- **Learning**: Model adjusts its parameters to make better predictions

#### Understanding the Loss Function

**Cross-Entropy Loss:**
The cross-entropy loss measures how well our predicted probability distribution matches the true distribution (which is a one-hot vector for the correct word).

$$
\ell_{ce}(u, x_t) = -\log(\text{softmax}(u)_{x_t})
$$

**The Cross-Entropy Intuition:**
- **Predicted distribution**: Model's guess about next word probabilities
- **True distribution**: One-hot vector (1 for correct word, 0 for others)
- **Cross-entropy**: Measures how different these distributions are
- **Minimization**: Want predicted distribution to match true distribution

**Why This Works:**
- If the model assigns high probability to the correct word, $-\log(p)$ is small
- If the model assigns low probability to the correct word, $-\log(p)$ is large
- The loss encourages the model to assign high probability to correct words

**The Penalty System:**
- **Good prediction**: High probability for correct word → small penalty
- **Bad prediction**: Low probability for correct word → large penalty
- **Perfect prediction**: Probability = 1 → penalty = 0
- **Terrible prediction**: Probability ≈ 0 → penalty → ∞

**Example:**
For the sequence "The cat sat" and $u_2$ (predicting the word after "The"):
- If the model predicts $p(\text{cat}) = 0.8$, loss = $-\log(0.8) = 0.22$
- If the model predicts $p(\text{cat}) = 0.1$, loss = $-\log(0.1) = 2.30$

**The Learning Signal:**
- **High confidence, correct**: Small loss, model is doing well
- **High confidence, wrong**: Large loss, model is overconfident
- **Low confidence, correct**: Medium loss, model is uncertain
- **Low confidence, wrong**: Large loss, model is wrong and uncertain

#### Training Process

1. **Forward Pass**: For each position $t$, compute $u_t = f_\theta(x_0, \ldots, x_{t-1})$
2. **Loss Computation**: Compute cross-entropy loss for each position
3. **Backward Pass**: Compute gradients with respect to $\theta$
4. **Parameter Update**: Update $\theta$ using gradient descent

**The Training Loop:**
- **Step 1**: Take a batch of text sequences
- **Step 2**: For each position, predict the next word
- **Step 3**: Compare predictions with actual next words
- **Step 4**: Compute loss (how wrong the predictions are)
- **Step 5**: Compute gradients (how to change parameters)
- **Step 6**: Update parameters to reduce loss
- **Step 7**: Repeat with next batch

**The Batch Processing:**
- **Single sequence**: Process one text sequence at a time
- **Batch processing**: Process multiple sequences simultaneously
- **Parallelization**: Can use GPU to process many sequences in parallel
- **Efficiency**: Much faster than processing sequences one by one

**The Gradient Flow:**
- **Loss computation**: How wrong is the model at each position?
- **Gradient computation**: How should each parameter change?
- **Parameter update**: Adjust parameters to reduce loss
- **Convergence**: Parameters converge to values that minimize loss

## Text Generation with Language Models

**Autoregressive text decoding / generation.** Given an autoregressive Transformer, we can simply sample text from it sequentially. Given a prefix $x_1, \ldots, x_t$, we generate text completion $x_{t+1}, \ldots, x_T$ sequentially using the conditional distribution.

$$
x_{t+1} \sim \mathrm{softmax}(f_\theta(x_0, x_1, \cdots, x_t)) \tag{14.9}
$$
$$
x_{t+2} \sim \mathrm{softmax}(f_\theta(x_0, x_1, \cdots, x_{t+1})) \tag{14.10}
$$
$$
\vdots \tag{14.11}
$$
$$
x_T \sim \mathrm{softmax}(f_\theta(x_0, x_1, \cdots, x_{T-1})) \tag{14.12}
$$

### The Big Picture: How Text Generation Works

**The Generation Process:**
Text generation is like having a conversation with the model. You give it some starting text (prefix), and it continues the conversation by predicting what comes next, one word at a time.

**The Autoregressive Nature:**
- **Sequential generation**: Generate one word at a time
- **Context building**: Each new word depends on all previous words
- **Feedback loop**: Generated words become input for next predictions
- **Cumulative context**: Context grows as generation proceeds

**The Generation Intuition:**
- **Starting point**: Given some initial text (prompt)
- **Prediction**: Model predicts probability distribution over next word
- **Sampling**: Choose next word based on probabilities
- **Extension**: Add chosen word to text and repeat

Note that each generated token is used as the input to the model when generating the following tokens. In practice, people often introduce a parameter $\tau > 0$ named *temperature* to further adjust the entropy/sharpness of the generated distribution,

$$
x_{t+1} \sim \mathrm{softmax}(f_\theta(x_0, x_1, \cdots, x_t)/\tau) \tag{14.13}
$$
$$
x_{t+2} \sim \mathrm{softmax}(f_\theta(x_0, x_1, \cdots, x_{t+1})/\tau) \tag{14.14}
$$
$$
\vdots \tag{14.15}
$$
$$
x_T \sim \mathrm{softmax}(f_\theta(x_0, x_1, \cdots, x_{T-1})/\tau) \tag{14.16}
$$

### Understanding Temperature

The temperature parameter $\tau$ controls the randomness of text generation:

**The Temperature Analogy:**
- **Temperature**: Like adjusting the "creativity" or "confidence" of the model
- **Low temperature**: Model is very confident and predictable
- **High temperature**: Model is more uncertain and creative
- **Optimal temperature**: Balance between coherence and diversity

**Low Temperature ($\tau < 1$):**
- Makes the distribution more "peaked" (concentrated)
- Model becomes more confident in its predictions
- Generated text is more deterministic and focused
- Risk of repetitive or boring text

**The Low Temperature Effect:**
- **Sharpening**: Amplifies differences between probabilities
- **Confidence**: Model becomes more certain about its predictions
- **Consistency**: Generated text is more predictable
- **Repetition**: May get stuck in loops or patterns

**High Temperature ($\tau > 1$):**
- Makes the distribution more "flat" (spread out)
- Model becomes less confident in its predictions
- Generated text is more diverse and creative
- Risk of incoherent or nonsensical text

**The High Temperature Effect:**
- **Flattening**: Reduces differences between probabilities
- **Uncertainty**: Model becomes less certain about predictions
- **Diversity**: Generated text is more varied
- **Creativity**: May produce unexpected but interesting text

**Example:**
Consider logits $u = [2.0, 1.0, 0.5]$:
- $\tau = 0.5$: $\text{softmax}(u/0.5) = [0.88, 0.11, 0.01]$ (very confident)
- $\tau = 1.0$: $\text{softmax}(u/1.0) = [0.63, 0.23, 0.14]$ (balanced)
- $\tau = 2.0$: $\text{softmax}(u/2.0) = [0.42, 0.31, 0.27]$ (uncertain)

**The Temperature Scaling Process:**
1. **Original logits**: [2.0, 1.0, 0.5]
2. **Divide by temperature**: [2.0/τ, 1.0/τ, 0.5/τ]
3. **Apply softmax**: Convert to probabilities
4. **Sample**: Choose next word based on probabilities

When $\tau = 1$, the text is sampled from the original conditional probability defined by the model. With a decreasing $\tau$, the generated text gradually becomes more "deterministic". $\tau \to 0$ reduces to greedy decoding, where we generate the most probable next token from the conditional probability.

**The Temperature Extremes:**
- **$\tau \to 0$**: Greedy decoding (always choose most probable word)
- **$\tau = 1$**: Standard sampling (use model's original probabilities)
- **$\tau \to \infty$**: Uniform sampling (equal probability for all words)

### Generation Strategies

**Greedy Decoding ($\tau \to 0$):**
- Always choose the most probable next word
- Fast and deterministic
- Often produces repetitive or boring text

**The Greedy Decoding Analogy:**
- **Strategy**: Always take the safest, most obvious choice
- **Speed**: Very fast (no randomness to compute)
- **Quality**: Often repetitive or predictable
- **Use case**: When you want consistent, predictable output

**Random Sampling ($\tau = 1$):**
- Sample according to the model's probability distribution
- More diverse output
- Can be less coherent

**The Random Sampling Analogy:**
- **Strategy**: Let the model's confidence guide choices
- **Diversity**: More varied output than greedy
- **Coherence**: Generally maintains reasonable coherence
- **Use case**: When you want natural, varied text

**Top-k Sampling:**
- Only consider the top $k$ most probable words
- Sample from this restricted set
- Balances diversity and coherence

**The Top-k Sampling Analogy:**
- **Strategy**: Consider only the most reasonable options
- **Control**: Limits choices to top k candidates
- **Balance**: Good balance between diversity and coherence
- **Use case**: When you want controlled creativity

**Nucleus Sampling (Top-p):**
- Consider words until cumulative probability reaches $p$
- Sample from this dynamic set
- Often produces better results than fixed top-k

**The Nucleus Sampling Analogy:**
- **Strategy**: Consider options until you have enough probability mass
- **Adaptive**: Number of options varies based on model confidence
- **Quality**: Often produces better text than fixed top-k
- **Use case**: When you want high-quality, diverse text

**The Sampling Strategy Comparison:**
- **Greedy**: Fastest, most predictable, often repetitive
- **Random**: Natural diversity, good coherence
- **Top-k**: Controlled diversity, consistent quality
- **Nucleus**: Best quality, adaptive diversity

---

## 14.3.1 Zero-shot learning and in-context learning

For language models, there are many ways to adapt a pretrained model to downstream tasks. In this notes, we discuss three of them: finetuning, zero-shot learning, and in-context learning.

### The Big Picture: Adaptation Methods for Language Models

**The Adaptation Challenge:**
Once we have a pretrained language model, we need to use it for specific tasks. Different tasks require different adaptation strategies, depending on the amount of labeled data available and the specific requirements.

**The Adaptation Spectrum:**
- **Finetuning**: Lots of labeled data, best performance
- **In-context learning**: Few labeled examples, good performance
- **Zero-shot learning**: No labeled data, variable performance

**The Key Insight:**
Language models can adapt to new tasks without traditional training by using their general language understanding and the structure of the task itself.

### Finetuning

**Finetuning** is not very common for the autoregressive language models that we introduced in Section 14.3 but much more common for other variants such as masked language models which has similar input-output interfaces but are pretrained differently [Devlin et al., 2019]. 

Finetuning means taking a model that has already learned a lot from a huge dataset (pretraining), and then continuing to train it on a smaller, task-specific dataset. Think of it like a student who has read many books (pretraining) and then studies specifically for a test (finetuning). The model adapts its knowledge to do well on the new task.

**The Finetuning Analogy:**
- **Pretraining**: Like getting a general education (reading many books)
- **Finetuning**: Like studying for a specific exam (focused learning)
- **Adaptation**: Like using general knowledge to learn specific skills
- **Specialization**: Like becoming an expert in a particular area

**The Finetuning Process:**
1. **Start with pretrained model**: Model already knows general language patterns
2. **Add task-specific data**: Small dataset for the specific task
3. **Continue training**: Update model parameters on task data
4. **Task specialization**: Model becomes good at the specific task

#### Mathematical Formulation

The finetuning method is the same as introduced generally in Section 14.1—the only question is how we define the prediction task with an additional linear head. One option is to treat $c_{T+1} = \phi_\theta(x_1, \cdots, x_T)$ as the representation and use $w^\top c_{T+1} = w^\top \phi_\theta(x_1, \cdots, x_T)$ to predict the task label. As described in Section 14.1, we initialize $\theta$ to the pretrained model $\hat{\theta}$ and then optimize both $w$ and $\theta$.

**The Finetuning Mathematics:**
- **Representation extraction**: $c_{T+1} = \phi_\theta(x_1, \cdots, x_T)$
- **Task prediction**: $y = w^\top c_{T+1}$
- **Parameter initialization**: $\theta \leftarrow \hat{\theta}$ (pretrained weights)
- **Joint optimization**: Update both $w$ and $\theta$

**The Representation Learning:**
- **$c_{T+1}$**: Final representation from the language model
- **$w$**: Task-specific weights to be learned
- **$y$**: Task prediction (classification, regression, etc.)
- **Learning**: Both representation and task weights are updated

#### When to Use Finetuning

**Advantages:**
- **Best Performance**: Can achieve highest accuracy on the target task
- **Task-Specific Adaptation**: Model can learn task-specific patterns
- **Flexibility**: Can adapt to any task format
- **Optimization**: Can optimize for task-specific metrics

**Disadvantages:**
- **Requires Labeled Data**: Needs significant amount of task-specific data
- **Computationally Expensive**: Requires training the entire model
- **Risk of Catastrophic Forgetting**: May lose general knowledge
- **Task-Specific**: Need separate model for each task

**The Catastrophic Forgetting Problem:**
- **What happens**: Model forgets general knowledge while learning specific task
- **Why it happens**: Parameters optimized for specific task may not preserve general knowledge
- **Mitigation**: Use smaller learning rates, regularization, or parameter-efficient methods
- **Trade-off**: Task performance vs. general knowledge preservation

**When to Use:**
- You have a large labeled dataset for your specific task
- You want the best possible performance
- You have sufficient computational resources
- The task is very different from pretraining

- **Why does this work?** The pretrained model already knows a lot about language, so it only needs to "tune" itself to the specifics of the new task. This is much faster and requires less data than training from scratch.
- **Tip:** Use finetuning when you have a moderate or large labeled dataset for your specific task, and you want the best possible performance.

---

### Zero-shot Learning

**Zero-shot** adaptation or zero-shot learning is the setting where there is no input-output pairs from the downstream tasks. For language problems tasks, typically the task is formatted as a question or a cloze test form via natural language. 

#### How Zero-shot Learning Works

- **Analogy:** Imagine you ask a well-read person a question they've never seen before, but they can answer it because they have broad general knowledge.
- **How does it work?** The model is given a prompt (like a question) and must generate the answer using only what it learned during pretraining. No additional training is done for the new task.

**The Zero-shot Intuition:**
- **No examples**: Task is presented without any training examples
- **General knowledge**: Model uses its broad understanding from pretraining
- **Task understanding**: Model must understand what the task is asking for
- **Answer generation**: Model generates appropriate response

**The Zero-shot Process:**
1. **Task formulation**: Convert task into natural language prompt
2. **Model understanding**: Model interprets what the task requires
3. **Knowledge application**: Model applies relevant knowledge
4. **Answer generation**: Model generates appropriate response

#### Task Formatting

The key insight is that we can format many tasks as natural language questions or prompts. The model then generates text that answers the question.

For example, we can format an example as a question:

$x_{\text{task}} = (x_{\text{task},1}, \cdots, x_{\text{task},R}) = \text{"Is the speed of light a universal constant?"}$

Then, we compute the most likely next word predicted by the language model given this question, that is, computing $\text{argmax}_{x_{T+1}} p(x_{T+1} \mid x_{\text{task},1}, \cdots, x_{\text{task},R})$. In this case, if the most likely next word $x_{T+1}$ is "No", then we solve the task. (The speed of light is only a constant in vacuum.)

**The Task Formatting Strategy:**
- **Natural language**: Convert task into question or instruction
- **Clear instructions**: Make it obvious what the model should do
- **Expected format**: Specify the format of the expected answer
- **Context provision**: Provide necessary context for the task

**The Prompt Engineering:**
- **Question format**: "What is the capital of France?"
- **Instruction format**: "Translate this to French: Hello world"
- **Cloze format**: "The capital of France is _____"
- **Multiple choice**: "Is this positive or negative? The movie was great."

#### Examples of Zero-shot Tasks

**Classification Tasks:**
- **Sentiment Analysis**: "The movie was terrible. Sentiment: [positive/negative]"
- **Topic Classification**: "This article discusses climate change. Topic: [politics/science/sports]"
- **Language Detection**: "Bonjour, comment allez-vous? Language: [English/French/Spanish]"

**Generation Tasks:**
- **Translation**: "Translate to French: Hello world →"
- **Summarization**: "Summarize this text: [text] →"
- **Question Answering**: "What is the capital of France? Answer:"

**Reasoning Tasks:**
- **Mathematical Reasoning**: "What is 15 + 27? Answer:"
- **Logical Reasoning**: "If all A are B, and all B are C, then all A are C. True or false?"
- **Common Sense**: "Can a person run faster than a car? Answer:"

**The Task Diversity:**
- **Language tasks**: Translation, summarization, question answering
- **Reasoning tasks**: Math, logic, common sense
- **Creative tasks**: Writing, poetry, code generation
- **Analysis tasks**: Sentiment, topic classification, fact checking

#### Advantages and Limitations

**Advantages:**
- **No Training Required**: Works immediately without any additional training
- **No Labeled Data**: Can work on tasks with no labeled examples
- **Rapid Deployment**: Can be applied to new tasks instantly
- **Cost Effective**: No computational cost for adaptation

**The Zero-shot Benefits:**
- **Immediate use**: No waiting for training to complete
- **No data collection**: Can work with tasks where data is hard to collect
- **Scalability**: Can handle many tasks with one model
- **Accessibility**: Makes AI accessible to more users

**Limitations:**
- **Inconsistent Performance**: Results can be unreliable
- **Limited Control**: Cannot fine-tune behavior for specific tasks
- **Prompt Sensitivity**: Performance heavily depends on prompt formulation
- **Knowledge Cutoff**: Limited to knowledge from pretraining

**The Zero-shot Challenges:**
- **Unpredictable**: Performance varies significantly across tasks
- **Prompt engineering**: Requires careful prompt design
- **No learning**: Cannot improve with more examples
- **Knowledge limits**: Cannot access information beyond pretraining

- **Why does this work?** During pretraining, the model has seen so many examples of language that it can often generalize to new questions or tasks, even if it has never seen them before.
- **Tip:** Zero-shot is useful when you have no labeled data for your task, or want to quickly test what a model can do "out of the box."

---

### In-context Learning

**In-context learning** is mostly used for few-shot settings where we have a few labeled examples $(x^{(1)}_{\text{task}}, y^{(1)}_{\text{task}}), \cdots, (x^{(n_{\text{task}})}_{\text{task}}, y^{(n_{\text{task}})}_{\text{task}})$. 

#### The In-context Learning Paradigm

- **Analogy:** Imagine you show a person a few examples of a new kind of puzzle, and then ask them to solve a similar one. They use the examples as hints to figure out the pattern.
- **How does it work?** Given a test example $x_{\text{test}}$, we construct a document $(x_1, \cdots, x_T)$, which is more commonly called a "prompt" in this context, by concatenating the labeled examples and the test example in some format. The model is not retrained; instead, it "reads" the prompt and tries to continue it in a way that matches the pattern.

**The In-context Learning Intuition:**
- **Few examples**: Just a handful of labeled examples (1-10)
- **Pattern recognition**: Model learns the pattern from examples
- **No training**: Model parameters remain unchanged
- **Temporary learning**: "Learning" happens only during inference

**The In-context Process:**
1. **Example collection**: Gather a few labeled examples
2. **Prompt construction**: Format examples into a coherent prompt
3. **Pattern learning**: Model learns pattern from examples
4. **Test application**: Apply learned pattern to new example

#### Prompt Construction

For example, we may construct the prompt as follows:

$$
\begin{align*}
x_1, \cdots, x_T \quad = \quad & \text{"Q: 2 ~ 3 = ?"} \quad x^{(1)}_{\text{task}} \\
& \text{"A: 5"} \quad y^{(1)}_{\text{task}} \\
& \text{"Q: 6 ~ 7 = ?"} \quad x^{(2)}_{\text{task}} \\
& \text{"A: 13"} \quad y^{(2)}_{\text{task}} \\
& \cdots \\
& \text{"Q: 15 ~ 2 = ?"} \quad x_{\text{test}}
\end{align*}
$$

Then, we let the pretrained model generate the most likely $x_{T+1}, x_{T+2}, \cdots$. In this case, if the model can "learn" that the symbol $\sim$ means addition from the few examples, we will obtain the following which suggests the answer is 17.

$$
x_{T+1}, x_{T+2}, \cdots = \text{"A: 17"}.
$$

**The Prompt Construction Strategy:**
- **Consistent format**: Use same format for all examples
- **Clear pattern**: Make the pattern obvious to the model
- **Relevant examples**: Choose examples similar to the test case
- **Appropriate number**: Usually 1-10 examples (more isn't always better)

**The Pattern Learning Process:**
1. **Input examples**: Model sees several input-output pairs
2. **Pattern extraction**: Model identifies the underlying pattern
3. **Test application**: Model applies pattern to new input
4. **Output generation**: Model generates appropriate output

#### Understanding In-context Learning

**Key Insights:**
1. **Pattern Recognition**: The model learns patterns from the examples in the prompt
2. **No Parameter Updates**: The model's parameters remain unchanged
3. **Temporary Learning**: The "learning" happens only during inference
4. **Few-shot Capability**: Works with just a few examples (typically 1-10)

**The In-context Learning Mechanism:**
- **Attention mechanism**: Model attends to relevant parts of the prompt
- **Pattern matching**: Model finds similarities between examples
- **Generalization**: Model applies learned pattern to new cases
- **Context utilization**: Model uses prompt context for predictions

**Why It Works:**
- **Pretraining Knowledge**: The model has seen similar patterns during pretraining
- **Pattern Completion**: The model tries to complete the pattern in the prompt
- **Context Utilization**: The examples provide context for the task
- **Attention Power**: Transformer attention can focus on relevant examples

**The Learning Without Learning Paradox:**
- **No training**: Model parameters don't change
- **Learning happens**: Model behavior adapts to the prompt
- **Temporary adaptation**: Adaptation only lasts for this prompt
- **Pattern completion**: Model completes the pattern it sees

#### Examples of In-context Learning

**Text Classification:**
```
Input: "I love this movie!"
Output: positive

Input: "This is terrible."
Output: negative

Input: "The food was okay."
Output: [model generates: negative]
```

**Translation:**
```
Input: "Hello" → "Hola"
Input: "Goodbye" → "Adiós"
Input: "Thank you" → [model generates: "Gracias"]
```

**Mathematical Reasoning:**
```
Input: "2 + 3 = 5"
Input: "7 + 4 = 11"
Input: "15 + 8 = [model generates: 23]"
```

**The Example Quality:**
- **Relevance**: Examples should be similar to the test case
- **Diversity**: Examples should cover different cases
- **Clarity**: Examples should be clear and unambiguous
- **Consistency**: Examples should follow the same format

#### Advantages and Limitations

**Advantages:**
- **No Training Required**: Works without additional training
- **Few Examples Needed**: Can work with just a few labeled examples
- **Rapid Prototyping**: Easy to test new tasks quickly
- **Interpretable**: The prompt shows exactly what the model is learning from

**The In-context Benefits:**
- **Immediate use**: No training time required
- **Low data requirement**: Works with minimal labeled data
- **Flexibility**: Can adapt to many different tasks
- **Transparency**: Prompt shows exactly what the model learns

**Limitations:**
- **Prompt Engineering**: Requires careful prompt design
- **Inconsistent Performance**: Results can vary significantly
- **Limited Context**: Limited by the model's context window
- **No Learning**: Cannot improve with more examples (unlike traditional learning)

**The In-context Challenges:**
- **Prompt sensitivity**: Small changes can affect performance
- **Context limits**: Limited by model's maximum sequence length
- **No improvement**: Performance doesn't improve with more examples
- **Unpredictable**: Results can be inconsistent

#### Best Practices for In-context Learning

1. **Clear Formatting**: Use consistent formatting for examples
2. **Relevant Examples**: Choose examples similar to the target task
3. **Appropriate Number**: Use 1-10 examples (more isn't always better)
4. **Clear Instructions**: Include explicit instructions when possible
5. **Consistent Style**: Maintain consistent style across examples

**The Prompt Engineering Guidelines:**
- **Format consistency**: Use same format for all examples
- **Example quality**: Choose high-quality, relevant examples
- **Instruction clarity**: Make instructions clear and specific
- **Style consistency**: Maintain consistent writing style
- **Length optimization**: Balance between clarity and context limits

- **Why does this work?** The model has learned to pick up on patterns in the prompt, even if it is not explicitly trained for the new task. This is a powerful way to use large language models for new problems with very little data.
- **Tip:** In-context learning is great for rapid prototyping and for tasks where you have only a handful of labeled examples.

---

### Comparison of Adaptation Methods

| Method | Data Required | Performance | Speed | Flexibility | Control |
|--------|---------------|-------------|-------|-------------|---------|
| **Finetuning** | Large labeled dataset | Best | Slow | High | High |
| **Zero-shot** | No data | Variable | Fast | Medium | Low |
| **In-context** | Few examples | Good | Fast | High | Medium |

**The Method Comparison:**
- **Data requirements**: How much labeled data is needed
- **Performance**: How well the method works on the task
- **Speed**: How quickly the method can be applied
- **Flexibility**: How easily it adapts to different tasks
- **Control**: How much control you have over the adaptation

**Summary:**
- **Finetuning:** Best when you have a moderate/large labeled dataset and want top performance. The model is retrained for your task.
- **Zero-shot:** No labeled data needed. The model uses its general knowledge to answer new questions.
- **In-context learning:** Give the model a few examples in the prompt. The model "figures out" the pattern and applies it to the new example, without retraining.

These methods make large language models extremely flexible and powerful for a wide range of tasks.

### Practical Considerations

**When to Use Each Method:**

1. **Use Finetuning When:**
   - You have >1000 labeled examples
   - You need the best possible performance
   - You have computational resources
   - The task is very different from pretraining

2. **Use Zero-shot When:**
   - You have no labeled data
   - You want to quickly test a model's capabilities
   - You're doing rapid prototyping
   - The task is similar to what the model saw during pretraining

3. **Use In-context Learning When:**
   - You have 1-10 labeled examples
   - You want to avoid training
   - You're doing rapid prototyping
   - The task has clear patterns that can be demonstrated

**The Decision Framework:**
- **Data availability**: More data → finetuning
- **Performance needs**: High performance → finetuning
- **Speed requirements**: Fast deployment → zero-shot or in-context
- **Resource constraints**: Limited compute → zero-shot or in-context

**Hybrid Approaches:**
- **Prompt Engineering + Zero-shot**: Carefully design prompts for better zero-shot performance
- **In-context + Finetuning**: Use in-context learning to bootstrap, then finetune
- **Ensemble Methods**: Combine predictions from multiple adaptation methods

**The Hybrid Strategy:**
- **Bootstrap**: Use in-context learning to get initial performance
- **Evaluate**: Assess if performance is sufficient
- **Decide**: Choose whether to finetune for better performance
- **Optimize**: Use best method for the specific use case

## From Theoretical Understanding to Practical Implementation

We've now explored **large language models** - specialized foundation models for text that leverage the sequential and contextual nature of language. We've seen how language modeling works through the chain rule of probability, how Transformer architectures process text, and how these models can generate coherent text and adapt to new tasks through finetuning, zero-shot learning, and in-context learning.

**The Journey So Far:**
- **Foundation models**: General-purpose models trained on massive data
- **Language modeling**: Predicting next words in sequences
- **Transformer architecture**: Parallel processing with attention
- **Text generation**: Creating coherent text with various strategies
- **Adaptation methods**: Using models for specific tasks

However, while understanding the theoretical foundations of self-supervised learning and large language models is essential, true mastery comes from **practical implementation**. The concepts we've learned - contrastive learning, language modeling, text generation, and adaptation methods - need to be applied to real problems to develop intuition and practical skills.

**The Theory-to-Practice Bridge:**
- **Understanding**: Theoretical knowledge provides foundation
- **Implementation**: Practical coding builds intuition
- **Experimentation**: Hands-on work reveals nuances
- **Mastery**: Combining theory and practice leads to expertise

This motivates our exploration of **hands-on coding** - the practical implementation of all the self-supervised learning and language model concepts we've learned. We'll put our theoretical knowledge into practice by implementing contrastive learning for computer vision, building language models for text generation, and developing the practical skills needed to create foundation models that can adapt to various downstream tasks.

**The Practical Learning Goals:**
- **Implementation skills**: Ability to code the concepts we've learned
- **Intuition building**: Deep understanding through hands-on experience
- **Problem solving**: Ability to apply concepts to real problems
- **Tool mastery**: Proficiency with modern AI frameworks and tools

The transition from theoretical understanding to practical implementation represents the bridge from knowledge to application - taking our understanding of how self-supervised learning and language models work and turning it into practical tools for building powerful AI systems.

In the next section, we'll implement complete systems for self-supervised learning and language models, experiment with different techniques, and develop the practical skills needed for real-world applications in computer vision and natural language processing.

---

**Previous: [Self-Supervised Learning](01_pretraining.md)** - Understand the fundamental techniques for learning from unlabeled data.

**Next: [Hands-on Coding](03_hands-on_coding.md)** - Implement self-supervised learning and language model techniques with practical examples.

[^5]: In the practical implementations, typically all the data are concatenated into a single sequence in some order, and each example typically corresponds a sub-sequence of consecutive words which may correspond to a subset of a document or may span across multiple documents.

[^6]: Technically, words may be decomposed into tokens which could be words or sub-words (combinations of letters), but this note omits this technicality. In fact most common words are a single token themselves.



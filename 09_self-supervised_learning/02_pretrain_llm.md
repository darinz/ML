# 14.3 Pretrained large language models

Natural language processing is another area where pretraining models are particularly successful. In language problems, an example typically corresponds to a document or generally a sequence (or trunk) of words,[^5] denoted by $x = (x_1, \cdots, x_T)$ where $T$ is the length of the document/sequence, $x_i \in \{1, \cdots, V\}$ are words in the document, and $V$ is the vocabulary size.[^6]

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
$`
Vocabulary: ["the", "cat", "sat", "on", "mat"]
Word IDs: [1, 2, 3, 4, 5]
Embeddings: 
  e_1 (the) = [0.1, 0.2, 0.3, ...]
  e_2 (cat) = [0.4, 0.1, 0.8, ...]
  e_3 (sat) = [0.2, 0.9, 0.1, ...]
  ...
$`

## The Transformer Architecture

The most commonly used model is the Transformer [Vaswani et al., 2017]. In this subsection, we will introduce the input-output interface of a Transformer, but treat the intermediate computation in the Transformer as a blackbox. We refer the students to the transformer paper or more advanced courses for more details.

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

<img src="./img/transformer_output.png" width="400px"/>

**Figure 14.1 (description):** The inputs to the Transformer are the embeddings $e_{x_0}, e_{x_1}, \ldots, e_{x_T}$ corresponding to the tokens $x_0, x_1, \ldots, x_T$. The Transformer $f_\theta(x)$ outputs a sequence of vectors $u_1, u_2, \ldots, u_{T+1}$, each of which is used to predict the next token in the sequence.

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

**Properties:**
- All outputs are positive (probabilities)
- Sum to 1 (valid probability distribution)
- Preserves relative ordering of logits
- Amplifies differences between large and small values

**Example:**
If $u_t = [2.0, 1.0, 0.5]$ for a 3-word vocabulary:
- $\exp(u_t) = [7.39, 2.72, 1.65]$
- $\sum \exp(u_t) = 11.76$
- $\text{softmax}(u_t) = [0.63, 0.23, 0.14]$

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

#### Understanding the Loss Function

**Cross-Entropy Loss:**
The cross-entropy loss measures how well our predicted probability distribution matches the true distribution (which is a one-hot vector for the correct word).

$$
\ell_{ce}(u, x_t) = -\log(\text{softmax}(u)_{x_t})
$$

**Why This Works:**
- If the model assigns high probability to the correct word, $-\log(p)$ is small
- If the model assigns low probability to the correct word, $-\log(p)$ is large
- The loss encourages the model to assign high probability to correct words

**Example:**
For the sequence "The cat sat" and $u_2$ (predicting the word after "The"):
- If the model predicts $p(\text{cat}) = 0.8$, loss = $-\log(0.8) = 0.22$
- If the model predicts $p(\text{cat}) = 0.1$, loss = $-\log(0.1) = 2.30$

#### Training Process

1. **Forward Pass**: For each position $t$, compute $u_t = f_\theta(x_0, \ldots, x_{t-1})$
2. **Loss Computation**: Compute cross-entropy loss for each position
3. **Backward Pass**: Compute gradients with respect to $\theta$
4. **Parameter Update**: Update $\theta$ using gradient descent

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

**Low Temperature ($\tau < 1$):**
- Makes the distribution more "peaked" (concentrated)
- Model becomes more confident in its predictions
- Generated text is more deterministic and focused
- Risk of repetitive or boring text

**High Temperature ($\tau > 1$):**
- Makes the distribution more "flat" (spread out)
- Model becomes less confident in its predictions
- Generated text is more diverse and creative
- Risk of incoherent or nonsensical text

**Example:**
Consider logits $u = [2.0, 1.0, 0.5]$:
- $\tau = 0.5$: $\text{softmax}(u/0.5) = [0.88, 0.11, 0.01]$ (very confident)
- $\tau = 1.0$: $\text{softmax}(u/1.0) = [0.63, 0.23, 0.14]$ (balanced)
- $\tau = 2.0$: $\text{softmax}(u/2.0) = [0.42, 0.31, 0.27]$ (uncertain)

When $\tau = 1$, the text is sampled from the original conditional probability defined by the model. With a decreasing $\tau$, the generated text gradually becomes more "deterministic". $\tau \to 0$ reduces to greedy decoding, where we generate the most probable next token from the conditional probability.

### Generation Strategies

**Greedy Decoding ($\tau \to 0$):**
- Always choose the most probable next word
- Fast and deterministic
- Often produces repetitive or boring text

**Random Sampling ($\tau = 1$):**
- Sample according to the model's probability distribution
- More diverse output
- Can be less coherent

**Top-k Sampling:**
- Only consider the top $k$ most probable words
- Sample from this restricted set
- Balances diversity and coherence

**Nucleus Sampling (Top-p):**
- Consider words until cumulative probability reaches $p$
- Sample from this dynamic set
- Often produces better results than fixed top-k

---

## 14.3.1 Zero-shot learning and in-context learning

For language models, there are many ways to adapt a pretrained model to downstream tasks. In this notes, we discuss three of them: finetuning, zero-shot learning, and in-context learning.

### Finetuning

**Finetuning** is not very common for the autoregressive language models that we introduced in Section 14.3 but much more common for other variants such as masked language models which has similar input-output interfaces but are pretrained differently [Devlin et al., 2019]. 

Finetuning means taking a model that has already learned a lot from a huge dataset (pretraining), and then continuing to train it on a smaller, task-specific dataset. Think of it like a student who has read many books (pretraining) and then studies specifically for a test (finetuning). The model adapts its knowledge to do well on the new task.

#### Mathematical Formulation

The finetuning method is the same as introduced generally in Section 14.1—the only question is how we define the prediction task with an additional linear head. One option is to treat $c_{T+1} = \phi_\theta(x_1, \cdots, x_T)$ as the representation and use $w^\top c_{T+1} = w^\top \phi_\theta(x_1, \cdots, x_T)$ to predict the task label. As described in Section 14.1, we initialize $\theta$ to the pretrained model $\hat{\theta}$ and then optimize both $w$ and $\theta$.

#### When to Use Finetuning

**Advantages:**
- **Best Performance**: Can achieve highest accuracy on the target task
- **Task-Specific Adaptation**: Model can learn task-specific patterns
- **Flexibility**: Can adapt to any task format

**Disadvantages:**
- **Requires Labeled Data**: Needs significant amount of task-specific data
- **Computationally Expensive**: Requires training the entire model
- **Risk of Catastrophic Forgetting**: May lose general knowledge
- **Task-Specific**: Need separate model for each task

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

#### Task Formatting

The key insight is that we can format many tasks as natural language questions or prompts. The model then generates text that answers the question.

For example, we can format an example as a question:

$x_{\text{task}} = (x_{\text{task},1}, \cdots, x_{\text{task},R}) = \text{"Is the speed of light a universal constant?"}$

Then, we compute the most likely next word predicted by the language model given this question, that is, computing $\text{argmax}_{x_{T+1}} p(x_{T+1} \mid x_{\text{task},1}, \cdots, x_{\text{task},R})$. In this case, if the most likely next word $x_{T+1}$ is "No", then we solve the task. (The speed of light is only a constant in vacuum.)

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

#### Advantages and Limitations

**Advantages:**
- **No Training Required**: Works immediately without any additional training
- **No Labeled Data**: Can work on tasks with no labeled examples
- **Rapid Deployment**: Can be applied to new tasks instantly
- **Cost Effective**: No computational cost for adaptation

**Limitations:**
- **Inconsistent Performance**: Results can be unreliable
- **Limited Control**: Cannot fine-tune behavior for specific tasks
- **Prompt Sensitivity**: Performance heavily depends on prompt formulation
- **Knowledge Cutoff**: Limited to knowledge from pretraining

- **Why does this work?** During pretraining, the model has seen so many examples of language that it can often generalize to new questions or tasks, even if it has never seen them before.
- **Tip:** Zero-shot is useful when you have no labeled data for your task, or want to quickly test what a model can do "out of the box."

---

### In-context Learning

**In-context learning** is mostly used for few-shot settings where we have a few labeled examples $(x^{(1)}_{\text{task}}, y^{(1)}_{\text{task}}), \cdots, (x^{(n_{\text{task}})}_{\text{task}}, y^{(n_{\text{task}})}_{\text{task}})$. 

#### The In-context Learning Paradigm

- **Analogy:** Imagine you show a person a few examples of a new kind of puzzle, and then ask them to solve a similar one. They use the examples as hints to figure out the pattern.
- **How does it work?** Given a test example $x_{\text{test}}$, we construct a document $(x_1, \cdots, x_T)$, which is more commonly called a "prompt" in this context, by concatenating the labeled examples and the test example in some format. The model is not retrained; instead, it "reads" the prompt and tries to continue it in a way that matches the pattern.

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

#### Understanding In-context Learning

**Key Insights:**
1. **Pattern Recognition**: The model learns patterns from the examples in the prompt
2. **No Parameter Updates**: The model's parameters remain unchanged
3. **Temporary Learning**: The "learning" happens only during inference
4. **Few-shot Capability**: Works with just a few examples (typically 1-10)

**Why It Works:**
- **Pretraining Knowledge**: The model has seen similar patterns during pretraining
- **Pattern Completion**: The model tries to complete the pattern in the prompt
- **Context Utilization**: The examples provide context for the task

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

#### Advantages and Limitations

**Advantages:**
- **No Training Required**: Works without additional training
- **Few Examples Needed**: Can work with just a few labeled examples
- **Rapid Prototyping**: Easy to test new tasks quickly
- **Interpretable**: The prompt shows exactly what the model is learning from

**Limitations:**
- **Prompt Engineering**: Requires careful prompt design
- **Inconsistent Performance**: Results can vary significantly
- **Limited Context**: Limited by the model's context window
- **No Learning**: Cannot improve with more examples (unlike traditional learning)

#### Best Practices for In-context Learning

1. **Clear Formatting**: Use consistent formatting for examples
2. **Relevant Examples**: Choose examples similar to the target task
3. **Appropriate Number**: Use 1-10 examples (more isn't always better)
4. **Clear Instructions**: Include explicit instructions when possible
5. **Consistent Style**: Maintain consistent style across examples

- **Why does this work?** The model has learned to pick up on patterns in the prompt, even if it is not explicitly trained for the new task. This is a powerful way to use large language models for new problems with very little data.
- **Tip:** In-context learning is great for rapid prototyping and for tasks where you have only a handful of labeled examples.

---

### Comparison of Adaptation Methods

| Method | Data Required | Performance | Speed | Flexibility | Control |
|--------|---------------|-------------|-------|-------------|---------|
| **Finetuning** | Large labeled dataset | Best | Slow | High | High |
| **Zero-shot** | No data | Variable | Fast | Medium | Low |
| **In-context** | Few examples | Good | Fast | High | Medium |

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

**Hybrid Approaches:**
- **Prompt Engineering + Zero-shot**: Carefully design prompts for better zero-shot performance
- **In-context + Finetuning**: Use in-context learning to bootstrap, then finetune
- **Ensemble Methods**: Combine predictions from multiple adaptation methods

[^5]: In the practical implementations, typically all the data are concatenated into a single sequence in some order, and each example typically corresponds a sub-sequence of consecutive words which may correspond to a subset of a document or may span across multiple documents.
[^6]: Technically, words may be decomposed into tokens which could be words or sub-words (combinations of letters), but this note omits this technicality. In fact most common words are a single token themselves.



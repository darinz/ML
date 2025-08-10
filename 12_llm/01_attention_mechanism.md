# Attention Mechanisms

## Overview

Attention mechanisms are the foundational innovation that powers modern transformer architectures and large language models. This guide provides a deep dive into the theory, mathematics, and implementation of attention mechanisms, from basic concepts to advanced variants.

## Table of Contents

- [Introduction to Attention](#introduction-to-attention)
- [Mathematical Foundations](#mathematical-foundations)
- [Types of Attention](#types-of-attention)
- [Multi-Head Attention](#multi-head-attention)
- [Positional Encoding](#positional-encoding)
- [Implementation Details](#implementation-details)
- [Advanced Attention Variants](#advanced-attention-variants)
- [Practical Examples](#practical-examples)
- [Performance Considerations](#performance-considerations)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

## Introduction to Attention

### What is Attention?

Attention is a mechanism that allows neural networks to focus on specific parts of the input when processing each element. Instead of treating all input elements equally, attention computes a weighted combination of input elements, where the weights are learned dynamically based on the content.

### Why Attention?

**Key Benefits:**
- **Parallelization**: Unlike RNNs, attention can process sequences in parallel
- **Long-range Dependencies**: Can capture relationships between distant elements
- **Interpretability**: Attention weights provide insights into model decisions
- **Flexibility**: Can be applied to various sequence lengths and modalities

### The Intuition

Think of attention like a spotlight that can focus on different parts of a stage (input sequence) depending on what's currently being performed (current element being processed). The spotlight's intensity (attention weight) determines how much focus each part receives.

## Mathematical Foundations

### Query-Key-Value Framework

The attention mechanism operates on three fundamental components:

- **Query (Q)**: What we're looking for
- **Key (K)**: What each input element offers
- **Value (V)**: The actual content of each input element

### Scaled Dot-Product Attention

The core attention formula is:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

**Step-by-step breakdown:**

1. **Compute Attention Scores**: $`QK^T`$
   - Measures similarity between queries and keys
   - Higher scores indicate stronger relationships

2. **Scale the Scores**: $`\frac{QK^T}{\sqrt{d_k}}`$
   - Prevents softmax from entering regions with small gradients
   - $`d_k`$ is the dimension of keys

3. **Apply Softmax**: $`\text{softmax}(\cdot)`$
   - Converts scores to probabilities (attention weights)
   - Ensures weights sum to 1

4. **Weighted Sum**: $`(\text{weights}) \times V`$
   - Combines values according to attention weights

### Detailed Mathematical Formulation

For a sequence of length $`n`$ and embedding dimension $`d`$:

```math
\begin{align}
Q &= XW_Q \in \mathbb{R}^{n \times d_k} \\
K &= XW_K \in \mathbb{R}^{n \times d_k} \\
V &= XW_V \in \mathbb{R}^{n \times d_v} \\
\text{Attention}(Q, K, V) &= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\end{align}
```

Where:
- $`X \in \mathbb{R}^{n \times d}`$ is the input sequence
- $`W_Q, W_K, W_V`$ are learned weight matrices
- $`d_k`$ and $`d_v`$ are the dimensions of keys and values respectively

## Types of Attention

### 1. Self-Attention

Self-attention computes attention within the same sequence, allowing each position to attend to all positions in the sequence.

**Use Cases:**
- Understanding relationships within a single sequence
- Capturing long-range dependencies
- Learning contextual representations

**Example:**
```python
# Self-attention in a sentence
sentence = "The cat sat on the mat"
# Each word attends to all other words to understand context
```

### 2. Cross-Attention

Cross-attention computes attention between two different sequences, typically used in encoder-decoder architectures.

**Use Cases:**
- Machine translation (encoder output → decoder)
- Question answering (question → context)
- Summarization (source → target)

**Example:**
```python
# Cross-attention in translation
source = "Hello world"
target = "Hola mundo"
# Target words attend to source words for translation
```

### 3. Masked Attention

Masked attention prevents the model from looking at future tokens, essential for autoregressive generation.

**Use Cases:**
- Language modeling
- Text generation
- Causal modeling

**Implementation:**
```python
# Create causal mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
attention_scores = attention_scores.masked_fill(mask == 1, float('-inf'))
```

## Multi-Head Attention

### Motivation

Single attention can only focus on one type of relationship at a time. Multi-head attention allows the model to attend to different types of relationships simultaneously.

### Mathematical Formulation

```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_h)W^O
```

Where each head is:
```math
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

### Implementation Structure

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention for each head
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.w_o(attention_output)
        return output
```

## Positional Encoding

### The Problem

Since attention processes all positions in parallel, it has no inherent notion of sequence order. Positional encoding provides this crucial information.

### Sinusoidal Positional Encoding

The original transformer uses sinusoidal positional encoding:

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
```

**Properties:**
- **Uniqueness**: Each position has a unique encoding
- **Extrapolation**: Can handle sequences longer than training
- **Relative Positions**: Encodes relative position information

### Implementation

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### Alternative Positional Encodings

**1. Learned Positional Embeddings**
```python
self.pos_embedding = nn.Embedding(max_len, d_model)
```

**2. Relative Positional Encoding**
- Encodes relative distances between positions
- More efficient for long sequences

**3. RoPE (Rotary Position Embedding)**
- Applies rotation matrices to embeddings
- Better extrapolation to longer sequences

## Implementation Details

### Complete Attention Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask=None):
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # Compute weighted sum
        output = torch.matmul(attn, v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        
        # Pass through the pre-attention projection
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # Transpose for attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting
        
        q, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        
        q = self.layer_norm(q)
        
        return q, attn
```

### Attention Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, save_path=None):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Tensor of shape (seq_len, seq_len)
        tokens: List of token strings
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights.detach().numpy(), 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                annot=True,
                fmt='.2f')
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# Example usage
tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
attention_weights = torch.randn(6, 6)  # Example weights
visualize_attention(attention_weights, tokens)
```

## Advanced Attention Variants

### 1. Local Attention

Limits attention to a local window around each position, reducing computational complexity.

```python
def local_attention(q, k, v, window_size=64):
    seq_len = q.size(1)
    local_weights = torch.zeros_like(q)
    
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        
        # Compute attention only within window
        local_q = q[:, i:i+1, :]
        local_k = k[:, start:end, :]
        local_v = v[:, start:end, :]
        
        attn_weights = torch.matmul(local_q, local_k.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        local_weights[:, i, start:end] = attn_weights.squeeze(1)
    
    return torch.matmul(local_weights, v)
```

### 2. Sparse Attention

Only computes attention for a subset of position pairs, further reducing complexity.

```python
def sparse_attention(q, k, v, sparsity_pattern):
    """
    sparsity_pattern: Boolean tensor indicating which attention connections to compute
    """
    attn_weights = torch.matmul(q, k.transpose(-2, -1))
    attn_weights = attn_weights.masked_fill(~sparsity_pattern, float('-inf'))
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)
```

### 3. Linear Attention

Reformulates attention to avoid quadratic complexity:

```python
def linear_attention(q, k, v):
    """
    Linear attention using kernel feature maps
    """
    # Apply kernel feature map
    q = F.elu(q) + 1
    k = F.elu(k) + 1
    
    # Compute KV and Q(KV)
    kv = torch.matmul(k.transpose(-2, -1), v)
    qkv = torch.matmul(q, kv)
    
    # Normalize
    k_sum = torch.sum(k, dim=-2, keepdim=True)
    qk_sum = torch.matmul(q, k_sum.transpose(-2, -1))
    
    return qkv / (qk_sum + 1e-8)
```

## Practical Examples

### Example 1: Simple Self-Attention

```python
import torch
import torch.nn as nn

# Simple self-attention implementation
def simple_self_attention(x, d_k=64):
    """
    x: Input tensor of shape (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = x.shape
    
    # Create Q, K, V (simplified - in practice, these would be learned)
    Q = nn.Linear(d_model, d_k)(x)
    K = nn.Linear(d_model, d_k)(x)
    V = nn.Linear(d_model, d_model)(x)
    
    # Compute attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# Test the implementation
x = torch.randn(2, 10, 512)  # batch_size=2, seq_len=10, d_model=512
output, weights = simple_self_attention(x)
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
```

### Example 2: Attention for Text Classification

```python
class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.attention = MultiHeadAttention(num_heads, d_model, d_model//num_heads, d_model//num_heads)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Apply self-attention
        x, _ = self.attention(x, x, x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        return x

# Usage example
model = AttentionClassifier(vocab_size=10000, d_model=256, num_classes=5)
input_tensor = torch.randint(0, 10000, (32, 50))  # batch_size=32, seq_len=50
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # (32, 5)
```

### Example 3: Attention for Sequence-to-Sequence

```python
class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, src, tgt):
        # Encode source sequence
        encoder_output, (hidden, cell) = self.encoder(src)
        
        # Decode with attention
        decoder_output, _ = self.decoder(tgt, (hidden, cell))
        
        # Apply cross-attention between decoder and encoder
        attended_output, attention_weights = self.attention(
            decoder_output, encoder_output, encoder_output
        )
        
        # Generate output
        output = self.output_layer(attended_output)
        return output, attention_weights
```

## Performance Considerations

### Computational Complexity

**Standard Attention**: $`O(n^2)`$ where $`n`$ is sequence length
- **Memory**: $`O(n^2)`$ for attention weights
- **Time**: $`O(n^2 \times d)`$ for matrix multiplications

**Optimization Strategies:**
1. **Linear Attention**: $`O(n)`$ complexity
2. **Sparse Attention**: $`O(n \times k)`$ where $`k`$ is sparsity factor
3. **Local Attention**: $`O(n \times w)`$ where $`w`$ is window size

### Memory Optimization

```python
# Gradient checkpointing for memory efficiency
from torch.utils.checkpoint import checkpoint

def memory_efficient_attention(q, k, v):
    return checkpoint(scaled_dot_product_attention, q, k, v)

# Flash Attention (if available)
try:
    from flash_attn import flash_attn_func
    def flash_attention(q, k, v):
        return flash_attn_func(q, k, v)
except ImportError:
    print("Flash Attention not available, using standard attention")
```

### Scaling to Long Sequences

```python
class LongSequenceAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size=1024):
        super().__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        
    def forward(self, x):
        seq_len = x.size(1)
        
        if seq_len <= self.window_size:
            # Standard attention for short sequences
            return self.attention(x, x, x)
        else:
            # Sliding window attention for long sequences
            outputs = []
            for i in range(0, seq_len, self.window_size // 2):
                end = min(i + self.window_size, seq_len)
                window_x = x[:, i:end, :]
                window_output, _ = self.attention(window_x, window_x, window_x)
                outputs.append(window_output)
            
            # Combine outputs (simplified - in practice, need proper overlap handling)
            return torch.cat(outputs, dim=1)
```

## Common Pitfalls and Solutions

### 1. Gradient Vanishing/Exploding

**Problem**: Attention weights can cause gradient issues in deep networks.

**Solution**: Proper scaling and normalization
```python
# Use scaled dot-product attention
attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

# Layer normalization
x = nn.LayerNorm(d_model)(x + attention_output)
```

### 2. Attention Collapse

**Problem**: All attention weights become uniform, losing the ability to focus.

**Solution**: Regularization and proper initialization
```python
# Add dropout to attention weights
attention_weights = F.dropout(attention_weights, p=0.1, training=self.training)

# Proper initialization
nn.init.xavier_uniform_(self.w_q.weight)
nn.init.xavier_uniform_(self.w_k.weight)
nn.init.xavier_uniform_(self.w_v.weight)
```

### 3. Memory Issues with Long Sequences

**Problem**: Quadratic memory usage with sequence length.

**Solution**: Use memory-efficient attention variants
```python
# Chunked attention for very long sequences
def chunked_attention(q, k, v, chunk_size=1024):
    outputs = []
    for i in range(0, q.size(1), chunk_size):
        chunk_q = q[:, i:i+chunk_size, :]
        chunk_output, _ = self.attention(chunk_q, k, v)
        outputs.append(chunk_output)
    return torch.cat(outputs, dim=1)
```

### 4. Training Instability

**Problem**: Attention training can be unstable, especially with large models.

**Solution**: Proper learning rate scheduling and warmup
```python
# Learning rate warmup
def get_learning_rate(step, d_model, warmup_steps=4000):
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Conclusion

Attention mechanisms have revolutionized deep learning by providing a powerful way to model relationships in sequential data. Understanding the mathematical foundations, implementation details, and practical considerations is crucial for building effective transformer-based models.

The key takeaways are:
1. **Attention enables parallel processing** of sequences while capturing long-range dependencies
2. **Multi-head attention** allows focusing on different types of relationships simultaneously
3. **Positional encoding** is essential for maintaining sequence order information
4. **Performance optimization** is crucial for scaling to long sequences
5. **Proper implementation** requires attention to numerical stability and memory efficiency

As you build attention-based models, remember to start simple and gradually add complexity, always monitoring for the common pitfalls discussed in this guide.

---

**Next Steps:**
- Explore transformer architecture implementations
- Study advanced attention variants like Flash Attention
- Practice with real-world applications
- Experiment with different positional encoding schemes

## From Attention Mechanisms to Complete Architectures

We've now explored **attention mechanisms** - the foundational innovation that powers modern transformer architectures and large language models. We've seen how the query-key-value framework enables parallel processing of sequences, how multi-head attention captures different types of relationships simultaneously, and how positional encoding maintains sequence order information.

However, while attention mechanisms provide the core computational unit, **real-world language models** require complete architectures that combine attention with other essential components. Consider a machine translation system - it needs not just attention, but also encoders to understand the source language, decoders to generate the target language, and mechanisms to coordinate between them.

This motivates our exploration of **transformer architecture** - the complete framework that combines attention mechanisms with encoder-decoder structures, feed-forward networks, layer normalization, and residual connections. We'll see how the original transformer architecture was designed for sequence-to-sequence tasks, how modern variants like BERT and GPT serve different purposes, and how these architectures enable the powerful language models that have revolutionized AI.

The transition from attention mechanisms to transformer architecture represents the bridge from core components to complete systems - taking our understanding of how attention works and building it into full architectures that can understand, generate, and manipulate language.

In the next section, we'll explore transformer architecture, understanding how attention mechanisms are integrated into complete models for various language tasks.

---

**Next: [Transformer Architecture](02_transformer_architecture.md)** - Learn how to build complete transformer models for language understanding and generation. 
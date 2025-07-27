# Transformer Architecture

## Overview

The Transformer architecture, introduced in "Attention Is All You Need," revolutionized natural language processing by replacing recurrent and convolutional components with attention mechanisms. This guide provides a deep dive into the transformer's architectural components, variants, and implementation details.

## Table of Contents

- [Introduction to Transformers](#introduction-to-transformers)
- [Encoder-Decoder Architecture](#encoder-decoder-architecture)
- [Core Components](#core-components)
- [Architectural Variants](#architectural-variants)
- [Implementation Details](#implementation-details)
- [Training Considerations](#training-considerations)
- [Practical Examples](#practical-examples)
- [Performance Optimization](#performance-optimization)
- [Common Architectures](#common-architectures)

## Introduction to Transformers

### What is a Transformer?

A transformer is a neural network architecture that uses attention mechanisms to process sequential data. Unlike RNNs and CNNs, transformers can process entire sequences in parallel, making them highly efficient for modern hardware.

### Key Innovations

**Revolutionary Features:**
- **Parallel Processing**: All positions processed simultaneously
- **Self-Attention**: Captures relationships between any positions
- **No Recurrence**: Eliminates sequential dependencies
- **Scalability**: Can handle long sequences effectively

### Why Transformers Work

**Advantages over Previous Architectures:**
- **Long-range Dependencies**: Can model relationships across entire sequences
- **Parallelization**: Training and inference can be parallelized
- **Interpretability**: Attention weights provide insights into model decisions
- **Flexibility**: Can be adapted for various sequence lengths and modalities

## Encoder-Decoder Architecture

### Original Transformer Structure

The original transformer uses an encoder-decoder architecture designed for sequence-to-sequence tasks like machine translation.

```math
\text{Transformer}(X, Y) = \text{Decoder}(\text{Encoder}(X), Y)
```

### Encoder Stack

The encoder processes the input sequence and creates a rich representation that captures relationships between all input elements.

**Encoder Architecture:**
```math
\text{Encoder}(X) = \text{LayerNorm}_N \circ \text{FFN}_N \circ \text{Attention}_N \circ \ldots \circ \text{LayerNorm}_1 \circ \text{FFN}_1 \circ \text{Attention}_1(X)
```

**Key Components:**
1. **Multi-Head Self-Attention**: Captures relationships within input
2. **Feed-Forward Network**: Position-wise transformations
3. **Layer Normalization**: Stabilizes training
4. **Residual Connections**: Helps with gradient flow

### Decoder Stack

The decoder generates the output sequence autoregressively, using both self-attention and cross-attention to encoder outputs.

**Decoder Architecture:**
```math
\text{Decoder}(X, Y) = \text{LayerNorm}_N \circ \text{FFN}_N \circ \text{CrossAttention}_N \circ \text{MaskedAttention}_N \circ \ldots \circ \text{LayerNorm}_1 \circ \text{FFN}_1 \circ \text{CrossAttention}_1 \circ \text{MaskedAttention}_1(Y, X)
```

**Key Components:**
1. **Masked Self-Attention**: Prevents looking at future tokens
2. **Cross-Attention**: Attends to encoder outputs
3. **Feed-Forward Network**: Position-wise transformations
4. **Layer Normalization**: Stabilizes training

## Core Components

### 1. Multi-Head Self-Attention

Self-attention allows each position to attend to all positions in the sequence, capturing complex relationships.

**Mathematical Formulation:**
```math
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
```

Where each head computes:
```math
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

**Implementation:**
```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Linear transformations and reshape
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply final linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        return output, attention_weights
```

### 2. Feed-Forward Network

The feed-forward network applies position-wise transformations to each position independently.

**Mathematical Formulation:**
```math
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
```

**Implementation:**
```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

### 3. Layer Normalization

Layer normalization stabilizes training by normalizing activations within each layer.

**Mathematical Formulation:**
```math
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
```

Where:
- $`\mu`$ and $`\sigma^2`$ are computed over the last dimension
- $`\gamma`$ and $`\beta`$ are learnable parameters
- $`\epsilon`$ is a small constant for numerical stability

### 4. Residual Connections

Residual connections help with gradient flow and training stability.

**Implementation:**
```python
class ResidualConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

## Architectural Variants

### 1. Encoder-Only Models (BERT-style)

Encoder-only models are designed for understanding tasks where the model needs to process the entire input sequence.

**Characteristics:**
- **Bidirectional**: Can attend to all positions in both directions
- **Understanding Tasks**: Classification, NER, QA, etc.
- **No Generation**: Cannot generate text autoregressively

**Popular Models:**
- **BERT**: Bidirectional encoder for language understanding
- **RoBERTa**: Robustly optimized BERT
- **DeBERTa**: Decoding-enhanced BERT with disentangled attention

**Implementation:**
```python
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 2. Decoder-Only Models (GPT-style)

Decoder-only models are designed for generation tasks where the model predicts the next token autoregressively.

**Characteristics:**
- **Unidirectional**: Can only attend to previous positions
- **Generation Tasks**: Language modeling, text generation
- **Causal Masking**: Prevents looking at future tokens

**Popular Models:**
- **GPT**: Generative pre-trained transformer
- **GPT-2**: Larger GPT with improved training
- **GPT-3**: Massive scale language model
- **GPT-4**: Advanced multimodal model

**Implementation:**
```python
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.norm(x)
        return self.output_projection(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Masked self-attention with residual connection
        attn_output, _ = self.masked_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x
```

### 3. Encoder-Decoder Models (T5-style)

Encoder-decoder models are designed for sequence-to-sequence tasks where the input and output are different sequences.

**Characteristics:**
- **Bidirectional Encoder**: Processes input in both directions
- **Unidirectional Decoder**: Generates output autoregressively
- **Cross-Attention**: Decoder attends to encoder outputs

**Popular Models:**
- **T5**: Text-to-text transfer transformer
- **BART**: Bidirectional and autoregressive transformer
- **mT5**: Multilingual T5

**Implementation:**
```python
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len=512, dropout=0.1):
        super().__init__()
        self.encoder = EncoderOnlyTransformer(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = DecoderOnlyTransformer(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, tgt_mask)
        return decoder_output

class DecoderLayerWithCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Masked self-attention
        attn_output, _ = self.masked_attention(x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention to encoder
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

## Implementation Details

### Complete Transformer Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, 
                 num_heads=8, d_ff=2048, max_len=512, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and Decoder
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def generate_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0)
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)
        
        # Encode
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        for encoder_layer in self.encoder:
            src = encoder_layer(src, src_mask)
        
        # Decode
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, src, tgt_mask, src_mask)
        
        # Output projection
        output = self.output_projection(tgt)
        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask, src_mask):
        # Masked self-attention
        attn_output, _ = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply final linear transformation
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```

## Training Considerations

### Loss Functions

**Cross-Entropy Loss for Language Modeling:**
```python
def compute_loss(logits, targets, ignore_index=0):
    """
    Compute cross-entropy loss for language modeling.
    
    Args:
        logits: Model output of shape (batch_size, seq_len, vocab_size)
        targets: Target tokens of shape (batch_size, seq_len)
        ignore_index: Token index to ignore in loss computation
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
    
    # Reshape for loss computation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    
    return loss_fn(logits_flat, targets_flat)
```

### Learning Rate Scheduling

**Transformer Learning Rate Schedule:**
```python
class TransformerScheduler:
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

# Usage
scheduler = TransformerScheduler(d_model=512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler_fn = scheduler
```

### Regularization Techniques

**Label Smoothing:**
```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.smoothing / (self.vocab_size - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Mask ignored indices
        mask = (targets != self.ignore_index).unsqueeze(1)
        smooth_targets = smooth_targets * mask.float()
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1)
        
        # Average over non-ignored tokens
        return loss.sum() / mask.sum()
```

## Practical Examples

### Example 1: Simple Language Model

```python
class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=8, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_model * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # Create causal mask
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        
        # Embed and encode
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask=mask)
        
        x = self.norm(x)
        return self.output_projection(x)

# Training example
model = SimpleLanguageModel(vocab_size=10000)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Shift sequences for language modeling
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        
        logits = model(input_seq)
        loss = criterion(logits.view(-1, vocab_size), target_seq.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

### Example 2: Text Classification with BERT-style Model

```python
class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=256, num_layers=6, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_model * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # Global average pooling
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        return self.classifier(x)

# Usage
model = BERTClassifier(vocab_size=10000, num_classes=5)
input_tensor = torch.randint(0, 10000, (32, 128))  # batch_size=32, seq_len=128
mask = (input_tensor != 0)  # Create mask for padding
output = model(input_tensor, mask)
print(f"Output shape: {output.shape}")  # (32, 5)
```

### Example 3: Sequence-to-Sequence Translation

```python
class TranslationModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8):
        super().__init__()
        self.transformer = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
    def forward(self, src, tgt):
        return self.transformer(src, tgt)
    
    def generate(self, src, max_len=50, start_token=1, end_token=2):
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # Encode source
            src_mask = self.transformer.generate_src_mask(src)
            src = self.transformer.src_embedding(src) * math.sqrt(self.transformer.d_model)
            src = self.transformer.pos_encoding(src)
            src = self.transformer.dropout(src)
            
            for encoder_layer in self.transformer.encoder:
                src = encoder_layer(src, src_mask)
            
            # Initialize target sequence
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
            
            for _ in range(max_len - 1):
                tgt_mask = self.transformer.generate_tgt_mask(tgt)
                
                # Decode
                tgt_embed = self.transformer.tgt_embedding(tgt) * math.sqrt(self.transformer.d_model)
                tgt_embed = self.transformer.pos_encoding(tgt_embed)
                tgt_embed = self.transformer.dropout(tgt_embed)
                
                for decoder_layer in self.transformer.decoder:
                    tgt_embed = decoder_layer(tgt_embed, src, tgt_mask, src_mask)
                
                tgt_embed = self.transformer.norm(tgt_embed)
                logits = self.transformer.output_projection(tgt_embed)
                
                # Get next token
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Check if all sequences have ended
                if (tgt == end_token).any(dim=1).all():
                    break
            
            return tgt

# Usage
model = TranslationModel(src_vocab_size=5000, tgt_vocab_size=5000)
src_seq = torch.randint(0, 5000, (1, 10))  # Source sequence
translation = model.generate(src_seq)
print(f"Translation: {translation}")
```

## Performance Optimization

### Memory Efficiency

**Gradient Checkpointing:**
```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientTransformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, use_checkpoint=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x, mask)
            else:
                x = layer(x, mask)
        return x
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

# Training loop with mixed precision
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(batch)
        loss = criterion(output, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Model Parallelism

```python
class ParallelTransformer(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, d_ff, num_gpus=2):
        super().__init__()
        self.num_gpus = num_gpus
        
        # Split layers across GPUs
        layers_per_gpu = num_layers // num_gpus
        self.layers = nn.ModuleList()
        
        for i in range(num_gpus):
            start_layer = i * layers_per_gpu
            end_layer = (i + 1) * layers_per_gpu if i < num_gpus - 1 else num_layers
            
            gpu_layers = nn.ModuleList([
                EncoderLayer(d_model, num_heads, d_ff)
                for _ in range(end_layer - start_layer)
            ]).to(f'cuda:{i}')
            
            self.layers.append(gpu_layers)
    
    def forward(self, x, mask=None):
        for gpu_layers in self.layers:
            for layer in gpu_layers:
                x = layer(x, mask)
        return x
```

## Common Architectures

### 1. BERT Architecture

```python
class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.segment_embedding = nn.Embedding(2, d_model)  # For sentence pairs
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        if token_type_ids is not None:
            x = x + self.segment_embedding(token_type_ids)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)
```

### 2. GPT Architecture

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        
        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask.unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x, mask=causal_mask)
        
        x = self.norm(x)
        return self.output_projection(x)
```

### 3. T5 Architecture

```python
class T5(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072):
        super().__init__()
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderLayerWithCrossAttention(d_model, num_heads, d_ff, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, target_ids=None, attention_mask=None):
        # Encode
        x = self.shared_embedding(input_ids)
        x = self.pos_encoding(x)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
        
        for layer in self.encoder:
            x = layer(x, mask)
        
        if target_ids is not None:
            # Decode
            y = self.shared_embedding(target_ids)
            y = self.pos_encoding(y)
            
            causal_mask = torch.tril(torch.ones(y.size(1), y.size(1))).unsqueeze(0)
            
            for layer in self.decoder:
                y = layer(y, x, causal_mask, mask)
            
            y = self.norm(y)
            return self.output_projection(y)
        else:
            return x
```

## Conclusion

The Transformer architecture has become the foundation of modern natural language processing and artificial intelligence. Understanding its components, variants, and implementation details is crucial for building effective language models and AI systems.

**Key Takeaways:**
1. **Attention is the core innovation** that enables parallel processing and long-range dependencies
2. **Different architectural variants** serve different purposes (understanding vs. generation)
3. **Proper implementation** requires attention to normalization, residual connections, and training stability
4. **Performance optimization** is essential for scaling to large models and long sequences
5. **Architectural choices** depend on the specific task and requirements

**Next Steps:**
- Explore advanced attention mechanisms like Flash Attention
- Study model scaling techniques and efficiency improvements
- Practice with real-world applications and datasets
- Experiment with different architectural variants for specific tasks

---

**References:**
- "Attention Is All You Need" - Vaswani et al.
- "BERT: Pre-training of Deep Bidirectional Transformers" - Devlin et al.
- "Language Models are Unsupervised Multitask Learners" - Radford et al.
- "Exploring the Limits of Transfer Learning" - Raffel et al. 
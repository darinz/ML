# Transformers and Large Language Models

[![Transformers](https://img.shields.io/badge/Transformers-Attention%20Mechanisms-blue.svg)](https://en.wikipedia.org/wiki/Transformer_(machine_learning))
[![LLM](https://img.shields.io/badge/LLM-Large%20Language%20Models-green.svg)](https://en.wikipedia.org/wiki/Large_language_model)
[![Attention](https://img.shields.io/badge/Attention-Self--Attention-purple.svg)](https://en.wikipedia.org/wiki/Attention_(machine_learning))

Comprehensive materials covering transformer architectures and large language models, from attention mechanisms to modern training techniques and applications.

## Overview

Transformers and language models have become the foundation of modern AI systems, powering applications from machine translation to conversational AI through attention-based architectures.

## Materials

### Theory
- **[01_attention_mechanism.md](01_attention_mechanism.md)** - Attention mechanism fundamentals and mathematical foundations
- **[02_transformer_architecture.md](02_transformer_architecture.md)** - Complete transformer architecture and components
- **[03_large_language_models.md](03_large_language_models.md)** - LLM scaling, training, and optimization
- **[04_training_and_optimization.md](04_training_and_optimization.md)** - Advanced training techniques and optimization
- **[05_applications_and_use_cases.md](05_applications_and_use_cases.md)** - Real-world applications and use cases
- **[06_hands-on_coding.md](06_hands-on_coding.md)** - Practical implementation guide

### Core Components
- **[code/attention.py](code/attention.py)** - Multi-head attention implementation
- **[code/transformer.py](code/transformer.py)** - Complete transformer architecture
- **[code/positional_encoding.py](code/positional_encoding.py)** - Positional encoding methods
- **[code/training.py](code/training.py)** - Training loop and optimization

### Advanced Features
- **[code/flash_attention.py](code/flash_attention.py)** - Memory-efficient attention
- **[code/rope_encoding.py](code/rope_encoding.py)** - Rotary positional encoding
- **[code/model_parallel.py](code/model_parallel.py)** - Distributed training examples
- **[code/quantization.py](code/quantization.py)** - Model compression techniques

### Applications
- **[code/text_classification.py](code/text_classification.py)** - BERT-style classification
- **[code/text_generation.py](code/text_generation.py)** - GPT-style generation
- **[code/translation.py](code/translation.py)** - Sequence-to-sequence translation
- **[code/summarization.py](code/summarization.py)** - Text summarization models

### Supporting Files
- **code/requirements.txt** - Python dependencies

## Key Concepts

### Attention Mechanism
**Scaled Dot-Product Attention**: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

**Multi-Head Attention**: Parallel attention mechanisms for different representation subspaces

### Transformer Architecture
**Encoder-Decoder Structure**:
- **Encoder**: Multi-head self-attention + feed-forward networks
- **Decoder**: Masked self-attention + cross-attention + feed-forward

**Key Components**: Layer normalization, residual connections, positional encoding

### Large Language Models
**Scaling Laws**: Chinchilla scaling for optimal model size vs. training compute

**Training Techniques**: Mixed precision, gradient checkpointing, model parallelism, ZeRO optimization

**Pre-training Objectives**: MLM (BERT), CLM (GPT), span corruption (T5)

## Applications

- **Machine Translation**: Sequence-to-sequence translation
- **Text Classification**: Sentiment analysis, topic classification
- **Text Generation**: Creative writing, code generation, dialogue systems
- **Question Answering**: Extractive and generative QA
- **Text Summarization**: Abstractive and extractive summarization
- **Multimodal**: Vision-language models, audio processing

## Getting Started

1. Read `01_attention_mechanism.md` for attention fundamentals
2. Study `02_transformer_architecture.md` for architecture details
3. Learn `03_large_language_models.md` for LLM concepts
4. Explore `04_training_and_optimization.md` for training techniques
5. Understand `05_applications_and_use_cases.md` for applications
6. Follow `06_hands-on_coding.md` for implementation

## Prerequisites

- Deep learning fundamentals
- Natural language processing basics
- Linear algebra and matrix operations
- Python, PyTorch, NumPy

## Installation

```bash
pip install -r code/requirements.txt
```

## Quick Start

```python
# Basic Attention
from code.attention import MultiHeadAttention
attention = MultiHeadAttention(d_model=512, num_heads=8)

# Transformer
from code.transformer import Transformer
model = Transformer(src_vocab_size=30000, tgt_vocab_size=30000, d_model=512)

# Flash Attention
from code.flash_attention import FlashAttention
flash_attn = FlashAttention(softmax_scale=1.0)

# RoPE Encoding
from code.rope_encoding import RoPE
rope = RoPE(dim=512, max_position_embeddings=2048)
```

## Reference Papers

- **Attention Is All You Need**: Original transformer paper
- **BERT**: Bidirectional transformer for understanding
- **GPT**: Autoregressive transformer for generation
- **T5**: Text-to-text transfer transformer
- **Flash Attention**: Memory-efficient attention
- **RoPE**: Rotary positional encoding

## Modern Developments

- **Efficient Training**: Reducing computational requirements
- **Model Compression**: Quantization and distillation
- **Multimodal Models**: Integrating vision, audio, and text
- **Reasoning Capabilities**: Chain-of-thought and logical reasoning
- **Specialized Models**: Domain-specific language models 
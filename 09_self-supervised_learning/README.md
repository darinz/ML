# Self-Supervised Learning & Foundation Models

[![Self-Supervised](https://img.shields.io/badge/Self--Supervised-Learning-blue.svg)](https://en.wikipedia.org/wiki/Self-supervised_learning)
[![Foundation Models](https://img.shields.io/badge/Foundation-Models-green.svg)](https://en.wikipedia.org/wiki/Foundation_model)
[![LLM](https://img.shields.io/badge/LLM-Large%20Language%20Models-purple.svg)](https://en.wikipedia.org/wiki/Large_language_model)

Comprehensive introduction to self-supervised learning, foundation models, and large language models with theoretical concepts and practical implementations.

## Overview

Self-supervised learning leverages unlabeled data to learn useful representations through pretext tasks, while foundation models provide adaptable base models for multiple downstream tasks.

## Materials

### Theory
- **[01_pretraining.md](01_pretraining.md)** - Self-supervised learning, contrastive learning, adaptation methods, practical considerations
- **[02_pretrain_llm.md](02_pretrain_llm.md)** - Large language model pretraining, Transformer architectures, text generation, adaptation strategies
- **[03_hands-on_coding.md](03_hands-on_coding.md)** - Practical implementation guide

### Implementation
- **[code/pretraining_examples.py](code/pretraining_examples.py)** - Self-supervised learning implementations, contrastive learning, data augmentation
- **[code/pretrain_llm_examples.py](code/pretrain_llm_examples.py)** - Language modeling, Transformer interface, text generation, adaptation methods

### Interactive Notebook
- **[02_pretrain_llm.ipynb](02_pretrain_llm.ipynb)** - Jupyter notebook with interactive LLM examples

### Supporting Files
- **code/requirements.txt** - Python dependencies
- **img/** - Visualizations and diagrams

## Key Concepts

### Self-Supervised Learning
**Objective**: Learn useful representations from unlabeled data

**Methods**:
- **Contrastive Learning**: Make similar views close, different views far
- **Masked Prediction**: Predict masked parts of input
- **Pretext Tasks**: Design tasks that require understanding

### Foundation Models
**Characteristics**:
- Large-scale pretraining on broad data
- Adaptable to multiple downstream tasks
- Transfer learning capabilities

### Large Language Models (LLMs)
**Architecture**: Transformer-based models

**Training**: Autoregressive language modeling

**Adaptation Methods**:
- **Finetuning**: Update model parameters
- **Zero-shot**: No parameter updates
- **In-context Learning**: Adaptation via prompting

## Applications

- **Computer Vision**: Image representation learning
- **Natural Language Processing**: Text understanding and generation
- **Multimodal Learning**: Combining vision and language
- **Transfer Learning**: Adapting to new domains
- **Few-shot Learning**: Learning with limited examples

## Getting Started

1. Read `01_pretraining.md` for self-supervised learning fundamentals
2. Study `02_pretrain_llm.md` for LLM concepts
3. Use `03_hands-on_coding.md` for practical guidance
4. Run Python examples to see algorithms in action
5. Explore the Jupyter notebook for interactive learning

## Prerequisites

- Deep learning fundamentals
- PyTorch experience
- Understanding of neural networks
- Basic NLP concepts (for LLM sections)

## Installation

```bash
pip install -r code/requirements.txt
```

## Running Examples

```bash
python code/pretraining_examples.py
python code/pretrain_llm_examples.py
```

## Quick Start Code

```python
# Contrastive Learning
from code.pretraining_examples import contrastive_learning_example
features, loss = contrastive_learning_example()

# Language Modeling
from code.pretrain_llm_examples import language_modeling_example
model, generated_text = language_modeling_example()

# Using Transformers
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

## Method Comparison

| Method | Data Requirement | Training Cost | Adaptation | Use Case |
|--------|------------------|---------------|------------|----------|
| Supervised | Labeled data | High | Finetuning | Specific tasks |
| Self-Supervised | Unlabeled data | High | Linear probe/Finetuning | Representation learning |
| Foundation Models | Massive data | Very high | Prompting/Finetuning | General purpose | 
# Self-Supervised Learning & Foundation Models

This module provides a comprehensive introduction to self-supervised learning, foundation models, and large language models (LLMs). It covers both theoretical concepts and practical implementations, with a focus on modern machine learning techniques that leverage unlabeled data and pretraining.

## Contents

- **01_pretraining.md**: In-depth explanation of self-supervised learning, pretraining, contrastive learning, adaptation methods (linear probe, finetuning), and practical considerations. Includes mathematical derivations, geometric intuition, and best practices.
- **02_pretrain_llm.md**: Detailed coverage of large language model pretraining, the chain rule for language modeling, Transformer architectures, text generation, adaptation strategies (finetuning, zero-shot, in-context learning), and prompt engineering.
- **pretraining_examples.py**: Comprehensive Python implementations of self-supervised learning concepts, including:
  - Supervised pretraining (ImageNet-style)
  - Contrastive learning (SimCLR-style)
  - Data augmentation for contrastive learning
  - Linear probe and finetuning adaptation
  - Feature visualization and analysis
  - Practical notes and best practices
  - All code is annotated for educational clarity and includes visualizations and comparisons
- **pretrain_llm_examples.py**: Python examples demonstrating:
  - Language modeling and the chain rule
  - Transformer input/output interface
  - Autoregressive text generation with temperature sampling
  - Conceptual finetuning with HuggingFace Trainer
  - Zero-shot and in-context learning (prompting)
  - Practical notes and best practices for LLMs

## Key Concepts

- **Self-Supervised Learning**: Learning useful representations from unlabeled data by designing pretext tasks (e.g., contrastive learning, masked prediction).
- **Foundation Models**: Large models pretrained on broad data, adaptable to many downstream tasks (e.g., vision, language).
- **Contrastive Learning**: Learning by making representations of different views of the same data similar, and different data dissimilar.
- **Adaptation Methods**: Linear probe (feature extraction) and finetuning (full model training) for downstream tasks.
- **Large Language Models (LLMs)**: Pretrained models for text, using the chain rule, Transformers, and autoregressive generation.
- **In-Context Learning**: Adapting to new tasks via prompting, without parameter updates.

## Educational Features

- **Detailed Explanations**: All markdown and code files are annotated with step-by-step guides, mathematical derivations, and conceptual insights.
- **Practical Implementations**: Python files provide hands-on examples, visualizations, and comparisons with best practices.
- **Best Practices**: Each section includes practical notes for real-world applications, including data requirements, computational considerations, and evaluation tips.

## Prerequisites

- Python 3.7+
- PyTorch
- scikit-learn
- matplotlib, seaborn
- transformers (for LLM examples)

Install requirements with:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn transformers
```

## How to Use

- Read the markdown files for conceptual understanding and mathematical background.
- Run the Python scripts to see practical implementations and visualizations.
- Use the code as a template for your own self-supervised learning or LLM projects.

## Further Reading

- [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Stanford CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Stanford CS229: Machine Learning](http://cs229.stanford.edu/)

---

*This folder is part of a larger machine learning curriculum. For more topics, see the main project README.* 
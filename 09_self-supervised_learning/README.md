# Self-Supervised Learning and Foundation Models

This folder provides a comprehensive, educational introduction to self-supervised learning, pretraining, and foundation models, with a special focus on large language models (LLMs) and modern deep learning techniques. It is designed for students, researchers, and practitioners who want to understand both the mathematical foundations and practical implementations of these cutting-edge topics.

## Folder Structure & Contents

- **01_pretraining.md**  
  Detailed notes on the motivation, mathematical formulation, and practical methods for self-supervised pretraining, especially in computer vision. Covers supervised pretraining, contrastive learning (e.g., SimCLR), and adaptation strategies (linear probe, finetuning).

- **pretraining_examples.py**  
  Python code examples for all major pretraining concepts: data normalization, covariance matrix, eigen-decomposition, PCA, contrastive learning, and adaptation methods. Includes synthetic data and visualization for hands-on learning.

- **02_pretrain_llm.md**  
  In-depth notes on large language models (LLMs), language modeling, the Transformer architecture, autoregressive text generation, and adaptation methods (finetuning, zero-shot, in-context learning). Includes step-by-step explanations, analogies, and practical tips.

- **pretrain_llm_examples.py**  
  Python code demonstrating language modeling, Transformer input/output, text generation with temperature, conceptual finetuning (using HuggingFace), and zero-shot/in-context learning via prompting. Uses HuggingFace Transformers for real-world relevance.

- **requirements.txt**  
  List of Python dependencies for running the code examples (PyTorch, HuggingFace Transformers, numpy, matplotlib, scikit-learn, etc.).

- **img/**  
  Folder containing supporting images and diagrams for the markdown notes.

## Main Topics Covered

- **Self-supervised learning:** Learning useful representations from unlabeled data by creating surrogate tasks (e.g., contrastive learning, masked prediction).
- **Pretraining and adaptation:** The two-phase paradigm of foundation models—first learn general features from large datasets, then adapt to specific tasks with little or no labeled data.
- **Foundation models:** Large, versatile models (like GPT, BERT, CLIP) that can be adapted to many tasks via finetuning, zero-shot, or in-context learning.
- **Large language models (LLMs):** Mathematical and practical foundations of language modeling, the Transformer architecture, and modern adaptation techniques.

## Quickstart: Running the Code

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run pretraining examples (vision, contrastive, adaptation):**
   ```bash
   python pretraining_examples.py
   ```
3. **Run language modeling and LLM examples:**
   ```bash
   python pretrain_llm_examples.py
   ```

> All code is designed for educational clarity, with synthetic or small real data for quick experimentation. GPU is recommended for large models, but CPU will work for small demos.

## Prerequisites
- Basic Python and PyTorch
- Familiarity with machine learning concepts (supervised learning, neural networks)
- No prior experience with Transformers or LLMs required—notes and code are beginner-friendly!

## Learning Approach
- **Step-by-step explanations:** All notes and code are written to be accessible, with analogies, diagrams, and practical tips.
- **Mathematical rigor + code:** Each concept is explained both mathematically and with runnable code.
- **Hands-on:** Try modifying the code, prompts, or parameters to see how models behave in practice.

## Further Reading
- [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Attention is All You Need (Transformer)](https://arxiv.org/abs/1706.03762)
- [On the Opportunities and Risks of Foundation Models (Bommasani et al., 2021)](https://arxiv.org/abs/2108.07258)

---

**Explore, experiment, and enjoy learning about the future of AI!** 
# Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()

A comprehensive learning resource providing a broad introduction to machine learning and statistical pattern recognition. This collection covers fundamental concepts, modern algorithms, and practical applications in the field of machine learning.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Curriculum](#curriculum)
  - [I. Supervised Learning](#i-supervised-learning)
  - [II. Neural Networks and Deep Learning](#ii-neural-networks-and-deep-learning)
  - [III. Model Evaluation and Optimization](#iii-model-evaluation-and-optimization)
  - [IV. Unsupervised Learning](#iv-unsupervised-learning)
  - [V. Reinforcement Learning](#v-reinforcement-learning)
  - [VI. Recent Advances and Applications](#vi-recent-advances-and-applications)
- [Reference Materials](#reference-materials)
- [Development Guidelines](#development-guidelines)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)


## Overview

This learning resource serves as a comprehensive guide for machine learning, covering topics from basic supervised learning to advanced deep learning and reinforcement learning techniques. The material is designed to provide both theoretical understanding and practical implementation skills for students, researchers, and practitioners in the field.

### Learning Approach

The curriculum follows a progressive learning path that builds from fundamental concepts to advanced applications. Each topic includes theoretical foundations, mathematical derivations, and practical implementation examples. The material emphasizes both understanding the underlying principles and developing hands-on coding skills.

### Key Features

- **Comprehensive Coverage**: From classical machine learning algorithms to cutting-edge deep learning techniques
- **Practical Implementation**: Code examples and implementation guidelines for real-world applications
- **Mathematical Rigor**: Detailed derivations and proofs for understanding algorithm foundations
- **Modern Applications**: Coverage of recent advances including transformers, large language models, and self-supervised learning
- **Industry Relevance**: Focus on techniques and applications used in current industry practice

### Target Audience

This resource is designed for:
- **Students**: Learning machine learning fundamentals and advanced topics
- **Researchers**: Exploring cutting-edge techniques and implementations
- **Practitioners**: Applying machine learning in real-world scenarios
- **Educators**: Teaching machine learning concepts and methodologies

### Learning Outcomes

Upon completing this curriculum, learners will be able to:
- Understand fundamental machine learning concepts and algorithms
- Implement and optimize various ML models from scratch
- Apply deep learning techniques to complex problems
- Evaluate and improve model performance effectively
- Stay current with recent advances in the field
- Apply machine learning to real-world applications across domains

## Prerequisites

To successfully work with the material in this repository, you should have:

- **Programming Skills**: Ability to write non-trivial programs in Python with NumPy
- **Mathematics**: Familiarity with probability theory, multivariable calculus, and linear algebra
- **Computer Science**: Basic understanding of algorithms and data structures

### Recommended Review Materials

- [Python Review](https://github.com/darinz/Toolkit/tree/main/Python) - Essential Python concepts
- [NumPy Review](https://github.com/darinz/Toolkit/tree/main/NumPy) - Numerical computing with NumPy
- [Math Review](https://github.com/darinz/Math) - Mathematical foundations for machine learning

## Curriculum

### I. Supervised Learning

Supervised learning involves training models on labeled data to make predictions or classifications. This section covers fundamental algorithms that learn from input-output pairs, including linear models, classification techniques, and generative approaches for pattern recognition and prediction tasks.

#### 01. Linear Models
- Linear Regression
- Classification and Logistic Regression
- Generalized Linear Models
- Ridge and Lasso Regression

#### 02. Advanced Classification
- Support Vector Machines (SVM)
- Kernel Methods
- Decision Trees and Random Forests
- Ensemble Methods (Bagging, Boosting, AdaBoost)

#### 03. Generative Learning
- Generative Learning Algorithms
- Gaussian Discriminant Analysis (GDA)
- Naive Bayes Classifiers

### II. Neural Networks and Deep Learning

Neural networks and deep learning represent the cutting edge of machine learning, enabling complex pattern recognition through multi-layered architectures. This section explores artificial neural networks, from basic perceptrons to advanced architectures like transformers, covering both theoretical foundations and practical applications in computer vision, natural language processing, and beyond.

#### 04. Fundamentals
- Neural Network Architecture (MLP)
- Multi-Class Loss Functions
- Backpropagation Algorithm
- Activation Functions and Optimization

#### 05. Modern Architectures
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)

#### 06. Transformers and Language Models
- Self-Attention Mechanism
- Transformer Architecture
- BERT, GPT, and Modern Language Models
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [CS224N Self-Attention, Transformers Notes](./reference/CS224N_Self-Attention-Transformers-2023_Draft.pdf)

#### 07. Computer Vision
- Image Classification
- Object Detection
- Semantic Segmentation
- Transfer Learning and Pre-trained Models

### III. Model Evaluation and Optimization

Model evaluation and optimization are crucial for building effective machine learning systems. This section covers techniques for assessing model performance, preventing overfitting, and optimizing training processes. Topics include evaluation metrics, regularization strategies, and advanced optimization algorithms essential for practical machine learning applications.

#### 08. Generalization and Regularization
- Bias-Variance Tradeoff
- Overfitting and Underfitting
- Regularization Techniques
- Model Selection and Validation

#### 09. Evaluation Metrics
- Classification Metrics (Accuracy, Precision, Recall, F1-Score)
- Regression Metrics (MSE, MAE, R²)
- ROC Curves and AUC
- Cross-Validation Strategies

#### 10. Optimization
- Gradient Descent Variants
- Stochastic Optimization
- Learning Rate Scheduling
- Advanced Optimizers (Adam, RMSprop)

### IV. Unsupervised Learning

Unsupervised learning discovers hidden patterns and structures in data without predefined labels. This section explores algorithms for clustering, dimensionality reduction, and generative modeling. These techniques are fundamental for data exploration, feature learning, and understanding underlying data distributions in various domains.

#### 11. Clustering
- K-Means Algorithm
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models

#### 12. Dimensionality Reduction
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- t-SNE and UMAP
- Autoencoders

#### 13. Modern Unsupervised Learning
- Self-Supervised Learning
- Contrastive Learning
- Foundation Models
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)

### V. Reinforcement Learning

Reinforcement learning enables agents to learn optimal behaviors through interaction with environments and feedback from rewards. This section covers algorithms for sequential decision-making, from basic dynamic programming to advanced policy gradient methods. Applications include robotics, game playing, autonomous systems, and control theory.

#### 14. Fundamentals
- Markov Decision Processes (MDPs)
- Value Functions and Policy Functions
- Dynamic Programming

#### 15. Model-Free Methods
- Monte Carlo Methods
- Temporal Difference Learning
- Q-Learning and SARSA
- Deep Q-Networks (DQN)

#### 16. Policy-Based Methods
- Policy Gradient Methods
- REINFORCE Algorithm
- Actor-Critic Methods
- Proximal Policy Optimization (PPO)

#### 17. Advanced Control
- Linear Quadratic Regulator (LQR)
- Differential Dynamic Programming (DDP)
- Linear Quadratic Gaussian (LQG)

### VI. Recent Advances and Applications

Recent advances in machine learning have revolutionized various fields and applications. This section explores cutting-edge developments including large language models, advanced computer vision techniques, and practical applications across domains. Topics cover the latest innovations in AI and their real-world implementations in industry and research.

#### 18. Large Language Models
- Transformer Architecture Evolution
- Prompt Engineering
- Fine-tuning Strategies
- Multimodal Models

#### 19. Computer Vision Advances
- Vision Transformers (ViT)
- Self-Supervised Learning in Vision
- Contrastive Learning (SimCLR, MoCo)
- Foundation Models for Vision

#### 20. Practical Applications
- Natural Language Processing
- Computer Vision
- Speech Recognition
- Robotics and Control
- Bioinformatics
- Recommendation Systems

## Reference Materials

- [CS229 Machine Learning Course](https://cs229.stanford.edu/) - Stanford University
- [CS224N Self-Attention, Transformers Notes](./reference/CS224N_Self-Attention-Transformers-2023_Draft.pdf)
- [CS229 Decision Trees Notes](./reference/CS229_Decision-Trees-Notes.pdf)
- [CS229 Lecture Notes](./reference/CS229_Lecture-Notes.pdf)
- [CS229 Linear Algebra Review](./reference/CS229_Linear_Algebra_Review.pdf)
- [CS229 Probability Review](./reference/CS229_Probability_Review.pdf)

## Development Guidelines

See [DEVELOPMENT_GUIDELINES.md](./DEVELOPMENT_GUIDELINES.md) for comprehensive information on code quality, testing, debugging, and best practices for machine learning development in this project.

## Contributing

This repository is under active development. Learning materials will be added as they become available. Contributions are welcome!

## Acknowledgments

This repository is based on Stanford University's [CS229 Machine Learning course](https://cs229.stanford.edu/) and incorporates additional modern machine learning concepts and applications.
# Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()

A comprehensive learning resource providing a broad introduction to machine learning and statistical pattern recognition. This collection covers fundamental concepts, modern algorithms, and practical applications in the field of machine learning.

## Table of Contents

- [Overview](#overview)
- [Real-World Applications](#real-world-applications)
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

The curriculum follows a progressive learning path that builds from fundamental concepts to advanced applications. Each topic includes theoretical foundations, mathematical derivations, and practical implementation examples. The material emphasizes both understanding the underlying principles and developing hands-on coding skills.es

### Learning Outcomes

Upon completing this curriculum, learners will be able to:
- Understand fundamental machine learning concepts and algorithms
- Implement and optimize various ML models from scratch
- Apply deep learning techniques to complex problems
- Evaluate and improve model performance effectively
- Stay current with recent advances in the field
- Apply machine learning to real-world applications across domains

## Real-World Applications

Machine learning is transforming industries and solving complex problems across the globe. To see how the algorithms and techniques covered in this curriculum are applied in practice, explore our comprehensive guide to **[Machine Learning Applications](./ML_APPLICATIONS.md)**.

This guide showcases real-world applications across 12 major domains:

- **Healthcare & Medicine**: Medical imaging, drug discovery, personalized medicine
- **Finance & Banking**: Fraud detection, algorithmic trading, risk assessment
- **Technology & Software**: NLP, computer vision, software development
- **Transportation & Logistics**: Autonomous vehicles, supply chain optimization
- **Retail & E-commerce**: Recommendation systems, marketing optimization
- **Manufacturing & Industry**: Predictive maintenance, process optimization
- **Entertainment & Media**: Content creation, recommendation algorithms
- **Environmental Science**: Climate modeling, conservation, renewable energy
- **Education & Learning**: Personalized learning, educational technology
- **Security & Cybersecurity**: Threat detection, digital forensics
- **Agriculture & Food**: Precision agriculture, food safety
- **Energy & Utilities**: Smart grids, resource optimization

Understanding these applications will help you see the practical impact of machine learning and guide your learning journey toward domains that interest you most.

## Prerequisites

To successfully work with the material in this repository, you should have:

- **Programming Skills**: Ability to write non-trivial programs in Python with NumPy
- **Mathematics**: Familiarity with probability theory, multivariable calculus, and linear algebra
- **Computer Science**: Basic understanding of algorithms and data structures

### Recommended Review Materials

- [Python Review](https://github.com/darinz/Toolkit/tree/main/Python) - Essential Python concepts
- [NumPy Review](https://github.com/darinz/Toolkit/tree/main/NumPy) - Numerical computing with NumPy
- [Math Review](https://github.com/darinz/Math) - Mathematical foundations for machine learning

For comprehensive foundational materials, see the **[00. Math, Python, and NumPy Review](#00-math-python-and-numpy-review)** section in the curriculum below.

## Curriculum

### 00. Math, Python, and NumPy Review

Essential prerequisites for machine learning, covering mathematical foundations, Python programming, and numerical computing with NumPy.

**[📁 View Materials](./00_math_python_numpy_review/)**

#### 00.1 Linear Algebra Review
- Vector operations and matrix algebra
- Eigenvalues and eigenvectors
- Linear transformations and projections
- Applications in machine learning

#### 00.2 Probability Review
- Probability theory fundamentals
- Random variables and distributions
- Bayes' theorem and conditional probability
- Statistical inference concepts

#### 00.3 Python and NumPy
- Python programming essentials
- NumPy for numerical computing
- Data manipulation and visualization
- Practical exercises and examples

#### 00.4 Practice Problems
- Mathematical problem sets
- Programming exercises
- Solutions and explanations

### I. Supervised Learning

Supervised learning involves training models on labeled data to make predictions or classifications. This section covers fundamental algorithms that learn from input-output pairs, including linear models, classification techniques, and generative approaches for pattern recognition and prediction tasks.


#### 01. [Linear Models](./01_linear_models/)
- **[Linear Regression](./01_linear_models/01_linear_regression/)** - Basic linear regression implementation and theory
- **[Classification and Logistic Regression](./01_linear_models/02_classification_logistic_regression/)** - Binary and multi-class classification with logistic regression
- **[Generalized Linear Models](./01_linear_models/03_generalized_linear_models/)** - Extension of linear models to various response distributions
- **[Ridge and Lasso Regression](./01_linear_models/04_ridge_lasso_regression/)** - Regularization techniques for linear models

*Note: This section is currently under development.*

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
- [CS224N Self-Attention, Transformers Notes](./reference/CS224N_Self-Attention-Transformers-2023_Draft.pdf)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Transformer Architecture
- BERT, GPT, and Modern Language Models


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

### Core Course Materials
- [CS229 Machine Learning Course](https://cs229.stanford.edu/) - Stanford University
- [CS229 Lecture Notes](./reference/CS229_Lecture-Notes.pdf) - Comprehensive course notes
- [CS229 Decision Trees Notes](./reference/CS229_Decision-Trees-Notes.pdf) - Decision tree algorithms and theory

### Mathematical Foundations
- [Exponential Family Chapter 8](./reference/exponential_family_chapter8.pdf) - Exponential family distributions
- [Gradient and Hessian Lecture 4 Extra Materials](./reference/gradient-hessian_lecture-4-extra-materials.pdf) - Advanced optimization concepts

### Advanced Topics
- [CS224N Self-Attention, Transformers Notes](./reference/CS224N_Self-Attention-Transformers-2023_Draft.pdf) - Modern transformer architectures
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide to transformers

### Additional Resources
- [Python Review](https://github.com/darinz/Toolkit/tree/main/Python) - Essential Python concepts
- [NumPy Review](https://github.com/darinz/Toolkit/tree/main/NumPy) - Numerical computing with NumPy
- [Math Review](https://github.com/darinz/Math) - Mathematical foundations for machine learning
- [CIML v0.99 All](./reference/ciml-v0_99-all.pdf) - Comprehensive machine learning textbook

### Cheatsheets and Quick References
- [Super Cheatsheet Machine Learning](./reference/super-cheatsheet-machine-learning.pdf) - Comprehensive ML reference
- [Deep Learning Cheatsheet](./reference/cheatsheet-deep-learning.pdf) - Neural networks and deep learning
- [Supervised Learning Cheatsheet](./reference/cheatsheet-supervised-learning.pdf) - Supervised learning algorithms
- [Unsupervised Learning Cheatsheet](./reference/cheatsheet-unsupervised-learning.pdf) - Unsupervised learning techniques
- [Machine Learning Tips and Tricks](./reference/cheatsheet-machine-learning-tips-and-tricks.pdf) - Practical ML advice
- [Python for Data Science Cheatsheet](https://www.datacamp.com/community/data-science-cheatsheets) - Python, NumPy, Pandas, Matplotlib
- [Scikit-learn Cheatsheet](https://scikit-learn.org/stable/tutorial/machine_learning_map/) - Machine learning algorithm selection guide

## Development Guidelines

See [DEVELOPMENT_GUIDELINES.md](./DEVELOPMENT_GUIDELINES.md) for comprehensive information on code quality, testing, debugging, and best practices for machine learning development in this project.

## Contributing

This repository is under active development. Learning materials will be added as they become available. Contributions are welcome!

## Acknowledgments

This repository is based on Stanford University's [CS229 Machine Learning course](https://cs229.stanford.edu/) and incorporates additional modern machine learning concepts and applications.
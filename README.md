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

The curriculum follows a progressive learning path that builds from fundamental concepts to advanced applications. Each topic includes theoretical foundations, mathematical derivations, and practical implementation examples. The material emphasizes both understanding the underlying principles and developing hands-on coding skills.

#### Active Learning Strategies
- **Rederive all calculations**: Don't just read the math—work through every derivation step-by-step on paper or in a notebook
- **Implement from scratch**: Copy and paste code examples into your own Jupyter notebook or Python environment, then modify and experiment
- **Build incrementally**: Start with simple implementations and gradually add complexity
- **Test your understanding**: After each concept, try to explain it to yourself or others without referring to notes

#### Retention Techniques
- **Spaced repetition**: Review previous concepts regularly, especially before moving to related topics
- **Create your own examples**: Apply algorithms to datasets you're interested in or create synthetic data
- **Document your learning**: Keep a learning journal with key insights, common pitfalls, and personal examples
- **Teach others**: Explain concepts to peers or write blog posts about what you've learned
- **Build a portfolio**: Implement projects that combine multiple concepts from different sections

#### Practical Implementation
- **Use version control**: Track your code changes and experiments with Git
- **Experiment systematically**: Vary one parameter at a time and document the results
- **Visualize everything**: Create plots and graphs to understand data distributions, model behavior, and results
- **Compare implementations**: Try different approaches to the same problem and analyze trade-offs

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
- [Machine Learning Notation Guide](./NOTATION.md) - Mathematical notation and symbols used throughout the curriculum

For comprehensive foundational materials, see the **[00. Math, Python, and NumPy Review](#00-math-python-and-numpy-review)** section in the curriculum below.

## Curriculum

### 00. Math, Python, and NumPy Review

Essential prerequisites covering linear algebra, probability theory, Python programming, and numerical computing with NumPy.

**[View Materials](./00_math_python_numpy_review/)**

- **Linear Algebra**: Vector operations, matrix algebra, eigenvalues, linear transformations
- **Probability**: Probability theory, random variables, Bayes' theorem, statistical inference  
- **Python & NumPy**: Programming essentials, numerical computing, data manipulation
- **Practice Problems**: Mathematical exercises and programming challenges with solutions

### I. Supervised Learning

Supervised learning involves training models on labeled data to make predictions or classifications. This section covers fundamental algorithms that learn from input-output pairs, including linear models, classification techniques, and generative approaches for pattern recognition and prediction tasks.

#### 01. [Linear Models](./01_linear_models/)
Comprehensive coverage of fundamental linear models including:
- **[Linear Regression](./01_linear_models/01_linear_regression/)** - Least squares, gradient descent, normal equations, and probabilistic interpretation
- **[Classification and Logistic Regression](./01_linear_models/02_classification_logistic_regression/)** - Binary and multi-class classification with sigmoid and softmax functions
- **[Generalized Linear Models](./01_linear_models/03_generalized_linear_models/)** - Exponential family distributions and unified framework for various response types

#### 02. [Generative Learning](./02_generative_learning/)
Generative approaches that model the joint distribution p(x,y) through p(x|y) and p(y):
- **[Gaussian Discriminant Analysis (GDA)](./02_generative_learning/01_gda.md)** - Multivariate normal modeling with shared covariance
- **[Naive Bayes Classifiers](./02_generative_learning/02_naive_bayes.md)** - Bernoulli and multinomial variants with feature independence
- **Implementation Examples** - Complete Python implementations with parameter estimation and prediction

#### 03. [Advanced Classification](./03_advanced_classification/)
Comprehensive coverage of Support Vector Machines (SVM) and kernel methods for non-linear classification:
- **[Kernel Methods](./03_advanced_classification/01_kernel_methods.md)** - Feature maps, kernel trick, polynomial and RBF kernels, computational efficiency
- **[Kernel Properties](./03_advanced_classification/02_kernel_properties.md)** - Mercer's theorem, kernel validation, kernel matrix construction
- **[SVM Margins](./03_advanced_classification/03_svm_margins.md)** - Functional vs geometric margins, optimal margin classifiers, Lagrange duality
- **[SVM Optimal Margin](./03_advanced_classification/04_svm_optimal_margin.md)** - Dual formulation, support vectors, SMO algorithm, KKT conditions
- **[SVM Regularization](./03_advanced_classification/05_svm_regularization.md)** - Soft margin SVM, slack variables, regularization parameter C
- **Implementation Examples** - Complete Python implementations with kernel methods, SMO algorithm, and regularization techniques

### II. Deep Learning

Neural networks and deep learning represent the cutting edge of machine learning, enabling complex pattern recognition through multi-layered architectures. This section explores artificial neural networks, from basic perceptrons to advanced architectures like transformers, covering both theoretical foundations and practical applications in computer vision, natural language processing, and beyond.

#### 04. Deep Learning
- Neural Network Architecture (MLP)
- Multi-Class Loss Functions
- Backpropagation Algorithm
- Activation Functions and Optimization

### III. Generalization and Regularization

Model evaluation and optimization are crucial for building effective machine learning systems. This section covers techniques for assessing model performance, preventing overfitting, and optimizing training processes. Topics include evaluation metrics, regularization strategies, and advanced optimization algorithms essential for practical machine learning applications.

#### 05. Generalization
- Bias-Variance Tradeoff
- Overfitting and Underfitting
- Regularization Techniques
- Model Selection and Validation

#### 06. Regularization and model selection
- Classification Metrics (Accuracy, Precision, Recall, F1-Score)
- Regression Metrics (MSE, MAE, R²)
- ROC Curves and AUC
- Cross-Validation Strategies

### IV. Unsupervised Learning

Unsupervised learning discovers hidden patterns and structures in data without predefined labels. This section explores algorithms for clustering, dimensionality reduction, and generative modeling. These techniques are fundamental for data exploration, feature learning, and understanding underlying data distributions in various domains.

#### 07. Clustering and k-means
- K-Means Algorithm
- Hierarchical Clustering
- DBSCAN
- Gaussian Mixture Models

#### 08. Dimensionality Reduction
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- t-SNE and UMAP
- Autoencoders

#### 09. Modern Unsupervised Learning
- Self-Supervised Learning
- Contrastive Learning
- Foundation Models
- Generative Adversarial Networks (GANs)
- Variational Autoencoders (VAEs)

### V. Reinforcement Learning

Reinforcement learning enables agents to learn optimal behaviors through interaction with environments and feedback from rewards. This section covers algorithms for sequential decision-making, from basic dynamic programming to advanced policy gradient methods. Applications include robotics, game playing, autonomous systems, and control theory.

#### 10. Fundamentals
- Markov Decision Processes (MDPs)
- Value Functions and Policy Functions
- Dynamic Programming
- [RL: Sutton and Barto textbook](http://incompleteideas.net/book/the-book-2nd.html)

#### 11. Model-Free Methods
- Monte Carlo Methods
- Temporal Difference Learning
- Q-Learning and SARSA
- Deep Q-Networks (DQN)

#### 12. Policy-Based Methods
- Policy Gradient Methods
- REINFORCE Algorithm
- Actor-Critic Methods
- Proximal Policy Optimization (PPO)

#### 13. Advanced Control
- Linear Quadratic Regulator (LQR)
- Differential Dynamic Programming (DDP)
- Linear Quadratic Gaussian (LQG)

#### 14 Multi-Armed Bandits
- Multi-armed bandits: [Bandit Algorithms textbook](https://tor-lattimore.com/downloads/book/book.pdf), [informal notes](https://courses.cs.washington.edu/courses/cse541/24sp/resources/lecture_notes.pdf)
- Linear bandits: [linear bandits paper](https://papers.nips.cc/paper_files/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html), [generalized linear bandits paper](https://papers.nips.cc/paper_files/paper/2010/hash/c2626d850c80ea07e7511bbae4c76f4b-Abstract.html), [pure exploration/BAI paper](https://arxiv.org/abs/1409.6110)
- Contextual bandits: [contextual bandits survey paper](https://www.ambujtewari.com/research/tewari17ads.pdf)

### VI. Recent Advances and Applications

Recent advances in machine learning have revolutionized various fields and applications. This section explores cutting-edge developments including large language models, advanced computer vision techniques, and practical applications across domains. Topics cover the latest innovations in AI and their real-world implementations in industry and research.

#### 15. Transformers and Language Models
- Self-Attention Mechanism
- Large Language Models (LLMs): [original transformer paper](https://arxiv.org/abs/1706.03762), [wikipedia on LLMs](https://en.wikipedia.org/wiki/Large_language_model)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [CS224N Self-Attention, Transformers Notes](./reference/CS224N_Self-Attention-Transformers-2023_Draft.pdf)
- Transformer Architecture
- BERT, GPT, and Modern Language Models

#### 16 Reinforcement Learning for training Large Language Models
- Reinforcement Learning (RL): [OpenAI's spinning up](https://spinningup.openai.com/en/latest/)
- RL for LLMs: [KL-control](https://arxiv.org/abs/1611.02796), [reward model](https://arxiv.org/abs/1706.03741), [InstructGPT paper (ChatGPT)](https://arxiv.org/abs/2203.02155), [recent DeepSeek R1 paper](https://arxiv.org/abs/2501.12948)

#### 17. Computer Vision Advances
- Vision Transformers (ViT)
- Self-Supervised Learning in Vision
- Contrastive Learning (SimCLR, MoCo)
- Foundation Models for Vision

#### 18. Practical Applications
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

This repository is based on Stanford University's [CS229 Machine Learning course](https://cs229.stanford.edu/) and incorporates additional modern machine learning concepts and applications. Additional inspiration comes from the University of Washington's [CSE446: Machine Learning course](https://courses.cs.washington.edu/courses/cse446/).
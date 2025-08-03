# Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange.svg)]()
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Comprehensive-brightgreen.svg)](https://en.wikipedia.org/wiki/Machine_learning)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Neural%20Networks-red.svg)](https://en.wikipedia.org/wiki/Deep_learning)
[![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-RL-purple.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Transformers](https://img.shields.io/badge/Transformers-LLMs-yellow.svg)](https://arxiv.org/abs/1706.03762)
[![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CV-cyan.svg)](https://en.wikipedia.org/wiki/Computer_vision)
[![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-lightblue.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Stanford CS229](https://img.shields.io/badge/Stanford-CS229%20Based-navy.svg)](https://cs229.stanford.edu/)
[![UW CSE446](https://img.shields.io/badge/UW-CSE446%20Inspired-purple.svg)](https://courses.cs.washington.edu/courses/cse446/)

> **Note**: Some files have both `.md` and `.ipynb` formats due to LaTeX rendering issues on GitHub. Use `.ipynb` files for proper math rendering, or view `.md` files in an editor with LaTeX support.

A comprehensive learning resource providing a broad introduction to machine learning and statistical pattern recognition. This collection covers fundamental concepts, modern algorithms, and practical applications in the field of machine learning.

> This is a math-heavy version focused on theoretical foundations and mathematical derivations. While there are less math-intensive machine learning material, understanding the mathematical principles behind algorithms is invaluable for research and comprehending academic papers. This foundation enables deeper insights into model behavior and more effective algorithm selection and tuning.

## Complementary Learning Resources

For additional practical machine learning topics not covered in this curriculum, explore **[Practical Statistical Learning (PSL)](https://github.com/darinz/PSL)**. This complementary resource covers:

- **Regression and Classification Trees**: Decision tree algorithms, random forests, and ensemble methods
- **Recommender Systems**: Collaborative filtering, content-based methods, and hybrid approaches
- **Statistical Learning Methods**: Additional statistical approaches and techniques

**[Applied Machine Learning (AML)](https://github.com/darinz/AML)**. This curriculum focuses on practical implementation and application of machine learning algorithms. AML is ideal for learners seeking hands-on experience and applied skills in building ML systems.

## Table of Contents

- [Overview](#overview)
- [Real-World Applications](#real-world-applications)
- [Capstone Project](#capstone-project)
- [Prerequisites](#prerequisites)
- [Curriculum](#curriculum)
  - [Math, Python, and NumPy Review](#math-python-and-numpy-review)
  - [I. Supervised Learning](#i-supervised-learning)
  - [II. Deep Learning](#ii-deep-learning)
  - [III. Generalization and Regularization](#iii-generalization-and-regularization)
  - [IV. Unsupervised Learning](#iv-unsupervised-learning)
  - [V. Reinforcement Learning](#v-reinforcement-learning-and-control)
  - [VI. Recent Advances and Applications](#vi-recent-advances-and-applications)
- [Problem Sets](#problem-sets)
- [Reference Materials](#reference-materials)
- [Development Guidelines](#development-guidelines)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)


## Overview

A comprehensive machine learning curriculum covering supervised learning, deep learning, reinforcement learning, and modern applications. This math-heavy approach emphasizes theoretical foundations with practical implementation.

### Learning Outcomes

Upon completing this curriculum, learners will be able to:
- Pursue ML roles (MLE, Data Scientist, Research Scientist)
- Implement ML models from scratch
- Apply deep learning to complex problems
- Evaluate and improve model performance
- Stay current with recent advances

### Learning Approach

(1) **Active Learning**: Rederive calculations, implement from scratch, build incrementally, test understanding

(2) **Retention**: Use spaced repetition, create examples, document learning, teach others, build portfolio

(3) **Implementation**: Use version control, experiment systematically, visualize everything, compare approaches

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

## Capstone Project

To apply and demonstrate your understanding of the concepts covered in this learning path, we recommend completing a comprehensive machine learning project. Our **[Project Guide](./PROJECT.md)** provides detailed guidance on:

- **Project Types**: Application, algorithmic, and theoretical project options
- **Project Examples**: Real-world implementations across different domains
- **Project Structure**: Step-by-step methodology from problem definition to final report
- **Deliverables**: Code, documentation, and presentation requirements
- **Evaluation Criteria**: Technical quality, originality, and communication standards
- **Getting Started**: Practical steps to begin your project journey
- **Conference Submissions**: Information on submitting your work to machine learning conferences

The project serves as a capstone experience that combines all the theoretical knowledge and practical skills you've developed throughout this curriculum. It's an opportunity to work on a real-world problem that interests you and build a portfolio piece that demonstrates your machine learning expertise.

**[View Project Guide →](./PROJECT.md)**

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

For comprehensive foundational materials, see the **[Math, Python, and NumPy Review](#math-python-and-numpy-review)** section in the curriculum below.

## Curriculum

### Math, Python, and NumPy Review

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
- **[Generalized Linear Models](./01_linear_models/03_generalized_linear_models/)** - Comprehensive exponential family distributions, systematic GLM construction, and unified framework for various response types with real-world applications

#### 02. [Generative Learning](./02_generative_learning/)
Generative approaches that model the joint distribution $`p(x,y)`$ through $`p(x|y)`$ and $`p(y)`$:
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

Deep learning extends supervised learning to non-linear models through artificial neural networks. This section covers the fundamental building blocks of modern neural networks, including network architectures, backpropagation for gradient computation, and efficient vectorized implementations for training on large datasets.


#### 04. [Deep Learning](./04_deep_learning/)
Comprehensive coverage of deep learning fundamentals including:
- **[Non-Linear Models](./04_deep_learning/01_non-linear_models.md)** - Moving beyond linear models, loss functions for regression and classification
- **[Neural Networks](./04_deep_learning/02_neural_networks.md)** - Single neurons to multi-layer architectures, biological inspiration
- **[Modern Neural Network Modules](./04_deep_learning/03_modules.md)** - Residual connections, layer normalization, convolutional layers
- **[Backpropagation](./04_deep_learning/04_backpropagation.md)** - Efficient gradient computation, chain rule applications, computational graphs
- **[Vectorization](./04_deep_learning/05_vectorization.md)** - Parallel computation, matrix operations, broadcasting for scalable implementations
- **Implementation Examples** - Complete Python implementations with practical examples and visualizations

### III. Generalization and Regularization

Understanding model generalization and preventing overfitting are fundamental to successful machine learning. This section explores the bias-variance tradeoff, the double descent phenomenon in modern models, and sample complexity bounds. It also covers regularization techniques, implicit regularization effects in neural networks, cross-validation for model selection, and Bayesian approaches to regularization.

#### 05. [Generalization](./05_generalization/)
**Enhanced comprehensive coverage of generalization theory and practice including:**
- **[Bias-Variance Tradeoff](./05_generalization/01_bias-variance_tradeoﬀ.md)** - **Enhanced** with detailed mathematical derivations, intuitive explanations, case studies (linear, polynomial, quadratic models), practical implications, and modern extensions connecting to deep learning
- **[Double Descent Phenomenon](./05_generalization/02_double_descent.md)** - **Enhanced** with comprehensive coverage of classical vs. modern regimes, model-wise and sample-wise double descent, interpolation threshold analysis, implicit regularization effects, and regularization strategies
- **[Sample Complexity Bounds](./05_generalization/03_complexity_bounds.md)** - **Enhanced** with detailed theoretical foundations including concentration inequalities, union bound applications, empirical vs. generalization error analysis, VC dimension visualization, and learning curve demonstrations
- **Enhanced Implementation Examples** - **Comprehensive Python implementations** with:
  - `bias_variance_decomposition_examples.py` - Modular design with type hints, multiple demonstrations, interactive visualizations, and educational output
  - `double_descent_examples.py` - Advanced demonstrations of modern ML phenomena with regularization effects and implicit regularization analysis
  - `complexity_bounds_examples.py` - Theoretical concepts with practical implementations including Monte Carlo simulations and visual proofs
- **Educational Features** - Progress indicators, comprehensive annotations, cross-references to markdown theory, and self-study friendly structure

#### 06. [Regularization, Model Selection, and Bayesian Methods](./06_regularization_model_selection/)
Comprehensive notes and code for:
- **[Regularization](./06_regularization_model_selection/01_regularization.md)**
- **[Model Selection & Cross-Validation](./06_regularization_model_selection/02_model_selection.md)**
- **Python Examples**: 
  - [regularization_examples.py](./06_regularization_model_selection/regularization_examples.py)
  - [model_selection_and_bayes_examples.py](./06_regularization_model_selection/model_selection_and_bayes_examples.py)

### IV. Unsupervised Learning

Unsupervised learning discovers hidden patterns and structures in data without predefined labels. This section covers clustering algorithms including k-means, expectation-maximization (EM) methods, dimensionality reduction techniques like PCA, ICA, and autoencoders for feature learning, and modern self-supervised learning approaches including foundation models for pretraining and adaptation.

#### 07. [Clustering and EM Algorithms](./07_clustering_em/)
**Enhanced comprehensive coverage of unsupervised learning algorithms for clustering and probabilistic modeling including:**
- **[K-means Clustering](./07_clustering_em/01_clustering.md)** - **Enhanced** with detailed mathematical derivations, geometric intuition, convergence analysis, initialization strategies (k-means++), and practical considerations
- **[EM for Mixture of Gaussians](./07_clustering_em/02_em_mixture_of_gaussians.md)** - **Enhanced** with comprehensive coverage of mixture models, EM algorithm derivation, Jensen's inequality, parameter estimation, and comparison with k-means
- **[General EM Algorithm](./07_clustering_em/03_general_em.md)** - **Enhanced** with detailed framework for latent variable models, ELBO derivation, coordinate ascent, KL divergence, and variational inference principles
- **[Variational Auto-Encoders](./07_clustering_em/04_variational_auto-encoder.md)** - **Enhanced** with comprehensive coverage of variational inference, generative models, mean field approximation, reparameterization trick, and encoder-decoder architecture
- **Enhanced Implementation Examples** - **Comprehensive Python implementations** with:
  - `kmeans_examples.py` - Multiple initialization methods, convergence monitoring, evaluation metrics, and visualization
  - `em_mog_examples.py` - Complete EM implementation with ELBO computation, initialization strategies, and comparison with k-means
  - `general_em_examples.py` - Flexible EM framework with Jensen's inequality demonstrations, KL divergence analysis, and multiple model support
  - `variational_auto_encoder_examples.py` - Full VAE implementation with PyTorch, generative model, encoder, ELBO computation, and sample generation
- **Educational Features** - Progress indicators, comprehensive annotations, cross-references to markdown theory, and self-study friendly structure

#### 08. [Dimensionality Reduction](./08_dimensionality_reduction/)
Comprehensive coverage of linear dimensionality reduction techniques including:
- **[Principal Component Analysis (PCA)](./08_dimensionality_reduction/01_pca.md)** – Finds new axes (principal components) that maximize variance and decorrelate the data. Useful for visualization, compression, and noise reduction. Includes [pca_examples.py](./08_dimensionality_reduction/pca_examples.py) for hands-on code.
- **[Independent Component Analysis (ICA)](./08_dimensionality_reduction/02_ica.md)** – Finds statistically independent components in the data, especially useful for separating mixed signals (e.g., the cocktail party problem). Includes [ica_examples.py](./08_dimensionality_reduction/ica_examples.py) for hands-on code.
- **Figures and diagrams** in [img/](./08_dimensionality_reduction/img/)

For installation and running instructions, see the [08_dimensionality_reduction/README.md](./08_dimensionality_reduction/README.md).

#### 09. [Self-Supervised Learning and Foundation Models](./09_self-supervised_learning/)
Comprehensive coverage of modern self-supervised learning and foundation models including:
- **[Pretraining Methods](./09_self-supervised_learning/01_pretraining.md)** – Self-supervised learning fundamentals, contrastive learning (SimCLR), and adaptation strategies (linear probe, finetuning) for computer vision applications
- **[Large Language Models (LLMs)](./09_self-supervised_learning/02_pretrain_llm.md)** – Language modeling, Transformer architecture, autoregressive text generation, and modern adaptation techniques (finetuning, zero-shot, in-context learning)
- **Implementation Examples** – Complete Python implementations with [pretraining_examples.py](./09_self-supervised_learning/pretraining_examples.py) for vision tasks and [pretrain_llm_examples.py](./09_self-supervised_learning/pretrain_llm_examples.py) for language modeling
- **Figures and diagrams** in [img/](./09_self-supervised_learning/img/)

For installation and running instructions, see the [09_self-supervised_learning/README.md](./09_self-supervised_learning/README.md).

### V. Reinforcement Learning and Control

Comprehensive coverage of reinforcement learning and optimal control, including foundational theory, advanced control methods, and policy gradient algorithms. This section provides both mathematical derivations and practical implementation examples.

**10. [Reinforcement Learning](./10_reinforcement_learning/)**

- **[Markov Decision Processes (MDP)](./10_reinforcement_learning/01_markov_decision_processes.md)** – Formal definition of MDPs, value functions, Bellman equations, dynamic programming, and solution methods (value iteration, policy iteration)
- **[Continuous State MDPs](./10_reinforcement_learning/02_continuous_state_mdp.md)** – Extension of MDPs to continuous state/action spaces, discretization, and function approximation
- **[Advanced Control: LQR, DDP, LQG](./10_reinforcement_learning/03_advanced_control.md)** – Linear Quadratic Regulation (LQR), Differential Dynamic Programming (DDP), Linear Quadratic Gaussian (LQG) control, and connections to reinforcement learning
- **[Policy Gradient Methods (REINFORCE)](./10_reinforcement_learning/04_policy_gradient.md)** – Direct policy optimization, score function estimator, REINFORCE algorithm, variance reduction, and practical considerations
- **Implementation Examples**:
  - [markov_decision_processes_examples.py](./10_reinforcement_learning/markov_decision_processes_examples.py)
  - [continuous_state_mdp_examples.py](./10_reinforcement_learning/continuous_state_mdp_examples.py)
  - [advanced_control_examples.py](./10_reinforcement_learning/advanced_control_examples.py)
  - [policy_gradient_examples.py](./10_reinforcement_learning/policy_gradient_examples.py)
- **Figures and diagrams** in [img/](./10_reinforcement_learning/img/)

For installation and running instructions, see the [10_reinforcement_learning/README.md](./10_reinforcement_learning/README.md).

### VI. Recent Advances and Applications

Recent advances in machine learning have revolutionized various fields and applications. This section covers transformer architectures and large language models, multi-armed bandits for sequential decision-making, reinforcement learning techniques for training language models, modern computer vision advances including vision transformers and self-supervised learning, and practical applications across natural language processing, computer vision, speech recognition, robotics, healthcare, and recommendation systems.

#### 11. [Multi-Armed Bandits](./11_bandits/)
Comprehensive coverage of multi-armed bandits, a fundamental framework for sequential decision-making under uncertainty. This section provides both theoretical foundations and practical implementations for balancing exploration and exploitation in dynamic environments.

**Core Topics:**
- **[Classical Multi-Armed Bandits](./11_bandits/01_classical_multi_armed_bandits.md)** - Problem formulation, epsilon-greedy, UCB, Thompson sampling, regret analysis, and theoretical guarantees
- **[Linear Bandits](./11_bandits/02_linear_bandits.md)** - Linear reward functions, LinUCB, linear Thompson sampling, feature engineering, and regret bounds
- **[Contextual Bandits](./11_bandits/03_contextual_bandits.md)** - Context-dependent rewards, contextual UCB, neural bandits, and real-world applications
- **[Best Arm Identification](./11_bandits/04_best_arm_identification.md)** - Pure exploration problems, successive elimination, racing algorithms, LUCB, and sample complexity analysis
- **[Applications and Use Cases](./11_bandits/05_applications_and_use_cases.md)** - Online advertising, recommendation systems, clinical trials, dynamic pricing, and practical implementations

**Applications**: Online advertising, recommendation systems, clinical trials, dynamic pricing, A/B testing, and resource allocation

For comprehensive coverage including theoretical foundations, algorithmic implementations, and real-world applications, see the [bandits/README.md](./11_bandits/README.md).

#### 12. [Transformers and Language Models](./12_llm/)
Comprehensive coverage of transformer architectures and large language models (LLMs) that have revolutionized natural language processing and artificial intelligence. This section provides both theoretical foundations and practical implementations.

**Core Topics:**
- **[Attention Mechanisms](./12_llm/01_attention_mechanism.md)** - Self-attention fundamentals, QKV framework, multi-head attention, scaled dot-product attention, and attention weight computation
- **[Transformer Architecture](./12_llm/02_transformer_architecture.md)** - Encoder-decoder structure, positional encoding, layer normalization, residual connections, and architectural variants (BERT, GPT, T5)
- **[Large Language Models](./12_llm/03_large_language_models.md)** - Model scaling laws, training techniques, pre-training objectives, and modern LLM architectures
- **[Training and Optimization](./12_llm/04_training_and_optimization.md)** - Optimization strategies, regularization techniques, evaluation metrics, and training stability
- **[Applications and Use Cases](./12_llm/05_applications_and_use_cases.md)** - NLP tasks, generative AI, multimodal applications, and real-world deployment

#### 13. [Computer Vision Advances](./13_vision/)
Comprehensive coverage of modern computer vision advances including vision transformers, self-supervised learning, contrastive learning, and foundation models. This section provides both theoretical foundations and practical implementations for cutting-edge computer vision techniques.

**Core Topics:**
- **[Vision Transformers (ViT)](./13_vision/01_vision_transformers.md)** - Transformer architecture adaptation for vision, patch embedding, self-attention mechanisms, and architectural variants (DeiT, Swin, ConvNeXt)
- **[Self-Supervised Learning](./13_vision/02_self_supervised_learning.md)** - Pretext tasks (inpainting, jigsaw, rotation, colorization), representation learning, and transfer learning strategies
- **[Contrastive Learning](./13_vision/03_contrastive_learning.md)** - Modern contrastive methods including SimCLR, MoCo, BYOL, and DINO with theoretical foundations and practical implementations
- **[Foundation Models for Vision](./13_vision/04_foundation_models.md)** - CLIP for vision-language understanding, SAM for universal segmentation, DALL-E for text-to-image generation, and zero-shot learning capabilities

**Applications**: Image classification, object detection, segmentation, image generation, medical imaging, and multi-modal vision-language tasks

#### 14. [Reinforcement Learning for Training Large Language Models](./14_rlhf/)
Comprehensive coverage of reinforcement learning techniques for training and aligning large language models (LLMs) with human preferences. This rapidly evolving field has revolutionized how we create more helpful, harmless, and honest AI systems.

**Core Topics:**
- **[Fundamentals of RL for Language Models](./14_rlhf/01_fundamentals_of_rl_for_language_models.md)** - Problem formulation, MDP framework, language generation specifics, and key challenges in RL for LLMs
- **[Human Feedback Collection](./14_rlhf/02_human_feedback_collection.md)** - Preference data collection strategies, annotation guidelines, quality control, and bias mitigation techniques
- **[Reward Modeling](./14_rlhf/03_reward_modeling.md)** - Reward function learning, preference learning objectives, reward model validation, and evaluation metrics
- **[Policy Optimization](./14_rlhf/04_policy_optimization.md)** - Policy gradient methods, PPO for language models, TRPO, and optimization techniques
- **[Alignment Techniques](./14_rlhf/05_alignment_techniques.md)** - Direct Preference Optimization (DPO), Constitutional AI, red teaming, and advanced alignment methods

**Applications**: Conversational AI, content generation, code generation, educational systems, and safety-aligned language models

**Reference Materials**: [Sutton and Barto RL Textbook](http://incompleteideas.net/book/the-book-2nd.html), [OpenAI's Spinning Up](https://spinningup.openai.com/en/latest/), [InstructGPT Paper](https://arxiv.org/abs/2203.02155), [DPO Paper](https://arxiv.org/abs/2305.18290), [Constitutional AI](https://arxiv.org/abs/2212.08073)

For comprehensive coverage including theoretical foundations, practical implementations, and ethical considerations, see the [rlhf/README.md](./14_rlhf/README.md).

#### 15. Practical Applications
- Natural Language Processing: [Neural-Machine-Translation](https://github.com/darinz/Neural-Machine-Translation)
- Computer Vision: [Computer Vision for Perception](https://github.com/darinz/AI-Robotic-IoT/blob/main/docs/05_Technical_Approach_CV.md)
- Speech Recognition: [Voice-Based Command & Interaction](https://github.com/darinz/AI-Robotic-IoT/blob/main/docs/04_Technical_Approach_NLP.md)
- Robotics and Control: [AI, Robotic, and IoT](https://github.com/darinz/AI-Robotic-IoT)
- Healthcare: [Text2Mol](https://github.com/darinz/DLH-Text2Mol)
- Recommendation Systems: [Movie Recommendation System](https://github.com/darinz/Movie-Rec-Sys), [Item-Based Collaborative Filtering](https://github.com/darinz/Movie-Recommender)

## Problem Sets

Comprehensive problem sets with solutions covering all curriculum topics. Each includes detailed problems, starter code, datasets, and complete solutions.

**[View Problem Sets](./problem-sets/)**

### Practice Problem Series

**Series 1: Foundational Concepts** `[practice-1/](./practice-1/)`
9 practice sets covering probability, linear algebra, ML fundamentals, neural networks, clustering, reinforcement learning, SVMs, ensemble methods, and advanced applications.

**Series 2: Advanced Machine Learning** `[practice-2/](./practice-2/)`
9 practice sets on neural networks, clustering, deep learning, ML theory, applied problems, and comprehensive exam preparation.

**Features:** Complete problems/solutions, interactive Jupyter notebooks, visualizations, progressive difficulty

### Problem Set Series

**Series 01: Foundational ML** `[01/](./problem-sets/01/)`
- PS1: Linear Regression & Locally Weighted Regression
- PS2: Classification & Spam Detection  
- PS3: Learning Theory & K-Means Clustering
- PS4: Dimensionality Reduction & Reinforcement Learning
- PS5: Comprehensive Review

**Series 02: Advanced ML & Deep Learning** `[02/](./problem-sets/02/)`
- PS1: Optimization & Regularization
- PS2: Cross-Validation & Lasso Regression
- PS3: Gradient Descent & Stochastic Methods
- PS4: Kernel Methods & Neural Networks
- PS5: Neural Network Theory & Backpropagation
- PS6: Dimensionality Reduction & Matrix Decompositions

**Series 03: Modern ML Applications** `[03/](./problem-sets/03/)`
- PS1: Mathematical Foundations
- PS2: ML Fundamentals & Bias-Variance Tradeoff
- PS3: Optimization, Convexity & Regularization
- PS4: Kernel Methods & Neural Networks with PyTorch
- PS5: Computer Vision Applications & CNN Implementation

**Features:** Complete solutions, multiple languages (Python/MATLAB/PyTorch), real datasets, progressive difficulty

## Reference Materials

### Core Course Materials
- [CS229 Machine Learning Course](https://cs229.stanford.edu/) - Stanford University
- [CS229 Lecture Notes](./reference/CS229_Lecture-Notes.pdf) - Comprehensive course notes
- [CS229 Decision Trees Notes](./reference/CS229_Decision-Trees-Notes.pdf) - Decision tree algorithms and theory
- [CSE446: Machine Learning Course](https://courses.cs.washington.edu/courses/cse446/) - University of Washington

### Mathematical Foundations
- [Exponential Family Materials](./01_linear_models/03_generalized_linear_models/exponential_family/) - Comprehensive exponential family reference materials from MIT, Princeton, Berkeley, Columbia, and Purdue
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
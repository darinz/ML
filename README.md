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

A comprehensive learning resource providing a broad introduction to machine learning and statistical pattern recognition. This collection covers fundamental concepts, modern algorithms, and practical applications in the field of machine learning.

> **Note**: This is a math-heavy version focused on theoretical foundations and mathematical derivations. While we plan to add applied machine learning materials in the future (which will be less math-intensive), understanding the mathematical principles behind algorithms is invaluable for research and comprehending academic papers. This foundation enables deeper insights into model behavior and more effective algorithm selection and tuning.

## Complementary Learning Resources

For additional practical machine learning topics not covered in this theoretical curriculum, explore **[Practical Statistical Learning (PSL)](https://github.com/darinz/PSL)**. This complementary resource covers:

- **Regression and Classification Trees**: Decision tree algorithms, CART, random forests, and ensemble methods
- **Recommender Systems**: Collaborative filtering, content-based methods, and hybrid approaches
- **Statistical Learning Methods**: Additional statistical approaches and techniques

PSL provides a more applied, hands-on approach that complements the theoretical foundations covered in this repository.

**[Applied Machine Learning (AML)](https://github.com/darinz/AML)**. This curriculum focuses on practical implementation and application of machine learning algorithms, covering end-to-end workflows, real-world projects, and deployment strategies. AML is ideal for learners seeking hands-on experience and applied skills in building and deploying ML systems.

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
- [Reference Materials](#reference-materials)
- [Development Guidelines](#development-guidelines)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)


## Overview

This learning resource serves as a comprehensive guide for machine learning, covering topics from basic supervised learning to advanced deep learning and reinforcement learning techniques. The material is designed to provide both theoretical understanding and practical implementation skills for students, researchers, and practitioners in the field.

### Learning Outcomes

Upon completing this curriculum, learners will be able to:
- Develop sufficient skills and knowledge to pursue research or professional roles such as Machine Learning Engineer (MLE), Data Scientist, Research Scientist, or similar ML-focused positions
- Understand fundamental machine learning concepts and algorithms
- Implement and optimize various ML models from scratch
- Apply deep learning techniques to complex problems
- Evaluate and improve model performance effectively
- Stay current with recent advances in the field
- Apply machine learning to real-world applications across domains

### Learning Approach

The curriculum follows a progressive learning path that builds from fundamental concepts to advanced applications. Each topic includes theoretical foundations, mathematical derivations, and practical implementation examples. The material emphasizes both understanding the underlying principles and developing hands-on coding skills.

#### Active Learning Strategies
- **Rederive all calculations**: Don't just read the math—work through every derivation step-by-step on paper or in a notebook
- **Implement from scratch**: Retype code examples into your own Jupyter notebook or Python environment, then modify and experiment
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

**[10. Reinforcement Learning](./10_reinforcement_learning/)**

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

#### 11. Transformers and Language Models
- [CS224N Self-Attention and Transformers Notes](./reference/CS224N_Self-Attention-Transformers-2023_Draft.pdf)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- Transformer: [original transformer paper](https://arxiv.org/abs/1706.03762)
- LLMs: [Wikipedia on LLMs](https://en.wikipedia.org/wiki/Large_language_model)

#### 12. Multi-Armed Bandits
- Multi-armed bandits: [Bandit Algorithms textbook](https://tor-lattimore.com/downloads/book/book.pdf), [informal notes](https://courses.cs.washington.edu/courses/cse541/24sp/resources/lecture_notes.pdf)
- Linear bandits: [linear bandits paper](https://papers.nips.cc/paper_files/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html), [generalized linear bandits paper](https://papers.nips.cc/paper_files/paper/2010/hash/c2626d850c80ea07e7511bbae4c76f4b-Abstract.html), [pure exploration/BAI paper](https://arxiv.org/abs/1409.6110)
- Contextual bandits: [contextual bandits survey paper](https://www.ambujtewari.com/research/tewari17ads.pdf)

#### 13. Reinforcement Learning for training Large Language Models
- Reinforcement Learning Textbook: [Sutton and Barto](http://incompleteideas.net/book/the-book-2nd.html)
- Deep Reinforcement Learning: [OpenAI's spinning up](https://spinningup.openai.com/en/latest/)
- Reinforcement Learning for LLMs: [KL-control](https://arxiv.org/abs/1611.02796), [reward model](https://arxiv.org/abs/1706.03741), [InstructGPT paper (ChatGPT)](https://arxiv.org/abs/2203.02155), [recent DeepSeek R1 paper](https://arxiv.org/abs/2501.12948)

#### 14. Computer Vision Advances
- Vision Transformers (ViT): [original ViT paper](https://arxiv.org/abs/2010.11929), [ViT tutorial](https://pytorch.org/hub/pytorch_vision_vit/), [ViT implementation](https://github.com/lucidrains/vit-pytorch)
- Self-Supervised Learning in Vision: [survey paper](https://arxiv.org/abs/1902.06162), [BYOL paper](https://arxiv.org/abs/2006.07733), [DINO paper](https://arxiv.org/abs/2104.14294)
- Contrastive Learning (SimCLR, MoCo): [SimCLR paper](https://arxiv.org/abs/2002.05709), [MoCo paper](https://arxiv.org/abs/1911.05722), [MoCo v2](https://arxiv.org/abs/2003.04297), [MoCo v3](https://arxiv.org/abs/2104.02057)
- Foundation Models for Vision: [CLIP paper](https://arxiv.org/abs/2103.00020), [DALL-E paper](https://arxiv.org/abs/2102.12092), [SAM paper](https://arxiv.org/abs/2304.02643), [Segment Anything](https://github.com/facebookresearch/segment-anything)

#### 15. Practical Applications
- Natural Language Processing: [Neural-Machine-Translation](https://github.com/darinz/Neural-Machine-Translation)
- Computer Vision: [Computer Vision for Perception](https://github.com/darinz/AI-Robotic-IoT/blob/main/docs/05_Technical_Approach_CV.md)
- Speech Recognition: [Voice-Based Command & Interaction](https://github.com/darinz/AI-Robotic-IoT/blob/main/docs/04_Technical_Approach_NLP.md)
- Robotics and Control: [AI, Robotic, and IoT](https://github.com/darinz/AI-Robotic-IoT)
- Healthcare: [Text2Mol](https://github.com/darinz/DLH-Text2Mol)
- Recommendation Systems: [Movie Recommendation System](https://github.com/darinz/Movie-Rec-Sys), [Item-Based Collaborative Filtering](https://github.com/darinz/Movie-Recommender)

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
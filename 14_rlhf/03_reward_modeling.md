# Reward Modeling

This guide provides a comprehensive overview of reward modeling for reinforcement learning from human feedback (RLHF) systems. We'll explore the mathematical foundations, training objectives, validation methods, and practical implementation considerations for learning reward functions from human preferences.

## From Data Collection to Reward Modeling

We've now explored **human feedback collection** - the systematic process of gathering, structuring, and validating human preferences and judgments about model outputs. We've seen how different types of feedback (binary preferences, rankings, ratings, natural language explanations) capture different aspects of human judgment, how to design effective annotation guidelines and quality control measures, and how to mitigate biases and ensure diverse perspectives in the feedback collection process.

However, while collecting high-quality human feedback is essential, **raw feedback data** is not directly usable for training language models. Consider having thousands of preference judgments - we need a way to convert these relative preferences into a reward function that can guide policy optimization and provide consistent feedback during training.

This motivates our exploration of **reward modeling** - the process of learning a function that maps prompt-response pairs to scalar reward values, capturing human preferences and judgments. We'll see how to design neural network architectures that can learn from preference data, how to formulate training objectives that capture the relative nature of human preferences, how to validate and calibrate reward models, and how to address challenges like reward hacking and distributional shift.

The transition from human feedback collection to reward modeling represents the bridge from raw preference data to learnable reward signals - taking our understanding of how to collect human feedback and applying it to the challenge of building reward functions that can guide effective policy optimization.

In this section, we'll explore reward modeling, understanding how to convert human preferences into reward functions that enable successful RLHF training.

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Reward Model Architecture](#reward-model-architecture)
- [Training Objectives](#training-objectives)
- [Loss Functions](#loss-functions)
- [Validation and Evaluation](#validation-and-evaluation)
- [Implementation Examples](#implementation-examples)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)

## Overview

Reward modeling is the process of learning a function that maps prompt-response pairs to scalar reward values, capturing human preferences and judgments. Unlike traditional supervised learning where we have ground truth labels, reward modeling learns from relative preferences and rankings provided by human annotators.

### Key Concepts

**1. Preference Learning**: Learn from relative preferences rather than absolute labels
**2. Reward Function**: $`R_\phi(x, y)`$ that assigns scalar rewards to prompt-response pairs
**3. Human Alignment**: Capture human values and preferences in the reward function
**4. Generalization**: Learn to predict preferences for unseen prompt-response pairs

### Problem Formulation

Given a dataset of human preferences:
```math
\mathcal{D} = \{(x_i, y_{i,w}, y_{i,l})\}_{i=1}^N
```

Where:
- $`x_i`$: Input prompt/context
- $`y_{i,w}`$: Preferred (winning) response
- $`y_{i,l}`$: Less preferred (losing) response

Goal: Learn reward function $`R_\phi(x, y)`$ such that:
```math
R_\phi(x, y_w) > R_\phi(x, y_l)
```

## Mathematical Foundations

### Preference Learning Framework

**Assumption**: Human preferences follow a Bradley-Terry model:
```math
P(y_w \succ y_l | x) = \frac{\exp(R_\phi(x, y_w))}{\exp(R_\phi(x, y_w)) + \exp(R_\phi(x, y_l))} = \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))
```

Where:
- $`\succ`$: "is preferred to"
- $`\sigma`$: Sigmoid function
- $`R_\phi(x, y)`$: Learned reward function

### Maximum Likelihood Estimation

**Objective**: Maximize likelihood of observed preferences:
```math
\mathcal{L}(\phi) = \sum_{i=1}^N \log P(y_{i,w} \succ y_{i,l} | x_i)
```

**Log-likelihood**:
```math
\mathcal{L}(\phi) = \sum_{i=1}^N \log \sigma(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))
```

### Gradient-Based Optimization

**Gradient**:
```math
\nabla_\phi \mathcal{L}(\phi) = \sum_{i=1}^N (1 - \sigma(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))) \nabla_\phi(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))
```

## Reward Model Architecture

### Basic Architecture

**Standard Reward Model**:
```math
R_\phi(x, y) = f_\phi(\text{encode}(x, y))
```

Where:
- $`x`$: Input prompt/context
- $`y`$: Generated response
- $`f_\phi`$: Neural network with parameters $`\phi`$
- $`\text{encode}`$: Encoder for prompt-response pairs

### Encoder Architectures

**1. Concatenation-based**: Concatenate prompt and response tokens and encode together
**2. Separate Encoders**: Encode prompt and response separately, then fuse representations
**3. Cross-Attention Architecture**: Use cross-attention between prompt and response

**Implementation:** See `reward_model.py` for complete reward model architectures:
- `RewardModel` - Basic concatenation-based reward model
- `SeparateEncoderRewardModel` - Separate encoders for prompt and response
- `MultiObjectiveRewardModel` - Multi-objective reward modeling
- `predict_reward()` - Reward prediction utilities

### Advanced Architectures

**1. Multi-Objective Reward Model**: Separate heads for different objectives (helpfulness, harmlessness, honesty)
**2. Hierarchical Reward Model**: Level-specific reward heads for different aspects of responses

**Implementation:** See `reward_model.py` for advanced architectures:
- `MultiObjectiveRewardModel` - Multi-objective reward modeling
- Support for multiple reward heads
- Weighted combination of objectives

## Training Objectives

### Preference Learning Loss

**Standard Preference Loss**:
```math
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))
```

**Implementation:** See `reward_model.py` for training objectives:
- `RewardModelTrainer` - Complete training pipeline
- `preference_loss()` - Standard preference learning loss
- `ranking_loss()` - Multi-response ranking loss
- `train_step()` - Training step implementation

### Ranking Loss

**Multi-Response Ranking**:
```math
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_1, \ldots, y_k) \sim \mathcal{D}} \sum_{i=1}^{k-1} \log \sigma(R_\phi(x, y_i) - R_\phi(x, y_{i+1}))
```

**Implementation:** See `reward_model.py` for ranking loss:
- `ranking_loss()` - Multi-response ranking loss
- Support for ordered preference lists
- Pairwise ranking computation

### Contrastive Loss

**Contrastive Learning Approach**:
```math
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \frac{\exp(R_\phi(x, y_w)/\tau)}{\exp(R_\phi(x, y_w)/\tau) + \exp(R_\phi(x, y_l)/\tau)}
```

Where $`\tau`$ is the temperature parameter.

**Implementation:** See `reward_model.py` for contrastive learning:
- Temperature-scaled contrastive loss
- Support for different temperature parameters
- Contrastive training utilities

## Loss Functions

### Regularization Techniques

**1. L2 Regularization**: Add L2 penalty to model parameters
**2. Dropout Regularization**: Apply dropout to prevent overfitting
**3. Label Smoothing**: Smooth preference labels for better generalization

**Implementation:** See `reward_model.py` for regularization:
- L2 regularization support
- Dropout layers in model architectures
- Label smoothing utilities

### Advanced Loss Functions

**1. Focal Loss for Hard Examples**: Focus on difficult preference pairs
**2. Triplet Loss**: Learn relative distances between responses

**Implementation:** See `reward_model.py` for advanced loss functions:
- Focal loss for hard examples
- Triplet loss implementation
- Custom loss function support

## Validation and Evaluation

### Evaluation Metrics

**1. Preference Accuracy**: Percentage of correctly predicted preferences
**2. Ranking Correlation**: Spearman correlation with human rankings
**3. Calibration Metrics**: Expected calibration error

**Implementation:** See `evaluation.py` for comprehensive evaluation:
- `RewardModelEvaluator` - Complete evaluation pipeline
- `preference_accuracy()` - Preference prediction accuracy
- `ranking_correlation()` - Ranking correlation metrics
- `calibration_error()` - Calibration evaluation

### Robustness Evaluation

**1. Out-of-Distribution Testing**: Evaluate on unseen data distributions
**2. Adversarial Testing**: Test robustness against adversarial examples

**Implementation:** See `evaluation.py` for robustness evaluation:
- Out-of-distribution testing utilities
- Adversarial robustness evaluation
- Robustness gap computation

## Implementation Examples

### Complete Reward Model Training

**Implementation:** See `reward_model.py` for complete training pipeline:
- `RewardModelTrainer` - Complete training pipeline
- `train_step()` - Training step implementation
- `evaluate()` - Model evaluation utilities
- `save_model()` and `load_model()` - Model persistence

### Reward Model Inference

**Implementation:** See `reward_model.py` for inference utilities:
- `RewardModelInference` - Complete inference pipeline
- `predict_reward()` - Single prediction utilities
- `rank_responses()` - Response ranking capabilities
- `batch_predict()` - Batch prediction utilities

## Advanced Techniques

### Multi-Task Reward Learning

**Implementation:** See `reward_model.py` for multi-task learning:
- `MultiObjectiveRewardModel` - Multi-task reward modeling
- Support for multiple objectives
- Weighted combination of tasks
- Task-specific loss functions

### Uncertainty-Aware Reward Models

**Implementation:** See `reward_model.py` for uncertainty modeling:
- Uncertainty quantification in reward predictions
- Confidence-aware training
- Uncertainty-based sample selection

### Calibrated Reward Models

**Implementation:** See `reward_model.py` for calibration:
- Temperature scaling for calibration
- Calibration loss functions
- Calibrated prediction utilities

## Best Practices

### 1. Data Quality

- **Diverse Training Data**: Ensure coverage across different topics, styles, and difficulty levels
- **Quality Control**: Use multiple annotators and agreement monitoring
- **Bias Mitigation**: Address systematic biases in annotation
- **Validation Split**: Maintain separate validation set for model selection

### 2. Model Architecture

- **Appropriate Encoder**: Choose encoder architecture based on task requirements
- **Regularization**: Use dropout, L2 regularization, and other techniques
- **Architecture Search**: Experiment with different architectures for optimal performance
- **Scalability**: Consider computational requirements for large-scale deployment

### 3. Training Strategy

- **Learning Rate Scheduling**: Use appropriate learning rate schedules
- **Early Stopping**: Monitor validation metrics to prevent overfitting
- **Gradient Clipping**: Prevent gradient explosion in large models
- **Mixed Precision**: Use mixed precision training for efficiency

### 4. Evaluation

- **Multiple Metrics**: Use preference accuracy, ranking correlation, and calibration
- **Robustness Testing**: Evaluate on out-of-distribution and adversarial examples
- **Human Evaluation**: Validate automated metrics with human judgments
- **Continuous Monitoring**: Monitor model performance in production

### 5. Deployment

- **Model Compression**: Consider quantization and distillation for deployment
- **Caching**: Cache reward predictions for efficiency
- **Monitoring**: Implement monitoring for model drift and performance
- **Versioning**: Maintain model versions and rollback capabilities

## Summary

Reward modeling is a critical component of RLHF systems that enables learning from human preferences. Key aspects include:

1. **Mathematical Foundation**: Preference learning with Bradley-Terry model
2. **Architecture Design**: Various encoder architectures for different requirements
3. **Training Objectives**: Preference loss, ranking loss, and contrastive learning
4. **Validation**: Comprehensive evaluation with multiple metrics
5. **Advanced Techniques**: Multi-task learning, uncertainty quantification, and calibration
6. **Best Practices**: Data quality, model architecture, training strategy, and deployment

Effective reward modeling enables the training of language models that better align with human values and preferences, ultimately leading to more useful, safe, and honest AI systems.

---

**Note**: This guide provides the theoretical and practical foundations for reward modeling. For specific implementation details and advanced techniques, refer to the implementation examples and external resources referenced in the main README.

## From Reward Functions to Policy Optimization

We've now explored **reward modeling** - the process of learning a function that maps prompt-response pairs to scalar reward values, capturing human preferences and judgments. We've seen how to design neural network architectures that can learn from preference data, how to formulate training objectives that capture the relative nature of human preferences, how to validate and calibrate reward models, and how to address challenges like reward hacking and distributional shift.

However, while having a well-trained reward model is crucial, **the reward function alone** cannot improve language model behavior. Consider having a perfect reward model that can evaluate any response - we still need an optimization algorithm that can update the language model policy to maximize expected reward while maintaining language quality and preventing catastrophic forgetting.

This motivates our exploration of **policy optimization** - the core component of RLHF that updates the language model policy to maximize expected reward from human feedback. We'll see how policy gradient methods like REINFORCE and actor-critic approaches work for language models, how trust region methods like PPO and TRPO ensure stable policy updates, how to handle the unique challenges of language generation (sequential decisions, sparse rewards, high dimensionality), and how to balance reward maximization with maintaining language quality.

The transition from reward modeling to policy optimization represents the bridge from evaluation to improvement - taking our understanding of how to evaluate responses with reward functions and applying it to the challenge of optimizing language model policies to produce better responses.

In the next section, we'll explore policy optimization, understanding how to update language model policies to maximize expected reward while maintaining language quality.

---

**Previous: [Human Feedback Collection](02_human_feedback_collection.md)** - Learn how to collect and structure human preferences for RLHF training.

**Next: [Policy Optimization](04_policy_optimization.md)** - Learn how to optimize language model policies using reward functions. 
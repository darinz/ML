# Reward Modeling

This guide provides a comprehensive overview of reward modeling for reinforcement learning from human feedback (RLHF) systems. We'll explore the mathematical foundations, training objectives, validation methods, and practical implementation considerations for learning reward functions from human preferences.

### The Big Picture: What is Reward Modeling?

**The Reward Modeling Problem:**
Imagine you have thousands of human judgments like "Response A is better than Response B" or "Response A gets 4/5 stars." How do you turn these relative preferences into a function that can evaluate any response and give it a score? This is what reward modeling does.

**The Intuitive Analogy:**
Think of reward modeling like training a food critic. You show the critic many pairs of dishes and ask "Which one is better?" Over time, the critic learns to evaluate any dish and give it a score. Similarly, a reward model learns to evaluate any prompt-response pair and assign a reward score.

**Why Reward Modeling Matters:**
- **Converts preferences to scores**: Turns relative judgments into absolute scores
- **Enables optimization**: Provides the signal needed to improve language models
- **Scales human feedback**: One reward model can evaluate millions of responses
- **Maintains consistency**: Provides consistent evaluation across different responses

### The Reward Modeling Pipeline

**Step 1: Collect Preferences**
- Humans compare responses: "A is better than B"
- Gather thousands of such comparisons
- Ensure diverse, high-quality preferences

**Step 2: Design Reward Model**
- Choose architecture (concatenation, separate encoders, etc.)
- Define how to encode prompt-response pairs
- Design the scoring function

**Step 3: Train the Model**
- Use preference data to train the reward model
- Optimize to predict human preferences correctly
- Validate on held-out data

**Step 4: Deploy and Monitor**
- Use trained model to score new responses
- Monitor performance and retrain as needed
- Ensure reward model aligns with human values

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

### Understanding Reward Modeling Intuitively

**The Learning Problem:**
Traditional supervised learning: "This response gets a score of 4.2"
Reward modeling: "Response A is better than Response B"

**The Challenge:**
How do you learn to assign scores when you only have relative preferences? The key insight is that if A is preferred to B, then the reward model should assign a higher score to A than to B.

**The Learning Process:**
1. **Show preference pairs**: "A is better than B"
2. **Model makes predictions**: Assigns scores to both A and B
3. **Compare predictions**: Check if model correctly predicts A > B
4. **Update model**: Adjust parameters to make correct predictions
5. **Repeat**: Keep learning from more preference pairs

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

**Intuitive Understanding:**
This says: "For every preference pair, the preferred response should get a higher reward than the less preferred response."

## Mathematical Foundations

### Understanding the Mathematical Framework

**The Core Idea:**
We want to learn a function that assigns higher scores to responses that humans prefer. But we only have relative preferences, not absolute scores.

**The Solution:**
Use a probabilistic model that says "the probability that A is preferred to B depends on the difference in their reward scores."

### Preference Learning Framework

**Assumption**: Human preferences follow a Bradley-Terry model:
```math
P(y_w \succ y_l | x) = \frac{\exp(R_\phi(x, y_w))}{\exp(R_\phi(x, y_w)) + \exp(R_\phi(x, y_l))} = \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))
```

Where:
- $`\succ`$: "is preferred to"
- $`\sigma`$: Sigmoid function
- $`R_\phi(x, y)`$: Learned reward function

**Intuitive Understanding:**
This formula says: "The probability that humans prefer response A over response B is a function of how much higher A's reward score is than B's reward score."

**Why This Makes Sense:**
- **Large difference**: If A has much higher reward than B, probability approaches 1
- **Small difference**: If A and B have similar rewards, probability approaches 0.5
- **Negative difference**: If B has higher reward than A, probability approaches 0

**The Sigmoid Function:**
```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

This function:
- Maps any real number to (0, 1)
- Is symmetric around 0
- Approaches 1 for large positive values
- Approaches 0 for large negative values

### Maximum Likelihood Estimation

**Objective**: Maximize likelihood of observed preferences:
```math
\mathcal{L}(\phi) = \sum_{i=1}^N \log P(y_{i,w} \succ y_{i,l} | x_i)
```

**Intuitive Understanding:**
We want to find reward function parameters that make the observed preferences as likely as possible.

**Log-likelihood**:
```math
\mathcal{L}(\phi) = \sum_{i=1}^N \log \sigma(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))
```

**Why Log-Likelihood:**
- **Numerical stability**: Prevents underflow with small probabilities
- **Computational efficiency**: Sums are easier than products
- **Optimization**: Gradients are easier to compute

### Gradient-Based Optimization

**Gradient**:
```math
\nabla_\phi \mathcal{L}(\phi) = \sum_{i=1}^N (1 - \sigma(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))) \nabla_\phi(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))
```

**Intuitive Understanding:**
The gradient tells us how to change the reward function parameters to make the observed preferences more likely.

**Breaking Down the Gradient:**
1. **$`(1 - \sigma(...))`$**: How wrong our current prediction is
2. **$`\nabla_\phi(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))`$**: How changing parameters affects the reward difference
3. **Product**: How much to adjust parameters

## Reward Model Architecture

### Understanding Architecture Choices

**The Architecture Problem:**
How do you design a neural network that takes a prompt and response as input and outputs a single reward score?

**Key Considerations:**
- **Input representation**: How to encode prompt-response pairs
- **Model capacity**: How complex the model should be
- **Computational efficiency**: How fast inference needs to be
- **Interpretability**: How to understand what the model learned

### Basic Architecture

**Standard Reward Model**:
```math
R_\phi(x, y) = f_\phi(\text{encode}(x, y))
```

Where:
- $`x`$: Input prompt/context
- $`y`$: Generated response
- $`f_\phi`$: Neural network with parameters $`\phi``
- $`\text{encode}`$: Encoder for prompt-response pairs

**Intuitive Understanding:**
1. **Encode**: Convert prompt and response into a numerical representation
2. **Process**: Pass through neural network to extract features
3. **Score**: Output a single reward score

### Encoder Architectures

**1. Concatenation-based**: Concatenate prompt and response tokens and encode together
- **Intuitive**: Treat prompt and response as one long sequence
- **Pros**: Simple, captures interactions between prompt and response
- **Cons**: May not scale well to very long sequences

**2. Separate Encoders**: Encode prompt and response separately, then fuse representations
- **Intuitive**: Encode each part separately, then combine
- **Pros**: More flexible, can handle different lengths
- **Cons**: May miss some prompt-response interactions

**3. Cross-Attention Architecture**: Use cross-attention between prompt and response
- **Intuitive**: Let prompt and response "attend" to each other
- **Pros**: Captures rich interactions, state-of-the-art performance
- **Cons**: More complex, higher computational cost

**Implementation:** See `code/reward_model.py` for complete reward model architectures:
- `RewardModel` - Basic concatenation-based reward model
- `SeparateEncoderRewardModel` - Separate encoders for prompt and response
- `MultiObjectiveRewardModel` - Multi-objective reward modeling
- `predict_reward()` - Reward prediction utilities

### Advanced Architectures

**1. Multi-Objective Reward Model**: Separate heads for different objectives (helpfulness, harmlessness, honesty)
- **Intuitive**: Like having separate critics for different aspects
- **Benefits**: Can balance multiple objectives, more interpretable
- **Challenges**: Need to weight different objectives

**2. Hierarchical Reward Model**: Level-specific reward heads for different aspects of responses
- **Intuitive**: Evaluate different levels (sentence, paragraph, overall)
- **Benefits**: More fine-grained evaluation
- **Challenges**: More complex training and inference

**Implementation:** See `code/reward_model.py` for advanced architectures:
- `MultiObjectiveRewardModel` - Multi-objective reward modeling
- Support for multiple reward heads
- Weighted combination of objectives

## Training Objectives

### Understanding Training Objectives

**The Training Challenge:**
How do you formulate a loss function that encourages the reward model to correctly predict human preferences?

**Key Principles:**
- **Preference consistency**: Preferred responses should get higher scores
- **Margin maximization**: Create clear separation between preferred and non-preferred
- **Regularization**: Prevent overfitting to training data

### Preference Learning Loss

**Standard Preference Loss**:
```math
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))
```

**Intuitive Understanding:**
This loss encourages the model to assign higher scores to preferred responses. The negative log-likelihood means we want to maximize the probability of correct predictions.

**Why This Works:**
- **Correct prediction**: If $`R_\phi(x, y_w) > R_\phi(x, y_l)`$, loss is small
- **Incorrect prediction**: If $`R_\phi(x, y_w) < R_\phi(x, y_l)`$, loss is large
- **Uncertain prediction**: If scores are close, loss is moderate

**Implementation:** See `code/reward_model.py` for training objectives:
- `RewardModelTrainer` - Complete training pipeline
- `preference_loss()` - Standard preference learning loss
- `ranking_loss()` - Multi-response ranking loss
- `train_step()` - Training step implementation

### Ranking Loss

**Multi-Response Ranking**:
```math
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_1, \ldots, y_k) \sim \mathcal{D}} \sum_{i=1}^{k-1} \log \sigma(R_\phi(x, y_i) - R_\phi(x, y_{i+1}))
```

**Intuitive Understanding:**
This loss handles cases where we have more than two responses to rank. It ensures that each response gets a higher score than the next one in the ranking.

**When to Use:**
- **Multiple responses**: Have more than two responses to compare
- **Fine-grained ranking**: Need to preserve order of multiple responses
- **Efficient training**: Can learn from multiple comparisons at once

**Implementation:** See `code/reward_model.py` for ranking loss:
- `ranking_loss()` - Multi-response ranking loss
- Support for ordered preference lists
- Pairwise ranking computation

### Contrastive Loss

**Contrastive Learning Approach**:
```math
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \frac{\exp(R_\phi(x, y_w)/\tau)}{\exp(R_\phi(x, y_w)/\tau) + \exp(R_\phi(x, y_l)/\tau)}
```

Where $`\tau`$ is the temperature parameter.

**Intuitive Understanding:**
This loss treats preference learning as a contrastive learning problem. It pushes preferred responses away from non-preferred responses in the reward space.

**Temperature Parameter:**
- **Low temperature** ($`\tau \approx 0.1`$): Sharp distinction between preferred and non-preferred
- **High temperature** ($`\tau \approx 1.0`$): Softer distinction, more uncertainty

**Implementation:** See `code/reward_model.py` for contrastive learning:
- Temperature-scaled contrastive loss
- Support for different temperature parameters
- Contrastive training utilities

## Loss Functions

### Understanding Loss Function Design

**The Loss Function Challenge:**
How do you design a loss function that not only learns preferences but also generalizes well and prevents overfitting?

**Key Considerations:**
- **Preference learning**: Correctly predict human preferences
- **Generalization**: Work well on unseen data
- **Regularization**: Prevent overfitting
- **Robustness**: Handle noisy or inconsistent preferences

### Regularization Techniques

**1. L2 Regularization**: Add L2 penalty to model parameters
- **Intuitive**: Penalize large parameter values
- **Effect**: Prevents overfitting, encourages simpler models
- **Implementation**: Add $`\lambda \|\phi\|_2^2`$ to loss function

**2. Dropout Regularization**: Apply dropout to prevent overfitting
- **Intuitive**: Randomly disable neurons during training
- **Effect**: Forces model to be robust to missing information
- **Implementation**: Add dropout layers in neural network

**3. Label Smoothing**: Smooth preference labels for better generalization
- **Intuitive**: Add uncertainty to preference labels
- **Effect**: Prevents overconfidence, improves generalization
- **Implementation**: Convert hard preferences to soft probabilities

**Implementation:** See `code/reward_model.py` for regularization:
- L2 regularization support
- Dropout layers in model architectures
- Label smoothing utilities

### Advanced Loss Functions

**1. Focal Loss for Hard Examples**: Focus on difficult preference pairs
- **Intuitive**: Pay more attention to cases where model is wrong
- **Benefit**: Better learning from challenging examples
- **Implementation**: Weight loss by prediction confidence

**2. Triplet Loss**: Learn relative distances between responses
- **Intuitive**: Learn that A is closer to B than to C
- **Benefit**: Better understanding of response relationships
- **Implementation**: Minimize distance between similar responses, maximize between different

**Implementation:** See `code/reward_model.py` for advanced loss functions:
- Focal loss for hard examples
- Triplet loss implementation
- Custom loss function support

## Validation and Evaluation

### Understanding Evaluation Metrics

**The Evaluation Challenge:**
How do you know if your reward model is actually learning human preferences correctly?

**Key Questions:**
- **Accuracy**: Does the model predict preferences correctly?
- **Calibration**: Are the predicted probabilities well-calibrated?
- **Robustness**: Does the model work on different types of data?
- **Generalization**: Does it work on unseen prompts and responses?

### Evaluation Metrics

**1. Preference Accuracy**: Percentage of correctly predicted preferences
- **Intuitive**: How often does the model get the preference right?
- **Calculation**: Count correct predictions / total predictions
- **Limitation**: Doesn't capture confidence or calibration

**2. Ranking Correlation**: Spearman correlation with human rankings
- **Intuitive**: How well do model rankings match human rankings?
- **Calculation**: Compute correlation between model and human rankings
- **Benefit**: Captures ordinal relationships

**3. Calibration Metrics**: Expected calibration error
- **Intuitive**: Do predicted probabilities match actual frequencies?
- **Calculation**: Compare predicted vs. actual preference probabilities
- **Benefit**: Important for uncertainty quantification

**Implementation:** See `code/evaluation.py` for comprehensive evaluation:
- `RewardModelEvaluator` - Complete evaluation pipeline
- `preference_accuracy()` - Preference prediction accuracy
- `ranking_correlation()` - Ranking correlation metrics
- `calibration_error()` - Calibration evaluation

### Robustness Evaluation

**1. Out-of-Distribution Testing**: Evaluate on unseen data distributions
- **Intuitive**: Test on data that's different from training data
- **Purpose**: Check if model generalizes to new domains
- **Methods**: Different topics, styles, difficulty levels

**2. Adversarial Testing**: Test robustness against adversarial examples
- **Intuitive**: Test with deliberately crafted difficult examples
- **Purpose**: Check if model is robust to edge cases
- **Methods**: Perturb inputs, test with challenging examples

**Implementation:** See `code/evaluation.py` for robustness evaluation:
- Out-of-distribution testing utilities
- Adversarial robustness evaluation
- Robustness gap computation

## Implementation Examples

### Complete Reward Model Training

**Implementation:** See `code/reward_model.py` for complete training pipeline:
- `RewardModelTrainer` - Complete training pipeline
- `train_step()` - Training step implementation
- `evaluate()` - Model evaluation utilities
- `save_model()` and `load_model()` - Model persistence

### Reward Model Inference

**Implementation:** See `code/reward_model.py` for inference utilities:
- `RewardModelInference` - Complete inference pipeline
- `predict_reward()` - Single prediction utilities
- `rank_responses()` - Response ranking capabilities
- `batch_predict()` - Batch prediction utilities

## Advanced Techniques

### Multi-Task Reward Learning

**Intuitive Understanding:**
Instead of learning one reward function, learn multiple reward functions for different objectives (helpfulness, harmlessness, honesty) and combine them.

**Benefits:**
- **Interpretability**: Can understand what each objective contributes
- **Flexibility**: Can adjust weights for different use cases
- **Robustness**: Less likely to overfit to single objective

**Implementation:** See `code/reward_model.py` for multi-task learning:
- `MultiObjectiveRewardModel` - Multi-task reward modeling
- Support for multiple objectives
- Weighted combination of tasks
- Task-specific loss functions

### Uncertainty-Aware Reward Models

**Intuitive Understanding:**
Instead of just predicting a reward score, also predict how uncertain we are about that prediction.

**Benefits:**
- **Better decision making**: Can account for uncertainty in policy optimization
- **Active learning**: Can identify examples where we need more human feedback
- **Robustness**: Can handle cases where preferences are unclear

**Implementation:** See `code/reward_model.py` for uncertainty modeling:
- Uncertainty quantification in reward predictions
- Confidence-aware training
- Uncertainty-based sample selection

### Calibrated Reward Models

**Intuitive Understanding:**
Ensure that when the model predicts a 70% probability of preference, it's actually right 70% of the time.

**Benefits:**
- **Reliable probabilities**: Can trust the predicted preference probabilities
- **Better decision making**: More reliable for downstream tasks
- **Uncertainty quantification**: Better understanding of model confidence

**Implementation:** See `code/reward_model.py` for calibration:
- Temperature scaling for calibration
- Calibration loss functions
- Calibrated prediction utilities

## Best Practices

### 1. Data Quality

- **Diverse Training Data**: Ensure coverage across different topics, styles, and difficulty levels
- **Quality Control**: Use multiple annotators and agreement monitoring
- **Bias Mitigation**: Address systematic biases in annotation
- **Validation Split**: Maintain separate validation set for model selection

**Implementation Tips:**
- Collect data from diverse sources and annotators
- Monitor inter-annotator agreement
- Regular data quality audits
- Balance data across different categories

### 2. Model Architecture

- **Appropriate Encoder**: Choose encoder architecture based on task requirements
- **Regularization**: Use dropout, L2 regularization, and other techniques
- **Architecture Search**: Experiment with different architectures for optimal performance
- **Scalability**: Consider computational requirements for large-scale deployment

**Implementation Tips:**
- Start with simple architectures and scale up
- Use cross-validation to compare architectures
- Monitor training and validation metrics
- Consider inference speed requirements

### 3. Training Strategy

- **Learning Rate Scheduling**: Use appropriate learning rate schedules
- **Early Stopping**: Monitor validation metrics to prevent overfitting
- **Gradient Clipping**: Prevent gradient explosion in large models
- **Mixed Precision**: Use mixed precision training for efficiency

**Implementation Tips:**
- Use learning rate warmup and decay
- Monitor validation metrics during training
- Implement gradient clipping for stability
- Use mixed precision for faster training

### 4. Evaluation

- **Multiple Metrics**: Use preference accuracy, ranking correlation, and calibration
- **Robustness Testing**: Evaluate on out-of-distribution and adversarial examples
- **Human Evaluation**: Validate automated metrics with human judgments
- **Continuous Monitoring**: Monitor model performance in production

**Implementation Tips:**
- Track multiple evaluation metrics
- Regular robustness testing
- Periodic human evaluation
- Set up monitoring dashboards

### 5. Deployment

- **Model Compression**: Consider quantization and distillation for deployment
- **Caching**: Cache reward predictions for efficiency
- **Monitoring**: Implement monitoring for model drift and performance
- **Versioning**: Maintain model versions and rollback capabilities

**Implementation Tips:**
- Quantize models for faster inference
- Implement prediction caching
- Monitor for data drift
- Maintain model versioning system

## Summary

Reward modeling is a critical component of RLHF systems that enables learning from human preferences. Key aspects include:

1. **Mathematical Foundation**: Preference learning with Bradley-Terry model
2. **Architecture Design**: Various encoder architectures for different requirements
3. **Training Objectives**: Preference loss, ranking loss, and contrastive learning
4. **Validation**: Comprehensive evaluation with multiple metrics
5. **Advanced Techniques**: Multi-task learning, uncertainty quantification, and calibration
6. **Best Practices**: Data quality, model architecture, training strategy, and deployment

Effective reward modeling enables the training of language models that better align with human values and preferences, ultimately leading to more useful, safe, and honest AI systems.

**Key Takeaways:**
- Reward modeling converts human preferences into learnable reward functions
- The Bradley-Terry model provides a principled framework for preference learning
- Different architectures and loss functions serve different use cases
- Comprehensive evaluation is essential for reliable reward models
- Advanced techniques enable better uncertainty quantification and calibration

**The Broader Impact:**
Reward modeling has fundamentally changed how we train AI systems by:
- **Enabling preference-based learning**: Learning from human judgments rather than labels
- **Supporting subjective objectives**: Handling goals that can't be easily quantified
- **Enabling continuous improvement**: Systems that can get better with more feedback
- **Advancing AI alignment**: Training systems to behave according to human values

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
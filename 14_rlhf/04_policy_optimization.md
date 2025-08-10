# Policy Optimization

This guide provides a comprehensive overview of policy optimization methods for reinforcement learning from human feedback (RLHF) systems. We'll explore policy gradient methods, proximal policy optimization (PPO), trust region policy optimization (TRPO), and their applications to language model training.

## From Reward Functions to Policy Optimization

We've now explored **reward modeling** - the process of learning a function that maps prompt-response pairs to scalar reward values, capturing human preferences and judgments. We've seen how to design neural network architectures that can learn from preference data, how to formulate training objectives that capture the relative nature of human preferences, how to validate and calibrate reward models, and how to address challenges like reward hacking and distributional shift.

However, while having a well-trained reward model is crucial, **the reward function alone** cannot improve language model behavior. Consider having a perfect reward model that can evaluate any response - we still need an optimization algorithm that can update the language model policy to maximize expected reward while maintaining language quality and preventing catastrophic forgetting.

This motivates our exploration of **policy optimization** - the core component of RLHF that updates the language model policy to maximize expected reward from human feedback. We'll see how policy gradient methods like REINFORCE and actor-critic approaches work for language models, how trust region methods like PPO and TRPO ensure stable policy updates, how to handle the unique challenges of language generation (sequential decisions, sparse rewards, high dimensionality), and how to balance reward maximization with maintaining language quality.

The transition from reward modeling to policy optimization represents the bridge from evaluation to improvement - taking our understanding of how to evaluate responses with reward functions and applying it to the challenge of optimizing language model policies to produce better responses.

In this section, we'll explore policy optimization, understanding how to update language model policies to maximize expected reward while maintaining language quality.

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Policy Gradient Methods](#policy-gradient-methods)
- [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Trust Region Policy Optimization (TRPO)](#trust-region-policy-optimization-trpo)
- [Language Model Specifics](#language-model-specifics)
- [Implementation Examples](#implementation-examples)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)

## Overview

Policy optimization is the core component of RLHF that updates the language model policy to maximize expected reward from human feedback. Unlike traditional supervised learning that minimizes prediction error, policy optimization maximizes reward signals from human evaluators or learned reward models.

### Key Concepts

**1. Policy Gradient**: Direct optimization of policy parameters using gradient ascent
**2. Trust Region**: Constraining policy updates to prevent catastrophic changes
**3. Advantage Estimation**: Estimating relative value of actions for stable updates
**4. KL Control**: Preventing policy from deviating too far from reference model

### Problem Formulation

**Objective**: Maximize expected reward:
```math
J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s, a)]
```

Where:
- $`\pi_\theta`$: Language model policy with parameters $`\theta`$
- $`\rho_\pi`$: State distribution under policy $`\pi`$
- $`R(s, a)`$: Reward function (human feedback or learned reward model)

**Language Model Context**:
- **State**: Current conversation context or prompt $`s_t`$
- **Action**: Next token to generate $`a_t`$ from vocabulary $`\mathcal{V}`$
- **Policy**: $`\pi_\theta(a_t|s_t) = P(y_t | x, y_1, y_2, \ldots, y_{t-1})`$

## Mathematical Foundations

### Policy Gradient Theorem

**Theorem**: For any differentiable policy $`\pi_\theta`$, the gradient of the objective is:
```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s,a) \nabla_\theta \log \pi_\theta(a|s)]
```

**Proof Sketch**:
```math
\begin{align}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s,a)] \\
&= \nabla_\theta \int \rho_\pi(s) \pi_\theta(a|s) R(s,a) \, ds \, da \\
&= \int \rho_\pi(s) \nabla_\theta \pi_\theta(a|s) R(s,a) \, ds \, da \\
&= \int \rho_\pi(s) \pi_\theta(a|s) \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} R(s,a) \, ds \, da \\
&= \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s,a) \nabla_\theta \log \pi_\theta(a|s)]
\end{align}
```

### Advantage Function

**Definition**: The advantage function measures how much better an action is compared to the average:
```math
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
```

Where:
- $`Q^\pi(s, a)`$: Action-value function
- $`V^\pi(s)`$: State-value function

**Properties**:
- $`\mathbb{E}_{a \sim \pi} [A^\pi(s, a)] = 0`$: Average advantage is zero
- $`A^\pi(s, a) > 0`$: Action is better than average
- $`A^\pi(s, a) < 0`$: Action is worse than average

### Policy Gradient with Advantage

**Improved Policy Gradient**:
```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [A^\pi(s, a) \nabla_\theta \log \pi_\theta(a|s)]
```

**Benefits**:
- **Variance Reduction**: Advantage function reduces gradient variance
- **Stable Updates**: More stable policy updates
- **Better Convergence**: Faster and more reliable convergence

## Policy Gradient Methods

### REINFORCE Algorithm

**Basic REINFORCE**:
```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s,a) \nabla_\theta \log \pi_\theta(a|s)]
```

**Implementation:** See `policy_optimization.py` for REINFORCE implementation:
- `REINFORCETrainer` - Complete REINFORCE trainer for language models
- `reinforce_loss()` - REINFORCE loss computation
- `train_step()` - Training step implementation
- `generate_responses()` - Response generation utilities

### REINFORCE with Baseline

**Baseline Subtraction**:
```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [(R(s,a) - b(s)) \nabla_\theta \log \pi_\theta(a|s)]
```

Where $`b(s)`$ is a baseline function.

**Implementation:** See `policy_optimization.py` for baseline methods:
- Baseline estimation utilities
- Advantage computation with baselines
- Variance reduction techniques

### Actor-Critic Methods

**Actor-Critic Architecture**:
- **Actor**: Policy network $`\pi_\theta(a|s)`$
- **Critic**: Value network $`V_\phi(s)`$

**Advantage Estimation**:
```math
A_t = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
```

**Implementation:** See `policy_optimization.py` for actor-critic methods:
- Actor-critic training utilities
- Advantage estimation with value functions
- Value function learning

## Proximal Policy Optimization (PPO)

### PPO-Clip Objective

**PPO-Clip Loss**:
```math
L(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]
```

Where:
- $`r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}`$: Probability ratio
- $`A_t`$: Advantage estimate
- $`\epsilon`$: Clipping parameter (typically 0.2)

**Intuition**:
- **Ratio Clipping**: Prevents large policy updates
- **Conservative Updates**: Ensures policy doesn't change too much
- **Stable Training**: More stable than vanilla policy gradients

### PPO Implementation

**Implementation:** See `policy_optimization.py` for complete PPO implementation:
- `PPOTrainer` - Complete PPO trainer for language models
- `compute_advantages()` - Generalized Advantage Estimation (GAE)
- `compute_kl_divergence()` - KL divergence computation
- `ppo_loss()` - PPO loss with clipping
- `train_step()` - Complete PPO training step
- `generate_responses()` - Response generation
- `save_model()` and `load_model()` - Model persistence

### PPO for Language Models

**Token-level PPO**: Apply PPO at the token level for fine-grained control

**Sequence-level PPO**: Apply PPO at the sequence level for natural reward structure

**Implementation:** See `policy_optimization.py` for language model adaptations:
- Token-level and sequence-level optimization
- Language model specific training utilities
- Response generation and evaluation

## Trust Region Policy Optimization (TRPO)

### TRPO Objective

**TRPO Problem**:
```math
\max_\theta \mathbb{E}_t \left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t\right]
\text{ subject to } \mathbb{E}_t [\text{KL}(\pi_{\theta_{old}} \| \pi_\theta)] \leq \delta
```

Where:
- $`\delta`$: Trust region constraint threshold
- $`\text{KL}(\pi_{\theta_{old}} \| \pi_\theta)`$: KL divergence between old and new policies

### TRPO Implementation

**Implementation:** See `policy_optimization.py` for complete TRPO implementation:
- `TRPOTrainer` - Complete TRPO trainer
- `conjugate_gradient()` - Conjugate gradient optimization
- `fisher_vector_product()` - Fisher information matrix operations
- `compute_kl()` - KL divergence computation
- `trpo_step()` - TRPO training step
- `get_log_probs()` and `get_ref_log_probs()` - Log probability utilities

## Language Model Specifics

### Token-Level vs Sequence-Level Optimization

**Token-Level Optimization**:
- **Advantage**: Fine-grained control over each token
- **Challenge**: Sparse rewards at token level
- **Implementation**: Apply RL to each token generation step

**Sequence-Level Optimization**:
- **Advantage**: Natural reward structure
- **Challenge**: Credit assignment problem
- **Implementation**: Apply RL to complete sequences

### KL Divergence Control

**Purpose**: Prevent policy from deviating too far from reference model

**Implementation:** See `policy_optimization.py` for KL control:
- KL divergence computation and monitoring
- Adaptive KL penalty coefficients
- KL constraint enforcement

### Entropy Regularization

**Purpose**: Encourage exploration and prevent premature convergence

**Implementation:** See `policy_optimization.py` for entropy regularization:
- Entropy computation and regularization
- Exploration encouragement
- Convergence prevention

## Implementation Examples

### Complete PPO Training Loop

**Implementation:** See `policy_optimization.py` for complete training pipeline:
- `PolicyOptimizationPipeline` - Complete RLHF training pipeline
- Support for PPO, TRPO, and REINFORCE methods
- `train_epoch()` - Complete training loop
- `evaluate()` - Model evaluation utilities
- `save_model()` and `load_model()` - Model persistence

### Advanced PPO with GAE

**Implementation:** See `policy_optimization.py` for advanced PPO:
- `PPOTrainer` - Advanced PPO with GAE
- `compute_advantages()` - Generalized Advantage Estimation
- Value function learning and integration
- Advanced training utilities

## Advanced Techniques

### Multi-Objective PPO

**Implementation:** See `policy_optimization.py` for multi-objective optimization:
- Multi-objective loss computation
- Weighted combination of objectives
- Objective-specific training

### Conservative Policy Iteration

**Implementation:** See `policy_optimization.py` for conservative methods:
- Natural policy gradient computation
- Fisher information matrix operations
- Conservative update strategies

## Best Practices

### 1. Hyperparameter Tuning

**Key Hyperparameters**:
- **Learning Rate**: Start with 1e-5, adjust based on convergence
- **Clip Epsilon**: 0.1-0.3 for PPO, 0.01-0.05 for TRPO
- **KL Coefficient**: 0.01-0.1 for KL penalty
- **Batch Size**: 32-128 for language models
- **PPO Epochs**: 4-10 epochs per batch

### 2. Training Stability

**Techniques**:
- **Gradient Clipping**: Prevent gradient explosion
- **Learning Rate Scheduling**: Reduce learning rate over time
- **Early Stopping**: Stop when KL divergence exceeds threshold
- **Reward Normalization**: Normalize rewards to zero mean, unit variance

### 3. Evaluation

**Metrics**:
- **Average Reward**: Monitor reward improvement
- **KL Divergence**: Track policy deviation from reference
- **Perplexity**: Monitor language model quality
- **Human Evaluation**: Validate with human judgments

### 4. Implementation Considerations

**Language Model Specifics**:
- **Token-Level vs Sequence-Level**: Choose based on reward structure
- **KL Control**: Essential for preventing catastrophic forgetting
- **Reward Scaling**: Important for stable training
- **Batch Processing**: Efficient handling of variable-length sequences

### 5. Debugging

**Common Issues**:
- **Reward Hacking**: Models optimizing for proxy objectives
- **Mode Collapse**: Policy converging to limited responses
- **Catastrophic Forgetting**: Losing pre-trained knowledge
- **Unstable Training**: Oscillating or diverging rewards

**Solutions**:
- **Regularization**: KL penalties and entropy regularization
- **Monitoring**: Track multiple metrics during training
- **Checkpointing**: Save models at regular intervals
- **Validation**: Use held-out data for evaluation

## Summary

Policy optimization is a critical component of RLHF that enables language models to learn from human feedback. Key aspects include:

1. **Policy Gradient Methods**: REINFORCE and actor-critic approaches
2. **Trust Region Methods**: PPO and TRPO for stable updates
3. **Language Model Adaptations**: Token-level and sequence-level optimization
4. **Regularization**: KL control and entropy regularization
5. **Advanced Techniques**: Multi-objective optimization and conservative updates
6. **Best Practices**: Hyperparameter tuning, stability, and evaluation

Effective policy optimization enables the training of language models that better align with human values and preferences, ultimately leading to more useful, safe, and honest AI systems.

---

**Note**: This guide provides the theoretical and practical foundations for policy optimization in RLHF. For specific implementation details and advanced techniques, refer to the implementation examples and external resources referenced in the main README.

## From Policy Optimization to Advanced Alignment

We've now explored **policy optimization** - the core component of RLHF that updates the language model policy to maximize expected reward from human feedback. We've seen how policy gradient methods like REINFORCE and actor-critic approaches work for language models, how trust region methods like PPO and TRPO ensure stable policy updates, how to handle the unique challenges of language generation (sequential decisions, sparse rewards, high dimensionality), and how to balance reward maximization with maintaining language quality.

However, while policy optimization enables basic RLHF training, **the standard RLHF pipeline** has limitations in ensuring comprehensive alignment. Consider a model trained with RLHF - it may be helpful and harmless in most cases, but it might still have blind spots, be vulnerable to adversarial attacks, or lack explicit mechanisms for self-correction and safety.

This motivates our exploration of **advanced alignment techniques** - methods that go beyond standard RLHF to ensure more robust, safe, and beneficial AI behavior. We'll see how Direct Preference Optimization (DPO) eliminates the need for separate reward models, how Constitutional AI enables self-critique and revision, how Red Teaming systematically identifies and addresses safety vulnerabilities, and how multi-objective alignment balances competing objectives like helpfulness, harmlessness, and honesty.

The transition from policy optimization to advanced alignment represents the bridge from basic RLHF to comprehensive AI safety - taking our understanding of how to optimize language model policies and applying it to the challenge of building AI systems that are not just capable but also safe, honest, and beneficial to society.

In the next section, we'll explore advanced alignment techniques, understanding how to build more robust and safe AI systems that go beyond standard RLHF.

---

**Previous: [Reward Modeling](03_reward_modeling.md)** - Learn how to convert human preferences into reward functions.

**Next: [Alignment Techniques](05_alignment_techniques.md)** - Learn advanced techniques for ensuring AI safety and alignment. 
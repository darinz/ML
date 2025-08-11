# Reinforcement Learning for Training Large Language Models

[![RLHF](https://img.shields.io/badge/RLHF-Reinforcement%20Learning%20from%20Human%20Feedback-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback)
[![Alignment](https://img.shields.io/badge/Alignment-AI%20Alignment-green.svg)](https://en.wikipedia.org/wiki/AI_alignment)
[![LLM](https://img.shields.io/badge/LLM-Large%20Language%20Models-purple.svg)](https://en.wikipedia.org/wiki/Large_language_model)

Comprehensive materials covering reinforcement learning for language models, including RLHF, reward modeling, policy optimization, and alignment techniques.

## Overview

RL for language models represents a paradigm shift from supervised learning to preference-based learning, using human feedback to create more helpful, harmless, and honest AI systems.

## Materials

### Theory
- **[01_fundamentals_of_rl_for_language_models.md](01_fundamentals_of_rl_for_language_models.md)** - Core concepts and problem formulation
- **[02_human_feedback_collection.md](02_human_feedback_collection.md)** - Data collection strategies and annotation guidelines
- **[03_reward_modeling.md](03_reward_modeling.md)** - Reward function learning and validation
- **[04_policy_optimization.md](04_policy_optimization.md)** - Policy gradient methods and optimization techniques
- **[05_alignment_techniques.md](05_alignment_techniques.md)** - Advanced alignment methods and safety considerations
- **[06_hands-on_coding.md](06_hands-on_coding.md)** - Practical implementation guide

### Core RLHF Components
- **[code/reward_model.py](code/reward_model.py)** - Reward model implementation
- **[code/policy_optimization.py](code/policy_optimization.py)** - PPO/TRPO for language models
- **[code/preference_data.py](code/preference_data.py)** - Preference data processing
- **[code/evaluation.py](code/evaluation.py)** - Reward model and policy evaluation

### Advanced Techniques
- **[code/dpo.py](code/dpo.py)** - Direct Preference Optimization
- **[code/constitutional_ai.py](code/constitutional_ai.py)** - Constitutional AI framework
- **[code/red_teaming.py](code/red_teaming.py)** - Adversarial testing tools
- **[code/alignment_eval.py](code/alignment_eval.py)** - Alignment evaluation metrics

### Practical Applications
- **[code/chatbot_rlhf.py](code/chatbot_rlhf.py)** - RLHF for conversational AI
- **[code/summarization_rl.py](code/summarization_rl.py)** - RL for text summarization
- **[code/code_generation.py](code/code_generation.py)** - RL for code generation
- **[code/safety_alignment.py](code/safety_alignment.py)** - Safety-focused alignment

### Supporting Files
- **code/requirements.txt** - Python dependencies
- **code/environment.yaml** - Conda environment setup

## Key Concepts

### Problem Formulation
**MDP for Language Models**:
- **State**: Conversation context or prompt
- **Action**: Next token to generate
- **Reward**: Human preference score or learned reward function
- **Policy**: Language model parameters $\pi_\theta$

### Reward Modeling
**Preference Learning**: $\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))$

**Reward Function**: $R_\phi(x, y) = f_\phi(\text{encode}(x, y))$

### Policy Optimization
**PPO-Clip**: $L(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$

**DPO**: $\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$

### Alignment Techniques
**Constitutional AI**: Self-critique and revision framework
**Red Teaming**: Adversarial testing for safety
**Value Learning**: Explicit value alignment techniques

## Applications

- **Conversational AI**: Chatbots and virtual assistants
- **Content Generation**: Writing assistance and creative tools
- **Code Generation**: Programming assistance
- **Text Summarization**: Abstractive summarization with RL
- **Safety Alignment**: Making models more helpful and harmless

## Getting Started

1. Read `01_fundamentals_of_rl_for_language_models.md` for RL basics
2. Study `02_human_feedback_collection.md` for data collection
3. Learn `03_reward_modeling.md` for reward learning
4. Explore `04_policy_optimization.md` for policy optimization
5. Understand `05_alignment_techniques.md` for alignment
6. Follow `06_hands-on_coding.md` for implementation

## Prerequisites

- Reinforcement learning fundamentals
- Deep learning and transformers
- Natural language processing
- Python, PyTorch, transformers library

## Installation

```bash
pip install -r code/requirements.txt
# or
conda env create -f code/environment.yaml
```

## Quick Start

```python
# Reward Model
from code.reward_model import RewardModel
reward_model = RewardModel(encoder='bert-base', hidden_dim=768)

# Policy Optimization
from code.policy_optimization import PPOTrainer
trainer = PPOTrainer(model, reward_model, tokenizer)

# DPO
from code.dpo import DPOTrainer
trainer = DPOTrainer(model, ref_model, tokenizer, beta=0.1)

# Constitutional AI
from code.constitutional_ai import ConstitutionalAI
ai = ConstitutionalAI(model, principles=['helpful', 'harmless', 'honest'])
```

## Reference Papers

- **RLHF**: Training language models to follow instructions with human feedback
- **DPO**: Direct Preference Optimization
- **Constitutional AI**: Self-critique and revision
- **Red Teaming**: Adversarial testing for language models
- **PPO**: Proximal Policy Optimization Algorithms

## Ethical Considerations

- **Bias Amplification**: RLHF may amplify existing biases
- **Reward Hacking**: Models optimizing for proxy objectives
- **Value Alignment**: Whose values should models reflect?
- **Transparency**: Understanding model decision-making 
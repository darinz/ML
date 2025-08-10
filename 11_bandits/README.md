# Multi-Armed Bandits

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Bandits](https://img.shields.io/badge/Bandits-Multi--Armed-green.svg)](https://en.wikipedia.org/wiki/Multi-armed_bandit)
[![RL](https://img.shields.io/badge/RL-Reinforcement%20Learning-purple.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)

Comprehensive materials covering multi-armed bandits for sequential decision-making under uncertainty, balancing exploration and exploitation.

## Overview

Multi-armed bandits provide the theoretical foundation for balancing exploration and exploitation in dynamic environments, with applications from clinical trials to online advertising.

## Materials

### Theory
- **[01_classical_multi_armed_bandits.md](01_classical_multi_armed_bandits.md)** - Classical bandit algorithms, regret analysis, theoretical guarantees
- **[02_linear_bandits.md](02_linear_bandits.md)** - Linear bandit algorithms, LinUCB, linear Thompson sampling
- **[03_contextual_bandits.md](03_contextual_bandits.md)** - Contextual bandits with state-dependent rewards
- **[04_best_arm_identification.md](04_best_arm_identification.md)** - Best arm identification, pure exploration problems
- **[05_applications_and_use_cases.md](05_applications_and_use_cases.md)** - Real-world applications and practical implementations
- **[06_hands-on_coding.md](06_hands-on_coding.md)** - Practical implementation guide

### Basic Algorithms
- **[epsilon_greedy.py](epsilon_greedy.py)** - Epsilon-greedy algorithm implementation
- **[ucb.py](ucb.py)** - Upper Confidence Bound algorithm
- **[thompson_sampling.py](thompson_sampling.py)** - Thompson sampling for Bernoulli bandits
- **[regret_analysis.py](regret_analysis.py)** - Regret computation and visualization

### Advanced Algorithms
- **[linucb.py](linucb.py)** - Linear UCB implementation
- **[linear_thompson.py](linear_thompson.py)** - Linear Thompson sampling
- **[contextual_bandits.py](contextual_bandits.py)** - Contextual bandit framework
- **[neural_bandits.py](neural_bandits.py)** - Deep learning for bandits
- **[meta_bandits.py](meta_bandits.py)** - Meta-learning approaches
- **[bai_algorithms.py](bai_algorithms.py)** - Best arm identification methods
- **[multi_objective.py](multi_objective.py)** - Multi-objective bandits

### Practical Applications
- **[recommendation_system.py](recommendation_system.py)** - Movie recommendation with bandits
- **[ad_selection.py](ad_selection.py)** - Online advertising simulation
- **[clinical_trials.py](clinical_trials.py)** - Adaptive clinical trial design
- **[dynamic_pricing.py](dynamic_pricing.py)** - Price optimization simulation

### Supporting Files
- **requirements.txt** - Python dependencies

## Key Concepts

### Classical Multi-Armed Bandits
**Problem**: Choose from $K$ arms with unknown reward distributions

**Regret**: $R(T) = \sum_{t=1}^T \mu^* - \mu_{a_t}$

**Algorithms**:
- **Epsilon-Greedy**: Random exploration with probability $\epsilon$
- **UCB**: $a_t = \arg\max_i (\hat{\mu}_i + \sqrt{\frac{2 \log (t)}{n_i}})$
- **Thompson Sampling**: Sample from posterior distributions

### Linear Bandits
**Model**: $r_t = \langle \theta^*, x_{a_t} \rangle + \eta_t$

**LinUCB**: $a_t = \arg\max_i (\langle \hat{\theta}_t, x_i \rangle + \alpha \sqrt{x_i^T A_t^{-1} x_i})$

### Contextual Bandits
**Model**: $r_t = \langle \theta^*, x_{a_t, t} \rangle + \eta_t$

**Challenge**: Context vectors change over time

### Best Arm Identification
**Goal**: Identify best arm with high confidence

**Algorithms**: Successive elimination, LUCB, racing algorithms

## Applications

- **Online Advertising**: Ad selection and bidding optimization
- **Recommendation Systems**: Content and product recommendations
- **Clinical Trials**: Adaptive treatment assignment
- **Dynamic Pricing**: Price optimization and revenue management
- **Resource Allocation**: Optimal allocation under uncertainty

## Getting Started

1. Read `01_classical_multi_armed_bandits.md` for fundamentals
2. Study `02_linear_bandits.md` for linear extensions
3. Learn `03_contextual_bandits.md` for contextual settings
4. Explore `04_best_arm_identification.md` for pure exploration
5. Use `05_applications_and_use_cases.md` for practical applications
6. Follow `06_hands-on_coding.md` for implementation guidance

## Prerequisites

- Probability theory and statistics
- Basic optimization concepts
- Python programming and NumPy
- Understanding of confidence intervals

## Installation

```bash
pip install -r requirements.txt
```

## Running Examples

```bash
python epsilon_greedy.py
python ucb.py
python thompson_sampling.py
python linucb.py
python recommendation_system.py
python ad_selection.py
```

## Quick Start Code

```python
# Epsilon-Greedy
from epsilon_greedy import EpsilonGreedy
bandit = EpsilonGreedy(n_arms=10, epsilon=0.1)
action = bandit.select_action()
reward = environment.step(action)
bandit.update(action, reward)

# UCB
from ucb import UCB
bandit = UCB(n_arms=10)
action = bandit.select_action()

# Linear UCB
from linucb import LinUCB
bandit = LinUCB(n_arms=10, d=5, alpha=1.0)
action = bandit.select_action(context)
```

## Algorithm Comparison

| Algorithm | Regret Bound | Assumptions | Use Case |
|-----------|--------------|-------------|----------|
| Epsilon-Greedy | $O(T^{2/3})$ | None | Simple exploration |
| UCB | $O(\sqrt{KT \log T})$ | Bounded rewards | Theoretical guarantees |
| Thompson Sampling | $O(\sqrt{KT \log T})$ | Bayesian | Practical performance |
| LinUCB | $O(d\sqrt{T \log T})$ | Linear rewards | Feature-based decisions | 
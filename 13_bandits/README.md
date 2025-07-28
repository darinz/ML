# Multi-Armed Bandits

This section contains comprehensive materials covering multi-armed bandits, a fundamental framework for sequential decision-making under uncertainty. Multi-armed bandits provide the theoretical foundation for balancing exploration and exploitation in dynamic environments, with applications ranging from clinical trials to online advertising and recommendation systems.

## Overview

Multi-armed bandits represent a simplified form of reinforcement learning where an agent must repeatedly choose from a set of actions (arms) to maximize cumulative reward over time. The key challenge is balancing exploration (trying new actions to learn their rewards) with exploitation (choosing actions known to be good).

### Learning Objectives

Upon completing this section, you will understand:
- The fundamental trade-off between exploration and exploitation
- Classical bandit algorithms and their theoretical guarantees
- Linear and contextual bandit extensions
- Best arm identification and pure exploration problems
- Practical applications in recommendation systems and A/B testing
- Regret analysis and performance bounds

## Table of Contents

- [Classical Multi-Armed Bandits](#classical-multi-armed-bandits)
- [Linear Bandits](#linear-bandits)
- [Contextual Bandits](#contextual-bandits)
- [Best Arm Identification](#best-arm-identification)
- [Applications and Use Cases](#applications-and-use-cases)
- [Implementation Examples](#implementation-examples)
- [Reference Materials](#reference-materials)

## Documentation Files

- [01_classical_multi_armed_bandits.md](01_classical_multi_armed_bandits.md) - Comprehensive guide to classical multi-armed bandit algorithms
- [02_linear_bandits.md](02_linear_bandits.md) - Linear bandit algorithms and theoretical foundations
- [03_contextual_bandits.md](03_contextual_bandits.md) - Contextual bandits with state-dependent rewards
- [04_best_arm_identification.md](04_best_arm_identification.md) - Best arm identification and pure exploration problems
- [05_applications_and_use_cases.md](05_applications_and_use_cases.md) - Real-world applications and practical implementations

## Classical Multi-Armed Bandits

📖 **Detailed Guide**: [01_classical_multi_armed_bandits.md](01_classical_multi_armed_bandits.md)

### Problem Formulation

In the classical multi-armed bandit problem, we have:
- $`K`$ arms (actions) with unknown reward distributions
- At each time step $`t`$, we choose an arm $`a_t \in \{1, 2, \ldots, K\}`$
- We receive a reward $`r_t`$ drawn from the distribution of arm $`a_t`$
- Goal: Maximize cumulative reward $`\sum_{t=1}^T r_t`$ over $`T`$ rounds

**Key Concepts:**
- **Regret**: Difference between optimal cumulative reward and achieved reward
- **Exploration vs. Exploitation**: Trade-off between learning and earning
- **Sample Complexity**: Number of samples needed to identify the best arm

### Fundamental Algorithms

**Epsilon-Greedy:**
- With probability $`\epsilon`$: choose random arm (exploration)
- With probability $`1-\epsilon`$: choose arm with highest empirical mean (exploitation)

**Upper Confidence Bound (UCB):**
```math
a_t = \arg\max_{i} \left(\hat{\mu}_i + \sqrt{\frac{2 \log t}{n_i}}\right)
```
Where:
- $`\hat{\mu}_i`$: Empirical mean of arm $`i`$
- $`n_i`$: Number of times arm $`i`` has been pulled
- $`t`$: Current time step

**Thompson Sampling:**
- Maintain posterior distributions over arm rewards
- Sample from posteriors to select arms
- Update posteriors based on observed rewards

### Regret Analysis

**Theoretical Guarantees:**
- **UCB**: $`O(\sqrt{KT \log T})`$ regret bound
- **Thompson Sampling**: Similar theoretical guarantees with Bayesian assumptions
- **Lower Bounds**: $`\Omega(\sqrt{KT})`$ for any algorithm

## Linear Bandits

📖 **Detailed Guide**: [02_linear_bandits.md](02_linear_bandits.md)

### Problem Extension

Linear bandits extend the classical setting by assuming rewards are linear functions of arm features:

```math
r_t = \langle \theta^*, x_{a_t} \rangle + \eta_t
```

Where:
- $`\theta^* \in \mathbb{R}^d`$: Unknown parameter vector
- $`x_{a_t} \in \mathbb{R}^d`$: Feature vector of chosen arm
- $`\eta_t`$: Noise term

### Algorithms

**LinUCB:**
```math
a_t = \arg\max_{i} \left(\langle \hat{\theta}_t, x_i \rangle + \alpha \sqrt{x_i^T A_t^{-1} x_i}\right)
```

Where:
- $`\hat{\theta}_t`$: Least squares estimate of $`\theta^*`$
- $`A_t`$: Design matrix
- $`\alpha`$: Exploration parameter

**Linear Thompson Sampling:**
- Maintain Gaussian posterior over $`\theta^*`$
- Sample $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, A_t^{-1})`$
- Choose $`a_t = \arg\max_i \langle \theta_t, x_i \rangle`$

### Regret Bounds

**Theoretical Results:**
- **LinUCB**: $`O(d\sqrt{T \log T})`$ regret bound
- **Linear Thompson Sampling**: Similar guarantees under Bayesian assumptions
- **Lower Bounds**: $`\Omega(d\sqrt{T})`$ for any algorithm

## Contextual Bandits

📖 **Detailed Guide**: [03_contextual_bandits.md](03_contextual_bandits.md)

### Problem Setting

Contextual bandits introduce context (state) information that changes over time:

```math
r_t = \langle \theta^*, x_{a_t, t} \rangle + \eta_t
```

Where $`x_{a_t, t}`$ depends on both the arm and the current context.

### Key Challenges

**Contextual Information:**
- Context vectors change over time
- Need to learn context-dependent reward functions
- Balance exploration across different contexts

**Algorithmic Approaches:**
- **Contextual UCB**: Extend UCB to handle contexts
- **Contextual Thompson Sampling**: Bayesian approach with context
- **Neural Bandits**: Deep learning for complex reward functions

### Applications

**Real-World Use Cases:**
- **Online Advertising**: Choose ads based on user context
- **Recommendation Systems**: Recommend items based on user features
- **Clinical Trials**: Adaptive treatment assignment
- **Dynamic Pricing**: Price optimization with customer features

## Best Arm Identification

📖 **Detailed Guide**: [04_best_arm_identification.md](04_best_arm_identification.md)

### Pure Exploration

In best arm identification (BAI), the goal is to identify the best arm with high confidence, rather than maximizing cumulative reward.

**Problem Formulation:**
- Fixed budget of $`T`$ total pulls
- Goal: Identify arm with highest mean reward
- Success probability: $`P(\hat{i}^* = i^*) \geq 1-\delta`$

### Algorithms

**Successive Elimination:**
1. Pull each arm $`n_0`$ times
2. Eliminate arms with low empirical means
3. Continue until one arm remains

**Racing Algorithms:**
- Maintain confidence intervals for all arms
- Stop when one arm is clearly best
- Adaptive allocation based on uncertainty

**LUCB (Lower-Upper Confidence Bound):**
- Maintain confidence intervals for all arms
- Pull arms with highest upper bound and highest lower bound
- Stop when intervals separate

### Sample Complexity

**Theoretical Results:**
- **Gap-dependent bounds**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta})`$
- **Gap-independent bounds**: $`O(K \log \frac{1}{\delta})`$
- Where $`\Delta_i = \mu_{i^*} - \mu_i`$ is the gap

## Applications and Use Cases

📖 **Detailed Guide**: [05_applications_and_use_cases.md](05_applications_and_use_cases.md)

### Online Advertising

**Ad Selection:**
- Choose from multiple ad creatives
- Optimize for click-through rate (CTR)
- Balance exploration of new ads with exploitation of known good ads

**Bidding Optimization:**
- Real-time bidding in ad auctions
- Learn optimal bid amounts
- Contextual information from user profiles

### Recommendation Systems

**Content Recommendation:**
- Choose from multiple content options
- Learn user preferences over time
- Handle cold-start problems for new users/items

**Product Recommendations:**
- E-commerce product suggestions
- Balance popular items with niche products
- Personalize based on user behavior

### Clinical Trials

**Adaptive Trials:**
- Allocate patients to treatment arms
- Learn treatment effectiveness
- Ethical considerations for patient welfare

**Drug Discovery:**
- Screen multiple drug candidates
- Optimize experimental design
- Resource allocation in pharmaceutical research

### Dynamic Pricing

**Price Optimization:**
- Set prices for products/services
- Learn demand elasticity
- Balance revenue with market share

**Revenue Management:**
- Hotel room pricing
- Airline ticket pricing
- Real-time price adjustments

## Implementation Examples

### Basic Bandit Algorithms

**Core Implementations:**
- `epsilon_greedy.py`: Epsilon-greedy algorithm
- `ucb.py`: Upper Confidence Bound algorithm
- `thompson_sampling.py`: Thompson sampling for Bernoulli bandits
- `regret_analysis.py`: Regret computation and visualization

### Linear Bandits

**Linear Extensions:**
- `linucb.py`: Linear UCB implementation
- `linear_thompson.py`: Linear Thompson sampling
- `feature_engineering.py`: Feature extraction and preprocessing
- `contextual_bandits.py`: Contextual bandit framework

### Advanced Algorithms

**Modern Techniques:**
- `neural_bandits.py`: Deep learning for bandits
- `meta_bandits.py`: Meta-learning approaches
- `bai_algorithms.py`: Best arm identification methods
- `multi_objective.py`: Multi-objective bandits

### Practical Applications

**Real-World Examples:**
- `recommendation_system.py`: Movie recommendation with bandits
- `ad_selection.py`: Online advertising simulation
- `clinical_trials.py`: Adaptive clinical trial design
- `dynamic_pricing.py`: Price optimization simulation

## Reference Materials

### Core Textbooks and Resources

**Foundational Materials:**
- **[Bandit Algorithms Textbook](https://tor-lattimore.com/downloads/book/book.pdf)**: Comprehensive textbook by Tor Lattimore and Csaba Szepesvári
- **[Informal Notes](https://courses.cs.washington.edu/courses/cse541/24sp/resources/lecture_notes.pdf)**: University of Washington course notes

### Research Papers

**Linear Bandits:**
- **[Linear Bandits Paper](https://papers.nips.cc/paper_files/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html)**: Theoretical foundations of linear bandits
- **[Generalized Linear Bandits](https://papers.nips.cc/paper_files/paper/2010/hash/c2626d850c80ea07e7511bbae4c76f4b-Abstract.html)**: Extension to generalized linear models
- **[Pure Exploration/BAI Paper](https://arxiv.org/abs/1409.6110)**: Best arm identification algorithms

**Contextual Bandits:**
- **[Contextual Bandits Survey](https://www.ambujtewari.com/research/tewari17ads.pdf)**: Comprehensive survey by Ambuj Tewari

### Educational Resources

**Learning Materials:**
- **Stanford CS234**: Reinforcement Learning course with bandit coverage
- **UC Berkeley CS285**: Deep Reinforcement Learning with bandit foundations
- **University of Washington CSE541**: Advanced topics in machine learning

### Implementation Libraries

**Practical Tools:**
- **[Vowpal Wabbit](https://vowpalwabbit.org/)**: Fast online learning library
- **[Contextual Bandits](https://github.com/david-abel/simple_rl)**: Simple RL library with bandits
- **[BanditLib](https://github.com/banditlib/banditlib)**: C++ bandit algorithms library

## Getting Started

### Prerequisites

Before diving into multi-armed bandits, ensure you have:
- **Probability Theory**: Understanding of random variables and expectations
- **Optimization**: Basic optimization concepts and algorithms
- **Statistics**: Confidence intervals and hypothesis testing
- **Python Programming**: NumPy, Matplotlib, statistical libraries

### Installation

```bash
# Core dependencies
pip install numpy scipy matplotlib seaborn
pip install pandas scikit-learn

# Additional utilities
pip install jupyter ipywidgets
pip install tqdm for progress bars
```

### Quick Start

1. **Understand the Problem**: Start with classical multi-armed bandits
2. **Implement Basic Algorithms**: Code epsilon-greedy and UCB
3. **Study Linear Extensions**: Learn linear bandits with features
4. **Explore Applications**: Apply to recommendation systems
5. **Advanced Topics**: Best arm identification and contextual bandits

## Theoretical Foundations

### Regret Analysis

**Cumulative Regret:**
```math
R(T) = \sum_{t=1}^T \mu^* - \mu_{a_t}
```

Where $`\mu^* = \max_i \mu_i`$ is the optimal arm's mean reward.

**Expected Regret:**
```math
\mathbb{E}[R(T)] = \sum_{i \neq i^*} \Delta_i \mathbb{E}[n_i(T)]
```

Where $`n_i(T)`$ is the number of times arm $`i`$ is pulled.

### Concentration Inequalities

**Hoeffding's Inequality:**
For bounded random variables $`X_i \in [a, b]`$:
```math
P\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mu\right| \geq \epsilon\right) \leq 2\exp\left(-\frac{2n\epsilon^2}{(b-a)^2}\right)
```

**Chernoff Bounds:**
For Bernoulli random variables:
```math
P(\hat{\mu} \geq \mu + \epsilon) \leq \exp(-n \text{KL}(\mu + \epsilon \| \mu))
```

### Algorithm Analysis

**UCB Regret Bound:**
For UCB algorithm with $`\alpha = 2`$:
```math
\mathbb{E}[R(T)] \leq \sum_{i \neq i^*} \frac{8 \log T}{\Delta_i} + \left(1 + \frac{\pi^2}{3}\right) \sum_{i=1}^K \Delta_i
```

## Future Directions

### Emerging Research Areas

**Recent Developments:**
- **Non-stationary Bandits**: Handling changing reward distributions
- **Structured Bandits**: Exploiting structure in action spaces
- **Multi-objective Bandits**: Balancing multiple objectives
- **Federated Bandits**: Distributed learning across agents
- **Safe Bandits**: Ensuring safety constraints during learning

### Open Problems

**Research Challenges:**
- **Non-linear Reward Functions**: Beyond linear approximations
- **High-dimensional Contexts**: Feature selection and dimensionality reduction
- **Correlated Arms**: Exploiting correlations between actions
- **Adversarial Bandits**: Robust algorithms against adversarial rewards
- **Bandits with Constraints**: Resource constraints and fairness

---

**Note**: This section is under active development. Content will be added progressively as materials become available. Check back regularly for updates and new implementations. 
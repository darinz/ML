# Classical Multi-Armed Bandits

## Introduction

Classical multi-armed bandits represent the foundational framework for sequential decision-making under uncertainty. The problem gets its name from the analogy of a gambler facing multiple slot machines (arms) and trying to maximize their cumulative winnings over time. This framework provides the theoretical foundation for balancing exploration and exploitation in dynamic environments.

### The Core Challenge

The fundamental challenge in multi-armed bandits is the **exploration-exploitation trade-off**:
- **Exploration**: Trying different arms to learn their reward distributions
- **Exploitation**: Choosing arms that are known to provide high rewards

This trade-off appears in numerous real-world scenarios:
- Clinical trials: testing different treatments
- Online advertising: selecting ad creatives
- Recommendation systems: choosing content to show users
- A/B testing: comparing different website designs

## Problem Formulation

### Mathematical Setup

In the classical multi-armed bandit problem, we have:

- **$`K`$ arms** (actions) with unknown reward distributions
- At each time step $`t`$, we choose an arm $`a_t \in \{1, 2, \ldots, K\}`$
- We receive a reward $`r_t`$ drawn from the distribution of arm $`a_t`$
- **Goal**: Maximize cumulative reward $`\sum_{t=1}^T r_t`$ over $`T`$ rounds

### Key Definitions

**Reward Distributions:**
- Each arm $`i`$ has an unknown reward distribution with mean $`\mu_i`$
- Rewards are typically bounded: $`r_t \in [0, 1]`$ or $`r_t \in [a, b]`$
- The optimal arm is $`i^* = \arg\max_i \mu_i`$ with mean $`\mu^* = \mu_{i^*}`$

**Regret:**
The **cumulative regret** measures the difference between optimal and achieved performance:

```math
R(T) = \sum_{t=1}^T \mu^* - \mu_{a_t}
```

**Expected Regret:**
```math
\mathbb{E}[R(T)] = \sum_{i \neq i^*} \Delta_i \mathbb{E}[n_i(T)]
```

Where:
- $`\Delta_i = \mu^* - \mu_i`$ is the **gap** between arm $`i`$ and the optimal arm
- $`n_i(T)`$ is the number of times arm $`i`$ is pulled up to time $`T`$

### Problem Variants

**Stochastic Bandits:**
- Rewards are drawn from fixed, unknown distributions
- Most common and well-studied variant

**Adversarial Bandits:**
- Rewards are chosen by an adversary
- No statistical assumptions on reward generation

**Non-stationary Bandits:**
- Reward distributions change over time
- Requires algorithms that can adapt to changing environments

## Fundamental Algorithms

### 1. Epsilon-Greedy

The simplest exploration strategy that balances exploration and exploitation through a fixed parameter.

**Algorithm:**
- With probability $`\epsilon`$: choose random arm (exploration)
- With probability $`1-\epsilon`$: choose arm with highest empirical mean (exploitation)

**Implementation:**
See [`epsilon_greedy.py`](epsilon_greedy.py) for the complete implementation.

```python
# Key implementation details:
def epsilon_greedy(arms, epsilon, T):
    # With probability epsilon: choose random arm (exploration)
    # With probability 1-epsilon: choose best empirical arm (exploitation)
    # Update empirical means incrementally
```

**Analysis:**
- **Pros**: Simple, easy to implement
- **Cons**: Fixed exploration rate, suboptimal regret bounds
- **Regret**: $`O(T^{2/3})`$ with optimal $`\epsilon`$ decay

### 2. Upper Confidence Bound (UCB)

UCB uses confidence intervals to balance exploration and exploitation optimally.

**Algorithm:**
```math
a_t = \arg\max_{i} \left(\hat{\mu}_i + \sqrt{\frac{2 \log t}{n_i}}\right)
```

Where:
- $`\hat{\mu}_i`$: Empirical mean of arm $`i`$
- $`n_i`$: Number of times arm $`i`$ has been pulled
- $`t`$: Current time step

**Intuition:**
- First term: exploitation (choose arm with high empirical mean)
- Second term: exploration (choose arm with high uncertainty)
- The exploration term decreases as we pull an arm more

**Implementation:**
See [`ucb.py`](ucb.py) for the complete implementation.

```python
# Key implementation details:
def ucb(arms, T):
    # Pull each arm once initially
    # Calculate UCB values: empirical_mean + sqrt(2*log(t)/pulls)
    # Choose arm with highest UCB value
    # Update empirical means incrementally
```

**Theoretical Guarantees:**
- **Regret Bound**: $`O(\sqrt{KT \log T})`$
- **Gap-dependent bound**: $`O(\sum_{i \neq i^*} \frac{\log T}{\Delta_i})`$
- **Optimal up to logarithmic factors**

### 3. Thompson Sampling

Thompson sampling is a Bayesian approach that maintains posterior distributions over arm rewards.

**Algorithm:**
1. Maintain posterior distributions over arm rewards
2. Sample from posteriors to select arms
3. Update posteriors based on observed rewards

**For Bernoulli Rewards:**
- Prior: Beta distribution $`\text{Beta}(1, 1)`$ (uniform)
- Posterior: $`\text{Beta}(1 + S_i, 1 + F_i)`$
- Where $`S_i`$ = successes, $`F_i`$ = failures for arm $`i`$

**Implementation:**
See [`thompson_sampling.py`](thompson_sampling.py) for the complete implementation.

```python
# Key implementation details:
def thompson_sampling(arms, T):
    # Maintain Beta(alpha, beta) posteriors for each arm
    # Sample from posteriors to select arms
    # Update posteriors based on observed rewards (0 or 1)
    # Return empirical means from final posteriors
```

**Theoretical Guarantees:**
- **Regret Bound**: $`O(\sqrt{KT \log T})`$ (similar to UCB)
- **Bayesian optimal** under certain assumptions
- **Often performs better in practice** than UCB

## Theoretical Analysis

### Regret Bounds

**UCB Regret Analysis:**
For UCB algorithm with exploration parameter $`\alpha = 2`$:

```math
\mathbb{E}[R(T)] \leq \sum_{i \neq i^*} \frac{8 \log T}{\Delta_i} + \left(1 + \frac{\pi^2}{3}\right) \sum_{i=1}^K \Delta_i
```

**Key Insights:**
- **Gap-dependent**: Regret depends on gaps $`\Delta_i`$
- **Logarithmic**: Regret grows as $`\log T`$ for each suboptimal arm
- **Constant term**: Initial exploration cost

**Lower Bounds:**
- **Minimax lower bound**: $`\Omega(\sqrt{KT})`$ for any algorithm
- **Gap-dependent lower bound**: $`\Omega(\sum_{i \neq i^*} \frac{\log T}{\Delta_i})`$

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

**Application to UCB:**
The exploration term $`\sqrt{\frac{2 \log t}{n_i}}`$ in UCB is derived from Hoeffding's inequality, ensuring that the true mean lies within the confidence interval with high probability.

### Sample Complexity

**Definition:**
The number of samples needed to identify the best arm with high confidence.

**For Best Arm Identification:**
- **Successive Elimination**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta})`$
- **Racing Algorithms**: Similar bounds with adaptive allocation

## Practical Considerations

### Parameter Tuning

**Epsilon-Greedy:**
- **Fixed epsilon**: Simple but suboptimal
- **Decaying epsilon**: $`\epsilon_t = \min(1, \frac{cK}{d^2 t})`$ for better performance
- **Adaptive epsilon**: Adjust based on observed performance

**UCB:**
- **Exploration parameter**: $`\alpha`$ controls exploration vs. exploitation
- **Typical values**: $`\alpha = 2`$ (theoretical), $`\alpha = 1`$ (practical)
- **Tuning**: Cross-validation or theoretical analysis

**Thompson Sampling:**
- **Prior choice**: Uniform Beta(1,1) is often sufficient
- **Non-informative priors**: Can improve performance with domain knowledge
- **Hyperparameter-free**: Often works well without tuning

### Implementation Details

**Numerical Stability:**
```python
# Avoid division by zero in UCB
if pulls[i] == 0:
    ucb_value = float('inf')
else:
    ucb_value = empirical_means[i] + np.sqrt(2 * np.log(t) / pulls[i])
```

**Memory Efficiency:**
```python
# Incremental mean update
def update_mean(old_mean, old_count, new_value):
    return (old_mean * old_count + new_value) / (old_count + 1)
```

**Parallelization:**
- **Batch updates**: Process multiple rewards simultaneously
- **Distributed bandits**: Multiple agents sharing information
- **Asynchronous updates**: Handle delayed feedback

### Common Pitfalls

**1. Insufficient Exploration:**
- Using too small epsilon in epsilon-greedy
- Setting exploration parameter too low in UCB
- Not accounting for reward variance

**2. Over-exploration:**
- Using too large epsilon
- Setting exploration parameter too high
- Wasting samples on clearly suboptimal arms

**3. Non-stationary Environments:**
- Classical algorithms assume stationary rewards
- Need adaptation mechanisms for changing environments
- Consider sliding windows or discounting

**4. Reward Scaling:**
- Algorithms assume bounded rewards $`[0, 1]`$
- Need proper scaling for different reward ranges
- Consider normalization or clipping

## Advanced Topics

### Non-stationary Bandits

**Problem:**
Reward distributions change over time, requiring algorithms that can adapt.

**Solutions:**
- **Sliding window UCB**: Only use recent observations
- **Discounting**: Give more weight to recent rewards
- **Change detection**: Detect when distributions change

### Correlated Arms

**Problem:**
Arms are not independent; pulling one arm provides information about others.

**Solutions:**
- **Correlation-aware algorithms**: Exploit arm correlations
- **Feature-based approaches**: Use arm features to share information
- **Hierarchical bandits**: Group similar arms together

### Multi-objective Bandits

**Problem:**
Multiple objectives to optimize simultaneously (e.g., revenue and user satisfaction).

**Solutions:**
- **Pareto optimality**: Find non-dominated solutions
- **Scalarization**: Combine objectives into single metric
- **Preference learning**: Learn user preferences over objectives

## Applications

### Online Advertising

**Ad Selection:**
- Choose from multiple ad creatives
- Optimize for click-through rate (CTR)
- Balance exploration of new ads with exploitation of known good ads

**Implementation:**
See [`ad_bandit.py`](ad_bandit.py) for the complete implementation.

```python
# Key implementation details:
class AdBandit:
    # UCB-based ad selection
    # Track empirical click-through rates
    # Update statistics incrementally
```

### Recommendation Systems

**Content Recommendation:**
- Choose from multiple content options
- Learn user preferences over time
- Handle cold-start problems for new users/items

**Clinical Trials:**
- Allocate patients to treatment arms
- Learn treatment effectiveness
- Ethical considerations for patient welfare

## Implementation Examples

### Basic Bandit Environment

See [`bandit_environment.py`](bandit_environment.py) for the complete implementation.

```python
# Key components:
class BernoulliArm:
    # Bernoulli arm with success probability p
    
class BanditEnvironment:
    # Multi-armed bandit environment
    # Calculate cumulative regret
```

### Algorithm Comparison

See [`bandit_environment.py`](bandit_environment.py) for the complete implementation.

```python
# Key functionality:
def compare_algorithms(env, algorithms, T=1000, n_runs=100):
    # Compare different bandit algorithms
    # Run multiple independent trials
    # Return average regrets for each algorithm

# Example usage:
# arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
# env = BanditEnvironment(arm_means)
# algorithms = {'Epsilon-Greedy': ..., 'UCB': ..., 'Thompson Sampling': ...}
# results = compare_algorithms(env, algorithms)
```

### Visualization

See [`bandit_environment.py`](bandit_environment.py) for the complete implementation.

```python
# Key functionality:
def plot_regret_comparison(results, T):
    # Plot regret comparison of different algorithms
    # Show cumulative regret over time
    # Include legend and proper labeling

# Usage:
# plot_regret_comparison(results, 1000)
```

## Summary

Classical multi-armed bandits provide a fundamental framework for sequential decision-making under uncertainty. The key insights are:

1. **Exploration-Exploitation Trade-off**: The core challenge of balancing learning and earning
2. **Algorithm Diversity**: Different approaches (epsilon-greedy, UCB, Thompson sampling) offer various trade-offs
3. **Theoretical Guarantees**: UCB and Thompson sampling achieve near-optimal regret bounds
4. **Practical Considerations**: Parameter tuning, numerical stability, and application-specific adaptations
5. **Wide Applicability**: From online advertising to clinical trials

The classical bandit framework serves as the foundation for more advanced variants like linear bandits, contextual bandits, and best arm identification problems.

## Further Reading

- **Bandit Algorithms Textbook**: Comprehensive treatment by Tor Lattimore and Csaba Szepesvári
- **Research Papers**: Original UCB and Thompson sampling papers
- **Online Courses**: Stanford CS234, UC Berkeley CS285
- **Implementation Libraries**: Vowpal Wabbit, Contextual Bandits

---

**Note**: This guide covers the fundamentals of classical multi-armed bandits. For more advanced topics, see the sections on Linear Bandits, Contextual Bandits, and Best Arm Identification.

## From Independent Arms to Structured Learning

We've now explored **classical multi-armed bandits** - the foundational framework for sequential decision-making under uncertainty. We've seen how the exploration-exploitation trade-off manifests in various algorithms like epsilon-greedy, UCB, and Thompson sampling, and how these methods provide theoretical guarantees for regret minimization.

However, while classical bandits treat each arm independently, **real-world problems** often exhibit structure that can be exploited for more efficient learning. Consider a recommendation system where products have features - similar products likely have similar reward distributions, and learning about one product can inform our understanding of similar products.

This motivates our exploration of **linear bandits** - extending the bandit framework to handle structured action spaces where rewards are linear functions of arm features. We'll see how LinUCB and linear Thompson sampling can exploit arm similarities, how feature-based learning reduces sample complexity, and how these methods enable efficient learning in high-dimensional action spaces.

The transition from classical to linear bandits represents the bridge from independent learning to structured learning - taking our understanding of the exploration-exploitation trade-off and extending it to leverage the structure inherent in many real-world problems.

In the next section, we'll explore linear bandits, understanding how they exploit arm similarities and enable more efficient learning through feature-based representations.

---

**Next: [Linear Bandits](02_linear_bandits.md)** - Learn how to exploit structured action spaces for more efficient learning. 
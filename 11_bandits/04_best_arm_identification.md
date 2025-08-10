# Best Arm Identification

## Introduction

Best Arm Identification (BAI) represents a fundamental shift in the multi-armed bandit paradigm from cumulative reward maximization to pure exploration. Unlike traditional bandit algorithms that balance exploration and exploitation, BAI algorithms focus exclusively on identifying the best arm with high confidence, regardless of the cumulative reward achieved during the learning process.

### Motivation

Traditional bandit algorithms optimize for cumulative reward, but many real-world scenarios prioritize accurate identification over immediate performance:

- **Clinical Trials**: Identify the most effective treatment, not maximize short-term outcomes
- **A/B Testing**: Determine the best website design or feature
- **Drug Discovery**: Find the most promising drug candidate
- **Product Development**: Select the best product variant for mass production
- **Algorithm Selection**: Choose the best algorithm for a specific task

### Key Differences from Traditional Bandits

**Traditional Bandits (Regret Minimization):**
- Goal: Maximize cumulative reward $`\sum_{t=1}^T r_t`$
- Metric: Cumulative regret $`R(T) = \sum_{t=1}^T (\mu^* - \mu_{a_t})`$
- Trade-off: Exploration vs. exploitation

**Best Arm Identification (Pure Exploration):**
- Goal: Identify best arm with high confidence
- Metric: Success probability $`P(\hat{i}^* = i^*)`$
- Focus: Pure exploration, no exploitation needed

## Problem Formulation

### Mathematical Setup

In the best arm identification problem, we have:

- **$`K`$ arms** with unknown reward distributions
- **Fixed budget** of $`T`$ total pulls
- **Goal**: Identify arm with highest mean reward $`i^* = \arg\max_i \mu_i`$
- **Success criterion**: $`P(\hat{i}^* = i^*) \geq 1-\delta`$ where $`\delta`$ is the failure probability

### Key Definitions

**Gaps:**
- $`\Delta_i = \mu_{i^*} - \mu_i`$: Gap between optimal arm and arm $`i`$
- $`\Delta_{\min} = \min_{i \neq i^*} \Delta_i`$: Minimum gap
- $`\Delta_{\max} = \max_{i \neq i^*} \Delta_i`$: Maximum gap

**Sample Complexity:**
- **Gap-dependent**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta})`$
- **Gap-independent**: $`O(K \log \frac{1}{\delta})`$

**Confidence Intervals:**
- $`\text{CI}_i(t) = [\hat{\mu}_i(t) \pm \beta_i(t)]`$: Confidence interval for arm $`i`$ at time $`t`$
- $`\beta_i(t)`$: Confidence radius (depends on algorithm)

### Problem Variants

**Fixed Budget:**
- Total number of pulls $`T`$ is fixed
- Goal: Maximize success probability

**Fixed Confidence:**
- Target success probability $`1-\delta`$ is fixed
- Goal: Minimize expected number of pulls

**Fixed Budget and Confidence:**
- Both $`T`$ and $`\delta`$ are fixed
- Goal: Achieve success probability $`\geq 1-\delta`$ within $`T`$ pulls

## Fundamental Algorithms

### 1. Successive Elimination

Successive Elimination is a simple and intuitive algorithm that eliminates arms progressively based on empirical comparisons.

**Algorithm:**
1. **Initialization**: Pull each arm $`n_0`$ times
2. **Elimination**: Eliminate arms with low empirical means
3. **Iteration**: Continue until one arm remains

**Implementation:**
See [`code/successive_elimination.py`](code/successive_elimination.py) for the complete implementation.

```python
# Key implementation details:
class SuccessiveElimination:
    # Initialize statistics and active arms set
    # Pull arms until n0 times, then eliminate worst arm
    # Continue until only one arm remains
```

**Theoretical Guarantees:**
- **Sample Complexity**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{K}{\delta})`$
- **Success Probability**: $`P(\hat{i}^* = i^*) \geq 1-\delta`$

### 2. Racing Algorithms

Racing algorithms maintain confidence intervals for all arms and stop when one arm is clearly the best.

**Algorithm:**
1. **Initialization**: Pull each arm $`n_0`$ times
2. **Confidence Intervals**: Maintain confidence intervals for all arms
3. **Stopping Criterion**: Stop when one arm's lower bound exceeds all others' upper bounds
4. **Adaptive Allocation**: Pull arms with highest uncertainty

**Implementation:**
See [`code/racing_algorithm.py`](code/racing_algorithm.py) for the complete implementation.

```python
# Key implementation details:
class RacingAlgorithm:
    # Maintain confidence intervals for all arms
    # Pull arms with highest uncertainty
    # Stop when one arm's lower bound exceeds all others' upper bounds
```

### 3. LUCB (Lower-Upper Confidence Bound)

LUCB is a sophisticated algorithm that pulls the arm with highest upper bound and the arm with highest lower bound among the remaining arms.

**Algorithm:**
1. **Initialization**: Pull each arm $`n_0`$ times
2. **Arm Selection**: Pull arms with highest upper bound and highest lower bound
3. **Stopping Criterion**: Stop when intervals separate

**Implementation:**
See [`code/lucb.py`](code/lucb.py) for the complete implementation.

```python
# Key implementation details:
class LUCB:
    # Pull arm with highest upper bound and arm with highest lower bound
    # Use LUCB-specific confidence intervals
    # Stop when intervals separate
```

### 4. Sequential Halving

Sequential Halving is an efficient algorithm that eliminates half of the remaining arms in each round.

**Algorithm:**
1. **Initialization**: Start with all arms
2. **Rounds**: In each round, pull remaining arms equally
3. **Elimination**: Eliminate bottom half of arms based on empirical means
4. **Termination**: Continue until one arm remains

**Implementation:**
See [`sequential_halving.py`](sequential_halving.py) for the complete implementation.

```python
# Key implementation details:
class SequentialHalving:
    # Eliminate half of remaining arms in each round
    # Pull arms equally within each round
    # Continue until one arm remains
```

## Theoretical Analysis

### Sample Complexity Bounds

**Gap-dependent Bounds:**
For algorithms with gap-dependent sample complexity:

```math
\mathbb{E}[N] \leq O\left(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta}\right)
```

**Gap-independent Bounds:**
For algorithms with gap-independent sample complexity:

```math
\mathbb{E}[N] \leq O\left(K \log \frac{1}{\delta}\right)
```

**Lower Bounds:**
For any BAI algorithm:

```math
\mathbb{E}[N] \geq \Omega\left(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta}\right)
```

### Confidence Interval Analysis

**Hoeffding-based Intervals:**
For bounded rewards $`r_i \in [0, 1]`$:

```math
P(|\hat{\mu}_i - \mu_i| \geq \epsilon) \leq 2\exp(-2n_i \epsilon^2)
```

**Chernoff-based Intervals:**
For Bernoulli rewards:

```math
P(|\hat{\mu}_i - \mu_i| \geq \epsilon) \leq 2\exp(-n_i \text{KL}(\mu_i + \epsilon \| \mu_i))
```

**Union Bound:**
For multiple arms and time steps:

```math
P(\exists i, t : |\hat{\mu}_i(t) - \mu_i| \geq \beta_i(t)) \leq \delta
```

### Algorithm Comparison

**Successive Elimination:**
- **Pros**: Simple, intuitive, good theoretical guarantees
- **Cons**: May not be optimal for all gap structures
- **Sample Complexity**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{K}{\delta})`$

**Racing Algorithms:**
- **Pros**: Adaptive allocation, good empirical performance
- **Cons**: More complex implementation
- **Sample Complexity**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{K}{\delta})`$

**LUCB:**
- **Pros**: Optimal sample complexity, theoretical guarantees
- **Cons**: Complex implementation, may be conservative
- **Sample Complexity**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta})`$

**Sequential Halving:**
- **Pros**: Simple, efficient for large action spaces
- **Cons**: Fixed budget requirement, may not be optimal
- **Sample Complexity**: $`O(K \log \frac{1}{\delta})`$

## Practical Considerations

### Parameter Tuning

**Confidence Level ($`\delta`$):**
- **Typical values**: $`\delta = 0.1, 0.05, 0.01`$
- **Trade-off**: Lower $`\delta`$ requires more samples but higher confidence
- **Application-specific**: Choose based on cost of incorrect identification

**Initial Sample Size ($`n_0`$):**
- **Theoretical**: $`n_0 = O(\log \frac{K}{\delta})`$
- **Practical**: $`n_0 = 10-50`$ often works well
- **Adaptive**: Can be adjusted based on observed gaps

### Stopping Criteria

**Fixed Budget:**
- Stop when $`T`$ pulls are exhausted
- Return best arm based on empirical means

**Fixed Confidence:**
- Stop when confidence intervals separate
- Return arm with highest lower bound

**Adaptive Stopping:**
- Stop when success probability exceeds threshold
- Requires online estimation of success probability

### Numerical Stability

**Confidence Interval Calculation:**
See [`bai_utils.py`](bai_utils.py) for the complete implementation.

```python
# Key functionality:
def stable_confidence_radius(pulls, delta, method='hoeffding'):
    # Calculate stable confidence radius using Hoeffding or Chernoff bounds

def stable_mean_update(old_mean, old_count, new_value):
    # Stable incremental mean update
```

## Advanced Topics

### Contextual Best Arm Identification

**Problem Extension:**
Identify the best arm for each context, not just globally.

**Algorithms:**
- **Contextual Successive Elimination**: Eliminate arms per context
- **Contextual Racing**: Maintain confidence intervals per context
- **Contextual LUCB**: Extend LUCB to handle contexts

### Linear Best Arm Identification

**Problem Setting:**
Identify the best arm when rewards are linear functions of features.

**Algorithms:**
- **Linear Successive Elimination**: Use linear confidence intervals
- **Linear Racing**: Maintain ellipsoidal confidence regions
- **Linear LUCB**: Extend LUCB to linear bandits

### Multi-objective Best Arm Identification

**Problem:**
Identify the best arm when there are multiple objectives to optimize.

**Approaches:**
- **Pareto optimality**: Find non-dominated arms
- **Scalarization**: Combine objectives into single metric
- **Preference learning**: Learn user preferences over objectives

### Best Arm Identification with Constraints

**Resource Constraints:**
- **Budget constraints**: Limited total cost
- **Time constraints**: Limited time horizon
- **Safety constraints**: Ensure safe exploration

**Algorithms:**
- **Constrained BAI**: Add constraint handling to BAI algorithms
- **Safe exploration**: Ensure constraints are satisfied during identification

## Applications

### A/B Testing

**Website Design Testing:**
See [`bai_applications.py`](bai_applications.py) for the complete implementation.

```python
# Key functionality:
class ABTestBAI:
    # Use LUCB for A/B testing
    # Convert multiple metrics to single reward
    # Stop when best variant is identified
```

### Clinical Trials

**Treatment Comparison:**
See [`bai_applications.py`](bai_applications.py) for the complete implementation.

```python
# Key functionality:
class ClinicalTrialBAI:
    # Use Successive Elimination for clinical trials
    # Track patient outcomes
    # Stop when best treatment is identified
```

### Product Development

**Feature Selection:**
See [`bai_applications.py`](bai_applications.py) for the complete implementation.

```python
# Key functionality:
class FeatureSelectionBAI:
    # Use Racing Algorithm for feature selection
    # Evaluate feature performance
    # Stop when best feature is identified
```

### Algorithm Selection

**Machine Learning Algorithm Selection:**
See [`bai_applications.py`](bai_applications.py) for the complete implementation.

```python
# Key functionality:
class AlgorithmSelectionBAI:
    # Use Sequential Halving for algorithm selection
    # Train and evaluate algorithms
    # Stop when best algorithm is identified
```

## Implementation Examples

### Complete BAI Environment

See [`bai_environment.py`](bai_environment.py) for the complete implementation.

```python
# Key components:
class BAIEnvironment:
    # BAI environment with arm means and noise
    # Pull arms and observe rewards
    # Calculate gaps between optimal and suboptimal arms

def run_bai_experiment(env, algorithm, max_pulls=1000):
    # Run BAI experiment
    # Select arms, observe rewards, check completion
```

### Algorithm Comparison

See [`bai_environment.py`](bai_environment.py) for the complete implementation.

```python
# Key functionality:
def compare_bai_algorithms(env, algorithms, n_runs=100, max_pulls=1000):
    # Compare different BAI algorithms
    # Run multiple independent trials
    # Return success rates and sample complexities

# Example usage:
# arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
# env = BAIEnvironment(arm_means)
# algorithms = {'Successive Elimination': ..., 'Racing': ..., 'LUCB': ..., 'Sequential Halving': ...}
# results = compare_bai_algorithms(env, algorithms)
```

### Visualization

See [`bai_environment.py`](bai_environment.py) for the complete implementation.

```python
# Key functionality:
def plot_bai_results(results):
    # Plot success rate comparison
    # Plot sample complexity comparison
    # Include error bars and proper labeling

# Usage:
# plot_bai_results(results)
```

## Summary

Best Arm Identification provides a powerful framework for pure exploration problems where accurate identification is more important than cumulative reward. Key insights include:

1. **Pure Exploration Focus**: Algorithms prioritize identification over immediate performance
2. **Confidence-based Stopping**: Stop when confidence intervals separate
3. **Theoretical Guarantees**: Optimal sample complexity bounds
4. **Practical Algorithms**: Successive Elimination, Racing, LUCB, Sequential Halving
5. **Wide Applicability**: A/B testing, clinical trials, algorithm selection

BAI algorithms bridge the gap between traditional bandits and pure exploration problems, providing both theoretical guarantees and practical effectiveness for identification tasks.

## Further Reading

- **Pure Exploration/BAI Paper**: Theoretical foundations and algorithms
- **Best Arm Identification Survey**: Comprehensive overview of BAI methods
- **Contextual BAI**: Extensions to contextual settings
- **Linear BAI**: Extensions to linear bandits

---

**Note**: This guide covers the fundamentals of Best Arm Identification. For more advanced topics, see the sections on Contextual BAI, Linear BAI, and Multi-objective BAI.

## From Pure Exploration to Real-World Applications

We've now explored **Best Arm Identification (BAI)** - a fundamental shift in the multi-armed bandit paradigm from cumulative reward maximization to pure exploration. We've seen how algorithms like Successive Elimination, Racing, LUCB, and Sequential Halving focus exclusively on identifying the best arm with high confidence, regardless of the cumulative reward achieved during the learning process.

However, while understanding BAI algorithms is valuable, **the true impact** of multi-armed bandits lies in their real-world applications. Consider the algorithms we've learned - from classical bandits to linear and contextual bandits, and now best arm identification - these theoretical frameworks become powerful when applied to solve actual problems in advertising, healthcare, e-commerce, and beyond.

This motivates our exploration of **applications and use cases** - the practical implementation of bandit algorithms across diverse domains. We'll see how bandits optimize ad selection and bidding in online advertising, how they enable personalized recommendations in e-commerce and content platforms, how they improve clinical trials and drug discovery in healthcare, how they optimize pricing strategies in dynamic markets, and how they enhance A/B testing and algorithm selection processes.

The transition from best arm identification to applications represents the bridge from pure exploration to practical impact - taking our understanding of how to identify optimal actions and applying it to real-world scenarios where intelligent decision-making under uncertainty provides significant value.

In the next section, we'll explore applications and use cases, understanding how bandit algorithms solve real-world problems and create value across diverse domains.

## Complete Example Usage

See [`bai_example.py`](bai_example.py) for a complete example demonstrating how to use all BAI algorithms together, including:

- Environment setup and algorithm comparison
- Performance analysis and ranking
- Single algorithm demonstrations
- Confidence interval visualization

The example includes comprehensive analysis of success rates, sample complexities, and efficiency metrics across all BAI algorithms.

---

**Previous: [Contextual Bandits](03_contextual_bandits.md)** - Learn how to adapt bandit algorithms to changing environments.

**Next: [Applications and Use Cases](05_applications_and_use_cases.md)** - Explore real-world applications of multi-armed bandits. 
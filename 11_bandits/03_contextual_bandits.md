 # Contextual Bandits

## Introduction

Contextual bandits represent a natural evolution of multi-armed bandits by incorporating dynamic context information that influences reward distributions. Unlike classical bandits where arms have fixed reward distributions, contextual bandits adapt to changing environments where the optimal action depends on the current context.

## From Static Features to Dynamic Contexts

We've now explored **linear bandits** - extending the bandit framework to handle structured action spaces where rewards are linear functions of arm features. We've seen how LinUCB and linear Thompson sampling can exploit arm similarities, how feature-based learning reduces sample complexity, and how these methods enable efficient learning in high-dimensional action spaces.

However, while linear bandits leverage the structure of arm features, **real-world environments** are often dynamic and context-dependent. Consider an online advertising system where the effectiveness of an ad depends not just on the ad itself, but also on the user's current context - their demographics, browsing history, time of day, and other contextual factors.

This motivates our exploration of **contextual bandits** - extending the bandit framework to handle dynamic contexts where the optimal action depends on the current state. We'll see how contextual UCB and contextual Thompson sampling adapt to changing environments, how personalization enables tailored decisions, and how these methods handle the complexity of real-world applications.

The transition from linear to contextual bandits represents the bridge from static structure to dynamic adaptation - taking our understanding of feature-based learning and extending it to handle the temporal and contextual dynamics inherent in many real-world problems.

In this section, we'll explore contextual bandits, understanding how they adapt to changing contexts and enable personalized decision-making.

### Motivation

The classical multi-armed bandit framework assumes stationary reward distributions, which is often unrealistic in practice. Contextual bandits address this limitation by:

- **Adapting to changing environments**: Context provides information about the current state
- **Personalizing decisions**: Actions are tailored to specific contexts
- **Handling non-stationarity**: Reward distributions change with context
- **Enabling real-world applications**: Most practical scenarios involve contextual information

### Key Applications

Contextual bandits are essential in scenarios where:
- **User context matters**: Different users respond differently to the same action
- **Temporal dynamics**: Optimal actions change over time
- **Personalization**: Tailoring decisions to individual characteristics
- **Adaptive systems**: Learning and adapting to changing environments

## Problem Formulation

### Mathematical Setup

Contextual bandits introduce context (state) information that changes over time:

```math
r_t = \langle \theta^*, x_{a_t, t} \rangle + \eta_t
```

Where:
- $`\theta^* \in \mathbb{R}^d`$: Unknown parameter vector (true reward model)
- $`x_{a_t, t} \in \mathbb{R}^d`$: Feature vector of chosen arm at time $`t`$ in context $`t`$
- $`\eta_t`$: Noise term (typically sub-Gaussian)
- $`a_t \in \{1, 2, \ldots, K\}`$: Chosen arm at time $`t`$
- $`c_t`$: Context at time $`t`$ (may be implicit in $`x_{a_t, t}`$)

### Key Assumptions

**Contextual Reward Structure:**
- Rewards are linear in context-arm features: $`\mathbb{E}[r_t | a_t, c_t] = \langle \theta^*, x_{a_t, t} \rangle`$
- The parameter vector $`\theta^*`$ is unknown but fixed
- Feature vectors $`x_{i, t}`$ depend on both arm $`i`$ and context $`c_t`$

**Context Generation:**
- Contexts $`c_t`$ may be drawn from a distribution or chosen by an adversary
- Contexts can be arbitrary (adversarial) or have structure (stochastic)
- Feature vectors $`x_{i, t}`$ are revealed for all arms $`i`$ at time $`t`$

**Noise Assumptions:**
- Noise terms $`\eta_t`$ are conditionally sub-Gaussian given the history
- $`\mathbb{E}[\eta_t | \mathcal{F}_{t-1}, a_t, c_t] = 0`$ (zero mean)
- $`\mathbb{E}[\exp(\lambda \eta_t) | \mathcal{F}_{t-1}, a_t, c_t] \leq \exp(\frac{\lambda^2 \sigma^2}{2})`$ (sub-Gaussian)

### Problem Variants

**Stochastic Contexts:**
- Contexts are drawn from a fixed distribution
- Allows for more optimistic regret bounds
- Common in recommendation systems

**Adversarial Contexts:**
- Contexts are chosen by an adversary
- More challenging theoretical guarantees
- Common in online advertising

**Contextual Linear Bandits:**
- Feature vectors are linear combinations of context and arm features
- $`x_{a_t, t} = \phi(c_t, a_t)`$ where $`\phi`$ is a known feature map

**General Contextual Bandits:**
- No specific structure on reward functions
- Requires more sophisticated algorithms (e.g., neural bandits)

## Fundamental Algorithms

### 1. Contextual UCB (CUCB)

Contextual UCB extends the UCB principle to handle changing contexts by maintaining context-dependent confidence intervals.

**Algorithm:**
```math
a_t = \arg\max_{i} \left(\langle \hat{\theta}_t, x_{i, t} \rangle + \alpha \sqrt{x_{i, t}^T A_t^{-1} x_{i, t}}\right)
```

Where:
- $`\hat{\theta}_t`$: Least squares estimate of $`\theta^*`$
- $`A_t = \lambda I + \sum_{s=1}^{t-1} x_{a_s, s} x_{a_s, s}^T`$: Design matrix
- $`\alpha`$: Exploration parameter (typically $`\alpha = \sqrt{d \log T}`$)

**Intuition:**
- **Exploitation term**: $`\langle \hat{\theta}_t, x_{i, t} \rangle`$ (choose arm with high predicted reward in current context)
- **Exploration term**: $`\alpha \sqrt{x_{i, t}^T A_t^{-1} x_{i, t}}`$ (choose arm with high uncertainty in current context)
- **Context adaptation**: Confidence intervals adapt to the current context

**Implementation:**
See [`contextual_ucb.py`](contextual_ucb.py) for the complete implementation.

```python
# Key implementation details:
class ContextualUCB:
    # Initialize design matrix A and cumulative rewards b
    # Select arm: argmax(θ̂ᵀx + α√(xᵀA⁻¹x)) for current context
    # Update: A += xxᵀ, b += reward * x
```

### 2. Contextual Thompson Sampling (CTS)

Contextual Thompson Sampling maintains a Gaussian posterior over the parameter vector and samples from this posterior to select actions in each context.

**Algorithm:**
1. Maintain Gaussian posterior: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, A_t^{-1})`$
2. Sample parameter: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, \sigma^2 A_t^{-1})`$
3. Choose action: $`a_t = \arg\max_i \langle \theta_t, x_{i, t} \rangle`$

**Implementation:**
See [`contextual_thompson_sampling.py`](contextual_thompson_sampling.py) for the complete implementation.

```python
# Key implementation details:
class ContextualThompsonSampling:
    # Maintain Gaussian posterior over parameter vector
    # Sample from posterior to select arms in current context
    # Update posterior based on observed rewards
```

### 3. Neural Contextual Bandits

Neural contextual bandits use deep neural networks to model complex, non-linear reward functions that depend on context.

**Algorithm:**
1. Maintain neural network model for reward prediction
2. Use uncertainty quantification (e.g., dropout, ensemble methods)
3. Select actions based on predicted rewards and uncertainty

**Implementation:**
See [`neural_contextual_bandit.py`](neural_contextual_bandit.py) for the complete implementation.

```python
# Key implementation details:
class NeuralContextualBandit:
    # Use deep neural networks to model complex reward functions
    # Maintain uncertainty quantification through dropout
    # Select actions based on predicted rewards and uncertainty
```

### 4. LinUCB with Disjoint Models

LinUCB with disjoint models maintains separate linear models for each arm, allowing for more flexible context-arm interactions.

**Algorithm:**
For each arm $`i`$, maintain separate parameter estimate $`\hat{\theta}_i`$:

```math
a_t = \arg\max_{i} \left(\langle \hat{\theta}_{i, t}, x_t \rangle + \alpha \sqrt{x_t^T A_{i, t}^{-1} x_t}\right)
```

**Implementation:**
See [`disjoint_linucb.py`](disjoint_linucb.py) for the complete implementation.

```python
# Key implementation details:
class DisjointLinUCB:
    # Maintain separate linear models for each arm
    # Allow flexible context-arm interactions
    # Update only the model for the chosen arm
```

## Theoretical Analysis

### Regret Bounds

**Contextual UCB Regret Analysis:**
For contextual UCB with appropriate exploration parameter:

```math
\mathbb{E}[R(T)] \leq O(d\sqrt{T \log T})
```

**Key Insights:**
- **Dimension-dependent**: Regret scales with feature dimension $`d`$
- **Context adaptation**: Algorithm adapts to changing contexts
- **Sublinear growth**: Achieves $`O(\sqrt{T})`$ regret

**Neural Bandit Bounds:**
For neural contextual bandits with $`L`$-layer networks:

```math
\mathbb{E}[R(T)] \leq O(\sqrt{T \log T} \cdot \text{poly}(L, d))
```

### Context Assumptions

**Stochastic Contexts:**
- Contexts drawn from fixed distribution
- Allows for more optimistic bounds
- Common in recommendation systems

**Adversarial Contexts:**
- Contexts chosen by adversary
- More challenging theoretical guarantees
- Common in online advertising

**Contextual Gap:**
The contextual gap $`\Delta_t(i) = \max_j \langle \theta^*, x_{j, t} \rangle - \langle \theta^*, x_{i, t} \rangle`$ measures the suboptimality of arm $`i`$ in context $`t`$.

### Sample Complexity

**Contextual Best Arm Identification:**
For identifying the best arm in each context:

```math
\mathbb{E}[N] \leq O\left(\frac{d^2 \log T}{\Delta_{\min}^2}\right)
```

Where $`\Delta_{\min}`$ is the minimum contextual gap across all contexts.

## Practical Considerations

### Context Engineering

**Feature Extraction:**
See [`context_utils.py`](context_utils.py) for the complete implementation.

```python
# Key functionality:
def extract_user_features(user_data):
    # Extract demographics, behavioral, and contextual features
    
def extract_item_features(item_data):
    # Extract item characteristics and content features
```

**Context-Arm Feature Combination:**
See [`context_utils.py`](context_utils.py) for the complete implementation.

```python
# Key functionality:
def combine_context_arm_features(context, arm_features):
    # Combine context and arm features (concatenation or interactions)
    
def create_contextual_features(context, all_arms):
    # Create contextual features for all arms
```

### Parameter Tuning

**Exploration Parameter ($`\alpha`$):**
- **Theoretical**: $`\alpha = \sqrt{d \log T}`$
- **Practical**: $`\alpha = 1.0`$ often works well
- **Context-dependent**: May need different values for different contexts

**Regularization ($`\lambda`$):**
- **Ridge regression**: $`\lambda = 1.0`$ is often sufficient
- **Feature scaling**: Adjust based on feature magnitudes
- **Numerical stability**: Larger $`\lambda`$ for ill-conditioned problems

### Context Adaptation

**Context Clustering:**
```python
# Key functionality:
class ContextualBanditWithClustering:
    # Assign contexts to clusters for efficient learning
    # Maintain separate models for each cluster
    # Update only the relevant cluster model
```

## Advanced Topics

### Contextual Bandits with Constraints

**Resource Constraints:**
- **Budget constraints**: Limited total cost across contexts
- **Time constraints**: Limited time horizon
- **Safety constraints**: Ensure safe exploration in each context

**Algorithms:**
- **Constrained Contextual UCB**: Add constraint handling to CUCB
- **Safe contextual exploration**: Ensure constraints are satisfied
- **Multi-objective contextual bandits**: Balance multiple objectives

### Non-stationary Contextual Bandits

**Problem:**
Contexts and reward functions change over time, requiring adaptation.

**Solutions:**
- **Sliding window**: Only use recent observations
- **Discounting**: Give more weight to recent contexts
- **Change detection**: Detect when contexts change significantly

### Federated Contextual Bandits

**Distributed Learning:**
- Multiple agents learning from different contexts
- Privacy-preserving learning
- Communication-efficient algorithms

**Applications:**
- **Mobile recommendation**: Learn across devices
- **Healthcare**: Learn across hospitals while preserving privacy
- **E-commerce**: Learn across different regions

## Applications

### Online Advertising

**Ad Selection with User Context:**
See [`contextual_applications.py`](contextual_applications.py) for the complete implementation.

```python
# Key functionality:
class ContextualAdSelector:
    # Use Contextual UCB for ad selection
    # Combine user context with ad features
    # Update model with click feedback
```

### Recommendation Systems

**Personalized Content Recommendation:**
See [`contextual_applications.py`](contextual_applications.py) for the complete implementation.

```python
# Key functionality:
class PersonalizedRecommender:
    # Use Contextual Thompson Sampling for recommendations
    # Combine user context with item features
    # Update model with user engagement
```

### Clinical Trials

**Adaptive Treatment Assignment:**
See [`contextual_applications.py`](contextual_applications.py) for the complete implementation.

```python
# Key functionality:
class AdaptiveClinicalTrial:
    # Use Neural Contextual Bandit for treatment assignment
    # Convert patient features to contextual features
    # Update model with treatment outcomes
```

### Dynamic Pricing

**Price Optimization with Customer Features:**
See [`contextual_applications.py`](contextual_applications.py) for the complete implementation.

```python
# Key functionality:
class DynamicPricer:
    # Use Contextual UCB for dynamic pricing
    # Create contextual features for each price level
    # Update model with purchase decisions
```

## Implementation Examples

### Complete Contextual Bandit Environment

See [`contextual_bandit_environment.py`](contextual_bandit_environment.py) for the complete implementation.

```python
# Key components:
class ContextualBanditEnvironment:
    # Contextual bandit environment with dynamic contexts
    # Generate contexts and arm features
    # Calculate context-dependent regret

def run_contextual_bandit_experiment(env, algorithm, T=1000, n_arms=10):
    # Run contextual bandit experiment
    # Generate contexts, select arms, observe rewards
```

### Algorithm Comparison

See [`contextual_bandit_environment.py`](contextual_bandit_environment.py) for the complete implementation.

```python
# Key functionality:
def compare_contextual_algorithms(env, algorithms, T=1000, n_arms=10, n_runs=50):
    # Compare different contextual bandit algorithms
    # Run multiple independent trials
    # Return average regrets for each algorithm

# Example usage:
# d = 10, theta_star = normalized random vector
# context_generator = function that generates contexts
# algorithms = {'Contextual UCB': ..., 'Contextual TS': ..., 'Neural Bandit': ...}
# results = compare_contextual_algorithms(env, algorithms)
```

### Visualization

See [`contextual_bandit_environment.py`](contextual_bandit_environment.py) for the complete implementation.

```python
# Key functionality:
def plot_contextual_bandit_results(results, T):
    # Plot cumulative regret comparison
    # Plot regret rate (regret / sqrt(t))
    # Include legend and proper labeling

# Usage:
# plot_contextual_bandit_results(results, 1000)
```

## Summary

Contextual bandits provide a powerful framework for adaptive decision-making in dynamic environments. Key insights include:

1. **Context Adaptation**: Algorithms adapt to changing contexts and user characteristics
2. **Personalization**: Decisions are tailored to specific contexts and users
3. **Theoretical Guarantees**: Sublinear regret bounds with context-dependent analysis
4. **Practical Algorithms**: CUCB, CTS, and neural bandits for different scenarios
5. **Wide Applicability**: From online advertising to clinical trials

Contextual bandits bridge the gap between classical bandits and full reinforcement learning, providing both theoretical guarantees and practical effectiveness for real-world applications.

## Further Reading

- **Contextual Bandits Survey**: Comprehensive survey by Ambuj Tewari
- **Linear Contextual Bandits**: Theoretical foundations and algorithms
- **Neural Contextual Bandits**: Deep learning approaches
- **Online Learning Resources**: Stanford CS234, UC Berkeley CS285

---

**Note**: This guide covers the fundamentals of contextual bandits. For more advanced topics, see the sections on Neural Bandits, Federated Bandits, and Constrained Contextual Bandits.

## From Cumulative Reward to Pure Exploration

We've now explored **contextual bandits** - extending the bandit framework to handle dynamic contexts where the optimal action depends on the current state. We've seen how contextual UCB and contextual Thompson sampling adapt to changing environments, how personalization enables tailored decisions, and how these methods handle the complexity of real-world applications.

However, while traditional bandit algorithms focus on maximizing cumulative reward through exploration-exploitation balance, **many real-world scenarios** prioritize accurate identification over immediate performance. Consider a clinical trial where the goal is to identify the most effective treatment, or an A/B test where the objective is to determine the best website design - in these cases, we care more about making the right final decision than about maximizing rewards during the learning process.

This motivates our exploration of **best arm identification (BAI)** - a fundamental shift in the bandit paradigm from cumulative reward maximization to pure exploration. We'll see how BAI algorithms focus exclusively on identifying the best arm with high confidence, how they use different stopping criteria and sampling strategies, and how these methods enable efficient identification in scenarios where accuracy is more important than immediate performance.

The transition from contextual bandits to best arm identification represents the bridge from adaptive learning to pure exploration - taking our understanding of bandit algorithms and applying it to scenarios where the goal is identification rather than cumulative reward maximization.

In the next section, we'll explore best arm identification, understanding how to design algorithms for pure exploration problems.

---

**Previous: [Linear Bandits](02_linear_bandits.md)** - Learn how to exploit structured action spaces for more efficient learning.

**Next: [Best Arm Identification](04_best_arm_identification.md)** - Learn algorithms for pure exploration and identification problems.
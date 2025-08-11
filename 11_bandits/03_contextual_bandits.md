# Contextual Bandits

## Introduction

Contextual bandits represent a natural evolution of multi-armed bandits by incorporating dynamic context information that influences reward distributions. Unlike classical bandits where arms have fixed reward distributions, contextual bandits adapt to changing environments where the optimal action depends on the current context.

### The Big Picture: What are Contextual Bandits?

**The Contextual Bandit Problem:**
Imagine you're a personal shopper who needs to recommend products to different customers. Each customer has unique preferences (context), and the same product might work great for one customer but poorly for another. You need to learn which products work best for which types of customers. This is the contextual bandit problem - adapting your recommendations based on the current context.

**The Intuitive Analogy:**
Think of contextual bandits like a smart thermostat. The thermostat doesn't just set one temperature - it adapts based on context: time of day, whether someone is home, the weather outside, etc. Similarly, contextual bandits don't just choose the same action every time - they adapt their choices based on the current situation.

**Why Contextual Bandits Matter:**
- **Personalization**: Tailor decisions to individual users
- **Adaptation**: Respond to changing environments
- **Real-world relevance**: Most practical problems involve context
- **Efficiency**: Learn faster by exploiting context structure

### The Key Insight

**From Static to Dynamic Decision Making:**
- **Classical bandits**: Same action works best for everyone
- **Linear bandits**: Actions have features, but context is fixed
- **Contextual bandits**: Best action depends on current context

**The Personalization Advantage:**
- **Without context**: "Show everyone the same ad"
- **With context**: "Show tech ads to young users, luxury ads to high-income users"

## From Static Features to Dynamic Contexts

We've now explored **linear bandits** - extending the bandit framework to handle structured action spaces where rewards are linear functions of arm features. We've seen how LinUCB and linear Thompson sampling can exploit arm similarities, how feature-based learning reduces sample complexity, and how these methods enable efficient learning in high-dimensional action spaces.

However, while linear bandits leverage the structure of arm features, **real-world environments** are often dynamic and context-dependent. Consider an online advertising system where the effectiveness of an ad depends not just on the ad itself, but also on the user's current context - their demographics, browsing history, time of day, and other contextual factors.

This motivates our exploration of **contextual bandits** - extending the bandit framework to handle dynamic contexts where the optimal action depends on the current state. We'll see how contextual UCB and contextual Thompson sampling adapt to changing environments, how personalization enables tailored decisions, and how these methods handle the complexity of real-world applications.

The transition from linear to contextual bandits represents the bridge from static structure to dynamic adaptation - taking our understanding of feature-based learning and extending it to handle the temporal and contextual dynamics inherent in many real-world problems.

In this section, we'll explore contextual bandits, understanding how they adapt to changing contexts and enable personalized decision-making.

### Understanding the Motivation

**The Static World Limitation:**
Linear bandits assume that the relationship between features and rewards is fixed. But in reality, this relationship often depends on context.

**The Contextual Bandit Solution:**
By incorporating context information, we can learn how the optimal action changes with the situation.

**The Personalization Advantage:**
- **Learning user preferences**: Different users like different things
- **Adapting to time**: Morning vs. evening preferences
- **Responding to environment**: Weather, location, device type
- **Handling dynamics**: Preferences that change over time

### Key Applications

Contextual bandits are essential in scenarios where:
- **User context matters**: Different users respond differently to the same action
- **Temporal dynamics**: Optimal actions change over time
- **Personalization**: Tailoring decisions to individual characteristics
- **Adaptive systems**: Learning and adapting to changing environments

**Real-World Examples:**
- **Online advertising**: Show different ads to different users
- **Recommendation systems**: Recommend content based on user context
- **Clinical trials**: Adapt treatments based on patient characteristics
- **Dynamic pricing**: Adjust prices based on customer context

## Problem Formulation

### Understanding the Problem Setup

**The Basic Scenario:**
At each time step, you observe a context (like user information), then choose an action from a set of available actions. The reward depends on both the action and the context.

**Key Questions:**
- How do you learn the relationship between context, actions, and rewards?
- How do you adapt your strategy as contexts change?
- How do you balance exploration and exploitation in different contexts?

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

**Intuitive Understanding:**
This says: "The reward depends on both the action and the current context. The feature vector combines information about both the action and the context."

**The Learning Process:**
1. **Observe context**: See current user/situation information
2. **Choose action**: Pick action based on context and current knowledge
3. **Observe reward**: Get feedback for the chosen action in this context
4. **Update model**: Use reward to improve understanding of context-action relationships
5. **Repeat**: Continue adapting to new contexts

**Example: Online Advertising**
- Context: [user_age, user_location, time_of_day, device_type]
- Action: [ad_category, ad_size, ad_color]
- Feature: [user_age, user_location, time_of_day, device_type, ad_category, ad_size, ad_color]
- Reward: Click (1) or no click (0)

### Key Assumptions

**Contextual Reward Structure:**
- Rewards are linear in context-arm features: $`\mathbb{E}[r_t | a_t, c_t] = \langle \theta^*, x_{a_t, t} \rangle`$
- The parameter vector $`\theta^*`$ is unknown but fixed
- Feature vectors $`x_{i, t}`$ depend on both arm $`i`$ and context $`c_t`$

**Intuitive Understanding:**
- **Linear relationship**: Context and action features combine linearly to determine rewards
- **Fixed parameters**: The underlying relationship doesn't change (though contexts do)
- **Context-dependent features**: The same action looks different in different contexts

**Context Generation:**
- Contexts $`c_t`$ may be drawn from a distribution or chosen by an adversary
- Contexts can be arbitrary (adversarial) or have structure (stochastic)
- Feature vectors $`x_{i, t}`$ are revealed for all arms $`i`$ at time $`t`$

**Intuitive Understanding:**
- **Stochastic contexts**: Like users arriving randomly with different characteristics
- **Adversarial contexts**: Like someone trying to make your algorithm perform poorly
- **Revealed features**: You can see what all actions would look like in the current context

**Noise Assumptions:**
- Noise terms $`\eta_t`$ are conditionally sub-Gaussian given the history
- $`\mathbb{E}[\eta_t | \mathcal{F}_{t-1}, a_t, c_t] = 0`$ (zero mean)
- $`\mathbb{E}[\exp(\lambda \eta_t) | \mathcal{F}_{t-1}, a_t, c_t] \leq \exp(\frac{\lambda^2 \sigma^2}{2})`$ (sub-Gaussian)

**Intuitive Understanding:**
- **Zero mean**: Noise doesn't systematically favor any context-action combination
- **Sub-Gaussian**: Noise is well-behaved and not too extreme
- **Conditional independence**: Noise doesn't depend on previous choices or contexts

### Problem Variants

**Stochastic Contexts:**
- Contexts are drawn from a fixed distribution
- Allows for more optimistic regret bounds
- Common in recommendation systems
- **Intuitive**: Like users with predictable characteristics

**Adversarial Contexts:**
- Contexts are chosen by an adversary
- More challenging theoretical guarantees
- Common in online advertising
- **Intuitive**: Like someone trying to make your algorithm fail

**Contextual Linear Bandits:**
- Feature vectors are linear combinations of context and arm features
- $`x_{a_t, t} = \phi(c_t, a_t)`$ where $`\phi`$ is a known feature map
- **Intuitive**: Like combining user features and product features in a structured way

**General Contextual Bandits:**
- No specific structure on reward functions
- Requires more sophisticated algorithms (e.g., neural bandits)
- **Intuitive**: Like complex, non-linear relationships between context, actions, and rewards

## Fundamental Algorithms

### Understanding Algorithm Design

**The Algorithm Challenge:**
How do you design a strategy that learns context-action relationships efficiently while adapting to changing contexts?

**Key Principles:**
- **Context adaptation**: Learn how actions perform in different contexts
- **Uncertainty quantification**: Measure uncertainty in context-action relationships
- **Dynamic exploration**: Explore more in unfamiliar contexts
- **Personalization**: Tailor decisions to current context

### 1. Contextual UCB (CUCB)

Contextual UCB extends the UCB principle to handle changing contexts by maintaining context-dependent confidence intervals.

**Intuitive Understanding:**
Contextual UCB is like a smart personal assistant who learns your preferences in different situations. When you're at work, they recommend professional content. When you're relaxing, they recommend entertainment. They're always learning and adapting to your current context.

**Algorithm:**
```math
a_t = \arg\max_{i} \left(\langle \hat{\theta}_t, x_{i, t} \rangle + \alpha \sqrt{x_{i, t}^T A_t^{-1} x_{i, t}}\right)
```

Where:
- $`\hat{\theta}_t`$: Least squares estimate of $`\theta^*`$
- $`A_t = \lambda I + \sum_{s=1}^{t-1} x_{a_s, s} x_{a_s, s}^T`$: Design matrix
- $`\alpha`$: Exploration parameter (typically $`\alpha = \sqrt{d \log T}`$)

**Breaking Down the Formula:**
- **$`\langle \hat{\theta}_t, x_{i, t} \rangle`$**: Predicted reward for action i in current context (exploitation)
- **$`\alpha \sqrt{x_{i, t}^T A_t^{-1} x_{i, t}}`$**: Uncertainty bonus for action i in current context (exploration)
- **Context adaptation**: Both terms depend on the current context

**Why This Works:**
- **Context-aware exploitation**: Choose actions that work well in the current context
- **Context-aware exploration**: Explore more in contexts we haven't seen much
- **Adaptive learning**: Learn faster by exploiting context structure
- **Personalization**: Naturally adapts to different user types

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

**Intuitive Understanding:**
Contextual Thompson Sampling is like a Bayesian personal trainer who maintains beliefs about how different exercises work for different people. They sample from their beliefs to decide what exercise to recommend to you today, then update their beliefs based on how it goes.

**Algorithm:**
1. Maintain Gaussian posterior: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, A_t^{-1})`$
2. Sample parameter: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, \sigma^2 A_t^{-1})`$
3. Choose action: $`a_t = \arg\max_i \langle \theta_t, x_{i, t} \rangle`$

**The Bayesian Approach:**
- **Prior beliefs**: Start with initial beliefs about context-action relationships
- **Sample**: Draw a random parameter vector from current beliefs
- **Choose**: Pick the action that looks best in current context with this parameter
- **Update**: Update beliefs based on observed reward
- **Repeat**: Continue sampling and updating

**Why Sampling Works:**
- **Context exploration**: Sampling explores different context-action relationships
- **Context exploitation**: Sampling from confident parameters picks good actions
- **Natural adaptation**: Automatically adapts to changing contexts
- **Uncertainty handling**: Naturally handles uncertainty in context-action relationships

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

**Intuitive Understanding:**
Neural contextual bandits are like a sophisticated AI that can learn complex patterns in how different users respond to different actions. Instead of assuming simple linear relationships, they can capture complex, non-linear interactions between context and actions.

**Algorithm:**
1. Maintain neural network model for reward prediction
2. Use uncertainty quantification (e.g., dropout, ensemble methods)
3. Select actions based on predicted rewards and uncertainty

**Why Neural Networks Help:**
- **Complex patterns**: Can learn non-linear context-action relationships
- **Feature learning**: Automatically learn useful features from raw context
- **Scalability**: Can handle high-dimensional contexts
- **Flexibility**: Can model complex reward functions

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

**Intuitive Understanding:**
Disjoint LinUCB is like having separate experts for each action. Each expert learns how their specific action performs in different contexts, without sharing information with other experts.

**Algorithm:**
For each arm $`i`$, maintain separate parameter estimate $`\hat{\theta}_i`$:

```math
a_t = \arg\max_{i} \left(\langle \hat{\theta}_{i, t}, x_t \rangle + \alpha \sqrt{x_t^T A_{i, t}^{-1} x_t}\right)
```

**Why Disjoint Models Help:**
- **Flexible interactions**: Each arm can have different context relationships
- **No interference**: Learning about one arm doesn't affect others
- **Simplicity**: Easier to implement and understand
- **Robustness**: Less sensitive to model misspecification

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

### Understanding Regret Analysis

**The Analysis Challenge:**
How do we mathematically analyze how well contextual bandit algorithms perform? What guarantees can we provide?

**Key Questions:**
- How does regret grow with time and context complexity?
- How does regret depend on context assumptions?
- What are the fundamental limits of any algorithm?

### Regret Bounds

**Contextual UCB Regret Analysis:**
For contextual UCB with appropriate exploration parameter:

```math
\mathbb{E}[R(T)] \leq O(d\sqrt{T \log T})
```

**Intuitive Understanding:**
This bound says: "Your expected regret grows as the square root of time, multiplied by the feature dimension and a logarithmic factor."

**Breaking Down the Bound:**
- **$`\sqrt{T}`$**: Regret grows sublinearly with time
- **$`d`$**: Regret scales with feature dimension
- **$`\log T`$**: Logarithmic factor from confidence intervals

**Key Insights:**
- **Dimension-dependent**: Regret scales with feature dimension $`d`$
- **Context adaptation**: Algorithm adapts to changing contexts
- **Sublinear growth**: Achieves $`O(\sqrt{T})`$ regret

**Neural Bandit Bounds:**
For neural contextual bandits with $`L`$-layer networks:

```math
\mathbb{E}[R(T)] \leq O(\sqrt{T \log T} \cdot \text{poly}(L, d))
```

**Intuitive Understanding:**
Neural bandits can achieve similar regret bounds, but the constants depend on network complexity.

### Context Assumptions

**Stochastic Contexts:**
- Contexts drawn from fixed distribution
- Allows for more optimistic bounds
- Common in recommendation systems
- **Intuitive**: Like users with predictable characteristics

**Adversarial Contexts:**
- Contexts chosen by adversary
- More challenging theoretical guarantees
- Common in online advertising
- **Intuitive**: Like someone trying to make your algorithm fail

**Contextual Gap:**
The contextual gap $`\Delta_t(i) = \max_j \langle \theta^*, x_{j, t} \rangle - \langle \theta^*, x_{i, t} \rangle`$ measures the suboptimality of arm $`i`$ in context $`t`$.

**Intuitive Understanding:**
The contextual gap measures how much worse an action is compared to the best action in a specific context. It's like measuring how much you lose by choosing the wrong restaurant for a specific occasion.

### Sample Complexity

**Contextual Best Arm Identification:**
For identifying the best arm in each context:

```math
\mathbb{E}[N] \leq O\left(\frac{d^2 \log T}{\Delta_{\min}^2}\right)
```

Where $`\Delta_{\min}`$ is the minimum contextual gap across all contexts.

**Intuitive Understanding:**
This tells us how many samples we need to identify the best action in each context with high confidence.

## Practical Considerations

### Understanding Practical Challenges

**The Implementation Challenge:**
How do you implement contextual bandit algorithms in practice? What issues arise in real-world applications?

**Key Considerations:**
- **Context engineering**: How to design good context features
- **Feature combination**: How to combine context and action features
- **Parameter tuning**: How to set algorithm parameters
- **Context adaptation**: How to handle changing contexts

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

**Intuitive Understanding:**
Good context features are like good predictors of user behavior. Age, location, time of day, and browsing history are much better predictors than random numbers.

**Context-Arm Feature Combination:**
See [`context_utils.py`](context_utils.py) for the complete implementation.

```python
# Key functionality:
def combine_context_arm_features(context, arm_features):
    # Combine context and arm features (concatenation or interactions)
    
def create_contextual_features(context, all_arms):
    # Create contextual features for all arms
```

**Why Feature Combination Matters:**
- **Concatenation**: Simple but may miss interactions
- **Interactions**: Can capture complex context-action relationships
- **Feature engineering**: Domain knowledge can improve performance

### Parameter Tuning

**Exploration Parameter ($`\alpha`$):**
- **Theoretical**: $`\alpha = \sqrt{d \log T}`$
- **Practical**: $`\alpha = 1.0`$ often works well
- **Context-dependent**: May need different values for different contexts

**Intuitive Understanding:**
- **Higher α**: More exploration, more conservative
- **Lower α**: Less exploration, more aggressive
- **Context-dependent**: Some contexts may need more exploration than others

**Regularization ($`\lambda`$):**
- **Ridge regression**: $`\lambda = 1.0`$ is often sufficient
- **Feature scaling**: Adjust based on feature magnitudes
- **Numerical stability**: Larger $`\lambda`$ for ill-conditioned problems

**Intuitive Understanding:**
Regularization prevents overfitting to specific contexts and ensures numerical stability.

### Context Adaptation

**Context Clustering:**
```python
# Key functionality:
class ContextualBanditWithClustering:
    # Assign contexts to clusters for efficient learning
    # Maintain separate models for each cluster
    # Update only the relevant cluster model
```

**Intuitive Understanding:**
Instead of learning a separate model for every possible context, group similar contexts together and learn shared models.

## Advanced Topics

### Contextual Bandits with Constraints

**Resource Constraints:**
- **Budget constraints**: Limited total cost across contexts
- **Time constraints**: Limited time horizon
- **Safety constraints**: Ensure safe exploration in each context

**Intuitive Understanding:**
Real-world problems often have constraints. You can't spend unlimited money on ads, or you need to ensure treatments are safe for patients.

**Algorithms:**
- **Constrained Contextual UCB**: Add constraint handling to CUCB
- **Safe contextual exploration**: Ensure constraints are satisfied
- **Multi-objective contextual bandits**: Balance multiple objectives

### Non-stationary Contextual Bandits

**Problem:**
Contexts and reward functions change over time, requiring adaptation.

**Intuitive Understanding:**
Like user preferences that change over time, or market conditions that evolve.

**Solutions:**
- **Sliding window**: Only use recent observations
- **Discounting**: Give more weight to recent contexts
- **Change detection**: Detect when contexts change significantly

### Federated Contextual Bandits

**Distributed Learning:**
- Multiple agents learning from different contexts
- Privacy-preserving learning
- Communication-efficient algorithms

**Intuitive Understanding:**
Like multiple stores learning customer preferences while keeping data private, or multiple hospitals learning treatment effectiveness while preserving patient privacy.

**Applications:**
- **Mobile recommendation**: Learn across devices
- **Healthcare**: Learn across hospitals while preserving privacy
- **E-commerce**: Learn across different regions

## Applications

### Understanding Real-World Applications

**The Application Challenge:**
How do contextual bandit algorithms apply to real-world problems? What are the practical considerations?

**Key Applications:**
- **Online advertising**: Personalize ad selection
- **Recommendation systems**: Personalize content recommendations
- **Clinical trials**: Adapt treatments to patients
- **Dynamic pricing**: Adjust prices based on customer context

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

**Intuitive Understanding:**
Like choosing which ad to show based on user demographics, browsing history, and time of day. Learning that "tech ads work well with young users during work hours" helps optimize ad selection.

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

**Intuitive Understanding:**
Like Netflix recommending different movies based on your viewing history, time of day, and device type. Learning that "you prefer action movies in the evening on your TV" helps provide better recommendations.

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

**Intuitive Understanding:**
Like assigning treatments based on patient characteristics (age, symptoms, medical history) and treatment features (dosage, frequency, type). Learning that "high-dose treatments work better for severe cases" helps optimize treatment assignment.

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

**Intuitive Understanding:**
Like adjusting prices based on customer characteristics, time of day, and product features. Learning that "premium customers are willing to pay more for luxury items" helps optimize pricing.

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

**Key Takeaways:**
- Contextual bandits adapt decisions based on changing contexts
- Personalization enables tailored decisions for different users and situations
- Contextual UCB, Thompson Sampling, and neural bandits provide different trade-offs
- Theoretical guarantees show sublinear regret with context adaptation
- Practical implementation requires careful context engineering and feature combination

**The Broader Impact:**
Contextual bandits have fundamentally changed how we approach personalized decision-making by:
- **Enabling personalization**: Tailoring decisions to individual users and contexts
- **Supporting adaptive systems**: Learning and adapting to changing environments
- **Providing theoretical foundations**: Rigorous guarantees for contextual learning
- **Enabling real-world applications**: Practical tools for personalized recommendations

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
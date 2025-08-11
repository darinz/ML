# Linear Bandits

## Introduction

Linear bandits represent a significant extension of classical multi-armed bandits by incorporating structured information about the arms. Instead of treating each arm as an independent entity, linear bandits assume that rewards are linear functions of arm features, enabling more efficient learning through information sharing between similar arms.

### The Big Picture: What are Linear Bandits?

**The Linear Bandit Problem:**
Imagine you're choosing between 1000 different restaurants, each described by features like "price range," "cuisine type," "distance," and "rating." Instead of trying each restaurant independently, you can learn that "expensive Italian restaurants near downtown tend to be good." This is the linear bandit problem - learning patterns in features to make better decisions.

**The Intuitive Analogy:**
Think of linear bandits like learning to predict house prices. Instead of learning the price of each house individually, you learn that "houses with more bedrooms, in better neighborhoods, tend to cost more." Once you learn this pattern, you can predict the price of any house based on its features.

**Why Linear Bandits Matter:**
- **Efficiency**: Learn patterns instead of individual values
- **Generalization**: Predict rewards for new arms you've never seen
- **Scalability**: Handle thousands of arms with few features
- **Structure**: Exploit similarities between arms

### The Key Insight

**From Independent to Structured Learning:**
- **Classical bandits**: Each arm is independent, no information sharing
- **Linear bandits**: Arms share structure through features, enabling information sharing

**The Efficiency Gain:**
- **Classical**: Need to try each arm many times (scales with K)
- **Linear**: Learn the pattern, apply to all arms (scales with d)

## From Independent Arms to Structured Learning

We've now explored **classical multi-armed bandits** - the foundational framework for sequential decision-making under uncertainty. We've seen how the exploration-exploitation trade-off manifests in various algorithms like epsilon-greedy, UCB, and Thompson sampling, and how these methods provide theoretical guarantees for regret minimization.

However, while classical bandits treat each arm independently, **real-world problems** often exhibit structure that can be exploited for more efficient learning. Consider a recommendation system where products have features - similar products likely have similar reward distributions, and learning about one product can inform our understanding of similar products.

This motivates our exploration of **linear bandits** - extending the bandit framework to handle structured action spaces where rewards are linear functions of arm features. We'll see how LinUCB and linear Thompson sampling can exploit arm similarities, how feature-based learning reduces sample complexity, and how these methods enable efficient learning in high-dimensional action spaces.

The transition from classical to linear bandits represents the bridge from independent learning to structured learning - taking our understanding of the exploration-exploitation trade-off and extending it to leverage the structure inherent in many real-world problems.

In this section, we'll explore linear bandits, understanding how they exploit arm similarities and enable more efficient learning through feature-based representations.

### Understanding the Motivation

**The Classical Bandit Limitation:**
In classical bandits, if you have 1000 arms, you need to try each arm many times to learn its reward. This is inefficient when arms are similar.

**The Linear Bandit Solution:**
If arms can be described by features (like restaurant features), you can learn the relationship between features and rewards, then apply this knowledge to all arms.

**The Information Sharing Advantage:**
- **Learning about one expensive Italian restaurant** helps you understand all expensive Italian restaurants
- **Learning about one action movie** helps you understand all action movies
- **Learning about one treatment** helps you understand similar treatments

### Key Applications

Linear bandits are particularly valuable in scenarios with:
- **Large action spaces**: Too many arms to explore independently
- **Structured actions**: Arms can be described by feature vectors
- **Contextual information**: User/context features influence rewards
- **Cold-start problems**: New arms can be evaluated based on features

**Real-World Examples:**
- **Recommendation systems**: Products with features (genre, price, rating)
- **Online advertising**: Ads with features (category, size, color)
- **Clinical trials**: Treatments with features (dosage, frequency, duration)
- **Resource allocation**: Options with features (cost, benefit, risk)

## Problem Formulation

### Understanding the Problem Setup

**The Basic Scenario:**
Instead of having K independent arms, you have K arms described by d-dimensional feature vectors. The reward for each arm is a linear combination of its features plus noise.

**Key Questions:**
- How do you learn the relationship between features and rewards?
- How do you balance exploration and exploitation in feature space?
- How do you handle the uncertainty in your parameter estimates?

### Mathematical Setup

Linear bandits extend the classical setting by assuming rewards are linear functions of arm features:

```math
r_t = \langle \theta^*, x_{a_t} \rangle + \eta_t
```

Where:
- $`\theta^* \in \mathbb{R}^d`$: Unknown parameter vector (true reward model)
- $`x_{a_t} \in \mathbb{R}^d`$: Feature vector of chosen arm at time $`t`$
- $`\eta_t`$: Noise term (typically sub-Gaussian)
- $`a_t \in \{1, 2, \ldots, K\}`$: Chosen arm at time $`t`$

**Intuitive Understanding:**
This says: "The reward for an arm is a weighted sum of its features, where the weights (θ*) are unknown and we need to learn them."

**The Learning Process:**
1. **Start**: Know nothing about the parameter vector θ*
2. **Choose**: Pick an arm based on current parameter estimate
3. **Observe**: Get reward from chosen arm
4. **Update**: Use reward to improve parameter estimate
5. **Repeat**: Continue learning the parameter vector

**Example: Restaurant Features**
- Features: [price_level, cuisine_type, distance, rating]
- Parameter: [0.5, 0.3, -0.2, 0.8] (weights for each feature)
- Reward: 0.5×price + 0.3×cuisine + (-0.2)×distance + 0.8×rating + noise

### Key Assumptions

**Linear Reward Structure:**
- Rewards are linear in arm features: $`\mathbb{E}[r_t | a_t] = \langle \theta^*, x_{a_t} \rangle`$
- The parameter vector $`\theta^*`$ is unknown but fixed
- Feature vectors $`x_i`$ are known for all arms $`i`$

**Intuitive Understanding:**
- **Linear relationship**: Features combine additively to determine rewards
- **Fixed parameters**: The relationship doesn't change over time
- **Known features**: We know the features of each arm beforehand

**Noise Assumptions:**
- Noise terms $`\eta_t`$ are conditionally sub-Gaussian
- $`\mathbb{E}[\eta_t | \mathcal{F}_{t-1}] = 0`$ (zero mean)
- $`\mathbb{E}[\exp(\lambda \eta_t) | \mathcal{F}_{t-1}] \leq \exp(\frac{\lambda^2 \sigma^2}{2})`$ (sub-Gaussian)

**Intuitive Understanding:**
- **Zero mean**: Noise doesn't systematically bias rewards up or down
- **Sub-Gaussian**: Noise is well-behaved (not too extreme)
- **Conditional independence**: Noise doesn't depend on previous choices

**Bounded Features:**
- Feature vectors are bounded: $`\|x_i\|_2 \leq L`$ for all arms $`i`$
- Parameter vector is bounded: $`\|\theta^*\|_2 \leq S`$

**Intuitive Understanding:**
- **Bounded features**: No feature can be infinitely large
- **Bounded parameters**: The relationship between features and rewards is reasonable
- **Practical necessity**: Ensures algorithms work properly

### Problem Variants

**Fixed Action Set:**
- Action set $`\mathcal{A} = \{x_1, x_2, \ldots, x_K\}`$ is fixed
- All feature vectors are known in advance
- **Intuitive**: Like having a fixed menu of restaurants

**Continuous Action Set:**
- Action set $`\mathcal{A} \subset \mathbb{R}^d`` is continuous
- Can choose any point in the action set
- More complex but allows infinite action spaces
- **Intuitive**: Like being able to design custom restaurants

**Contextual Linear Bandits:**
- Feature vectors depend on context: $`x_{a_t, t}`$
- Context changes over time
- Requires adaptation to changing contexts
- **Intuitive**: Like restaurants that change their menu based on the day

## Fundamental Algorithms

### Understanding Algorithm Design

**The Algorithm Challenge:**
How do you design a strategy that learns the parameter vector efficiently while balancing exploration and exploitation?

**Key Principles:**
- **Parameter estimation**: Learn θ* from observed rewards
- **Uncertainty quantification**: Measure uncertainty in parameter estimates
- **Optimistic exploration**: Assume unknown parameters might be favorable
- **Feature-based selection**: Choose arms based on feature-parameter interaction

### 1. LinUCB (Linear Upper Confidence Bound)

LinUCB extends the UCB principle to linear bandits by maintaining confidence ellipsoids around the parameter estimate.

**Intuitive Understanding:**
LinUCB is like a cautious scientist who estimates the relationship between features and rewards, but adds an "uncertainty bonus" to account for the fact that their estimate might be wrong.

**Algorithm:**
```math
a_t = \arg\max_{i} \left(\langle \hat{\theta}_t, x_i \rangle + \alpha \sqrt{x_i^T A_t^{-1} x_i}\right)
```

Where:
- $`\hat{\theta}_t`$: Least squares estimate of $`\theta^*`$
- $`A_t = \lambda I + \sum_{s=1}^{t-1} x_{a_s} x_{a_s}^T`$: Design matrix
- $`\alpha`$: Exploration parameter (typically $`\alpha = \sqrt{d \log T}`$)

**Breaking Down the Formula:**
- **$`\langle \hat{\theta}_t, x_i \rangle`$**: Predicted reward based on current parameter estimate (exploitation)
- **$`\alpha \sqrt{x_i^T A_t^{-1} x_i}`$**: Uncertainty bonus based on feature vector (exploration)
- **$`A_t^{-1}`$**: Inverse of design matrix, captures parameter uncertainty

**Why This Works:**
- **High uncertainty features**: Arms with features we haven't seen much get high exploration bonus
- **Low uncertainty features**: Arms with familiar features rely mostly on parameter estimate
- **Automatic balance**: Naturally balances exploration and exploitation

**The Design Matrix Intuition:**
- **$`A_t`$**: Measures how much we've learned about different feature combinations
- **$`A_t^{-1}`$**: Inverse measures our uncertainty about parameter estimates
- **$`x_i^T A_t^{-1} x_i`$**: How uncertain we are about the reward for arm i

**Implementation:**
See [`linucb.py`](linucb.py) for the complete implementation.

```python
# Key implementation details:
class LinUCB:
    # Initialize design matrix A and cumulative rewards b
    # Select arm: argmax(θ̂ᵀx + α√(xᵀA⁻¹x))
    # Update: A += xxᵀ, b += reward * x
```

**Theoretical Guarantees:**
- **Regret Bound**: $`O(d\sqrt{T \log T})`$ with high probability
- **Gap-dependent bound**: $`O(\frac{d^2 \log T}{\Delta_{\min}})`$ where $`\Delta_{\min}`$ is the minimum gap
- **Optimal up to logarithmic factors**

**When to Use:**
- **Theoretical guarantees**: When you need proven performance bounds
- **Stable performance**: When you want predictable behavior
- **Feature-rich problems**: When arms have meaningful features

### 2. Linear Thompson Sampling

Linear Thompson Sampling maintains a Gaussian posterior over the parameter vector and samples from this posterior to select actions.

**Intuitive Understanding:**
Linear Thompson Sampling is like a Bayesian scientist who maintains beliefs about the parameter vector and samples from these beliefs to make decisions. Instead of being optimistic, it's probabilistic.

**Algorithm:**
1. Maintain Gaussian posterior: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, A_t^{-1})`$
2. Sample parameter: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, \sigma^2 A_t^{-1})`$
3. Choose action: $`a_t = \arg\max_i \langle \theta_t, x_i \rangle`$

**The Bayesian Approach:**
- **Prior beliefs**: Start with initial beliefs about parameter vector
- **Sample**: Draw a random parameter vector from current beliefs
- **Choose**: Pick the arm that looks best with this sampled parameter
- **Update**: Update beliefs based on observed reward
- **Repeat**: Continue sampling and updating

**Posterior Update:**
- **Prior**: $`\theta \sim \mathcal{N}(0, \lambda I)`$ (regularization)
- **Likelihood**: $`r_t | \theta, x_{a_t} \sim \mathcal{N}(\langle \theta, x_{a_t} \rangle, \sigma^2)`$
- **Posterior**: $`\theta | \mathcal{F}_t \sim \mathcal{N}(\hat{\theta}_t, A_t^{-1})`$

**Intuitive Understanding:**
- **Gaussian prior**: Start with beliefs that parameters are around zero
- **Gaussian likelihood**: Assume rewards are normally distributed around predicted values
- **Gaussian posterior**: Updated beliefs after seeing new data
- **Sampling**: Draw random parameters to explore uncertainty

**Why Sampling Works:**
- **Exploration**: Sampling from uncertain parameters gives diverse exploration
- **Exploitation**: Sampling from confident parameters usually picks good arms
- **Natural balance**: Uncertainty naturally decreases with more observations
- **Bayesian optimal**: Theoretically optimal under certain assumptions

**Implementation:**
See [`linear_thompson_sampling.py`](linear_thompson_sampling.py) for the complete implementation.

```python
# Key implementation details:
class LinearThompsonSampling:
    # Maintain Gaussian posterior over parameter vector
    # Sample from posterior to select arms
    # Update posterior based on observed rewards
```

**Theoretical Guarantees:**
- **Regret Bound**: $`O(d\sqrt{T \log T})`$ under Bayesian assumptions
- **Often performs better in practice** than LinUCB
- **Automatic exploration** through posterior sampling

**When to Use:**
- **Practical performance**: When you want the best empirical performance
- **Bayesian framework**: When you have prior knowledge about parameters
- **Complex models**: When you want to extend to more sophisticated models

### 3. OFUL (Optimism in the Face of Uncertainty for Linear Bandits)

OFUL is a more sophisticated algorithm that constructs confidence ellipsoids and uses optimism to guide exploration.

**Intuitive Understanding:**
OFUL is like an optimistic explorer who assumes the unknown parameter vector might be the best possible one within their confidence region, then chooses the arm that would be best under this optimistic assumption.

**Algorithm:**
```math
a_t = \arg\max_{i} \max_{\theta \in \mathcal{C}_t} \langle \theta, x_i \rangle
```

Where $`\mathcal{C}_t`$ is the confidence ellipsoid:
```math
\mathcal{C}_t = \{\theta : \|\theta - \hat{\theta}_t\|_{A_t} \leq \beta_t\}
```

**Breaking Down the Algorithm:**
- **$`\mathcal{C}_t`$**: Confidence region where we think θ* might be
- **$`\max_{\theta \in \mathcal{C}_t}`$**: Find the most optimistic parameter in confidence region
- **$`\arg\max_i`$**: Choose the arm that's best under this optimistic parameter

**Why Optimism Works:**
- **Exploration**: Optimistic parameters encourage trying new arms
- **Exploitation**: As confidence region shrinks, optimism becomes realistic
- **Theoretical guarantees**: Provides strong regret bounds
- **Adaptive exploration**: Automatically adjusts exploration based on uncertainty

**Implementation:**
See [`oful.py`](oful.py) for the complete implementation.

```python
# Key implementation details:
class OFUL:
    # Construct confidence ellipsoids around parameter estimate
    # Use optimism to guide exploration
    # Calculate confidence radius based on concentration inequalities
```

## Theoretical Analysis

### Understanding Regret Analysis

**The Analysis Challenge:**
How do we mathematically analyze how well linear bandit algorithms perform? What guarantees can we provide?

**Key Questions:**
- How does regret grow with time and dimension?
- How does regret depend on the problem structure?
- What are the fundamental limits of any algorithm?

### Regret Bounds

**LinUCB Regret Analysis:**
For LinUCB with appropriate exploration parameter $`\alpha`$:

```math
\mathbb{E}[R(T)] \leq O\left(d\sqrt{T \log T}\right)
```

**Intuitive Understanding:**
This bound says: "Your expected regret grows as the square root of time, multiplied by the dimension and a logarithmic factor."

**Breaking Down the Bound:**
- **$`\sqrt{T}`$**: Regret grows sublinearly with time (much better than linear)
- **$`d`$**: Regret scales with feature dimension (not number of arms)
- **$`\log T`$**: Logarithmic factor from confidence interval construction

**Key Insights:**
- **Dimension-dependent**: Regret scales with $`d`$ instead of $`K`$
- **Sublinear growth**: $`O(\sqrt{T})`$ instead of linear
- **Logarithmic factors**: Due to confidence interval construction

**Gap-dependent Bounds:**
For arms with minimum gap $`\Delta_{\min}`$:

```math
\mathbb{E}[R(T)] \leq O\left(\frac{d^2 \log T}{\Delta_{\min}}\right)
```

**Intuitive Understanding:**
When there's a clear difference between the best arm and others, regret grows only logarithmically with time.

### Lower Bounds

**Minimax Lower Bound:**
For any algorithm in linear bandits:

```math
\mathbb{E}[R(T)] \geq \Omega(d\sqrt{T})
```

**Intuitive Understanding:**
No algorithm can do better than this - it's a fundamental limit of the problem.

**Gap-dependent Lower Bound:**
```math
\mathbb{E}[R(T)] \geq \Omega\left(\frac{d \log T}{\Delta_{\min}}\right)
```

**Intuitive Understanding:**
Even with clear gaps between arms, you need at least logarithmic regret to learn the parameters.

### Concentration Inequalities

**Self-Normalized Martingale:**
For linear bandits, we use self-normalized martingale concentration:

```math
P\left(\|\theta^* - \hat{\theta}_t\|_{A_t} \geq \beta_t\right) \leq \delta
```

Where $`\beta_t = \sqrt{2 \log \frac{\det(A_t)^{1/2}}{\delta \lambda^{d/2}}} + \sqrt{\lambda} S`$

**Intuitive Understanding:**
This inequality tells us how likely it is that our parameter estimate is far from the true parameter. It's crucial for constructing confidence intervals.

**Why This Matters:**
- **Confidence intervals**: Basis for UCB-style algorithms
- **Regret bounds**: Essential for theoretical analysis
- **Algorithm design**: Guides exploration parameter selection

## Practical Considerations

### Understanding Practical Challenges

**The Implementation Challenge:**
How do you implement linear bandit algorithms in practice? What issues arise in real-world applications?

**Key Considerations:**
- **Feature engineering**: How to design good features
- **Parameter tuning**: How to set algorithm parameters
- **Numerical stability**: Avoiding computational issues
- **Scalability**: Handling large-scale problems

### Feature Engineering

**Feature Selection:**
- **Domain knowledge**: Use relevant features for the problem
- **Dimensionality reduction**: PCA, feature selection methods
- **Kernel methods**: Extend to non-linear reward functions

**Intuitive Understanding:**
Good features are like good predictors. If you're predicting house prices, features like "number of bedrooms" and "location" are much better than "color of the front door."

**Feature Normalization:**
See [`feature_utils.py`](feature_utils.py) for the complete implementation.

```python
# Key functionality:
def normalize_features(features):
    # Normalize feature vectors to unit norm
    
def standardize_features(features):
    # Standardize features to zero mean and unit variance
```

**Why Normalization Matters:**
- **Numerical stability**: Prevents computational issues
- **Fair comparison**: All features contribute equally
- **Algorithm performance**: Many algorithms assume normalized features

### Parameter Tuning

**Exploration Parameter ($`\alpha`$):**
- **Theoretical**: $`\alpha = \sqrt{d \log T}`$
- **Practical**: $`\alpha = 1.0`$ often works well
- **Cross-validation**: Tune on validation data

**Intuitive Understanding:**
- **Higher α**: More exploration, more conservative
- **Lower α**: Less exploration, more aggressive
- **α = 1.0**: Often a good practical choice

**Regularization ($`\lambda`$):**
- **Ridge regression**: $`\lambda = 1.0`$ is often sufficient
- **Feature scaling**: Adjust based on feature magnitudes
- **Numerical stability**: Larger $`\lambda`$ for ill-conditioned problems

**Intuitive Understanding:**
Regularization prevents overfitting and ensures numerical stability. It's like adding a small penalty for large parameter values.

### Numerical Stability

**Matrix Inversion:**
See [`feature_utils.py`](feature_utils.py) for the complete implementation.

```python
# Key functionality:
def stable_solve(A, b):
    # Stable matrix solve with regularization
    
def check_conditioning(A):
    # Check condition number of design matrix
```

**Intuitive Understanding:**
Matrix inversion can be numerically unstable. We need to ensure the design matrix is well-conditioned and use stable solvers.

**Common Issues:**
- **Ill-conditioned matrices**: When features are highly correlated
- **Singular matrices**: When we haven't seen enough diverse features
- **Numerical errors**: Accumulation of floating-point errors

## Advanced Topics

### Generalized Linear Bandits

**Problem Extension:**
Instead of linear rewards, use generalized linear models:

```math
\mathbb{E}[r_t | a_t] = g(\langle \theta^*, x_{a_t} \rangle)
```

Where $`g`$ is a known link function (e.g., sigmoid for logistic regression).

**Intuitive Understanding:**
Like linear bandits, but the relationship between features and rewards is non-linear. For example, the probability of clicking an ad might be a sigmoid function of the feature combination.

**Algorithms:**
- **GLM-UCB**: Extend UCB to generalized linear models
- **GLM-TS**: Thompson sampling with GLM likelihood
- **Online Newton Step**: Second-order optimization methods

### Kernel Bandits

**Non-linear Extensions:**
Use kernel functions to capture non-linear reward functions:

```math
\mathbb{E}[r_t | a_t] = f(x_{a_t}) = \sum_{i=1}^{t-1} \alpha_i k(x_i, x_{a_t})
```

**Intuitive Understanding:**
Instead of linear combinations of features, use kernel functions to capture complex non-linear relationships. Like using similarity functions between arms.

**Algorithms:**
- **GP-UCB**: Gaussian process upper confidence bound
- **Kernel-TS**: Thompson sampling with Gaussian processes
- **Random Fourier Features**: Efficient approximation for large datasets

### Bandits with Constraints

**Resource Constraints:**
- **Budget constraints**: Limited total cost
- **Time constraints**: Limited time horizon
- **Safety constraints**: Ensure safe exploration

**Intuitive Understanding:**
Real-world problems often have constraints. You can't spend unlimited money on ads, or you need to ensure treatments are safe for patients.

**Algorithms:**
- **Constrained UCB**: Add constraint handling to UCB
- **Safe exploration**: Ensure constraints are satisfied
- **Multi-objective bandits**: Balance multiple objectives

## Applications

### Understanding Real-World Applications

**The Application Challenge:**
How do linear bandit algorithms apply to real-world problems? What are the practical considerations?

**Key Applications:**
- **Recommendation systems**: Products with features
- **Online advertising**: Ads with features
- **Clinical trials**: Treatments with features
- **Resource allocation**: Options with features

### Recommendation Systems

**Content Recommendation:**
See [`application_examples.py`](application_examples.py) for the complete implementation.

```python
# Key functionality:
class ContentRecommender:
    # Use LinUCB for content recommendation
    # Combine user context with item features
    # Update model with user feedback
```

**Intuitive Understanding:**
Like Netflix recommending movies based on features like genre, year, rating, and cast. Learning that "action movies from the 90s with high ratings tend to be liked" helps recommend similar movies.

### Online Advertising

**Ad Selection with User Features:**
See [`application_examples.py`](application_examples.py) for the complete implementation.

```python
# Key functionality:
class AdSelector:
    # Use Linear Thompson Sampling for ad selection
    # Combine user and ad features
    # Update model with click feedback
```

**Intuitive Understanding:**
Like choosing which ad to show based on features like ad category, user demographics, and time of day. Learning that "tech ads work well with young users during work hours" helps optimize ad selection.

### Clinical Trials

**Adaptive Treatment Assignment:**
See [`application_examples.py`](application_examples.py) for the complete implementation.

```python
# Key functionality:
class AdaptiveTrial:
    # Use LinUCB for adaptive treatment assignment
    # Combine patient and treatment features
    # Update model with treatment outcomes
```

**Intuitive Understanding:**
Like assigning treatments based on patient features (age, symptoms, medical history) and treatment features (dosage, frequency, type). Learning that "high-dose treatments work better for severe cases" helps optimize treatment assignment.

## Implementation Examples

### Complete Example Usage

See [`linear_bandit_example.py`](linear_bandit_example.py) for a complete example that demonstrates how to use all the linear bandit algorithms together.

```python
# Run a complete experiment comparing LinUCB, Linear Thompson Sampling, and OFUL
# python code/linear_bandit_example.py
```

### Complete Linear Bandit Environment

See [`linear_bandit_environment.py`](linear_bandit_environment.py) for the complete implementation.

```python
# Key components:
class LinearBanditEnvironment:
    # Linear bandit environment with noise
    # Pull arms and observe rewards
    # Calculate cumulative regret

def run_linear_bandit_experiment(env, algorithm, T=1000):
    # Run linear bandit experiment
    # Select arms, observe rewards, update algorithm
```

### Algorithm Comparison

See [`linear_bandit_environment.py`](linear_bandit_environment.py) for the complete implementation.

```python
# Key functionality:
def compare_linear_algorithms(env, algorithms, T=1000, n_runs=50):
    # Compare different linear bandit algorithms
    # Run multiple independent trials
    # Return average regrets for each algorithm

# Example usage:
# d = 5, K = 10, theta_star = normalized random vector
# arms = normalized random feature vectors
# algorithms = {'LinUCB': ..., 'Linear TS': ..., 'OFUL': ...}
# results = compare_linear_algorithms(env, algorithms)
```

### Visualization

See [`linear_bandit_environment.py`](linear_bandit_environment.py) for the complete implementation.

```python
# Key functionality:
def plot_linear_bandit_results(results, T):
    # Plot cumulative regret comparison
    # Plot regret rate (regret / sqrt(t))
    # Include legend and proper labeling

# Usage:
# plot_linear_bandit_results(results, 1000)
```

## Summary

Linear bandits provide a powerful framework for sequential decision-making with structured action spaces. Key insights include:

1. **Feature-based Learning**: Exploit arm similarities through feature representations
2. **Efficient Exploration**: Confidence ellipsoids guide exploration in feature space
3. **Theoretical Guarantees**: Sublinear regret bounds scaling with dimension $`d`$
4. **Practical Algorithms**: LinUCB, Linear Thompson Sampling, and OFUL
5. **Wide Applicability**: Recommendation systems, advertising, clinical trials

Linear bandits bridge the gap between classical bandits and more complex contextual settings, providing both theoretical guarantees and practical effectiveness.

**Key Takeaways:**
- Linear bandits exploit structure in arm features for more efficient learning
- Feature-based learning reduces sample complexity from K to d
- LinUCB, Linear Thompson Sampling, and OFUL provide different trade-offs
- Theoretical guarantees show sublinear regret scaling with dimension
- Practical implementation requires careful feature engineering and parameter tuning

**The Broader Impact:**
Linear bandits have fundamentally changed how we approach structured decision-making by:
- **Enabling efficient learning**: Learning patterns instead of individual values
- **Supporting large-scale applications**: Handling thousands of arms efficiently
- **Providing theoretical foundations**: Rigorous guarantees for structured learning
- **Inspiring advanced methods**: Foundation for contextual and kernel bandits

## Further Reading

- **Linear Bandits Paper**: Theoretical foundations by Dani et al.
- **Generalized Linear Bandits**: Extension to GLM by Filippi et al.
- **Bandit Algorithms Textbook**: Comprehensive treatment by Lattimore and Szepesvári
- **Online Learning Resources**: Stanford CS234, UC Berkeley CS285

---

**Note**: This guide covers the fundamentals of linear bandits. For more advanced topics, see the sections on Contextual Bandits, Kernel Bandits, and Generalized Linear Bandits.

## From Static Features to Dynamic Contexts

We've now explored **linear bandits** - extending the bandit framework to handle structured action spaces where rewards are linear functions of arm features. We've seen how LinUCB and linear Thompson sampling can exploit arm similarities, how feature-based learning reduces sample complexity, and how these methods enable efficient learning in high-dimensional action spaces.

However, while linear bandits leverage the structure of arm features, **real-world environments** are often dynamic and context-dependent. Consider an online advertising system where the effectiveness of an ad depends not just on the ad itself, but also on the user's current context - their demographics, browsing history, time of day, and other contextual factors.

This motivates our exploration of **contextual bandits** - extending the bandit framework to handle dynamic contexts where the optimal action depends on the current state. We'll see how contextual UCB and contextual Thompson sampling adapt to changing environments, how personalization enables tailored decisions, and how these methods handle the complexity of real-world applications.

The transition from linear to contextual bandits represents the bridge from static structure to dynamic adaptation - taking our understanding of feature-based learning and extending it to handle the temporal and contextual dynamics inherent in many real-world problems.

In the next section, we'll explore contextual bandits, understanding how they adapt to changing contexts and enable personalized decision-making.

---

**Previous: [Classical Multi-Armed Bandits](01_classical_multi_armed_bandits.md)** - Understand the foundational framework for sequential decision making.

**Next: [Contextual Bandits](03_contextual_bandits.md)** - Learn how to adapt to dynamic contexts and personalize decisions. 
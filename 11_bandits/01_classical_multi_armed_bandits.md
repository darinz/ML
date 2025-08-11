# Classical Multi-Armed Bandits

## Introduction

Classical multi-armed bandits represent the foundational framework for sequential decision-making under uncertainty. The problem gets its name from the analogy of a gambler facing multiple slot machines (arms) and trying to maximize their cumulative winnings over time. This framework provides the theoretical foundation for balancing exploration and exploitation in dynamic environments.

### The Big Picture: What are Multi-Armed Bandits?

**The Bandit Problem:**
Imagine you're in a casino with 5 slot machines, each with different (but unknown) payout rates. You have $100 and want to maximize your winnings. How do you decide which machines to play? This is the multi-armed bandit problem.

**The Intuitive Analogy:**
Think of multi-armed bandits like trying different restaurants in a new city. You want to find the best restaurant, but you also want to enjoy good meals while you're learning. You face a trade-off: try new places (exploration) or go back to places you know are good (exploitation).

**Why Bandits Matter:**
- **Sequential decisions**: Each choice affects future opportunities
- **Uncertainty**: We don't know the true rewards beforehand
- **Learning**: We improve our decisions over time
- **Real-world applications**: From medicine to advertising to recommendations

### The Core Challenge

The fundamental challenge in multi-armed bandits is the **exploration-exploitation trade-off**:
- **Exploration**: Trying different arms to learn their reward distributions
- **Exploitation**: Choosing arms that are known to provide high rewards

**The Trade-off Dilemma:**
- **Too much exploration**: You waste time on bad options
- **Too much exploitation**: You might miss better options
- **Just right**: You learn efficiently while maximizing rewards

This trade-off appears in numerous real-world scenarios:
- **Clinical trials**: Testing different treatments on patients
- **Online advertising**: Selecting which ads to show users
- **Recommendation systems**: Choosing content to recommend
- **A/B testing**: Comparing different website designs
- **Resource allocation**: Distributing limited resources across options

## Problem Formulation

### Understanding the Problem Setup

**The Basic Scenario:**
You have K different options (arms), each with an unknown reward distribution. At each time step, you choose one option and receive a reward. Your goal is to maximize the total reward over time.

**Key Questions:**
- How do you balance trying new options vs. using known good options?
- How do you measure how well you're doing?
- What algorithms can solve this problem effectively?

### Mathematical Setup

In the classical multi-armed bandit problem, we have:

- **$`K`$ arms** (actions) with unknown reward distributions
- At each time step $`t`$, we choose an arm $`a_t \in \{1, 2, \ldots, K\}`$
- We receive a reward $`r_t`$ drawn from the distribution of arm $`a_t`$
- **Goal**: Maximize cumulative reward $`\sum_{t=1}^T r_t`$ over $`T`$ rounds

**Intuitive Understanding:**
This says: "You have K different options, each with some unknown probability of giving you a reward. You want to maximize your total rewards over T tries."

**The Learning Process:**
1. **Start**: Know nothing about any arm
2. **Choose**: Pick an arm based on current knowledge
3. **Observe**: Get a reward from the chosen arm
4. **Update**: Use the reward to improve your knowledge
5. **Repeat**: Keep choosing and learning

### Key Definitions

**Reward Distributions:**
- Each arm $`i`$ has an unknown reward distribution with mean $`\mu_i`$
- Rewards are typically bounded: $`r_t \in [0, 1]`$ or $`r_t \in [a, b]`$
- The optimal arm is $`i^* = \arg\max_i \mu_i`$ with mean $`\mu^* = \mu_{i^*}`$

**Intuitive Understanding:**
- **Reward distribution**: Each arm is like a biased coin - it has some probability of giving you a reward
- **Mean reward**: The average reward you'd get if you pulled this arm many times
- **Optimal arm**: The arm with the highest average reward

**Regret:**
The **cumulative regret** measures the difference between optimal and achieved performance:

```math
R(T) = \sum_{t=1}^T \mu^* - \mu_{a_t}
```

**Intuitive Understanding:**
Regret measures "how much worse did I do compared to the best possible strategy?" It's like asking "how much money did I lose by not always choosing the best arm?"

**Expected Regret:**
```math
\mathbb{E}[R(T)] = \sum_{i \neq i^*} \Delta_i \mathbb{E}[n_i(T)]
```

Where:
- $`\Delta_i = \mu^* - \mu_i`$ is the **gap** between arm $`i`$ and the optimal arm
- $`n_i(T)`$ is the number of times arm $`i`$ is pulled up to time $`T`$

**Intuitive Understanding:**
This says: "My expected regret is the sum over all bad arms of (how much worse this arm is) × (how many times I'll pull it)."

**Why This Makes Sense:**
- **Large gaps**: Arms much worse than optimal contribute more to regret
- **Frequent pulls**: Arms pulled more often contribute more to regret
- **Goal**: Minimize regret by pulling bad arms less often

### Problem Variants

**Stochastic Bandits:**
- Rewards are drawn from fixed, unknown distributions
- Most common and well-studied variant
- **Intuitive**: Like slot machines with fixed but unknown payout rates

**Adversarial Bandits:**
- Rewards are chosen by an adversary
- No statistical assumptions on reward generation
- **Intuitive**: Like playing against someone who's trying to make you lose

**Non-stationary Bandits:**
- Reward distributions change over time
- Requires algorithms that can adapt to changing environments
- **Intuitive**: Like slot machines where the payout rates change over time

## Fundamental Algorithms

### Understanding Algorithm Design

**The Algorithm Challenge:**
How do you design a strategy that balances exploration and exploitation to minimize regret?

**Key Principles:**
- **Optimism in the face of uncertainty**: Assume unknown arms might be good
- **Confidence intervals**: Quantify uncertainty about each arm
- **Adaptive exploration**: Explore more when uncertain, exploit when confident

### 1. Epsilon-Greedy

The simplest exploration strategy that balances exploration and exploitation through a fixed parameter.

**Intuitive Understanding:**
Epsilon-greedy is like a student who mostly studies the subject they know best (exploitation) but occasionally tries a new subject (exploration) to see if it might be better.

**Algorithm:**
- With probability $`\epsilon`$: choose random arm (exploration)
- With probability $`1-\epsilon`$: choose arm with highest empirical mean (exploitation)

**The Learning Process:**
1. **Initialize**: Start with no knowledge of any arm
2. **Choose**: Either explore (random) or exploit (best known)
3. **Observe**: Get reward from chosen arm
4. **Update**: Update empirical mean for that arm
5. **Repeat**: Continue until time horizon

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
- **Pros**: Simple, easy to implement, works reasonably well
- **Cons**: Fixed exploration rate, doesn't adapt to uncertainty
- **Regret**: $`O(T^{2/3})`$ with optimal $`\epsilon`$ decay

**When to Use:**
- **Simple problems**: When you need a quick, simple solution
- **Baseline comparison**: As a benchmark for more sophisticated algorithms
- **Educational purposes**: To understand the exploration-exploitation trade-off

### 2. Upper Confidence Bound (UCB)

UCB uses confidence intervals to balance exploration and exploitation optimally.

**Intuitive Understanding:**
UCB is like a cautious optimist. It assumes each arm might be the best one, but it's more optimistic about arms it hasn't tried much (high uncertainty). It's like saying "This arm might be amazing, and I haven't tried it much, so let me give it a chance."

**Algorithm:**
```math
a_t = \arg\max_{i} \left(\hat{\mu}_i + \sqrt{\frac{2 \log t}{n_i}}\right)
```

Where:
- $`\hat{\mu}_i`$: Empirical mean of arm $`i`$
- $`n_i`$: Number of times arm $`i`$ has been pulled
- $`t`$: Current time step

**Breaking Down the Formula:**
- **$`\hat{\mu}_i`$**: What we think this arm's reward is (exploitation)
- **$`\sqrt{\frac{2 \log t}{n_i}}`$**: How uncertain we are about this arm (exploration)
- **Sum**: Total "optimistic estimate" of the arm's value

**Why This Works:**
- **High uncertainty**: Arms pulled few times get high exploration bonus
- **Low uncertainty**: Arms pulled many times rely mostly on empirical mean
- **Logarithmic growth**: Exploration bonus grows slowly over time
- **Automatic balance**: Naturally balances exploration and exploitation

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

**When to Use:**
- **Theoretical guarantees**: When you need proven performance bounds
- **Stable performance**: When you want consistent, predictable behavior
- **Academic applications**: When you need to compare against theoretical results

### 3. Thompson Sampling

Thompson sampling is a Bayesian approach that maintains posterior distributions over arm rewards.

**Intuitive Understanding:**
Thompson sampling is like a scientist who maintains beliefs about each arm's true reward and updates these beliefs based on evidence. Instead of being optimistic, it samples from its current beliefs to make decisions.

**The Bayesian Approach:**
1. **Prior beliefs**: Start with initial beliefs about each arm
2. **Sample**: Draw a sample from current beliefs for each arm
3. **Choose**: Pick the arm with the highest sampled value
4. **Update**: Update beliefs based on observed reward
5. **Repeat**: Continue sampling and updating

**For Bernoulli Rewards:**
- Prior: Beta distribution $`\text{Beta}(1, 1)`$ (uniform)
- Posterior: $`\text{Beta}(1 + S_i, 1 + F_i)`$
- Where $`S_i`$ = successes, $`F_i`$ = failures for arm $`i`$

**Intuitive Understanding:**
- **Beta distribution**: Models our uncertainty about the probability of success
- **Prior**: Start with uniform belief (any probability equally likely)
- **Posterior**: Update beliefs based on observed successes and failures
- **Sampling**: Draw a random probability from our current belief

**Why Sampling Works:**
- **Exploration**: Sampling from uncertain arms gives them a chance
- **Exploitation**: Sampling from confident arms usually picks the best
- **Natural balance**: Uncertainty naturally decreases with more observations
- **Bayesian optimal**: Theoretically optimal under certain assumptions

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

**When to Use:**
- **Practical performance**: When you want the best empirical performance
- **Bayesian framework**: When you have prior knowledge about arms
- **Complex models**: When you want to extend to more sophisticated reward models

## Theoretical Analysis

### Understanding Regret Analysis

**The Analysis Challenge:**
How do we mathematically analyze how well these algorithms perform? What guarantees can we provide?

**Key Questions:**
- How does regret grow with time?
- How does regret depend on the problem structure?
- What are the fundamental limits of any algorithm?

### Regret Bounds

**UCB Regret Analysis:**
For UCB algorithm with exploration parameter $`\alpha = 2`$:

```math
\mathbb{E}[R(T)] \leq \sum_{i \neq i^*} \frac{8 \log T}{\Delta_i} + \left(1 + \frac{\pi^2}{3}\right) \sum_{i=1}^K \Delta_i
```

**Intuitive Understanding:**
This bound says: "Your expected regret is at most (some constant) × (logarithm of time) × (sum of inverse gaps) plus a constant term."

**Breaking Down the Bound:**
- **$`\frac{8 \log T}{\Delta_i}`$**: How much regret you accumulate from pulling arm i
- **$`\log T`$**: Regret grows logarithmically with time
- **$`\frac{1}{\Delta_i}`$**: Harder problems (smaller gaps) lead to more regret
- **Constant term**: Initial exploration cost

**Key Insights:**
- **Gap-dependent**: Regret depends on gaps $`\Delta_i`$
- **Logarithmic**: Regret grows as $`\log T`$ for each suboptimal arm
- **Constant term**: Initial exploration cost

**Lower Bounds:**
- **Minimax lower bound**: $`\Omega(\sqrt{KT})`$ for any algorithm
- **Gap-dependent lower bound**: $`\Omega(\sum_{i \neq i^*} \frac{\log T}{\Delta_i})`$

**Intuitive Understanding:**
These lower bounds say: "No matter how clever your algorithm is, you can't do better than this." They tell us what's fundamentally possible.

### Concentration Inequalities

**Why Concentration Inequalities Matter:**
Concentration inequalities tell us how likely it is that our empirical estimates are close to the true values. This is crucial for understanding why UCB works.

**Hoeffding's Inequality:**
For bounded random variables $`X_i \in [a, b]`$:

```math
P\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mu\right| \geq \epsilon\right) \leq 2\exp\left(-\frac{2n\epsilon^2}{(b-a)^2}\right)
```

**Intuitive Understanding:**
This says: "The probability that our sample average is far from the true mean decreases exponentially with the number of samples."

**Chernoff Bounds:**
For Bernoulli random variables:

```math
P(\hat{\mu} \geq \mu + \epsilon) \leq \exp(-n \text{KL}(\mu + \epsilon \| \mu))
```

**Application to UCB:**
The exploration term $`\sqrt{\frac{2 \log t}{n_i}}`$ in UCB is derived from Hoeffding's inequality, ensuring that the true mean lies within the confidence interval with high probability.

**Intuitive Understanding:**
UCB uses these inequalities to create confidence intervals. The exploration term is chosen so that the true mean is within the interval with high probability.

### Sample Complexity

**Definition:**
The number of samples needed to identify the best arm with high confidence.

**Intuitive Understanding:**
Sample complexity asks: "How many times do I need to pull each arm to be confident about which one is best?"

**For Best Arm Identification:**
- **Successive Elimination**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta})`$
- **Racing Algorithms**: Similar bounds with adaptive allocation

**Why This Matters:**
- **Resource constraints**: Limited time or budget
- **Clinical trials**: Limited number of patients
- **A/B testing**: Limited number of users

## Practical Considerations

### Understanding Practical Challenges

**The Implementation Challenge:**
How do you implement these algorithms in practice? What issues arise in real-world applications?

**Key Considerations:**
- **Parameter tuning**: How to set algorithm parameters
- **Numerical stability**: Avoiding computational issues
- **Memory efficiency**: Handling large-scale problems
- **Real-world constraints**: Dealing with practical limitations

### Parameter Tuning

**Epsilon-Greedy:**
- **Fixed epsilon**: Simple but suboptimal
- **Decaying epsilon**: $`\epsilon_t = \min(1, \frac{cK}{d^2 t})`$ for better performance
- **Adaptive epsilon**: Adjust based on observed performance

**Intuitive Understanding:**
- **Fixed epsilon**: Always explore with same probability
- **Decaying epsilon**: Explore less as you learn more
- **Adaptive epsilon**: Adjust exploration based on uncertainty

**UCB:**
- **Exploration parameter**: $`\alpha`$ controls exploration vs. exploitation
- **Typical values**: $`\alpha = 2`$ (theoretical), $`\alpha = 1`$ (practical)
- **Tuning**: Cross-validation or theoretical analysis

**Intuitive Understanding:**
- **Higher α**: More exploration, more conservative
- **Lower α**: Less exploration, more aggressive
- **α = 2**: Theoretical optimal for worst-case scenarios
- **α = 1**: Often better in practice

**Thompson Sampling:**
- **Prior choice**: Uniform Beta(1,1) is often sufficient
- **Non-informative priors**: Can improve performance with domain knowledge
- **Hyperparameter-free**: Often works well without tuning

**Intuitive Understanding:**
- **Uniform prior**: Start with no knowledge about any arm
- **Informative prior**: Start with some knowledge about arms
- **No tuning needed**: Often works well out of the box

### Implementation Details

**Numerical Stability:**
```python
# Avoid division by zero in UCB
if pulls[i] == 0:
    ucb_value = float('inf')
else:
    ucb_value = empirical_means[i] + np.sqrt(2 * np.log(t) / pulls[i])
```

**Intuitive Understanding:**
When we haven't pulled an arm yet, we're maximally uncertain about it, so we give it infinite value to ensure we try it.

**Memory Efficiency:**
```python
# Incremental mean update
def update_mean(old_mean, old_count, new_value):
    return (old_mean * old_count + new_value) / (old_count + 1)
```

**Intuitive Understanding:**
Instead of storing all rewards and recomputing the mean, we can update it incrementally. This saves memory and is more efficient.

*Note: These implementation details are handled in the separate Python files: [`ucb.py`](ucb.py), [`epsilon_greedy.py`](epsilon_greedy.py), and [`thompson_sampling.py`](thompson_sampling.py).*

**Parallelization:**
- **Batch updates**: Process multiple rewards simultaneously
- **Distributed bandits**: Multiple agents sharing information
- **Asynchronous updates**: Handle delayed feedback

### Common Pitfalls

**1. Insufficient Exploration:**
- Using too small epsilon in epsilon-greedy
- Setting exploration parameter too low in UCB
- Not accounting for reward variance

**Intuitive Understanding:**
If you don't explore enough, you might get stuck with a suboptimal arm and never discover the best one.

**2. Over-exploration:**
- Using too large epsilon
- Setting exploration parameter too high
- Wasting samples on clearly suboptimal arms

**Intuitive Understanding:**
If you explore too much, you waste time on bad options instead of exploiting the good ones you've found.

**3. Non-stationary Environments:**
- Classical algorithms assume stationary rewards
- Need adaptation mechanisms for changing environments
- Consider sliding windows or discounting

**Intuitive Understanding:**
If the reward distributions change over time, you need algorithms that can adapt to these changes.

**4. Reward Scaling:**
- Algorithms assume bounded rewards $`[0, 1]`$
- Need proper scaling for different reward ranges
- Consider normalization or clipping

**Intuitive Understanding:**
If your rewards are outside the expected range, you need to scale them appropriately for the algorithms to work well.

## Advanced Topics

### Non-stationary Bandits

**Problem:**
Reward distributions change over time, requiring algorithms that can adapt.

**Intuitive Understanding:**
Like slot machines where the payout rates change over time, or restaurants where the quality varies.

**Solutions:**
- **Sliding window UCB**: Only use recent observations
- **Discounting**: Give more weight to recent rewards
- **Change detection**: Detect when distributions change

### Correlated Arms

**Problem:**
Arms are not independent; pulling one arm provides information about others.

**Intuitive Understanding:**
Like restaurants where similar cuisines might have similar quality, or products where similar features predict similar performance.

**Solutions:**
- **Correlation-aware algorithms**: Exploit arm correlations
- **Feature-based approaches**: Use arm features to share information
- **Hierarchical bandits**: Group similar arms together

### Multi-objective Bandits

**Problem:**
Multiple objectives to optimize simultaneously (e.g., revenue and user satisfaction).

**Intuitive Understanding:**
Like choosing a restaurant where you want both good food and good service, or an ad that maximizes both clicks and conversions.

**Solutions:**
- **Pareto optimality**: Find non-dominated solutions
- **Scalarization**: Combine objectives into single metric
- **Preference learning**: Learn user preferences over objectives

## Applications

### Understanding Real-World Applications

**The Application Challenge:**
How do bandit algorithms apply to real-world problems? What are the practical considerations?

**Key Applications:**
- **Online advertising**: Choosing which ads to show
- **Recommendation systems**: Suggesting content to users
- **Clinical trials**: Testing treatments on patients
- **Resource allocation**: Distributing limited resources

### Online Advertising

**Ad Selection:**
- Choose from multiple ad creatives
- Optimize for click-through rate (CTR)
- Balance exploration of new ads with exploitation of known good ads

**Intuitive Understanding:**
Like a marketing manager who has several ad designs and wants to maximize clicks while learning which ads work best.

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

**Intuitive Understanding:**
Like Netflix choosing which movies to recommend to you, balancing showing you movies they think you'll like with trying new ones to learn your preferences.

**Clinical Trials:**
- Allocate patients to treatment arms
- Learn treatment effectiveness
- Ethical considerations for patient welfare

**Intuitive Understanding:**
Like a doctor who has several treatment options and wants to learn which works best while ensuring patients get good care.

## Implementation Examples

### Complete Example Usage

See [`example_usage.py`](example_usage.py) for a complete example that demonstrates how to use all the bandit algorithms together.

```python
# Run a complete experiment comparing epsilon-greedy, UCB, and Thompson sampling
# python code/example_usage.py
```

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

**Key Takeaways:**
- Multi-armed bandits model sequential decision-making under uncertainty
- The exploration-exploitation trade-off is fundamental to the problem
- UCB and Thompson sampling provide near-optimal theoretical guarantees
- Practical implementation requires careful attention to numerical stability and parameter tuning
- Bandit algorithms have wide applications in real-world problems

**The Broader Impact:**
Multi-armed bandits have fundamentally changed how we approach sequential decision-making by:
- **Providing principled approaches**: Theoretical foundations for exploration-exploitation
- **Enabling efficient learning**: Algorithms that learn optimally from limited data
- **Supporting real-world applications**: Practical tools for online learning problems
- **Inspiring advanced methods**: Foundation for contextual and linear bandits

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
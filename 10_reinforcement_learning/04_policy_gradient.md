# Policy Gradient Methods: REINFORCE and Beyond

## Introduction

Policy gradient methods represent a fundamental approach to reinforcement learning that directly optimizes policy parameters using gradient ascent. Unlike value-based methods that learn value functions and derive policies from them, policy gradient methods work directly with parameterized policies and optimize them to maximize expected returns.

**Key Advantages:**
- **Model-free**: No need to learn transition dynamics or value functions
- **Continuous action spaces**: Naturally handles continuous control problems
- **Stochastic policies**: Can represent exploration and uncertainty
- **Direct optimization**: Optimizes the objective of interest directly

**Applications:**
- Robot control and manipulation
- Game playing (AlphaGo, Dota 2)
- Autonomous driving
- Natural language processing
- Financial trading

## From Model-Based Control to Model-Free Learning

We've now explored **advanced control methods** - specialized techniques that combine the principles of reinforcement learning with classical control theory. We've seen how Linear Quadratic Regulation (LQR) provides exact solutions for linear systems, how Differential Dynamic Programming (DDP) handles nonlinear systems through iterative optimization, and how Linear Quadratic Gaussian (LQG) control addresses partial observability through optimal state estimation.

However, while these model-based control methods are powerful when we have good models of the system dynamics, **many real-world problems** involve systems where the dynamics are unknown, complex, or difficult to model accurately. In these cases, we need methods that can learn optimal behavior directly from experience without requiring explicit models of the environment.

This motivates our exploration of **policy gradient methods** - model-free reinforcement learning techniques that directly optimize policy parameters using gradient ascent. We'll see how REINFORCE learns policies from experience, how variance reduction techniques improve learning efficiency, and how these methods enable learning in complex, unknown environments where model-based approaches are not feasible.

The transition from model-based control to model-free learning represents the bridge from structured optimization to adaptive learning - taking our understanding of optimal control and extending it to scenarios where system models are unknown or unreliable.

In this chapter, we'll explore policy gradient methods, understanding how they learn optimal policies directly from experience without requiring explicit models of the environment.

---

## 17.1 The Policy Gradient Framework

### Problem Setup

We consider a **finite-horizon** Markov Decision Process (MDP) with:
- **State space**: $S$ (can be discrete or continuous)
- **Action space**: $A$ (can be discrete or continuous)
- **Transition dynamics**: $P_{sa}(s')$ (unknown to the agent)
- **Reward function**: $R(s, a)$ (can be queried but not known analytically)
- **Horizon**: $T < \infty$ (finite episode length)

### Parameterized Policies

We work with **stochastic policies** parameterized by $\theta \in \mathbb{R}^d$:

$$
\pi_\theta(a|s) = P(a_t = a | s_t = s, \theta)
$$

**Key Properties:**
- $\pi_\theta(a|s) \geq 0$ for all $a, s$
- $\sum_{a \in A} \pi_\theta(a|s) = 1$ for all $s$
- Differentiable with respect to $\theta$

**Examples of Policy Parameterizations:**

1. **Softmax Policy** (discrete actions):
   $$
   \pi_\theta(a|s) = \frac{e^{f_\theta(s, a)}}{\sum_{a'} e^{f_\theta(s, a')}}
   $$
   Where $f_\theta(s, a)$ is a neural network or linear function.

2. **Gaussian Policy** (continuous actions):
   $$
   \pi_\theta(a|s) = \mathcal{N}(a; \mu_\theta(s), \sigma_\theta^2(s))
   $$
   Where $\mu_\theta(s)$ and $\sigma_\theta(s)$ are parameterized functions.

### Objective Function

Our goal is to maximize the **expected return**:

$$
\eta(\theta) \triangleq \mathbb{E} \left[ \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t) \right] \tag{17.1}
$$

Where:
- $\gamma \in [0, 1]$ is the discount factor
- The expectation is over trajectories $\tau = (s_0, a_0, \ldots, s_{T-1}, a_{T-1}, s_T)$
- $s_0 \sim \mu$ (initial state distribution)
- $s_{t+1} \sim P_{s_t a_t}$ (environment dynamics)
- $a_t \sim \pi_\theta(\cdot|s_t)$ (policy)

**Connection to Value Functions:**
$$
\eta(\theta) = \mathbb{E}_{s_0 \sim \mu} \left[ V^{\pi_\theta}(s_0) \right]
$$

### Why Policy Gradient Methods?

1. **Model-free learning**: No need to learn $P_{sa}$ or $R(s, a)$
2. **Continuous action spaces**: Natural handling of continuous control
3. **Exploration**: Stochastic policies naturally explore
4. **Direct optimization**: Optimizes the objective of interest
5. **Flexibility**: Can incorporate domain knowledge into policy structure

---

## 17.2 The Policy Gradient Theorem

### The Core Challenge

We want to compute:
$$
\nabla_\theta \eta(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim P_\theta} \left[ \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t) \right]
$$

The challenge is that the expectation is over a distribution $P_\theta$ that depends on $\theta$, making direct differentiation difficult.

### The Log-Derivative Trick

The key insight is the **log-derivative trick** (also called the REINFORCE trick):

$$
\nabla_\theta \mathbb{E}_{\tau \sim P_\theta} [f(\tau)] = \mathbb{E}_{\tau \sim P_\theta} \left[ (\nabla_\theta \log P_\theta(\tau)) f(\tau) \right] \tag{17.3}
$$

**Derivation:**
$$
\nabla_\theta \mathbb{E}_{\tau \sim P_\theta} [f(\tau)] = \nabla_\theta \int P_\theta(\tau) f(\tau) d\tau
= \int \nabla_\theta (P_\theta(\tau) f(\tau)) d\tau \quad \text{(swap integration with gradient)}
= \int (\nabla_\theta P_\theta(\tau)) f(\tau) d\tau \quad \text{(because $f$ does not depend on $\theta$)}
= \int P_\theta(\tau) \frac{\nabla_\theta P_\theta(\tau)}{P_\theta(\tau)} f(\tau) d\tau
= \int P_\theta(\tau) (\nabla_\theta \log P_\theta(\tau)) f(\tau) d\tau
= \mathbb{E}_{\tau \sim P_\theta} \left[ (\nabla_\theta \log P_\theta(\tau)) f(\tau) \right]
$$

**Intuition:** We can estimate the gradient using only samples from the current policy, without needing to know the environment dynamics.

### Trajectory Probability Decomposition

For a trajectory $\tau = (s_0, a_0, \ldots, s_{T-1}, a_{T-1}, s_T)$:

$$
P_\theta(\tau) = \mu(s_0) \pi_\theta(a_0|s_0) P_{s_0 a_0}(s_1) \pi_\theta(a_1|s_1) P_{s_1 a_1}(s_2) \cdots P_{s_{T-1} a_{T-1}}(s_T) \tag{17.6}
$$

Taking the logarithm:
$$
\log P_\theta(\tau) = \log \mu(s_0) + \log \pi_\theta(a_0|s_0) + \log P_{s_0 a_0}(s_1) + \log \pi_\theta(a_1|s_1)
+ \log P_{s_1 a_1}(s_2) + \cdots + \log P_{s_{T-1} a_{T-1}}(s_T) \tag{17.7}
$$

**Key Insight:** When we take the gradient with respect to $\theta$, only the policy terms survive:

$$
\nabla_\theta \log P_\theta(\tau) = \nabla_\theta \log \pi_\theta(a_0|s_0) + \nabla_\theta \log \pi_\theta(a_1|s_1) + \cdots + \nabla_\theta \log \pi_\theta(a_{T-1}|s_{T-1})
$$

The environment terms ($\log P_{s_t a_t}(s_{t+1})$) don't depend on $\theta$ and thus have zero gradient.

### The Policy Gradient Formula

Combining the log-derivative trick with the trajectory decomposition:

$$
\nabla_\theta \eta(\theta) = \mathbb{E}_{\tau \sim P_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \cdot \left( \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t) \right) \right] \tag{17.8}
$$

**Interpretation:**
- $\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$: Direction that increases the probability of the taken actions
- $\sum_{t=0}^{T-1} \gamma^t R(s_t, a_t)$: Total reward of the trajectory
- High-reward trajectories get their actions reinforced more strongly

---

## 17.3 Variance Reduction with Baselines

### The Variance Problem

The basic policy gradient estimator can have very high variance, making learning slow and unstable. This happens because:

1. **High variance in returns**: Different trajectories can have very different total rewards
2. **Credit assignment**: All actions in a trajectory get the same weight (the total return)
3. **Exploration noise**: Random exploration can lead to poor trajectories

### The Baseline Trick

We can subtract a **baseline** $B(s_t)$ from the reward without changing the expected gradient:

$$
\nabla_\theta \eta(\theta) = \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (R_{\geq t} - B(s_t)) \right]
$$

Where $R_{\geq t} = \sum_{j=t}^{T-1} \gamma^{j-t} R(s_j, a_j)$ is the return from time $t$ onward.

**Why this works:**
$$
\mathbb{E}_{\tau \sim P_\theta} [\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot B(s_t)]
= \mathbb{E} [\mathbb{E} [\nabla_\theta \log \pi_\theta(a_t|s_t) | s_0, a_0, \ldots, s_{t-1}, a_{t-1}, s_t] B(s_t)]
= 0
$$

Because $\mathbb{E}_{a_t \sim \pi_\theta(\cdot|s_t)} [\nabla_\theta \log \pi_\theta(a_t|s_t)] = 0$ for any fixed state $s_t$.

### Optimal Baseline

The optimal baseline that minimizes variance is the **value function**:

$$
B^*(s_t) = V^{\pi_\theta}(s_t) = \mathbb{E} \left[ \sum_{j=t}^{T-1} \gamma^{j-t} R(s_j, a_j) | s_t \right]
$$

**Intuition:** We only reinforce actions that perform better than expected from that state.

### Practical Baseline Choices

1. **State-dependent baseline**: $B(s_t) = V^{\pi_\theta}(s_t)$ (estimated)
2. **Constant baseline**: $B(s_t) = \mathbb{E}[R_{\geq t}]$ (average return)
3. **Advantage function**: $A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)$

---

## 17.4 The REINFORCE Algorithm

### Basic REINFORCE

**Algorithm 1: Vanilla REINFORCE**

For iteration $i = 1, 2, \ldots$:

1. **Collect trajectories**: Run policy $\pi_{\theta_i}$ to collect $N$ trajectories $\{\tau^{(1)}, \ldots, \tau^{(N)}\}$

2. **Compute returns**: For each trajectory, compute $R_{\geq t}^{(i)} = \sum_{j=t}^{T-1} \gamma^{j-t} R(s_j^{(i)}, a_j^{(i)})$

3. **Estimate gradient**:
   $$
   \hat{g}_i = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot R_{\geq t}^{(i)}
   $$

4. **Update parameters**:
   $$
   \theta_{i+1} = \theta_i + \alpha_i \hat{g}_i
   $$

### REINFORCE with Baseline

**Algorithm 2: REINFORCE with Baseline**

For iteration $i = 1, 2, \ldots$:

1. **Collect trajectories**: Run policy $\pi_{\theta_i}$ to collect $N$ trajectories

2. **Compute returns**: $R_{\geq t}^{(i)} = \sum_{j=t}^{T-1} \gamma^{j-t} R(s_j^{(i)}, a_j^{(i)})$

3. **Fit baseline**: Find $B$ that minimizes:
   $$
   \sum_{i=1}^N \sum_{t=0}^{T-1} (R_{\geq t}^{(i)} - B(s_t^{(i)}))^2 \tag{17.12}
   $$

4. **Estimate gradient**:
   $$
   \hat{g}_i = \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t^{(i)}|s_t^{(i)}) \cdot (R_{\geq t}^{(i)} - B(s_t^{(i)}))
   $$

5. **Update parameters**: $\theta_{i+1} = \theta_i + \alpha_i \hat{g}_i$

### Implementation Details

#### Policy Network Architecture

For discrete actions:
```python
class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        logits = self.fc2(x)
        return torch.softmax(logits, dim=-1)
```

For continuous actions:
```python
class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        mean = self.fc2(x)
        std = torch.exp(self.log_std)
        return mean, std
```

#### Log Probability Computation

For discrete actions:
```python
def log_prob_discrete(policy, state, action):
    probs = policy(state)
    return torch.log(probs.gather(1, action.unsqueeze(1))).squeeze(1)
```

For continuous actions:
```python
def log_prob_continuous(policy, state, action):
    mean, std = policy(state)
    dist = torch.distributions.Normal(mean, std)
    return dist.log_prob(action).sum(dim=-1)
```

---

## 17.5 Advanced Policy Gradient Methods

### Actor-Critic Methods

Actor-critic methods combine policy gradients with value function estimation:

$$
\nabla_\theta \eta(\theta) = \mathbb{E}_{\tau \sim P_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^{\pi_\theta}(s_t, a_t) \right]
$$

Where $A^{\pi_\theta}(s_t, a_t) = Q^{\pi_\theta}(s_t, a_t) - V^{\pi_\theta}(s_t)$ is the advantage function.

**Advantages:**
- Lower variance than REINFORCE
- Better sample efficiency
- Can use function approximation

### Natural Policy Gradient

The natural policy gradient uses the Fisher information matrix to compute the steepest ascent direction:

$$
\theta_{i+1} = \theta_i + \alpha_i F^{-1}(\theta_i) \nabla_\theta \eta(\theta_i)
$$

Where $F(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^T]$ is the Fisher information matrix.

### Trust Region Policy Optimization (TRPO)

TRPO constrains policy updates to prevent too large changes:

$$
\max_\theta \mathbb{E} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A^{\pi_{\theta_{old}}}(s, a) \right]
$$

Subject to:
$$
\mathbb{E} \left[ KL(\pi_{\theta_{old}}(\cdot|s) \| \pi_\theta(\cdot|s)) \right] \leq \delta
$$

### Proximal Policy Optimization (PPO)

PPO uses a clipped objective to prevent large policy updates:

$$
L(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio.

---

## 17.6 Practical Considerations

### Hyperparameter Tuning

1. **Learning rate**: Start with $\alpha = 0.01$ and adjust based on performance
2. **Batch size**: Larger batches reduce variance but increase computational cost
3. **Discount factor**: $\gamma = 0.99$ is common for most problems
4. **Baseline**: Use value function estimation for better performance

### Exploration Strategies

1. **Entropy regularization**: Add entropy bonus to encourage exploration
2. **Action noise**: Add noise to continuous actions
3. **Temperature scaling**: Adjust softmax temperature for exploration

### Common Issues and Solutions

#### High Variance
- Use baselines or advantage functions
- Increase batch size
- Use actor-critic methods

#### Poor Convergence
- Tune learning rate carefully
- Use adaptive learning rates
- Normalize advantages

#### Policy Collapse
- Add entropy regularization
- Use trust region methods
- Monitor policy entropy

### Performance Monitoring

1. **Average return**: Track mean episode return
2. **Policy entropy**: Monitor exploration level
3. **Gradient norm**: Check for gradient explosion
4. **Value function loss**: Monitor baseline quality

---

## 17.7 Theoretical Analysis

### Convergence Properties

**Theorem (Policy Gradient Convergence):** Under mild conditions, REINFORCE converges to a local optimum of $\eta(\theta)$.

**Assumptions:**
- Policy is differentiable and Lipschitz continuous
- Rewards are bounded
- Learning rate satisfies Robbins-Monro conditions

### Sample Complexity

The sample complexity of policy gradient methods depends on:
- Problem horizon $T$
- State/action space size
- Policy class complexity
- Desired accuracy $\epsilon$

**Typical bound:** $O(T^2 / \epsilon^2)$ samples for $\epsilon$-optimal policy

### Variance Analysis

The variance of the policy gradient estimator is:
$$
\text{Var}[\hat{g}] = \mathbb{E} \left[ \|\nabla_\theta \log \pi_\theta(a|s)\|^2 \cdot (R_{\geq t} - B(s_t))^2 \right]
$$

This motivates the use of baselines and advantage functions for variance reduction.

---

## 17.8 Applications and Examples

### CartPole Control

```python
# Simple REINFORCE implementation for CartPole
def reinforce_cartpole():
    policy = DiscretePolicy(state_dim=4, action_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    
    for episode in range(1000):
        # Collect trajectory
        states, actions, rewards = collect_trajectory(policy)
        
        # Compute returns
        returns = compute_returns(rewards, gamma=0.99)
        
        # Compute policy gradient
        log_probs = [log_prob_discrete(policy, s, a) for s, a in zip(states, actions)]
        loss = -torch.mean(torch.stack(log_probs) * torch.tensor(returns))
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Continuous Control

```python
# REINFORCE for continuous control
def reinforce_continuous():
    policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim)
    value_net = ValueNetwork(state_dim=state_dim)  # Baseline
    
    for episode in range(1000):
        # Collect trajectory
        states, actions, rewards = collect_trajectory(policy)
        
        # Fit baseline
        fit_baseline(value_net, states, rewards)
        
        # Compute advantages
        advantages = compute_advantages(states, rewards, value_net)
        
        # Update policy
        update_policy(policy, states, actions, advantages)
```

### Advanced Applications

1. **Robotics**: Manipulation tasks, locomotion
2. **Games**: Atari games, board games
3. **Autonomous systems**: Self-driving cars, drones
4. **Natural language**: Dialogue systems, text generation

---

## Summary and Key Insights

### Core Principles

1. **Direct policy optimization**: Optimize policy parameters directly
2. **Sample-based learning**: Use trajectories to estimate gradients
3. **Variance reduction**: Use baselines and advantage functions
4. **Exploration**: Stochastic policies naturally explore

### Algorithm Comparison

| Method | Variance | Sample Efficiency | Complexity |
|--------|----------|-------------------|------------|
| REINFORCE | High | Low | Simple |
| REINFORCE + Baseline | Medium | Medium | Simple |
| Actor-Critic | Low | High | Medium |
| TRPO/PPO | Low | High | Complex |

### Best Practices

1. **Start simple**: Use REINFORCE with baseline
2. **Tune hyperparameters**: Learning rate is crucial
3. **Monitor performance**: Track returns and policy entropy
4. **Use appropriate baselines**: Value functions work well
5. **Consider advanced methods**: Actor-critic for better performance

### Future Directions

1. **Sample efficiency**: Reduce number of environment interactions
2. **Stability**: Improve convergence guarantees
3. **Scalability**: Handle high-dimensional state/action spaces
4. **Multi-agent**: Extend to multi-agent settings
5. **Hierarchical**: Combine with hierarchical RL

Policy gradient methods remain fundamental to modern reinforcement learning, providing the foundation for many advanced algorithms and applications.

## From Theoretical Understanding to Practical Implementation

We've now explored **policy gradient methods** - model-free reinforcement learning techniques that directly optimize policy parameters using gradient ascent. We've seen how REINFORCE learns policies from experience, how variance reduction techniques improve learning efficiency, and how these methods enable learning in complex, unknown environments where model-based approaches are not feasible.

However, while understanding the theoretical foundations of reinforcement learning and policy gradient methods is essential, true mastery comes from **practical implementation**. The concepts we've learned - MDPs, value functions, continuous state spaces, advanced control, and policy gradients - need to be applied to real problems to develop intuition and practical skills.

This motivates our exploration of **hands-on coding** - the practical implementation of all the reinforcement learning concepts we've learned. We'll put our theoretical knowledge into practice by implementing value and policy iteration, building continuous state RL systems, applying advanced control methods, and developing policy gradient algorithms for real-world problems.

The transition from theoretical understanding to practical implementation represents the bridge from knowledge to application - taking our understanding of how reinforcement learning works and turning it into practical tools for building intelligent agents that learn from experience.

In the next section, we'll implement complete reinforcement learning systems, experiment with different algorithms, and develop the practical skills needed for real-world applications in robotics, control, and autonomous systems.

---

**Previous: [Advanced Control Methods](03_advanced_control.md)** - Learn specialized control techniques for structured systems.

**Next: [Hands-on Coding](05_hands-on_coding.md)** - Implement reinforcement learning algorithms with practical examples.


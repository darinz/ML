# Policy Gradient Methods: REINFORCE and Beyond

## The Big Picture: Why Policy Gradient Methods Matter

**The Learning Challenge:**
Imagine trying to teach a robot to walk, a computer to play chess, or an AI to have a conversation. In these cases, we don't have a perfect model of how the world works, and we can't simply provide step-by-step instructions. Instead, we need the system to learn from experience - to try different actions, see what happens, and gradually improve its behavior.

**The Intuitive Analogy:**
Think of the difference between:
- **Value-based learning**: Like learning to evaluate chess positions (what's good/bad)
- **Policy-based learning**: Like learning to play chess moves directly (what to do)

**Why Policy Gradients Matter:**
- **Direct learning**: Learn what to do, not just what's good
- **Continuous actions**: Handle complex, continuous control problems
- **Model-free**: Work without knowing how the environment works
- **Natural exploration**: Stochastic policies naturally try different things
- **Flexible representation**: Can incorporate domain knowledge into policy structure

### The Key Insight

**From Evaluation to Action:**
- **Value methods**: Learn "how good is this situation?" then choose best action
- **Policy methods**: Learn "what should I do in this situation?" directly

**The Learning Paradigm Shift:**
- **Indirect learning**: Value function → Policy → Action
- **Direct learning**: Policy → Action
- **End-to-end**: Optimize the final objective directly

## Introduction

Policy gradient methods represent a fundamental approach to reinforcement learning that directly optimizes policy parameters using gradient ascent. Unlike value-based methods that learn value functions and derive policies from them, policy gradient methods work directly with parameterized policies and optimize them to maximize expected returns.

**The Direct Optimization Advantage:**
- **No intermediate step**: Don't need to learn value functions first
- **End-to-end learning**: Optimize the final objective directly
- **Flexible objectives**: Can optimize any differentiable objective
- **Natural exploration**: Stochastic policies explore automatically

**Key Advantages:**
- **Model-free**: No need to learn transition dynamics or value functions
- **Continuous action spaces**: Naturally handles continuous control problems
- **Stochastic policies**: Can represent exploration and uncertainty
- **Direct optimization**: Optimizes the objective of interest directly

**The Model-Free Advantage:**
- **Unknown dynamics**: Don't need to know how states transition
- **Complex systems**: Can handle systems too complex to model
- **Adaptive learning**: Learn and adapt as the environment changes
- **Real-world applicability**: Work in messy, unpredictable environments

**Applications:**
- Robot control and manipulation
- Game playing (AlphaGo, Dota 2)
- Autonomous driving
- Natural language processing
- Financial trading

**The Application Spectrum:**
- **Robotics**: Complex, continuous control problems
- **Games**: Strategic decision-making with uncertainty
- **Language**: Creative, open-ended generation tasks
- **Finance**: Risk management and portfolio optimization

## From Model-Based Control to Model-Free Learning

We've now explored **advanced control methods** - specialized techniques that combine the principles of reinforcement learning with classical control theory. We've seen how Linear Quadratic Regulation (LQR) provides exact solutions for linear systems, how Differential Dynamic Programming (DDP) handles nonlinear systems through iterative optimization, and how Linear Quadratic Gaussian (LQG) control addresses partial observability through optimal state estimation.

However, while these model-based control methods are powerful when we have good models of the system dynamics, **many real-world problems** involve systems where the dynamics are unknown, complex, or difficult to model accurately. In these cases, we need methods that can learn optimal behavior directly from experience without requiring explicit models of the environment.

**The Model Limitation Problem:**
- **Perfect models**: Rare in real-world applications
- **Complex dynamics**: Too complicated to model accurately
- **Unknown environments**: Can't predict what will happen
- **Adaptive requirements**: Environment changes over time

This motivates our exploration of **policy gradient methods** - model-free reinforcement learning techniques that directly optimize policy parameters using gradient ascent. We'll see how REINFORCE learns policies from experience, how variance reduction techniques improve learning efficiency, and how these methods enable learning in complex, unknown environments where model-based approaches are not feasible.

**The Learning Revolution:**
- **From models to experience**: Learn from data instead of equations
- **From prediction to action**: Focus on what to do, not what will happen
- **From optimization to learning**: Use gradient descent on policy parameters
- **From certainty to exploration**: Embrace uncertainty and learn from it

The transition from model-based control to model-free learning represents the bridge from structured optimization to adaptive learning - taking our understanding of optimal control and extending it to scenarios where system models are unknown or unreliable.

In this chapter, we'll explore policy gradient methods, understanding how they learn optimal policies directly from experience without requiring explicit models of the environment.

---

## Understanding the Policy Gradient Framework

### The Big Picture: What is Policy Gradient Learning?

**The Policy Gradient Problem:**
How do we learn a policy (a mapping from states to actions) that maximizes expected reward when we don't know how the environment works? This is like learning to play a new game without knowing the rules - we can only observe what happens when we make moves.

**The Intuitive Analogy:**
- **Policy**: Like a strategy guide that tells you what to do in each situation
- **Gradient**: Like a compass that points in the direction of improvement
- **Learning**: Like gradually improving your strategy based on results

**The Key Insight:**
We can use gradient ascent to directly optimize policy parameters based on the rewards we receive from the environment.

### 17.1 The Policy Gradient Framework

#### Problem Setup

We consider a **finite-horizon** Markov Decision Process (MDP) with:
- **State space**: $S$ (can be discrete or continuous)
- **Action space**: $A$ (can be discrete or continuous)
- **Transition dynamics**: $P_{sa}(s')$ (unknown to the agent)
- **Reward function**: $R(s, a)$ (can be queried but not known analytically)
- **Horizon**: $T < \infty$ (finite episode length)

**The Unknown Environment Challenge:**
- **Unknown dynamics**: Don't know how states transition
- **Unknown rewards**: Don't know reward function analytically
- **Learning goal**: Find good policy through trial and error
- **Finite episodes**: Each learning episode has a fixed length

**The Trial-and-Error Analogy:**
- **Unknown rules**: Like playing a new board game
- **Trial and error**: Try different moves and see what happens
- **Learning**: Gradually understand what works and what doesn't
- **Improvement**: Refine strategy based on results

#### Parameterized Policies

We work with **stochastic policies** parameterized by $\theta \in \mathbb{R}^d$:

$$
\pi_\theta(a|s) = P(a_t = a | s_t = s, \theta)
$$

**The Parameterized Policy Intuition:**
- **$\theta$**: Like the "settings" that control how the policy behaves
- **$\pi_\theta(a|s)$**: Probability of taking action $a$ in state $s$
- **Stochastic**: Sometimes choose different actions for same state
- **Differentiable**: Can compute gradients with respect to $\theta$

**The Policy as a Function:**
- **Input**: Current state $s$
- **Output**: Probability distribution over actions
- **Parameters**: $\theta$ controls the shape of this function
- **Learning**: Adjust $\theta$ to make good actions more likely

**Key Properties:**
- $\pi_\theta(a|s) \geq 0$ for all $a, s$ (probabilities are non-negative)
- $\sum_{a \in A} \pi_\theta(a|s) = 1$ for all $s$ (probabilities sum to 1)
- Differentiable with respect to $\theta$ (can compute gradients)

**The Probability Distribution Analogy:**
- **Policy**: Like a biased coin that decides what to do
- **Parameters**: Like the bias of the coin
- **Learning**: Like adjusting the bias to get better outcomes
- **Stochasticity**: Like the randomness that allows exploration

**Examples of Policy Parameterizations:**

1. **Softmax Policy** (discrete actions):
   $$
   \pi_\theta(a|s) = \frac{e^{f_\theta(s, a)}}{\sum_{a'} e^{f_\theta(s, a')}}
   $$
   Where $f_\theta(s, a)$ is a neural network or linear function.

   **The Softmax Intuition:**
   - **$f_\theta(s, a)$**: Like a "score" for each action
   - **Exponential**: Makes high scores much more likely
   - **Normalization**: Ensures probabilities sum to 1
   - **Smooth**: Small changes in scores lead to smooth changes in probabilities

   **The Restaurant Menu Analogy:**
   - **Scores**: Like ratings for different dishes
   - **Softmax**: Like choosing dishes based on ratings
   - **High ratings**: Much more likely to be chosen
   - **Exploration**: Sometimes try lower-rated dishes

2. **Gaussian Policy** (continuous actions):
   $$
   \pi_\theta(a|s) = \mathcal{N}(a; \mu_\theta(s), \sigma_\theta^2(s))
   $$
   Where $\mu_\theta(s)$ and $\sigma_\theta(s)$ are parameterized functions.

   **The Gaussian Policy Intuition:**
   - **$\mu_\theta(s)$**: Like the "recommended" action for state $s$
   - **$\sigma_\theta(s)$**: Like the "uncertainty" or "exploration level"
   - **Normal distribution**: Actions cluster around the mean
   - **Exploration**: Sometimes try actions far from the mean

   **The Archery Analogy:**
   - **Target**: Like the optimal action $\mu_\theta(s)$
   - **Arrow spread**: Like the exploration $\sigma_\theta(s)$
   - **Aiming**: Try to hit the target (optimal action)
   - **Learning**: Adjust aim and reduce spread over time

#### Objective Function

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

**The Expected Return Intuition:**
- **$\eta(\theta)$**: Average total reward when following policy $\pi_\theta$
- **Expectation**: Average over many different episodes
- **Discounting**: Future rewards worth less than immediate rewards
- **Optimization goal**: Find $\theta$ that maximizes $\eta(\theta)$

**The Investment Portfolio Analogy:**
- **$\eta(\theta)$**: Like expected return on investment portfolio
- **$\theta$**: Like portfolio allocation (stocks, bonds, etc.)
- **Optimization**: Find allocation that maximizes expected return
- **Risk**: Stochastic policies introduce variability in returns

**Connection to Value Functions:**
$$
\eta(\theta) = \mathbb{E}_{s_0 \sim \mu} \left[ V^{\pi_\theta}(s_0) \right]
$$

**The Value Function Connection:**
- **$V^{\pi_\theta}(s_0)$**: Expected return starting from state $s_0$
- **$\eta(\theta)$**: Average over all possible starting states
- **Equivalence**: Maximizing expected return = maximizing average value
- **Bridge**: Connects policy gradients to value function methods

#### Why Policy Gradient Methods?

1. **Model-free learning**: No need to learn $P_{sa}$ or $R(s, a)$
2. **Continuous action spaces**: Natural handling of continuous control
3. **Exploration**: Stochastic policies naturally explore
4. **Direct optimization**: Optimizes the objective of interest
5. **Flexibility**: Can incorporate domain knowledge into policy structure

**The Model-Free Advantage:**
- **Unknown dynamics**: Don't need to know how states transition
- **Unknown rewards**: Don't need analytical reward function
- **Data-driven**: Learn purely from experience
- **Adaptive**: Can handle changing environments

**The Continuous Action Advantage:**
- **Natural representation**: Continuous actions are natural for many problems
- **Smooth control**: Can make small, precise adjustments
- **No discretization**: Don't need to approximate continuous actions
- **Robust**: Small changes in parameters lead to small changes in behavior

**The Exploration Advantage:**
- **Natural exploration**: Stochastic policies try different actions
- **No explicit exploration**: Don't need separate exploration strategy
- **Adaptive exploration**: Exploration level can be learned
- **Balanced**: Balance between exploitation and exploration

**The Direct Optimization Advantage:**
- **End-to-end**: Optimize final objective directly
- **No intermediate steps**: Don't need to learn value functions first
- **Flexible objectives**: Can optimize any differentiable objective
- **Clear target**: Objective function is what we actually care about

**The Flexibility Advantage:**
- **Domain knowledge**: Can incorporate prior knowledge into policy structure
- **Constraints**: Can enforce constraints through policy design
- **Multi-objective**: Can optimize multiple objectives simultaneously
- **Hierarchical**: Can design hierarchical policy structures

---

## Understanding the Policy Gradient Theorem

### The Big Picture: How Do We Compute Policy Gradients?

**The Gradient Challenge:**
We want to find the direction that improves our policy, but the objective function depends on the policy in a complex way through the expectation over trajectories.

**The Intuitive Analogy:**
- **Objective**: Like trying to maximize your average score in a game
- **Policy**: Like your strategy for playing the game
- **Gradient**: Like figuring out which way to adjust your strategy
- **Challenge**: Your score depends on your strategy in a complex way

**The Key Insight:**
We can use the "log-derivative trick" to compute gradients using only samples from the current policy.

### 17.2 The Policy Gradient Theorem

#### The Core Challenge

We want to compute:
$$
\nabla_\theta \eta(\theta) = \nabla_\theta \mathbb{E}_{\tau \sim P_\theta} \left[ \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t) \right]
$$

The challenge is that the expectation is over a distribution $P_\theta$ that depends on $\theta$, making direct differentiation difficult.

**The Differentiation Challenge:**
- **$P_\theta$**: Distribution over trajectories depends on policy parameters
- **$\theta$**: Policy parameters affect which trajectories are likely
- **Direct differentiation**: Can't easily differentiate through expectation
- **Solution needed**: Need a trick to compute gradients from samples

**The Sampling Analogy:**
- **Trajectories**: Like different ways a game could play out
- **Policy parameters**: Like settings that affect which outcomes are likely
- **Gradient**: Like figuring out how to change settings to get better outcomes
- **Challenge**: Can't directly compute how settings affect outcomes

#### The Log-Derivative Trick

The key insight is the **log-derivative trick** (also called the REINFORCE trick):

$$
\nabla_\theta \mathbb{E}_{\tau \sim P_\theta} [f(\tau)] = \mathbb{E}_{\tau \sim P_\theta} \left[ (\nabla_\theta \log P_\theta(\tau)) f(\tau) \right] \tag{17.3}
$$

**The Log-Derivative Intuition:**
- **$f(\tau)$**: Function of trajectory (like total reward)
- **$\nabla_\theta \log P_\theta(\tau)$**: Direction that makes trajectory more likely
- **Product**: High-reward trajectories get their probability increased
- **Expectation**: Average over all possible trajectories

**The Weighted Learning Analogy:**
- **High-reward trajectories**: Get "pushed" in the direction of higher probability
- **Low-reward trajectories**: Get "pushed" in the direction of lower probability
- **Net effect**: Policy moves toward actions that lead to high rewards
- **Sampling**: Use actual trajectories to estimate this effect

**Derivation:**
$$
\nabla_\theta \mathbb{E}_{\tau \sim P_\theta} [f(\tau)] = \nabla_\theta \int P_\theta(\tau) f(\tau) d\tau
= \int \nabla_\theta (P_\theta(\tau) f(\tau)) d\tau \quad \text{(swap integration with gradient)}
= \int (\nabla_\theta P_\theta(\tau)) f(\tau) d\tau \quad \text{(because $f$ does not depend on $\theta$)}
= \int P_\theta(\tau) \frac{\nabla_\theta P_\theta(\tau)}{P_\theta(\tau)} f(\tau) d\tau
= \int P_\theta(\tau) (\nabla_\theta \log P_\theta(\tau)) f(\tau) d\tau
= \mathbb{E}_{\tau \sim P_\theta} \left[ (\nabla_\theta \log P_\theta(\tau)) f(\tau) \right]
$$

**The Mathematical Intuition:**
- **Step 1**: Write expectation as integral
- **Step 2**: Swap gradient and integral (valid under mild conditions)
- **Step 3**: Use product rule and fact that $f$ doesn't depend on $\theta$
- **Step 4**: Multiply and divide by $P_\theta(\tau)$
- **Step 5**: Recognize $\nabla_\theta \log P_\theta(\tau) = \frac{\nabla_\theta P_\theta(\tau)}{P_\theta(\tau)}$
- **Step 6**: Rewrite as expectation

**Intuition:** We can estimate the gradient using only samples from the current policy, without needing to know the environment dynamics.

**The Sampling Advantage:**
- **No model needed**: Don't need to know $P_{sa}$ or $R(s, a)$
- **Data-driven**: Use actual experience to estimate gradients
- **Online learning**: Can update policy after each episode
- **Robust**: Works even with complex, unknown environments

#### Trajectory Probability Decomposition

For a trajectory $\tau = (s_0, a_0, \ldots, s_{T-1}, a_{T-1}, s_T)$:

$$
P_\theta(\tau) = \mu(s_0) \pi_\theta(a_0|s_0) P_{s_0 a_0}(s_1) \pi_\theta(a_1|s_1) P_{s_1 a_1}(s_2) \cdots P_{s_{T-1} a_{T-1}}(s_T) \tag{17.6}
$$

**The Trajectory Decomposition Intuition:**
- **$\mu(s_0)$**: Probability of starting in state $s_0$
- **$\pi_\theta(a_0|s_0)$**: Probability of taking action $a_0$ in state $s_0$
- **$P_{s_0 a_0}(s_1)$**: Probability of transitioning to state $s_1$
- **Pattern**: Alternating between policy decisions and environment transitions

**The Chain of Events Analogy:**
- **Start**: Random initial state
- **Decide**: Policy chooses action
- **Transition**: Environment moves to new state
- **Repeat**: Continue until episode ends
- **Probability**: Product of all these probabilities

Taking the logarithm:
$$
\log P_\theta(\tau) = \log \mu(s_0) + \log \pi_\theta(a_0|s_0) + \log P_{s_0 a_0}(s_1) + \log \pi_\theta(a_1|s_1)
+ \log P_{s_1 a_1}(s_2) + \cdots + \log P_{s_{T-1} a_{T-1}}(s_T) \tag{17.7}
$$

**The Log Decomposition Intuition:**
- **Log of product**: Becomes sum of logs
- **Additive structure**: Each term contributes independently
- **Gradient computation**: Only policy terms depend on $\theta$
- **Environment terms**: Don't affect gradient computation

**Key Insight:** When we take the gradient with respect to $\theta$, only the policy terms survive:

$$
\nabla_\theta \log P_\theta(\tau) = \nabla_\theta \log \pi_\theta(a_0|s_0) + \nabla_\theta \log \pi_\theta(a_1|s_1) + \cdots + \nabla_\theta \log \pi_\theta(a_{T-1}|s_{T-1})
$$

The environment terms ($\log P_{s_t a_t}(s_{t+1})$) don't depend on $\theta$ and thus have zero gradient.

**The Gradient Decomposition Intuition:**
- **Policy terms**: $\nabla_\theta \log \pi_\theta(a_t|s_t)$ - how to make action $a_t$ more likely in state $s_t$
- **Environment terms**: $\nabla_\theta \log P_{s_t a_t}(s_{t+1})$ - zero because environment doesn't depend on $\theta$
- **Net effect**: Only policy decisions affect the gradient
- **Learning**: Adjust policy to make good actions more likely

#### The Policy Gradient Formula

Combining the log-derivative trick with the trajectory decomposition:

$$
\nabla_\theta \eta(\theta) = \mathbb{E}_{\tau \sim P_\theta} \left[ \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \cdot \left( \sum_{t=0}^{T-1} \gamma^t R(s_t, a_t) \right) \right] \tag{17.8}
$$

**The Policy Gradient Intuition:**
- **$\sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t)$**: Direction that increases probability of all taken actions
- **$\sum_{t=0}^{T-1} \gamma^t R(s_t, a_t)$**: Total reward of the trajectory
- **Product**: High-reward trajectories get their actions reinforced more strongly
- **Learning**: Actions that lead to high rewards become more likely

**The Reinforcement Analogy:**
- **High-reward trajectory**: Like a successful strategy
- **Action probabilities**: Like how likely you are to try each move
- **Reinforcement**: Successful strategies get their moves "reinforced"
- **Learning**: Gradually shift toward successful strategies

**The Credit Assignment Problem:**
- **All actions**: Every action in a trajectory gets the same weight (total reward)
- **Problem**: Don't know which actions were actually good
- **Solution**: Use baselines or advantage functions (coming next)
- **Trade-off**: Simple but high variance

---

**Next: [Variance Reduction with Baselines](04_policy_gradient.md#173-variance-reduction-with-baselines)** - Learn how to reduce variance in policy gradient estimates.


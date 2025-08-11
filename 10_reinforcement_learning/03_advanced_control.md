# Advanced Control Methods: LQR, DDP, and LQG

## The Big Picture: Why Advanced Control Methods Matter

**The Control Challenge:**
Imagine trying to land a rocket on Mars, fly a drone through a forest, or balance a robot on one leg. These are complex control problems where simple approaches like "turn left when too far right" don't work. We need sophisticated methods that can handle the complexity of real-world systems.

**The Intuitive Analogy:**
Think of the difference between:
- **Simple control**: Like driving a car on a straight road (basic feedback)
- **Advanced control**: Like landing a plane in crosswinds (complex, multi-variable optimization)

**Why These Methods Matter:**
- **Real-world complexity**: Most systems are nonlinear, noisy, and partially observable
- **Performance requirements**: Need optimal or near-optimal solutions
- **Safety critical**: Failures can be catastrophic (rocket crashes, robot falls)
- **Efficiency**: Need to use minimal energy, time, or resources

### The Key Insight

**From Simple to Sophisticated:**
- **Basic RL**: Learn from trial and error (like learning to ride a bike)
- **Advanced Control**: Use mathematical structure for optimal solutions (like engineering a self-balancing bike)

**The Structure Advantage:**
- **Model-based**: Leverage known physics and dynamics
- **Optimal solutions**: Find the mathematically best strategy
- **Efficiency**: Much faster than learning from scratch
- **Reliability**: Predictable performance with guarantees

## Introduction

This chapter covers three fundamental advanced control methods that extend beyond basic reinforcement learning:

1. **Linear Quadratic Regulation (LQR)** - Optimal control for linear systems with quadratic costs
2. **Differential Dynamic Programming (DDP)** - Iterative trajectory optimization for nonlinear systems
3. **Linear Quadratic Gaussian (LQG)** - Optimal control under partial observability

These methods are essential in robotics, aerospace, and control theory, providing both theoretical insights and practical solutions for complex control problems.

**The Method Hierarchy:**
- **LQR**: The foundation - exact solutions for linear systems
- **DDP**: The extension - handles nonlinear systems through approximation
- **LQG**: The completion - handles uncertainty and partial observations

**The Engineering Analogy:**
- **LQR**: Like designing a suspension system for a car (linear spring-damper)
- **DDP**: Like designing an active suspension that adapts to road conditions (nonlinear, adaptive)
- **LQG**: Like designing a suspension that works with noisy sensors (handles uncertainty)

## From Value Function Approximation to Advanced Control

We've now explored **continuous state MDPs** - extending the MDP framework to handle infinite or continuous state spaces. We've seen how discretization can approximate continuous problems, how value function approximation enables learning in high-dimensional spaces, and how fitted value iteration provides practical algorithms for solving complex control problems.

However, while value function approximation provides powerful tools for handling continuous state spaces, **real-world control problems** often require more sophisticated techniques that leverage the structure of the underlying system. Many physical systems have known dynamics, cost structures, and constraints that can be exploited for more efficient and robust control.

This motivates our exploration of **advanced control methods** - specialized techniques that combine the principles of reinforcement learning with classical control theory. We'll see how Linear Quadratic Regulation (LQR) provides exact solutions for linear systems, how Differential Dynamic Programming (DDP) handles nonlinear systems through iterative optimization, and how Linear Quadratic Gaussian (LQG) control addresses partial observability through optimal state estimation.

The transition from value function approximation to advanced control represents the bridge from general-purpose learning algorithms to domain-specific optimization techniques - taking our understanding of continuous state MDPs and applying it to structured control problems with known dynamics and cost functions.

In this chapter, we'll explore LQR, DDP, and LQG control methods, understanding how they leverage system structure for more efficient and robust control.

---

## Understanding Finite-Horizon Markov Decision Processes

### The Big Picture: Why Finite-Horizon Problems Matter

**The Time Constraint Problem:**
Many real-world problems have natural time limits. A rocket has limited fuel, a robot has a deadline, or a game has a fixed number of moves. These finite-horizon problems require different approaches than infinite-horizon problems.

**The Intuitive Analogy:**
- **Infinite-horizon**: Like planning for retirement (long-term, steady strategy)
- **Finite-horizon**: Like planning a vacation (short-term, time-dependent strategy)

**The Key Insight:**
In finite-horizon problems, the optimal strategy changes over time because the remaining time affects the value of different actions.

### 16.1 Finite-Horizon Markov Decision Processes

#### Motivation and Context

In Chapter 15, we explored infinite-horizon MDPs with stationary policies. However, many real-world problems have finite time horizons and require time-dependent strategies. Consider a rocket landing on Mars - the optimal control strategy changes dramatically as the rocket approaches the surface.

**The Rocket Landing Example:**
- **Early in flight**: Focus on trajectory optimization and fuel efficiency
- **Mid-flight**: Balance trajectory with landing preparation
- **Final approach**: Focus entirely on safe landing, regardless of fuel cost
- **Last few seconds**: Emergency procedures if needed

**The Time-Dependent Strategy:**
- **Time 0**: "I have lots of time, I can be patient"
- **Time T/2**: "I need to start thinking about the end"
- **Time T-1**: "This is my last chance to get it right"

#### Mathematical Framework

We previously defined the **optimal Bellman equation** for infinite-horizon MDPs:

```math
V^{\pi^*}(s) = R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^{\pi^*}(s')
```

From this, we recovered the optimal policy:

```math
\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s')
```

**The Infinite-Horizon Intuition:**
- **Stationary policy**: The same strategy works forever
- **Discount factor**: Future rewards are worth less than immediate rewards
- **Convergence**: The value function converges to a steady state

#### Generalizing to Finite-Horizon Setting

For finite-horizon problems, we make several key generalizations:

##### 1. Continuous and Discrete State Spaces

We use expectation notation that works for both discrete and continuous spaces:

```math
\mathbb{E}_{s' \sim P_{sa}} \left[ V^{\pi^*}(s') \right]
```

**Intuitive Understanding:**
This notation is universal - for discrete spaces, it becomes a sum; for continuous spaces, it becomes an integral.

**The Universal Language Analogy:**
- **Discrete spaces**: Like counting discrete objects (apples, oranges)
- **Continuous spaces**: Like measuring continuous quantities (weight, temperature)
- **Expectation notation**: Like having a universal unit that works for both

##### 2. State-Action Dependent Rewards

We extend rewards to depend on both states and actions: $R : S \times A \to \mathbb{R}$

This changes the optimal action computation to:

```math
\pi^*(s) = \arg\max_{a \in A} R(s, a) + \gamma \mathbb{E}_{s' \sim P_{sa}} \left[ V^{\pi^*}(s') \right]
```

**The Action Cost Intuition:**
- **State-only rewards**: Like getting points for being in good positions
- **State-action rewards**: Like getting points for good positions AND good moves

**Practical Example:** In a car control problem, the reward might be:
- High reward for staying in lane with smooth steering
- Negative reward for jerky movements or leaving the lane

**The Driving Analogy:**
- **State reward**: Points for being in the correct lane
- **Action reward**: Penalty for sudden steering movements
- **Combined reward**: Balance between position and smoothness

##### 3. Finite Time Horizon

We define a **finite-horizon MDP** as the tuple:

```math
(S, A, P_{sa}, T, R)
```

Where $T > 0$ is the **time horizon**. The payoff becomes:

```math
R(s_0, a_0) + R(s_1, a_1) + \cdots + R(s_T, a_T)
```

**The Finite Sum Intuition:**
- **Infinite sum**: Need discounting to ensure convergence
- **Finite sum**: Always converges, no discounting needed
- **Time pressure**: Every action matters because time is limited

**Key Insight:** No discount factor $\gamma$ is needed because we have a finite sum!

#### Why Remove the Discount Factor?

The discount factor $\gamma$ was introduced to ensure convergence of infinite sums:

```math
\left| \sum_{t=0}^\infty R(s_t) \gamma^t \right| \leq \bar{R} \sum_{t=0}^\infty \gamma^t = \frac{\bar{R}}{1-\gamma}
```

For finite sums, convergence is guaranteed without discounting.

**The Convergence Analogy:**
- **Infinite series**: Like an infinite bank account that might not converge
- **Finite series**: Like a fixed-term investment that always has a final value
- **Discount factor**: Like interest rate that makes infinite sums manageable

#### Non-Stationary Policies

A crucial insight is that **optimal policies become time-dependent** in finite-horizon settings:

```math
\pi^{(t)} : S \to A
```

**The Time-Dependent Strategy Intuition:**
- **Early in episode**: Can afford to be patient and plan long-term
- **Middle of episode**: Need to balance immediate and future rewards
- **Near the end**: Focus on immediate rewards since future is limited

**Intuitive Example:** Consider a grid world with two goals (+1 and +10):
- Early in the episode: Aim for the +10 goal (long-term planning)
- Near the end: If closer to +1 goal, switch strategy to maximize immediate reward (short-term optimization)

**The Chess Endgame Analogy:**
- **Opening**: Develop pieces, control center (long-term strategy)
- **Middlegame**: Create tactical opportunities (medium-term planning)
- **Endgame**: Focus on immediate material advantage (short-term tactics)

#### Time-Dependent Dynamics

We can extend to **time-dependent dynamics**:

```math
s_{t+1} \sim P^{(t)}_{s_t, a_t}
```

This models real-world scenarios where system dynamics change over time (e.g., fuel consumption, changing traffic conditions).

**The Changing World Analogy:**
- **Static dynamics**: Like playing chess on a fixed board
- **Time-dependent dynamics**: Like playing chess where the board changes over time

**Real-World Examples:**
- **Rocket**: Fuel consumption changes mass and dynamics
- **Car**: Weather conditions affect road friction
- **Robot**: Battery depletion affects motor performance

#### Value Function Definition

The value function at time $t$ for policy $\pi$ is:

```math
V_t(s) = \mathbb{E} \left[ R^{(t)}(s_t, a_t) + \cdots + R^{(T)}(s_T, a_T) \mid s_t = s, \pi \right]
```

The optimal value function is:

```math
V^*_t(s) = \max_{\pi} V_t^{\pi}(s)
```

**The Time-Dependent Value Intuition:**
- **$V_t(s)$**: How much reward can I expect from state $s$ with $T-t$ time steps remaining?
- **Time pressure**: Less time remaining means less opportunity for long-term planning
- **Urgency**: Value functions change as deadline approaches

**The Countdown Analogy:**
- **Time T**: "I have lots of time to get this right"
- **Time T/2**: "I need to start making progress"
- **Time 1**: "This is my last chance"

#### Dynamic Programming Solution

The beauty of finite-horizon problems is that they naturally fit the **dynamic programming** paradigm:

**The Backward Planning Intuition:**
- **Start from the end**: What should I do in the final state?
- **Work backwards**: Given what I'll do later, what should I do now?
- **Optimal substructure**: Each decision depends only on future optimal decisions

##### Step 1: Terminal Condition
At the final time step $T$:

```math
\forall s \in S: \quad V^*_T(s) := \max_{a \in A} R^{(T)}(s, a)
```

**The Final Decision Intuition:**
- **No future**: At the last step, only immediate reward matters
- **Simple choice**: Pick the action that gives the highest immediate reward
- **No planning**: No need to consider future consequences

##### Step 2: Backward Recursion
For $0 \leq t < T$:

```math
\forall t < T,\ s \in S: \quad V^*_t(s) := \max_{a \in A} \left[ R^{(t)}(s, a) + \mathbb{E}_{s' \sim P^{(t)}_{sa}} \left[ V^*_{t+1}(s') \right] \right]
```

**The Backward Induction Intuition:**
- **Current reward**: What do I get for this action right now?
- **Future value**: What's the best I can do from the next state?
- **Optimal choice**: Pick action that maximizes current + future value

**The Investment Analogy:**
- **Current reward**: Like immediate return on investment
- **Future value**: Like expected future returns
- **Optimal choice**: Like choosing the investment with best total return

#### Algorithm: Backward Induction

1. Compute $V^*_T$ using the terminal condition
2. For $t = T-1, \ldots, 0$:
   - Compute $V^*_t$ using $V^*_{t+1}$ via backward recursion

**The Algorithm Intuition:**
- **Step 1**: Figure out what to do at the very end
- **Step 2**: Work backwards, using future knowledge to make current decisions
- **Result**: Optimal strategy for every state at every time

**Computational Complexity:** $O(T \cdot |S| \cdot |A|)$ for discrete spaces

**The Complexity Breakdown:**
- **T**: Number of time steps
- **|S|**: Number of states
- **|A|**: Number of actions
- **Total**: For each time step, for each state, for each action

#### Connection to Infinite-Horizon Value Iteration

Standard value iteration can be viewed as a special case. Running value iteration for $T$ steps gives a $\gamma^T$ approximation of the optimal value function.

**The Approximation Intuition:**
- **Finite-horizon**: Exact solution for T steps
- **Infinite-horizon**: Approximate solution, gets better with more iterations
- **Connection**: Finite-horizon is like "stopping early" in infinite-horizon

**Theorem (Convergence):** Let $B$ denote the Bellman update and $\|f(x)\|_\infty := \sup_x |f(x)|$. Then:

```math
\|V_{t+1} - V^*\|_\infty = \|B(V_t) - V^*\|_\infty 
\leq \gamma \|V_t - V^*\|_\infty 
\leq \gamma^t \|V_1 - V^*\|_\infty
```

This shows that the Bellman operator $B$ is a $\gamma$-contracting operator.

**The Contraction Intuition:**
- **Contracting operator**: Each iteration brings us closer to the optimal solution
- **$\gamma$-contracting**: The error shrinks by a factor of $\gamma$ each iteration
- **Convergence**: Eventually, we get arbitrarily close to the optimal solution

---

## Understanding Linear Quadratic Regulation (LQR)

### The Big Picture: What is LQR?

**The LQR Problem:**
How do we find the optimal control strategy for a linear system with quadratic costs? This is like finding the perfect steering strategy for a car that responds linearly to inputs and where we want to minimize both position error and control effort.

**The Intuitive Analogy:**
- **Linear system**: Like a car that responds predictably to steering inputs
- **Quadratic cost**: Like wanting to stay in the center of the lane (position error) while using smooth steering (control effort)
- **Optimal control**: Like finding the perfect steering strategy that balances these goals

**Why LQR Matters:**
- **Exact solutions**: No approximation needed for linear systems
- **Wide applicability**: Many systems can be approximated as linear
- **Computational efficiency**: Fast, closed-form solutions
- **Theoretical foundation**: Basis for more complex methods

### 16.2 Linear Quadratic Regulation (LQR)

#### Introduction and Motivation

LQR is one of the most important and widely-used control methods. It provides **exact, closed-form solutions** for a specific but practically important class of problems.

**The Mathematical Beauty:**
- **Linear dynamics**: Simple, predictable system behavior
- **Quadratic costs**: Natural way to penalize deviations and control effort
- **Optimal solution**: Can be computed exactly using matrix algebra
- **Closed-form**: No iterative optimization needed

**Key Applications:**
- Robot arm control
- Aircraft autopilot systems
- Inverted pendulum stabilization
- Car lane-keeping systems

**The Control Hierarchy:**
- **PID control**: Simple, heuristic approach
- **LQR**: Optimal, model-based approach
- **Model Predictive Control**: Advanced, constraint-aware approach

#### Problem Setup

##### State and Action Spaces
We work in continuous spaces:
```math
S = \mathbb{R}^d, \quad A = \mathbb{R}^d
```

**The Continuous Space Intuition:**
- **Discrete spaces**: Like having a finite number of positions
- **Continuous spaces**: Like being able to be anywhere in space
- **Real-world systems**: Most physical systems have continuous states

##### Linear Dynamics with Gaussian Noise
```math
s_{t+1} = A_t s_t + B_t a_t + w_t
```

Where:
- $A_t \in \mathbb{R}^{d \times d}$: State transition matrix
- $B_t \in \mathbb{R}^{d \times d}$: Control input matrix  
- $w_t \sim \mathcal{N}(0, \Sigma_t)$: Gaussian process noise

**The Linear Dynamics Intuition:**
- **$A_t s_t$**: How the state evolves naturally (like a car coasting)
- **$B_t a_t$**: How control inputs affect the state (like steering the car)
- **$w_t$**: Random disturbances (like wind or road bumps)

**The Car Analogy:**
- **State**: Position and velocity of the car
- **$A_t$**: How position and velocity change naturally (physics)
- **$B_t$**: How steering and acceleration affect position and velocity
- **$w_t$**: Wind, road conditions, measurement errors

**Remarkable Result:** The optimal policy is **independent of the noise** (as long as it has zero mean)!

**The Noise Independence Intuition:**
- **Zero mean noise**: Disturbances average out over time
- **Optimal policy**: Focuses on the predictable part of the system
- **Robustness**: Works regardless of noise level (as long as mean is zero)

##### Quadratic Cost Function
```math
R^{(t)}(s_t, a_t) = -s_t^\top U_t s_t - a_t^\top W_t a_t
```

Where $U_t, W_t \in \mathbb{R}^{d \times d}$ are positive definite matrices.

**The Quadratic Cost Intuition:**
- **$-s_t^\top U_t s_t$**: Penalty for being far from desired state (usually origin)
- **$-a_t^\top W_t a_t$**: Penalty for using large control inputs
- **Balance**: Trade-off between accuracy and effort

**The Balancing Act:**
- **High $U_t$**: Really want to stay close to target (aggressive control)
- **High $W_t$**: Really want to use small inputs (conservative control)
- **Balanced**: Find sweet spot between accuracy and effort

**Example:** For $U_t = W_t = I_d$:
```math
R_t = -\|s_t\|^2 - \|a_t\|^2
```

This models a car trying to stay centered in a lane with smooth steering.

**The Lane-Keeping Analogy:**
- **$-\|s_t\|^2$**: Penalty for being off-center (position error)
- **$-\|a_t\|^2$**: Penalty for jerky steering (control effort)
- **Optimal behavior**: Smooth steering that keeps car centered

#### LQR Algorithm

##### Step 1: System Identification (if needed)

If the system parameters are unknown, estimate them using linear regression:

```math
\underset{A, B}{\arg\min} \sum_{i=1}^n \sum_{t=0}^{T-1} \left\| s_{t+1}^{(i)} - \left( A s_t^{(i)} + B a_t^{(i)} \right) \right\|^2
```

**The System Identification Intuition:**
- **Unknown system**: Like not knowing how a car responds to steering
- **Data collection**: Drive the car and record what happens
- **Linear regression**: Find the best linear model that fits the data
- **Model validation**: Test the model on new data

The noise covariance $\Sigma$ can be estimated using techniques from Gaussian Discriminant Analysis.

##### Step 2: Optimal Policy Computation

Given the system parameters, we solve for the optimal policy using dynamic programming.

**Key Insight:** The optimal value function is **quadratic** in the state!

**The Quadratic Value Function Intuition:**
- **Linear dynamics**: State changes linearly with inputs
- **Quadratic cost**: Cost is quadratic in state and action
- **Quadratic value**: Value function inherits quadratic structure
- **Mathematical beauty**: Quadratic functions have simple optimal solutions

### Mathematical Derivation

#### Step 1: Terminal Condition
At time $T$:
$$
V^\ast_T(s_T) = \max_{a_T \in A} R_T(s_T, a_T)
= \max_{a_T \in A} -s_T^\top U_T s_T - a_T^\top W_T a_T
= -s_T^\top U_T s_T \qquad \text{(maximized for $a_T = 0$)}
$$

**The Terminal Condition Intuition:**
- **Last time step**: No future to consider
- **Optimal action**: $a_T = 0$ (no control input)
- **Value function**: Just the cost of the final state
- **Quadratic form**: $V_T(s) = s^\top \Phi_T s$ where $\Phi_T = -U_T$

#### Step 2: Backward Recursion

**Fact 1:** If $V^*_{t+1}$ is quadratic, then $V^*_t$ is also quadratic.

**Fact 2:** The optimal policy is **linear** in the state.

**The Mathematical Induction Intuition:**
- **Base case**: Terminal value function is quadratic
- **Induction step**: If $V_{t+1}$ is quadratic, then $V_t$ is quadratic
- **Conclusion**: All value functions are quadratic
- **Implication**: Optimal policy is linear (derivative of quadratic is linear)

**Derivation:** For $t < T$, assuming we know $V^*_{t+1}$:

```math
V^*_t(s_t) = s_t^\top \Phi_t s_t + \Psi_t
= \max_{a_t} \left[ R^{(t)}(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim P^{(t)}_{sa}} [V^*_{t+1}(s_{t+1})] \right]
= \max_{a_t} \left[ -s_t^\top U_t s_t - a_t^\top W_t a_t + \mathbb{E}_{s_{t+1} \sim \mathcal{N}(A_t s_t + B_t a_t, \Sigma_t)} [s_{t+1}^\top \Phi_{t+1} s_{t+1} + \Psi_{t+1}] \right]
```

**The Backward Recursion Intuition:**
- **Current cost**: $-s_t^\top U_t s_t - a_t^\top W_t a_t$
- **Future value**: Expected value of next state
- **Optimization**: Find action that maximizes current + future value
- **Quadratic structure**: Maintains quadratic form through recursion

**Key Identity:** For $w_t \sim \mathcal{N}(0, \Sigma_t)$:
```math
\mathbb{E} [w_t^\top \Phi_{t+1} w_t] = \mathrm{Tr}(\Sigma_t \Phi_{t+1})
```

**The Noise Expectation Intuition:**
- **Gaussian noise**: Random disturbances with known statistics
- **Quadratic form**: Noise appears in quadratic terms
- **Expectation**: Average over all possible noise realizations
- **Trace formula**: Mathematical result for Gaussian quadratic forms

#### Optimal Action Computation

The expression above is quadratic in $a_t$ and can be optimized analytically:

```math
a^*_t = \left[ (B_t^\top \Phi_{t+1} B_t - W_t)^{-1} B_t^\top \Phi_{t+1} A_t \right] \cdot s_t = L_t \cdot s_t
```

Where:
```math
L_t := (B_t^\top \Phi_{t+1} B_t - W_t)^{-1} B_t^\top \Phi_{t+1} A_t
```

**The Linear Policy Intuition:**
- **Quadratic optimization**: Maximum of quadratic function
- **Linear solution**: Optimal action is linear in state
- **Gain matrix**: $L_t$ determines how much to control based on state
- **Feedback control**: Control input depends on current state

**Remarkable Result:** The optimal policy is **linear** in the state!

**The Linear Feedback Analogy:**
- **State feedback**: Control input proportional to state error
- **Gain scheduling**: Different gains for different time steps
- **Optimal gains**: Mathematically optimal feedback coefficients

### Discrete Riccati Equations

Solving for $\Phi_t$ and $\Psi_t$ yields the **Discrete Riccati Equations**:

```math
\Phi_t = A_t^\top \left( \Phi_{t+1} - \Phi_{t+1} B_t (B_t^\top \Phi_{t+1} B_t - W_t)^{-1} B_t^\top \Phi_{t+1} \right) A_t - U_t
```

```math
\Psi_t = -\mathrm{tr}(\Sigma_t \Phi_{t+1}) + \Psi_{t+1}
```

**The Riccati Equation Intuition:**
- **Backward recursion**: Each $\Phi_t$ depends on $\Phi_{t+1}$
- **Matrix algebra**: Complex but computable matrix operations
- **Optimal gains**: Riccati equations give optimal feedback gains
- **Time-varying**: Gains change over time (non-stationary policy)

**Key Insights:**
1. $\Phi_t$ depends only on $\Phi_{t+1}$, not on $\Psi_t$ or $\Sigma_t$
2. The optimal policy is independent of the noise!
3. Only $\Psi_t$ depends on $\Sigma_t$

**The Noise Independence Intuition:**
- **$\Phi_t$**: Determines optimal feedback gains
- **$\Psi_t$**: Determines expected cost due to noise
- **Policy**: Only depends on gains, not on noise cost
- **Robustness**: Optimal policy works regardless of noise level

### Complete LQR Algorithm

1. **Initialize:** $\Phi_T := -U_T$ and $\Psi_T := 0$
2. **Backward Pass:** For $t = T-1, \ldots, 0$:
   - Update $\Phi_t$ and $\Psi_t$ using the Riccati equations
3. **Forward Pass:** For $t = 0, \ldots, T-1$:
   - Compute optimal action: $a^*_t = L_t s_t$

**The Algorithm Intuition:**
- **Backward pass**: Compute optimal gains for each time step
- **Forward pass**: Apply optimal control using computed gains
- **Offline computation**: All gains computed before execution
- **Online execution**: Simple linear feedback during operation

**Computational Complexity:** $O(T \cdot d^3)$ where $d$ is the state dimension.

**The Complexity Breakdown:**
- **T**: Number of time steps
- **d³**: Matrix operations for each time step
- **Total**: Reasonable for moderate state dimensions

### Practical Considerations

#### Stability Conditions
- The system must be **stabilizable** (controllable to the origin)
- Cost matrices $U_t, W_t$ must be positive definite
- Convergence is guaranteed if a stabilizing policy exists

**The Stability Intuition:**
- **Stabilizable**: Can control system to any desired state
- **Positive definite costs**: Ensures well-defined optimization problem
- **Convergence**: Riccati equations converge to optimal solution

#### Implementation Notes
- Since the optimal policy doesn't depend on $\Psi_t$, we can skip computing $\Psi_t$ for efficiency
- The Riccati equations can be solved efficiently using matrix operations
- LQR provides globally optimal solutions for linear systems

**The Implementation Efficiency:**
- **Skip $\Psi_t$**: Only need gains, not noise cost
- **Matrix operations**: Use optimized linear algebra libraries
- **Global optimality**: No local optima to worry about

---

## Understanding the Transition to Nonlinear Systems

### The Big Picture: Why Linearization Matters

**The Nonlinear Reality:**
Most real-world systems are nonlinear. A pendulum swings nonlinearly, a car's dynamics change with speed, and a robot's joints have complex interactions. However, LQR only works for linear systems.

**The Intuitive Analogy:**
- **Linear systems**: Like a spring that responds proportionally to force
- **Nonlinear systems**: Like a spring that gets stiffer as you stretch it
- **Linearization**: Like approximating the nonlinear spring as linear near a specific point

**The Key Insight:**
We can use LQR on nonlinear systems by approximating them as linear near a reference trajectory.

### 16.3 From Nonlinear Dynamics to LQR

#### Motivation

While LQR provides elegant solutions, most real-world systems are nonlinear. However, many nonlinear problems can be **approximated** using LQR through linearization.

**The Approximation Strategy:**
- **Local linearization**: Approximate nonlinear system as linear near operating point
- **Reference trajectory**: Follow a nominal path where linearization is valid
- **Feedback control**: Use LQR to correct deviations from nominal trajectory

**Examples:**
- Inverted pendulum
- Robot manipulators
- Aircraft dynamics
- Chemical processes

**The Operating Point Analogy:**
- **Nonlinear system**: Like a complex machine with many operating modes
- **Operating point**: Like a specific setting where the machine works well
- **Linear approximation**: Like a simple model that works near that setting

#### Linearization Approach

##### Intuitive Idea

If a system spends most of its time near a reference trajectory, we can approximate the nonlinear dynamics using a **first-order Taylor expansion**.

**The Taylor Expansion Intuition:**
- **Nonlinear function**: Like a curved road
- **Linear approximation**: Like approximating the road as straight near a point
- **Accuracy**: Good approximation near the point, gets worse farther away

##### Mathematical Formulation

Consider nonlinear dynamics:
```math
s_{t+1} = F(s_t, a_t)
```

Around a reference point $(\bar{s}_t, \bar{a}_t)$, we linearize:

```math
s_{t+1} \approx F(\bar{s}_t, \bar{a}_t) + \nabla_s F(\bar{s}_t, \bar{a}_t) \cdot (s_t - \bar{s}_t) + \nabla_a F(\bar{s}_t, \bar{a}_t) \cdot (a_t - \bar{a}_t)
```

This can be rewritten as:
```math
s_{t+1} \approx A s_t + B a_t + \kappa
```

Where:
- $A = \nabla_s F(\bar{s}_t, \bar{a}_t)$
- $B = \nabla_a F(\bar{s}_t, \bar{a}_t)$
- $\kappa = F(\bar{s}_t, \bar{a}_t) - A \bar{s}_t - B \bar{a}_t$

**The Linearization Process:**
1. **Choose reference point**: Pick $(\bar{s}_t, \bar{a}_t)$ near expected operation
2. **Compute derivatives**: Find $\nabla_s F$ and $\nabla_a F$ at reference point
3. **Form linear approximation**: $s_{t+1} \approx A s_t + B a_t + \kappa$
4. **Apply LQR**: Use linear approximation for control design

##### Handling the Constant Term

The constant term $\kappa$ can be eliminated by augmenting the state space:
```math
\tilde{s}_t = \begin{pmatrix} s_t \\ 1 \end{pmatrix}
```

This is the same trick used in linear regression to handle bias terms.

**The State Augmentation Intuition:**
- **Constant term**: Like a bias that doesn't depend on state or action
- **Augmented state**: Include constant as part of state vector
- **Linear form**: Eliminates constant term from dynamics
- **Standard LQR**: Can now apply standard LQR methods

### Example: Inverted Pendulum

Consider an inverted pendulum with state $s_t = [\theta_t, \dot{\theta}_t]^\top$:

**Nonlinear Dynamics:**
```math
\begin{pmatrix}
    \theta_{t+1} \\
    \dot{\theta}_{t+1}
\end{pmatrix}
= F\left(
    \begin{pmatrix}
        \theta_t \\
        \dot{\theta}_t
    \end{pmatrix}, a_t
\right)
```

**Linearization around $\theta = 0$ (upright position):**
```math
F(s_t, a_t) \approx F(0, 0) + \nabla_s F(0, 0) \cdot s_t + \nabla_a F(0, 0) \cdot a_t
```

This gives us the linear approximation needed for LQR.

**The Pendulum Analogy:**
- **Upright position**: $\theta = 0$ (desired equilibrium)
- **Small deviations**: Near upright, pendulum behaves linearly
- **Large deviations**: Far from upright, nonlinear effects dominate
- **Control strategy**: Keep pendulum near upright where linearization is valid

---

## 16.4 Differential Dynamic Programming (DDP)

### Motivation

While simple linearization works for systems near equilibrium, many problems require following complex trajectories. DDP addresses this by iteratively improving trajectory approximations.

**Applications:**
- Rocket landing
- Robot path planning
- Trajectory optimization
- Motion planning

### DDP Algorithm Overview

DDP is an **iterative trajectory optimization** method that:
1. Starts with a nominal trajectory
2. Linearizes around this trajectory
3. Solves the resulting LQR problem
4. Updates the trajectory
5. Repeats until convergence

### Step-by-Step Derivation

#### Step 1: Nominal Trajectory

Start with a nominal trajectory using a simple controller:
```math
s^*_0, a^*_0 \to s^*_1, a^*_1 \to \ldots \to s^*_T
```

This could be a simple proportional controller or even random actions.

#### Step 2: Linearization Around Trajectory

At each trajectory point $(s^*_t, a^*_t)$, linearize the dynamics:

```math
s_{t+1} \approx F(s^*_t, a^*_t) + \nabla_s F(s^*_t, a^*_t)(s_t - s^*_t) + \nabla_a F(s^*_t, a^*_t)(a_t - a^*_t)
```

This gives us time-varying linear dynamics:
```math
s_{t+1} = A_t \cdot s_t + B_t \cdot a_t
```

#### Step 3: Cost Function Linearization

Similarly, linearize the cost function using second-order Taylor expansion:

```math
\begin{align*}
R(s_t, a_t) \approx & R(s^*_t, a^*_t) + \nabla_s R(s^*_t, a^*_t)(s_t - s^*_t) + \nabla_a R(s^*_t, a^*_t)(a_t - a^*_t) \\
& + \frac{1}{2}(s_t - s^*_t)^\top H_{ss}(s_t - s^*_t) + (s_t - s^*_t)^\top H_{sa}(a_t - a^*_t) \\
& + \frac{1}{2}(a_t - a^*_t)^\top H_{aa}(a_t - a^*_t)
\end{align*}
```

Where $H_{xy}$ are Hessian blocks evaluated at $(s^*_t, a^*_t)$.

This can be rewritten in quadratic form:
```math
R_t(s_t, a_t) = -s_t^\top U_t s_t - a_t^\top W_t a_t
```

#### Step 4: LQR Solution

Now we have a standard LQR problem that can be solved using the methods from Section 16.2.

#### Step 5: Trajectory Update

Use the new optimal policy to generate an improved trajectory:
```math
s^*_0, \pi_0(s^*_0) \to s^*_1, \pi_1(s^*_1) \to \ldots \to s^*_T
```

**Important:** Use the **true nonlinear dynamics** $F$ for trajectory generation, not the linear approximation.

#### Step 6: Iteration

Repeat steps 2-5 until convergence or maximum iterations reached.

### Convergence Properties

- DDP typically converges in 5-20 iterations
- Each iteration improves the trajectory
- Convergence is not guaranteed but is common in practice
- The method can get stuck in local optima

### Practical Considerations

#### Line Search
To improve convergence, often use a line search:
```math
a_t = a^*_t + \alpha \Delta a_t
```

Where $\alpha \in (0, 1]$ is a step size.

#### Regularization
Add regularization to the cost function to ensure numerical stability:
```math
R_t(s_t, a_t) = -s_t^\top U_t s_t - a_t^\top W_t a_t - \lambda \|a_t\|^2
```

#### Multiple Shooting
For long trajectories, use multiple shooting to improve numerical stability.

---

## 16.5 Linear Quadratic Gaussian (LQG)

### Motivation

In real-world applications, we often cannot observe the full state directly. Instead, we receive noisy, partial observations. LQG extends LQR to handle **partial observability**.

**Examples:**
- Self-driving cars (camera images, not full state)
- Robot navigation (sensor readings, not exact position)
- Financial systems (market data, not underlying state)

### Partially Observable MDPs (POMDPs)

#### Problem Formulation

A POMDP adds an observation layer to the MDP framework:

```math
(S, O, A, P_{sa}, T, R)
```

Where:
- $O$: Observation space
- $o_t \sim O(o \mid s_t)$: Observation distribution

#### Belief State

The key insight is to maintain a **belief state** - a distribution over possible states:

```math
b_t(s) = P(s_t = s \mid o_1, \ldots, o_t)
```

The policy then maps belief states to actions: $\pi : \Delta(S) \to A$

### LQG Problem Setup

#### System Dynamics
```math
s_{t+1} = A \cdot s_t + B \cdot a_t + w_t
```

#### Observation Model
```math
y_t = C \cdot s_t + v_t
```

Where:
- $C \in \mathbb{R}^{n \times d}$: Observation matrix
- $v_t \sim \mathcal{N}(0, \Sigma_y)$: Observation noise
- $n < d$: Partial observability (fewer observations than state dimensions)

#### Cost Function
Unchanged from LQR:
```math
R^{(t)}(s_t, a_t) = -s_t^\top U_t s_t - a_t^\top W_t a_t
```

### LQG Solution Strategy

The LQG solution follows a **separation principle**:

1. **State Estimation:** Use Kalman filter to estimate the state
2. **Optimal Control:** Apply LQR using the estimated state

#### Step 1: State Estimation via Kalman Filter

The Kalman filter provides the optimal state estimate:
```math
s_t | y_1, \ldots, y_t \sim \mathcal{N}(s_{t|t}, \Sigma_{t|t})
```

#### Step 2: Optimal Control

Apply the LQR policy using the estimated state:
```math
a_t = L_t s_{t|t}
```

### Kalman Filter Derivation

#### Predict Step

Given the current belief state $s_{t|t} \sim \mathcal{N}(s_{t|t}, \Sigma_{t|t})$:

**State Prediction:**
```math
s_{t+1|t} = A \cdot s_{t|t}
```

**Covariance Prediction:**
```math
\Sigma_{t+1|t} = A \cdot \Sigma_{t|t} \cdot A^\top + \Sigma_s
```

#### Update Step

When new observation $y_{t+1}$ arrives:

**State Update:**
```math
s_{t+1|t+1} = s_{t+1|t} + K_t (y_{t+1} - C s_{t+1|t})
```

**Covariance Update:**
```math
\Sigma_{t+1|t+1} = \Sigma_{t+1|t} - K_t C \Sigma_{t+1|t}
```

**Kalman Gain:**
```math
K_t = \Sigma_{t+1|t} C^T (C \Sigma_{t+1|t} C^T + \Sigma_y)^{-1}
```

### Intuitive Understanding

#### Why Kalman Filter Works

The Kalman filter optimally combines:
1. **Model prediction** (what we expect based on dynamics)
2. **New observation** (what sensors tell us)

The **Kalman gain** $K_t$ determines how much to trust each source:
- High $K_t$: Trust the observation more
- Low $K_t$: Trust the model prediction more

#### Mathematical Intuition

The Kalman gain minimizes the posterior variance:
```math
K_t = \arg\min_{K} \text{Var}(s_{t+1} | y_1, \ldots, y_{t+1})
```

This gives the optimal linear estimator under Gaussian assumptions.

### Complete LQG Algorithm

#### Forward Pass (Kalman Filter)
1. Initialize: $s_{0|0} = \mu_0$, $\Sigma_{0|0} = \Sigma_0$
2. For $t = 0, \ldots, T-1$:
   - Predict: $s_{t+1|t}$, $\Sigma_{t+1|t}$
   - Update: $s_{t+1|t+1}$, $\Sigma_{t+1|t+1}$

#### Backward Pass (LQR)
1. Initialize: $\Phi_T = -U_T$
2. For $t = T-1, \ldots, 0$:
   - Compute $L_t$ using Riccati equations
   - Update $\Phi_t$

#### Control Execution
For $t = 0, \ldots, T-1$:
```math
a_t = L_t s_{t|t}
```

### Computational Complexity

- **Kalman Filter:** $O(d^3)$ per time step
- **LQR Backward Pass:** $O(T \cdot d^3)$
- **Total:** $O(T \cdot d^3)$

### Practical Considerations

#### Robustness
- LQG assumes Gaussian noise - may not hold in practice
- Extensions like Extended Kalman Filter (EKF) handle nonlinear observations
- Unscented Kalman Filter (UKF) provides better nonlinear approximations

#### Tuning
- Process noise covariance $\Sigma_s$ affects how much to trust the model
- Observation noise covariance $\Sigma_y$ affects how much to trust sensors
- These parameters often need tuning for good performance

#### Limitations
- Assumes linear dynamics and Gaussian noise
- May not work well with highly nonlinear systems
- Requires good initial state estimate

### Historical Context

The Kalman filter was famously used in the **Apollo Lunar Module** for navigation. It provided real-time state estimation despite limited computational resources, demonstrating the power of optimal estimation theory.

---

## Summary and Connections

### Method Comparison

| Method | Dynamics | Observability | Solution Type | Applications |
|--------|----------|---------------|---------------|--------------|
| LQR | Linear | Full | Exact | Robot control, aircraft |
| DDP | Nonlinear | Full | Iterative | Trajectory optimization |
| LQG | Linear | Partial | Exact | Sensor fusion, robotics |

### Key Insights

1. **LQR** provides exact solutions for linear systems with quadratic costs
2. **DDP** extends this to nonlinear systems through iterative linearization
3. **LQG** handles partial observability through optimal state estimation
4. All methods build on the principle of **dynamic programming**

### Practical Guidelines

1. **Start with LQR** for linear systems
2. **Use DDP** when nonlinearities are significant
3. **Apply LQG** when full state observation is impossible
4. **Combine methods** for complex problems (e.g., DDP with state estimation)

### Further Reading

- **Optimal Control Theory:** Bellman, Pontryagin's principle
- **Estimation Theory:** Kalman filter variants, particle filters
- **Robotics Applications:** Model predictive control, receding horizon control
- **Deep Learning Integration:** Neural network policies, end-to-end learning

These methods form the foundation of modern control theory and continue to be essential tools in robotics, aerospace, and autonomous systems.

## From Model-Based Control to Model-Free Learning

We've now explored **advanced control methods** - specialized techniques that combine the principles of reinforcement learning with classical control theory. We've seen how Linear Quadratic Regulation (LQR) provides exact solutions for linear systems, how Differential Dynamic Programming (DDP) handles nonlinear systems through iterative optimization, and how Linear Quadratic Gaussian (LQG) control addresses partial observability through optimal state estimation.

However, while these model-based control methods are powerful when we have good models of the system dynamics, **many real-world problems** involve systems where the dynamics are unknown, complex, or difficult to model accurately. In these cases, we need methods that can learn optimal behavior directly from experience without requiring explicit models of the environment.

This motivates our exploration of **policy gradient methods** - model-free reinforcement learning techniques that directly optimize policy parameters using gradient ascent. We'll see how REINFORCE learns policies from experience, how variance reduction techniques improve learning efficiency, and how these methods enable learning in complex, unknown environments where model-based approaches are not feasible.

The transition from model-based control to model-free learning represents the bridge from structured optimization to adaptive learning - taking our understanding of optimal control and extending it to scenarios where system models are unknown or unreliable.

In the next section, we'll explore policy gradient methods, understanding how they learn optimal policies directly from experience without requiring explicit models of the environment.

---

**Previous: [Continuous State MDPs](02_continuous_state_mdp.md)** - Learn how to handle infinite and continuous state spaces.

**Next: [Differential Dynamic Programming (DDP)](03_advanced_control.md#164-differential-dynamic-programming-ddp)** - Learn iterative trajectory optimization for nonlinear systems.

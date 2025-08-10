# Advanced Control Methods: LQR, DDP, and LQG

## Introduction

This chapter covers three fundamental advanced control methods that extend beyond basic reinforcement learning:

1. **Linear Quadratic Regulation (LQR)** - Optimal control for linear systems with quadratic costs
2. **Differential Dynamic Programming (DDP)** - Iterative trajectory optimization for nonlinear systems
3. **Linear Quadratic Gaussian (LQG)** - Optimal control under partial observability

These methods are essential in robotics, aerospace, and control theory, providing both theoretical insights and practical solutions for complex control problems.

## From Value Function Approximation to Advanced Control

We've now explored **continuous state MDPs** - extending the MDP framework to handle infinite or continuous state spaces. We've seen how discretization can approximate continuous problems, how value function approximation enables learning in high-dimensional spaces, and how fitted value iteration provides practical algorithms for solving complex control problems.

However, while value function approximation provides powerful tools for handling continuous state spaces, **real-world control problems** often require more sophisticated techniques that leverage the structure of the underlying system. Many physical systems have known dynamics, cost structures, and constraints that can be exploited for more efficient and robust control.

This motivates our exploration of **advanced control methods** - specialized techniques that combine the principles of reinforcement learning with classical control theory. We'll see how Linear Quadratic Regulation (LQR) provides exact solutions for linear systems, how Differential Dynamic Programming (DDP) handles nonlinear systems through iterative optimization, and how Linear Quadratic Gaussian (LQG) control addresses partial observability through optimal state estimation.

The transition from value function approximation to advanced control represents the bridge from general-purpose learning algorithms to domain-specific optimization techniques - taking our understanding of continuous state MDPs and applying it to structured control problems with known dynamics and cost functions.

In this chapter, we'll explore LQR, DDP, and LQG control methods, understanding how they leverage system structure for more efficient and robust control.

---

## 16.1 Finite-Horizon Markov Decision Processes

### Motivation and Context

In Chapter 15, we explored infinite-horizon MDPs with stationary policies. However, many real-world problems have finite time horizons and require time-dependent strategies. Consider a rocket landing on Mars - the optimal control strategy changes dramatically as the rocket approaches the surface.

### Mathematical Framework

We previously defined the **optimal Bellman equation** for infinite-horizon MDPs:

```math
V^{\pi^*}(s) = R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^{\pi^*}(s')
```

From this, we recovered the optimal policy:

```math
\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s')
```

### Generalizing to Finite-Horizon Setting

For finite-horizon problems, we make several key generalizations:

#### 1. Continuous and Discrete State Spaces

We use expectation notation that works for both discrete and continuous spaces:

```math
\mathbb{E}_{s' \sim P_{sa}} \left[ V^{\pi^*}(s') \right]
```

**Intuition:** This notation is universal - for discrete spaces, it becomes a sum; for continuous spaces, it becomes an integral.

#### 2. State-Action Dependent Rewards

We extend rewards to depend on both states and actions: $R : S \times A \to \mathbb{R}$

This changes the optimal action computation to:

```math
\pi^*(s) = \arg\max_{a \in A} R(s, a) + \gamma \mathbb{E}_{s' \sim P_{sa}} \left[ V^{\pi^*}(s') \right]
```

**Practical Example:** In a car control problem, the reward might be:
- High reward for staying in lane with smooth steering
- Negative reward for jerky movements or leaving the lane

#### 3. Finite Time Horizon

We define a **finite-horizon MDP** as the tuple:

```math
(S, A, P_{sa}, T, R)
```

Where $T > 0$ is the **time horizon**. The payoff becomes:

```math
R(s_0, a_0) + R(s_1, a_1) + \cdots + R(s_T, a_T)
```

**Key Insight:** No discount factor $\gamma$ is needed because we have a finite sum!

### Why Remove the Discount Factor?

The discount factor $\gamma$ was introduced to ensure convergence of infinite sums:

```math
\left| \sum_{t=0}^\infty R(s_t) \gamma^t \right| \leq \bar{R} \sum_{t=0}^\infty \gamma^t = \frac{\bar{R}}{1-\gamma}
```

For finite sums, convergence is guaranteed without discounting.

### Non-Stationary Policies

A crucial insight is that **optimal policies become time-dependent** in finite-horizon settings:

```math
\pi^{(t)} : S \to A
```

**Intuitive Example:** Consider a grid world with two goals (+1 and +10):
- Early in the episode: Aim for the +10 goal
- Near the end: If closer to +1 goal, switch strategy to maximize immediate reward

### Time-Dependent Dynamics

We can extend to **time-dependent dynamics**:

```math
s_{t+1} \sim P^{(t)}_{s_t, a_t}
```

This models real-world scenarios where system dynamics change over time (e.g., fuel consumption, changing traffic conditions).

### Value Function Definition

The value function at time $t$ for policy $\pi$ is:

```math
V_t(s) = \mathbb{E} \left[ R^{(t)}(s_t, a_t) + \cdots + R^{(T)}(s_T, a_T) \mid s_t = s, \pi \right]
```

The optimal value function is:

```math
V^*_t(s) = \max_{\pi} V_t^{\pi}(s)
```

### Dynamic Programming Solution

The beauty of finite-horizon problems is that they naturally fit the **dynamic programming** paradigm:

#### Step 1: Terminal Condition
At the final time step $T$:

```math
\forall s \in S: \quad V^*_T(s) := \max_{a \in A} R^{(T)}(s, a)
```

#### Step 2: Backward Recursion
For $0 \leq t < T$:

```math
\forall t < T,\ s \in S: \quad V^*_t(s) := \max_{a \in A} \left[ R^{(t)}(s, a) + \mathbb{E}_{s' \sim P^{(t)}_{sa}} \left[ V^*_{t+1}(s') \right] \right]
```

#### Algorithm: Backward Induction

1. Compute $V^*_T$ using the terminal condition
2. For $t = T-1, \ldots, 0$:
   - Compute $V^*_t$ using $V^*_{t+1}$ via backward recursion

**Computational Complexity:** $O(T \cdot |S| \cdot |A|)$ for discrete spaces

### Connection to Infinite-Horizon Value Iteration

Standard value iteration can be viewed as a special case. Running value iteration for $T$ steps gives a $\gamma^T$ approximation of the optimal value function.

**Theorem (Convergence):** Let $B$ denote the Bellman update and $\|f(x)\|_\infty := \sup_x |f(x)|$. Then:

```math
\|V_{t+1} - V^*\|_\infty = \|B(V_t) - V^*\|_\infty 
\leq \gamma \|V_t - V^*\|_\infty 
\leq \gamma^t \|V_1 - V^*\|_\infty
```

This shows that the Bellman operator $B$ is a $\gamma$-contracting operator.

---

## 16.2 Linear Quadratic Regulation (LQR)

### Introduction and Motivation

LQR is one of the most important and widely-used control methods. It provides **exact, closed-form solutions** for a specific but practically important class of problems.

**Key Applications:**
- Robot arm control
- Aircraft autopilot systems
- Inverted pendulum stabilization
- Car lane-keeping systems

### Problem Setup

#### State and Action Spaces
We work in continuous spaces:
```math
S = \mathbb{R}^d, \quad A = \mathbb{R}^d
```

#### Linear Dynamics with Gaussian Noise
```math
s_{t+1} = A_t s_t + B_t a_t + w_t
```

Where:
- $A_t \in \mathbb{R}^{d \times d}$: State transition matrix
- $B_t \in \mathbb{R}^{d \times d}$: Control input matrix  
- $w_t \sim \mathcal{N}(0, \Sigma_t)$: Gaussian process noise

**Remarkable Result:** The optimal policy is **independent of the noise** (as long as it has zero mean)!

#### Quadratic Cost Function
```math
R^{(t)}(s_t, a_t) = -s_t^\top U_t s_t - a_t^\top W_t a_t
```

Where $U_t, W_t \in \mathbb{R}^{d \times d}$ are positive definite matrices.

**Intuition:** This encourages:
- States close to the origin (minimize $\|s_t\|^2$)
- Small control inputs (minimize $\|a_t\|^2$)

**Example:** For $U_t = W_t = I_d$:
```math
R_t = -\|s_t\|^2 - \|a_t\|^2
```

This models a car trying to stay centered in a lane with smooth steering.

### LQR Algorithm

#### Step 1: System Identification (if needed)

If the system parameters are unknown, estimate them using linear regression:

```math
\underset{A, B}{\arg\min} \sum_{i=1}^n \sum_{t=0}^{T-1} \left\| s_{t+1}^{(i)} - \left( A s_t^{(i)} + B a_t^{(i)} \right) \right\|^2
```

The noise covariance $\Sigma$ can be estimated using techniques from Gaussian Discriminant Analysis.

#### Step 2: Optimal Policy Computation

Given the system parameters, we solve for the optimal policy using dynamic programming.

**Key Insight:** The optimal value function is **quadratic** in the state!

### Mathematical Derivation

#### Step 1: Terminal Condition
At time $T$:
$$
V^\ast_T(s_T) = \max_{a_T \in A} R_T(s_T, a_T)
= \max_{a_T \in A} -s_T^\top U_T s_T - a_T^\top W_T a_T
= -s_T^\top U_T s_T \qquad \text{(maximized for $a_T = 0$)}
$$

#### Step 2: Backward Recursion

**Fact 1:** If $V^*_{t+1}$ is quadratic, then $V^*_t$ is also quadratic.

**Fact 2:** The optimal policy is **linear** in the state.

**Derivation:** For $t < T$, assuming we know $V^*_{t+1}$:

```math
V^*_t(s_t) = s_t^\top \Phi_t s_t + \Psi_t
= \max_{a_t} \left[ R^{(t)}(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim P^{(t)}_{sa}} [V^*_{t+1}(s_{t+1})] \right]
= \max_{a_t} \left[ -s_t^\top U_t s_t - a_t^\top W_t a_t + \mathbb{E}_{s_{t+1} \sim \mathcal{N}(A_t s_t + B_t a_t, \Sigma_t)} [s_{t+1}^\top \Phi_{t+1} s_{t+1} + \Psi_{t+1}] \right]
```

**Key Identity:** For $w_t \sim \mathcal{N}(0, \Sigma_t)$:
```math
\mathbb{E} [w_t^\top \Phi_{t+1} w_t] = \mathrm{Tr}(\Sigma_t \Phi_{t+1})
```

#### Optimal Action Computation

The expression above is quadratic in $a_t$ and can be optimized analytically:

```math
a^*_t = \left[ (B_t^\top \Phi_{t+1} B_t - W_t)^{-1} B_t^\top \Phi_{t+1} A_t \right] \cdot s_t = L_t \cdot s_t
```

Where:
```math
L_t := (B_t^\top \Phi_{t+1} B_t - W_t)^{-1} B_t^\top \Phi_{t+1} A_t
```

**Remarkable Result:** The optimal policy is **linear** in the state!

### Discrete Riccati Equations

Solving for $\Phi_t$ and $\Psi_t$ yields the **Discrete Riccati Equations**:

```math
\Phi_t = A_t^\top \left( \Phi_{t+1} - \Phi_{t+1} B_t (B_t^\top \Phi_{t+1} B_t - W_t)^{-1} B_t^\top \Phi_{t+1} \right) A_t - U_t
```

```math
\Psi_t = -\mathrm{tr}(\Sigma_t \Phi_{t+1}) + \Psi_{t+1}
```

**Key Insights:**
1. $\Phi_t$ depends only on $\Phi_{t+1}$, not on $\Psi_t$ or $\Sigma_t$
2. The optimal policy is independent of the noise!
3. Only $\Psi_t$ depends on $\Sigma_t$

### Complete LQR Algorithm

1. **Initialize:** $\Phi_T := -U_T$ and $\Psi_T := 0$
2. **Backward Pass:** For $t = T-1, \ldots, 0$:
   - Update $\Phi_t$ and $\Psi_t$ using the Riccati equations
3. **Forward Pass:** For $t = 0, \ldots, T-1$:
   - Compute optimal action: $a^*_t = L_t s_t$

**Computational Complexity:** $O(T \cdot d^3)$ where $d$ is the state dimension.

### Practical Considerations

#### Stability Conditions
- The system must be **stabilizable** (controllable to the origin)
- Cost matrices $U_t, W_t$ must be positive definite
- Convergence is guaranteed if a stabilizing policy exists

#### Implementation Notes
- Since the optimal policy doesn't depend on $\Psi_t$, we can skip computing $\Psi_t$ for efficiency
- The Riccati equations can be solved efficiently using matrix operations
- LQR provides globally optimal solutions for linear systems

---

## 16.3 From Nonlinear Dynamics to LQR

### Motivation

While LQR provides elegant solutions, most real-world systems are nonlinear. However, many nonlinear problems can be **approximated** using LQR through linearization.

**Examples:**
- Inverted pendulum
- Robot manipulators
- Aircraft dynamics
- Chemical processes

### Linearization Approach

#### Intuitive Idea

If a system spends most of its time near a reference trajectory, we can approximate the nonlinear dynamics using a **first-order Taylor expansion**.

#### Mathematical Formulation

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

#### Handling the Constant Term

The constant term $\kappa$ can be eliminated by augmenting the state space:
```math
\tilde{s}_t = \begin{pmatrix} s_t \\ 1 \end{pmatrix}
```

This is the same trick used in linear regression to handle bias terms.

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

**Next: [Policy Gradient Methods](04_policy_gradient.md)** - Learn model-free reinforcement learning techniques.

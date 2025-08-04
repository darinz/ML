# Reinforcement Learning: Markov Decision Processes (MDPs)

This section introduces the foundational concepts of reinforcement learning (RL) through the lens of Markov Decision Processes (MDPs). The goal is to provide both mathematical rigor and intuitive understanding, so you can apply these ideas to real-world problems.

---

## What is Reinforcement Learning?

In supervised learning, algorithms learn to mimic provided labels $`y`$ for each input $`x`$. But in many real-world scenarios—like teaching a robot to walk—there are no explicit labels for the "right" action. Instead, we only know if the agent is doing well or poorly, based on a **reward function**.

**Key idea:**
- The agent interacts with an environment, receives rewards, and must learn a policy to maximize its long-term reward.
- There is no teacher providing the correct answer at each step; the agent must discover good strategies through trial and error.

**Real-world analogy:**
- Training a dog: You can't tell the dog exactly what to do at every moment, but you can reward it for good behavior and discourage bad behavior. Over time, the dog learns which actions lead to more rewards.

---

## 15.1 Markov Decision Processes (MDPs)

An MDP is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision maker.

**Formal definition:**
A Markov decision process is a tuple $`(S, A, \{P_{sa}\}, \gamma, R)`$, where:
- $`S`$: set of **states** (e.g., all possible positions and orientations of a robot)
- $`A`$: set of **actions** (e.g., all possible moves the robot can make)
- $`P_{sa}`$: **state transition probabilities**; for each $`s \in S`$ and $`a \in A`$, $`P_{sa}`$ is a distribution over $S$ (i.e., what is the probability of ending up in each possible next state if you take action $a$ in state $s$?)
- $`\gamma \in [0, 1)`$: **discount factor** (how much we care about future rewards)
- $`R : S \times A \to \mathbb{R}`$: **reward function** (how much immediate reward do we get for taking action $a$ in state $s$?)

**Intuitive explanation:**
- The agent starts in some state $`s_0`$.
- At each time step, it chooses an action $`a_t`$.
- The environment transitions to a new state $`s_{t+1}`$ according to $`P_{s_t a_t}`$.
- The agent receives a reward $`R(s_t, a_t)`$.
- The process repeats.

**Diagram:**

```math
s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} s_3 \xrightarrow{a_3} \ldots
```

**Total payoff:**

```math
R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \cdots
```

The discount factor $`\gamma`$ ensures that immediate rewards are more valuable than distant future rewards. This models the idea that "a reward today is worth more than a reward tomorrow."

---

## The Goal: Maximizing Expected Return

The agent's objective is to choose actions over time to maximize the expected sum of discounted rewards:

```math
\mathbb{E} \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \right]
```

**Why discount?**
- Encourages the agent to seek rewards sooner rather than later.
- Ensures the sum converges for infinite-horizon problems.
- Models uncertainty about the future (e.g., the environment might end at any time).

---

## Policies and Value Functions

**Policy ($`\pi`$):**
- A function $`\pi : S \to A`$ mapping states to actions.
- The agent "executes" a policy by always choosing $`a = \pi(s)`$ in state $s$.

**Value function ($`V^{\pi}`$):**
- Measures how good it is to start in state $s$ and follow policy $\pi$ thereafter.

```math
V^{\pi}(s) = \mathbb{E} \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \mid s_0 = s, \pi \right]
```

**Bellman equation for $`V^{\pi}`$:**

```math
V^{\pi}(s) = R(s) + \gamma \sum_{s' \in S} P_{s \pi(s)}(s') V^{\pi}(s')
```

**Step-by-step intuition:**
- The value of a state is the immediate reward plus the expected discounted value of the next state, assuming we follow policy $\pi$.
- This recursive relationship allows us to solve for $V^{\pi}$ efficiently (especially for small, finite MDPs).

---

## The Optimal Value Function and Policy

**Optimal value function ($`V^*`$):**

```math
V^*(s) = \max_{\pi} V^{\pi}(s)
```

- $V^*(s)$ is the best possible expected sum of discounted rewards starting from $s$.

**Bellman optimality equation:**

```math
V^*(s) = R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s')
```

- The optimal value is the immediate reward plus the best possible expected future value, over all possible actions.

**Optimal policy ($`\pi^*`$):**

```math
\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s')
```

- The optimal policy always chooses the action that leads to the highest expected value.

**Key property:**
- The same optimal policy $\pi^*$ is optimal for all states $s$.

---

## Algorithms: Value Iteration and Policy Iteration

### Value Iteration

A dynamic programming algorithm to compute $V^*$ and $\pi^*$ for finite MDPs.

**Algorithm:**
1. Initialize $V(s) := 0$ for all $s$.
2. Repeat until convergence:
    - For each state $s$:

```math
V(s) := R(s) + \max_{a \in A} \gamma \sum_{s'} P_{sa}(s') V(s')
```

3. After convergence, extract the optimal policy:

```math
\pi^*(s) = \arg\max_{a \in A} \sum_{s'} P_{sa}(s') V(s')
```

**Intuition:**
- At each step, update your estimate of the value of each state by considering the best action you could take.
- The process "backs up" value information from future states to current states.

### Policy Iteration

Another dynamic programming algorithm for finite MDPs.

**Algorithm:**
1. Initialize $\pi$ randomly.
2. Repeat until convergence:
    - **Policy evaluation:** Compute $V^{\pi}$ for the current policy (by solving the Bellman equations).
    - **Policy improvement:** For each state $s$, update $\pi(s)$ to the action that maximizes expected value:

    ```math
    \pi(s) := \arg\max_{a \in A} \sum_{s'} P_{sa}(s') V(s')
    ```

**Comparison:**
- Value iteration updates value estimates directly, always assuming the best action.
- Policy iteration alternates between evaluating the current policy and improving it.
- Both converge to the optimal policy for finite MDPs.

---

## Learning the Model: Estimating $P_{sa}$ and $R$

In many real-world problems, the agent does not know the transition probabilities or reward function in advance. Instead, it must estimate them from experience.

**How to estimate $P_{sa}$?**
- Run many episodes, record $(s, a, s')$ transitions.
- Estimate $P_{sa}(s')$ as:

```math
P_{sa}(s') = \frac{\#\text{times took action } a \text{ in state } s \text{ and got to } s'}{\#\text{times took action } a \text{ in state } s}
```

- If you have no data for $(s, a)$, use a uniform distribution as a prior.

**How to estimate $R(s)$?**
- Average the observed rewards received in state $s$ (or for $(s, a)$ if rewards depend on actions).

**Practical tip:**
- Keep running totals for each $(s, a, s')$ and $(s, a)$ pair to efficiently update your estimates as you gather more data.

---

## Putting It All Together: Model-Based RL Loop

1. **Initialize $\pi$ randomly.**
2. **Repeat:**
    1. Execute $\pi$ in the environment for several episodes, collecting data.
    2. Update your estimates of $P_{sa}$ and $R$ using the new data.
    3. Use value iteration or policy iteration with the estimated model to compute a new policy.
    4. Update $\pi$ to the new policy.

**Key insight:**
- This loop alternates between exploring the environment, updating your model, and planning the best actions based on your current knowledge.
- Over time, your policy improves as your model and value estimates become more accurate.

---

## Summary and Best Practices

- MDPs provide a powerful framework for modeling sequential decision-making under uncertainty.
- Value and policy iteration are foundational algorithms for solving MDPs.
- In practice, model parameters are often unknown and must be estimated from data.
- Always keep track of your experience and update your model as you learn.
- Use discounting to ensure your value estimates are well-behaved and to encourage the agent to seek rewards sooner.

**Analogy:**
- Think of RL as learning to play a new game: you try different strategies, learn from the outcomes, and gradually improve your play as you understand the rules and consequences better.
# Reinforcement learning

We now begin our study of reinforcement learning and adaptive control.

In supervised learning, we saw algorithms that tried to make their outputs mimic the labels $`y`$. In that setting, the labels gave an unambiguous "right answer" for each of the inputs $`x`$. In contrast, for many sequential decision making and control problems, it is very difficult to provide this type of explicit supervision to a learning algorithm. For example, if we have just built a four-legged robot and are trying to program it to walk, then initially we have no idea what the "correct" actions to take are to make it walk, and so do not know how to provide explicit supervision for a learning algorithm to try to mimic.

In the reinforcement learning framework, we will instead provide our algorithms only a reward function, which indicates to the learning agent when it is doing well, and when it is doing poorly. In the four-legged walking example, the reward function might give the robot positive rewards for moving forwards, and negative rewards for either moving backwards or falling over. It will then be the learning algorithm's job to figure out how to choose actions over time so as to obtain large rewards.

Reinforcement learning has been successful in applications as diverse as autonomous helicopter flight, robot legged locomotion, cell-phone network routing, marketing strategy selection, factory control, and efficient web-page indexing. Our study of reinforcement learning will begin with a definition of the **Markov decision processes (MDP)**, which provides the formalism in which RL problems are usually posed.

## 15.1 Markov decision processes

A Markov decision process is a tuple $`(S, A, \{P_{sa}\}, \gamma, R)`$, where:

- $`S`$ is a set of **states**. (For example, in autonomous helicopter flight, $`S`$ might be the set of all possible positions and orientations of the helicopter.)
- $`A`$ is a set of **actions**. (For example, the set of all possible directions in which you can push the helicopter's control sticks.)
- $`P_{sa}`$ are the state transition probabilities. For each state $`s \in S`$ and action $`a \in A`$, $`P_{sa}`$ is a distribution over the state space. We'll say more about this later, but briefly, $`P_{sa}`$ gives the distribution over what states we will transition to if we take action $`a`$ in state $`s`$.
- $`\gamma \in [0, 1)`$ is called the **discount factor**.
- $`R : S \times A \to \mathbb{R}`$ is the **reward function**. (Rewards are sometimes also written as a function of a state $`S`$ only, in which case we would have $`R : S \to \mathbb{R}`$.)

---

### Dynamics of an MDP

The dynamics of an MDP proceeds as follows: We start in some state $`s_0`$, and get to choose some action $`a_0 \in A`$ to take in the MDP. As a result of our choice, the state of the MDP randomly transitions to some successor state $`s_1`$, drawn according to $`s_1 \sim P_{s_0 a_0}`$. Then, we get to pick another action $`a_1`$. As a result of this action, the state transitions again, now to some $`s_2 \sim P_{s_1 a_1}`$. We then pick $`a_2`$, and so on.

Pictorially, we can represent this process as follows:

```math
s_0 \xrightarrow{a_0} s_1 \xrightarrow{a_1} s_2 \xrightarrow{a_2} s_3 \xrightarrow{a_3} \ldots
```

Upon visiting the sequence of states $`s_0, s_1, \ldots`$ with actions $`a_0, a_1, \ldots`$, our total payoff is given by

```math
R(s_0, a_0) + \gamma R(s_1, a_1) + \gamma^2 R(s_2, a_2) + \cdots
```

Or, when we are writing rewards as a function of the states only, this becomes

```math
R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots
```

For most of our development, we will use the simpler state-rewards $`R(s)`$, though the generalization to state-action rewards $`R(s, a)`$ offers no special difficulties.

---

Our goal in reinforcement learning is to choose actions over time so as to maximize the expected value of the total payoff:

```math
\mathbb{E} \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \right]
```

Note that the reward at timestep $`t`$ is **discounted** by a factor of $`\gamma^t`$. Thus, to make this expectation large, we would like to accrue positive rewards as soon as possible (and postpone negative rewards as long as possible). In economic applications where $`R(\cdot)`$ is the amount of money made, $`\gamma`$ also has a natural interpretation in terms of the interest rate (where a dollar today is worth more than a dollar tomorrow).

---

### Policy and Value Function

**A policy** is any function $`\pi : S \mapsto A`$ mapping from the states to the actions. We say that we are **executing** some policy $`\pi`$ if, whenever we are in state $`s`$, we take action $`a = \pi(s)`$. We also define the **value function** for a policy $`\pi`$ according to

```math
V^{\pi}(s) = \mathbb{E} \left[ R(s_0) + \gamma R(s_1) + \gamma^2 R(s_2) + \cdots \mid s_0 = s, \pi \right]
```

$`V^{\pi}(s)`$ is simply the expected sum of discounted rewards upon starting in state $`s`$, and taking actions according to $`\pi`$.

Given a fixed policy $`\pi`$, its value function $`V^{\pi}`$ satisfies the **Bellman equations**:

```math
V^{\pi}(s) = R(s) + \gamma \sum_{s' \in S} P_{s \pi(s)}(s') V^{\pi}(s')
```

This says that the expected sum of discounted rewards $`V^{\pi}(s)`$ for starting in $`s`$ consists of two terms: First, the **immediate reward** $`R(s)`$ that we get right away simply for starting in state $`s`$, and second, the expected sum of future discounted rewards. Examining the second term in more detail, we see that the summation term above can be rewritten $`\mathbb{E}_{s' \sim P_{s \pi(s)}}[V^{\pi}(s')]`$. This is the expected sum of discounted rewards for starting in state $`s'`$, where $`s'`$ is distributed according $`P_{s \pi(s)}`$, which is the distribution over where we will end up after taking the first action $`\pi(s)`$ in the MDP from state $`s`$. Thus, the second term above gives the expected sum of discounted rewards obtained *after* the first step in the MDP.

Bellman's equations can be used to efficiently solve for $`V^{\pi}`$. Specifically, in a finite-state MDP ($`|S| < \infty`$), we can write down one such equation for $`V^{\pi}(s)`$ for every state $`s`$. This gives us a set of $`|S|`$ linear equations in $`|S|`$ variables (the unknown $`V^{\pi}(s)`$'s, one for each state), which can be efficiently solved for the $`V^{\pi}(s)`$'s.

[^1]: This notation in which we condition on $`\pi`$ isn't technically correct because $`\pi`$ isn't a random variable, but this is quite standard in the literature.

---

We also define the **optimal value function** according to

```math
V^*(s) = \max_{\pi} V^{\pi}(s)
```

In other words, this is the best possible expected sum of discounted rewards that can be attained using any policy. There is also a version of Bellman's equations for the optimal value function:

```math
V^*(s) = R(s) + \max_{a \in A} \gamma \sum_{s' \in S} P_{sa}(s') V^*(s')
```

The first term above is the immediate reward as before. The second term is the maximum over all actions $`a`$ of the expected future sum of discounted rewards we'll get upon after action $`a`$. You should make sure you understand this equation and see why it makes sense.

We also define a policy $`\pi^* : S \mapsto A`$ as follows:

```math
\pi^*(s) = \arg\max_{a \in A} \sum_{s' \in S} P_{sa}(s') V^*(s')
```

**(15.3)**

Note that $`\pi^*(s)`$ gives the action $`a`$ that attains the maximum in the "max" in Equation (15.2).

It is a fact that for every state $`s`$ and every policy $`\pi`$, we have

```math
V^*(s) = V^{\pi^*}(s) \geq V^{\pi}(s)
```

The first equality says that the $`V^{\pi^*}`$, the value function for $`\pi^*`$, is equal to the optimal value function $`V^*`$ for every state $`s`$. Further, the inequality above says that $`\pi^*`$'s value is at least as large as the value of any other policy. In other words, $`\pi^*`$ as defined in Equation (15.3) is the optimal policy.

Note that $`\pi^*`$ has the interesting property that it is the optimal policy for *all* states $`s`$. Specifically, it is not the case that if we were starting in some state $`s`$ then there'd be some optimal policy for that state, and if we were starting in some other state $`s'`$ then there'd be some other policy that's optimal policy for $`s'`$. The same policy $`\pi^*`$ attains the maximum in Equation (15.1) for all states $`s`$. This means that we can use the same policy $`\pi^*`$ no matter what the initial state of our MDP is.

## 15.2 Value iteration and policy iteration

We now describe two efficient algorithms for solving finite-state MDPs. For now, we will consider only MDPs with finite state and action spaces ($`|S| < \infty`$, $`|A| < \infty`$). In this section, we will also assume that we know the state transition probabilities $`\{P_{sa}\}`$ and the reward function $`R`$.

The first algorithm, **value iteration**, is as follows:

**Algorithm 4** Value Iteration

1. For each state $`s`$, initialize $`V(s) := 0`$.
2. For until convergence **do**
3. &nbsp;&nbsp;&nbsp;For every state, update

```math
V(s) := R(s) + \max_{a \in A} \gamma \sum_{s'} P_{sa}(s') V(s')
```
**(15.4)**

This algorithm can be thought of as repeatedly trying to update the estimated value function using Bellman Equations (15.2).

There are two possible ways of performing the updates in the inner loop of the algorithm. In the first, we can first compute the new values for $`V(s)`$ for every state $`s`$, and then overwrite all the old values with the new values. This is called a **synchronous** update. In this case, the algorithm can be viewed as implementing a "Bellman backup operator" that takes a current estimate of the value function, and maps it to a new estimate. (See homework problem for details.) Alternatively, we can also perform **asynchronous** updates. Here, we would loop over the states (in some order), updating the values one at a time.

Under either synchronous or asynchronous updates, it can be shown that value iteration will cause $`V`$ to converge to $`V^*`$. Having found $`V^*`$, we can then use Equation (15.3) to find the optimal policy.

Apart from value iteration, there is a second standard algorithm for finding an optimal policy for an MDP. The **policy iteration** algorithm proceeds as follows:

Thus, the inner-loop repeatedly computes the value function for the current policy, and then updates the policy using the current value function. (The policy $`\pi`$ found in step (b) is also called the policy that is **greedy with respect to** $`V`$.) Note that step (a) can be done via solving Bellman's equations as described earlier, which in the case of a fixed policy, is just a set of $`|S|`$ linear equations in $`|S|`$ variables.

After at most a *finite* number of iterations of this algorithm, $`V`$ will converge to $`V^*`$, and $`\pi`$ will converge to $`\pi^*`$.[^2]

[^2]: Note that value iteration cannot reach the exact $`V^*`$ in a finite number of iterations.

**Algorithm 5** Policy Iteration

1. Initialize $`\pi`$ randomly.
2. For until convergence **do**
3. &nbsp;&nbsp;&nbsp;Let $`V := V^{\pi}`$.  ▷ typically by linear system solver
4. &nbsp;&nbsp;&nbsp;For each state $`s`$, let

```math
\pi(s) := \arg\max_{a \in A} \sum_{s'} P_{sa}(s') V(s')
```

Both value iteration and policy iteration are standard algorithms for solving MDPs, and there isn't currently universal agreement over which algorithm is better. For small MDPs, policy iteration is often very fast and converges with very few iterations. However, for MDPs with large state spaces, solving for $`V^{\pi}`$ explicitly would involve solving a large system of linear equations, and could be difficult (and note that one has to solve the linear system multiple times in policy iteration). In these problems, value iteration may be preferred. For this reason, in practice value iteration seems to be used more often than policy iteration. For some more discussions on the comparison and connection of value iteration and policy iteration, please see Section 15.5.

## 15.3 Learning a model for an MDP

So far, we have discussed MDPs and algorithms for MDPs assuming that the state transition probabilities and rewards are known. In many realistic problems, we are not given state transition probabilities and rewards explicitly, but must instead estimate them from data. (Usually, $`S`$, $`A`$ and $`\gamma`$ are known.)

> **Intuition:** In real-world situations, you often don't have a perfect map of how your environment works. For example, a robot may not know exactly how its actions will affect its position, or a game-playing agent may not know the rules in advance. Instead, the agent must learn how the world works by trying things out and observing what happens.

For example, suppose that, for the inverted pendulum problem (see problem set 4), we had a number of trials in the MDP, that proceeded as follows:

```math
\begin{align*}
& s_0^{(1)} \xrightarrow{a_0^{(1)}} s_1^{(1)} \xrightarrow{a_1^{(1)}} s_2^{(1)} \xrightarrow{a_2^{(1)}} s_3^{(1)} \xrightarrow{a_3^{(1)}} \cdots \\
& s_0^{(2)} \xrightarrow{a_0^{(2)}} s_1^{(2)} \xrightarrow{a_1^{(2)}} s_2^{(2)} \xrightarrow{a_2^{(2)}} s_3^{(2)} \xrightarrow{a_3^{(2)}} \cdots \\
& \qquad \vdots
\end{align*}
```

Here, $`s_i^{(j)}`$ is the state we were at time $`i`$ of trial $`j`$, and $`a_i^{(j)}`$ is the corresponding action that was taken from that state. In practice, each of the trials above might be run until the MDP terminates (such as if the pole falls over in the inverted pendulum problem), or it might be run for some large but finite number of timesteps.

> **Explanation:** Think of each trial as an episode where the agent interacts with the environment. At each step, it records what state it was in, what action it took, and what state it ended up in next. Over many episodes, the agent builds up a history of its experiences, which it can use to estimate how the environment works.

Given this "experience" in the MDP consisting of a number of trials, we can then easily derive the maximum likelihood estimates for the state transition probabilities:

```math
P_{sa}(s') = \frac{\#\text{times took we action } a \text{ in state } s \text{ and got to } s'}{\#\text{times we took action } a \text{ in state } s}
```
<span style="float:right;">(15.5)</span>

> **Intuitive meaning:** This formula is just a fancy way of saying: "If I was in state $`s`$ and took action $`a`$ 100 times, and 30 of those times I ended up in state $`s'`$, then I estimate the probability of ending up in $`s'`$ after taking $`a`$ in $`s`$ to be 30/100 = 0.3."

Or, if the ratio above is "0/0"—corresponding to the case of never having taken action $`a`$ in state $`s`$ before—the we might simply estimate $`P_{sa}(s')`$ to be $`1/|S|`$. (I.e., estimate $`P_{sa}`$ to be the uniform distribution over all states.)

> **Why uniform?** If you have no data at all about what happens when you take action $`a`$ in state $`s`$, it's reasonable to assume that all outcomes are equally likely, until you get more information.

Note that, if we gain more experience (observe more trials) in the MDP, there is an efficient way to update our estimated state transition probabilities using the new experience. Specifically, if we keep around the counts for both the numerator and denominator terms of (15.5), then as we observe more trials, we can simply keep accumulating those counts. Computing the ratio of these counts then gives our estimate of $`P_{sa}`$.

> **Practical tip:** You don't need to start over every time you get new data. Just keep a running total of how many times you've seen each state-action pair, and how many times each possible next state occurred. This makes your estimates more accurate as you gather more experience.

Using a similar procedure, if $`R`$ is unknown, we can also pick our estimate of the expected immediate reward $`R(s)`$ in state $`s`$ to be the average reward observed in state $`s`$.

> **Reward estimation:** If you don't know the reward function in advance, just keep track of the rewards you actually receive in each state (or state-action pair), and use the average as your estimate.

Having learned a model for the MDP, we can then use either value iteration or policy iteration to solve the MDP using the estimated transition probabilities and rewards. For example, putting together model learning and value iteration, here is one possible algorithm for learning in an MDP with unknown state transition probabilities:

**Algorithm: Learning in an MDP with Unknown State Transition Probabilities**

1. **Initialize $`\pi`$ randomly.**
   - *Start with any policy, even if it's just picking actions at random. The goal is to improve it over time as you learn more about the environment.*
2. **Repeat:**
    1. **Execute $`\pi`$ in the MDP for some number of trials.**
       - *Let the agent interact with the environment using its current policy. Each trial is like an episode where the agent collects data about what happens when it follows $`\pi`$.*
    2. **Using the accumulated experience in the MDP, update our estimates for $`P_{sa}`$ (and $`R`$, if applicable).**
       - *After each batch of experience, use the new data to refine your estimates of how the environment works. The more data you have, the better your estimates will be.*
    3. **Apply value iteration with the estimated state transition probabilities and rewards to get a new estimated value function $`V`$.**
       - *Use your current model of the environment to figure out how good each state is, assuming you act optimally from now on. Value iteration is a way to compute these values efficiently.*
    4. **Update $`\pi`$ to be the greedy policy with respect to $`V`$.**
       - *Change your policy so that, in each state, you pick the action that looks best according to your current value estimates. This is called acting "greedily" with respect to $`V`$.*

> **Summary:** This process is a loop: try things out, update your understanding of the world, plan the best actions based on what you know, and then try again. Over time, your policy should get better and better as your model and value estimates improve.

> **Optimization tip:** For this particular algorithm, there is one simple optimization that can make it run much more quickly. Specifically, in the inner loop of the algorithm where we apply value iteration, if instead of initializing value iteration with $`V = 0`$, we initialize it with the solution found during the previous iteration of our algorithm, then that will provide value iteration with a much better initial starting point and make it converge more quickly. This is like picking up where you left off, rather than starting from scratch every time.
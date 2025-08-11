# Continuous State Markov Decision Processes (MDPs)

This section explores how to handle Markov Decision Processes (MDPs) when the state space is infinite or continuous—a common scenario in robotics, control, and real-world applications. The focus is on building intuition, understanding the math, and learning practical strategies.

---

## The Big Picture: Why Continuous State Spaces Matter

**The Real-World Challenge:**
Imagine trying to teach a robot to walk, a car to drive, or a drone to fly. In these cases, the "state" of the system isn't just a few discrete options like "left," "right," or "center." Instead, it's described by continuous variables like position, velocity, angle, and acceleration that can take any value within a range.

**The Intuitive Analogy:**
Think of the difference between:
- **Discrete states**: Like choosing between "hot," "warm," or "cold" for temperature
- **Continuous states**: Like the actual temperature reading (72.3°F, 73.1°F, etc.)

**Why This Matters:**
- **Real-world precision**: Most physical systems have continuous states
- **Infinite possibilities**: You can't list every possible state
- **Mathematical complexity**: Standard discrete methods don't work
- **Practical necessity**: Most robotics and control problems are continuous

### The Key Insight

**From Finite to Infinite:**
- **Finite MDPs**: Like playing chess with a fixed board and pieces
- **Continuous MDPs**: Like navigating through real space where you can be anywhere

**The Scaling Problem:**
- **Small discrete spaces**: Easy to solve with tables and enumeration
- **Large continuous spaces**: Require approximation and generalization
- **The curse of dimensionality**: Problems get exponentially harder with more dimensions

## From Discrete to Continuous State Spaces

We've now explored **Markov Decision Processes (MDPs)** - the foundational mathematical framework for reinforcement learning. We've seen how MDPs model sequential decision-making under uncertainty, how value and policy iteration algorithms can solve finite MDPs optimally, and how these methods provide the theoretical foundation for learning optimal behavior through interaction with environments.

However, while finite MDPs provide excellent intuition and work well for problems with small, discrete state spaces, **real-world problems** often involve continuous state variables that cannot be easily discretized. Consider a robot navigating through space - its position, velocity, and orientation are all continuous variables that can take infinitely many values.

This motivates our exploration of **continuous state MDPs** - extending the MDP framework to handle infinite or continuous state spaces. We'll see how discretization can approximate continuous problems, how value function approximation enables learning in high-dimensional spaces, and how these techniques bridge the gap between theoretical MDPs and practical applications in robotics, control, and real-world decision-making.

The transition from discrete to continuous state spaces represents the bridge from theoretical foundations to practical applications - taking our understanding of MDPs and extending it to handle the complexity and richness of real-world problems.

In this section, we'll explore discretization techniques, value function approximation methods, and practical algorithms for solving continuous state MDPs.

---

## Understanding the Continuous State Challenge

### The Big Picture: What Makes Continuous States Different?

**The Fundamental Problem:**
In discrete MDPs, we can create a table with one row for each state and one column for each action. But in continuous MDPs, there are infinitely many states, so we can't use tables anymore.

**The Intuitive Analogy:**
- **Discrete MDP**: Like having a map with a finite number of cities
- **Continuous MDP**: Like having a map with every possible point on Earth

**Key Challenges:**
1. **Storage**: Can't store a value for every possible state
2. **Computation**: Can't iterate through all states
3. **Generalization**: Need to predict values for unseen states
4. **Curse of dimensionality**: Problems get exponentially harder

### Why Continuous State Spaces Matter

In many real-world problems, the state of a system is described by continuous variables (like position, velocity, or angle), not just a small set of discrete states. For example:
- A car's state: $(x, y, \theta, \dot{x}, \dot{y}, \dot{\theta})$ (position, orientation, velocities)
- A robot arm: joint angles and velocities
- A drone: 3D position, orientation, and velocities

**The Robot Navigation Example:**
- **Discrete approach**: Divide the room into 10×10 grid squares
- **Continuous reality**: Robot can be at any precise position (x=3.247, y=7.891)
- **The problem**: 10×10 grid misses most of the actual positions

**Key challenge:**
- You can't enumerate all possible states—there are infinitely many!
- Standard dynamic programming algorithms for finite MDPs don't scale to this setting.

**The Infinite Possibilities Problem:**
- **1D position**: Infinite possible locations on a line
- **2D position**: Infinite possible points on a plane
- **6D robot state**: Infinite possible combinations of position, orientation, and velocity

---

## Understanding Discretization

### The Big Picture: What is Discretization?

**The Discretization Problem:**
How do we handle infinite continuous states when our algorithms only work with finite discrete states?

**The Intuitive Analogy:**
Think of discretization like creating a map:
- **Continuous reality**: Every possible location on Earth
- **Discrete approximation**: A map with cities marked as dots
- **The trade-off**: Simplicity vs. accuracy

**The Key Insight:**
We sacrifice some precision to make the problem solvable with our existing tools.

### 15.4.1 Discretization

**Discretization** is the simplest way to handle continuous states:
- Chop up the continuous space into a grid of small, finite cells.
- Treat each cell as a single "state" in a standard MDP.

**The Grid Creation Process:**
1. **Define boundaries**: What's the range of each state variable?
2. **Choose resolution**: How fine should the grid be?
3. **Create cells**: Divide the space into equal-sized regions
4. **Map states**: Assign each continuous state to its nearest cell

**Example:**
- For 2D states $(s_1, s_2)$, use a grid:

  <img src="./img/grid_cell.png" width="300px"/>

- Each grid cell $\bar{s}$ is a discrete state.
- Approximate the continuous MDP by a discrete one $(\bar{S}, A, \{P_{sa}\}, \gamma, R)$.
- Use value or policy iteration to solve for $V^\ast(\bar{s})$ and $\pi^\ast(\bar{s})$.
- When the real system is in a continuous state $s$, map it to the nearest grid cell $\bar{s}$ and use $\pi^*(\bar{s})$.

**The State Mapping Process:**
1. **Continuous state**: Robot at position (3.2, 7.8)
2. **Find nearest cell**: Which grid cell contains this point?
3. **Discrete state**: Use the cell as the discrete state
4. **Policy lookup**: What action does the policy recommend for this cell?

**Intuitive analogy:**
- Imagine overlaying graph paper on a map and treating each square as a city. You plan routes between cities, not every possible point.

**The Map Planning Analogy:**
- **Continuous space**: Every possible location on Earth
- **Grid overlay**: Divide the world into squares
- **City planning**: Plan routes between major cities (grid centers)
- **Approximation**: Assume all points in a square are the same

**Limitations:**
- The value function is assumed to be constant within each cell (piecewise constant).
- This can be a poor approximation for smooth functions.
- You need a very fine grid for good accuracy, which quickly becomes impractical.

**The Staircase Problem:**
- **Smooth reality**: Like a smooth hill
- **Discrete approximation**: Like stairs going up the hill
- **The issue**: Stairs miss the smoothness of the real hill

**Visual example:**
- Fitting a smooth curve with a staircase (piecewise constant) approximation:

  <img src="./img/supervised_learning.png" width="350px"/>
  <img src="./img/piecewise.png" width="350px"/>

- The more steps you use, the closer you get, but it's never as smooth as the real thing.

**The Resolution Trade-off:**
- **Coarse grid**: Fast computation, poor accuracy
- **Fine grid**: Good accuracy, slow computation
- **The sweet spot**: Balance between speed and accuracy

**Curse of dimensionality:**
- If $S = \mathbb{R}^d$ and you discretize each dimension into $k$ values, you get $k^d$ states.
- For $d=10$ and $k=100$, that's $10^{20}$ states—impossible to store or compute!

**The Exponential Explosion:**
- **1D**: 100 states = 100 cells
- **2D**: 100×100 = 10,000 cells
- **3D**: 100×100×100 = 1,000,000 cells
- **10D**: 100^10 = 10^20 cells (impossible!)

**Rule of thumb:**
- Discretization is great for 1D or 2D problems, sometimes 3D or 4D if you're clever.
- For higher dimensions, it's infeasible.

**The Practical Guidelines:**
- **1-2D**: Use fine discretization
- **3-4D**: Use coarse discretization or smart sampling
- **5+D**: Use function approximation instead

---

## Understanding Value Function Approximation

### The Big Picture: What is Value Function Approximation?

**The Approximation Problem:**
Instead of storing a value for every possible state, can we learn a function that predicts the value for any state?

**The Intuitive Analogy:**
- **Table approach**: Like memorizing the price of every possible item
- **Function approach**: Like learning a rule "price = base_cost + weight × rate"

**The Key Insight:**
We can use machine learning techniques to learn patterns in the value function, allowing us to generalize to unseen states.

### 15.4.2 Value Function Approximation

To handle large or continuous state spaces, we use **value function approximation**:
- Instead of a table of values for each state, learn a function $V(s)$ that predicts the value for any state $s$.
- This is similar to regression in supervised learning.

**The Function Learning Process:**
1. **Collect data**: Sample states and their true values
2. **Choose model**: Pick a function approximator (linear, neural network, etc.)
3. **Train model**: Fit the function to the data
4. **Use model**: Predict values for new states

**Why is this powerful?**
- The function can generalize to unseen states.
- You can use linear regression, neural networks, or other function approximators.

**The Generalization Advantage:**
- **Seen states**: Function predicts values for states we've encountered
- **Unseen states**: Function predicts values for new states based on patterns
- **Continuous space**: Function works for any point in the space

**The Learning Analogy:**
- **Experience**: Like seeing many examples of good and bad chess positions
- **Pattern recognition**: Like learning what makes a position good
- **Generalization**: Like being able to evaluate new positions you've never seen

### Using a Model or Simulator

Assume you have a **model** or **simulator** for the MDP:
- A black box that takes $(s_t, a_t)$ and outputs $s_{t+1}$ sampled from $P_{s_t a_t}$.
- You can get such a model from physics, engineering, or by learning from data.

**The Simulator Advantage:**
- **Safe exploration**: Try policies without breaking real hardware
- **Fast iteration**: Simulate thousands of trials quickly
- **Perfect knowledge**: Know the true dynamics (in simulation)
- **Cost effective**: Much cheaper than real-world testing

**The Flight Simulator Analogy:**
- **Real flying**: Expensive, dangerous, limited practice time
- **Flight simulator**: Safe, cheap, unlimited practice
- **Transfer learning**: Skills learned in simulator transfer to real flying
- **Model fidelity**: Better simulator = better transfer

**Practical note:**
- Simulators are widely used in robotics and control, because they let you test and learn policies without risking real hardware.

**Learning a model from data:**
- Run $n$ trials, each for $T$ timesteps, recording $(s_t, a_t, s_{t+1})$.
- Fit a model (e.g., linear regression):

  $s_{t+1} = A s_t + B a_t$

- Or, for more complex systems, use non-linear models or neural networks.

**The Model Learning Process:**
1. **Data collection**: Record state-action-next_state triplets
2. **Model selection**: Choose appropriate function class
3. **Training**: Fit model to predict next states
4. **Validation**: Test model accuracy on held-out data

**The Physics vs. Data Trade-off:**
- **Physics-based models**: Use known equations of motion
- **Data-driven models**: Learn from observed behavior
- **Hybrid approaches**: Combine physics with learned corrections

---

### Understanding Fitted Value Iteration

### The Big Picture: What is Fitted Value Iteration?

**The Algorithm Challenge:**
How do we combine value iteration with function approximation to solve continuous MDPs?

**The Intuitive Analogy:**
- **Standard value iteration**: Like updating a spreadsheet with exact values
- **Fitted value iteration**: Like updating a machine learning model with approximate values

**The Key Insight:**
Instead of updating a table, we update a function approximator using supervised learning.

### Fitted Value Iteration

**Fitted value iteration** is a powerful algorithm for approximating the value function in continuous state MDPs.

**Key idea:**
- Use supervised learning to fit $V(s)$ to targets computed from the Bellman update.
- Repeat this process until convergence.

**The Learning Loop:**
1. **Generate targets**: Use current value function to compute target values
2. **Train model**: Fit function approximator to these targets
3. **Update function**: Replace old value function with new one
4. **Repeat**: Continue until convergence

**Algorithm:**
1. Randomly sample $n$ states $s^{(1)}, \ldots, s^{(n)}$ from $S$.
2. Initialize $\theta := 0$ (parameters of $V$).
3. Repeat:
    - For each $i = 1, \ldots, n$:
        - Compute $`y^{(i)} := R(s^{(i)}) + \gamma \max _a \mathbb{E} _{s' \sim P_{s^{(i)} a}} [V(s')]`$.
    - Fit $\theta$ by minimizing the squared error:

      $\min_{\theta} \sum_{i=1}^n (\theta^T \phi(s^{(i)}) - y^{(i)})^2$

    - (Here, $\phi(s)$ is a feature mapping of the state.)

**The Step-by-Step Process:**
1. **Sample states**: Pick representative states from the space
2. **Compute targets**: Use Bellman equation to compute target values
3. **Train regressor**: Fit function to predict targets from states
4. **Update parameters**: Adjust function parameters to reduce error
5. **Check convergence**: Stop when function stops changing

**Intuitive explanation:**
- This is like using regression to fit a function to a set of data points, except here the data points are states and their "target values" are computed using the Bellman update.
- The process is a loop: use your current value function to generate targets, fit a new value function to those targets, and repeat.

**The Teacher-Student Analogy:**
- **Teacher (current function)**: Provides target values for each state
- **Student (new function)**: Learns to predict these target values
- **Feedback loop**: Student becomes teacher for next iteration
- **Improvement**: Each iteration makes the function more accurate

**Why does this work?**
- The function approximator generalizes across the state space.
- You can use more powerful models (e.g., neural networks) for better generalization.

**The Generalization Power:**
- **Local learning**: Learn from specific state examples
- **Global prediction**: Predict values for entire state space
- **Smooth interpolation**: Fill in gaps between training points
- **Feature learning**: Discover useful patterns in state representation

**Practical note:**
- The better your function approximator, the better your value function will generalize to unseen states. This is why deep learning is often used in modern RL.

**The Model Choice Trade-off:**
- **Linear models**: Simple, fast, limited expressiveness
- **Neural networks**: Complex, slow, high expressiveness
- **Kernel methods**: Medium complexity, good generalization
- **Decision trees**: Interpretable, piecewise constant

**Caveats:**
- Fitted value iteration does not always converge, but in practice it often works well.
- The choice of features $\phi$ and the regression algorithm is crucial.

**The Convergence Challenge:**
- **Theoretical guarantee**: No guarantee of convergence
- **Practical reality**: Often converges in practice
- **Stability tricks**: Use target networks, experience replay, etc.
- **Monitoring**: Track training loss and policy performance

---

## Understanding Policy Extraction

### The Big Picture: How Do We Extract Policies from Value Functions?

**The Policy Problem:**
Once we have a value function, how do we convert it into a policy that tells us what action to take?

**The Intuitive Analogy:**
- **Value function**: Like knowing how good each position is in a game
- **Policy**: Like knowing what move to make from each position
- **Policy extraction**: Like converting position evaluation into move selection

### Policy Extraction in Continuous Spaces

Once you have an approximate value function $V(s)$, you can extract a policy:

```math
\pi(s) = \arg\max_a \mathbb{E}_{s' \sim P_{sa}} [V(s')]
```

**The Policy Extraction Process:**
1. **Current state**: Start from state $s$
2. **Action evaluation**: For each possible action $a$
3. **Next state prediction**: Use model to predict next states $s'$
4. **Value computation**: Compute $V(s')$ for each next state
5. **Expectation calculation**: Average over possible next states
6. **Action selection**: Choose action with highest expected value

**The Decision-Making Analogy:**
- **Current situation**: Like being at a crossroads
- **Possible actions**: Like the different roads you can take
- **Future outcomes**: Like where each road leads
- **Value assessment**: Like how good each destination is
- **Best choice**: Like picking the road to the best destination

- For each action $a$, sample possible next states $s'$ using the model, compute $V(s')$, and pick the action with the highest expected value.
- If the model is deterministic, you can just use $V(f(s, a))$ where $f$ is the deterministic transition function.

**The Sampling vs. Deterministic Trade-off:**
- **Stochastic systems**: Sample multiple next states to estimate expectation
- **Deterministic systems**: Use exact next state for precise calculation
- **Computational cost**: Sampling is more expensive but handles uncertainty
- **Accuracy**: Deterministic is exact but assumes perfect knowledge

**Practical tip:**
- In high dimensions, sampling all actions and next states can be expensive. Use approximations or restrict the action space if needed.

**The Computational Challenge:**
- **Action space size**: More actions = more computation
- **State space size**: More complex states = more computation
- **Approximation strategies**: Discretize actions, use gradients, etc.
- **Efficiency tricks**: Parallel sampling, early termination, etc.

---

## Understanding the Connection to Supervised Learning

### The Big Picture: How Does RL Connect to Supervised Learning?

**The Learning Connection:**
Reinforcement learning and supervised learning are more similar than they appear. Both involve learning from data to make predictions.

**The Key Insight:**
Value function approximation turns RL into a supervised learning problem.

### Connections to Supervised Learning

- Value function approximation is essentially a regression problem.
- The targets are computed using the Bellman update, but the fitting step is just like supervised learning.
- This connection is why modern RL often uses deep learning: neural networks are powerful function approximators.

**The Supervised Learning Analogy:**
- **Input**: State $s$ (like features in supervised learning)
- **Output**: Value $V(s)$ (like target in supervised learning)
- **Training data**: State-value pairs $(s, V(s))$
- **Loss function**: Squared error between predicted and target values
- **Learning algorithm**: Same as supervised regression

**The Data Generation Process:**
- **Supervised learning**: Human provides labeled examples
- **Reinforcement learning**: Algorithm generates its own labels using Bellman equation
- **Self-supervised**: The agent teaches itself through experience
- **Iterative improvement**: Better value function → better targets → better value function

**The Modern RL Revolution:**
- **Deep learning**: Powerful function approximators for complex patterns
- **Experience replay**: Stabilize learning with historical data
- **Target networks**: Prevent instability in value function updates
- **Actor-critic methods**: Separate value and policy learning

---

## Summary and Best Practices

### Key Takeaways

- **Discretization is simple but suffers from the curse of dimensionality**: Good for low-dimensional problems, impractical for high dimensions
- **Value function approximation enables RL in large or continuous state spaces**: Scales to complex problems through generalization
- **Fitted value iteration is a practical and powerful algorithm, but requires careful choice of features and models**: Balance between expressiveness and computational efficiency
- **Always leverage simulators or models when available—they make RL much more tractable**: Safe, fast, and cost-effective learning
- **Use deep learning for complex, high-dimensional problems, but start with simple models to build intuition**: Progressive complexity for better understanding

### The Broader Impact

Continuous state MDPs have enabled RL to tackle real-world problems by:
- **Scaling to complex systems**: Handling high-dimensional state spaces
- **Leveraging function approximation**: Using machine learning for generalization
- **Combining simulation and reality**: Safe learning with real-world deployment
- **Enabling modern AI applications**: Robotics, autonomous vehicles, game playing

**Analogy:**
- Think of value function approximation as learning to predict the value of a chess position: you can't memorize every possible position, but you can learn a function that generalizes from experience.

**The Chess Master Analogy:**
- **Novice**: Memorizes specific positions and moves
- **Expert**: Learns patterns and principles that apply to any position
- **Generalization**: Can evaluate new positions never seen before
- **Intuition**: Develops feel for what makes positions good or bad

### Best Practices

1. **Start with discretization**: For simple problems, discretization provides good intuition
2. **Choose appropriate resolution**: Balance accuracy with computational cost
3. **Use function approximation**: For complex problems, learn smooth value functions
4. **Leverage simulators**: Safe and fast learning environment
5. **Monitor convergence**: Ensure algorithms are actually improving
6. **Validate on real systems**: Test learned policies in the real world

**The Development Process:**
- **Prototype**: Start with simple discretization
- **Scale**: Move to function approximation for complex problems
- **Optimize**: Fine-tune features and model architecture
- **Deploy**: Test in real-world environment
- **Iterate**: Continuously improve based on performance

## From Value Function Approximation to Advanced Control

We've now explored **continuous state MDPs** - extending the MDP framework to handle infinite or continuous state spaces. We've seen how discretization can approximate continuous problems, how value function approximation enables learning in high-dimensional spaces, and how fitted value iteration provides practical algorithms for solving complex control problems.

However, while value function approximation provides powerful tools for handling continuous state spaces, **real-world control problems** often require more sophisticated techniques that leverage the structure of the underlying system. Many physical systems have known dynamics, cost structures, and constraints that can be exploited for more efficient and robust control.

This motivates our exploration of **advanced control methods** - specialized techniques that combine the principles of reinforcement learning with classical control theory. We'll see how Linear Quadratic Regulation (LQR) provides exact solutions for linear systems, how Differential Dynamic Programming (DDP) handles nonlinear systems through iterative optimization, and how Linear Quadratic Gaussian (LQG) control addresses partial observability through optimal state estimation.

The transition from value function approximation to advanced control represents the bridge from general-purpose learning algorithms to domain-specific optimization techniques - taking our understanding of continuous state MDPs and applying it to structured control problems with known dynamics and cost functions.

In the next section, we'll explore LQR, DDP, and LQG control methods, understanding how they leverage system structure for more efficient and robust control.

---

**Previous: [Markov Decision Processes](01_markov_decision_processes.md)** - Understand the foundational framework for sequential decision making.

**Next: [Advanced Control Methods](03_advanced_control.md)** - Learn specialized control techniques for structured systems.


# Reinforcement Learning & Control Theory

[![RL](https://img.shields.io/badge/RL-Reinforcement%20Learning-blue.svg)](https://en.wikipedia.org/wiki/Reinforcement_learning)
[![Control](https://img.shields.io/badge/Control-Control%20Theory-green.svg)](https://en.wikipedia.org/wiki/Control_theory)
[![MDP](https://img.shields.io/badge/MDP-Markov%20Decision%20Processes-purple.svg)](https://en.wikipedia.org/wiki/Markov_decision_process)

Advanced lecture notes and Python implementations for reinforcement learning and control theory with mathematical rigor and practical demonstrations.

> **For an in-depth treatment of Reinforcement Learning:** Check out the comprehensive [Reinforcement Learning repository](https://github.com/darinz/RL) which covers everything from fundamental RL algorithms to advanced deep reinforcement learning, including policy gradients, actor-critic methods, and modern applications in robotics and game playing.

## Overview

Comprehensive coverage of MDPs, continuous state spaces, advanced control methods, and policy gradient algorithms for decision-making under uncertainty.

## Materials

### Theory
- **[01_markov_decision_processes.md](01_markov_decision_processes.md)** - MDP fundamentals, value iteration, policy iteration
- **[02_continuous_state_mdp.md](02_continuous_state_mdp.md)** - Continuous state/action MDPs, value function approximation
- **[03_advanced_control.md](03_advanced_control.md)** - LQR, DDP, LQG, Kalman filter
- **[04_policy_gradient.md](04_policy_gradient.md)** - Policy gradient methods, REINFORCE, variance reduction
- **[05_hands-on_coding.md](05_hands-on_coding.md)** - Practical implementation guide

### Implementation
- **[code/markov_decision_processes_examples.py](code/markov_decision_processes_examples.py)** - Value iteration, policy iteration, MDP demonstrations
- **[code/continuous_state_mdp_examples.py](code/continuous_state_mdp_examples.py)** - Discretization, value function regression, fitted value iteration
- **[code/advanced_control_examples.py](code/advanced_control_examples.py)** - LQR, DDP, LQG, Kalman filter algorithms
- **[code/policy_gradient_examples.py](code/policy_gradient_examples.py)** - REINFORCE, policy gradient with baseline

### Supporting Files
- **code/requirements.txt** - Python dependencies
- **code/environment.yaml** - Conda environment setup
- **img/** - Figures and diagrams

## Key Concepts

### Markov Decision Processes (MDPs)
**Components**: States, actions, transitions, rewards, discount factor

**Value Function**: $V^\pi(s) = \mathbb{E}[\sum_{t=0}^{\infty} \gamma^t R_t | s_0 = s, \pi]$

**Optimal Policy**: $\pi^*(s) = \arg\max_a Q^*(s,a)$

### Continuous State MDPs
**Challenge**: Infinite state space

**Solutions**:
- Discretization of continuous spaces
- Value function approximation
- Fitted value iteration

### Advanced Control
**Linear Quadratic Regulation (LQR)**:
- Optimal control for linear systems
- Quadratic cost functions
- Riccati equation solution

**Differential Dynamic Programming (DDP)**:
- Local trajectory optimization
- Second-order approximation
- Iterative refinement

### Policy Gradient Methods
**Objective**: Maximize expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$

**REINFORCE**: $\nabla_\theta J(\theta) = \mathbb{E}[R(\tau) \nabla_\theta \log \pi_\theta(\tau)]$

**Variance Reduction**: Baseline subtraction, advantage estimation

## Applications

- **Robotics**: Motion planning and control
- **Game Playing**: Strategy optimization
- **Autonomous Systems**: Decision making under uncertainty
- **Resource Management**: Optimal allocation strategies
- **Finance**: Portfolio optimization and trading

## Getting Started

1. Read `01_markov_decision_processes.md` for MDP fundamentals
2. Study `02_continuous_state_mdp.md` for continuous spaces
3. Learn `03_advanced_control.md` for control methods
4. Explore `04_policy_gradient.md` for policy optimization
5. Use `05_hands-on_coding.md` for practical guidance
6. Run Python examples to see algorithms in action

## Prerequisites

- Probability and statistics
- Linear algebra and calculus
- Python programming and NumPy
- Understanding of optimization concepts

## Installation

```bash
pip install -r code/requirements.txt
```

Or use conda:
```bash
conda env create -f code/environment.yaml
```

## Running Examples

```bash
python code/markov_decision_processes_examples.py
python code/continuous_state_mdp_examples.py
python code/advanced_control_examples.py
python code/policy_gradient_examples.py
```

## Quick Start Code

```python
# Value Iteration
from code.markov_decision_processes_examples import value_iteration
V, policy = value_iteration(P, R, gamma=0.9, max_iter=1000)

# Policy Gradient
from code.policy_gradient_examples import reinforce
theta = reinforce(env, n_episodes=1000, learning_rate=0.01)

# LQR Control
from code.advanced_control_examples import lqr_control
K, P = lqr_control(A, B, Q, R)
```

## Method Comparison

| Method | State Space | Action Space | Convergence | Use Case |
|--------|-------------|--------------|-------------|----------|
| Value Iteration | Discrete | Discrete | Guaranteed | Small MDPs |
| Policy Gradient | Continuous | Continuous | Local | Large spaces |
| LQR | Continuous | Continuous | Global | Linear systems |
| DDP | Continuous | Continuous | Local | Trajectory optimization |
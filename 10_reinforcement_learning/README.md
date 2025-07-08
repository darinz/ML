# Reinforcement Learning & Control Theory: Advanced Topics

This directory contains advanced lecture notes, explanations, and Python implementations for key topics in reinforcement learning (RL) and control theory. The materials are designed for educational purposes, with a focus on clarity, mathematical rigor, and practical demonstration.

## Table of Contents

### Theory & Notes (Markdown)
- **01_markov_decision_processes.md** — Fundamentals of Markov Decision Processes (MDPs)
- **02_continuous_state_mdp.md** — Continuous state/action MDPs and value function approximation
- **03_advanced_control.md** — Advanced control: LQR, DDP, LQG, Kalman filter, and more
- **04_policy_gradient.md** — Policy gradient methods, REINFORCE, and variance reduction

### Python Example Implementations
- **markov_decision_processes_examples.py** — Value iteration, policy iteration, and MDP demos
- **continuous_state_mdp_examples.py** — Discretization, value function regression, fitted value iteration
- **advanced_control_examples.py** — LQR, DDP, LQG, Kalman filter, and related algorithms
- **policy_gradient_examples.py** — REINFORCE, policy gradient with baseline, and toy MDP demos

## Getting Started

### Requirements
- Python 3.7+
- numpy
- (Optional, for baseline regression) scikit-learn

Install dependencies (if needed):
```bash
pip install numpy scikit-learn
```

### Running the Example Scripts
Each `.py` file is self-contained and can be run directly. For example:
```bash
python policy_gradient_examples.py
```
This will run demonstrations of the REINFORCE algorithm and policy gradient with baseline on a simple toy MDP, printing results and learned policy parameters.

### Structure & Educational Focus
- **Markdown files** provide step-by-step derivations, intuitive explanations, and mathematical details for each topic.
- **Python scripts** implement the algorithms described in the notes, with clear comments and example usage.
- The code is designed for learning and experimentation, not for production or large-scale RL tasks.

## Topics Covered
- Finite-horizon and continuous-state MDPs
- Value iteration, policy iteration, and value function approximation
- Linear Quadratic Regulation (LQR), Differential Dynamic Programming (DDP)
- Kalman filter and Linear Quadratic Gaussian (LQG) control
- Policy gradient methods (REINFORCE), variance reduction, and baselines

## Folder Contents
- `img/` — Figures and diagrams for the notes
- `.md` files — Explanatory lecture notes and derivations
- `.py` files — Python implementations and demonstrations
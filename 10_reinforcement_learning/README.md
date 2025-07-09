# Reinforcement Learning & Control Theory: Advanced Topics

This directory contains advanced lecture notes and Python implementations for key topics in reinforcement learning (RL) and control theory. All materials have been enhanced for educational clarity, mathematical rigor, and practical demonstration.

## Table of Contents

### Theory & Notes (Markdown)
- **01_markov_decision_processes.md** — Fundamentals of Markov Decision Processes (MDPs) with detailed derivations and explanations
- **02_continuous_state_mdp.md** — Continuous state/action MDPs, value function approximation, and practical notes
- **03_advanced_control.md** — Advanced control: LQR, DDP, LQG, Kalman filter, with step-by-step math and intuition
- **04_policy_gradient.md** — Policy gradient methods, REINFORCE, variance reduction, and practical implementation details

### Python Example Implementations
- **markov_decision_processes_examples.py** — Value iteration, policy iteration, and MDP demos, with comprehensive annotations and usage examples
- **continuous_state_mdp_examples.py** — Discretization, value function regression, fitted value iteration, with detailed code and explanations
- **advanced_control_examples.py** — LQR, DDP, LQG, Kalman filter, and related algorithms, all annotated and demonstrated
- **policy_gradient_examples.py** — REINFORCE, policy gradient with baseline, and toy MDP demos, with step-by-step code and comments

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

## Structure & Educational Focus
- **Markdown files** provide step-by-step derivations, intuitive explanations, and mathematical details for each topic. All notes have been enhanced for clarity and depth, with practical insights and worked examples.
- **Python scripts** implement the algorithms described in the notes, with clear comments, usage examples, and educational annotations. All code is designed for learning and experimentation, not for production or large-scale RL tasks.

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
# Classical Multi-Armed Bandits

This directory contains implementations of classical multi-armed bandit algorithms and supporting code.

## File Structure

### Core Algorithms
- [`epsilon_greedy.py`](epsilon_greedy.py) - Epsilon-greedy algorithm implementation
- [`ucb.py`](ucb.py) - Upper Confidence Bound (UCB) algorithm implementation  
- [`thompson_sampling.py`](thompson_sampling.py) - Thompson sampling algorithm implementation

### Environment and Utilities
- [`bandit_environment.py`](bandit_environment.py) - Bandit environment, comparison functions, and visualization
- [`ad_bandit.py`](ad_bandit.py) - Specialized bandit for online advertising applications

### Examples and Documentation
- [`example_usage.py`](example_usage.py) - Complete example demonstrating all algorithms
- [`01_classical_multi_armed_bandits.md`](01_classical_multi_armed_bandits.md) - Comprehensive theoretical and practical guide

## Quick Start

1. **Run the complete example:**
   ```bash
   python example_usage.py
   ```

2. **Use individual algorithms:**
   ```python
   from epsilon_greedy import epsilon_greedy
   from ucb import ucb
   from thompson_sampling import thompson_sampling
   from bandit_environment import BanditEnvironment
   
   # Create environment
   env = BanditEnvironment([0.1, 0.2, 0.3, 0.4, 0.5])
   
   # Run algorithms
   empirical_means = epsilon_greedy(env.arms, epsilon=0.1, T=1000)
   ```

## Algorithm Overview

### Epsilon-Greedy
- **Pros**: Simple, easy to implement
- **Cons**: Fixed exploration rate, suboptimal regret bounds
- **Regret**: O(T^(2/3)) with optimal epsilon decay

### UCB (Upper Confidence Bound)
- **Pros**: Optimal regret bounds, principled exploration
- **Cons**: Requires tuning of exploration parameter
- **Regret**: O(√(KT log T))

### Thompson Sampling
- **Pros**: Bayesian optimal, often performs better in practice
- **Cons**: Requires prior specification
- **Regret**: O(√(KT log T)) (similar to UCB)

## Dependencies

- numpy
- matplotlib
- random (built-in)

## Usage Examples

### Basic Usage
```python
from bandit_environment import BanditEnvironment
from epsilon_greedy import epsilon_greedy

# Create environment with 5 arms
arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
env = BanditEnvironment(arm_means)

# Run epsilon-greedy
empirical_means = epsilon_greedy(env.arms, epsilon=0.1, T=1000)
print(f"Empirical means: {empirical_means}")
```

### Algorithm Comparison
```python
from bandit_environment import compare_algorithms, plot_regret_comparison

# Define algorithms
algorithms = {
    'Epsilon-Greedy': lambda env, T: epsilon_greedy(env.arms, 0.1, T),
    'UCB': lambda env, T: ucb(env.arms, T),
    'Thompson Sampling': lambda env, T: thompson_sampling(env.arms, T)
}

# Compare algorithms
results = compare_algorithms(env, algorithms, T=1000, n_runs=100)

# Plot results
plot_regret_comparison(results, 1000)
```

### Online Advertising
```python
from ad_bandit import AdBandit

# Create ad bandit
bandit = AdBandit(n_ads=10)

# Select ad for user
ad_id = bandit.select_ad(user_context)

# Update with observed click
bandit.update(ad_id, click=1)  # or click=0
```

## Theory

For detailed theoretical background, mathematical formulations, and analysis, see [`01_classical_multi_armed_bandits.md`](01_classical_multi_armed_bandits.md).

Key concepts covered:
- Exploration-exploitation trade-off
- Regret analysis and bounds
- Concentration inequalities
- Practical considerations and tuning
- Applications to real-world problems 
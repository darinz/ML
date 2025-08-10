# Classical Multi-Armed Bandits

This directory contains implementations of classical multi-armed bandit algorithms and supporting code.

## File Structure

### Core Algorithms (Classical Bandits)
- [`epsilon_greedy.py`](epsilon_greedy.py) - Epsilon-greedy algorithm implementation
- [`ucb.py`](ucb.py) - Upper Confidence Bound (UCB) algorithm implementation  
- [`thompson_sampling.py`](thompson_sampling.py) - Thompson sampling algorithm implementation

### Core Algorithms (Linear Bandits)
- [`linucb.py`](linucb.py) - Linear Upper Confidence Bound (LinUCB) algorithm
- [`linear_thompson_sampling.py`](linear_thompson_sampling.py) - Linear Thompson Sampling algorithm
- [`oful.py`](oful.py) - Optimism in the Face of Uncertainty for Linear Bandits (OFUL)

### Environment and Utilities
- [`bandit_environment.py`](bandit_environment.py) - Classical bandit environment, comparison functions, and visualization
- [`linear_bandit_environment.py`](linear_bandit_environment.py) - Linear bandit environment and utilities
- [`feature_utils.py`](feature_utils.py) - Feature engineering utilities
- [`ad_bandit.py`](ad_bandit.py) - Specialized bandit for online advertising applications
- [`application_examples.py`](application_examples.py) - Real-world application examples

### Examples and Documentation
- [`example_usage.py`](example_usage.py) - Complete example demonstrating classical bandit algorithms
- [`linear_bandit_example.py`](linear_bandit_example.py) - Complete example demonstrating linear bandit algorithms
- [`01_classical_multi_armed_bandits.md`](01_classical_multi_armed_bandits.md) - Comprehensive guide to classical bandits
- [`02_linear_bandits.md`](02_linear_bandits.md) - Comprehensive guide to linear bandits

## Quick Start

1. **Run the complete example:**
   ```bash
   python example_usage.py
   ```

2. **Use classical bandit algorithms:**
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

3. **Use linear bandit algorithms:**
   ```python
   from linucb import LinUCB
   from linear_thompson_sampling import LinearThompsonSampling
   from linear_bandit_environment import LinearBanditEnvironment
   
   # Create environment
   d = 5  # Feature dimension
   theta_star = np.random.randn(d)  # True parameter
   arms = np.random.randn(10, d)    # Feature vectors
   env = LinearBanditEnvironment(theta_star, arms)
   
   # Run algorithms
   algorithm = LinUCB(d, alpha=1.0)
   chosen_arms, rewards = run_linear_bandit_experiment(env, algorithm, T=1000)
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

### Linear Bandit Algorithms

### LinUCB
- **Pros**: Optimal regret bounds, principled exploration
- **Cons**: Requires tuning of exploration parameter
- **Regret**: O(d√(T log T))

### Linear Thompson Sampling
- **Pros**: Bayesian optimal, often performs better in practice
- **Cons**: Requires prior specification
- **Regret**: O(d√(T log T)) (similar to LinUCB)

### OFUL
- **Pros**: Sophisticated confidence ellipsoid construction
- **Cons**: More complex implementation
- **Regret**: O(d√(T log T))

## Dependencies

- numpy
- matplotlib
- scipy
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
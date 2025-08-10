# Classical Multi-Armed Bandits

This directory contains implementations of classical multi-armed bandit algorithms and supporting code.

## File Structure

### Core Algorithms (Classical Bandits)
- [`code/epsilon_greedy.py`](code/epsilon_greedy.py) - Epsilon-greedy algorithm implementation
- [`code/ucb.py`](code/ucb.py) - Upper Confidence Bound (UCB) algorithm implementation  
- [`code/thompson_sampling.py`](code/thompson_sampling.py) - Thompson sampling algorithm implementation

### Core Algorithms (Linear Bandits)
- [`code/linucb.py`](code/linucb.py) - Linear Upper Confidence Bound (LinUCB) algorithm
- [`code/linear_thompson_sampling.py`](code/linear_thompson_sampling.py) - Linear Thompson Sampling algorithm
- [`code/oful.py`](code/oful.py) - Optimism in the Face of Uncertainty for Linear Bandits (OFUL)

### Environment and Utilities
- [`code/bandit_environment.py`](code/bandit_environment.py) - Classical bandit environment, comparison functions, and visualization
- [`code/linear_bandit_environment.py`](code/linear_bandit_environment.py) - Linear bandit environment and utilities
- [`code/feature_utils.py`](code/feature_utils.py) - Feature engineering utilities
- [`code/ad_bandit.py`](code/ad_bandit.py) - Specialized bandit for online advertising applications
- [`code/application_examples.py`](code/application_examples.py) - Real-world application examples

### Examples and Documentation
- [`code/example_usage.py`](code/example_usage.py) - Complete example demonstrating classical bandit algorithms
- [`code/linear_bandit_example.py`](code/linear_bandit_example.py) - Complete example demonstrating linear bandit algorithms
- [`01_classical_multi_armed_bandits.md`](01_classical_multi_armed_bandits.md) - Comprehensive guide to classical bandits
- [`02_linear_bandits.md`](02_linear_bandits.md) - Comprehensive guide to linear bandits

## Quick Start

1. **Run the complete example:**
   ```bash
   python code/example_usage.py
   ```

2. **Use classical bandit algorithms:**
   ```python
   from code.epsilon_greedy import epsilon_greedy
   from code.ucb import ucb
   from code.thompson_sampling import thompson_sampling
   from code.bandit_environment import BanditEnvironment
   
   # Create environment
   env = BanditEnvironment([0.1, 0.2, 0.3, 0.4, 0.5])
   
   # Run algorithms
   empirical_means = epsilon_greedy(env.arms, epsilon=0.1, T=1000)
   ```

3. **Use linear bandit algorithms:**
   ```python
   from code.linucb import LinUCB
   from code.linear_thompson_sampling import LinearThompsonSampling
   from code.linear_bandit_environment import LinearBanditEnvironment
   
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
from code.bandit_environment import BanditEnvironment
from code.epsilon_greedy import epsilon_greedy

# Create environment with 5 arms
arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
env = BanditEnvironment(arm_means)

# Run epsilon-greedy
empirical_means = epsilon_greedy(env.arms, epsilon=0.1, T=1000)
print(f"Empirical means: {empirical_means}")
```

### Classical Bandit Algorithm Comparison
```python
from code.bandit_environment import compare_algorithms, plot_regret_comparison

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

### Linear Bandit Algorithm Comparison
```python
from code.linear_bandit_environment import compare_linear_algorithms, plot_linear_bandit_results

# Define algorithms
algorithms = {
    'LinUCB': lambda d: LinUCB(d, alpha=1.0),
    'Linear TS': lambda d: LinearThompsonSampling(d, sigma=1.0),
    'OFUL': lambda d: OFUL(d, delta=0.1)
}

# Compare algorithms
results = compare_linear_algorithms(env, algorithms)
```

### Online Advertising
```python
from code.ad_bandit import AdBandit

# Create ad bandit
bandit = AdBandit(n_ads=10)

# Select ad for user
ad_id = bandit.select_ad(user_context)

# Update with observed click
bandit.update(ad_id, click=1)  # or click=0
```

## Theory

For detailed theoretical background, mathematical formulations, and analysis, see:
- [`01_classical_multi_armed_bandits.md`](01_classical_multi_armed_bandits.md) - Classical bandits
- [`02_linear_bandits.md`](02_linear_bandits.md) - Linear bandits

Key concepts covered:
- Exploration-exploitation trade-off
- Regret analysis and bounds
- Concentration inequalities
- Feature-based learning and structured action spaces
- Practical considerations and tuning
- Applications to real-world problems 
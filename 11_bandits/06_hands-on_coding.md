# Multi-Armed Bandits: Hands-On Learning Guide

[![Bandits](https://img.shields.io/badge/Bandits-Multi--Armed%20Bandits-blue.svg)](https://en.wikipedia.org/wiki/Multi-armed_bandit)
[![Exploration](https://img.shields.io/badge/Exploration-Exploitation%20Trade--off-green.svg)](https://en.wikipedia.org/wiki/Exploration-exploitation_dilemma)
[![Optimization](https://img.shields.io/badge/Optimization-Sequential%20Decision%20Making-yellow.svg)](https://en.wikipedia.org/wiki/Sequential_decision_making)
[![Hands-on Learning](https://img.shields.io/badge/Learning-Hands--on%20Experience-green.svg)](https://en.wikipedia.org/wiki/Experiential_learning)

## From Exploration-Exploitation to Intelligent Decision Making

We've explored the elegant framework of **Multi-Armed Bandits**, which addresses the fundamental challenge of sequential decision-making under uncertainty. Understanding these concepts is crucial because bandits provide the mathematical foundation for balancing exploration and exploitation in dynamic environments, from clinical trials to online advertising and recommendation systems.

However, true understanding comes from **hands-on implementation**. This practical guide will help you translate the theoretical concepts into working code, experiment with different bandit algorithms, and develop the intuition needed to build intelligent systems that learn from experience while maximizing rewards.

## From Theoretical Understanding to Hands-On Mastery

We've now explored **applications and use cases** - the practical implementation of bandit algorithms across diverse domains. We've seen how bandits optimize ad selection and bidding in online advertising, how they enable personalized recommendations in e-commerce and content platforms, how they improve clinical trials and drug discovery in healthcare, how they optimize pricing strategies in dynamic markets, and how they enhance A/B testing and algorithm selection processes.

However, while understanding applications is valuable, **true mastery** comes from hands-on implementation. Consider building a recommendation system that adapts to user preferences, or implementing an A/B testing framework that efficiently identifies the best variant - these require not just theoretical knowledge but practical skills in implementing bandit algorithms, handling real data, and optimizing performance.

This motivates our exploration of **hands-on coding** - the practical implementation of all the bandit concepts we've learned. We'll put our theoretical knowledge into practice by implementing classical bandit algorithms like epsilon-greedy, UCB, and Thompson sampling, building linear and contextual bandits, applying best arm identification techniques, and developing practical applications for recommendation systems, A/B testing, and other real-world scenarios.

The transition from applications to hands-on coding represents the bridge from understanding to implementation - taking our knowledge of how bandits solve real-world problems and turning it into practical tools for building intelligent decision-making systems.

In this practical guide, we'll implement complete bandit systems, experiment with different algorithms and applications, and develop the practical skills needed for real-world deployment of multi-armed bandits.

## Learning Objectives

By completing this hands-on learning guide, you will:

1. **Master classical bandit algorithms** through interactive implementations of epsilon-greedy, UCB, and Thompson sampling
2. **Implement linear bandits** using LinUCB and linear Thompson sampling
3. **Apply contextual bandits** for state-dependent decision making
4. **Understand best arm identification** and pure exploration problems
5. **Develop intuition for exploration-exploitation** through practical experimentation
6. **Build practical applications** for recommendation systems and A/B testing

## Quick Start

### Prerequisites
- Basic Python knowledge (variables, functions, arrays)
- Familiarity with probability and statistics
- Understanding of linear algebra (matrices, vectors)
- Completion of probability and statistics modules (recommended)

### Estimated Time
- **Setup**: 30 minutes
- **Lesson 1**: 3-4 hours
- **Lesson 2**: 3-4 hours
- **Lesson 3**: 3-4 hours
- **Lesson 4**: 2-3 hours
- **Total**: 12-15 hours

---

## Environment Setup

### Option 1: Using Conda (Recommended)

#### Step 1: Install Miniconda
```bash
# Download Miniconda for your OS
# Windows: https://docs.conda.io/en/latest/miniconda.html
# macOS: https://docs.conda.io/en/latest/miniconda.html
# Linux: https://docs.conda.io/en/latest/miniconda.html

# Verify installation
conda --version
```

#### Step 2: Create Environment
```bash
# Navigate to the bandits directory
cd 11_bandits

# Create a new conda environment
conda env create -f code/environment.yaml

# Activate the environment
conda activate bandits-lesson

# Verify installation
python -c "import numpy, matplotlib, scipy; print('All packages installed successfully!')"
```

### Option 2: Using pip

#### Step 1: Create Virtual Environment
```bash
# Navigate to the bandits directory
cd 11_bandits

# Create virtual environment
python -m venv bandits-env

# Activate environment
# On Windows:
bandits-env\Scripts\activate
# On macOS/Linux:
source bandits-env/bin/activate

# Install requirements
pip install -r code/requirements.txt

# Verify installation
python -c "import numpy, matplotlib, scipy; print('All packages installed successfully!')"
```

### Option 3: Using Jupyter Notebooks

#### Step 1: Install Jupyter
```bash
# After setting up environment above
pip install jupyter notebook

# Launch Jupyter
jupyter notebook
```

#### Step 2: Create New Notebook
```python
# In a new notebook cell, import required packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from typing import List, Dict, Tuple, Optional, Callable
import random
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
```

---

## Lesson Structure

### Lesson 1: Classical Multi-Armed Bandits (3-4 hours)
**Files**: `code/epsilon_greedy.py`, `code/ucb.py`, `code/thompson_sampling.py`

#### Learning Goals
- Understand the exploration-exploitation trade-off
- Master epsilon-greedy algorithm implementation
- Implement Upper Confidence Bound (UCB) algorithms
- Apply Thompson sampling for Bayesian bandits
- Build practical applications for A/B testing

#### Hands-On Activities

**Activity 1.1: Understanding the Bandit Problem**
```python
# Explore the fundamental bandit problem
from code.epsilon_greedy import EpsilonGreedy

# Create a simple bandit environment
arm_means = [0.1, 0.3, 0.5, 0.2, 0.4]  # True reward means
n_arms = len(arm_means)

# Initialize epsilon-greedy algorithm
bandit = EpsilonGreedy(n_arms=n_arms, epsilon=0.1)

print(f"Bandit problem with {n_arms} arms")
print(f"True arm means: {arm_means}")
print(f"Best arm: {np.argmax(arm_means)}")

# Key insight: Bandits balance learning arm rewards with maximizing cumulative reward
```

**Activity 1.2: Epsilon-Greedy Implementation**
```python
# Implement and test epsilon-greedy algorithm
from code.epsilon_greedy import run_epsilon_greedy_experiment, compare_epsilon_values

# Run epsilon-greedy experiment
cumulative_rewards, regrets = run_epsilon_greedy_experiment(
    arm_means=arm_means,
    n_steps=1000,
    epsilon=0.1,
    n_runs=100
)

print(f"Average cumulative reward: {np.mean(cumulative_rewards[-1]):.2f}")
print(f"Average final regret: {np.mean(regrets[-1]):.2f}")

# Compare different epsilon values
epsilon_values = [0.01, 0.1, 0.3]
results = compare_epsilon_values(arm_means, epsilon_values, n_steps=1000)

print("Epsilon comparison:")
for eps, result in zip(epsilon_values, results['cumulative_rewards']):
    print(f"  ε={eps}: {np.mean(result[-1]):.2f}")

# Key insight: Epsilon controls the exploration-exploitation trade-off
```

**Activity 1.3: Upper Confidence Bound (UCB)**
```python
# Implement UCB algorithm
from code.ucb import UCB, run_ucb_experiment, compare_ucb_variants

# Create UCB bandit
ucb_bandit = UCB(n_arms=n_arms, alpha=2.0)

# Run UCB experiment
cumulative_rewards, regrets = run_ucb_experiment(
    arm_means=arm_means,
    n_steps=1000,
    alpha=2.0,
    n_runs=100
)

print(f"UCB average cumulative reward: {np.mean(cumulative_rewards[-1]):.2f}")
print(f"UCB average final regret: {np.mean(regrets[-1]):.2f}")

# Compare UCB variants
ucb_results = compare_ucb_variants(arm_means, n_steps=1000)

print("UCB variants comparison:")
for variant, result in ucb_results['cumulative_rewards'].items():
    print(f"  {variant}: {np.mean(result[-1]):.2f}")

# Key insight: UCB uses confidence intervals to balance exploration and exploitation
```

**Activity 1.4: Thompson Sampling**
```python
# Implement Thompson sampling
from code.thompson_sampling import ThompsonSampling, run_thompson_experiment

# Create Thompson sampling bandit
thompson_bandit = ThompsonSampling(n_arms=n_arms)

# Run Thompson sampling experiment
cumulative_rewards, regrets = run_thompson_experiment(
    arm_means=arm_means,
    n_steps=1000,
    n_runs=100,
    reward_type='bernoulli'
)

print(f"Thompson sampling average cumulative reward: {np.mean(cumulative_rewards[-1]):.2f}")
print(f"Thompson sampling average final regret: {np.mean(regrets[-1]):.2f}")

# Compare with other algorithms
print("Algorithm comparison:")
print(f"  Epsilon-Greedy: {np.mean(results['cumulative_rewards'][1][-1]):.2f}")
print(f"  UCB: {np.mean(cumulative_rewards[-1]):.2f}")
print(f"  Thompson Sampling: {np.mean(cumulative_rewards[-1]):.2f}")

# Key insight: Thompson sampling uses Bayesian inference for action selection
```

**Activity 1.5: Regret Analysis**
```python
# Analyze regret performance
from code.regret_analysis import analyze_regret, plot_regret_comparison

# Compare regret across algorithms
algorithms = {
    'Epsilon-Greedy': regrets,
    'UCB': regrets,
    'Thompson Sampling': regrets
}

regret_analysis = analyze_regret(algorithms, n_steps=1000)
plot_regret_comparison(regret_analysis, n_steps=1000)

# Key insight: Regret analysis helps understand algorithm performance
```

#### Experimentation Tasks
1. **Experiment with different epsilon values**: Study how exploration rate affects performance
2. **Test various UCB parameters**: Compare different alpha values and UCB variants
3. **Analyze Thompson sampling behavior**: Study posterior distributions and sampling
4. **Compare algorithm performance**: Observe regret curves and convergence rates

#### Check Your Understanding
- [ ] Can you explain the exploration-exploitation trade-off?
- [ ] Do you understand how epsilon-greedy works?
- [ ] Can you implement UCB with confidence intervals?
- [ ] Do you see how Thompson sampling uses Bayesian inference?

---

### Lesson 2: Linear Bandits (3-4 hours)
**Files**: `code/linucb.py`, `code/linear_thompson.py`, `code/feature_engineering.py`

#### Learning Goals
- Understand linear bandit framework
- Master LinUCB algorithm implementation
- Implement linear Thompson sampling
- Apply feature engineering for bandits
- Build practical applications for recommendation systems

#### Hands-On Activities

**Activity 2.1: Linear Bandit Framework**
```python
# Explore linear bandit framework
from code.linucb import LinUCB

# Create linear bandit environment
n_arms = 5
context_dim = 10
theta_star = np.random.randn(context_dim)  # True parameter vector

# Initialize LinUCB
linucb_bandit = LinUCB(n_arms=n_arms, context_dim=context_dim, alpha=1.0)

print(f"Linear bandit with {n_arms} arms and {context_dim}-dimensional context")
print(f"True parameter vector: {theta_star}")

# Key insight: Linear bandits assume rewards are linear functions of features
```

**Activity 2.2: LinUCB Implementation**
```python
# Implement and test LinUCB
from code.linucb import run_linucb_experiment

# Generate random arm features
arm_features = np.random.randn(n_arms, context_dim)

# Run LinUCB experiment
results = run_linucb_experiment(
    arm_features=arm_features,
    theta_star=theta_star,
    n_steps=1000,
    alpha=1.0,
    n_runs=50
)

print(f"LinUCB average cumulative reward: {np.mean(results['cumulative_rewards'][-1]):.2f}")
print(f"LinUCB average final regret: {np.mean(results['regrets'][-1]):.2f}")

# Key insight: LinUCB extends UCB to handle linear reward functions
```

**Activity 2.3: Linear Thompson Sampling**
```python
# Implement linear Thompson sampling
from code.linear_thompson import LinearThompsonSampling, run_linear_thompson_experiment

# Create linear Thompson sampling bandit
linear_thompson = LinearThompsonSampling(
    n_arms=n_arms, 
    context_dim=context_dim,
    nu=1.0
)

# Run linear Thompson sampling experiment
results_thompson = run_linear_thompson_experiment(
    arm_features=arm_features,
    theta_star=theta_star,
    n_steps=1000,
    nu=1.0,
    n_runs=50
)

print(f"Linear Thompson average cumulative reward: {np.mean(results_thompson['cumulative_rewards'][-1]):.2f}")

# Compare LinUCB vs Linear Thompson
print("Linear bandit comparison:")
print(f"  LinUCB: {np.mean(results['cumulative_rewards'][-1]):.2f}")
print(f"  Linear Thompson: {np.mean(results_thompson['cumulative_rewards'][-1]):.2f}")

# Key insight: Linear Thompson sampling combines Bayesian inference with linear models
```

**Activity 2.4: Feature Engineering**
```python
# Explore feature engineering for bandits
from code.feature_engineering import (
    create_polynomial_features, create_rbf_features, 
    create_fourier_features, compare_feature_representations
)

# Create different feature representations
context = np.random.randn(context_dim)

# Polynomial features
poly_features = create_polynomial_features(degree=2)
poly_context = poly_features(context)

# RBF features
centers = np.random.randn(5, context_dim)
rbf_features = create_rbf_features(centers, sigma=1.0)
rbf_context = rbf_features(context)

# Fourier features
freqs = np.random.randn(5, context_dim)
fourier_features = create_fourier_features(freqs)
fourier_context = fourier_features(context)

print(f"Original context dimension: {context_dim}")
print(f"Polynomial features dimension: {len(poly_context)}")
print(f"RBF features dimension: {len(rbf_context)}")
print(f"Fourier features dimension: {len(fourier_context)}")

# Compare feature representations
feature_comparison = compare_feature_representations(
    arm_features, theta_star, n_steps=500
)

# Key insight: Feature engineering can improve bandit performance
```

#### Experimentation Tasks
1. **Experiment with different context dimensions**: Study how dimensionality affects performance
2. **Test various feature representations**: Compare polynomial, RBF, and Fourier features
3. **Analyze parameter sensitivity**: Study how alpha and nu affect LinUCB and Thompson sampling
4. **Compare linear vs classical bandits**: Observe when linear assumptions help

#### Check Your Understanding
- [ ] Can you explain the linear bandit framework?
- [ ] Do you understand how LinUCB works?
- [ ] Can you implement linear Thompson sampling?
- [ ] Do you see the importance of feature engineering?

---

### Lesson 3: Contextual Bandits (3-4 hours)
**Files**: `contextual_bandits.py`, `neural_bandits.py`, `multi_objective.py`

#### Learning Goals
- Understand contextual bandit framework
- Master contextual UCB and Thompson sampling
- Implement neural contextual bandits
- Apply multi-objective bandit algorithms
- Build practical applications for personalization

#### Hands-On Activities

**Activity 3.1: Contextual Bandit Framework**
```python
# Explore contextual bandit framework
from contextual_bandits import ContextualBandit, ContextualUCB

# Create contextual bandit environment
n_arms = 3
context_dim = 5

# Initialize contextual UCB
contextual_ucb = ContextualUCB(n_arms=n_arms, context_dim=context_dim, alpha=2.0)

print(f"Contextual bandit with {n_arms} arms and {context_dim}-dimensional context")

# Key insight: Contextual bandits adapt to changing environments
```

**Activity 3.2: Contextual UCB and Thompson Sampling**
```python
# Implement contextual algorithms
from contextual_bandits import (
    ContextualThompsonSampling, run_contextual_experiment,
    compare_contextual_algorithms
)

# Create contextual Thompson sampling
contextual_thompson = ContextualThompsonSampling(
    n_arms=n_arms, context_dim=context_dim, nu=1.0
)

# Define context generator and reward function
def context_generator():
    return np.random.randn(context_dim)

def reward_function(arm, context):
    # Simple linear reward function
    theta_arm = np.random.randn(context_dim)
    return np.dot(theta_arm, context) + np.random.normal(0, 0.1)

# Run contextual experiment
results = run_contextual_experiment(
    bandit=contextual_ucb,
    context_generator=context_generator,
    reward_function=reward_function,
    n_steps=1000
)

print(f"Contextual UCB cumulative reward: {results['cumulative_reward']:.2f}")

# Key insight: Contextual bandits use state information for better decisions
```

**Activity 3.3: Neural Contextual Bandits**
```python
# Implement neural contextual bandits
from neural_bandits import NeuralContextualBandit, run_neural_bandit_experiment

# Create neural contextual bandit
neural_bandit = NeuralContextualBandit(
    n_arms=n_arms, 
    context_dim=context_dim, 
    hidden_dim=64
)

# Run neural bandit experiment
neural_results = run_neural_bandit_experiment(
    bandit=neural_bandit,
    context_generator=context_generator,
    reward_function=reward_function,
    n_steps=1000
)

print(f"Neural bandit cumulative reward: {neural_results['cumulative_reward']:.2f}")

# Compare contextual algorithms
comparison = compare_contextual_algorithms(n_arms, context_dim, n_steps=1000)

print("Contextual algorithm comparison:")
for algorithm, result in comparison.items():
    print(f"  {algorithm}: {result['cumulative_reward']:.2f}")

# Key insight: Neural networks can capture complex reward functions
```

**Activity 3.4: Multi-Objective Bandits**
```python
# Implement multi-objective bandits
from multi_objective import (
    MultiObjectiveBandit, run_multi_objective_experiment
)

# Create multi-objective bandit
n_objectives = 2
multi_bandit = MultiObjectiveBandit(
    n_arms=n_arms, 
    n_objectives=n_objectives,
    preference_vector=[0.7, 0.3]  # Weight for each objective
)

# Run multi-objective experiment
multi_results = run_multi_objective_experiment(
    bandit=multi_bandit,
    n_steps=1000
)

print(f"Multi-objective bandit performance: {multi_results['final_reward']:.2f}")

# Key insight: Multi-objective bandits balance multiple competing objectives
```

#### Experimentation Tasks
1. **Experiment with different context distributions**: Study how context affects performance
2. **Test various neural architectures**: Compare different hidden layer sizes
3. **Analyze multi-objective trade-offs**: Study how preference vectors affect decisions
4. **Compare contextual vs non-contextual**: Observe when context helps

#### Check Your Understanding
- [ ] Can you explain the contextual bandit framework?
- [ ] Do you understand how contextual UCB works?
- [ ] Can you implement neural contextual bandits?
- [ ] Do you see the benefits of multi-objective optimization?

---

### Lesson 4: Best Arm Identification and Applications (2-3 hours)
**Files**: `bai_algorithms.py`, `recommendation_system.py`, `ad_selection.py`

#### Learning Goals
- Understand best arm identification problems
- Master pure exploration algorithms
- Implement recommendation systems
- Apply bandits to advertising
- Build practical applications for real-world problems

#### Hands-On Activities

**Activity 4.1: Best Arm Identification**
```python
# Explore best arm identification
from bai_algorithms import (
    SuccessiveElimination, RacingAlgorithm, run_bai_experiment
)

# Create best arm identification environment
arm_means = [0.1, 0.3, 0.5, 0.2, 0.4]
delta = 0.1  # Confidence parameter

# Initialize successive elimination
successive_elim = SuccessiveElimination(n_arms=len(arm_means), delta=delta)

# Run BAI experiment
bai_results = run_bai_experiment(
    algorithm=successive_elim,
    arm_means=arm_means,
    delta=delta
)

print(f"Best arm identified: {bai_results['best_arm']}")
print(f"True best arm: {np.argmax(arm_means)}")
print(f"Sample complexity: {bai_results['sample_complexity']}")

# Key insight: BAI focuses purely on exploration to find the best arm
```

**Activity 4.2: Recommendation Systems**
```python
# Implement bandit-based recommendation system
from recommendation_system import (
    BanditRecommender, run_recommendation_experiment
)

# Create bandit recommender
n_items = 10
n_users = 100
recommender = BanditRecommender(n_items=n_items, n_users=n_users)

# Run recommendation experiment
rec_results = run_recommendation_experiment(
    recommender=recommender,
    n_steps=1000
)

print(f"Recommendation system cumulative reward: {rec_results['cumulative_reward']:.2f}")
print(f"Average user satisfaction: {rec_results['user_satisfaction']:.3f}")

# Key insight: Bandits enable personalized recommendations
```

**Activity 4.3: Advertising Selection**
```python
# Implement bandit-based ad selection
from ad_selection import (
    AdSelectionBandit, run_ad_selection_experiment
)

# Create ad selection bandit
n_ads = 5
n_user_segments = 3
ad_bandit = AdSelectionBandit(n_ads=n_ads, n_segments=n_user_segments)

# Run ad selection experiment
ad_results = run_ad_selection_experiment(
    bandit=ad_bandit,
    n_steps=1000
)

print(f"Ad selection cumulative reward: {ad_results['cumulative_reward']:.2f}")
print(f"Click-through rate: {ad_results['ctr']:.3f}")

# Key insight: Bandits optimize ad selection for different user segments
```

**Activity 4.4: Clinical Trials**
```python
# Implement bandit-based clinical trials
from clinical_trials import (
    ClinicalTrialBandit, run_clinical_trial_experiment
)

# Create clinical trial bandit
n_treatments = 4
n_patients = 200
clinical_bandit = ClinicalTrialBandit(
    n_treatments=n_treatments, 
    n_patients=n_patients
)

# Run clinical trial experiment
clinical_results = run_clinical_trial_experiment(
    bandit=clinical_bandit,
    n_steps=n_patients
)

print(f"Clinical trial cumulative reward: {clinical_results['cumulative_reward']:.2f}")
print(f"Best treatment identified: {clinical_results['best_treatment']}")

# Key insight: Bandits enable ethical and efficient clinical trials
```

#### Experimentation Tasks
1. **Experiment with different confidence levels**: Study how delta affects sample complexity
2. **Test various recommendation strategies**: Compare different bandit algorithms for recommendations
3. **Analyze ad selection performance**: Study how user segmentation affects CTR
4. **Compare BAI vs cumulative reward**: Observe the exploration-exploitation trade-off

#### Check Your Understanding
- [ ] Can you explain best arm identification problems?
- [ ] Do you understand pure exploration algorithms?
- [ ] Can you implement bandit-based recommendation systems?
- [ ] Do you see the practical applications of bandits?

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Algorithm Not Converging
```python
# Problem: Bandit algorithm doesn't converge to optimal arm
# Solution: Check parameter settings and reward distributions
def diagnose_bandit_convergence(bandit, arm_means, n_steps=1000):
    """Diagnose bandit convergence issues."""
    # Run algorithm
    rewards = []
    actions = []
    
    for step in range(n_steps):
        action = bandit.select_arm()
        reward = np.random.binomial(1, arm_means[action])
        bandit.update(action, reward)
        
        rewards.append(reward)
        actions.append(action)
    
    # Analyze convergence
    window_size = 100
    recent_actions = actions[-window_size:]
    action_counts = np.bincount(recent_actions, minlength=len(arm_means))
    
    best_arm = np.argmax(arm_means)
    best_arm_fraction = action_counts[best_arm] / window_size
    
    print(f"Best arm fraction in last {window_size} steps: {best_arm_fraction:.3f}")
    print(f"Expected fraction for random: {1/len(arm_means):.3f}")
    
    if best_arm_fraction < 0.5:
        print("Warning: Algorithm may not be converging to optimal arm")
    
    return best_arm_fraction
```

#### Issue 2: High Variance in Performance
```python
# Problem: High variance in bandit performance across runs
# Solution: Use proper initialization and regularization
def stable_bandit_training(bandit_class, arm_means, n_runs=100, n_steps=1000):
    """Stable bandit training with multiple runs."""
    all_rewards = []
    
    for run in range(n_runs):
        # Reset bandit for each run
        bandit = bandit_class(n_arms=len(arm_means))
        
        run_rewards = []
        for step in range(n_steps):
            action = bandit.select_arm()
            reward = np.random.binomial(1, arm_means[action])
            bandit.update(action, reward)
            run_rewards.append(reward)
        
        all_rewards.append(run_rewards)
    
    # Analyze stability
    final_rewards = [np.sum(rewards) for rewards in all_rewards]
    mean_reward = np.mean(final_rewards)
    std_reward = np.std(final_rewards)
    
    print(f"Mean cumulative reward: {mean_reward:.2f}")
    print(f"Standard deviation: {std_reward:.2f}")
    print(f"Coefficient of variation: {std_reward/mean_reward:.3f}")
    
    return mean_reward, std_reward
```

#### Issue 3: Contextual Bandit Performance Issues
```python
# Problem: Contextual bandit performs poorly
# Solution: Check feature engineering and context distribution
def diagnose_contextual_bandit(bandit, context_generator, reward_function, n_steps=1000):
    """Diagnose contextual bandit issues."""
    contexts = []
    rewards = []
    actions = []
    
    for step in range(n_steps):
        context = context_generator()
        action = bandit.select_arm(context)
        reward = reward_function(action, context)
        bandit.update(action, reward, context)
        
        contexts.append(context)
        rewards.append(reward)
        actions.append(action)
    
    # Analyze context distribution
    contexts_array = np.array(contexts)
    context_mean = np.mean(contexts_array, axis=0)
    context_std = np.std(contexts_array, axis=0)
    
    print(f"Context mean: {context_mean}")
    print(f"Context std: {context_std}")
    
    # Check for context drift
    early_contexts = contexts_array[:n_steps//2]
    late_contexts = contexts_array[n_steps//2:]
    
    early_mean = np.mean(early_contexts, axis=0)
    late_mean = np.mean(late_contexts, axis=0)
    
    context_drift = np.linalg.norm(late_mean - early_mean)
    print(f"Context drift: {context_drift:.3f}")
    
    if context_drift > 0.1:
        print("Warning: Significant context drift detected")
    
    return context_drift
```

#### Issue 4: Linear Bandit Numerical Issues
```python
# Problem: Linear bandit has numerical instability
# Solution: Use regularization and proper matrix operations
def robust_linear_bandit(n_arms, context_dim, regularization=1e-6):
    """Robust linear bandit with regularization."""
    # Initialize with regularization
    A_inv = np.eye(context_dim) / regularization
    b = np.zeros(context_dim)
    
    def update_linear_bandit(context, action, reward):
        nonlocal A_inv, b
        
        # Sherman-Morrison update with regularization
        x = context
        A_inv_x = A_inv @ x
        denominator = 1 + x.T @ A_inv_x
        
        if denominator > 1e-10:  # Check numerical stability
            A_inv = A_inv - np.outer(A_inv_x, A_inv_x) / denominator
            b = b + reward * x
        else:
            # Fallback to regularized update
            A_inv = np.linalg.inv(A_inv + np.outer(x, x) + regularization * np.eye(context_dim))
            b = b + reward * x
    
    return update_linear_bandit
```

#### Issue 5: Thompson Sampling Posterior Issues
```python
# Problem: Thompson sampling posteriors become degenerate
# Solution: Use proper priors and numerical stability
def stable_thompson_sampling(n_arms, prior_alpha=1.0, prior_beta=1.0):
    """Stable Thompson sampling with proper priors."""
    # Initialize with proper priors
    alpha = np.full(n_arms, prior_alpha)
    beta = np.full(n_arms, prior_beta)
    
    def select_arm():
        # Sample with numerical stability
        samples = []
        for arm in range(n_arms):
            try:
                sample = np.random.beta(alpha[arm], beta[arm])
                samples.append(sample)
            except ValueError:
                # Fallback for degenerate posteriors
                sample = alpha[arm] / (alpha[arm] + beta[arm])
                samples.append(sample)
        return np.argmax(samples)
    
    def update_arm(arm, reward):
        nonlocal alpha, beta
        
        # Update with bounds checking
        if reward == 1:
            alpha[arm] += 1
        else:
            beta[arm] += 1
        
        # Ensure numerical stability
        alpha[arm] = max(alpha[arm], 1e-6)
        beta[arm] = max(beta[arm], 1e-6)
    
    return select_arm, update_arm
```

---

## Assessment and Progress Tracking

### Self-Assessment Checklist

#### Classical Bandits Level
- [ ] I can explain the exploration-exploitation trade-off
- [ ] I understand how epsilon-greedy works
- [ ] I can implement UCB with confidence intervals
- [ ] I can apply Thompson sampling with Bayesian inference

#### Linear Bandits Level
- [ ] I can explain the linear bandit framework
- [ ] I understand how LinUCB works
- [ ] I can implement linear Thompson sampling
- [ ] I can apply feature engineering techniques

#### Contextual Bandits Level
- [ ] I can explain the contextual bandit framework
- [ ] I understand how contextual UCB works
- [ ] I can implement neural contextual bandits
- [ ] I can apply multi-objective optimization

#### Applications Level
- [ ] I can explain best arm identification problems
- [ ] I understand pure exploration algorithms
- [ ] I can implement bandit-based recommendation systems
- [ ] I can apply bandits to real-world problems

### Progress Tracking

#### Week 1: Classical and Linear Bandits
- **Goal**: Complete Lessons 1 and 2
- **Deliverable**: Working classical and linear bandit implementations
- **Assessment**: Can you implement epsilon-greedy, UCB, and Thompson sampling?

#### Week 2: Contextual Bandits and Applications
- **Goal**: Complete Lessons 3 and 4
- **Deliverable**: Contextual bandit and application implementations
- **Assessment**: Can you implement contextual bandits and apply them to real problems?

---

## Extension Projects

### Project 1: Advanced Bandit Algorithms
**Goal**: Implement cutting-edge bandit algorithms

**Tasks**:
1. Implement EXP4 for adversarial bandits
2. Add kernel bandits for non-linear rewards
3. Create hierarchical bandits for structured problems
4. Build meta-bandits for algorithm selection
5. Add bandits with delayed feedback

**Skills Developed**:
- Advanced bandit algorithms
- Adversarial learning
- Kernel methods
- Hierarchical modeling

### Project 2: Bandit Applications
**Goal**: Build real-world bandit applications

**Tasks**:
1. Implement A/B testing framework with bandits
2. Add recommendation system with contextual bandits
3. Create dynamic pricing system
4. Build clinical trial optimization
5. Add portfolio optimization with bandits

**Skills Developed**:
- Real-world applications
- System design
- Performance optimization
- User experience design

### Project 3: Bandit Research
**Goal**: Conduct original bandit research

**Tasks**:
1. Implement novel bandit algorithms
2. Add theoretical analysis and proofs
3. Create comprehensive experiments
4. Build evaluation frameworks
5. Write research papers

**Skills Developed**:
- Research methodology
- Theoretical analysis
- Experimental design
- Academic writing

---

## Additional Resources

### Books
- **"Bandit Algorithms"** by Tor Lattimore and Csaba Szepesvári
- **"Reinforcement Learning: An Introduction"** by Richard S. Sutton and Andrew G. Barto
- **"Introduction to Online Convex Optimization"** by Elad Hazan

### Online Courses
- **Stanford CS234**: Reinforcement Learning
- **Berkeley CS285**: Deep Reinforcement Learning
- **MIT 6.832**: Underactuated Robotics

### Practice Environments
- **OpenAI Gym**: Standard RL environments
- **Vowpal Wabbit**: Online learning library
- **BanditLib**: Bandit algorithm library
- **Contextual Bandits**: Real-world datasets

### Advanced Topics
- **Adversarial Bandits**: Learning against adversaries
- **Kernel Bandits**: Non-linear reward functions
- **Hierarchical Bandits**: Structured decision problems
- **Meta-Learning**: Learning to learn with bandits

---

## Conclusion: The Future of Intelligent Decision Making

Congratulations on completing this comprehensive journey through Multi-Armed Bandits! We've explored the fundamental techniques for sequential decision-making under uncertainty.

### The Complete Picture

**1. Classical Bandits** - We started with the exploration-exploitation trade-off and basic algorithms.

**2. Linear Bandits** - We built systems that handle structured reward functions.

**3. Contextual Bandits** - We implemented algorithms that adapt to changing environments.

**4. Applications** - We explored real-world applications from recommendations to clinical trials.

### Key Insights

- **Exploration vs Exploitation**: Bandits provide the mathematical framework for this fundamental trade-off
- **Uncertainty Handling**: Proper uncertainty quantification is crucial for good decisions
- **Adaptive Learning**: Bandits enable systems that learn and improve over time
- **Practical Applications**: Bandits have wide-ranging applications in the real world
- **Theoretical Guarantees**: Bandit algorithms come with strong theoretical foundations

### Looking Forward

This bandit foundation prepares you for advanced topics:
- **Deep Bandits**: Neural networks for complex reward functions
- **Multi-Agent Bandits**: Systems with multiple interacting agents
- **Federated Bandits**: Distributed learning across multiple agents
- **Fair Bandits**: Ensuring fairness in decision-making
- **Safe Bandits**: Guaranteeing safety in critical applications

The principles we've learned here - exploration-exploitation, uncertainty quantification, and adaptive learning - will serve you well throughout your AI and machine learning journey.

### Next Steps

1. **Apply bandit techniques** to your own projects
2. **Explore advanced bandit algorithms** and research frontiers
3. **Build real-world applications** using bandits
4. **Contribute to open source** bandit libraries
5. **Continue learning** about sequential decision making

Remember: Multi-Armed Bandits are not just algorithms - they're a fundamental approach to intelligent decision-making under uncertainty. Keep exploring, building, and applying these concepts to create smarter, more adaptive systems!

---

**Previous: [Applications and Use Cases](05_applications_and_use_cases.md)** - Explore real-world applications of multi-armed bandits.

**Next: [Transformers and Large Language Models](../12_llm/README.md)** - Explore attention mechanisms and transformer architectures.

---

## Environment Files

### requirements.txt
```
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0
jupyter>=1.0.0
notebook>=6.4.0
ipykernel>=6.0.0
nb_conda_kernels>=2.3.0
```

### environment.yaml
```yaml
name: bandits-lesson
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy>=1.21.0
  - matplotlib>=3.5.0
  - scipy>=1.7.0
  - scikit-learn>=1.0.0
  - pandas>=1.3.0
  - seaborn>=0.11.0
  - jupyter>=1.0.0
  - notebook>=6.4.0
  - pip
  - pip:
    - ipykernel
    - nb_conda_kernels
```

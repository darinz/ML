# Best Arm Identification

## Introduction

Best Arm Identification (BAI) represents a fundamental shift in the multi-armed bandit paradigm from cumulative reward maximization to pure exploration. Unlike traditional bandit algorithms that balance exploration and exploitation, BAI algorithms focus exclusively on identifying the best arm with high confidence, regardless of the cumulative reward achieved during the learning process.

### Motivation

Traditional bandit algorithms optimize for cumulative reward, but many real-world scenarios prioritize accurate identification over immediate performance:

- **Clinical Trials**: Identify the most effective treatment, not maximize short-term outcomes
- **A/B Testing**: Determine the best website design or feature
- **Drug Discovery**: Find the most promising drug candidate
- **Product Development**: Select the best product variant for mass production
- **Algorithm Selection**: Choose the best algorithm for a specific task

### Key Differences from Traditional Bandits

**Traditional Bandits (Regret Minimization):**
- Goal: Maximize cumulative reward $`\sum_{t=1}^T r_t`$
- Metric: Cumulative regret $`R(T) = \sum_{t=1}^T (\mu^* - \mu_{a_t})`$
- Trade-off: Exploration vs. exploitation

**Best Arm Identification (Pure Exploration):**
- Goal: Identify best arm with high confidence
- Metric: Success probability $`P(\hat{i}^* = i^*)`$
- Focus: Pure exploration, no exploitation needed

## Problem Formulation

### Mathematical Setup

In the best arm identification problem, we have:

- **$`K`$ arms** with unknown reward distributions
- **Fixed budget** of $`T`$ total pulls
- **Goal**: Identify arm with highest mean reward $`i^* = \arg\max_i \mu_i`$
- **Success criterion**: $`P(\hat{i}^* = i^*) \geq 1-\delta`$ where $`\delta`$ is the failure probability

### Key Definitions

**Gaps:**
- $`\Delta_i = \mu_{i^*} - \mu_i`$: Gap between optimal arm and arm $`i`$
- $`\Delta_{\min} = \min_{i \neq i^*} \Delta_i`$: Minimum gap
- $`\Delta_{\max} = \max_{i \neq i^*} \Delta_i`$: Maximum gap

**Sample Complexity:**
- **Gap-dependent**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta})`$
- **Gap-independent**: $`O(K \log \frac{1}{\delta})`$

**Confidence Intervals:**
- $`\text{CI}_i(t) = [\hat{\mu}_i(t) \pm \beta_i(t)]`$: Confidence interval for arm $`i`$ at time $`t`$
- $`\beta_i(t)`$: Confidence radius (depends on algorithm)

### Problem Variants

**Fixed Budget:**
- Total number of pulls $`T`$ is fixed
- Goal: Maximize success probability

**Fixed Confidence:**
- Target success probability $`1-\delta`$ is fixed
- Goal: Minimize expected number of pulls

**Fixed Budget and Confidence:**
- Both $`T`$ and $`\delta`$ are fixed
- Goal: Achieve success probability $`\geq 1-\delta`$ within $`T`$ pulls

## Fundamental Algorithms

### 1. Successive Elimination

Successive Elimination is a simple and intuitive algorithm that eliminates arms progressively based on empirical comparisons.

**Algorithm:**
1. **Initialization**: Pull each arm $`n_0`$ times
2. **Elimination**: Eliminate arms with low empirical means
3. **Iteration**: Continue until one arm remains

**Implementation:**
```python
import numpy as np
from scipy.stats import norm

class SuccessiveElimination:
    def __init__(self, n_arms, delta=0.1, n0=None):
        self.n_arms = n_arms
        self.delta = delta
        self.n0 = n0 if n0 else max(1, int(np.log(n_arms / delta)))
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.active_arms = set(range(n_arms))
        
    def select_arm(self):
        """Select arm to pull"""
        if len(self.active_arms) == 1:
            return list(self.active_arms)[0]
        
        # Pull arms that haven't been pulled n0 times
        for arm in self.active_arms:
            if self.pulls[arm] < self.n0:
                return arm
        
        # All arms have been pulled n0 times, eliminate worst arm
        worst_arm = min(self.active_arms, key=lambda i: self.empirical_means[i])
        self.active_arms.remove(worst_arm)
        
        # Continue pulling remaining arms
        return list(self.active_arms)[0]
    
    def update(self, arm, reward):
        """Update algorithm with observed reward"""
        self.pulls[arm] += 1
        self.empirical_means[arm] = ((self.empirical_means[arm] * (self.pulls[arm] - 1) + reward) / self.pulls[arm])
    
    def get_best_arm(self):
        """Return the identified best arm"""
        if len(self.active_arms) == 1:
            return list(self.active_arms)[0]
        else:
            return np.argmax(self.empirical_means)
    
    def is_complete(self):
        """Check if algorithm has completed"""
        return len(self.active_arms) == 1
```

**Theoretical Guarantees:**
- **Sample Complexity**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{K}{\delta})`$
- **Success Probability**: $`P(\hat{i}^* = i^*) \geq 1-\delta`$

### 2. Racing Algorithms

Racing algorithms maintain confidence intervals for all arms and stop when one arm is clearly the best.

**Algorithm:**
1. **Initialization**: Pull each arm $`n_0`$ times
2. **Confidence Intervals**: Maintain confidence intervals for all arms
3. **Stopping Criterion**: Stop when one arm's lower bound exceeds all others' upper bounds
4. **Adaptive Allocation**: Pull arms with highest uncertainty

**Implementation:**
```python
class RacingAlgorithm:
    def __init__(self, n_arms, delta=0.1, n0=None):
        self.n_arms = n_arms
        self.delta = delta
        self.n0 = n0 if n0 else max(1, int(np.log(n_arms / delta)))
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.total_pulls = 0
        
    def get_confidence_radius(self, arm):
        """Calculate confidence radius for arm"""
        if self.pulls[arm] == 0:
            return float('inf')
        
        # Hoeffding-based confidence interval
        beta = np.sqrt(np.log(2 * self.n_arms / self.delta) / (2 * self.pulls[arm]))
        return beta
    
    def get_confidence_intervals(self):
        """Get confidence intervals for all arms"""
        intervals = []
        for arm in range(self.n_arms):
            radius = self.get_confidence_radius(arm)
            lower = self.empirical_means[arm] - radius
            upper = self.empirical_means[arm] + radius
            intervals.append((lower, upper))
        return intervals
    
    def select_arm(self):
        """Select arm to pull"""
        # Initial phase: pull each arm n0 times
        for arm in range(self.n_arms):
            if self.pulls[arm] < self.n0:
                return arm
        
        # Racing phase: pull arm with highest uncertainty
        intervals = self.get_confidence_intervals()
        
        # Find arm with highest upper bound
        best_arm = max(range(self.n_arms), key=lambda i: intervals[i][1])
        
        # Find arm with highest lower bound (different from best)
        other_arms = [i for i in range(self.n_arms) if i != best_arm]
        if other_arms:
            challenger = max(other_arms, key=lambda i: intervals[i][0])
            
            # Pull the arm with highest uncertainty
            if intervals[best_arm][1] - intervals[best_arm][0] > intervals[challenger][1] - intervals[challenger][0]:
                return best_arm
            else:
                return challenger
        
        return best_arm
    
    def update(self, arm, reward):
        """Update algorithm with observed reward"""
        self.pulls[arm] += 1
        self.total_pulls += 1
        self.empirical_means[arm] = ((self.empirical_means[arm] * (self.pulls[arm] - 1) + reward) / self.pulls[arm])
    
    def get_best_arm(self):
        """Return the identified best arm"""
        return np.argmax(self.empirical_means)
    
    def is_complete(self):
        """Check if algorithm has completed"""
        intervals = self.get_confidence_intervals()
        
        # Check if one arm's lower bound exceeds all others' upper bounds
        for i in range(self.n_arms):
            lower_i = intervals[i][0]
            all_others_upper = [intervals[j][1] for j in range(self.n_arms) if j != i]
            if all(lower_i > upper_j for upper_j in all_others_upper):
                return True
        
        return False
```

### 3. LUCB (Lower-Upper Confidence Bound)

LUCB is a sophisticated algorithm that pulls the arm with highest upper bound and the arm with highest lower bound among the remaining arms.

**Algorithm:**
1. **Initialization**: Pull each arm $`n_0`$ times
2. **Arm Selection**: Pull arms with highest upper bound and highest lower bound
3. **Stopping Criterion**: Stop when intervals separate

**Implementation:**
```python
class LUCB:
    def __init__(self, n_arms, delta=0.1, n0=None):
        self.n_arms = n_arms
        self.delta = delta
        self.n0 = n0 if n0 else max(1, int(np.log(n_arms / delta)))
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.total_pulls = 0
        
    def get_confidence_radius(self, arm):
        """Calculate confidence radius for arm"""
        if self.pulls[arm] == 0:
            return float('inf')
        
        # LUCB-specific confidence interval
        beta = np.sqrt(np.log(4 * self.n_arms * self.total_pulls**2 / self.delta) / (2 * self.pulls[arm]))
        return beta
    
    def get_confidence_intervals(self):
        """Get confidence intervals for all arms"""
        intervals = []
        for arm in range(self.n_arms):
            radius = self.get_confidence_radius(arm)
            lower = self.empirical_means[arm] - radius
            upper = self.empirical_means[arm] + radius
            intervals.append((lower, upper))
        return intervals
    
    def select_arm(self):
        """Select arm to pull using LUCB"""
        # Initial phase: pull each arm n0 times
        for arm in range(self.n_arms):
            if self.pulls[arm] < self.n0:
                return arm
        
        # LUCB phase: pull arms with highest upper and lower bounds
        intervals = self.get_confidence_intervals()
        
        # Find arm with highest upper bound
        h = max(range(self.n_arms), key=lambda i: intervals[i][1])
        
        # Find arm with highest lower bound among others
        others = [i for i in range(self.n_arms) if i != h]
        l = max(others, key=lambda i: intervals[i][0])
        
        # Pull the arm with higher uncertainty
        if intervals[h][1] - intervals[h][0] > intervals[l][1] - intervals[l][0]:
            return h
        else:
            return l
    
    def update(self, arm, reward):
        """Update algorithm with observed reward"""
        self.pulls[arm] += 1
        self.total_pulls += 1
        self.empirical_means[arm] = ((self.empirical_means[arm] * (self.pulls[arm] - 1) + reward) / self.pulls[arm])
    
    def get_best_arm(self):
        """Return the identified best arm"""
        return np.argmax(self.empirical_means)
    
    def is_complete(self):
        """Check if algorithm has completed"""
        intervals = self.get_confidence_intervals()
        
        # Find arm with highest upper bound
        h = max(range(self.n_arms), key=lambda i: intervals[i][1])
        
        # Find arm with highest lower bound among others
        others = [i for i in range(self.n_arms) if i != h]
        l = max(others, key=lambda i: intervals[i][0])
        
        # Check if intervals separate
        return intervals[h][0] > intervals[l][1]
```

### 4. Sequential Halving

Sequential Halving is an efficient algorithm that eliminates half of the remaining arms in each round.

**Algorithm:**
1. **Initialization**: Start with all arms
2. **Rounds**: In each round, pull remaining arms equally
3. **Elimination**: Eliminate bottom half of arms based on empirical means
4. **Termination**: Continue until one arm remains

**Implementation:**
```python
class SequentialHalving:
    def __init__(self, n_arms, budget):
        self.n_arms = n_arms
        self.budget = budget
        
        # Calculate number of rounds
        self.n_rounds = int(np.log2(n_arms))
        
        # Initialize statistics
        self.empirical_means = np.zeros(n_arms)
        self.pulls = np.zeros(n_arms, dtype=int)
        self.active_arms = list(range(n_arms))
        self.current_round = 0
        self.pulls_per_arm = 0
        
    def select_arm(self):
        """Select arm to pull"""
        if len(self.active_arms) == 1:
            return self.active_arms[0]
        
        # Calculate pulls per arm for current round
        if self.current_round == 0:
            self.pulls_per_arm = self.budget // (len(self.active_arms) * self.n_rounds)
        else:
            self.pulls_per_arm = self.budget // (len(self.active_arms) * (self.n_rounds - self.current_round))
        
        # Find arm that needs more pulls
        for arm in self.active_arms:
            if self.pulls[arm] < self.pulls_per_arm:
                return arm
        
        # All arms have been pulled enough, eliminate bottom half
        self._eliminate_bottom_half()
        return self.active_arms[0]  # Return first remaining arm
    
    def _eliminate_bottom_half(self):
        """Eliminate bottom half of active arms"""
        # Sort arms by empirical means
        sorted_arms = sorted(self.active_arms, key=lambda i: self.empirical_means[i], reverse=True)
        
        # Keep top half
        keep_count = len(self.active_arms) // 2
        self.active_arms = sorted_arms[:keep_count]
        self.current_round += 1
    
    def update(self, arm, reward):
        """Update algorithm with observed reward"""
        self.pulls[arm] += 1
        self.empirical_means[arm] = ((self.empirical_means[arm] * (self.pulls[arm] - 1) + reward) / self.pulls[arm])
    
    def get_best_arm(self):
        """Return the identified best arm"""
        if len(self.active_arms) == 1:
            return self.active_arms[0]
        else:
            return max(self.active_arms, key=lambda i: self.empirical_means[i])
    
    def is_complete(self):
        """Check if algorithm has completed"""
        return len(self.active_arms) == 1
```

## Theoretical Analysis

### Sample Complexity Bounds

**Gap-dependent Bounds:**
For algorithms with gap-dependent sample complexity:

```math
\mathbb{E}[N] \leq O\left(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta}\right)
```

**Gap-independent Bounds:**
For algorithms with gap-independent sample complexity:

```math
\mathbb{E}[N] \leq O\left(K \log \frac{1}{\delta}\right)
```

**Lower Bounds:**
For any BAI algorithm:

```math
\mathbb{E}[N] \geq \Omega\left(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta}\right)
```

### Confidence Interval Analysis

**Hoeffding-based Intervals:**
For bounded rewards $`r_i \in [0, 1]`$:

```math
P(|\hat{\mu}_i - \mu_i| \geq \epsilon) \leq 2\exp(-2n_i \epsilon^2)
```

**Chernoff-based Intervals:**
For Bernoulli rewards:

```math
P(|\hat{\mu}_i - \mu_i| \geq \epsilon) \leq 2\exp(-n_i \text{KL}(\mu_i + \epsilon \| \mu_i))
```

**Union Bound:**
For multiple arms and time steps:

```math
P(\exists i, t : |\hat{\mu}_i(t) - \mu_i| \geq \beta_i(t)) \leq \delta
```

### Algorithm Comparison

**Successive Elimination:**
- **Pros**: Simple, intuitive, good theoretical guarantees
- **Cons**: May not be optimal for all gap structures
- **Sample Complexity**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{K}{\delta})`$

**Racing Algorithms:**
- **Pros**: Adaptive allocation, good empirical performance
- **Cons**: More complex implementation
- **Sample Complexity**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{K}{\delta})`$

**LUCB:**
- **Pros**: Optimal sample complexity, theoretical guarantees
- **Cons**: Complex implementation, may be conservative
- **Sample Complexity**: $`O(\sum_{i \neq i^*} \frac{1}{\Delta_i^2} \log \frac{1}{\delta})`$

**Sequential Halving:**
- **Pros**: Simple, efficient for large action spaces
- **Cons**: Fixed budget requirement, may not be optimal
- **Sample Complexity**: $`O(K \log \frac{1}{\delta})`$

## Practical Considerations

### Parameter Tuning

**Confidence Level ($`\delta`$):**
- **Typical values**: $`\delta = 0.1, 0.05, 0.01`$
- **Trade-off**: Lower $`\delta`$ requires more samples but higher confidence
- **Application-specific**: Choose based on cost of incorrect identification

**Initial Sample Size ($`n_0`$):**
- **Theoretical**: $`n_0 = O(\log \frac{K}{\delta})`$
- **Practical**: $`n_0 = 10-50`$ often works well
- **Adaptive**: Can be adjusted based on observed gaps

### Stopping Criteria

**Fixed Budget:**
- Stop when $`T`$ pulls are exhausted
- Return best arm based on empirical means

**Fixed Confidence:**
- Stop when confidence intervals separate
- Return arm with highest lower bound

**Adaptive Stopping:**
- Stop when success probability exceeds threshold
- Requires online estimation of success probability

### Numerical Stability

**Confidence Interval Calculation:**
```python
def stable_confidence_radius(pulls, delta, method='hoeffding'):
    """Calculate stable confidence radius"""
    if pulls == 0:
        return float('inf')
    
    if method == 'hoeffding':
        return np.sqrt(np.log(2 / delta) / (2 * pulls))
    elif method == 'chernoff':
        # For Bernoulli rewards
        return np.sqrt(np.log(1 / delta) / pulls)
    else:
        raise ValueError(f"Unknown method: {method}")
```

**Empirical Mean Update:**
```python
def stable_mean_update(old_mean, old_count, new_value):
    """Stable incremental mean update"""
    return (old_mean * old_count + new_value) / (old_count + 1)
```

## Advanced Topics

### Contextual Best Arm Identification

**Problem Extension:**
Identify the best arm for each context, not just globally.

**Algorithms:**
- **Contextual Successive Elimination**: Eliminate arms per context
- **Contextual Racing**: Maintain confidence intervals per context
- **Contextual LUCB**: Extend LUCB to handle contexts

### Linear Best Arm Identification

**Problem Setting:**
Identify the best arm when rewards are linear functions of features.

**Algorithms:**
- **Linear Successive Elimination**: Use linear confidence intervals
- **Linear Racing**: Maintain ellipsoidal confidence regions
- **Linear LUCB**: Extend LUCB to linear bandits

### Multi-objective Best Arm Identification

**Problem:**
Identify the best arm when there are multiple objectives to optimize.

**Approaches:**
- **Pareto optimality**: Find non-dominated arms
- **Scalarization**: Combine objectives into single metric
- **Preference learning**: Learn user preferences over objectives

### Best Arm Identification with Constraints

**Resource Constraints:**
- **Budget constraints**: Limited total cost
- **Time constraints**: Limited time horizon
- **Safety constraints**: Ensure safe exploration

**Algorithms:**
- **Constrained BAI**: Add constraint handling to BAI algorithms
- **Safe exploration**: Ensure constraints are satisfied during identification

## Applications

### A/B Testing

**Website Design Testing:**
```python
class ABTestBAI:
    def __init__(self, n_variants, delta=0.05):
        self.bai = LUCB(n_variants, delta)
        self.metrics = ['conversion_rate', 'revenue_per_user', 'session_duration']
    
    def run_test(self, user_traffic):
        """Run A/B test using BAI"""
        for user in user_traffic:
            # Select variant to show
            variant = self.bai.select_arm()
            
            # Show variant and collect metrics
            metrics = self._show_variant_and_collect_metrics(user, variant)
            
            # Convert metrics to reward
            reward = self._convert_metrics_to_reward(metrics)
            
            # Update BAI algorithm
            self.bai.update(variant, reward)
            
            # Check if test is complete
            if self.bai.is_complete():
                break
        
        return self.bai.get_best_arm()
    
    def _convert_metrics_to_reward(self, metrics):
        """Convert multiple metrics to single reward"""
        # Weighted combination of metrics
        weights = [0.5, 0.3, 0.2]  # conversion, revenue, duration
        reward = sum(w * m for w, m in zip(weights, metrics))
        return reward
```

### Clinical Trials

**Treatment Comparison:**
```python
class ClinicalTrialBAI:
    def __init__(self, n_treatments, delta=0.01):
        self.bai = SuccessiveElimination(n_treatments, delta)
        self.patient_outcomes = []
    
    def run_trial(self, patients):
        """Run clinical trial using BAI"""
        for patient in patients:
            # Assign treatment
            treatment = self.bai.select_arm()
            
            # Administer treatment and observe outcome
            outcome = self._administer_treatment_and_observe(patient, treatment)
            
            # Update BAI algorithm
            self.bai.update(treatment, outcome)
            self.patient_outcomes.append((treatment, outcome))
            
            # Check if trial is complete
            if self.bai.is_complete():
                break
        
        best_treatment = self.bai.get_best_arm()
        return best_treatment, self.patient_outcomes
    
    def _administer_treatment_and_observe(self, patient, treatment):
        """Administer treatment and observe outcome"""
        # Simulate treatment administration
        # In practice, this would involve actual treatment and follow-up
        baseline = patient.get_baseline_health()
        treatment_effect = self._get_treatment_effect(treatment, patient)
        outcome = baseline + treatment_effect + np.random.normal(0, 0.1)
        return outcome
```

### Product Development

**Feature Selection:**
```python
class FeatureSelectionBAI:
    def __init__(self, n_features, delta=0.1):
        self.bai = RacingAlgorithm(n_features, delta)
        self.feature_performances = {}
    
    def evaluate_features(self, test_cases):
        """Evaluate features using BAI"""
        for test_case in test_cases:
            # Select feature to evaluate
            feature = self.bai.select_arm()
            
            # Evaluate feature performance
            performance = self._evaluate_feature_performance(feature, test_case)
            
            # Update BAI algorithm
            self.bai.update(feature, performance)
            
            # Check if evaluation is complete
            if self.bai.is_complete():
                break
        
        best_feature = self.bai.get_best_arm()
        return best_feature, self.feature_performances
    
    def _evaluate_feature_performance(self, feature, test_case):
        """Evaluate performance of a feature on a test case"""
        # Simulate feature evaluation
        # In practice, this would involve actual testing
        base_performance = test_case.get_base_performance()
        feature_boost = self._get_feature_boost(feature, test_case)
        performance = base_performance + feature_boost + np.random.normal(0, 0.05)
        return performance
```

### Algorithm Selection

**Machine Learning Algorithm Selection:**
```python
class AlgorithmSelectionBAI:
    def __init__(self, algorithms, delta=0.05):
        self.n_algorithms = len(algorithms)
        self.bai = SequentialHalving(self.n_algorithms, budget=1000)
        self.algorithms = algorithms
        self.performance_history = {}
    
    def select_best_algorithm(self, dataset):
        """Select best algorithm using BAI"""
        # Split dataset for evaluation
        train_data, test_data = self._split_dataset(dataset)
        
        while not self.bai.is_complete():
            # Select algorithm to evaluate
            alg_idx = self.bai.select_arm()
            algorithm = self.algorithms[alg_idx]
            
            # Train and evaluate algorithm
            performance = self._train_and_evaluate(algorithm, train_data, test_data)
            
            # Update BAI algorithm
            self.bai.update(alg_idx, performance)
        
        best_alg_idx = self.bai.get_best_arm()
        return self.algorithms[best_alg_idx]
    
    def _train_and_evaluate(self, algorithm, train_data, test_data):
        """Train and evaluate an algorithm"""
        # Train algorithm
        algorithm.train(train_data)
        
        # Evaluate on test data
        predictions = algorithm.predict(test_data.X)
        performance = self._calculate_performance(predictions, test_data.y)
        
        return performance
    
    def _calculate_performance(self, predictions, true_values):
        """Calculate performance metric"""
        # Example: accuracy for classification
        correct = sum(1 for p, t in zip(predictions, true_values) if p == t)
        return correct / len(predictions)
```

## Implementation Examples

### Complete BAI Environment

```python
import numpy as np
import matplotlib.pyplot as plt

class BAIEnvironment:
    def __init__(self, arm_means, noise_std=0.1):
        self.arm_means = arm_means
        self.noise_std = noise_std
        self.n_arms = len(arm_means)
        self.optimal_arm = np.argmax(arm_means)
        self.optimal_mean = arm_means[self.optimal_arm]
        
    def pull_arm(self, arm_idx):
        """Pull arm and return reward"""
        expected_reward = self.arm_means[arm_idx]
        noise = np.random.normal(0, self.noise_std)
        return expected_reward + noise
    
    def get_gaps(self):
        """Calculate gaps between optimal arm and others"""
        gaps = []
        for i in range(self.n_arms):
            if i != self.optimal_arm:
                gap = self.optimal_mean - self.arm_means[i]
                gaps.append(gap)
        return gaps

def run_bai_experiment(env, algorithm, max_pulls=1000):
    """Run BAI experiment"""
    pulls_used = 0
    success = False
    
    while pulls_used < max_pulls and not algorithm.is_complete():
        # Select arm
        arm_idx = algorithm.select_arm()
        
        # Pull arm and observe reward
        reward = env.pull_arm(arm_idx)
        
        # Update algorithm
        algorithm.update(arm_idx, reward)
        pulls_used += 1
        
        # Check if correct arm is identified
        if algorithm.is_complete():
            identified_arm = algorithm.get_best_arm()
            success = (identified_arm == env.optimal_arm)
            break
    
    return success, pulls_used, algorithm.get_best_arm()
```

### Algorithm Comparison

```python
def compare_bai_algorithms(env, algorithms, n_runs=100, max_pulls=1000):
    """Compare different BAI algorithms"""
    results = {}
    
    for name, algorithm_class in algorithms.items():
        success_rates = []
        sample_complexities = []
        
        for run in range(n_runs):
            # Create fresh algorithm instance
            algorithm = algorithm_class(env.n_arms)
            
            # Run experiment
            success, pulls, identified_arm = run_bai_experiment(env, algorithm, max_pulls)
            
            success_rates.append(success)
            sample_complexities.append(pulls)
        
        results[name] = {
            'success_rate': np.mean(success_rates),
            'avg_sample_complexity': np.mean(sample_complexities),
            'std_sample_complexity': np.std(sample_complexities)
        }
    
    return results

# Example usage
arm_means = [0.1, 0.2, 0.3, 0.4, 0.5]
env = BAIEnvironment(arm_means)

algorithms = {
    'Successive Elimination': lambda n: SuccessiveElimination(n, delta=0.1),
    'Racing': lambda n: RacingAlgorithm(n, delta=0.1),
    'LUCB': lambda n: LUCB(n, delta=0.1),
    'Sequential Halving': lambda n: SequentialHalving(n, budget=1000)
}

results = compare_bai_algorithms(env, algorithms)
```

### Visualization

```python
def plot_bai_results(results):
    """Plot BAI algorithm comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success rate comparison
    names = list(results.keys())
    success_rates = [results[name]['success_rate'] for name in names]
    
    ax1.bar(names, success_rates)
    ax1.set_ylabel('Success Rate')
    ax1.set_title('BAI Algorithm Success Rate Comparison')
    ax1.set_ylim(0, 1)
    
    # Sample complexity comparison
    sample_complexities = [results[name]['avg_sample_complexity'] for name in names]
    std_complexities = [results[name]['std_sample_complexity'] for name in names]
    
    ax2.bar(names, sample_complexities, yerr=std_complexities, capsize=5)
    ax2.set_ylabel('Average Sample Complexity')
    ax2.set_title('BAI Algorithm Sample Complexity Comparison')
    
    plt.tight_layout()
    plt.show()

plot_bai_results(results)
```

## Summary

Best Arm Identification provides a powerful framework for pure exploration problems where accurate identification is more important than cumulative reward. Key insights include:

1. **Pure Exploration Focus**: Algorithms prioritize identification over immediate performance
2. **Confidence-based Stopping**: Stop when confidence intervals separate
3. **Theoretical Guarantees**: Optimal sample complexity bounds
4. **Practical Algorithms**: Successive Elimination, Racing, LUCB, Sequential Halving
5. **Wide Applicability**: A/B testing, clinical trials, algorithm selection

BAI algorithms bridge the gap between traditional bandits and pure exploration problems, providing both theoretical guarantees and practical effectiveness for identification tasks.

## Further Reading

- **Pure Exploration/BAI Paper**: Theoretical foundations and algorithms
- **Best Arm Identification Survey**: Comprehensive overview of BAI methods
- **Contextual BAI**: Extensions to contextual settings
- **Linear BAI**: Extensions to linear bandits

---

**Note**: This guide covers the fundamentals of Best Arm Identification. For more advanced topics, see the sections on Contextual BAI, Linear BAI, and Multi-objective BAI.

## From Pure Exploration to Real-World Applications

We've now explored **Best Arm Identification (BAI)** - a fundamental shift in the multi-armed bandit paradigm from cumulative reward maximization to pure exploration. We've seen how algorithms like Successive Elimination, Racing, LUCB, and Sequential Halving focus exclusively on identifying the best arm with high confidence, regardless of the cumulative reward achieved during the learning process.

However, while understanding BAI algorithms is valuable, **the true impact** of multi-armed bandits lies in their real-world applications. Consider the algorithms we've learned - from classical bandits to linear and contextual bandits, and now best arm identification - these theoretical frameworks become powerful when applied to solve actual problems in advertising, healthcare, e-commerce, and beyond.

This motivates our exploration of **applications and use cases** - the practical implementation of bandit algorithms across diverse domains. We'll see how bandits optimize ad selection and bidding in online advertising, how they enable personalized recommendations in e-commerce and content platforms, how they improve clinical trials and drug discovery in healthcare, how they optimize pricing strategies in dynamic markets, and how they enhance A/B testing and algorithm selection processes.

The transition from best arm identification to applications represents the bridge from pure exploration to practical impact - taking our understanding of how to identify optimal actions and applying it to real-world scenarios where intelligent decision-making under uncertainty provides significant value.

In the next section, we'll explore applications and use cases, understanding how bandit algorithms solve real-world problems and create value across diverse domains.

---

**Previous: [Contextual Bandits](03_contextual_bandits.md)** - Learn how to adapt bandit algorithms to changing environments.

**Next: [Applications and Use Cases](05_applications_and_use_cases.md)** - Explore real-world applications of multi-armed bandits. 
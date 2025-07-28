# Linear Bandits

## Introduction

Linear bandits represent a significant extension of classical multi-armed bandits by incorporating structured information about the arms. Instead of treating each arm as an independent entity, linear bandits assume that rewards are linear functions of arm features, enabling more efficient learning through information sharing between similar arms.

### Motivation

The classical multi-armed bandit framework treats each arm independently, which can be inefficient when arms share common structure. Linear bandits address this limitation by:

- **Exploiting arm similarities**: Arms with similar features likely have similar rewards
- **Reducing sample complexity**: Learning in $`d`$-dimensional feature space instead of $`K`$-dimensional arm space
- **Enabling generalization**: Predict rewards for unseen arms based on their features
- **Handling large action spaces**: Efficient when $`d \ll K`$

### Key Applications

Linear bandits are particularly valuable in scenarios with:
- **Large action spaces**: Too many arms to explore independently
- **Structured actions**: Arms can be described by feature vectors
- **Contextual information**: User/context features influence rewards
- **Cold-start problems**: New arms can be evaluated based on features

## Problem Formulation

### Mathematical Setup

Linear bandits extend the classical setting by assuming rewards are linear functions of arm features:

```math
r_t = \langle \theta^*, x_{a_t} \rangle + \eta_t
```

Where:
- $`\theta^* \in \mathbb{R}^d`$: Unknown parameter vector (true reward model)
- $`x_{a_t} \in \mathbb{R}^d`$: Feature vector of chosen arm at time $`t`$
- $`\eta_t`$: Noise term (typically sub-Gaussian)
- $`a_t \in \{1, 2, \ldots, K\}`$: Chosen arm at time $`t`$

### Key Assumptions

**Linear Reward Structure:**
- Rewards are linear in arm features: $`\mathbb{E}[r_t | a_t] = \langle \theta^*, x_{a_t} \rangle`$
- The parameter vector $`\theta^*`$ is unknown but fixed
- Feature vectors $`x_i`$ are known for all arms $`i`$

**Noise Assumptions:**
- Noise terms $`\eta_t`$ are conditionally sub-Gaussian
- $`\mathbb{E}[\eta_t | \mathcal{F}_{t-1}] = 0`$ (zero mean)
- $`\mathbb{E}[\exp(\lambda \eta_t) | \mathcal{F}_{t-1}] \leq \exp(\frac{\lambda^2 \sigma^2}{2})`$ (sub-Gaussian)

**Bounded Features:**
- Feature vectors are bounded: $`\|x_i\|_2 \leq L`$ for all arms $`i`$
- Parameter vector is bounded: $`\|\theta^*\|_2 \leq S`$

### Problem Variants

**Fixed Action Set:**
- Action set $`\mathcal{A} = \{x_1, x_2, \ldots, x_K\}`$ is fixed
- All feature vectors are known in advance

**Continuous Action Set:**
- Action set $`\mathcal{A} \subset \mathbb{R}^d`` is continuous
- Can choose any point in the action set
- More complex but allows infinite action spaces

**Contextual Linear Bandits:**
- Feature vectors depend on context: $`x_{a_t, t}`$
- Context changes over time
- Requires adaptation to changing contexts

## Fundamental Algorithms

### 1. LinUCB (Linear Upper Confidence Bound)

LinUCB extends the UCB principle to linear bandits by maintaining confidence ellipsoids around the parameter estimate.

**Algorithm:**
```math
a_t = \arg\max_{i} \left(\langle \hat{\theta}_t, x_i \rangle + \alpha \sqrt{x_i^T A_t^{-1} x_i}\right)
```

Where:
- $`\hat{\theta}_t`$: Least squares estimate of $`\theta^*`$
- $`A_t = \lambda I + \sum_{s=1}^{t-1} x_{a_s} x_{a_s}^T`$: Design matrix
- $`\alpha`$: Exploration parameter (typically $`\alpha = \sqrt{d \log T}`$)

**Intuition:**
- **Exploitation term**: $`\langle \hat{\theta}_t, x_i \rangle`$ (choose arm with high predicted reward)
- **Exploration term**: $`\alpha \sqrt{x_i^T A_t^{-1} x_i}`$ (choose arm with high uncertainty)
- **Confidence ellipsoid**: $`A_t^{-1}`$ captures uncertainty in parameter estimate

**Implementation:**
```python
import numpy as np
from scipy.linalg import solve

class LinUCB:
    def __init__(self, d, alpha=1.0, lambda_reg=1.0):
        self.d = d
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Initialize design matrix and parameter estimate
        self.A = lambda_reg * np.eye(d)  # Regularized design matrix
        self.b = np.zeros(d)  # Cumulative rewards
        self.theta_hat = np.zeros(d)  # Parameter estimate
        
    def select_arm(self, arms):
        """Select arm using LinUCB algorithm"""
        # Update parameter estimate
        self.theta_hat = solve(self.A, self.b)
        
        # Calculate UCB values for all arms
        ucb_values = []
        for x in arms:
            # Exploitation term
            exploitation = np.dot(self.theta_hat, x)
            
            # Exploration term
            exploration = self.alpha * np.sqrt(np.dot(x, solve(self.A, x)))
            
            ucb_values.append(exploitation + exploration)
        
        return np.argmax(ucb_values)
    
    def update(self, arm_idx, reward, arm_features):
        """Update algorithm with observed reward"""
        x = arm_features[arm_idx]
        
        # Update design matrix and cumulative rewards
        self.A += np.outer(x, x)
        self.b += reward * x
```

**Theoretical Guarantees:**
- **Regret Bound**: $`O(d\sqrt{T \log T})`$ with high probability
- **Gap-dependent bound**: $`O(\frac{d^2 \log T}{\Delta_{\min}})`$ where $`\Delta_{\min}`$ is the minimum gap
- **Optimal up to logarithmic factors**

### 2. Linear Thompson Sampling

Linear Thompson Sampling maintains a Gaussian posterior over the parameter vector and samples from this posterior to select actions.

**Algorithm:**
1. Maintain Gaussian posterior: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, A_t^{-1})`$
2. Sample parameter: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, \sigma^2 A_t^{-1})`$
3. Choose action: $`a_t = \arg\max_i \langle \theta_t, x_i \rangle`$

**Posterior Update:**
- **Prior**: $`\theta \sim \mathcal{N}(0, \lambda I)`$ (regularization)
- **Likelihood**: $`r_t | \theta, x_{a_t} \sim \mathcal{N}(\langle \theta, x_{a_t} \rangle, \sigma^2)`$
- **Posterior**: $`\theta | \mathcal{F}_t \sim \mathcal{N}(\hat{\theta}_t, A_t^{-1})`$

**Implementation:**
```python
import numpy as np
from scipy.stats import multivariate_normal

class LinearThompsonSampling:
    def __init__(self, d, sigma=1.0, lambda_reg=1.0):
        self.d = d
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        
        # Initialize posterior parameters
        self.A = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.theta_hat = np.zeros(d)
        
    def select_arm(self, arms):
        """Select arm using Linear Thompson Sampling"""
        # Update parameter estimate
        self.theta_hat = np.linalg.solve(self.A, self.b)
        
        # Sample from posterior
        posterior_cov = self.sigma**2 * np.linalg.inv(self.A)
        theta_sample = multivariate_normal.rvs(
            mean=self.theta_hat, 
            cov=posterior_cov
        )
        
        # Choose arm with highest sampled reward
        predicted_rewards = [np.dot(theta_sample, x) for x in arms]
        return np.argmax(predicted_rewards)
    
    def update(self, arm_idx, reward, arm_features):
        """Update algorithm with observed reward"""
        x = arm_features[arm_idx]
        
        # Update posterior parameters
        self.A += np.outer(x, x) / (self.sigma**2)
        self.b += reward * x / (self.sigma**2)
```

**Theoretical Guarantees:**
- **Regret Bound**: $`O(d\sqrt{T \log T})`$ under Bayesian assumptions
- **Often performs better in practice** than LinUCB
- **Automatic exploration** through posterior sampling

### 3. OFUL (Optimism in the Face of Uncertainty for Linear Bandits)

OFUL is a more sophisticated algorithm that constructs confidence ellipsoids and uses optimism to guide exploration.

**Algorithm:**
```math
a_t = \arg\max_{i} \max_{\theta \in \mathcal{C}_t} \langle \theta, x_i \rangle
```

Where $`\mathcal{C}_t`$ is the confidence ellipsoid:
```math
\mathcal{C}_t = \{\theta : \|\theta - \hat{\theta}_t\|_{A_t} \leq \beta_t\}
```

**Implementation:**
```python
class OFUL:
    def __init__(self, d, delta=0.1, lambda_reg=1.0):
        self.d = d
        self.delta = delta
        self.lambda_reg = lambda_reg
        
        self.A = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.theta_hat = np.zeros(d)
        
    def select_arm(self, arms):
        """Select arm using OFUL algorithm"""
        # Update parameter estimate
        self.theta_hat = np.linalg.solve(self.A, self.b)
        
        # Calculate confidence radius
        beta = self._calculate_beta()
        
        # Find optimistic parameter within confidence ellipsoid
        theta_opt = self._find_optimistic_parameter(arms, beta)
        
        # Choose arm with highest optimistic reward
        predicted_rewards = [np.dot(theta_opt, x) for x in arms]
        return np.argmax(predicted_rewards)
    
    def _calculate_beta(self):
        """Calculate confidence radius"""
        t = np.trace(self.A) - self.d * self.lambda_reg
        return np.sqrt(2 * np.log(1 / self.delta) + self.d * np.log(1 + t / (self.d * self.lambda_reg)))
    
    def _find_optimistic_parameter(self, arms, beta):
        """Find optimistic parameter within confidence ellipsoid"""
        # This is a simplified version - in practice, this requires solving an optimization problem
        # For simplicity, we'll use the UCB approach
        return self.theta_hat + beta * np.linalg.solve(self.A, np.random.randn(self.d))
```

## Theoretical Analysis

### Regret Bounds

**LinUCB Regret Analysis:**
For LinUCB with appropriate exploration parameter $`\alpha`$:

```math
\mathbb{E}[R(T)] \leq O\left(d\sqrt{T \log T}\right)
```

**Key Insights:**
- **Dimension-dependent**: Regret scales with $`d`$ instead of $`K`$
- **Sublinear growth**: $`O(\sqrt{T})`$ instead of linear
- **Logarithmic factors**: Due to confidence interval construction

**Gap-dependent Bounds:**
For arms with minimum gap $`\Delta_{\min}`$:

```math
\mathbb{E}[R(T)] \leq O\left(\frac{d^2 \log T}{\Delta_{\min}}\right)
```

### Lower Bounds

**Minimax Lower Bound:**
For any algorithm in linear bandits:

```math
\mathbb{E}[R(T)] \geq \Omega(d\sqrt{T})
```

**Gap-dependent Lower Bound:**
```math
\mathbb{E}[R(T)] \geq \Omega\left(\frac{d \log T}{\Delta_{\min}}\right)
```

### Concentration Inequalities

**Self-Normalized Martingale:**
For linear bandits, we use self-normalized martingale concentration:

```math
P\left(\|\theta^* - \hat{\theta}_t\|_{A_t} \geq \beta_t\right) \leq \delta
```

Where $`\beta_t = \sqrt{2 \log \frac{\det(A_t)^{1/2}}{\delta \lambda^{d/2}}} + \sqrt{\lambda} S`$

## Practical Considerations

### Feature Engineering

**Feature Selection:**
- **Domain knowledge**: Use relevant features for the problem
- **Dimensionality reduction**: PCA, feature selection methods
- **Kernel methods**: Extend to non-linear reward functions

**Feature Normalization:**
```python
def normalize_features(features):
    """Normalize feature vectors to unit norm"""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / norms

def standardize_features(features):
    """Standardize features to zero mean and unit variance"""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / std
```

### Parameter Tuning

**Exploration Parameter ($`\alpha`$):**
- **Theoretical**: $`\alpha = \sqrt{d \log T}`$
- **Practical**: $`\alpha = 1.0`$ often works well
- **Cross-validation**: Tune on validation data

**Regularization ($`\lambda`$):**
- **Ridge regression**: $`\lambda = 1.0`$ is often sufficient
- **Feature scaling**: Adjust based on feature magnitudes
- **Numerical stability**: Larger $`\lambda`$ for ill-conditioned problems

### Numerical Stability

**Matrix Inversion:**
```python
def stable_solve(A, b):
    """Stable matrix solve with regularization"""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Add small regularization if matrix is singular
        A_reg = A + 1e-6 * np.eye(A.shape[0])
        return np.linalg.solve(A_reg, b)
```

**Conditioning:**
```python
def check_conditioning(A):
    """Check condition number of design matrix"""
    eigenvals = np.linalg.eigvals(A)
    condition_number = np.max(eigenvals) / np.min(eigenvals)
    return condition_number
```

## Advanced Topics

### Generalized Linear Bandits

**Problem Extension:**
Instead of linear rewards, use generalized linear models:

```math
\mathbb{E}[r_t | a_t] = g(\langle \theta^*, x_{a_t} \rangle)
```

Where $`g`$ is a known link function (e.g., sigmoid for logistic regression).

**Algorithms:**
- **GLM-UCB**: Extend UCB to generalized linear models
- **GLM-TS**: Thompson sampling with GLM likelihood
- **Online Newton Step**: Second-order optimization methods

### Kernel Bandits

**Non-linear Extensions:**
Use kernel functions to capture non-linear reward functions:

```math
\mathbb{E}[r_t | a_t] = f(x_{a_t}) = \sum_{i=1}^{t-1} \alpha_i k(x_i, x_{a_t})
```

**Algorithms:**
- **GP-UCB**: Gaussian process upper confidence bound
- **Kernel-TS**: Thompson sampling with Gaussian processes
- **Random Fourier Features**: Efficient approximation for large datasets

### Bandits with Constraints

**Resource Constraints:**
- **Budget constraints**: Limited total cost
- **Time constraints**: Limited time horizon
- **Safety constraints**: Ensure safe exploration

**Algorithms:**
- **Constrained UCB**: Add constraint handling to UCB
- **Safe exploration**: Ensure constraints are satisfied
- **Multi-objective bandits**: Balance multiple objectives

## Applications

### Recommendation Systems

**Content Recommendation:**
```python
class ContentRecommender:
    def __init__(self, n_items, feature_dim):
        self.bandit = LinUCB(feature_dim)
        self.item_features = self._extract_features(n_items)
    
    def recommend(self, user_context):
        """Recommend content based on user context"""
        # Combine user context with item features
        contextual_features = self._combine_features(user_context, self.item_features)
        
        # Select item using bandit algorithm
        item_idx = self.bandit.select_arm(contextual_features)
        return item_idx
    
    def update(self, item_idx, user_context, reward):
        """Update model with user feedback"""
        contextual_features = self._combine_features(user_context, self.item_features)
        self.bandit.update(item_idx, reward, contextual_features)
```

### Online Advertising

**Ad Selection with User Features:**
```python
class AdSelector:
    def __init__(self, n_ads, user_feature_dim):
        self.bandit = LinearThompsonSampling(user_feature_dim)
        self.ad_features = self._extract_ad_features(n_ads)
    
    def select_ad(self, user_features):
        """Select ad based on user features"""
        # Combine user and ad features
        combined_features = self._combine_user_ad_features(user_features, self.ad_features)
        
        # Select ad using bandit algorithm
        ad_idx = self.bandit.select_arm(combined_features)
        return ad_idx
    
    def update(self, ad_idx, user_features, click):
        """Update model with click feedback"""
        combined_features = self._combine_user_ad_features(user_features, self.ad_features)
        self.bandit.update(ad_idx, click, combined_features)
```

### Clinical Trials

**Adaptive Treatment Assignment:**
```python
class AdaptiveTrial:
    def __init__(self, n_treatments, patient_feature_dim):
        self.bandit = LinUCB(patient_feature_dim)
        self.treatment_features = self._extract_treatment_features(n_treatments)
    
    def assign_treatment(self, patient_features):
        """Assign treatment based on patient features"""
        # Combine patient and treatment features
        combined_features = self._combine_patient_treatment_features(
            patient_features, self.treatment_features
        )
        
        # Select treatment using bandit algorithm
        treatment_idx = self.bandit.select_arm(combined_features)
        return treatment_idx
    
    def update(self, treatment_idx, patient_features, outcome):
        """Update model with treatment outcome"""
        combined_features = self._combine_patient_treatment_features(
            patient_features, self.treatment_features
        )
        self.bandit.update(treatment_idx, outcome, combined_features)
```

## Implementation Examples

### Complete Linear Bandit Environment

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearBanditEnvironment:
    def __init__(self, theta_star, arms, noise_std=0.1):
        self.theta_star = theta_star
        self.arms = arms
        self.noise_std = noise_std
        self.d = len(theta_star)
        
    def pull_arm(self, arm_idx):
        """Pull arm and return reward"""
        x = self.arms[arm_idx]
        expected_reward = np.dot(self.theta_star, x)
        noise = np.random.normal(0, self.noise_std)
        return expected_reward + noise
    
    def get_optimal_reward(self):
        """Get optimal expected reward"""
        rewards = [np.dot(self.theta_star, x) for x in self.arms]
        return max(rewards)
    
    def get_regret(self, chosen_arms, rewards):
        """Calculate cumulative regret"""
        optimal_reward = self.get_optimal_reward()
        cumulative_optimal = np.cumsum([optimal_reward] * len(rewards))
        cumulative_rewards = np.cumsum(rewards)
        return cumulative_optimal - cumulative_rewards

def run_linear_bandit_experiment(env, algorithm, T=1000):
    """Run linear bandit experiment"""
    chosen_arms = []
    rewards = []
    
    for t in range(T):
        # Select arm
        arm_idx = algorithm.select_arm(env.arms)
        
        # Pull arm and observe reward
        reward = env.pull_arm(arm_idx)
        
        # Update algorithm
        algorithm.update(arm_idx, reward, env.arms)
        
        chosen_arms.append(arm_idx)
        rewards.append(reward)
    
    return chosen_arms, rewards
```

### Algorithm Comparison

```python
def compare_linear_algorithms(env, algorithms, T=1000, n_runs=50):
    """Compare different linear bandit algorithms"""
    results = {}
    
    for name, algorithm_class in algorithms.items():
        regrets = []
        for run in range(n_runs):
            # Create fresh algorithm instance
            algorithm = algorithm_class(env.d)
            
            # Run experiment
            chosen_arms, rewards = run_linear_bandit_experiment(env, algorithm, T)
            
            # Calculate regret
            regret = env.get_regret(chosen_arms, rewards)
            regrets.append(regret)
        
        results[name] = np.mean(regrets, axis=0)
    
    return results

# Example usage
d = 5
K = 10
theta_star = np.random.randn(d)
theta_star = theta_star / np.linalg.norm(theta_star)  # Normalize

# Generate random arms
arms = np.random.randn(K, d)
arms = arms / np.linalg.norm(arms, axis=1, keepdims=True)  # Normalize

env = LinearBanditEnvironment(theta_star, arms)

algorithms = {
    'LinUCB': lambda d: LinUCB(d, alpha=1.0),
    'Linear TS': lambda d: LinearThompsonSampling(d, sigma=1.0),
    'OFUL': lambda d: OFUL(d, delta=0.1)
}

results = compare_linear_algorithms(env, algorithms)
```

### Visualization

```python
def plot_linear_bandit_results(results, T):
    """Plot regret comparison for linear bandit algorithms"""
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative regret
    plt.subplot(2, 1, 1)
    for name, regret in results.items():
        plt.plot(range(1, T+1), regret, label=name, linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.title('Linear Bandit Algorithm Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot regret rate (regret / sqrt(t))
    plt.subplot(2, 1, 2)
    for name, regret in results.items():
        regret_rate = regret / np.sqrt(range(1, T+1))
        plt.plot(range(1, T+1), regret_rate, label=name, linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Regret Rate (R(t) / √t)')
    plt.title('Regret Rate Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_linear_bandit_results(results, 1000)
```

## Summary

Linear bandits provide a powerful framework for sequential decision-making with structured action spaces. Key insights include:

1. **Feature-based Learning**: Exploit arm similarities through feature representations
2. **Efficient Exploration**: Confidence ellipsoids guide exploration in feature space
3. **Theoretical Guarantees**: Sublinear regret bounds scaling with dimension $`d`$
4. **Practical Algorithms**: LinUCB, Linear Thompson Sampling, and OFUL
5. **Wide Applicability**: Recommendation systems, advertising, clinical trials

Linear bandits bridge the gap between classical bandits and more complex contextual settings, providing both theoretical guarantees and practical effectiveness.

## Further Reading

- **Linear Bandits Paper**: Theoretical foundations by Dani et al.
- **Generalized Linear Bandits**: Extension to GLM by Filippi et al.
- **Bandit Algorithms Textbook**: Comprehensive treatment by Lattimore and Szepesvári
- **Online Learning Resources**: Stanford CS234, UC Berkeley CS285

---

**Note**: This guide covers the fundamentals of linear bandits. For more advanced topics, see the sections on Contextual Bandits, Kernel Bandits, and Generalized Linear Bandits. 
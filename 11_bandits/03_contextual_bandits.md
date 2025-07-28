 # Contextual Bandits

## Introduction

Contextual bandits represent a natural evolution of multi-armed bandits by incorporating dynamic context information that influences reward distributions. Unlike classical bandits where arms have fixed reward distributions, contextual bandits adapt to changing environments where the optimal action depends on the current context.

### Motivation

The classical multi-armed bandit framework assumes stationary reward distributions, which is often unrealistic in practice. Contextual bandits address this limitation by:

- **Adapting to changing environments**: Context provides information about the current state
- **Personalizing decisions**: Actions are tailored to specific contexts
- **Handling non-stationarity**: Reward distributions change with context
- **Enabling real-world applications**: Most practical scenarios involve contextual information

### Key Applications

Contextual bandits are essential in scenarios where:
- **User context matters**: Different users respond differently to the same action
- **Temporal dynamics**: Optimal actions change over time
- **Personalization**: Tailoring decisions to individual characteristics
- **Adaptive systems**: Learning and adapting to changing environments

## Problem Formulation

### Mathematical Setup

Contextual bandits introduce context (state) information that changes over time:

```math
r_t = \langle \theta^*, x_{a_t, t} \rangle + \eta_t
```

Where:
- $`\theta^* \in \mathbb{R}^d`$: Unknown parameter vector (true reward model)
- $`x_{a_t, t} \in \mathbb{R}^d`$: Feature vector of chosen arm at time $`t`$ in context $`t`$
- $`\eta_t`$: Noise term (typically sub-Gaussian)
- $`a_t \in \{1, 2, \ldots, K\}`$: Chosen arm at time $`t`$
- $`c_t`$: Context at time $`t`$ (may be implicit in $`x_{a_t, t}`$)

### Key Assumptions

**Contextual Reward Structure:**
- Rewards are linear in context-arm features: $`\mathbb{E}[r_t | a_t, c_t] = \langle \theta^*, x_{a_t, t} \rangle`$
- The parameter vector $`\theta^*`$ is unknown but fixed
- Feature vectors $`x_{i, t}`$ depend on both arm $`i`$ and context $`c_t`$

**Context Generation:**
- Contexts $`c_t`$ may be drawn from a distribution or chosen by an adversary
- Contexts can be arbitrary (adversarial) or have structure (stochastic)
- Feature vectors $`x_{i, t}`$ are revealed for all arms $`i`$ at time $`t`$

**Noise Assumptions:**
- Noise terms $`\eta_t`$ are conditionally sub-Gaussian given the history
- $`\mathbb{E}[\eta_t | \mathcal{F}_{t-1}, a_t, c_t] = 0`$ (zero mean)
- $`\mathbb{E}[\exp(\lambda \eta_t) | \mathcal{F}_{t-1}, a_t, c_t] \leq \exp(\frac{\lambda^2 \sigma^2}{2})`$ (sub-Gaussian)

### Problem Variants

**Stochastic Contexts:**
- Contexts are drawn from a fixed distribution
- Allows for more optimistic regret bounds
- Common in recommendation systems

**Adversarial Contexts:**
- Contexts are chosen by an adversary
- More challenging theoretical guarantees
- Common in online advertising

**Contextual Linear Bandits:**
- Feature vectors are linear combinations of context and arm features
- $`x_{a_t, t} = \phi(c_t, a_t)`$ where $`\phi`$ is a known feature map

**General Contextual Bandits:**
- No specific structure on reward functions
- Requires more sophisticated algorithms (e.g., neural bandits)

## Fundamental Algorithms

### 1. Contextual UCB (CUCB)

Contextual UCB extends the UCB principle to handle changing contexts by maintaining context-dependent confidence intervals.

**Algorithm:**
```math
a_t = \arg\max_{i} \left(\langle \hat{\theta}_t, x_{i, t} \rangle + \alpha \sqrt{x_{i, t}^T A_t^{-1} x_{i, t}}\right)
```

Where:
- $`\hat{\theta}_t`$: Least squares estimate of $`\theta^*`$
- $`A_t = \lambda I + \sum_{s=1}^{t-1} x_{a_s, s} x_{a_s, s}^T`$: Design matrix
- $`\alpha`$: Exploration parameter (typically $`\alpha = \sqrt{d \log T}`$)

**Intuition:**
- **Exploitation term**: $`\langle \hat{\theta}_t, x_{i, t} \rangle`$ (choose arm with high predicted reward in current context)
- **Exploration term**: $`\alpha \sqrt{x_{i, t}^T A_t^{-1} x_{i, t}}`$ (choose arm with high uncertainty in current context)
- **Context adaptation**: Confidence intervals adapt to the current context

**Implementation:**
```python
import numpy as np
from scipy.linalg import solve

class ContextualUCB:
    def __init__(self, d, alpha=1.0, lambda_reg=1.0):
        self.d = d
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Initialize design matrix and parameter estimate
        self.A = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.theta_hat = np.zeros(d)
        
    def select_arm(self, context_features):
        """Select arm using Contextual UCB algorithm"""
        # Update parameter estimate
        self.theta_hat = solve(self.A, self.b)
        
        # Calculate UCB values for all arms in current context
        ucb_values = []
        for x in context_features:
            # Exploitation term
            exploitation = np.dot(self.theta_hat, x)
            
            # Exploration term
            exploration = self.alpha * np.sqrt(np.dot(x, solve(self.A, x)))
            
            ucb_values.append(exploitation + exploration)
        
        return np.argmax(ucb_values)
    
    def update(self, arm_idx, reward, context_features):
        """Update algorithm with observed reward"""
        x = context_features[arm_idx]
        
        # Update design matrix and cumulative rewards
        self.A += np.outer(x, x)
        self.b += reward * x
```

### 2. Contextual Thompson Sampling (CTS)

Contextual Thompson Sampling maintains a Gaussian posterior over the parameter vector and samples from this posterior to select actions in each context.

**Algorithm:**
1. Maintain Gaussian posterior: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, A_t^{-1})`$
2. Sample parameter: $`\theta_t \sim \mathcal{N}(\hat{\theta}_t, \sigma^2 A_t^{-1})`$
3. Choose action: $`a_t = \arg\max_i \langle \theta_t, x_{i, t} \rangle`$

**Implementation:**
```python
import numpy as np
from scipy.stats import multivariate_normal

class ContextualThompsonSampling:
    def __init__(self, d, sigma=1.0, lambda_reg=1.0):
        self.d = d
        self.sigma = sigma
        self.lambda_reg = lambda_reg
        
        # Initialize posterior parameters
        self.A = lambda_reg * np.eye(d)
        self.b = np.zeros(d)
        self.theta_hat = np.zeros(d)
        
    def select_arm(self, context_features):
        """Select arm using Contextual Thompson Sampling"""
        # Update parameter estimate
        self.theta_hat = np.linalg.solve(self.A, self.b)
        
        # Sample from posterior
        posterior_cov = self.sigma**2 * np.linalg.inv(self.A)
        theta_sample = multivariate_normal.rvs(
            mean=self.theta_hat, 
            cov=posterior_cov
        )
        
        # Choose arm with highest sampled reward in current context
        predicted_rewards = [np.dot(theta_sample, x) for x in context_features]
        return np.argmax(predicted_rewards)
    
    def update(self, arm_idx, reward, context_features):
        """Update algorithm with observed reward"""
        x = context_features[arm_idx]
        
        # Update posterior parameters
        self.A += np.outer(x, x) / (self.sigma**2)
        self.b += reward * x / (self.sigma**2)
```

### 3. Neural Contextual Bandits

Neural contextual bandits use deep neural networks to model complex, non-linear reward functions that depend on context.

**Algorithm:**
1. Maintain neural network model for reward prediction
2. Use uncertainty quantification (e.g., dropout, ensemble methods)
3. Select actions based on predicted rewards and uncertainty

**Implementation:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralContextualBandit:
    def __init__(self, input_dim, hidden_dim=64, num_arms=10, dropout_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_arms = num_arms
        self.dropout_rate = dropout_rate
        
        # Neural network model
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_arms)
        )
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def select_arm(self, context_features):
        """Select arm using neural contextual bandit"""
        # Convert context features to tensor
        context_tensor = torch.FloatTensor(context_features).unsqueeze(0)
        
        # Get predictions from neural network
        with torch.no_grad():
            predictions = self.model(context_tensor)
            
        # Add exploration noise (Thompson sampling approximation)
        noise = torch.randn_like(predictions) * 0.1
        predictions += noise
        
        return torch.argmax(predictions).item()
    
    def update(self, arm_idx, reward, context_features):
        """Update neural network with observed reward"""
        context_tensor = torch.FloatTensor(context_features).unsqueeze(0)
        
        # Create target vector
        target = torch.zeros(self.num_arms)
        target[arm_idx] = reward
        target = target.unsqueeze(0)
        
        # Forward pass
        predictions = self.model(context_tensor)
        
        # Compute loss and update
        loss = self.criterion(predictions, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

### 4. LinUCB with Disjoint Models

LinUCB with disjoint models maintains separate linear models for each arm, allowing for more flexible context-arm interactions.

**Algorithm:**
For each arm $`i`$, maintain separate parameter estimate $`\hat{\theta}_i`$:

```math
a_t = \arg\max_{i} \left(\langle \hat{\theta}_{i, t}, x_t \rangle + \alpha \sqrt{x_t^T A_{i, t}^{-1} x_t}\right)
```

**Implementation:**
```python
class DisjointLinUCB:
    def __init__(self, d, num_arms, alpha=1.0, lambda_reg=1.0):
        self.d = d
        self.num_arms = num_arms
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Separate models for each arm
        self.A = [lambda_reg * np.eye(d) for _ in range(num_arms)]
        self.b = [np.zeros(d) for _ in range(num_arms)]
        self.theta_hat = [np.zeros(d) for _ in range(num_arms)]
        
    def select_arm(self, context):
        """Select arm using disjoint LinUCB"""
        ucb_values = []
        
        for i in range(self.num_arms):
            # Update parameter estimate for arm i
            self.theta_hat[i] = np.linalg.solve(self.A[i], self.b[i])
            
            # Calculate UCB value for arm i
            exploitation = np.dot(self.theta_hat[i], context)
            exploration = self.alpha * np.sqrt(np.dot(context, np.linalg.solve(self.A[i], context)))
            ucb_values.append(exploitation + exploration)
        
        return np.argmax(ucb_values)
    
    def update(self, arm_idx, reward, context):
        """Update model for specific arm"""
        # Update design matrix and cumulative rewards for the chosen arm
        self.A[arm_idx] += np.outer(context, context)
        self.b[arm_idx] += reward * context
```

## Theoretical Analysis

### Regret Bounds

**Contextual UCB Regret Analysis:**
For contextual UCB with appropriate exploration parameter:

```math
\mathbb{E}[R(T)] \leq O(d\sqrt{T \log T})
```

**Key Insights:**
- **Dimension-dependent**: Regret scales with feature dimension $`d`$
- **Context adaptation**: Algorithm adapts to changing contexts
- **Sublinear growth**: Achieves $`O(\sqrt{T})`$ regret

**Neural Bandit Bounds:**
For neural contextual bandits with $`L`$-layer networks:

```math
\mathbb{E}[R(T)] \leq O(\sqrt{T \log T} \cdot \text{poly}(L, d))
```

### Context Assumptions

**Stochastic Contexts:**
- Contexts drawn from fixed distribution
- Allows for more optimistic bounds
- Common in recommendation systems

**Adversarial Contexts:**
- Contexts chosen by adversary
- More challenging theoretical guarantees
- Common in online advertising

**Contextual Gap:**
The contextual gap $`\Delta_t(i) = \max_j \langle \theta^*, x_{j, t} \rangle - \langle \theta^*, x_{i, t} \rangle`$ measures the suboptimality of arm $`i`$ in context $`t`$.

### Sample Complexity

**Contextual Best Arm Identification:**
For identifying the best arm in each context:

```math
\mathbb{E}[N] \leq O\left(\frac{d^2 \log T}{\Delta_{\min}^2}\right)
```

Where $`\Delta_{\min}`$ is the minimum contextual gap across all contexts.

## Practical Considerations

### Context Engineering

**Feature Extraction:**
```python
def extract_user_features(user_data):
    """Extract user features for contextual bandit"""
    features = []
    
    # Demographics
    features.extend([
        user_data.get('age', 0) / 100.0,  # Normalize age
        user_data.get('gender', 0),  # Binary or categorical
        user_data.get('location', 0)  # Location encoding
    ])
    
    # Behavioral features
    features.extend([
        user_data.get('session_duration', 0) / 3600.0,  # Hours
        user_data.get('page_views', 0) / 100.0,  # Normalize
        user_data.get('purchase_history', 0)  # Purchase count
    ])
    
    # Contextual features
    features.extend([
        user_data.get('time_of_day', 0) / 24.0,  # Hour of day
        user_data.get('day_of_week', 0) / 7.0,  # Day of week
        user_data.get('device_type', 0)  # Device encoding
    ])
    
    return np.array(features)

def extract_item_features(item_data):
    """Extract item features for contextual bandit"""
    features = []
    
    # Item characteristics
    features.extend([
        item_data.get('category', 0),  # Category encoding
        item_data.get('price', 0) / 1000.0,  # Normalize price
        item_data.get('rating', 0) / 5.0,  # Normalize rating
        item_data.get('popularity', 0) / 1000.0  # Normalize popularity
    ])
    
    # Content features
    features.extend([
        item_data.get('content_length', 0) / 1000.0,
        item_data.get('has_image', 0),  # Binary
        item_data.get('has_video', 0)  # Binary
    ])
    
    return np.array(features)
```

**Context-Arm Feature Combination:**
```python
def combine_context_arm_features(context, arm_features):
    """Combine context and arm features"""
    # Simple concatenation
    combined = np.concatenate([context, arm_features])
    
    # Or use interaction features
    interactions = np.outer(context, arm_features).flatten()
    combined = np.concatenate([context, arm_features, interactions])
    
    return combined

def create_contextual_features(context, all_arms):
    """Create contextual features for all arms"""
    contextual_features = []
    
    for arm_features in all_arms:
        combined = combine_context_arm_features(context, arm_features)
        contextual_features.append(combined)
    
    return contextual_features
```

### Parameter Tuning

**Exploration Parameter ($`\alpha`$):**
- **Theoretical**: $`\alpha = \sqrt{d \log T}`$
- **Practical**: $`\alpha = 1.0`$ often works well
- **Context-dependent**: May need different values for different contexts

**Regularization ($`\lambda`$):**
- **Ridge regression**: $`\lambda = 1.0`$ is often sufficient
- **Feature scaling**: Adjust based on feature magnitudes
- **Numerical stability**: Larger $`\lambda`$ for ill-conditioned problems

### Context Adaptation

**Context Clustering:**
```python
class ContextualBanditWithClustering:
    def __init__(self, d, num_clusters=10):
        self.d = d
        self.num_clusters = num_clusters
        self.cluster_models = [LinUCB(d) for _ in range(num_clusters)]
        self.cluster_centers = None
        
    def assign_cluster(self, context):
        """Assign context to nearest cluster"""
        if self.cluster_centers is None:
            return np.random.randint(self.num_clusters)
        
        distances = [np.linalg.norm(context - center) for center in self.cluster_centers]
        return np.argmin(distances)
    
    def select_arm(self, context_features):
        """Select arm using clustered contextual bandit"""
        cluster_idx = self.assign_cluster(context_features[0])  # Use first arm's context
        return self.cluster_models[cluster_idx].select_arm(context_features)
    
    def update(self, arm_idx, reward, context_features):
        """Update model for assigned cluster"""
        cluster_idx = self.assign_cluster(context_features[0])
        self.cluster_models[cluster_idx].update(arm_idx, reward, context_features)
```

## Advanced Topics

### Contextual Bandits with Constraints

**Resource Constraints:**
- **Budget constraints**: Limited total cost across contexts
- **Time constraints**: Limited time horizon
- **Safety constraints**: Ensure safe exploration in each context

**Algorithms:**
- **Constrained Contextual UCB**: Add constraint handling to CUCB
- **Safe contextual exploration**: Ensure constraints are satisfied
- **Multi-objective contextual bandits**: Balance multiple objectives

### Non-stationary Contextual Bandits

**Problem:**
Contexts and reward functions change over time, requiring adaptation.

**Solutions:**
- **Sliding window**: Only use recent observations
- **Discounting**: Give more weight to recent contexts
- **Change detection**: Detect when contexts change significantly

### Federated Contextual Bandits

**Distributed Learning:**
- Multiple agents learning from different contexts
- Privacy-preserving learning
- Communication-efficient algorithms

**Applications:**
- **Mobile recommendation**: Learn across devices
- **Healthcare**: Learn across hospitals while preserving privacy
- **E-commerce**: Learn across different regions

## Applications

### Online Advertising

**Ad Selection with User Context:**
```python
class ContextualAdSelector:
    def __init__(self, n_ads, user_feature_dim):
        self.bandit = ContextualUCB(user_feature_dim)
        self.ad_features = self._extract_ad_features(n_ads)
    
    def select_ad(self, user_context):
        """Select ad based on user context"""
        # Combine user context with ad features
        contextual_features = self._combine_user_ad_features(user_context, self.ad_features)
        
        # Select ad using contextual bandit
        ad_idx = self.bandit.select_arm(contextual_features)
        return ad_idx
    
    def update(self, ad_idx, user_context, click):
        """Update model with click feedback"""
        contextual_features = self._combine_user_ad_features(user_context, self.ad_features)
        self.bandit.update(ad_idx, click, contextual_features)
    
    def _combine_user_ad_features(self, user_context, ad_features):
        """Combine user and ad features for contextual learning"""
        combined_features = []
        for ad_feat in ad_features:
            # Simple concatenation
            combined = np.concatenate([user_context, ad_feat])
            combined_features.append(combined)
        return combined_features
```

### Recommendation Systems

**Personalized Content Recommendation:**
```python
class PersonalizedRecommender:
    def __init__(self, n_items, user_feature_dim):
        self.bandit = ContextualThompsonSampling(user_feature_dim)
        self.item_features = self._extract_item_features(n_items)
    
    def recommend(self, user_context):
        """Recommend content based on user context"""
        # Combine user context with item features
        contextual_features = self._combine_user_item_features(user_context, self.item_features)
        
        # Select item using contextual bandit
        item_idx = self.bandit.select_arm(contextual_features)
        return item_idx
    
    def update(self, item_idx, user_context, engagement):
        """Update model with user engagement"""
        contextual_features = self._combine_user_item_features(user_context, self.item_features)
        self.bandit.update(item_idx, engagement, contextual_features)
```

### Clinical Trials

**Adaptive Treatment Assignment:**
```python
class AdaptiveClinicalTrial:
    def __init__(self, n_treatments, patient_feature_dim):
        self.bandit = NeuralContextualBandit(patient_feature_dim, num_arms=n_treatments)
    
    def assign_treatment(self, patient_features):
        """Assign treatment based on patient features"""
        # Convert patient features to contextual features
        contextual_features = self._create_contextual_features(patient_features)
        
        # Select treatment using neural contextual bandit
        treatment_idx = self.bandit.select_arm(contextual_features)
        return treatment_idx
    
    def update(self, treatment_idx, patient_features, outcome):
        """Update model with treatment outcome"""
        contextual_features = self._create_contextual_features(patient_features)
        self.bandit.update(treatment_idx, outcome, contextual_features)
    
    def _create_contextual_features(self, patient_features):
        """Create contextual features for all treatments"""
        # For neural bandits, we need features for each treatment
        contextual_features = []
        for treatment in range(self.bandit.num_arms):
            # Combine patient features with treatment encoding
            treatment_features = np.zeros(self.bandit.num_arms)
            treatment_features[treatment] = 1.0
            combined = np.concatenate([patient_features, treatment_features])
            contextual_features.append(combined)
        return contextual_features
```

### Dynamic Pricing

**Price Optimization with Customer Features:**
```python
class DynamicPricer:
    def __init__(self, n_price_levels, customer_feature_dim):
        self.bandit = ContextualUCB(customer_feature_dim)
        self.price_levels = np.linspace(10, 100, n_price_levels)
    
    def set_price(self, customer_features):
        """Set price based on customer features"""
        # Create contextual features for each price level
        contextual_features = self._create_price_features(customer_features)
        
        # Select price using contextual bandit
        price_idx = self.bandit.select_arm(contextual_features)
        return self.price_levels[price_idx]
    
    def update(self, price_idx, customer_features, purchase):
        """Update model with purchase decision"""
        contextual_features = self._create_price_features(customer_features)
        self.bandit.update(price_idx, purchase, contextual_features)
    
    def _create_price_features(self, customer_features):
        """Create contextual features for each price level"""
        contextual_features = []
        for price in self.price_levels:
            # Combine customer features with price
            price_feature = price / 100.0  # Normalize price
            combined = np.concatenate([customer_features, [price_feature]])
            contextual_features.append(combined)
        return contextual_features
```

## Implementation Examples

### Complete Contextual Bandit Environment

```python
import numpy as np
import matplotlib.pyplot as plt

class ContextualBanditEnvironment:
    def __init__(self, theta_star, context_generator, noise_std=0.1):
        self.theta_star = theta_star
        self.context_generator = context_generator
        self.noise_std = noise_std
        self.d = len(theta_star)
        
    def generate_context(self):
        """Generate context for current time step"""
        return self.context_generator()
    
    def generate_arm_features(self, context, n_arms):
        """Generate arm features based on context"""
        arm_features = []
        for i in range(n_arms):
            # Combine context with arm-specific features
            arm_specific = np.random.randn(self.d - len(context))
            combined = np.concatenate([context, arm_specific])
            arm_features.append(combined)
        return arm_features
    
    def pull_arm(self, arm_idx, context_features):
        """Pull arm and return reward"""
        x = context_features[arm_idx]
        expected_reward = np.dot(self.theta_star, x)
        noise = np.random.normal(0, self.noise_std)
        return expected_reward + noise
    
    def get_optimal_reward(self, context_features):
        """Get optimal expected reward for current context"""
        rewards = [np.dot(self.theta_star, x) for x in context_features]
        return max(rewards)
    
    def get_regret(self, chosen_arms, rewards, optimal_rewards):
        """Calculate cumulative regret"""
        cumulative_optimal = np.cumsum(optimal_rewards)
        cumulative_rewards = np.cumsum(rewards)
        return cumulative_optimal - cumulative_rewards

def run_contextual_bandit_experiment(env, algorithm, T=1000, n_arms=10):
    """Run contextual bandit experiment"""
    chosen_arms = []
    rewards = []
    optimal_rewards = []
    
    for t in range(T):
        # Generate context
        context = env.generate_context()
        
        # Generate arm features for current context
        context_features = env.generate_arm_features(context, n_arms)
        
        # Select arm
        arm_idx = algorithm.select_arm(context_features)
        
        # Pull arm and observe reward
        reward = env.pull_arm(arm_idx, context_features)
        
        # Get optimal reward for comparison
        optimal_reward = env.get_optimal_reward(context_features)
        
        # Update algorithm
        algorithm.update(arm_idx, reward, context_features)
        
        chosen_arms.append(arm_idx)
        rewards.append(reward)
        optimal_rewards.append(optimal_reward)
    
    return chosen_arms, rewards, optimal_rewards
```

### Algorithm Comparison

```python
def compare_contextual_algorithms(env, algorithms, T=1000, n_arms=10, n_runs=50):
    """Compare different contextual bandit algorithms"""
    results = {}
    
    for name, algorithm_class in algorithms.items():
        regrets = []
        for run in range(n_runs):
            # Create fresh algorithm instance
            algorithm = algorithm_class(env.d)
            
            # Run experiment
            chosen_arms, rewards, optimal_rewards = run_contextual_bandit_experiment(
                env, algorithm, T, n_arms
            )
            
            # Calculate regret
            regret = env.get_regret(chosen_arms, rewards, optimal_rewards)
            regrets.append(regret)
        
        results[name] = np.mean(regrets, axis=0)
    
    return results

# Example usage
d = 10
theta_star = np.random.randn(d)
theta_star = theta_star / np.linalg.norm(theta_star)  # Normalize

# Context generator (stochastic contexts)
def context_generator():
    return np.random.randn(5)  # 5-dimensional context

env = ContextualBanditEnvironment(theta_star, context_generator)

algorithms = {
    'Contextual UCB': lambda d: ContextualUCB(d, alpha=1.0),
    'Contextual TS': lambda d: ContextualThompsonSampling(d, sigma=1.0),
    'Neural Bandit': lambda d: NeuralContextualBandit(d, num_arms=10)
}

results = compare_contextual_algorithms(env, algorithms)
```

### Visualization

```python
def plot_contextual_bandit_results(results, T):
    """Plot regret comparison for contextual bandit algorithms"""
    plt.figure(figsize=(12, 8))
    
    # Plot cumulative regret
    plt.subplot(2, 1, 1)
    for name, regret in results.items():
        plt.plot(range(1, T+1), regret, label=name, linewidth=2)
    
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Regret')
    plt.title('Contextual Bandit Algorithm Comparison')
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

plot_contextual_bandit_results(results, 1000)
```

## Summary

Contextual bandits provide a powerful framework for adaptive decision-making in dynamic environments. Key insights include:

1. **Context Adaptation**: Algorithms adapt to changing contexts and user characteristics
2. **Personalization**: Decisions are tailored to specific contexts and users
3. **Theoretical Guarantees**: Sublinear regret bounds with context-dependent analysis
4. **Practical Algorithms**: CUCB, CTS, and neural bandits for different scenarios
5. **Wide Applicability**: From online advertising to clinical trials

Contextual bandits bridge the gap between classical bandits and full reinforcement learning, providing both theoretical guarantees and practical effectiveness for real-world applications.

## Further Reading

- **Contextual Bandits Survey**: Comprehensive survey by Ambuj Tewari
- **Linear Contextual Bandits**: Theoretical foundations and algorithms
- **Neural Contextual Bandits**: Deep learning approaches
- **Online Learning Resources**: Stanford CS234, UC Berkeley CS285

---

**Note**: This guide covers the fundamentals of contextual bandits. For more advanced topics, see the sections on Neural Bandits, Federated Bandits, and Constrained Contextual Bandits.
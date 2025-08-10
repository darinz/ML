# Applications and Use Cases

## Introduction

Multi-armed bandits have found widespread applications across diverse domains, from online advertising to healthcare. This guide explores the major application areas where bandit algorithms provide significant value, along with practical implementations and real-world considerations.

## From Pure Exploration to Real-World Applications

We've now explored **Best Arm Identification (BAI)** - a fundamental shift in the multi-armed bandit paradigm from cumulative reward maximization to pure exploration. We've seen how algorithms like Successive Elimination, Racing, LUCB, and Sequential Halving focus exclusively on identifying the best arm with high confidence, regardless of the cumulative reward achieved during the learning process.

However, while understanding BAI algorithms is valuable, **the true impact** of multi-armed bandits lies in their real-world applications. Consider the algorithms we've learned - from classical bandits to linear and contextual bandits, and now best arm identification - these theoretical frameworks become powerful when applied to solve actual problems in advertising, healthcare, e-commerce, and beyond.

This motivates our exploration of **applications and use cases** - the practical implementation of bandit algorithms across diverse domains. We'll see how bandits optimize ad selection and bidding in online advertising, how they enable personalized recommendations in e-commerce and content platforms, how they improve clinical trials and drug discovery in healthcare, how they optimize pricing strategies in dynamic markets, and how they enhance A/B testing and algorithm selection processes.

The transition from best arm identification to applications represents the bridge from pure exploration to practical impact - taking our understanding of how to identify optimal actions and applying it to real-world scenarios where intelligent decision-making under uncertainty provides significant value.

In this section, we'll explore applications and use cases, understanding how bandit algorithms solve real-world problems and create value across diverse domains.

## Online Advertising

### Ad Selection

**Problem:** Choose the best ad creative from multiple options to maximize click-through rate (CTR).

**Bandit Approach:**
- **Arms**: Different ad creatives
- **Reward**: Click (1) or no click (0)
- **Context**: User demographics, browsing history, device type

**Implementation:**
```python
import numpy as np
from contextual_bandits import ContextualUCB

class AdSelectionBandit:
    def __init__(self, n_ads, user_feature_dim):
        self.bandit = ContextualUCB(user_feature_dim)
        self.ad_features = self._extract_ad_features(n_ads)
        
    def select_ad(self, user_context):
        """Select ad based on user context"""
        contextual_features = self._combine_features(user_context, self.ad_features)
        ad_idx = self.bandit.select_arm(contextual_features)
        return ad_idx
    
    def update(self, ad_idx, user_context, click):
        """Update with click feedback"""
        contextual_features = self._combine_features(user_context, self.ad_features)
        self.bandit.update(ad_idx, click, contextual_features)
    
    def _extract_ad_features(self, n_ads):
        """Extract features for each ad"""
        features = []
        for i in range(n_ads):
            # Ad-specific features: category, color, text length, etc.
            ad_features = [
                i % 5,  # Category
                (i // 5) % 3,  # Color scheme
                np.random.randint(10, 100),  # Text length
                np.random.choice([0, 1]),  # Has image
                np.random.choice([0, 1])   # Has video
            ]
            features.append(ad_features)
        return np.array(features)
    
    def _combine_features(self, user_context, ad_features):
        """Combine user and ad features"""
        combined_features = []
        for ad_feat in ad_features:
            # Simple concatenation
            combined = np.concatenate([user_context, ad_feat])
            combined_features.append(combined)
        return combined_features
```

### Bidding Optimization

**Problem:** Learn optimal bid amounts in real-time bidding auctions.

**Bandit Approach:**
- **Arms**: Different bid amounts
- **Reward**: Profit (revenue - cost) if win, 0 if lose
- **Context**: User profile, ad slot, time of day

**Implementation:**
```python
class BiddingBandit:
    def __init__(self, n_bid_levels, user_feature_dim):
        self.bandit = ContextualThompsonSampling(user_feature_dim)
        self.bid_levels = np.linspace(0.1, 10.0, n_bid_levels)
        
    def select_bid(self, user_context, reserve_price):
        """Select bid amount based on user context"""
        # Create contextual features for each bid level
        contextual_features = []
        for bid in self.bid_levels:
            # Combine user context with bid information
            bid_features = [bid, reserve_price, bid/reserve_price]
            combined = np.concatenate([user_context, bid_features])
            contextual_features.append(combined)
        
        bid_idx = self.bandit.select_arm(contextual_features)
        return self.bid_levels[bid_idx]
    
    def update(self, bid_idx, user_context, reserve_price, won, revenue, cost):
        """Update with auction outcome"""
        reward = revenue - cost if won else 0
        
        contextual_features = []
        for bid in self.bid_levels:
            bid_features = [bid, reserve_price, bid/reserve_price]
            combined = np.concatenate([user_context, bid_features])
            contextual_features.append(combined)
        
        self.bandit.update(bid_idx, reward, contextual_features)
```

## Recommendation Systems

### Content Recommendation

**Problem:** Recommend content (articles, videos, music) based on user preferences.

**Bandit Approach:**
- **Arms**: Different content items
- **Reward**: User engagement (view time, likes, shares)
- **Context**: User profile, content features, time of day

**Implementation:**
```python
class ContentRecommender:
    def __init__(self, n_items, user_feature_dim):
        self.bandit = NeuralContextualBandit(user_feature_dim, num_arms=n_items)
        self.item_features = self._extract_item_features(n_items)
        
    def recommend(self, user_context, n_recommendations=5):
        """Recommend content items"""
        contextual_features = self._create_contextual_features(user_context)
        
        # Get top-k recommendations
        recommendations = []
        for _ in range(n_recommendations):
            item_idx = self.bandit.select_arm(contextual_features)
            recommendations.append(item_idx)
        
        return recommendations
    
    def update(self, item_idx, user_context, engagement):
        """Update with user engagement"""
        contextual_features = self._create_contextual_features(user_context)
        self.bandit.update(item_idx, engagement, contextual_features)
    
    def _extract_item_features(self, n_items):
        """Extract features for each content item"""
        features = []
        for i in range(n_items):
            item_features = [
                i % 10,  # Category
                np.random.randint(100, 10000),  # Length
                np.random.choice([0, 1]),  # Has image
                np.random.choice([0, 1]),  # Has video
                np.random.uniform(0, 5),  # Average rating
                np.random.randint(0, 1000)  # Popularity
            ]
            features.append(item_features)
        return np.array(features)
    
    def _create_contextual_features(self, user_context):
        """Create contextual features for all items"""
        contextual_features = []
        for item_feat in self.item_features:
            # Combine user context with item features
            combined = np.concatenate([user_context, item_feat])
            contextual_features.append(combined)
        return contextual_features
```

### Product Recommendations

**Problem:** Recommend products in e-commerce based on user behavior.

**Bandit Approach:**
- **Arms**: Different products
- **Reward**: Purchase (1) or no purchase (0)
- **Context**: User history, product features, price sensitivity

**Implementation:**
```python
class ProductRecommender:
    def __init__(self, n_products, user_feature_dim):
        self.bandit = LinUCB(user_feature_dim)
        self.product_features = self._extract_product_features(n_products)
        
    def recommend_products(self, user_context, n_recommendations=10):
        """Recommend products based on user context"""
        contextual_features = self._create_contextual_features(user_context)
        
        # Get recommendations using bandit
        recommendations = []
        for _ in range(n_recommendations):
            product_idx = self.bandit.select_arm(contextual_features)
            recommendations.append(product_idx)
        
        return recommendations
    
    def update(self, product_idx, user_context, purchased):
        """Update with purchase feedback"""
        contextual_features = self._create_contextual_features(user_context)
        self.bandit.update(product_idx, purchased, contextual_features)
    
    def _extract_product_features(self, n_products):
        """Extract features for each product"""
        features = []
        for i in range(n_products):
            product_features = [
                i % 20,  # Category
                np.random.uniform(10, 1000),  # Price
                np.random.uniform(0, 5),  # Rating
                np.random.randint(0, 1000),  # Sales count
                np.random.choice([0, 1]),  # In stock
                np.random.choice([0, 1])   # On sale
            ]
            features.append(product_features)
        return np.array(features)
```

## Clinical Trials

### Adaptive Clinical Trials

**Problem:** Allocate patients to treatment arms while learning effectiveness.

**Bandit Approach:**
- **Arms**: Different treatments
- **Reward**: Patient outcome (improvement, survival time)
- **Context**: Patient demographics, medical history

**Implementation:**
```python
class AdaptiveClinicalTrial:
    def __init__(self, n_treatments, patient_feature_dim):
        self.bandit = ContextualThompsonSampling(patient_feature_dim)
        self.treatment_features = self._extract_treatment_features(n_treatments)
        
    def assign_treatment(self, patient_features):
        """Assign treatment based on patient characteristics"""
        contextual_features = self._create_contextual_features(patient_features)
        treatment_idx = self.bandit.select_arm(contextual_features)
        return treatment_idx
    
    def update(self, treatment_idx, patient_features, outcome):
        """Update with treatment outcome"""
        contextual_features = self._create_contextual_features(patient_features)
        self.bandit.update(treatment_idx, outcome, contextual_features)
    
    def get_best_treatment(self, patient_features):
        """Get the best treatment for a patient"""
        contextual_features = self._create_contextual_features(patient_features)
        return self.bandit.select_arm(contextual_features)
    
    def _extract_treatment_features(self, n_treatments):
        """Extract features for each treatment"""
        features = []
        for i in range(n_treatments):
            treatment_features = [
                i % 3,  # Treatment type
                np.random.uniform(0.1, 2.0),  # Dosage
                np.random.choice([0, 1]),  # Oral/IV
                np.random.uniform(0, 24),  # Frequency
                np.random.choice([0, 1])   # Experimental/Control
            ]
            features.append(treatment_features)
        return np.array(features)
```

### Drug Discovery

**Problem:** Screen multiple drug candidates efficiently.

**Bandit Approach:**
- **Arms**: Different drug compounds
- **Reward**: Efficacy score (0-1)
- **Context**: Compound features, target properties

**Implementation:**
```python
class DrugDiscoveryBandit:
    def __init__(self, n_compounds, target_feature_dim):
        self.bandit = LinUCB(target_feature_dim)
        self.compound_features = self._extract_compound_features(n_compounds)
        
    def select_compound(self, target_features):
        """Select compound to test"""
        contextual_features = self._create_contextual_features(target_features)
        compound_idx = self.bandit.select_arm(contextual_features)
        return compound_idx
    
    def update(self, compound_idx, target_features, efficacy):
        """Update with efficacy results"""
        contextual_features = self._create_contextual_features(target_features)
        self.bandit.update(compound_idx, efficacy, contextual_features)
    
    def _extract_compound_features(self, n_compounds):
        """Extract features for each compound"""
        features = []
        for i in range(n_compounds):
            compound_features = [
                i % 10,  # Chemical class
                np.random.uniform(100, 1000),  # Molecular weight
                np.random.uniform(0, 10),  # LogP
                np.random.uniform(0, 20),  # H-bond donors
                np.random.uniform(0, 20),  # H-bond acceptors
                np.random.uniform(0, 10)   # Rotatable bonds
            ]
            features.append(compound_features)
        return np.array(features)
```

## Dynamic Pricing

### Price Optimization

**Problem:** Set optimal prices to maximize revenue.

**Bandit Approach:**
- **Arms**: Different price levels
- **Reward**: Revenue (price × demand)
- **Context**: Customer features, market conditions

**Implementation:**
```python
class DynamicPricer:
    def __init__(self, n_price_levels, customer_feature_dim):
        self.bandit = ContextualUCB(customer_feature_dim)
        self.price_levels = np.linspace(10, 100, n_price_levels)
        
    def set_price(self, customer_features, market_conditions):
        """Set price based on customer and market"""
        contextual_features = self._create_contextual_features(customer_features, market_conditions)
        price_idx = self.bandit.select_arm(contextual_features)
        return self.price_levels[price_idx]
    
    def update(self, price_idx, customer_features, market_conditions, demand, revenue):
        """Update with demand and revenue data"""
        contextual_features = self._create_contextual_features(customer_features, market_conditions)
        self.bandit.update(price_idx, revenue, contextual_features)
    
    def _create_contextual_features(self, customer_features, market_conditions):
        """Create contextual features for each price level"""
        contextual_features = []
        for price in self.price_levels:
            # Combine customer, market, and price features
            price_features = [price, price/50.0, np.log(price)]  # Normalize price
            combined = np.concatenate([customer_features, market_conditions, price_features])
            contextual_features.append(combined)
        return contextual_features
```

### Revenue Management

**Problem:** Optimize pricing for perishable inventory (hotels, airlines).

**Bandit Approach:**
- **Arms**: Different pricing strategies
- **Reward**: Profit margin
- **Context**: Time to departure, occupancy, demand forecast

**Implementation:**
```python
class RevenueManager:
    def __init__(self, n_pricing_strategies, market_feature_dim):
        self.bandit = NeuralContextualBandit(market_feature_dim, num_arms=n_pricing_strategies)
        
    def set_pricing_strategy(self, market_features):
        """Select pricing strategy based on market conditions"""
        contextual_features = self._create_contextual_features(market_features)
        strategy_idx = self.bandit.select_arm(contextual_features)
        return strategy_idx
    
    def update(self, strategy_idx, market_features, profit_margin):
        """Update with profit margin data"""
        contextual_features = self._create_contextual_features(market_features)
        self.bandit.update(strategy_idx, profit_margin, contextual_features)
    
    def _create_contextual_features(self, market_features):
        """Create contextual features for each pricing strategy"""
        contextual_features = []
        for strategy in range(self.bandit.num_arms):
            # Combine market features with strategy encoding
            strategy_features = np.zeros(self.bandit.num_arms)
            strategy_features[strategy] = 1.0
            combined = np.concatenate([market_features, strategy_features])
            contextual_features.append(combined)
        return contextual_features
```

## A/B Testing

### Website Optimization

**Problem:** Test different website designs to maximize conversion rate.

**Bandit Approach:**
- **Arms**: Different website variants
- **Reward**: Conversion (1) or no conversion (0)
- **Context**: User segment, traffic source

**Implementation:**
```python
class ABTestBandit:
    def __init__(self, n_variants, user_feature_dim):
        self.bandit = SuccessiveElimination(n_variants, delta=0.05)
        self.variant_features = self._extract_variant_features(n_variants)
        
    def select_variant(self, user_context):
        """Select website variant to show"""
        # For BAI, we focus on identification rather than cumulative reward
        variant_idx = self.bandit.select_arm()
        return variant_idx
    
    def update(self, variant_idx, user_context, converted):
        """Update with conversion data"""
        self.bandit.update(variant_idx, converted)
    
    def get_best_variant(self):
        """Get the identified best variant"""
        return self.bandit.get_best_arm()
    
    def is_complete(self):
        """Check if A/B test is complete"""
        return self.bandit.is_complete()
    
    def _extract_variant_features(self, n_variants):
        """Extract features for each website variant"""
        features = []
        for i in range(n_variants):
            variant_features = [
                i % 3,  # Layout type
                (i // 3) % 2,  # Color scheme
                (i // 6) % 2,  # Button style
                np.random.randint(1, 10),  # Content length
                np.random.choice([0, 1])   # Has video
            ]
            features.append(variant_features)
        return np.array(features)
```

## Algorithm Selection

### Machine Learning Model Selection

**Problem:** Choose the best algorithm for a specific dataset.

**Bandit Approach:**
- **Arms**: Different ML algorithms
- **Reward**: Performance metric (accuracy, F1-score)
- **Context**: Dataset characteristics

**Implementation:**
```python
class AlgorithmSelector:
    def __init__(self, algorithms, dataset_feature_dim):
        self.algorithms = algorithms
        self.bandit = RacingAlgorithm(len(algorithms), delta=0.1)
        
    def select_algorithm(self, dataset_features):
        """Select algorithm to evaluate"""
        algorithm_idx = self.bandit.select_arm()
        return self.algorithms[algorithm_idx]
    
    def update(self, algorithm_idx, dataset_features, performance):
        """Update with algorithm performance"""
        self.bandit.update(algorithm_idx, performance)
    
    def get_best_algorithm(self):
        """Get the identified best algorithm"""
        return self.algorithms[self.bandit.get_best_arm()]
    
    def is_complete(self):
        """Check if algorithm selection is complete"""
        return self.bandit.is_complete()
```

## Implementation Examples

### Complete Application Framework

```python
import numpy as np
import matplotlib.pyplot as plt

class BanditApplication:
    def __init__(self, application_type, n_arms, feature_dim):
        self.application_type = application_type
        self.n_arms = n_arms
        self.feature_dim = feature_dim
        
        # Initialize appropriate bandit based on application
        if application_type == "ad_selection":
            self.bandit = AdSelectionBandit(n_arms, feature_dim)
        elif application_type == "recommendation":
            self.bandit = ContentRecommender(n_arms, feature_dim)
        elif application_type == "clinical_trial":
            self.bandit = AdaptiveClinicalTrial(n_arms, feature_dim)
        elif application_type == "dynamic_pricing":
            self.bandit = DynamicPricer(n_arms, feature_dim)
        else:
            raise ValueError(f"Unknown application type: {application_type}")
    
    def run_experiment(self, n_steps=1000):
        """Run bandit experiment"""
        rewards = []
        regrets = []
        
        for step in range(n_steps):
            # Generate context
            context = self._generate_context()
            
            # Select action
            action = self._select_action(context)
            
            # Get reward
            reward = self._get_reward(action, context)
            
            # Update bandit
            self._update_bandit(action, context, reward)
            
            rewards.append(reward)
            
            # Calculate regret (if optimal action is known)
            optimal_reward = self._get_optimal_reward(context)
            regret = optimal_reward - reward
            regrets.append(regret)
        
        return rewards, regrets
    
    def _generate_context(self):
        """Generate context based on application type"""
        if self.application_type == "ad_selection":
            return self._generate_user_context()
        elif self.application_type == "recommendation":
            return self._generate_user_context()
        elif self.application_type == "clinical_trial":
            return self._generate_patient_context()
        elif self.application_type == "dynamic_pricing":
            return self._generate_customer_context()
    
    def _generate_user_context(self):
        """Generate user context for advertising/recommendation"""
        return np.random.randn(self.feature_dim)
    
    def _generate_patient_context(self):
        """Generate patient context for clinical trials"""
        return np.random.randn(self.feature_dim)
    
    def _generate_customer_context(self):
        """Generate customer context for pricing"""
        return np.random.randn(self.feature_dim)
    
    def _select_action(self, context):
        """Select action based on application type"""
        if self.application_type == "ad_selection":
            return self.bandit.select_ad(context)
        elif self.application_type == "recommendation":
            return self.bandit.recommend(context)[0]  # Return first recommendation
        elif self.application_type == "clinical_trial":
            return self.bandit.assign_treatment(context)
        elif self.application_type == "dynamic_pricing":
            return self.bandit.set_price(context, {})
    
    def _get_reward(self, action, context):
        """Get reward based on application type"""
        # Simulate reward generation
        base_reward = 0.3 + 0.4 * np.dot(context[:3], [1, 0.5, 0.2])
        noise = np.random.normal(0, 0.1)
        return np.clip(base_reward + noise, 0, 1)
    
    def _update_bandit(self, action, context, reward):
        """Update bandit based on application type"""
        if self.application_type == "ad_selection":
            self.bandit.update(action, context, reward)
        elif self.application_type == "recommendation":
            self.bandit.update(action, context, reward)
        elif self.application_type == "clinical_trial":
            self.bandit.update(action, context, reward)
        elif self.application_type == "dynamic_pricing":
            self.bandit.update(action, context, {}, 1, reward)  # Simplified update
    
    def _get_optimal_reward(self, context):
        """Get optimal reward for regret calculation"""
        # Simplified optimal reward calculation
        return 0.7 + 0.3 * np.dot(context[:3], [1, 0.5, 0.2])
```

### Performance Comparison

```python
def compare_applications():
    """Compare different bandit applications"""
    applications = [
        ("ad_selection", 10, 5),
        ("recommendation", 20, 8),
        ("clinical_trial", 5, 6),
        ("dynamic_pricing", 15, 4)
    ]
    
    results = {}
    
    for app_type, n_arms, feature_dim in applications:
        print(f"Running {app_type} experiment...")
        
        app = BanditApplication(app_type, n_arms, feature_dim)
        rewards, regrets = app.run_experiment(n_steps=500)
        
        results[app_type] = {
            'avg_reward': np.mean(rewards),
            'cumulative_regret': np.sum(regrets),
            'final_regret': np.mean(regrets[-100:])  # Last 100 steps
        }
    
    return results

# Run comparison
results = compare_applications()

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Average reward comparison
app_names = list(results.keys())
avg_rewards = [results[name]['avg_reward'] for name in app_names]
ax1.bar(app_names, avg_rewards)
ax1.set_ylabel('Average Reward')
ax1.set_title('Application Performance Comparison')

# Cumulative regret comparison
cumulative_regrets = [results[name]['cumulative_regret'] for name in app_names]
ax2.bar(app_names, cumulative_regrets)
ax2.set_ylabel('Cumulative Regret')
ax2.set_title('Application Regret Comparison')

plt.tight_layout()
plt.show()
```

## Summary

Multi-armed bandits provide powerful solutions for a wide range of real-world applications:

1. **Online Advertising**: Optimize ad selection and bidding strategies
2. **Recommendation Systems**: Personalize content and product recommendations
3. **Clinical Trials**: Adaptively allocate treatments while learning effectiveness
4. **Dynamic Pricing**: Optimize prices based on demand and customer characteristics
5. **A/B Testing**: Efficiently identify the best variant
6. **Algorithm Selection**: Choose the best ML algorithm for specific tasks

Each application domain has unique characteristics that influence algorithm choice and implementation details. Understanding these domain-specific considerations is crucial for successful bandit deployments.

## Further Reading

- **Domain-specific papers**: Research papers for each application area
- **Industry case studies**: Real-world deployment examples
- **Implementation guides**: Best practices for production systems
- **Evaluation frameworks**: Metrics and methodologies for bandit evaluation

---

**Note**: This guide covers the major application areas for multi-armed bandits. For more specialized applications, see the domain-specific literature and case studies.

## From Theoretical Understanding to Hands-On Implementation

We've now explored **applications and use cases** - the practical implementation of bandit algorithms across diverse domains. We've seen how bandits optimize ad selection and bidding in online advertising, how they enable personalized recommendations in e-commerce and content platforms, how they improve clinical trials and drug discovery in healthcare, how they optimize pricing strategies in dynamic markets, and how they enhance A/B testing and algorithm selection processes.

However, while understanding applications is valuable, **true mastery** comes from hands-on implementation. Consider building a recommendation system that adapts to user preferences, or implementing an A/B testing framework that efficiently identifies the best variant - these require not just theoretical knowledge but practical skills in implementing bandit algorithms, handling real data, and optimizing performance.

This motivates our exploration of **hands-on coding** - the practical implementation of all the bandit concepts we've learned. We'll put our theoretical knowledge into practice by implementing classical bandit algorithms like epsilon-greedy, UCB, and Thompson sampling, building linear and contextual bandits, applying best arm identification techniques, and developing practical applications for recommendation systems, A/B testing, and other real-world scenarios.

The transition from applications to hands-on coding represents the bridge from understanding to implementation - taking our knowledge of how bandits solve real-world problems and turning it into practical tools for building intelligent decision-making systems.

In the next section, we'll implement complete bandit systems, experiment with different algorithms and applications, and develop the practical skills needed for real-world deployment of multi-armed bandits.

---

**Previous: [Best Arm Identification](04_best_arm_identification.md)** - Learn pure exploration algorithms for identifying optimal actions.

**Next: [Hands-on Coding](06_hands-on_coding.md)** - Implement complete bandit algorithms and applications with practical examples. 
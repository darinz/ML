# Policy Optimization

This guide provides a comprehensive overview of policy optimization methods for reinforcement learning from human feedback (RLHF) systems. We'll explore policy gradient methods, proximal policy optimization (PPO), trust region policy optimization (TRPO), and their applications to language model training.

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Policy Gradient Methods](#policy-gradient-methods)
- [Proximal Policy Optimization (PPO)](#proximal-policy-optimization-ppo)
- [Trust Region Policy Optimization (TRPO)](#trust-region-policy-optimization-trpo)
- [Language Model Specifics](#language-model-specifics)
- [Implementation Examples](#implementation-examples)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)

## Overview

Policy optimization is the core component of RLHF that updates the language model policy to maximize expected reward from human feedback. Unlike traditional supervised learning that minimizes prediction error, policy optimization maximizes reward signals from human evaluators or learned reward models.

### Key Concepts

**1. Policy Gradient**: Direct optimization of policy parameters using gradient ascent
**2. Trust Region**: Constraining policy updates to prevent catastrophic changes
**3. Advantage Estimation**: Estimating relative value of actions for stable updates
**4. KL Control**: Preventing policy from deviating too far from reference model

### Problem Formulation

**Objective**: Maximize expected reward:
```math
J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s, a)]
```

Where:
- $`\pi_\theta`$: Language model policy with parameters $`\theta`$
- $`\rho_\pi`$: State distribution under policy $`\pi`$
- $`R(s, a)`$: Reward function (human feedback or learned reward model)

**Language Model Context**:
- **State**: Current conversation context or prompt $`s_t`$
- **Action**: Next token to generate $`a_t`$ from vocabulary $`\mathcal{V}`$
- **Policy**: $`\pi_\theta(a_t|s_t) = P(y_t | x, y_1, y_2, \ldots, y_{t-1})`$

## Mathematical Foundations

### Policy Gradient Theorem

**Theorem**: For any differentiable policy $`\pi_\theta`$, the gradient of the objective is:
```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s,a) \nabla_\theta \log \pi_\theta(a|s)]
```

**Proof Sketch**:
```math
\begin{align}
\nabla_\theta J(\theta) &= \nabla_\theta \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s,a)] \\
&= \nabla_\theta \int \rho_\pi(s) \pi_\theta(a|s) R(s,a) \, ds \, da \\
&= \int \rho_\pi(s) \nabla_\theta \pi_\theta(a|s) R(s,a) \, ds \, da \\
&= \int \rho_\pi(s) \pi_\theta(a|s) \frac{\nabla_\theta \pi_\theta(a|s)}{\pi_\theta(a|s)} R(s,a) \, ds \, da \\
&= \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s,a) \nabla_\theta \log \pi_\theta(a|s)]
\end{align}
```

### Advantage Function

**Definition**: The advantage function measures how much better an action is compared to the average:
```math
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
```

Where:
- $`Q^\pi(s, a)`$: Action-value function
- $`V^\pi(s)`$: State-value function

**Properties**:
- $`\mathbb{E}_{a \sim \pi} [A^\pi(s, a)] = 0`$: Average advantage is zero
- $`A^\pi(s, a) > 0`$: Action is better than average
- $`A^\pi(s, a) < 0`$: Action is worse than average

### Policy Gradient with Advantage

**Improved Policy Gradient**:
```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [A^\pi(s, a) \nabla_\theta \log \pi_\theta(a|s)]
```

**Benefits**:
- **Variance Reduction**: Advantage function reduces gradient variance
- **Stable Updates**: More stable policy updates
- **Better Convergence**: Faster and more reliable convergence

## Policy Gradient Methods

### REINFORCE Algorithm

**Basic REINFORCE**:
```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [R(s,a) \nabla_\theta \log \pi_\theta(a|s)]
```

**Implementation**:
```python
def reinforce_loss(log_probs, rewards):
    """
    Compute REINFORCE loss for language generation
    
    Args:
        log_probs: Log probabilities of generated tokens [batch_size, seq_len]
        rewards: Rewards for each sequence [batch_size]
    
    Returns:
        loss: Policy gradient loss
    """
    # Compute log probability of each sequence
    seq_log_probs = log_probs.sum(dim=1)  # [batch_size]
    
    # Compute loss (negative because we want to maximize reward)
    loss = -(seq_log_probs * rewards).mean()
    
    return loss

class REINFORCETrainer:
    def __init__(self, model, reward_model, learning_rate=1e-5):
        self.model = model
        self.reward_model = reward_model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def train_step(self, prompts, responses, rewards):
        """
        Perform one training step with REINFORCE
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            rewards: Rewards for each response
        """
        # Tokenize inputs
        inputs = self.tokenizer(prompts + responses, return_tensors='pt', padding=True)
        
        # Get log probabilities
        outputs = self.model(**inputs)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # Compute REINFORCE loss
        loss = reinforce_loss(log_probs, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### REINFORCE with Baseline

**Baseline Subtraction**:
```math
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho_\pi, a \sim \pi_\theta} [(R(s,a) - b(s)) \nabla_\theta \log \pi_\theta(a|s)]
```

Where $`b(s)`$ is a baseline function.

**Implementation**:
```python
def reinforce_with_baseline(log_probs, rewards, baseline):
    """
    Compute REINFORCE loss with baseline subtraction
    
    Args:
        log_probs: Log probabilities of generated tokens
        rewards: Rewards for each sequence
        baseline: Baseline values for each sequence
    
    Returns:
        loss: Policy gradient loss with baseline
    """
    seq_log_probs = log_probs.sum(dim=1)
    
    # Subtract baseline from rewards
    advantage = rewards - baseline
    
    # Compute loss
    loss = -(seq_log_probs * advantage).mean()
    
    return loss

class BaselineEstimator:
    def __init__(self, model):
        self.model = model
    
    def estimate_baseline(self, states):
        """
        Estimate baseline values for given states
        
        Args:
            states: State representations
        
        Returns:
            baseline: Estimated baseline values
        """
        with torch.no_grad():
            baseline = self.model(states)
        
        return baseline
```

### Actor-Critic Methods

**Actor-Critic Architecture**:
- **Actor**: Policy network $`\pi_\theta(a|s)`$
- **Critic**: Value network $`V_\phi(s)`$

**Advantage Estimation**:
```math
A_t = R_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
```

**Implementation**:
```python
class ActorCriticTrainer:
    def __init__(self, actor, critic, learning_rate=1e-4):
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=learning_rate)
    
    def train_step(self, states, actions, rewards, next_states):
        """
        Perform one training step with actor-critic
        
        Args:
            states: Current states
            actions: Taken actions
            rewards: Received rewards
            next_states: Next states
        """
        # Compute advantages
        current_values = self.critic(states)
        next_values = self.critic(next_states)
        advantages = rewards + 0.99 * next_values - current_values
        
        # Actor loss (policy gradient)
        log_probs = self.actor.get_log_probs(states, actions)
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # Critic loss (value function regression)
        critic_loss = torch.nn.functional.mse_loss(current_values, rewards + 0.99 * next_values)
        
        # Update networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
```

## Proximal Policy Optimization (PPO)

### PPO-Clip Objective

**PPO-Clip Loss**:
```math
L(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]
```

Where:
- $`r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}`$: Probability ratio
- $`A_t`$: Advantage estimate
- $`\epsilon`$: Clipping parameter (typically 0.2)

**Intuition**:
- **Ratio Clipping**: Prevents large policy updates
- **Conservative Updates**: Ensures policy doesn't change too much
- **Stable Training**: More stable than vanilla policy gradients

### PPO Implementation

```python
class PPOTrainer:
    def __init__(self, model, ref_model, reward_model, tokenizer, 
                 learning_rate=1e-5, clip_epsilon=0.2, kl_coef=0.1):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
    
    def compute_advantages(self, rewards, values, gamma=0.99):
        """
        Compute advantages using GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            gamma: Discount factor
        
        Returns:
            advantages: Computed advantages
        """
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * 0.95 * gae
            advantages[t] = gae
        
        return advantages
    
    def ppo_loss(self, log_probs, old_log_probs, advantages, rewards, kl_div):
        """
        Compute PPO loss with KL penalty
        
        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities
            advantages: Advantage estimates
            rewards: Rewards
            kl_div: KL divergence from reference model
        
        Returns:
            loss: PPO loss
        """
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # PPO-clip loss
        clip_adv = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        ppo_loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        # Add KL penalty
        kl_penalty = self.kl_coef * kl_div
        
        return ppo_loss + kl_penalty
    
    def train_step(self, prompts, responses, rewards):
        """
        Perform one PPO training step
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            rewards: Rewards for responses
        """
        # Tokenize inputs
        inputs = self.tokenizer(prompts + responses, return_tensors='pt', padding=True)
        
        # Get current policy log probabilities
        outputs = self.model(**inputs)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # Get old policy log probabilities (from reference model)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            old_log_probs = ref_outputs.logits.log_softmax(dim=-1)
        
        # Compute KL divergence
        kl_div = torch.nn.functional.kl_div(
            log_probs, old_log_probs, reduction='batchmean'
        )
        
        # Compute advantages (simplified - in practice use GAE)
        advantages = rewards - rewards.mean()
        
        # Compute PPO loss
        loss = self.ppo_loss(log_probs, old_log_probs, advantages, rewards, kl_div)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

### PPO for Language Models

**Token-level PPO**:
```python
class TokenLevelPPO:
    def __init__(self, model, ref_model, reward_model, tokenizer):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
    
    def token_level_ppo_loss(self, prompt_ids, response_ids, rewards):
        """
        Compute PPO loss at token level
        
        Args:
            prompt_ids: Input prompt token IDs
            response_ids: Generated response token IDs
            rewards: Rewards for each token
        
        Returns:
            loss: Token-level PPO loss
        """
        # Get log probabilities for each token
        inputs = torch.cat([prompt_ids, response_ids], dim=1)
        outputs = self.model(inputs)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # Get reference log probabilities
        with torch.no_grad():
            ref_outputs = self.ref_model(inputs)
            ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)
        
        # Compute ratio for each token
        ratio = torch.exp(log_probs - ref_log_probs)
        
        # Apply PPO clipping
        clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        advantages = rewards.unsqueeze(-1)  # [batch_size, seq_len]
        
        ppo_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
        
        return ppo_loss
```

**Sequence-level PPO**:
```python
class SequenceLevelPPO:
    def __init__(self, model, ref_model, reward_model, tokenizer):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
    
    def sequence_level_ppo_loss(self, prompt_ids, response_ids, rewards):
        """
        Compute PPO loss at sequence level
        
        Args:
            prompt_ids: Input prompt token IDs
            response_ids: Generated response token IDs
            rewards: Rewards for each sequence
        
        Returns:
            loss: Sequence-level PPO loss
        """
        # Get sequence log probabilities
        inputs = torch.cat([prompt_ids, response_ids], dim=1)
        outputs = self.model(inputs)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # Sum log probabilities over sequence
        seq_log_probs = log_probs.sum(dim=1)  # [batch_size]
        
        # Get reference log probabilities
        with torch.no_grad():
            ref_outputs = self.ref_model(inputs)
            ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)
            ref_seq_log_probs = ref_log_probs.sum(dim=1)
        
        # Compute ratio
        ratio = torch.exp(seq_log_probs - ref_seq_log_probs)
        
        # Apply PPO clipping
        clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        advantages = rewards
        
        ppo_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
        
        return ppo_loss
```

## Trust Region Policy Optimization (TRPO)

### TRPO Objective

**TRPO Problem**:
```math
\max_\theta \mathbb{E}_t \left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t\right]
\text{ subject to } \mathbb{E}_t [\text{KL}(\pi_{\theta_{old}} \| \pi_\theta)] \leq \delta
```

Where:
- $`\delta`$: Trust region constraint threshold
- $`\text{KL}(\pi_{\theta_{old}} \| \pi_\theta)`$: KL divergence between old and new policies

### TRPO Implementation

```python
class TRPOTrainer:
    def __init__(self, model, ref_model, reward_model, tokenizer, 
                 max_kl=0.01, damping=0.1):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.max_kl = max_kl
        self.damping = damping
    
    def conjugate_gradient(self, states, actions, advantages, max_iter=10):
        """
        Solve linear system using conjugate gradient
        
        Args:
            states: State representations
            actions: Action representations
            advantages: Advantage estimates
            max_iter: Maximum iterations
        
        Returns:
            step: Policy update step
        """
        # Initialize
        x = torch.zeros_like(self.model.parameters())
        r = advantages  # Initial residual
        p = r.clone()
        
        for _ in range(max_iter):
            # Compute Ap (Fisher-vector product)
            Ap = self.fisher_vector_product(p, states, actions)
            
            # Compute step size
            alpha = torch.dot(r, r) / torch.dot(p, Ap)
            
            # Update x and residual
            x = x + alpha * p
            r_new = r - alpha * Ap
            
            # Check convergence
            if torch.norm(r_new) < 1e-8:
                break
            
            # Update search direction
            beta = torch.dot(r_new, r_new) / torch.dot(r, r)
            p = r_new + beta * p
            r = r_new
        
        return x
    
    def fisher_vector_product(self, v, states, actions):
        """
        Compute Fisher-vector product
        
        Args:
            v: Vector to multiply
            states: State representations
            actions: Action representations
        
        Returns:
            Fv: Fisher-vector product
        """
        # Compute KL divergence
        kl_div = self.compute_kl(states, actions)
        
        # Compute gradient of KL
        kl_grad = torch.autograd.grad(kl_div, self.model.parameters(), 
                                     create_graph=True)
        
        # Compute Fisher-vector product
        Fv = torch.autograd.grad(torch.dot(kl_grad, v), self.model.parameters())
        
        return torch.stack(Fv)
    
    def compute_kl(self, states, actions):
        """
        Compute KL divergence between current and reference policies
        
        Args:
            states: State representations
            actions: Action representations
        
        Returns:
            kl_div: KL divergence
        """
        # Get current policy log probabilities
        current_log_probs = self.model.get_log_probs(states, actions)
        
        # Get reference policy log probabilities
        with torch.no_grad():
            ref_log_probs = self.ref_model.get_log_probs(states, actions)
        
        # Compute KL divergence
        kl_div = torch.nn.functional.kl_div(
            current_log_probs, ref_log_probs, reduction='batchmean'
        )
        
        return kl_div
    
    def trpo_step(self, prompts, responses, rewards):
        """
        Perform one TRPO step
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            rewards: Rewards for responses
        """
        # Tokenize inputs
        inputs = self.tokenizer(prompts + responses, return_tensors='pt', padding=True)
        
        # Compute advantages
        advantages = rewards - rewards.mean()
        
        # Compute policy gradient
        log_probs = self.model.get_log_probs(inputs['input_ids'], inputs['attention_mask'])
        policy_loss = -(log_probs * advantages).mean()
        
        # Compute gradient
        grad = torch.autograd.grad(policy_loss, self.model.parameters())
        
        # Solve for step direction using conjugate gradient
        step = self.conjugate_gradient(inputs['input_ids'], inputs['attention_mask'], advantages)
        
        # Scale step to satisfy KL constraint
        kl_div = self.compute_kl(inputs['input_ids'], inputs['attention_mask'])
        scale = torch.sqrt(2 * self.max_kl / kl_div)
        step = step * torch.clamp(scale, max=1.0)
        
        # Apply step
        for param, step_param in zip(self.model.parameters(), step):
            param.data += step_param
        
        return policy_loss.item()
```

## Language Model Specifics

### Token-Level vs Sequence-Level Optimization

**Token-Level Optimization**:
- **Advantage**: Fine-grained control over each token
- **Challenge**: Sparse rewards at token level
- **Implementation**: Apply RL to each token generation step

**Sequence-Level Optimization**:
- **Advantage**: Natural reward structure
- **Challenge**: Credit assignment problem
- **Implementation**: Apply RL to complete sequences

### KL Divergence Control

**Purpose**: Prevent policy from deviating too far from reference model

**Implementation**:
```python
def kl_penalty_loss(policy_loss, kl_div, kl_coef=0.1, target_kl=0.01):
    """
    Compute loss with KL penalty
    
    Args:
        policy_loss: Main policy loss
        kl_div: KL divergence from reference model
        kl_coef: KL penalty coefficient
        target_kl: Target KL divergence
    
    Returns:
        total_loss: Combined loss
    """
    # Adaptive KL penalty
    if kl_div > 2 * target_kl:
        kl_coef *= 1.5
    elif kl_div < 0.5 * target_kl:
        kl_coef *= 0.5
    
    total_loss = policy_loss + kl_coef * kl_div
    
    return total_loss
```

### Entropy Regularization

**Purpose**: Encourage exploration and prevent premature convergence

**Implementation**:
```python
def entropy_regularized_loss(policy_loss, log_probs, entropy_coef=0.01):
    """
    Add entropy regularization to policy loss
    
    Args:
        policy_loss: Main policy loss
        log_probs: Policy log probabilities
        entropy_coef: Entropy coefficient
    
    Returns:
        total_loss: Loss with entropy regularization
    """
    # Compute entropy
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    
    # Add entropy regularization
    total_loss = policy_loss - entropy_coef * entropy
    
    return total_loss
```

## Implementation Examples

### Complete PPO Training Loop

```python
class PPOTrainingLoop:
    def __init__(self, model, ref_model, reward_model, tokenizer, 
                 learning_rate=1e-5, batch_size=32, ppo_epochs=4):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.ppo_epochs = ppo_epochs
    
    def generate_responses(self, prompts, max_length=100):
        """Generate responses using current policy"""
        responses = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def compute_rewards(self, prompts, responses):
        """Compute rewards using reward model"""
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            reward = self.reward_model.predict_reward(prompt, response)
            rewards.append(reward)
        
        return torch.tensor(rewards)
    
    def ppo_epoch(self, prompts, responses, rewards):
        """Perform one PPO epoch"""
        # Tokenize all data
        all_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(all_texts, return_tensors='pt', padding=True, truncation=True)
        
        # Get log probabilities
        outputs = self.model(**inputs)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # Get reference log probabilities
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)
        
        # Compute advantages
        advantages = rewards - rewards.mean()
        
        # PPO training
        for _ in range(self.ppo_epochs):
            # Compute ratio
            ratio = torch.exp(log_probs - ref_log_probs)
            
            # PPO-clip loss
            clip_ratio = torch.clamp(ratio, 0.8, 1.2)
            ppo_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
            
            # KL penalty
            kl_div = torch.nn.functional.kl_div(
                log_probs, ref_log_probs, reduction='batchmean'
            )
            total_loss = ppo_loss + 0.1 * kl_div
            
            # Update model
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
    def train(self, prompts, num_iterations=1000):
        """Main training loop"""
        for iteration in range(num_iterations):
            # Generate responses
            responses = self.generate_responses(prompts)
            
            # Compute rewards
            rewards = self.compute_rewards(prompts, responses)
            
            # PPO update
            self.ppo_epoch(prompts, responses, rewards)
            
            # Log progress
            if iteration % 100 == 0:
                avg_reward = rewards.mean().item()
                print(f"Iteration {iteration}, Average Reward: {avg_reward:.4f}")
```

### Advanced PPO with GAE

```python
class AdvancedPPOTrainer:
    def __init__(self, model, ref_model, reward_model, value_model, tokenizer):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.value_model = value_model
        self.tokenizer = tokenizer
    
    def compute_gae(self, rewards, values, gamma=0.99, lambda_=0.95):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            gamma: Discount factor
            lambda_: GAE parameter
        
        Returns:
            advantages: GAE advantages
            returns: Returns
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lambda_ * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def train_step(self, prompts, responses, rewards):
        """Advanced PPO training step with GAE"""
        # Tokenize inputs
        inputs = self.tokenizer(prompts + responses, return_tensors='pt', padding=True)
        
        # Get policy log probabilities
        outputs = self.model(**inputs)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # Get value estimates
        values = self.value_model(**inputs)
        
        # Get reference log probabilities
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)
        
        # Compute GAE advantages
        advantages, returns = self.compute_gae(rewards, values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO loss
        ratio = torch.exp(log_probs - ref_log_probs)
        clip_ratio = torch.clamp(ratio, 0.8, 1.2)
        ppo_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()
        
        # Value loss
        value_loss = torch.nn.functional.mse_loss(values, returns)
        
        # KL penalty
        kl_div = torch.nn.functional.kl_div(
            log_probs, ref_log_probs, reduction='batchmean'
        )
        
        # Total loss
        total_loss = ppo_loss + 0.5 * value_loss + 0.1 * kl_div
        
        return total_loss
```

## Advanced Techniques

### Multi-Objective PPO

```python
class MultiObjectivePPO:
    def __init__(self, model, ref_model, reward_models, tokenizer):
        self.model = model
        self.ref_model = ref_model
        self.reward_models = reward_models  # Dictionary of reward models
        self.tokenizer = tokenizer
    
    def multi_objective_loss(self, prompts, responses, objectives):
        """
        Compute multi-objective PPO loss
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            objectives: Dictionary of objective weights
        
        Returns:
            total_loss: Combined multi-objective loss
        """
        total_loss = 0
        
        for objective_name, weight in objectives.items():
            if objective_name in self.reward_models:
                reward_model = self.reward_models[objective_name]
                rewards = reward_model.compute_rewards(prompts, responses)
                
                # Compute PPO loss for this objective
                ppo_loss = self.compute_ppo_loss(prompts, responses, rewards)
                total_loss += weight * ppo_loss
        
        return total_loss
```

### Conservative Policy Iteration

```python
class ConservativePolicyIteration:
    def __init__(self, model, ref_model, reward_model, tokenizer, 
                 max_kl=0.01, damping=0.1):
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.max_kl = max_kl
        self.damping = damping
    
    def natural_policy_gradient(self, states, actions, advantages):
        """
        Compute natural policy gradient
        
        Args:
            states: State representations
            actions: Action representations
            advantages: Advantage estimates
        
        Returns:
            natural_grad: Natural policy gradient
        """
        # Compute Fisher information matrix
        kl_div = self.compute_kl(states, actions)
        fisher_grad = torch.autograd.grad(kl_div, self.model.parameters(), 
                                        create_graph=True)
        
        # Solve linear system for natural gradient
        natural_grad = self.solve_linear_system(fisher_grad, advantages)
        
        return natural_grad
    
    def solve_linear_system(self, fisher_grad, advantages):
        """Solve linear system using conjugate gradient"""
        # Implementation of conjugate gradient solver
        # This is a simplified version
        return fisher_grad  # Placeholder
```

## Best Practices

### 1. Hyperparameter Tuning

**Key Hyperparameters**:
- **Learning Rate**: Start with 1e-5, adjust based on convergence
- **Clip Epsilon**: 0.1-0.3 for PPO, 0.01-0.05 for TRPO
- **KL Coefficient**: 0.01-0.1 for KL penalty
- **Batch Size**: 32-128 for language models
- **PPO Epochs**: 4-10 epochs per batch

### 2. Training Stability

**Techniques**:
- **Gradient Clipping**: Prevent gradient explosion
- **Learning Rate Scheduling**: Reduce learning rate over time
- **Early Stopping**: Stop when KL divergence exceeds threshold
- **Reward Normalization**: Normalize rewards to zero mean, unit variance

### 3. Evaluation

**Metrics**:
- **Average Reward**: Monitor reward improvement
- **KL Divergence**: Track policy deviation from reference
- **Perplexity**: Monitor language model quality
- **Human Evaluation**: Validate with human judgments

### 4. Implementation Considerations

**Language Model Specifics**:
- **Token-Level vs Sequence-Level**: Choose based on reward structure
- **KL Control**: Essential for preventing catastrophic forgetting
- **Reward Scaling**: Important for stable training
- **Batch Processing**: Efficient handling of variable-length sequences

### 5. Debugging

**Common Issues**:
- **Reward Hacking**: Models optimizing for proxy objectives
- **Mode Collapse**: Policy converging to limited responses
- **Catastrophic Forgetting**: Losing pre-trained knowledge
- **Unstable Training**: Oscillating or diverging rewards

**Solutions**:
- **Regularization**: KL penalties and entropy regularization
- **Monitoring**: Track multiple metrics during training
- **Checkpointing**: Save models at regular intervals
- **Validation**: Use held-out data for evaluation

## Summary

Policy optimization is a critical component of RLHF that enables language models to learn from human feedback. Key aspects include:

1. **Policy Gradient Methods**: REINFORCE and actor-critic approaches
2. **Trust Region Methods**: PPO and TRPO for stable updates
3. **Language Model Adaptations**: Token-level and sequence-level optimization
4. **Regularization**: KL control and entropy regularization
5. **Advanced Techniques**: Multi-objective optimization and conservative updates
6. **Best Practices**: Hyperparameter tuning, stability, and evaluation

Effective policy optimization enables the training of language models that better align with human values and preferences, ultimately leading to more useful, safe, and honest AI systems.

---

**Note**: This guide provides the theoretical and practical foundations for policy optimization in RLHF. For specific implementation details and advanced techniques, refer to the implementation examples and external resources referenced in the main README. 
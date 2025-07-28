# Reward Modeling

This guide provides a comprehensive overview of reward modeling for reinforcement learning from human feedback (RLHF) systems. We'll explore the mathematical foundations, training objectives, validation methods, and practical implementation considerations for learning reward functions from human preferences.

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Reward Model Architecture](#reward-model-architecture)
- [Training Objectives](#training-objectives)
- [Loss Functions](#loss-functions)
- [Validation and Evaluation](#validation-and-evaluation)
- [Implementation Examples](#implementation-examples)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)

## Overview

Reward modeling is the process of learning a function that maps prompt-response pairs to scalar reward values, capturing human preferences and judgments. Unlike traditional supervised learning where we have ground truth labels, reward modeling learns from relative preferences and rankings provided by human annotators.

### Key Concepts

**1. Preference Learning**: Learn from relative preferences rather than absolute labels
**2. Reward Function**: $`R_\phi(x, y)`$ that assigns scalar rewards to prompt-response pairs
**3. Human Alignment**: Capture human values and preferences in the reward function
**4. Generalization**: Learn to predict preferences for unseen prompt-response pairs

### Problem Formulation

Given a dataset of human preferences:
```math
\mathcal{D} = \{(x_i, y_{i,w}, y_{i,l})\}_{i=1}^N
```

Where:
- $`x_i`$: Input prompt/context
- $`y_{i,w}`$: Preferred (winning) response
- $`y_{i,l}`$: Less preferred (losing) response

Goal: Learn reward function $`R_\phi(x, y)`$ such that:
```math
R_\phi(x, y_w) > R_\phi(x, y_l)
```

## Mathematical Foundations

### Preference Learning Framework

**Assumption**: Human preferences follow a Bradley-Terry model:
```math
P(y_w \succ y_l | x) = \frac{\exp(R_\phi(x, y_w))}{\exp(R_\phi(x, y_w)) + \exp(R_\phi(x, y_l))} = \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))
```

Where:
- $`\succ`$: "is preferred to"
- $`\sigma`$: Sigmoid function
- $`R_\phi(x, y)`$: Learned reward function

### Maximum Likelihood Estimation

**Objective**: Maximize likelihood of observed preferences:
```math
\mathcal{L}(\phi) = \sum_{i=1}^N \log P(y_{i,w} \succ y_{i,l} | x_i)
```

**Log-likelihood**:
```math
\mathcal{L}(\phi) = \sum_{i=1}^N \log \sigma(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))
```

### Gradient-Based Optimization

**Gradient**:
```math
\nabla_\phi \mathcal{L}(\phi) = \sum_{i=1}^N (1 - \sigma(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))) \nabla_\phi(R_\phi(x_i, y_{i,w}) - R_\phi(x_i, y_{i,l}))
```

## Reward Model Architecture

### Basic Architecture

**Standard Reward Model**:
```math
R_\phi(x, y) = f_\phi(\text{encode}(x, y))
```

Where:
- $`x`$: Input prompt/context
- $`y`$: Generated response
- $`f_\phi`$: Neural network with parameters $`\phi`$
- $`\text{encode}`$: Encoder for prompt-response pairs

### Encoder Architectures

**1. Concatenation-based**:
```python
class ConcatenationRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
        
    def forward(self, prompt_ids, response_ids, attention_mask=None):
        # Concatenate prompt and response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        
        # Get embeddings
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Pool over sequence length
        pooled = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        # Predict reward
        reward = self.reward_head(pooled).squeeze(-1)  # [batch_size]
        
        return reward
```

**2. Separate Encoders**:
```python
class SeparateEncoderRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.prompt_encoder = base_model
        self.response_encoder = base_model
        self.fusion_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        
    def forward(self, prompt_ids, response_ids, attention_mask=None):
        # Encode prompt and response separately
        prompt_outputs = self.prompt_encoder(prompt_ids, attention_mask=attention_mask)
        response_outputs = self.response_encoder(response_ids, attention_mask=attention_mask)
        
        # Pool embeddings
        prompt_pooled = prompt_outputs.last_hidden_state.mean(dim=1)
        response_pooled = response_outputs.last_hidden_state.mean(dim=1)
        
        # Concatenate and fuse
        combined = torch.cat([prompt_pooled, response_pooled], dim=1)
        fused = self.fusion_layer(combined)
        
        # Predict reward
        reward = self.reward_head(fused).squeeze(-1)
        
        return reward
```

**3. Cross-Attention Architecture**:
```python
class CrossAttentionRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.base_model = base_model
        self.cross_attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.reward_head = nn.Linear(hidden_size, 1)
        
    def forward(self, prompt_ids, response_ids, attention_mask=None):
        # Encode prompt and response
        prompt_outputs = self.base_model(prompt_ids, attention_mask=attention_mask)
        response_outputs = self.base_model(response_ids, attention_mask=attention_mask)
        
        # Cross-attention between prompt and response
        prompt_hidden = prompt_outputs.last_hidden_state  # [batch, seq_len, hidden]
        response_hidden = response_outputs.last_hidden_state  # [batch, seq_len, hidden]
        
        # Apply cross-attention
        attended_response, _ = self.cross_attention(
            response_hidden, prompt_hidden, prompt_hidden
        )
        
        # Pool attended response
        pooled = attended_response.mean(dim=1)
        
        # Predict reward
        reward = self.reward_head(pooled).squeeze(-1)
        
        return reward
```

### Advanced Architectures

**1. Multi-Objective Reward Model**:
```python
class MultiObjectiveRewardModel(nn.Module):
    def __init__(self, base_model, objectives=['helpfulness', 'harmlessness', 'honesty']):
        super().__init__()
        self.base_model = base_model
        self.objectives = objectives
        
        # Separate heads for each objective
        self.objective_heads = nn.ModuleDict({
            obj: nn.Linear(base_model.config.hidden_size, 1) 
            for obj in objectives
        })
        
        # Fusion layer for combined reward
        self.fusion_layer = nn.Linear(len(objectives), 1)
        
    def forward(self, prompt_ids, response_ids, attention_mask=None):
        # Encode prompt-response pair
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        
        # Predict rewards for each objective
        objective_rewards = {}
        for obj in self.objectives:
            objective_rewards[obj] = self.objective_heads[obj](pooled).squeeze(-1)
        
        # Combine rewards
        combined_rewards = torch.stack([objective_rewards[obj] for obj in self.objectives], dim=1)
        total_reward = self.fusion_layer(combined_rewards).squeeze(-1)
        
        return total_reward, objective_rewards
```

**2. Hierarchical Reward Model**:
```python
class HierarchicalRewardModel(nn.Module):
    def __init__(self, base_model, num_levels=3):
        super().__init__()
        self.base_model = base_model
        self.num_levels = num_levels
        
        # Level-specific reward heads
        self.level_heads = nn.ModuleList([
            nn.Linear(base_model.config.hidden_size, 1) 
            for _ in range(num_levels)
        ])
        
        # Level weights
        self.level_weights = nn.Parameter(torch.ones(num_levels))
        
    def forward(self, prompt_ids, response_ids, attention_mask=None):
        # Encode prompt-response pair
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Split sequence into levels
        seq_len = hidden_states.size(1)
        level_size = seq_len // self.num_levels
        
        level_rewards = []
        for i in range(self.num_levels):
            start_idx = i * level_size
            end_idx = start_idx + level_size if i < self.num_levels - 1 else seq_len
            
            level_hidden = hidden_states[:, start_idx:end_idx, :]
            level_pooled = level_hidden.mean(dim=1)
            level_reward = self.level_heads[i](level_pooled).squeeze(-1)
            level_rewards.append(level_reward)
        
        # Weighted combination
        level_rewards = torch.stack(level_rewards, dim=1)
        weights = torch.softmax(self.level_weights, dim=0)
        total_reward = (level_rewards * weights.unsqueeze(0)).sum(dim=1)
        
        return total_reward, level_rewards
```

## Training Objectives

### Preference Learning Loss

**Standard Preference Loss**:
```math
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))
```

**Implementation**:
```python
def preference_loss(reward_model, prompt_ids, chosen_ids, rejected_ids, attention_mask=None):
    """
    Compute preference learning loss
    
    Args:
        reward_model: Reward model
        prompt_ids: Input prompt token IDs
        chosen_ids: Preferred response token IDs
        rejected_ids: Less preferred response token IDs
        attention_mask: Attention mask
    
    Returns:
        loss: Preference learning loss
    """
    # Get rewards for chosen and rejected responses
    chosen_rewards = reward_model(prompt_ids, chosen_ids, attention_mask)
    rejected_rewards = reward_model(prompt_ids, rejected_ids, attention_mask)
    
    # Compute preference loss
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    
    return loss
```

### Ranking Loss

**Multi-Response Ranking**:
```math
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_1, \ldots, y_k) \sim \mathcal{D}} \sum_{i=1}^{k-1} \log \sigma(R_\phi(x, y_i) - R_\phi(x, y_{i+1}))
```

**Implementation**:
```python
def ranking_loss(reward_model, prompt_ids, response_ids_list, attention_mask=None):
    """
    Compute ranking loss for multiple responses
    
    Args:
        reward_model: Reward model
        prompt_ids: Input prompt token IDs
        response_ids_list: List of response token IDs (ordered by preference)
        attention_mask: Attention mask
    
    Returns:
        loss: Ranking loss
    """
    # Get rewards for all responses
    rewards = []
    for response_ids in response_ids_list:
        reward = reward_model(prompt_ids, response_ids, attention_mask)
        rewards.append(reward)
    
    rewards = torch.stack(rewards, dim=1)  # [batch_size, num_responses]
    
    # Compute pairwise ranking losses
    loss = 0
    for i in range(len(response_ids_list) - 1):
        loss += -torch.log(torch.sigmoid(rewards[:, i] - rewards[:, i+1])).mean()
    
    return loss
```

### Contrastive Loss

**Contrastive Learning Approach**:
```math
\mathcal{L}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \log \frac{\exp(R_\phi(x, y_w)/\tau)}{\exp(R_\phi(x, y_w)/\tau) + \exp(R_\phi(x, y_l)/\tau)}
```

Where $`\tau`$ is the temperature parameter.

**Implementation**:
```python
def contrastive_loss(reward_model, prompt_ids, chosen_ids, rejected_ids, temperature=0.1, attention_mask=None):
    """
    Compute contrastive loss
    
    Args:
        reward_model: Reward model
        prompt_ids: Input prompt token IDs
        chosen_ids: Preferred response token IDs
        rejected_ids: Less preferred response token IDs
        temperature: Temperature parameter
        attention_mask: Attention mask
    
    Returns:
        loss: Contrastive loss
    """
    # Get rewards
    chosen_rewards = reward_model(prompt_ids, chosen_ids, attention_mask)
    rejected_rewards = reward_model(prompt_ids, rejected_ids, attention_mask)
    
    # Compute contrastive loss
    logits = torch.stack([chosen_rewards, rejected_rewards], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    return loss
```

## Loss Functions

### Regularization Techniques

**1. L2 Regularization**:
```python
def l2_regularized_loss(model, loss, weight_decay=0.01):
    """Add L2 regularization to loss"""
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    
    return loss + weight_decay * l2_reg
```

**2. Dropout Regularization**:
```python
class RegularizedRewardModel(nn.Module):
    def __init__(self, base_model, dropout_rate=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_rate)
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(self, prompt_ids, response_ids, attention_mask=None):
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Apply dropout
        pooled = self.dropout(hidden_states.mean(dim=1))
        reward = self.reward_head(pooled).squeeze(-1)
        
        return reward
```

**3. Label Smoothing**:
```python
def label_smoothed_preference_loss(reward_model, prompt_ids, chosen_ids, rejected_ids, 
                                  smoothing=0.1, attention_mask=None):
    """
    Compute preference loss with label smoothing
    
    Args:
        smoothing: Label smoothing parameter
        ...: Other arguments same as preference_loss
    """
    chosen_rewards = reward_model(prompt_ids, chosen_ids, attention_mask)
    rejected_rewards = reward_model(prompt_ids, rejected_ids, attention_mask)
    
    # Apply label smoothing
    logits = chosen_rewards - rejected_rewards
    smoothed_labels = torch.full_like(logits, 1.0 - smoothing)
    
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, smoothed_labels)
    
    return loss
```

### Advanced Loss Functions

**1. Focal Loss for Hard Examples**:
```python
def focal_preference_loss(reward_model, prompt_ids, chosen_ids, rejected_ids, 
                         gamma=2.0, alpha=0.25, attention_mask=None):
    """
    Compute focal loss for preference learning
    
    Args:
        gamma: Focusing parameter
        alpha: Balancing parameter
        ...: Other arguments same as preference_loss
    """
    chosen_rewards = reward_model(prompt_ids, chosen_ids, attention_mask)
    rejected_rewards = reward_model(prompt_ids, rejected_ids, attention_mask)
    
    logits = chosen_rewards - rejected_rewards
    probs = torch.sigmoid(logits)
    
    # Focal loss
    pt = probs
    focal_weight = (1 - pt) ** gamma
    loss = -alpha * focal_weight * torch.log(pt)
    
    return loss.mean()
```

**2. Triplet Loss**:
```python
def triplet_loss(reward_model, prompt_ids, anchor_ids, positive_ids, negative_ids, 
                margin=0.1, attention_mask=None):
    """
    Compute triplet loss for reward learning
    
    Args:
        margin: Margin for triplet loss
        ...: Other arguments
    """
    anchor_rewards = reward_model(prompt_ids, anchor_ids, attention_mask)
    positive_rewards = reward_model(prompt_ids, positive_ids, attention_mask)
    negative_rewards = reward_model(prompt_ids, negative_ids, attention_mask)
    
    # Triplet loss
    positive_dist = torch.abs(anchor_rewards - positive_rewards)
    negative_dist = torch.abs(anchor_rewards - negative_rewards)
    
    loss = torch.clamp(positive_dist - negative_dist + margin, min=0)
    
    return loss.mean()
```

## Validation and Evaluation

### Evaluation Metrics

**1. Preference Accuracy**:
```python
def preference_accuracy(reward_model, test_data):
    """
    Compute preference accuracy
    
    Args:
        reward_model: Trained reward model
        test_data: Test dataset with preferences
    
    Returns:
        accuracy: Preference prediction accuracy
    """
    correct = 0
    total = 0
    
    for prompt_ids, chosen_ids, rejected_ids in test_data:
        chosen_reward = reward_model(prompt_ids, chosen_ids)
        rejected_reward = reward_model(prompt_ids, rejected_ids)
        
        # Check if model correctly predicts preference
        if chosen_reward > rejected_reward:
            correct += 1
        total += 1
    
    return correct / total
```

**2. Ranking Correlation**:
```python
def ranking_correlation(reward_model, test_data):
    """
    Compute Spearman correlation with human rankings
    
    Args:
        reward_model: Trained reward model
        test_data: Test dataset with human rankings
    
    Returns:
        correlation: Spearman correlation coefficient
    """
    from scipy.stats import spearmanr
    
    predicted_rankings = []
    human_rankings = []
    
    for prompt_ids, response_ids_list, human_ranking in test_data:
        # Get model rewards
        rewards = []
        for response_ids in response_ids_list:
            reward = reward_model(prompt_ids, response_ids)
            rewards.append(reward.item())
        
        # Get predicted ranking
        predicted_ranking = np.argsort(np.argsort(rewards))
        
        predicted_rankings.extend(predicted_ranking)
        human_rankings.extend(human_ranking)
    
    correlation, p_value = spearmanr(predicted_rankings, human_rankings)
    
    return correlation
```

**3. Calibration Metrics**:
```python
def calibration_error(reward_model, test_data, num_bins=10):
    """
    Compute calibration error
    
    Args:
        reward_model: Trained reward model
        test_data: Test dataset
        num_bins: Number of bins for calibration
    
    Returns:
        calibration_error: Expected calibration error
    """
    # Collect predictions and true preferences
    predictions = []
    true_preferences = []
    
    for prompt_ids, chosen_ids, rejected_ids in test_data:
        chosen_reward = reward_model(prompt_ids, chosen_ids)
        rejected_reward = reward_model(prompt_ids, rejected_ids)
        
        pred_prob = torch.sigmoid(chosen_reward - rejected_reward).item()
        predictions.append(pred_prob)
        true_preferences.append(1.0)  # Always 1 for preference data
    
    # Compute calibration error
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = np.logical_and(predictions > bin_lower, predictions <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_conf = np.mean(predictions[in_bin])
            bin_acc = np.mean(np.array(true_preferences)[in_bin])
            calibration_error += np.sum(in_bin) * np.abs(bin_conf - bin_acc)
    
    return calibration_error / len(predictions)
```

### Robustness Evaluation

**1. Out-of-Distribution Testing**:
```python
def ood_evaluation(reward_model, in_dist_data, ood_data):
    """
    Evaluate robustness on out-of-distribution data
    
    Args:
        reward_model: Trained reward model
        in_dist_data: In-distribution test data
        ood_data: Out-of-distribution test data
    
    Returns:
        robustness_metrics: Dictionary of robustness metrics
    """
    # Evaluate on in-distribution data
    in_dist_accuracy = preference_accuracy(reward_model, in_dist_data)
    
    # Evaluate on out-of-distribution data
    ood_accuracy = preference_accuracy(reward_model, ood_data)
    
    # Compute robustness gap
    robustness_gap = in_dist_accuracy - ood_accuracy
    
    return {
        'in_dist_accuracy': in_dist_accuracy,
        'ood_accuracy': ood_accuracy,
        'robustness_gap': robustness_gap
    }
```

**2. Adversarial Testing**:
```python
def adversarial_evaluation(reward_model, test_data, attack_method='pgd'):
    """
    Evaluate robustness against adversarial attacks
    
    Args:
        reward_model: Trained reward model
        test_data: Test dataset
        attack_method: Adversarial attack method
    
    Returns:
        adversarial_metrics: Dictionary of adversarial robustness metrics
    """
    if attack_method == 'pgd':
        return pgd_attack_evaluation(reward_model, test_data)
    elif attack_method == 'fgsm':
        return fgsm_attack_evaluation(reward_model, test_data)
    else:
        raise ValueError(f"Unknown attack method: {attack_method}")

def pgd_attack_evaluation(reward_model, test_data, epsilon=0.1, steps=10):
    """Evaluate against PGD attack"""
    # Implementation of PGD attack evaluation
    # This would involve perturbing input embeddings and checking robustness
    pass
```

## Implementation Examples

### Complete Reward Model Training

```python
class RewardModelTrainer:
    def __init__(self, model, tokenizer, learning_rate=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
    def train_epoch(self, train_dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            prompt_ids = batch['prompt_ids']
            chosen_ids = batch['chosen_ids']
            rejected_ids = batch['rejected_ids']
            attention_mask = batch.get('attention_mask')
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = preference_loss(self.model, prompt_ids, chosen_ids, rejected_ids, attention_mask)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(train_dataloader)
    
    def evaluate(self, eval_dataloader):
        """Evaluate reward model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        labels = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                prompt_ids = batch['prompt_ids']
                chosen_ids = batch['chosen_ids']
                rejected_ids = batch['rejected_ids']
                attention_mask = batch.get('attention_mask')
                
                # Compute loss
                loss = preference_loss(self.model, prompt_ids, chosen_ids, rejected_ids, attention_mask)
                total_loss += loss.item()
                
                # Compute predictions
                chosen_rewards = self.model(prompt_ids, chosen_ids, attention_mask)
                rejected_rewards = self.model(prompt_ids, rejected_ids, attention_mask)
                
                preds = (chosen_rewards > rejected_rewards).float()
                predictions.extend(preds.cpu().numpy())
                labels.extend([1.0] * len(preds))  # Always 1 for preference data
        
        accuracy = np.mean(np.array(predictions) == np.array(labels))
        return {
            'loss': total_loss / len(eval_dataloader),
            'accuracy': accuracy
        }
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

### Reward Model Inference

```python
class RewardModelInference:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
    
    def predict_reward(self, prompt, response):
        """
        Predict reward for a prompt-response pair
        
        Args:
            prompt: Input prompt text
            response: Generated response text
        
        Returns:
            reward: Predicted reward value
        """
        # Tokenize
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt', padding=True)
        response_tokens = self.tokenizer(response, return_tensors='pt', padding=True)
        
        # Predict reward
        with torch.no_grad():
            reward = self.model(prompt_tokens['input_ids'], response_tokens['input_ids'])
        
        return reward.item()
    
    def rank_responses(self, prompt, responses):
        """
        Rank multiple responses for a prompt
        
        Args:
            prompt: Input prompt text
            responses: List of response texts
        
        Returns:
            ranked_responses: Responses ranked by predicted reward
        """
        rewards = []
        for response in responses:
            reward = self.predict_reward(prompt, response)
            rewards.append(reward)
        
        # Sort by reward (descending)
        ranked_indices = np.argsort(rewards)[::-1]
        ranked_responses = [(responses[i], rewards[i]) for i in ranked_indices]
        
        return ranked_responses
    
    def batch_predict(self, prompt_response_pairs):
        """
        Predict rewards for multiple prompt-response pairs
        
        Args:
            prompt_response_pairs: List of (prompt, response) tuples
        
        Returns:
            rewards: List of predicted rewards
        """
        prompts, responses = zip(*prompt_response_pairs)
        
        # Tokenize in batch
        prompt_tokens = self.tokenizer(list(prompts), return_tensors='pt', padding=True)
        response_tokens = self.tokenizer(list(responses), return_tensors='pt', padding=True)
        
        # Predict rewards
        with torch.no_grad():
            rewards = self.model(prompt_tokens['input_ids'], response_tokens['input_ids'])
        
        return rewards.cpu().numpy()
```

## Advanced Techniques

### Multi-Task Reward Learning

```python
class MultiTaskRewardModel(nn.Module):
    def __init__(self, base_model, tasks=['helpfulness', 'harmlessness', 'honesty']):
        super().__init__()
        self.base_model = base_model
        self.tasks = tasks
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task: nn.Linear(base_model.config.hidden_size, 1) 
            for task in tasks
        })
        
        # Task weights
        self.task_weights = nn.Parameter(torch.ones(len(tasks)))
        
    def forward(self, prompt_ids, response_ids, attention_mask=None):
        # Encode prompt-response pair
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        
        # Predict rewards for each task
        task_rewards = {}
        for task in self.tasks:
            task_rewards[task] = self.task_heads[task](pooled).squeeze(-1)
        
        # Weighted combination
        weights = torch.softmax(self.task_weights, dim=0)
        combined_rewards = torch.stack([task_rewards[task] for task in self.tasks], dim=1)
        total_reward = (combined_rewards * weights.unsqueeze(0)).sum(dim=1)
        
        return total_reward, task_rewards

def multi_task_loss(model, prompt_ids, chosen_ids, rejected_ids, task_labels, attention_mask=None):
    """Compute multi-task preference loss"""
    total_reward, task_rewards = model(prompt_ids, chosen_ids, attention_mask)
    _, rejected_task_rewards = model(prompt_ids, rejected_ids, attention_mask)
    
    # Compute loss for each task
    total_loss = 0
    for task in model.tasks:
        chosen_task_reward = task_rewards[task]
        rejected_task_reward = rejected_task_rewards[task]
        
        task_loss = -torch.log(torch.sigmoid(chosen_task_reward - rejected_task_reward)).mean()
        total_loss += task_loss
    
    return total_loss
```

### Uncertainty-Aware Reward Models

```python
class UncertaintyRewardModel(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
        self.uncertainty_head = nn.Linear(hidden_size, 1)
        
    def forward(self, prompt_ids, response_ids, attention_mask=None):
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        
        # Predict reward and uncertainty
        reward = self.reward_head(pooled).squeeze(-1)
        uncertainty = torch.sigmoid(self.uncertainty_head(pooled)).squeeze(-1)
        
        return reward, uncertainty

def uncertainty_aware_loss(model, prompt_ids, chosen_ids, rejected_ids, attention_mask=None):
    """Compute uncertainty-aware preference loss"""
    chosen_reward, chosen_uncertainty = model(prompt_ids, chosen_ids, attention_mask)
    rejected_reward, rejected_uncertainty = model(prompt_ids, rejected_ids, attention_mask)
    
    # Weight by uncertainty (lower uncertainty = higher weight)
    chosen_weight = 1.0 - chosen_uncertainty
    rejected_weight = 1.0 - rejected_uncertainty
    
    # Weighted preference loss
    logits = chosen_reward - rejected_reward
    weights = chosen_weight * rejected_weight
    
    loss = -weights * torch.log(torch.sigmoid(logits))
    
    return loss.mean()
```

### Calibrated Reward Models

```python
class CalibratedRewardModel(nn.Module):
    def __init__(self, base_model, temperature=1.0):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, prompt_ids, response_ids, attention_mask=None):
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state.mean(dim=1)
        
        # Apply temperature scaling
        reward = self.reward_head(pooled).squeeze(-1) / self.temperature
        
        return reward

def calibration_loss(model, prompt_ids, chosen_ids, rejected_ids, 
                   target_prob=0.8, attention_mask=None):
    """Compute calibration loss"""
    chosen_reward = model(prompt_ids, chosen_ids, attention_mask)
    rejected_reward = model(prompt_ids, rejected_ids, attention_mask)
    
    # Compute predicted probability
    pred_prob = torch.sigmoid(chosen_reward - rejected_reward)
    
    # Calibration loss: encourage predictions to match target probability
    calibration_loss = torch.nn.functional.mse_loss(pred_prob, torch.full_like(pred_prob, target_prob))
    
    return calibration_loss
```

## Best Practices

### 1. Data Quality

- **Diverse Training Data**: Ensure coverage across different topics, styles, and difficulty levels
- **Quality Control**: Use multiple annotators and agreement monitoring
- **Bias Mitigation**: Address systematic biases in annotation
- **Validation Split**: Maintain separate validation set for model selection

### 2. Model Architecture

- **Appropriate Encoder**: Choose encoder architecture based on task requirements
- **Regularization**: Use dropout, L2 regularization, and other techniques
- **Architecture Search**: Experiment with different architectures for optimal performance
- **Scalability**: Consider computational requirements for large-scale deployment

### 3. Training Strategy

- **Learning Rate Scheduling**: Use appropriate learning rate schedules
- **Early Stopping**: Monitor validation metrics to prevent overfitting
- **Gradient Clipping**: Prevent gradient explosion in large models
- **Mixed Precision**: Use mixed precision training for efficiency

### 4. Evaluation

- **Multiple Metrics**: Use preference accuracy, ranking correlation, and calibration
- **Robustness Testing**: Evaluate on out-of-distribution and adversarial examples
- **Human Evaluation**: Validate automated metrics with human judgments
- **Continuous Monitoring**: Monitor model performance in production

### 5. Deployment

- **Model Compression**: Consider quantization and distillation for deployment
- **Caching**: Cache reward predictions for efficiency
- **Monitoring**: Implement monitoring for model drift and performance
- **Versioning**: Maintain model versions and rollback capabilities

## Summary

Reward modeling is a critical component of RLHF systems that enables learning from human preferences. Key aspects include:

1. **Mathematical Foundation**: Preference learning with Bradley-Terry model
2. **Architecture Design**: Various encoder architectures for different requirements
3. **Training Objectives**: Preference loss, ranking loss, and contrastive learning
4. **Validation**: Comprehensive evaluation with multiple metrics
5. **Advanced Techniques**: Multi-task learning, uncertainty quantification, and calibration
6. **Best Practices**: Data quality, model architecture, training strategy, and deployment

Effective reward modeling enables the training of language models that better align with human values and preferences, ultimately leading to more useful, safe, and honest AI systems.

---

**Note**: This guide provides the theoretical and practical foundations for reward modeling. For specific implementation details and advanced techniques, refer to the implementation examples and external resources referenced in the main README. 
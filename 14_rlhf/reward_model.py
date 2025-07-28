"""
Reward Model Implementation for RLHF

This module provides implementations of reward models for reinforcement learning
from human feedback (RLHF). It includes various architectures and training methods
for learning reward functions from human preferences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Basic reward model for RLHF.
    
    This model takes prompt-response pairs and outputs scalar reward values
    that capture human preferences.
    """
    
    def __init__(self, base_model_name: str, hidden_size: int = 768):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Reward head
        self.reward_head = nn.Linear(hidden_size, 1)
        
        # Initialize reward head
        nn.init.xavier_uniform_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the reward model.
        
        Args:
            input_ids: Token IDs for prompt-response pairs
            attention_mask: Attention mask
            
        Returns:
            rewards: Scalar reward values
        """
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Pool over sequence length (mean pooling)
        if attention_mask is not None:
            # Masked mean pooling
            masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Predict reward
        rewards = self.reward_head(pooled).squeeze(-1)
        
        return rewards
    
    def predict_reward(self, prompt: str, response: str) -> float:
        """
        Predict reward for a prompt-response pair.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            reward: Predicted reward value
        """
        # Concatenate prompt and response
        text = prompt + response
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        # Predict reward
        with torch.no_grad():
            reward = self.forward(inputs['input_ids'], inputs['attention_mask'])
        
        return reward.item()


class SeparateEncoderRewardModel(nn.Module):
    """
    Reward model with separate encoders for prompt and response.
    """
    
    def __init__(self, base_model_name: str, hidden_size: int = 768):
        super().__init__()
        self.prompt_encoder = AutoModel.from_pretrained(base_model_name)
        self.response_encoder = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Fusion layer
        self.fusion_layer = nn.Linear(2 * hidden_size, hidden_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        
        # Initialize
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)
        nn.init.xavier_uniform_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
    
    def forward(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                prompt_mask: Optional[torch.Tensor] = None,
                response_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with separate encoders.
        
        Args:
            prompt_ids: Token IDs for prompts
            response_ids: Token IDs for responses
            prompt_mask: Attention mask for prompts
            response_mask: Attention mask for responses
            
        Returns:
            rewards: Scalar reward values
        """
        # Encode prompts and responses separately
        prompt_outputs = self.prompt_encoder(prompt_ids, attention_mask=prompt_mask)
        response_outputs = self.response_encoder(response_ids, attention_mask=response_mask)
        
        # Pool embeddings
        if prompt_mask is not None:
            prompt_pooled = (prompt_outputs.last_hidden_state * prompt_mask.unsqueeze(-1)).sum(dim=1) / prompt_mask.sum(dim=1, keepdim=True)
        else:
            prompt_pooled = prompt_outputs.last_hidden_state.mean(dim=1)
        
        if response_mask is not None:
            response_pooled = (response_outputs.last_hidden_state * response_mask.unsqueeze(-1)).sum(dim=1) / response_mask.sum(dim=1, keepdim=True)
        else:
            response_pooled = response_outputs.last_hidden_state.mean(dim=1)
        
        # Concatenate and fuse
        combined = torch.cat([prompt_pooled, response_pooled], dim=1)
        fused = self.fusion_layer(combined)
        
        # Predict reward
        rewards = self.reward_head(fused).squeeze(-1)
        
        return rewards


class MultiObjectiveRewardModel(nn.Module):
    """
    Multi-objective reward model for different alignment objectives.
    """
    
    def __init__(self, base_model_name: str, objectives: List[str], hidden_size: int = 768):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.objectives = objectives
        
        # Separate heads for each objective
        self.objective_heads = nn.ModuleDict({
            obj: nn.Linear(hidden_size, 1) for obj in objectives
        })
        
        # Fusion layer for combined reward
        self.fusion_layer = nn.Linear(len(objectives), 1)
        
        # Initialize
        for head in self.objective_heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
        
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        nn.init.zeros_(self.fusion_layer.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for multi-objective reward model.
        
        Args:
            input_ids: Token IDs for prompt-response pairs
            attention_mask: Attention mask
            
        Returns:
            total_reward: Combined reward
            objective_rewards: Individual objective rewards
        """
        # Encode prompt-response pairs
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


class RewardModelTrainer:
    """
    Trainer for reward models using preference learning.
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 1e-5, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
    
    def preference_loss(self, chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute preference learning loss.
        
        Args:
            chosen_rewards: Rewards for preferred responses
            rejected_rewards: Rewards for less preferred responses
            
        Returns:
            loss: Preference learning loss
        """
        # Preference loss using Bradley-Terry model
        logits = chosen_rewards - rejected_rewards
        loss = -torch.log(torch.sigmoid(logits)).mean()
        
        return loss
    
    def ranking_loss(self, rewards_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute ranking loss for multiple responses.
        
        Args:
            rewards_list: List of rewards for responses (ordered by preference)
            
        Returns:
            loss: Ranking loss
        """
        loss = 0
        for i in range(len(rewards_list) - 1):
            # Pairwise ranking loss
            logits = rewards_list[i] - rewards_list[i + 1]
            loss += -torch.log(torch.sigmoid(logits)).mean()
        
        return loss / (len(rewards_list) - 1)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Perform one training step.
        
        Args:
            batch: Training batch with chosen and rejected responses
            
        Returns:
            loss: Training loss
        """
        self.model.train()
        
        # Get inputs
        chosen_ids = batch['chosen_ids'].to(self.device)
        rejected_ids = batch['rejected_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        chosen_rewards = self.model(chosen_ids, attention_mask)
        rejected_rewards = self.model(rejected_ids, attention_mask)
        
        # Compute loss
        loss = self.preference_loss(chosen_rewards, rejected_rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """
        Evaluate reward model.
        
        Args:
            eval_dataloader: Evaluation dataloader
            
        Returns:
            metrics: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                chosen_ids = batch['chosen_ids'].to(self.device)
                rejected_ids = batch['rejected_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass
                chosen_rewards = self.model(chosen_ids, attention_mask)
                rejected_rewards = self.model(rejected_ids, attention_mask)
                
                # Compute loss
                loss = self.preference_loss(chosen_rewards, rejected_rewards)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = (chosen_rewards > rejected_rewards).float()
                correct_predictions += predictions.sum().item()
                total_predictions += predictions.size(0)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'loss': total_loss / len(eval_dataloader),
            'accuracy': accuracy
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


class RewardModelInference:
    """
    Inference interface for reward models.
    """
    
    def __init__(self, model: nn.Module, tokenizer, device: str = 'cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def predict_reward(self, prompt: str, response: str) -> float:
        """
        Predict reward for a prompt-response pair.
        
        Args:
            prompt: Input prompt
            response: Generated response
            
        Returns:
            reward: Predicted reward value
        """
        # Concatenate prompt and response
        text = prompt + response
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Predict reward
        with torch.no_grad():
            reward = self.model(input_ids, attention_mask)
        
        return reward.item()
    
    def rank_responses(self, prompt: str, responses: List[str]) -> List[Tuple[str, float]]:
        """
        Rank multiple responses for a prompt.
        
        Args:
            prompt: Input prompt
            responses: List of responses to rank
            
        Returns:
            ranked_responses: Responses ranked by predicted reward
        """
        rewards = []
        for response in responses:
            reward = self.predict_reward(prompt, response)
            rewards.append(reward)
        
        # Sort by reward (descending)
        ranked_pairs = sorted(zip(responses, rewards), key=lambda x: x[1], reverse=True)
        
        return ranked_pairs
    
    def batch_predict(self, prompt_response_pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Predict rewards for multiple prompt-response pairs.
        
        Args:
            prompt_response_pairs: List of (prompt, response) tuples
            
        Returns:
            rewards: List of predicted rewards
        """
        rewards = []
        for prompt, response in prompt_response_pairs:
            reward = self.predict_reward(prompt, response)
            rewards.append(reward)
        
        return rewards


def create_reward_model(model_type: str = 'basic', **kwargs) -> nn.Module:
    """
    Factory function to create reward models.
    
    Args:
        model_type: Type of reward model ('basic', 'separate', 'multi_objective')
        **kwargs: Additional arguments
        
    Returns:
        model: Reward model
    """
    if model_type == 'basic':
        return RewardModel(**kwargs)
    elif model_type == 'separate':
        return SeparateEncoderRewardModel(**kwargs)
    elif model_type == 'multi_objective':
        return MultiObjectiveRewardModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Example usage
    model = create_reward_model('basic', base_model_name='gpt2')
    trainer = RewardModelTrainer(model)
    inference = RewardModelInference(model, model.tokenizer)
    
    # Test prediction
    reward = inference.predict_reward("What is machine learning?", "Machine learning is a subset of AI.")
    print(f"Predicted reward: {reward}") 
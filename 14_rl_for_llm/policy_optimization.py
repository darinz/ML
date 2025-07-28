"""
Policy Optimization Implementation for RLHF

This module provides implementations of policy optimization methods for reinforcement
learning from human feedback (RLHF). It includes PPO, TRPO, and other methods
specifically adapted for language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer for language models.
    """
    
    def __init__(self, model: nn.Module, ref_model: nn.Module, reward_model: nn.Module,
                 tokenizer, learning_rate: float = 1e-5, clip_epsilon: float = 0.2,
                 kl_coef: float = 0.1, device: str = 'cuda'):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Training metrics
        self.train_losses = []
        self.kl_divs = []
        self.rewards = []
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor,
                          gamma: float = 0.99, lambda_: float = 0.95) -> torch.Tensor:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Reward sequence
            values: Value estimates
            gamma: Discount factor
            lambda_: GAE parameter
            
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
            gae = delta + gamma * lambda_ * gae
            advantages[t] = gae
        
        return advantages
    
    def compute_kl_divergence(self, log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between current and reference policies.
        
        Args:
            log_probs: Current policy log probabilities
            ref_log_probs: Reference policy log probabilities
            
        Returns:
            kl_div: KL divergence
        """
        kl_div = F.kl_div(log_probs, ref_log_probs, reduction='batchmean')
        return kl_div
    
    def ppo_loss(self, log_probs: torch.Tensor, old_log_probs: torch.Tensor,
                 advantages: torch.Tensor, kl_div: torch.Tensor) -> torch.Tensor:
        """
        Compute PPO loss with KL penalty.
        
        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities
            advantages: Advantage estimates
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
    
    def train_step(self, prompts: List[str], responses: List[str], rewards: List[float]) -> Dict[str, float]:
        """
        Perform one PPO training step.
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            rewards: Rewards for responses
            
        Returns:
            metrics: Training metrics
        """
        self.model.train()
        
        # Tokenize inputs
        all_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(all_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get current policy log probabilities
        outputs = self.model(input_ids, attention_mask=attention_mask)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # Get reference policy log probabilities
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            ref_log_probs = ref_outputs.logits.log_softmax(dim=-1)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(log_probs, ref_log_probs)
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # Compute advantages (simplified - in practice use GAE)
        advantages = rewards_tensor - rewards_tensor.mean()
        
        # Compute PPO loss
        loss = self.ppo_loss(log_probs, ref_log_probs, advantages, kl_div)
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record metrics
        self.train_losses.append(loss.item())
        self.kl_divs.append(kl_div.item())
        self.rewards.append(rewards_tensor.mean().item())
        
        return {
            'loss': loss.item(),
            'kl_div': kl_div.item(),
            'mean_reward': rewards_tensor.mean().item()
        }
    
    def generate_responses(self, prompts: List[str], max_length: int = 100) -> List[str]:
        """
        Generate responses using current policy.
        
        Args:
            prompts: Input prompts
            max_length: Maximum response length
            
        Returns:
            responses: Generated responses
        """
        self.model.eval()
        responses = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
        return responses
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class TRPOTrainer:
    """
    Trust Region Policy Optimization (TRPO) trainer for language models.
    """
    
    def __init__(self, model: nn.Module, ref_model: nn.Module, reward_model: nn.Module,
                 tokenizer, max_kl: float = 0.01, damping: float = 0.1, device: str = 'cuda'):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_kl = max_kl
        self.damping = damping
    
    def conjugate_gradient(self, states: torch.Tensor, actions: torch.Tensor,
                          advantages: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
        """
        Solve linear system using conjugate gradient.
        
        Args:
            states: State representations
            actions: Action representations
            advantages: Advantage estimates
            max_iter: Maximum iterations
            
        Returns:
            step: Policy update step
        """
        # Initialize
        x = torch.zeros_like(advantages)
        r = advantages.clone()  # Initial residual
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
    
    def fisher_vector_product(self, v: torch.Tensor, states: torch.Tensor,
                            actions: torch.Tensor) -> torch.Tensor:
        """
        Compute Fisher-vector product.
        
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
        kl_grad = torch.autograd.grad(kl_div, self.model.parameters(), create_graph=True)
        
        # Compute Fisher-vector product
        Fv = torch.autograd.grad(torch.dot(kl_grad, v), self.model.parameters())
        
        return torch.stack(Fv)
    
    def compute_kl(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between current and reference policies.
        
        Args:
            states: State representations
            actions: Action representations
            
        Returns:
            kl_div: KL divergence
        """
        # Get current policy log probabilities
        current_log_probs = self.get_log_probs(states, actions)
        
        # Get reference policy log probabilities
        with torch.no_grad():
            ref_log_probs = self.get_ref_log_probs(states, actions)
        
        # Compute KL divergence
        kl_div = F.kl_div(current_log_probs, ref_log_probs, reduction='batchmean')
        
        return kl_div
    
    def get_log_probs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for current policy."""
        # Simplified implementation
        outputs = self.model(states)
        log_probs = outputs.logits.log_softmax(dim=-1)
        return log_probs
    
    def get_ref_log_probs(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for reference policy."""
        with torch.no_grad():
            outputs = self.ref_model(states)
            log_probs = outputs.logits.log_softmax(dim=-1)
        return log_probs
    
    def trpo_step(self, prompts: List[str], responses: List[str], rewards: List[float]) -> Dict[str, float]:
        """
        Perform one TRPO step.
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            rewards: Rewards for responses
            
        Returns:
            metrics: Training metrics
        """
        # Tokenize inputs
        all_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(all_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Compute advantages
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        advantages = rewards_tensor - rewards_tensor.mean()
        
        # Compute policy gradient
        log_probs = self.get_log_probs(input_ids, attention_mask)
        policy_loss = -(log_probs * advantages).mean()
        
        # Compute gradient
        grad = torch.autograd.grad(policy_loss, self.model.parameters())
        
        # Solve for step direction using conjugate gradient
        step = self.conjugate_gradient(input_ids, attention_mask, advantages)
        
        # Scale step to satisfy KL constraint
        kl_div = self.compute_kl(input_ids, attention_mask)
        scale = torch.sqrt(2 * self.max_kl / kl_div)
        step = step * torch.clamp(scale, max=1.0)
        
        # Apply step
        for param, step_param in zip(self.model.parameters(), step):
            param.data += step_param
        
        return {
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'mean_reward': rewards_tensor.mean().item()
        }


class REINFORCETrainer:
    """
    REINFORCE trainer for language models.
    """
    
    def __init__(self, model: nn.Module, reward_model: nn.Module, tokenizer,
                 learning_rate: float = 1e-5, device: str = 'cuda'):
        self.model = model.to(device)
        self.reward_model = reward_model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def reinforce_loss(self, log_probs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute REINFORCE loss.
        
        Args:
            log_probs: Log probabilities of generated tokens
            rewards: Rewards for each sequence
            
        Returns:
            loss: REINFORCE loss
        """
        # Compute log probability of each sequence
        seq_log_probs = log_probs.sum(dim=1)  # [batch_size]
        
        # Compute loss (negative because we want to maximize reward)
        loss = -(seq_log_probs * rewards).mean()
        
        return loss
    
    def train_step(self, prompts: List[str], responses: List[str], rewards: List[float]) -> float:
        """
        Perform one REINFORCE training step.
        
        Args:
            prompts: Input prompts
            responses: Generated responses
            rewards: Rewards for responses
            
        Returns:
            loss: Training loss
        """
        self.model.train()
        
        # Tokenize inputs
        all_texts = [p + r for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(all_texts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get log probabilities
        outputs = self.model(input_ids, attention_mask=attention_mask)
        log_probs = outputs.logits.log_softmax(dim=-1)
        
        # Convert rewards to tensor
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        
        # Compute REINFORCE loss
        loss = self.reinforce_loss(log_probs, rewards_tensor)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def generate_responses(self, prompts: List[str], max_length: int = 100) -> List[str]:
        """
        Generate responses using current policy.
        
        Args:
            prompts: Input prompts
            max_length: Maximum response length
            
        Returns:
            responses: Generated responses
        """
        self.model.eval()
        responses = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs['input_ids'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        
        return responses


class PolicyOptimizationPipeline:
    """
    Complete policy optimization pipeline for RLHF.
    """
    
    def __init__(self, model_name: str, reward_model: nn.Module, method: str = 'ppo',
                 device: str = 'cuda', **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_model = reward_model
        self.device = device
        
        # Initialize trainer based on method
        if method == 'ppo':
            self.trainer = PPOTrainer(self.model, self.ref_model, self.reward_model,
                                     self.tokenizer, device=device, **kwargs)
        elif method == 'trpo':
            self.trainer = TRPOTrainer(self.model, self.ref_model, self.reward_model,
                                      self.tokenizer, device=device, **kwargs)
        elif method == 'reinforce':
            self.trainer = REINFORCETrainer(self.model, self.reward_model,
                                           self.tokenizer, device=device, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
    
    def train_epoch(self, prompts: List[str], num_iterations: int = 100) -> List[Dict[str, float]]:
        """
        Train for one epoch.
        
        Args:
            prompts: Training prompts
            num_iterations: Number of training iterations
            
        Returns:
            metrics: Training metrics
        """
        metrics = []
        
        for iteration in range(num_iterations):
            # Generate responses
            responses = self.trainer.generate_responses(prompts)
            
            # Compute rewards
            rewards = []
            for prompt, response in zip(prompts, responses):
                reward = self.reward_model.predict_reward(prompt, response)
                rewards.append(reward)
            
            # Training step
            if self.method == 'ppo':
                step_metrics = self.trainer.train_step(prompts, responses, rewards)
            elif self.method == 'trpo':
                step_metrics = self.trainer.trpo_step(prompts, responses, rewards)
            elif self.method == 'reinforce':
                loss = self.trainer.train_step(prompts, responses, rewards)
                step_metrics = {'loss': loss, 'mean_reward': np.mean(rewards)}
            
            metrics.append(step_metrics)
            
            # Log progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Mean Reward: {np.mean(rewards):.4f}")
        
        return metrics
    
    def evaluate(self, eval_prompts: List[str]) -> Dict[str, float]:
        """
        Evaluate trained model.
        
        Args:
            eval_prompts: Evaluation prompts
            
        Returns:
            metrics: Evaluation metrics
        """
        # Generate responses
        responses = self.trainer.generate_responses(eval_prompts)
        
        # Compute rewards
        rewards = []
        for prompt, response in zip(eval_prompts, responses):
            reward = self.reward_model.predict_reward(prompt, response)
            rewards.append(reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        self.trainer.save_model(path)
    
    def load_model(self, path: str):
        """Load trained model."""
        self.trainer.load_model(path)


if __name__ == "__main__":
    # Example usage
    from reward_model import RewardModel
    
    # Create reward model
    reward_model = RewardModel('gpt2')
    
    # Create policy optimization pipeline
    pipeline = PolicyOptimizationPipeline('gpt2', reward_model, method='ppo')
    
    # Training prompts
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "How does deep learning work?"
    ]
    
    # Train for one epoch
    metrics = pipeline.train_epoch(prompts, num_iterations=50)
    print(f"Training completed. Final metrics: {metrics[-1]}")
    
    # Evaluate
    eval_metrics = pipeline.evaluate(prompts)
    print(f"Evaluation metrics: {eval_metrics}") 
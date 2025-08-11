"""
Direct Preference Optimization (DPO) Implementation

This module provides a complete implementation of Direct Preference Optimization
for reinforcement learning from human feedback (RLHF).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class DPOTrainer:
    """
    Direct Preference Optimization trainer.
    
    DPO eliminates the need for a separate reward model by directly optimizing
    the policy to match human preferences.
    """
    
    def __init__(self, model: nn.Module, ref_model: nn.Module, tokenizer,
                 beta: float = 0.1, learning_rate: float = 1e-5, device: str = 'cuda'):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # Training metrics
        self.train_losses = []
        self.kl_divs = []
    
    def dpo_loss(self, prompt_ids: torch.Tensor, chosen_ids: torch.Tensor,
                 rejected_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute DPO loss.
        
        Args:
            prompt_ids: Input prompt token IDs
            chosen_ids: Preferred response token IDs
            rejected_ids: Less preferred response token IDs
            attention_mask: Attention mask
            
        Returns:
            loss: DPO loss
        """
        # Get log probabilities for chosen and rejected responses
        chosen_log_probs = self.get_log_probs(prompt_ids, chosen_ids, attention_mask)
        rejected_log_probs = self.get_log_probs(prompt_ids, rejected_ids, attention_mask)
        
        # Get reference log probabilities
        with torch.no_grad():
            ref_chosen_log_probs = self.get_ref_log_probs(prompt_ids, chosen_ids, attention_mask)
            ref_rejected_log_probs = self.get_ref_log_probs(prompt_ids, rejected_ids, attention_mask)
        
        # Compute log ratios
        chosen_log_ratio = chosen_log_probs - ref_chosen_log_probs
        rejected_log_ratio = rejected_log_probs - ref_rejected_log_probs
        
        # Compute DPO loss
        logits = self.beta * (chosen_log_ratio - rejected_log_ratio)
        loss = -torch.log(torch.sigmoid(logits)).mean()
        
        return loss
    
    def get_log_probs(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get log probabilities for current policy.
        
        Args:
            prompt_ids: Prompt token IDs
            response_ids: Response token IDs
            attention_mask: Attention mask
            
        Returns:
            log_probs: Log probabilities
        """
        # Concatenate prompt and response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        if attention_mask is not None:
            input_attention_mask = torch.cat([attention_mask, torch.ones_like(response_ids)], dim=1)
        else:
            input_attention_mask = None
        
        # Get model outputs
        outputs = self.model(input_ids, attention_mask=input_attention_mask)
        logits = outputs.logits
        
        # Get log probabilities for response tokens only
        response_logits = logits[:, prompt_ids.size(1):, :]
        log_probs = F.log_softmax(response_logits, dim=-1)
        
        # Sum log probabilities over response tokens
        response_mask = torch.ones_like(response_ids)
        log_probs = (log_probs * response_mask.unsqueeze(-1)).sum(dim=1)
        
        return log_probs
    
    def get_ref_log_probs(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get log probabilities for reference policy.
        
        Args:
            prompt_ids: Prompt token IDs
            response_ids: Response token IDs
            attention_mask: Attention mask
            
        Returns:
            log_probs: Reference log probabilities
        """
        # Concatenate prompt and response
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        if attention_mask is not None:
            input_attention_mask = torch.cat([attention_mask, torch.ones_like(response_ids)], dim=1)
        else:
            input_attention_mask = None
        
        # Get reference model outputs
        with torch.no_grad():
            outputs = self.ref_model(input_ids, attention_mask=input_attention_mask)
            logits = outputs.logits
        
        # Get log probabilities for response tokens only
        response_logits = logits[:, prompt_ids.size(1):, :]
        log_probs = F.log_softmax(response_logits, dim=-1)
        
        # Sum log probabilities over response tokens
        response_mask = torch.ones_like(response_ids)
        log_probs = (log_probs * response_mask.unsqueeze(-1)).sum(dim=1)
        
        return log_probs
    
    def train_step(self, prompts: List[str], chosen_responses: List[str],
                   rejected_responses: List[str]) -> Dict[str, float]:
        """
        Perform one DPO training step.
        
        Args:
            prompts: Input prompts
            chosen_responses: Preferred responses
            rejected_responses: Less preferred responses
            
        Returns:
            metrics: Training metrics
        """
        self.model.train()
        
        # Tokenize inputs
        prompt_ids = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(self.device)
        chosen_ids = self.tokenizer(chosen_responses, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(self.device)
        rejected_ids = self.tokenizer(rejected_responses, return_tensors='pt', padding=True, truncation=True)['input_ids'].to(self.device)
        
        # Compute DPO loss
        loss = self.dpo_loss(prompt_ids, chosen_ids, rejected_ids)
        
        # Compute KL divergence for monitoring
        with torch.no_grad():
            chosen_log_probs = self.get_log_probs(prompt_ids, chosen_ids)
            ref_chosen_log_probs = self.get_ref_log_probs(prompt_ids, chosen_ids)
            kl_div = F.kl_div(chosen_log_probs, ref_chosen_log_probs, reduction='batchmean')
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        # Record metrics
        self.train_losses.append(loss.item())
        self.kl_divs.append(kl_div.item())
        
        return {
            'loss': loss.item(),
            'kl_div': kl_div.item()
        }
    
    def evaluate(self, eval_data: List[Dict]) -> Dict[str, float]:
        """
        Evaluate DPO model.
        
        Args:
            eval_data: Evaluation data
            
        Returns:
            metrics: Evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for item in eval_data:
                prompt = item['prompt']
                chosen_response = item['chosen_response']
                rejected_response = item['rejected_response']
                
                # Tokenize
                prompt_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
                chosen_ids = self.tokenizer(chosen_response, return_tensors='pt')['input_ids'].to(self.device)
                rejected_ids = self.tokenizer(rejected_response, return_tensors='pt')['input_ids'].to(self.device)
                
                # Compute loss
                loss = self.dpo_loss(prompt_ids, chosen_ids, rejected_ids)
                total_loss += loss.item()
                
                # Compute preference accuracy
                chosen_log_probs = self.get_log_probs(prompt_ids, chosen_ids)
                rejected_log_probs = self.get_log_probs(prompt_ids, rejected_ids)
                
                # Check if model correctly predicts preference
                if chosen_log_probs > rejected_log_probs:
                    correct_predictions += 1
                total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'loss': total_loss / len(eval_data),
            'accuracy': accuracy
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'beta': self.beta
        }, path)
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.beta = checkpoint.get('beta', self.beta)


class DPODataset(Dataset):
    """
    Dataset for DPO training.
    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item['prompt']
        chosen_response = item['chosen_response']
        rejected_response = item['rejected_response']
        
        # Tokenize
        chosen_text = prompt + chosen_response
        rejected_text = prompt + rejected_response
        
        chosen_inputs = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        rejected_inputs = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'prompt': prompt,
            'chosen_response': chosen_response,
            'rejected_response': rejected_response,
            'chosen_ids': chosen_inputs['input_ids'].squeeze(0),
            'rejected_ids': rejected_inputs['input_ids'].squeeze(0),
            'chosen_mask': chosen_inputs['attention_mask'].squeeze(0),
            'rejected_mask': rejected_inputs['attention_mask'].squeeze(0)
        }


class DPOPipeline:
    """
    Complete DPO training pipeline.
    """
    
    def __init__(self, model_name: str, beta: float = 0.1, learning_rate: float = 1e-5,
                 device: str = 'cuda'):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        
        # Initialize trainer
        self.trainer = DPOTrainer(self.model, self.ref_model, self.tokenizer,
                                 beta=beta, learning_rate=learning_rate, device=device)
    
    def train(self, train_data: List[Dict], val_data: List[Dict],
              num_epochs: int = 3, batch_size: int = 8) -> List[Dict]:
        """
        Train DPO model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            training_history: Training history
        """
        # Create dataloaders
        train_dataset = DPODataset(train_data, self.tokenizer)
        val_dataset = DPODataset(val_data, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        training_history = []
        
        for epoch in range(num_epochs):
            epoch_metrics = {'epoch': epoch}
            
            # Training
            train_losses = []
            for batch in train_loader:
                prompts = batch['prompt']
                chosen_responses = batch['chosen_response']
                rejected_responses = batch['rejected_response']
                
                metrics = self.trainer.train_step(prompts, chosen_responses, rejected_responses)
                train_losses.append(metrics['loss'])
            
            epoch_metrics['train_loss'] = np.mean(train_losses)
            
            # Validation
            val_metrics = self.trainer.evaluate(val_data)
            epoch_metrics.update(val_metrics)
            
            training_history.append(epoch_metrics)
            
            print(f"Epoch {epoch}: Train Loss: {epoch_metrics['train_loss']:.4f}, "
                  f"Val Loss: {epoch_metrics['loss']:.4f}, "
                  f"Val Accuracy: {epoch_metrics['accuracy']:.4f}")
        
        return training_history
    
    def generate_response(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate response using trained model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum response length
            
        Returns:
            response: Generated response
        """
        self.model.eval()
        
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
        return response
    
    def save_model(self, path: str):
        """Save trained model."""
        self.trainer.save_model(path)
    
    def load_model(self, path: str):
        """Load trained model."""
        self.trainer.load_model(path)


class AdaptiveDPO(DPOTrainer):
    """
    Adaptive DPO with dynamic beta adjustment.
    """
    
    def __init__(self, model: nn.Module, ref_model: nn.Module, tokenizer,
                 initial_beta: float = 0.1, target_kl: float = 0.01,
                 learning_rate: float = 1e-5, device: str = 'cuda'):
        super().__init__(model, ref_model, tokenizer, initial_beta, learning_rate, device)
        self.target_kl = target_kl
        self.beta_history = [initial_beta]
    
    def adaptive_beta_update(self, current_kl: float):
        """
        Update beta based on current KL divergence.
        
        Args:
            current_kl: Current KL divergence
        """
        if current_kl > 2 * self.target_kl:
            self.beta *= 1.5
        elif current_kl < 0.5 * self.target_kl:
            self.beta *= 0.5
        
        self.beta_history.append(self.beta)
    
    def train_step(self, prompts: List[str], chosen_responses: List[str],
                   rejected_responses: List[str]) -> Dict[str, float]:
        """
        Perform one adaptive DPO training step.
        
        Args:
            prompts: Input prompts
            chosen_responses: Preferred responses
            rejected_responses: Less preferred responses
            
        Returns:
            metrics: Training metrics
        """
        # Perform standard DPO step
        metrics = super().train_step(prompts, chosen_responses, rejected_responses)
        
        # Update beta adaptively
        self.adaptive_beta_update(metrics['kl_div'])
        
        metrics['beta'] = self.beta
        return metrics


if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model and tokenizer
    model_name = 'gpt2'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create DPO trainer
    trainer = DPOTrainer(model, ref_model, tokenizer, beta=0.1)
    
    # Sample training data
    train_data = [
        {
            'prompt': 'What is machine learning?',
            'chosen_response': 'Machine learning is a subset of AI that enables computers to learn from data.',
            'rejected_response': 'Machine learning is cool.'
        },
        {
            'prompt': 'Explain neural networks.',
            'chosen_response': 'Neural networks are computational models inspired by biological neurons.',
            'rejected_response': 'Neural networks are networks.'
        }
    ]
    
    # Training loop
    for epoch in range(5):
        for item in train_data:
            metrics = trainer.train_step(
                [item['prompt']],
                [item['chosen_response']],
                [item['rejected_response']]
            )
            print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}, KL: {metrics['kl_div']:.4f}")
    
    # Save model
    trainer.save_model('dpo_model.pt')
    print("Training completed and model saved!") 
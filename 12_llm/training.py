"""
Training Loop and Optimization Implementation
===========================================

This module provides comprehensive training utilities for transformer models,
including training loops, optimization strategies, and evaluation functions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import math
import time
import wandb
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm


class TransformerTrainer:
    """
    Comprehensive trainer for transformer models.
    
    This class provides a complete training pipeline with various
    optimization strategies and monitoring capabilities.
    """
    
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, 
                 val_dataloader: Optional[DataLoader] = None,
                 config: Dict[str, Any] = None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup mixed precision
        self.scaler = GradScaler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Initialize wandb if configured
        if self.config.get('use_wandb', False):
            wandb.init(project=self.config.get('project_name', 'transformer-training'))
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine')
        warmup_steps = self.config.get('warmup_steps', 4000)
        total_steps = len(self.train_dataloader) * self.config.get('num_epochs', 100)
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        elif scheduler_type == 'linear':
            return optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=warmup_steps, gamma=0.1
            )
        elif scheduler_type == 'transformer':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                else:
                    return (warmup_steps / step) ** 0.5
            
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch of data
        
        Returns:
            Dictionary containing loss and other metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
        # Backward pass
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        max_grad_norm = self.config.get('max_grad_norm', 1.0)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Scheduler step
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validation step.
        
        Returns:
            Dictionary containing validation metrics
        """
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        epoch_loss = 0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            # Training step
            metrics = self.train_step(batch)
            
            # Update metrics
            epoch_loss += metrics['loss']
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'lr': f"{metrics['learning_rate']:.6f}"
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                wandb.log({
                    'train_loss': metrics['loss'],
                    'learning_rate': metrics['learning_rate'],
                    'global_step': self.global_step
                })
        
        avg_loss = epoch_loss / num_batches
        return {'train_loss': avg_loss}
    
    def train(self, num_epochs: int, save_path: str = None) -> None:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save best model
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Print metrics
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Save best model
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                if save_path:
                    self.save_checkpoint(save_path)
            
            # Save checkpoint periodically
            if save_path and (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(f"{save_path}_epoch_{epoch + 1}")
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from {path}")


class LanguageModelTrainer(TransformerTrainer):
    """
    Specialized trainer for language modeling tasks.
    
    This includes additional metrics like perplexity and
    language model specific optimizations.
    """
    
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, 
                 val_dataloader: Optional[DataLoader] = None,
                 config: Dict[str, Any] = None):
        super().__init__(model, train_dataloader, val_dataloader, config)
    
    def compute_perplexity(self, dataloader: DataLoader) -> float:
        """
        Compute perplexity on a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
        
        Returns:
            Perplexity value
        """
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Count tokens (assuming labels are provided)
                if 'labels' in batch:
                    num_tokens = (batch['labels'] != -100).sum().item()
                else:
                    num_tokens = batch['input_ids'].numel()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def validate(self) -> Dict[str, float]:
        """Override validation to include perplexity."""
        val_metrics = super().validate()
        
        if self.val_dataloader is not None:
            perplexity = self.compute_perplexity(self.val_dataloader)
            val_metrics['perplexity'] = perplexity
        
        return val_metrics


class ClassificationTrainer(TransformerTrainer):
    """
    Specialized trainer for classification tasks.
    
    This includes accuracy metrics and classification-specific
    optimizations.
    """
    
    def __init__(self, model: nn.Module, train_dataloader: DataLoader, 
                 val_dataloader: Optional[DataLoader] = None,
                 config: Dict[str, Any] = None):
        super().__init__(model, train_dataloader, val_dataloader, config)
    
    def compute_accuracy(self, dataloader: DataLoader) -> float:
        """
        Compute accuracy on a dataset.
        
        Args:
            dataloader: DataLoader for the dataset
        
        Returns:
            Accuracy value
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                
                # Compute predictions
                predictions = torch.argmax(logits, dim=-1)
                labels = batch['labels']
                
                # Count correct predictions
                correct += (predictions == labels).sum().item()
                total += labels.numel()
        
        accuracy = correct / total
        return accuracy
    
    def validate(self) -> Dict[str, float]:
        """Override validation to include accuracy."""
        val_metrics = super().validate()
        
        if self.val_dataloader is not None:
            accuracy = self.compute_accuracy(self.val_dataloader)
            val_metrics['accuracy'] = accuracy
        
        return val_metrics


def create_trainer(trainer_type: str, model: nn.Module, train_dataloader: DataLoader,
                  val_dataloader: Optional[DataLoader] = None,
                  config: Dict[str, Any] = None) -> TransformerTrainer:
    """
    Factory function to create trainers.
    
    Args:
        trainer_type: Type of trainer ('transformer', 'language_model', 'classification')
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        config: Training configuration
    
    Returns:
        Trainer instance
    """
    if trainer_type == 'transformer':
        return TransformerTrainer(model, train_dataloader, val_dataloader, config)
    elif trainer_type == 'language_model':
        return LanguageModelTrainer(model, train_dataloader, val_dataloader, config)
    elif trainer_type == 'classification':
        return ClassificationTrainer(model, train_dataloader, val_dataloader, config)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'optimizer': 'adamw',
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'scheduler': 'transformer',
        'warmup_steps': 4000,
        'max_grad_norm': 1.0,
        'num_epochs': 100,
        'save_interval': 10,
        'use_wandb': True,
        'project_name': 'transformer-training'
    }
    
    # Create dummy model and data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 2)
        
        def forward(self, input_ids, labels=None):
            outputs = self.linear(input_ids.float())
            if labels is not None:
                loss = nn.CrossEntropyLoss()(outputs, labels)
                return type('obj', (object,), {'loss': loss})()
            return outputs
    
    # Create dummy data
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=1000):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, 10, (10,)),
                'labels': torch.randint(0, 2, (10,))
            }
    
    # Create data loaders
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(100)
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model and trainer
    model = DummyModel()
    trainer = create_trainer('language_model', model, train_dataloader, val_dataloader, config)
    
    # Train
    trainer.train(num_epochs=5, save_path='checkpoint.pt') 
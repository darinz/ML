# Model Parallel Training for Transformers

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import os
from typing import Dict, Any, Optional, List
import math

class ModelParallelTransformer(nn.Module):
    """
    Model parallel transformer implementation.
    
    This demonstrates how to split transformer layers across multiple GPUs
    for training very large models that don't fit on a single device.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 12,
                 num_heads: int = 8, d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1, num_gpus: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.num_gpus = num_gpus
        
        # Embedding layer (on GPU 0)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Split transformer layers across GPUs
        self.layers_per_gpu = num_layers // num_gpus
        self.transformer_layers = nn.ModuleList()
        
        for i in range(num_gpus):
            start_layer = i * self.layers_per_gpu
            end_layer = (i + 1) * self.layers_per_gpu if i < num_gpus - 1 else num_layers
            
            gpu_layers = nn.ModuleList([
                TransformerLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(end_layer - start_layer)
            ])
            self.transformer_layers.append(gpu_layers)
        
        # Output projection (on last GPU)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with model parallelism.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            output: Transformer output
        """
        # Embedding and positional encoding (GPU 0)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through transformer layers on different GPUs
        for gpu_id, gpu_layers in enumerate(self.transformer_layers):
            device = torch.device(f'cuda:{gpu_id}')
            x = x.to(device)
            
            for layer in gpu_layers:
                layer = layer.to(device)
                x = layer(x)
            
            # Move to next GPU if not the last one
            if gpu_id < len(self.transformer_layers) - 1:
                x = x.to(torch.device(f'cuda:{gpu_id + 1}'))
        
        # Output projection (on last GPU)
        output = self.output_projection(x)
        return output

class TransformerLayer(nn.Module):
    """Single transformer layer for model parallel training."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention for model parallel training."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of multi-head attention."""
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.w_o(context)
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]

class DistributedTrainer:
    """
    Distributed trainer for model parallel training.
    
    Handles data parallel and model parallel training across multiple GPUs.
    """
    
    def __init__(self, model: nn.Module, world_size: int, rank: int,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 lr: float = 1e-4):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(f'cuda:{rank}')
        
        # Move model to device
        self.model.to(self.device)
        
        # Wrap with DDP for data parallelism
        self.model = DDP(self.model, device_ids=[rank])
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
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
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
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
        """Validation step."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.train_step(batch)
            
            epoch_loss += metrics['loss']
            num_batches += 1
            self.global_step += 1
            
            if batch_idx % 100 == 0 and self.rank == 0:
                print(f'Step {self.global_step}: Loss = {metrics["loss"]:.4f}')
        
        avg_loss = epoch_loss / num_batches
        return {'train_loss': avg_loss}
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """Main training loop."""
        if self.rank == 0:
            print(f"Starting distributed training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Print metrics (only on rank 0)
            if self.rank == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print(f'  Train Loss: {train_metrics["train_loss"]:.4f}')
                if val_metrics:
                    print(f'  Val Loss: {val_metrics["val_loss"]:.4f}')
            
            # Save checkpoint
            if save_path and self.rank == 0:
                if val_metrics and val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    self.save_checkpoint(save_path)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }, path)
        print(f"Checkpoint saved to {path}")

def setup_distributed_training(rank: int, world_size: int):
    """
    Setup distributed training environment.
    
    Args:
        rank: Process rank
        world_size: Number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)

def cleanup_distributed_training():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def create_model_parallel_model(vocab_size: int, num_gpus: int = 2):
    """
    Create model parallel transformer.
    
    Args:
        vocab_size: Vocabulary size
        num_gpus: Number of GPUs for model parallelism
    
    Returns:
        model: Model parallel transformer
    """
    model = ModelParallelTransformer(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=12,
        num_heads=8,
        d_ff=2048,
        num_gpus=num_gpus
    )
    return model

def create_distributed_dataloader(dataset, batch_size: int, world_size: int, rank: int):
    """
    Create distributed data loader.
    
    Args:
        dataset: Dataset
        batch_size: Batch size per GPU
        world_size: Number of processes
        rank: Process rank
    
    Returns:
        dataloader: Distributed data loader
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader

def train_model_parallel(rank: int, world_size: int, vocab_size: int = 30000):
    """
    Train model parallel transformer.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        vocab_size: Vocabulary size
    """
    # Setup distributed training
    setup_distributed_training(rank, world_size)
    
    # Create model
    model = create_model_parallel_model(vocab_size, world_size)
    
    # Create dummy dataset and dataloader
    # In practice, use real dataset
    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randint(0, vocab_size, (1000, 128)),
        torch.randint(0, vocab_size, (1000, 128))
    )
    
    train_loader = create_distributed_dataloader(
        dummy_dataset, batch_size=32, world_size=world_size, rank=rank
    )
    
    # Create trainer
    trainer = DistributedTrainer(
        model=model,
        world_size=world_size,
        rank=rank,
        train_loader=train_loader,
        lr=1e-4
    )
    
    # Train
    trainer.train(num_epochs=10)
    
    # Cleanup
    cleanup_distributed_training()

def run_model_parallel_training(world_size: int = 2):
    """
    Run model parallel training with multiple processes.
    
    Args:
        world_size: Number of processes/GPUs
    """
    mp.spawn(
        train_model_parallel,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # Example usage
    run_model_parallel_training(world_size=2) 
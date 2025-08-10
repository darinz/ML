"""
Training Techniques for Large Language Models.

This module implements various training techniques and optimizations
for efficiently training large transformer models.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torch.distributed.optim import ZeroRedundancyOptimizer

class MixedPrecisionTrainer:
    """
    Mixed Precision Trainer for efficient LLM training.
    
    Uses lower precision (FP16/BF16) to reduce memory usage and speed up training.
    """
    
    def __init__(self, model, optimizer, device='cuda'):
        """
        Initialize Mixed Precision Trainer.
        
        Args:
            model: The model to train
            optimizer: The optimizer to use
            device: Device to train on
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()
        
    def train_step(self, batch, targets):
        """
        Perform a single training step with mixed precision.
        
        Args:
            batch: Input batch
            targets: Target labels
            
        Returns:
            loss: Training loss
        """
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = self.model(batch)
            loss = torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)), 
                targets.view(-1)
            )
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()

class MemoryEfficientLLM(nn.Module):
    """
    Memory Efficient Large Language Model using gradient checkpointing.
    
    Trades compute for memory by recomputing intermediate activations.
    """
    
    def __init__(self, vocab_size, d_model, num_layers, use_checkpoint=True):
        """
        Initialize Memory Efficient LLM.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer layers
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Import transformer layer from other module
        from encoder_decoder_layers import EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads=16, d_ff=d_model*4)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Forward pass with optional gradient checkpointing.
        
        Args:
            x: Input tensor
            
        Returns:
            output: Model output
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = checkpoint(layer, x)
            else:
                x = layer(x)
        
        x = self.norm(x)
        return self.output_projection(x)

class ModelParallelLLM(nn.Module):
    """
    Model Parallel Large Language Model.
    
    Distributes model layers across multiple devices for memory efficiency.
    """
    
    def __init__(self, vocab_size, d_model, num_layers, num_gpus=4):
        """
        Initialize Model Parallel LLM.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_gpus: Number of GPUs to distribute across
        """
        super().__init__()
        self.num_gpus = num_gpus
        self.layers_per_gpu = num_layers // num_gpus
        
        # Embedding on first GPU
        self.embedding = nn.Embedding(vocab_size, d_model).to('cuda:0')
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model).to('cuda:0')
        
        # Distribute layers across GPUs
        self.layers = nn.ModuleList()
        for i in range(num_gpus):
            start_layer = i * self.layers_per_gpu
            end_layer = (i + 1) * self.layers_per_gpu if i < num_gpus - 1 else num_layers
            
            # Import transformer layer from other module
            from encoder_decoder_layers import EncoderLayer
            gpu_layers = nn.ModuleList([
                EncoderLayer(d_model, num_heads=16, d_ff=d_model*4)
                for _ in range(end_layer - start_layer)
            ]).to(f'cuda:{i}')
            
            self.layers.append(gpu_layers)
        
        # Output projection on last GPU
        self.norm = nn.LayerNorm(d_model).to(f'cuda:{num_gpus-1}')
        self.output_projection = nn.Linear(d_model, vocab_size).to(f'cuda:{num_gpus-1}')
    
    def forward(self, x):
        """
        Forward pass across multiple GPUs.
        
        Args:
            x: Input tensor
            
        Returns:
            output: Model output
        """
        # Move input to first GPU
        x = x.to('cuda:0')
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Process through layers on different GPUs
        for gpu_layers in self.layers:
            for layer in gpu_layers:
                x = layer(x)
            # Move to next GPU if not last
            if gpu_layers != self.layers[-1]:
                x = x.to(f'cuda:{self.layers.index(gpu_layers) + 1}')
        
        x = self.norm(x)
        return self.output_projection(x)

class ZeROLLM(nn.Module):
    """
    ZeRO (Zero Redundancy Optimizer) Large Language Model.
    
    Uses ZeRO optimization for memory efficiency.
    """
    
    def __init__(self, vocab_size, d_model, num_layers):
        """
        Initialize ZeRO LLM.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer layers
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Import transformer layer from other module
        from encoder_decoder_layers import EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads=16, d_ff=d_model*4)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """
        Forward pass of ZeRO LLM.
        
        Args:
            x: Input tensor
            
        Returns:
            output: Model output
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return self.output_projection(x)

def setup_zero_optimizer(model, lr=1e-4):
    """
    Setup ZeRO optimizer for a model.
    
    Args:
        model: The model to optimize
        lr: Learning rate
        
    Returns:
        optimizer: ZeRO optimizer
    """
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    return optimizer

class CosineAnnealingWarmupScheduler:
    """
    Cosine Annealing with Warmup learning rate scheduler.
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        """
        Initialize scheduler.
        
        Args:
            optimizer: The optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, step):
        """
        Update learning rate for current step.
        
        Args:
            step: Current training step
        """
        if step < self.warmup_steps:
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def init_weights(module):
    """
    Initialize weights for transformer modules.
    
    Args:
        module: Module to initialize
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def train_with_gradient_accumulation(model, dataloader, optimizer, accumulation_steps=4):
    """
    Train with gradient accumulation for large batch sizes.
    
    Args:
        model: Model to train
        dataloader: Data loader
        optimizer: Optimizer
        accumulation_steps: Number of steps to accumulate gradients
    """
    model.train()
    optimizer.zero_grad()
    
    for i, (batch, targets) in enumerate(dataloader):
        loss = model(batch, targets)
        loss = loss / accumulation_steps  # Scale loss
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# Example usage
def demonstrate_training_techniques():
    """Demonstrate various training techniques."""
    print("Training Techniques Demonstration")
    print("=" * 40)
    
    # Example 1: Mixed Precision Training
    print("1. Mixed Precision Training")
    vocab_size, d_model, num_layers = 50000, 2048, 24
    
    # Create model (simplified for demonstration)
    model = MemoryEfficientLLM(vocab_size, d_model, num_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = MixedPrecisionTrainer(model, optimizer)
    
    print(f"   Model created with {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters")
    print()
    
    # Example 2: Memory Efficient Training
    print("2. Memory Efficient Training")
    model_checkpoint = MemoryEfficientLLM(vocab_size, d_model, num_layers, use_checkpoint=True)
    print(f"   Gradient checkpointing enabled: {model_checkpoint.use_checkpoint}")
    print()
    
    # Example 3: Model Parallel Training
    print("3. Model Parallel Training")
    if torch.cuda.device_count() >= 2:
        model_parallel = ModelParallelLLM(vocab_size, d_model, num_layers, num_gpus=2)
        print(f"   Model distributed across {model_parallel.num_gpus} GPUs")
    else:
        print("   Not enough GPUs for model parallelism demonstration")
    print()
    
    # Example 4: ZeRO Optimization
    print("4. ZeRO Optimization")
    model_zero = ZeROLLM(vocab_size, d_model, num_layers)
    optimizer_zero = setup_zero_optimizer(model_zero)
    print("   ZeRO optimizer configured")
    print()
    
    # Example 5: Learning Rate Scheduling
    print("5. Learning Rate Scheduling")
    scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_steps=4000, total_steps=100000)
    print("   Cosine annealing with warmup scheduler created")
    print()
    
    # Example 6: Weight Initialization
    print("6. Weight Initialization")
    model.apply(init_weights)
    print("   Weights initialized with proper initialization scheme")

if __name__ == "__main__":
    import math
    demonstrate_training_techniques()

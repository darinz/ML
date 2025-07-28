# Image Inpainting for Self-Supervised Learning

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
import numpy as np
import random

class InpaintingModel(nn.Module):
    """
    Image Inpainting Model for Self-Supervised Learning.
    
    This model learns to fill in masked regions of images,
    which serves as a pretext task for representation learning.
    """
    
    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, 
                 num_layers: int = 4, mask_ratio: float = 0.15):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mask_ratio = mask_ratio
        
        # Encoder
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_dim * (2 ** i)
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            in_ch = out_ch
        
        # Decoder
        self.decoder = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            out_ch = hidden_dim * (2 ** i) if i > 0 else in_channels
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            in_ch = out_ch
        
        # Final reconstruction layer
        self.final_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of inpainting model.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            mask: Optional mask indicating regions to inpaint
        
        Returns:
            reconstructed: Reconstructed images
        """
        # Encoder
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(features) - 1:
                # Skip connection
                skip_feature = features[-(i + 2)]
                x = x + skip_feature
        
        # Final reconstruction
        reconstructed = self.final_layer(x)
        
        return reconstructed
    
    def create_random_mask(self, batch_size: int, height: int, width: int, 
                          device: torch.device) -> torch.Tensor:
        """
        Create random mask for inpainting.
        
        Args:
            batch_size: Batch size
            height: Image height
            width: Image width
            device: Device to create mask on
        
        Returns:
            mask: Binary mask where 1 indicates masked regions
        """
        mask = torch.zeros(batch_size, 1, height, width, device=device)
        
        # Create random rectangular masks
        for b in range(batch_size):
            num_masks = random.randint(1, 5)
            for _ in range(num_masks):
                # Random mask size
                mask_h = random.randint(height // 8, height // 4)
                mask_w = random.randint(width // 8, width // 4)
                
                # Random position
                top = random.randint(0, height - mask_h)
                left = random.randint(0, width - mask_w)
                
                # Apply mask
                mask[b, 0, top:top+mask_h, left:left+mask_w] = 1
        
        return mask
    
    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to input images.
        
        Args:
            x: Input images
            mask: Binary mask
        
        Returns:
            masked_x: Masked images
        """
        # Expand mask to match input channels
        mask = mask.expand(-1, x.size(1), -1, -1)
        
        # Apply mask (set masked regions to 0)
        masked_x = x * (1 - mask)
        
        return masked_x

class TransformerInpaintingModel(nn.Module):
    """
    Transformer-based Inpainting Model.
    
    Uses transformer architecture for image inpainting.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12,
                 mlp_ratio: float = 4.0, mask_ratio: float = 0.15):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mask_ratio = mask_ratio
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Transformer encoder
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        # Decoder head
        self.decoder_head = nn.Linear(embed_dim, patch_size * patch_size * in_channels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for layer in self.transformer_layers:
            nn.init.trunc_normal_(layer.attention.proj.weight, std=0.02)
            nn.init.trunc_normal_(layer.mlp.fc2.weight, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of transformer inpainting model.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            mask: Optional mask indicating regions to inpaint
        
        Returns:
            reconstructed: Reconstructed images
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        patches = self.patch_embed(x)  # (B, embed_dim, grid_h, grid_w)
        patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add position embedding
        patches = patches + self.pos_embed
        
        # Apply mask if provided
        if mask is not None:
            # Reshape mask to patches
            mask = F.interpolate(mask, size=(H // self.patch_size, W // self.patch_size), 
                               mode='nearest')
            mask = mask.flatten(1)  # (B, num_patches)
            
            # Replace masked patches with mask token
            mask_tokens = self.mask_token.expand(B, self.num_patches, -1)
            patches = torch.where(mask.unsqueeze(-1).bool(), mask_tokens, patches)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            patches = layer(patches)
        
        # Decode patches
        reconstructed_patches = self.decoder_head(patches)  # (B, num_patches, patch_size^2 * channels)
        
        # Reshape to image
        reconstructed = reconstructed_patches.view(B, self.num_patches, C, self.patch_size, self.patch_size)
        reconstructed = reconstructed.permute(0, 2, 1, 3, 4).contiguous()
        reconstructed = reconstructed.view(B, C, H, W)
        
        return reconstructed
    
    def create_random_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Create random mask for transformer inpainting.
        
        Args:
            batch_size: Batch size
            device: Device to create mask on
        
        Returns:
            mask: Binary mask
        """
        num_patches = self.num_patches
        num_masked = int(num_patches * self.mask_ratio)
        
        mask = torch.zeros(batch_size, num_patches, device=device)
        
        for b in range(batch_size):
            # Randomly select patches to mask
            masked_indices = torch.randperm(num_patches)[:num_masked]
            mask[b, masked_indices] = 1
        
        return mask

class TransformerLayer(nn.Module):
    """Single transformer layer for inpainting."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of transformer layer."""
        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # MLP
        mlp_output = self.mlp(x)
        x = self.norm2(x + mlp_output)
        
        return x

class InpaintingTrainer:
    """
    Trainer for inpainting models.
    """
    
    def __init__(self, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 lr: float = 1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = batch.to(self.device)
        
        # Create mask
        if isinstance(self.model, TransformerInpaintingModel):
            mask = self.model.create_random_mask(batch.size(0), self.device)
            masked_batch = batch  # For transformer, mask is applied in forward pass
        else:
            mask = self.model.create_random_mask(batch.size(0), batch.size(2), batch.size(3), self.device)
            masked_batch = self.model.apply_mask(batch, mask)
        
        # Forward pass
        reconstructed = self.model(masked_batch, mask)
        
        # Calculate loss
        loss = self.criterion(reconstructed, batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
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
                batch = batch.to(self.device)
                
                # Create mask
                if isinstance(self.model, TransformerInpaintingModel):
                    mask = self.model.create_random_mask(batch.size(0), self.device)
                    masked_batch = batch
                else:
                    mask = self.model.create_random_mask(batch.size(0), batch.size(2), batch.size(3), self.device)
                    masked_batch = self.model.apply_mask(batch, mask)
                
                # Forward pass
                reconstructed = self.model(masked_batch, mask)
                
                # Calculate loss
                loss = self.criterion(reconstructed, batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """Main training loop."""
        print(f"Starting inpainting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                num_batches += 1
                self.global_step += 1
                
                if batch_idx % 100 == 0:
                    print(f'Step {self.global_step}: Loss = {metrics["loss"]:.4f}')
            
            avg_loss = epoch_loss / num_batches
            
            # Validation
            val_metrics = self.validate()
            
            # Print metrics
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'  Train Loss: {avg_loss:.4f}')
            if val_metrics:
                print(f'  Val Loss: {val_metrics["val_loss"]:.4f}')
            
            # Save checkpoint
            if save_path and val_metrics and val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.save_checkpoint(save_path)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }, path)
        print(f"Checkpoint saved to {path}")

def create_inpainting_model(model_type: str = 'cnn', **kwargs):
    """
    Create inpainting model based on type.
    
    Args:
        model_type: Type of model ('cnn', 'transformer')
        **kwargs: Additional arguments for the model
    
    Returns:
        model: Inpainting model
    """
    if model_type == 'cnn':
        return InpaintingModel(**kwargs)
    elif model_type == 'transformer':
        return TransformerInpaintingModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def visualize_inpainting(original: torch.Tensor, masked: torch.Tensor, 
                        reconstructed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Visualize inpainting results.
    
    Args:
        original: Original images
        masked: Masked images
        reconstructed: Reconstructed images
        mask: Binary mask
    
    Returns:
        visualization: Concatenated visualization
    """
    # Normalize images to [0, 1]
    original = (original - original.min()) / (original.max() - original.min())
    masked = (masked - masked.min()) / (masked.max() - masked.min())
    reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())
    
    # Expand mask to 3 channels for visualization
    if mask.size(1) == 1:
        mask = mask.expand(-1, 3, -1, -1)
    
    # Concatenate images
    visualization = torch.cat([original, masked, reconstructed, mask], dim=3)
    
    return visualization

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    channels = 3
    height = width = 224
    
    # Create input
    x = torch.randn(batch_size, channels, height, width)
    
    # Test different inpainting models
    models = {
        'cnn': InpaintingModel(in_channels=3, hidden_dim=64),
        'transformer': TransformerInpaintingModel(img_size=224, patch_size=16)
    }
    
    for name, model in models.items():
        # Create mask
        if isinstance(model, TransformerInpaintingModel):
            mask = model.create_random_mask(batch_size, torch.device('cpu'))
            output = model(x, mask)
        else:
            mask = model.create_random_mask(batch_size, height, width, torch.device('cpu'))
            masked_x = model.apply_mask(x, mask)
            output = model(masked_x, mask)
        
        print(f"{name} inpainting output shape: {output.shape}")
    
    print("Inpainting implementations created successfully!") 
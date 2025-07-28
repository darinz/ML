# Vision Transformer (ViT) Implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
import numpy as np
from einops import rearrange, repeat

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.
    
    This is a complete implementation of the Vision Transformer architecture
    as described in "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 num_classes: int = 1000, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 representation_size: Optional[int] = None, distilled: bool = False,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0, drop_path_rate: float = 0.0,
                 norm_layer: Optional[nn.Module] = None, act_layer: Optional[nn.Module] = None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, norm_layer=norm_layer
        )
        num_patches = self.patch_embed.num_patches
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 and distilled else nn.Identity()
        
        # Weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize patch embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.trunc_normal_(block.attn.proj.weight, std=0.02)
            nn.init.trunc_normal_(block.mlp.fc2.weight, std=0.02)
        
        # Initialize classifier
        if hasattr(self.head, 'weight'):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
        if hasattr(self.head_dist, 'weight'):
            nn.init.trunc_normal_(self.head_dist.weight, std=0.02)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extraction layers.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            features: Extracted features
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add distillation token if using distillation
        if self.dist_token is not None:
            dist_tokens = self.dist_token.expand(B, -1, -1)
            x = torch.cat((x, dist_tokens), dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Vision Transformer.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            logits: Classification logits
        """
        x = self.forward_features(x)
        
        if self.dist_token is not None:
            # Use both class token and distillation token
            x, x_dist = x[:, 0], x[:, 1]
            x = self.pre_logits(x)
            x_dist = self.pre_logits(x_dist)
            
            if self.training:
                return self.head(x), self.head_dist(x_dist)
            else:
                # During inference, average the predictions
                return (self.head(x) + self.head_dist(x_dist)) / 2
        else:
            # Use only class token
            x = x[:, 0]  # Take class token
            x = self.pre_logits(x)
            return self.head(x)

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    
    Converts images to patches and projects them to embedding space.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        # Convolutional patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of patch embedding.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            patches: Embedded patches of shape (batch_size, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Convolutional embedding
        x = self.proj(x)  # (batch_size, embed_dim, grid_h, grid_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        
        return x

class Block(nn.Module):
    """
    Transformer Block.
    
    Contains multi-head self-attention and MLP layers.
    """
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False,
                 drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0,
                 act_layer: nn.Module = nn.GELU, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, dim)
        
        Returns:
            output: Output tensor
        """
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

class Attention(nn.Module):
    """
    Multi-head Self-Attention.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of attention.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, dim)
        
        Returns:
            output: Attention output
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MLP(nn.Module):
    """
    MLP as used in Vision Transformer.
    """
    
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLP."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    
    def __init__(self, drop_prob: float = None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with drop path."""
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

def create_vit_model(variant: str = 'base', num_classes: int = 1000, img_size: int = 224):
    """
    Create Vision Transformer model.
    
    Args:
        variant: Model variant ('tiny', 'small', 'base', 'large', 'huge')
        num_classes: Number of classes
        img_size: Input image size
    
    Returns:
        model: Vision Transformer model
    """
    configs = {
        'tiny': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
        'huge': {'embed_dim': 1280, 'depth': 32, 'num_heads': 16}
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}")
    
    config = configs[variant]
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0
    )
    
    return model

# Import required modules
from functools import partial
from collections import OrderedDict

if __name__ == "__main__":
    # Example usage
    model = create_vit_model('base', num_classes=1000)
    
    # Create dummy input
    batch_size = 4
    channels = 3
    height = width = 224
    x = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Vision Transformer implementation created successfully!") 
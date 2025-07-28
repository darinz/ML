# Multi-Head Self-Attention for Vision Transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-Attention for Vision Transformers.
    
    This implements the standard multi-head attention mechanism
    used in Vision Transformers.
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, dim)
            mask: Optional attention mask
        
        Returns:
            output: Attention output
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class RelativePositionAttention(nn.Module):
    """
    Relative Position Attention.
    
    Incorporates relative positional information in attention.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, max_relative_position: int = 32):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.max_relative_position = max_relative_position
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Relative position embeddings
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_relative_position - 1) * num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Generate relative position index
        coords_h = torch.arange(max_relative_position)
        coords_w = torch.arange(max_relative_position)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += max_relative_position - 1
        relative_coords[:, :, 1] += max_relative_position - 1
        relative_coords[:, :, 0] *= 2 * max_relative_position - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of relative position attention.
        
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
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class LocalAttention(nn.Module):
    """
    Local Attention.
    
    Restricts attention to local windows for efficiency.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7,
                 qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Define a simple local window
        self.register_buffer("attn_mask", self._create_attn_mask())
    
    def _create_attn_mask(self) -> torch.Tensor:
        """Create attention mask for local windows."""
        H, W = self.window_size, self.window_size
        mask = torch.zeros((H * W, H * W))
        
        # Allow attention within local window
        for i in range(H * W):
            for j in range(H * W):
                if abs(i // W - j // W) <= 1 and abs(i % W - j % W) <= 1:
                    mask[i, j] = 1
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of local attention.
        
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
        
        # Apply local attention mask
        mask = self.attn_mask.to(attn.device)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class AxialAttention(nn.Module):
    """
    Axial Attention.
    
    Applies attention along different axes separately.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Separate attention for height and width
        self.qkv_h = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_w = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        Forward pass of axial attention.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, dim)
            height: Height of the feature map
            width: Width of the feature map
        
        Returns:
            output: Attention output
        """
        B, N, C = x.shape
        assert N == height * width, f"Number of patches {N} doesn't match height*width {height*width}"
        
        # Reshape to spatial dimensions
        x = x.view(B, height, width, C)
        
        # Height attention
        x_h = x.transpose(1, 2)  # (B, W, H, C)
        x_h = x_h.reshape(B * width, height, C)
        qkv_h = self.qkv_h(x_h).reshape(B * width, height, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        attn_h = self.attn_drop(attn_h)
        
        x_h = (attn_h @ v_h).transpose(1, 2).reshape(B, width, height, C)
        x_h = x_h.transpose(1, 2)  # (B, H, W, C)
        
        # Width attention
        x_w = x_h.reshape(B * height, width, C)
        qkv_w = self.qkv_w(x_w).reshape(B * height, width, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_w, k_w, v_w = qkv_w[0], qkv_w[1], qkv_w[2]
        
        attn_w = (q_w @ k_w.transpose(-2, -1)) * self.scale
        attn_w = attn_w.softmax(dim=-1)
        attn_w = self.attn_drop(attn_w)
        
        x_w = (attn_w @ v_w).transpose(1, 2).reshape(B, height, width, C)
        
        # Reshape back to patches
        x = x_w.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class SparseAttention(nn.Module):
    """
    Sparse Attention.
    
    Uses sparse attention patterns for efficiency.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False,
                 attn_drop: float = 0.0, proj_drop: float = 0.0, sparsity: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.sparsity = sparsity
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of sparse attention.
        
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
        
        # Apply sparsity mask
        if self.training:
            # During training, randomly mask attention
            mask = torch.rand_like(attn) > self.sparsity
            attn = attn.masked_fill(mask == 0, float('-inf'))
        else:
            # During inference, keep top-k attention weights
            top_k = int(N * (1 - self.sparsity))
            top_k_values, _ = torch.topk(attn, k=top_k, dim=-1)
            threshold = top_k_values[:, :, :, -1:]
            attn = attn.masked_fill(attn < threshold, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class LinearAttention(nn.Module):
    """
    Linear Attention.
    
    Approximates attention with linear complexity.
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
        Forward pass of linear attention.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, dim)
        
        Returns:
            output: Attention output
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Linear attention approximation
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Compute attention efficiently
        kv = k.transpose(-2, -1) @ v  # (B, num_heads, head_dim, head_dim)
        qkv = q @ kv  # (B, num_heads, N, head_dim)
        
        # Normalize
        k_sum = k.sum(dim=-2, keepdim=True)  # (B, num_heads, 1, head_dim)
        qk = q @ k_sum.transpose(-2, -1)  # (B, num_heads, N, 1)
        qkv = qkv / (qk + 1e-8)
        
        x = qkv.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

def create_attention(attention_type: str = 'standard', **kwargs):
    """
    Create attention module based on type.
    
    Args:
        attention_type: Type of attention ('standard', 'relative', 'local', 'axial', 'sparse', 'linear')
        **kwargs: Additional arguments for the attention module
    
    Returns:
        attention: Attention module
    """
    if attention_type == 'standard':
        return MultiHeadAttention(**kwargs)
    elif attention_type == 'relative':
        return RelativePositionAttention(**kwargs)
    elif attention_type == 'local':
        return LocalAttention(**kwargs)
    elif attention_type == 'axial':
        return AxialAttention(**kwargs)
    elif attention_type == 'sparse':
        return SparseAttention(**kwargs)
    elif attention_type == 'linear':
        return LinearAttention(**kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

def visualize_attention_weights(attention_weights: torch.Tensor, 
                              patch_size: int = 16, img_size: int = 224) -> torch.Tensor:
    """
    Visualize attention weights as an image.
    
    Args:
        attention_weights: Attention weights of shape (num_patches, num_patches)
        patch_size: Size of patches
        img_size: Size of original image
    
    Returns:
        attention_map: Visualized attention map
    """
    num_patches = attention_weights.shape[0]
    grid_size = int(math.sqrt(num_patches))
    
    # Reshape to grid
    attention_weights = attention_weights.view(grid_size, grid_size, grid_size, grid_size)
    
    # Upsample to image size
    attention_map = F.interpolate(
        attention_weights.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode='bilinear',
        align_corners=False
    )
    
    return attention_map.squeeze()

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    num_patches = 196  # 14x14 patches
    dim = 768
    num_heads = 12
    
    # Create input
    x = torch.randn(batch_size, num_patches, dim)
    
    # Test different attention mechanisms
    attention_modules = {
        'standard': MultiHeadAttention(dim, num_heads),
        'relative': RelativePositionAttention(dim, num_heads),
        'local': LocalAttention(dim, num_heads, window_size=7),
        'sparse': SparseAttention(dim, num_heads, sparsity=0.5),
        'linear': LinearAttention(dim, num_heads)
    }
    
    for name, attention in attention_modules.items():
        output = attention(x)
        print(f"{name} attention output shape: {output.shape}")
    
    # Test axial attention
    height = width = 14
    axial_attention = AxialAttention(dim, num_heads)
    output = axial_attention(x, height, width)
    print(f"axial attention output shape: {output.shape}")
    
    print("Attention implementations created successfully!") 
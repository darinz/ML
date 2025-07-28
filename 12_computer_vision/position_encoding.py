# Position Encoding for Vision Transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import numpy as np

class SinusoidalPositionEncoding(nn.Module):
    """
    Sinusoidal Position Encoding.
    
    Standard sinusoidal position encoding used in transformers.
    """
    
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
        """
        Add sinusoidal position encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x with position encoding added
        """
        return x + self.pe[:, :x.size(1)]

class LearnedPositionEncoding(nn.Module):
    """
    Learned Position Encoding.
    
    Learnable position embeddings for vision transformers.
    """
    
    def __init__(self, num_patches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned position encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, d_model)
        
        Returns:
            x with position encoding added
        """
        return self.dropout(x + self.pos_embed)

class RelativePositionEncoding(nn.Module):
    """
    Relative Position Encoding.
    
    Encodes relative positions between patches.
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Relative position embeddings
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_relative_position - 1) * 2, d_model)
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
        Add relative position encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, d_model)
        
        Returns:
            x with relative position encoding added
        """
        B, N, C = x.shape
        
        # Get relative position embeddings
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        
        # Add to input
        x = x + relative_position_bias.unsqueeze(0)
        
        return x

class RotaryPositionEncoding(nn.Module):
    """
    Rotary Position Encoding (RoPE).
    
    Applies rotation matrices to embeddings based on position.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create rotation matrices
        self._create_rotation_matrices()
    
    def _create_rotation_matrices(self):
        """Create rotation matrices for RoPE."""
        position = torch.arange(0, self.max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                           -(math.log(10000.0) / self.d_model))
        
        # Create rotation angles
        angles = position * div_term.unsqueeze(0)
        
        # Create rotation matrices
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        
        # Reshape for broadcasting
        self.register_buffer('cos_angles', cos_angles.unsqueeze(0))  # (1, max_len, d_model//2)
        self.register_buffer('sin_angles', sin_angles.unsqueeze(0))  # (1, max_len, d_model//2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x with rotary position encoding applied
        """
        batch_size, seq_len, d_model = x.shape
        
        # Split into even and odd dimensions
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        
        # Apply rotation
        cos_angles = self.cos_angles[:, :seq_len, :]
        sin_angles = self.sin_angles[:, :seq_len, :]
        
        x_rotated_even = x_even * cos_angles - x_odd * sin_angles
        x_rotated_odd = x_even * sin_angles + x_odd * cos_angles
        
        # Interleave back
        x_rotated = torch.zeros_like(x)
        x_rotated[:, :, 0::2] = x_rotated_even
        x_rotated[:, :, 1::2] = x_rotated_odd
        
        return x_rotated

class ALiBiPositionEncoding(nn.Module):
    """
    Attention with Linear Biases (ALiBi).
    
    Adds learned linear biases to attention scores.
    """
    
    def __init__(self, num_heads: int, max_len: int = 5000):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        
        # Create ALiBi slopes
        slopes = torch.Tensor(self._get_slopes(num_heads))
        self.register_buffer('slopes', slopes)
        
        # Create position indices
        position_indices = torch.arange(max_len)
        self.register_buffer('position_indices', position_indices)
    
    def _get_slopes(self, num_heads: int) -> list:
        """Get ALiBi slopes for different heads."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            return (get_slopes_power_of_2(closest_power_of_2) + 
                   self._get_slopes(2*closest_power_of_2)[0::2][:num_heads-closest_power_of_2])
    
    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Add ALiBi biases to attention scores.
        
        Args:
            attention_scores: Attention scores of shape (batch_size, num_heads, seq_len, seq_len)
        
        Returns:
            attention_scores with ALiBi biases added
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # Create position matrix
        positions = torch.arange(seq_len, device=attention_scores.device)
        position_matrix = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Add ALiBi biases
        alibi_biases = self.slopes.unsqueeze(1).unsqueeze(1) * position_matrix.unsqueeze(0)
        attention_scores = attention_scores + alibi_biases
        
        return attention_scores

class T5PositionEncoding(nn.Module):
    """
    T5-style Position Encoding.
    
    Uses relative position embeddings like in T5.
    """
    
    def __init__(self, d_model: int, max_relative_distance: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance
        
        # Relative position embeddings
        self.relative_attention_bias = nn.Embedding(
            2 * max_relative_distance + 1, d_model
        )
        
        # Initialize embeddings
        nn.init.normal_(self.relative_attention_bias.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add T5-style position encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x with T5 position encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create relative position indices
        positions = torch.arange(seq_len, device=x.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions = relative_positions.clamp(-self.max_relative_distance, self.max_relative_distance)
        relative_positions = relative_positions + self.max_relative_distance
        
        # Get relative position embeddings
        relative_embeddings = self.relative_attention_bias(relative_positions)
        
        # Add to input
        x = x + relative_embeddings
        
        return x

class VisionPositionEncoding(nn.Module):
    """
    Vision-specific Position Encoding.
    
    Combines 2D spatial information with sequence position.
    """
    
    def __init__(self, d_model: int, grid_size: int = 14, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.grid_size = grid_size
        self.dropout = nn.Dropout(p=dropout)
        
        # 2D position embeddings
        self.row_embed = nn.Parameter(torch.randn(1, grid_size, d_model // 2))
        self.col_embed = nn.Parameter(torch.randn(1, grid_size, d_model // 2))
        
        # Initialize embeddings
        nn.init.trunc_normal_(self.row_embed, std=0.02)
        nn.init.trunc_normal_(self.col_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add vision-specific position encoding.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches, d_model)
        
        Returns:
            x with vision position encoding added
        """
        batch_size, num_patches, d_model = x.shape
        grid_size = int(math.sqrt(num_patches))
        
        # Reshape to grid
        x = x.view(batch_size, grid_size, grid_size, d_model)
        
        # Add row and column embeddings
        row_embeddings = self.row_embed.expand(batch_size, -1, grid_size, -1)
        col_embeddings = self.col_embed.expand(batch_size, grid_size, -1, -1)
        
        # Concatenate row and column embeddings
        pos_embeddings = torch.cat([row_embeddings, col_embeddings], dim=-1)
        
        # Add to input
        x = x + pos_embeddings
        x = self.dropout(x)
        
        # Reshape back to patches
        x = x.view(batch_size, num_patches, d_model)
        
        return x

def create_position_encoding(encoding_type: str = 'learned', **kwargs):
    """
    Create position encoding based on type.
    
    Args:
        encoding_type: Type of position encoding ('sinusoidal', 'learned', 'relative', 'rotary', 'alibi', 't5', 'vision')
        **kwargs: Additional arguments for the position encoding
    
    Returns:
        position_encoding: Position encoding module
    """
    if encoding_type == 'sinusoidal':
        return SinusoidalPositionEncoding(**kwargs)
    elif encoding_type == 'learned':
        return LearnedPositionEncoding(**kwargs)
    elif encoding_type == 'relative':
        return RelativePositionEncoding(**kwargs)
    elif encoding_type == 'rotary':
        return RotaryPositionEncoding(**kwargs)
    elif encoding_type == 'alibi':
        return ALiBiPositionEncoding(**kwargs)
    elif encoding_type == 't5':
        return T5PositionEncoding(**kwargs)
    elif encoding_type == 'vision':
        return VisionPositionEncoding(**kwargs)
    else:
        raise ValueError(f"Unknown position encoding type: {encoding_type}")

def interpolate_position_encoding(pos_embed: torch.Tensor, 
                                num_patches: int, 
                                num_extra_tokens: int = 1) -> torch.Tensor:
    """
    Interpolate position embeddings for different sequence lengths.
    
    Args:
        pos_embed: Position embeddings
        num_patches: Number of patches
        num_extra_tokens: Number of extra tokens (e.g., class token)
    
    Returns:
        interpolated_pos_embed: Interpolated position embeddings
    """
    pos_embed = pos_embed.reshape(1, -1, pos_embed.shape[-1])
    
    # Interpolate to target length
    target_length = num_patches + num_extra_tokens
    interpolated_pos_embed = F.interpolate(
        pos_embed.transpose(1, 2),
        size=target_length,
        mode='linear',
        align_corners=False
    ).transpose(1, 2)
    
    return interpolated_pos_embed

def create_2d_position_encoding(height: int, width: int, d_model: int) -> torch.Tensor:
    """
    Create 2D position encoding for vision tasks.
    
    Args:
        height: Height of the feature map
        width: Width of the feature map
        d_model: Embedding dimension
    
    Returns:
        pos_encoding: 2D position encoding
    """
    pos_encoding = torch.zeros(1, height * width, d_model)
    
    # Create 2D grid
    y_pos = torch.arange(height).float()
    x_pos = torch.arange(width).float()
    
    # Normalize positions
    y_pos = y_pos / (height - 1) * 2 - 1
    x_pos = x_pos / (width - 1) * 2 - 1
    
    # Create position embeddings
    for i in range(height):
        for j in range(width):
            pos = i * width + j
            pos_encoding[0, pos, :d_model//2] = y_pos[i]
            pos_encoding[0, pos, d_model//2:] = x_pos[j]
    
    return pos_encoding

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    num_patches = 196  # 14x14 patches
    d_model = 768
    
    # Create input
    x = torch.randn(batch_size, num_patches, d_model)
    
    # Test different position encodings
    position_encodings = {
        'sinusoidal': SinusoidalPositionEncoding(d_model),
        'learned': LearnedPositionEncoding(num_patches, d_model),
        'relative': RelativePositionEncoding(d_model),
        'rotary': RotaryPositionEncoding(d_model),
        'vision': VisionPositionEncoding(d_model, grid_size=14)
    }
    
    for name, pos_encoding in position_encodings.items():
        output = pos_encoding(x)
        print(f"{name} position encoding output shape: {output.shape}")
    
    # Test ALiBi
    num_heads = 12
    alibi = ALiBiPositionEncoding(num_heads)
    attention_scores = torch.randn(batch_size, num_heads, num_patches, num_patches)
    output = alibi(attention_scores)
    print(f"ALiBi output shape: {output.shape}")
    
    print("Position encoding implementations created successfully!") 
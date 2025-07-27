"""
Positional Encoding Implementation
=================================

This module provides various positional encoding methods for transformer models,
including sinusoidal, learned, and rotary positional encodings.
"""

import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as used in the original transformer.
    
    This provides the model with information about the position of tokens
    in the sequence using sine and cosine functions of different frequencies.
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
        Add sinusoidal positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x + positional_encoding: Tensor with positional information
        """
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding.
    
    This learns positional embeddings as parameters of the model,
    which can be more flexible than sinusoidal encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x + positional_encoding: Tensor with positional information
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_encoding = self.pe(positions)
        return x + pos_encoding


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    This applies rotation matrices to embeddings based on position,
    which provides better extrapolation to longer sequences.
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
        Apply rotary positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x with rotary positional encoding applied
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


class RelativePositionalEncoding(nn.Module):
    """
    Relative positional encoding.
    
    This encodes relative distances between positions rather than
    absolute positions, which can be more efficient for long sequences.
    """
    
    def __init__(self, d_model: int, max_relative_position: int = 32):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # Create relative position embeddings
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply relative positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x with relative positional encoding applied
        """
        batch_size, seq_len, d_model = x.shape
        
        # Create relative position indices
        range_vec = torch.arange(seq_len, device=x.device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Shift to non-negative indices
        distance_mat_clipped += self.max_relative_position
        
        # Get relative position embeddings
        relative_position_embeddings = self.relative_position_embeddings(
            distance_mat_clipped
        )
        
        # Add to input
        return x + relative_position_embeddings


class ALiBiPositionalEncoding(nn.Module):
    """
    Attention with Linear Biases (ALiBi) positional encoding.
    
    This adds learned biases to attention scores based on relative positions,
    which can help with extrapolation to longer sequences.
    """
    
    def __init__(self, num_heads: int, max_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        
        # Create ALiBi slopes
        slopes = torch.Tensor(self._get_slopes(num_heads))
        self.register_buffer('slopes', slopes)
        
        # Create bias matrix
        bias = torch.arange(max_len).unsqueeze(0).repeat(max_len, 1)
        bias = bias - bias.transpose(0, 1)
        bias = bias.unsqueeze(0).unsqueeze(0)  # (1, 1, max_len, max_len)
        self.register_buffer('bias', bias)
    
    def _get_slopes(self, num_heads: int) -> list:
        """Get ALiBi slopes for different attention heads."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(slopes[:num_heads-closest_power_of_2])
            return slopes
    
    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Add ALiBi biases to attention scores.
        
        Args:
            attention_scores: Attention scores of shape (batch_size, num_heads, seq_len, seq_len)
        
        Returns:
            attention_scores with ALiBi biases added
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # Get bias for current sequence length
        bias = self.bias[:, :, :seq_len, :seq_len]
        
        # Add slopes to bias
        slopes = self.slopes[:num_heads].view(1, num_heads, 1, 1)
        bias = bias * slopes
        
        return attention_scores + bias


class T5PositionalEncoding(nn.Module):
    """
    T5-style relative positional encoding.
    
    This uses a simplified relative positional encoding scheme
    that only considers relative distances up to a certain threshold.
    """
    
    def __init__(self, d_model: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.d_model = d_model
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        
        # Create relative position embeddings
        self.relative_attention_bias = nn.Embedding(num_buckets, 1)
    
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """
        Convert relative positions to buckets.
        
        Args:
            relative_position: Relative position tensor
        
        Returns:
            bucket indices
        """
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        
        # Handle positions beyond max_distance
        relative_buckets = 0
        if relative_position > 0:
            relative_buckets = num_buckets // 2
        else:
            relative_position = -relative_position
        
        # Use log space for positions within max_distance
        is_small = relative_position < max_distance
        relative_position_if_large = max_distance + torch.log(relative_position.float() / max_distance) / math.log(max_distance / num_buckets) * (num_buckets - max_distance)
        relative_position_if_large = torch.clamp(relative_position_if_large, max=max_distance - 1)
        relative_buckets_if_large = relative_buckets + relative_position_if_large
        
        return torch.where(is_small, relative_position + relative_buckets, relative_buckets_if_large)
    
    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """
        Add T5 relative positional encoding to attention scores.
        
        Args:
            attention_scores: Attention scores of shape (batch_size, num_heads, seq_len, seq_len)
        
        Returns:
            attention_scores with relative positional encoding added
        """
        batch_size, num_heads, seq_len, _ = attention_scores.shape
        
        # Create relative position matrix
        context_position = torch.arange(seq_len, device=attention_scores.device)
        memory_position = torch.arange(seq_len, device=attention_scores.device)
        relative_position = memory_position.unsqueeze(0) - context_position.unsqueeze(1)
        
        # Convert to buckets
        relative_position_bucket = self._relative_position_bucket(relative_position)
        
        # Get bias values
        bias = self.relative_attention_bias(relative_position_bucket).squeeze(-1)
        
        # Add bias to attention scores
        return attention_scores + bias.unsqueeze(0).unsqueeze(0)


def create_positional_encoding(encoding_type: str, d_model: int, **kwargs) -> nn.Module:
    """
    Factory function to create positional encoding modules.
    
    Args:
        encoding_type: Type of positional encoding ('sinusoidal', 'learned', 'rotary', 'relative', 'alibi', 't5')
        d_model: Model dimension
        **kwargs: Additional arguments for specific encoding types
    
    Returns:
        Positional encoding module
    """
    if encoding_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(d_model, **kwargs)
    elif encoding_type == 'learned':
        return LearnedPositionalEncoding(d_model, **kwargs)
    elif encoding_type == 'rotary':
        return RotaryPositionalEncoding(d_model, **kwargs)
    elif encoding_type == 'relative':
        return RelativePositionalEncoding(d_model, **kwargs)
    elif encoding_type == 'alibi':
        return ALiBiPositionalEncoding(**kwargs)
    elif encoding_type == 't5':
        return T5PositionalEncoding(d_model, **kwargs)
    else:
        raise ValueError(f"Unknown positional encoding type: {encoding_type}")


# Example usage
if __name__ == "__main__":
    # Test different positional encodings
    batch_size, seq_len, d_model = 2, 10, 512
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test sinusoidal encoding
    sinusoidal_pe = SinusoidalPositionalEncoding(d_model)
    x_sinusoidal = sinusoidal_pe(x)
    print(f"Sinusoidal PE output shape: {x_sinusoidal.shape}")
    
    # Test learned encoding
    learned_pe = LearnedPositionalEncoding(d_model)
    x_learned = learned_pe(x)
    print(f"Learned PE output shape: {x_learned.shape}")
    
    # Test rotary encoding
    rotary_pe = RotaryPositionalEncoding(d_model)
    x_rotary = rotary_pe(x)
    print(f"Rotary PE output shape: {x_rotary.shape}")
    
    # Test relative encoding
    relative_pe = RelativePositionalEncoding(d_model)
    x_relative = relative_pe(x)
    print(f"Relative PE output shape: {x_relative.shape}")
    
    # Test ALiBi encoding
    num_heads = 8
    attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    alibi_pe = ALiBiPositionalEncoding(num_heads)
    scores_with_alibi = alibi_pe(attention_scores)
    print(f"ALiBi PE output shape: {scores_with_alibi.shape}")
    
    # Test T5 encoding
    t5_pe = T5PositionalEncoding(d_model)
    scores_with_t5 = t5_pe(attention_scores)
    print(f"T5 PE output shape: {scores_with_t5.shape}")
    
    # Test factory function
    factory_pe = create_positional_encoding('sinusoidal', d_model)
    x_factory = factory_pe(x)
    print(f"Factory PE output shape: {x_factory.shape}") 
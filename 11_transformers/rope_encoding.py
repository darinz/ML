"""
Rotary Position Embedding (RoPE) Implementation
===============================================

This module provides implementations of Rotary Position Embedding (RoPE),
a modern positional encoding technique that provides better extrapolation
to longer sequences.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation.
    
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


class RotaryEmbedding(nn.Module):
    """
    Rotary embedding for attention mechanisms.
    
    This applies rotary transformations to queries and keys
    in attention computations.
    """
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Generate rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for queries and keys.
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, dim)
            seq_len: Optional sequence length override
        
        Returns:
            cos_emb: Cosine embeddings
            sin_emb: Sine embeddings
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        return cos_emb, sin_emb


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    Args:
        x: Input tensor
    
    Returns:
        Rotated tensor
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to queries and keys.
    
    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine embeddings
        sin: Sine embeddings
    
    Returns:
        q_rotated: Rotated query tensor
        k_rotated: Rotated key tensor
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryAttention(nn.Module):
    """
    Attention module with rotary positional encoding.
    
    This integrates RoPE directly into the attention mechanism.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(self.d_k)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with rotary attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: Attention output
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply rotary embeddings
        cos_emb, sin_emb = self.rotary_emb(Q, seq_len)
        Q_rot, K_rot = apply_rotary_pos_emb(Q, K, cos_emb, sin_emb)
        
        # Compute attention
        scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = torch.dropout(attention_weights, p=self.dropout, train=self.training)
        
        # Compute output
        output = torch.matmul(attention_weights, V)
        
        # Reshape and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output


class DynamicRotaryEmbedding(nn.Module):
    """
    Dynamic rotary embedding that can handle variable sequence lengths.
    
    This generates rotary embeddings on-the-fly for any sequence length.
    """
    
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Generate rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dynamic rotary embeddings.
        
        Args:
            x: Input tensor
            seq_len: Sequence length (inferred from x if not provided)
        
        Returns:
            cos_emb: Cosine embeddings
            sin_emb: Sine embeddings
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        return cos_emb, sin_emb


class RelativeRotaryEmbedding(nn.Module):
    """
    Relative rotary embedding for relative position encoding.
    
    This applies rotary transformations based on relative positions
    rather than absolute positions.
    """
    
    def __init__(self, dim: int, max_relative_position: int = 32, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_relative_position = max_relative_position
        self.base = base
        
        # Generate rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute relative rotary embeddings.
        
        Args:
            x: Input tensor
        
        Returns:
            cos_emb: Cosine embeddings
            sin_emb: Sine embeddings
        """
        seq_len = x.shape[-2]
        
        # Create relative position matrix
        range_vec = torch.arange(seq_len, device=x.device)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # Compute frequencies
        freqs = torch.einsum("i,j->ij", distance_mat_clipped.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        return cos_emb, sin_emb


def create_rotary_encoding(encoding_type: str, dim: int, **kwargs) -> nn.Module:
    """
    Factory function to create rotary encoding modules.
    
    Args:
        encoding_type: Type of rotary encoding ('rotary', 'dynamic', 'relative')
        dim: Embedding dimension
        **kwargs: Additional arguments
    
    Returns:
        Rotary encoding module
    """
    if encoding_type == 'rotary':
        return RotaryEmbedding(dim, **kwargs)
    elif encoding_type == 'dynamic':
        return DynamicRotaryEmbedding(dim, **kwargs)
    elif encoding_type == 'relative':
        return RelativeRotaryEmbedding(dim, **kwargs)
    else:
        raise ValueError(f"Unknown rotary encoding type: {encoding_type}")


# Example usage
if __name__ == "__main__":
    # Test rotary positional encoding
    batch_size, seq_len, d_model = 2, 10, 512
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test basic RoPE
    rope = RotaryPositionalEncoding(d_model)
    x_rope = rope(x)
    print(f"RoPE output shape: {x_rope.shape}")
    
    # Test rotary attention
    rotary_attention = RotaryAttention(d_model, num_heads=8)
    attention_output = rotary_attention(x)
    print(f"Rotary attention output shape: {attention_output.shape}")
    
    # Test dynamic rotary embedding
    dynamic_rope = DynamicRotaryEmbedding(d_model // 8)  # For attention heads
    cos_emb, sin_emb = dynamic_rope(x)
    print(f"Dynamic RoPE cos shape: {cos_emb.shape}")
    print(f"Dynamic RoPE sin shape: {sin_emb.shape}")
    
    # Test relative rotary embedding
    relative_rope = RelativeRotaryEmbedding(d_model // 8)
    rel_cos_emb, rel_sin_emb = relative_rope(x)
    print(f"Relative RoPE cos shape: {rel_cos_emb.shape}")
    print(f"Relative RoPE sin shape: {rel_sin_emb.shape}")
    
    # Test factory function
    factory_rope = create_rotary_encoding('rotary', d_model // 8)
    factory_cos, factory_sin = factory_rope(x)
    print(f"Factory RoPE cos shape: {factory_cos.shape}")
    print(f"Factory RoPE sin shape: {factory_sin.shape}") 
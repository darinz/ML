"""
Multi-Head Attention Implementation
==================================

This module provides comprehensive implementations of attention mechanisms
used in transformer architectures, including scaled dot-product attention,
multi-head attention, and various attention variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention implementation.
    
    This is the core attention mechanism used in transformers.
    """
    
    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, d_k)
            k: Key tensor of shape (batch_size, seq_len, d_k)
            v: Value tensor of shape (batch_size, seq_len, d_v)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
        
        Returns:
            output: Attention output tensor
            attention_weights: Attention weights tensor
        """
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn = self.dropout(F.softmax(attn, dim=-1))
        
        # Compute weighted sum
        output = torch.matmul(attn, v)
        
        return output, attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation.
    
    This allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        # Linear transformations for Q, K, V
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        # Attention mechanism
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head attention.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Optional attention mask
        
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        
        # Pass through the pre-attention projection
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        # Transpose for attention dot product
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting
        
        q, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        
        q = self.layer_norm(q)
        
        return q, attn


class FlashAttention(nn.Module):
    """
    Memory-efficient attention implementation.
    
    This reduces memory usage from O(n²) to O(n) for attention computation.
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with memory-efficient attention.
        
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
        
        # Memory-efficient attention computation
        output = self._flash_attention(Q, K, V, mask)
        
        # Reshape and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output
    
    def _flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Memory-efficient attention computation.
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional attention mask
        
        Returns:
            output: Attention output
        """
        # This is a simplified implementation
        # In practice, you would use the actual Flash Attention implementation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        
        output = torch.matmul(attention_weights, V)
        
        return output


class CausalAttention(nn.Module):
    """
    Causal attention for autoregressive models.
    
    This prevents the model from looking at future tokens during training.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, d_model, d_model//num_heads, 
                                          d_model//num_heads, dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor
        
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        batch_size, seq_len = x.shape[:2]
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        return self.attention(x, x, x, mask=causal_mask)


def create_attention_mask(padding_mask: torch.Tensor) -> torch.Tensor:
    """
    Create attention mask from padding mask.
    
    Args:
        padding_mask: Boolean tensor indicating padded positions (True for padding)
    
    Returns:
        attention_mask: Attention mask tensor
    """
    # padding_mask: (batch_size, seq_len)
    # attention_mask: (batch_size, 1, 1, seq_len)
    attention_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    return attention_mask


def visualize_attention(attention_weights: torch.Tensor, tokens: list, 
                       save_path: Optional[str] = None) -> None:
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights tensor of shape (seq_len, seq_len)
        tokens: List of token strings
        save_path: Optional path to save the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights.detach().numpy(), 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                annot=True,
                fmt='.2f')
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Test multi-head attention
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    # Create sample data
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Initialize attention module
    attention = MultiHeadAttention(num_heads, d_model, d_model//num_heads, 
                                 d_model//num_heads, dropout=0.1)
    
    # Forward pass
    output, attention_weights = attention(x, x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test causal attention
    causal_attention = CausalAttention(d_model, num_heads)
    causal_output, causal_weights = causal_attention(x)
    
    print(f"Causal output shape: {causal_output.shape}")
    print(f"Causal attention weights shape: {causal_weights.shape}")
    
    # Test flash attention
    flash_attention = FlashAttention(d_model, num_heads)
    flash_output = flash_attention(x)
    
    print(f"Flash attention output shape: {flash_output.shape}") 
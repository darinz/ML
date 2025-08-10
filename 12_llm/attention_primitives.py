"""
Basic Attention Primitives.

This module implements the fundamental attention mechanisms including
scaled dot-product attention and simple self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention implementation.
    
    This is the core attention mechanism that computes attention weights
    based on the similarity between queries and keys, then applies them to values.
    """
    
    def __init__(self, temperature, attn_dropout=0.1):
        """
        Initialize Scaled Dot-Product Attention.
        
        Args:
            temperature: Scaling factor (usually sqrt(d_k))
            attn_dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Forward pass of scaled dot-product attention.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, d_k)
            k: Key tensor of shape (batch_size, seq_len, d_k)
            v: Value tensor of shape (batch_size, seq_len, d_v)
            mask: Optional mask tensor of shape (batch_size, seq_len, seq_len)
            
        Returns:
            output: Weighted sum of values
            attn: Attention weights
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

def simple_self_attention(x, d_k=64):
    """
    Simple self-attention implementation for demonstration.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)
        d_k: Dimension of keys and queries
        
    Returns:
        output: Self-attention output
        attention_weights: Attention weight matrix
    """
    batch_size, seq_len, d_model = x.shape
    
    # Create Q, K, V (simplified - in practice, these would be learned)
    Q = nn.Linear(d_model, d_k)(x)
    K = nn.Linear(d_model, d_k)(x)
    V = nn.Linear(d_model, d_model)(x)
    
    # Compute attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

def create_causal_mask(seq_len):
    """
    Create causal mask for autoregressive attention.
    
    Args:
        seq_len: Length of sequence
        
    Returns:
        mask: Causal mask tensor
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask

def apply_attention_mask(attention_scores, mask):
    """
    Apply attention mask to attention scores.
    
    Args:
        attention_scores: Raw attention scores
        mask: Mask tensor (1 for allowed, 0 for masked)
        
    Returns:
        masked_scores: Attention scores with mask applied
    """
    return attention_scores.masked_fill(mask == 1, float('-inf'))

# Example usage functions
def demonstrate_attention_primitives():
    """Demonstrate basic attention primitives."""
    print("Attention Primitives Demonstration")
    print("=" * 40)
    
    # Example 1: Scaled Dot-Product Attention
    print("1. Scaled Dot-Product Attention")
    batch_size, seq_len, d_k, d_v = 2, 10, 64, 64
    
    # Create random Q, K, V
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_v)
    
    # Initialize attention
    attention = ScaledDotProductAttention(temperature=math.sqrt(d_k))
    
    # Apply attention
    output, attn_weights = attention(q, k, v)
    print(f"   Input shapes: Q{q.shape}, K{k.shape}, V{v.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    print()
    
    # Example 2: Simple Self-Attention
    print("2. Simple Self-Attention")
    x = torch.randn(2, 10, 512)  # batch_size=2, seq_len=10, d_model=512
    output, weights = simple_self_attention(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Attention weights shape: {weights.shape}")
    print()
    
    # Example 3: Causal Masking
    print("3. Causal Masking")
    seq_len = 5
    causal_mask = create_causal_mask(seq_len)
    print(f"   Causal mask for sequence length {seq_len}:")
    print(f"   {causal_mask}")
    print()
    
    # Example 4: Attention with Masking
    print("4. Attention with Masking")
    attention_scores = torch.randn(1, seq_len, seq_len)
    masked_scores = apply_attention_mask(attention_scores, causal_mask)
    print(f"   Original scores shape: {attention_scores.shape}")
    print(f"   Masked scores shape: {masked_scores.shape}")
    print(f"   Masked scores contain -inf: {torch.isinf(masked_scores).any()}")

if __name__ == "__main__":
    demonstrate_attention_primitives()

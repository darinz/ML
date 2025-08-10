import torch
import torch.nn as nn
import torch.nn.functional as F

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

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


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for transformer architectures.
    
    This module adds positional information to input embeddings using
    sinusoidal functions, allowing the model to understand sequence order.
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize Positional Encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass of positional encoding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Input with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned Positional Embedding as an alternative to sinusoidal encoding.
    
    This module learns positional embeddings as parameters rather than
    using fixed sinusoidal functions.
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Initialize Learned Positional Embedding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x):
        """
        Forward pass of learned positional embedding.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Input with learned positional embedding added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb 
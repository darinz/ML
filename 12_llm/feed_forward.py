import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Feed-Forward Network for transformer architectures.
    
    This module applies position-wise transformations to each position independently,
    typically consisting of two linear transformations with a ReLU activation.
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize Feed-Forward Network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Transformed tensor of same shape as input
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class ResidualConnection(nn.Module):
    """
    Residual Connection with Layer Normalization.
    
    This module implements the residual connection pattern used in transformers,
    combining the input with the output of a sublayer through layer normalization.
    """
    
    def __init__(self, size, dropout=0.1):
        """
        Initialize Residual Connection.
        
        Args:
            size: Size of the layer (d_model)
            dropout: Dropout rate
        """
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor
            sublayer: Sublayer function to apply
            
        Returns:
            output: Residual connection output
        """
        return x + self.dropout(sublayer(self.norm(x)))

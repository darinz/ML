import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention, MultiHeadSelfAttention
from feed_forward import FeedForward

class EncoderLayer(nn.Module):
    """
    Encoder Layer for transformer architectures.
    
    This layer consists of multi-head self-attention followed by a feed-forward network,
    with residual connections and layer normalization.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize Encoder Layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor
            
        Returns:
            output: Encoded tensor of same shape as input
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """
    Decoder Layer for transformer architectures.
    
    This layer consists of masked self-attention, cross-attention to encoder,
    and a feed-forward network, with residual connections and layer normalization.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize Decoder Layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.masked_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Forward pass of decoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Output from encoder of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Mask for target sequence
            src_mask: Mask for source sequence
            
        Returns:
            output: Decoded tensor of same shape as input
        """
        # Masked self-attention
        attn_output, _ = self.masked_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention to encoder
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class DecoderLayerWithCrossAttention(nn.Module):
    """
    Decoder Layer with Cross-Attention for encoder-decoder models.
    
    This is a specialized decoder layer that includes cross-attention
    to the encoder outputs, typically used in sequence-to-sequence models.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize Decoder Layer with Cross-Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.masked_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        """
        Forward pass of decoder layer with cross-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Output from encoder of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Mask for target sequence
            src_mask: Mask for source sequence
            
        Returns:
            output: Decoded tensor of same shape as input
        """
        # Masked self-attention
        attn_output, _ = self.masked_attention(x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention to encoder
        cross_attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

"""
Complete Transformer Architecture Implementation
=============================================

This module provides a complete implementation of the transformer architecture
as described in "Attention Is All You Need", including encoder, decoder,
and the full transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    This provides the model with information about the position of tokens
    in the sequence, which is necessary since transformers process all
    positions in parallel.
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
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x + positional_encoding: Tensor with positional information
        """
        return x + self.pe[:, :x.size(1)]


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    
    This consists of two linear transformations with a ReLU activation
    in between, and is applied to each position separately.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Transformed tensor
        """
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """
    Single encoder layer of the transformer.
    
    This consists of multi-head self-attention followed by a feed-forward
    network, with residual connections and layer normalization.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(num_heads, d_model, d_model//num_heads, 
                                               d_model//num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor
            mask: Optional attention mask
        
        Returns:
            output: Encoder layer output
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer of the transformer.
    
    This consists of masked multi-head self-attention, cross-attention
    to the encoder, and a feed-forward network.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_attention = MultiHeadAttention(num_heads, d_model, d_model//num_heads, 
                                                 d_model//num_heads, dropout)
        self.cross_attention = MultiHeadAttention(num_heads, d_model, d_model//num_heads, 
                                                d_model//num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of decoder layer.
        
        Args:
            x: Input tensor
            encoder_output: Output from encoder
            tgt_mask: Mask for target sequence (causal mask)
            src_mask: Mask for source sequence
        
        Returns:
            output: Decoder layer output
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


class Encoder(nn.Module):
    """
    Complete encoder stack of the transformer.
    
    This consists of multiple encoder layers stacked together.
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            mask: Optional attention mask
        
        Returns:
            output: Encoder output
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


class Decoder(nn.Module):
    """
    Complete decoder stack of the transformer.
    
    This consists of multiple decoder layers stacked together.
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, 
                 num_heads: int, d_ff: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                tgt_mask: Optional[torch.Tensor] = None, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of decoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            encoder_output: Output from encoder
            tgt_mask: Mask for target sequence (causal mask)
            src_mask: Mask for source sequence
        
        Returns:
            output: Decoder output
        """
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        
        return self.norm(x)


class Transformer(nn.Module):
    """
    Complete transformer model.
    
    This implements the full transformer architecture with encoder and decoder.
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, 
                 num_layers: int = 6, num_heads: int = 8, d_ff: int = 2048, 
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Encoder and decoder
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Generate source mask for padding.
        
        Args:
            src: Source tensor
        
        Returns:
            mask: Source mask
        """
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def generate_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Generate target mask for causal attention.
        
        Args:
            tgt: Target tensor
        
        Returns:
            mask: Target mask
        """
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0)
        return tgt_mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of transformer.
        
        Args:
            src: Source tensor of shape (batch_size, src_len)
            tgt: Target tensor of shape (batch_size, tgt_len)
        
        Returns:
            output: Transformer output
        """
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)
        
        # Encode
        encoder_output = self.encoder(src, src_mask)
        
        # Decode
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        return output


class GPTModel(nn.Module):
    """
    GPT-style decoder-only transformer model.
    
    This is used for language modeling and text generation.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 12, 
                 num_heads: int = 12, d_ff: int = 3072, max_len: int = 2048, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of GPT model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
        
        Returns:
            output: Model output
        """
        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        
        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask.unsqueeze(1)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, x, causal_mask, causal_mask)  # Self-attention only
        
        x = self.norm(x)
        return self.output_projection(x)


class BERTModel(nn.Module):
    """
    BERT-style encoder-only transformer model.
    
    This is used for understanding tasks like classification and NER.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 12, 
                 num_heads: int = 12, d_ff: int = 3072, max_len: int = 512, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.segment_embedding = nn.Embedding(2, d_model)  # For sentence pairs
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of BERT model.
        
        Args:
            input_ids: Input token IDs
            token_type_ids: Optional token type IDs for sentence pairs
            attention_mask: Optional attention mask
        
        Returns:
            output: Model output
        """
        # Embedding
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Add segment embeddings if provided
        if token_type_ids is not None:
            x = x + self.segment_embedding(token_type_ids)
        
        x = self.dropout(x)
        
        # Create attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)


# Example usage
if __name__ == "__main__":
    # Test transformer
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    # Create sample data
    src = torch.randint(1, src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))
    
    # Initialize transformer
    transformer = Transformer(src_vocab_size, tgt_vocab_size)
    
    # Forward pass
    output = transformer(src, tgt)
    print(f"Transformer output shape: {output.shape}")
    
    # Test GPT model
    vocab_size = 50000
    seq_len = 20
    
    gpt_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    gpt_model = GPTModel(vocab_size)
    gpt_output = gpt_model(gpt_input)
    print(f"GPT output shape: {gpt_output.shape}")
    
    # Test BERT model
    bert_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    bert_model = BERTModel(vocab_size)
    bert_output = bert_model(bert_input)
    print(f"BERT output shape: {bert_output.shape}") 
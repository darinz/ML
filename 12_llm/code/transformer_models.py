import torch
import torch.nn as nn
import math
from positional_encoding import PositionalEncoding
from encoder_decoder_layers import EncoderLayer, DecoderLayer, DecoderLayerWithCrossAttention
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    
    This implements the original transformer architecture from "Attention Is All You Need"
    with encoder-decoder structure for tasks like machine translation.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, 
                 num_heads=8, d_ff=2048, max_len=512, dropout=0.1):
        """
        Initialize Transformer model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            num_layers: Number of encoder/decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and Decoder
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_src_mask(self, src):
        """Generate source mask for padding tokens."""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def generate_tgt_mask(self, tgt):
        """Generate target mask for causal attention."""
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0)
        return tgt_mask
    
    def forward(self, src, tgt):
        """
        Forward pass of transformer.
        
        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            tgt: Target sequence of shape (batch_size, tgt_seq_len)
            
        Returns:
            output: Logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        src_mask = self.generate_src_mask(src)
        tgt_mask = self.generate_tgt_mask(tgt)
        
        # Encode
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)
        src = self.dropout(src)
        
        for encoder_layer in self.encoder:
            src = encoder_layer(src, src_mask)
        
        # Decode
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        for decoder_layer in self.decoder:
            tgt = decoder_layer(tgt, src, tgt_mask, src_mask)
        
        # Output projection
        output = self.output_projection(tgt)
        return output

class EncoderOnlyTransformer(nn.Module):
    """
    Encoder-only transformer for understanding tasks (BERT-style).
    
    This model is designed for tasks where the model needs to process
    the entire input sequence to understand it.
    """
    
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=8, d_ff=1024, max_len=512, dropout=0.1):
        """
        Initialize Encoder-only Transformer.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Forward pass of encoder-only transformer.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            output: Encoded representations of shape (batch_size, seq_len, d_model)
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)

class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only transformer for generation tasks (GPT-style).
    
    This model is designed for tasks where the model generates text
    autoregressively, one token at a time.
    """
    
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=8, d_ff=1024, max_len=512, dropout=0.1):
        """
        Initialize Decoder-only Transformer.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        """
        Forward pass of decoder-only transformer.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            output: Logits of shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, None, mask, None)  # No encoder output for decoder-only
            
        x = self.norm(x)
        return self.output_projection(x)

class EncoderDecoderTransformer(nn.Module):
    """
    Encoder-decoder transformer for sequence-to-sequence tasks (T5-style).
    
    This model combines an encoder for understanding input with a decoder
    for generating output, typically used for translation, summarization, etc.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=512, dropout=0.1):
        """
        Initialize Encoder-Decoder Transformer.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            num_layers: Number of encoder/decoder layers
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.encoder = EncoderOnlyTransformer(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = DecoderOnlyTransformer(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of encoder-decoder transformer.
        
        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            tgt: Target sequence of shape (batch_size, tgt_seq_len)
            src_mask: Source attention mask
            tgt_mask: Target attention mask
            
        Returns:
            output: Decoder output
        """
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, tgt_mask)
        return decoder_output

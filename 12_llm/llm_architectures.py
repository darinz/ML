"""
Large Language Model Architecture Variants.

This module implements various LLM architecture variants including
GPT-style, BERT-style, and T5-style models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTModel(nn.Module):
    """
    GPT-style autoregressive model for text generation.
    
    Uses decoder-only architecture with causal masking.
    """
    
    def __init__(self, vocab_size, d_model=2048, num_layers=24, num_heads=16, max_len=2048):
        """
        Initialize GPT Model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_len: Maximum sequence length
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Import decoder layer from other module
        from encoder_decoder_layers import DecoderLayer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_model*4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of GPT model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            
        Returns:
            output: Model output logits
        """
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        
        if attention_mask is not None:
            causal_mask = causal_mask * attention_mask.unsqueeze(1)
        
        for layer in self.layers:
            x = layer(x, None, causal_mask, None)  # No encoder output for decoder-only
        
        x = self.norm(x)
        return self.output_projection(x)

class BERTModel(nn.Module):
    """
    BERT-style bidirectional model for understanding tasks.
    
    Uses encoder-only architecture with bidirectional attention.
    """
    
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, max_len=512):
        """
        Initialize BERT Model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_len: Maximum sequence length
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.segment_embedding = nn.Embedding(2, d_model)  # For sentence pairs
        
        # Import encoder layer from other module
        from encoder_decoder_layers import EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_model*4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Forward pass of BERT model.
        
        Args:
            input_ids: Input token IDs
            token_type_ids: Token type IDs for sentence pairs
            attention_mask: Optional attention mask
            
        Returns:
            output: Model output representations
        """
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        if token_type_ids is not None:
            x = x + self.segment_embedding(token_type_ids)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.norm(x)

class T5Model(nn.Module):
    """
    T5-style text-to-text transfer model.
    
    Uses encoder-decoder architecture for sequence-to-sequence tasks.
    """
    
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, max_len=512):
        """
        Initialize T5 Model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of encoder/decoder layers
            num_heads: Number of attention heads
            max_len: Maximum sequence length
        """
        super().__init__()
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Import encoder and decoder layers from other module
        from encoder_decoder_layers import EncoderLayer, DecoderLayerWithCrossAttention
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_model*4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderLayerWithCrossAttention(d_model, num_heads, d_model*4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, target_ids=None, attention_mask=None):
        """
        Forward pass of T5 model.
        
        Args:
            input_ids: Input token IDs
            target_ids: Target token IDs (for training)
            attention_mask: Optional attention mask
            
        Returns:
            output: Model output (logits if training, representations if not)
        """
        # Encode
        x = self.shared_embedding(input_ids)
        x = self.pos_encoding(x)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            mask = None
        
        for layer in self.encoder:
            x = layer(x, mask)
        
        if target_ids is not None:
            # Decode
            y = self.shared_embedding(target_ids)
            y = self.pos_encoding(y)
            
            causal_mask = torch.tril(torch.ones(y.size(1), y.size(1))).unsqueeze(0)
            
            for layer in self.decoder:
                y = layer(y, x, causal_mask, mask)
            
            y = self.norm(y)
            return self.output_projection(y)
        else:
            return x

class SimpleLanguageModel(nn.Module):
    """
    Simple Language Model for demonstration purposes.
    
    A simplified version of a decoder-only language model.
    """
    
    def __init__(self, vocab_size, d_model=256, num_layers=6, num_heads=8, max_len=512):
        """
        Initialize Simple Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_len: Maximum sequence length
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Import decoder layer from other module
        from encoder_decoder_layers import DecoderLayer
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_model * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        Forward pass of simple language model.
        
        Args:
            x: Input token IDs
            
        Returns:
            output: Model output logits
        """
        # Create causal mask
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
        
        # Embed and encode
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, None, mask, None)  # No encoder output
        
        x = self.norm(x)
        return self.output_projection(x)

class BERTClassifier(nn.Module):
    """
    BERT-style classifier for text classification tasks.
    
    Uses BERT encoder with classification head.
    """
    
    def __init__(self, vocab_size, num_classes, d_model=256, num_layers=6, num_heads=8):
        """
        Initialize BERT Classifier.
        
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of classification classes
            d_model: Model dimension
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Import encoder layer from other module
        from encoder_decoder_layers import EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_model * 4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x, mask=None):
        """
        Forward pass of BERT classifier.
        
        Args:
            x: Input token IDs
            mask: Optional attention mask
            
        Returns:
            output: Classification logits
        """
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        
        # Global average pooling
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        return self.classifier(x)

class TranslationModel(nn.Module):
    """
    Translation Model for sequence-to-sequence tasks.
    
    Uses encoder-decoder architecture for machine translation.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8):
        """
        Initialize Translation Model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            num_layers: Number of encoder/decoder layers
            num_heads: Number of attention heads
        """
        super().__init__()
        
        # Import transformer from other module
        from transformer_models import Transformer
        self.transformer = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
    def forward(self, src, tgt):
        """
        Forward pass of translation model.
        
        Args:
            src: Source sequence
            tgt: Target sequence
            
        Returns:
            output: Translation logits
        """
        return self.transformer(src, tgt)
    
    def generate(self, src, max_len=50, start_token=1, end_token=2):
        """
        Generate translation for source sequence.
        
        Args:
            src: Source sequence
            max_len: Maximum generation length
            start_token: Start token ID
            end_token: End token ID
            
        Returns:
            generated: Generated target sequence
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            device = src.device
            
            # Encode source
            src_mask = self.transformer.generate_src_mask(src)
            src = self.transformer.src_embedding(src) * math.sqrt(self.transformer.d_model)
            src = self.transformer.pos_encoding(src)
            src = self.transformer.dropout(src)
            
            for encoder_layer in self.transformer.encoder:
                src = encoder_layer(src, src_mask)
            
            # Initialize target sequence
            tgt = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
            
            for _ in range(max_len - 1):
                tgt_mask = self.transformer.generate_tgt_mask(tgt)
                
                # Decode
                tgt_embed = self.transformer.tgt_embedding(tgt) * math.sqrt(self.transformer.d_model)
                tgt_embed = self.transformer.pos_encoding(tgt_embed)
                tgt_embed = self.transformer.dropout(tgt_embed)
                
                for decoder_layer in self.transformer.decoder:
                    tgt_embed = decoder_layer(tgt_embed, src, tgt_mask, src_mask)
                
                tgt_embed = self.transformer.norm(tgt_embed)
                logits = self.transformer.output_projection(tgt_embed)
                
                # Get next token
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Check if all sequences have ended
                if (tgt == end_token).any(dim=1).all():
                    break
            
            return tgt

# Example usage functions
def demonstrate_llm_architectures():
    """Demonstrate various LLM architectures."""
    print("LLM Architecture Variants Demonstration")
    print("=" * 40)
    
    # Example 1: GPT Model
    print("1. GPT-style Model")
    vocab_size, d_model, num_layers = 50000, 2048, 24
    gpt_model = GPTModel(vocab_size, d_model, num_layers)
    print(f"   Parameters: {sum(p.numel() for p in gpt_model.parameters())/1e9:.1f}B")
    print(f"   Architecture: Decoder-only with causal masking")
    print()
    
    # Example 2: BERT Model
    print("2. BERT-style Model")
    bert_model = BERTModel(vocab_size, d_model=768, num_layers=12)
    print(f"   Parameters: {sum(p.numel() for p in bert_model.parameters())/1e6:.1f}M")
    print(f"   Architecture: Encoder-only with bidirectional attention")
    print()
    
    # Example 3: T5 Model
    print("3. T5-style Model")
    t5_model = T5Model(vocab_size, d_model=768, num_layers=12)
    print(f"   Parameters: {sum(p.numel() for p in t5_model.parameters())/1e6:.1f}M")
    print(f"   Architecture: Encoder-decoder for sequence-to-sequence")
    print()
    
    # Example 4: Simple Language Model
    print("4. Simple Language Model")
    simple_model = SimpleLanguageModel(vocab_size, d_model=256, num_layers=6)
    print(f"   Parameters: {sum(p.numel() for p in simple_model.parameters())/1e6:.1f}M")
    print(f"   Architecture: Simplified decoder-only")
    print()
    
    # Example 5: BERT Classifier
    print("5. BERT Classifier")
    num_classes = 5
    classifier = BERTClassifier(vocab_size, num_classes, d_model=256, num_layers=6)
    print(f"   Parameters: {sum(p.numel() for p in classifier.parameters())/1e6:.1f}M")
    print(f"   Output classes: {num_classes}")
    print()
    
    # Example 6: Translation Model
    print("6. Translation Model")
    src_vocab_size, tgt_vocab_size = 5000, 5000
    translation_model = TranslationModel(src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6)
    print(f"   Parameters: {sum(p.numel() for p in translation_model.parameters())/1e6:.1f}M")
    print(f"   Source vocab: {src_vocab_size}, Target vocab: {tgt_vocab_size}")
    print()
    
    # Example 7: Model Comparison
    print("7. Model Comparison")
    models = {
        'GPT': gpt_model,
        'BERT': bert_model,
        'T5': t5_model,
        'Simple': simple_model,
        'Classifier': classifier,
        'Translation': translation_model
    }
    
    print("   Model Size Comparison:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        if params > 1e9:
            print(f"     {name}: {params/1e9:.1f}B parameters")
        else:
            print(f"     {name}: {params/1e6:.1f}M parameters")

if __name__ == "__main__":
    demonstrate_llm_architectures()

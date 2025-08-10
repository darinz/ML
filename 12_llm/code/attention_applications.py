"""
Attention Applications and Practical Examples.

This module provides practical examples of attention mechanisms in various
applications including text classification, sequence-to-sequence, and more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionClassifier(nn.Module):
    """
    Text classification model using attention mechanisms.
    """
    
    def __init__(self, vocab_size, d_model, num_classes, num_heads=8):
        """
        Initialize Attention Classifier.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_classes: Number of classification classes
            num_heads: Number of attention heads
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Import multi-head attention from other module
        from multi_head_attention import MultiHeadAttention
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        """
        Forward pass of attention classifier.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            output: Classification logits
        """
        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Apply self-attention
        x, _ = self.attention(x, x, x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        return x

class Seq2SeqAttention(nn.Module):
    """
    Sequence-to-sequence model with attention mechanism.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize Seq2Seq Attention model.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Import multi-head attention from other module
        from multi_head_attention import MultiHeadAttention
        self.attention = MultiHeadAttention(hidden_dim, num_heads=8)
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, src, tgt):
        """
        Forward pass of seq2seq attention model.
        
        Args:
            src: Source sequence
            tgt: Target sequence
            
        Returns:
            output: Decoder output
            attention_weights: Attention weights
        """
        # Encode source sequence
        encoder_output, (hidden, cell) = self.encoder(src)
        
        # Decode with attention
        decoder_output, _ = self.decoder(tgt, (hidden, cell))
        
        # Apply cross-attention between decoder and encoder
        attended_output, attention_weights = self.attention(
            decoder_output, encoder_output, encoder_output
        )
        
        # Generate output
        output = self.output_layer(attended_output)
        return output, attention_weights

class AttentionLanguageModel(nn.Module):
    """
    Language model using attention mechanisms.
    """
    
    def __init__(self, vocab_size, d_model, num_layers, num_heads=8, max_len=512):
        """
        Initialize Attention Language Model.
        
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
        
        # Import multi-head attention from other module
        from multi_head_attention import MultiHeadAttention
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        Forward pass of attention language model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            output: Language model logits
        """
        # Create causal mask for autoregressive generation
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        # Apply attention layers
        for attention_layer in self.attention_layers:
            x, _ = attention_layer(x, x, x, mask)
        
        # Normalize and project to vocabulary
        x = self.norm(x)
        x = self.output_projection(x)
        
        return x

class AttentionQuestionAnswering(nn.Module):
    """
    Question answering model using attention mechanisms.
    """
    
    def __init__(self, vocab_size, d_model, num_heads=8):
        """
        Initialize Attention QA model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Import multi-head attention from other module
        from multi_head_attention import MultiHeadAttention
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        
        self.answer_start = nn.Linear(d_model, 1)
        self.answer_end = nn.Linear(d_model, 1)
        
    def forward(self, question, context):
        """
        Forward pass of attention QA model.
        
        Args:
            question: Question tensor
            context: Context tensor
            
        Returns:
            start_logits: Start position logits
            end_logits: End position logits
        """
        # Embed question and context
        question_emb = self.embedding(question)
        context_emb = self.embedding(context)
        
        # Add positional encoding
        question_emb = self.pos_encoding(question_emb)
        context_emb = self.pos_encoding(context_emb)
        
        # Self-attention on question
        question_attended, _ = self.self_attention(question_emb, question_emb, question_emb)
        
        # Cross-attention from question to context
        context_attended, _ = self.cross_attention(question_attended, context_emb, context_emb)
        
        # Predict answer span
        start_logits = self.answer_start(context_attended).squeeze(-1)
        end_logits = self.answer_end(context_attended).squeeze(-1)
        
        return start_logits, end_logits

def simple_self_attention_example():
    """
    Simple self-attention example for demonstration.
    """
    print("Simple Self-Attention Example")
    print("=" * 30)
    
    # Import simple self-attention from other module
    from attention_primitives import simple_self_attention
    
    # Create example input
    x = torch.randn(2, 10, 512)  # batch_size=2, seq_len=10, d_model=512
    output, weights = simple_self_attention(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Attention weights sum to 1: {torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))}")

def attention_classifier_example():
    """
    Attention classifier example.
    """
    print("\nAttention Classifier Example")
    print("=" * 30)
    
    # Create model
    model = AttentionClassifier(vocab_size=10000, d_model=256, num_classes=5)
    
    # Create example input
    input_tensor = torch.randint(0, 10000, (32, 50))  # batch_size=32, seq_len=50
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # (32, 5)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def seq2seq_attention_example():
    """
    Sequence-to-sequence attention example.
    """
    print("\nSeq2Seq Attention Example")
    print("=" * 30)
    
    # Create model
    model = Seq2SeqAttention(input_dim=256, hidden_dim=512, output_dim=256)
    
    # Create example inputs
    src = torch.randn(16, 20, 256)  # batch_size=16, src_len=20, input_dim=256
    tgt = torch.randn(16, 15, 256)  # batch_size=16, tgt_len=15, input_dim=256
    
    output, attention_weights = model(src, tgt)
    
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

def language_model_example():
    """
    Attention language model example.
    """
    print("\nAttention Language Model Example")
    print("=" * 30)
    
    # Create model
    model = AttentionLanguageModel(vocab_size=10000, d_model=256, num_layers=4)
    
    # Create example input
    input_tensor = torch.randint(0, 10000, (8, 64))  # batch_size=8, seq_len=64
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")  # (8, 64, 10000)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def question_answering_example():
    """
    Attention question answering example.
    """
    print("\nAttention Question Answering Example")
    print("=" * 30)
    
    # Create model
    model = AttentionQuestionAnswering(vocab_size=10000, d_model=256)
    
    # Create example inputs
    question = torch.randint(0, 10000, (4, 10))  # batch_size=4, question_len=10
    context = torch.randint(0, 10000, (4, 100))  # batch_size=4, context_len=100
    
    start_logits, end_logits = model(question, context)
    
    print(f"Question shape: {question.shape}")
    print(f"Context shape: {context.shape}")
    print(f"Start logits shape: {start_logits.shape}")
    print(f"End logits shape: {end_logits.shape}")

# Example usage functions
def demonstrate_attention_applications():
    """Demonstrate various attention applications."""
    print("Attention Applications Demonstration")
    print("=" * 40)
    
    # Run all examples
    simple_self_attention_example()
    attention_classifier_example()
    seq2seq_attention_example()
    language_model_example()
    question_answering_example()
    
    print("\n" + "=" * 40)
    print("All attention applications demonstrated successfully!")

if __name__ == "__main__":
    demonstrate_attention_applications()

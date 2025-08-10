"""
Pre-training Objectives for Large Language Models.

This module implements various pre-training objectives including
Masked Language Modeling (MLM), Causal Language Modeling (CLM),
and Span Corruption for training large language models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLMTrainer:
    """
    Masked Language Modeling (MLM) trainer for BERT-style pre-training.
    
    Randomly masks tokens and predicts them based on context.
    """
    
    def __init__(self, model, vocab_size, mask_token_id, mask_prob=0.15):
        """
        Initialize MLM Trainer.
        
        Args:
            model: The model to train
            vocab_size: Size of vocabulary
            mask_token_id: ID of the [MASK] token
            mask_prob: Probability of masking tokens
        """
        self.model = model
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob
        
    def create_mlm_targets(self, input_ids):
        """
        Create MLM targets by masking random tokens.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            masked_input_ids: Input with masked tokens
            labels: Target labels for masked positions
        """
        batch_size, seq_len = input_ids.shape
        targets = input_ids.clone()
        
        # Create mask for random tokens
        mask = torch.rand(batch_size, seq_len) < self.mask_prob
        
        # Replace masked tokens with [MASK]
        input_ids[mask] = self.mask_token_id
        
        # Create target labels (only for masked positions)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[mask] = targets[mask]
        
        return input_ids, labels
    
    def compute_mlm_loss(self, logits, labels):
        """
        Compute MLM loss only for masked positions.
        
        Args:
            logits: Model output logits
            labels: Target labels
            
        Returns:
            loss: MLM loss
        """
        loss_fct = nn.CrossEntropyLoss()
        
        # Only compute loss for masked positions
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, self.vocab_size)
        active_labels = labels.view(-1)[active_loss]
        
        loss = loss_fct(active_logits, active_labels)
        return loss

class CLMTrainer:
    """
    Causal Language Modeling (CLM) trainer for GPT-style pre-training.
    
    Predicts the next token in the sequence.
    """
    
    def __init__(self, model, vocab_size):
        """
        Initialize CLM Trainer.
        
        Args:
            model: The model to train
            vocab_size: Size of vocabulary
        """
        self.model = model
        self.vocab_size = vocab_size
        
    def create_clm_targets(self, input_ids):
        """
        Create CLM targets by shifting sequence.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            inputs: Input sequence (without last token)
            targets: Target sequence (without first token)
        """
        # Input: [A, B, C, D]
        # Target: [B, C, D, E]
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        return inputs, targets
    
    def compute_clm_loss(self, logits, targets):
        """
        Compute CLM loss for all positions.
        
        Args:
            logits: Model output logits
            targets: Target token IDs
            
        Returns:
            loss: CLM loss
        """
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits.view(-1, self.vocab_size), targets.view(-1))

class SpanCorruptionTrainer:
    """
    Span Corruption trainer for T5-style pre-training.
    
    Masks spans of text instead of individual tokens.
    """
    
    def __init__(self, model, vocab_size, mask_token_id, span_length=3, mask_prob=0.15):
        """
        Initialize Span Corruption Trainer.
        
        Args:
            model: The model to train
            vocab_size: Size of vocabulary
            mask_token_id: ID of the [MASK] token
            span_length: Length of spans to mask
            mask_prob: Probability of masking spans
        """
        self.model = model
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.span_length = span_length
        self.mask_prob = mask_prob
        
    def create_span_targets(self, input_ids):
        """
        Create span corruption targets.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            masked_input_ids: Input with masked spans
            labels: Target labels for masked positions
        """
        batch_size, seq_len = input_ids.shape
        targets = input_ids.clone()
        
        # Create mask for span corruption
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        for i in range(batch_size):
            pos = 0
            while pos < seq_len:
                # Decide whether to mask this position
                if torch.rand(1) < self.mask_prob:
                    # Mask span starting at this position
                    span_end = min(pos + self.span_length, seq_len)
                    mask[i, pos:span_end] = True
                    pos = span_end
                else:
                    pos += 1
        
        # Replace masked spans with [MASK]
        input_ids[mask] = self.mask_token_id
        
        # Create target labels
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        labels[mask] = targets[mask]
        
        return input_ids, labels
    
    def compute_mlm_loss(self, logits, labels):
        """
        Compute loss for masked positions (same as MLM).
        
        Args:
            logits: Model output logits
            labels: Target labels
            
        Returns:
            loss: Span corruption loss
        """
        loss_fct = nn.CrossEntropyLoss()
        
        # Only compute loss for masked positions
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, self.vocab_size)
        active_labels = labels.view(-1)[active_loss]
        
        loss = loss_fct(active_logits, active_labels)
        return loss

class PrefixLanguageModel(nn.Module):
    """
    Prefix Language Model combining bidirectional and autoregressive modeling.
    
    Uses a hybrid approach where prefix positions can attend to all positions,
    while non-prefix positions can only attend to previous positions.
    """
    
    def __init__(self, vocab_size, d_model, num_layers, prefix_length=64):
        """
        Initialize Prefix Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of transformer layers
            prefix_length: Length of prefix that can attend bidirectionally
        """
        super().__init__()
        self.prefix_length = prefix_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Import positional encoding from other module
        from positional_encoding import PositionalEncoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Import transformer layer from other module
        from encoder_decoder_layers import EncoderLayer
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads=16, d_ff=d_model*4)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        """
        Forward pass with prefix attention mask.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            output: Model output logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask
        # Prefix positions can attend to all positions
        # Non-prefix positions can only attend to previous positions
        attention_mask = torch.ones(batch_size, seq_len, seq_len)
        
        for i in range(batch_size):
            for j in range(seq_len):
                if j < self.prefix_length:
                    # Prefix: can attend to all positions
                    attention_mask[i, j, :] = 1
                else:
                    # Non-prefix: can only attend to previous positions
                    attention_mask[i, j, j+1:] = 0
        
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.norm(x)
        return self.output_projection(x)

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for improved training stability.
    """
    
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        """
        Initialize Label Smoothing Loss.
        
        Args:
            vocab_size: Size of vocabulary
            smoothing: Smoothing factor
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model output logits
            targets: Target token IDs
            
        Returns:
            loss: Label smoothing loss
        """
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)
        
        # Create smoothed targets
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.smoothing / (self.vocab_size - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Mask ignored indices
        mask = (targets != self.ignore_index).unsqueeze(1)
        smooth_targets = smooth_targets * mask.float()
        
        # Compute loss
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1)
        
        # Average over non-ignored tokens
        return loss.sum() / mask.sum()

# Example usage functions
def demonstrate_pretraining_objectives():
    """Demonstrate various pre-training objectives."""
    print("Pre-training Objectives Demonstration")
    print("=" * 40)
    
    # Example 1: MLM Training
    print("1. Masked Language Modeling (MLM)")
    vocab_size, d_model, num_layers = 50000, 768, 12
    mask_token_id = 103  # [MASK] token ID
    
    # Create model (simplified for demonstration)
    from training_techniques import MemoryEfficientLLM
    model = MemoryEfficientLLM(vocab_size, d_model, num_layers)
    trainer = MLMTrainer(model, vocab_size, mask_token_id)
    
    # Create sample data
    batch_size, seq_len = 4, 128
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create MLM targets
    masked_inputs, labels = trainer.create_mlm_targets(input_ids)
    print(f"   Original sequence length: {seq_len}")
    print(f"   Masked tokens: {(labels != -100).sum().item()}")
    print()
    
    # Example 2: CLM Training
    print("2. Causal Language Modeling (CLM)")
    clm_trainer = CLMTrainer(model, vocab_size)
    
    # Create CLM targets
    inputs, targets = clm_trainer.create_clm_targets(input_ids)
    print(f"   Input sequence length: {inputs.size(1)}")
    print(f"   Target sequence length: {targets.size(1)}")
    print()
    
    # Example 3: Span Corruption
    print("3. Span Corruption")
    span_trainer = SpanCorruptionTrainer(model, vocab_size, mask_token_id, span_length=3)
    
    # Create span corruption targets
    masked_spans, span_labels = span_trainer.create_span_targets(input_ids)
    print(f"   Span length: {span_trainer.span_length}")
    print(f"   Masked spans: {(span_labels != -100).sum().item()}")
    print()
    
    # Example 4: Prefix Language Model
    print("4. Prefix Language Model")
    prefix_model = PrefixLanguageModel(vocab_size, d_model, num_layers, prefix_length=32)
    print(f"   Prefix length: {prefix_model.prefix_length}")
    print(f"   Total parameters: {sum(p.numel() for p in prefix_model.parameters())/1e6:.1f}M")
    print()
    
    # Example 5: Label Smoothing
    print("5. Label Smoothing")
    label_smoothing = LabelSmoothingLoss(vocab_size, smoothing=0.1)
    print(f"   Smoothing factor: {label_smoothing.smoothing}")
    print()
    
    # Example 6: Training Loop
    print("6. Training Loop Example")
    print("   # MLM Training")
    print("   for batch in dataloader:")
    print("       masked_inputs, labels = trainer.create_mlm_targets(batch)")
    print("       logits = model(masked_inputs)")
    print("       loss = trainer.compute_mlm_loss(logits, labels)")
    print("       loss.backward()")
    print()
    print("   # CLM Training")
    print("   for batch in dataloader:")
    print("       inputs, targets = clm_trainer.create_clm_targets(batch)")
    print("       logits = model(inputs)")
    print("       loss = clm_trainer.compute_clm_loss(logits, targets)")
    print("       loss.backward()")

if __name__ == "__main__":
    demonstrate_pretraining_objectives()

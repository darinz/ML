# Text Generation with Transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import math
import random
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM

class GPTTextGenerator(nn.Module):
    """
    GPT-style text generator implementation.
    
    This implements autoregressive text generation using transformer architecture.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 12,
                 num_heads: int = 8, d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for text generation.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            mask: Optional attention mask
        
        Returns:
            logits: Token prediction logits
        """
        # Token embedding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, mask)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, prompt: torch.Tensor, max_length: int = 100, 
                 temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                 do_sample: bool = True, pad_token_id: int = 0,
                 eos_token_id: int = 1) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            prompt: Input prompt tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
        
        Returns:
            generated_tokens: Generated token sequence
        """
        self.eval()
        batch_size = prompt.size(0)
        device = prompt.device
        
        # Initialize with prompt
        generated = prompt.clone()
        
        with torch.no_grad():
            for _ in range(max_length - prompt.size(1)):
                # Get model predictions
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply nucleus sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for end-of-sequence
                if (next_token == eos_token_id).any():
                    break
        
        return generated

class DecoderLayer(nn.Module):
    """Single decoder layer for text generation."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MaskedMultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Masked self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class MaskedMultiHeadAttention(nn.Module):
    """Masked multi-head attention for autoregressive generation."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of masked multi-head attention."""
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1, self.d_model)
        
        # Final linear transformation
        output = self.w_o(context)
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
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
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]

class TextGenerationTrainer:
    """
    Trainer for text generation models.
    """
    
    def __init__(self, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                 lr: float = 1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Create causal mask
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(self.device)
        
        # Forward pass
        logits = self.model(input_ids, mask)
        
        # Calculate loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Create causal mask
                seq_len = input_ids.size(1)
                mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                mask = mask.to(self.device)
                
                # Forward pass
                logits = self.model(input_ids, mask)
                
                # Calculate loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """Main training loop."""
        print(f"Starting text generation training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                num_batches += 1
                self.global_step += 1
                
                if batch_idx % 100 == 0:
                    print(f'Step {self.global_step}: Loss = {metrics["loss"]:.4f}')
            
            avg_loss = epoch_loss / num_batches
            
            # Validation
            val_metrics = self.validate()
            
            # Print metrics
            print(f'Epoch {epoch + 1}/{num_epochs}')
            print(f'  Train Loss: {avg_loss:.4f}')
            if val_metrics:
                print(f'  Val Loss: {val_metrics["val_loss"]:.4f}')
            
            # Save checkpoint
            if save_path and val_metrics and val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.save_checkpoint(save_path)
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }, path)
        print(f"Checkpoint saved to {path}")

class HuggingFaceTextGenerator:
    """
    Text generator using Hugging Face transformers.
    """
    
    def __init__(self, model_name: str = 'gpt2'):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                     do_sample: bool = True, num_return_sequences: int = 1) -> List[str]:
        """
        Generate text using Hugging Face model.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to generate
        
        Returns:
            generated_texts: List of generated text sequences
        """
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def generate_with_beam_search(self, prompt: str, max_length: int = 100,
                                 num_beams: int = 5, early_stopping: bool = True) -> List[str]:
        """
        Generate text using beam search.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            num_beams: Number of beams
            early_stopping: Whether to stop early
        
        Returns:
            generated_texts: List of generated text sequences
        """
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate with beam search
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts

class CreativeTextGenerator:
    """
    Creative text generation with various techniques.
    """
    
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def generate_with_prompt_engineering(self, prompt: str, 
                                       max_length: int = 100,
                                       temperature: float = 0.8) -> str:
        """
        Generate text with prompt engineering.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
        
        Returns:
            generated_text: Generated text
        """
        # Enhance prompt with creative instructions
        enhanced_prompt = f"Write a creative story: {prompt}"
        
        # Tokenize
        inputs = self.tokenizer.encode(enhanced_prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.9
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def generate_with_style_transfer(self, prompt: str, style: str,
                                   max_length: int = 100) -> str:
        """
        Generate text with style transfer.
        
        Args:
            prompt: Input prompt
            style: Target style (e.g., 'poetic', 'formal', 'casual')
            max_length: Maximum generation length
        
        Returns:
            generated_text: Generated text in target style
        """
        # Create style-specific prompt
        style_prompts = {
            'poetic': f"Write in a poetic style: {prompt}",
            'formal': f"Write in a formal academic style: {prompt}",
            'casual': f"Write in a casual conversational style: {prompt}",
            'technical': f"Write in a technical style: {prompt}"
        }
        
        style_prompt = style_prompts.get(style, prompt)
        
        # Tokenize
        inputs = self.tokenizer.encode(style_prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.9
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def generate_with_constraints(self, prompt: str, constraints: List[str],
                                max_length: int = 100) -> str:
        """
        Generate text with specific constraints.
        
        Args:
            prompt: Input prompt
            constraints: List of constraints to follow
            max_length: Maximum generation length
        
        Returns:
            generated_text: Generated text following constraints
        """
        # Create constraint-aware prompt
        constraint_text = " ".join([f"Must include: {c}" for c in constraints])
        enhanced_prompt = f"{prompt}. {constraint_text}"
        
        # Tokenize
        inputs = self.tokenizer.encode(enhanced_prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=0.8,
                do_sample=True,
                top_k=50,
                top_p=0.9
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

def create_text_generation_model(vocab_size: int):
    """
    Create text generation model.
    
    Args:
        vocab_size: Vocabulary size
    
    Returns:
        model: Text generation model
    """
    model = GPTTextGenerator(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048
    )
    return model

def create_huggingface_generator(model_name: str = 'gpt2'):
    """
    Create Hugging Face text generator.
    
    Args:
        model_name: Model name
    
    Returns:
        generator: Hugging Face text generator
    """
    generator = HuggingFaceTextGenerator(model_name)
    return generator

if __name__ == "__main__":
    # Example usage
    vocab_size = 30000
    
    # Create custom model
    model = create_text_generation_model(vocab_size)
    
    # Create Hugging Face generator
    hf_generator = create_huggingface_generator()
    
    # Generate text
    prompt = "Once upon a time"
    generated_texts = hf_generator.generate_text(prompt, max_length=50)
    
    print("Generated text:", generated_texts[0])
    print("Text generation examples created successfully!") 
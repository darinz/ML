# Machine Translation with Transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import math
import random
import numpy as np
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer

class TranslationTransformer(nn.Module):
    """
    Sequence-to-sequence transformer for machine translation.
    
    This implements the original transformer architecture with encoder-decoder
    structure for machine translation tasks.
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_layers: int = 6, num_heads: int = 8, d_ff: int = 2048,
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)
        
        # Decoder
        self.decoder = Decoder(d_model, num_layers, num_heads, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source token indices
            src_mask: Source attention mask
        
        Returns:
            encoder_output: Encoded source sequence
        """
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = self.pos_encoding(src_embedded)
        encoder_output = self.encoder(src_embedded, src_mask)
        return encoder_output
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target token indices
            encoder_output: Encoder output
            tgt_mask: Target attention mask
            src_mask: Source attention mask
        
        Returns:
            decoder_output: Decoded target sequence
        """
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        decoder_output = self.decoder(tgt_embedded, encoder_output, tgt_mask, src_mask)
        return decoder_output
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for translation.
        
        Args:
            src: Source token indices
            tgt: Target token indices
            src_mask: Source attention mask
            tgt_mask: Target attention mask
        
        Returns:
            output: Translation logits
        """
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Decode target
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # Output projection
        output = self.output_projection(decoder_output)
        
        return output
    
    def translate(self, src: torch.Tensor, max_length: int = 100,
                 temperature: float = 1.0, beam_size: int = 5,
                 src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Translate source sequence to target sequence.
        
        Args:
            src: Source token indices
            max_length: Maximum translation length
            temperature: Sampling temperature
            beam_size: Beam search size
            src_mask: Source attention mask
        
        Returns:
            translated: Translated token sequence
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode source
        encoder_output = self.encode(src, src_mask)
        
        # Initialize target with start token
        tgt = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)  # Start token
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Create target mask
                tgt_mask = self._create_causal_mask(tgt.size(1)).to(device)
                
                # Decode
                decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
                logits = self.output_projection(decoder_output[:, -1, :])
                
                # Sample next token
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to target
                tgt = torch.cat([tgt, next_token], dim=1)
                
                # Check for end token
                if (next_token == 3).any():  # End token
                    break
        
        return tgt
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for decoder."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

class Encoder(nn.Module):
    """Transformer encoder for translation."""
    
    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through encoder layers."""
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    """Transformer decoder for translation."""
    
    def __init__(self, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through decoder layers."""
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x

class EncoderLayer(nn.Module):
    """Single encoder layer."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
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
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class DecoderLayer(nn.Module):
    """Single decoder layer."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + attn_output)
        
        # Cross-attention
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + cross_attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)
        
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention for translation."""
    
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
        """Forward pass of multi-head attention."""
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

class TranslationTrainer:
    """
    Trainer for translation models.
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
        src = batch['src'].to(self.device)
        tgt = batch['tgt'].to(self.device)
        tgt_input = tgt[:, :-1]  # Remove last token for input
        tgt_output = tgt[:, 1:]  # Remove first token for output
        
        # Create masks
        src_mask = self._create_padding_mask(src)
        tgt_mask = self._create_causal_mask(tgt_input.size(1))
        
        # Forward pass
        logits = self.model(src, tgt_input, src_mask, tgt_mask)
        
        # Calculate loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
        
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
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Create masks
                src_mask = self._create_padding_mask(src)
                tgt_mask = self._create_causal_mask(tgt_input.size(1))
                
                # Forward pass
                logits = self.model(src, tgt_input, src_mask, tgt_mask)
                
                # Calculate loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def _create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask for attention."""
        return (x != 0).unsqueeze(1).unsqueeze(2)
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for decoder."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """Main training loop."""
        print(f"Starting translation training for {num_epochs} epochs...")
        
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

class HuggingFaceTranslator:
    """
    Translator using Hugging Face transformers.
    """
    
    def __init__(self, model_name: str = 'Helsinki-NLP/opus-mt-en-fr'):
        self.model_name = model_name
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def translate(self, text: str, max_length: int = 100) -> str:
        """
        Translate text using Hugging Face model.
        
        Args:
            text: Input text to translate
            max_length: Maximum translation length
        
        Returns:
            translated_text: Translated text
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', padding=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode output
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    
    def translate_batch(self, texts: List[str], max_length: int = 100) -> List[str]:
        """
        Translate a batch of texts.
        
        Args:
            texts: List of texts to translate
            max_length: Maximum translation length
        
        Returns:
            translated_texts: List of translated texts
        """
        # Tokenize inputs
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)
        
        # Generate translations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode outputs
        translated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return translated_texts

class MultiLanguageTranslator:
    """
    Multi-language translator supporting multiple language pairs.
    """
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def add_language_pair(self, src_lang: str, tgt_lang: str, model_name: str):
        """
        Add a language pair to the translator.
        
        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            model_name: Hugging Face model name
        """
        key = f"{src_lang}-{tgt_lang}"
        
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            
            model.to(self.device)
            
            self.models[key] = model
            self.tokenizers[key] = tokenizer
            
            print(f"Added language pair: {src_lang} -> {tgt_lang}")
        except Exception as e:
            print(f"Failed to load model for {src_lang} -> {tgt_lang}: {e}")
    
    def translate(self, text: str, src_lang: str, tgt_lang: str, 
                 max_length: int = 100) -> str:
        """
        Translate text between specified languages.
        
        Args:
            text: Input text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            max_length: Maximum translation length
        
        Returns:
            translated_text: Translated text
        """
        key = f"{src_lang}-{tgt_lang}"
        
        if key not in self.models:
            raise ValueError(f"Language pair {src_lang} -> {tgt_lang} not available")
        
        model = self.models[key]
        tokenizer = self.tokenizers[key]
        
        # Tokenize input
        inputs = tokenizer(text, return_tensors='pt', padding=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )
        
        # Decode output
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    
    def get_available_language_pairs(self) -> List[str]:
        """Get list of available language pairs."""
        return list(self.models.keys())

def create_translation_model(src_vocab_size: int, tgt_vocab_size: int):
    """
    Create translation transformer model.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
    
    Returns:
        model: Translation transformer model
    """
    model = TranslationTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048
    )
    return model

def create_huggingface_translator(model_name: str = 'Helsinki-NLP/opus-mt-en-fr'):
    """
    Create Hugging Face translator.
    
    Args:
        model_name: Model name
    
    Returns:
        translator: Hugging Face translator
    """
    translator = HuggingFaceTranslator(model_name)
    return translator

def create_multi_language_translator():
    """
    Create multi-language translator.
    
    Returns:
        translator: Multi-language translator
    """
    translator = MultiLanguageTranslator()
    
    # Add common language pairs
    language_pairs = [
        ('en', 'fr', 'Helsinki-NLP/opus-mt-en-fr'),
        ('en', 'de', 'Helsinki-NLP/opus-mt-en-de'),
        ('en', 'es', 'Helsinki-NLP/opus-mt-en-es'),
        ('fr', 'en', 'Helsinki-NLP/opus-mt-fr-en'),
        ('de', 'en', 'Helsinki-NLP/opus-mt-de-en'),
        ('es', 'en', 'Helsinki-NLP/opus-mt-es-en')
    ]
    
    for src_lang, tgt_lang, model_name in language_pairs:
        translator.add_language_pair(src_lang, tgt_lang, model_name)
    
    return translator

if __name__ == "__main__":
    # Example usage
    src_vocab_size = 30000
    tgt_vocab_size = 30000
    
    # Create custom model
    model = create_translation_model(src_vocab_size, tgt_vocab_size)
    
    # Create Hugging Face translator
    translator = create_huggingface_translator()
    
    # Create multi-language translator
    multi_translator = create_multi_language_translator()
    
    # Example translation
    text = "Hello, how are you?"
    translated = translator.translate(text)
    
    print(f"Original: {text}")
    print(f"Translated: {translated}")
    print("Translation examples created successfully!") 
# Text Summarization with Transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import math
import random
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

class SummarizationTransformer(nn.Module):
    """
    Transformer model for text summarization.
    
    This implements both extractive and abstractive summarization
    using transformer architecture.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 6,
                 num_heads: int = 8, d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1, summarization_type: str = 'abstractive'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.summarization_type = summarization_type
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder = Encoder(d_model, num_layers, num_heads, d_ff, dropout)
        
        if summarization_type == 'abstractive':
            # Decoder for abstractive summarization
            self.decoder = Decoder(d_model, num_layers, num_heads, d_ff, dropout)
            self.output_projection = nn.Linear(d_model, vocab_size)
        else:
            # Extractive summarization components
            self.sentence_encoder = nn.Linear(d_model, d_model)
            self.classifier = nn.Linear(d_model, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for summarization.
        
        Args:
            input_ids: Input token indices
            target_ids: Target token indices (for abstractive)
            attention_mask: Attention mask
        
        Returns:
            output: Summarization output
        """
        # Token embedding and positional encoding
        embedded = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        embedded = self.pos_encoding(embedded)
        
        # Encode input
        encoder_output = self.encoder(embedded, attention_mask)
        
        if self.summarization_type == 'abstractive':
            # Abstractive summarization
            if target_ids is not None:
                # Training mode
                target_embedded = self.token_embedding(target_ids) * math.sqrt(self.d_model)
                target_embedded = self.pos_encoding(target_embedded)
                
                # Create causal mask for decoder
                seq_len = target_ids.size(1)
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                causal_mask = causal_mask.to(target_ids.device)
                
                decoder_output = self.decoder(target_embedded, encoder_output, causal_mask, attention_mask)
                output = self.output_projection(decoder_output)
            else:
                # Inference mode - generate summary
                output = self.generate_summary(encoder_output, attention_mask)
        else:
            # Extractive summarization
            sentence_scores = self.classifier(encoder_output)
            output = torch.sigmoid(sentence_scores)
        
        return output
    
    def generate_summary(self, encoder_output: torch.Tensor, 
                        attention_mask: Optional[torch.Tensor] = None,
                        max_length: int = 150, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate abstractive summary.
        
        Args:
            encoder_output: Encoder output
            attention_mask: Attention mask
            max_length: Maximum summary length
            temperature: Sampling temperature
        
        Returns:
            summary_ids: Generated summary token indices
        """
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Start with start token
        summary_ids = torch.full((batch_size, 1), 2, dtype=torch.long, device=device)  # Start token
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Get current summary embeddings
                summary_embedded = self.token_embedding(summary_ids) * math.sqrt(self.d_model)
                summary_embedded = self.pos_encoding(summary_embedded)
                
                # Create causal mask
                seq_len = summary_ids.size(1)
                causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
                causal_mask = causal_mask.to(device)
                
                # Decode
                decoder_output = self.decoder(summary_embedded, encoder_output, causal_mask, attention_mask)
                logits = self.output_projection(decoder_output[:, -1, :])
                
                # Sample next token
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to summary
                summary_ids = torch.cat([summary_ids, next_token], dim=1)
                
                # Check for end token
                if (next_token == 3).any():  # End token
                    break
        
        return summary_ids

class Encoder(nn.Module):
    """Transformer encoder for summarization."""
    
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
    """Transformer decoder for abstractive summarization."""
    
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
    """Multi-head attention for summarization."""
    
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

class ExtractiveSummarizer:
    """
    Extractive summarization using TF-IDF and sentence ranking.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except:
            pass
    
    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """
        Generate extractive summary.
        
        Args:
            text: Input text
            num_sentences: Number of sentences in summary
        
        Returns:
            summary: Extractive summary
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores (sum of TF-IDF scores)
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        
        # Get top sentences
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_indices = sorted(top_indices)  # Maintain original order
        
        # Extract summary
        summary_sentences = [sentences[i] for i in top_indices]
        summary = ' '.join(summary_sentences)
        
        return summary
    
    def summarize_with_keywords(self, text: str, keywords: List[str], 
                              num_sentences: int = 3) -> str:
        """
        Generate extractive summary with keyword bias.
        
        Args:
            text: Input text
            keywords: Keywords to prioritize
            num_sentences: Number of sentences in summary
        
        Returns:
            summary: Extractive summary
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Create TF-IDF vectors
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores with keyword bias
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        
        # Add keyword bonus
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for keyword in keywords if keyword.lower() in sentence_lower)
            sentence_scores[i] += keyword_count * 0.5
        
        # Get top sentences
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_indices = sorted(top_indices)  # Maintain original order
        
        # Extract summary
        summary_sentences = [sentences[i] for i in top_indices]
        summary = ' '.join(summary_sentences)
        
        return summary

class HuggingFaceSummarizer:
    """
    Summarizer using Hugging Face transformers.
    """
    
    def __init__(self, model_name: str = 't5-base'):
        self.model_name = model_name
        
        if 't5' in model_name.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif 'bart' in model_name.lower():
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def summarize(self, text: str, max_length: int = 150, 
                 min_length: int = 50, num_beams: int = 4) -> str:
        """
        Generate abstractive summary.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
        
        Returns:
            summary: Generated summary
        """
        # Prepare input
        if 't5' in self.model_name.lower():
            # T5 requires a prefix
            input_text = f"summarize: {text}"
        else:
            input_text = text
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors='pt', 
                               max_length=1024, truncation=True).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode output
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def summarize_batch(self, texts: List[str], max_length: int = 150,
                       min_length: int = 50, num_beams: int = 4) -> List[str]:
        """
        Summarize a batch of texts.
        
        Args:
            texts: List of texts to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
        
        Returns:
            summaries: List of generated summaries
        """
        summaries = []
        
        for text in texts:
            summary = self.summarize(text, max_length, min_length, num_beams)
            summaries.append(summary)
        
        return summaries

class SummarizationTrainer:
    """
    Trainer for summarization models.
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
        if model.summarization_type == 'abstractive':
            self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        else:
            self.criterion = nn.BCELoss()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        if self.model.summarization_type == 'abstractive':
            target_ids = batch['target_ids'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, target_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), 
                                target_ids.view(-1))
        else:
            # Extractive summarization
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            scores = self.model(input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            loss = self.criterion(scores.squeeze(-1), labels.float())
        
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
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                if self.model.summarization_type == 'abstractive':
                    target_ids = batch['target_ids'].to(self.device)
                    
                    # Forward pass
                    logits = self.model(input_ids, target_ids, attention_mask)
                    
                    # Calculate loss
                    loss = self.criterion(logits.view(-1, logits.size(-1)), 
                                        target_ids.view(-1))
                else:
                    # Extractive summarization
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    scores = self.model(input_ids, attention_mask=attention_mask)
                    
                    # Calculate loss
                    loss = self.criterion(scores.squeeze(-1), labels.float())
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """Main training loop."""
        print(f"Starting summarization training for {num_epochs} epochs...")
        
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

def create_summarization_model(vocab_size: int, summarization_type: str = 'abstractive'):
    """
    Create summarization transformer model.
    
    Args:
        vocab_size: Vocabulary size
        summarization_type: Type of summarization ('abstractive' or 'extractive')
    
    Returns:
        model: Summarization transformer model
    """
    model = SummarizationTransformer(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        summarization_type=summarization_type
    )
    return model

def create_huggingface_summarizer(model_name: str = 't5-base'):
    """
    Create Hugging Face summarizer.
    
    Args:
        model_name: Model name
    
    Returns:
        summarizer: Hugging Face summarizer
    """
    summarizer = HuggingFaceSummarizer(model_name)
    return summarizer

def create_extractive_summarizer():
    """
    Create extractive summarizer.
    
    Returns:
        summarizer: Extractive summarizer
    """
    summarizer = ExtractiveSummarizer()
    return summarizer

if __name__ == "__main__":
    # Example usage
    vocab_size = 30000
    
    # Create custom models
    abstractive_model = create_summarization_model(vocab_size, 'abstractive')
    extractive_model = create_summarization_model(vocab_size, 'extractive')
    
    # Create Hugging Face summarizer
    hf_summarizer = create_huggingface_summarizer()
    
    # Create extractive summarizer
    extractive_summarizer = create_extractive_summarizer()
    
    # Example summarization
    text = """
    Artificial intelligence has made significant progress in recent years. 
    Machine learning algorithms can now perform tasks that were previously 
    thought to be impossible. Deep learning models have achieved remarkable 
    results in image recognition, natural language processing, and game playing. 
    The field continues to advance rapidly with new breakthroughs occurring regularly.
    """
    
    # Abstractive summarization
    abstractive_summary = hf_summarizer.summarize(text)
    
    # Extractive summarization
    extractive_summary = extractive_summarizer.summarize(text)
    
    print("Original text:", text.strip())
    print("Abstractive summary:", abstractive_summary)
    print("Extractive summary:", extractive_summary)
    print("Summarization examples created successfully!") 
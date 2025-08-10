# Model Quantization for Transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub, prepare_qat_fx, convert_fx
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization.qconfig import get_default_qconfig, QConfig
from torch.ao.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize
from typing import Dict, Any, Optional, Tuple, List
import copy
import math

class QuantizedTransformer(nn.Module):
    """
    Quantized transformer implementation.
    
    Demonstrates various quantization techniques for transformer models.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 6,
                 num_heads: int = 8, d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1, quantize: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.quantize = quantize
        
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, num_heads, d_ff, dropout)
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional quantization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        
        Returns:
            output: Transformer output
        """
        # Quantize input if enabled
        if self.quantize:
            x = self.quant(x)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Output projection
        output = self.output_projection(x)
        
        # Dequantize output if enabled
        if self.quantize:
            output = self.dequant(output)
        
        return output

class TransformerLayer(nn.Module):
    """Single transformer layer with quantization support."""
    
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention with quantization support."""
    
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
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
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

class QuantizationManager:
    """
    Manager for model quantization.
    
    Handles different quantization techniques and configurations.
    """
    
    def __init__(self, model: nn.Module, qconfig: Optional[QConfig] = None):
        self.model = model
        self.qconfig = qconfig or get_default_qconfig('fbgemm')
        self.quantized_model = None
    
    def prepare_for_quantization(self) -> nn.Module:
        """
        Prepare model for quantization.
        
        Returns:
            prepared_model: Model prepared for quantization
        """
        # Create a copy of the model
        prepared_model = copy.deepcopy(self.model)
        
        # Set quantization configuration
        prepared_model.qconfig = self.qconfig
        
        # Prepare for quantization
        prepared_model = prepare_fx(prepared_model, {'': self.qconfig})
        
        return prepared_model
    
    def quantize_model(self, prepared_model: nn.Module) -> nn.Module:
        """
        Convert prepared model to quantized model.
        
        Args:
            prepared_model: Model prepared for quantization
        
        Returns:
            quantized_model: Quantized model
        """
        self.quantized_model = convert_fx(prepared_model)
        return self.quantized_model
    
    def quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization to model.
        
        Args:
            model: Model to quantize
        
        Returns:
            quantized_model: Dynamically quantized model
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    def quantize_static(self, model: nn.Module, calibration_data: List[torch.Tensor]) -> nn.Module:
        """
        Apply static quantization to model.
        
        Args:
            model: Model to quantize
            calibration_data: Data for calibration
        
        Returns:
            quantized_model: Statically quantized model
        """
        # Prepare model for static quantization
        model.eval()
        model.qconfig = get_default_qconfig('fbgemm')
        
        # Prepare for quantization
        prepared_model = prepare_fx(model, {'': model.qconfig})
        
        # Calibrate with data
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)
        
        # Convert to quantized model
        quantized_model = convert_fx(prepared_model)
        return quantized_model

class PostTrainingQuantizer:
    """
    Post-training quantization utilities.
    """
    
    @staticmethod
    def quantize_linear_layers(model: nn.Module, dtype: torch.dtype = torch.qint8) -> nn.Module:
        """
        Quantize linear layers in the model.
        
        Args:
            model: Model to quantize
            dtype: Quantization data type
        
        Returns:
            quantized_model: Model with quantized linear layers
        """
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=dtype
        )
        return quantized_model
    
    @staticmethod
    def quantize_attention_layers(model: nn.Module) -> nn.Module:
        """
        Quantize attention layers specifically.
        
        Args:
            model: Model to quantize
        
        Returns:
            quantized_model: Model with quantized attention
        """
        # Find attention layers and quantize them
        for name, module in model.named_modules():
            if isinstance(module, MultiHeadAttention):
                # Quantize linear layers in attention
                module.w_q = torch.quantization.quantize_dynamic(
                    module.w_q, {nn.Linear}, dtype=torch.qint8
                )
                module.w_k = torch.quantization.quantize_dynamic(
                    module.w_k, {nn.Linear}, dtype=torch.qint8
                )
                module.w_v = torch.quantization.quantize_dynamic(
                    module.w_v, {nn.Linear}, dtype=torch.qint8
                )
                module.w_o = torch.quantization.quantize_dynamic(
                    module.w_o, {nn.Linear}, dtype=torch.qint8
                )
        
        return model

class QuantizationAwareTrainer:
    """
    Trainer for quantization-aware training (QAT).
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
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs
        
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
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """Main training loop."""
        print(f"Starting QAT training for {num_epochs} epochs...")
        
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

class QuantizationEvaluator:
    """
    Evaluator for quantized models.
    """
    
    def __init__(self, original_model: nn.Module, quantized_model: nn.Module):
        self.original_model = original_model
        self.quantized_model = quantized_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.original_model.to(self.device)
        self.quantized_model.to(self.device)
    
    def compare_models(self, test_data: List[torch.Tensor]) -> Dict[str, float]:
        """
        Compare original and quantized models.
        
        Args:
            test_data: Test data for evaluation
        
        Returns:
            metrics: Comparison metrics
        """
        self.original_model.eval()
        self.quantized_model.eval()
        
        original_outputs = []
        quantized_outputs = []
        
        with torch.no_grad():
            for data in test_data:
                data = data.to(self.device)
                
                # Original model
                orig_output = self.original_model(data)
                original_outputs.append(orig_output)
                
                # Quantized model
                quant_output = self.quantized_model(data)
                quantized_outputs.append(quant_output)
        
        # Calculate metrics
        metrics = {}
        
        # Accuracy comparison
        orig_accuracy = self._calculate_accuracy(original_outputs, test_data)
        quant_accuracy = self._calculate_accuracy(quantized_outputs, test_data)
        metrics['accuracy_drop'] = orig_accuracy - quant_accuracy
        
        # Output similarity
        similarity = self._calculate_similarity(original_outputs, quantized_outputs)
        metrics['output_similarity'] = similarity
        
        # Model size comparison
        orig_size = self._get_model_size(self.original_model)
        quant_size = self._get_model_size(self.quantized_model)
        metrics['size_reduction'] = (orig_size - quant_size) / orig_size
        
        return metrics
    
    def _calculate_accuracy(self, outputs: List[torch.Tensor], 
                           targets: List[torch.Tensor]) -> float:
        """Calculate accuracy for outputs."""
        correct = 0
        total = 0
        
        for output, target in zip(outputs, targets):
            pred = output.argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += target.numel()
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_similarity(self, outputs1: List[torch.Tensor], 
                             outputs2: List[torch.Tensor]) -> float:
        """Calculate similarity between two sets of outputs."""
        total_similarity = 0.0
        count = 0
        
        for out1, out2 in zip(outputs1, outputs2):
            # Cosine similarity
            similarity = F.cosine_similarity(out1.flatten(), out2.flatten(), dim=0)
            total_similarity += similarity.item()
            count += 1
        
        return total_similarity / count if count > 0 else 0.0
    
    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes."""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return param_size + buffer_size

def create_quantized_transformer(vocab_size: int, quantize: bool = True):
    """
    Create quantized transformer model.
    
    Args:
        vocab_size: Vocabulary size
        quantize: Whether to enable quantization
    
    Returns:
        model: Quantized transformer model
    """
    model = QuantizedTransformer(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        quantize=quantize
    )
    return model

def quantize_transformer_post_training(model: nn.Module) -> nn.Module:
    """
    Apply post-training quantization to transformer.
    
    Args:
        model: Original transformer model
    
    Returns:
        quantized_model: Quantized model
    """
    # Quantize linear layers
    quantized_model = PostTrainingQuantizer.quantize_linear_layers(model)
    
    # Quantize attention layers
    quantized_model = PostTrainingQuantizer.quantize_attention_layers(quantized_model)
    
    return quantized_model

def prepare_qat_model(model: nn.Module, qconfig: Optional[QConfig] = None) -> nn.Module:
    """
    Prepare model for quantization-aware training.
    
    Args:
        model: Original model
        qconfig: Quantization configuration
    
    Returns:
        qat_model: Model prepared for QAT
    """
    if qconfig is None:
        qconfig = get_default_qconfig('fbgemm')
    
    # Set quantization configuration
    model.qconfig = qconfig
    
    # Prepare for QAT
    qat_model = prepare_qat_fx(model, {'': qconfig})
    
    return qat_model

def convert_qat_to_quantized(qat_model: nn.Module) -> nn.Module:
    """
    Convert QAT model to quantized model.
    
    Args:
        qat_model: QAT model
    
    Returns:
        quantized_model: Quantized model
    """
    quantized_model = convert_fx(qat_model)
    return quantized_model

if __name__ == "__main__":
    # Example usage
    vocab_size = 30000
    
    # Create original model
    original_model = create_quantized_transformer(vocab_size, quantize=False)
    
    # Post-training quantization
    quantized_model = quantize_transformer_post_training(original_model)
    
    # QAT preparation
    qat_model = prepare_qat_model(original_model)
    
    print("Quantization examples created successfully!") 
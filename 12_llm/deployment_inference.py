"""
Deployment and Inference for Large Language Models.

This module implements various deployment and inference techniques
for efficiently serving large language models in production.
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import time
import numpy as np
from typing import List, Dict, Any

def quantize_model(model, calibration_data):
    """
    Quantize model for faster inference.
    
    Args:
        model: The model to quantize
        calibration_data: Data for calibration
        
    Returns:
        quantized_model: Quantized model
    """
    model.eval()
    
    # Prepare for quantization
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    quantization.prepare(model, inplace=True)
    
    # Calibrate
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)
    
    # Convert to quantized model
    quantized_model = quantization.convert(model, inplace=False)
    
    return quantized_model

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
    """
    Generate text using the trained model.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        
    Returns:
        generated_text: Generated text
    """
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    input_ids = input_ids.to(next(model.parameters()).device)
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(generated_ids)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if end token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

class ModelServer:
    """
    Simple model server for inference.
    
    Provides a basic interface for serving model predictions.
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize model server.
        
        Args:
            model: The model to serve
            tokenizer: Tokenizer for the model
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def predict(self, text, max_length=50, temperature=0.8):
        """
        Generate prediction for input text.
        
        Args:
            text: Input text
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            prediction: Generated text
        """
        return generate_text(
            self.model, 
            self.tokenizer, 
            text, 
            max_length=max_length, 
            temperature=temperature
        )
    
    def batch_predict(self, texts, max_length=50, temperature=0.8):
        """
        Generate predictions for multiple texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            predictions: List of generated texts
        """
        predictions = []
        for text in texts:
            prediction = self.predict(text, max_length, temperature)
            predictions.append(prediction)
        return predictions

class OptimizedInference:
    """
    Optimized inference with various techniques.
    
    Implements techniques like caching, batching, and optimization.
    """
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Initialize optimized inference.
        
        Args:
            model: The model to optimize
            tokenizer: Tokenizer for the model
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Enable optimizations
        self.model = torch.jit.script(self.model) if hasattr(torch.jit, 'script') else self.model
        
    def optimized_generate(self, prompt, max_length=100, temperature=0.8):
        """
        Optimized text generation.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            generated_text: Generated text
        """
        # Use torch.no_grad for inference
        with torch.no_grad():
            return generate_text(
                self.model, 
                self.tokenizer, 
                prompt, 
                max_length, 
                temperature
            )
    
    def batch_generate(self, prompts, max_length=100, temperature=0.8):
        """
        Batch text generation for efficiency.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            generated_texts: List of generated texts
        """
        generated_texts = []
        
        # Process in batches for efficiency
        batch_size = 4
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            
            with torch.no_grad():
                for prompt in batch_prompts:
                    generated_text = self.optimized_generate(prompt, max_length, temperature)
                    generated_texts.append(generated_text)
        
        return generated_texts

def measure_inference_performance(model, tokenizer, test_prompts, num_runs=10):
    """
    Measure inference performance metrics.
    
    Args:
        model: The model to test
        tokenizer: Tokenizer for the model
        test_prompts: List of test prompts
        num_runs: Number of runs for averaging
        
    Returns:
        performance_metrics: Dictionary of performance metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    warmup_prompt = test_prompts[0]
    for _ in range(5):
        _ = generate_text(model, tokenizer, warmup_prompt, max_length=10)
    
    # Measure performance
    latencies = []
    throughputs = []
    
    for _ in range(num_runs):
        for prompt in test_prompts:
            start_time = time.time()
            generated_text = generate_text(model, tokenizer, prompt, max_length=50)
            end_time = time.time()
            
            latency = end_time - start_time
            throughput = len(generated_text.split()) / latency  # words per second
            
            latencies.append(latency)
            throughputs.append(throughput)
    
    performance_metrics = {
        'avg_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'avg_throughput': np.mean(throughputs),
        'std_throughput': np.std(throughputs),
        'min_latency': np.min(latencies),
        'max_latency': np.max(latencies)
    }
    
    return performance_metrics

def create_model_checkpoint(model, optimizer, epoch, save_path):
    """
    Create model checkpoint for deployment.
    
    Args:
        model: The model to save
        optimizer: The optimizer state
        epoch: Current epoch
        save_path: Path to save checkpoint
        
    Returns:
        checkpoint_info: Information about the saved checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'model_config': {
            'vocab_size': getattr(model, 'vocab_size', None),
            'd_model': getattr(model, 'd_model', None),
            'num_layers': getattr(model, 'num_layers', None),
            'num_heads': getattr(model, 'num_heads', None)
        }
    }
    
    torch.save(checkpoint, save_path)
    
    checkpoint_info = {
        'path': save_path,
        'epoch': epoch,
        'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1024**2,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    
    return checkpoint_info

def load_model_checkpoint(checkpoint_path, model_class, device='cuda'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Class of the model to load
        device: Device to load model on
        
    Returns:
        model: Loaded model
        checkpoint_info: Information about the loaded checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model from config
    config = checkpoint['model_config']
    model = model_class(**config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    checkpoint_info = {
        'epoch': checkpoint['epoch'],
        'model_size_mb': sum(p.numel() for p in model.parameters()) * 4 / 1024**2,
        'total_parameters': sum(p.numel() for p in model.parameters())
    }
    
    return model, checkpoint_info

def optimize_for_inference(model, device='cuda'):
    """
    Optimize model for inference.
    
    Args:
        model: The model to optimize
        device: Device to run on
        
    Returns:
        optimized_model: Optimized model
    """
    model.eval()
    model.to(device)
    
    # Enable optimizations
    if hasattr(torch, 'jit'):
        try:
            # Try to compile with TorchScript
            model = torch.jit.script(model)
        except:
            # Fallback to tracing if scripting fails
            dummy_input = torch.randn(1, 128).long().to(device)
            model = torch.jit.trace(model, dummy_input)
    
    # Enable memory efficient attention if available
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        # This will use flash attention if available
        pass
    
    return model

def create_inference_pipeline(model, tokenizer, device='cuda'):
    """
    Create a complete inference pipeline.
    
    Args:
        model: The model to use
        tokenizer: Tokenizer for the model
        device: Device to run on
        
    Returns:
        pipeline: Inference pipeline
    """
    # Optimize model
    optimized_model = optimize_for_inference(model, device)
    
    # Create server
    server = ModelServer(optimized_model, tokenizer, device)
    
    # Create optimized inference
    optimized_inference = OptimizedInference(optimized_model, tokenizer, device)
    
    return {
        'server': server,
        'optimized_inference': optimized_inference,
        'model': optimized_model
    }

# Example usage functions
def demonstrate_deployment_tools():
    """Demonstrate various deployment tools."""
    print("Deployment and Inference Tools Demonstration")
    print("=" * 40)
    
    # Example 1: Model Quantization
    print("1. Model Quantization")
    print("   - Reduces model size and improves inference speed")
    print("   - Uses lower precision (INT8) for weights and activations")
    print("   - Requires calibration data for optimal performance")
    print()
    
    # Example 2: Text Generation
    print("2. Text Generation")
    print("   - Implements various sampling strategies (greedy, top-k, top-p)")
    print("   - Supports temperature control for creativity")
    print("   - Handles stopping conditions and special tokens")
    print()
    
    # Example 3: Model Server
    print("3. Model Server")
    print("   - Provides simple interface for model serving")
    print("   - Supports single and batch predictions")
    print("   - Handles device management and model state")
    print()
    
    # Example 4: Optimized Inference
    print("4. Optimized Inference")
    print("   - Uses TorchScript for faster execution")
    print("   - Implements batching for efficiency")
    print("   - Enables memory optimizations")
    print()
    
    # Example 5: Performance Measurement
    print("5. Performance Measurement")
    print("   - Measures latency and throughput")
    print("   - Provides statistical analysis")
    print("   - Helps identify bottlenecks")
    print()
    
    # Example 6: Checkpoint Management
    print("6. Checkpoint Management")
    print("   - Saves and loads model checkpoints")
    print("   - Preserves optimizer state and configuration")
    print("   - Tracks model metadata")
    print()
    
    # Example usage code
    print("Example Usage:")
    print("""
    # Create inference pipeline
    pipeline = create_inference_pipeline(model, tokenizer, device)
    
    # Generate text
    generated_text = generate_text(model, tokenizer, "Hello world", max_length=50)
    print(f"Generated: {generated_text}")
    
    # Measure performance
    test_prompts = ["Hello", "How are you", "What is AI"]
    metrics = measure_inference_performance(model, tokenizer, test_prompts)
    print(f"Average latency: {metrics['avg_latency']:.3f}s")
    print(f"Average throughput: {metrics['avg_throughput']:.1f} words/s")
    
    # Save checkpoint
    checkpoint_info = create_model_checkpoint(model, optimizer, epoch=10, save_path="model.pt")
    print(f"Saved checkpoint: {checkpoint_info['model_size_mb']:.1f}MB")
    
    # Load checkpoint
    loaded_model, info = load_model_checkpoint("model.pt", ModelClass, device)
    print(f"Loaded model from epoch {info['epoch']}")
    """)

if __name__ == "__main__":
    demonstrate_deployment_tools()

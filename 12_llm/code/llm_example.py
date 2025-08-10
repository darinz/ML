"""
Complete Example Usage for Large Language Models.

This script demonstrates how to use all the LLM components together
with a complete example including model creation, training, evaluation, and deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def run_complete_llm_example():
    """Run a complete LLM example demonstrating all components."""
    
    print("Large Language Models Complete Example")
    print("=" * 50)
    
    # 1. Scaling Laws Analysis
    print("\n1. Scaling Laws Analysis")
    from scaling_laws import demonstrate_scaling_laws
    demonstrate_scaling_laws()
    
    # 2. Model Architecture Creation
    print("\n2. Model Architecture Creation")
    from llm_architectures import demonstrate_llm_architectures
    demonstrate_llm_architectures()
    
    # 3. Training Techniques
    print("\n3. Training Techniques")
    from training_techniques import demonstrate_training_techniques
    demonstrate_training_techniques()
    
    # 4. Pre-training Objectives
    print("\n4. Pre-training Objectives")
    from pretraining_objectives import demonstrate_pretraining_objectives
    demonstrate_pretraining_objectives()
    
    # 5. Evaluation and Monitoring
    print("\n5. Evaluation and Monitoring")
    from evaluation_monitoring import demonstrate_evaluation_tools
    demonstrate_evaluation_tools()
    
    # 6. Deployment and Inference
    print("\n6. Deployment and Inference")
    from deployment_inference import demonstrate_deployment_tools
    demonstrate_deployment_tools()
    
    # 7. Ethical Considerations
    print("\n7. Ethical Considerations")
    from ethical_considerations import demonstrate_ethical_tools
    demonstrate_ethical_tools()
    
    print("\nComplete LLM example finished!")
    print("All components have been demonstrated successfully.")

def create_simple_training_example():
    """Create a simple training example with a small model."""
    
    print("\nSimple Training Example")
    print("=" * 30)
    
    # Create a small model
    from llm_architectures import SimpleLanguageModel
    vocab_size = 1000
    d_model = 128
    num_layers = 4
    
    model = SimpleLanguageModel(vocab_size, d_model, num_layers)
    print(f"Model created with {sum(p.numel() for p in model.parameters())/1e3:.1f}K parameters")
    
    # Create dummy data
    batch_size = 8
    seq_len = 32
    num_batches = 10
    
    # Generate random sequences
    data = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size * num_batches, seq_len))
    
    # Create data loader
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, vocab_size), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 2 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    print("Training completed!")
    return model

def demonstrate_inference_pipeline():
    """Demonstrate a complete inference pipeline."""
    
    print("\nInference Pipeline Demonstration")
    print("=" * 35)
    
    # Create a simple model
    from llm_architectures import SimpleLanguageModel
    model = SimpleLanguageModel(vocab_size=1000, d_model=128, num_layers=4)
    
    # Create a simple tokenizer (for demonstration)
    class SimpleTokenizer:
        def __init__(self, vocab_size=1000):
            self.vocab_size = vocab_size
            self.eos_token_id = vocab_size - 1
        
        def encode(self, text, return_tensors='pt', add_special_tokens=False):
            # Simple encoding: convert characters to integers
            tokens = [ord(c) % self.vocab_size for c in text]
            if return_tensors == 'pt':
                return torch.tensor([tokens])
            return tokens
        
        def decode(self, token_ids, skip_special_tokens=True):
            # Simple decoding: convert integers back to characters
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return ''.join([chr(t) if t < 128 else '?' for t in token_ids])
    
    tokenizer = SimpleTokenizer()
    
    # Create inference pipeline
    from deployment_inference import create_inference_pipeline
    pipeline = create_inference_pipeline(model, tokenizer, device='cpu')
    
    # Test text generation
    test_prompts = ["Hello", "How are", "What is"]
    
    print("Testing text generation:")
    for prompt in test_prompts:
        try:
            generated = pipeline['server'].predict(prompt, max_length=10, temperature=0.8)
            print(f"Prompt: '{prompt}' -> Generated: '{generated}'")
        except Exception as e:
            print(f"Error generating for '{prompt}': {e}")
    
    print("Inference pipeline demonstration completed!")

def demonstrate_evaluation_metrics():
    """Demonstrate evaluation metrics calculation."""
    
    print("\nEvaluation Metrics Demonstration")
    print("=" * 35)
    
    # Create a simple model and data
    from llm_architectures import SimpleLanguageModel
    model = SimpleLanguageModel(vocab_size=1000, d_model=128, num_layers=4)
    
    # Create dummy dataloader
    batch_size = 4
    seq_len = 16
    num_batches = 5
    
    data = torch.randint(0, 1000, (batch_size * num_batches, seq_len))
    targets = torch.randint(0, 1000, (batch_size * num_batches, seq_len))
    
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate perplexity
    from evaluation_monitoring import calculate_perplexity
    try:
        perplexity = calculate_perplexity(model, dataloader, device='cpu')
        print(f"Model perplexity: {perplexity:.2f}")
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
    
    # Calculate model efficiency
    from evaluation_monitoring import calculate_model_efficiency
    try:
        efficiency_metrics = calculate_model_efficiency(model, (1, 128), device='cpu')
        print(f"Model efficiency metrics:")
        for key, value in efficiency_metrics.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error calculating efficiency metrics: {e}")
    
    print("Evaluation metrics demonstration completed!")

def demonstrate_ethical_analysis():
    """Demonstrate ethical analysis tools."""
    
    print("\nEthical Analysis Demonstration")
    print("=" * 35)
    
    # Create a simple model and tokenizer
    from llm_architectures import SimpleLanguageModel
    model = SimpleLanguageModel(vocab_size=1000, d_model=128, num_layers=4)
    
    class SimpleTokenizer:
        def __init__(self, vocab_size=1000):
            self.vocab_size = vocab_size
            self.eos_token_id = vocab_size - 1
        
        def encode(self, text, return_tensors='pt', add_special_tokens=False):
            tokens = [ord(c) % self.vocab_size for c in text]
            if return_tensors == 'pt':
                return torch.tensor([tokens])
            return tokens
    
    tokenizer = SimpleTokenizer()
    
    # Test content filtering
    from ethical_considerations import ContentFilter
    content_filter = ContentFilter()
    
    test_texts = [
        "This is a normal text.",
        "This text contains harmful content.",
        "Another normal sentence."
    ]
    
    print("Testing content filtering:")
    for text in test_texts:
        filtered = content_filter.filter_text(text)
        toxicity = content_filter.get_toxicity_score(text)
        print(f"Text: '{text}'")
        print(f"  Filtered: '{filtered}'")
        print(f"  Toxicity score: {toxicity:.3f}")
    
    # Test bias detection
    from ethical_considerations import BiasDetector
    bias_detector = BiasDetector(model, tokenizer)
    
    test_prompts = ["The person", "A worker", "Someone"]
    
    try:
        gender_bias = bias_detector.analyze_gender_bias(test_prompts)
        print(f"\nGender bias analysis:")
        print(f"  Bias scores: {gender_bias['bias_scores']}")
        print(f"  Bias ratio: {gender_bias['bias_ratio']:.2f}")
        print(f"  Is biased: {gender_bias['is_biased']}")
    except Exception as e:
        print(f"Error in bias detection: {e}")
    
    print("Ethical analysis demonstration completed!")

if __name__ == "__main__":
    # Run complete example
    run_complete_llm_example()
    
    # Run individual demonstrations
    print("\n" + "="*60)
    print("INDIVIDUAL COMPONENT DEMONSTRATIONS")
    print("="*60)
    
    # Simple training example
    model = create_simple_training_example()
    
    # Inference pipeline
    demonstrate_inference_pipeline()
    
    # Evaluation metrics
    demonstrate_evaluation_metrics()
    
    # Ethical analysis
    demonstrate_ethical_analysis()
    
    print("\n" + "="*60)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("="*60)

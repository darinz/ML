"""
Evaluation and Monitoring for Large Language Models.

This module implements various evaluation metrics and monitoring tools
for assessing LLM performance and behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_perplexity(model, dataloader, device):
    """
    Calculate perplexity on validation set.
    
    Args:
        model: The language model
        dataloader: Data loader for validation set
        device: Device to run evaluation on
        
    Returns:
        perplexity: Perplexity score
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-100,
                reduction='sum'
            )
            
            # Count non-ignored tokens
            num_tokens = (targets != -100).sum().item()
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def visualize_attention(model, input_text, tokenizer, layer_idx=0, head_idx=0):
    """
    Visualize attention weights for a specific layer and head.
    
    Args:
        model: The language model
        input_text: Input text to visualize attention for
        tokenizer: Tokenizer for the model
        layer_idx: Index of layer to visualize
        head_idx: Index of attention head to visualize
    """
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(input_text)
    input_ids = torch.tensor([tokens]).to(next(model.parameters()).device)
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attention_weights = outputs.attentions[layer_idx][0, head_idx]
    
    # Create visualization
    token_labels = tokenizer.convert_ids_to_tokens(tokens)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights.cpu().numpy(), 
                xticklabels=token_labels, 
                yticklabels=token_labels,
                cmap='Blues')
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.show()

def calculate_accuracy(model, dataloader, device):
    """
    Calculate accuracy for classification tasks.
    
    Args:
        model: The classification model
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        
    Returns:
        accuracy: Accuracy score
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            outputs = model(input_ids)
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = correct / total
    return accuracy

def calculate_bleu_score(predictions, references, tokenizer):
    """
    Calculate BLEU score for translation tasks.
    
    Args:
        predictions: List of predicted sequences
        references: List of reference sequences
        tokenizer: Tokenizer for tokenization
        
    Returns:
        bleu_score: BLEU score
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    smoothie = SmoothingFunction().method1
    
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        # Tokenize predictions and references
        pred_tokens = tokenizer.tokenize(pred)
        ref_tokens = [tokenizer.tokenize(ref)]
        
        # Calculate BLEU score
        score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothie)
        bleu_scores.append(score)
    
    return np.mean(bleu_scores)

def monitor_training_metrics(train_losses, val_losses, learning_rates, save_path=None):
    """
    Monitor and visualize training metrics.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        learning_rates: List of learning rates
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot learning rate
    ax2.plot(learning_rates, color='green')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def calculate_model_efficiency(model, input_size, device):
    """
    Calculate model efficiency metrics.
    
    Args:
        model: The model to evaluate
        input_size: Size of input tensor
        device: Device to run evaluation on
        
    Returns:
        metrics: Dictionary of efficiency metrics
    """
    model.eval()
    
    # Measure inference time
    input_tensor = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Measure inference time
    torch.cuda.synchronize()
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    with torch.no_grad():
        _ = model(input_tensor)
    end_time.record()
    
    torch.cuda.synchronize()
    inference_time = start_time.elapsed_time(end_time)
    
    # Calculate memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    metrics = {
        'inference_time_ms': inference_time,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / 1024**2  # Assuming FP32
    }
    
    return metrics

def evaluate_model_robustness(model, dataloader, device, noise_levels=[0.0, 0.1, 0.2, 0.3]):
    """
    Evaluate model robustness to input noise.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        noise_levels: List of noise levels to test
        
    Returns:
        robustness_scores: Dictionary of robustness scores
    """
    model.eval()
    robustness_scores = {}
    
    for noise_level in noise_levels:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input_ids, targets in dataloader:
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                
                # Add noise to input
                if noise_level > 0:
                    noise = torch.randn_like(input_ids.float()) * noise_level
                    input_ids = input_ids + noise.long()
                    input_ids = torch.clamp(input_ids, 0, input_ids.max())
                
                outputs = model(input_ids)
                _, predicted = torch.max(outputs.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = correct / total
        robustness_scores[f'noise_{noise_level}'] = accuracy
    
    return robustness_scores

def calculate_confidence_metrics(model, dataloader, device):
    """
    Calculate confidence metrics for model predictions.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        
    Returns:
        confidence_metrics: Dictionary of confidence metrics
    """
    model.eval()
    confidences = []
    correct_predictions = []
    
    with torch.no_grad():
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            outputs = model(input_ids)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confidences.extend(confidence.cpu().numpy())
            correct_predictions.extend((predicted == targets).cpu().numpy())
    
    confidences = np.array(confidences)
    correct_predictions = np.array(correct_predictions)
    
    # Calculate confidence metrics
    avg_confidence = np.mean(confidences)
    confidence_correct = np.mean(confidences[correct_predictions])
    confidence_incorrect = np.mean(confidences[~correct_predictions])
    
    # Calculate calibration error
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_accuracy = correct_predictions[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            calibration_error += (bin_confidence - bin_accuracy) ** 2
    
    calibration_error = np.sqrt(calibration_error / n_bins)
    
    confidence_metrics = {
        'average_confidence': avg_confidence,
        'confidence_correct': confidence_correct,
        'confidence_incorrect': confidence_incorrect,
        'calibration_error': calibration_error
    }
    
    return confidence_metrics

def plot_attention_analysis(model, input_text, tokenizer, save_path=None):
    """
    Create comprehensive attention analysis visualization.
    
    Args:
        model: The language model
        input_text: Input text to analyze
        tokenizer: Tokenizer for the model
        save_path: Optional path to save the plot
    """
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(input_text)
    input_ids = torch.tensor([tokens]).to(next(model.parameters()).device)
    
    # Get attention weights for all layers and heads
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attentions = outputs.attentions
    
    num_layers = len(attentions)
    num_heads = attentions[0].size(1)
    
    # Create subplot grid
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(4*num_heads, 4*num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    token_labels = tokenizer.convert_ids_to_tokens(tokens)
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attention_weights = attentions[layer_idx][0, head_idx].cpu().numpy()
            
            ax = axes[layer_idx, head_idx]
            sns.heatmap(attention_weights, 
                       xticklabels=token_labels if layer_idx == num_layers-1 else [],
                       yticklabels=token_labels if head_idx == 0 else [],
                       cmap='Blues',
                       ax=ax)
            ax.set_title(f'L{layer_idx+1}H{head_idx+1}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# Example usage functions
def demonstrate_evaluation_tools():
    """Demonstrate various evaluation tools."""
    print("Evaluation and Monitoring Tools Demonstration")
    print("=" * 40)
    
    # Example 1: Perplexity Calculation
    print("1. Perplexity Calculation")
    print("   - Measures how well the model predicts the next token")
    print("   - Lower perplexity indicates better performance")
    print("   - Formula: exp(cross_entropy_loss)")
    print()
    
    # Example 2: Attention Visualization
    print("2. Attention Visualization")
    print("   - Visualizes attention weights between tokens")
    print("   - Helps understand model behavior")
    print("   - Can identify patterns in attention")
    print()
    
    # Example 3: Model Efficiency
    print("3. Model Efficiency Metrics")
    print("   - Inference time measurement")
    print("   - Memory usage tracking")
    print("   - Parameter count analysis")
    print()
    
    # Example 4: Robustness Evaluation
    print("4. Robustness Evaluation")
    print("   - Tests model performance under noise")
    print("   - Evaluates model stability")
    print("   - Identifies potential vulnerabilities")
    print()
    
    # Example 5: Confidence Analysis
    print("5. Confidence Analysis")
    print("   - Measures prediction confidence")
    print("   - Evaluates calibration quality")
    print("   - Helps identify overconfident predictions")
    print()
    
    # Example 6: Training Monitoring
    print("6. Training Monitoring")
    print("   - Tracks training and validation loss")
    print("   - Monitors learning rate schedule")
    print("   - Helps identify training issues")
    print()
    
    # Example usage code
    print("Example Usage:")
    print("""
    # Calculate perplexity
    perplexity = calculate_perplexity(model, val_dataloader, device)
    print(f"Validation Perplexity: {perplexity:.2f}")
    
    # Visualize attention
    visualize_attention(model, "The cat sat on the mat", tokenizer)
    
    # Monitor training
    monitor_training_metrics(train_losses, val_losses, learning_rates)
    
    # Evaluate efficiency
    metrics = calculate_model_efficiency(model, (1, 128), device)
    print(f"Inference time: {metrics['inference_time_ms']:.2f}ms")
    print(f"Memory usage: {metrics['memory_allocated_gb']:.2f}GB")
    """)

if __name__ == "__main__":
    demonstrate_evaluation_tools()

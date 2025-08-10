"""
Scaling Laws for Large Language Models.

This module implements various scaling laws and utilities for determining
optimal model sizes, data requirements, and compute budgets for LLMs.
"""

import math

def compute_optimal_scaling(compute_budget_flops):
    """
    Compute optimal model size and data size given compute budget using Chinchilla scaling laws.
    
    Args:
        compute_budget_flops: Total compute budget in FLOPs
    
    Returns:
        optimal_params: Optimal number of parameters
        optimal_tokens: Optimal number of training tokens
    """
    # Chinchilla scaling constants
    A = 6.9e13
    B = 1.4e13
    alpha = 0.28
    beta = 3.65
    
    # Compute optimal parameters
    optimal_params = A * (compute_budget_flops ** alpha)
    
    # Compute optimal tokens
    optimal_tokens = B * (optimal_params ** beta)
    
    return int(optimal_params), int(optimal_tokens)

def estimate_data_requirements(model_size_billions, tokens_per_epoch=1e12):
    """
    Estimate data requirements for different model sizes.
    
    Args:
        model_size_billions: Model size in billions of parameters
        tokens_per_epoch: Tokens per training epoch
    
    Returns:
        epochs_needed: Number of training epochs
        total_tokens: Total tokens needed
    """
    # Chinchilla optimal data ratio
    optimal_tokens_per_param = 20  # tokens per parameter
    
    total_tokens_needed = model_size_billions * 1e9 * optimal_tokens_per_param
    epochs_needed = total_tokens_needed / tokens_per_epoch
    
    return epochs_needed, total_tokens_needed

def estimate_compute_requirements(model_size_billions, sequence_length=2048, batch_size=1):
    """
    Estimate compute requirements for training.
    
    Args:
        model_size_billions: Model size in billions of parameters
        sequence_length: Training sequence length
        batch_size: Training batch size
    
    Returns:
        flops_per_token: FLOPs per token
        memory_gb: Memory requirements in GB
    """
    params = model_size_billions * 1e9
    
    # FLOPs per token (2 forward + 1 backward = 3x forward)
    flops_per_token = 3 * 2 * params * sequence_length
    
    # Memory estimation (parameters + activations)
    param_memory = params * 4  # 4 bytes per parameter (FP32)
    activation_memory = batch_size * sequence_length * params * 4  # activations
    total_memory_gb = (param_memory + activation_memory) / 1e9
    
    return flops_per_token, total_memory_gb

def calculate_training_time(compute_requirements_flops, hardware_flops_per_second):
    """
    Calculate estimated training time given compute requirements and hardware.
    
    Args:
        compute_requirements_flops: Total compute requirements in FLOPs
        hardware_flops_per_second: Hardware compute capacity in FLOPs/second
    
    Returns:
        training_time_seconds: Estimated training time in seconds
        training_time_days: Estimated training time in days
    """
    training_time_seconds = compute_requirements_flops / hardware_flops_per_second
    training_time_days = training_time_seconds / (24 * 3600)
    
    return training_time_seconds, training_time_days

def analyze_scaling_efficiency(model_sizes_billions, compute_budget_flops):
    """
    Analyze scaling efficiency for different model sizes.
    
    Args:
        model_sizes_billions: List of model sizes in billions
        compute_budget_flops: Available compute budget
    
    Returns:
        dict: Analysis results for each model size
    """
    results = {}
    
    for size in model_sizes_billions:
        # Calculate requirements
        flops_per_token, memory_gb = estimate_compute_requirements(size)
        epochs_needed, total_tokens = estimate_data_requirements(size)
        
        # Calculate total compute needed
        total_compute_needed = flops_per_token * total_tokens
        
        # Calculate efficiency (how much of budget is used)
        efficiency = total_compute_needed / compute_budget_flops
        
        results[size] = {
            'flops_per_token': flops_per_token,
            'memory_gb': memory_gb,
            'epochs_needed': epochs_needed,
            'total_tokens': total_tokens,
            'total_compute_needed': total_compute_needed,
            'efficiency': efficiency
        }
    
    return results

# Example usage functions
def demonstrate_scaling_laws():
    """Demonstrate scaling laws with example calculations."""
    print("Scaling Laws Demonstration")
    print("=" * 40)
    
    # Example 1: Optimal scaling for given compute budget
    compute_budget = 1e24  # 1 ZettaFLOP
    params, tokens = compute_optimal_scaling(compute_budget)
    print(f"Compute Budget: {compute_budget/1e24:.1f} ZettaFLOPs")
    print(f"Optimal Parameters: {params/1e9:.1f}B")
    print(f"Optimal Tokens: {tokens/1e12:.1f}T")
    print()
    
    # Example 2: Data requirements for different model sizes
    model_sizes = [1, 7, 70, 175, 540]  # billions of parameters
    print("Data Requirements Analysis:")
    for size in model_sizes:
        epochs, tokens = estimate_data_requirements(size)
        print(f"{size:3d}B model: {epochs:6.1f} epochs, {tokens/1e12:5.1f}T tokens")
    print()
    
    # Example 3: Compute requirements
    print("Compute Requirements Analysis:")
    for size in [1, 7, 70, 175]:
        flops, memory = estimate_compute_requirements(size)
        print(f"{size:3d}B model: {flops/1e12:5.1f}T FLOPs/token, {memory:6.1f}GB memory")
    print()
    
    # Example 4: Scaling efficiency analysis
    compute_budget = 1e24
    model_sizes = [1, 7, 70, 175]
    results = analyze_scaling_efficiency(model_sizes, compute_budget)
    
    print("Scaling Efficiency Analysis:")
    for size, result in results.items():
        print(f"{size:3d}B model: {result['efficiency']*100:5.1f}% budget efficiency")

if __name__ == "__main__":
    demonstrate_scaling_laws()

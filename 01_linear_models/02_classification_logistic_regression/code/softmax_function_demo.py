import numpy as np
import matplotlib.pyplot as plt

def demonstrate_softmax_function():
    """Demonstrate the softmax function step by step"""
    
    # Example logits (scores)
    logits = np.array([2.0, 1.0, 0.1])
    
    print("Softmax Function Demonstration")
    print("=" * 40)
    print(f"Input logits: {logits}")
    print()
    
    # Step 1: Exponentiation
    exp_logits = np.exp(logits)
    print("Step 1: Exponentiation")
    print(f"exp(logits) = {exp_logits}")
    print()
    
    # Step 2: Sum of exponentials
    sum_exp = np.sum(exp_logits)
    print("Step 2: Sum of exponentials")
    print(f"sum = {sum_exp}")
    print()
    
    # Step 3: Normalization
    probabilities = exp_logits / sum_exp
    print("Step 3: Normalization")
    print(f"Probabilities = {probabilities}")
    print(f"Sum of probabilities = {np.sum(probabilities):.6f}")
    print()
    
    # Verify properties
    print("Softmax Properties:")
    print(f"All probabilities ≥ 0: {np.all(probabilities >= 0)}")
    print(f"Sum = 1: {np.abs(np.sum(probabilities) - 1) < 1e-10}")
    print(f"Ordering preserved: {np.argmax(logits) == np.argmax(probabilities)}")
    print()
    
    # Compare with different logits
    print("Effect of Logit Changes:")
    print("-" * 30)
    
    # Case 1: Large differences
    logits_large = np.array([5.0, 1.0, 0.1])
    probs_large = np.exp(logits_large) / np.sum(np.exp(logits_large))
    print(f"Large differences: {logits_large} → {probs_large}")
    
    # Case 2: Small differences
    logits_small = np.array([1.1, 1.0, 0.9])
    probs_small = np.exp(logits_small) / np.sum(np.exp(logits_small))
    print(f"Small differences: {logits_small} → {probs_small}")
    
    # Case 3: Equal logits
    logits_equal = np.array([1.0, 1.0, 1.0])
    probs_equal = np.exp(logits_equal) / np.sum(np.exp(logits_equal))
    print(f"Equal logits: {logits_equal} → {probs_equal}")
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Softmax transformation
    plt.subplot(1, 3, 1)
    x_pos = np.arange(len(logits))
    plt.bar(x_pos - 0.2, logits, width=0.4, label='Logits', alpha=0.7)
    plt.bar(x_pos + 0.2, probabilities, width=0.4, label='Probabilities', alpha=0.7)
    plt.xlabel('Class')
    plt.ylabel('Value')
    plt.title('Softmax Transformation')
    plt.legend()
    plt.xticks(x_pos, [f'Class {i}' for i in range(len(logits))])
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Effect of logit magnitude
    plt.subplot(1, 3, 2)
    logit_scales = [0.1, 1.0, 5.0, 10.0]
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, scale in enumerate(logit_scales):
        scaled_logits = logits * scale
        scaled_probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
        plt.plot(range(len(scaled_probs)), scaled_probs, 'o-', 
                color=colors[i], label=f'Scale: {scale}', linewidth=2, markersize=6)
    
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Effect of Logit Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Temperature scaling
    plt.subplot(1, 3, 3)
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    for i, temp in enumerate(temperatures):
        temp_probs = np.exp(logits / temp) / np.sum(np.exp(logits / temp))
        plt.plot(range(len(temp_probs)), temp_probs, 'o-', 
                color=colors[i % len(colors)], label=f'T: {temp}', 
                linewidth=2, markersize=6)
    
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Temperature Scaling Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("Key Insights:")
    print("-" * 20)
    print("1. Exponentiation amplifies differences between logits")
    print("2. Normalization ensures probabilities sum to 1")
    print("3. Larger logit differences lead to more confident predictions")
    print("4. Temperature scaling controls prediction sharpness")
    print("5. Softmax preserves ordering of input logits")
    
    return logits, probabilities

if __name__ == "__main__":
    softmax_demo = demonstrate_softmax_function()

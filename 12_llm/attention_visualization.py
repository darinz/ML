import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens, save_path=None):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Tensor of shape (seq_len, seq_len)
        tokens: List of token strings
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights.detach().numpy(), 
                xticklabels=tokens, 
                yticklabels=tokens,
                cmap='Blues',
                annot=True,
                fmt='.2f')
    plt.title('Attention Weights Heatmap')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# Example usage
if __name__ == "__main__":
    tokens = ['The', 'cat', 'sat', 'on', 'the', 'mat']
    attention_weights = torch.randn(6, 6)  # Example weights
    visualize_attention(attention_weights, tokens)

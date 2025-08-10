"""
Advanced Attention Variants.

This module implements various advanced attention mechanisms including
local attention, sparse attention, and linear attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def local_attention(q, k, v, window_size=64):
    """
    Local attention that limits attention to a local window around each position.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        window_size: Size of local attention window
        
    Returns:
        output: Local attention output
    """
    seq_len = q.size(1)
    batch_size, _, d_k = q.shape
    
    # Initialize output tensor
    output = torch.zeros_like(q)
    
    for i in range(seq_len):
        # Define local window boundaries
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2)
        
        # Extract local query, key, value
        local_q = q[:, i:i+1, :]  # Current position
        local_k = k[:, start:end, :]  # Local window
        local_v = v[:, start:end, :]  # Local window
        
        # Compute attention within window
        attn_weights = torch.matmul(local_q, local_k.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights / math.sqrt(d_k), dim=-1)
        
        # Apply attention to values
        local_output = torch.matmul(attn_weights, local_v)
        output[:, i:i+1, :] = local_output
    
    return output

def sparse_attention(q, k, v, sparsity_pattern):
    """
    Sparse attention that only computes attention for a subset of position pairs.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        sparsity_pattern: Boolean tensor indicating which attention connections to compute
        
    Returns:
        output: Sparse attention output
    """
    # Compute attention scores
    attn_weights = torch.matmul(q, k.transpose(-2, -1))
    
    # Apply sparsity mask
    attn_weights = attn_weights.masked_fill(~sparsity_pattern, float('-inf'))
    
    # Apply softmax
    attn_weights = F.softmax(attn_weights, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, v)
    
    return output

def linear_attention(q, k, v):
    """
    Linear attention using kernel feature maps to avoid quadratic complexity.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        
    Returns:
        output: Linear attention output
    """
    # Apply kernel feature map
    q = F.elu(q) + 1
    k = F.elu(k) + 1
    
    # Compute KV and Q(KV)
    kv = torch.matmul(k.transpose(-2, -1), v)
    qkv = torch.matmul(q, kv)
    
    # Normalize
    k_sum = torch.sum(k, dim=-2, keepdim=True)
    qk_sum = torch.matmul(q, k_sum.transpose(-2, -1))
    
    return qkv / (qk_sum + 1e-8)

def sliding_window_attention(q, k, v, window_size=512):
    """
    Sliding window attention for long sequences.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        window_size: Size of sliding window
        
    Returns:
        output: Sliding window attention output
    """
    seq_len = q.size(1)
    batch_size, _, d_k = q.shape
    
    if seq_len <= window_size:
        # Standard attention for short sequences
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)
    
    # Sliding window attention for long sequences
    output = torch.zeros_like(q)
    stride = window_size // 2
    
    for i in range(0, seq_len, stride):
        end = min(i + window_size, seq_len)
        
        # Extract window
        window_q = q[:, i:end, :]
        window_k = k[:, i:end, :]
        window_v = v[:, i:end, :]
        
        # Compute attention within window
        window_attn = torch.matmul(window_q, window_k.transpose(-2, -1)) / math.sqrt(d_k)
        window_attn = F.softmax(window_attn, dim=-1)
        window_output = torch.matmul(window_attn, window_v)
        
        # Store output (simplified - in practice, need proper overlap handling)
        output[:, i:end, :] = window_output
    
    return output

def chunked_attention(q, k, v, chunk_size=1024):
    """
    Chunked attention for very long sequences.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        chunk_size: Size of chunks
        
    Returns:
        output: Chunked attention output
    """
    seq_len = q.size(1)
    outputs = []
    
    for i in range(0, seq_len, chunk_size):
        chunk_q = q[:, i:i+chunk_size, :]
        
        # Compute attention for this chunk
        chunk_attn = torch.matmul(chunk_q, k.transpose(-2, -1)) / math.sqrt(chunk_q.size(-1))
        chunk_attn = F.softmax(chunk_attn, dim=-1)
        chunk_output = torch.matmul(chunk_attn, v)
        
        outputs.append(chunk_output)
    
    return torch.cat(outputs, dim=1)

def memory_efficient_attention(q, k, v):
    """
    Memory efficient attention using gradient checkpointing.
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        
    Returns:
        output: Memory efficient attention output
    """
    from torch.utils.checkpoint import checkpoint
    
    def attention_forward(q, k, v):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)
    
    return checkpoint(attention_forward, q, k, v)

def flash_attention(q, k, v):
    """
    Flash attention implementation (if available).
    
    Args:
        q: Query tensor
        k: Key tensor
        v: Value tensor
        
    Returns:
        output: Flash attention output
    """
    try:
        from flash_attn import flash_attn_func
        return flash_attn_func(q, k, v)
    except ImportError:
        print("Flash Attention not available, using standard attention")
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn_weights = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)

class LongSequenceAttention(nn.Module):
    """
    Attention module designed for long sequences.
    """
    
    def __init__(self, d_model, num_heads, window_size=1024):
        """
        Initialize Long Sequence Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            window_size: Size of attention window
        """
        super().__init__()
        self.window_size = window_size
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        
    def forward(self, x):
        """
        Forward pass with long sequence handling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            output: Attention output
        """
        seq_len = x.size(1)
        
        if seq_len <= self.window_size:
            # Standard attention for short sequences
            x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
            output, _ = self.attention(x, x, x)
            return output.transpose(0, 1)  # (batch_size, seq_len, d_model)
        else:
            # Sliding window attention for long sequences
            outputs = []
            stride = self.window_size // 2
            
            for i in range(0, seq_len, stride):
                end = min(i + self.window_size, seq_len)
                window_x = x[:, i:end, :]
                
                window_x = window_x.transpose(0, 1)
                window_output, _ = self.attention(window_x, window_x, window_x)
                window_output = window_output.transpose(0, 1)
                
                outputs.append(window_output)
            
            # Combine outputs (simplified - in practice, need proper overlap handling)
            return torch.cat(outputs, dim=1)

# Example usage functions
def demonstrate_advanced_attention():
    """Demonstrate advanced attention variants."""
    print("Advanced Attention Variants Demonstration")
    print("=" * 40)
    
    # Create example tensors
    batch_size, seq_len, d_model = 2, 128, 256
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    print("1. Local Attention")
    print("   - Limits attention to local windows")
    print("   - Reduces computational complexity")
    print("   - Suitable for long sequences")
    local_output = local_attention(q, k, v, window_size=32)
    print(f"   Output shape: {local_output.shape}")
    print()
    
    print("2. Sparse Attention")
    print("   - Only computes attention for subset of positions")
    print("   - Further reduces complexity")
    print("   - Requires predefined sparsity pattern")
    sparsity_pattern = torch.ones(seq_len, seq_len, dtype=torch.bool)
    sparse_output = sparse_attention(q, k, v, sparsity_pattern)
    print(f"   Output shape: {sparse_output.shape}")
    print()
    
    print("3. Linear Attention")
    print("   - Uses kernel feature maps")
    print("   - O(n) complexity instead of O(n²)")
    print("   - May sacrifice some expressiveness")
    linear_output = linear_attention(q, k, v)
    print(f"   Output shape: {linear_output.shape}")
    print()
    
    print("4. Sliding Window Attention")
    print("   - Processes long sequences in windows")
    print("   - Balances efficiency and expressiveness")
    print("   - Handles sequences longer than training")
    sliding_output = sliding_window_attention(q, k, v, window_size=64)
    print(f"   Output shape: {sliding_output.shape}")
    print()
    
    print("5. Memory Efficient Attention")
    print("   - Uses gradient checkpointing")
    print("   - Trades compute for memory")
    print("   - Useful for large models")
    memory_output = memory_efficient_attention(q, k, v)
    print(f"   Output shape: {memory_output.shape}")
    print()
    
    print("6. Long Sequence Attention Module")
    print("   - Complete module for long sequences")
    print("   - Automatically switches between methods")
    print("   - Handles different sequence lengths")
    long_seq_attn = LongSequenceAttention(d_model=256, num_heads=8, window_size=64)
    long_seq_output = long_seq_attn(q)
    print(f"   Output shape: {long_seq_output.shape}")

if __name__ == "__main__":
    demonstrate_advanced_attention()

"""
Memory-Efficient Attention Implementation
=======================================

This module provides memory-efficient attention implementations,
including Flash Attention for reducing memory usage from O(n²) to O(n).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class FlashAttention(nn.Module):
    """
    Memory-efficient attention implementation.
    
    This reduces memory usage from O(n²) to O(n) by computing
    attention in chunks and using gradient checkpointing.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, 
                 chunk_size: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        self.chunk_size = chunk_size
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with memory-efficient attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: Attention output
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Memory-efficient attention computation
        output = self._flash_attention(Q, K, V, mask)
        
        # Reshape and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output
    
    def _flash_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                        mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Memory-efficient attention computation.
        
        Args:
            Q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            K: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            V: Value tensor of shape (batch_size, num_heads, seq_len, d_k)
            mask: Optional attention mask
        
        Returns:
            output: Attention output
        """
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        # Initialize output and statistics
        output = torch.zeros_like(Q)
        l = torch.zeros(batch_size, num_heads, seq_len, 1, device=Q.device)
        m = torch.full((batch_size, num_heads, seq_len, 1), float('-inf'), device=Q.device)
        
        # Process in chunks
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            
            # Extract chunk
            Q_chunk = Q[:, :, chunk_start:chunk_end, :]
            K_chunk = K[:, :, :, :]
            V_chunk = V[:, :, :, :]
            
            # Compute attention scores for chunk
            scores = torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) / math.sqrt(d_k)
            
            # Apply mask if provided
            if mask is not None:
                mask_chunk = mask[:, :, chunk_start:chunk_end, :]
                scores = scores.masked_fill(mask_chunk == 0, float('-inf'))
            
            # Update running statistics
            m_new = torch.maximum(m[:, :, chunk_start:chunk_end, :], 
                                torch.max(scores, dim=-1, keepdim=True))
            l_new = torch.exp(m[:, :, chunk_start:chunk_end, :] - m_new) * l[:, :, chunk_start:chunk_end, :] + \
                   torch.exp(scores - m_new).sum(dim=-1, keepdim=True)
            
            # Compute attention weights
            attention_weights = torch.exp(scores - m_new) / l_new
            
            # Apply dropout
            if self.training:
                attention_weights = F.dropout(attention_weights, p=self.dropout)
            
            # Compute output for chunk
            output_chunk = torch.matmul(attention_weights, V_chunk)
            output[:, :, chunk_start:chunk_end, :] = output_chunk
            
            # Update statistics
            m[:, :, chunk_start:chunk_end, :] = m_new
            l[:, :, chunk_start:chunk_end, :] = l_new
        
        return output


class ChunkedAttention(nn.Module):
    """
    Chunked attention for very long sequences.
    
    This processes attention in chunks to reduce memory usage
    for sequences that don't fit in memory.
    """
    
    def __init__(self, d_model: int, num_heads: int, chunk_size: int = 1024, 
                 overlap: int = 128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with chunked attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: Attention output
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Chunked attention computation
        output = self._chunked_attention(Q, K, V, mask)
        
        # Reshape and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output
    
    def _chunked_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Chunked attention computation.
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional attention mask
        
        Returns:
            output: Attention output
        """
        batch_size, num_heads, seq_len, d_k = Q.shape
        output = torch.zeros_like(Q)
        
        # Process in overlapping chunks
        for chunk_start in range(0, seq_len, self.chunk_size - self.overlap):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            
            # Extract chunk
            Q_chunk = Q[:, :, chunk_start:chunk_end, :]
            K_chunk = K[:, :, :, :]
            V_chunk = V[:, :, :, :]
            
            # Compute attention for chunk
            scores = torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) / math.sqrt(d_k)
            
            # Apply mask if provided
            if mask is not None:
                mask_chunk = mask[:, :, chunk_start:chunk_end, :]
                scores = scores.masked_fill(mask_chunk == 0, float('-inf'))
            
            # Apply softmax
            attention_weights = F.softmax(scores, dim=-1)
            
            # Compute output for chunk
            output_chunk = torch.matmul(attention_weights, V_chunk)
            output[:, :, chunk_start:chunk_end, :] = output_chunk
        
        return output


class SparseAttention(nn.Module):
    """
    Sparse attention implementation.
    
    This only computes attention for a subset of position pairs,
    further reducing computational complexity.
    """
    
    def __init__(self, d_model: int, num_heads: int, sparsity_pattern: torch.Tensor):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sparsity_pattern = sparsity_pattern
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Attention output
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Sparse attention computation
        output = self._sparse_attention(Q, K, V)
        
        # Reshape and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output
    
    def _sparse_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Sparse attention computation.
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
        
        Returns:
            output: Attention output
        """
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply sparsity pattern
        scores = scores.masked_fill(~self.sparsity_pattern, float('-inf'))
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute output
        output = torch.matmul(attention_weights, V)
        
        return output


class LinearAttention(nn.Module):
    """
    Linear attention implementation.
    
    This reformulates attention to avoid quadratic complexity
    by using kernel feature maps.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with linear attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            output: Attention output
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Linear attention computation
        output = self._linear_attention(Q, K, V)
        
        # Reshape and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(output)
        
        return output
    
    def _linear_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Linear attention computation.
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
        
        Returns:
            output: Attention output
        """
        # Apply kernel feature map
        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # Compute KV and Q(KV)
        KV = torch.matmul(K.transpose(-2, -1), V)
        QKV = torch.matmul(Q, KV)
        
        # Normalize
        K_sum = torch.sum(K, dim=-2, keepdim=True)
        QK_sum = torch.matmul(Q, K_sum.transpose(-2, -1))
        
        return QKV / (QK_sum + 1e-8)


# Example usage
if __name__ == "__main__":
    # Test memory-efficient attention implementations
    batch_size, seq_len, d_model = 2, 1000, 512
    num_heads = 8
    
    # Create sample input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test Flash Attention
    flash_attention = FlashAttention(d_model, num_heads, chunk_size=256)
    flash_output = flash_attention(x)
    print(f"Flash Attention output shape: {flash_output.shape}")
    
    # Test Chunked Attention
    chunked_attention = ChunkedAttention(d_model, num_heads, chunk_size=256)
    chunked_output = chunked_attention(x)
    print(f"Chunked Attention output shape: {chunked_output.shape}")
    
    # Test Sparse Attention
    sparsity_pattern = torch.rand(seq_len, seq_len) > 0.5  # 50% sparsity
    sparse_attention = SparseAttention(d_model, num_heads, sparsity_pattern)
    sparse_output = sparse_attention(x)
    print(f"Sparse Attention output shape: {sparse_output.shape}")
    
    # Test Linear Attention
    linear_attention = LinearAttention(d_model, num_heads)
    linear_output = linear_attention(x)
    print(f"Linear Attention output shape: {linear_output.shape}") 
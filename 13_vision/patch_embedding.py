# Patch Embedding for Vision Transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math
import numpy as np

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    
    Converts images to patches and projects them to embedding space.
    This is a core component of Vision Transformers.
    """
    
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224, 
                 patch_size: Union[int, Tuple[int, int]] = 16, 
                 in_channels: int = 3, embed_dim: int = 768,
                 norm_layer: Optional[nn.Module] = None,
                 flatten: bool = True):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten = flatten
        
        # Calculate number of patches
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        # Convolutional patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of patch embedding.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            patches: Embedded patches of shape (batch_size, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Convolutional embedding
        x = self.proj(x)  # (batch_size, embed_dim, grid_h, grid_w)
        
        if self.flatten:
            x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
            x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        x = self.norm(x)
        return x

class OverlappingPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding.
    
    Creates overlapping patches for better feature extraction.
    """
    
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 stride: Union[int, Tuple[int, int]] = 8,
                 in_channels: int = 3, embed_dim: int = 768,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches with overlap
        self.num_patches = ((self.img_size[0] - self.patch_size[0]) // self.stride[0] + 1) * \
                          ((self.img_size[1] - self.patch_size[1]) // self.stride[1] + 1)
        
        # Convolutional patch embedding with overlap
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of overlapping patch embedding.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            patches: Embedded patches of shape (batch_size, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Convolutional embedding with overlap
        x = self.proj(x)  # (batch_size, embed_dim, grid_h, grid_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        
        return x

class HybridPatchEmbed(nn.Module):
    """
    Hybrid Patch Embedding.
    
    Combines CNN features with patch embedding for better performance.
    """
    
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_channels: int = 3, embed_dim: int = 768,
                 cnn_channels: int = 64, norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.cnn_channels = cnn_channels
        
        # Calculate number of patches
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        # CNN feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, embed_dim, kernel_size=1)
        )
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of hybrid patch embedding.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            patches: Embedded patches of shape (batch_size, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch_size, embed_dim, height, width)
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, embed_dim, grid_h, grid_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        
        return x

class PositionalPatchEmbed(nn.Module):
    """
    Positional Patch Embedding.
    
    Includes positional information in patch embedding.
    """
    
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_channels: int = 3, embed_dim: int = 768,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        
        # Patch embedding
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of positional patch embedding.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            patches: Embedded patches with positional information
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Patch embedding
        x = self.proj(x)  # (batch_size, embed_dim, grid_h, grid_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        return x

class AdaptivePatchEmbed(nn.Module):
    """
    Adaptive Patch Embedding.
    
    Adapts patch size based on image content.
    """
    
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 min_patch_size: int = 8, max_patch_size: int = 32,
                 in_channels: int = 3, embed_dim: int = 768,
                 norm_layer: Optional[nn.Module] = None):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Multiple patch embeddings for different sizes
        self.patch_embeddings = nn.ModuleDict({
            f'patch_{size}': nn.Conv2d(in_channels, embed_dim, kernel_size=size, stride=size)
            for size in range(min_patch_size, max_patch_size + 1, 4)
        })
        
        # Attention mechanism for patch size selection
        self.patch_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of adaptive patch embedding.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            patches: Adaptively embedded patches
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # Get embeddings for different patch sizes
        embeddings = []
        for name, embed in self.patch_embeddings.items():
            emb = embed(x)
            emb = emb.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
            embeddings.append(emb)
        
        # Stack embeddings
        stacked_embeddings = torch.stack(embeddings, dim=1)  # (batch_size, num_sizes, num_patches, embed_dim)
        
        # Use attention to select best patch size
        batch_size, num_sizes, num_patches, embed_dim = stacked_embeddings.shape
        stacked_embeddings = stacked_embeddings.view(batch_size * num_patches, num_sizes, embed_dim)
        
        # Self-attention for patch size selection
        attended_embeddings, _ = self.patch_attention(
            stacked_embeddings, stacked_embeddings, stacked_embeddings
        )
        
        # Take the first embedding (or could use weighted average)
        selected_embeddings = attended_embeddings[:, 0, :]  # (batch_size * num_patches, embed_dim)
        selected_embeddings = selected_embeddings.view(batch_size, num_patches, embed_dim)
        
        # Normalize
        selected_embeddings = self.norm(selected_embeddings)
        
        return selected_embeddings

def create_patch_embedding(embedding_type: str = 'standard', **kwargs):
    """
    Create patch embedding based on type.
    
    Args:
        embedding_type: Type of patch embedding ('standard', 'overlapping', 'hybrid', 'positional', 'adaptive')
        **kwargs: Additional arguments for the embedding
    
    Returns:
        patch_embed: Patch embedding module
    """
    if embedding_type == 'standard':
        return PatchEmbed(**kwargs)
    elif embedding_type == 'overlapping':
        return OverlappingPatchEmbed(**kwargs)
    elif embedding_type == 'hybrid':
        return HybridPatchEmbed(**kwargs)
    elif embedding_type == 'positional':
        return PositionalPatchEmbed(**kwargs)
    elif embedding_type == 'adaptive':
        return AdaptivePatchEmbed(**kwargs)
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")

def visualize_patches(image: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Visualize patches from an image.
    
    Args:
        image: Input image of shape (channels, height, width)
        patch_size: Size of patches
    
    Returns:
        patches: Visualized patches
    """
    C, H, W = image.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    
    # Extract patches
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(C, num_patches_h, num_patches_w, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4)  # (num_patches_h, num_patches_w, C, patch_size, patch_size)
    
    return patches

def patch_to_image(patches: torch.Tensor, patch_size: int, img_size: int) -> torch.Tensor:
    """
    Reconstruct image from patches.
    
    Args:
        patches: Patches of shape (num_patches_h, num_patches_w, channels, patch_size, patch_size)
        patch_size: Size of patches
        img_size: Size of reconstructed image
    
    Returns:
        image: Reconstructed image
    """
    num_patches_h, num_patches_w, C, _, _ = patches.shape
    
    # Reshape patches
    patches = patches.view(num_patches_h * num_patches_w, C, patch_size, patch_size)
    
    # Reconstruct image
    image = patches.view(num_patches_h, num_patches_w, C, patch_size, patch_size)
    image = image.permute(2, 0, 3, 1, 4).contiguous()
    image = image.view(C, num_patches_h * patch_size, num_patches_w * patch_size)
    
    return image

if __name__ == "__main__":
    # Example usage
    batch_size = 4
    channels = 3
    height = width = 224
    embed_dim = 768
    
    # Create input
    x = torch.randn(batch_size, channels, height, width)
    
    # Test different patch embeddings
    patch_embeddings = {
        'standard': PatchEmbed(img_size=224, patch_size=16, embed_dim=embed_dim),
        'overlapping': OverlappingPatchEmbed(img_size=224, patch_size=16, stride=8, embed_dim=embed_dim),
        'hybrid': HybridPatchEmbed(img_size=224, patch_size=16, embed_dim=embed_dim),
        'positional': PositionalPatchEmbed(img_size=224, patch_size=16, embed_dim=embed_dim)
    }
    
    for name, embed in patch_embeddings.items():
        output = embed(x)
        print(f"{name} output shape: {output.shape}")
    
    print("Patch embedding implementations created successfully!") 
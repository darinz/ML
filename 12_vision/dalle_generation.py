"""
DALL-E Style Image Generation

This module implements a DALL-E style image generation model that generates images
from text descriptions using a transformer architecture.

References:
- Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., &
  Sutskever, I. (2021). Zero-shot text-to-image generation. ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional, Union
import random


class DALLETextEncoder(nn.Module):
    """Text encoder for DALL-E."""
    
    def __init__(self, vocab_size: int = 49408, max_length: int = 256, 
                 embed_dim: int = 768, num_layers: int = 12, num_heads: int = 12):
        """
        Initialize DALL-E text encoder.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.ln_final = nn.LayerNorm(embed_dim)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text: Input text tokens (batch_size, seq_length)
            
        Returns:
            Text representations (batch_size, embed_dim)
        """
        # Create position indices
        positions = torch.arange(text.size(1), device=text.device).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(text) + self.position_embedding(positions)
        
        # Transformer
        x = x.transpose(0, 1)  # (seq_length, batch_size, embed_dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, seq_length, embed_dim)
        
        # Layer normalization
        x = self.ln_final(x)
        
        # Pooling (use [EOS] token)
        x = x[:, -1, :]  # (batch_size, embed_dim)
        
        return x


class DALLEImageEncoder(nn.Module):
    """Image encoder for DALL-E (discrete VAE)."""
    
    def __init__(self, image_size: int = 256, num_tokens: int = 8192, 
                 embed_dim: int = 768, num_layers: int = 3):
        """
        Initialize DALL-E image encoder.
        
        Args:
            image_size: Size of input images
            num_tokens: Number of discrete tokens
            embed_dim: Embedding dimension
            num_layers: Number of encoder layers
        """
        super().__init__()
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        
        # Convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Quantization layer
        self.quantizer = nn.Linear(embed_dim, num_tokens)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            
        Returns:
            Tuple of (discrete_tokens, continuous_features)
        """
        # Encode images
        features = self.encoder(x)
        
        # Reshape for quantization
        b, c, h, w = features.shape
        features = features.view(b, c, -1).transpose(1, 2)  # (batch_size, h*w, embed_dim)
        
        # Quantize to discrete tokens
        logits = self.quantizer(features)
        discrete_tokens = torch.argmax(logits, dim=-1)
        
        return discrete_tokens, features


class DALLEImageDecoder(nn.Module):
    """Image decoder for DALL-E (discrete VAE)."""
    
    def __init__(self, image_size: int = 256, num_tokens: int = 8192, 
                 embed_dim: int = 768, num_layers: int = 3):
        """
        Initialize DALL-E image decoder.
        
        Args:
            image_size: Size of output images
            num_tokens: Number of discrete tokens
            embed_dim: Embedding dimension
            num_layers: Number of decoder layers
        """
        super().__init__()
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        
        # Convolutional decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, discrete_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            discrete_tokens: Discrete tokens (batch_size, seq_length)
            
        Returns:
            Reconstructed images (batch_size, 3, height, width)
        """
        # Embed tokens
        embeddings = self.token_embedding(discrete_tokens)
        
        # Reshape for convolution
        b, seq_len, c = embeddings.shape
        h = w = int(np.sqrt(seq_len))
        embeddings = embeddings.transpose(1, 2).view(b, c, h, w)
        
        # Decode images
        images = self.decoder(embeddings)
        
        return images


class DALLETransformer(nn.Module):
    """Transformer for DALL-E image generation."""
    
    def __init__(self, vocab_size: int = 8192, max_length: int = 1024, 
                 embed_dim: int = 768, num_layers: int = 24, num_heads: int = 12):
        """
        Initialize DALL-E transformer.
        
        Args:
            vocab_size: Size of vocabulary (text + image tokens)
            max_length: Maximum sequence length
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, text_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text_tokens: Text tokens (batch_size, text_length)
            image_tokens: Image tokens (batch_size, image_length)
            
        Returns:
            Predicted image tokens (batch_size, image_length, vocab_size)
        """
        # Concatenate text and image tokens
        combined_tokens = torch.cat([text_tokens, image_tokens], dim=1)
        
        # Create position indices
        positions = torch.arange(combined_tokens.size(1), device=combined_tokens.device).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(combined_tokens) + self.position_embedding(positions)
        
        # Transformer
        x = x.transpose(0, 1)  # (seq_length, batch_size, embed_dim)
        x = self.transformer(x, x)
        x = x.transpose(0, 1)  # (batch_size, seq_length, embed_dim)
        
        # Output projection (only for image tokens)
        text_length = text_tokens.size(1)
        image_features = x[:, text_length:, :]
        logits = self.output_projection(image_features)
        
        return logits


class DALLEModel(nn.Module):
    """Complete DALL-E model."""
    
    def __init__(self, vocab_size: int = 8192, max_length: int = 1024, 
                 embed_dim: int = 768, num_tokens: int = 8192):
        """
        Initialize DALL-E model.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            embed_dim: Embedding dimension
            num_tokens: Number of image tokens
        """
        super().__init__()
        
        # Text encoder
        self.text_encoder = DALLETextEncoder(vocab_size=vocab_size//2, embed_dim=embed_dim)
        
        # Image encoder (discrete VAE)
        self.image_encoder = DALLEImageEncoder(num_tokens=num_tokens, embed_dim=embed_dim)
        
        # Image decoder (discrete VAE)
        self.image_decoder = DALLEImageDecoder(num_tokens=num_tokens, embed_dim=embed_dim)
        
        # Transformer
        self.transformer = DALLETransformer(vocab_size=vocab_size, embed_dim=embed_dim)
        
    def forward(self, text: torch.Tensor, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text: Input text tokens (batch_size, text_length)
            images: Input images (batch_size, 3, height, width)
            
        Returns:
            Tuple of (predicted_tokens, reconstructed_images)
        """
        # Encode text
        text_features = self.text_encoder(text)
        
        # Encode images to discrete tokens
        discrete_tokens, image_features = self.image_encoder(images)
        
        # Generate image tokens
        predicted_logits = self.transformer(text, discrete_tokens)
        predicted_tokens = torch.argmax(predicted_logits, dim=-1)
        
        # Decode images
        reconstructed_images = self.image_decoder(discrete_tokens)
        
        return predicted_tokens, reconstructed_images
    
    def generate(self, text: torch.Tensor, max_length: int = 1024, 
                temperature: float = 1.0) -> torch.Tensor:
        """
        Generate images from text.
        
        Args:
            text: Input text tokens (batch_size, text_length)
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated image tokens (batch_size, image_length)
        """
        self.eval()
        
        with torch.no_grad():
            # Encode text
            text_features = self.text_encoder(text)
            
            # Initialize image tokens
            batch_size = text.size(0)
            image_tokens = torch.zeros(batch_size, 0, dtype=torch.long, device=text.device)
            
            # Autoregressive generation
            for _ in range(max_length):
                # Get predictions
                logits = self.transformer(text, image_tokens)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                image_tokens = torch.cat([image_tokens, next_token], dim=1)
                
                # Stop if end token is generated
                if (next_token == 1).any():  # Assuming 1 is end token
                    break
            
            return image_tokens


class DALLELoss(nn.Module):
    """Loss function for DALL-E."""
    
    def __init__(self):
        """Initialize DALL-E loss."""
        super().__init__()
        self.reconstruction_loss = nn.MSELoss()
        self.token_loss = nn.CrossEntropyLoss()
    
    def forward(self, predicted_tokens: torch.Tensor, target_tokens: torch.Tensor,
                reconstructed_images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        """
        Compute DALL-E loss.
        
        Args:
            predicted_tokens: Predicted image tokens
            target_tokens: Target image tokens
            reconstructed_images: Reconstructed images
            target_images: Target images
            
        Returns:
            Loss value
        """
        # Token prediction loss
        token_loss = self.token_loss(predicted_tokens.view(-1, predicted_tokens.size(-1)), 
                                   target_tokens.view(-1))
        
        # Image reconstruction loss
        reconstruction_loss = self.reconstruction_loss(reconstructed_images, target_images)
        
        # Total loss
        total_loss = token_loss + reconstruction_loss
        
        return total_loss


def generate_image_from_text(model: nn.Module, 
                           text: str,
                           tokenizer,
                           device: str = 'cuda',
                           image_size: int = 256) -> torch.Tensor:
    """
    Generate image from text description.
    
    Args:
        model: Trained DALL-E model
        text: Text description
        tokenizer: Text tokenizer
        device: Device to use
        image_size: Size of generated image
        
    Returns:
        Generated image (3, height, width)
    """
    model.eval()
    
    with torch.no_grad():
        # Tokenize text
        text_tokens = tokenizer.tokenize(text).unsqueeze(0).to(device)
        
        # Generate image tokens
        image_tokens = model.generate(text_tokens)
        
        # Decode image
        generated_image = model.image_decoder(image_tokens)
        
        return generated_image.squeeze(0)


def generate_image_variations(model: nn.Module, 
                            image: torch.Tensor,
                            device: str = 'cuda') -> torch.Tensor:
    """
    Generate variations of an input image.
    
    Args:
        model: Trained DALL-E model
        image: Input image (3, height, width)
        device: Device to use
        
    Returns:
        Generated image variation (3, height, width)
    """
    model.eval()
    
    with torch.no_grad():
        # Encode image to tokens
        discrete_tokens, _ = model.image_encoder(image.unsqueeze(0).to(device))
        
        # Generate variations
        # In practice, this would involve more sophisticated sampling
        variation_tokens = discrete_tokens + torch.randint(-1, 2, discrete_tokens.shape, device=device)
        variation_tokens = torch.clamp(variation_tokens, 0, model.image_encoder.num_tokens - 1)
        
        # Decode variation
        variation_image = model.image_decoder(variation_tokens)
        
        return variation_image.squeeze(0)


def interpolate_images(model: nn.Module, 
                     image1: torch.Tensor,
                     image2: torch.Tensor,
                     alpha: float,
                     device: str = 'cuda') -> torch.Tensor:
    """
    Interpolate between two images in token space.
    
    Args:
        model: Trained DALL-E model
        image1: First image (3, height, width)
        image2: Second image (3, height, width)
        alpha: Interpolation factor (0 to 1)
        device: Device to use
        
    Returns:
        Interpolated image (3, height, width)
    """
    model.eval()
    
    with torch.no_grad():
        # Encode images to tokens
        tokens1, _ = model.image_encoder(image1.unsqueeze(0).to(device))
        tokens2, _ = model.image_encoder(image2.unsqueeze(0).to(device))
        
        # Interpolate tokens
        interpolated_tokens = alpha * tokens1.float() + (1 - alpha) * tokens2.float()
        interpolated_tokens = torch.round(interpolated_tokens).long()
        
        # Decode interpolated image
        interpolated_image = model.image_decoder(interpolated_tokens)
        
        return interpolated_image.squeeze(0)


# Example usage
if __name__ == "__main__":
    # Example of how to use the DALL-E implementation
    
    # Create model
    model = DALLEModel(vocab_size=8192, max_length=1024, embed_dim=768, num_tokens=8192)
    
    # Create dummy data
    batch_size = 2
    text_length = 50
    image_size = 256
    
    text_tokens = torch.randint(0, 4096, (batch_size, text_length))
    images = torch.randn(batch_size, 3, image_size, image_size)
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    predicted_tokens, reconstructed_images = model(text_tokens.to(device), images.to(device))
    print(f'Predicted tokens shape: {predicted_tokens.shape}')
    print(f'Reconstructed images shape: {reconstructed_images.shape}')
    
    # Test generation
    dummy_text = "a photo of a cat"
    # In practice, you would use a proper tokenizer
    dummy_tokens = torch.randint(0, 4096, (1, 10)).to(device)
    
    generated_tokens = model.generate(dummy_tokens, max_length=256)
    print(f'Generated tokens shape: {generated_tokens.shape}')
    
    # Test image generation from text
    # generated_image = generate_image_from_text(model, "a beautiful sunset", None, device)
    # print(f'Generated image shape: {generated_image.shape}') 
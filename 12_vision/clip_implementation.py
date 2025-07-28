"""
CLIP: Contrastive Language-Image Pre-training

This module implements CLIP, a vision-language model that learns aligned representations
of images and text through contrastive learning.

References:
- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
  Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning
  transferable visual representations from natural language supervision. ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional, Union
import random
import re


class CLIPImageEncoder(nn.Module):
    """Image encoder for CLIP."""
    
    def __init__(self, encoder: str = 'resnet50', projection_dim: int = 512):
        """
        Initialize CLIP image encoder.
        
        Args:
            encoder: Encoder architecture ('resnet50', 'vit', etc.)
            projection_dim: Dimension of projection output
        """
        super().__init__()
        
        if encoder == 'resnet50':
            import torchvision.models as models
            self.encoder = models.resnet50(pretrained=False)
            # Remove the final classification layer
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
            feature_dim = 2048
        elif encoder == 'resnet101':
            import torchvision.models as models
            self.encoder = models.resnet101(pretrained=False)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        
        # Projection head
        self.projection = nn.Linear(feature_dim, projection_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (batch_size, channels, height, width)
            
        Returns:
            Image representations (batch_size, projection_dim)
        """
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        representations = self.projection(features)
        representations = F.normalize(representations, dim=1)
        
        return representations


class CLIPTextEncoder(nn.Module):
    """Text encoder for CLIP."""
    
    def __init__(self, vocab_size: int = 49408, max_length: int = 77, 
                 projection_dim: int = 512, num_layers: int = 12, num_heads: int = 8):
        """
        Initialize CLIP text encoder.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            projection_dim: Dimension of projection output
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        super().__init__()
        self.max_length = max_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, projection_dim)
        self.position_embedding = nn.Embedding(max_length, projection_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim,
            nhead=num_heads,
            dim_feedforward=projection_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.ln_final = nn.LayerNorm(projection_dim)
        
        # Projection head
        self.projection = nn.Linear(projection_dim, projection_dim)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            text: Input text tokens (batch_size, seq_length)
            
        Returns:
            Text representations (batch_size, projection_dim)
        """
        # Create position indices
        positions = torch.arange(text.size(1), device=text.device).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(text) + self.position_embedding(positions)
        
        # Transformer
        x = x.transpose(0, 1)  # (seq_length, batch_size, projection_dim)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, seq_length, projection_dim)
        
        # Layer normalization
        x = self.ln_final(x)
        
        # Pooling (use [EOS] token)
        x = x[:, -1, :]  # (batch_size, projection_dim)
        
        # Projection
        representations = self.projection(x)
        representations = F.normalize(representations, dim=1)
        
        return representations


class CLIPModel(nn.Module):
    """CLIP model with image and text encoders."""
    
    def __init__(self, image_encoder: str = 'resnet50', vocab_size: int = 49408,
                 max_length: int = 77, projection_dim: int = 512, temperature: float = 0.07):
        """
        Initialize CLIP model.
        
        Args:
            image_encoder: Image encoder architecture
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            projection_dim: Dimension of projection output
            temperature: Temperature parameter for contrastive learning
        """
        super().__init__()
        self.temperature = temperature
        
        # Image encoder
        self.image_encoder = CLIPImageEncoder(image_encoder, projection_dim)
        
        # Text encoder
        self.text_encoder = CLIPTextEncoder(vocab_size, max_length, projection_dim)
        
    def forward(self, images: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images (batch_size, channels, height, width)
            text: Input text tokens (batch_size, seq_length)
            
        Returns:
            Tuple of (image_representations, text_representations)
        """
        image_representations = self.image_encoder(images)
        text_representations = self.text_encoder(text)
        
        return image_representations, text_representations
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images only."""
        return self.image_encoder(images)
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text only."""
        return self.text_encoder(text)


class CLIPLoss(nn.Module):
    """Contrastive loss for CLIP."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize CLIP loss.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_representations: torch.Tensor, 
                text_representations: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            image_representations: Image representations (batch_size, projection_dim)
            text_representations: Text representations (batch_size, projection_dim)
            
        Returns:
            Loss value
        """
        # Compute similarity matrix
        logits = torch.mm(image_representations, text_representations.T) / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Image-to-text loss
        image_loss = F.cross_entropy(logits, labels)
        
        # Text-to-image loss
        text_loss = F.cross_entropy(logits.T, labels)
        
        # Total loss
        loss = (image_loss + text_loss) / 2
        
        return loss


class SimpleTokenizer:
    """Simple tokenizer for CLIP."""
    
    def __init__(self, vocab_size: int = 49408):
        """
        Initialize simple tokenizer.
        
        Args:
            vocab_size: Size of vocabulary
        """
        self.vocab_size = vocab_size
        
        # Simple word-based tokenization
        self.word_to_id = {}
        self.id_to_word = {}
        
        # Add special tokens
        self.word_to_id['<SOS>'] = 0
        self.word_to_id['<EOS>'] = 1
        self.word_to_id['<PAD>'] = 2
        self.word_to_id['<UNK>'] = 3
        
        self.id_to_word[0] = '<SOS>'
        self.id_to_word[1] = '<EOS>'
        self.id_to_word[2] = '<PAD>'
        self.id_to_word[3] = '<UNK>'
        
        # Add common words
        common_words = ['a', 'the', 'is', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                       'and', 'or', 'but', 'not', 'this', 'that', 'these', 'those',
                       'dog', 'cat', 'bird', 'car', 'house', 'tree', 'flower', 'book']
        
        for i, word in enumerate(common_words):
            if i + 4 < vocab_size:
                self.word_to_id[word] = i + 4
                self.id_to_word[i + 4] = word
    
    def tokenize(self, text: str, max_length: int = 77) -> torch.Tensor:
        """
        Tokenize text.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Tokenized text (seq_length,)
        """
        # Simple word-based tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Convert words to IDs
        tokens = [self.word_to_id.get('<SOS>', 0)]
        
        for word in words:
            if len(tokens) >= max_length - 1:
                break
            token_id = self.word_to_id.get(word, self.word_to_id.get('<UNK>', 3))
            tokens.append(token_id)
        
        tokens.append(self.word_to_id.get('<EOS>', 1))
        
        # Pad to max_length
        while len(tokens) < max_length:
            tokens.append(self.word_to_id.get('<PAD>', 2))
        
        return torch.tensor(tokens, dtype=torch.long)


class CLIPDataset:
    """Dataset wrapper for CLIP."""
    
    def __init__(self, dataset, tokenizer, max_length: int = 77):
        """
        Initialize CLIP dataset.
        
        Args:
            dataset: Base dataset with (image, text) pairs
            tokenizer: Text tokenizer
            max_length: Maximum sequence length
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get image-text pair."""
        image, text = self.dataset[idx]
        
        # Tokenize text
        tokens = self.tokenizer.tokenize(text, self.max_length)
        
        return {
            'image': image,
            'text': tokens,
            'text_string': text
        }


def train_clip(model: nn.Module, 
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               num_epochs: int = 100,
               learning_rate: float = 0.001,
               device: str = 'cuda'):
    """
    Train CLIP model.
    
    Args:
        model: CLIP model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    criterion = CLIPLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            text = batch['text'].to(device)
            
            optimizer.zero_grad()
            image_representations, text_representations = model(images, text)
            loss = criterion(image_representations, text_representations)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                text = batch['text'].to(device)
                
                image_representations, text_representations = model(images, text)
                loss = criterion(image_representations, text_representations)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_clip_model.pth')
        
        scheduler.step()


def evaluate_clip(model: nn.Module, 
                 test_loader: torch.utils.data.DataLoader,
                 device: str = 'cuda') -> float:
    """
    Evaluate CLIP model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    criterion = CLIPLoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            text = batch['text'].to(device)
            
            image_representations, text_representations = model(images, text)
            loss = criterion(image_representations, text_representations)
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader)
    return avg_loss


def zero_shot_classification(model: nn.Module, 
                           images: torch.Tensor,
                           class_names: List[str],
                           tokenizer,
                           device: str = 'cuda') -> torch.Tensor:
    """
    Perform zero-shot classification.
    
    Args:
        model: Trained CLIP model
        images: Input images (batch_size, channels, height, width)
        class_names: List of class names
        tokenizer: Text tokenizer
        device: Device to use
        
    Returns:
        Predicted class indices (batch_size,)
    """
    model.eval()
    
    with torch.no_grad():
        # Encode images
        image_representations = model.encode_image(images.to(device))
        
        # Encode class names
        text_representations = []
        for class_name in class_names:
            tokens = tokenizer.tokenize(f"a photo of a {class_name}").to(device)
            text_repr = model.encode_text(tokens.unsqueeze(0))
            text_representations.append(text_repr)
        
        text_representations = torch.cat(text_representations, dim=0)
        
        # Compute similarities
        similarities = torch.mm(image_representations, text_representations.T)
        
        # Get predictions
        predictions = torch.argmax(similarities, dim=1)
        
        return predictions


def image_text_retrieval(model: nn.Module, 
                        images: torch.Tensor,
                        texts: List[str],
                        tokenizer,
                        device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform image-text retrieval.
    
    Args:
        model: Trained CLIP model
        images: Input images (batch_size, channels, height, width)
        texts: List of text queries
        tokenizer: Text tokenizer
        device: Device to use
        
    Returns:
        Tuple of (image_to_text_rankings, text_to_image_rankings)
    """
    model.eval()
    
    with torch.no_grad():
        # Encode images
        image_representations = model.encode_image(images.to(device))
        
        # Encode texts
        text_representations = []
        for text in texts:
            tokens = tokenizer.tokenize(text).to(device)
            text_repr = model.encode_text(tokens.unsqueeze(0))
            text_representations.append(text_repr)
        
        text_representations = torch.cat(text_representations, dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(image_representations, text_representations.T)
        
        # Get rankings
        image_to_text_rankings = torch.argsort(similarity_matrix, dim=1, descending=True)
        text_to_image_rankings = torch.argsort(similarity_matrix.T, dim=1, descending=True)
        
        return image_to_text_rankings, text_to_image_rankings


# Example usage
if __name__ == "__main__":
    # Example of how to use the CLIP implementation
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    
    # Create dataset (you would need to create a proper image-text dataset)
    # For demonstration, we'll create a dummy dataset
    class DummyCLIPDataset:
        def __init__(self, size=1000):
            self.size = size
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Dummy image
            image = torch.randn(3, 224, 224)
            image = self.transform(image)
            
            # Dummy text
            texts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]
            text = random.choice(texts)
            
            return image, text
    
    # Create datasets
    train_dataset = DummyCLIPDataset(1000)
    test_dataset = DummyCLIPDataset(100)
    
    # Create CLIP datasets
    train_clip_dataset = CLIPDataset(train_dataset, tokenizer)
    test_clip_dataset = CLIPDataset(test_dataset, tokenizer)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_clip_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_clip_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = CLIPModel(image_encoder='resnet50', vocab_size=1000, max_length=77, projection_dim=512)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_clip(model, train_loader, test_loader, num_epochs=10, device=device)
    
    # Evaluate model
    loss = evaluate_clip(model, test_loader, device=device)
    print(f'Test Loss: {loss:.4f}')
    
    # Zero-shot classification
    class_names = ["dog", "cat", "bird"]
    dummy_images = torch.randn(5, 3, 224, 224)
    predictions = zero_shot_classification(model, dummy_images, class_names, tokenizer, device)
    print(f'Zero-shot predictions: {predictions}')
    
    # Image-text retrieval
    texts = ["a photo of a dog", "a photo of a cat", "a photo of a bird"]
    image_to_text, text_to_image = image_text_retrieval(model, dummy_images, texts, tokenizer, device)
    print(f'Image-to-text rankings: {image_to_text}')
    print(f'Text-to-image rankings: {text_to_image}') 
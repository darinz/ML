# Contrastive Learning

## Overview

Contrastive learning has emerged as a powerful paradigm for learning visual representations by training models to distinguish between similar and dissimilar data points. By maximizing agreement between different views of the same data while minimizing agreement with views from different data, contrastive learning can learn rich, transferable representations without manual supervision.

### Key Principles

**Core Concepts:**
- **Positive Pairs**: Different views of the same data point
- **Negative Pairs**: Views from different data points
- **Representation Learning**: Learning features that capture semantic similarity
- **Temperature Scaling**: Controlling the sharpness of similarity distributions
- **Data Augmentation**: Creating diverse views for robust learning

## Table of Contents

- [Theoretical Foundations](#theoretical-foundations)
- [SimCLR Framework](#simclr-framework)
- [MoCo (Momentum Contrast)](#moco-momentum-contrast)
- [Advanced Contrastive Methods](#advanced-contrastive-methods)
- [Implementation Examples](#implementation-examples)
- [Evaluation and Applications](#evaluation-and-applications)

## Theoretical Foundations

### Contrastive Learning Objective

The fundamental goal of contrastive learning is to learn representations where similar data points are close together and dissimilar data points are far apart in the representation space.

**Mathematical Formulation:**
```math
\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j^+)/\tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k)/\tau)}
```

Where:
- $`z_i`$: Representation of anchor point
- $`z_j^+`$: Representation of positive pair
- $`z_k`$: Representations of all points (including negatives)
- $`\tau`$: Temperature parameter
- $`\text{sim}`$: Similarity function (typically cosine similarity)

### InfoNCE Loss

The InfoNCE (Information Noise Contrastive Estimation) loss is a widely used contrastive learning objective:

```math
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}_{(x_i, x_j) \sim p_{\text{pos}}} \left[ \log \frac{\exp(\text{sim}(f(x_i), f(x_j))/\tau)}{\sum_{k=1}^{N} \exp(\text{sim}(f(x_i), f(x_k))/\tau)} \right]
```

**Key Properties:**
- **Mutual Information Maximization**: Maximizes mutual information between positive pairs
- **Temperature Control**: $`\tau`$ controls the sharpness of the similarity distribution
- **Negative Sampling**: Requires large number of negative samples for effectiveness

### Representation Learning Theory

**Why Contrastive Learning Works:**
1. **Invariance Learning**: Models learn to be invariant to augmentations
2. **Semantic Similarity**: Similar objects have similar representations
3. **Transfer Learning**: Learned representations transfer to downstream tasks
4. **Robustness**: Representations are robust to noise and variations

## SimCLR Framework

### Architecture Overview

SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is a straightforward yet effective contrastive learning framework.

**Key Components:**
- **Data Augmentation**: Create two views of each image
- **Encoder Network**: Extract representations from augmented views
- **Projection Head**: Project representations to comparison space
- **Contrastive Loss**: Maximize similarity of positive pairs

### Implementation

**Complete SimCLR Implementation:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy

class SimCLR(nn.Module):
    """
    SimCLR: Simple Framework for Contrastive Learning of Visual Representations.
    
    This implementation follows the original SimCLR paper with:
    - ResNet encoder
    - MLP projection head
    - InfoNCE loss
    - Strong data augmentation
    """
    
    def __init__(self, encoder, projection_dim=128, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        
        # Projection head (MLP)
        self.projection = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection head weights."""
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x1, x2):
        """
        Forward pass of SimCLR.
        
        Args:
            x1: First augmented view
            x2: Second augmented view
        
        Returns:
            z1, z2: Projected representations
        """
        # Encode both views
        h1 = self.encoder(x1)  # [batch_size, encoder_dim]
        h2 = self.encoder(x2)  # [batch_size, encoder_dim]
        
        # Project to comparison space
        z1 = self.projection(h1)  # [batch_size, projection_dim]
        z2 = self.projection(h2)  # [batch_size, projection_dim]
        
        # Normalize for cosine similarity
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        return z1, z2
    
    def contrastive_loss(self, z1, z2):
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            z1, z2: Normalized representations from two views
        
        Returns:
            loss: InfoNCE loss
        """
        batch_size = z1.shape[0]
        
        # Concatenate all representations
        representations = torch.cat([z1, z2], dim=0)  # [2*batch_size, projection_dim]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels for positive pairs
        # For each sample, its positive pair is at index i + batch_size
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        labels = labels[~mask].view(2 * batch_size, -1)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss

class SimCLRAugmentation:
    """
    Strong data augmentation for SimCLR.
    
    Implements the augmentation strategy from the original paper:
    - Random crop and resize
    - Random horizontal flip
    - Color jittering
    - Random grayscale
    - Gaussian blur
    """
    
    def __init__(self, size=224, s=1.0):
        self.size = size
        self.s = s
        
        # Color jittering parameters
        self.color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        
        # Gaussian blur
        self.blur = transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        
        # Main transform
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([self.blur], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, x):
        """Apply augmentation twice to create two views."""
        return self.transform(x), self.transform(x)

class SimCLRTrainer:
    """
    Trainer for SimCLR model.
    
    Handles training loop, evaluation, and model saving.
    """
    
    def __init__(self, model, train_loader, val_loader=None, lr=1e-3, weight_decay=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            # Get augmented views
            aug1, aug2 = data[0].to(self.device), data[1].to(self.device)
            
            # Forward pass
            z1, z2 = self.model(aug1, aug2)
            
            # Compute loss
            loss = self.model.contrastive_loss(z1, z2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                aug1, aug2 = data[0].to(self.device), data[1].to(self.device)
                z1, z2 = self.model(aug1, aug2)
                loss = self.model.contrastive_loss(z1, z2)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs, save_path=None):
        """Main training loop."""
        print(f"Starting SimCLR training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate()
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            if val_loss is not None:
                print(f'  Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss is not None and val_loss < self.best_loss:
                self.best_loss = val_loss
                if save_path:
                    self.save_checkpoint(save_path)
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }, path)
        print(f"Checkpoint saved to {path}")
```

### Data Augmentation Strategy

**Key Augmentation Techniques:**
1. **Random Crop and Resize**: Scale between 0.2 and 1.0
2. **Random Horizontal Flip**: 50% probability
3. **Color Jittering**: Brightness, contrast, saturation, hue
4. **Random Grayscale**: 20% probability
5. **Gaussian Blur**: Random blur with varying sigma

**Implementation:**
```python
def create_simclr_augmentation(size=224, s=1.0):
    """
    Create SimCLR-style data augmentation.
    
    Args:
        size: Output image size
        s: Strength of color jittering
    
    Returns:
        transform: Augmentation transform
    """
    # Color jittering
    color_jitter = transforms.ColorJitter(
        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
    )
    
    # Gaussian blur
    blur = transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    
    # Main transform
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([blur], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform
```

## MoCo (Momentum Contrast)

### Architecture Overview

MoCo addresses the challenge of maintaining a large dictionary of negative samples for contrastive learning by introducing a momentum encoder and a queue-based memory bank.

**Key Innovations:**
- **Momentum Encoder**: Slowly updated encoder for consistency
- **Queue**: Large queue of negative samples
- **Momentum Update**: Exponential moving average of encoder parameters

### Implementation

**Complete MoCo Implementation:**
```python
class MoCo(nn.Module):
    """
    MoCo: Momentum Contrast for Unsupervised Visual Representation Learning.
    
    This implementation includes:
    - Momentum encoder
    - Queue-based negative sampling
    - InfoNCE loss
    """
    
    def __init__(self, encoder, queue_size=65536, momentum=0.999, temperature=0.07):
        super().__init__()
        self.encoder_q = encoder  # Query encoder
        self.encoder_k = copy.deepcopy(encoder)  # Key encoder
        
        # Freeze key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        
        # Queue for negative samples
        self.register_buffer("queue", torch.randn(encoder.output_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # Parameters
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        # Projection heads
        self.projection_q = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.projection_k = copy.deepcopy(self.projection_q)
        
        # Freeze key projection
        for param in self.projection_k.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def momentum_update(self):
        """Update key encoder with momentum."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """
        Dequeue and enqueue keys.
        
        Args:
            keys: Key representations to enqueue
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace keys at ptr
        if ptr + batch_size > self.queue_size:
            batch_size = self.queue_size - ptr
            keys = keys[:batch_size]
        
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q, im_k):
        """
        Forward pass of MoCo.
        
        Args:
            im_q: Query images
            im_k: Key images
        
        Returns:
            logits: Similarity logits
            labels: Ground truth labels
        """
        # Query encoder
        q = self.encoder_q(im_q)
        q = self.projection_q(q)
        q = F.normalize(q, dim=1)
        
        # Key encoder
        with torch.no_grad():
            k = self.encoder_k(im_k)
            k = self.projection_k(k)
            k = F.normalize(k, dim=1)
        
        # Compute logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.temperature
        
        # Labels: positive pairs are the first column
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        # Dequeue and enqueue
        self.dequeue_and_enqueue(k)
        
        return logits, labels

class MoCoTrainer:
    """
    Trainer for MoCo model.
    """
    
    def __init__(self, model, train_loader, val_loader=None, lr=1e-3):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 200
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            # Get query and key images
            im_q, im_k = data[0].to(self.device), data[1].to(self.device)
            
            # Forward pass
            logits, labels = self.model(im_q, im_k)
            
            # Compute loss
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Update momentum encoder
            self.model.momentum_update()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}: Loss = {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs):
        """Main training loop."""
        print(f"Starting MoCo training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            print(f'Epoch {epoch+1}/{num_epochs}: Loss = {train_loss:.4f}')
```

### MoCo Variants

**MoCo v2 Improvements:**
```python
class MoCoV2(MoCo):
    """
    MoCo v2 with improved training strategy.
    
    Key improvements:
    - MLP projection head
    - Stronger data augmentation
    - Cosine learning rate scheduling
    """
    
    def __init__(self, encoder, queue_size=65536, momentum=0.999, temperature=0.2):
        super().__init__(encoder, queue_size, momentum, temperature)
        
        # MLP projection head (improved from linear)
        self.projection_q = nn.Sequential(
            nn.Linear(encoder.output_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )
        self.projection_k = copy.deepcopy(self.projection_q)
        
        # Freeze key projection
        for param in self.projection_k.parameters():
            param.requires_grad = False
```

**MoCo v3 Simplifications:**
```python
class MoCoV3(MoCo):
    """
    MoCo v3 with simplified architecture.
    
    Key changes:
    - Simplified projection head
    - Better initialization
    - Improved training stability
    """
    
    def __init__(self, encoder, queue_size=65536, momentum=0.999, temperature=0.2):
        super().__init__(encoder, queue_size, momentum, temperature)
        
        # Simplified projection head
        self.projection_q = nn.Linear(encoder.output_dim, 256)
        self.projection_k = copy.deepcopy(self.projection_q)
        
        # Freeze key projection
        for param in self.projection_k.parameters():
            param.requires_grad = False
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for m in [self.projection_q, self.projection_k]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
```

## Advanced Contrastive Methods

### CLIP (Contrastive Language-Image Pre-training)

**Architecture Overview:**
CLIP learns aligned representations between images and text using contrastive learning.

**Implementation:**
```python
class CLIP(nn.Module):
    """
    CLIP: Contrastive Language-Image Pre-training.
    
    Learns aligned representations between images and text.
    """
    
    def __init__(self, image_encoder, text_encoder, projection_dim=512, temperature=0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature
        
        # Projection layers
        self.image_projection = nn.Linear(image_encoder.output_dim, projection_dim)
        self.text_projection = nn.Linear(text_encoder.output_dim, projection_dim)
    
    def encode_image(self, images):
        """Encode images."""
        features = self.image_encoder(images)
        projected = self.image_projection(features)
        normalized = F.normalize(projected, dim=1)
        return normalized
    
    def encode_text(self, text):
        """Encode text."""
        features = self.text_encoder(text)
        projected = self.text_projection(features)
        normalized = F.normalize(projected, dim=1)
        return normalized
    
    def forward(self, images, text):
        """
        Forward pass of CLIP.
        
        Args:
            images: Batch of images
            text: Batch of text
        
        Returns:
            logits: Similarity logits
            labels: Ground truth labels
        """
        # Encode images and text
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Create labels (diagonal)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        return logits, labels
    
    def contrastive_loss(self, logits, labels):
        """Compute CLIP contrastive loss."""
        # Image-to-text loss
        loss_i2t = F.cross_entropy(logits, labels)
        
        # Text-to-image loss
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # Average loss
        loss = (loss_i2t + loss_t2i) / 2
        return loss
```

### DALL-E Style Generation

**Overview:**
DALL-E generates images from text descriptions using a discrete VAE and transformer architecture.

**Key Components:**
```python
class DiscreteVAE(nn.Module):
    """
    Discrete VAE for image tokenization.
    
    Compresses images to discrete tokens for transformer processing.
    """
    
    def __init__(self, num_tokens=8192, token_dim=512):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, token_dim)
        )
        
        # Codebook
        self.codebook = nn.Parameter(torch.randn(num_tokens, token_dim))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 32 * 32),  # Reconstruct to 32x32
            nn.Sigmoid()
        )
    
    def encode(self, x):
        """Encode images to discrete tokens."""
        features = self.encoder(x)
        
        # Find closest codebook entries
        distances = torch.cdist(features, self.codebook)
        indices = torch.argmin(distances, dim=1)
        
        return indices
    
    def decode(self, indices):
        """Decode discrete tokens to images."""
        embeddings = self.codebook[indices]
        reconstructed = self.decoder(embeddings)
        return reconstructed.view(-1, 3, 32, 32)
    
    def forward(self, x):
        """Forward pass."""
        indices = self.encode(x)
        reconstructed = self.decode(indices)
        return reconstructed, indices

class DALLETransformer(nn.Module):
    """
    Transformer for DALL-E style generation.
    
    Generates image tokens conditioned on text.
    """
    
    def __init__(self, vocab_size, d_model=512, num_layers=12, num_heads=8):
        super().__init__()
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(1024, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, text_tokens, image_tokens=None):
        """
        Forward pass.
        
        Args:
            text_tokens: Text token indices
            image_tokens: Image token indices (for training)
        
        Returns:
            logits: Token predictions
        """
        batch_size = text_tokens.shape[0]
        
        if image_tokens is not None:
            # Training: concatenate text and image tokens
            tokens = torch.cat([text_tokens, image_tokens], dim=1)
        else:
            # Inference: only text tokens
            tokens = text_tokens
        
        # Embeddings
        token_embeddings = self.token_embedding(tokens)
        position_embeddings = self.position_embedding(
            torch.arange(tokens.shape[1], device=tokens.device)
        ).unsqueeze(0)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Transformer
        hidden_states = self.transformer(embeddings)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        return logits
```

## Implementation Examples

### Training Pipeline

**Complete Training Setup:**
```python
def setup_contrastive_training(model_type='simclr', batch_size=256):
    """
    Setup contrastive learning training.
    
    Args:
        model_type: Type of contrastive model ('simclr', 'moco', 'clip')
        batch_size: Training batch size
    
    Returns:
        model: Contrastive learning model
        train_loader: Training data loader
        trainer: Model trainer
    """
    # Create model
    if model_type == 'simclr':
        encoder = create_resnet_encoder()
        model = SimCLR(encoder)
        augmentation = SimCLRAugmentation()
    elif model_type == 'moco':
        encoder = create_resnet_encoder()
        model = MoCo(encoder)
        augmentation = SimCLRAugmentation()
    elif model_type == 'clip':
        image_encoder = create_vision_transformer()
        text_encoder = create_text_transformer()
        model = CLIP(image_encoder, text_encoder)
        augmentation = CLIPAugmentation()
    
    # Create dataset and dataloader
    dataset = ContrastiveDataset(
        data_dir='path/to/data',
        augmentation=augmentation
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create trainer
    if model_type == 'simclr':
        trainer = SimCLRTrainer(model, train_loader)
    elif model_type == 'moco':
        trainer = MoCoTrainer(model, train_loader)
    elif model_type == 'clip':
        trainer = CLIPTrainer(model, train_loader)
    
    return model, train_loader, trainer

def create_resnet_encoder():
    """Create ResNet encoder for contrastive learning."""
    import torchvision.models as models
    
    # Load pre-trained ResNet
    resnet = models.resnet50(pretrained=False)
    
    # Remove classification head
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    encoder.output_dim = 2048  # ResNet-50 feature dimension
    
    return encoder

def create_vision_transformer():
    """Create Vision Transformer for CLIP."""
    from transformers import ViTModel
    
    vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
    vit.output_dim = 768  # ViT-Base feature dimension
    
    return vit

def create_text_transformer():
    """Create text transformer for CLIP."""
    from transformers import BertModel
    
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert.output_dim = 768  # BERT-Base feature dimension
    
    return bert
```

### Evaluation Protocols

**Linear Evaluation:**
```python
def linear_evaluation(encoder, train_loader, val_loader, num_classes=10):
    """
    Evaluate learned representations with linear classifier.
    
    Args:
        encoder: Pre-trained encoder
        train_loader: Training data loader
        val_loader: Validation data loader
        num_classes: Number of classes
    
    Returns:
        accuracy: Classification accuracy
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.eval()
    
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    
    # Linear classifier
    classifier = nn.Linear(encoder.output_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    for epoch in range(50):
        classifier.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                features = encoder(data)
            
            optimizer.zero_grad()
            output = classifier(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            features = encoder(data)
            output = classifier(features)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    return accuracy
```

## Evaluation and Applications

### Transfer Learning

**Feature Extraction:**
```python
def extract_features(encoder, dataloader):
    """
    Extract features from pre-trained encoder.
    
    Args:
        encoder: Pre-trained encoder
        dataloader: Data loader
    
    Returns:
        features: Extracted features
        labels: Corresponding labels
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device)
    encoder.eval()
    
    features = []
    labels = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            feature = encoder(data)
            features.append(feature.cpu())
            labels.append(target)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return features, labels
```

### Zero-Shot Learning

**CLIP Zero-Shot Classification:**
```python
def zero_shot_classification(clip_model, image, class_names, template="a photo of a {}"):
    """
    Perform zero-shot classification using CLIP.
    
    Args:
        clip_model: Pre-trained CLIP model
        image: Input image
        class_names: List of class names
        template: Text template for class names
    
    Returns:
        predictions: Class predictions
        probabilities: Prediction probabilities
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model.to(device)
    clip_model.eval()
    
    # Prepare text prompts
    text_prompts = [template.format(name) for name in class_names]
    
    # Tokenize text
    tokenizer = clip_model.tokenizer
    text_tokens = tokenizer(text_prompts, padding=True, return_tensors='pt').to(device)
    
    # Encode image and text
    with torch.no_grad():
        image_features = clip_model.encode_image(image.unsqueeze(0))
        text_features = clip_model.encode_text(text_tokens)
        
        # Compute similarities
        similarities = torch.matmul(image_features, text_features.T)
        probabilities = torch.softmax(similarities, dim=1)
        
        # Get predictions
        predictions = torch.argmax(probabilities, dim=1)
    
    return predictions, probabilities
```

### Image Retrieval

**Text-to-Image Retrieval:**
```python
def text_to_image_retrieval(clip_model, text_query, image_database, top_k=5):
    """
    Retrieve images matching text query.
    
    Args:
        clip_model: Pre-trained CLIP model
        text_query: Text query
        image_database: Database of images
        top_k: Number of top results to return
    
    Returns:
        top_images: Top matching images
        similarities: Similarity scores
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model.to(device)
    clip_model.eval()
    
    # Encode text query
    tokenizer = clip_model.tokenizer
    text_tokens = tokenizer([text_query], return_tensors='pt').to(device)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        
        # Encode all images
        image_features = []
        for image in image_database:
            features = clip_model.encode_image(image.unsqueeze(0))
            image_features.append(features)
        
        image_features = torch.cat(image_features, dim=0)
        
        # Compute similarities
        similarities = torch.matmul(text_features, image_features.T)
        
        # Get top-k results
        top_similarities, top_indices = torch.topk(similarities, top_k, dim=1)
    
    top_images = [image_database[i] for i in top_indices[0]]
    return top_images, top_similarities[0]
```

## Conclusion

Contrastive learning has revolutionized representation learning in computer vision by enabling models to learn powerful representations from unlabeled data. Key innovations include:

1. **SimCLR**: Simple yet effective framework with strong augmentation
2. **MoCo**: Momentum encoder and queue-based negative sampling
3. **CLIP**: Multi-modal contrastive learning for image-text alignment
4. **Advanced Methods**: DALL-E and other generative contrastive approaches

**Key Takeaways:**
- Contrastive learning can learn rich representations without labels
- Data augmentation is crucial for effective contrastive learning
- Temperature scaling controls the sharpness of similarity distributions
- Learned representations transfer well to downstream tasks

**Future Directions:**
- **Multi-modal Learning**: Combining vision with other modalities
- **Efficiency Improvements**: Reducing computational requirements
- **Better Negative Sampling**: More effective negative sample strategies
- **Real-world Applications**: Deploying in production systems

---

**References:**
- "A Simple Framework for Contrastive Learning of Visual Representations" - Chen et al.
- "Momentum Contrast for Unsupervised Visual Representation Learning" - He et al.
- "Learning Transferable Visual Representations" - Radford et al. (CLIP)
- "Zero-Shot Text-to-Image Generation" - Ramesh et al. (DALL-E) 
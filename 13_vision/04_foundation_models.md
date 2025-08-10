# Foundation Models for Vision

## Overview

Foundation models represent a paradigm shift in computer vision, where large-scale pre-trained models can be applied to a wide range of downstream tasks with minimal fine-tuning. These models leverage massive datasets and computational resources to learn general-purpose visual representations that transfer effectively across domains.

## From Representation Learning to Foundation Models

We've now explored **contrastive learning** - a paradigm that learns visual representations by training models to distinguish between positive pairs (different views of the same data) and negative pairs (views from different data). We've seen how frameworks like SimCLR, MoCo, and BYOL enable effective contrastive learning, how data augmentation creates diverse views for robust learning, and how contrastive learning has become the dominant approach for self-supervised visual representation learning.

However, while contrastive learning provides powerful representations, **the true potential** of modern computer vision lies in foundation models - large-scale pre-trained models that can be applied to a wide range of downstream tasks with minimal fine-tuning. Consider CLIP, which can perform zero-shot classification on any visual category, or SAM, which can segment any object in any image - these models demonstrate capabilities that go far beyond traditional supervised learning.

This motivates our exploration of **foundation models for vision** - large-scale models that leverage massive datasets and computational resources to learn general-purpose visual representations. We'll see how CLIP enables zero-shot classification and retrieval through vision-language alignment, how SAM provides universal segmentation capabilities, how DALL-E demonstrates text-to-image generation, and how these models represent a paradigm shift in computer vision.

The transition from contrastive learning to foundation models represents the bridge from representation learning to general-purpose AI - taking our understanding of learning visual representations and applying it to building models that can handle multiple vision tasks with unprecedented flexibility.

In this section, we'll explore foundation models, understanding how large-scale pre-training enables zero-shot capabilities and multi-task performance.

### Key Characteristics

**Core Features:**
- **Scale**: Trained on massive datasets with billions of parameters
- **Generalization**: Strong performance across diverse tasks and domains
- **Zero-shot Capabilities**: Can perform tasks without task-specific training
- **Multi-modal Integration**: Often combine vision with language or other modalities
- **Prompt-based Interaction**: Can be controlled through natural language prompts

## Table of Contents

- [CLIP (Contrastive Language-Image Pre-training)](#clip-contrastive-language-image-pre-training)
- [SAM (Segment Anything Model)](#sam-segment-anything-model)
- [DALL-E and Image Generation](#dall-e-and-image-generation)
- [Implementation Examples](#implementation-examples)
- [Applications and Use Cases](#applications-and-use-cases)

## CLIP (Contrastive Language-Image Pre-training)

### Architecture Overview

CLIP learns aligned representations between images and text using contrastive learning on a massive dataset of image-text pairs.

**Key Components:**
- **Image Encoder**: Vision transformer or ResNet backbone
- **Text Encoder**: Transformer for text processing
- **Contrastive Learning**: Aligns image and text representations
- **Zero-shot Transfer**: Enables classification without training

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import clip

class CLIPModel(nn.Module):
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for m in [self.image_projection, self.text_projection]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode_image(self, images):
        """Encode images to normalized features."""
        features = self.image_encoder(images)
        projected = self.image_projection(features)
        normalized = F.normalize(projected, dim=1)
        return normalized
    
    def encode_text(self, text):
        """Encode text to normalized features."""
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

class CLIPZeroShotClassifier:
    """
    Zero-shot classifier using CLIP.
    """
    
    def __init__(self, clip_model, class_names, template="a photo of a {}"):
        self.clip_model = clip_model
        self.class_names = class_names
        self.template = template
        
        # Prepare text prompts
        self.text_prompts = [template.format(name) for name in class_names]
        
        # Tokenize text
        self.tokenizer = clip_model.tokenizer
        self.text_tokens = self.tokenizer(self.text_prompts, padding=True, return_tensors='pt')
        
        # Encode text features (cached)
        with torch.no_grad():
            self.text_features = clip_model.encode_text(self.text_tokens)
    
    def predict(self, images):
        """
        Predict class labels for images.
        
        Args:
            images: Batch of images
        
        Returns:
            predictions: Predicted class indices
            probabilities: Prediction probabilities
        """
        device = images.device
        self.text_features = self.text_features.to(device)
        
        # Encode images
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            
            # Compute similarities
            similarities = torch.matmul(image_features, self.text_features.T)
            probabilities = torch.softmax(similarities, dim=1)
            
            # Get predictions
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions, probabilities

def create_clip_model(image_encoder_type='vit', text_encoder_type='bert'):
    """
    Create CLIP model with specified encoders.
    
    Args:
        image_encoder_type: Type of image encoder ('vit', 'resnet')
        text_encoder_type: Type of text encoder ('bert', 'gpt')
    
    Returns:
        clip_model: CLIP model
    """
    # Image encoder
    if image_encoder_type == 'vit':
        from transformers import ViTModel
        image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        image_encoder.output_dim = 768
    elif image_encoder_type == 'resnet':
        import torchvision.models as models
        image_encoder = models.resnet50(pretrained=False)
        image_encoder = nn.Sequential(*list(image_encoder.children())[:-1])
        image_encoder.output_dim = 2048
    
    # Text encoder
    if text_encoder_type == 'bert':
        text_encoder = AutoModel.from_pretrained('bert-base-uncased')
        text_encoder.output_dim = 768
    elif text_encoder_type == 'gpt':
        text_encoder = AutoModel.from_pretrained('gpt2')
        text_encoder.output_dim = 768
    
    # Create CLIP model
    clip_model = CLIPModel(image_encoder, text_encoder)
    
    return clip_model
```

### Training and Fine-tuning

```python
class CLIPTrainer:
    """
    Trainer for CLIP model.
    """
    
    def __init__(self, model, train_loader, val_loader=None, lr=1e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * 100
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, text) in enumerate(self.train_loader):
            images = images.to(self.device)
            text = text.to(self.device)
            
            # Forward pass
            logits, labels = self.model(images, text)
            
            # Compute loss
            loss = self.model.contrastive_loss(logits, labels)
            
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
    
    def train(self, num_epochs):
        """Main training loop."""
        print(f"Starting CLIP training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            print(f'Epoch {epoch+1}/{num_epochs}: Loss = {train_loss:.4f}')
```

## SAM (Segment Anything Model)

### Architecture Overview

SAM is a foundation model for image segmentation that can segment any object in any image using various types of prompts.

**Key Components:**
- **Image Encoder**: Vision transformer for image processing
- **Prompt Encoder**: Encodes point, box, or text prompts
- **Mask Decoder**: Generates segmentation masks
- **Ambitious Data Engine**: Large-scale data collection strategy

### Implementation

```python
class SAMImageEncoder(nn.Module):
    """
    Image encoder for SAM.
    
    Uses Vision Transformer to process images.
    """
    
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=12,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        """
        Forward pass of image encoder.
        
        Args:
            x: Input images of shape (batch_size, 3, img_size, img_size)
        
        Returns:
            features: Image features of shape (batch_size, num_patches, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, embed_dim, H, W)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        features = self.transformer(x)
        
        return features

class SAMPromptEncoder(nn.Module):
    """
    Prompt encoder for SAM.
    
    Encodes point, box, and text prompts.
    """
    
    def __init__(self, embed_dim=256, image_embedding_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        
        # Point embedding
        self.point_embed = nn.Embedding(1, embed_dim)
        
        # Box embedding
        self.box_embed = nn.Linear(4, embed_dim)
        
        # Text embedding (simplified)
        self.text_embed = nn.Linear(512, embed_dim)  # Assuming 512-dim text features
        
        # Prompt transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.prompt_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    def forward(self, points=None, boxes=None, text=None):
        """
        Forward pass of prompt encoder.
        
        Args:
            points: Point coordinates of shape (batch_size, num_points, 2)
            boxes: Box coordinates of shape (batch_size, num_boxes, 4)
            text: Text features of shape (batch_size, text_dim)
        
        Returns:
            prompt_embeddings: Encoded prompts
        """
        batch_size = 1
        if points is not None:
            batch_size = points.shape[0]
        elif boxes is not None:
            batch_size = boxes.shape[0]
        elif text is not None:
            batch_size = text.shape[0]
        
        prompt_embeddings = []
        
        # Encode points
        if points is not None:
            point_embeddings = self.point_embed(torch.zeros_like(points[:, :, 0]))
            prompt_embeddings.append(point_embeddings)
        
        # Encode boxes
        if boxes is not None:
            box_embeddings = self.box_embed(boxes)
            prompt_embeddings.append(box_embeddings)
        
        # Encode text
        if text is not None:
            text_embeddings = self.text_embed(text).unsqueeze(1)
            prompt_embeddings.append(text_embeddings)
        
        # Concatenate all embeddings
        if prompt_embeddings:
            prompt_embeddings = torch.cat(prompt_embeddings, dim=1)
            
            # Apply transformer
            prompt_embeddings = self.prompt_transformer(prompt_embeddings)
        else:
            prompt_embeddings = torch.zeros(batch_size, 0, self.embed_dim)
        
        return prompt_embeddings

class SAMMaskDecoder(nn.Module):
    """
    Mask decoder for SAM.
    
    Generates segmentation masks from image and prompt features.
    """
    
    def __init__(self, image_embed_dim=768, prompt_embed_dim=256, mask_dim=256):
        super().__init__()
        self.image_embed_dim = image_embed_dim
        self.prompt_embed_dim = prompt_embed_dim
        self.mask_dim = mask_dim
        
        # Image feature projection
        self.image_projection = nn.Linear(image_embed_dim, mask_dim)
        
        # Prompt feature projection
        self.prompt_projection = nn.Linear(prompt_embed_dim, mask_dim)
        
        # Mask prediction head
        self.mask_head = nn.Sequential(
            nn.Linear(mask_dim, mask_dim),
            nn.ReLU(),
            nn.Linear(mask_dim, mask_dim),
            nn.ReLU(),
            nn.Linear(mask_dim, 1),
            nn.Sigmoid()
        )
        
        # IoU prediction head
        self.iou_head = nn.Sequential(
            nn.Linear(mask_dim, mask_dim),
            nn.ReLU(),
            nn.Linear(mask_dim, 1)
        )
    
    def forward(self, image_embeddings, prompt_embeddings):
        """
        Forward pass of mask decoder.
        
        Args:
            image_embeddings: Image features of shape (batch_size, num_patches, image_embed_dim)
            prompt_embeddings: Prompt features of shape (batch_size, num_prompts, prompt_embed_dim)
        
        Returns:
            masks: Predicted masks of shape (batch_size, num_prompts, H, W)
            iou_scores: IoU scores of shape (batch_size, num_prompts)
        """
        batch_size, num_patches, _ = image_embeddings.shape
        num_prompts = prompt_embeddings.shape[1]
        
        # Project features
        image_features = self.image_projection(image_embeddings)  # (batch_size, num_patches, mask_dim)
        prompt_features = self.prompt_projection(prompt_embeddings)  # (batch_size, num_prompts, mask_dim)
        
        # Expand for broadcasting
        image_features = image_features.unsqueeze(1).expand(-1, num_prompts, -1, -1)
        prompt_features = prompt_features.unsqueeze(2).expand(-1, -1, num_patches, -1)
        
        # Combine features
        combined_features = image_features + prompt_features
        
        # Predict masks
        mask_logits = self.mask_head(combined_features)  # (batch_size, num_prompts, num_patches, 1)
        masks = mask_logits.squeeze(-1).view(batch_size, num_prompts, int(num_patches**0.5), int(num_patches**0.5))
        
        # Predict IoU scores
        iou_scores = self.iou_head(prompt_features.mean(dim=2))  # (batch_size, num_prompts, 1)
        iou_scores = iou_scores.squeeze(-1)
        
        return masks, iou_scores

class SAM(nn.Module):
    """
    SAM: Segment Anything Model.
    
    Complete SAM implementation with image encoder, prompt encoder, and mask decoder.
    """
    
    def __init__(self, img_size=1024):
        super().__init__()
        self.img_size = img_size
        
        # Image encoder
        self.image_encoder = SAMImageEncoder(img_size=img_size)
        
        # Prompt encoder
        self.prompt_encoder = SAMPromptEncoder()
        
        # Mask decoder
        self.mask_decoder = SAMMaskDecoder()
    
    def forward(self, images, points=None, boxes=None, text=None):
        """
        Forward pass of SAM.
        
        Args:
            images: Input images
            points: Point prompts
            boxes: Box prompts
            text: Text prompts
        
        Returns:
            masks: Predicted masks
            iou_scores: IoU scores
        """
        # Encode images
        image_embeddings = self.image_encoder(images)
        
        # Encode prompts
        prompt_embeddings = self.prompt_encoder(points, boxes, text)
        
        # Decode masks
        masks, iou_scores = self.mask_decoder(image_embeddings, prompt_embeddings)
        
        return masks, iou_scores

class SAMInference:
    """
    Inference interface for SAM.
    """
    
    def __init__(self, sam_model):
        self.sam_model = sam_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sam_model.to(self.device)
        self.sam_model.eval()
    
    def segment_points(self, image, points):
        """
        Segment image using point prompts.
        
        Args:
            image: Input image
            points: Point coordinates
        
        Returns:
            masks: Segmentation masks
        """
        with torch.no_grad():
            masks, iou_scores = self.sam_model(
                image.unsqueeze(0),
                points=points.unsqueeze(0)
            )
        
        return masks[0], iou_scores[0]
    
    def segment_boxes(self, image, boxes):
        """
        Segment image using box prompts.
        
        Args:
            image: Input image
            boxes: Box coordinates
        
        Returns:
            masks: Segmentation masks
        """
        with torch.no_grad():
            masks, iou_scores = self.sam_model(
                image.unsqueeze(0),
                boxes=boxes.unsqueeze(0)
            )
        
        return masks[0], iou_scores[0]
    
    def segment_text(self, image, text):
        """
        Segment image using text prompts.
        
        Args:
            image: Input image
            text: Text description
        
        Returns:
            masks: Segmentation masks
        """
        with torch.no_grad():
            masks, iou_scores = self.sam_model(
                image.unsqueeze(0),
                text=text.unsqueeze(0)
            )
        
        return masks[0], iou_scores[0]
```

## DALL-E and Image Generation

### Architecture Overview

DALL-E generates high-quality images from text descriptions using a discrete VAE and transformer architecture.

**Key Components:**
- **Discrete VAE**: Compresses images to discrete tokens
- **Text Encoder**: Processes text descriptions
- **Transformer**: Generates image tokens autoregressively
- **Text Conditioning**: Conditions generation on text prompts

### Implementation

```python
class DiscreteVAE(nn.Module):
    """
    Discrete VAE for image tokenization.
    
    Compresses images to discrete tokens for transformer processing.
    """
    
    def __init__(self, num_tokens=8192, token_dim=512, img_size=256):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.img_size = img_size
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, token_dim)
        )
        
        # Codebook
        self.codebook = nn.Parameter(torch.randn(num_tokens, token_dim))
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(token_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.codebook, std=0.02)
    
    def encode(self, x):
        """
        Encode images to discrete tokens.
        
        Args:
            x: Input images of shape (batch_size, 3, img_size, img_size)
        
        Returns:
            indices: Discrete token indices
        """
        features = self.encoder(x)
        
        # Find closest codebook entries
        distances = torch.cdist(features, self.codebook)
        indices = torch.argmin(distances, dim=1)
        
        return indices
    
    def decode(self, indices):
        """
        Decode discrete tokens to images.
        
        Args:
            indices: Discrete token indices
        
        Returns:
            reconstructed: Reconstructed images
        """
        embeddings = self.codebook[indices]
        reconstructed = self.decoder(embeddings)
        return reconstructed
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images
        
        Returns:
            reconstructed: Reconstructed images
            indices: Discrete token indices
        """
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
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
    
    def generate(self, text_tokens, max_length=256, temperature=1.0):
        """
        Generate image tokens autoregressively.
        
        Args:
            text_tokens: Text token indices
            max_length: Maximum generation length
            temperature: Sampling temperature
        
        Returns:
            image_tokens: Generated image tokens
        """
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Start with text tokens
        current_tokens = text_tokens.clone()
        
        for i in range(max_length - text_tokens.shape[1]):
            # Forward pass
            logits = self.forward(current_tokens)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
        
        # Return image tokens (everything after text tokens)
        image_tokens = current_tokens[:, text_tokens.shape[1]:]
        return image_tokens

class DALLETextEncoder(nn.Module):
    """
    Text encoder for DALL-E.
    
    Processes text descriptions for image generation.
    """
    
    def __init__(self, vocab_size=50257, d_model=512, num_layers=12):
        super().__init__()
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(1024, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, text_tokens):
        """
        Forward pass of text encoder.
        
        Args:
            text_tokens: Text token indices
        
        Returns:
            text_features: Text features
        """
        # Embeddings
        token_embeddings = self.token_embedding(text_tokens)
        position_embeddings = self.position_embedding(
            torch.arange(text_tokens.shape[1], device=text_tokens.device)
        ).unsqueeze(0)
        
        # Combine embeddings
        embeddings = token_embeddings + position_embeddings
        
        # Transformer
        text_features = self.transformer(embeddings)
        
        return text_features

class DALLETextToImage(nn.Module):
    """
    Complete DALL-E text-to-image model.
    """
    
    def __init__(self, vocab_size=50257, d_model=512, num_tokens=8192):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_tokens = num_tokens
        
        # Text encoder
        self.text_encoder = DALLETextEncoder(vocab_size, d_model)
        
        # Discrete VAE
        self.vae = DiscreteVAE(num_tokens, d_model)
        
        # Image generation transformer
        self.transformer = DALLETransformer(vocab_size + num_tokens, d_model)
    
    def forward(self, text_tokens, image_tokens=None):
        """
        Forward pass.
        
        Args:
            text_tokens: Text token indices
            image_tokens: Image token indices (for training)
        
        Returns:
            logits: Token predictions
        """
        return self.transformer(text_tokens, image_tokens)
    
    def generate(self, text_tokens, max_length=256, temperature=1.0):
        """
        Generate image from text.
        
        Args:
            text_tokens: Text token indices
            max_length: Maximum generation length
            temperature: Sampling temperature
        
        Returns:
            images: Generated images
        """
        # Generate image tokens
        image_tokens = self.transformer.generate(text_tokens, max_length, temperature)
        
        # Decode to images
        images = self.vae.decode(image_tokens)
        
        return images

class DALLETextTokenizer:
    """
    Simple text tokenizer for DALL-E.
    
    In practice, use a proper tokenizer like BPE.
    """
    
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.pad_token = 0
        self.eos_token = 1
        self.unk_token = 2
    
    def encode(self, text):
        """
        Encode text to token indices.
        
        Args:
            text: Input text
        
        Returns:
            tokens: Token indices
        """
        # Simple character-level tokenization (for demonstration)
        # In practice, use a proper tokenizer
        tokens = [ord(c) % (self.vocab_size - 3) + 3 for c in text]
        tokens.append(self.eos_token)
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens):
        """
        Decode token indices to text.
        
        Args:
            tokens: Token indices
        
        Returns:
            text: Decoded text
        """
        # Simple character-level decoding
        text = ""
        for token in tokens:
            if token == self.eos_token:
                break
            if token >= 3:
                text += chr(token - 3)
        return text
```

## Implementation Examples

### Foundation Model Training

```python
def train_foundation_model(model_type='clip', train_loader=None, num_epochs=100):
    """
    Train foundation model.
    
    Args:
        model_type: Type of foundation model ('clip', 'sam', 'dalle')
        train_loader: Training data loader
        num_epochs: Number of training epochs
    
    Returns:
        model: Trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'clip':
        # Create CLIP model
        image_encoder = create_vision_transformer()
        text_encoder = create_text_transformer()
        model = CLIPModel(image_encoder, text_encoder)
        
        # Create trainer
        trainer = CLIPTrainer(model, train_loader)
        
    elif model_type == 'sam':
        # Create SAM model
        model = SAM()
        
        # Create trainer (simplified)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
    elif model_type == 'dalle':
        # Create DALL-E model
        model = DALLETextToImage()
        
        # Create trainer (simplified)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training loop
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if model_type == 'clip':
                images, text = batch
                images = images.to(device)
                text = text.to(device)
                
                logits, labels = model(images, text)
                loss = model.contrastive_loss(logits, labels)
                
            elif model_type == 'sam':
                images, masks, prompts = batch
                images = images.to(device)
                masks = masks.to(device)
                
                pred_masks, iou_scores = model(images, **prompts)
                loss = F.binary_cross_entropy(pred_masks, masks)
                
            elif model_type == 'dalle':
                text_tokens, image_tokens = batch
                text_tokens = text_tokens.to(device)
                image_tokens = image_tokens.to(device)
                
                logits = model(text_tokens, image_tokens)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), image_tokens.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}')
    
    return model

def create_vision_transformer():
    """Create Vision Transformer for foundation models."""
    from transformers import ViTModel
    
    vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
    vit.output_dim = 768
    return vit

def create_text_transformer():
    """Create text transformer for foundation models."""
    from transformers import AutoModel
    
    bert = AutoModel.from_pretrained('bert-base-uncased')
    bert.output_dim = 768
    return bert
```

### Zero-shot Applications

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

def segment_anything(sam_model, image, points=None, boxes=None, text=None):
    """
    Segment anything using SAM.
    
    Args:
        sam_model: Pre-trained SAM model
        image: Input image
        points: Point prompts
        boxes: Box prompts
        text: Text prompts
    
    Returns:
        masks: Segmentation masks
        iou_scores: IoU scores
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam_model.to(device)
    sam_model.eval()
    
    with torch.no_grad():
        masks, iou_scores = sam_model(
            image.unsqueeze(0),
            points=points.unsqueeze(0) if points is not None else None,
            boxes=boxes.unsqueeze(0) if boxes is not None else None,
            text=text.unsqueeze(0) if text is not None else None
        )
    
    return masks[0], iou_scores[0]

def generate_image(dalle_model, text, max_length=256, temperature=1.0):
    """
    Generate image from text using DALL-E.
    
    Args:
        dalle_model: Pre-trained DALL-E model
        text: Text description
        max_length: Maximum generation length
        temperature: Sampling temperature
    
    Returns:
        images: Generated images
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dalle_model.to(device)
    dalle_model.eval()
    
    # Tokenize text
    tokenizer = DALLETextTokenizer()
    text_tokens = tokenizer.encode(text).unsqueeze(0).to(device)
    
    # Generate image
    with torch.no_grad():
        images = dalle_model.generate(text_tokens, max_length, temperature)
    
    return images
```

## Applications and Use Cases

### Multi-modal Applications

**Image-Text Retrieval:**
```python
def image_text_retrieval(clip_model, query_text, image_database, top_k=5):
    """
    Retrieve images matching text query.
    
    Args:
        clip_model: Pre-trained CLIP model
        query_text: Text query
        image_database: Database of images
        top_k: Number of top results
    
    Returns:
        top_images: Top matching images
        similarities: Similarity scores
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model.to(device)
    clip_model.eval()
    
    # Encode text query
    tokenizer = clip_model.tokenizer
    text_tokens = tokenizer([query_text], return_tensors='pt').to(device)
    
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

### Creative Applications

**Text-to-Image Generation:**
```python
def creative_image_generation(dalle_model, prompts, num_images=4):
    """
    Generate creative images from text prompts.
    
    Args:
        dalle_model: Pre-trained DALL-E model
        prompts: List of text prompts
        num_images: Number of images per prompt
    
    Returns:
        generated_images: List of generated images
    """
    generated_images = []
    
    for prompt in prompts:
        for i in range(num_images):
            # Add some randomness to prompts
            random_prompt = f"{prompt} (variation {i+1})"
            
            # Generate image
            images = generate_image(dalle_model, random_prompt, temperature=0.8)
            generated_images.append(images[0])
    
    return generated_images
```

### Medical and Scientific Applications

**Medical Image Analysis:**
```python
def medical_image_analysis(sam_model, medical_image, anatomical_landmarks):
    """
    Analyze medical images using SAM.
    
    Args:
        sam_model: Pre-trained SAM model
        medical_image: Medical image
        anatomical_landmarks: Points of interest
    
    Returns:
        segmentations: Anatomical segmentations
    """
    # Segment anatomical structures
    masks, iou_scores = segment_anything(
        sam_model, 
        medical_image, 
        points=anatomical_landmarks
    )
    
    return masks, iou_scores
```

## Conclusion

Foundation models represent a significant advancement in computer vision, enabling:

1. **Zero-shot Capabilities**: Models can perform tasks without task-specific training
2. **Multi-modal Understanding**: Integration of vision with language and other modalities
3. **Prompt-based Control**: Natural language control of model behavior
4. **Transfer Learning**: Strong performance across diverse domains

**Key Takeaways:**
- Foundation models leverage massive datasets and computational resources
- CLIP enables zero-shot classification and retrieval
- SAM provides universal segmentation capabilities
- DALL-E demonstrates text-to-image generation

**Future Directions:**
- **Scaling**: Larger models with more data and compute
- **Efficiency**: Reducing computational requirements
- **Robustness**: Improving reliability and safety
- **Accessibility**: Making foundation models more accessible

---

**References:**
- "Learning Transferable Visual Representations" - Radford et al. (CLIP)
- "Segment Anything" - Kirillov et al. (SAM)
- "Zero-Shot Text-to-Image Generation" - Ramesh et al. (DALL-E)

## From Theoretical Understanding to Practical Implementation

We've now explored **foundation models for vision** - large-scale models that leverage massive datasets and computational resources to learn general-purpose visual representations. We've seen how CLIP enables zero-shot classification and retrieval through vision-language alignment, how SAM provides universal segmentation capabilities, how DALL-E demonstrates text-to-image generation, and how these models represent a paradigm shift in computer vision.

However, while understanding foundation models is valuable, **true mastery** comes from hands-on implementation. Consider building a system that can classify images without training on specific categories, or implementing a segmentation model that can segment any object in any image - these require not just theoretical knowledge but practical skills in implementing vision transformers, contrastive learning, and foundation models.

This motivates our exploration of **hands-on coding** - the practical implementation of all the computer vision concepts we've learned. We'll put our theoretical knowledge into practice by implementing Vision Transformers from scratch, building self-supervised learning systems, applying contrastive learning frameworks, and developing practical applications using foundation models for classification, segmentation, and generation.

The transition from foundation models to hands-on coding represents the bridge from understanding to implementation - taking our knowledge of how modern computer vision works and turning it into practical tools for building intelligent vision systems.

In the next section, we'll implement complete computer vision systems, experiment with different architectures and learning paradigms, and develop the practical skills needed for real-world applications in computer vision and AI.

---

**Previous: [Contrastive Learning](03_contrastive_learning.md)** - Learn how to train models by comparing similar and dissimilar data.

**Next: [Hands-on Coding](05_hands-on_coding.md)** - Implement vision transformers and modern computer vision techniques with practical examples. 
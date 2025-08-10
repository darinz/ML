# Computer Vision: Hands-On Learning Guide

[![Vision](https://img.shields.io/badge/Vision-Computer%20Vision-blue.svg)](https://en.wikipedia.org/wiki/Computer_vision)
[![Transformers](https://img.shields.io/badge/Transformers-Vision%20Transformers-green.svg)](https://en.wikipedia.org/wiki/Vision_transformer)
[![Self-Supervised](https://img.shields.io/badge/Self--Supervised-Learning-yellow.svg)](https://en.wikipedia.org/wiki/Self-supervised_learning)
[![Hands-on Learning](https://img.shields.io/badge/Learning-Hands--on%20Experience-green.svg)](https://en.wikipedia.org/wiki/Experiential_learning)

## From Vision Transformers to Foundation Models

We've explored the revolutionary framework of **Computer Vision**, which addresses the fundamental challenge of enabling machines to understand and process visual information through transformer architectures, self-supervised learning, and foundation models. Understanding these concepts is crucial because modern computer vision has transformed how machines perceive the world, enabling breakthroughs in image recognition, generation, and understanding.

However, true understanding comes from **hands-on implementation**. This practical guide will help you translate the theoretical concepts into working code, experiment with different vision architectures, and develop the intuition needed to build intelligent systems that can see, understand, and generate visual content.

## Learning Objectives

By completing this hands-on learning guide, you will:

1. **Master Vision Transformers** through interactive implementations of patch embedding and attention mechanisms
2. **Implement self-supervised learning** techniques including rotation, jigsaw, and inpainting
3. **Apply contrastive learning** with SimCLR, MoCo, and BYOL frameworks
4. **Build foundation models** including CLIP and DINO for vision-language understanding
5. **Develop practical applications** for classification, segmentation, and generation
6. **Understand modern vision techniques** including zero-shot learning and few-shot adaptation

## Quick Start

### Prerequisites
- Basic Python knowledge (variables, functions, arrays)
- Familiarity with PyTorch (tensors, neural networks, autograd)
- Understanding of computer vision concepts (images, convolutions)
- Completion of deep learning and neural networks modules (recommended)

### Estimated Time
- **Setup**: 30 minutes
- **Lesson 1**: 4-5 hours
- **Lesson 2**: 4-5 hours
- **Lesson 3**: 4-5 hours
- **Lesson 4**: 3-4 hours
- **Total**: 16-19 hours

---

## Environment Setup

### Option 1: Using Conda (Recommended)

#### Step 1: Install Miniconda
```bash
# Download Miniconda for your OS
# Windows: https://docs.conda.io/en/latest/miniconda.html
# macOS: https://docs.conda.io/en/latest/miniconda.html
# Linux: https://docs.conda.io/en/latest/miniconda.html

# Verify installation
conda --version
```

#### Step 2: Create Environment
```bash
# Navigate to the vision directory
cd 13_vision

# Create a new conda environment
conda env create -f environment.yaml

# Activate the environment
conda activate vision-lesson

# Verify installation
python -c "import torch, torchvision, numpy; print('All packages installed successfully!')"
```

### Option 2: Using pip

#### Step 1: Create Virtual Environment
```bash
# Navigate to the vision directory
cd 13_vision

# Create virtual environment
python -m venv vision-env

# Activate environment
# On Windows:
vision-env\Scripts\activate
# On macOS/Linux:
source vision-env/bin/activate

# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import torch, torchvision, numpy; print('All packages installed successfully!')"
```

### Option 3: Using Jupyter Notebooks

#### Step 1: Install Jupyter
```bash
# After setting up environment above
pip install jupyter notebook

# Launch Jupyter
jupyter notebook
```

#### Step 2: Create New Notebook
```python
# In a new notebook cell, import required packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import math
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set plotting style
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

---

## Lesson Structure

### Lesson 1: Vision Transformers (4-5 hours)
**Files**: `vision_transformer.py`, `patch_embedding.py`, `attention.py`, `position_encoding.py`

#### Learning Goals
- Understand Vision Transformer architecture
- Master patch embedding and linear projection
- Implement multi-head self-attention for vision
- Apply positional encoding techniques
- Build practical applications for image classification

#### Hands-On Activities

**Activity 1.1: Understanding Vision Transformer Architecture**
```python
# Explore the core Vision Transformer architecture
from vision_transformer import VisionTransformer, create_vit_model

# Create a Vision Transformer model
vit_model = create_vit_model(
    variant='base',
    num_classes=1000,
    img_size=224
)

print(f"ViT model created with {sum(p.numel() for p in vit_model.parameters())} parameters")

# Create sample input
batch_size = 4
input_images = torch.randn(batch_size, 3, 224, 224)

# Forward pass
output = vit_model(input_images)

print(f"Input shape: {input_images.shape}")
print(f"Output shape: {output.shape}")

# Key insight: Vision Transformers treat images as sequences of patches
```

**Activity 1.2: Patch Embedding Implementation**
```python
# Implement patch embedding
from patch_embedding import PatchEmbedding

# Create patch embedding
patch_embed = PatchEmbedding(
    img_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=768
)

# Apply patch embedding
patches = patch_embed(input_images)

print(f"Original image shape: {input_images.shape}")
print(f"Patches shape: {patches.shape}")
print(f"Number of patches: {patches.shape[1]}")

# Visualize patches
def visualize_patches(image, patch_size=16):
    """Visualize how an image is divided into patches."""
    h, w = image.shape[1], image.shape[2]
    patches_h = h // patch_size
    patches_w = w // patch_size
    
    fig, axes = plt.subplots(patches_h, patches_w, figsize=(10, 10))
    for i in range(patches_h):
        for j in range(patches_w):
            patch = image[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            axes[i, j].imshow(patch.permute(1, 2, 0))
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.show()

# Visualize patches for first image
visualize_patches(input_images[0])

# Key insight: Patch embedding converts spatial structure to sequence representation
```

**Activity 1.3: Multi-Head Self-Attention for Vision**
```python
# Implement attention mechanism for vision
from attention import MultiHeadAttention

# Create multi-head attention
d_model = 768
num_heads = 12
attention = MultiHeadAttention(
    d_model=d_model,
    num_heads=num_heads,
    dropout=0.1
)

# Apply attention to patches
attention_output, attention_weights = attention(patches)

print(f"Patches shape: {patches.shape}")
print(f"Attention output shape: {attention_output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")

# Visualize attention weights
plt.figure(figsize=(8, 8))
plt.imshow(attention_weights[0, 0].detach().numpy(), cmap='viridis')
plt.title('Attention Weights (First Head)')
plt.colorbar()
plt.show()

# Key insight: Self-attention enables global relationships between patches
```

**Activity 1.4: Positional Encoding**
```python
# Implement positional encoding for vision
from position_encoding import PositionalEncoding

# Create positional encoding
pos_encoding = PositionalEncoding(
    d_model=d_model,
    max_len=196  # 14x14 patches
)

# Add positional encoding to patches
patches_with_pos = pos_encoding(patches)

print(f"Patches without position: {patches.shape}")
print(f"Patches with position: {patches_with_pos.shape}")

# Visualize positional encoding
plt.figure(figsize=(10, 6))
plt.imshow(pos_encoding.pe[0].T, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Positional Encoding for Vision')
plt.xlabel('Position')
plt.ylabel('Dimension')
plt.show()

# Key insight: Positional encoding provides spatial location information
```

**Activity 1.5: Complete Vision Transformer**
```python
# Build and test complete Vision Transformer
from vision_transformer import VisionTransformer

# Create complete ViT
vit = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=10,
    embed_dim=768,
    depth=12,
    num_heads=12
)

# Forward pass
logits = vit(input_images)
predictions = torch.softmax(logits, dim=-1)

print(f"ViT output shape: {logits.shape}")
print(f"Predictions shape: {predictions.shape}")
print(f"Predicted class: {torch.argmax(predictions, dim=1)}")

# Key insight: Vision Transformers can achieve state-of-the-art performance on image classification
```

#### Experimentation Tasks
1. **Experiment with different patch sizes**: Study how patch size affects performance
2. **Test various attention heads**: Compare different numbers of attention heads
3. **Analyze attention patterns**: Visualize attention weights for different images
4. **Compare with CNNs**: Observe the differences between ViT and CNN architectures

#### Check Your Understanding
- [ ] Can you explain how Vision Transformers work?
- [ ] Do you understand patch embedding and linear projection?
- [ ] Can you implement multi-head attention for vision?
- [ ] Do you see the benefits of positional encoding?

---

### Lesson 2: Self-Supervised Learning (4-5 hours)
**Files**: `rotation.py`, `jigsaw.py`, `inpainting.py`, `colorization.py`

#### Learning Goals
- Understand self-supervised learning principles
- Master pretext task implementations
- Implement rotation prediction
- Apply jigsaw puzzle solving
- Build inpainting and colorization models

#### Hands-On Activities

**Activity 2.1: Rotation Prediction**
```python
# Implement rotation prediction as pretext task
from rotation import RotationPrediction, RotationDataset

# Create rotation prediction model
rotation_model = RotationPrediction(
    encoder='resnet18',
    num_rotations=4
)

# Create rotation dataset
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply rotation augmentation
rotated_images, rotation_labels = RotationDataset.apply_rotations(input_images)

print(f"Original images shape: {input_images.shape}")
print(f"Rotated images shape: {rotated_images.shape}")
print(f"Rotation labels: {rotation_labels}")

# Forward pass
rotation_logits = rotation_model(rotated_images)
rotation_predictions = torch.softmax(rotation_logits, dim=-1)

print(f"Rotation predictions shape: {rotation_predictions.shape}")

# Key insight: Rotation prediction teaches models to understand spatial orientation
```

**Activity 2.2: Jigsaw Puzzle Solving**
```python
# Implement jigsaw puzzle solving
from jigsaw import JigsawPuzzle, JigsawDataset

# Create jigsaw puzzle model
jigsaw_model = JigsawPuzzle(
    encoder='resnet18',
    num_permutations=100
)

# Create jigsaw puzzle
puzzle_images, permutation_labels = JigsawDataset.create_puzzle(input_images, grid_size=3)

print(f"Puzzle images shape: {puzzle_images.shape}")
print(f"Permutation labels: {permutation_labels}")

# Forward pass
jigsaw_logits = jigsaw_model(puzzle_images)
jigsaw_predictions = torch.softmax(jigsaw_logits, dim=-1)

print(f"Jigsaw predictions shape: {jigsaw_predictions.shape}")

# Key insight: Jigsaw puzzles teach models spatial relationships and context
```

**Activity 2.3: Image Inpainting**
```python
# Implement image inpainting
from inpainting import InpaintingModel, InpaintingDataset

# Create inpainting model
inpainting_model = InpaintingModel(
    encoder='resnet18',
    decoder_channels=[512, 256, 128, 64, 3]
)

# Create masked images
masked_images, masks, original_images = InpaintingDataset.create_masks(input_images, mask_ratio=0.3)

print(f"Masked images shape: {masked_images.shape}")
print(f"Masks shape: {masks.shape}")

# Forward pass
reconstructed_images = inpainting_model(masked_images)

print(f"Reconstructed images shape: {reconstructed_images.shape}")

# Visualize inpainting results
def visualize_inpainting(original, masked, reconstructed, mask):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(original[0].permute(1, 2, 0))
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(masked[0].permute(1, 2, 0))
    axes[0, 1].set_title('Masked')
    axes[1, 0].imshow(reconstructed[0].permute(1, 2, 0))
    axes[1, 0].set_title('Reconstructed')
    axes[1, 1].imshow(mask[0], cmap='gray')
    axes[1, 1].set_title('Mask')
    plt.tight_layout()
    plt.show()

visualize_inpainting(original_images, masked_images, reconstructed_images, masks)

# Key insight: Inpainting teaches models to understand image structure and context
```

**Activity 2.4: Image Colorization**
```python
# Implement image colorization
from colorization import ColorizationModel, ColorizationDataset

# Create colorization model
colorization_model = ColorizationModel(
    encoder='resnet18',
    decoder_channels=[512, 256, 128, 64, 2]  # 2 channels for a*b color space
)

# Convert to grayscale and create colorization task
grayscale_images, color_targets = ColorizationDataset.convert_to_grayscale(input_images)

print(f"Grayscale images shape: {grayscale_images.shape}")
print(f"Color targets shape: {color_targets.shape}")

# Forward pass
predicted_colors = colorization_model(grayscale_images)

print(f"Predicted colors shape: {predicted_colors.shape}")

# Convert back to RGB for visualization
def lab_to_rgb(lab_images):
    """Convert LAB to RGB for visualization."""
    # Simplified conversion - in practice, use proper color space conversion
    return lab_images

reconstructed_rgb = lab_to_rgb(predicted_colors)

# Key insight: Colorization teaches models to understand color relationships and context
```

#### Experimentation Tasks
1. **Experiment with different pretext tasks**: Compare rotation, jigsaw, and inpainting
2. **Test various augmentation strategies**: Study how augmentation affects learning
3. **Analyze learned representations**: Visualize features learned by different pretext tasks
4. **Compare self-supervised vs supervised**: Observe the benefits of self-supervised learning

#### Check Your Understanding
- [ ] Can you explain self-supervised learning principles?
- [ ] Do you understand how pretext tasks work?
- [ ] Can you implement rotation prediction?
- [ ] Do you see the benefits of self-supervised learning?

---

### Lesson 3: Contrastive Learning (4-5 hours)
**Files**: `simclr.py`, `moco.py`, `byol.py`, `dino.py`

#### Learning Goals
- Understand contrastive learning principles
- Master SimCLR implementation
- Implement MoCo framework
- Apply BYOL and DINO techniques
- Build practical applications for representation learning

#### Hands-On Activities

**Activity 3.1: SimCLR Implementation**
```python
# Implement SimCLR contrastive learning
from simclr import SimCLRModel, SimCLRLoss, SimCLRTransform

# Create SimCLR model
simclr_model = SimCLRModel(
    encoder='resnet18',
    projection_dim=128
)

# Create SimCLR transforms
simclr_transform = SimCLRTransform(image_size=224, s=1.0)

# Apply augmentations
augmented_pairs = []
for image in input_images:
    aug1, aug2 = simclr_transform(image)
    augmented_pairs.append((aug1, aug2))

augmented_pairs = torch.stack([torch.stack(pair) for pair in augmented_pairs])

print(f"Augmented pairs shape: {augmented_pairs.shape}")

# Forward pass
projections = simclr_model(augmented_pairs.view(-1, 3, 224, 224))
projections = projections.view(batch_size, 2, -1)

print(f"Projections shape: {projections.shape}")

# Key insight: SimCLR learns representations by maximizing agreement between augmented views
```

**Activity 3.2: SimCLR Loss Computation**
```python
# Implement SimCLR loss
from simclr import SimCLRLoss

# Create SimCLR loss
simclr_loss = SimCLRLoss(temperature=0.5)

# Compute loss
loss = simclr_loss(projections[:, 0], projections[:, 1])

print(f"SimCLR loss: {loss.item():.4f}")

# Key insight: Contrastive loss encourages similar views to have similar representations
```

**Activity 3.3: MoCo Implementation**
```python
# Implement MoCo (Momentum Contrast)
from moco import MoCoModel, MoCoLoss

# Create MoCo model
moco_model = MoCoModel(
    encoder='resnet18',
    projection_dim=128,
    queue_size=65536,
    momentum=0.999
)

# Create query and key encodings
query_encodings = moco_model.encode_query(augmented_pairs[:, 0])
key_encodings = moco_model.encode_key(augmented_pairs[:, 1])

print(f"Query encodings shape: {query_encodings.shape}")
print(f"Key encodings shape: {key_encodings.shape}")

# Compute MoCo loss
moco_loss = MoCoLoss(temperature=0.07)
loss = moco_loss(query_encodings, key_encodings)

print(f"MoCo loss: {loss.item():.4f}")

# Key insight: MoCo uses a momentum encoder and queue to improve contrastive learning
```

**Activity 3.4: BYOL Implementation**
```python
# Implement BYOL (Bootstrap Your Own Latent)
from byol import BYOLModel, BYOLLoss

# Create BYOL model
byol_model = BYOLModel(
    encoder='resnet18',
    projection_dim=256,
    prediction_dim=256
)

# Forward pass through online and target networks
online_projection, online_prediction = byol_model.online_network(augmented_pairs[:, 0])
target_projection = byol_model.target_network(augmented_pairs[:, 1])

print(f"Online projection shape: {online_projection.shape}")
print(f"Online prediction shape: {online_prediction.shape}")
print(f"Target projection shape: {target_projection.shape}")

# Compute BYOL loss
byol_loss = BYOLLoss()
loss = byol_loss(online_prediction, target_projection)

print(f"BYOL loss: {loss.item():.4f}")

# Key insight: BYOL learns representations without negative pairs
```

**Activity 3.5: DINO Implementation**
```python
# Implement DINO (Self-Distillation with No Labels)
from dino import DINOModel, DINOLoss

# Create DINO model
dino_model = DINOModel(
    encoder='resnet18',
    projection_dim=256,
    num_prototypes=65536
)

# Forward pass
student_output = dino_model.student_network(augmented_pairs[:, 0])
teacher_output = dino_model.teacher_network(augmented_pairs[:, 1])

print(f"Student output shape: {student_output.shape}")
print(f"Teacher output shape: {teacher_output.shape}")

# Compute DINO loss
dino_loss = DINOLoss(temperature=0.1)
loss = dino_loss(student_output, teacher_output)

print(f"DINO loss: {loss.item():.4f}")

# Key insight: DINO uses self-distillation to learn representations without labels
```

#### Experimentation Tasks
1. **Experiment with different contrastive methods**: Compare SimCLR, MoCo, BYOL, and DINO
2. **Test various augmentation strategies**: Study how augmentation affects contrastive learning
3. **Analyze learned representations**: Visualize features learned by different methods
4. **Compare contrastive vs supervised**: Observe the benefits of contrastive learning

#### Check Your Understanding
- [ ] Can you explain contrastive learning principles?
- [ ] Do you understand how SimCLR works?
- [ ] Can you implement MoCo framework?
- [ ] Do you see the benefits of contrastive learning?

---

### Lesson 4: Foundation Models and Applications (3-4 hours)
**Files**: `clip_implementation.py`, `zero_shot_classification.py`, `dalle_generation.py`, `sam_segmentation.py`

#### Learning Goals
- Understand foundation models for vision
- Master CLIP implementation
- Implement zero-shot classification
- Apply vision-language models
- Build practical applications for modern vision tasks

#### Hands-On Activities

**Activity 4.1: CLIP Implementation**
```python
# Implement CLIP (Contrastive Language-Image Pre-training)
from clip_implementation import CLIPModel, CLIPLoss, SimpleTokenizer

# Create CLIP model
clip_model = CLIPModel(
    image_encoder='resnet50',
    vocab_size=49408,
    max_length=77,
    projection_dim=512
)

# Create tokenizer
tokenizer = SimpleTokenizer(vocab_size=49408)

# Create sample text
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
text_tokens = torch.stack([tokenizer.tokenize(text) for text in texts])

print(f"Text tokens shape: {text_tokens.shape}")

# Forward pass
image_features = clip_model.encode_image(input_images)
text_features = clip_model.encode_text(text_tokens)

print(f"Image features shape: {image_features.shape}")
print(f"Text features shape: {text_features.shape}")

# Key insight: CLIP learns aligned representations of images and text
```

**Activity 4.2: Zero-Shot Classification**
```python
# Implement zero-shot classification
from zero_shot_classification import ZeroShotClassifier

# Create zero-shot classifier
zero_shot_classifier = ZeroShotClassifier(
    model=clip_model,
    tokenizer=tokenizer
)

# Define class names
class_names = ["cat", "dog", "bird", "car", "tree"]

# Perform zero-shot classification
predictions = zero_shot_classifier.classify(
    images=input_images,
    class_names=class_names
)

print(f"Zero-shot predictions shape: {predictions.shape}")
print(f"Predicted classes: {torch.argmax(predictions, dim=1)}")

# Key insight: Zero-shot classification enables classification without training data
```

**Activity 4.3: DALL-E Style Generation**
```python
# Implement DALL-E style image generation
from dalle_generation import DALLEModel, DALLEGenerator

# Create DALL-E model
dalle_model = DALLEModel(
    text_encoder='transformer',
    image_decoder='transformer',
    vocab_size=49408,
    image_size=256
)

# Create text prompts
prompts = ["a cat sitting on a chair", "a dog running in a park"]

# Generate images
generated_images = dalle_model.generate(
    prompts=prompts,
    num_images=1,
    temperature=0.8
)

print(f"Generated images shape: {generated_images.shape}")

# Visualize generated images
def visualize_generated_images(images, prompts):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for i, (image, prompt) in enumerate(zip(images, prompts)):
        axes[i].imshow(image.permute(1, 2, 0))
        axes[i].set_title(prompt)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

visualize_generated_images(generated_images, prompts)

# Key insight: DALL-E generates images from text descriptions
```

**Activity 4.4: SAM Segmentation**
```python
# Implement SAM (Segment Anything Model)
from sam_segmentation import SAMModel, SAMPredictor

# Create SAM model
sam_model = SAMModel(
    encoder='vit_h',
    decoder='mask_decoder',
    prompt_encoder='point_prompt_encoder'
)

# Create predictor
sam_predictor = SAMPredictor(sam_model)

# Set image
sam_predictor.set_image(input_images[0])

# Create point prompts
point_coords = torch.tensor([[100, 100], [200, 200]])  # Example points
point_labels = torch.tensor([1, 1])  # Foreground points

# Predict masks
masks, scores, logits = sam_predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels
)

print(f"Masks shape: {masks.shape}")
print(f"Scores shape: {scores.shape}")

# Visualize segmentation
def visualize_segmentation(image, masks, scores):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.permute(1, 2, 0))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(masks[0], cmap='gray')
    axes[1].set_title(f'Segmentation (Score: {scores[0]:.3f})')
    axes[1].axis('off')
    
    axes[2].imshow(image.permute(1, 2, 0))
    axes[2].imshow(masks[0], alpha=0.5, cmap='jet')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_segmentation(input_images[0], masks, scores)

# Key insight: SAM enables interactive segmentation with point prompts
```

#### Experimentation Tasks
1. **Experiment with different foundation models**: Compare CLIP, DALL-E, and SAM
2. **Test various zero-shot tasks**: Study how well models generalize
3. **Analyze generation quality**: Compare different generation parameters
4. **Compare foundation vs task-specific**: Observe the benefits of foundation models

#### Check Your Understanding
- [ ] Can you explain foundation models for vision?
- [ ] Do you understand how CLIP works?
- [ ] Can you implement zero-shot classification?
- [ ] Do you see the applications of vision-language models?

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue 1: Memory Issues with Large Models
```python
# Problem: Out of memory when training large vision models
# Solution: Use gradient checkpointing and mixed precision
def memory_efficient_vision_training(model, train_loader, optimizer):
    """Memory efficient vision model training."""
    from torch.cuda.amp import GradScaler, autocast
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Mixed precision training
    scaler = GradScaler()
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            loss = model(batch)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    return model
```

#### Issue 2: Vision Transformer Training Issues
```python
# Problem: Vision Transformer training doesn't converge
# Solution: Use proper learning rate scheduling and data augmentation
def robust_vit_training(model, train_loader, num_epochs=100):
    """Robust Vision Transformer training."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        warmup_steps = 1000
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / 100000)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            loss = model(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}: Loss = {loss.item():.4f}")
    
    return model
```

#### Issue 3: Contrastive Learning Convergence
```python
# Problem: Contrastive learning doesn't converge
# Solution: Use proper temperature and batch size
def stable_contrastive_training(model, train_loader, temperature=0.07, batch_size=256):
    """Stable contrastive learning training."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(100):
        for batch in train_loader:
            # Ensure proper batch size for contrastive learning
            if batch[0].size(0) < batch_size:
                continue
                
            optimizer.zero_grad()
            
            # Forward pass
            features = model(batch[0])
            
            # Compute contrastive loss
            loss = contrastive_loss(features, temperature=temperature)
            
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                print(f"Epoch {epoch}, Loss = {loss.item():.4f}")
    
    return model
```

#### Issue 4: Foundation Model Performance
```python
# Problem: Foundation models perform poorly on specific tasks
# Solution: Use proper prompting and fine-tuning
def improve_foundation_model_performance(model, task_data, task_type='classification'):
    """Improve foundation model performance on specific tasks."""
    
    if task_type == 'classification':
        # Use better prompts
        prompts = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a photograph of a {}"
        ]
        
        # Ensemble predictions
        all_predictions = []
        for prompt in prompts:
            predictions = model.classify_with_prompt(task_data, prompt)
            all_predictions.append(predictions)
        
        # Average predictions
        final_predictions = torch.stack(all_predictions).mean(dim=0)
        
    elif task_type == 'generation':
        # Use better generation parameters
        generated_images = model.generate(
            prompts=task_data,
            temperature=0.8,
            top_p=0.9,
            num_images=4
        )
        
        # Select best image based on quality
        final_images = select_best_images(generated_images)
    
    return final_predictions if task_type == 'classification' else final_images
```

---

## Assessment and Progress Tracking

### Self-Assessment Checklist

#### Vision Transformers Level
- [ ] I can explain Vision Transformer architecture
- [ ] I understand patch embedding and linear projection
- [ ] I can implement multi-head attention for vision
- [ ] I can apply positional encoding techniques

#### Self-Supervised Learning Level
- [ ] I can explain self-supervised learning principles
- [ ] I understand how pretext tasks work
- [ ] I can implement rotation prediction
- [ ] I can apply inpainting and colorization

#### Contrastive Learning Level
- [ ] I can explain contrastive learning principles
- [ ] I understand how SimCLR works
- [ ] I can implement MoCo framework
- [ ] I can apply BYOL and DINO techniques

#### Foundation Models Level
- [ ] I can explain foundation models for vision
- [ ] I understand how CLIP works
- [ ] I can implement zero-shot classification
- [ ] I can apply vision-language models

### Progress Tracking

#### Week 1: Vision Transformers and Self-Supervised Learning
- **Goal**: Complete Lessons 1 and 2
- **Deliverable**: Working Vision Transformer and self-supervised learning implementations
- **Assessment**: Can you implement Vision Transformers and self-supervised learning?

#### Week 2: Contrastive Learning and Foundation Models
- **Goal**: Complete Lessons 3 and 4
- **Deliverable**: Contrastive learning and foundation model implementations
- **Assessment**: Can you implement contrastive learning and foundation models?

---

## Extension Projects

### Project 1: Advanced Vision Architectures
**Goal**: Implement cutting-edge vision architectures

**Tasks**:
1. Implement Swin Transformer with hierarchical structure
2. Add ConvNeXt for modern CNN design
3. Create MaxViT for multi-axis attention
4. Build EfficientViT for mobile deployment
5. Add vision-language models (BLIP, CoCa)

**Skills Developed**:
- Advanced vision architectures
- Hierarchical transformers
- Mobile vision models
- Multi-modal learning

### Project 2: Vision Applications
**Goal**: Build real-world vision applications

**Tasks**:
1. Implement object detection with Vision Transformers
2. Add instance segmentation with SAM
3. Create video understanding systems
4. Build medical image analysis
5. Add autonomous driving perception

**Skills Developed**:
- Real-world applications
- Computer vision systems
- Performance optimization
- Domain-specific solutions

### Project 3: Vision Research
**Goal**: Conduct original vision research

**Tasks**:
1. Implement novel attention mechanisms
2. Add efficient training techniques
3. Create evaluation frameworks
4. Build interpretability tools
5. Write research papers

**Skills Developed**:
- Research methodology
- Novel algorithm development
- Experimental design
- Academic writing

---

## Additional Resources

### Books
- **"Vision Transformers"** by Dosovitskiy et al.
- **"Self-Supervised Learning in Computer Vision"** by Jure Zbontar
- **"Contrastive Learning"** by Ting Chen

### Online Courses
- **Stanford CS231N**: Convolutional Neural Networks for Visual Recognition
- **Berkeley CS294**: Deep Reinforcement Learning
- **MIT 6.S191**: Introduction to Deep Learning

### Practice Environments
- **ImageNet**: Large-scale image classification dataset
- **COCO**: Object detection and segmentation dataset
- **Open Images**: Large-scale object detection dataset
- **Papers With Code**: Latest research implementations

### Advanced Topics
- **Multi-Modal Learning**: Combining vision with other modalities
- **Efficient Vision**: Techniques for mobile and edge deployment
- **3D Vision**: Understanding 3D structure from images
- **Video Understanding**: Temporal modeling in vision

---

## Conclusion: The Future of Computer Vision

Congratulations on completing this comprehensive journey through Computer Vision! We've explored the fundamental techniques for building intelligent vision systems.

### The Complete Picture

**1. Vision Transformers** - We started with the revolutionary architecture that transformed computer vision.

**2. Self-Supervised Learning** - We built systems that learn representations without labels.

**3. Contrastive Learning** - We implemented frameworks for learning visual representations.

**4. Foundation Models** - We explored models that can handle multiple vision tasks.

### Key Insights

- **Transformers are Powerful**: Vision Transformers have revolutionized computer vision
- **Self-Supervision is Key**: Learning without labels enables better representations
- **Contrastive Learning Works**: Comparing different views leads to better features
- **Foundation Models Scale**: Large models can handle multiple tasks effectively
- **Applications are Endless**: Computer vision has applications in every domain

### Looking Forward

This computer vision foundation prepares you for advanced topics:
- **Multi-Modal AI**: Combining vision with language, audio, and other modalities
- **3D Vision**: Understanding 3D structure and geometry
- **Video Understanding**: Temporal modeling and video analysis
- **Robotics**: Vision for autonomous systems
- **Medical Imaging**: Specialized vision for healthcare

The principles we've learned here - attention mechanisms, self-supervised learning, and foundation models - will serve you well throughout your AI journey.

### Next Steps

1. **Apply vision techniques** to your own projects
2. **Explore advanced architectures** and research frontiers
3. **Build real-world applications** using computer vision
4. **Contribute to open source** vision libraries
5. **Continue learning** about computer vision

Remember: Computer Vision is not just algorithms - it's a fundamental approach to enabling machines to see and understand the world. Keep exploring, building, and applying these concepts to create more intelligent and capable vision systems!

---

## Environment Files

### requirements.txt
```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.20.0
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
seaborn>=0.11.0
jupyter>=1.0.0
notebook>=6.4.0
ipykernel>=6.0.0
nb_conda_kernels>=2.3.0
opencv-python>=4.5.0
pillow>=8.0.0
einops>=0.6.0
```

### environment.yaml
```yaml
name: vision-lesson
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - numpy>=1.21.0
  - matplotlib>=3.5.0
  - scipy>=1.7.0
  - scikit-learn>=1.0.0
  - pandas>=1.3.0
  - seaborn>=0.11.0
  - jupyter>=1.0.0
  - notebook>=6.4.0
  - opencv>=4.5.0
  - pillow>=8.0.0
  - pip
  - pip:
    - transformers>=4.20.0
    - einops>=0.6.0
    - ipykernel
    - nb_conda_kernels
```

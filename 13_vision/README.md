# Computer Vision Advances

[![Vision Transformers](https://img.shields.io/badge/ViT-Vision%20Transformers-blue.svg)](https://en.wikipedia.org/wiki/Vision_transformer)
[![Self-Supervised](https://img.shields.io/badge/Self--Supervised-Learning-green.svg)](https://en.wikipedia.org/wiki/Self-supervised_learning)
[![Contrastive](https://img.shields.io/badge/Contrastive-Learning-purple.svg)](https://en.wikipedia.org/wiki/Contrastive_learning)

Comprehensive materials covering modern computer vision advances including Vision Transformers, self-supervised learning, contrastive learning, and foundation models.

## Overview

Modern computer vision has been revolutionized by transformer architectures and self-supervised learning, enabling breakthroughs in image understanding, generation, and segmentation.

## Materials

### Theory
- **[01_vision_transformers.md](01_vision_transformers.md)** - Vision Transformer architecture and implementation
- **[02_self_supervised_learning.md](02_self_supervised_learning.md)** - Self-supervised learning techniques for vision
- **[03_contrastive_learning.md](03_contrastive_learning.md)** - Contrastive learning frameworks and methods
- **[04_foundation_models.md](04_foundation_models.md)** - Foundation models for computer vision
- **[05_hands-on_coding.md](05_hands-on_coding.md)** - Practical implementation guide

### Vision Transformer Components
- **[vision_transformer.py](vision_transformer.py)** - Complete ViT implementation
- **[patch_embedding.py](patch_embedding.py)** - Image patching and embedding
- **[attention.py](attention.py)** - Multi-head self-attention for vision
- **[position_encoding.py](position_encoding.py)** - Position embedding methods

### Self-Supervised Learning
- **[inpainting.py](inpainting.py)** - Image inpainting implementation
- **[jigsaw.py](jigsaw.py)** - Jigsaw puzzle solving
- **[rotation.py](rotation.py)** - Rotation prediction
- **[colorization.py](colorization.py)** - Image colorization

### Contrastive Learning
- **[simclr.py](simclr.py)** - SimCLR implementation
- **[moco.py](moco.py)** - MoCo framework
- **[byol.py](byol.py)** - BYOL training
- **[dino.py](dino.py)** - DINO self-distillation

### Foundation Models
- **[clip_implementation.py](clip_implementation.py)** - CLIP model implementation
- **[sam_segmentation.py](sam_segmentation.py)** - SAM for segmentation
- **[dalle_generation.py](dalle_generation.py)** - DALL-E style generation
- **[zero_shot_classification.py](zero_shot_classification.py)** - Zero-shot learning

### Supporting Files
- **requirements.txt** - Python dependencies
- **environment.yaml** - Conda environment setup

## Key Concepts

### Vision Transformers
**Patch Embedding**: $z_0 = [x_{\text{class}}; x_p^1 E; \ldots; x_p^N E] + E_{\text{pos}}$

**Transformer Block**: $z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$

**Advantages**: Global attention, scalability, transfer learning

### Self-Supervised Learning
**Pretext Tasks**: Inpainting, jigsaw, rotation, colorization

**BYOL**: $q_\theta(z_t) = f_\theta(g_\theta(x_t))$

**DINO**: Self-distillation without labels

### Contrastive Learning
**SimCLR Loss**: $\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$

**MoCo**: Momentum encoder with queue of negatives

### Foundation Models
**CLIP**: Contrastive language-image pre-training
**SAM**: Segment anything model
**DALL-E**: Text-to-image generation

## Applications

- **Image Classification**: State-of-the-art classification with ViT
- **Object Detection**: DETR, Mask2Former
- **Image Segmentation**: SAM, universal segmentation
- **Image Generation**: DALL-E, Stable Diffusion
- **Medical Imaging**: Diagnosis, radiology, pathology

## Getting Started

1. Read `01_vision_transformers.md` for ViT fundamentals
2. Study `02_self_supervised_learning.md` for pretext tasks
3. Learn `03_contrastive_learning.md` for contrastive methods
4. Explore `04_foundation_models.md` for foundation models
5. Follow `05_hands-on_coding.md` for implementation

## Prerequisites

- Deep learning and neural networks
- Computer vision basics
- Transformer architecture
- Python, PyTorch, OpenCV

## Installation

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yaml
```

## Quick Start

```python
# Vision Transformer
from vision_transformer import VisionTransformer
model = VisionTransformer(image_size=224, patch_size=16, num_classes=1000)

# SimCLR
from simclr import SimCLR
model = SimCLR(encoder='resnet50', projection_dim=128)

# CLIP
from clip_implementation import CLIP
model = CLIP(image_encoder='vit_base', text_encoder='transformer')
```

## Reference Papers

- **ViT**: An Image is Worth 16x16 Words
- **SimCLR**: Simple Framework for Contrastive Learning
- **MoCo**: Momentum Contrast for Unsupervised Learning
- **CLIP**: Learning Transferable Visual Representations
- **SAM**: Segment Anything
- **DALL-E**: Zero-Shot Text-to-Image Generation 
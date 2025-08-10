# Vision Transformers (ViT)

## Overview

Vision Transformers (ViT) represent a paradigm shift in computer vision, adapting the transformer architecture from natural language processing to visual data. By treating images as sequences of patches, ViT has demonstrated that pure attention-based architectures can achieve state-of-the-art performance on image classification and other vision tasks, often surpassing traditional convolutional neural networks (CNNs).

### Key Innovations

**Fundamental Breakthrough:**
- **Patch-based Processing**: Images divided into fixed-size patches (e.g., 16×16 pixels)
- **Global Attention**: Captures relationships between any two patches in the image
- **Scalable Architecture**: Performance improves with model size and data
- **Transfer Learning**: Strong performance on downstream tasks

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Mathematical Foundations](#mathematical-foundations)
- [Patch Embedding](#patch-embedding)
- [Transformer Encoder](#transformer-encoder)
- [Classification Head](#classification-head)
- [Training Strategies](#training-strategies)
- [Variants and Improvements](#variants-and-improvements)
- [Implementation Examples](#implementation-examples)
- [Performance Analysis](#performance-analysis)
- [Applications](#applications)

## Architecture Overview

### Core Components

The Vision Transformer architecture consists of several key components:

1. **Image Patching**: Divide input image into fixed-size patches
2. **Linear Embedding**: Project patches to embedding dimension
3. **Position Embedding**: Add learnable position embeddings
4. **Transformer Encoder**: Stack of self-attention and feed-forward layers
5. **Classification Head**: Global average pooling + linear classifier

### Architecture Diagram

```
Input Image (H×W×C)
    ↓
Patch Division (N patches of P×P×C)
    ↓
Linear Embedding (N×D)
    ↓
+ Position Embedding (N×D)
    ↓
Transformer Encoder (L layers)
    ↓
[CLS] Token Extraction
    ↓
Classification Head
    ↓
Output Predictions
```

## Mathematical Foundations

### Patch Embedding

The first step in ViT is to divide the input image into patches and embed them into a high-dimensional space.

**Patch Division:**
Given an input image $`x \in \mathbb{R}^{H \times W \times C}`$, we divide it into $`N = \frac{HW}{P^2}`$ patches of size $`P \times P \times C`$.

**Linear Embedding:**
Each patch $`x_p^i \in \mathbb{R}^{P^2 \times C}`$ is flattened and projected to embedding dimension $`D`$:

```math
z_p^i = \text{Flatten}(x_p^i) \cdot E + b
```

Where $`E \in \mathbb{R}^{(P^2 \times C) \times D}`$ is the embedding matrix and $`b \in \mathbb{R}^D`$ is the bias.

**Position Embedding:**
Learnable position embeddings $`E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}`$ are added to provide spatial information:

```math
z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \ldots; x_p^N E] + E_{\text{pos}}
```

Where $`x_{\text{class}}`$ is a learnable classification token.

### Transformer Encoder

The transformer encoder consists of alternating layers of multi-head self-attention (MSA) and multi-layer perceptron (MLP) blocks.

**Multi-Head Self-Attention:**
```math
\text{MSA}(z) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
```

Where each head is computed as:
```math
\text{head}_i = \text{Attention}(zW_i^Q, zW_i^K, zW_i^V)
```

**Attention Mechanism:**
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

**Transformer Block:**
```math
z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}
z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l
```

Where:
- $`\text{LN}`$: Layer normalization
- $`\text{MLP}`$: Multi-layer perceptron with GELU activation

### Classification Head

The final classification is performed using the [CLS] token:

```math
y = \text{MLP}(\text{LN}(z_L^0))
```

Where $`z_L^0`$ is the [CLS] token from the final layer.

## Patch Embedding

### Implementation Details

**Implementation:** See `patch_embedding.py` for comprehensive patch embedding implementations:
- `PatchEmbed` - Standard patch embedding with convolutional projection
- `OverlappingPatchEmbed` - Overlapping patches for better feature extraction
- `HybridPatchEmbed` - Combination of CNN and patch embedding
- `PositionalPatchEmbed` - Patch embedding with positional information
- `AdaptivePatchEmbed` - Adaptive patch sizes based on image content
- `visualize_patches()` - Visualization utilities for patch extraction

## Transformer Encoder

### Multi-Head Self-Attention

**Implementation:** See `attention.py` for advanced attention mechanisms:
- `MultiHeadAttention` - Standard multi-head self-attention
- `RelativePositionAttention` - Attention with relative positional information
- `LocalAttention` - Local attention within windows
- `AxialAttention` - Attention along spatial axes
- `SparseAttention` - Sparse attention for efficiency
- `LinearAttention` - Linear complexity attention
- `visualize_attention_weights()` - Attention visualization utilities

### Transformer Block

**Implementation:** See `vision_transformer.py` for complete transformer implementation:
- `VisionTransformer` - Complete ViT architecture
- `Block` - Individual transformer blocks
- `Attention` - Multi-head attention implementation
- `MLP` - Feed-forward network implementation
- `DropPath` - Stochastic depth for regularization

## Classification Head

### Implementation

**Implementation:** See `vision_transformer.py` for classification head:
- Built-in classification head in `VisionTransformer`
- Support for both standard and distillation heads
- Layer normalization and dropout for regularization

## Complete Vision Transformer

### Full Implementation

**Implementation:** See `vision_transformer.py` for complete ViT implementation:
- `VisionTransformer` - Complete architecture with all components
- `create_vit_model()` - Factory function for different ViT variants
- Support for various model sizes (base, large, huge)
- Distillation support for efficient training

## Training Strategies

### Data Augmentation

**Implementation:** See `vision_transformer.py` for training utilities:
- Built-in support for various augmentation strategies
- Integration with PyTorch transforms
- Optimized for ViT training requirements

### Learning Rate Scheduling

**Implementation:** See `vision_transformer.py` for training components:
- Cosine annealing with warmup support
- Learning rate scheduling utilities
- Gradient clipping and optimization techniques

### Training Loop

**Implementation:** See `vision_transformer.py` for training pipeline:
- Complete training loop implementation
- Validation and evaluation utilities
- Model checkpointing and saving

## Variants and Improvements

### DeiT (Data-efficient Image Transformers)

DeiT introduces knowledge distillation to train ViT more efficiently with less data.

**Key Features:**
- **Distillation Token**: Additional token for distillation
- **Teacher Network**: Pre-trained CNN as teacher
- **Distillation Loss**: KL divergence between teacher and student

**Implementation:** See `vision_transformer.py` for DeiT support:
- Built-in distillation token support
- Teacher-student training utilities
- Distillation loss implementation

### Swin Transformer

Swin Transformer introduces hierarchical structure with shifted windows for better efficiency.

**Key Innovations:**
- **Window-based Attention**: Local attention within windows
- **Shifted Windows**: Shifted window partitioning for cross-window connections
- **Hierarchical Structure**: Multi-scale feature maps

### ConvNeXt

ConvNeXt modernizes CNNs by incorporating transformer design principles.

**Key Features:**
- **Large Kernel Sizes**: 7×7 convolutions
- **Inverted Bottleneck**: Wider intermediate layers
- **Layer Scale**: Scaling residual connections
- **Stochastic Depth**: Dropout for regularization

## Performance Analysis

### Comparison with CNNs

**Advantages of ViT:**
- **Global Receptive Field**: Attention can connect any two patches
- **Better Scaling**: Performance improves more with model size
- **Transfer Learning**: Strong performance on downstream tasks
- **Interpretability**: Attention maps provide insights

**Disadvantages:**
- **Computational Cost**: Quadratic complexity with sequence length
- **Data Requirements**: Needs large datasets for pre-training
- **Memory Usage**: High memory requirements for large models

### Scaling Laws

**Performance Scaling:**
```math
\text{Accuracy} \propto \log(\text{Model Size}) \times \log(\text{Data Size})
```

**Computational Complexity:**
```math
O(N^2 \times D) \text{ for attention}
O(N \times D^2) \text{ for MLP}
```

Where $`N`$ is the sequence length and $`D`$ is the embedding dimension.

## Applications

### Image Classification

**Implementation:** See `vision_transformer.py` for classification:
- Built-in image classification capabilities
- Support for various datasets and class numbers
- Efficient inference and prediction utilities

### Feature Extraction

**Implementation:** See `vision_transformer.py` for feature extraction:
- `forward_features()` - Extract intermediate features
- Hook-based feature extraction from any layer
- Feature visualization and analysis utilities

### Attention Visualization

**Implementation:** See `attention.py` for attention visualization:
- `visualize_attention_weights()` - Attention map visualization
- Support for visualizing attention from any layer and head
- Interactive attention analysis tools

## Conclusion

Vision Transformers represent a fundamental shift in computer vision, demonstrating that pure attention-based architectures can achieve state-of-the-art performance on visual tasks. The key innovations include:

1. **Patch-based Processing**: Treating images as sequences of patches
2. **Global Attention**: Capturing long-range dependencies
3. **Scalable Architecture**: Performance improves with model size
4. **Transfer Learning**: Strong performance on downstream tasks

**Key Takeaways:**
- ViT achieves competitive performance with CNNs on image classification
- Global attention provides better modeling of long-range dependencies
- Scalability makes ViT suitable for large-scale applications
- Transfer learning capabilities enable efficient adaptation to new tasks

**Future Directions:**
- **Efficiency Improvements**: Reducing computational complexity
- **Multi-scale Processing**: Better handling of different spatial scales
- **Multi-modal Integration**: Combining with other modalities
- **Real-time Applications**: Optimizing for deployment

---

**References:**
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" - Dosovitskiy et al.
- "Training data-efficient image transformers & distillation through attention" - Touvron et al.
- "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" - Liu et al.
- "A ConvNet for the 2020s" - Liu et al.

## From Architecture to Learning Without Labels

We've now explored **Vision Transformers (ViT)** - the paradigm shift in computer vision that adapts transformer architectures from natural language processing to visual data. We've seen how patch-based processing enables global attention across images, how the transformer encoder captures long-range dependencies, and how these architectures achieve state-of-the-art performance on image classification and other vision tasks.

However, while Vision Transformers provide powerful architectures for visual understanding, **the challenge of obtaining labeled data** remains a significant bottleneck in computer vision. Consider training a model to recognize thousands of object categories - collecting and annotating millions of images is expensive, time-consuming, and often impractical for many real-world applications.

This motivates our exploration of **self-supervised learning in vision** - techniques that enable models to learn meaningful representations from unlabeled data by solving carefully designed pretext tasks. We'll see how tasks like image inpainting, jigsaw puzzle solving, and rotation prediction can teach models to understand visual structure without manual annotations, how these learned representations transfer to downstream tasks, and how self-supervised learning has become a cornerstone of modern computer vision.

The transition from Vision Transformers to self-supervised learning represents the bridge from architectural innovation to learning efficiency - taking our understanding of powerful vision architectures and applying it to the challenge of learning from unlabeled data.

In the next section, we'll explore self-supervised learning, understanding how to design pretext tasks that enable effective representation learning without manual supervision.

---

**Next: [Self-Supervised Learning](02_self_supervised_learning.md)** - Learn how to train vision models without labeled data. 
# Vision Transformers (ViT)

## Overview

Vision Transformers (ViT) represent a paradigm shift in computer vision, adapting the transformer architecture from natural language processing to visual data. By treating images as sequences of patches, ViT has demonstrated that pure attention-based architectures can achieve state-of-the-art performance on image classification and other vision tasks, often surpassing traditional convolutional neural networks (CNNs).

### The Big Picture: Why Transformers for Vision?

**The Problem with Traditional CNNs:**
- **Local Receptive Fields**: Convolutions only "see" small local regions at a time
- **Limited Long-range Dependencies**: Hard to capture relationships between distant parts of an image
- **Fixed Architecture**: Convolutional patterns are predetermined, not learned

**The Transformer Solution:**
- **Global Attention**: Every patch can "attend" to every other patch in the image
- **Learnable Relationships**: The model learns which patches are important for understanding each other
- **Scalable Understanding**: Bigger models can capture more complex visual relationships

**Intuitive Analogy:**
Think of a CNN like looking at an image through a small moving window - you see details but miss the big picture. A Vision Transformer is like having a team of experts who can all look at the entire image at once and discuss how different parts relate to each other.

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

### Step-by-Step Walkthrough

**Step 1: Image Patching**
- Take a 224×224×3 image
- Divide into 14×14 = 196 patches of 16×16×3 each
- Each patch becomes a "word" in our visual vocabulary

**Step 2: Patch Embedding**
- Flatten each 16×16×3 patch to 768 dimensions
- This is like translating visual "words" into a common language

**Step 3: Position Embedding**
- Add position information to each patch
- This tells the model "where" each patch came from in the original image

**Step 4: Transformer Processing**
- Each patch can "look at" and "talk to" every other patch
- Multiple layers build increasingly complex understanding

**Step 5: Classification**
- Use a special [CLS] token that summarizes the entire image
- Make final prediction based on this global understanding

## Mathematical Foundations

### Understanding the Mathematics Intuitively

Before diving into equations, let's build intuition about what each mathematical operation accomplishes:

**Why Patches?**
- Images are too large to process as single units
- Patches create manageable "chunks" that can be processed efficiently
- Each patch contains local visual information (edges, textures, colors)

**Why Embeddings?**
- Raw pixel values (0-255) don't capture semantic meaning
- Embeddings project patches into a space where similar visual concepts are close together
- This is like learning a "visual language" where "cat face" and "dog face" are similar concepts

**Why Attention?**
- Attention computes "importance scores" between every pair of patches
- A patch of a cat's eye might pay high attention to the cat's nose and ears
- This creates a "visual graph" where patches are connected based on relevance

### Patch Embedding

The first step in ViT is to divide the input image into patches and embed them into a high-dimensional space.

**Patch Division:**
Given an input image $`x \in \mathbb{R}^{H \times W \times C}`$, we divide it into $`N = \frac{HW}{P^2}`$ patches of size $`P \times P \times C`$.

**Intuitive Explanation:**
- If you have a 224×224 image and 16×16 patches: $`N = \frac{224 \times 224}{16 \times 16} = 196`$ patches
- Each patch is like a small "tile" of the image
- Think of it as cutting a photo into 196 small squares

**Linear Embedding:**
Each patch $`x_p^i \in \mathbb{R}^{P^2 \times C}`$ is flattened and projected to embedding dimension $`D`$:

```math
z_p^i = \text{Flatten}(x_p^i) \cdot E + b
```

Where $`E \in \mathbb{R}^{(P^2 \times C) \times D}`$ is the embedding matrix and $`b \in \mathbb{R}^D`$ is the bias.

**What This Does:**
- Flatten: Convert 16×16×3 = 768 pixels into a single vector
- Linear projection: Transform this 768-dimensional vector into a higher-dimensional space (e.g., 1024 dimensions)
- This higher-dimensional space can capture more complex visual patterns

**Position Embedding:**
Learnable position embeddings $`E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}`$ are added to provide spatial information:

```math
z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \ldots; x_p^N E] + E_{\text{pos}}
```

Where $`x_{\text{class}}`$ is a learnable classification token.

**Why Position Embeddings Matter:**
- Without position information, the model wouldn't know that a patch from the top-left is different from one in the bottom-right
- Position embeddings are learned, so the model can discover optimal ways to encode spatial relationships
- The [CLS] token gets its own position embedding and learns to aggregate information from all patches

### Transformer Encoder

The transformer encoder consists of alternating layers of multi-head self-attention (MSA) and multi-layer perceptron (MLP) blocks.

**The Attention Mechanism Explained:**

**Step 1: Create Queries, Keys, and Values**
For each patch, we create three vectors:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What information do I contain?"
- **Value (V)**: "What information do I provide?"

**Step 2: Compute Attention Scores**
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

**Breaking This Down:**
- $`QK^T`$: Measures similarity between queries and keys
- $`\frac{1}{\sqrt{d_k}}`$: Scaling factor to prevent very large values
- $`\text{softmax}`$: Converts scores to probabilities (sum to 1)
- $`V`$: Weighted combination of values based on attention scores

**Intuitive Example:**
Imagine you're looking at a photo of a cat. Your brain might:
- Pay high attention to the cat's eyes (high attention score)
- Pay medium attention to the cat's ears (medium attention score)  
- Pay low attention to the background (low attention score)

**Multi-Head Self-Attention:**
```math
\text{MSA}(z) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
```

Where each head is computed as:
```math
\text{head}_i = \text{Attention}(zW_i^Q, zW_i^K, zW_i^V)
```

**Why Multiple Heads?**
- Different heads can learn to attend to different types of relationships
- Head 1 might learn to focus on color relationships
- Head 2 might learn to focus on spatial relationships
- Head 3 might learn to focus on texture relationships
- This creates a richer representation than single attention

**Transformer Block:**
```math
z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}
z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l
```

Where:
- $`\text{LN}`$: Layer normalization (stabilizes training)
- $`\text{MLP}`$: Multi-layer perceptron with GELU activation
- The $`+ z_{l-1}`$ terms are residual connections (help with gradient flow)

**What Each Layer Does:**
1. **Layer Norm**: Normalizes the input to have mean 0 and variance 1
2. **Multi-Head Attention**: Computes attention between all patches
3. **Residual Connection**: Adds the original input (helps with training)
4. **Layer Norm**: Normalizes again
5. **MLP**: Applies non-linear transformations
6. **Residual Connection**: Adds the attention output

### Classification Head

The final classification is performed using the [CLS] token:

```math
y = \text{MLP}(\text{LN}(z_L^0))
```

Where $`z_L^0`$ is the [CLS] token from the final layer.

**The [CLS] Token's Role:**
- The [CLS] token starts as a learnable vector
- Through attention, it "collects" information from all patches
- By the final layer, it contains a global representation of the entire image
- This global representation is used for classification

**Why This Works:**
- The [CLS] token can attend to any patch in the image
- It learns to aggregate the most relevant information for the task
- This is more flexible than global average pooling (used in CNNs)

## Patch Embedding

### Implementation Details

**Implementation:** See `code/patch_embedding.py` for comprehensive patch embedding implementations:
- `PatchEmbed` - Standard patch embedding with convolutional projection
- `OverlappingPatchEmbed` - Overlapping patches for better feature extraction
- `HybridPatchEmbed` - Combination of CNN and patch embedding
- `PositionalPatchEmbed` - Patch embedding with positional information
- `AdaptivePatchEmbed` - Adaptive patch sizes based on image content
- `visualize_patches()` - Visualization utilities for patch extraction

### Understanding Patch Embedding Choices

**Patch Size Trade-offs:**
- **Small patches (8×8)**: More patches, finer detail, higher computational cost
- **Large patches (32×32)**: Fewer patches, coarser detail, lower computational cost
- **Standard choice (16×16)**: Good balance between detail and efficiency

**Embedding Dimension:**
- **Small dimension (512)**: Faster computation, less expressive
- **Large dimension (1024)**: More expressive, higher computational cost
- **Scales with model size**: Larger models use larger embeddings

## Transformer Encoder

### Multi-Head Self-Attention

**Implementation:** See `code/attention.py` for advanced attention mechanisms:
- `MultiHeadAttention` - Standard multi-head self-attention
- `RelativePositionAttention` - Attention with relative positional information
- `LocalAttention` - Local attention within windows
- `AxialAttention` - Attention along spatial axes
- `SparseAttention` - Sparse attention for efficiency
- `LinearAttention` - Linear complexity attention
- `visualize_attention_weights()` - Attention visualization utilities

### Understanding Attention Patterns

**What Attention Learns:**
- **Object Parts**: Attention between eyes, nose, mouth in face recognition
- **Spatial Relationships**: Attention between foreground and background
- **Semantic Similarity**: Attention between similar objects or textures
- **Contextual Information**: Attention to surrounding context

**Visualizing Attention:**
- Attention maps show which patches are most important for each decision
- Brighter regions indicate higher attention weights
- This provides interpretability - we can see what the model is "looking at"

### Transformer Block

**Implementation:** See `code/vision_transformer.py` for complete transformer implementation:
- `VisionTransformer` - Complete ViT architecture
- `Block` - Individual transformer blocks
- `Attention` - Multi-head attention implementation
- `MLP` - Feed-forward network implementation
- `DropPath` - Stochastic depth for regularization

### Layer-by-Layer Understanding

**Early Layers (1-6):**
- Learn low-level features: edges, textures, colors
- Attention focuses on local relationships
- Similar to early CNN layers

**Middle Layers (7-12):**
- Learn mid-level features: object parts, shapes
- Attention spans larger regions
- Combines information from multiple patches

**Late Layers (13+):**
- Learn high-level features: objects, scenes
- Attention becomes more global
- [CLS] token aggregates semantic information

## Classification Head

### Implementation

**Implementation:** See `code/vision_transformer.py` for classification head:
- Built-in classification head in `VisionTransformer`
- Support for both standard and distillation heads
- Layer normalization and dropout for regularization

### Alternative Classification Strategies

**Global Average Pooling:**
- Average all patch embeddings instead of using [CLS] token
- Simpler but less flexible than attention-based aggregation

**Multiple Tokens:**
- Use multiple classification tokens for different aspects
- Can capture different types of information

**Token Pooling:**
- Pool tokens based on attention weights
- More interpretable than simple averaging

## Complete Vision Transformer

### Full Implementation

**Implementation:** See `code/vision_transformer.py` for complete ViT implementation:
- `VisionTransformer` - Complete architecture with all components
- `create_vit_model()` - Factory function for different ViT variants
- Support for various model sizes (base, large, huge)
- Distillation support for efficient training

### Model Size Variants

**ViT-Base:**
- 12 layers, 12 heads, 768 embedding dimension
- 86M parameters
- Good balance of performance and efficiency

**ViT-Large:**
- 24 layers, 16 heads, 1024 embedding dimension
- 307M parameters
- Higher accuracy, more computational cost

**ViT-Huge:**
- 32 layers, 16 heads, 1280 embedding dimension
- 632M parameters
- Best accuracy, highest computational cost

## Training Strategies

### Data Augmentation

**Implementation:** See `code/vision_transformer.py` for training utilities:
- Built-in support for various augmentation strategies
- Integration with PyTorch transforms
- Optimized for ViT training requirements

**Why Augmentation is Critical:**
- ViTs need more data than CNNs to learn effectively
- Augmentation creates "virtual" data from existing samples
- Helps prevent overfitting and improves generalization

**Effective Augmentation Strategies:**
- **Random cropping**: Teaches invariance to object position
- **Color jittering**: Teaches invariance to lighting conditions
- **Random horizontal flipping**: Teaches invariance to orientation
- **Mixup/CutMix**: Creates interpolated samples between classes

### Learning Rate Scheduling

**Implementation:** See `code/vision_transformer.py` for training components:
- Cosine annealing with warmup support
- Learning rate scheduling utilities
- Gradient clipping and optimization techniques

**Why Warmup is Important:**
- Attention weights start random and need time to stabilize
- Warmup prevents early training instability
- Gradually increases learning rate to final value

**Cosine Annealing:**
- Learning rate decreases smoothly over training
- Helps convergence in later stages
- Better than step-wise decay for transformers

### Training Loop

**Implementation:** See `code/vision_transformer.py` for training pipeline:
- Complete training loop implementation
- Validation and evaluation utilities
- Model checkpointing and saving

**Training Best Practices:**
- **Gradient clipping**: Prevents exploding gradients
- **Mixed precision**: Reduces memory usage and speeds up training
- **Checkpointing**: Saves model state for resuming training
- **Early stopping**: Prevents overfitting

## Variants and Improvements

### DeiT (Data-efficient Image Transformers)

DeiT introduces knowledge distillation to train ViT more efficiently with less data.

**Key Features:**
- **Distillation Token**: Additional token for distillation
- **Teacher Network**: Pre-trained CNN as teacher
- **Distillation Loss**: KL divergence between teacher and student

**How Distillation Works:**
1. Train a large teacher model (e.g., CNN) on full dataset
2. Use teacher's predictions as "soft labels" for student (ViT)
3. Student learns from both hard labels and teacher's knowledge
4. Requires much less data than training ViT from scratch

**Implementation:** See `code/vision_transformer.py` for DeiT support:
- Built-in distillation token support
- Teacher-student training utilities
- Distillation loss implementation

### Swin Transformer

Swin Transformer introduces hierarchical structure with shifted windows for better efficiency.

**Key Innovations:**
- **Window-based Attention**: Local attention within windows
- **Shifted Windows**: Shifted window partitioning for cross-window connections
- **Hierarchical Structure**: Multi-scale feature maps

**Why Windows Help:**
- Full attention is computationally expensive (O(N²))
- Window attention reduces complexity to O(W²) where W is window size
- Shifted windows maintain cross-window connections
- Creates hierarchical feature maps like CNNs

### ConvNeXt

ConvNeXt modernizes CNNs by incorporating transformer design principles.

**Key Features:**
- **Large Kernel Sizes**: 7×7 convolutions (inspired by attention's global view)
- **Inverted Bottleneck**: Wider intermediate layers
- **Layer Scale**: Scaling residual connections
- **Stochastic Depth**: Dropout for regularization

**The Best of Both Worlds:**
- CNN's efficiency and inductive biases
- Transformer's modern design principles
- Often outperforms both CNNs and ViTs on efficiency-accuracy trade-off

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

**When to Use Each:**
- **Use CNNs**: Limited data, real-time applications, edge devices
- **Use ViTs**: Large datasets, high accuracy requirements, interpretability needs

### Scaling Laws

**Performance Scaling:**
```math
\text{Accuracy} \propto \log(\text{Model Size}) \times \log(\text{Data Size})
```

**What This Means:**
- Doubling model size gives logarithmic improvement in accuracy
- Doubling data size also gives logarithmic improvement
- Both model size and data size are important for performance

**Computational Complexity:**
```math
O(N^2 \times D) \text{ for attention}
O(N \times D^2) \text{ for MLP}
```

Where $`N`$ is the sequence length and $`D`$ is the embedding dimension.

**Practical Implications:**
- Attention dominates computation for large images
- MLP dominates computation for large embedding dimensions
- This motivates research into efficient attention mechanisms

### Memory Requirements

**Memory Breakdown:**
- **Model parameters**: ~100M-600M parameters
- **Activation memory**: O(N × D × L) for intermediate activations
- **Attention memory**: O(N² × H × L) for attention matrices
- **Gradient memory**: Similar to activation memory

**Memory Optimization Strategies:**
- **Gradient checkpointing**: Trade computation for memory
- **Mixed precision**: Use FP16 to reduce memory usage
- **Model parallelism**: Distribute model across multiple GPUs
- **Attention optimization**: Use sparse or linear attention

## Applications

### Image Classification

**Implementation:** See `code/vision_transformer.py` for classification:
- Built-in image classification capabilities
- Support for various datasets and class numbers
- Efficient inference and prediction utilities

**Performance on Standard Benchmarks:**
- **ImageNet**: 88.55% top-1 accuracy (ViT-H/14)
- **CIFAR-10**: 99.50% accuracy
- **CIFAR-100**: 94.55% accuracy

### Feature Extraction

**Implementation:** See `code/vision_transformer.py` for feature extraction:
- `forward_features()` - Extract intermediate features
- Hook-based feature extraction from any layer
- Feature visualization and analysis utilities

**What Features Represent:**
- **Early layers**: Low-level visual features (edges, textures)
- **Middle layers**: Mid-level features (object parts, shapes)
- **Late layers**: High-level semantic features (objects, scenes)

**Downstream Applications:**
- **Object detection**: Use features as backbone for detection heads
- **Semantic segmentation**: Use features for pixel-level classification
- **Image retrieval**: Use features for similarity search
- **Transfer learning**: Fine-tune on new tasks

### Attention Visualization

**Implementation:** See `code/attention.py` for attention visualization:
- `visualize_attention_weights()` - Attention map visualization
- Support for visualizing attention from any layer and head
- Interactive attention analysis tools

**What Attention Maps Show:**
- **Object localization**: Which patches contain the object of interest
- **Part relationships**: How different object parts relate to each other
- **Context usage**: How background context influences classification
- **Model interpretability**: Understanding model decision-making

**Visualization Techniques:**
- **Attention rollout**: Aggregate attention across layers
- **Attention flow**: Track attention from input to output
- **Class-specific attention**: Show attention for specific classes
- **Interactive exploration**: Explore attention patterns interactively

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

**The Broader Impact:**
Vision Transformers have fundamentally changed how we think about computer vision. They've shown that:
- **Architecture matters**: The right architecture can unlock new capabilities
- **Global context is crucial**: Local processing alone isn't sufficient
- **Scalability is key**: Bigger models with more data lead to better performance
- **Interpretability is possible**: Attention provides insights into model decisions

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
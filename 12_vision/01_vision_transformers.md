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

**Patch Division:**
```python
def create_patches(image, patch_size=16):
    """
    Divide image into patches.
    
    Args:
        image: Input image of shape (batch_size, channels, height, width)
        patch_size: Size of each patch
    
    Returns:
        patches: Tensor of shape (batch_size, num_patches, patch_size^2 * channels)
    """
    batch_size, channels, height, width = image.shape
    num_patches = (height // patch_size) * (width // patch_size)
    
    # Reshape to extract patches
    patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(batch_size, num_patches, patch_size * patch_size * channels)
    
    return patches
```

**Linear Embedding:**
```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positions = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Create patches
        patches = create_patches(x, self.patch_size)
        
        # Linear projection
        embeddings = self.projection(patches)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.positions
        
        return embeddings
```

## Transformer Encoder

### Multi-Head Self-Attention

**Implementation:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear transformation for Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(batch_size, seq_len, embed_dim)
        
        # Final linear transformation
        output = self.projection(context)
        
        return output
```

### Transformer Block

**Complete Implementation:**
```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x
```

## Classification Head

### Implementation

```python
class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Extract [CLS] token
        cls_token = x[:, 0]
        
        # Apply layer normalization
        cls_token = self.norm(cls_token)
        
        # Apply dropout
        cls_token = self.dropout(cls_token)
        
        # Classification
        logits = self.classifier(cls_token)
        
        return logits
```

## Complete Vision Transformer

### Full Implementation

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.classifier = ClassificationHead(embed_dim, num_classes, dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
```

## Training Strategies

### Data Augmentation

**Effective Augmentation for ViT:**
```python
def get_vit_augmentation():
    """Get data augmentation pipeline for ViT training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
```

### Learning Rate Scheduling

**Cosine Annealing with Warmup:**
```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Create cosine learning rate schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Training Loop

**Complete Training Implementation:**
```python
def train_vit(model, train_loader, val_loader, num_epochs=100, lr=1e-4):
    """Train Vision Transformer."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=1000, num_training_steps=len(train_loader) * num_epochs
    )
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
```

## Variants and Improvements

### DeiT (Data-efficient Image Transformers)

DeiT introduces knowledge distillation to train ViT more efficiently with less data.

**Key Features:**
- **Distillation Token**: Additional token for distillation
- **Teacher Network**: Pre-trained CNN as teacher
- **Distillation Loss**: KL divergence between teacher and student

**Implementation:**
```python
class DeiT(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_token = nn.Parameter(torch.randn(1, 1, kwargs['embed_dim']))
        self.distillation_head = ClassificationHead(kwargs['embed_dim'], kwargs['num_classes'])
    
    def forward(self, x):
        # Add distillation token
        batch_size = x.shape[0]
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        
        # Patch embedding
        x = self.patch_embed(x)
        x = torch.cat([x[:, :1], distillation_tokens, x[:, 1:]], dim=1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification and distillation
        cls_logits = self.classifier(x[:, 0])  # [CLS] token
        dist_logits = self.distillation_head(x[:, 1])  # Distillation token
        
        return cls_logits, dist_logits
```

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

**Implementation:**
```python
def classify_image(model, image_path, class_names):
    """Classify an image using ViT."""
    # Load and preprocess image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
    
    return class_names[predicted_class], probabilities[0][predicted_class].item()
```

### Feature Extraction

**Extracting Features:**
```python
def extract_features(model, image):
    """Extract features from intermediate layers."""
    features = []
    
    def hook_fn(module, input, output):
        features.append(output)
    
    # Register hooks for intermediate layers
    hooks = []
    for block in model.blocks:
        hooks.append(block.register_forward_hook(hook_fn))
    
    # Forward pass
    with torch.no_grad():
        model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return features
```

### Attention Visualization

**Visualizing Attention Maps:**
```python
def visualize_attention(model, image, layer_idx=0, head_idx=0):
    """Visualize attention maps from a specific layer and head."""
    attention_maps = []
    
    def attention_hook(module, input, output):
        attention_maps.append(output)
    
    # Register hook for attention
    hook = model.blocks[layer_idx].attention.register_forward_hook(attention_hook)
    
    # Forward pass
    with torch.no_grad():
        model(image)
    
    # Extract attention weights
    attention = attention_maps[0][0, head_idx]  # [seq_len, seq_len]
    
    # Remove hook
    hook.remove()
    
    return attention
```

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
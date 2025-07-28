# Self-Supervised Learning in Vision

## Overview

Self-supervised learning has revolutionized computer vision by enabling models to learn meaningful representations from unlabeled data. By solving carefully designed pretext tasks, models can capture visual structure and semantics without manual annotations, leading to powerful representations that transfer well to downstream tasks.

### Key Principles

**Core Concepts:**
- **Pretext Tasks**: Tasks that can be solved without labels but require understanding visual structure
- **Representation Learning**: Learning features that are useful for downstream tasks
- **Transfer Learning**: Applying learned representations to new tasks with minimal labeled data
- **Unsupervised Pre-training**: Learning from large amounts of unlabeled data

## Table of Contents

- [Pretext Tasks](#pretext-tasks)
- [Contrastive Learning](#contrastive-learning)
- [Modern Self-Supervised Methods](#modern-self-supervised-methods)
- [Implementation Examples](#implementation-examples)
- [Evaluation and Transfer](#evaluation-and-transfer)
- [Applications](#applications)

## Pretext Tasks

### Overview

Pretext tasks are self-supervised learning objectives that can be solved without manual labels but require understanding visual structure and semantics.

**Key Characteristics:**
- **Automatic Supervision**: Labels are generated automatically from the data
- **Visual Understanding**: Tasks require understanding of visual structure
- **Transferable Features**: Learned representations transfer to downstream tasks

### Image Inpainting

**Task Definition:**
Fill in masked regions of an image using surrounding context.

**Implementation:**
```python
class InpaintingPretextTask(nn.Module):
    def __init__(self, mask_ratio=0.15, mask_size=32):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_size = mask_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def create_mask(self, image_size):
        """Create random rectangular masks."""
        mask = torch.ones(image_size)
        num_masks = int(self.mask_ratio * (image_size[1] * image_size[2]) / (self.mask_size ** 2))
        
        for _ in range(num_masks):
            x = torch.randint(0, image_size[1] - self.mask_size, (1,))
            y = torch.randint(0, image_size[2] - self.mask_size, (1,))
            mask[:, x:x+self.mask_size, y:y+self.mask_size] = 0
        
        return mask
    
    def forward(self, x):
        # Create mask
        mask = self.create_mask(x.shape).to(x.device)
        
        # Apply mask
        masked_x = x * mask
        
        # Encode
        features = self.encoder(masked_x)
        
        # Decode
        reconstructed = self.decoder(features)
        
        return reconstructed, x, mask
```

### Jigsaw Puzzle Solving

**Task Definition:**
Reconstruct the original image from shuffled patches.

**Implementation:**
```python
class JigsawPretextTask(nn.Module):
    def __init__(self, num_patches=9, num_permutations=100):
        super().__init__()
        self.num_patches = num_patches
        self.num_permutations = num_permutations
        self.patch_size = 64
        
        # Permutation classifier
        self.classifier = nn.Sequential(
            nn.Linear(3 * self.patch_size * self.patch_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_permutations)
        )
    
    def create_patches(self, x):
        """Divide image into patches."""
        batch_size = x.shape[0]
        patches = []
        
        for i in range(int(math.sqrt(self.num_patches))):
            for j in range(int(math.sqrt(self.num_patches))):
                patch = x[:, :, 
                         i*self.patch_size:(i+1)*self.patch_size,
                         j*self.patch_size:(j+1)*self.patch_size]
                patches.append(patch.flatten(1))
        
        return torch.stack(patches, dim=1)  # [batch_size, num_patches, patch_dim]
    
    def create_permutation(self):
        """Create a random permutation of patches."""
        return torch.randperm(self.num_patches)
    
    def forward(self, x):
        # Create patches
        patches = self.create_patches(x)
        
        # Create permutation
        perm = self.create_permutation()
        
        # Apply permutation
        permuted_patches = patches[:, perm]
        
        # Flatten for classification
        permuted_input = permuted_patches.flatten(1)
        
        # Predict permutation
        logits = self.classifier(permuted_input)
        
        return logits, perm
```

### Rotation Prediction

**Task Definition:**
Predict the rotation angle applied to an image (0°, 90°, 180°, 270°).

**Implementation:**
```python
class RotationPretextTask(nn.Module):
    def __init__(self, num_rotations=4):
        super().__init__()
        self.num_rotations = num_rotations
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_rotations)
        )
    
    def rotate_image(self, x, rotation_idx):
        """Apply rotation to image."""
        angles = [0, 90, 180, 270]
        angle = angles[rotation_idx]
        
        if angle == 0:
            return x
        elif angle == 90:
            return x.transpose(2, 3).flip(2)
        elif angle == 180:
            return x.flip(2, 3)
        elif angle == 270:
            return x.transpose(2, 3).flip(3)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Create rotated versions
        rotated_images = []
        labels = []
        
        for i in range(self.num_rotations):
            rotated = self.rotate_image(x, i)
            rotated_images.append(rotated)
            labels.extend([i] * batch_size)
        
        # Stack rotated images
        rotated_batch = torch.cat(rotated_images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long, device=x.device)
        
        # Predict rotation
        logits = self.encoder(rotated_batch)
        
        return logits, labels
```

### Colorization

**Task Definition:**
Predict color channels from grayscale input.

**Implementation:**
```python
class ColorizationPretextTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, padding=1),  # a, b channels
            nn.Tanh()
        )
    
    def rgb_to_lab(self, x):
        """Convert RGB to LAB color space."""
        # Simplified conversion (in practice, use proper color space conversion)
        # This is a placeholder for the actual RGB to LAB conversion
        return x
    
    def lab_to_rgb(self, l, ab):
        """Convert LAB to RGB color space."""
        # Simplified conversion (in practice, use proper color space conversion)
        # This is a placeholder for the actual LAB to RGB conversion
        return torch.cat([l, ab], dim=1)
    
    def forward(self, x):
        # Convert to LAB
        lab = self.rgb_to_lab(x)
        l_channel = lab[:, 0:1]  # Lightness channel
        ab_channels = lab[:, 1:]  # a, b channels
        
        # Encode grayscale
        features = self.encoder(l_channel)
        
        # Decode color
        predicted_ab = self.decoder(features)
        
        return predicted_ab, ab_channels
```

## Contrastive Learning

### Overview

Contrastive learning learns representations by maximizing agreement between different augmented views of the same image while minimizing agreement with views from different images.

**Key Components:**
- **Data Augmentation**: Create multiple views of the same image
- **Encoder Network**: Extract representations from augmented views
- **Projection Head**: Project representations to comparison space
- **Contrastive Loss**: Maximize similarity of positive pairs, minimize similarity of negative pairs

### SimCLR Framework

**Architecture:**
```python
class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim=128, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x1, x2):
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        # Project to comparison space
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        return z1, z2
    
    def contrastive_loss(self, z1, z2):
        """Compute contrastive loss."""
        batch_size = z1.shape[0]
        
        # Concatenate all representations
        representations = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature
        
        # Create labels for positive pairs
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        labels = labels[~mask].view(2 * batch_size, -1)
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
```

**Data Augmentation:**
```python
class SimCLRAugmentation:
    def __init__(self, size=224):
        self.size = size
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)
```

### MoCo (Momentum Contrast)

**Key Innovations:**
- **Momentum Encoder**: Slowly updated encoder for consistency
- **Queue**: Large queue of negative samples
- **Momentum Update**: Exponential moving average of encoder parameters

**Implementation:**
```python
class MoCo(nn.Module):
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
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
    
    @torch.no_grad()
    def momentum_update(self):
        """Update key encoder with momentum."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """Dequeue and enqueue keys."""
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
        # Query encoder
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)
        
        # Key encoder
        with torch.no_grad():
            k = self.encoder_k(im_k)
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
```

## Modern Self-Supervised Methods

### BYOL (Bootstrap Your Own Latent)

**Key Innovation:**
BYOL uses two networks (online and target) where the target network is an exponential moving average of the online network.

**Implementation:**
```python
class BYOL(nn.Module):
    def __init__(self, encoder, projection_dim=256, prediction_dim=256, momentum=0.996):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Online network
        self.online_projector = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, prediction_dim)
        )
        
        # Target network
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Freeze target network
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def momentum_update(self):
        """Update target network with momentum."""
        for param_online, param_target in zip(self.online_projector.parameters(), 
                                            self.target_projector.parameters()):
            param_target.data = param_target.data * self.momentum + param_online.data * (1. - self.momentum)
    
    def forward(self, x1, x2):
        # Online network
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        
        z1_online = self.online_projector(h1)
        z2_online = self.online_projector(h2)
        
        p1 = self.online_predictor(z1_online)
        p2 = self.online_predictor(z2_online)
        
        # Target network
        with torch.no_grad():
            z1_target = self.target_projector(h1)
            z2_target = self.target_projector(h2)
        
        # Normalize
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1_target = F.normalize(z1_target, dim=1)
        z2_target = F.normalize(z2_target, dim=1)
        
        return p1, p2, z1_target, z2_target
    
    def loss(self, p1, p2, z1_target, z2_target):
        """Compute BYOL loss."""
        loss1 = 2 - 2 * F.cosine_similarity(p1, z2_target, dim=1)
        loss2 = 2 - 2 * F.cosine_similarity(p2, z1_target, dim=1)
        
        return (loss1 + loss2).mean()
```

### DINO (Self-Distillation with No Labels)

**Key Features:**
- **Multi-crop Strategy**: Different views of the same image
- **Centering and Sharpening**: Stabilize training
- **Knowledge Distillation**: Student learns from teacher

**Implementation:**
```python
class DINO(nn.Module):
    def __init__(self, encoder, projection_dim=256, momentum=0.996, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        self.temperature = temperature
        
        # Student network
        self.student_head = nn.Sequential(
            nn.Linear(encoder.output_dim, 512),
            nn.GELU(),
            nn.Linear(512, projection_dim)
        )
        
        # Teacher network
        self.teacher_head = copy.deepcopy(self.student_head)
        
        # Freeze teacher
        for param in self.teacher_head.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def momentum_update(self):
        """Update teacher network with momentum."""
        for param_student, param_teacher in zip(self.student_head.parameters(), 
                                              self.teacher_head.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + param_student.data * (1. - self.momentum)
    
    def forward(self, crops):
        # Student forward pass
        student_outputs = []
        for crop in crops:
            features = self.encoder(crop)
            output = self.student_head(features)
            student_outputs.append(output)
        
        # Teacher forward pass (only for global crops)
        teacher_outputs = []
        with torch.no_grad():
            for i in range(len(crops) // 2):  # Only global crops
                features = self.encoder(crops[i])
                output = self.teacher_head(features)
                teacher_outputs.append(output)
        
        return student_outputs, teacher_outputs
    
    def loss(self, student_outputs, teacher_outputs):
        """Compute DINO loss."""
        total_loss = 0
        num_crops = len(student_outputs)
        num_global_crops = len(teacher_outputs)
        
        # Compute loss for each global crop
        for i in range(num_global_crops):
            teacher_output = teacher_outputs[i]
            
            # Compute loss with all student outputs
            for j in range(num_crops):
                student_output = student_outputs[j]
                
                # Cross-entropy loss
                logits = torch.matmul(student_output, teacher_output.T) / self.temperature
                labels = torch.arange(student_output.shape[0], device=student_output.device)
                loss = F.cross_entropy(logits, labels)
                
                total_loss += loss
        
        return total_loss / (num_global_crops * num_crops)
```

## Implementation Examples

### Training Pipeline

**Complete Training Loop:**
```python
def train_self_supervised(model, train_loader, num_epochs=100, lr=1e-3):
    """Train self-supervised model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Create augmented views
            if isinstance(model, (SimCLR, MoCo)):
                aug1, aug2 = data[0], data[1]
                optimizer.zero_grad()
                
                if isinstance(model, SimCLR):
                    z1, z2 = model(aug1, aug2)
                    loss = model.contrastive_loss(z1, z2)
                else:  # MoCo
                    logits, labels = model(aug1, aug2)
                    loss = F.cross_entropy(logits, labels)
                
                loss.backward()
                optimizer.step()
                
                # Update momentum encoder
                if isinstance(model, MoCo):
                    model.momentum_update()
            
            elif isinstance(model, BYOL):
                aug1, aug2 = data[0], data[1]
                optimizer.zero_grad()
                
                p1, p2, z1_target, z2_target = model(aug1, aug2)
                loss = model.loss(p1, p2, z1_target, z2_target)
                
                loss.backward()
                optimizer.step()
                
                # Update momentum encoder
                model.momentum_update()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}')
```

### Evaluation and Transfer

**Linear Evaluation:**
```python
def linear_evaluation(encoder, train_loader, val_loader, num_classes=10):
    """Evaluate learned representations with linear classifier."""
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

## Applications

### Feature Extraction

**Extracting Features:**
```python
def extract_features(encoder, dataloader):
    """Extract features from pre-trained encoder."""
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

### Downstream Tasks

**Fine-tuning for Classification:**
```python
def fine_tune_classifier(encoder, num_classes, train_loader, val_loader):
    """Fine-tune encoder for classification."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add classification head
    classifier = nn.Linear(encoder.output_dim, num_classes)
    model = nn.Sequential(encoder, classifier).to(device)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model
```

## Conclusion

Self-supervised learning has emerged as a powerful paradigm for learning visual representations without manual annotations. Key innovations include:

1. **Pretext Tasks**: Tasks that can be solved automatically but require visual understanding
2. **Contrastive Learning**: Learning by comparing different views of the same image
3. **Modern Methods**: BYOL, DINO, and other advanced techniques
4. **Transfer Learning**: Applying learned representations to downstream tasks

**Key Takeaways:**
- Self-supervised learning can learn powerful representations from unlabeled data
- Contrastive learning has been particularly successful for visual representation learning
- Modern methods like BYOL and DINO achieve state-of-the-art performance
- Learned representations transfer well to downstream tasks

**Future Directions:**
- **Multi-modal Learning**: Combining vision with other modalities
- **Efficiency Improvements**: Reducing computational requirements
- **Better Pretext Tasks**: Designing more effective self-supervised objectives
- **Real-world Applications**: Deploying in production systems

---

**References:**
- "A Simple Framework for Contrastive Learning of Visual Representations" - Chen et al.
- "Momentum Contrast for Unsupervised Visual Representation Learning" - He et al.
- "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning" - Grill et al.
- "Emerging Properties in Self-Supervised Vision Transformers" - Caron et al. 
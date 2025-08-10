"""
MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

This module implements MoCo, a contrastive learning framework that uses a momentum encoder
and a queue of negative samples to learn visual representations.

References:
- He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum contrast for
  unsupervised visual representation learning. CVPR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
import random
from collections import deque


class MoCoTransform:
    """Data augmentation for MoCo."""
    
    def __init__(self, image_size: int = 224, s: float = 1.0):
        """
        Initialize MoCo transforms.
        
        Args:
            image_size: Size of the image
            s: Strength of color jittering
        """
        self.image_size = image_size
        self.s = s
        
        # Color jittering
        color_jitter = transforms.ColorJitter(
            0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s
        )
        
        # Data augmentation pipeline
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        """Apply two random augmentations."""
        return self.transform(x), self.transform(x)


class MoCoModel(nn.Module):
    """MoCo model with query encoder and momentum key encoder."""
    
    def __init__(self, encoder: str = 'resnet18', projection_dim: int = 128, 
                 queue_size: int = 65536, momentum: float = 0.999):
        """
        Initialize MoCo model.
        
        Args:
            encoder: Encoder architecture ('resnet18', 'resnet50', etc.)
            projection_dim: Dimension of projection head output
            queue_size: Size of the queue for negative samples
            momentum: Momentum parameter for key encoder update
        """
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        
        # Query encoder
        self.encoder_q = self._build_encoder(encoder)
        self.projection_q = self._build_projection_head(encoder, projection_dim)
        
        # Key encoder (momentum encoder)
        self.encoder_k = self._build_encoder(encoder)
        self.projection_k = self._build_projection_head(encoder, projection_dim)
        
        # Initialize key encoder with query encoder weights
        self._init_key_encoder()
        
        # Queue for negative samples
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def _build_encoder(self, encoder: str) -> nn.Module:
        """Build encoder network."""
        if encoder == 'resnet18':
            import torchvision.models as models
            encoder_model = models.resnet18(pretrained=False)
            # Remove the final classification layer
            encoder_model = nn.Sequential(*list(encoder_model.children())[:-1])
        elif encoder == 'resnet50':
            import torchvision.models as models
            encoder_model = models.resnet50(pretrained=False)
            encoder_model = nn.Sequential(*list(encoder_model.children())[:-1])
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        
        return encoder_model
    
    def _build_projection_head(self, encoder: str, projection_dim: int) -> nn.Module:
        """Build projection head."""
        if encoder == 'resnet18':
            feature_dim = 512
        elif encoder == 'resnet50':
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        
        return nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def _init_key_encoder(self):
        """Initialize key encoder with query encoder weights."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        for param_q, param_k in zip(self.projection_q.parameters(), self.projection_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Dequeue and enqueue keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr
        if ptr + batch_size > self.queue_size:
            batch_size = self.queue_size - ptr
            keys = keys[:batch_size]
        
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            im_q: Query images (batch_size, channels, height, width)
            im_k: Key images (batch_size, channels, height, width)
            
        Returns:
            Tuple of (query representations, key representations)
        """
        # Query encoder
        q = self.encoder_q(im_q)
        q = q.view(q.size(0), -1)
        q = self.projection_q(q)
        q = F.normalize(q, dim=1)
        
        # Key encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()
            
            k = self.encoder_k(im_k)
            k = k.view(k.size(0), -1)
            k = self.projection_k(k)
            k = F.normalize(k, dim=1)
        
        return q, k


class MoCoLoss(nn.Module):
    """InfoNCE loss for MoCo."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            q: Query representations (batch_size, projection_dim)
            k: Key representations (batch_size, projection_dim)
            queue: Queue of negative samples (projection_dim, queue_size)
            
        Returns:
            Loss value
        """
        batch_size = q.size(0)
        
        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        
        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])
        
        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        
        # Labels: positive key is the first
        labels = torch.zeros(batch_size, dtype=torch.long, device=q.device)
        
        # Temperature
        logits = logits / self.temperature
        
        return self.criterion(logits, labels)


class MoCoDataset:
    """Dataset wrapper for MoCo."""
    
    def __init__(self, dataset, transform):
        """
        Initialize MoCo dataset.
        
        Args:
            dataset: Base dataset
            transform: MoCo transform
        """
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get two augmented views of the same image."""
        image, label = self.dataset[idx]
        view_q, view_k = self.transform(image)
        
        return {
            'view_q': view_q,
            'view_k': view_k,
            'label': label
        }


def train_moco(model: nn.Module, 
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               num_epochs: int = 100,
               learning_rate: float = 0.001,
               temperature: float = 0.07,
               device: str = 'cuda'):
    """
    Train MoCo model.
    
    Args:
        model: MoCo model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        temperature: Temperature parameter
        device: Device to train on
    """
    model = model.to(device)
    criterion = MoCoLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            view_q = batch['view_q'].to(device)
            view_k = batch['view_k'].to(device)
            
            optimizer.zero_grad()
            q, k = model(view_q, view_k)
            loss = criterion(q, k, model.queue)
            loss.backward()
            optimizer.step()
            
            # Update queue
            model._dequeue_and_enqueue(k)
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                view_q = batch['view_q'].to(device)
                view_k = batch['view_k'].to(device)
                
                q, k = model(view_q, view_k)
                loss = criterion(q, k, model.queue)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_moco_model.pth')
        
        scheduler.step()


def evaluate_moco(model: nn.Module, 
                 test_loader: torch.utils.data.DataLoader,
                 temperature: float = 0.07,
                 device: str = 'cuda') -> float:
    """
    Evaluate MoCo model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        temperature: Temperature parameter
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    criterion = MoCoLoss(temperature=temperature)
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            view_q = batch['view_q'].to(device)
            view_k = batch['view_k'].to(device)
            
            q, k = model(view_q, view_k)
            loss = criterion(q, k, model.queue)
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader)
    return avg_loss


def extract_features(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader,
                    device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from the query encoder for downstream tasks.
    
    Args:
        model: Trained MoCo model
        dataloader: Data loader
        device: Device to use
        
    Returns:
        Tuple of (features, labels)
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            view_q = batch['view_q'].to(device)
            labels_batch = batch['label']
            
            # Extract features from query encoder
            h = model.encoder_q(view_q)
            h = h.view(h.size(0), -1)  # Flatten
            
            features.append(h.cpu())
            labels.extend(labels_batch)
    
    return torch.cat(features, dim=0), torch.tensor(labels)


def compute_representations(model: nn.Module, 
                          dataloader: torch.utils.data.DataLoader,
                          device: str = 'cuda') -> torch.Tensor:
    """
    Compute representations for all samples.
    
    Args:
        model: Trained MoCo model
        dataloader: Data loader
        device: Device to use
        
    Returns:
        Representations tensor
    """
    model.eval()
    representations = []
    
    with torch.no_grad():
        for batch in dataloader:
            view_q = batch['view_q'].to(device)
            
            # Get representations from query encoder
            h = model.encoder_q(view_q)
            h = h.view(h.size(0), -1)  # Flatten
            z = model.projection_q(h)
            z = F.normalize(z, dim=1)
            
            representations.append(z.cpu())
    
    return torch.cat(representations, dim=0)


def visualize_representations(representations: torch.Tensor, 
                            labels: torch.Tensor,
                            method: str = 'tsne'):
    """
    Visualize learned representations.
    
    Args:
        representations: Learned representations
        labels: Ground truth labels
        method: Visualization method ('tsne', 'pca')
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Reduce dimensionality
    representations_2d = reducer.fit_transform(representations.numpy())
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(representations_2d[:, 0], representations_2d[:, 1], 
                         c=labels.numpy(), cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f'MoCo Representations ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()


def linear_evaluation(model: nn.Module, 
                     train_loader: torch.utils.data.DataLoader,
                     test_loader: torch.utils.data.DataLoader,
                     num_classes: int,
                     device: str = 'cuda') -> Tuple[float, float]:
    """
    Linear evaluation on learned representations.
    
    Args:
        model: Trained MoCo model
        train_loader: Training data loader
        test_loader: Test data loader
        num_classes: Number of classes
        device: Device to use
        
    Returns:
        Tuple of (train_accuracy, test_accuracy)
    """
    # Extract features
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)
    
    # Train linear classifier
    linear_classifier = nn.Linear(train_features.size(1), num_classes)
    linear_classifier = linear_classifier.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(linear_classifier.parameters(), lr=0.001)
    
    # Training
    linear_classifier.train()
    for epoch in range(100):
        optimizer.zero_grad()
        logits = linear_classifier(train_features.to(device))
        loss = criterion(logits, train_labels.to(device))
        loss.backward()
        optimizer.step()
    
    # Evaluation
    linear_classifier.eval()
    with torch.no_grad():
        # Train accuracy
        train_logits = linear_classifier(train_features.to(device))
        _, train_pred = torch.max(train_logits, 1)
        train_acc = (train_pred == train_labels.to(device)).float().mean().item()
        
        # Test accuracy
        test_logits = linear_classifier(test_features.to(device))
        _, test_pred = torch.max(test_logits, 1)
        test_acc = (test_pred == test_labels.to(device)).float().mean().item()
    
    return train_acc, test_acc


# Example usage
if __name__ == "__main__":
    # Example of how to use the MoCo implementation
    
    # Create dataset
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    
    # MoCo transform
    moco_transform = MoCoTransform(image_size=32, s=1.0)
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True)
    test_dataset = CIFAR10(root='./data', train=False, download=True)
    
    # Create MoCo datasets
    train_moco_dataset = MoCoDataset(train_dataset, moco_transform)
    test_moco_dataset = MoCoDataset(test_dataset, moco_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_moco_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_moco_dataset, batch_size=128, shuffle=False)
    
    # Create model
    model = MoCoModel(encoder='resnet18', projection_dim=128, queue_size=1024, momentum=0.999)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_moco(model, train_loader, test_loader, num_epochs=50, device=device)
    
    # Evaluate model
    loss = evaluate_moco(model, test_loader, device=device)
    print(f'Test Loss: {loss:.4f}')
    
    # Extract features
    features, labels = extract_features(model, test_loader, device)
    print(f'Extracted features shape: {features.shape}')
    print(f'Labels shape: {labels.shape}')
    
    # Visualize representations
    representations = compute_representations(model, test_loader, device)
    visualize_representations(representations, labels, method='tsne')
    
    # Linear evaluation
    train_acc, test_acc = linear_evaluation(model, train_loader, test_loader, 10, device)
    print(f'Linear Evaluation - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}') 
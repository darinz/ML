"""
SimCLR: Simple Framework for Contrastive Learning of Visual Representations

This module implements SimCLR, a contrastive learning framework that learns representations
by maximizing agreement between different augmented views of the same image.

References:
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for
  contrastive learning of visual representations. ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
import random


class SimCLRTransform:
    """Data augmentation for SimCLR."""
    
    def __init__(self, image_size: int = 224, s: float = 1.0):
        """
        Initialize SimCLR transforms.
        
        Args:
            image_size: Size of the image
            s: Strength of color jittering
        """
        self.image_size = image_size
        self.s = s
        
        # Color jittering
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        
        # Data augmentation pipeline
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        """Apply two random augmentations."""
        return self.transform(x), self.transform(x)


class SimCLRModel(nn.Module):
    """SimCLR model with encoder and projection head."""
    
    def __init__(self, encoder: str = 'resnet18', projection_dim: int = 128):
        """
        Initialize SimCLR model.
        
        Args:
            encoder: Encoder architecture ('resnet18', 'resnet50', etc.)
            projection_dim: Dimension of projection head output
        """
        super().__init__()
        
        # Encoder
        if encoder == 'resnet18':
            import torchvision.models as models
            self.encoder = models.resnet18(pretrained=False)
            # Remove the final classification layer
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
            feature_dim = 512
        elif encoder == 'resnet50':
            import torchvision.models as models
            self.encoder = models.resnet50(pretrained=False)
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (batch_size, channels, height, width)
            
        Returns:
            Projected representations (batch_size, projection_dim)
        """
        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        
        # Project
        z = self.projection_head(h)
        
        # Normalize
        z = F.normalize(z, dim=1)
        
        return z


class SimCLRLoss(nn.Module):
    """NT-Xent loss for SimCLR."""
    
    def __init__(self, temperature: float = 0.5):
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.
        
        Args:
            z_i: Representations of first augmented views (batch_size, projection_dim)
            z_j: Representations of second augmented views (batch_size, projection_dim)
            
        Returns:
            Loss value
        """
        batch_size = z_i.size(0)
        
        # Concatenate representations
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, projection_dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(z, z.T) / self.temperature  # (2*batch_size, 2*batch_size)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # Positive pairs: (i, j) and (j, i)
        positives = torch.cat([
            torch.diag(similarity_matrix[:batch_size, batch_size:]),
            torch.diag(similarity_matrix[batch_size:, :batch_size])
        ])
        
        # Negative pairs: all other pairs
        negatives = similarity_matrix
        
        # Compute logits
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
        
        # Labels: first column is positive
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z.device)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class SimCLRDataset:
    """Dataset wrapper for SimCLR."""
    
    def __init__(self, dataset, transform):
        """
        Initialize SimCLR dataset.
        
        Args:
            dataset: Base dataset
            transform: SimCLR transform
        """
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get two augmented views of the same image."""
        image, label = self.dataset[idx]
        view_i, view_j = self.transform(image)
        
        return {
            'view_i': view_i,
            'view_j': view_j,
            'label': label
        }


def train_simclr(model: nn.Module, 
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 num_epochs: int = 100,
                 learning_rate: float = 0.001,
                 temperature: float = 0.5,
                 device: str = 'cuda'):
    """
    Train SimCLR model.
    
    Args:
        model: SimCLR model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        temperature: Temperature parameter
        device: Device to train on
    """
    model = model.to(device)
    criterion = SimCLRLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            view_i = batch['view_i'].to(device)
            view_j = batch['view_j'].to(device)
            
            optimizer.zero_grad()
            z_i = model(view_i)
            z_j = model(view_j)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                view_i = batch['view_i'].to(device)
                view_j = batch['view_j'].to(device)
                
                z_i = model(view_i)
                z_j = model(view_j)
                loss = criterion(z_i, z_j)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_simclr_model.pth')
        
        scheduler.step()


def evaluate_simclr(model: nn.Module, 
                   test_loader: torch.utils.data.DataLoader,
                   temperature: float = 0.5,
                   device: str = 'cuda') -> float:
    """
    Evaluate SimCLR model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        temperature: Temperature parameter
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    criterion = SimCLRLoss(temperature=temperature)
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            view_i = batch['view_i'].to(device)
            view_j = batch['view_j'].to(device)
            
            z_i = model(view_i)
            z_j = model(view_j)
            loss = criterion(z_i, z_j)
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader)
    return avg_loss


def extract_features(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader,
                    device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from the encoder for downstream tasks.
    
    Args:
        model: Trained SimCLR model
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
            view_i = batch['view_i'].to(device)
            labels_batch = batch['label']
            
            # Extract features from encoder
            h = model.encoder(view_i)
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
        model: Trained SimCLR model
        dataloader: Data loader
        device: Device to use
        
    Returns:
        Representations tensor
    """
    model.eval()
    representations = []
    
    with torch.no_grad():
        for batch in dataloader:
            view_i = batch['view_i'].to(device)
            
            # Get representations
            z = model(view_i)
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
    plt.title(f'SimCLR Representations ({method.upper()})')
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
        model: Trained SimCLR model
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
    # Example of how to use the SimCLR implementation
    
    # Create dataset
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    
    # SimCLR transform
    simclr_transform = SimCLRTransform(image_size=32, s=1.0)
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True)
    test_dataset = CIFAR10(root='./data', train=False, download=True)
    
    # Create SimCLR datasets
    train_simclr_dataset = SimCLRDataset(train_dataset, simclr_transform)
    test_simclr_dataset = SimCLRDataset(test_dataset, simclr_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_simclr_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_simclr_dataset, batch_size=128, shuffle=False)
    
    # Create model
    model = SimCLRModel(encoder='resnet18', projection_dim=128)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_simclr(model, train_loader, test_loader, num_epochs=50, device=device)
    
    # Evaluate model
    loss = evaluate_simclr(model, test_loader, device=device)
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
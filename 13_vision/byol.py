"""
BYOL: Bootstrap Your Own Latent for Self-Supervised Learning

This module implements BYOL, a self-supervised learning method that uses two networks
to predict each other's representations without negative samples.

References:
- Grill, J. B., Strub, F., Altché, F., Tallec, C., Richemond, P. H., Buchatskaya, E.,
  Doersch, C., Pires, B. A., Guo, Z. D., Azar, M. G., Piot, B., Kavukcuoglu, K.,
  Munos, R., & Valko, M. (2020). Bootstrap your own latent: A new approach to
  self-supervised learning. NeurIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
import random


class BYOLTransform:
    """Data augmentation for BYOL."""
    
    def __init__(self, image_size: int = 224, s: float = 1.0):
        """
        Initialize BYOL transforms.
        
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


class BYOLModel(nn.Module):
    """BYOL model with online and target networks."""
    
    def __init__(self, encoder: str = 'resnet18', projection_dim: int = 256, 
                 prediction_dim: int = 256, momentum: float = 0.996):
        """
        Initialize BYOL model.
        
        Args:
            encoder: Encoder architecture ('resnet18', 'resnet50', etc.)
            projection_dim: Dimension of projection head output
            prediction_dim: Dimension of prediction head output
            momentum: Momentum parameter for target network update
        """
        super().__init__()
        self.momentum = momentum
        
        # Online network
        self.online_encoder = self._build_encoder(encoder)
        self.online_projector = self._build_projector(encoder, projection_dim)
        self.online_predictor = self._build_predictor(projection_dim, prediction_dim)
        
        # Target network
        self.target_encoder = self._build_encoder(encoder)
        self.target_projector = self._build_projector(encoder, projection_dim)
        
        # Initialize target network with online network weights
        self._init_target_network()
        
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
    
    def _build_projector(self, encoder: str, projection_dim: int) -> nn.Module:
        """Build projection head."""
        if encoder == 'resnet18':
            feature_dim = 512
        elif encoder == 'resnet50':
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        
        return nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim)
        )
    
    def _build_predictor(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build prediction head."""
        return nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim)
        )
    
    def _init_target_network(self):
        """Initialize target network with online network weights."""
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
        
        for param_online, param_target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_target.data.copy_(param_online.data)
            param_target.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_target_network(self):
        """Momentum update of the target network."""
        for param_online, param_target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_target.data = param_target.data * self.momentum + param_online.data * (1. - self.momentum)
        
        for param_online, param_target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_target.data = param_target.data * self.momentum + param_online.data * (1. - self.momentum)
    
    def forward(self, view_1: torch.Tensor, view_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            view_1: First augmented view (batch_size, channels, height, width)
            view_2: Second augmented view (batch_size, channels, height, width)
            
        Returns:
            Tuple of (prediction_1, prediction_2)
        """
        # Online network for view_1
        online_1 = self.online_encoder(view_1)
        online_1 = online_1.view(online_1.size(0), -1)
        online_1 = self.online_projector(online_1)
        online_1 = F.normalize(online_1, dim=1)
        pred_1 = self.online_predictor(online_1)
        pred_1 = F.normalize(pred_1, dim=1)
        
        # Online network for view_2
        online_2 = self.online_encoder(view_2)
        online_2 = online_2.view(online_2.size(0), -1)
        online_2 = self.online_projector(online_2)
        online_2 = F.normalize(online_2, dim=1)
        pred_2 = self.online_predictor(online_2)
        pred_2 = F.normalize(pred_2, dim=1)
        
        # Target network for view_1
        with torch.no_grad():
            self._momentum_update_target_network()
            
            target_1 = self.target_encoder(view_1)
            target_1 = target_1.view(target_1.size(0), -1)
            target_1 = self.target_projector(target_1)
            target_1 = F.normalize(target_1, dim=1)
        
        # Target network for view_2
        with torch.no_grad():
            target_2 = self.target_encoder(view_2)
            target_2 = target_2.view(target_2.size(0), -1)
            target_2 = self.target_projector(target_2)
            target_2 = F.normalize(target_2, dim=1)
        
        return pred_1, pred_2, target_1, target_2


class BYOLLoss(nn.Module):
    """BYOL loss function."""
    
    def __init__(self):
        """Initialize BYOL loss."""
        super().__init__()
    
    def forward(self, pred_1: torch.Tensor, pred_2: torch.Tensor, 
                target_1: torch.Tensor, target_2: torch.Tensor) -> torch.Tensor:
        """
        Compute BYOL loss.
        
        Args:
            pred_1: Predictions for view_1
            pred_2: Predictions for view_2
            target_1: Targets for view_1
            target_2: Targets for view_2
            
        Returns:
            Loss value
        """
        # MSE loss between predictions and targets
        loss_1 = F.mse_loss(pred_1, target_2.detach())
        loss_2 = F.mse_loss(pred_2, target_1.detach())
        
        return loss_1 + loss_2


class BYOLDataset:
    """Dataset wrapper for BYOL."""
    
    def __init__(self, dataset, transform):
        """
        Initialize BYOL dataset.
        
        Args:
            dataset: Base dataset
            transform: BYOL transform
        """
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get two augmented views of the same image."""
        image, label = self.dataset[idx]
        view_1, view_2 = self.transform(image)
        
        return {
            'view_1': view_1,
            'view_2': view_2,
            'label': label
        }


def train_byol(model: nn.Module, 
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               num_epochs: int = 100,
               learning_rate: float = 0.001,
               device: str = 'cuda'):
    """
    Train BYOL model.
    
    Args:
        model: BYOL model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    criterion = BYOLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            view_1 = batch['view_1'].to(device)
            view_2 = batch['view_2'].to(device)
            
            optimizer.zero_grad()
            pred_1, pred_2, target_1, target_2 = model(view_1, view_2)
            loss = criterion(pred_1, pred_2, target_1, target_2)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                view_1 = batch['view_1'].to(device)
                view_2 = batch['view_2'].to(device)
                
                pred_1, pred_2, target_1, target_2 = model(view_1, view_2)
                loss = criterion(pred_1, pred_2, target_1, target_2)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_byol_model.pth')
        
        scheduler.step()


def evaluate_byol(model: nn.Module, 
                 test_loader: torch.utils.data.DataLoader,
                 device: str = 'cuda') -> float:
    """
    Evaluate BYOL model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    criterion = BYOLLoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            view_1 = batch['view_1'].to(device)
            view_2 = batch['view_2'].to(device)
            
            pred_1, pred_2, target_1, target_2 = model(view_1, view_2)
            loss = criterion(pred_1, pred_2, target_1, target_2)
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader)
    return avg_loss


def extract_features(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader,
                    device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from the online encoder for downstream tasks.
    
    Args:
        model: Trained BYOL model
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
            view_1 = batch['view_1'].to(device)
            labels_batch = batch['label']
            
            # Extract features from online encoder
            h = model.online_encoder(view_1)
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
        model: Trained BYOL model
        dataloader: Data loader
        device: Device to use
        
    Returns:
        Representations tensor
    """
    model.eval()
    representations = []
    
    with torch.no_grad():
        for batch in dataloader:
            view_1 = batch['view_1'].to(device)
            
            # Get representations from online encoder
            h = model.online_encoder(view_1)
            h = h.view(h.size(0), -1)  # Flatten
            z = model.online_projector(h)
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
    plt.title(f'BYOL Representations ({method.upper()})')
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
        model: Trained BYOL model
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
    # Example of how to use the BYOL implementation
    
    # Create dataset
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    
    # BYOL transform
    byol_transform = BYOLTransform(image_size=32, s=1.0)
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True)
    test_dataset = CIFAR10(root='./data', train=False, download=True)
    
    # Create BYOL datasets
    train_byol_dataset = BYOLDataset(train_dataset, byol_transform)
    test_byol_dataset = BYOLDataset(test_dataset, byol_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_byol_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_byol_dataset, batch_size=128, shuffle=False)
    
    # Create model
    model = BYOLModel(encoder='resnet18', projection_dim=256, prediction_dim=256, momentum=0.996)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_byol(model, train_loader, test_loader, num_epochs=50, device=device)
    
    # Evaluate model
    loss = evaluate_byol(model, test_loader, device=device)
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
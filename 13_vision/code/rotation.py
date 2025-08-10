"""
Rotation Prediction for Self-Supervised Learning

This module implements rotation prediction as a pretext task for self-supervised learning.
The model learns to predict the rotation angle applied to an image, helping it understand
spatial relationships and object orientation.

References:
- Gidaris, S., Singh, P., & Komodakis, N. (2018). Unsupervised representation learning
  by predicting image rotations. ICLR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
import random


class RotationDataset:
    """Dataset wrapper that creates rotated images for rotation prediction."""
    
    def __init__(self, dataset, rotation_angles: List[int] = [0, 90, 180, 270]):
        """
        Initialize rotation dataset.
        
        Args:
            dataset: Base dataset (e.g., ImageFolder)
            rotation_angles: List of rotation angles in degrees
        """
        self.dataset = dataset
        self.rotation_angles = rotation_angles
        self.num_classes = len(rotation_angles)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a rotated image sample."""
        image, label = self.dataset[idx]
        
        # Randomly select rotation angle
        rotation_idx = random.randint(0, len(self.rotation_angles) - 1)
        rotation_angle = self.rotation_angles[rotation_idx]
        
        # Apply rotation
        rotated_image = self._rotate_image(image, rotation_angle)
        
        return {
            'image': rotated_image,
            'rotation_label': rotation_idx,
            'rotation_angle': rotation_angle,
            'original_label': label
        }
    
    def _rotate_image(self, image: torch.Tensor, angle: int) -> torch.Tensor:
        """Rotate image by given angle."""
        if angle == 0:
            return image
        elif angle == 90:
            return image.transpose(1, 2).flip(1)
        elif angle == 180:
            return image.flip(1, 2)
        elif angle == 270:
            return image.transpose(1, 2).flip(2)
        else:
            raise ValueError(f"Unsupported rotation angle: {angle}")


class RotationPredictionModel(nn.Module):
    """Model for predicting image rotation."""
    
    def __init__(self, num_classes: int = 4, backbone: str = 'resnet18'):
        """
        Initialize rotation prediction model.
        
        Args:
            num_classes: Number of rotation classes (typically 4 for 0, 90, 180, 270)
            backbone: Backbone architecture ('resnet18', 'resnet50', etc.)
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Load backbone
        if backbone == 'resnet18':
            import torchvision.models as models
            self.backbone = models.resnet18(pretrained=False)
            # Remove the final classification layer
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 512
        elif backbone == 'resnet50':
            import torchvision.models as models
            self.backbone = models.resnet50(pretrained=False)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Rotation classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (batch_size, channels, height, width)
            
        Returns:
            Logits for rotation classification
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


class RotationLoss(nn.Module):
    """Loss function for rotation prediction."""
    
    def __init__(self):
        """Initialize loss function."""
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Target rotation labels (batch_size,)
            
        Returns:
            Loss value
        """
        return self.cross_entropy(logits, targets)


def train_rotation_model(model: nn.Module, 
                        train_loader: torch.utils.data.DataLoader,
                        val_loader: torch.utils.data.DataLoader,
                        num_epochs: int = 100,
                        learning_rate: float = 0.001,
                        device: str = 'cuda'):
    """
    Train rotation prediction model.
    
    Args:
        model: Rotation prediction model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    criterion = RotationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            targets = batch['rotation_label'].to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = batch['rotation_label'].to(device)
                
                logits = model(images)
                loss = criterion(logits, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_rotation_model.pth')
        
        scheduler.step()


def evaluate_rotation_model(model: nn.Module, 
                          test_loader: torch.utils.data.DataLoader,
                          device: str = 'cuda') -> Tuple[float, float]:
    """
    Evaluate rotation prediction model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    criterion = RotationLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            targets = batch['rotation_label'].to(device)
            
            logits = model(images)
            loss = criterion(logits, targets)
            
            test_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            test_total += targets.size(0)
            test_correct += (predicted == targets).sum().item()
    
    accuracy = 100 * test_correct / test_total
    avg_loss = test_loss / len(test_loader)
    
    return accuracy, avg_loss


def extract_features(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader,
                    device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from the backbone for downstream tasks.
    
    Args:
        model: Trained rotation model
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
            images = batch['image'].to(device)
            original_labels = batch['original_label']
            
            # Extract features from backbone
            backbone_features = model.backbone(images)
            feature_vectors = model.classifier[:-1](backbone_features)  # Remove final classification layer
            
            features.append(feature_vectors.cpu())
            labels.extend(original_labels)
    
    return torch.cat(features, dim=0), torch.tensor(labels)


def visualize_rotation_predictions(model: nn.Module, 
                                 dataloader: torch.utils.data.DataLoader,
                                 device: str = 'cuda',
                                 num_samples: int = 8):
    """
    Visualize rotation predictions.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    images_list = []
    predictions_list = []
    targets_list = []
    angles_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(images_list) >= num_samples:
                break
                
            images = batch['image'].to(device)
            targets = batch['rotation_label']
            angles = batch['rotation_angle']
            
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            
            for i in range(min(len(images), num_samples - len(images_list))):
                images_list.append(images[i].cpu())
                predictions_list.append(predicted[i].cpu())
                targets_list.append(targets[i])
                angles_list.append(angles[i])
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    rotation_angles = [0, 90, 180, 270]
    
    for i in range(min(num_samples, len(images_list))):
        img = images_list[i].permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]
        
        pred_angle = rotation_angles[predictions_list[i]]
        true_angle = angles_list[i]
        
        axes[i].imshow(img)
        axes[i].set_title(f'Pred: {pred_angle}°, True: {true_angle}°')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_rotation_performance(model: nn.Module, 
                               dataloader: torch.utils.data.DataLoader,
                               device: str = 'cuda') -> dict:
    """
    Analyze rotation prediction performance by angle.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use
        
    Returns:
        Dictionary with performance metrics by angle
    """
    model.eval()
    angle_correct = {0: 0, 90: 0, 180: 0, 270: 0}
    angle_total = {0: 0, 90: 0, 180: 0, 270: 0}
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            targets = batch['rotation_label']
            angles = batch['rotation_angle']
            
            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            
            for i in range(len(images)):
                angle = angles[i].item()
                target = targets[i].item()
                pred = predicted[i].item()
                
                angle_total[angle] += 1
                if target == pred:
                    angle_correct[angle] += 1
    
    # Calculate accuracy by angle
    results = {}
    for angle in [0, 90, 180, 270]:
        if angle_total[angle] > 0:
            accuracy = 100 * angle_correct[angle] / angle_total[angle]
            results[angle] = {
                'accuracy': accuracy,
                'correct': angle_correct[angle],
                'total': angle_total[angle]
            }
    
    return results


# Example usage
if __name__ == "__main__":
    # Example of how to use the rotation prediction implementation
    
    # Create dataset
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create rotation datasets
    train_rotation_dataset = RotationDataset(train_dataset)
    test_rotation_dataset = RotationDataset(test_dataset)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_rotation_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_rotation_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = RotationPredictionModel(num_classes=4, backbone='resnet18')
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_rotation_model(model, train_loader, test_loader, num_epochs=50, device=device)
    
    # Evaluate model
    accuracy, loss = evaluate_rotation_model(model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Loss: {loss:.4f}')
    
    # Analyze performance by angle
    angle_results = analyze_rotation_performance(model, test_loader, device)
    print("\nPerformance by rotation angle:")
    for angle, metrics in angle_results.items():
        print(f"{angle}°: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")
    
    # Extract features for downstream tasks
    features, labels = extract_features(model, test_loader, device)
    print(f'Extracted features shape: {features.shape}')
    print(f'Labels shape: {labels.shape}')
    
    # Visualize predictions
    visualize_rotation_predictions(model, test_loader, device) 
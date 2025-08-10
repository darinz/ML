"""
DINO: Self-Distillation with No Labels

This module implements DINO, a self-supervised learning method that uses knowledge
distillation without labels to learn visual representations.

References:
- Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., &
  Joulin, A. (2021). Emerging properties in self-supervised vision transformers.
  ICCV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
import random


class DINOTransform:
    """Data augmentation for DINO with multi-crop strategy."""
    
    def __init__(self, image_size: int = 224, global_crops_scale: Tuple[float, float] = (0.4, 1.0),
                 local_crops_scale: Tuple[float, float] = (0.05, 0.4), local_crops_number: int = 8):
        """
        Initialize DINO transforms.
        
        Args:
            image_size: Size of the image
            global_crops_scale: Scale range for global crops
            local_crops_scale: Scale range for local crops
            local_crops_number: Number of local crops
        """
        self.image_size = image_size
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        
        # Color jittering
        color_jitter = transforms.ColorJitter(
            0.4, 0.4, 0.4, 0.1
        )
        
        # Global transforms
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Local transforms
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        """Apply multi-crop augmentation."""
        crops = []
        crops.append(self.global_transform_1(x))
        crops.append(self.global_transform_2(x))
        
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(x))
        
        return crops


class DINOModel(nn.Module):
    """DINO model with student and teacher networks."""
    
    def __init__(self, encoder: str = 'resnet18', projection_dim: int = 256, 
                 momentum: float = 0.996, center_momentum: float = 0.9):
        """
        Initialize DINO model.
        
        Args:
            encoder: Encoder architecture ('resnet18', 'resnet50', etc.)
            projection_dim: Dimension of projection head output
            momentum: Momentum parameter for teacher network update
            center_momentum: Momentum parameter for center update
        """
        super().__init__()
        self.momentum = momentum
        self.center_momentum = center_momentum
        
        # Student network
        self.student_encoder = self._build_encoder(encoder)
        self.student_head = self._build_head(encoder, projection_dim)
        
        # Teacher network
        self.teacher_encoder = self._build_encoder(encoder)
        self.teacher_head = self._build_head(encoder, projection_dim)
        
        # Initialize teacher network with student network weights
        self._init_teacher_network()
        
        # Center for teacher outputs
        self.register_buffer("center", torch.zeros(1, projection_dim))
        
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
    
    def _build_head(self, encoder: str, projection_dim: int) -> nn.Module:
        """Build projection head."""
        if encoder == 'resnet18':
            feature_dim = 512
        elif encoder == 'resnet50':
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported encoder: {encoder}")
        
        return nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, projection_dim)
        )
    
    def _init_teacher_network(self):
        """Initialize teacher network with student network weights."""
        for param_student, param_teacher in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            param_teacher.data.copy_(param_student.data)
            param_teacher.requires_grad = False
        
        for param_student, param_teacher in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_teacher.data.copy_(param_student.data)
            param_teacher.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update_teacher_network(self):
        """Momentum update of the teacher network."""
        for param_student, param_teacher in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + param_student.data * (1. - self.momentum)
        
        for param_student, param_teacher in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + param_student.data * (1. - self.momentum)
    
    def forward(self, crops: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            crops: List of augmented crops
            
        Returns:
            Tuple of (student_outputs, teacher_outputs)
        """
        # Student network
        student_outputs = []
        for crop in crops:
            h = self.student_encoder(crop)
            h = h.view(h.size(0), -1)
            z = self.student_head(h)
            student_outputs.append(z)
        
        # Teacher network (only for global crops)
        teacher_outputs = []
        with torch.no_grad():
            self._momentum_update_teacher_network()
            
            for i in range(2):  # Only global crops
                h = self.teacher_encoder(crops[i])
                h = h.view(h.size(0), -1)
                z = self.teacher_head(h)
                teacher_outputs.append(z)
        
        return torch.stack(student_outputs), torch.stack(teacher_outputs)


class DINOLoss(nn.Module):
    """DINO loss function."""
    
    def __init__(self, temperature: float = 0.1, center_momentum: float = 0.9):
        """
        Initialize DINO loss.
        
        Args:
            temperature: Temperature parameter for softmax
            center_momentum: Momentum parameter for center update
        """
        super().__init__()
        self.temperature = temperature
        self.center_momentum = center_momentum
    
    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor, 
                center: torch.Tensor) -> torch.Tensor:
        """
        Compute DINO loss.
        
        Args:
            student_output: Student network output
            teacher_output: Teacher network output
            center: Center for teacher outputs
            
        Returns:
            Loss value
        """
        # Center and sharpen teacher output
        teacher_output = teacher_output - center
        teacher_output = F.softmax(teacher_output / self.temperature, dim=-1)
        
        # Student output
        student_output = F.log_softmax(student_output / self.temperature, dim=-1)
        
        # KL divergence loss
        loss = F.kl_div(student_output, teacher_output, reduction='batchmean')
        
        return loss


class DINODataset:
    """Dataset wrapper for DINO."""
    
    def __init__(self, dataset, transform):
        """
        Initialize DINO dataset.
        
        Args:
            dataset: Base dataset
            transform: DINO transform
        """
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get multi-crop augmented views of the same image."""
        image, label = self.dataset[idx]
        crops = self.transform(image)
        
        return {
            'crops': crops,
            'label': label
        }


def train_dino(model: nn.Module, 
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               num_epochs: int = 100,
               learning_rate: float = 0.001,
               temperature: float = 0.1,
               device: str = 'cuda'):
    """
    Train DINO model.
    
    Args:
        model: DINO model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        temperature: Temperature parameter
        device: Device to train on
    """
    model = model.to(device)
    criterion = DINOLoss(temperature=temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            crops = [crop.to(device) for crop in batch['crops']]
            
            optimizer.zero_grad()
            student_outputs, teacher_outputs = model(crops)
            
            # Compute loss for each global crop
            loss = 0
            for i in range(2):  # Global crops
                for j in range(2):  # Global crops
                    if i != j:
                        loss += criterion(student_outputs[i], teacher_outputs[j], model.center)
            
            loss = loss / 2  # Average over pairs
            loss.backward()
            optimizer.step()
            
            # Update center
            with torch.no_grad():
                teacher_outputs_mean = torch.stack(teacher_outputs).mean(0)
                model.center = model.center * model.center_momentum + teacher_outputs_mean * (1 - model.center_momentum)
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                crops = [crop.to(device) for crop in batch['crops']]
                
                student_outputs, teacher_outputs = model(crops)
                
                # Compute loss for each global crop
                loss = 0
                for i in range(2):  # Global crops
                    for j in range(2):  # Global crops
                        if i != j:
                            loss += criterion(student_outputs[i], teacher_outputs[j], model.center)
                
                loss = loss / 2  # Average over pairs
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_dino_model.pth')
        
        scheduler.step()


def evaluate_dino(model: nn.Module, 
                 test_loader: torch.utils.data.DataLoader,
                 temperature: float = 0.1,
                 device: str = 'cuda') -> float:
    """
    Evaluate DINO model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        temperature: Temperature parameter
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    criterion = DINOLoss(temperature=temperature)
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            crops = [crop.to(device) for crop in batch['crops']]
            
            student_outputs, teacher_outputs = model(crops)
            
            # Compute loss for each global crop
            loss = 0
            for i in range(2):  # Global crops
                for j in range(2):  # Global crops
                    if i != j:
                        loss += criterion(student_outputs[i], teacher_outputs[j], model.center)
            
            loss = loss / 2  # Average over pairs
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader)
    return avg_loss


def extract_features(model: nn.Module, 
                    dataloader: torch.utils.data.DataLoader,
                    device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from the student encoder for downstream tasks.
    
    Args:
        model: Trained DINO model
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
            crops = [crop.to(device) for crop in batch['crops']]
            labels_batch = batch['label']
            
            # Use first crop for feature extraction
            h = model.student_encoder(crops[0])
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
        model: Trained DINO model
        dataloader: Data loader
        device: Device to use
        
    Returns:
        Representations tensor
    """
    model.eval()
    representations = []
    
    with torch.no_grad():
        for batch in dataloader:
            crops = [crop.to(device) for crop in batch['crops']]
            
            # Use first crop for representation
            h = model.student_encoder(crops[0])
            h = h.view(h.size(0), -1)  # Flatten
            z = model.student_head(h)
            
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
    plt.title(f'DINO Representations ({method.upper()})')
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
        model: Trained DINO model
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
    # Example of how to use the DINO implementation
    
    # Create dataset
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    
    # DINO transform
    dino_transform = DINOTransform(image_size=32, global_crops_scale=(0.4, 1.0),
                                  local_crops_scale=(0.05, 0.4), local_crops_number=8)
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True)
    test_dataset = CIFAR10(root='./data', train=False, download=True)
    
    # Create DINO datasets
    train_dino_dataset = DINODataset(train_dataset, dino_transform)
    test_dino_dataset = DINODataset(test_dataset, dino_transform)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dino_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dino_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = DINOModel(encoder='resnet18', projection_dim=256, momentum=0.996, center_momentum=0.9)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dino(model, train_loader, test_loader, num_epochs=50, device=device)
    
    # Evaluate model
    loss = evaluate_dino(model, test_loader, device=device)
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
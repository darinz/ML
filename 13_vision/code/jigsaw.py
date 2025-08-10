"""
Jigsaw Puzzle Solving for Self-Supervised Learning

This module implements jigsaw puzzle solving as a pretext task for self-supervised learning.
The model learns to reconstruct the original image from shuffled patches, helping it
understand spatial relationships and object structure.

References:
- Noroozi, M., & Favaro, P. (2016). Unsupervised learning of visual representations
  by solving jigsaw puzzles. ECCV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
import random


class JigsawPuzzleDataset:
    """Dataset wrapper that creates jigsaw puzzles from images."""
    
    def __init__(self, dataset, grid_size: int = 3, patch_size: int = 64):
        """
        Initialize jigsaw puzzle dataset.
        
        Args:
            dataset: Base dataset (e.g., ImageFolder)
            grid_size: Size of the grid (e.g., 3 for 3x3 grid)
            patch_size: Size of each patch in pixels
        """
        self.dataset = dataset
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches = grid_size ** 2
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a jigsaw puzzle sample."""
        image, label = self.dataset[idx]
        
        # Create patches
        patches = self._extract_patches(image)
        
        # Create permutation
        permutation = list(range(self.num_patches))
        random.shuffle(permutation)
        
        # Shuffle patches
        shuffled_patches = [patches[i] for i in permutation]
        
        return {
            'shuffled_patches': torch.stack(shuffled_patches),
            'permutation': torch.tensor(permutation, dtype=torch.long),
            'original_label': label
        }
    
    def _extract_patches(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Extract patches from image."""
        patches = []
        h, w = image.shape[-2:]
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                h_start = i * self.patch_size
                h_end = h_start + self.patch_size
                w_start = j * self.patch_size
                w_end = w_start + self.patch_size
                
                patch = image[:, h_start:h_end, w_start:w_end]
                patches.append(patch)
        
        return patches


class JigsawPuzzleModel(nn.Module):
    """Model for solving jigsaw puzzles."""
    
    def __init__(self, num_classes: int = 1000, grid_size: int = 3, 
                 patch_size: int = 64, embedding_dim: int = 512):
        """
        Initialize jigsaw puzzle model.
        
        Args:
            num_classes: Number of permutation classes
            grid_size: Size of the grid
            patch_size: Size of each patch
            embedding_dim: Dimension of patch embeddings
        """
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.num_patches = grid_size ** 2
        
        # Patch encoder (CNN for each patch)
        self.patch_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim)
        )
        
        # Permutation classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * self.num_patches, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, shuffled_patches: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            shuffled_patches: Tensor of shape (batch_size, num_patches, channels, height, width)
            
        Returns:
            Logits for permutation classification
        """
        batch_size = shuffled_patches.shape[0]
        
        # Encode each patch
        patch_embeddings = []
        for i in range(self.num_patches):
            patch = shuffled_patches[:, i]  # (batch_size, channels, height, width)
            embedding = self.patch_encoder(patch)  # (batch_size, embedding_dim)
            patch_embeddings.append(embedding)
        
        # Concatenate all patch embeddings
        all_embeddings = torch.cat(patch_embeddings, dim=1)  # (batch_size, num_patches * embedding_dim)
        
        # Classify permutation
        logits = self.classifier(all_embeddings)
        
        return logits


class JigsawPuzzleLoss(nn.Module):
    """Loss function for jigsaw puzzle solving."""
    
    def __init__(self, num_classes: int = 1000):
        """
        Initialize loss function.
        
        Args:
            num_classes: Number of permutation classes
        """
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.num_classes = num_classes
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            logits: Model predictions (batch_size, num_classes)
            targets: Target permutations (batch_size,)
            
        Returns:
            Loss value
        """
        return self.cross_entropy(logits, targets)


def generate_permutations(grid_size: int = 3, num_permutations: int = 1000) -> List[List[int]]:
    """
    Generate a set of permutations for jigsaw puzzle solving.
    
    Args:
        grid_size: Size of the grid
        num_permutations: Number of permutations to generate
        
    Returns:
        List of permutations
    """
    num_patches = grid_size ** 2
    permutations = []
    
    # Identity permutation (original order)
    permutations.append(list(range(num_patches)))
    
    # Generate random permutations
    for _ in range(num_permutations - 1):
        perm = list(range(num_patches))
        random.shuffle(perm)
        if perm not in permutations:
            permutations.append(perm)
    
    return permutations


def train_jigsaw_puzzle(model: nn.Module, 
                       train_loader: torch.utils.data.DataLoader,
                       val_loader: torch.utils.data.DataLoader,
                       num_epochs: int = 100,
                       learning_rate: float = 0.001,
                       device: str = 'cuda'):
    """
    Train jigsaw puzzle model.
    
    Args:
        model: Jigsaw puzzle model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    criterion = JigsawPuzzleLoss()
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
            shuffled_patches = batch['shuffled_patches'].to(device)
            targets = batch['permutation'].to(device)
            
            optimizer.zero_grad()
            logits = model(shuffled_patches)
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
                shuffled_patches = batch['shuffled_patches'].to(device)
                targets = batch['permutation'].to(device)
                
                logits = model(shuffled_patches)
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
            torch.save(model.state_dict(), 'best_jigsaw_model.pth')
        
        scheduler.step()


def evaluate_jigsaw_model(model: nn.Module, 
                         test_loader: torch.utils.data.DataLoader,
                         device: str = 'cuda') -> Tuple[float, float]:
    """
    Evaluate jigsaw puzzle model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    criterion = JigsawPuzzleLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            shuffled_patches = batch['shuffled_patches'].to(device)
            targets = batch['permutation'].to(device)
            
            logits = model(shuffled_patches)
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
    Extract features from the patch encoder for downstream tasks.
    
    Args:
        model: Trained jigsaw model
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
            shuffled_patches = batch['shuffled_patches'].to(device)
            original_labels = batch['original_label']
            
            # Extract features from patch encoder
            batch_size = shuffled_patches.shape[0]
            patch_embeddings = []
            
            for i in range(model.num_patches):
                patch = shuffled_patches[:, i]
                embedding = model.patch_encoder(patch)
                patch_embeddings.append(embedding)
            
            # Use mean of patch embeddings as image representation
            image_features = torch.stack(patch_embeddings, dim=1).mean(dim=1)
            
            features.append(image_features.cpu())
            labels.extend(original_labels)
    
    return torch.cat(features, dim=0), torch.tensor(labels)


# Example usage
if __name__ == "__main__":
    # Example of how to use the jigsaw puzzle implementation
    
    # Create dataset
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((96, 96)),  # Resize for 3x3 grid with 32x32 patches
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create jigsaw puzzle datasets
    train_jigsaw_dataset = JigsawPuzzleDataset(train_dataset, grid_size=3, patch_size=32)
    test_jigsaw_dataset = JigsawPuzzleDataset(test_dataset, grid_size=3, patch_size=32)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_jigsaw_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_jigsaw_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = JigsawPuzzleModel(num_classes=1000, grid_size=3, patch_size=32)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_jigsaw_puzzle(model, train_loader, test_loader, num_epochs=50, device=device)
    
    # Evaluate model
    accuracy, loss = evaluate_jigsaw_model(model, test_loader, device)
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Loss: {loss:.4f}')
    
    # Extract features for downstream tasks
    features, labels = extract_features(model, test_loader, device)
    print(f'Extracted features shape: {features.shape}')
    print(f'Labels shape: {labels.shape}') 
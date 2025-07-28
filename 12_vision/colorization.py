"""
Image Colorization for Self-Supervised Learning

This module implements image colorization as a pretext task for self-supervised learning.
The model learns to predict color from grayscale images, helping it understand color
relationships and object appearance.

References:
- Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful image colorization. ECCV.
- Larsson, G., Maire, M., & Shakhnarovich, G. (2016). Learning representations for
  automatic colorization. ECCV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional
import random
import colorsys


class ColorizationDataset:
    """Dataset wrapper that creates grayscale-color pairs for colorization."""
    
    def __init__(self, dataset, color_space: str = 'lab'):
        """
        Initialize colorization dataset.
        
        Args:
            dataset: Base dataset (e.g., ImageFolder)
            color_space: Color space to use ('lab', 'hsv', 'rgb')
        """
        self.dataset = dataset
        self.color_space = color_space
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a colorization sample."""
        image, label = self.dataset[idx]
        
        # Convert to grayscale and color channels
        grayscale, color_channels = self._split_channels(image)
        
        return {
            'grayscale': grayscale,
            'color_channels': color_channels,
            'original_image': image,
            'original_label': label
        }
    
    def _split_channels(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split image into grayscale and color channels."""
        if self.color_space == 'lab':
            return self._split_lab_channels(image)
        elif self.color_space == 'hsv':
            return self._split_hsv_channels(image)
        elif self.color_space == 'rgb':
            return self._split_rgb_channels(image)
        else:
            raise ValueError(f"Unsupported color space: {self.color_space}")
    
    def _split_lab_channels(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split LAB image into L (grayscale) and AB (color) channels."""
        # Convert RGB to LAB
        lab_image = self._rgb_to_lab(image)
        
        # L channel (grayscale)
        grayscale = lab_image[0:1]  # L channel
        
        # AB channels (color)
        color_channels = lab_image[1:]  # A and B channels
        
        return grayscale, color_channels
    
    def _split_hsv_channels(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split HSV image into V (grayscale) and HS (color) channels."""
        # Convert RGB to HSV
        hsv_image = self._rgb_to_hsv(image)
        
        # V channel (grayscale)
        grayscale = hsv_image[2:3]  # V channel
        
        # HS channels (color)
        color_channels = hsv_image[:2]  # H and S channels
        
        return grayscale, color_channels
    
    def _split_rgb_channels(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split RGB image into grayscale and color channels."""
        # Convert to grayscale
        grayscale = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        grayscale = grayscale.unsqueeze(0)
        
        # Color channels (use all RGB channels)
        color_channels = image
        
        return grayscale, color_channels
    
    def _rgb_to_lab(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Convert RGB to LAB color space."""
        # Simple RGB to LAB conversion (approximate)
        # In practice, you might want to use a more accurate conversion
        r, g, b = rgb_image[0], rgb_image[1], rgb_image[2]
        
        # Convert to XYZ
        x = 0.4124 * r + 0.3576 * g + 0.1805 * b
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        z = 0.0193 * r + 0.1192 * g + 0.9505 * b
        
        # Convert to LAB
        x = x / 0.95047
        y = y / 1.00000
        z = z / 1.08883
        
        x = torch.where(x > 0.008856, torch.pow(x, 1/3), (7.787 * x) + (16 / 116))
        y = torch.where(y > 0.008856, torch.pow(y, 1/3), (7.787 * y) + (16 / 116))
        z = torch.where(z > 0.008856, torch.pow(z, 1/3), (7.787 * z) + (16 / 116))
        
        l = (116 * y) - 16
        a = 500 * (x - y)
        b = 200 * (y - z)
        
        return torch.stack([l, a, b])
    
    def _rgb_to_hsv(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """Convert RGB to HSV color space."""
        r, g, b = rgb_image[0], rgb_image[1], rgb_image[2]
        
        cmax, cmax_idx = torch.max(rgb_image, dim=0)
        cmin = torch.min(rgb_image, dim=0)[0]
        diff = cmax - cmin
        
        h = torch.zeros_like(cmax)
        s = torch.zeros_like(cmax)
        v = cmax
        
        # Calculate hue
        for i in range(3):
            mask = (cmax_idx == i) & (diff != 0)
            if i == 0:  # Red
                h[mask] = (60 * ((g[mask] - b[mask]) / diff[mask]) % 360) / 360
            elif i == 1:  # Green
                h[mask] = (60 * (((b[mask] - r[mask]) / diff[mask]) + 2) % 360) / 360
            else:  # Blue
                h[mask] = (60 * (((r[mask] - g[mask]) / diff[mask]) + 4) % 360) / 360
        
        # Calculate saturation
        s = torch.where(cmax != 0, diff / cmax, torch.zeros_like(diff))
        
        return torch.stack([h, s, v])


class ColorizationModel(nn.Module):
    """Model for image colorization."""
    
    def __init__(self, color_space: str = 'lab', num_color_channels: int = 2):
        """
        Initialize colorization model.
        
        Args:
            color_space: Color space to use
            num_color_channels: Number of color channels to predict
        """
        super().__init__()
        self.color_space = color_space
        self.num_color_channels = num_color_channels
        
        # Encoder (processes grayscale input)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder (generates color channels)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, num_color_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def forward(self, grayscale: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            grayscale: Grayscale input (batch_size, 1, height, width)
            
        Returns:
            Predicted color channels (batch_size, num_color_channels, height, width)
        """
        features = self.encoder(grayscale)
        color_channels = self.decoder(features)
        return color_channels


class ColorizationLoss(nn.Module):
    """Loss function for colorization."""
    
    def __init__(self, loss_type: str = 'l2'):
        """
        Initialize loss function.
        
        Args:
            loss_type: Type of loss ('l2', 'l1', 'smooth_l1')
        """
        super().__init__()
        self.loss_type = loss_type
        
        if loss_type == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            predicted: Predicted color channels
            target: Target color channels
            
        Returns:
            Loss value
        """
        return self.loss_fn(predicted, target)


def reconstruct_image(grayscale: torch.Tensor, 
                     color_channels: torch.Tensor, 
                     color_space: str = 'lab') -> torch.Tensor:
    """
    Reconstruct full color image from grayscale and color channels.
    
    Args:
        grayscale: Grayscale channel
        color_channels: Predicted color channels
        color_space: Color space used
        
    Returns:
        Reconstructed RGB image
    """
    if color_space == 'lab':
        return _reconstruct_lab(grayscale, color_channels)
    elif color_space == 'hsv':
        return _reconstruct_hsv(grayscale, color_channels)
    elif color_space == 'rgb':
        return _reconstruct_rgb(grayscale, color_channels)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")


def _reconstruct_lab(grayscale: torch.Tensor, color_channels: torch.Tensor) -> torch.Tensor:
    """Reconstruct RGB from LAB channels."""
    # Combine L and AB channels
    lab_image = torch.cat([grayscale, color_channels], dim=1)
    
    # Convert LAB to RGB (simplified)
    l, a, b = lab_image[:, 0], lab_image[:, 1], lab_image[:, 2]
    
    # Convert to XYZ
    y = (l + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    
    x = torch.where(x > 0.206897, x**3, (x - 16/116) / 7.787)
    y = torch.where(y > 0.206897, y**3, (y - 16/116) / 7.787)
    z = torch.where(z > 0.206897, z**3, (z - 16/116) / 7.787)
    
    x = x * 0.95047
    y = y * 1.00000
    z = z * 1.08883
    
    # Convert to RGB
    r = 3.2406 * x - 1.5372 * y - 0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b = 0.0557 * x - 0.2040 * y + 1.0570 * z
    
    rgb = torch.stack([r, g, b], dim=1)
    rgb = torch.clamp(rgb, 0, 1)
    
    return rgb


def _reconstruct_hsv(grayscale: torch.Tensor, color_channels: torch.Tensor) -> torch.Tensor:
    """Reconstruct RGB from HSV channels."""
    # Combine V and HS channels
    hsv_image = torch.cat([color_channels, grayscale], dim=1)
    
    # Convert HSV to RGB
    h, s, v = hsv_image[:, 0], hsv_image[:, 1], hsv_image[:, 2]
    
    # Convert to RGB (simplified)
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c
    
    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)
    
    # Assign RGB values based on hue
    for i in range(6):
        mask = (h * 6 >= i) & (h * 6 < i + 1)
        if i == 0:
            r[mask] = c[mask]
            g[mask] = x[mask]
            b[mask] = 0
        elif i == 1:
            r[mask] = x[mask]
            g[mask] = c[mask]
            b[mask] = 0
        elif i == 2:
            r[mask] = 0
            g[mask] = c[mask]
            b[mask] = x[mask]
        elif i == 3:
            r[mask] = 0
            g[mask] = x[mask]
            b[mask] = c[mask]
        elif i == 4:
            r[mask] = x[mask]
            g[mask] = 0
            b[mask] = c[mask]
        else:
            r[mask] = c[mask]
            g[mask] = 0
            b[mask] = x[mask]
    
    rgb = torch.stack([r + m, g + m, b + m], dim=1)
    rgb = torch.clamp(rgb, 0, 1)
    
    return rgb


def _reconstruct_rgb(grayscale: torch.Tensor, color_channels: torch.Tensor) -> torch.Tensor:
    """Reconstruct RGB from grayscale and color channels."""
    # Simple reconstruction: use color channels directly
    return color_channels


def train_colorization_model(model: nn.Module, 
                           train_loader: torch.utils.data.DataLoader,
                           val_loader: torch.utils.data.DataLoader,
                           num_epochs: int = 100,
                           learning_rate: float = 0.001,
                           device: str = 'cuda'):
    """
    Train colorization model.
    
    Args:
        model: Colorization model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
    """
    model = model.to(device)
    criterion = ColorizationLoss(loss_type='l2')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            grayscale = batch['grayscale'].to(device)
            color_channels = batch['color_channels'].to(device)
            
            optimizer.zero_grad()
            predicted = model(grayscale)
            loss = criterion(predicted, color_channels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                grayscale = batch['grayscale'].to(device)
                color_channels = batch['color_channels'].to(device)
                
                predicted = model(grayscale)
                loss = criterion(predicted, color_channels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_colorization_model.pth')
        
        scheduler.step()


def evaluate_colorization_model(model: nn.Module, 
                              test_loader: torch.utils.data.DataLoader,
                              device: str = 'cuda') -> float:
    """
    Evaluate colorization model.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Average loss
    """
    model.eval()
    criterion = ColorizationLoss(loss_type='l2')
    test_loss = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            grayscale = batch['grayscale'].to(device)
            color_channels = batch['color_channels'].to(device)
            
            predicted = model(grayscale)
            loss = criterion(predicted, color_channels)
            test_loss += loss.item()
    
    avg_loss = test_loss / len(test_loader)
    return avg_loss


def visualize_colorization_results(model: nn.Module, 
                                 dataloader: torch.utils.data.DataLoader,
                                 device: str = 'cuda',
                                 num_samples: int = 8):
    """
    Visualize colorization results.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to use
        num_samples: Number of samples to visualize
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    grayscale_list = []
    original_list = []
    predicted_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            if len(grayscale_list) >= num_samples:
                break
                
            grayscale = batch['grayscale'].to(device)
            color_channels = batch['color_channels']
            original_image = batch['original_image']
            
            predicted = model(grayscale)
            
            for i in range(min(len(grayscale), num_samples - len(grayscale_list))):
                grayscale_list.append(grayscale[i].cpu())
                original_list.append(original_image[i])
                predicted_list.append(predicted[i].cpu())
    
    # Create visualization
    fig, axes = plt.subplots(3, min(num_samples, len(grayscale_list)), figsize=(16, 12))
    
    for i in range(min(num_samples, len(grayscale_list))):
        # Grayscale input
        gray_img = grayscale_list[i].squeeze()
        axes[0, i].imshow(gray_img, cmap='gray')
        axes[0, i].set_title('Grayscale Input')
        axes[0, i].axis('off')
        
        # Original image
        orig_img = original_list[i].permute(1, 2, 0)
        orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
        axes[1, i].imshow(orig_img)
        axes[1, i].set_title('Original')
        axes[1, i].axis('off')
        
        # Predicted colorization
        pred_img = reconstruct_image(grayscale_list[i], predicted_list[i])
        pred_img = pred_img.permute(1, 2, 0)
        pred_img = torch.clamp(pred_img, 0, 1)
        axes[2, i].imshow(pred_img)
        axes[2, i].set_title('Predicted')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example of how to use the colorization implementation
    
    # Create dataset
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create colorization datasets
    train_colorization_dataset = ColorizationDataset(train_dataset, color_space='lab')
    test_colorization_dataset = ColorizationDataset(test_dataset, color_space='lab')
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_colorization_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_colorization_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = ColorizationModel(color_space='lab', num_color_channels=2)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_colorization_model(model, train_loader, test_loader, num_epochs=50, device=device)
    
    # Evaluate model
    loss = evaluate_colorization_model(model, test_loader, device)
    print(f'Test Loss: {loss:.4f}')
    
    # Visualize results
    visualize_colorization_results(model, test_loader, device) 
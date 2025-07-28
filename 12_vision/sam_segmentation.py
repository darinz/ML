"""
SAM: Segment Anything Model

This module implements SAM, a foundation model for image segmentation that can segment
any object in any image using prompts.

References:
- Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T.,
  Whitehead, S., Berg, A. C., Lo, W. Y., Dollár, P., & Girshick, R. (2023). Segment
  anything. ICCV.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional, Union
import random


class SAMImageEncoder(nn.Module):
    """Image encoder for SAM."""
    
    def __init__(self, encoder: str = 'vit_b', patch_size: int = 16, embed_dim: int = 768):
        """
        Initialize SAM image encoder.
        
        Args:
            encoder: Encoder architecture ('vit_b', 'vit_l', 'vit_h')
            patch_size: Size of image patches
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, 64, 64))  # 1024x1024 -> 64x64
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=12,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(12)
        ])
        
        # Layer normalization
        self.ln = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (batch_size, 3, height, width)
            
        Returns:
            Image features (batch_size, embed_dim, h, w)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch_size, embed_dim, h, w)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Reshape for transformer
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (batch_size, h*w, embed_dim)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Layer normalization
        x = self.ln(x)
        
        # Reshape back
        x = x.transpose(1, 2).reshape(b, c, h, w)
        
        return x


class SAMPromptEncoder(nn.Module):
    """Prompt encoder for SAM."""
    
    def __init__(self, embed_dim: int = 256, image_embedding_size: Tuple[int, int] = (64, 64)):
        """
        Initialize SAM prompt encoder.
        
        Args:
            embed_dim: Embedding dimension
            image_embedding_size: Size of image embedding
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        
        # Point embedding
        self.point_embed = nn.Embedding(3, embed_dim)  # 0: background, 1: foreground, 2: ignore
        
        # Box embedding
        self.box_embed = nn.Linear(4, embed_dim)
        
        # Mask embedding
        self.mask_embed = nn.Conv2d(1, embed_dim, kernel_size=2, stride=2)
        
        # No mask embedding
        self.no_mask_embed = nn.Parameter(torch.randn(1, embed_dim, 1, 1))
        
    def forward(self, points: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            points: Point prompts (batch_size, num_points, 3) - (x, y, label)
            boxes: Box prompts (batch_size, num_boxes, 4) - (x1, y1, x2, y2)
            masks: Mask prompts (batch_size, 1, h, w)
            
        Returns:
            Prompt embeddings (batch_size, embed_dim, h, w)
        """
        batch_size = 1
        if points is not None:
            batch_size = points.size(0)
        elif boxes is not None:
            batch_size = boxes.size(0)
        elif masks is not None:
            batch_size = masks.size(0)
        
        h, w = self.image_embedding_size
        prompt_embed = torch.zeros(batch_size, self.embed_dim, h, w, device=self.device)
        
        # Point prompts
        if points is not None:
            point_embeddings = self.point_embed(points[:, :, 2].long())  # (batch_size, num_points, embed_dim)
            # Simple point embedding (in practice, this would be more sophisticated)
            for i in range(points.size(1)):
                x, y, label = points[0, i]
                if 0 <= x < w and 0 <= y < h:
                    prompt_embed[0, :, int(y), int(x)] = point_embeddings[0, i]
        
        # Box prompts
        if boxes is not None:
            box_embeddings = self.box_embed(boxes)  # (batch_size, num_boxes, embed_dim)
            # Simple box embedding (in practice, this would be more sophisticated)
            for i in range(boxes.size(1)):
                x1, y1, x2, y2 = boxes[0, i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                prompt_embed[0, :, y1:y2, x1:x2] = box_embeddings[0, i].unsqueeze(-1).unsqueeze(-1)
        
        # Mask prompts
        if masks is not None:
            mask_embeddings = self.mask_embed(masks)  # (batch_size, embed_dim, h//2, w//2)
            # Upsample to match image embedding size
            mask_embeddings = F.interpolate(mask_embeddings, size=(h, w), mode='bilinear')
            prompt_embed = prompt_embed + mask_embeddings
        
        # If no prompts, use no mask embedding
        if points is None and boxes is None and masks is None:
            prompt_embed = prompt_embed + self.no_mask_embed
        
        return prompt_embed


class SAMMaskDecoder(nn.Module):
    """Mask decoder for SAM."""
    
    def __init__(self, num_multimask_outputs: int = 3, iou_head_depth: int = 3,
                 iou_head_hidden_dim: int = 256):
        """
        Initialize SAM mask decoder.
        
        Args:
            num_multimask_outputs: Number of mask outputs
            iou_head_depth: Depth of IoU prediction head
            iou_head_hidden_dim: Hidden dimension of IoU prediction head
        """
        super().__init__()
        self.num_multimask_outputs = num_multimask_outputs
        
        # Transformer decoder layers
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=2
        )
        
        # IoU prediction head
        self.iou_prediction_head = nn.Sequential(
            nn.Linear(256, iou_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(iou_head_hidden_dim, iou_head_hidden_dim),
            nn.ReLU(),
            nn.Linear(iou_head_hidden_dim, num_multimask_outputs)
        )
        
        # Mask prediction head
        self.mask_prediction_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Output projection
        self.output_projection = nn.Linear(256, 1)
        
    def forward(self, image_embeddings: torch.Tensor, 
                prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            image_embeddings: Image embeddings (batch_size, embed_dim, h, w)
            prompt_embeddings: Prompt embeddings (batch_size, embed_dim, h, w)
            
        Returns:
            Tuple of (masks, iou_predictions)
        """
        # Reshape embeddings for transformer
        b, c, h, w = image_embeddings.shape
        image_embeddings = image_embeddings.flatten(2).transpose(1, 2)  # (batch_size, h*w, embed_dim)
        prompt_embeddings = prompt_embeddings.flatten(2).transpose(1, 2)  # (batch_size, h*w, embed_dim)
        
        # Transformer decoder
        decoded_features = self.transformer_decoder(
            prompt_embeddings, image_embeddings
        )
        
        # IoU prediction
        iou_predictions = self.iou_prediction_head(decoded_features.mean(dim=1))
        
        # Mask prediction
        mask_features = self.mask_prediction_head(decoded_features)
        mask_logits = self.output_projection(mask_features)
        
        # Reshape to spatial dimensions
        masks = mask_logits.transpose(1, 2).reshape(b, self.num_multimask_outputs, h, w)
        
        return masks, iou_predictions


class SAMModel(nn.Module):
    """Complete SAM model."""
    
    def __init__(self, image_encoder: str = 'vit_b', num_multimask_outputs: int = 3):
        """
        Initialize SAM model.
        
        Args:
            image_encoder: Image encoder architecture
            num_multimask_outputs: Number of mask outputs
        """
        super().__init__()
        
        # Image encoder
        self.image_encoder = SAMImageEncoder(image_encoder)
        
        # Prompt encoder
        self.prompt_encoder = SAMPromptEncoder()
        
        # Mask decoder
        self.mask_decoder = SAMMaskDecoder(num_multimask_outputs)
        
    def forward(self, images: torch.Tensor, 
                points: Optional[torch.Tensor] = None,
                boxes: Optional[torch.Tensor] = None,
                masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images (batch_size, 3, height, width)
            points: Point prompts (batch_size, num_points, 3)
            boxes: Box prompts (batch_size, num_boxes, 4)
            masks: Mask prompts (batch_size, 1, h, w)
            
        Returns:
            Tuple of (masks, iou_predictions)
        """
        # Encode images
        image_embeddings = self.image_encoder(images)
        
        # Encode prompts
        prompt_embeddings = self.prompt_encoder(points, boxes, masks)
        
        # Decode masks
        masks, iou_predictions = self.mask_decoder(image_embeddings, prompt_embeddings)
        
        return masks, iou_predictions


class SAMLoss(nn.Module):
    """Loss function for SAM."""
    
    def __init__(self, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        """
        Initialize SAM loss.
        
        Args:
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
        """
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, pred_masks: torch.Tensor, gt_masks: torch.Tensor,
                pred_ious: torch.Tensor, gt_ious: torch.Tensor) -> torch.Tensor:
        """
        Compute SAM loss.
        
        Args:
            pred_masks: Predicted masks (batch_size, num_masks, h, w)
            gt_masks: Ground truth masks (batch_size, num_masks, h, w)
            pred_ious: Predicted IoUs (batch_size, num_masks)
            gt_ious: Ground truth IoUs (batch_size, num_masks)
            
        Returns:
            Loss value
        """
        # Focal loss for masks
        pred_masks_sigmoid = torch.sigmoid(pred_masks)
        focal_loss = self._focal_loss(pred_masks_sigmoid, gt_masks)
        
        # MSE loss for IoU predictions
        iou_loss = F.mse_loss(pred_ious, gt_ious)
        
        # Total loss
        total_loss = focal_loss + iou_loss
        
        return total_loss
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = pred * target + (1 - pred) * (1 - target)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()


def segment_with_points(model: nn.Module, 
                       images: torch.Tensor,
                       points: torch.Tensor,
                       device: str = 'cuda') -> torch.Tensor:
    """
    Segment images using point prompts.
    
    Args:
        model: Trained SAM model
        images: Input images (batch_size, 3, height, width)
        points: Point prompts (batch_size, num_points, 3) - (x, y, label)
        device: Device to use
        
    Returns:
        Predicted masks (batch_size, num_masks, height, width)
    """
    model.eval()
    
    with torch.no_grad():
        masks, iou_predictions = model(images.to(device), points=points.to(device))
        
        # Apply sigmoid to get probabilities
        masks = torch.sigmoid(masks)
        
        return masks


def segment_with_boxes(model: nn.Module, 
                      images: torch.Tensor,
                      boxes: torch.Tensor,
                      device: str = 'cuda') -> torch.Tensor:
    """
    Segment images using box prompts.
    
    Args:
        model: Trained SAM model
        images: Input images (batch_size, 3, height, width)
        boxes: Box prompts (batch_size, num_boxes, 4) - (x1, y1, x2, y2)
        device: Device to use
        
    Returns:
        Predicted masks (batch_size, num_masks, height, width)
    """
    model.eval()
    
    with torch.no_grad():
        masks, iou_predictions = model(images.to(device), boxes=boxes.to(device))
        
        # Apply sigmoid to get probabilities
        masks = torch.sigmoid(masks)
        
        return masks


def segment_with_masks(model: nn.Module, 
                      images: torch.Tensor,
                      masks: torch.Tensor,
                      device: str = 'cuda') -> torch.Tensor:
    """
    Segment images using mask prompts.
    
    Args:
        model: Trained SAM model
        images: Input images (batch_size, 3, height, width)
        masks: Mask prompts (batch_size, 1, height, width)
        device: Device to use
        
    Returns:
        Predicted masks (batch_size, num_masks, height, width)
    """
    model.eval()
    
    with torch.no_grad():
        pred_masks, iou_predictions = model(images.to(device), masks=masks.to(device))
        
        # Apply sigmoid to get probabilities
        pred_masks = torch.sigmoid(pred_masks)
        
        return pred_masks


def auto_segment(model: nn.Module, 
                images: torch.Tensor,
                device: str = 'cuda') -> torch.Tensor:
    """
    Automatically segment images without prompts.
    
    Args:
        model: Trained SAM model
        images: Input images (batch_size, 3, height, width)
        device: Device to use
        
    Returns:
        Predicted masks (batch_size, num_masks, height, width)
    """
    model.eval()
    
    with torch.no_grad():
        masks, iou_predictions = model(images.to(device))
        
        # Apply sigmoid to get probabilities
        masks = torch.sigmoid(masks)
        
        return masks


# Example usage
if __name__ == "__main__":
    # Example of how to use the SAM implementation
    
    # Create model
    model = SAMModel(image_encoder='vit_b', num_multimask_outputs=3)
    
    # Create dummy data
    batch_size = 2
    height, width = 1024, 1024
    
    images = torch.randn(batch_size, 3, height, width)
    points = torch.tensor([
        [[100, 100, 1], [200, 200, 1]],  # Foreground points
        [[300, 300, 0], [400, 400, 0]]   # Background points
    ], dtype=torch.float)
    
    boxes = torch.tensor([
        [[50, 50, 150, 150]],   # Box 1
        [[250, 250, 350, 350]]  # Box 2
    ], dtype=torch.float)
    
    masks = torch.randn(batch_size, 1, height, width)
    
    # Test different segmentation methods
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Point-based segmentation
    point_masks = segment_with_points(model, images, points, device)
    print(f'Point masks shape: {point_masks.shape}')
    
    # Box-based segmentation
    box_masks = segment_with_boxes(model, images, boxes, device)
    print(f'Box masks shape: {box_masks.shape}')
    
    # Mask-based segmentation
    mask_masks = segment_with_masks(model, images, masks, device)
    print(f'Mask masks shape: {mask_masks.shape}')
    
    # Auto segmentation
    auto_masks = auto_segment(model, images, device)
    print(f'Auto masks shape: {auto_masks.shape}') 
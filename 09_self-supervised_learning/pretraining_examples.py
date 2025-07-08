"""
Self-supervised Learning and Foundation Models - Implementation Examples

This file contains practical implementations of:
1. Supervised pretraining (ImageNet-style classification)
2. Contrastive learning (SimCLR-style)
3. Linear probe adaptation
4. Finetuning adaptation

Each section includes educational examples with detailed comments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1. SUPERVISED PRETRAINING IMPLEMENTATION
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple CNN for supervised pretraining on ImageNet-style classification.
    This represents the model that learns to classify images into categories.
    """
    def __init__(self, num_classes=1000, feature_dim=128):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers (these will be our pretrained features)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classification layer (this will be removed after pretraining)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """
        Extract features from the penultimate layer (before classifier).
        This is what we use after pretraining for downstream tasks.
        """
        features = self.features(x)
        features = nn.Flatten()(features)
        return features

def supervised_pretraining_example():
    """
    Example of supervised pretraining on a small dataset.
    In practice, this would be done on ImageNet with millions of images.
    """
    print("=== SUPERVISED PRETRAINING EXAMPLE ===")
    
    # Create synthetic dataset (in practice, use ImageNet)
    # Simulate 1000 images with 10 classes
    num_samples = 1000
    num_classes = 10
    
    # Generate synthetic data
    X = torch.randn(num_samples, 3, 32, 32)  # RGB images
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Create model
    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (simplified)
    print("Training model on synthetic dataset...")
    for epoch in range(5):  # In practice, train for many epochs
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Extract pretrained features
    print("\nExtracting pretrained features...")
    with torch.no_grad():
        features = model.extract_features(X)
        print(f"Feature shape: {features.shape}")
        print(f"Features extracted from {features.shape[0]} images")
    
    return model, features

# ============================================================================
# 2. CONTRASTIVE LEARNING IMPLEMENTATION (SIMCLR-STYLE)
# ============================================================================

class ContrastiveModel(nn.Module):
    """
    Contrastive learning model (SimCLR-style).
    Learns representations without labels using data augmentation.
    """
    def __init__(self, feature_dim=128, projection_dim=64):
        super(ContrastiveModel, self).__init__()
        
        # Encoder (feature extractor)
        self.encoder = nn.Sequential(
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
            nn.Linear(256, feature_dim)
        )
        
        # Projection head (for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return features, projections
    
    def extract_features(self, x):
        """Extract features without projection (for downstream tasks)"""
        return self.encoder(x)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for SimCLR-style training.
    Pulls positive pairs together, pushes negative pairs apart.
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, projections, labels):
        """
        Args:
            projections: [2N, D] tensor of projections
            labels: [N] tensor indicating which pairs are positive
        """
        # Normalize projections
        projections = nn.functional.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create mask for positive pairs
        mask = torch.zeros_like(similarity_matrix)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    mask[i, j] = 1
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Only consider positive pairs
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        loss = -mean_log_prob.mean()
        
        return loss

def data_augmentation_example():
    """
    Example of data augmentation for contrastive learning.
    Shows how to create positive pairs from the same image.
    """
    print("\n=== DATA AUGMENTATION EXAMPLE ===")
    
    # Create a simple image
    original_image = torch.randn(1, 3, 32, 32)
    
    # Define augmentations
    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0))
    ]
    
    # Apply augmentations to create positive pairs
    augmented_pairs = []
    for i in range(3):  # Create 3 pairs
        aug1 = original_image.clone()
        aug2 = original_image.clone()
        
        for aug in augmentations:
            aug1 = aug(aug1)
            aug2 = aug(aug2)
        
        augmented_pairs.append((aug1, aug2))
    
    print(f"Created {len(augmented_pairs)} positive pairs from 1 original image")
    print(f"Each pair contains two different views of the same image")
    
    return augmented_pairs

def contrastive_learning_example():
    """
    Example of contrastive learning training.
    """
    print("\n=== CONTRASTIVE LEARNING EXAMPLE ===")
    
    # Create synthetic dataset
    num_samples = 100
    batch_size = 16
    
    # Generate synthetic images
    images = torch.randn(num_samples, 3, 32, 32)
    
    # Create positive pairs using data augmentation
    positive_pairs = []
    labels = []
    
    for i in range(0, num_samples, 2):
        # Create two augmentations of the same image
        aug1 = images[i] + torch.randn_like(images[i]) * 0.1
        aug2 = images[i] + torch.randn_like(images[i]) * 0.1
        
        positive_pairs.extend([aug1, aug2])
        labels.extend([i//2, i//2])  # Same label for positive pairs
    
    # Convert to tensors
    positive_pairs = torch.stack(positive_pairs)
    labels = torch.tensor(labels)
    
    # Create model and loss
    model = ContrastiveModel()
    criterion = ContrastiveLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("Training contrastive model...")
    for epoch in range(10):
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(positive_pairs), batch_size):
            batch_images = positive_pairs[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            optimizer.zero_grad()
            features, projections = model(batch_images)
            loss = criterion(projections, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/num_batches:.4f}")
    
    # Extract learned features
    print("\nExtracting contrastive features...")
    with torch.no_grad():
        features = model.extract_features(positive_pairs)
        print(f"Feature shape: {features.shape}")
    
    return model, features

# ============================================================================
# 3. ADAPTATION METHODS
# ============================================================================

def linear_probe_example(pretrained_model, features):
    """
    Linear probe adaptation example.
    Freezes the pretrained model and learns a linear classifier on top.
    """
    print("\n=== LINEAR PROBE EXAMPLE ===")
    
    # Simulate downstream task data
    num_task_samples = 200
    num_task_classes = 5
    
    # Generate task-specific labels
    task_labels = np.random.randint(0, num_task_classes, num_task_samples)
    
    # Use pretrained features as input
    task_features = features[:num_task_samples].numpy()
    
    # Train linear classifier
    print("Training linear classifier on pretrained features...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(task_features, task_labels)
    
    # Evaluate
    predictions = clf.predict(task_features)
    accuracy = accuracy_score(task_labels, predictions)
    print(f"Linear probe accuracy: {accuracy:.3f}")
    
    return clf, accuracy

def finetuning_example(pretrained_model):
    """
    Finetuning adaptation example.
    Continues training all parameters on the new task.
    """
    print("\n=== FINE-TUNING EXAMPLE ===")
    
    # Create new task dataset
    num_task_samples = 200
    num_task_classes = 5
    
    # Generate synthetic task data
    task_images = torch.randn(num_task_samples, 3, 32, 32)
    task_labels = torch.randint(0, num_task_classes, (num_task_samples,))
    
    # Create new classifier for the task
    task_classifier = nn.Linear(128, num_task_classes)  # 128 is feature dim
    
    # Combine pretrained features with new classifier
    class FinetunedModel(nn.Module):
        def __init__(self, pretrained_model, task_classifier):
            super().__init__()
            self.pretrained = pretrained_model
            self.task_classifier = task_classifier
        
        def forward(self, x):
            features = self.pretrained.extract_features(x)
            output = self.task_classifier(features)
            return output
    
    # Create finetuned model
    finetuned_model = FinetunedModel(pretrained_model, task_classifier)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(finetuned_model.parameters(), lr=0.001)
    
    # Finetuning loop
    print("Finetuning model on new task...")
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = finetuned_model(task_images)
        loss = criterion(outputs, task_labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Evaluate
    with torch.no_grad():
        outputs = finetuned_model(task_images)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == task_labels).float().mean()
        print(f"Finetuning accuracy: {accuracy:.3f}")
    
    return finetuned_model, accuracy

# ============================================================================
# 4. COMPARISON AND VISUALIZATION
# ============================================================================

def compare_adaptation_methods():
    """
    Compare linear probe vs finetuning performance.
    """
    print("\n=== ADAPTATION METHODS COMPARISON ===")
    
    # Train a pretrained model
    pretrained_model, features = supervised_pretraining_example()
    
    # Test linear probe
    linear_clf, linear_acc = linear_probe_example(pretrained_model, features)
    
    # Test finetuning
    finetuned_model, finetune_acc = finetuning_example(pretrained_model)
    
    # Compare results
    print(f"\nResults Comparison:")
    print(f"Linear Probe Accuracy: {linear_acc:.3f}")
    print(f"Finetuning Accuracy: {finetune_acc:.3f}")
    
    if finetune_acc > linear_acc:
        print("Finetuning performed better (as expected with more data)")
    else:
        print("Linear probe performed better (may indicate limited task data)")
    
    return linear_acc, finetune_acc

def visualize_features(features, labels, title="Feature Visualization"):
    """
    Visualize learned features using t-SNE or PCA.
    """
    try:
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        
        # Reduce dimensionality for visualization
        if features.shape[1] > 2:
            # Use PCA first, then t-SNE
            pca = PCA(n_components=50)
            features_pca = pca.fit_transform(features)
            
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(features_pca)
        else:
            features_2d = features
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(scatter)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("sklearn not available for visualization")

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """
    Run all examples to demonstrate the complete foundation model pipeline.
    """
    print("FOUNDATION MODELS AND SELF-SUPERVISED LEARNING EXAMPLES")
    print("=" * 60)
    
    # 1. Supervised pretraining
    pretrained_model, features = supervised_pretraining_example()
    
    # 2. Contrastive learning
    contrastive_model, contrastive_features = contrastive_learning_example()
    
    # 3. Data augmentation example
    augmented_pairs = data_augmentation_example()
    
    # 4. Adaptation methods
    linear_acc, finetune_acc = compare_adaptation_methods()
    
    # 5. Visualization (if matplotlib is available)
    try:
        # Visualize features from supervised pretraining
        labels = np.random.randint(0, 5, features.shape[0])
        visualize_features(features.numpy(), labels, 
                         "Supervised Pretraining Features")
        
        # Visualize features from contrastive learning
        contrastive_labels = np.random.randint(0, 5, contrastive_features.shape[0])
        visualize_features(contrastive_features.numpy(), contrastive_labels,
                         "Contrastive Learning Features")
        
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETED!")
    print("Key takeaways:")
    print("1. Supervised pretraining learns features from labeled data")
    print("2. Contrastive learning learns features without labels")
    print("3. Linear probe is fast but limited")
    print("4. Finetuning is more flexible but requires more data")
    print("5. Both methods leverage pretrained representations effectively")

if __name__ == "__main__":
    main() 
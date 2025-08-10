"""
Self-supervised Learning and Foundation Models - Comprehensive Implementation Examples
====================================================================================

This file provides comprehensive implementations of self-supervised learning concepts from the markdown file.
Each section demonstrates key concepts with detailed explanations and visualizations.

Key Concepts Covered:
1. The data labeling problem and motivation for self-supervised learning
2. Supervised pretraining (ImageNet-style classification)
3. Contrastive learning (SimCLR-style) with data augmentation
4. Linear probe adaptation (feature extraction)
5. Finetuning adaptation (full model training)
6. Comparison of adaptation methods
7. Practical applications and best practices

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
from sklearn.manifold import TSNE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def print_section_header(title):
    """Print a formatted section header for better readability."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

# ============================================================================
# SECTION 1: UNDERSTANDING THE DATA LABELING PROBLEM
# ============================================================================

print_section_header("UNDERSTANDING THE DATA LABELING PROBLEM")

def demonstrate_labeling_cost():
    """
    Demonstrate the high cost of data labeling with concrete examples.
    """
    print("Demonstrating the data labeling bottleneck...")
    
    # Example scenarios with realistic costs
    scenarios = {
        "Medical Imaging": {
            "time_per_image": 15,  # minutes
            "expert_cost_per_hour": 150,  # USD
            "dataset_size": 100000,
            "total_cost": None,
            "total_time": None
        },
        "Speech Transcription": {
            "time_per_minute": 4,  # minutes of human time per minute of audio
            "expert_cost_per_hour": 50,  # USD
            "dataset_size": 1000,  # hours of audio
            "total_cost": None,
            "total_time": None
        },
        "Text Annotation": {
            "time_per_sentence": 0.5,  # minutes
            "expert_cost_per_hour": 30,  # USD
            "dataset_size": 50000,  # sentences
            "total_cost": None,
            "total_time": None
        }
    }
    
    # Calculate costs
    for task, data in scenarios.items():
        if task == "Medical Imaging":
            data["total_time"] = data["time_per_image"] * data["dataset_size"] / 60  # hours
            data["total_cost"] = data["total_time"] * data["expert_cost_per_hour"]
        elif task == "Speech Transcription":
            data["total_time"] = data["time_per_minute"] * data["dataset_size"] / 60  # hours
            data["total_cost"] = data["total_time"] * data["expert_cost_per_hour"]
        elif task == "Text Annotation":
            data["total_time"] = data["time_per_sentence"] * data["dataset_size"] / 60  # hours
            data["total_cost"] = data["total_time"] * data["expert_cost_per_hour"]
    
    # Display results
    print("Data Labeling Costs:")
    print("-" * 50)
    for task, data in scenarios.items():
        print(f"{task}:")
        print(f"  Dataset size: {data['dataset_size']:,}")
        print(f"  Total time: {data['total_time']:,.0f} hours ({data['total_time']/8/365:.1f} person-years)")
        print(f"  Total cost: ${data['total_cost']:,.0f}")
        print()
    
    print("Key insights:")
    print("1. Labeling large datasets requires significant human effort")
    print("2. Expert annotation is expensive and time-consuming")
    print("3. This creates a bottleneck for supervised learning")
    print("4. Self-supervised learning can leverage unlabeled data instead")

demonstrate_labeling_cost()

# ============================================================================
# SECTION 2: SUPERVISED PRETRAINING IMPLEMENTATION
# ============================================================================

print_section_header("SUPERVISED PRETRAINING IMPLEMENTATION")

class SimpleCNN(nn.Module):
    """
    Simple CNN for supervised pretraining on ImageNet-style classification.
    
    This model represents the standard approach where we train on a large labeled dataset
    (like ImageNet) and then use the learned features for downstream tasks.
    
    Architecture:
    - Feature extraction layers (convolutional layers)
    - Classification head (linear layers)
    - After pretraining, we remove the classification head and keep only the features
    """
    def __init__(self, num_classes=1000, feature_dim=128):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction layers (these will be our pretrained features)
        # These layers learn to extract useful visual features
        self.features = nn.Sequential(
            # First conv block: 3 -> 64 channels
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            
            # Second conv block: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
            
            # Third conv block: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Classification layer (this will be removed after pretraining)
        # This learns to map features to class predictions
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_classes)
        )
    
    def forward(self, x):
        """Forward pass: extract features and classify"""
        features = self.features(x)
        output = self.classifier(features)
        return output
    
    def extract_features(self, x):
        """
        Extract features from the penultimate layer (before classifier).
        This is what we use after pretraining for downstream tasks.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            features: Extracted features [batch_size, 256]
        """
        features = self.features(x)
        features = nn.Flatten()(features)
        return features

def supervised_pretraining_example():
    """
    Example of supervised pretraining on a small dataset.
    In practice, this would be done on ImageNet with millions of images.
    
    This demonstrates the standard supervised learning approach where we:
    1. Train a model to classify images into predefined categories
    2. Use the learned features for other tasks
    """
    print("Demonstrating supervised pretraining...")
    
    # Create synthetic dataset (in practice, use ImageNet)
    # Simulate 1000 images with 10 classes
    num_samples = 1000
    num_classes = 10
    
    print(f"Creating synthetic dataset: {num_samples} images, {num_classes} classes")
    
    # Generate synthetic data
    # In practice, this would be real images from ImageNet
    X = torch.randn(num_samples, 3, 32, 32)  # RGB images
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Create model
    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model on synthetic dataset...")
    print("(In practice, this would be trained on ImageNet for many epochs)")
    
    # Training loop (simplified)
    losses = []
    for epoch in range(5):  # In practice, train for many epochs
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Supervised Pretraining Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Extract pretrained features
    print("\nExtracting pretrained features...")
    with torch.no_grad():
        features = model.extract_features(X)
        print(f"Feature shape: {features.shape}")
        print(f"Features extracted from {features.shape[0]} images")
        print(f"Each feature vector has {features.shape[1]} dimensions")
    
    # Analyze feature quality
    print("\nAnalyzing feature quality...")
    feature_norms = torch.norm(features, dim=1)
    print(f"Average feature norm: {feature_norms.mean():.3f}")
    print(f"Feature norm std: {feature_norms.std():.3f}")
    
    return model, features

# Run supervised pretraining
supervised_model, supervised_features = supervised_pretraining_example()

# ============================================================================
# SECTION 3: CONTRASTIVE LEARNING IMPLEMENTATION (SIMCLR-STYLE)
# ============================================================================

print_section_header("CONTRASTIVE LEARNING IMPLEMENTATION (SIMCLR-STYLE)")

class ContrastiveModel(nn.Module):
    """
    Contrastive learning model (SimCLR-style).
    
    This model learns representations without labels using data augmentation.
    The key idea is that different views of the same image should have similar
    representations, while views of different images should have different representations.
    
    Architecture:
    - Encoder: extracts features from images
    - Projection head: maps features to a lower-dimensional space for contrastive learning
    """
    def __init__(self, feature_dim=128, projection_dim=64):
        super(ContrastiveModel, self).__init__()
        
        # Encoder (feature extractor)
        # This learns to extract useful features from images
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
        # This maps features to a lower-dimensional space where contrastive loss is applied
        # After pretraining, we discard this and use only the encoder
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        """
        Forward pass: extract features and project them
        
        Args:
            x: Input images [batch_size, 3, height, width]
            
        Returns:
            features: Extracted features [batch_size, feature_dim]
            projections: Projected features [batch_size, projection_dim]
        """
        features = self.encoder(x)
        projections = self.projection(features)
        return features, projections
    
    def extract_features(self, x):
        """Extract features without projection (for downstream tasks)"""
        return self.encoder(x)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for SimCLR-style training.
    
    This loss function implements the InfoNCE loss, which:
    - Pulls positive pairs (different views of the same image) together
    - Pushes negative pairs (views of different images) apart
    
    Mathematical formulation:
    L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
    where z_i and z_j are positive pairs, and z_k are all other samples
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, projections, labels):
        """
        Compute contrastive loss.
        
        Args:
            projections: [2N, D] tensor of projections (N pairs, each with 2 views)
            labels: [N] tensor indicating which pairs are positive
            
        Returns:
            loss: Contrastive loss value
        """
        # Normalize projections to unit vectors
        projections = nn.functional.normalize(projections, dim=1)
        
        # Compute similarity matrix
        # similarity_matrix[i, j] = cosine similarity between projections i and j
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Create mask for positive pairs
        # mask[i, j] = 1 if samples i and j are positive pairs (same image, different views)
        mask = torch.zeros_like(similarity_matrix)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    mask[i, j] = 1
        
        # Compute InfoNCE loss
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
    
    The key insight is that we apply different random transformations to the same
    image to create two "views" that should have similar representations.
    """
    print("Demonstrating data augmentation for contrastive learning...")
    
    # Create a simple image
    original_image = torch.randn(1, 3, 32, 32)
    
    # Define augmentations
    # These transformations should preserve the semantic content of the image
    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontally
        transforms.RandomRotation(degrees=10),    # Rotate slightly
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust colors
        transforms.RandomResizedCrop(32, scale=(0.8, 1.0))    # Crop and resize
    ]
    
    # Apply augmentations to create positive pairs
    augmented_pairs = []
    for i in range(3):  # Create 3 pairs
        aug1 = original_image.clone()
        aug2 = original_image.clone()
        
        # Apply different random augmentations to each view
        for aug in augmentations:
            aug1 = aug(aug1)
            aug2 = aug(aug2)
        
        augmented_pairs.append((aug1, aug2))
    
    print(f"Created {len(augmented_pairs)} positive pairs from 1 original image")
    print(f"Each pair contains two different views of the same image")
    print(f"These pairs should have similar representations after training")
    
    # Visualize the augmentations
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(original_image[0].permute(1, 2, 0).numpy())
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Augmented pairs
    for i, (aug1, aug2) in enumerate(augmented_pairs):
        axes[0, i+1].imshow(aug1[0].permute(1, 2, 0).numpy())
        axes[0, i+1].set_title(f'View 1 (Pair {i+1})')
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(aug2[0].permute(1, 2, 0).numpy())
        axes[1, i+1].set_title(f'View 2 (Pair {i+1})')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return augmented_pairs

def contrastive_learning_example():
    """
    Example of contrastive learning training.
    
    This demonstrates how to train a model using contrastive learning:
    1. Create positive pairs using data augmentation
    2. Train the model to make positive pairs similar and negative pairs different
    3. Use the learned features for downstream tasks
    """
    print("Demonstrating contrastive learning training...")
    
    # Create synthetic dataset
    num_samples = 100
    batch_size = 16
    
    # Generate synthetic images
    images = torch.randn(num_samples, 3, 32, 32)
    
    # Create positive pairs using data augmentation
    positive_pairs = []
    labels = []
    
    for i in range(num_samples // 2):  # Create pairs
        # Create two augmented views of the same image
        aug1 = images[i].clone()
        aug2 = images[i].clone()
        
        # Apply random augmentations
        if torch.rand(1) > 0.5:
            aug1 = torch.flip(aug1, [2])  # Horizontal flip
        if torch.rand(1) > 0.5:
            aug2 = torch.flip(aug2, [2])  # Horizontal flip
        
        positive_pairs.extend([aug1, aug2])
        labels.extend([i, i])  # Same label for both views
    
    # Convert to tensors
    positive_pairs = torch.stack(positive_pairs)
    labels = torch.tensor(labels)
    
    print(f"Created {len(positive_pairs)} augmented views from {num_samples//2} images")
    print(f"Each image has 2 views, creating {len(positive_pairs)//2} positive pairs")
    
    # Create model and train
    model = ContrastiveModel()
    criterion = ContrastiveLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining contrastive model...")
    losses = []
    
    for epoch in range(10):
        optimizer.zero_grad()
        
        # Forward pass
        features, projections = model(positive_pairs)
        
        # Compute contrastive loss
        loss = criterion(projections, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Contrastive Learning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Extract features
    print("\nExtracting contrastive features...")
    with torch.no_grad():
        contrastive_features = model.extract_features(positive_pairs)
        print(f"Feature shape: {contrastive_features.shape}")
    
    return model, contrastive_features

# Run contrastive learning
contrastive_model, contrastive_features = contrastive_learning_example()

# ============================================================================
# SECTION 4: LINEAR PROBE ADAPTATION
# ============================================================================

print_section_header("LINEAR PROBE ADAPTATION")

def linear_probe_example(pretrained_model, features):
    """
    Example of linear probe adaptation.
    
    Linear probe is a simple adaptation method where we:
    1. Freeze the pretrained model
    2. Train only a linear classifier on top of the features
    3. This tests how good the learned features are
    
    Advantages:
    - Fast and simple
    - No risk of catastrophic forgetting
    - Good for evaluating feature quality
    
    Disadvantages:
    - Limited expressiveness (only linear relationships)
    - May not achieve best performance
    """
    print("Demonstrating linear probe adaptation...")
    
    # Create a downstream task (e.g., binary classification)
    num_samples = features.shape[0]
    num_classes = 2
    
    # Create synthetic labels for the downstream task
    # In practice, these would be real task-specific labels
    downstream_labels = torch.randint(0, num_classes, (num_samples,))
    
    print(f"Downstream task: {num_classes}-class classification")
    print(f"Number of samples: {num_samples}")
    print(f"Feature dimension: {features.shape[1]}")
    
    # Convert to numpy for sklearn
    features_np = features.detach().numpy()
    labels_np = downstream_labels.numpy()
    
    # Train linear classifier
    print("\nTraining linear classifier...")
    classifier = LogisticRegression(random_state=42, max_iter=1000)
    classifier.fit(features_np, labels_np)
    
    # Evaluate
    predictions = classifier.predict(features_np)
    accuracy = accuracy_score(labels_np, predictions)
    
    print(f"Linear probe accuracy: {accuracy:.3f}")
    
    # Analyze the linear classifier
    print(f"\nLinear classifier weights shape: {classifier.coef_.shape}")
    print(f"Number of parameters: {classifier.coef_.size + classifier.intercept_.size}")
    
    # Compare with random features
    random_features = torch.randn_like(features)
    random_features_np = random_features.detach().numpy()
    
    random_classifier = LogisticRegression(random_state=42, max_iter=1000)
    random_classifier.fit(random_features_np, labels_np)
    random_predictions = random_classifier.predict(random_features_np)
    random_accuracy = accuracy_score(labels_np, random_predictions)
    
    print(f"Random features accuracy: {random_accuracy:.3f}")
    print(f"Improvement: {accuracy - random_accuracy:.3f}")
    
    return classifier, accuracy

# Test linear probe on both models
print("Testing linear probe on supervised features...")
supervised_classifier, supervised_accuracy = linear_probe_example(supervised_model, supervised_features)

print("\nTesting linear probe on contrastive features...")
contrastive_classifier, contrastive_accuracy = linear_probe_example(contrastive_model, contrastive_features)

# ============================================================================
# SECTION 5: FINETUNING ADAPTATION
# ============================================================================

print_section_header("FINETUNING ADAPTATION")

def finetuning_example(pretrained_model):
    """
    Example of finetuning adaptation.
    
    Finetuning is a more powerful adaptation method where we:
    1. Initialize the model with pretrained weights
    2. Continue training on the downstream task
    3. Update all parameters (both features and task-specific head)
    
    Advantages:
    - Better performance than linear probe
    - Can learn task-specific features
    - More flexible
    
    Disadvantages:
    - Computationally expensive
    - Risk of catastrophic forgetting
    - Requires more labeled data
    """
    print("Demonstrating finetuning adaptation...")
    
    # Create a downstream task
    num_samples = 500
    num_classes = 3
    
    # Generate synthetic data for downstream task
    X_downstream = torch.randn(num_samples, 3, 32, 32)
    y_downstream = torch.randint(0, num_classes, (num_samples,))
    
    print(f"Downstream task: {num_classes}-class classification")
    print(f"Number of samples: {num_samples}")
    
    # Create finetuned model
    class FinetunedModel(nn.Module):
        def __init__(self, pretrained_model, num_classes):
            super(FinetunedModel, self).__init__()
            
            # Use pretrained encoder
            self.encoder = pretrained_model.encoder
            
            # Add new task-specific head
            self.task_head = nn.Sequential(
                nn.Linear(128, 64),  # 128 is the feature dimension
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            features = self.encoder(x)
            output = self.task_head(features)
            return output
    
    # Create model and initialize with pretrained weights
    finetuned_model = FinetunedModel(pretrained_model, num_classes)
    
    # Copy pretrained weights
    finetuned_model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
    
    print("Model initialized with pretrained weights")
    print("Training on downstream task...")
    
    # Train on downstream task
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(finetuned_model.parameters(), lr=0.001)
    
    losses = []
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = finetuned_model(X_downstream)
        loss = criterion(outputs, y_downstream)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.title('Finetuning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Evaluate
    with torch.no_grad():
        outputs = finetuned_model(X_downstream)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_downstream).float().mean()
    
    print(f"Finetuning accuracy: {accuracy.item():.3f}")
    
    return finetuned_model, accuracy.item()

# Test finetuning on both models
print("Testing finetuning on supervised model...")
supervised_finetuned, supervised_finetune_acc = finetuning_example(supervised_model)

print("\nTesting finetuning on contrastive model...")
contrastive_finetuned, contrastive_finetune_acc = finetuning_example(contrastive_model)

# ============================================================================
# SECTION 6: COMPARISON OF ADAPTATION METHODS
# ============================================================================

print_section_header("COMPARISON OF ADAPTATION METHODS")

def compare_adaptation_methods():
    """
    Compare different adaptation methods and their performance.
    
    This demonstrates the trade-offs between different adaptation strategies:
    - Linear probe: Fast, simple, but limited
    - Finetuning: Better performance, but more complex
    """
    print("Comparing adaptation methods...")
    
    # Collect results
    results = {
        'Supervised Pretraining': {
            'Linear Probe': supervised_accuracy,
            'Finetuning': supervised_finetune_acc
        },
        'Contrastive Learning': {
            'Linear Probe': contrastive_accuracy,
            'Finetuning': contrastive_finetune_acc
        }
    }
    
    # Display comparison table
    print("\nPerformance Comparison:")
    print("-" * 50)
    print(f"{'Method':<20} {'Linear Probe':<15} {'Finetuning':<15}")
    print("-" * 50)
    
    for pretraining_method, accuracies in results.items():
        print(f"{pretraining_method:<20} {accuracies['Linear Probe']:<15.3f} {accuracies['Finetuning']:<15.3f}")
    
    # Create visualization
    methods = list(results.keys())
    linear_probe_scores = [results[m]['Linear Probe'] for m in methods]
    finetuning_scores = [results[m]['Finetuning'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, linear_probe_scores, width, label='Linear Probe', alpha=0.8)
    plt.bar(x + width/2, finetuning_scores, width, label='Finetuning', alpha=0.8)
    
    plt.xlabel('Pretraining Method')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Adaptation Methods')
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, (lp, ft) in enumerate(zip(linear_probe_scores, finetuning_scores)):
        plt.text(i - width/2, lp + 0.01, f'{lp:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, ft + 0.01, f'{ft:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    print("\nKey Insights:")
    print("1. Finetuning generally outperforms linear probe")
    print("2. Both pretraining methods provide useful features")
    print("3. Choice depends on data availability and performance requirements")
    
    return results

# Run comparison
comparison_results = compare_adaptation_methods()

# ============================================================================
# SECTION 7: FEATURE VISUALIZATION AND ANALYSIS
# ============================================================================

print_section_header("FEATURE VISUALIZATION AND ANALYSIS")

def visualize_features(features, labels, title="Feature Visualization"):
    """
    Visualize learned features using t-SNE.
    
    This helps us understand what the model has learned by visualizing
    how similar samples are positioned in the feature space.
    """
    print(f"Visualizing features: {title}")
    
    # Use t-SNE to reduce dimensionality for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features.detach().numpy())
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Color by class
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels.numpy(), cmap='tab10', alpha=0.7)
    
    plt.title(f'{title} (t-SNE)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, label='Class')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze feature statistics
    print(f"\nFeature Statistics:")
    print(f"Mean feature norm: {torch.norm(features, dim=1).mean():.3f}")
    print(f"Feature norm std: {torch.norm(features, dim=1).std():.3f}")
    print(f"Feature correlation: {torch.corrcoef(features.T).mean():.3f}")

def analyze_feature_quality():
    """
    Analyze the quality of features learned by different methods.
    """
    print("Analyzing feature quality...")
    
    # Create synthetic labels for visualization
    num_samples = supervised_features.shape[0]
    synthetic_labels = torch.randint(0, 5, (num_samples,))
    
    # Visualize supervised features
    visualize_features(supervised_features, synthetic_labels, 
                      "Supervised Pretraining Features")
    
    # Visualize contrastive features
    visualize_features(contrastive_features, synthetic_labels, 
                      "Contrastive Learning Features")
    
    # Compare feature statistics
    print("\nFeature Quality Comparison:")
    print("-" * 40)
    
    supervised_norms = torch.norm(supervised_features, dim=1)
    contrastive_norms = torch.norm(contrastive_features, dim=1)
    
    print(f"{'Metric':<20} {'Supervised':<15} {'Contrastive':<15}")
    print("-" * 40)
    print(f"{'Mean norm':<20} {supervised_norms.mean():<15.3f} {contrastive_norms.mean():<15.3f}")
    print(f"{'Norm std':<20} {supervised_norms.std():<15.3f} {contrastive_norms.std():<15.3f}")
    print(f"{'Min norm':<20} {supervised_norms.min():<15.3f} {contrastive_norms.min():<15.3f}")
    print(f"{'Max norm':<20} {supervised_norms.max():<15.3f} {contrastive_norms.max():<15.3f}")

# Run feature analysis
analyze_feature_quality()

# ============================================================================
# SECTION 8: PRACTICAL CONSIDERATIONS AND BEST PRACTICES
# ============================================================================

print_section_header("PRACTICAL CONSIDERATIONS AND BEST PRACTICES")

def practical_considerations():
    """
    Discuss practical considerations and best practices for self-supervised learning.
    """
    print("Practical Considerations and Best Practices:")
    print("=" * 50)
    
    considerations = {
        "Data Requirements": {
            "Supervised Pretraining": "Large labeled dataset (e.g., ImageNet)",
            "Contrastive Learning": "Large unlabeled dataset",
            "Linear Probe": "Small labeled dataset for downstream task",
            "Finetuning": "Moderate to large labeled dataset for downstream task"
        },
        "Computational Cost": {
            "Supervised Pretraining": "High (training large model)",
            "Contrastive Learning": "Very high (large batches, multiple views)",
            "Linear Probe": "Low (only train linear classifier)",
            "Finetuning": "High (train entire model)"
        },
        "Performance": {
            "Supervised Pretraining": "Good baseline performance",
            "Contrastive Learning": "Often better than supervised",
            "Linear Probe": "Limited by linear assumption",
            "Finetuning": "Best performance when data available"
        },
        "When to Use": {
            "Supervised Pretraining": "When you have labeled data and want proven approach",
            "Contrastive Learning": "When you have unlabeled data and want best performance",
            "Linear Probe": "Quick evaluation or limited downstream data",
            "Finetuning": "When you need best performance and have sufficient data"
        }
    }
    
    for category, methods in considerations.items():
        print(f"\n{category}:")
        print("-" * 30)
        for method, description in methods.items():
            print(f"  {method}: {description}")
    
    print("\nBest Practices:")
    print("-" * 20)
    practices = [
        "Always normalize your data before training",
        "Use appropriate data augmentations for your domain",
        "Start with linear probe to evaluate feature quality",
        "Use finetuning when you have sufficient labeled data",
        "Monitor for catastrophic forgetting during finetuning",
        "Consider the computational cost vs. performance trade-off",
        "Validate your approach on a held-out test set",
        "Use appropriate evaluation metrics for your task"
    ]
    
    for i, practice in enumerate(practices, 1):
        print(f"{i}. {practice}")

# Run practical considerations
practical_considerations()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run all examples.
    """
    print("Self-Supervised Learning and Foundation Models")
    print("Comprehensive Implementation Examples")
    print("=" * 60)
    
    print("\nThis script demonstrates:")
    print("1. The data labeling problem and motivation")
    print("2. Supervised pretraining implementation")
    print("3. Contrastive learning implementation")
    print("4. Linear probe adaptation")
    print("5. Finetuning adaptation")
    print("6. Comparison of methods")
    print("7. Feature analysis and visualization")
    print("8. Practical considerations")
    
    print("\nAll examples use synthetic data for educational purposes.")
    print("In practice, you would use real datasets like ImageNet.")

if __name__ == "__main__":
    main() 
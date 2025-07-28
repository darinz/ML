"""
Zero-Shot Classification

This module implements zero-shot classification methods that can classify images
without training on specific classes, using pre-trained vision-language models.

References:
- Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
  Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning
  transferable visual representations from natural language supervision. ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
import random


class ZeroShotClassifier:
    """Zero-shot classifier using vision-language models."""
    
    def __init__(self, model_name: str = 'clip', device: str = 'cuda'):
        """
        Initialize zero-shot classifier.
        
        Args:
            model_name: Name of the model ('clip', 'open_clip', etc.)
            device: Device to use
        """
        self.model_name = model_name
        self.device = device
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load pre-trained vision-language model."""
        if self.model_name == 'clip':
            try:
                import clip
                model, preprocess = clip.load('ViT-B/32', device=self.device)
                return model
            except ImportError:
                print("CLIP not available, using dummy model")
                return self._create_dummy_model()
        else:
            return self._create_dummy_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for demonstration."""
        class DummyVisionLanguageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, 512)
                )
                self.text_encoder = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512)
                )
            
            def encode_image(self, image):
                return F.normalize(self.image_encoder(image), dim=1)
            
            def encode_text(self, text):
                return F.normalize(self.text_encoder(text), dim=1)
        
        return DummyVisionLanguageModel().to(self.device)
    
    def classify(self, images: torch.Tensor, class_names: List[str], 
                class_descriptions: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify images using zero-shot learning.
        
        Args:
            images: Input images (batch_size, 3, height, width)
            class_names: List of class names
            class_descriptions: Optional descriptions for each class
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        with torch.no_grad():
            # Encode images
            image_features = self.model.encode_image(images)
            
            # Prepare text prompts
            if class_descriptions is None:
                text_prompts = [f"a photo of a {class_name}" for class_name in class_names]
            else:
                text_prompts = class_descriptions
            
            # Encode text prompts
            text_features = []
            for prompt in text_prompts:
                # Simple text encoding for dummy model
                if hasattr(self.model, 'encode_text'):
                    text_feat = self.model.encode_text(self._encode_text_simple(prompt))
                else:
                    # For CLIP model
                    import clip
                    text_tokens = clip.tokenize(prompt).to(self.device)
                    text_feat = self.model.encode_text(text_tokens)
                text_features.append(text_feat)
            
            text_features = torch.cat(text_features, dim=0)
            
            # Compute similarities
            similarities = torch.mm(image_features, text_features.T)
            
            # Get predictions and probabilities
            probabilities = F.softmax(similarities, dim=1)
            predictions = torch.argmax(similarities, dim=1)
            
            return predictions, probabilities
    
    def _encode_text_simple(self, text: str) -> torch.Tensor:
        """Simple text encoding for dummy model."""
        # Convert text to simple numerical representation
        text_vector = torch.zeros(512)
        for i, char in enumerate(text[:512]):
            text_vector[i] = ord(char)
        return text_vector.unsqueeze(0).to(self.device)


class FewShotClassifier:
    """Few-shot classifier using vision-language models."""
    
    def __init__(self, base_model: str = 'clip', device: str = 'cuda'):
        """
        Initialize few-shot classifier.
        
        Args:
            base_model: Base model name
            device: Device to use
        """
        self.base_model = base_model
        self.device = device
        self.classifier = ZeroShotClassifier(base_model, device)
    
    def fit(self, support_images: torch.Tensor, support_labels: torch.Tensor,
            class_names: List[str]) -> None:
        """
        Fit the classifier using support examples.
        
        Args:
            support_images: Support images (num_support, 3, height, width)
            support_labels: Support labels (num_support,)
            class_names: List of class names
        """
        self.support_images = support_images
        self.support_labels = support_labels
        self.class_names = class_names
        
        # Encode support images
        with torch.no_grad():
            self.support_features = self.classifier.model.encode_image(support_images)
    
    def predict(self, query_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict classes for query images.
        
        Args:
            query_images: Query images (batch_size, 3, height, width)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        with torch.no_grad():
            # Encode query images
            query_features = self.classifier.model.encode_image(query_images)
            
            # Compute similarities with support features
            similarities = torch.mm(query_features, self.support_features.T)
            
            # Aggregate similarities by class
            num_classes = len(self.class_names)
            batch_size = query_images.size(0)
            
            class_similarities = torch.zeros(batch_size, num_classes, device=self.device)
            
            for i in range(num_classes):
                class_mask = (self.support_labels == i)
                if class_mask.any():
                    class_sim = similarities[:, class_mask].mean(dim=1)
                    class_similarities[:, i] = class_sim
            
            # Get predictions and probabilities
            probabilities = F.softmax(class_similarities, dim=1)
            predictions = torch.argmax(class_similarities, dim=1)
            
            return predictions, probabilities


class OpenSetClassifier:
    """Open-set classifier for unknown class detection."""
    
    def __init__(self, base_model: str = 'clip', device: str = 'cuda', threshold: float = 0.5):
        """
        Initialize open-set classifier.
        
        Args:
            base_model: Base model name
            device: Device to use
            threshold: Threshold for unknown class detection
        """
        self.base_model = base_model
        self.device = device
        self.threshold = threshold
        self.classifier = ZeroShotClassifier(base_model, device)
    
    def classify(self, images: torch.Tensor, class_names: List[str],
                class_descriptions: Optional[List[str]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classify images with unknown class detection.
        
        Args:
            images: Input images (batch_size, 3, height, width)
            class_names: List of known class names
            class_descriptions: Optional descriptions for each class
            
        Returns:
            Tuple of (predictions, probabilities, unknown_scores)
        """
        # Get standard predictions
        predictions, probabilities = self.classifier.classify(images, class_names, class_descriptions)
        
        # Compute unknown scores (confidence-based)
        max_probabilities = torch.max(probabilities, dim=1)[0]
        unknown_scores = 1.0 - max_probabilities
        
        # Detect unknown classes
        unknown_mask = unknown_scores > self.threshold
        
        return predictions, probabilities, unknown_scores


class HierarchicalClassifier:
    """Hierarchical zero-shot classifier."""
    
    def __init__(self, base_model: str = 'clip', device: str = 'cuda'):
        """
        Initialize hierarchical classifier.
        
        Args:
            base_model: Base model name
            device: Device to use
        """
        self.base_model = base_model
        self.device = device
        self.classifier = ZeroShotClassifier(base_model, device)
        self.hierarchy = {}
    
    def set_hierarchy(self, hierarchy: Dict[str, List[str]]):
        """
        Set the class hierarchy.
        
        Args:
            hierarchy: Dictionary mapping superclasses to subclasses
        """
        self.hierarchy = hierarchy
    
    def classify(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify images using hierarchical classification.
        
        Args:
            images: Input images (batch_size, 3, height, width)
            
        Returns:
            Tuple of (superclass_predictions, subclass_predictions)
        """
        batch_size = images.size(0)
        superclass_predictions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        subclass_predictions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # First, classify superclasses
        superclass_names = list(self.hierarchy.keys())
        superclass_pred, superclass_prob = self.classifier.classify(images, superclass_names)
        superclass_predictions = superclass_pred
        
        # Then, classify subclasses within predicted superclasses
        for i in range(batch_size):
            predicted_superclass = superclass_names[superclass_pred[i]]
            subclass_names = self.hierarchy[predicted_superclass]
            
            # Classify within the predicted superclass
            subclass_pred, _ = self.classifier.classify(images[i:i+1], subclass_names)
            subclass_predictions[i] = subclass_pred[0]
        
        return superclass_predictions, subclass_predictions


def evaluate_zero_shot_classifier(classifier: ZeroShotClassifier,
                                test_images: torch.Tensor,
                                test_labels: torch.Tensor,
                                class_names: List[str]) -> Dict[str, float]:
    """
    Evaluate zero-shot classifier.
    
    Args:
        classifier: Zero-shot classifier
        test_images: Test images
        test_labels: Test labels
        class_names: List of class names
        
    Returns:
        Dictionary of evaluation metrics
    """
    predictions, probabilities = classifier.classify(test_images, class_names)
    
    # Compute accuracy
    accuracy = (predictions == test_labels).float().mean().item()
    
    # Compute per-class accuracy
    per_class_accuracy = {}
    for i, class_name in enumerate(class_names):
        class_mask = (test_labels == i)
        if class_mask.any():
            class_acc = (predictions[class_mask] == test_labels[class_mask]).float().mean().item()
            per_class_accuracy[class_name] = class_acc
    
    # Compute confidence metrics
    max_probabilities = torch.max(probabilities, dim=1)[0]
    avg_confidence = max_probabilities.mean().item()
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'avg_confidence': avg_confidence
    }


def create_text_prompts(class_names: List[str], 
                       prompt_templates: Optional[List[str]] = None) -> List[str]:
    """
    Create text prompts for zero-shot classification.
    
    Args:
        class_names: List of class names
        prompt_templates: Optional list of prompt templates
        
    Returns:
        List of text prompts
    """
    if prompt_templates is None:
        prompt_templates = [
            "a photo of a {}",
            "a picture of a {}",
            "an image of a {}",
            "a photograph of a {}",
            "a {} in the image"
        ]
    
    prompts = []
    for class_name in class_names:
        for template in prompt_templates:
            prompts.append(template.format(class_name))
    
    return prompts


def ensemble_zero_shot_classification(classifiers: List[ZeroShotClassifier],
                                    images: torch.Tensor,
                                    class_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform ensemble zero-shot classification.
    
    Args:
        classifiers: List of zero-shot classifiers
        images: Input images
        class_names: List of class names
        
    Returns:
        Tuple of (ensemble_predictions, ensemble_probabilities)
    """
    all_probabilities = []
    
    for classifier in classifiers:
        _, probabilities = classifier.classify(images, class_names)
        all_probabilities.append(probabilities)
    
    # Average probabilities
    ensemble_probabilities = torch.stack(all_probabilities).mean(dim=0)
    ensemble_predictions = torch.argmax(ensemble_probabilities, dim=1)
    
    return ensemble_predictions, ensemble_probabilities


# Example usage
if __name__ == "__main__":
    # Example of how to use the zero-shot classification implementation
    
    # Create classifier
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = ZeroShotClassifier(model_name='clip', device=device)
    
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    class_names = ['cat', 'dog', 'bird', 'car']
    
    # Zero-shot classification
    predictions, probabilities = classifier.classify(images, class_names)
    print(f'Predictions: {predictions}')
    print(f'Probabilities shape: {probabilities.shape}')
    
    # Few-shot classification
    few_shot_classifier = FewShotClassifier(base_model='clip', device=device)
    
    # Create support examples
    support_images = torch.randn(8, 3, 224, 224)  # 2 examples per class
    support_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    
    # Fit the classifier
    few_shot_classifier.fit(support_images, support_labels, class_names)
    
    # Predict on query images
    query_images = torch.randn(2, 3, 224, 224)
    query_predictions, query_probabilities = few_shot_classifier.predict(query_images)
    print(f'Few-shot predictions: {query_predictions}')
    
    # Open-set classification
    open_set_classifier = OpenSetClassifier(base_model='clip', device=device, threshold=0.3)
    predictions, probabilities, unknown_scores = open_set_classifier.classify(images, class_names)
    print(f'Unknown scores: {unknown_scores}')
    
    # Hierarchical classification
    hierarchical_classifier = HierarchicalClassifier(base_model='clip', device=device)
    
    # Set hierarchy
    hierarchy = {
        'animal': ['cat', 'dog', 'bird'],
        'vehicle': ['car', 'truck', 'bicycle']
    }
    hierarchical_classifier.set_hierarchy(hierarchy)
    
    # Classify
    superclass_pred, subclass_pred = hierarchical_classifier.classify(images)
    print(f'Superclass predictions: {superclass_pred}')
    print(f'Subclass predictions: {subclass_pred}')
    
    # Evaluate classifier
    test_labels = torch.randint(0, len(class_names), (batch_size,))
    metrics = evaluate_zero_shot_classifier(classifier, images, test_labels, class_names)
    print(f'Evaluation metrics: {metrics}')
    
    # Create text prompts
    prompts = create_text_prompts(class_names)
    print(f'Text prompts: {prompts[:5]}')  # Show first 5 prompts
    
    # Ensemble classification
    classifiers = [classifier, classifier]  # Same classifier for demonstration
    ensemble_pred, ensemble_prob = ensemble_zero_shot_classification(classifiers, images, class_names)
    print(f'Ensemble predictions: {ensemble_pred}') 
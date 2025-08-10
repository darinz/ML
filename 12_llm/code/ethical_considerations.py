"""
Ethical Considerations for Large Language Models.

This module implements various tools and techniques for addressing
ethical concerns in LLM development and deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from typing import List, Dict, Any, Tuple

def detect_bias(model, tokenizer, test_prompts, target_groups):
    """
    Detect bias in model outputs.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        test_prompts: List of test prompts
        target_groups: List of target groups to test for bias
        
    Returns:
        bias_scores: Dictionary of bias scores for each group
    """
    model.eval()
    bias_scores = {}
    
    for group in target_groups:
        group_scores = []
        
        for prompt in test_prompts:
            # Generate completions
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            input_ids = input_ids.to(next(model.parameters()).device)
            
            with torch.no_grad():
                outputs = model(input_ids)
                probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
            
            # Calculate bias score for this group
            group_tokens = [tokenizer.encode(group, add_special_tokens=False)[0]]
            group_prob = probs[0, group_tokens].mean().item()
            group_scores.append(group_prob)
        
        bias_scores[group] = np.mean(group_scores)
    
    return bias_scores

def safety_filter(text, harmful_patterns):
    """
    Filter potentially harmful content.
    
    Args:
        text: Text to filter
        harmful_patterns: List of harmful patterns to check for
        
    Returns:
        is_safe: Boolean indicating if text is safe
    """
    text_lower = text.lower()
    
    for pattern in harmful_patterns:
        if pattern in text_lower:
            return False
    
    return True

def generate_safe_text(model, tokenizer, prompt, safety_filter_func, max_length=100):
    """
    Generate text with safety filtering.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        safety_filter_func: Function to check text safety
        max_length: Maximum generation length
        
    Returns:
        generated_text: Safe generated text or safety message
    """
    from deployment_inference import generate_text
    
    generated_text = generate_text(model, tokenizer, prompt, max_length=max_length)
    
    if safety_filter_func(generated_text):
        return generated_text
    else:
        return "I cannot generate that content."

class BiasDetector:
    """
    Comprehensive bias detection for language models.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize bias detector.
        
        Args:
            model: The language model to analyze
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        
    def analyze_gender_bias(self, test_prompts):
        """
        Analyze gender bias in model outputs.
        
        Args:
            test_prompts: List of test prompts
            
        Returns:
            bias_analysis: Dictionary containing bias analysis
        """
        gender_terms = {
            'male': ['he', 'him', 'his', 'man', 'men', 'boy', 'boys'],
            'female': ['she', 'her', 'hers', 'woman', 'women', 'girl', 'girls']
        }
        
        bias_scores = {}
        for gender, terms in gender_terms.items():
            scores = []
            for prompt in test_prompts:
                # Generate completion
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
                input_ids = input_ids.to(next(self.model.parameters()).device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
                
                # Calculate probability of gender terms
                gender_probs = []
                for term in terms:
                    term_ids = self.tokenizer.encode(term, add_special_tokens=False)
                    if term_ids:
                        gender_probs.append(probs[0, term_ids[0]].item())
                
                if gender_probs:
                    scores.append(np.mean(gender_probs))
            
            bias_scores[gender] = np.mean(scores) if scores else 0.0
        
        # Calculate bias ratio
        bias_ratio = bias_scores['male'] / bias_scores['female'] if bias_scores['female'] > 0 else float('inf')
        
        return {
            'bias_scores': bias_scores,
            'bias_ratio': bias_ratio,
            'is_biased': abs(bias_ratio - 1.0) > 0.2  # Threshold for bias detection
        }
    
    def analyze_racial_bias(self, test_prompts):
        """
        Analyze racial bias in model outputs.
        
        Args:
            test_prompts: List of test prompts
            
        Returns:
            bias_analysis: Dictionary containing bias analysis
        """
        racial_terms = {
            'white': ['white', 'caucasian', 'european'],
            'black': ['black', 'african', 'african-american'],
            'asian': ['asian', 'chinese', 'japanese', 'korean'],
            'hispanic': ['hispanic', 'latino', 'latina', 'mexican']
        }
        
        bias_scores = {}
        for race, terms in racial_terms.items():
            scores = []
            for prompt in test_prompts:
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
                input_ids = input_ids.to(next(self.model.parameters()).device)
                
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)
                
                race_probs = []
                for term in terms:
                    term_ids = self.tokenizer.encode(term, add_special_tokens=False)
                    if term_ids:
                        race_probs.append(probs[0, term_ids[0]].item())
                
                if race_probs:
                    scores.append(np.mean(race_probs))
            
            bias_scores[race] = np.mean(scores) if scores else 0.0
        
        return {
            'bias_scores': bias_scores,
            'max_bias': max(bias_scores.values()),
            'min_bias': min(bias_scores.values()),
            'bias_range': max(bias_scores.values()) - min(bias_scores.values())
        }

class ContentFilter:
    """
    Content filtering for harmful or inappropriate content.
    """
    
    def __init__(self, harmful_patterns=None, inappropriate_words=None):
        """
        Initialize content filter.
        
        Args:
            harmful_patterns: List of harmful patterns to filter
            inappropriate_words: List of inappropriate words to filter
        """
        self.harmful_patterns = harmful_patterns or [
            'harm', 'violence', 'hate', 'discrimination',
            'illegal', 'dangerous', 'harmful'
        ]
        
        self.inappropriate_words = inappropriate_words or [
            # Add inappropriate words here
        ]
    
    def filter_text(self, text):
        """
        Filter text for harmful content.
        
        Args:
            text: Text to filter
            
        Returns:
            filtered_text: Filtered text or safety message
        """
        text_lower = text.lower()
        
        # Check for harmful patterns
        for pattern in self.harmful_patterns:
            if pattern in text_lower:
                return "Content filtered for safety reasons."
        
        # Check for inappropriate words
        for word in self.inappropriate_words:
            if word in text_lower:
                return "Content filtered for inappropriate language."
        
        return text
    
    def get_toxicity_score(self, text):
        """
        Calculate toxicity score for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            toxicity_score: Score between 0 and 1
        """
        text_lower = text.lower()
        toxicity_indicators = 0
        total_words = len(text.split())
        
        # Count harmful indicators
        for pattern in self.harmful_patterns:
            if pattern in text_lower:
                toxicity_indicators += 1
        
        # Normalize score
        toxicity_score = min(toxicity_indicators / max(total_words, 1), 1.0)
        
        return toxicity_score

class FairnessMetrics:
    """
    Calculate fairness metrics for model predictions.
    """
    
    def __init__(self):
        """Initialize fairness metrics calculator."""
        pass
    
    def demographic_parity(self, predictions, sensitive_attributes):
        """
        Calculate demographic parity.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Sensitive attribute values
            
        Returns:
            parity_score: Demographic parity score
        """
        unique_attributes = np.unique(sensitive_attributes)
        positive_rates = {}
        
        for attr in unique_attributes:
            mask = sensitive_attributes == attr
            positive_rates[attr] = np.mean(predictions[mask])
        
        # Calculate disparity
        rates = list(positive_rates.values())
        disparity = max(rates) - min(rates)
        
        return {
            'positive_rates': positive_rates,
            'disparity': disparity,
            'is_fair': disparity < 0.1  # Threshold for fairness
        }
    
    def equalized_odds(self, predictions, true_labels, sensitive_attributes):
        """
        Calculate equalized odds.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            sensitive_attributes: Sensitive attribute values
            
        Returns:
            odds_analysis: Equalized odds analysis
        """
        unique_attributes = np.unique(sensitive_attributes)
        tpr_by_group = {}
        fpr_by_group = {}
        
        for attr in unique_attributes:
            mask = sensitive_attributes == attr
            group_predictions = predictions[mask]
            group_labels = true_labels[mask]
            
            # True positive rate
            tpr = np.sum((group_predictions == 1) & (group_labels == 1)) / np.sum(group_labels == 1)
            tpr_by_group[attr] = tpr
            
            # False positive rate
            fpr = np.sum((group_predictions == 1) & (group_labels == 0)) / np.sum(group_labels == 0)
            fpr_by_group[attr] = fpr
        
        # Calculate disparities
        tpr_disparity = max(tpr_by_group.values()) - min(tpr_by_group.values())
        fpr_disparity = max(fpr_by_group.values()) - min(fpr_by_group.values())
        
        return {
            'tpr_by_group': tpr_by_group,
            'fpr_by_group': fpr_by_group,
            'tpr_disparity': tpr_disparity,
            'fpr_disparity': fpr_disparity,
            'is_fair': tpr_disparity < 0.1 and fpr_disparity < 0.1
        }

class EthicalTraining:
    """
    Techniques for ethical training of language models.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize ethical training.
        
        Args:
            model: The model to train ethically
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def create_ethical_dataset(self, base_dataset, ethical_guidelines):
        """
        Create ethically balanced dataset.
        
        Args:
            base_dataset: Base dataset
            ethical_guidelines: Guidelines for ethical content
            
        Returns:
            ethical_dataset: Ethically balanced dataset
        """
        # This is a simplified implementation
        # In practice, you would implement more sophisticated filtering
        
        ethical_dataset = []
        
        for item in base_dataset:
            # Check if item meets ethical guidelines
            if self._meets_guidelines(item, ethical_guidelines):
                ethical_dataset.append(item)
        
        return ethical_dataset
    
    def _meets_guidelines(self, item, guidelines):
        """
        Check if item meets ethical guidelines.
        
        Args:
            item: Dataset item
            guidelines: Ethical guidelines
            
        Returns:
            meets_guidelines: Boolean indicating if item meets guidelines
        """
        # Simplified implementation
        # In practice, you would implement more sophisticated checking
        
        text = item.get('text', '')
        text_lower = text.lower()
        
        # Check for harmful content
        harmful_indicators = ['harm', 'violence', 'hate', 'discrimination']
        for indicator in harmful_indicators:
            if indicator in text_lower:
                return False
        
        return True
    
    def ethical_loss(self, outputs, targets, ethical_penalty=0.1):
        """
        Add ethical penalty to loss function.
        
        Args:
            outputs: Model outputs
            targets: Target labels
            ethical_penalty: Penalty for unethical predictions
            
        Returns:
            total_loss: Loss with ethical penalty
        """
        # Standard loss
        standard_loss = F.cross_entropy(outputs, targets)
        
        # Ethical penalty (simplified)
        # In practice, you would implement more sophisticated ethical penalties
        ethical_penalty_loss = 0.0
        
        # Add penalty for potentially harmful predictions
        probs = F.softmax(outputs, dim=1)
        harmful_token_ids = [self.tokenizer.encode('harm', add_special_tokens=False)[0]]
        
        for token_id in harmful_token_ids:
            if token_id < outputs.size(-1):
                harmful_prob = probs[:, token_id].mean()
                ethical_penalty_loss += harmful_prob * ethical_penalty
        
        total_loss = standard_loss + ethical_penalty_loss
        
        return total_loss

def create_ethical_guidelines():
    """
    Create comprehensive ethical guidelines.
    
    Returns:
        guidelines: Dictionary of ethical guidelines
    """
    guidelines = {
        'harmful_content': {
            'description': 'Avoid generating harmful or violent content',
            'examples': ['violence', 'hate speech', 'discrimination'],
            'severity': 'high'
        },
        'privacy': {
            'description': 'Respect user privacy and data protection',
            'examples': ['personal information', 'sensitive data'],
            'severity': 'high'
        },
        'accuracy': {
            'description': 'Ensure factual accuracy and avoid misinformation',
            'examples': ['false claims', 'misleading information'],
            'severity': 'medium'
        },
        'fairness': {
            'description': 'Ensure fair and unbiased outputs',
            'examples': ['gender bias', 'racial bias', 'ageism'],
            'severity': 'high'
        },
        'transparency': {
            'description': 'Be transparent about model capabilities and limitations',
            'examples': ['uncertainty', 'confidence levels'],
            'severity': 'medium'
        }
    }
    
    return guidelines

# Example usage functions
def demonstrate_ethical_tools():
    """Demonstrate various ethical tools."""
    print("Ethical Considerations Tools Demonstration")
    print("=" * 40)
    
    # Example 1: Bias Detection
    print("1. Bias Detection")
    print("   - Detects gender, racial, and other biases")
    print("   - Analyzes model outputs for fairness")
    print("   - Provides quantitative bias scores")
    print()
    
    # Example 2: Content Filtering
    print("2. Content Filtering")
    print("   - Filters harmful or inappropriate content")
    print("   - Calculates toxicity scores")
    print("   - Implements safety measures")
    print()
    
    # Example 3: Fairness Metrics
    print("3. Fairness Metrics")
    print("   - Demographic parity analysis")
    print("   - Equalized odds calculation")
    print("   - Fairness evaluation tools")
    print()
    
    # Example 4: Ethical Training
    print("4. Ethical Training")
    print("   - Ethical dataset creation")
    print("   - Ethical loss functions")
    print("   - Guidelines-based training")
    print()
    
    # Example 5: Safety Measures
    print("5. Safety Measures")
    print("   - Input validation and filtering")
    print("   - Output safety checks")
    print("   - Harmful content prevention")
    print()
    
    # Example usage code
    print("Example Usage:")
    print("""
    # Initialize bias detector
    bias_detector = BiasDetector(model, tokenizer)
    
    # Analyze gender bias
    gender_bias = bias_detector.analyze_gender_bias(test_prompts)
    print(f"Gender bias ratio: {gender_bias['bias_ratio']:.2f}")
    
    # Initialize content filter
    content_filter = ContentFilter()
    
    # Filter text
    filtered_text = content_filter.filter_text(generated_text)
    toxicity_score = content_filter.get_toxicity_score(generated_text)
    print(f"Toxicity score: {toxicity_score:.3f}")
    
    # Calculate fairness metrics
    fairness_calc = FairnessMetrics()
    parity_analysis = fairness_calc.demographic_parity(predictions, sensitive_attrs)
    print(f"Demographic parity: {parity_analysis['is_fair']}")
    
    # Ethical training
    ethical_trainer = EthicalTraining(model, tokenizer)
    ethical_loss = ethical_trainer.ethical_loss(outputs, targets)
    print(f"Ethical loss: {ethical_loss:.4f}")
    """)

if __name__ == "__main__":
    demonstrate_ethical_tools()

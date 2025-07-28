"""
Preference Data Processing for RLHF

This module provides implementations for processing preference data for reinforcement
learning from human feedback (RLHF). It includes data loading, preprocessing,
augmentation, and quality control methods.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
import random

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """
    Dataset for preference learning with prompt-response pairs.
    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get prompt and responses
        prompt = item['prompt']
        chosen_response = item['chosen_response']
        rejected_response = item['rejected_response']
        
        # Tokenize
        chosen_text = prompt + chosen_response
        rejected_text = prompt + rejected_response
        
        chosen_inputs = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        rejected_inputs = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'chosen_ids': chosen_inputs['input_ids'].squeeze(0),
            'rejected_ids': rejected_inputs['input_ids'].squeeze(0),
            'chosen_mask': chosen_inputs['attention_mask'].squeeze(0),
            'rejected_mask': rejected_inputs['attention_mask'].squeeze(0),
            'prompt': prompt,
            'chosen_response': chosen_response,
            'rejected_response': rejected_response
        }


class RankingDataset(Dataset):
    """
    Dataset for ranking-based preference learning.
    """
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = item['prompt']
        responses = item['responses']  # List of responses ordered by preference
        ranking = item['ranking']      # Human ranking
        
        # Tokenize all responses
        tokenized_responses = []
        for response in responses:
            text = prompt + response
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            tokenized_responses.append({
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0)
            })
        
        return {
            'prompt': prompt,
            'responses': responses,
            'ranking': ranking,
            'tokenized_responses': tokenized_responses
        }


class PreferenceDataProcessor:
    """
    Processor for preference data with various preprocessing and augmentation techniques.
    """
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_from_json(self, file_path: str) -> List[Dict]:
        """
        Load preference data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            data: List of preference data
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def load_from_csv(self, file_path: str) -> List[Dict]:
        """
        Load preference data from CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            data: List of preference data
        """
        df = pd.read_csv(file_path)
        
        data = []
        for _, row in df.iterrows():
            data.append({
                'prompt': row['prompt'],
                'chosen_response': row['chosen_response'],
                'rejected_response': row['rejected_response']
            })
        
        return data
    
    def validate_data(self, data: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Validate preference data and return valid entries with error messages.
        
        Args:
            data: Raw preference data
            
        Returns:
            valid_data: Valid data entries
            errors: List of error messages
        """
        valid_data = []
        errors = []
        
        for i, item in enumerate(data):
            try:
                # Check required fields
                required_fields = ['prompt', 'chosen_response', 'rejected_response']
                for field in required_fields:
                    if field not in item:
                        raise ValueError(f"Missing required field: {field}")
                
                # Check for empty strings
                if not item['prompt'].strip() or not item['chosen_response'].strip() or not item['rejected_response'].strip():
                    raise ValueError("Empty prompt or response")
                
                # Check for duplicate responses
                if item['chosen_response'] == item['rejected_response']:
                    raise ValueError("Chosen and rejected responses are identical")
                
                # Check length limits
                if len(item['prompt']) > 1000 or len(item['chosen_response']) > 1000 or len(item['rejected_response']) > 1000:
                    raise ValueError("Text too long")
                
                valid_data.append(item)
                
            except Exception as e:
                errors.append(f"Item {i}: {str(e)}")
        
        return valid_data, errors
    
    def augment_data(self, data: List[Dict], augmentation_factor: int = 2) -> List[Dict]:
        """
        Augment preference data using various techniques.
        
        Args:
            data: Original preference data
            augmentation_factor: Number of augmented samples per original sample
            
        Returns:
            augmented_data: Augmented data
        """
        augmented_data = []
        
        for item in data:
            # Add original item
            augmented_data.append(item)
            
            # Generate augmented versions
            for _ in range(augmentation_factor - 1):
                augmented_item = self._augment_item(item)
                augmented_data.append(augmented_item)
        
        return augmented_data
    
    def _augment_item(self, item: Dict) -> Dict:
        """
        Augment a single preference item.
        
        Args:
            item: Original item
            
        Returns:
            augmented_item: Augmented item
        """
        # Simple augmentation techniques
        augmented_item = item.copy()
        
        # Random capitalization
        if random.random() < 0.3:
            augmented_item['prompt'] = augmented_item['prompt'].lower()
        
        # Add random punctuation
        if random.random() < 0.2:
            puncts = ['.', '!', '?']
            augmented_item['prompt'] += random.choice(puncts)
        
        # Synonym replacement (simplified)
        if random.random() < 0.1:
            synonyms = {
                'what': 'how',
                'explain': 'describe',
                'tell': 'show'
            }
            for word, synonym in synonyms.items():
                if word in augmented_item['prompt'].lower():
                    augmented_item['prompt'] = augmented_item['prompt'].replace(word, synonym)
                    break
        
        return augmented_item
    
    def create_train_val_split(self, data: List[Dict], val_ratio: float = 0.2,
                              random_state: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        Create train/validation split.
        
        Args:
            data: Full dataset
            val_ratio: Validation set ratio
            random_state: Random seed
            
        Returns:
            train_data: Training data
            val_data: Validation data
        """
        train_data, val_data = train_test_split(
            data, test_size=val_ratio, random_state=random_state
        )
        
        return train_data, val_data
    
    def create_dataloaders(self, train_data: List[Dict], val_data: List[Dict],
                          batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch dataloaders for training and validation.
        
        Args:
            train_data: Training data
            val_data: Validation data
            batch_size: Batch size
            
        Returns:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        train_dataset = PreferenceDataset(train_data, self.tokenizer, self.max_length)
        val_dataset = PreferenceDataset(val_data, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader


class PreferenceDataCollector:
    """
    Collector for preference data with quality control and bias mitigation.
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.collected_data = []
        self.quality_metrics = []
    
    def collect_binary_preferences(self, prompts: List[str], responses_a: List[str],
                                 responses_b: List[str], preferences: List[int]) -> List[Dict]:
        """
        Collect binary preference data.
        
        Args:
            prompts: Input prompts
            responses_a: First set of responses
            responses_b: Second set of responses
            preferences: Human preferences (0 for A, 1 for B)
            
        Returns:
            data: Preference data
        """
        data = []
        
        for prompt, resp_a, resp_b, pref in zip(prompts, responses_a, responses_b, preferences):
            if pref == 0:
                chosen_response = resp_a
                rejected_response = resp_b
            else:
                chosen_response = resp_b
                rejected_response = resp_a
            
            data.append({
                'prompt': prompt,
                'chosen_response': chosen_response,
                'rejected_response': rejected_response
            })
        
        return data
    
    def collect_ranking_data(self, prompts: List[str], response_sets: List[List[str]],
                           rankings: List[List[int]]) -> List[Dict]:
        """
        Collect ranking-based preference data.
        
        Args:
            prompts: Input prompts
            response_sets: Sets of responses for each prompt
            rankings: Human rankings for each set
            
        Returns:
            data: Ranking data
        """
        data = []
        
        for prompt, responses, ranking in zip(prompts, response_sets, rankings):
            # Sort responses by ranking
            sorted_responses = [responses[i] for i in ranking]
            
            data.append({
                'prompt': prompt,
                'responses': sorted_responses,
                'ranking': ranking
            })
        
        return data
    
    def quality_control(self, data: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        Apply quality control measures to preference data.
        
        Args:
            data: Raw preference data
            
        Returns:
            filtered_data: Quality-controlled data
            issues: List of quality issues
        """
        filtered_data = []
        issues = []
        
        for i, item in enumerate(data):
            item_issues = []
            
            # Check response length
            chosen_len = len(item['chosen_response'])
            rejected_len = len(item['rejected_response'])
            
            if chosen_len < 10 or rejected_len < 10:
                item_issues.append("Response too short")
            
            if chosen_len > 500 or rejected_len > 500:
                item_issues.append("Response too long")
            
            # Check for repetitive content
            if self._is_repetitive(item['chosen_response']) or self._is_repetitive(item['rejected_response']):
                item_issues.append("Repetitive content")
            
            # Check for inappropriate content (simplified)
            inappropriate_words = ['inappropriate', 'harmful', 'dangerous']
            for word in inappropriate_words:
                if word in item['chosen_response'].lower() or word in item['rejected_response'].lower():
                    item_issues.append("Potentially inappropriate content")
                    break
            
            if not item_issues:
                filtered_data.append(item)
            else:
                issues.append(f"Item {i}: {'; '.join(item_issues)}")
        
        return filtered_data, issues
    
    def _is_repetitive(self, text: str, threshold: float = 0.3) -> bool:
        """
        Check if text is repetitive.
        
        Args:
            text: Text to check
            threshold: Repetition threshold
            
        Returns:
            is_repetitive: Whether text is repetitive
        """
        words = text.split()
        if len(words) < 5:
            return False
        
        # Check for repeated phrases
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+2])
            count = text.count(phrase)
            if count > 2:
                return True
        
        return False
    
    def bias_detection(self, data: List[Dict]) -> Dict[str, float]:
        """
        Detect potential biases in preference data.
        
        Args:
            data: Preference data
            
        Returns:
            bias_metrics: Bias detection metrics
        """
        bias_metrics = {}
        
        # Length bias
        chosen_lengths = [len(item['chosen_response']) for item in data]
        rejected_lengths = [len(item['rejected_response']) for item in data]
        
        bias_metrics['length_bias'] = np.mean(chosen_lengths) - np.mean(rejected_lengths)
        
        # Content bias (simplified)
        positive_words = ['good', 'great', 'excellent', 'helpful', 'useful']
        negative_words = ['bad', 'poor', 'terrible', 'useless', 'unhelpful']
        
        chosen_positive = sum(1 for item in data 
                            if any(word in item['chosen_response'].lower() for word in positive_words))
        rejected_positive = sum(1 for item in data 
                              if any(word in item['rejected_response'].lower() for word in positive_words))
        
        bias_metrics['positive_bias'] = (chosen_positive - rejected_positive) / len(data)
        
        return bias_metrics


class PreferenceDataAnalyzer:
    """
    Analyzer for preference data with statistical analysis and visualization.
    """
    
    def __init__(self):
        self.analysis_results = {}
    
    def analyze_dataset(self, data: List[Dict]) -> Dict[str, Union[float, List]]:
        """
        Analyze preference dataset.
        
        Args:
            data: Preference data
            
        Returns:
            analysis: Dataset analysis
        """
        analysis = {}
        
        # Basic statistics
        analysis['total_samples'] = len(data)
        
        # Prompt analysis
        prompt_lengths = [len(item['prompt']) for item in data]
        analysis['avg_prompt_length'] = np.mean(prompt_lengths)
        analysis['std_prompt_length'] = np.std(prompt_lengths)
        
        # Response analysis
        chosen_lengths = [len(item['chosen_response']) for item in data]
        rejected_lengths = [len(item['rejected_response']) for item in data]
        
        analysis['avg_chosen_length'] = np.mean(chosen_lengths)
        analysis['avg_rejected_length'] = np.mean(rejected_lengths)
        analysis['length_difference'] = np.mean(chosen_lengths) - np.mean(rejected_lengths)
        
        # Content analysis
        analysis['unique_prompts'] = len(set(item['prompt'] for item in data))
        analysis['prompt_diversity'] = analysis['unique_prompts'] / analysis['total_samples']
        
        return analysis
    
    def compute_agreement_metrics(self, annotations: List[List[int]]) -> Dict[str, float]:
        """
        Compute inter-annotator agreement metrics.
        
        Args:
            annotations: List of annotations from multiple annotators
            
        Returns:
            metrics: Agreement metrics
        """
        if len(annotations) < 2:
            return {'agreement': 1.0}
        
        # Simple agreement calculation
        agreements = 0
        total_pairs = 0
        
        for i in range(len(annotations[0])):
            for j in range(i + 1, len(annotations[0])):
                annotator_agreements = 0
                for annotator in annotations:
                    if annotator[i] < annotator[j]:
                        annotator_agreements += 1
                
                if annotator_agreements == len(annotations) or annotator_agreements == 0:
                    agreements += 1
                total_pairs += 1
        
        agreement_rate = agreements / total_pairs if total_pairs > 0 else 1.0
        
        return {
            'agreement_rate': agreement_rate,
            'total_pairs': total_pairs,
            'agreements': agreements
        }


def create_preference_data_loader(data_path: str, tokenizer, batch_size: int = 8,
                                max_length: int = 512, val_ratio: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """
    Create preference data loaders from file.
    
    Args:
        data_path: Path to data file
        tokenizer: Tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        val_ratio: Validation set ratio
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
    """
    processor = PreferenceDataProcessor(tokenizer, max_length)
    
    # Load data
    if data_path.endswith('.json'):
        data = processor.load_from_json(data_path)
    elif data_path.endswith('.csv'):
        data = processor.load_from_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Validate data
    valid_data, errors = processor.validate_data(data)
    if errors:
        logger.warning(f"Found {len(errors)} data issues: {errors[:5]}")
    
    # Create train/val split
    train_data, val_data = processor.create_train_val_split(valid_data, val_ratio)
    
    # Create dataloaders
    train_loader, val_loader = processor.create_dataloaders(train_data, val_data, batch_size)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    
    # Create sample data
    sample_data = [
        {
            'prompt': 'What is machine learning?',
            'chosen_response': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data.',
            'rejected_response': 'Machine learning is cool.'
        },
        {
            'prompt': 'Explain neural networks.',
            'chosen_response': 'Neural networks are computational models inspired by biological neurons.',
            'rejected_response': 'Neural networks are networks.'
        }
    ]
    
    # Process data
    processor = PreferenceDataProcessor(tokenizer)
    train_data, val_data = processor.create_train_val_split(sample_data)
    train_loader, val_loader = processor.create_dataloaders(train_data, val_data)
    
    print(f"Created dataloaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    # Analyze data
    analyzer = PreferenceDataAnalyzer()
    analysis = analyzer.analyze_dataset(sample_data)
    print(f"Dataset analysis: {analysis}") 
"""
Text Classification Implementation
================================

This module provides BERT-style text classification implementations
for various NLP tasks like sentiment analysis, topic classification,
and intent detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, classification_report
import json


class TextClassificationDataset(Dataset):
    """
    Dataset for text classification tasks.
    
    This handles tokenization and formatting for classification tasks.
    """
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, 
                 max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTClassifier(nn.Module):
    """
    BERT-based classifier for text classification tasks.
    
    This uses a pre-trained BERT model with a classification head
    on top for various NLP tasks.
    """
    
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BERT classifier.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for training
        
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }


class DistilBERTClassifier(nn.Module):
    """
    DistilBERT-based classifier for faster inference.
    
    This uses DistilBERT which is a distilled version of BERT
    that is faster and smaller while maintaining good performance.
    """
    
    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_classes)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of DistilBERT classifier.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels for training
        
        Returns:
            Dictionary containing logits and loss (if labels provided)
        """
        # Get DistilBERT outputs
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use [CLS] token for classification
        cls_output = hidden_states[:, 0, :]
        
        # Apply dropout and classification
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }


class TextClassificationTrainer:
    """
    Trainer for text classification models.
    
    This provides training, evaluation, and prediction functionality
    for text classification tasks.
    """
    
    def __init__(self, model: nn.Module, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
              optimizer, num_epochs: int = 3, save_path: str = None):
        """
        Train the classification model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            num_epochs: Number of training epochs
            save_path: Path to save best model
        """
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            total_loss = 0
            
            for batch in train_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            val_acc = self.evaluate(val_dataloader)
            avg_loss = total_loss / len(train_dataloader)
            
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Training Loss: {avg_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"Best model saved with accuracy: {val_acc:.4f}")
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: Data loader for evaluation
        
        Returns:
            Accuracy score
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch.pop('labels')
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs['logits']
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_predictions)
        return accuracy
    
    def predict(self, texts: List[str], batch_size: int = 32) -> List[int]:
        """
        Make predictions on new texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for prediction
        
        Returns:
            List of predicted class labels
        """
        self.model.eval()
        all_predictions = []
        
        # Create dataset and dataloader
        dataset = TextClassificationDataset(texts, [0] * len(texts), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch.pop('labels')  # Remove labels for prediction
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs['logits']
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
        
        return all_predictions


class SentimentAnalyzer:
    """
    Sentiment analysis classifier.
    
    This provides a complete pipeline for sentiment analysis
    using pre-trained transformer models.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DistilBERTClassifier(model_name, num_classes=3)  # Positive, Negative, Neutral
        self.trainer = TextClassificationTrainer(self.model, self.tokenizer)
        
        # Class labels
        self.labels = ['negative', 'neutral', 'positive']
    
    def train(self, texts: List[str], labels: List[int], 
              val_texts: List[str] = None, val_labels: List[int] = None,
              num_epochs: int = 3, batch_size: int = 16):
        """
        Train the sentiment analyzer.
        
        Args:
            texts: Training texts
            labels: Training labels (0: negative, 1: neutral, 2: positive)
            val_texts: Validation texts
            val_labels: Validation labels
            num_epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Create datasets
        train_dataset = TextClassificationDataset(texts, labels, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_texts and val_labels:
            val_dataset = TextClassificationDataset(val_texts, val_labels, self.tokenizer)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_dataloader = None
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Train
        self.trainer.train(train_dataloader, val_dataloader, optimizer, num_epochs)
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for given texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of prediction dictionaries with label and confidence
        """
        predictions = self.trainer.predict(texts)
        
        results = []
        for text, pred in zip(texts, predictions):
            results.append({
                'text': text,
                'sentiment': self.labels[pred],
                'label': pred
            })
        
        return results


class TopicClassifier:
    """
    Topic classification model.
    
    This provides a complete pipeline for topic classification
    using pre-trained transformer models.
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 topics: List[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.topics = topics or ['technology', 'sports', 'politics', 'entertainment', 'science']
        self.model = DistilBERTClassifier(model_name, num_classes=len(self.topics))
        self.trainer = TextClassificationTrainer(self.model, self.tokenizer)
    
    def train(self, texts: List[str], labels: List[int], 
              val_texts: List[str] = None, val_labels: List[int] = None,
              num_epochs: int = 3, batch_size: int = 16):
        """
        Train the topic classifier.
        
        Args:
            texts: Training texts
            labels: Training labels (indices corresponding to topics)
            val_texts: Validation texts
            val_labels: Validation labels
            num_epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Create datasets
        train_dataset = TextClassificationDataset(texts, labels, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_texts and val_labels:
            val_dataset = TextClassificationDataset(val_texts, val_labels, self.tokenizer)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_dataloader = None
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        # Train
        self.trainer.train(train_dataloader, val_dataloader, optimizer, num_epochs)
    
    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict topics for given texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of prediction dictionaries with topic and confidence
        """
        predictions = self.trainer.predict(texts)
        
        results = []
        for text, pred in zip(texts, predictions):
            results.append({
                'text': text,
                'topic': self.topics[pred],
                'label': pred
            })
        
        return results


# Example usage
if __name__ == "__main__":
    # Example sentiment analysis
    sentiment_analyzer = SentimentAnalyzer()
    
    # Sample data
    texts = [
        "I love this product! It's amazing.",
        "This is terrible, I hate it.",
        "It's okay, nothing special.",
        "The best thing I've ever bought!",
        "Worst purchase ever, don't buy it."
    ]
    
    labels = [2, 0, 1, 2, 0]  # positive, negative, neutral, positive, negative
    
    # Train the model
    print("Training sentiment analyzer...")
    sentiment_analyzer.train(texts, labels, num_epochs=2)
    
    # Make predictions
    test_texts = [
        "This movie is fantastic!",
        "I'm disappointed with the service.",
        "The food was average."
    ]
    
    predictions = sentiment_analyzer.predict(test_texts)
    for pred in predictions:
        print(f"Text: {pred['text']}")
        print(f"Sentiment: {pred['sentiment']}")
        print()
    
    # Example topic classification
    topic_classifier = TopicClassifier()
    
    # Sample data
    topic_texts = [
        "The new iPhone has amazing features.",
        "The team won the championship game.",
        "The president announced new policies.",
        "The movie received great reviews.",
        "Scientists discovered a new planet."
    ]
    
    topic_labels = [0, 1, 2, 3, 4]  # technology, sports, politics, entertainment, science
    
    # Train the model
    print("Training topic classifier...")
    topic_classifier.train(topic_texts, topic_labels, num_epochs=2)
    
    # Make predictions
    test_topic_texts = [
        "The latest Android update is available.",
        "The basketball game was intense.",
        "The new law was passed today."
    ]
    
    topic_predictions = topic_classifier.predict(test_topic_texts)
    for pred in topic_predictions:
        print(f"Text: {pred['text']}")
        print(f"Topic: {pred['topic']}")
        print() 
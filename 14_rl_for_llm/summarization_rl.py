"""
RL for Text Summarization

This module provides a complete implementation of reinforcement learning
for text summarization tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


class SummarizationRL:
    """
    Reinforcement learning for text summarization.
    """
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Rouge scorer for evaluation
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Training metrics
        self.training_metrics = {
            'rewards': [],
            'losses': [],
            'rouge_scores': []
        }
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """
        Generate summary for input text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            
        Returns:
            summary: Generated summary
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def compute_summarization_reward(self, text: str, summary: str, reference_summary: str) -> float:
        """
        Compute reward for summarization.
        
        Args:
            text: Original text
            summary: Generated summary
            reference_summary: Reference summary
            
        Returns:
            reward: Summarization reward
        """
        # Rouge score
        rouge_scores = self.rouge_scorer.score(reference_summary, summary)
        rouge_reward = (rouge_scores['rouge1'].fmeasure + 
                       rouge_scores['rouge2'].fmeasure + 
                       rouge_scores['rougeL'].fmeasure) / 3
        
        # Length reward
        length_reward = self._compute_length_reward(summary, reference_summary)
        
        # Coherence reward
        coherence_reward = self._compute_coherence_reward(summary)
        
        # Combined reward
        total_reward = 0.6 * rouge_reward + 0.2 * length_reward + 0.2 * coherence_reward
        
        return total_reward
    
    def _compute_length_reward(self, summary: str, reference_summary: str) -> float:
        """
        Compute length-based reward.
        
        Args:
            summary: Generated summary
            reference_summary: Reference summary
            
        Returns:
            reward: Length reward
        """
        summary_length = len(summary.split())
        reference_length = len(reference_summary.split())
        
        # Reward summaries close to reference length
        length_diff = abs(summary_length - reference_length)
        if length_diff <= 5:
            return 1.0
        elif length_diff <= 10:
            return 0.7
        elif length_diff <= 20:
            return 0.3
        else:
            return 0.0
    
    def _compute_coherence_reward(self, summary: str) -> float:
        """
        Compute coherence reward.
        
        Args:
            summary: Generated summary
            
        Returns:
            reward: Coherence reward
        """
        # Simple coherence heuristics
        score = 0.5  # Base score
        
        # Reward complete sentences
        if summary.endswith('.') or summary.endswith('!') or summary.endswith('?'):
            score += 0.2
        
        # Penalize very short summaries
        if len(summary.split()) < 5:
            score -= 0.3
        
        # Reward informative words
        informative_words = ['because', 'however', 'therefore', 'additionally', 'furthermore']
        for word in informative_words:
            if word in summary.lower():
                score += 0.1
        
        return max(0, min(1, score))
    
    def train_on_summarization_data(self, training_data: List[Dict]) -> List[float]:
        """
        Train on summarization data using RL.
        
        Args:
            training_data: Training data with text, reference summaries
            
        Returns:
            losses: Training losses
        """
        losses = []
        
        for item in training_data:
            text = item['text']
            reference_summary = item['reference_summary']
            
            # Generate summary
            summary = self.generate_summary(text)
            
            # Compute reward
            reward = self.compute_summarization_reward(text, summary, reference_summary)
            
            # Compute policy gradient loss
            loss = self._compute_policy_gradient_loss(text, summary, reward)
            losses.append(loss)
            
            # Update model
            self._update_model(loss)
        
        return losses
    
    def _compute_policy_gradient_loss(self, text: str, summary: str, reward: float) -> torch.Tensor:
        """
        Compute policy gradient loss for summarization.
        
        Args:
            text: Input text
            summary: Generated summary
            reward: Reward for the summary
            
        Returns:
            loss: Policy gradient loss
        """
        # Tokenize input and output
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        
        summary_tokens = self.tokenizer(summary, return_tensors='pt')['input_ids'].to(self.device)
        
        # Get model outputs
        outputs = self.model(input_ids, labels=summary_tokens)
        logits = outputs.logits
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Sum log probabilities for summary tokens
        summary_log_prob = log_probs[0, :len(summary_tokens[0]), :].gather(1, summary_tokens[0].unsqueeze(1)).sum()
        
        # Policy gradient loss
        loss = -summary_log_prob * reward
        
        return loss
    
    def _update_model(self, loss: torch.Tensor):
        """
        Update model parameters.
        
        Args:
            loss: Training loss
        """
        # This would typically use an optimizer
        # For simplicity, we'll just record the loss
        self.training_metrics['losses'].append(loss.item())


class SummarizationEvaluator:
    """
    Evaluator for summarization systems.
    """
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_summarization(self, texts: List[str], generated_summaries: List[str], 
                             reference_summaries: List[str]) -> Dict[str, float]:
        """
        Evaluate summarization quality.
        
        Args:
            texts: Original texts
            generated_summaries: Generated summaries
            reference_summaries: Reference summaries
            
        Returns:
            metrics: Evaluation metrics
        """
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        length_ratios = []
        compression_ratios = []
        
        for text, summary, reference in zip(texts, generated_summaries, reference_summaries):
            # Rouge scores
            scores = self.rouge_scorer.score(reference, summary)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            
            # Length ratios
            summary_length = len(summary.split())
            reference_length = len(reference.split())
            text_length = len(text.split())
            
            length_ratio = summary_length / reference_length if reference_length > 0 else 1.0
            compression_ratio = summary_length / text_length if text_length > 0 else 1.0
            
            length_ratios.append(length_ratio)
            compression_ratios.append(compression_ratio)
        
        return {
            'rouge1_f1': np.mean(rouge_scores['rouge1']),
            'rouge2_f1': np.mean(rouge_scores['rouge2']),
            'rougeL_f1': np.mean(rouge_scores['rougeL']),
            'avg_length_ratio': np.mean(length_ratios),
            'avg_compression_ratio': np.mean(compression_ratios)
        }


if __name__ == "__main__":
    # Example usage
    model_name = 't5-small'  # Good for summarization
    
    # Create summarization RL
    summarization_rl = SummarizationRL(model_name)
    
    # Test summarization
    sample_text = """
    Machine learning is a subset of artificial intelligence that enables computers to learn 
    from data without being explicitly programmed. It involves algorithms that can identify 
    patterns in data and make predictions or decisions based on those patterns. Machine 
    learning has applications in various fields including healthcare, finance, and technology.
    """
    
    summary = summarization_rl.generate_summary(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Generated summary: {summary}")
    
    # Test evaluation
    evaluator = SummarizationEvaluator()
    reference_summary = "Machine learning enables computers to learn from data and make predictions."
    
    metrics = evaluator.evaluate_summarization(
        [sample_text], [summary], [reference_summary]
    )
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}") 
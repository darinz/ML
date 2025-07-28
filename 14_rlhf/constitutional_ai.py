"""
Constitutional AI Framework Implementation

This module provides a complete implementation of Constitutional AI for self-critique
and revision of language model responses.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import json

logger = logging.getLogger(__name__)


class ConstitutionalAI:
    """
    Constitutional AI framework for self-critique and revision.
    
    This framework enables language models to evaluate and revise their own
    responses based on predefined principles.
    """
    
    def __init__(self, model: nn.Module, tokenizer, principles: List[str],
                 max_iterations: int = 3, device: str = 'cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.principles = principles
        self.max_iterations = max_iterations
        self.device = device
        
        # Set model to evaluation mode
        self.model.eval()
    
    def generate_aligned_response(self, prompt: str) -> Dict[str, str]:
        """
        Generate response using Constitutional AI framework.
        
        Args:
            prompt: Input prompt
            
        Returns:
            result: Dictionary containing final response and intermediate steps
        """
        # Step 1: Generate initial response
        initial_response = self._generate_response(prompt)
        
        # Step 2-4: Iterative critique and revision
        current_response = initial_response
        iterations = []
        
        for iteration in range(self.max_iterations):
            # Self-critique
            critique = self._self_critique(prompt, current_response)
            
            # Check if response is satisfactory
            if self._is_satisfactory(critique):
                break
            
            # Revise response
            revised_response = self._revise_response(prompt, current_response, critique)
            iterations.append({
                'iteration': iteration + 1,
                'response': current_response,
                'critique': critique,
                'revised_response': revised_response
            })
            
            current_response = revised_response
        
        return {
            'final_response': current_response,
            'initial_response': initial_response,
            'iterations': iterations,
            'num_iterations': len(iterations)
        }
    
    def _generate_response(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate initial response for a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum response length
            
        Returns:
            response: Generated response
        """
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _self_critique(self, prompt: str, response: str) -> str:
        """
        Generate self-critique based on principles.
        
        Args:
            prompt: Input prompt
            response: Current response
            
        Returns:
            critique: Self-critique
        """
        critique_prompt = f"""
        Evaluate the following response based on these principles:
        {self._format_principles()}
        
        Prompt: {prompt}
        Response: {response}
        
        Provide a critique of the response, identifying any issues with:
        1. Helpfulness - Does it address the user's need?
        2. Harmlessness - Is it safe and appropriate?
        3. Honesty - Is it truthful and accurate?
        4. Clarity - Is it clear and well-structured?
        
        Critique:
        """
        
        critique = self._generate_response(critique_prompt, max_length=200)
        return critique
    
    def _is_satisfactory(self, critique: str) -> bool:
        """
        Determine if response is satisfactory based on critique.
        
        Args:
            critique: Self-critique
            
        Returns:
            satisfactory: Boolean indicating if response is satisfactory
        """
        # Simple heuristic: check for positive indicators
        positive_indicators = [
            'good', 'appropriate', 'helpful', 'safe', 'honest', 'clear',
            'satisfactory', 'adequate', 'meets', 'addresses', 'correct'
        ]
        negative_indicators = [
            'harmful', 'inappropriate', 'misleading', 'unsafe', 'unclear',
            'incomplete', 'incorrect', 'problematic', 'concerning'
        ]
        
        critique_lower = critique.lower()
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in critique_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in critique_lower)
        
        # Response is satisfactory if positive indicators outweigh negative ones
        return positive_score > negative_score
    
    def _revise_response(self, prompt: str, current_response: str, critique: str) -> str:
        """
        Revise response based on critique.
        
        Args:
            prompt: Input prompt
            current_response: Current response
            critique: Self-critique
            
        Returns:
            revised_response: Revised response
        """
        revision_prompt = f"""
        Based on the following critique, revise the response to better align with the principles:
        {self._format_principles()}
        
        Original Prompt: {prompt}
        Original Response: {current_response}
        Critique: {critique}
        
        Provide a revised response that addresses the issues identified in the critique:
        """
        
        revised_response = self._generate_response(revision_prompt, max_length=150)
        return revised_response
    
    def _format_principles(self) -> str:
        """
        Format principles for prompts.
        
        Returns:
            principle_text: Formatted principles
        """
        principle_text = ""
        for i, principle in enumerate(self.principles, 1):
            principle_text += f"{i}. {principle}\n"
        return principle_text
    
    def batch_generate(self, prompts: List[str]) -> List[Dict[str, str]]:
        """
        Generate aligned responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            
        Returns:
            results: List of generation results
        """
        results = []
        for prompt in prompts:
            result = self.generate_aligned_response(prompt)
            results.append(result)
        
        return results
    
    def evaluate_alignment(self, test_prompts: List[str]) -> Dict[str, float]:
        """
        Evaluate alignment of generated responses.
        
        Args:
            test_prompts: Test prompts
            
        Returns:
            metrics: Alignment evaluation metrics
        """
        results = self.batch_generate(test_prompts)
        
        # Compute metrics
        avg_iterations = np.mean([r['num_iterations'] for r in results])
        satisfactory_count = sum(1 for r in results if r['num_iterations'] < self.max_iterations)
        satisfactory_rate = satisfactory_count / len(results) if results else 0
        
        # Compute response quality metrics
        response_lengths = [len(r['final_response'].split()) for r in results]
        avg_length = np.mean(response_lengths)
        
        return {
            'avg_iterations': avg_iterations,
            'satisfactory_rate': satisfactory_rate,
            'avg_response_length': avg_length,
            'total_prompts': len(test_prompts)
        }


class MultiAgentConstitutionalAI:
    """
    Multi-agent Constitutional AI with separate generator, critic, and reviser.
    """
    
    def __init__(self, generator: nn.Module, critic: nn.Module, reviser: nn.Module,
                 tokenizer, principles: List[str], max_iterations: int = 3, device: str = 'cuda'):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.reviser = reviser.to(device)
        self.tokenizer = tokenizer
        self.principles = principles
        self.max_iterations = max_iterations
        self.device = device
        
        # Set models to evaluation mode
        self.generator.eval()
        self.critic.eval()
        self.reviser.eval()
    
    def generate_aligned_response(self, prompt: str) -> Dict[str, str]:
        """
        Generate response using multi-agent Constitutional AI.
        
        Args:
            prompt: Input prompt
            
        Returns:
            result: Generation result
        """
        # Generate initial response
        initial_response = self._generate_response(prompt)
        
        # Iterative critique and revision
        current_response = initial_response
        iterations = []
        
        for iteration in range(self.max_iterations):
            # Generate critique
            critique = self._generate_critique(prompt, current_response)
            
            # Check if satisfactory
            if self._is_satisfactory(critique):
                break
            
            # Revise response
            revised_response = self._generate_revision(prompt, current_response, critique)
            iterations.append({
                'iteration': iteration + 1,
                'response': current_response,
                'critique': critique,
                'revised_response': revised_response
            })
            
            current_response = revised_response
        
        return {
            'final_response': current_response,
            'initial_response': initial_response,
            'iterations': iterations,
            'num_iterations': len(iterations)
        }
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using generator model."""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.generator.generate(
                input_ids,
                max_length=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def _generate_critique(self, prompt: str, response: str) -> str:
        """Generate critique using critic model."""
        critique_prompt = f"""
        Evaluate this response based on the principles:
        {self._format_principles()}
        
        Prompt: {prompt}
        Response: {response}
        
        Critique:
        """
        
        inputs = self.tokenizer(critique_prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.critic.generate(
                input_ids,
                max_length=200,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        critique = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return critique
    
    def _generate_revision(self, prompt: str, response: str, critique: str) -> str:
        """Generate revision using reviser model."""
        revision_prompt = f"""
        Revise this response based on the critique:
        
        Prompt: {prompt}
        Original Response: {response}
        Critique: {critique}
        
        Revised Response:
        """
        
        inputs = self.tokenizer(revision_prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.reviser.generate(
                input_ids,
                max_length=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        revision = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return revision
    
    def _is_satisfactory(self, critique: str) -> bool:
        """Check if critique indicates satisfactory response."""
        positive_indicators = ['good', 'appropriate', 'helpful', 'safe', 'honest']
        negative_indicators = ['harmful', 'inappropriate', 'misleading', 'unsafe']
        
        critique_lower = critique.lower()
        
        positive_score = sum(1 for indicator in positive_indicators if indicator in critique_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in critique_lower)
        
        return positive_score > negative_score
    
    def _format_principles(self) -> str:
        """Format principles for prompts."""
        principle_text = ""
        for i, principle in enumerate(self.principles, 1):
            principle_text += f"{i}. {principle}\n"
        return principle_text


class IterativeConstitutionalAI(ConstitutionalAI):
    """
    Iterative Constitutional AI with continuous improvement.
    """
    
    def __init__(self, model: nn.Module, tokenizer, principles: List[str],
                 max_iterations: int = 5, improvement_threshold: float = 0.1,
                 device: str = 'cuda'):
        super().__init__(model, tokenizer, principles, max_iterations, device)
        self.improvement_threshold = improvement_threshold
    
    def iterative_improvement(self, prompt: str) -> Dict[str, Union[List[str], List[float]]]:
        """
        Iteratively improve response quality.
        
        Args:
            prompt: Input prompt
            
        Returns:
            result: Iterative improvement results
        """
        responses = []
        scores = []
        
        # Generate initial response
        current_response = self._generate_response(prompt)
        responses.append(current_response)
        
        for iteration in range(self.max_iterations):
            # Evaluate current response
            current_score = self._evaluate_response(prompt, current_response)
            scores.append(current_score)
            
            # Check for improvement
            if len(scores) > 1:
                improvement = scores[-1] - scores[-2]
                if improvement < self.improvement_threshold:
                    break
            
            # Generate critique and revise
            critique = self._self_critique(prompt, current_response)
            current_response = self._revise_response(prompt, current_response, critique)
            responses.append(current_response)
        
        return {
            'responses': responses,
            'scores': scores,
            'final_response': current_response,
            'improvement_history': [scores[i] - scores[i-1] for i in range(1, len(scores))]
        }
    
    def _evaluate_response(self, prompt: str, response: str) -> float:
        """
        Evaluate response quality.
        
        Args:
            prompt: Input prompt
            response: Response to evaluate
            
        Returns:
            score: Quality score
        """
        # Simple evaluation based on length and content
        score = 0.5  # Base score
        
        # Reward appropriate length
        word_count = len(response.split())
        if 10 <= word_count <= 100:
            score += 0.2
        elif word_count < 5:
            score -= 0.3
        
        # Reward relevant content
        relevant_keywords = ['because', 'example', 'explain', 'how', 'what', 'why']
        if any(keyword in response.lower() for keyword in relevant_keywords):
            score += 0.2
        
        # Reward clarity indicators
        clarity_indicators = ['first', 'second', 'finally', 'therefore', 'however']
        if any(indicator in response.lower() for indicator in clarity_indicators):
            score += 0.1
        
        return max(0, min(1, score))


class ConstitutionalAITrainer:
    """
    Trainer for Constitutional AI models.
    """
    
    def __init__(self, model: nn.Module, tokenizer, principles: List[str],
                 learning_rate: float = 1e-5, device: str = 'cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.principles = principles
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    def train_on_critique_data(self, training_data: List[Dict]) -> List[float]:
        """
        Train model on critique-response pairs.
        
        Args:
            training_data: Training data with prompts, responses, and critiques
            
        Returns:
            losses: Training losses
        """
        losses = []
        
        for item in training_data:
            prompt = item['prompt']
            response = item['response']
            critique = item['critique']
            
            # Create training example
            training_text = f"Prompt: {prompt}\nResponse: {response}\nCritique: {critique}"
            
            # Tokenize
            inputs = self.tokenizer(training_text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Compute loss (language modeling)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.append(loss.item())
        
        return losses
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'principles': self.principles
        }, path)
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# Default principles for Constitutional AI
DEFAULT_PRINCIPLES = [
    "Helpfulness: Provide useful and relevant information",
    "Harmlessness: Avoid harmful or inappropriate content",
    "Honesty: Be truthful and accurate",
    "Transparency: Acknowledge limitations and uncertainties"
]


if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model and tokenizer
    model_name = 'gpt2'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create Constitutional AI
    constitutional_ai = ConstitutionalAI(model, tokenizer, DEFAULT_PRINCIPLES)
    
    # Test generation
    prompt = "What is the best way to learn machine learning?"
    result = constitutional_ai.generate_aligned_response(prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Final Response: {result['final_response']}")
    print(f"Number of iterations: {result['num_iterations']}")
    
    # Test batch generation
    prompts = [
        "Explain quantum computing",
        "What are the benefits of exercise?",
        "How do I start a business?"
    ]
    
    results = constitutional_ai.batch_generate(prompts)
    for i, result in enumerate(results):
        print(f"\nPrompt {i+1}: {prompts[i]}")
        print(f"Response: {result['final_response']}")
    
    # Evaluate alignment
    metrics = constitutional_ai.evaluate_alignment(prompts)
    print(f"\nAlignment metrics: {metrics}") 
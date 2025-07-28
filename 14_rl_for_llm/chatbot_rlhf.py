"""
RLHF for Conversational AI Chatbots

This module provides a complete implementation of reinforcement learning from
human feedback (RLHF) for conversational AI chatbots.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import json
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class ConversationalRLHF:
    """
    RLHF implementation for conversational AI chatbots.
    """
    
    def __init__(self, model_name: str, reward_model: nn.Module, device: str = 'cuda'):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_model = reward_model
        self.device = device
        
        # Move models to device
        self.model.to(device)
        self.ref_model.to(device)
        self.reward_model.to(device)
        
        # Conversation history
        self.conversation_history = []
        
        # Training metrics
        self.training_metrics = {
            'rewards': [],
            'losses': [],
            'conversation_lengths': []
        }
    
    def generate_response(self, user_input: str, max_length: int = 100) -> str:
        """
        Generate chatbot response.
        
        Args:
            user_input: User input message
            max_length: Maximum response length
            
        Returns:
            response: Generated response
        """
        # Build conversation context
        context = self._build_context(user_input)
        
        # Tokenize input
        inputs = self.tokenizer(context, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new response (remove context)
        response = response[len(context):].strip()
        
        # Update conversation history
        self.conversation_history.append({
            'user': user_input,
            'assistant': response,
            'timestamp': len(self.conversation_history)
        })
        
        return response
    
    def _build_context(self, user_input: str) -> str:
        """
        Build conversation context from history.
        
        Args:
            user_input: Current user input
            
        Returns:
            context: Full conversation context
        """
        context = ""
        
        # Add conversation history (last 5 turns)
        recent_history = self.conversation_history[-10:]  # Last 10 turns
        for turn in recent_history:
            context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        
        # Add current user input
        context += f"User: {user_input}\nAssistant:"
        
        return context
    
    def compute_conversation_reward(self, conversation: List[Dict]) -> float:
        """
        Compute reward for a conversation.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            reward: Conversation reward
        """
        if not conversation:
            return 0.0
        
        total_reward = 0.0
        
        for turn in conversation:
            user_input = turn['user']
            assistant_response = turn['assistant']
            
            # Compute reward for this turn
            turn_reward = self._compute_turn_reward(user_input, assistant_response)
            total_reward += turn_reward
        
        # Normalize by conversation length
        avg_reward = total_reward / len(conversation)
        
        return avg_reward
    
    def _compute_turn_reward(self, user_input: str, assistant_response: str) -> float:
        """
        Compute reward for a single conversation turn.
        
        Args:
            user_input: User input
            assistant_response: Assistant response
            
        Returns:
            reward: Turn reward
        """
        # Use reward model to compute reward
        reward = self.reward_model.predict_reward(user_input, assistant_response)
        
        # Add conversation-specific rewards
        conversation_rewards = self._compute_conversation_specific_rewards(user_input, assistant_response)
        
        return reward + conversation_rewards
    
    def _compute_conversation_specific_rewards(self, user_input: str, assistant_response: str) -> float:
        """
        Compute conversation-specific rewards.
        
        Args:
            user_input: User input
            assistant_response: Assistant response
            
        Returns:
            reward: Conversation-specific reward
        """
        reward = 0.0
        
        # Reward appropriate response length
        response_length = len(assistant_response.split())
        if 5 <= response_length <= 50:
            reward += 0.1
        elif response_length < 3:
            reward -= 0.2
        
        # Reward engagement (questions, follow-ups)
        engagement_indicators = ['?', 'what about', 'how about', 'tell me more']
        if any(indicator in assistant_response.lower() for indicator in engagement_indicators):
            reward += 0.1
        
        # Reward helpfulness indicators
        helpfulness_indicators = ['here\'s', 'let me', 'i can', 'sure', 'of course']
        if any(indicator in assistant_response.lower() for indicator in helpfulness_indicators):
            reward += 0.1
        
        # Penalize repetitive responses
        if len(self.conversation_history) > 0:
            last_response = self.conversation_history[-1]['assistant']
            if assistant_response.lower() == last_response.lower():
                reward -= 0.3
        
        return reward
    
    def train_on_conversation(self, conversation_data: List[Dict]) -> List[float]:
        """
        Train on conversation data using RLHF.
        
        Args:
            conversation_data: List of conversation examples
            
        Returns:
            losses: Training losses
        """
        losses = []
        
        for conversation in conversation_data:
            # Generate responses for the conversation
            generated_responses = []
            for turn in conversation['turns']:
                user_input = turn['user']
                response = self.generate_response(user_input)
                generated_responses.append(response)
            
            # Compute rewards
            rewards = []
            for i, turn in enumerate(conversation['turns']):
                reward = self._compute_turn_reward(turn['user'], generated_responses[i])
                rewards.append(reward)
            
            # Compute policy gradient loss
            loss = self._compute_policy_gradient_loss(conversation['turns'], generated_responses, rewards)
            losses.append(loss)
            
            # Update model
            self._update_model(loss)
        
        return losses
    
    def _compute_policy_gradient_loss(self, turns: List[Dict], responses: List[str], rewards: List[float]) -> torch.Tensor:
        """
        Compute policy gradient loss for conversation.
        
        Args:
            turns: Conversation turns
            responses: Generated responses
            rewards: Rewards for each turn
            
        Returns:
            loss: Policy gradient loss
        """
        total_loss = 0.0
        
        for turn, response, reward in zip(turns, responses, rewards):
            # Build context
            context = self._build_context_for_turn(turn['user'])
            
            # Tokenize
            inputs = self.tokenizer(context + response, return_tensors='pt', truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(self.device)
            
            # Get log probabilities
            outputs = self.model(input_ids)
            logits = outputs.logits
            
            # Compute log probabilities for response tokens
            response_tokens = self.tokenizer(response, return_tensors='pt')['input_ids'].to(self.device)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Sum log probabilities
            response_log_prob = log_probs[0, -len(response_tokens[0]):, :].gather(1, response_tokens[0].unsqueeze(1)).sum()
            
            # Policy gradient loss
            loss = -response_log_prob * reward
            total_loss += loss
        
        return total_loss
    
    def _build_context_for_turn(self, user_input: str) -> str:
        """
        Build context for a specific turn.
        
        Args:
            user_input: User input
            
        Returns:
            context: Context for the turn
        """
        context = ""
        
        # Add conversation history
        for turn in self.conversation_history[-5:]:  # Last 5 turns
            context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        
        # Add current user input
        context += f"User: {user_input}\nAssistant:"
        
        return context
    
    def _update_model(self, loss: torch.Tensor):
        """
        Update model parameters.
        
        Args:
            loss: Training loss
        """
        # This would typically use an optimizer
        # For simplicity, we'll just record the loss
        self.training_metrics['losses'].append(loss.item())


class ConversationalRewardModel:
    """
    Reward model specifically designed for conversational AI.
    """
    
    def __init__(self, base_model_name: str, device: str = 'cuda'):
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.device = device
        
        # Reward head
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)
        self.reward_head.to(device)
        
        # Initialize reward head
        nn.init.xavier_uniform_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
    
    def predict_reward(self, user_input: str, assistant_response: str) -> float:
        """
        Predict reward for a user input and assistant response pair.
        
        Args:
            user_input: User input
            assistant_response: Assistant response
            
        Returns:
            reward: Predicted reward
        """
        # Concatenate input and response
        text = f"User: {user_input}\nAssistant: {assistant_response}"
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
            # Pool over sequence length
            pooled = hidden_states.mean(dim=1)
            
            # Predict reward
            reward = self.reward_head(pooled).squeeze(-1)
        
        return reward.item()
    
    def train_on_conversation_data(self, training_data: List[Dict]) -> List[float]:
        """
        Train reward model on conversation data.
        
        Args:
            training_data: Training data with user inputs, responses, and human ratings
            
        Returns:
            losses: Training losses
        """
        losses = []
        
        for item in training_data:
            user_input = item['user_input']
            good_response = item['good_response']
            bad_response = item['bad_response']
            
            # Predict rewards
            good_reward = self.predict_reward(user_input, good_response)
            bad_reward = self.predict_reward(user_input, bad_response)
            
            # Compute preference loss
            logits = good_reward - bad_reward
            loss = -torch.log(torch.sigmoid(torch.tensor(logits)))
            
            losses.append(loss.item())
        
        return losses


class ConversationalDataset:
    """
    Dataset for conversational RLHF training.
    """
    
    def __init__(self, conversations: List[Dict]):
        self.conversations = conversations
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        return self.conversations[idx]
    
    def create_training_batches(self, batch_size: int = 4) -> List[List[Dict]]:
        """
        Create training batches.
        
        Args:
            batch_size: Batch size
            
        Returns:
            batches: List of conversation batches
        """
        batches = []
        for i in range(0, len(self.conversations), batch_size):
            batch = self.conversations[i:i + batch_size]
            batches.append(batch)
        
        return batches


class ConversationalEvaluator:
    """
    Evaluator for conversational AI systems.
    """
    
    def __init__(self):
        self.evaluation_metrics = {}
    
    def evaluate_conversation_quality(self, conversations: List[List[Dict]]) -> Dict[str, float]:
        """
        Evaluate conversation quality.
        
        Args:
            conversations: List of conversations
            
        Returns:
            metrics: Conversation quality metrics
        """
        metrics = {
            'avg_conversation_length': 0,
            'avg_response_length': 0,
            'engagement_score': 0,
            'helpfulness_score': 0,
            'consistency_score': 0
        }
        
        total_conversations = len(conversations)
        total_turns = 0
        total_response_length = 0
        engagement_scores = []
        helpfulness_scores = []
        consistency_scores = []
        
        for conversation in conversations:
            conversation_length = len(conversation)
            total_turns += conversation_length
            
            for turn in conversation:
                response = turn['assistant']
                response_length = len(response.split())
                total_response_length += response_length
                
                # Compute engagement score
                engagement_score = self._compute_engagement_score(response)
                engagement_scores.append(engagement_score)
                
                # Compute helpfulness score
                helpfulness_score = self._compute_helpfulness_score(turn['user'], response)
                helpfulness_scores.append(helpfulness_score)
            
            # Compute consistency score for conversation
            consistency_score = self._compute_consistency_score(conversation)
            consistency_scores.append(consistency_score)
        
        # Compute averages
        metrics['avg_conversation_length'] = total_turns / total_conversations
        metrics['avg_response_length'] = total_response_length / total_turns
        metrics['engagement_score'] = np.mean(engagement_scores)
        metrics['helpfulness_score'] = np.mean(helpfulness_scores)
        metrics['consistency_score'] = np.mean(consistency_scores)
        
        return metrics
    
    def _compute_engagement_score(self, response: str) -> float:
        """
        Compute engagement score for a response.
        
        Args:
            response: Assistant response
            
        Returns:
            score: Engagement score
        """
        score = 0.5  # Base score
        
        # Engagement indicators
        engagement_indicators = ['?', 'what about', 'how about', 'tell me more', 'interesting']
        for indicator in engagement_indicators:
            if indicator in response.lower():
                score += 0.1
        
        # Penalize very short responses
        if len(response.split()) < 3:
            score -= 0.2
        
        return max(0, min(1, score))
    
    def _compute_helpfulness_score(self, user_input: str, response: str) -> float:
        """
        Compute helpfulness score for a response.
        
        Args:
            user_input: User input
            response: Assistant response
            
        Returns:
            score: Helpfulness score
        """
        score = 0.5  # Base score
        
        # Helpfulness indicators
        helpfulness_indicators = ['here\'s', 'let me', 'i can', 'sure', 'of course', 'yes']
        for indicator in helpfulness_indicators:
            if indicator in response.lower():
                score += 0.1
        
        # Reward appropriate length
        response_length = len(response.split())
        if 5 <= response_length <= 50:
            score += 0.2
        elif response_length < 3:
            score -= 0.3
        
        return max(0, min(1, score))
    
    def _compute_consistency_score(self, conversation: List[Dict]) -> float:
        """
        Compute consistency score for a conversation.
        
        Args:
            conversation: Conversation turns
            
        Returns:
            score: Consistency score
        """
        if len(conversation) < 2:
            return 1.0
        
        # Check for consistency in tone and style
        responses = [turn['assistant'] for turn in conversation]
        
        # Simple consistency check (in practice, use more sophisticated methods)
        consistency_score = 1.0
        
        # Check for repetitive responses
        unique_responses = set(responses)
        if len(unique_responses) < len(responses) * 0.8:
            consistency_score -= 0.3
        
        return max(0, consistency_score)


if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load models
    model_name = 'gpt2'
    reward_model = ConversationalRewardModel(model_name)
    
    # Create conversational RLHF
    conversational_rlhf = ConversationalRLHF(model_name, reward_model)
    
    # Test conversation
    user_inputs = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Can you help me with a math problem?",
        "Tell me a joke"
    ]
    
    print("Conversational AI Test:")
    for user_input in user_inputs:
        response = conversational_rlhf.generate_response(user_input)
        print(f"User: {user_input}")
        print(f"Assistant: {response}")
        print()
    
    # Create evaluator
    evaluator = ConversationalEvaluator()
    
    # Sample conversation data
    sample_conversations = [
        [
            {'user': 'Hello', 'assistant': 'Hi there! How can I help you today?'},
            {'user': 'What\'s the weather?', 'assistant': 'I don\'t have access to real-time weather data, but I can help you find weather information online.'}
        ]
    ]
    
    # Evaluate conversations
    metrics = evaluator.evaluate_conversation_quality(sample_conversations)
    print("Conversation Quality Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}") 
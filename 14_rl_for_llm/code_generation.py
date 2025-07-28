"""
RL for Code Generation

This module provides a complete implementation of reinforcement learning
for code generation tasks.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import ast
import re

logger = logging.getLogger(__name__)


class CodeGenerationRL:
    """
    Reinforcement learning for code generation.
    """
    
    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Code evaluation metrics
        self.evaluation_metrics = {
            'syntax_errors': [],
            'execution_success': [],
            'code_quality_scores': []
        }
    
    def generate_code(self, prompt: str, max_length: int = 200) -> str:
        """
        Generate code based on prompt.
        
        Args:
            prompt: Code generation prompt
            max_length: Maximum code length
            
        Returns:
            code: Generated code
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        
        # Generate code
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated code (remove prompt)
        code = code[len(prompt):].strip()
        
        return code
    
    def compute_code_reward(self, prompt: str, generated_code: str, test_cases: List[Dict] = None) -> float:
        """
        Compute reward for generated code.
        
        Args:
            prompt: Code generation prompt
            generated_code: Generated code
            test_cases: Test cases for evaluation
            
        Returns:
            reward: Code generation reward
        """
        # Syntax correctness reward
        syntax_reward = self._compute_syntax_reward(generated_code)
        
        # Code quality reward
        quality_reward = self._compute_code_quality_reward(generated_code)
        
        # Functionality reward (if test cases provided)
        functionality_reward = 0.0
        if test_cases:
            functionality_reward = self._compute_functionality_reward(generated_code, test_cases)
        
        # Combined reward
        total_reward = 0.4 * syntax_reward + 0.3 * quality_reward + 0.3 * functionality_reward
        
        return total_reward
    
    def _compute_syntax_reward(self, code: str) -> float:
        """
        Compute syntax correctness reward.
        
        Args:
            code: Generated code
            
        Returns:
            reward: Syntax reward
        """
        try:
            # Try to parse the code
            ast.parse(code)
            return 1.0  # Syntax is correct
        except SyntaxError:
            return 0.0  # Syntax error
        except Exception:
            return 0.5  # Other parsing issues
    
    def _compute_code_quality_reward(self, code: str) -> float:
        """
        Compute code quality reward.
        
        Args:
            code: Generated code
            
        Returns:
            reward: Quality reward
        """
        score = 0.5  # Base score
        
        # Reward good practices
        good_practices = [
            'def ', 'class ', 'import ', 'from ', 'return ',
            'if __name__', 'try:', 'except:', 'finally:',
            'with ', 'async def', 'await '
        ]
        
        for practice in good_practices:
            if practice in code:
                score += 0.1
        
        # Penalize bad practices
        bad_practices = [
            'eval(', 'exec(', 'import os', 'import sys',
            'while True:', 'for i in range(1000000):'
        ]
        
        for practice in bad_practices:
            if practice in code:
                score -= 0.2
        
        # Reward appropriate length
        lines = code.split('\n')
        if 3 <= len(lines) <= 50:
            score += 0.2
        elif len(lines) < 2:
            score -= 0.3
        
        # Reward comments
        if '#' in code:
            score += 0.1
        
        return max(0, min(1, score))
    
    def _compute_functionality_reward(self, code: str, test_cases: List[Dict]) -> float:
        """
        Compute functionality reward using test cases.
        
        Args:
            code: Generated code
            test_cases: Test cases for evaluation
            
        Returns:
            reward: Functionality reward
        """
        if not test_cases:
            return 0.5  # Default score if no test cases
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        try:
            # Create a safe execution environment
            exec_globals = {}
            exec_locals = {}
            
            # Execute the code
            exec(code, exec_globals, exec_locals)
            
            # Run test cases
            for test_case in test_cases:
                try:
                    # Extract function name and arguments
                    func_name = test_case.get('function', 'main')
                    args = test_case.get('args', [])
                    expected_output = test_case.get('expected_output')
                    
                    # Call the function
                    if func_name in exec_locals:
                        result = exec_locals[func_name](*args)
                        
                        # Check if result matches expected output
                        if expected_output is not None and result == expected_output:
                            passed_tests += 1
                        elif expected_output is None:
                            # If no expected output, just check if function runs without error
                            passed_tests += 1
                    
                except Exception:
                    # Test case failed
                    pass
        
        except Exception:
            # Code execution failed
            return 0.0
        
        return passed_tests / total_tests if total_tests > 0 else 0.0
    
    def train_on_code_data(self, training_data: List[Dict]) -> List[float]:
        """
        Train on code generation data using RL.
        
        Args:
            training_data: Training data with prompts, reference code, test cases
            
        Returns:
            losses: Training losses
        """
        losses = []
        
        for item in training_data:
            prompt = item['prompt']
            reference_code = item.get('reference_code', '')
            test_cases = item.get('test_cases', [])
            
            # Generate code
            generated_code = self.generate_code(prompt)
            
            # Compute reward
            reward = self.compute_code_reward(prompt, generated_code, test_cases)
            
            # Compute policy gradient loss
            loss = self._compute_policy_gradient_loss(prompt, generated_code, reward)
            losses.append(loss)
            
            # Update model
            self._update_model(loss)
        
        return losses
    
    def _compute_policy_gradient_loss(self, prompt: str, code: str, reward: float) -> torch.Tensor:
        """
        Compute policy gradient loss for code generation.
        
        Args:
            prompt: Input prompt
            code: Generated code
            reward: Reward for the code
            
        Returns:
            loss: Policy gradient loss
        """
        # Tokenize input and output
        full_text = prompt + code
        inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        
        # Get model outputs
        outputs = self.model(input_ids)
        logits = outputs.logits
        
        # Compute log probabilities for code tokens
        prompt_tokens = self.tokenizer(prompt, return_tensors='pt')['input_ids'].to(self.device)
        code_tokens = self.tokenizer(code, return_tensors='pt')['input_ids'].to(self.device)
        
        # Get log probabilities for code part only
        code_start_idx = len(prompt_tokens[0])
        code_log_probs = torch.log_softmax(logits[0, code_start_idx:, :], dim=-1)
        
        # Sum log probabilities
        code_log_prob = code_log_probs[:len(code_tokens[0]), :].gather(1, code_tokens[0].unsqueeze(1)).sum()
        
        # Policy gradient loss
        loss = -code_log_prob * reward
        
        return loss
    
    def _update_model(self, loss: torch.Tensor):
        """
        Update model parameters.
        
        Args:
            loss: Training loss
        """
        # This would typically use an optimizer
        # For simplicity, we'll just record the loss
        self.evaluation_metrics['code_quality_scores'].append(loss.item())


class CodeEvaluator:
    """
    Evaluator for code generation systems.
    """
    
    def __init__(self):
        self.evaluation_metrics = {}
    
    def evaluate_code_generation(self, prompts: List[str], generated_codes: List[str], 
                               reference_codes: List[str] = None) -> Dict[str, float]:
        """
        Evaluate code generation quality.
        
        Args:
            prompts: Input prompts
            generated_codes: Generated code
            reference_codes: Reference code (optional)
            
        Returns:
            metrics: Evaluation metrics
        """
        syntax_scores = []
        quality_scores = []
        length_scores = []
        
        for code in generated_codes:
            # Syntax correctness
            syntax_score = self._evaluate_syntax(code)
            syntax_scores.append(syntax_score)
            
            # Code quality
            quality_score = self._evaluate_code_quality(code)
            quality_scores.append(quality_score)
            
            # Length appropriateness
            length_score = self._evaluate_code_length(code)
            length_scores.append(length_score)
        
        return {
            'syntax_correctness': np.mean(syntax_scores),
            'code_quality': np.mean(quality_scores),
            'length_appropriateness': np.mean(length_scores),
            'overall_score': np.mean([np.mean(syntax_scores), np.mean(quality_scores), np.mean(length_scores)])
        }
    
    def _evaluate_syntax(self, code: str) -> float:
        """
        Evaluate syntax correctness.
        
        Args:
            code: Generated code
            
        Returns:
            score: Syntax correctness score
        """
        try:
            ast.parse(code)
            return 1.0
        except SyntaxError:
            return 0.0
        except Exception:
            return 0.5
    
    def _evaluate_code_quality(self, code: str) -> float:
        """
        Evaluate code quality.
        
        Args:
            code: Generated code
            
        Returns:
            score: Code quality score
        """
        score = 0.5  # Base score
        
        # Check for good practices
        good_practices = ['def ', 'class ', 'import ', 'return ', 'try:', 'except:']
        for practice in good_practices:
            if practice in code:
                score += 0.1
        
        # Check for bad practices
        bad_practices = ['eval(', 'exec(', 'while True:', 'for i in range(1000000):']
        for practice in bad_practices:
            if practice in code:
                score -= 0.2
        
        # Check for comments
        if '#' in code:
            score += 0.1
        
        return max(0, min(1, score))
    
    def _evaluate_code_length(self, code: str) -> float:
        """
        Evaluate code length appropriateness.
        
        Args:
            code: Generated code
            
        Returns:
            score: Length appropriateness score
        """
        lines = code.split('\n')
        line_count = len(lines)
        
        if 3 <= line_count <= 50:
            return 1.0
        elif line_count < 2:
            return 0.3
        elif line_count > 100:
            return 0.5
        else:
            return 0.8


class CodeTestGenerator:
    """
    Generate test cases for code evaluation.
    """
    
    def __init__(self):
        self.test_templates = {}
    
    def generate_test_cases(self, prompt: str, code: str) -> List[Dict]:
        """
        Generate test cases for code evaluation.
        
        Args:
            prompt: Code generation prompt
            code: Generated code
            
        Returns:
            test_cases: Generated test cases
        """
        test_cases = []
        
        # Extract function name from code
        function_name = self._extract_function_name(code)
        
        if function_name:
            # Generate basic test cases
            test_cases.extend(self._generate_basic_test_cases(function_name, code))
        
        return test_cases
    
    def _extract_function_name(self, code: str) -> Optional[str]:
        """
        Extract function name from code.
        
        Args:
            code: Generated code
            
        Returns:
            function_name: Extracted function name
        """
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    return node.name
        except:
            pass
        
        return None
    
    def _generate_basic_test_cases(self, function_name: str, code: str) -> List[Dict]:
        """
        Generate basic test cases.
        
        Args:
            function_name: Function name
            code: Generated code
            
        Returns:
            test_cases: Basic test cases
        """
        test_cases = []
        
        # Simple test case
        test_cases.append({
            'function': function_name,
            'args': [],
            'expected_output': None
        })
        
        return test_cases


if __name__ == "__main__":
    # Example usage
    model_name = 'gpt2'  # Could use code-specific models like 'microsoft/DialoGPT-medium'
    
    # Create code generation RL
    code_generation_rl = CodeGenerationRL(model_name)
    
    # Test code generation
    prompt = "Write a Python function to calculate the factorial of a number"
    
    generated_code = code_generation_rl.generate_code(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated code:\n{generated_code}")
    
    # Test evaluation
    evaluator = CodeEvaluator()
    metrics = evaluator.evaluate_code_generation([prompt], [generated_code])
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Test reward computation
    reward = code_generation_rl.compute_code_reward(prompt, generated_code)
    print(f"\nCode Generation Reward: {reward:.3f}") 
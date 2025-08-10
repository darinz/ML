"""
Bandit Application Framework for running experiments across different domains.

This module provides a unified framework for running bandit experiments
across various application domains including advertising, recommendation,
clinical trials, and dynamic pricing.
"""

import numpy as np
import matplotlib.pyplot as plt
from ad_selection_bandit import AdSelectionBandit
from content_recommender import ContentRecommender
from clinical_trial_bandit import AdaptiveClinicalTrial
from dynamic_pricer import DynamicPricer

class BanditApplication:
    """
    Unified framework for bandit applications across different domains.
    """
    
    def __init__(self, application_type, n_arms, feature_dim):
        """
        Initialize Bandit Application.
        
        Args:
            application_type: Type of application ('ad_selection', 'recommendation', 'clinical_trial', 'dynamic_pricing')
            n_arms: Number of arms/actions
            feature_dim: Dimension of feature vectors
        """
        self.application_type = application_type
        self.n_arms = n_arms
        self.feature_dim = feature_dim
        
        # Initialize appropriate bandit based on application
        if application_type == "ad_selection":
            self.bandit = AdSelectionBandit(n_arms, feature_dim)
        elif application_type == "recommendation":
            self.bandit = ContentRecommender(n_arms, feature_dim)
        elif application_type == "clinical_trial":
            self.bandit = AdaptiveClinicalTrial(n_arms, feature_dim)
        elif application_type == "dynamic_pricing":
            self.bandit = DynamicPricer(n_arms, feature_dim)
        else:
            raise ValueError(f"Unknown application type: {application_type}")
    
    def run_experiment(self, n_steps=1000):
        """
        Run bandit experiment.
        
        Args:
            n_steps: Number of steps to run
            
        Returns:
            tuple: (rewards, regrets)
        """
        rewards = []
        regrets = []
        
        for step in range(n_steps):
            # Generate context
            context = self._generate_context()
            
            # Select action
            action = self._select_action(context)
            
            # Get reward
            reward = self._get_reward(action, context)
            
            # Update bandit
            self._update_bandit(action, context, reward)
            
            rewards.append(reward)
            
            # Calculate regret (if optimal action is known)
            optimal_reward = self._get_optimal_reward(context)
            regret = optimal_reward - reward
            regrets.append(regret)
        
        return rewards, regrets
    
    def _generate_context(self):
        """Generate context based on application type."""
        if self.application_type == "ad_selection":
            return self._generate_user_context()
        elif self.application_type == "recommendation":
            return self._generate_user_context()
        elif self.application_type == "clinical_trial":
            return self._generate_patient_context()
        elif self.application_type == "dynamic_pricing":
            return self._generate_customer_context()
    
    def _generate_user_context(self):
        """Generate user context for advertising/recommendation."""
        return np.random.randn(self.feature_dim)
    
    def _generate_patient_context(self):
        """Generate patient context for clinical trials."""
        return np.random.randn(self.feature_dim)
    
    def _generate_customer_context(self):
        """Generate customer context for pricing."""
        return np.random.randn(self.feature_dim)
    
    def _select_action(self, context):
        """Select action based on application type."""
        if self.application_type == "ad_selection":
            return self.bandit.select_ad(context)
        elif self.application_type == "recommendation":
            return self.bandit.recommend(context)[0]  # Return first recommendation
        elif self.application_type == "clinical_trial":
            return self.bandit.assign_treatment(context)
        elif self.application_type == "dynamic_pricing":
            return self.bandit.set_price(context, {})
    
    def _get_reward(self, action, context):
        """Get reward based on application type."""
        # Simulate reward generation
        base_reward = 0.3 + 0.4 * np.dot(context[:3], [1, 0.5, 0.2])
        noise = np.random.normal(0, 0.1)
        return np.clip(base_reward + noise, 0, 1)
    
    def _update_bandit(self, action, context, reward):
        """Update bandit based on application type."""
        if self.application_type == "ad_selection":
            self.bandit.update(action, context, reward)
        elif self.application_type == "recommendation":
            self.bandit.update(action, context, reward)
        elif self.application_type == "clinical_trial":
            self.bandit.update(action, context, reward)
        elif self.application_type == "dynamic_pricing":
            self.bandit.update(action, context, {}, 1, reward)  # Simplified update
    
    def _get_optimal_reward(self, context):
        """Get optimal reward for regret calculation."""
        # Simplified optimal reward calculation
        return 0.7 + 0.3 * np.dot(context[:3], [1, 0.5, 0.2])

def compare_applications():
    """
    Compare different bandit applications.
    
    Returns:
        dict: Results for each application type
    """
    applications = [
        ("ad_selection", 10, 5),
        ("recommendation", 20, 8),
        ("clinical_trial", 5, 6),
        ("dynamic_pricing", 15, 4)
    ]
    
    results = {}
    
    for app_type, n_arms, feature_dim in applications:
        print(f"Running {app_type} experiment...")
        
        app = BanditApplication(app_type, n_arms, feature_dim)
        rewards, regrets = app.run_experiment(n_steps=500)
        
        results[app_type] = {
            'avg_reward': np.mean(rewards),
            'cumulative_regret': np.sum(regrets),
            'final_regret': np.mean(regrets[-100:])  # Last 100 steps
        }
    
    return results

def plot_application_comparison(results):
    """
    Plot comparison of different bandit applications.
    
    Args:
        results: Dictionary of results from compare_applications
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Average reward comparison
    app_names = list(results.keys())
    avg_rewards = [results[name]['avg_reward'] for name in app_names]
    ax1.bar(app_names, avg_rewards)
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Application Performance Comparison')
    
    # Cumulative regret comparison
    cumulative_regrets = [results[name]['cumulative_regret'] for name in app_names]
    ax2.bar(app_names, cumulative_regrets)
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Application Regret Comparison')
    
    plt.tight_layout()
    plt.show()

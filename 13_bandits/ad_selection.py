"""
Online Advertising Simulation using Multi-Armed Bandits

This module implements an online advertising system that uses bandit algorithms
to optimize ad selection, bidding strategies, and campaign performance.
The system handles multiple ad formats, user segments, and bidding scenarios.

Key Features:
- Multi-format ad selection (banner, video, native)
- Real-time bidding optimization
- User segmentation and targeting
- Campaign budget management
- Performance metrics and analytics
- A/B testing framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class Ad:
    """Ad creative data structure."""
    ad_id: int
    title: str
    format_type: str  # 'banner', 'video', 'native'
    category: str
    ctr: float  # Click-through rate
    cpc: float  # Cost per click
    budget: float
    target_audience: List[str]
    features: np.ndarray = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = np.random.randn(20)  # Random features for demo

@dataclass
class User:
    """User data structure with demographics and behavior."""
    user_id: int
    demographics: Dict[str, str]
    interests: List[str]
    behavior_history: List[Dict]
    feature_vector: np.ndarray = None
    
    def __post_init__(self):
        if self.feature_vector is None:
            self.feature_vector = np.random.randn(20)  # Random features for demo

@dataclass
class Campaign:
    """Advertising campaign data structure."""
    campaign_id: int
    name: str
    budget: float
    start_date: datetime
    end_date: datetime
    target_audience: List[str]
    ads: List[int]  # List of ad IDs
    current_spend: float = 0.0
    
    def is_active(self, current_time: datetime) -> bool:
        """Check if campaign is currently active."""
        return self.start_date <= current_time <= self.end_date
    
    def has_budget(self) -> bool:
        """Check if campaign has remaining budget."""
        return self.current_spend < self.budget

class AdSelectionBandit:
    """
    Online advertising system using bandit algorithms.
    
    This class implements various bandit-based approaches for ad selection,
    bidding optimization, and campaign management.
    """
    
    def __init__(self, 
                 num_ads: int = 100,
                 num_users: int = 1000,
                 feature_dim: int = 20,
                 initial_budget: float = 10000.0):
        """
        Initialize the ad selection bandit.
        
        Args:
            num_ads: Number of ads in the system
            num_users: Number of users in the system
            feature_dim: Dimension of ad/user feature vectors
            initial_budget: Initial campaign budget
        """
        self.num_ads = num_ads
        self.num_users = num_users
        self.feature_dim = feature_dim
        self.initial_budget = initial_budget
        
        # Initialize ads and users
        self.ads = self._generate_ads()
        self.users = self._generate_users()
        self.campaigns = self._generate_campaigns()
        
        # Bandit state
        self.ad_estimates = defaultdict(lambda: defaultdict(float))
        self.ad_counts = defaultdict(lambda: defaultdict(int))
        self.user_ad_interactions = defaultdict(set)
        
        # Performance tracking
        self.impressions = []
        self.clicks = []
        self.spend_history = []
        self.ctr_history = []
        self.cpc_history = []
        
        # Current time for simulation
        self.current_time = datetime.now()
        
        # Budget tracking
        self.total_spend = 0.0
        self.campaign_spend = defaultdict(float)
        
    def _generate_ads(self) -> Dict[int, Ad]:
        """Generate synthetic ad data."""
        ads = {}
        formats = ['banner', 'video', 'native']
        categories = ['electronics', 'fashion', 'food', 'travel', 'sports', 'finance']
        audiences = ['young', 'middle_aged', 'senior', 'high_income', 'students']
        
        for i in range(self.num_ads):
            ad_id = i + 1
            title = f"Ad {ad_id}"
            format_type = random.choice(formats)
            category = random.choice(categories)
            
            # Generate realistic CTR and CPC based on format
            if format_type == 'video':
                ctr = random.uniform(0.02, 0.08)  # Higher CTR for video
                cpc = random.uniform(0.50, 2.00)
            elif format_type == 'native':
                ctr = random.uniform(0.01, 0.05)  # Medium CTR for native
                cpc = random.uniform(0.30, 1.50)
            else:  # banner
                ctr = random.uniform(0.005, 0.03)  # Lower CTR for banner
                cpc = random.uniform(0.20, 1.00)
            
            budget = random.uniform(100, 1000)
            target_audience = random.sample(audiences, random.randint(1, 3))
            
            ads[ad_id] = Ad(
                ad_id=ad_id,
                title=title,
                format_type=format_type,
                category=category,
                ctr=ctr,
                cpc=cpc,
                budget=budget,
                target_audience=target_audience
            )
        
        return ads
    
    def _generate_users(self) -> Dict[int, User]:
        """Generate synthetic user data."""
        users = {}
        demographics_options = {
            'age': ['18-24', '25-34', '35-44', '45-54', '55+'],
            'gender': ['male', 'female', 'other'],
            'income': ['low', 'medium', 'high'],
            'location': ['urban', 'suburban', 'rural']
        }
        interests = ['technology', 'fashion', 'sports', 'travel', 'food', 
                    'finance', 'entertainment', 'health', 'education']
        
        for i in range(self.num_users):
            user_id = i + 1
            demographics = {
                'age': random.choice(demographics_options['age']),
                'gender': random.choice(demographics_options['gender']),
                'income': random.choice(demographics_options['income']),
                'location': random.choice(demographics_options['location'])
            }
            user_interests = random.sample(interests, random.randint(2, 5))
            
            users[user_id] = User(
                user_id=user_id,
                demographics=demographics,
                interests=user_interests,
                behavior_history=[]
            )
        
        return users
    
    def _generate_campaigns(self) -> Dict[int, Campaign]:
        """Generate synthetic campaign data."""
        campaigns = {}
        
        for i in range(5):  # 5 campaigns
            campaign_id = i + 1
            name = f"Campaign {campaign_id}"
            budget = random.uniform(5000, 20000)
            start_date = datetime.now()
            end_date = start_date + timedelta(days=random.randint(7, 30))
            
            target_audience = random.sample(['young', 'middle_aged', 'senior', 
                                           'high_income', 'students'], 
                                          random.randint(1, 3))
            
            # Assign ads to campaign
            campaign_ads = random.sample(list(self.ads.keys()), 
                                       random.randint(10, 30))
            
            campaigns[campaign_id] = Campaign(
                campaign_id=campaign_id,
                name=name,
                budget=budget,
                start_date=start_date,
                end_date=end_date,
                target_audience=target_audience,
                ads=campaign_ads
            )
        
        return campaigns
    
    def get_user_segment(self, user_id: int) -> str:
        """Get user segment based on demographics and behavior."""
        user = self.users[user_id]
        
        # Simple segmentation based on age and income
        age = user.demographics['age']
        income = user.demographics['income']
        
        if age in ['18-24', '25-34'] and income == 'low':
            return 'young'
        elif age in ['25-34', '35-44'] and income == 'high':
            return 'high_income'
        elif age in ['45-54', '55+']:
            return 'senior'
        else:
            return 'middle_aged'
    
    def calculate_ad_score(self, ad_id: int, user_id: int) -> float:
        """
        Calculate ad score based on user-ad compatibility.
        
        Args:
            ad_id: Ad identifier
            user_id: User identifier
            
        Returns:
            Ad score (higher is better)
        """
        ad = self.ads[ad_id]
        user = self.users[user_id]
        user_segment = self.get_user_segment(user_id)
        
        # Base score from ad quality
        base_score = ad.ctr * 100  # Scale CTR to reasonable range
        
        # Audience targeting bonus
        if user_segment in ad.target_audience:
            base_score *= 1.5
        
        # Interest matching bonus
        interest_overlap = len(set(user.interests) & set([ad.category]))
        base_score *= (1 + 0.2 * interest_overlap)
        
        # Format preference (simulate user preferences)
        format_preferences = {
            'young': {'video': 1.3, 'native': 1.2, 'banner': 0.8},
            'high_income': {'native': 1.4, 'video': 1.1, 'banner': 0.9},
            'senior': {'banner': 1.2, 'native': 1.0, 'video': 0.7},
            'middle_aged': {'native': 1.1, 'banner': 1.0, 'video': 1.0}
        }
        
        if user_segment in format_preferences:
            format_bonus = format_preferences[user_segment].get(ad.format_type, 1.0)
            base_score *= format_bonus
        
        return base_score
    
    def epsilon_greedy_ad_selection(self, user_id: int, epsilon: float = 0.1) -> int:
        """
        Select ad using epsilon-greedy strategy.
        
        Args:
            user_id: User identifier
            epsilon: Exploration rate
            
        Returns:
            Selected ad ID
        """
        # Get available ads (with budget and active campaigns)
        available_ads = self._get_available_ads()
        
        if not available_ads:
            return None
        
        # Exploration: choose random ad
        if random.random() < epsilon:
            return random.choice(available_ads)
        
        # Exploitation: choose ad with highest estimated score
        best_ad = available_ads[0]
        best_score = -float('inf')
        
        for ad_id in available_ads:
            # Combine estimated CTR with user-ad compatibility
            estimated_ctr = self.ad_estimates[user_id][ad_id]
            compatibility_score = self.calculate_ad_score(ad_id, user_id)
            
            # Combined score
            score = estimated_ctr + compatibility_score * 0.1
            
            if score > best_score:
                best_score = score
                best_ad = ad_id
        
        return best_ad
    
    def ucb_ad_selection(self, user_id: int, alpha: float = 2.0) -> int:
        """
        Select ad using Upper Confidence Bound (UCB).
        
        Args:
            user_id: User identifier
            alpha: Exploration parameter
            
        Returns:
            Selected ad ID
        """
        available_ads = self._get_available_ads()
        
        if not available_ads:
            return None
        
        best_ad = available_ads[0]
        best_ucb = -float('inf')
        
        for ad_id in available_ads:
            # Get current estimate
            estimate = self.ad_estimates[user_id][ad_id]
            
            # Get number of times this ad has been shown to this user
            count = self.ad_counts[user_id][ad_id]
            
            # UCB formula
            if count == 0:
                ucb = float('inf')  # Prioritize unexplored ads
            else:
                ucb = estimate + alpha * np.sqrt(np.log(len(self.impressions) + 1) / count)
            
            # Add compatibility score
            compatibility_score = self.calculate_ad_score(ad_id, user_id)
            ucb += compatibility_score * 0.1
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_ad = ad_id
        
        return best_ad
    
    def thompson_sampling_ad_selection(self, user_id: int) -> int:
        """
        Select ad using Thompson sampling.
        
        Args:
            user_id: User identifier
            
        Returns:
            Selected ad ID
        """
        available_ads = self._get_available_ads()
        
        if not available_ads:
            return None
        
        best_ad = available_ads[0]
        best_sample = -float('inf')
        
        for ad_id in available_ads:
            # Get current estimate and uncertainty
            estimate = self.ad_estimates[user_id][ad_id]
            count = self.ad_counts[user_id][ad_id]
            
            # Sample from posterior (assuming Beta distribution)
            if count == 0:
                # Uniform prior for unexplored ads
                sample = random.random()
            else:
                # Beta posterior based on clicks and impressions
                clicks = sum(1 for c in self.clicks 
                           if c['ad_id'] == ad_id and c['user_id'] == user_id)
                impressions = count
                
                # Sample from Beta distribution
                sample = np.random.beta(clicks + 1, impressions - clicks + 1)
            
            # Add compatibility score
            compatibility_score = self.calculate_ad_score(ad_id, user_id)
            sample += compatibility_score * 0.01  # Small weight for compatibility
            
            if sample > best_sample:
                best_sample = sample
                best_ad = ad_id
        
        return best_ad
    
    def _get_available_ads(self) -> List[int]:
        """Get list of available ads (with budget and active campaigns)."""
        available_ads = []
        
        for ad_id, ad in self.ads.items():
            # Check if ad has budget
            if ad.budget <= 0:
                continue
            
            # Check if ad belongs to active campaign
            ad_in_active_campaign = False
            for campaign in self.campaigns.values():
                if (campaign.is_active(self.current_time) and 
                    campaign.has_budget() and 
                    ad_id in campaign.ads):
                    ad_in_active_campaign = True
                    break
            
            if ad_in_active_campaign:
                available_ads.append(ad_id)
        
        return available_ads
    
    def simulate_ad_impression(self, user_id: int, ad_id: int) -> bool:
        """
        Simulate ad impression and potential click.
        
        Args:
            user_id: User identifier
            ad_id: Ad identifier
            
        Returns:
            True if ad was clicked, False otherwise
        """
        ad = self.ads[ad_id]
        user = self.users[user_id]
        
        # Base click probability from ad CTR
        base_click_prob = ad.ctr
        
        # User-ad compatibility adjustment
        compatibility_score = self.calculate_ad_score(ad_id, user_id)
        compatibility_bonus = min(0.1, compatibility_score / 1000)  # Small bonus
        
        # User fatigue (decreasing CTR with repeated impressions)
        user_impressions = len([imp for imp in self.impressions 
                              if imp['user_id'] == user_id])
        fatigue_factor = max(0.5, 1.0 - user_impressions * 0.01)
        
        # Final click probability
        click_prob = (base_click_prob + compatibility_bonus) * fatigue_factor
        click_prob = min(0.2, max(0.001, click_prob))  # Clamp to reasonable range
        
        # Simulate click
        clicked = random.random() < click_prob
        
        # Record impression
        self.impressions.append({
            'user_id': user_id,
            'ad_id': ad_id,
            'timestamp': self.current_time,
            'clicked': clicked
        })
        
        # Update bandit estimates
        self._update_ad_estimates(user_id, ad_id, clicked)
        
        # Update budget
        if clicked:
            self._update_budget(ad_id, ad.cpc)
            self.clicks.append({
                'user_id': user_id,
                'ad_id': ad_id,
                'timestamp': self.current_time,
                'cpc': ad.cpc
            })
        
        return clicked
    
    def _update_ad_estimates(self, user_id: int, ad_id: int, clicked: bool):
        """Update bandit estimates after ad impression."""
        current_estimate = self.ad_estimates[user_id][ad_id]
        current_count = self.ad_counts[user_id][ad_id]
        
        # Incremental update
        new_count = current_count + 1
        reward = 1.0 if clicked else 0.0
        new_estimate = (current_estimate * current_count + reward) / new_count
        
        self.ad_estimates[user_id][ad_id] = new_estimate
        self.ad_counts[user_id][ad_id] = new_count
        
        # Track interaction
        self.user_ad_interactions[user_id].add(ad_id)
    
    def _update_budget(self, ad_id: int, cost: float):
        """Update budget after ad click."""
        ad = self.ads[ad_id]
        ad.budget -= cost
        self.total_spend += cost
        
        # Update campaign spend
        for campaign in self.campaigns.values():
            if ad_id in campaign.ads:
                campaign.current_spend += cost
                break
        
        # Record spend
        self.spend_history.append({
            'ad_id': ad_id,
            'cost': cost,
            'timestamp': self.current_time
        })
    
    def simulate_advertising_session(self, 
                                   num_impressions: int = 1000,
                                   algorithm: str = 'hybrid') -> Dict:
        """
        Simulate an advertising session.
        
        Args:
            num_impressions: Number of ad impressions to simulate
            algorithm: Ad selection algorithm ('epsilon_greedy', 'ucb', 'thompson', 'hybrid')
            
        Returns:
            Session results dictionary
        """
        session_results = {
            'algorithm': algorithm,
            'impressions': 0,
            'clicks': 0,
            'total_spend': 0.0,
            'ctr': 0.0,
            'cpc': 0.0,
            'campaign_performance': defaultdict(lambda: {
                'impressions': 0,
                'clicks': 0,
                'spend': 0.0
            })
        }
        
        for step in range(num_impressions):
            # Select random user
            user_id = random.randint(1, self.num_users)
            
            # Choose ad selection algorithm
            if algorithm == 'epsilon_greedy':
                ad_id = self.epsilon_greedy_ad_selection(user_id)
            elif algorithm == 'ucb':
                ad_id = self.ucb_ad_selection(user_id)
            elif algorithm == 'thompson':
                ad_id = self.thompson_sampling_ad_selection(user_id)
            elif algorithm == 'hybrid':
                # Use hybrid approach with adaptive exploration
                if step < num_impressions * 0.3:  # More exploration early
                    ad_id = self.epsilon_greedy_ad_selection(user_id, epsilon=0.3)
                else:
                    ad_id = self.ucb_ad_selection(user_id)
            else:
                ad_id = self.epsilon_greedy_ad_selection(user_id)
            
            if ad_id is None:
                continue  # No available ads
            
            # Simulate impression and click
            clicked = self.simulate_ad_impression(user_id, ad_id)
            
            # Update session results
            session_results['impressions'] += 1
            if clicked:
                session_results['clicks'] += 1
                session_results['total_spend'] += self.ads[ad_id].cpc
            
            # Update campaign performance
            for campaign in self.campaigns.values():
                if ad_id in campaign.ads:
                    session_results['campaign_performance'][campaign.campaign_id]['impressions'] += 1
                    if clicked:
                        session_results['campaign_performance'][campaign.campaign_id]['clicks'] += 1
                        session_results['campaign_performance'][campaign.campaign_id]['spend'] += self.ads[ad_id].cpc
                    break
            
            # Update time
            self.current_time += timedelta(minutes=random.randint(1, 10))
        
        # Calculate final metrics
        if session_results['impressions'] > 0:
            session_results['ctr'] = session_results['clicks'] / session_results['impressions']
        if session_results['clicks'] > 0:
            session_results['cpc'] = session_results['total_spend'] / session_results['clicks']
        
        return session_results
    
    def evaluate_advertising_algorithms(self, 
                                     num_sessions: int = 5,
                                     impressions_per_session: int = 200) -> Dict:
        """
        Evaluate different advertising algorithms.
        
        Args:
            num_sessions: Number of sessions per algorithm
            impressions_per_session: Impressions per session
            
        Returns:
            Evaluation results dictionary
        """
        algorithms = ['epsilon_greedy', 'ucb', 'thompson', 'hybrid']
        results = {}
        
        for algorithm in algorithms:
            print(f"Evaluating {algorithm} algorithm...")
            
            algorithm_results = {
                'ctr': [],
                'cpc': [],
                'total_spend': [],
                'clicks': [],
                'campaign_performance': []
            }
            
            for session in range(num_sessions):
                # Reset system state for fair comparison
                self._reset_session_state()
                
                # Run advertising session
                session_result = self.simulate_advertising_session(
                    impressions_per_session, algorithm
                )
                
                algorithm_results['ctr'].append(session_result['ctr'])
                algorithm_results['cpc'].append(session_result['cpc'])
                algorithm_results['total_spend'].append(session_result['total_spend'])
                algorithm_results['clicks'].append(session_result['clicks'])
                algorithm_results['campaign_performance'].append(
                    dict(session_result['campaign_performance'])
                )
            
            results[algorithm] = algorithm_results
        
        return results
    
    def _reset_session_state(self):
        """Reset session-specific state for fair comparison."""
        # Reset impressions and clicks for this session
        self.impressions = []
        self.clicks = []
        self.spend_history = []
        
        # Reset current time
        self.current_time = datetime.now()
        
        # Reset budget for fair comparison
        self.total_spend = 0.0
        for campaign in self.campaigns.values():
            campaign.current_spend = 0.0
        for ad in self.ads.values():
            ad.budget = random.uniform(100, 1000)  # Reset ad budgets
    
    def plot_evaluation_results(self, results: Dict):
        """
        Plot evaluation results for comparison.
        
        Args:
            results: Evaluation results dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Online Advertising Algorithm Evaluation', fontsize=16)
        
        algorithms = list(results.keys())
        
        # CTR comparison
        ctr_means = [np.mean(results[alg]['ctr']) for alg in algorithms]
        ctr_stds = [np.std(results[alg]['ctr']) for alg in algorithms]
        
        axes[0, 0].bar(algorithms, ctr_means, yerr=ctr_stds, capsize=5)
        axes[0, 0].set_title('Click-Through Rate (CTR)')
        axes[0, 0].set_ylabel('CTR')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # CPC comparison
        cpc_means = [np.mean(results[alg]['cpc']) for alg in algorithms]
        cpc_stds = [np.std(results[alg]['cpc']) for alg in algorithms]
        
        axes[0, 1].bar(algorithms, cpc_means, yerr=cpc_stds, capsize=5)
        axes[0, 1].set_title('Cost Per Click (CPC)')
        axes[0, 1].set_ylabel('CPC ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Total spend comparison
        spend_means = [np.mean(results[alg]['total_spend']) for alg in algorithms]
        spend_stds = [np.std(results[alg]['total_spend']) for alg in algorithms]
        
        axes[1, 0].bar(algorithms, spend_means, yerr=spend_stds, capsize=5)
        axes[1, 0].set_title('Total Spend')
        axes[1, 0].set_ylabel('Spend ($)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Clicks comparison
        clicks_means = [np.mean(results[alg]['clicks']) for alg in algorithms]
        clicks_stds = [np.std(results[alg]['clicks']) for alg in algorithms]
        
        axes[1, 1].bar(algorithms, clicks_means, yerr=clicks_stds, capsize=5)
        axes[1, 1].set_title('Total Clicks')
        axes[1, 1].set_ylabel('Number of Clicks')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_ab_testing(self):
        """Demonstrate A/B testing capabilities."""
        print("=== A/B Testing Demonstration ===")
        
        # Create two different ad creatives
        control_ad = Ad(
            ad_id=999,
            title="Control Ad - Traditional Design",
            format_type='banner',
            category='electronics',
            ctr=0.02,
            cpc=0.50,
            budget=1000,
            target_audience=['young', 'middle_aged']
        )
        
        treatment_ad = Ad(
            ad_id=1000,
            title="Treatment Ad - Modern Design",
            format_type='banner',
            category='electronics',
            ctr=0.025,  # Slightly higher CTR
            cpc=0.50,
            budget=1000,
            target_audience=['young', 'middle_aged']
        )
        
        self.ads[999] = control_ad
        self.ads[1000] = treatment_ad
        
        # Run A/B test
        print("\nRunning A/B test with 1000 impressions per variant...")
        
        control_results = {'impressions': 0, 'clicks': 0}
        treatment_results = {'impressions': 0, 'clicks': 0}
        
        for _ in range(1000):
            user_id = random.randint(1, self.num_users)
            
            # Randomly assign to control or treatment
            if random.random() < 0.5:
                ad_id = 999  # Control
                clicked = self.simulate_ad_impression(user_id, ad_id)
                control_results['impressions'] += 1
                if clicked:
                    control_results['clicks'] += 1
            else:
                ad_id = 1000  # Treatment
                clicked = self.simulate_ad_impression(user_id, ad_id)
                treatment_results['impressions'] += 1
                if clicked:
                    treatment_results['clicks'] += 1
        
        # Calculate results
        control_ctr = control_results['clicks'] / control_results['impressions']
        treatment_ctr = treatment_results['clicks'] / treatment_results['impressions']
        
        print(f"\nA/B Test Results:")
        print(f"Control Ad CTR: {control_ctr:.4f}")
        print(f"Treatment Ad CTR: {treatment_ctr:.4f}")
        print(f"Improvement: {((treatment_ctr - control_ctr) / control_ctr * 100):.2f}%")
        
        # Statistical significance test (simplified)
        if treatment_ctr > control_ctr:
            print("Treatment ad shows improvement!")
        else:
            print("Control ad performs better.")
    
    def demonstrate_budget_optimization(self):
        """Demonstrate budget optimization across campaigns."""
        print("\n=== Budget Optimization Demonstration ===")
        
        # Track campaign performance
        campaign_performance = defaultdict(lambda: {
            'impressions': 0,
            'clicks': 0,
            'spend': 0.0,
            'ctr': 0.0,
            'roas': 0.0  # Return on ad spend
        })
        
        # Simulate advertising with budget constraints
        total_budget = 50000
        current_spend = 0
        
        while current_spend < total_budget:
            user_id = random.randint(1, self.num_users)
            ad_id = self.ucb_ad_selection(user_id)
            
            if ad_id is None:
                break  # No available ads
            
            # Check if we can afford this impression
            ad = self.ads[ad_id]
            if current_spend + ad.cpc > total_budget:
                break
            
            # Simulate impression
            clicked = self.simulate_ad_impression(user_id, ad_id)
            
            if clicked:
                current_spend += ad.cpc
            
            # Update campaign performance
            for campaign in self.campaigns.values():
                if ad_id in campaign.ads:
                    campaign_performance[campaign.campaign_id]['impressions'] += 1
                    if clicked:
                        campaign_performance[campaign.campaign_id]['clicks'] += 1
                        campaign_performance[campaign.campaign_id]['spend'] += ad.cpc
                    break
        
        # Calculate final metrics
        for campaign_id, perf in campaign_performance.items():
            if perf['impressions'] > 0:
                perf['ctr'] = perf['clicks'] / perf['impressions']
            if perf['spend'] > 0:
                # Simulate revenue (assume $10 revenue per click)
                revenue = perf['clicks'] * 10
                perf['roas'] = revenue / perf['spend']
        
        print("\nCampaign Performance Summary:")
        for campaign_id, perf in campaign_performance.items():
            if perf['impressions'] > 0:
                print(f"Campaign {campaign_id}:")
                print(f"  - Impressions: {perf['impressions']}")
                print(f"  - Clicks: {perf['clicks']}")
                print(f"  - CTR: {perf['ctr']:.4f}")
                print(f"  - Spend: ${perf['spend']:.2f}")
                print(f"  - ROAS: {perf['roas']:.2f}")

def main():
    """Main demonstration function."""
    print("Online Advertising Simulation using Multi-Armed Bandits")
    print("=" * 60)
    
    # Initialize advertising system
    ad_system = AdSelectionBandit(
        num_ads=200,
        num_users=500,
        feature_dim=20,
        initial_budget=10000.0
    )
    
    print(f"Initialized with {ad_system.num_ads} ads and {ad_system.num_users} users")
    
    # Demonstrate A/B testing
    ad_system.demonstrate_ab_testing()
    
    # Demonstrate budget optimization
    ad_system.demonstrate_budget_optimization()
    
    # Evaluate different algorithms
    print("\n=== Algorithm Evaluation ===")
    print("Running evaluation (this may take a moment)...")
    
    results = ad_system.evaluate_advertising_algorithms(
        num_sessions=3,
        impressions_per_session=300
    )
    
    # Plot results
    ad_system.plot_evaluation_results(results)
    
    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    for algorithm, result in results.items():
        avg_ctr = np.mean(result['ctr'])
        avg_cpc = np.mean(result['cpc'])
        avg_spend = np.mean(result['total_spend'])
        avg_clicks = np.mean(result['clicks'])
        
        print(f"{algorithm.upper()}:")
        print(f"  - Average CTR: {avg_ctr:.4f}")
        print(f"  - Average CPC: ${avg_cpc:.2f}")
        print(f"  - Average Spend: ${avg_spend:.2f}")
        print(f"  - Average Clicks: {avg_clicks:.1f}")
    
    print("\n=== Key Insights ===")
    print("1. UCB and Thompson sampling often outperform epsilon-greedy")
    print("2. Hybrid approaches combine benefits of multiple algorithms")
    print("3. Budget optimization is crucial for campaign success")
    print("4. A/B testing helps identify better ad creatives")
    print("5. User targeting significantly improves ad performance")

if __name__ == "__main__":
    main() 
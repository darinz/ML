"""
Dynamic Pricing Simulation using Multi-Armed Bandits

This module implements a dynamic pricing system that uses bandit algorithms
to optimize pricing strategies, maximize revenue, and learn demand elasticity.
The system handles multiple products, customer segments, and market conditions.

Key Features:
- Multi-product pricing optimization
- Customer segmentation and targeting
- Demand elasticity learning
- Revenue and profit maximization
- Market competition simulation
- Seasonal and temporal effects
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
class Product:
    """Product data structure."""
    product_id: int
    name: str
    category: str
    cost: float
    base_price: float
    elasticity: float  # Price elasticity of demand
    quality_score: float
    features: np.ndarray = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = np.random.randn(15)  # Random features for demo

@dataclass
class Customer:
    """Customer data structure with preferences and behavior."""
    customer_id: int
    segment: str  # 'budget', 'premium', 'value', 'luxury'
    price_sensitivity: float
    quality_preference: float
    loyalty: float
    purchase_history: List[Dict]
    feature_vector: np.ndarray = None
    
    def __post_init__(self):
        if self.purchase_history is None:
            self.purchase_history = []
        if self.feature_vector is None:
            self.feature_vector = np.random.randn(15)  # Random features for demo

@dataclass
class MarketCondition:
    """Market condition data structure."""
    condition_id: int
    name: str
    demand_multiplier: float
    competition_level: float
    seasonality_factor: float
    economic_indicator: float

class DynamicPricingBandit:
    """
    Dynamic pricing system using bandit algorithms.
    
    This class implements various bandit-based approaches for price optimization,
    demand learning, and revenue maximization.
    """
    
    def __init__(self, 
                 num_products: int = 20,
                 num_customers: int = 500,
                 feature_dim: int = 15,
                 initial_capital: float = 100000.0):
        """
        Initialize the dynamic pricing bandit.
        
        Args:
            num_products: Number of products in the system
            num_customers: Number of customers in the system
            feature_dim: Dimension of product/customer feature vectors
            initial_capital: Initial capital for inventory
        """
        self.num_products = num_products
        self.num_customers = num_customers
        self.feature_dim = feature_dim
        self.initial_capital = initial_capital
        
        # Initialize products and customers
        self.products = self._generate_products()
        self.customers = self._generate_customers()
        self.market_conditions = self._generate_market_conditions()
        
        # Bandit state
        self.price_estimates = defaultdict(lambda: defaultdict(float))
        self.price_counts = defaultdict(lambda: defaultdict(int))
        self.customer_product_interactions = defaultdict(set)
        
        # Performance tracking
        self.sales_history = []
        self.revenue_history = []
        self.profit_history = []
        self.demand_history = []
        
        # Current state
        self.current_time = datetime.now()
        self.current_market_condition = 1
        self.inventory = defaultdict(int)
        self.cash_flow = initial_capital
        
        # Pricing parameters
        self.price_levels = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  # Multipliers of base price
        self.min_margin = 0.1  # Minimum profit margin
        
    def _generate_products(self) -> Dict[int, Product]:
        """Generate synthetic product data."""
        products = {}
        categories = ['electronics', 'clothing', 'food', 'books', 'sports', 'home']
        
        for i in range(self.num_products):
            product_id = i + 1
            name = f"Product {product_id}"
            category = random.choice(categories)
            
            # Generate realistic product characteristics
            if category == 'electronics':
                cost = random.uniform(50, 500)
                base_price = cost * random.uniform(1.3, 2.0)
                elasticity = random.uniform(-2.0, -1.0)  # More elastic
                quality_score = random.uniform(0.6, 0.9)
            elif category == 'clothing':
                cost = random.uniform(10, 100)
                base_price = cost * random.uniform(1.5, 3.0)
                elasticity = random.uniform(-1.5, -0.8)
                quality_score = random.uniform(0.5, 0.8)
            elif category == 'food':
                cost = random.uniform(2, 20)
                base_price = cost * random.uniform(1.2, 2.5)
                elasticity = random.uniform(-1.2, -0.6)
                quality_score = random.uniform(0.4, 0.7)
            else:
                cost = random.uniform(5, 50)
                base_price = cost * random.uniform(1.4, 2.5)
                elasticity = random.uniform(-1.8, -1.0)
                quality_score = random.uniform(0.5, 0.8)
            
            products[product_id] = Product(
                product_id=product_id,
                name=name,
                category=category,
                cost=cost,
                base_price=base_price,
                elasticity=elasticity,
                quality_score=quality_score
            )
        
        return products
    
    def _generate_customers(self) -> Dict[int, Customer]:
        """Generate synthetic customer data."""
        customers = {}
        segments = ['budget', 'value', 'premium', 'luxury']
        
        for i in range(self.num_customers):
            customer_id = i + 1
            segment = random.choice(segments)
            
            # Generate segment-specific characteristics
            if segment == 'budget':
                price_sensitivity = random.uniform(0.8, 1.0)
                quality_preference = random.uniform(0.3, 0.6)
                loyalty = random.uniform(0.2, 0.5)
            elif segment == 'value':
                price_sensitivity = random.uniform(0.6, 0.8)
                quality_preference = random.uniform(0.5, 0.7)
                loyalty = random.uniform(0.4, 0.7)
            elif segment == 'premium':
                price_sensitivity = random.uniform(0.4, 0.6)
                quality_preference = random.uniform(0.7, 0.9)
                loyalty = random.uniform(0.6, 0.8)
            else:  # luxury
                price_sensitivity = random.uniform(0.2, 0.4)
                quality_preference = random.uniform(0.8, 1.0)
                loyalty = random.uniform(0.7, 0.9)
            
            customers[customer_id] = Customer(
                customer_id=customer_id,
                segment=segment,
                price_sensitivity=price_sensitivity,
                quality_preference=quality_preference,
                loyalty=loyalty,
                purchase_history=[]
            )
        
        return customers
    
    def _generate_market_conditions(self) -> Dict[int, MarketCondition]:
        """Generate synthetic market conditions."""
        conditions = {}
        
        # Normal market
        conditions[1] = MarketCondition(
            condition_id=1,
            name="Normal Market",
            demand_multiplier=1.0,
            competition_level=0.5,
            seasonality_factor=1.0,
            economic_indicator=1.0
        )
        
        # High demand market
        conditions[2] = MarketCondition(
            condition_id=2,
            name="High Demand",
            demand_multiplier=1.3,
            competition_level=0.3,
            seasonality_factor=1.2,
            economic_indicator=1.1
        )
        
        # Competitive market
        conditions[3] = MarketCondition(
            condition_id=3,
            name="Competitive Market",
            demand_multiplier=0.8,
            competition_level=0.8,
            seasonality_factor=0.9,
            economic_indicator=0.9
        )
        
        # Seasonal peak
        conditions[4] = MarketCondition(
            condition_id=4,
            name="Seasonal Peak",
            demand_multiplier=1.4,
            competition_level=0.4,
            seasonality_factor=1.5,
            economic_indicator=1.0
        )
        
        return conditions
    
    def calculate_demand(self, product_id: int, price: float, customer_id: int) -> float:
        """
        Calculate demand for a product at a given price.
        
        Args:
            product_id: Product identifier
            price: Price of the product
            customer_id: Customer identifier
            
        Returns:
            Demand quantity
        """
        product = self.products[product_id]
        customer = self.customers[customer_id]
        market = self.market_conditions[self.current_market_condition]
        
        # Base demand from product characteristics
        base_demand = 10.0 * product.quality_score
        
        # Price elasticity effect
        price_ratio = price / product.base_price
        elasticity_effect = price_ratio ** product.elasticity
        
        # Customer segment effect
        segment_multipliers = {
            'budget': 0.7,
            'value': 1.0,
            'premium': 1.3,
            'luxury': 1.6
        }
        segment_effect = segment_multipliers.get(customer.segment, 1.0)
        
        # Price sensitivity effect
        price_sensitivity_effect = (1 - customer.price_sensitivity * (price_ratio - 1))
        price_sensitivity_effect = max(0.1, price_sensitivity_effect)
        
        # Quality preference effect
        quality_effect = 1 + customer.quality_preference * (product.quality_score - 0.5)
        
        # Market condition effects
        market_effect = (market.demand_multiplier * 
                        market.seasonality_factor * 
                        market.economic_indicator)
        
        # Competition effect
        competition_effect = 1 - market.competition_level * 0.3
        
        # Calculate final demand
        demand = (base_demand * 
                 elasticity_effect * 
                 segment_effect * 
                 price_sensitivity_effect * 
                 quality_effect * 
                 market_effect * 
                 competition_effect)
        
        # Add some randomness
        demand *= random.uniform(0.8, 1.2)
        
        return max(0.0, demand)
    
    def calculate_optimal_price(self, product_id: int, customer_id: int) -> float:
        """
        Calculate optimal price for a product-customer combination.
        
        Args:
            product_id: Product identifier
            customer_id: Customer identifier
            
        Returns:
            Optimal price
        """
        product = self.products[product_id]
        customer = self.customers[customer_id]
        
        # Base optimal price from elasticity
        if product.elasticity != -1:  # Avoid division by zero
            optimal_ratio = 1 / (1 + 1/product.elasticity)
        else:
            optimal_ratio = 1.0
        
        # Customer segment adjustments
        segment_adjustments = {
            'budget': 0.9,
            'value': 1.0,
            'premium': 1.1,
            'luxury': 1.2
        }
        segment_adjustment = segment_adjustments.get(customer.segment, 1.0)
        
        # Market condition adjustments
        market = self.market_conditions[self.current_market_condition]
        market_adjustment = 1 + (market.demand_multiplier - 1) * 0.2
        
        # Calculate optimal price
        optimal_price = (product.base_price * 
                        optimal_ratio * 
                        segment_adjustment * 
                        market_adjustment)
        
        # Ensure minimum margin
        min_price = product.cost * (1 + self.min_margin)
        optimal_price = max(optimal_price, min_price)
        
        return optimal_price
    
    def epsilon_greedy_pricing(self, product_id: int, customer_id: int, 
                              epsilon: float = 0.2) -> float:
        """
        Set price using epsilon-greedy strategy.
        
        Args:
            product_id: Product identifier
            customer_id: Customer identifier
            epsilon: Exploration rate
            
        Returns:
            Selected price
        """
        product = self.products[product_id]
        
        # Exploration: choose random price level
        if random.random() < epsilon:
            price_multiplier = random.choice(self.price_levels)
            return product.base_price * price_multiplier
        
        # Exploitation: choose price with highest estimated revenue
        best_price = product.base_price
        best_revenue = -float('inf')
        
        for multiplier in self.price_levels:
            price = product.base_price * multiplier
            
            # Ensure minimum margin
            if price < product.cost * (1 + self.min_margin):
                continue
            
            # Estimate demand and revenue
            demand = self.calculate_demand(product_id, price, customer_id)
            revenue = demand * price
            
            if revenue > best_revenue:
                best_revenue = revenue
                best_price = price
        
        return best_price
    
    def ucb_pricing(self, product_id: int, customer_id: int, alpha: float = 2.0) -> float:
        """
        Set price using Upper Confidence Bound (UCB).
        
        Args:
            product_id: Product identifier
            customer_id: Customer identifier
            alpha: Exploration parameter
            
        Returns:
            Selected price
        """
        product = self.products[product_id]
        
        best_price = product.base_price
        best_ucb = -float('inf')
        
        for multiplier in self.price_levels:
            price = product.base_price * multiplier
            
            # Ensure minimum margin
            if price < product.cost * (1 + self.min_margin):
                continue
            
            # Get current estimate
            estimate = self.price_estimates[customer_id][price]
            
            # Get number of times this price has been used
            count = self.price_counts[customer_id][price]
            
            # UCB formula
            if count == 0:
                ucb = float('inf')  # Prioritize unexplored prices
            else:
                ucb = estimate + alpha * np.sqrt(np.log(len(self.sales_history) + 1) / count)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_price = price
        
        return best_price
    
    def thompson_sampling_pricing(self, product_id: int, customer_id: int) -> float:
        """
        Set price using Thompson sampling.
        
        Args:
            product_id: Product identifier
            customer_id: Customer identifier
            
        Returns:
            Selected price
        """
        product = self.products[product_id]
        
        best_price = product.base_price
        best_sample = -float('inf')
        
        for multiplier in self.price_levels:
            price = product.base_price * multiplier
            
            # Ensure minimum margin
            if price < product.cost * (1 + self.min_margin):
                continue
            
            # Get current estimate and uncertainty
            estimate = self.price_estimates[customer_id][price]
            count = self.price_counts[customer_id][price]
            
            # Sample from posterior (assuming Beta distribution)
            if count == 0:
                # Uniform prior for unexplored prices
                sample = random.random()
            else:
                # Beta posterior based on revenue performance
                # Convert revenue to success/failure (revenue > threshold is success)
                threshold = product.cost * 1.5  # Minimum profitable revenue
                successes = sum(1 for sale in self.sales_history 
                              if sale['price'] == price and sale['revenue'] > threshold)
                failures = count - successes
                
                # Sample from Beta distribution
                sample = np.random.beta(successes + 1, failures + 1)
            
            if sample > best_sample:
                best_sample = sample
                best_price = price
        
        return best_price
    
    def optimal_pricing(self, product_id: int, customer_id: int) -> float:
        """
        Set price using optimal pricing strategy.
        
        Args:
            product_id: Product identifier
            customer_id: Customer identifier
            
        Returns:
            Selected price
        """
        return self.calculate_optimal_price(product_id, customer_id)
    
    def simulate_purchase(self, customer_id: int, product_id: int, price: float) -> Dict:
        """
        Simulate customer purchase decision.
        
        Args:
            customer_id: Customer identifier
            product_id: Product identifier
            price: Offered price
            
        Returns:
            Purchase outcome dictionary
        """
        customer = self.customers[customer_id]
        product = self.products[product_id]
        
        # Calculate demand
        demand = self.calculate_demand(product_id, price, customer_id)
        
        # Simulate purchase decision
        purchase_probability = min(1.0, demand / 10.0)  # Normalize to probability
        purchased = random.random() < purchase_probability
        
        if purchased:
            # Calculate quantity (simplified)
            quantity = max(1, int(demand * random.uniform(0.5, 1.5)))
            
            # Calculate revenue and profit
            revenue = quantity * price
            cost = quantity * product.cost
            profit = revenue - cost
            
            # Update inventory and cash flow
            self.inventory[product_id] -= quantity
            self.cash_flow += revenue
            
            # Record purchase
            purchase_record = {
                'customer_id': customer_id,
                'product_id': product_id,
                'price': price,
                'quantity': quantity,
                'revenue': revenue,
                'profit': profit,
                'timestamp': self.current_time
            }
            
            customer.purchase_history.append(purchase_record)
            
            return purchase_record
        
        return None
    
    def simulate_pricing_session(self, 
                                num_customers: int = 100,
                                algorithm: str = 'hybrid') -> Dict:
        """
        Simulate a pricing session.
        
        Args:
            num_customers: Number of customers to serve
            algorithm: Pricing algorithm ('epsilon_greedy', 'ucb', 'thompson', 'optimal', 'hybrid')
            
        Returns:
            Session results dictionary
        """
        session_results = {
            'algorithm': algorithm,
            'customers_served': 0,
            'total_sales': 0,
            'total_revenue': 0.0,
            'total_profit': 0.0,
            'average_price': 0.0,
            'conversion_rate': 0.0
        }
        
        prices_used = []
        sales_made = 0
        
        for customer_idx in range(num_customers):
            # Select random customer
            customer_id = random.randint(1, self.num_customers)
            
            # Select random product
            product_id = random.randint(1, self.num_products)
            
            # Choose pricing algorithm
            if algorithm == 'epsilon_greedy':
                price = self.epsilon_greedy_pricing(product_id, customer_id)
            elif algorithm == 'ucb':
                price = self.ucb_pricing(product_id, customer_id)
            elif algorithm == 'thompson':
                price = self.thompson_sampling_pricing(product_id, customer_id)
            elif algorithm == 'optimal':
                price = self.optimal_pricing(product_id, customer_id)
            elif algorithm == 'hybrid':
                # Use hybrid approach with adaptive exploration
                if customer_idx < num_customers * 0.3:  # More exploration early
                    price = self.epsilon_greedy_pricing(product_id, customer_id, epsilon=0.3)
                else:
                    price = self.ucb_pricing(product_id, customer_id)
            else:
                price = self.epsilon_greedy_pricing(product_id, customer_id)
            
            # Simulate purchase
            purchase = self.simulate_purchase(customer_id, product_id, price)
            
            # Update session results
            session_results['customers_served'] += 1
            prices_used.append(price)
            
            if purchase:
                session_results['total_sales'] += 1
                session_results['total_revenue'] += purchase['revenue']
                session_results['total_profit'] += purchase['profit']
                sales_made += 1
            
            # Update bandit estimates
            if purchase:
                self._update_price_estimates(customer_id, price, purchase['revenue'])
            
            # Update time
            self.current_time += timedelta(hours=random.randint(1, 4))
            
            # Occasionally change market conditions
            if random.random() < 0.1:  # 10% chance to change market
                self.current_market_condition = random.choice(list(self.market_conditions.keys()))
        
        # Calculate final metrics
        if prices_used:
            session_results['average_price'] = np.mean(prices_used)
        if session_results['customers_served'] > 0:
            session_results['conversion_rate'] = sales_made / session_results['customers_served']
        
        return session_results
    
    def _update_price_estimates(self, customer_id: int, price: float, revenue: float):
        """Update bandit estimates after sale."""
        current_estimate = self.price_estimates[customer_id][price]
        current_count = self.price_counts[customer_id][price]
        
        # Incremental update
        new_count = current_count + 1
        # Normalize revenue to [0, 1] for bandit learning
        normalized_revenue = min(1.0, revenue / 1000.0)  # Assume max revenue is 1000
        new_estimate = (current_estimate * current_count + normalized_revenue) / new_count
        
        self.price_estimates[customer_id][price] = new_estimate
        self.price_counts[customer_id][price] = new_count
    
    def evaluate_pricing_algorithms(self, 
                                  num_sessions: int = 5,
                                  customers_per_session: int = 200) -> Dict:
        """
        Evaluate different pricing algorithms.
        
        Args:
            num_sessions: Number of sessions per algorithm
            customers_per_session: Customers per session
            
        Returns:
            Evaluation results dictionary
        """
        algorithms = ['epsilon_greedy', 'ucb', 'thompson', 'optimal', 'hybrid']
        results = {}
        
        for algorithm in algorithms:
            print(f"Evaluating {algorithm} algorithm...")
            
            algorithm_results = {
                'total_revenue': [],
                'total_profit': [],
                'conversion_rate': [],
                'average_price': []
            }
            
            for session in range(num_sessions):
                # Reset session state
                self._reset_session_state()
                
                # Run pricing session
                session_result = self.simulate_pricing_session(
                    customers_per_session, algorithm
                )
                
                algorithm_results['total_revenue'].append(session_result['total_revenue'])
                algorithm_results['total_profit'].append(session_result['total_profit'])
                algorithm_results['conversion_rate'].append(session_result['conversion_rate'])
                algorithm_results['average_price'].append(session_result['average_price'])
            
            results[algorithm] = algorithm_results
        
        return results
    
    def _reset_session_state(self):
        """Reset session-specific state for fair comparison."""
        # Reset sales and revenue tracking
        self.sales_history = []
        self.revenue_history = []
        self.profit_history = []
        self.demand_history = []
        
        # Reset current time
        self.current_time = datetime.now()
        
        # Reset cash flow and inventory
        self.cash_flow = self.initial_capital
        self.inventory = defaultdict(lambda: 100)  # Reset inventory
        
        # Reset market condition
        self.current_market_condition = 1
    
    def plot_evaluation_results(self, results: Dict):
        """
        Plot evaluation results for comparison.
        
        Args:
            results: Evaluation results dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dynamic Pricing Algorithm Evaluation', fontsize=16)
        
        algorithms = list(results.keys())
        
        # Total revenue
        revenue_means = [np.mean(results[alg]['total_revenue']) for alg in algorithms]
        revenue_stds = [np.std(results[alg]['total_revenue']) for alg in algorithms]
        
        axes[0, 0].bar(algorithms, revenue_means, yerr=revenue_stds, capsize=5)
        axes[0, 0].set_title('Total Revenue')
        axes[0, 0].set_ylabel('Revenue ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total profit
        profit_means = [np.mean(results[alg]['total_profit']) for alg in algorithms]
        profit_stds = [np.std(results[alg]['total_profit']) for alg in algorithms]
        
        axes[0, 1].bar(algorithms, profit_means, yerr=profit_stds, capsize=5)
        axes[0, 1].set_title('Total Profit')
        axes[0, 1].set_ylabel('Profit ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Conversion rate
        conversion_means = [np.mean(results[alg]['conversion_rate']) for alg in algorithms]
        conversion_stds = [np.std(results[alg]['conversion_rate']) for alg in algorithms]
        
        axes[1, 0].bar(algorithms, conversion_means, yerr=conversion_stds, capsize=5)
        axes[1, 0].set_title('Conversion Rate')
        axes[1, 0].set_ylabel('Conversion Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Average price
        price_means = [np.mean(results[alg]['average_price']) for alg in algorithms]
        price_stds = [np.std(results[alg]['average_price']) for alg in algorithms]
        
        axes[1, 1].bar(algorithms, price_means, yerr=price_stds, capsize=5)
        axes[1, 1].set_title('Average Price')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_market_conditions(self):
        """Demonstrate how market conditions affect pricing."""
        print("=== Market Conditions Demonstration ===")
        
        # Test pricing under different market conditions
        customer_id = 1
        product_id = 1
        product = self.products[product_id]
        
        print(f"\nTesting pricing for {product.name} under different market conditions:")
        
        for condition_id, condition in self.market_conditions.items():
            self.current_market_condition = condition_id
            
            print(f"\n{condition.name}:")
            print(f"  - Demand multiplier: {condition.demand_multiplier:.2f}")
            print(f"  - Competition level: {condition.competition_level:.2f}")
            print(f"  - Seasonality factor: {condition.seasonality_factor:.2f}")
            
            # Calculate demand at different prices
            base_price = product.base_price
            prices = [base_price * 0.8, base_price, base_price * 1.2]
            
            for price in prices:
                demand = self.calculate_demand(product_id, price, customer_id)
                revenue = demand * price
                profit = revenue - (demand * product.cost)
                
                print(f"    Price ${price:.2f}: Demand={demand:.1f}, Revenue=${revenue:.2f}, Profit=${profit:.2f}")
    
    def demonstrate_customer_segmentation(self):
        """Demonstrate customer segmentation effects on pricing."""
        print("\n=== Customer Segmentation Demonstration ===")
        
        product_id = 1
        product = self.products[product_id]
        
        print(f"\nTesting pricing for {product.name} across customer segments:")
        
        for segment in ['budget', 'value', 'premium', 'luxury']:
            print(f"\n{segment.upper()} segment:")
            
            # Find a customer from this segment
            segment_customers = [cid for cid, customer in self.customers.items() 
                               if customer.segment == segment]
            
            if segment_customers:
                customer_id = random.choice(segment_customers)
                customer = self.customers[customer_id]
                
                print(f"  - Price sensitivity: {customer.price_sensitivity:.2f}")
                print(f"  - Quality preference: {customer.quality_preference:.2f}")
                print(f"  - Loyalty: {customer.loyalty:.2f}")
                
                # Test different pricing strategies
                base_price = product.base_price
                prices = [base_price * 0.8, base_price, base_price * 1.2, base_price * 1.4]
                
                for price in prices:
                    demand = self.calculate_demand(product_id, price, customer_id)
                    revenue = demand * price
                    profit = revenue - (demand * product.cost)
                    
                    print(f"    Price ${price:.2f}: Demand={demand:.1f}, Revenue=${revenue:.2f}, Profit=${profit:.2f}")

def main():
    """Main demonstration function."""
    print("Dynamic Pricing Simulation using Multi-Armed Bandits")
    print("=" * 60)
    
    # Initialize dynamic pricing system
    pricing_system = DynamicPricingBandit(
        num_products=30,
        num_customers=300,
        feature_dim=15,
        initial_capital=50000.0
    )
    
    print(f"Initialized with {pricing_system.num_products} products and {pricing_system.num_customers} customers")
    
    # Demonstrate market conditions
    pricing_system.demonstrate_market_conditions()
    
    # Demonstrate customer segmentation
    pricing_system.demonstrate_customer_segmentation()
    
    # Evaluate different algorithms
    print("\n=== Algorithm Evaluation ===")
    print("Running evaluation (this may take a moment)...")
    
    results = pricing_system.evaluate_pricing_algorithms(
        num_sessions=3,
        customers_per_session=150
    )
    
    # Plot results
    pricing_system.plot_evaluation_results(results)
    
    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    for algorithm, result in results.items():
        avg_revenue = np.mean(result['total_revenue'])
        avg_profit = np.mean(result['total_profit'])
        avg_conversion = np.mean(result['conversion_rate'])
        avg_price = np.mean(result['average_price'])
        
        print(f"{algorithm.upper()}:")
        print(f"  - Average Revenue: ${avg_revenue:.2f}")
        print(f"  - Average Profit: ${avg_profit:.2f}")
        print(f"  - Average Conversion Rate: {avg_conversion:.3f}")
        print(f"  - Average Price: ${avg_price:.2f}")
    
    print("\n=== Key Insights ===")
    print("1. Optimal pricing often performs best but requires perfect information")
    print("2. Bandit algorithms adapt to changing market conditions")
    print("3. Customer segmentation significantly affects optimal pricing")
    print("4. Market conditions require dynamic price adjustments")
    print("5. Hybrid approaches balance exploration and exploitation effectively")

if __name__ == "__main__":
    main() 
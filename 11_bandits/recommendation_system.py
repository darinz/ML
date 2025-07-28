"""
Movie Recommendation System using Multi-Armed Bandits

This module implements a movie recommendation system that uses bandit algorithms
to balance exploration of new movies with exploitation of known good recommendations.
The system handles cold-start problems and provides personalized recommendations.

Key Features:
- Collaborative filtering with bandit exploration
- Content-based features for movie similarity
- Hybrid approaches combining multiple signals
- Cold-start handling for new users and movies
- Real-time recommendation updates
- Performance evaluation with multiple metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

@dataclass
class Movie:
    """Movie data structure with features and metadata."""
    movie_id: int
    title: str
    genres: List[str]
    year: int
    rating: float
    num_ratings: int
    features: np.ndarray = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = np.random.randn(50)  # Random features for demo

@dataclass
class User:
    """User data structure with preferences and history."""
    user_id: int
    preferences: Dict[str, float] = None
    rating_history: Dict[int, float] = None
    feature_vector: np.ndarray = None
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.rating_history is None:
            self.rating_history = {}
        if self.feature_vector is None:
            self.feature_vector = np.random.randn(50)  # Random features for demo

class MovieRecommendationBandit:
    """
    Movie recommendation system using bandit algorithms.
    
    This class implements various bandit-based approaches for movie recommendations,
    including collaborative filtering, content-based filtering, and hybrid methods.
    """
    
    def __init__(self, 
                 num_movies: int = 1000,
                 num_users: int = 500,
                 feature_dim: int = 50,
                 exploration_rate: float = 0.1):
        """
        Initialize the movie recommendation bandit.
        
        Args:
            num_movies: Number of movies in the system
            num_users: Number of users in the system
            feature_dim: Dimension of movie/user feature vectors
            exploration_rate: Initial exploration rate for epsilon-greedy
        """
        self.num_movies = num_movies
        self.num_users = num_users
        self.feature_dim = feature_dim
        self.exploration_rate = exploration_rate
        
        # Initialize movies and users
        self.movies = self._generate_movies()
        self.users = self._generate_users()
        
        # Bandit state
        self.movie_estimates = defaultdict(lambda: defaultdict(float))
        self.movie_counts = defaultdict(lambda: defaultdict(int))
        self.user_movie_interactions = defaultdict(set)
        
        # Performance tracking
        self.recommendation_history = []
        self.rating_history = []
        self.regret_history = []
        
        # Content-based features
        self.movie_features = self._extract_movie_features()
        self.user_features = self._extract_user_features()
        
    def _generate_movies(self) -> Dict[int, Movie]:
        """Generate synthetic movie data."""
        movies = {}
        genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance', 
                 'Sci-Fi', 'Thriller', 'Documentary', 'Animation']
        
        for i in range(self.num_movies):
            movie_id = i + 1
            title = f"Movie {movie_id}"
            genres_list = random.sample(genres, random.randint(1, 3))
            year = random.randint(1990, 2024)
            rating = random.uniform(1.0, 5.0)
            num_ratings = random.randint(10, 10000)
            
            movies[movie_id] = Movie(
                movie_id=movie_id,
                title=title,
                genres=genres_list,
                year=year,
                rating=rating,
                num_ratings=num_ratings
            )
        
        return movies
    
    def _generate_users(self) -> Dict[int, User]:
        """Generate synthetic user data."""
        users = {}
        
        for i in range(self.num_users):
            user_id = i + 1
            users[user_id] = User(user_id=user_id)
        
        return users
    
    def _extract_movie_features(self) -> np.ndarray:
        """Extract content-based features from movies."""
        features = []
        for movie in self.movies.values():
            # Combine movie attributes into feature vector
            feature_vector = np.zeros(self.feature_dim)
            
            # Genre features (one-hot encoding)
            genre_mapping = {'Action': 0, 'Comedy': 1, 'Drama': 2, 'Horror': 3,
                           'Romance': 4, 'Sci-Fi': 5, 'Thriller': 6, 
                           'Documentary': 7, 'Animation': 8}
            
            for genre in movie.genres:
                if genre in genre_mapping:
                    feature_vector[genre_mapping[genre]] = 1.0
            
            # Year feature (normalized)
            feature_vector[9] = (movie.year - 1990) / 34.0
            
            # Rating feature
            feature_vector[10] = movie.rating / 5.0
            
            # Popularity feature
            feature_vector[11] = min(movie.num_ratings / 10000.0, 1.0)
            
            # Random features for diversity
            feature_vector[12:] = np.random.randn(self.feature_dim - 12)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _extract_user_features(self) -> np.ndarray:
        """Extract user preference features."""
        features = []
        
        for user in self.users.values():
            # Create user feature vector based on rating history
            feature_vector = np.zeros(self.feature_dim)
            
            if user.rating_history:
                # Average rating preference
                avg_rating = np.mean(list(user.rating_history.values()))
                feature_vector[0] = avg_rating / 5.0
                
                # Genre preferences
                genre_counts = defaultdict(int)
                for movie_id, rating in user.rating_history.items():
                    movie = self.movies[movie_id]
                    for genre in movie.genres:
                        genre_counts[genre] += rating
                
                # Normalize genre preferences
                total_rating = sum(genre_counts.values())
                if total_rating > 0:
                    for genre, count in genre_counts.items():
                        if genre in ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']:
                            genre_idx = {'Action': 1, 'Comedy': 2, 'Drama': 3, 
                                       'Horror': 4, 'Romance': 5}[genre]
                            feature_vector[genre_idx] = count / total_rating
            
            # Random features for cold-start users
            feature_vector[6:] = np.random.randn(self.feature_dim - 6)
            features.append(feature_vector)
        
        return np.array(features)
    
    def collaborative_filtering_score(self, user_id: int, movie_id: int) -> float:
        """
        Compute collaborative filtering score using user-movie interactions.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            
        Returns:
            Collaborative filtering score
        """
        user = self.users[user_id]
        
        if not user.rating_history:
            return 0.0
        
        # Find similar users
        similar_users = []
        for other_user_id, other_user in self.users.items():
            if other_user_id == user_id:
                continue
            
            # Compute similarity based on common rated movies
            common_movies = set(user.rating_history.keys()) & set(other_user.rating_history.keys())
            if len(common_movies) < 2:
                continue
            
            # Pearson correlation
            user_ratings = [user.rating_history[m] for m in common_movies]
            other_ratings = [other_user.rating_history[m] for m in common_movies]
            
            correlation = np.corrcoef(user_ratings, other_ratings)[0, 1]
            if not np.isnan(correlation):
                similar_users.append((other_user_id, correlation))
        
        # Sort by similarity
        similar_users.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Predict rating using similar users
        if similar_users:
            prediction = 0.0
            total_weight = 0.0
            
            for other_user_id, correlation in similar_users[:10]:  # Top 10 similar users
                other_user = self.users[other_user_id]
                if movie_id in other_user.rating_history:
                    weight = abs(correlation)
                    prediction += weight * other_user.rating_history[movie_id]
                    total_weight += weight
            
            if total_weight > 0:
                return prediction / total_weight
        
        return 0.0
    
    def content_based_score(self, user_id: int, movie_id: int) -> float:
        """
        Compute content-based filtering score.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            
        Returns:
            Content-based filtering score
        """
        user_idx = user_id - 1
        movie_idx = movie_id - 1
        
        if user_idx >= len(self.user_features) or movie_idx >= len(self.movie_features):
            return 0.0
        
        # Cosine similarity between user and movie features
        user_feature = self.user_features[user_idx]
        movie_feature = self.movie_features[movie_idx]
        
        similarity = np.dot(user_feature, movie_feature) / (
            np.linalg.norm(user_feature) * np.linalg.norm(movie_feature) + 1e-8
        )
        
        return max(0.0, similarity)  # Ensure non-negative
    
    def hybrid_score(self, user_id: int, movie_id: int, 
                    cf_weight: float = 0.6, cb_weight: float = 0.4) -> float:
        """
        Compute hybrid recommendation score.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            cf_weight: Weight for collaborative filtering
            cb_weight: Weight for content-based filtering
            
        Returns:
            Hybrid recommendation score
        """
        cf_score = self.collaborative_filtering_score(user_id, movie_id)
        cb_score = self.content_based_score(user_id, movie_id)
        
        # Normalize scores to [0, 1]
        cf_score = max(0.0, min(1.0, cf_score / 5.0))
        cb_score = max(0.0, min(1.0, cb_score))
        
        return cf_weight * cf_score + cb_weight * cb_score
    
    def epsilon_greedy_recommend(self, user_id: int, epsilon: float = None) -> int:
        """
        Recommend movie using epsilon-greedy strategy.
        
        Args:
            user_id: User identifier
            epsilon: Exploration rate (uses instance default if None)
            
        Returns:
            Recommended movie ID
        """
        if epsilon is None:
            epsilon = self.exploration_rate
        
        # Available movies (not rated by user)
        user = self.users[user_id]
        available_movies = [mid for mid in self.movies.keys() 
                          if mid not in user.rating_history]
        
        if not available_movies:
            return random.choice(list(self.movies.keys()))
        
        # Exploration: choose random movie
        if random.random() < epsilon:
            return random.choice(available_movies)
        
        # Exploitation: choose movie with highest estimated score
        best_movie = available_movies[0]
        best_score = -float('inf')
        
        for movie_id in available_movies:
            score = self.hybrid_score(user_id, movie_id)
            if score > best_score:
                best_score = score
                best_movie = movie_id
        
        return best_movie
    
    def ucb_recommend(self, user_id: int, alpha: float = 2.0) -> int:
        """
        Recommend movie using Upper Confidence Bound (UCB).
        
        Args:
            user_id: User identifier
            alpha: Exploration parameter
            
        Returns:
            Recommended movie ID
        """
        user = self.users[user_id]
        available_movies = [mid for mid in self.movies.keys() 
                          if mid not in user.rating_history]
        
        if not available_movies:
            return random.choice(list(self.movies.keys()))
        
        best_movie = available_movies[0]
        best_ucb = -float('inf')
        
        for movie_id in available_movies:
            # Get current estimate
            estimate = self.movie_estimates[user_id][movie_id]
            
            # Get number of times this movie has been recommended to this user
            count = self.movie_counts[user_id][movie_id]
            
            # UCB formula
            if count == 0:
                ucb = float('inf')  # Prioritize unexplored movies
            else:
                ucb = estimate + alpha * np.sqrt(np.log(len(self.recommendation_history) + 1) / count)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_movie = movie_id
        
        return best_movie
    
    def thompson_sampling_recommend(self, user_id: int) -> int:
        """
        Recommend movie using Thompson sampling.
        
        Args:
            user_id: User identifier
            
        Returns:
            Recommended movie ID
        """
        user = self.users[user_id]
        available_movies = [mid for mid in self.movies.keys() 
                          if mid not in user.rating_history]
        
        if not available_movies:
            return random.choice(list(self.movies.keys()))
        
        best_movie = available_movies[0]
        best_sample = -float('inf')
        
        for movie_id in available_movies:
            # Get current estimate and uncertainty
            estimate = self.movie_estimates[user_id][movie_id]
            count = self.movie_counts[user_id][movie_id]
            
            # Sample from posterior (assuming Beta distribution)
            if count == 0:
                # Uniform prior for unexplored movies
                sample = random.random()
            else:
                # Beta posterior based on successes and failures
                # Convert rating to success/failure (rating > 3.5 is success)
                successes = sum(1 for r in self.rating_history 
                              if r['movie_id'] == movie_id and r['rating'] > 3.5)
                failures = count - successes
                
                # Sample from Beta distribution
                sample = np.random.beta(successes + 1, failures + 1)
            
            if sample > best_sample:
                best_sample = sample
                best_movie = movie_id
        
        return best_movie
    
    def receive_rating(self, user_id: int, movie_id: int, rating: float):
        """
        Receive user rating and update bandit estimates.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            rating: User rating (1-5 scale)
        """
        # Update user rating history
        self.users[user_id].rating_history[movie_id] = rating
        
        # Update bandit estimates
        current_estimate = self.movie_estimates[user_id][movie_id]
        current_count = self.movie_counts[user_id][movie_id]
        
        # Incremental update
        new_count = current_count + 1
        new_estimate = (current_estimate * current_count + rating) / new_count
        
        self.movie_estimates[user_id][movie_id] = new_estimate
        self.movie_counts[user_id][movie_id] = new_count
        
        # Track interaction
        self.user_movie_interactions[user_id].add(movie_id)
        
        # Record for analysis
        self.rating_history.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': len(self.recommendation_history)
        })
    
    def simulate_recommendation_session(self, 
                                      user_id: int, 
                                      num_recommendations: int = 50,
                                      algorithm: str = 'hybrid') -> Dict:
        """
        Simulate a recommendation session for a user.
        
        Args:
            user_id: User identifier
            num_recommendations: Number of recommendations to make
            algorithm: Recommendation algorithm ('epsilon_greedy', 'ucb', 'thompson', 'hybrid')
            
        Returns:
            Session results dictionary
        """
        session_results = {
            'user_id': user_id,
            'algorithm': algorithm,
            'recommendations': [],
            'ratings': [],
            'cumulative_rating': 0.0,
            'exploration_count': 0
        }
        
        for step in range(num_recommendations):
            # Choose recommendation algorithm
            if algorithm == 'epsilon_greedy':
                movie_id = self.epsilon_greedy_recommend(user_id)
            elif algorithm == 'ucb':
                movie_id = self.ucb_recommend(user_id)
            elif algorithm == 'thompson':
                movie_id = self.thompson_sampling_recommend(user_id)
            elif algorithm == 'hybrid':
                # Use hybrid approach with adaptive exploration
                if step < num_recommendations * 0.3:  # More exploration early
                    movie_id = self.epsilon_greedy_recommend(user_id, epsilon=0.3)
                else:
                    movie_id = self.ucb_recommend(user_id)
            else:
                movie_id = self.epsilon_greedy_recommend(user_id)
            
            # Simulate user rating (real system would get actual rating)
            true_rating = self._simulate_user_rating(user_id, movie_id)
            
            # Receive rating and update
            self.receive_rating(user_id, movie_id, true_rating)
            
            # Track results
            session_results['recommendations'].append(movie_id)
            session_results['ratings'].append(true_rating)
            session_results['cumulative_rating'] += true_rating
            
            # Track exploration
            if algorithm == 'epsilon_greedy' and random.random() < self.exploration_rate:
                session_results['exploration_count'] += 1
        
        return session_results
    
    def _simulate_user_rating(self, user_id: int, movie_id: int) -> float:
        """
        Simulate user rating based on movie and user characteristics.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            
        Returns:
            Simulated rating (1-5 scale)
        """
        movie = self.movies[movie_id]
        user = self.users[user_id]
        
        # Base rating from movie quality
        base_rating = movie.rating
        
        # User preference adjustment
        user_adjustment = 0.0
        if user.rating_history:
            avg_user_rating = np.mean(list(user.rating_history.values()))
            user_adjustment = (avg_user_rating - 3.0) * 0.5
        
        # Genre preference adjustment
        genre_adjustment = 0.0
        if user.rating_history:
            genre_ratings = defaultdict(list)
            for mid, rating in user.rating_history.items():
                for genre in self.movies[mid].genres:
                    genre_ratings[genre].append(rating)
            
            for genre in movie.genres:
                if genre in genre_ratings:
                    avg_genre_rating = np.mean(genre_ratings[genre])
                    genre_adjustment += (avg_genre_rating - 3.0) * 0.3
        
        # Add noise
        noise = np.random.normal(0, 0.5)
        
        # Compute final rating
        final_rating = base_rating + user_adjustment + genre_adjustment + noise
        
        # Clamp to [1, 5]
        return max(1.0, min(5.0, final_rating))
    
    def evaluate_recommendations(self, 
                               num_users: int = 50,
                               recommendations_per_user: int = 20) -> Dict:
        """
        Evaluate recommendation system performance.
        
        Args:
            num_users: Number of users to evaluate
            recommendations_per_user: Number of recommendations per user
            
        Returns:
            Evaluation results dictionary
        """
        algorithms = ['epsilon_greedy', 'ucb', 'thompson', 'hybrid']
        results = {}
        
        for algorithm in algorithms:
            print(f"Evaluating {algorithm} algorithm...")
            
            algorithm_results = {
                'cumulative_ratings': [],
                'average_ratings': [],
                'exploration_rates': [],
                'coverage': []
            }
            
            for user_id in range(1, min(num_users + 1, self.num_users + 1)):
                # Reset user state for fair comparison
                self.users[user_id].rating_history = {}
                
                # Run recommendation session
                session = self.simulate_recommendation_session(
                    user_id, recommendations_per_user, algorithm
                )
                
                algorithm_results['cumulative_ratings'].append(session['cumulative_rating'])
                algorithm_results['average_ratings'].append(
                    session['cumulative_rating'] / recommendations_per_user
                )
                algorithm_results['exploration_rates'].append(
                    session['exploration_count'] / recommendations_per_user
                )
                
                # Calculate coverage (unique movies recommended)
                unique_movies = len(set(session['recommendations']))
                coverage = unique_movies / recommendations_per_user
                algorithm_results['coverage'].append(coverage)
            
            results[algorithm] = algorithm_results
        
        return results
    
    def plot_evaluation_results(self, results: Dict):
        """
        Plot evaluation results for comparison.
        
        Args:
            results: Evaluation results dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Movie Recommendation System Evaluation', fontsize=16)
        
        algorithms = list(results.keys())
        
        # Average ratings
        avg_ratings = [np.mean(results[alg]['average_ratings']) for alg in algorithms]
        std_ratings = [np.std(results[alg]['average_ratings']) for alg in algorithms]
        
        axes[0, 0].bar(algorithms, avg_ratings, yerr=std_ratings, capsize=5)
        axes[0, 0].set_title('Average Rating per Recommendation')
        axes[0, 0].set_ylabel('Average Rating')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Cumulative ratings
        cum_ratings = [np.mean(results[alg]['cumulative_ratings']) for alg in algorithms]
        std_cum = [np.std(results[alg]['cumulative_ratings']) for alg in algorithms]
        
        axes[0, 1].bar(algorithms, cum_ratings, yerr=std_cum, capsize=5)
        axes[0, 1].set_title('Cumulative Rating per Session')
        axes[0, 1].set_ylabel('Cumulative Rating')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Exploration rates
        exp_rates = [np.mean(results[alg]['exploration_rates']) for alg in algorithms]
        std_exp = [np.std(results[alg]['exploration_rates']) for alg in algorithms]
        
        axes[1, 0].bar(algorithms, exp_rates, yerr=std_exp, capsize=5)
        axes[1, 0].set_title('Exploration Rate')
        axes[1, 0].set_ylabel('Exploration Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Coverage
        coverage = [np.mean(results[alg]['coverage']) for alg in algorithms]
        std_cov = [np.std(results[alg]['coverage']) for alg in algorithms]
        
        axes[1, 1].bar(algorithms, coverage, yerr=std_cov, capsize=5)
        axes[1, 1].set_title('Recommendation Coverage')
        axes[1, 1].set_ylabel('Coverage Ratio')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def demonstrate_cold_start(self):
        """Demonstrate cold-start handling for new users and movies."""
        print("=== Cold-Start Demonstration ===")
        
        # New user cold-start
        new_user_id = self.num_users + 1
        self.users[new_user_id] = User(user_id=new_user_id)
        
        print(f"\n1. New User Cold-Start (User {new_user_id}):")
        print("   - No rating history")
        print("   - Must rely on content-based features")
        print("   - Will gradually build preference profile")
        
        # Simulate recommendations for new user
        session = self.simulate_recommendation_session(new_user_id, 10, 'hybrid')
        print(f"   - Average rating: {np.mean(session['ratings']):.2f}")
        print(f"   - Unique movies: {len(set(session['recommendations']))}")
        
        # New movie cold-start
        new_movie_id = self.num_movies + 1
        self.movies[new_movie_id] = Movie(
            movie_id=new_movie_id,
            title="New Movie",
            genres=['Action', 'Thriller'],
            year=2024,
            rating=3.5,
            num_ratings=0
        )
        
        print(f"\n2. New Movie Cold-Start (Movie {new_movie_id}):")
        print("   - No rating history")
        print("   - Must rely on content-based features")
        print("   - Will be explored through bandit algorithms")
        
        # Show how different algorithms handle cold-start
        algorithms = ['epsilon_greedy', 'ucb', 'thompson']
        for alg in algorithms:
            # Reset user for fair comparison
            test_user_id = 1
            self.users[test_user_id].rating_history = {}
            
            # Check if new movie gets recommended
            session = self.simulate_recommendation_session(test_user_id, 20, alg)
            new_movie_recommended = new_movie_id in session['recommendations']
            print(f"   - {alg}: New movie recommended = {new_movie_recommended}")
    
    def demonstrate_personalization(self):
        """Demonstrate personalization capabilities."""
        print("\n=== Personalization Demonstration ===")
        
        # Create users with different preferences
        action_lover = User(user_id=1001)
        comedy_lover = User(user_id=1002)
        drama_lover = User(user_id=1003)
        
        # Simulate some rating history to establish preferences
        action_movies = [mid for mid, movie in self.movies.items() 
                        if 'Action' in movie.genres][:5]
        comedy_movies = [mid for mid, movie in self.movies.items() 
                        if 'Comedy' in movie.genres][:5]
        drama_movies = [mid for mid, movie in self.movies.items() 
                       if 'Drama' in movie.genres][:5]
        
        # Action lover rates action movies highly
        for movie_id in action_movies:
            action_lover.rating_history[movie_id] = random.uniform(4.0, 5.0)
        
        # Comedy lover rates comedy movies highly
        for movie_id in comedy_movies:
            comedy_lover.rating_history[movie_id] = random.uniform(4.0, 5.0)
        
        # Drama lover rates drama movies highly
        for movie_id in drama_movies:
            drama_lover.rating_history[movie_id] = random.uniform(4.0, 5.0)
        
        self.users[1001] = action_lover
        self.users[1002] = comedy_lover
        self.users[1003] = drama_lover
        
        # Test personalization
        users = [1001, 1002, 1003]
        user_types = ['Action Lover', 'Comedy Lover', 'Drama Lover']
        
        for user_id, user_type in zip(users, user_types):
            print(f"\n{user_type} (User {user_id}):")
            
            # Get recommendations
            recommendations = []
            for _ in range(10):
                movie_id = self.epsilon_greedy_recommend(user_id, epsilon=0.2)
                recommendations.append(movie_id)
            
            # Analyze recommendations
            action_count = sum(1 for mid in recommendations 
                             if 'Action' in self.movies[mid].genres)
            comedy_count = sum(1 for mid in recommendations 
                             if 'Comedy' in self.movies[mid].genres)
            drama_count = sum(1 for mid in recommendations 
                            if 'Drama' in self.movies[mid].genres)
            
            print(f"   - Action movies: {action_count}/10")
            print(f"   - Comedy movies: {comedy_count}/10")
            print(f"   - Drama movies: {drama_count}/10")

def main():
    """Main demonstration function."""
    print("Movie Recommendation System using Multi-Armed Bandits")
    print("=" * 60)
    
    # Initialize recommendation system
    recommender = MovieRecommendationBandit(
        num_movies=500,
        num_users=200,
        feature_dim=50,
        exploration_rate=0.1
    )
    
    print(f"Initialized with {recommender.num_movies} movies and {recommender.num_users} users")
    
    # Demonstrate cold-start handling
    recommender.demonstrate_cold_start()
    
    # Demonstrate personalization
    recommender.demonstrate_personalization()
    
    # Evaluate different algorithms
    print("\n=== Algorithm Evaluation ===")
    print("Running evaluation (this may take a moment)...")
    
    results = recommender.evaluate_recommendations(
        num_users=30,
        recommendations_per_user=15
    )
    
    # Plot results
    recommender.plot_evaluation_results(results)
    
    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    for algorithm, result in results.items():
        avg_rating = np.mean(result['average_ratings'])
        avg_coverage = np.mean(result['coverage'])
        avg_exploration = np.mean(result['exploration_rates'])
        
        print(f"{algorithm.upper()}:")
        print(f"  - Average Rating: {avg_rating:.3f}")
        print(f"  - Coverage: {avg_coverage:.3f}")
        print(f"  - Exploration Rate: {avg_exploration:.3f}")
    
    print("\n=== Key Insights ===")
    print("1. Hybrid algorithms often perform best by combining multiple signals")
    print("2. Exploration is crucial for discovering new good movies")
    print("3. Personalization improves with more user interaction data")
    print("4. Cold-start problems can be mitigated with content-based features")
    print("5. Bandit algorithms naturally balance exploration and exploitation")

if __name__ == "__main__":
    main() 
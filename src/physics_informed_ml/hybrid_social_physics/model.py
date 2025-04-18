"""
Hybrid Social-Physics Model Implementation

This model combines the physics-informed differential equation approach
with social network influence to create a comprehensive recommendation system.

Key components:
1. Physics-informed differential equation (du/dt = G*F)
2. Social network influence weighting
3. Focused training on top-N most frequent items
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
import networkx as nx

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # physics_informed_ml directory
sys.path.append(parent_dir)

# Import from physics_informed_ml
from economist_model.preference_calculator import calculate_user_preference
from differential_equation.operator_estimator import OperatorEstimator

class HybridSocialPhysicsModel:
    """
    Hybrid model combining physics-informed differential equations with social influence.
    
    This model leverages:
    1. Differential equation for preference evolution (du/dt = G*F)
    2. Social network influence from connected users
    3. Focused approach on top-N most frequent items
    """
    
    def __init__(self, top_n_items=5, preference_evolution_weight=0.6, social_weight=0.3):
        """
        Initialize the hybrid model
        
        Args:
            top_n_items (int): Number of top items to focus on
            preference_evolution_weight (float): Weight for preference evolution in predictions
            social_weight (float): Weight for social influence in predictions
        """
        self.top_n_items = top_n_items
        self.preference_evolution_weight = preference_evolution_weight
        self.social_weight = social_weight
        self.transition_matrix = {}
        self.item_frequency = Counter()
        self.top_items = []
        self.fallback_item = None
        self.preference_operators = {}
        self.social_graph = None
        self.user_preferences = {}
        
    def fit(self, user_item_df, social_df=None):
        """
        Build the hybrid model with transition matrix, preference evolution, and social influence
        
        Args:
            user_item_df (DataFrame): DataFrame with user-item interactions 
                                     (user_id, item_id, view_time, click_rate, timestamp)
            social_df (DataFrame): DataFrame with social connections
                                   (user_id, friend_id, influence_weight)
            
        Returns:
            self: Fitted model
        """
        print(f"\n{'='*70}\nTraining Hybrid Social-Physics Model\n{'='*70}")
        
        # Extract relevant columns and ensure proper types
        df = user_item_df.copy()
        
        # Ensure timestamps are properly formatted
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Convert to Unix timestamp for numerical operations
        df['timestamp_unix'] = df['timestamp'].astype(int) / 10**9
        
        # Calculate time deltas
        df['time_delta'] = df.groupby('user_id')['timestamp_unix'].diff().fillna(1.0)
        df['time_delta'] = df['time_delta'].apply(lambda x: max(x, 0.001))  # Ensure positive values
        
        # Calculate preferences using economist's equation
        df['user_preference'] = df.apply(
            lambda row: calculate_user_preference(row['view_time'], row['click_rate']), 
            axis=1
        )
        
        # 1. Build transition matrices for each user
        self._build_transition_matrices(df)
        
        # 2. Build social network if provided
        if social_df is not None:
            self._build_social_network(social_df)
        
        # 3. Fit preference evolution model
        self._fit_preference_evolution(df)
        
        return self
    
    def _build_transition_matrices(self, df):
        """Build transition matrices for each user"""
        print("Building user transition matrices...")
        
        # Count item frequencies
        self.item_frequency = Counter(df['item_id'].tolist())
        
        # Identify top N items
        top_items = self.item_frequency.most_common(self.top_n_items)
        self.top_items = [item for item, _ in top_items]
        
        # Set fallback item (most common overall)
        if self.top_items:
            self.fallback_item = self.top_items[0]
            
        # Initialize transition matrices
        user_transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Process each user's interactions
        for user_id, user_df in df.groupby('user_id'):
            # Sort by timestamp to ensure correct sequence
            user_df = user_df.sort_values('timestamp')
            
            # Count transitions from each item to next item
            for i in range(len(user_df) - 1):
                current_item = user_df.iloc[i]['item_id']
                next_item = user_df.iloc[i+1]['item_id']
                user_transitions[user_id][current_item][next_item] += 1
                
            # Store preferences for this user
            self.user_preferences[user_id] = {}
            for _, row in user_df.iterrows():
                item_id = row['item_id']
                preference = row['user_preference']
                
                # Update running average of preferences for this item
                if item_id in self.user_preferences[user_id]:
                    old_pref = self.user_preferences[user_id][item_id][0]
                    old_count = self.user_preferences[user_id][item_id][1]
                    new_count = old_count + 1
                    new_pref = (old_pref * old_count + preference) / new_count
                    self.user_preferences[user_id][item_id] = (new_pref, new_count)
                else:
                    self.user_preferences[user_id][item_id] = (preference, 1)
        
        # Convert to probability transition matrices
        self.transition_matrix = defaultdict(dict)
        for user_id, user_transitions_dict in user_transitions.items():
            for current_item, next_items_dict in user_transitions_dict.items():
                total = sum(next_items_dict.values())
                
                if total > 0:  # Avoid division by zero
                    self.transition_matrix[user_id][current_item] = {
                        next_item: count/total for next_item, count in next_items_dict.items()
                    }
        
        print(f"Built transition matrices for {len(self.transition_matrix)} users")
        
    def _build_social_network(self, social_df):
        """Build social network from connection data"""
        print("Building social network...")
        
        # Create graph
        self.social_graph = nx.Graph()
        
        # Add nodes (users)
        unique_users = social_df['user_id'].unique()
        self.social_graph.add_nodes_from(unique_users)
        
        # Add edges with weights
        for _, row in social_df.iterrows():
            self.social_graph.add_edge(
                row['user_id'], 
                row['friend_id'], 
                weight=row.get('influence_weight', 1.0)
            )
            
        print(f"Built social graph with {self.social_graph.number_of_nodes()} users " 
              f"and {self.social_graph.number_of_edges()} connections")
    
    def _fit_preference_evolution(self, df):
        """Fit preference evolution operator for each user"""
        print("Training preference evolution models...")
        
        # Group by user for individual models
        for user_id, user_df in df.groupby('user_id'):
            # Sort by timestamp
            user_df = user_df.sort_values('timestamp')
            
            if len(user_df) < 5:  # Need minimum data to fit
                continue
                
            # Extract features for preference evolution
            preferences = user_df['user_preference'].tolist()
            factors = user_df[['view_time', 'click_rate']].values
            time_intervals = user_df['time_delta'].tolist()
            
            # Create and fit operator
            operator = OperatorEstimator()
            try:
                operator.fit(preferences, factors, time_intervals)
                self.preference_operators[user_id] = operator
            except Exception as e:
                print(f"Error fitting preference operator for user {user_id}: {e}")
        
        print(f"Trained preference evolution models for {len(self.preference_operators)} users")
    
    def predict(self, user_id, current_item, view_time=None, click_rate=None, 
                last_preference=0.5, time_delta=1.0):
        """
        Predict next item combining transition probabilities, preference evolution, and social influence
        
        Args:
            user_id: ID of the user
            current_item: Current item ID
            view_time: Current view time
            click_rate: Current click rate
            last_preference: Last calculated preference
            time_delta: Time since last interaction
            
        Returns:
            Predicted next item ID
        """
        # Predictions from different components
        transition_pred = self._get_transition_prediction(user_id, current_item)
        physics_pred = self._get_physics_prediction(user_id, current_item, 
                                                   view_time, click_rate,
                                                   last_preference, time_delta)
        social_pred = self._get_social_prediction(user_id, current_item)
        
        # If no valid predictions, return fallback
        if not any([transition_pred, physics_pred, social_pred]):
            return self.fallback_item
        
        # Combine predictions with weights
        combined_scores = defaultdict(float)
        
        # Base transition probability (1 - preference_evolution_weight - social_weight)
        base_weight = max(0, 1 - self.preference_evolution_weight - self.social_weight)
        
        if transition_pred:
            combined_scores[transition_pred] += base_weight
            
        if physics_pred:
            combined_scores[physics_pred] += self.preference_evolution_weight
            
        if social_pred:
            combined_scores[social_pred] += self.social_weight
            
        # Return item with highest combined score
        if combined_scores:
            return max(combined_scores.items(), key=lambda x: x[1])[0]
        
        # Fallback to most common item
        return self.fallback_item
    
    def _get_transition_prediction(self, user_id, current_item):
        """Get prediction based on transition matrix"""
        if user_id not in self.transition_matrix:
            return None
            
        if current_item not in self.transition_matrix[user_id]:
            return None
            
        transitions = self.transition_matrix[user_id][current_item]
        if not transitions:
            return None
            
        return max(transitions.items(), key=lambda x: x[1])[0]
    
    def _get_physics_prediction(self, user_id, current_item, view_time, click_rate,
                              last_preference, time_delta):
        """Get prediction based on physics-informed preference evolution"""
        if user_id not in self.preference_operators:
            return None
            
        if view_time is None or click_rate is None:
            return None
            
        try:
            # Predict preference evolution
            operator = self.preference_operators[user_id]
            factors = np.array([[view_time, click_rate]])
            time_intervals = np.array([time_delta])
            next_preference = operator.predict_preference_evolution(
                last_preference, factors, time_intervals)[1]
                
            # Find item with closest preference
            if user_id in self.user_preferences:
                closest_item = None
                min_diff = float('inf')
                
                for item, (pref, _) in self.user_preferences[user_id].items():
                    diff = abs(pref - next_preference)
                    if diff < min_diff:
                        min_diff = diff
                        closest_item = item
                        
                return closest_item
        except Exception as e:
            print(f"Error in physics prediction: {e}")
            
        return None
    
    def _get_social_prediction(self, user_id, current_item):
        """Get prediction based on social influence"""
        if not self.social_graph or user_id not in self.social_graph:
            return None
            
        # Get friends and their influence weights
        friends = []
        for friend_id in self.social_graph.neighbors(user_id):
            weight = self.social_graph[user_id][friend_id].get('weight', 1.0)
            friends.append((friend_id, weight))
            
        if not friends:
            return None
            
        # Get predictions from friends
        friend_predictions = defaultdict(float)
        for friend_id, influence_weight in friends:
            pred = self._get_transition_prediction(friend_id, current_item)
            if pred:
                friend_predictions[pred] += influence_weight
                
        # Return the most influenced prediction
        if friend_predictions:
            return max(friend_predictions.items(), key=lambda x: x[1])[0]
            
        return None
    
    def predict_top_k(self, user_id, current_item, k=5, **kwargs):
        """
        Get top-k predictions combining physics and social methods
        
        Args:
            user_id: ID of the user
            current_item: Current item ID
            k: Number of predictions to return
            **kwargs: Additional arguments for prediction
            
        Returns:
            List of top-k predicted items
        """
        # Get all predictions with scores
        all_scores = self._get_all_prediction_scores(user_id, current_item, **kwargs)
        
        # Sort by score
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_items = [item for item, _ in sorted_items[:k]]
        
        # If not enough predictions, add popular items
        if len(top_k_items) < k:
            for item in self.top_items:
                if item not in top_k_items:
                    top_k_items.append(item)
                    if len(top_k_items) >= k:
                        break
        
        return top_k_items[:k]
    
    def _get_all_prediction_scores(self, user_id, current_item, **kwargs):
        """Get scores for all possible next items"""
        scores = defaultdict(float)
        view_time = kwargs.get('view_time')
        click_rate = kwargs.get('click_rate')
        last_preference = kwargs.get('last_preference', 0.5)
        time_delta = kwargs.get('time_delta', 1.0)
        
        # Add transition scores
        base_weight = max(0, 1 - self.preference_evolution_weight - self.social_weight)
        if user_id in self.transition_matrix and current_item in self.transition_matrix[user_id]:
            for next_item, prob in self.transition_matrix[user_id][current_item].items():
                scores[next_item] += base_weight * prob
        
        # Add physics-based scores
        physics_pred = self._get_physics_prediction(
            user_id, current_item, view_time, click_rate, last_preference, time_delta
        )
        if physics_pred:
            scores[physics_pred] += self.preference_evolution_weight
        
        # Add social scores
        if self.social_graph and user_id in self.social_graph:
            for friend_id in self.social_graph.neighbors(user_id):
                influence_weight = self.social_graph[user_id][friend_id].get('weight', 1.0)
                
                if friend_id in self.transition_matrix and current_item in self.transition_matrix[friend_id]:
                    for next_item, prob in self.transition_matrix[friend_id][current_item].items():
                        scores[next_item] += self.social_weight * influence_weight * prob
        
        return scores
    
    def save(self, filepath):
        """Save the hybrid model to a file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load a hybrid model from a file"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

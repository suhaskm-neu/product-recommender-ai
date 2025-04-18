"""
Markov model integrated with differential equation-based preference evolution.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
import os
import sys

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from differential_equation.operator_estimator import OperatorEstimator
from economist_model.preference_calculator import calculate_user_preference

class DifferentialMarkovModel:
    def __init__(self, top_n_items=100, preference_evolution_weight=0.5):
        """
        Markov model integrated with differential equation-based preference evolution
        
        Args:
            top_n_items (int): Number of most frequent items to consider
            preference_evolution_weight (float): Weight for preference evolution in predictions
        """
        self.top_n_items = top_n_items
        self.preference_evolution_weight = preference_evolution_weight
        self.transition_matrix = {}
        self.item_frequency = Counter()
        self.top_items = []
        self.fallback_item = None
        self.preference_operator = OperatorEstimator()
        
    def fit(self, user_item_sequence, item_ids=None, view_times=None, click_rates=None, timestamps=None):
        """
        Build model with transition matrix and preference evolution operator
        
        Args:
            user_item_sequence (list): Sequence of item IDs
            item_ids (list): All item IDs in the dataset (can be None if same as user_item_sequence)
            view_times (list): View times for each interaction
            click_rates (list): Click rates for each interaction
            timestamps (list): Timestamps for each interaction
            
        Returns:
            self: Fitted model
        """
        print(f"Fitting differential equation Markov model with {len(user_item_sequence)} items")
        print(f"Top N items: {self.top_n_items}, Preference evolution weight: {self.preference_evolution_weight}")
        
        # If item_ids not provided, use the sequence
        if item_ids is None:
            item_ids = user_item_sequence
            
        # Count item frequency (for fallback prediction and top items)
        self.item_frequency = Counter(item_ids)
        print(f"Dataset has {len(self.item_frequency)} unique items")
        
        # Identify top N items by frequency
        top_items = self.item_frequency.most_common(self.top_n_items)
        self.top_items = [item for item, _ in top_items]
        
        # Set fallback item (most common overall)
        if self.top_items:
            self.fallback_item = self.top_items[0]
            print(f"Most frequent item (fallback): {self.fallback_item}, " 
                  f"frequency: {self.item_frequency[self.fallback_item]}")
        
        # Prepare preference evolution data if all required inputs are available
        if all(x is not None for x in [view_times, click_rates, timestamps]):
            print("Fitting preference evolution model using differential equation")
            
            # Calculate time intervals
            time_intervals = []
            for i in range(len(timestamps) - 1):
                delta = max(timestamps[i + 1] - timestamps[i], 0.001)  # Ensure positive
                time_intervals.append(delta)
            
            # Calculate initial preferences using economist's model
            preferences = [calculate_user_preference(vt, cr) for vt, cr in zip(view_times, click_rates)]
            
            # Create factor matrix F with [view_time, click_rate]
            factors = np.column_stack((view_times, click_rates))
            
            # Fit the preference evolution operator
            self.preference_operator.fit(preferences, factors, time_intervals)
            
            # Store estimated preferences for each item
            self.item_preferences = {}
            
            # Calculate average preference for each item
            temp_prefs = defaultdict(list)
            for i, item_id in enumerate(item_ids):
                temp_prefs[item_id].append(preferences[i])
                
            for item_id, pref_list in temp_prefs.items():
                self.item_preferences[item_id] = sum(pref_list) / len(pref_list)
                
            # Log preference stats for top items
            avg_pref = sum(preferences) / len(preferences)
            top_item_prefs = [self.item_preferences.get(item, 0) for item in self.top_items[:10]]
            print(f"Average preference: {avg_pref:.4f}")
            print(f"Top 10 items average preference: {sum(top_item_prefs)/len(top_item_prefs):.4f}")
        else:
            print("Warning: Missing data for preference evolution model")
            print("Using default preference values")
            self.item_preferences = {item: 0.5 for item in self.item_frequency}
        
        # Build transition matrix
        transition_counts = defaultdict(Counter)
        
        # Count transitions
        for i in range(len(user_item_sequence) - 1):
            current_item = user_item_sequence[i]
            next_item = user_item_sequence[i + 1]
            transition_counts[current_item][next_item] += 1
        
        # Convert to probabilities
        for current_item, next_items in transition_counts.items():
            total = sum(next_items.values())
            
            self.transition_matrix[current_item] = {}
            
            for next_item, count in next_items.items():
                # Base transition probability
                self.transition_matrix[current_item][next_item] = count / total
        
        print(f"Built transition matrix with {len(self.transition_matrix)} source items")
        return self
    
    def predict(self, current_item, view_time=None, click_rate=None, last_preference=0.5, time_delta=1.0):
        """
        Predict next item using both transition probabilities and preference evolution
        
        Args:
            current_item: Current item ID
            view_time: Current view time
            click_rate: Current click rate
            last_preference: Last calculated preference value
            time_delta: Time since last interaction
            
        Returns:
            Predicted next item ID
        """
        # Get base transition probabilities
        if current_item not in self.transition_matrix:
            # Fall back to the most common item
            return self.fallback_item
            
        transition_probs = self.transition_matrix[current_item]
        
        # Apply preference evolution if we have the required data
        if view_time is not None and click_rate is not None:
            try:
                # Predict preference evolution
                factors = np.array([[view_time, click_rate]])
                time_intervals = np.array([time_delta])
                next_preference = self.preference_operator.predict_preference_evolution(
                    last_preference, factors, time_intervals)[1]
                
                # Adjust transition probabilities with preference evolution
                adjusted_probs = {}
                for next_item, prob in transition_probs.items():
                    # Get item's base preference
                    item_pref = self.item_preferences.get(next_item, 0.5)
                    
                    # Calculate preference evolution factor (how much the preference changed)
                    evolution_factor = next_preference / max(last_preference, 0.001)
                    
                    # Apply evolution factor to item's preference
                    evolved_pref = item_pref * evolution_factor
                    evolved_pref = max(0.001, min(0.999, evolved_pref))
                    
                    # Weighted combination of transition probability and evolved preference
                    adjusted_probs[next_item] = (
                        (1 - self.preference_evolution_weight) * prob + 
                        self.preference_evolution_weight * evolved_pref
                    )
                    
                # Use adjusted probabilities
                if adjusted_probs:
                    return max(adjusted_probs.items(), key=lambda x: x[1])[0]
            except Exception as e:
                print(f"Error in preference evolution prediction: {e}")
                # Fall back to base transition matrix
                pass
        
        # Fall back to base transition probabilities
        if not transition_probs:
            return self.fallback_item
        
        # Return item with highest transition probability
        return max(transition_probs.items(), key=lambda x: x[1])[0]
    
    def predict_top_k(self, current_item, k=5, view_time=None, click_rate=None, last_preference=0.5, time_delta=1.0):
        """
        Predict the top k next items with preference evolution
        
        Args:
            current_item: Current item ID
            k (int): Number of predictions to return
            view_time: Current view time
            click_rate: Current click rate
            last_preference: Last calculated preference value
            time_delta: Time since last interaction
            
        Returns:
            list: Top k predicted items
        """
        # Get base transition probabilities
        if current_item not in self.transition_matrix:
            # Fall back to the most common items
            return self.top_items[:k]
            
        transition_probs = self.transition_matrix[current_item]
        
        # Apply preference evolution if we have the required data
        if view_time is not None and click_rate is not None:
            try:
                # Predict preference evolution
                factors = np.array([[view_time, click_rate]])
                time_intervals = np.array([time_delta])
                next_preference = self.preference_operator.predict_preference_evolution(
                    last_preference, factors, time_intervals)[1]
                
                # Adjust transition probabilities with preference evolution
                adjusted_probs = {}
                for next_item, prob in transition_probs.items():
                    # Get item's base preference
                    item_pref = self.item_preferences.get(next_item, 0.5)
                    
                    # Calculate preference evolution factor
                    evolution_factor = next_preference / max(last_preference, 0.001)
                    
                    # Apply evolution factor to item's preference
                    evolved_pref = item_pref * evolution_factor
                    evolved_pref = max(0.001, min(0.999, evolved_pref))
                    
                    # Weighted combination
                    adjusted_probs[next_item] = (
                        (1 - self.preference_evolution_weight) * prob + 
                        self.preference_evolution_weight * evolved_pref
                    )
                
                # Use adjusted probabilities
                if adjusted_probs:
                    sorted_items = sorted(adjusted_probs.items(), key=lambda x: x[1], reverse=True)
                    top_k_items = [item for item, _ in sorted_items[:k]]
                    
                    # If we don't have enough predictions, pad with popular items
                    if len(top_k_items) < k:
                        for item in self.top_items:
                            if item not in top_k_items:
                                top_k_items.append(item)
                                if len(top_k_items) >= k:
                                    break
                    
                    return top_k_items
                    
            except Exception as e:
                print(f"Error in preference evolution prediction: {e}")
                # Fall back to base transition matrix
                pass
        
        # Fall back to base transition probabilities
        if not transition_probs:
            return self.top_items[:k]
        
        # Sort by probability
        sorted_items = sorted(transition_probs.items(), key=lambda x: x[1], reverse=True)
        top_k_items = [item for item, _ in sorted_items[:k]]
        
        # If we don't have enough predictions, pad with popular items
        if len(top_k_items) < k:
            for item in self.top_items:
                if item not in top_k_items:
                    top_k_items.append(item)
                    if len(top_k_items) >= k:
                        break
        
        return top_k_items
    
    def save(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath (str): Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            DifferentialMarkovModel: Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

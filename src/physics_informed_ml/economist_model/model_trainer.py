"""
Markov Model enhanced with user preference information from the economist's equation.
"""
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
import os

class PreferenceEnhancedMarkovModel:
    def __init__(self, top_n_items=100, preference_weight=0.5):
        """
        Markov Model enhanced with user preference information
        
        Args:
            top_n_items (int): Number of most frequent items to consider
            preference_weight (float): Weight given to preference in transition probabilities
        """
        self.top_n_items = top_n_items
        self.preference_weight = preference_weight
        self.transition_matrix = {}
        self.preference_scores = {}
        self.item_frequency = Counter()
        self.top_items = []
        self.fallback_item = None
        
    def fit(self, user_item_sequence, item_ids=None, preferences=None):
        """
        Build transition matrix enhanced with preference information
        
        Args:
            user_item_sequence (list): Sequence of item IDs
            item_ids (list): All item IDs in the sequence (can be None if same as user_item_sequence)
            preferences (list): User preference scores for each item (can be None)
            
        Returns:
            self: Fitted model
        """
        print(f"Fitting preference-enhanced Markov model with {len(user_item_sequence)} items")
        print(f"Top N items: {self.top_n_items}, Preference weight: {self.preference_weight}")
        
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
        
        # Store preference scores for each item if provided
        if preferences is not None and len(preferences) == len(item_ids):
            print("Using provided preference scores")
            # Convert single values to lists for consistent handling
            temp_prefs = defaultdict(list)
            
            for item_id, pref in zip(item_ids, preferences):
                temp_prefs[item_id].append(pref)
            
            # Calculate average preference for each item
            for item_id, pref_list in temp_prefs.items():
                self.preference_scores[item_id] = sum(pref_list) / len(pref_list)
                
            # Log preference stats for top items
            avg_pref = sum(preferences) / len(preferences)
            top_item_prefs = [self.preference_scores.get(item, 0) for item in self.top_items[:10]]
            print(f"Average preference: {avg_pref:.4f}")
            print(f"Top 10 items average preference: {sum(top_item_prefs)/len(top_item_prefs):.4f}")
        else:
            print("No preference scores provided, using equal preferences")
            # Assign default preference of 0.5 to all items
            for item_id in self.item_frequency:
                self.preference_scores[item_id] = 0.5
        
        # Build transition matrix
        transition_counts = defaultdict(Counter)
        
        # Count transitions
        for i in range(len(user_item_sequence) - 1):
            current_item = user_item_sequence[i]
            next_item = user_item_sequence[i + 1]
            transition_counts[current_item][next_item] += 1
        
        # Convert to probabilities and apply preference weighting
        for current_item, next_items in transition_counts.items():
            total = sum(next_items.values())
            
            self.transition_matrix[current_item] = {}
            
            for next_item, count in next_items.items():
                # Base transition probability
                transition_prob = count / total
                
                # Get preference factor for the next item
                preference_factor = self.preference_scores.get(next_item, 0.5)
                
                # Weighted combination of transition probability and preference
                self.transition_matrix[current_item][next_item] = (
                    (1 - self.preference_weight) * transition_prob + 
                    self.preference_weight * preference_factor
                )
        
        print(f"Built transition matrix with {len(self.transition_matrix)} source items")
        return self
    
    def predict(self, current_item):
        """
        Predict the next item based on the enhanced transition matrix
        
        Args:
            current_item: The current item ID
            
        Returns:
            The predicted next item ID
        """
        if current_item not in self.transition_matrix:
            # Fall back to the most common item
            return self.fallback_item
            
        next_items = self.transition_matrix[current_item]
        if not next_items:
            return self.fallback_item
            
        # Return the item with highest enhanced probability
        return max(next_items.items(), key=lambda x: x[1])[0]
    
    def predict_top_k(self, current_item, k=5):
        """
        Predict the top k next items based on the enhanced transition matrix
        
        Args:
            current_item: The current item ID
            k (int): Number of predictions to return
            
        Returns:
            list: Top k predicted items
        """
        if current_item not in self.transition_matrix:
            # Fall back to the most common items
            return self.top_items[:k]
            
        next_items = self.transition_matrix[current_item]
        if not next_items:
            return self.top_items[:k]
            
        # Sort by probability and return top k
        sorted_items = sorted(next_items.items(), key=lambda x: x[1], reverse=True)
        top_k_items = [item for item, _ in sorted_items[:k]]
        
        # If we don't have enough predictions, pad with popular items
        if len(top_k_items) < k:
            # Add popular items not already in predictions
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
            PreferenceEnhancedMarkovModel: Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model

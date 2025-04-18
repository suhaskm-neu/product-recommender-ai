"""
Data utilities for the hybrid social-physics model

This module provides functions for loading, processing, and filtering
data for the hybrid model.
"""

import os
import pandas as pd
import numpy as np
from collections import Counter

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def load_interactions_data(data_path):
    """
    Load user-item interactions data
    
    Args:
        data_path (str): Path to the interaction data CSV
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    print_section(f"Loading Interactions Data")
    print(f"Loading from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} interactions with {df['user_id'].nunique()} users and {df['item_id'].nunique()} items")
        
        # Check for required columns
        required_cols = ['user_id', 'item_id', 'timestamp']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in data")
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure proper data types
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Check for and handle missing values
        for col in ['view_time', 'click_rate']:
            if col in df.columns and df[col].isnull().any():
                print(f"Warning: Found {df[col].isnull().sum()} missing values in '{col}', filling with median")
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def load_social_data(social_path):
    """
    Load social connections data
    
    Args:
        social_path (str): Path to the social connections CSV
        
    Returns:
        pandas.DataFrame or None: Loaded data or None if file not found/invalid
    """
    print_section(f"Loading Social Network Data")
    
    if not os.path.exists(social_path):
        print(f"Social connections file not found: {social_path}")
        print(f"Will proceed without social connections")
        return None
    
    try:
        df = pd.read_csv(social_path)
        print(f"Loaded {len(df)} social connections with {df['user_id'].nunique()} users")
        
        # Check for required columns
        required_cols = ['user_id', 'friend_id']
        for col in required_cols:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in social data")
                raise ValueError(f"Missing required column: {col}")
        
        # Add influence weight if not present
        if 'influence_weight' not in df.columns:
            print(f"Adding default influence weight of 1.0")
            df['influence_weight'] = 1.0
            
        return df
    
    except Exception as e:
        print(f"Error loading social data: {e}")
        print(f"Will proceed without social connections")
        return None

def filter_top_n_items(df, n=5):
    """
    Filter dataset to only include interactions with the top N most frequent items
    
    Args:
        df (pandas.DataFrame): DataFrame with item_id column
        n (int): Number of top items to keep
        
    Returns:
        pandas.DataFrame: Filtered DataFrame, and list of top items
    """
    print_section(f"Filtering to Top {n} Items")
    
    # Count item frequencies
    item_counts = df['item_id'].value_counts()
    
    # Get top N items
    top_items = item_counts.nlargest(n).index.tolist()
    print(f"Selected top {n} items out of {df['item_id'].nunique()} unique items")
    print(f"Top items represent {item_counts[top_items].sum() / len(df) * 100:.2f}% of all interactions")
    
    # Filter dataset to only include top items
    filtered_df = df[df['item_id'].isin(top_items)].copy()
    
    # Reset indices
    filtered_df = filtered_df.reset_index(drop=True)
    
    print(f"Filtered dataset contains {len(filtered_df)} interactions ({len(filtered_df)/len(df)*100:.2f}% of original)")
    
    # Calculate transition coverage
    calc_transition_coverage(filtered_df, top_items)
    
    return filtered_df, top_items

def calc_transition_coverage(df, items):
    """
    Calculate and print transition coverage statistics
    
    Args:
        df (pandas.DataFrame): DataFrame with item_id column
        items (list): List of items to analyze
    """
    # Sort by user and timestamp
    df = df.sort_values(['user_id', 'timestamp'])
    
    # Count transitions
    transitions = {}
    total_transitions = 0
    repeated_transitions = 0
    
    for user_id, user_df in df.groupby('user_id'):
        for i in range(len(user_df) - 1):
            current_item = user_df.iloc[i]['item_id']
            next_item = user_df.iloc[i+1]['item_id']
            
            # Only count transitions between items in our filtered set
            if current_item in items and next_item in items:
                transition = (current_item, next_item)
                transitions[transition] = transitions.get(transition, 0) + 1
                total_transitions += 1
    
    # Count repeated transitions
    for transition, count in transitions.items():
        if count > 1:
            repeated_transitions += (count - 1)
    
    # Calculate statistics
    unique_transitions = len(transitions)
    possible_transitions = len(items) * len(items)
    coverage_percent = (unique_transitions / possible_transitions) * 100 if possible_transitions > 0 else 0
    repeat_percent = (repeated_transitions / total_transitions) * 100 if total_transitions > 0 else 0
    
    print(f"\nTransition Statistics:")
    print(f"- Total transitions: {total_transitions}")
    print(f"- Unique transitions: {unique_transitions}")
    print(f"- Possible transitions: {possible_transitions}")
    print(f"- Coverage: {coverage_percent:.2f}%")
    print(f"- Repeated transitions: {repeated_transitions}")
    print(f"- Repeat percentage: {repeat_percent:.2f}%")

def train_test_split(df, test_size=0.2, chronological=True, random_state=42):
    """
    Split data into training and testing sets
    
    Args:
        df (pandas.DataFrame): DataFrame to split
        test_size (float): Proportion of data for testing
        chronological (bool): Whether to split chronologically
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, test_df)
    """
    if chronological:
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Split chronologically
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"\nSplit data chronologically:")
    else:
        # Random split
        from sklearn.model_selection import train_test_split as sklearn_split
        train_df, test_df = sklearn_split(df, test_size=test_size, random_state=random_state)
        
        print(f"\nSplit data randomly:")
        
    print(f"- Training set: {len(train_df)} records, {train_df['user_id'].nunique()} users")
    print(f"- Testing set: {len(test_df)} records, {test_df['user_id'].nunique()} users")
    
    return train_df, test_df

def prepare_test_sequences(test_df):
    """
    Prepare test sequences for evaluation
    
    Args:
        test_df (pandas.DataFrame): Test DataFrame
        
    Returns:
        list: List of dictionaries with test sequences
    """
    test_sequences = []
    
    for user_id, user_df in test_df.groupby('user_id'):
        # Sort by timestamp
        user_df = user_df.sort_values('timestamp')
        
        # Create sequence pairs
        for i in range(len(user_df) - 1):
            current_item = user_df.iloc[i]['item_id']
            next_item = user_df.iloc[i+1]['item_id']
            
            # Get additional features for current interaction
            row = user_df.iloc[i]
            
            test_sequences.append({
                'user_id': user_id,
                'current_item': current_item,
                'next_item': next_item,
                'view_time': row.get('view_time', 0.5),
                'click_rate': row.get('click_rate', 0.5),
                'time_delta': row.get('time_delta', 1.0)
            })
    
    return test_sequences

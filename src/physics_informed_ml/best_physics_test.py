"""
Focused Test for Physics-Informed Recommendation Models

This script tests the physics-informed models on a filtered dataset
containing only the top N most frequent items, similar to the approach
that worked well with the Markov transition matrix model.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Set up path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import custom modules
from utils.data_loader import load_and_preprocess_data, prepare_sequence_data
from utils.evaluation import evaluate_model, evaluate_detailed
from economist_model.preference_calculator import augment_dataset_with_preference, analyze_preference_distribution
from economist_model.model_trainer import PreferenceEnhancedMarkovModel
from differential_equation.model_trainer import DifferentialMarkovModel

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def filter_top_n_items(df, n=50):
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
    transition_coverage(filtered_df, top_items)
    
    return filtered_df, top_items

def transition_coverage(df, items):
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
    coverage_percent = (unique_transitions / possible_transitions) * 100
    repeat_percent = (repeated_transitions / total_transitions) * 100 if total_transitions > 0 else 0
    
    print(f"\nTransition Statistics:")
    print(f"- Total transitions: {total_transitions}")
    print(f"- Unique transitions: {unique_transitions}")
    print(f"- Possible transitions: {possible_transitions}")
    print(f"- Coverage: {coverage_percent:.2f}%")
    print(f"- Repeated transitions: {repeated_transitions}")
    print(f"- Repeat percentage: {repeat_percent:.2f}%")

def train_preference_model(train_df, test_df, preference_weight=0.5):
    """
    Train the economist's preference-enhanced Markov model
    
    Args:
        train_df (DataFrame): Training data
        test_df (DataFrame): Testing data
        preference_weight (float): Weight for preference in prediction
        
    Returns:
        tuple: (model, evaluation results)
    """
    print_section("Training Preference-Enhanced Markov Model")
    
    # Add preference feature to datasets
    print("Calculating user preferences using economist's equation")
    train_df_with_pref = augment_dataset_with_preference(train_df)
    test_df_with_pref = augment_dataset_with_preference(test_df)
    
    # Analyze preference distribution
    print("\nPreference Distribution Analysis:")
    pref_stats = analyze_preference_distribution(train_df_with_pref)
    
    # Track start time
    start_time = time.time()
    
    # Create and train model
    model = PreferenceEnhancedMarkovModel(top_n_items=len(train_df['item_id'].unique()), preference_weight=preference_weight)
    
    # Fit the model
    model.fit(
        train_df_with_pref['item_id'].tolist(),
        train_df_with_pref['item_id'].tolist(),
        train_df_with_pref['user_preference'].tolist()
    )
    
    # Calculate training time
    train_time = time.time() - start_time
    print(f"Model trained in {train_time:.2f} seconds")
    
    # Prepare test data
    test_sequences = []
    for user_id, user_df in test_df_with_pref.groupby('user_id'):
        user_df = user_df.sort_values('timestamp')
        items = user_df['item_id'].tolist()
        for i in range(len(items) - 1):
            test_sequences.append((items[i], items[i+1]))
    
    # Evaluate model
    print("\nEvaluating model performance:")
    results = evaluate_model(model, test_sequences)
    
    # Save model
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "focused_preference_markov.pkl")
    model.save(model_path)
    
    return model, results

def train_differential_model(train_df, test_df, preference_evolution_weight=0.5):
    """
    Train the mathematician's differential equation Markov model
    
    Args:
        train_df (DataFrame): Training data
        test_df (DataFrame): Testing data
        preference_evolution_weight (float): Weight for preference evolution
        
    Returns:
        tuple: (model, evaluation results)
    """
    print_section("Training Differential Equation Markov Model")
    
    # Ensure timestamps are properly formatted
    if train_df['timestamp'].dtype == 'object':
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    if test_df['timestamp'].dtype == 'object':
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Convert to Unix timestamp for numerical operations
    train_df['timestamp_unix'] = train_df['timestamp'].astype(int) / 10**9
    test_df['timestamp_unix'] = test_df['timestamp'].astype(int) / 10**9
    
    # Calculate time deltas
    train_df['time_delta'] = train_df.groupby('user_id')['timestamp_unix'].diff().fillna(1.0)
    test_df['time_delta'] = test_df.groupby('user_id')['timestamp_unix'].diff().fillna(1.0)
    
    # Handle non-positive time deltas
    train_df['time_delta'] = train_df['time_delta'].apply(lambda x: max(x, 0.001))
    test_df['time_delta'] = test_df['time_delta'].apply(lambda x: max(x, 0.001))
    
    print(f"Time delta statistics: min={train_df['time_delta'].min():.2f}, " 
          f"mean={train_df['time_delta'].mean():.2f}, max={train_df['time_delta'].max():.2f}")
    
    # Track start time
    start_time = time.time()
    
    # Create and train model - use all items since we've already filtered
    model = DifferentialMarkovModel(
        top_n_items=len(train_df['item_id'].unique()), 
        preference_evolution_weight=preference_evolution_weight
    )
    
    # Fit the model
    model.fit(
        train_df['item_id'].tolist(),
        train_df['item_id'].tolist(),
        train_df['view_time'].tolist(),
        train_df['click_rate'].tolist(),
        train_df['timestamp_unix'].tolist()
    )
    
    # Calculate training time
    train_time = time.time() - start_time
    print(f"Model trained in {train_time:.2f} seconds")
    
    # Prepare test data
    test_sequences = []
    for user_id, user_df in test_df.groupby('user_id'):
        user_df = user_df.sort_values('timestamp')
        items = user_df['item_id'].tolist()
        for i in range(len(items) - 1):
            test_sequences.append((items[i], items[i+1]))
    
    # Evaluate model
    print("\nEvaluating model performance:")
    results = evaluate_model(model, test_sequences)
    
    # Save model
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "focused_differential_markov.pkl")
    model.save(model_path)
    
    return model, results

def compare_models(preference_results, differential_results):
    """
    Compare the performance of both models
    
    Args:
        preference_results (dict): Results from preference model
        differential_results (dict): Results from differential model
    """
    print_section("Model Comparison")
    
    print(f"{'Metric':<20} {'Preference Model':<20} {'Differential Model':<20}")
    print("-" * 60)
    
    print(f"{'Exact Match':<20} {preference_results['exact_match_accuracy']:.4f}{'':>13} {differential_results['exact_match_accuracy']:.4f}{'':>13}")
    
    for k in sorted(preference_results['top_k_accuracy'].keys()):
        pref_acc = preference_results['top_k_accuracy'][k]
        diff_acc = differential_results['top_k_accuracy'][k]
        print(f"{'Top-' + str(k) + ' Accuracy':<20} {pref_acc:.4f}{'':>13} {diff_acc:.4f}{'':>13}")
    
    # Determine the winner
    pref_win = 0
    diff_win = 0
    
    # Compare exact match
    if preference_results['exact_match_accuracy'] > differential_results['exact_match_accuracy']:
        pref_win += 1
    elif differential_results['exact_match_accuracy'] > preference_results['exact_match_accuracy']:
        diff_win += 1
    
    # Compare top-k
    for k in preference_results['top_k_accuracy']:
        if preference_results['top_k_accuracy'][k] > differential_results['top_k_accuracy'][k]:
            pref_win += 1
        elif differential_results['top_k_accuracy'][k] > preference_results['top_k_accuracy'][k]:
            diff_win += 1
    
    print("\nOverall Winner:", end=" ")
    if pref_win > diff_win:
        print("Preference Model")
    elif diff_win > pref_win:
        print("Differential Model")
    else:
        print("Tie")
    
    # Create visualization directory if it doesn't exist
    vis_dir = os.path.join(current_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    
    # Extract metrics
    metrics = ['Exact Match'] + [f'Top-{k}' for k in sorted(preference_results['top_k_accuracy'].keys())]
    pref_scores = [preference_results['exact_match_accuracy']] + [preference_results['top_k_accuracy'][k] for k in sorted(preference_results['top_k_accuracy'].keys())]
    diff_scores = [differential_results['exact_match_accuracy']] + [differential_results['top_k_accuracy'][k] for k in sorted(differential_results['top_k_accuracy'].keys())]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, pref_scores, width, label='Preference Model')
    plt.bar(x + width/2, diff_scores, width, label='Differential Model')
    
    plt.xlabel('Metric')
    plt.ylabel('Accuracy')
    plt.title('Physics-Informed Models Performance Comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.savefig(os.path.join(vis_dir, 'focused_model_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {os.path.join(vis_dir, 'focused_model_comparison.png')}")
    
    return {'preference_win': pref_win, 'differential_win': diff_win}

def run_focused_test(data_path=None, top_n=50, test_size=0.2, random_state=42, 
                     preference_weight=0.5, preference_evolution_weight=0.5,
                     chronological_split=True):
    """
    Run the focused test pipeline
    
    Args:
        data_path: Path to the dataset
        top_n: Number of top items to include
        test_size: Proportion of data for testing
        random_state: Random seed
        preference_weight: Weight for the preference model
        preference_evolution_weight: Weight for the differential model
        chronological_split: Whether to split data chronologically
        
    Returns:
        dict: Results from both models
    """
    print_section(f"Running Focused Test with Top {top_n} Items")
    start_time = time.time()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df, _ = load_and_preprocess_data(data_path, 0, random_state)  # Load all data first
    
    # Filter to top N items
    df, top_items = filter_top_n_items(df, n=top_n)
    
    # Now split into train/test
    if chronological_split:
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Split chronologically
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"\nSplit data chronologically:")
    else:
        # Random split
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        
        print(f"\nSplit data randomly:")
        
    print(f"- Training set: {len(train_df)} records")
    print(f"- Testing set: {len(test_df)} records")
    
    # Train models
    preference_model, preference_results = train_preference_model(
        train_df, test_df, preference_weight
    )
    
    differential_model, differential_results = train_differential_model(
        train_df, test_df, preference_evolution_weight
    )
    
    # Compare models
    comparison = compare_models(preference_results, differential_results)
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    return {
        'preference_results': preference_results,
        'differential_results': differential_results,
        'comparison': comparison,
        'execution_time': total_time
    }

def main():
    """
    Main function to run the focused test
    """
    parser = argparse.ArgumentParser(description="Run focused test for physics-informed models")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="../../data/samples/user_0_processed.csv",
                        help="Path to the dataset (defaults to user_0_processed.csv)")
    parser.add_argument("--top_n", type=int, default=50,
                        help="Number of top items to include")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Proportion of data for testing")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed")
    
    # Model parameters
    parser.add_argument("--preference_weight", type=float, default=0.5,
                        help="Weight for preference in the economist's model")
    parser.add_argument("--preference_evolution_weight", type=float, default=0.5,
                        help="Weight for preference evolution in the differential model")
    parser.add_argument("--chronological_split", action="store_true",
                        help="Split data chronologically (default is random)")
    
    args = parser.parse_args()
    
    run_focused_test(
        data_path=args.data_path,
        top_n=args.top_n,
        test_size=args.test_size,
        random_state=args.random_state,
        preference_weight=args.preference_weight,
        preference_evolution_weight=args.preference_evolution_weight,
        chronological_split=args.chronological_split
    )

if __name__ == "__main__":
    main()

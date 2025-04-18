import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import joblib

"""
MULTI-USER FOCUSED MARKOV MODEL

This implementation tests whether the patterns observed in single-user data
(high accuracy with top-N recommendations when focusing on popular items)
hold true across multiple users.

Key insights from single-user model:
1. Full Model on Full Data: 40.29% accuracy
2. Full Model on Filtered Data: 23.42% accuracy
3. Top-N Recommendations:
   - Top-1: 61.82% accuracy
   - Top-3: 100% accuracy
   - Top-5: 100% accuracy

The approach works by:
- Focusing on the most frequent items rather than the entire catalog
- Using balanced evaluation on a filtered subset
- Implementing top-N recommendations for realistic scenarios
- Leveraging the full context while focusing evaluation on popular items
"""


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def load_and_analyze_data(data_path):
    """
    Load, preprocess, and analyze multi-user data.
    
    Args:
        data_path: Path to the CSV file with multi-user data
        
    Returns:
        Preprocessed DataFrame
    """
    print_section(f"Loading and analyzing multi-user data")
    start_time = time.time()
    
    # Load the data
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Ensure item_id and next_item_id are of the same type (int)
    df['item_id'] = df['item_id'].astype('int32')
    # Keep next_item_id as Int32 (nullable integer) to handle NaN values
    if 'next_item_id' in df.columns:
        df['next_item_id'] = df['next_item_id'].astype('Int32')
    
    # Print basic statistics
    print(f"Data shape: {df.shape}")
    print(f"Number of unique users: {df['user_id'].nunique()}")
    print(f"Number of unique items: {df['item_id'].nunique()}")
    if 'next_item_id' in df.columns:
        print(f"Number of unique next items: {df['next_item_id'].nunique()}")
    
    # Analyze item frequency distribution
    item_counts = df['item_id'].value_counts()
    print(f"\nItem frequency distribution:")
    print(f"Most common item appears {item_counts.max()} times")
    print(f"Least common item appears {item_counts.min()} times")
    print(f"Mean occurrences per item: {item_counts.mean():.2f}")
    print(f"Median occurrences per item: {item_counts.median():.2f}")
    
    # Count transitions if next_item_id is available
    if 'next_item_id' in df.columns:
        transition_counts = defaultdict(lambda: defaultdict(int))
        for i in range(len(df) - 1):
            # Only count transitions within the same user
            if i < len(df) - 1 and df.iloc[i]['user_id'] == df.iloc[i+1]['user_id']:
                current_item = df.iloc[i]['item_id']
                next_item = df.iloc[i]['next_item_id']
                if not pd.isna(next_item):
                    transition_counts[current_item][next_item] += 1
        
        # Count items with repeated transitions
        items_with_repeated_next = 0
        total_transitions = 0
        repeated_transitions = 0
        
        for current_item, next_items in transition_counts.items():
            if next_items:
                most_common_next = max(next_items.items(), key=lambda x: x[1])
                if most_common_next[1] > 1:
                    items_with_repeated_next += 1
                
                # Count total and repeated transitions
                item_transitions = sum(next_items.values())
                total_transitions += item_transitions
                repeated_transitions += item_transitions - len(next_items)
        
        print(f"Items with repeated transitions: {items_with_repeated_next} out of {len(transition_counts)} ({items_with_repeated_next/len(transition_counts)*100:.2f}%)")
        print(f"Total transitions: {total_transitions}")
        print(f"Repeated transitions: {repeated_transitions} ({repeated_transitions/total_transitions*100:.2f}%)")
    
    print(f"Data loaded and analyzed in {time.time() - start_time:.2f} seconds")
    
    return df


def build_transition_matrix(df):
    """
    Build a transition matrix (first-order Markov chain) for the entire dataset.
    
    Args:
        df: DataFrame with user interactions
        
    Returns:
        Dictionary mapping items to their most likely next items and transition counts
    """
    print_section("Building full transition matrix")
    start_time = time.time()
    
    # Count transitions from each item to next item
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    # Group by user_id to ensure we don't count transitions across different users
    user_groups = df.groupby('user_id')
    
    for user_id, user_df in user_groups:
        # Sort by timestamp if available
        if 'timestamp' in user_df.columns:
            user_df = user_df.sort_values('timestamp').reset_index(drop=True)
        
        for i in range(len(user_df) - 1):
            current_item = user_df.iloc[i]['item_id']
            next_item = user_df.iloc[i]['next_item_id']
            if not pd.isna(next_item):
                transition_counts[current_item][next_item] += 1
    
    # Create a model that predicts based on most frequent next item
    most_likely_next = {}
    for current_item, next_items in transition_counts.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created transition matrix for {len(most_likely_next)} items")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return most_likely_next, transition_counts


def focus_on_popular_items(df, top_n=100):
    """
    Filter the dataset to focus on the most frequent items.
    
    Args:
        df: DataFrame with user interactions
        top_n: Number of most frequent items to keep
        
    Returns:
        Filtered DataFrame
    """
    print_section(f"Focusing on top {top_n} most frequent items")
    start_time = time.time()
    
    # Get the most frequent items
    top_items = df['item_id'].value_counts().nlargest(top_n).index.tolist()
    
    # Filter data to only include rows where item_id is in the top items
    filtered_df = df[df['item_id'].isin(top_items)].copy()
    
    # Print statistics
    print(f"Original data shape: {df.shape}")
    print(f"Filtered data shape: {filtered_df.shape}")
    print(f"Retention rate: {len(filtered_df)/len(df)*100:.2f}%")
    
    # Check if we have enough data
    if len(filtered_df) < 100:
        print("Warning: Very little data left after filtering. Consider increasing top_n.")
    
    print(f"Data filtered in {time.time() - start_time:.2f} seconds")
    
    return filtered_df


def prepare_features(df):
    """
    Prepare features for the model, including previous item features.
    
    Args:
        df: DataFrame with user interactions
        
    Returns:
        DataFrame with added features
    """
    print_section("Preparing features")
    start_time = time.time()
    
    # Group by user_id to ensure we don't create features across different users
    user_groups = df.groupby('user_id')
    feature_dfs = []
    
    for user_id, user_df in user_groups:
        # Sort by timestamp if available
        if 'timestamp' in user_df.columns:
            user_df = user_df.sort_values('timestamp').reset_index(drop=True)
        
        # Add previous item features
        user_df['prev_item_id'] = user_df['item_id'].shift(1)
        user_df['prev_view_time'] = user_df['view_time'].shift(1)
        user_df['prev_click_rate'] = user_df['click_rate'].shift(1)
        
        # Drop first row which has NaN for previous features
        user_df = user_df.dropna(subset=['prev_item_id', 'prev_view_time', 'prev_click_rate'])
        
        feature_dfs.append(user_df)
    
    # Combine all user dataframes
    featured_df = pd.concat(feature_dfs, ignore_index=True)
    
    print(f"Added features: prev_item_id, prev_view_time, prev_click_rate")
    print(f"Data shape after feature preparation: {featured_df.shape}")
    print(f"Features prepared in {time.time() - start_time:.2f} seconds")
    
    return featured_df


def evaluate_transition_matrix(model, test_df, verbose=True):
    """
    Evaluate the transition matrix model.
    
    Args:
        model: Dictionary mapping items to their most likely next items
        test_df: Test DataFrame
        verbose: Whether to print detailed results
        
    Returns:
        Accuracy of the model
    """
    print_section("Evaluating transition matrix model")
    start_time = time.time()
    
    correct_predictions = 0
    total_predictions = 0
    
    # Group by user_id to ensure we don't evaluate across different users
    user_groups = test_df.groupby('user_id')
    
    for user_id, user_df in user_groups:
        for i, row in user_df.iterrows():
            current_item = row['item_id']
            true_next_item = row['next_item_id']
            
            if pd.isna(true_next_item):
                continue
                
            # Convert to same type for comparison
            true_next_item = int(true_next_item)
            
            if current_item in model:
                total_predictions += 1
                predicted_item = model[current_item]
                
                # Convert to same type for comparison
                if isinstance(predicted_item, float):
                    predicted_item = int(predicted_item)
                    
                if predicted_item == true_next_item:
                    correct_predictions += 1
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        if verbose:
            print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
    else:
        accuracy = 0
        if verbose:
            print("No predictions made with transition matrix approach")
    
    if verbose:
        print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    
    return accuracy


def get_top_n_recommendations(transition_counts, item_id, n=10):
    """
    Get the top N most likely next items for a given item.
    
    Args:
        transition_counts: Dictionary of transition counts
        item_id: Current item ID
        n: Number of recommendations to return
        
    Returns:
        List of top N recommended items
    """
    if item_id not in transition_counts:
        return []
    
    # Sort next items by count (descending)
    next_items = sorted(transition_counts[item_id].items(), key=lambda x: x[1], reverse=True)
    
    # Return top N items
    return [item for item, count in next_items[:n]]


def evaluate_top_n_accuracy(transition_counts, test_df, n_values=[1, 3, 5, 10]):
    """
    Evaluate the model using top-N recommendation accuracy.
    
    Args:
        transition_counts: Dictionary of transition counts
        test_df: Test DataFrame
        n_values: List of N values to evaluate
        
    Returns:
        Dictionary of accuracy scores for each N
    """
    print_section("Evaluating top-N recommendation accuracy")
    start_time = time.time()
    
    results = {}
    
    for n in n_values:
        correct = 0
        total = 0
        
        # Group by user_id to ensure we don't evaluate across different users
        user_groups = test_df.groupby('user_id')
        
        for user_id, user_df in user_groups:
            for i, row in user_df.iterrows():
                current_item = row['item_id']
                true_next_item = row['next_item_id']
                
                if pd.isna(true_next_item):
                    continue
                    
                # Convert to same type for comparison
                true_next_item = int(true_next_item)
                
                # Get top N recommendations
                recommendations = get_top_n_recommendations(transition_counts, current_item, n)
                
                if recommendations:  # If we have recommendations
                    total += 1
                    # Convert to same type for comparison
                    recommendations = [int(item) if isinstance(item, float) else item for item in recommendations]
                    if true_next_item in recommendations:
                        correct += 1
        
        accuracy = correct / total if total > 0 else 0
        results[n] = accuracy
        print(f"Top-{n} accuracy: {accuracy:.4f} ({correct}/{total})")
    
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    
    return results


def visualize_results(results, title="Model Performance", filename="multi_user_model_performance.png"):
    """
    Visualize the results of different models.
    
    Args:
        results: Dictionary of model results
        title: Title for the plot
        filename: Name of the output file
    """
    plt.figure(figsize=(10, 6))
    
    for model_name, accuracies in results.items():
        if isinstance(accuracies, dict):  # For top-N results
            x = list(accuracies.keys())
            y = list(accuracies.values())
            plt.plot(x, y, marker='o', label=model_name)
        else:  # For single accuracy value
            plt.axhline(y=accuracies, linestyle='--', label=f"{model_name} ({accuracies:.4f})")
    
    plt.title(title)
    plt.xlabel("Top-N Recommendations" if any(isinstance(v, dict) for v in results.values()) else "Models")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plots_dir = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(f"{plots_dir}/{filename}")
    print(f"Plot saved to {plots_dir}/{filename}")
    
    plt.close()


def run_multi_user_focused_model(data_path, top_n_items=100, test_size=0.2, random_state=42):
    """
    Run the multi-user focused Markov model pipeline.
    
    Args:
        data_path: Path to the data file
        top_n_items: Number of most frequent items to focus on
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of model results
    """
    print_section(f"Running Multi-User Focused Markov Model Pipeline")
    overall_start_time = time.time()
    
    # Step 1: Load and analyze data
    df = load_and_analyze_data(data_path)
    
    # Step 2: Build full transition matrix (for comparison)
    full_model, full_transition_counts = build_transition_matrix(df)
    
    # Step 3: Focus on popular items
    filtered_df = focus_on_popular_items(df, top_n=top_n_items)
    
    # Step 4: Prepare features
    featured_df = prepare_features(filtered_df)
    
    # Step 5: Split data while preserving user groups
    # Get unique user IDs
    unique_users = featured_df['user_id'].unique()
    
    # Split users into train and test sets
    train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=random_state)
    
    # Split data based on user assignment
    train_df = featured_df[featured_df['user_id'].isin(train_users)].copy()
    test_df = featured_df[featured_df['user_id'].isin(test_users)].copy()
    
    print(f"Training users: {len(train_users)}, Test users: {len(test_users)}")
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Step 6: Build focused transition matrix
    focused_model, focused_transition_counts = build_transition_matrix(train_df)
    print(f"Created focused transition matrix for {len(focused_model)} items")
    
    # Step 7: Evaluate models
    # Evaluate full model on full test data
    full_test_df = df[df['user_id'].isin(test_users)].copy()
    full_accuracy = evaluate_transition_matrix(full_model, full_test_df, verbose=True)
    
    # Evaluate focused model on filtered test data
    focused_accuracy = evaluate_transition_matrix(focused_model, test_df, verbose=True)
    
    # Evaluate top-N accuracy for full model
    full_top_n_results = evaluate_top_n_accuracy(full_transition_counts, full_test_df)
    
    # Evaluate top-N accuracy for focused model
    focused_top_n_results = evaluate_top_n_accuracy(focused_transition_counts, test_df)
    
    # Step 8: Visualize results
    results = {
        "Full Model": full_accuracy,
        "Focused Model": focused_accuracy,
        "Full Model Top-N": full_top_n_results,
        "Focused Model Top-N": focused_top_n_results
    }
    
    visualize_results(results, title=f"Multi-User Markov Model Performance (Top {top_n_items} Items)")
    
    # Compare with single-user results
    single_user_results = {
        "Single-User Full Model": 0.4029,
        "Single-User Focused Model": 0.2342,
        "Single-User Top-N": {1: 0.6182, 3: 1.0, 5: 1.0, 10: 1.0},
        "Multi-User Full Model": full_accuracy,
        "Multi-User Focused Model": focused_accuracy,
        "Multi-User Top-N": focused_top_n_results
    }
    
    visualize_results(single_user_results, 
                     title="Single-User vs Multi-User Performance",
                     filename="single_vs_multi_user_comparison.png")
    
    # Step 9: Save models
    models_dir = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/models/multi_user"
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(full_model, f"{models_dir}/full_transition_matrix.joblib")
    joblib.dump(focused_model, f"{models_dir}/focused_transition_matrix.joblib")
    
    print(f"\nModels saved to:")
    print(f"- Full Transition Matrix: {models_dir}/full_transition_matrix.joblib")
    print(f"- Focused Transition Matrix: {models_dir}/focused_transition_matrix.joblib")
    
    total_time = time.time() - overall_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    return results


def main():
    """
    Main function to run the multi-user focused Markov model.
    """
    # Configuration
    data_path = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/data/samples/multi_user_sample.csv"
    top_n_items = 100  # Focus on top 100 items
    test_size = 0.2
    random_state = 42
    
    # Run the model
    results = run_multi_user_focused_model(
        data_path=data_path,
        top_n_items=top_n_items,
        test_size=test_size,
        random_state=random_state
    )
    
    # Print summary
    print_section("Summary of Results")
    for model_name, accuracy in results.items():
        if isinstance(accuracy, dict):  # For top-N results
            print(f"{model_name}:")
            for n, acc in accuracy.items():
                print(f"  Top-{n}: {acc:.4f}")
        else:  # For single accuracy value
            print(f"{model_name}: {accuracy:.4f}")


if __name__ == "__main__":
    main()
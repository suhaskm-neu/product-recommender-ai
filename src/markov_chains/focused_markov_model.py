import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import joblib

"""
FOCUSED MARKOV MODEL

This implementation incorporates the key insights from the successful minimal_model.py approach:

1. Focus on Popular Items: Focus on the most frequent items rather than the entire catalog
2. Strategic Evaluation: Evaluate on a balanced subset rather than the full sparse dataset
3. Random Sampling: Use random sampling to ensure training and test sets have similar distributions
4. Transition Matrix: Leverage the simple but effective Markov chain approach
5. Detailed Analysis: Provide comprehensive metrics and visualizations

The key insight is that with extremely sparse data (50,000+ items with minimal repetition),
focusing on a subset where patterns actually exist is more effective than trying to model everything.
"""

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}\n{title}\n{'='*70}")


def load_and_analyze_data(data_path, user_id=0):
    """
    Load, preprocess, and analyze data for a specific user.
    
    Args:
        data_path: Path to the CSV file
        user_id: User ID to filter for (default: 0)
        
    Returns:
        Preprocessed DataFrame for the specified user
    """
    print_section(f"Loading and analyzing data for user {user_id}")
    start_time = time.time()
    
    # Load the data
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Filter for the specified user if needed
    if 'user_id' in df.columns:
        df = df[df['user_id'] == user_id].copy()
    
    # Ensure item_id and next_item_id are of the same type (int)
    df['item_id'] = df['item_id'].astype('int32')
    # Keep next_item_id as Int32 (nullable integer) to handle NaN values
    df['next_item_id'] = df['next_item_id'].astype('Int32')
    
    # Print basic statistics
    print(f"Data shape: {df.shape}")
    print(f"Number of unique items: {df['item_id'].nunique()}")
    print(f"Number of unique next items: {df['next_item_id'].nunique()}")
    
    # Analyze item frequency distribution
    item_counts = df['item_id'].value_counts()
    print(f"\nItem frequency distribution:")
    print(f"Most common item appears {item_counts.max()} times")
    print(f"Least common item appears {item_counts.min()} times")
    print(f"Mean occurrences per item: {item_counts.mean():.2f}")
    print(f"Median occurrences per item: {item_counts.median():.2f}")
    
    # Count transitions
    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(df) - 1):
        current_item = df.iloc[i]['item_id']
        next_item = df.iloc[i]['next_item_id']
        if not pd.isna(next_item):
            transition_counts[current_item][next_item] += 1
    
    # Count items with repeated transitions
    items_with_repeated_next = 0
    for current_item, next_items in transition_counts.items():
        if next_items:
            most_common_next = max(next_items.items(), key=lambda x: x[1])
            if most_common_next[1] > 1:
                items_with_repeated_next += 1
    
    print(f"Items with repeated transitions: {items_with_repeated_next} out of {len(transition_counts)} ({items_with_repeated_next/len(transition_counts)*100:.2f}%)")
    
    print(f"Data loaded and analyzed in {time.time() - start_time:.2f} seconds")
    
    return df


def build_transition_matrix(df):
    """
    Build a transition matrix (first-order Markov chain) for the entire dataset.
    
    Args:
        df: DataFrame with user interactions
        
    Returns:
        Dictionary mapping items to their most likely next items
    """
    print_section("Building full transition matrix")
    start_time = time.time()
    
    # Count transitions from each item to next item
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(df) - 1):
        current_item = df.iloc[i]['item_id']
        next_item = df.iloc[i]['next_item_id']
        if not pd.isna(next_item):
            transition_counts[current_item][next_item] += 1
    
    # Create a model that predicts based on most frequent next item
    most_likely_next = {}
    for current_item, next_items in transition_counts.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created transition matrix for {len(most_likely_next)} items")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return most_likely_next


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
    
    # Add previous item features
    df['prev_item_id'] = df['item_id'].shift(1)
    df['prev_view_time'] = df['view_time'].shift(1)
    df['prev_click_rate'] = df['click_rate'].shift(1)
    
    # Drop first row which has NaN for previous features
    df = df.dropna()
    
    print(f"Added features: prev_item_id, prev_view_time, prev_click_rate")
    print(f"Data shape after feature preparation: {df.shape}")
    print(f"Features prepared in {time.time() - start_time:.2f} seconds")
    
    return df


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
    
    for i, row in test_df.iterrows():
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


def evaluate_top_n_accuracy(model, test_df, n_values=[1, 3, 5, 10]):
    """
    Evaluate the model using top-N recommendation accuracy.
    
    Args:
        model: Dictionary mapping items to their most likely next items
        test_df: Test DataFrame
        n_values: List of N values to evaluate
        
    Returns:
        Dictionary of accuracy scores for each N
    """
    print_section("Evaluating top-N recommendation accuracy")
    start_time = time.time()
    
    # Build transition counts for top-N recommendations
    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(test_df) - 1):
        current_item = test_df.iloc[i]['item_id']
        next_item = test_df.iloc[i]['next_item_id']
        if not pd.isna(next_item):
            transition_counts[current_item][next_item] += 1
    
    results = {}
    
    for n in n_values:
        correct = 0
        total = 0
        
        for i, row in test_df.iterrows():
            current_item = row['item_id']
            true_next_item = row['next_item_id']
            
            if pd.isna(true_next_item):
                continue
                
            # Convert to same type for comparison
            true_next_item = int(true_next_item)
            
            if current_item in transition_counts:
                # Get top N recommendations
                next_items = sorted(transition_counts[current_item].items(), key=lambda x: x[1], reverse=True)
                recommendations = [item for item, count in next_items[:n]]
                
                if recommendations:
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


def visualize_results(results, title="Model Performance"):
    """
    Visualize the results of different models.
    
    Args:
        results: Dictionary of model results
        title: Title for the plot
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
    os.makedirs("../../plots", exist_ok=True)
    plt.savefig("../../plots/model_performance.png")
    print(f"Plot saved to ../../plots/model_performance.png")
    
    plt.close()


def run_focused_markov_model(data_path, user_id=0, top_n_items=100, test_size=0.2, random_state=42):
    """
    Run the focused Markov model pipeline.
    
    Args:
        data_path: Path to the data file
        user_id: User ID to filter for
        top_n_items: Number of most frequent items to focus on
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of model results
    """
    print_section(f"Running Focused Markov Model Pipeline")
    overall_start_time = time.time()
    
    # Step 1: Load and analyze data
    df = load_and_analyze_data(data_path, user_id)
    
    # Step 2: Build full transition matrix (for comparison)
    full_model = build_transition_matrix(df)
    
    # Step 3: Focus on popular items
    filtered_df = focus_on_popular_items(df, top_n=top_n_items)
    
    # Step 4: Prepare features
    featured_df = prepare_features(filtered_df)
    
    # Step 5: Split data
    X = featured_df[['item_id', 'prev_item_id', 'view_time', 'click_rate', 'prev_view_time', 'prev_click_rate']]
    y = featured_df['next_item_id']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Combine X_test and y_test for evaluation
    test_df = X_test.copy()
    test_df['next_item_id'] = y_test.values
    
    # Step 6: Build focused transition matrix
    focused_transition_counts = defaultdict(lambda: defaultdict(int))
    train_df = X_train.copy()
    train_df['next_item_id'] = y_train.values
    
    for i in range(len(train_df)):
        current_item = train_df.iloc[i]['item_id']
        next_item = train_df.iloc[i]['next_item_id']
        if not pd.isna(next_item):
            focused_transition_counts[current_item][next_item] += 1
    
    focused_model = {}
    for current_item, next_items in focused_transition_counts.items():
        if next_items:
            focused_model[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created focused transition matrix for {len(focused_model)} items")
    
    # Step 7: Evaluate models
    # Evaluate full model on full test data
    full_accuracy_on_full = evaluate_transition_matrix(full_model, df.iloc[int(len(df)*(1-test_size)):], verbose=False)
    print(f"Full model on full test data: {full_accuracy_on_full:.4f}")
    
    # Evaluate full model on filtered test data
    full_accuracy_on_filtered = evaluate_transition_matrix(full_model, test_df)
    
    # Evaluate focused model on filtered test data
    focused_accuracy = evaluate_transition_matrix(focused_model, test_df)
    
    # Evaluate top-N accuracy
    top_n_results = evaluate_top_n_accuracy(focused_model, test_df)
    
    # Step 8: Visualize results
    results = {
        "Full Model (Full Data)": full_accuracy_on_full,
        "Full Model (Filtered Data)": full_accuracy_on_filtered,
        "Focused Model": focused_accuracy,
        "Top-N Recommendations": top_n_results
    }
    
    visualize_results(results, title=f"Markov Model Performance (Top {top_n_items} Items)")
    
    # Step 9: Save models
    os.makedirs("../../models/single_user", exist_ok=True)
    joblib.dump(full_model, f"../../models/single_user/full_transition_matrix_user_{user_id}.joblib")
    joblib.dump(focused_model, f"../../models/single_user/focused_transition_matrix_user_{user_id}.joblib")
    
    print(f"\nModels saved to:")
    print(f"- Full Transition Matrix: ../../models/single_user/full_transition_matrix_user_{user_id}.joblib")
    print(f"- Focused Transition Matrix: ../../models/single_user/focused_transition_matrix_user_{user_id}.joblib")
    
    total_time = time.time() - overall_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    return results


def main():
    """
    Main function to run the focused Markov model.
    """
    # Configuration
    data_path = "../../data/samples/user_0_processed.csv"
    user_id = 0
    top_n_items = 100  # Focus on top 100 items
    test_size = 0.2
    random_state = 42
    
    # Run the model
    results = run_focused_markov_model(
        data_path=data_path,
        user_id=user_id,
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

import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import joblib
import random

# ============================================================================
# DATA LOADING AND ANALYSIS
# ============================================================================

def load_and_analyze_data(data_path, user_id=0):
    """
    Load, preprocess, and analyze data for a specific user.
    
    Args:
        data_path: Path to the CSV file
        user_id: User ID to filter for (default: 0)
        
    Returns:
        Preprocessed DataFrame for the specified user
    """
    print(f"\n{'='*50}\nLoading and analyzing data for user {user_id}\n{'='*50}")
    start_time = time.time()
    
    # Load the data
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Filter for the specified user if needed
    if 'user_id' in df.columns:
        df = df[df['user_id'] == user_id].copy()
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Ensure item_id and next_item_id are of the same type (int)
    df['item_id'] = df['item_id'].astype('int32')
    # Keep next_item_id as Int32 (nullable integer) to handle NaN values
    df['next_item_id'] = df['next_item_id'].astype('Int32')
    
    # Print basic statistics
    print(f"Data shape: {df.shape}")
    print(f"Number of unique items: {df['item_id'].nunique()}")
    print(f"Number of unique next items: {df['next_item_id'].nunique()}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Analyze item frequency distribution
    item_counts = df['item_id'].value_counts()
    print(f"\nItem frequency distribution:")
    print(f"Most common item appears {item_counts.max()} times")
    print(f"Least common item appears {item_counts.min()} times")
    print(f"Mean occurrences per item: {item_counts.mean():.2f}")
    print(f"Median occurrences per item: {item_counts.median():.2f}")
    
    # Analyze transition patterns
    print(f"\nAnalyzing transition patterns...")
    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(df) - 1):
        current_item = df.iloc[i]['item_id']
        next_item = df.iloc[i]['next_item_id']
        if not pd.isna(next_item):
            transition_counts[current_item][next_item] += 1
    
    # Count how many items have repeated transitions
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
    
    # Check for sequence patterns (pairs of items that appear together)
    sequence_patterns = defaultdict(int)
    for i in range(len(df) - 2):
        seq = (df.iloc[i]['item_id'], df.iloc[i+1]['item_id'])
        sequence_patterns[seq] += 1
    
    repeated_sequences = {seq: count for seq, count in sequence_patterns.items() if count > 1}
    print(f"Sequences that appear more than once: {len(repeated_sequences)} out of {len(sequence_patterns)} ({len(repeated_sequences)/len(sequence_patterns)*100:.2f}%)")
    
    print(f"Data loaded and analyzed in {time.time() - start_time:.2f} seconds")
    
    return df

# ============================================================================
# DATA SPLITTING STRATEGIES
# ============================================================================

def split_train_test_chronological(df, test_size=0.2):
    """
    Split the data into training and test sets, maintaining chronological order.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data to use for testing
        
    Returns:
        train_df, test_df: Training and test DataFrames
    """
    print(f"\n{'='*50}\nSplitting data chronologically\n{'='*50}")
    
    # Calculate split index to maintain chronological order
    split_idx = int(len(df) * (1 - test_size))
    
    # Split the data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
    return train_df, test_df


def split_train_test_random(df, test_size=0.2, random_state=42):
    """
    Split the data randomly into training and test sets.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, test_df: Training and test DataFrames
    """
    print(f"\n{'='*50}\nSplitting data randomly\n{'='*50}")
    
    # Generate indices for train/test split
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    # Split the data
    train_df = df.iloc[train_indices].copy().reset_index(drop=True)
    test_df = df.iloc[test_indices].copy().reset_index(drop=True)
    
    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
    return train_df, test_df


def split_train_test_by_session(df, test_size=0.2, session_gap_minutes=30, random_state=42):
    """
    Split the data by session, keeping sessions intact.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data to use for testing
        session_gap_minutes: Time gap in minutes to define a new session
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, test_df: Training and test DataFrames
    """
    print(f"\n{'='*50}\nSplitting data by session\n{'='*50}")
    
    # Define sessions based on time gaps
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['time_diff'] = df['timestamp'].diff()
    session_gap = session_gap_minutes * 60  # Convert to seconds
    df['new_session'] = (df['time_diff'] > session_gap) | (df['time_diff'].isna())
    df['session_id'] = df['new_session'].cumsum()
    
    # Get unique sessions
    sessions = df['session_id'].unique()
    
    # Randomly select sessions for test set
    random.seed(random_state)
    test_sessions = random.sample(list(sessions), int(len(sessions) * test_size))
    
    # Split the data
    train_df = df[~df['session_id'].isin(test_sessions)].copy().reset_index(drop=True)
    test_df = df[df['session_id'].isin(test_sessions)].copy().reset_index(drop=True)
    
    # Drop temporary columns
    train_df = train_df.drop(['time_diff', 'new_session', 'session_id'], axis=1)
    test_df = test_df.drop(['time_diff', 'new_session', 'session_id'], axis=1)
    
    print(f"Total sessions: {len(sessions)}")
    print(f"Training set: {len(train_df)} records from {len(sessions) - len(test_sessions)} sessions")
    print(f"Test set: {len(test_df)} records from {len(test_sessions)} sessions")
    
    return train_df, test_df

# ============================================================================
# MODEL 1: BASIC TRANSITION MATRIX (FIRST-ORDER MARKOV CHAIN)
# ============================================================================

def build_basic_transition_matrix(train_df):
    """
    Build a basic transition matrix (first-order Markov chain).
    
    Args:
        train_df: Training DataFrame
        
    Returns:
        Dictionary mapping items to their most likely next items
    """
    print(f"\n{'='*50}\nBuilding basic transition matrix\n{'='*50}")
    start_time = time.time()
    
    # Count transitions from each item to next item
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(train_df) - 1):
        current_item = train_df.iloc[i]['item_id']
        next_item = train_df.iloc[i]['next_item_id']
        if not pd.isna(next_item):
            transition_counts[current_item][next_item] += 1
    
    # Create a model that predicts based on most frequent next item
    most_likely_next = {}
    for current_item, next_items in transition_counts.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created transition matrix for {len(most_likely_next)} items")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return transition_counts, most_likely_next


# ============================================================================
# MODEL 2: HIGHER-ORDER MARKOV CHAIN (SEQUENCE MODELING)
# ============================================================================

def build_higher_order_markov(train_df, sequence_length=2):
    """
    Build a higher-order Markov chain that considers sequences of items.
    
    Args:
        train_df: Training DataFrame
        sequence_length: Number of previous items to consider
        
    Returns:
        Dictionary mapping sequences to their most likely next items
    """
    print(f"\n{'='*50}\nBuilding higher-order Markov chain (sequence length: {sequence_length})\n{'='*50}")
    start_time = time.time()
    
    # Count transitions from each sequence to next item
    sequence_transitions = defaultdict(lambda: defaultdict(int))
    
    # We need at least sequence_length+1 items to create a valid sequence
    if len(train_df) <= sequence_length:
        print("Not enough data for the specified sequence length")
        return {}, {}
    
    # Create sequences and count transitions
    for i in range(len(train_df) - sequence_length):
        # Create a tuple of the current sequence
        current_sequence = tuple(train_df.iloc[i:i+sequence_length]['item_id'].values)
        # Get the next item from the next_item_id column of the last item in the sequence
        next_item = train_df.iloc[i+sequence_length-1]['next_item_id']
        
        # Skip if next_item is NaN (last item in a sequence)
        if pd.isna(next_item):
            continue
            
        sequence_transitions[current_sequence][next_item] += 1
    
    # Create a model that predicts based on most frequent next item for each sequence
    most_likely_next = {}
    for sequence, next_items in sequence_transitions.items():
        if next_items:  # If there are any transitions from this sequence
            most_likely_next[sequence] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created higher-order model with {len(most_likely_next)} sequences")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return sequence_transitions, most_likely_next


# ============================================================================
# MODEL 3: TIME-DECAY WEIGHTED TRANSITION MATRIX
# ============================================================================

def build_time_decay_matrix(train_df, decay_factor=0.9):
    """
    Build a transition matrix with time decay weighting (more recent = more weight).
    
    Args:
        train_df: Training DataFrame
        decay_factor: Weight multiplier for each step back in time (0-1)
        
    Returns:
        Dictionary mapping items to their most likely next items
    """
    print(f"\n{'='*50}\nBuilding time-decay weighted transition matrix (decay factor: {decay_factor})\n{'='*50}")
    start_time = time.time()
    
    # Count transitions with time decay
    transition_weights = defaultdict(lambda: defaultdict(float))
    
    # Calculate weights based on position in the sequence
    # More recent interactions get higher weights
    for i in range(len(train_df) - 1):
        position_weight = decay_factor ** (len(train_df) - 2 - i)  # Newer items get weight closer to 1
        current_item = train_df.iloc[i]['item_id']
        next_item = train_df.iloc[i]['next_item_id']
        if not pd.isna(next_item):
            transition_weights[current_item][next_item] += position_weight
    
    # Create a model that predicts based on highest weighted next item
    most_likely_next = {}
    for current_item, next_items in transition_weights.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created time-decay model for {len(most_likely_next)} items")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return transition_weights, most_likely_next


# ============================================================================
# MODEL 4: HYBRID APPROACH (TRANSITION MATRIX + ITEM POPULARITY)
# ============================================================================

def build_hybrid_model(train_df, popularity_weight=0.3):
    """
    Build a hybrid model combining transition matrix with item popularity.
    
    Args:
        train_df: Training DataFrame
        popularity_weight: Weight given to item popularity (0-1)
        
    Returns:
        Dictionary mapping items to their most likely next items
    """
    print(f"\n{'='*50}\nBuilding hybrid model (popularity weight: {popularity_weight})\n{'='*50}")
    start_time = time.time()
    
    # Calculate item popularity
    item_counts = Counter(train_df['item_id'])
    total_items = sum(item_counts.values())
    item_popularity = {item: count/total_items for item, count in item_counts.items()}
    
    # Count transitions
    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(train_df) - 1):
        current_item = train_df.iloc[i]['item_id']
        next_item = train_df.iloc[i]['next_item_id']
        if not pd.isna(next_item):
            transition_counts[current_item][next_item] += 1
    
    # Normalize transition counts to probabilities
    transition_probs = defaultdict(dict)
    for current_item, next_items in transition_counts.items():
        total = sum(next_items.values())
        for next_item, count in next_items.items():
            # Combine transition probability with item popularity
            transition_prob = count / total
            pop_prob = item_popularity.get(next_item, 0)
            
            # Weighted combination
            combined_prob = (1 - popularity_weight) * transition_prob + popularity_weight * pop_prob
            transition_probs[current_item][next_item] = combined_prob
    
    # Create a model that predicts based on highest combined probability
    most_likely_next = {}
    for current_item, next_items in transition_probs.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created hybrid model for {len(most_likely_next)} items")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return transition_probs, most_likely_next


# ============================================================================
# MODEL 5: FALLBACK MODEL
# ============================================================================

def build_fallback_model(train_df):
    """
    Build a model with fallback strategy: first-order Markov -> item popularity.
    
    Args:
        train_df: Training DataFrame
        
    Returns:
        Tuple of (transition matrix, popularity ranking) for fallback
    """
    print(f"\n{'='*50}\nBuilding fallback model\n{'='*50}")
    start_time = time.time()
    
    # Build primary model (first-order Markov)
    transition_counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(train_df) - 1):
        current_item = train_df.iloc[i]['item_id']
        next_item = train_df.iloc[i]['next_item_id']
        if not pd.isna(next_item):
            transition_counts[current_item][next_item] += 1
    
    most_likely_next = {}
    for current_item, next_items in transition_counts.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    # Build fallback model (item popularity)
    item_counts = Counter(train_df['next_item_id'].dropna())
    popular_items = [item for item, _ in item_counts.most_common()]
    
    print(f"Created fallback model with {len(most_likely_next)} primary transitions")
    print(f"Fallback list contains {len(popular_items)} items")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return transition_counts, most_likely_next, popular_items


# ============================================================================
# TOP-N RECOMMENDATION EVALUATION
# ============================================================================

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


def evaluate_top_n_recommendations(transition_counts, test_df, n_values=[1, 3, 5, 10]):
    """
    Evaluate the model using top-N recommendation accuracy.
    
    Args:
        transition_counts: Dictionary of transition counts
        test_df: Test DataFrame
        n_values: List of N values to evaluate
        
    Returns:
        Dictionary of accuracy scores for each N
    """
    print(f"\n{'='*50}\nEvaluating top-N recommendations\n{'='*50}")
    start_time = time.time()
    
    results = {}
    
    for n in n_values:
        correct = 0
        total = 0
        
        for i in range(len(test_df) - 1):
            current_item = test_df.iloc[i]['item_id']
            true_next_item = test_df.iloc[i]['next_item_id']
            
            # Skip if true_next_item is NaN (last item in a sequence)
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


def evaluate_sequence_model_top_n(sequence_transitions, test_df, sequence_length=2, n_values=[1, 3, 5, 10]):
    """
    Evaluate the higher-order Markov model using top-N recommendation accuracy.
    
    Args:
        sequence_transitions: Dictionary of sequence transition counts
        test_df: Test DataFrame
        sequence_length: Length of sequences
        n_values: List of N values to evaluate
        
    Returns:
        Dictionary of accuracy scores for each N
    """
    print(f"\n{'='*50}\nEvaluating higher-order model with top-N recommendations\n{'='*50}")
    start_time = time.time()
    
    results = {}
    
    for n in n_values:
        correct = 0
        total = 0
        
        for i in range(len(test_df) - sequence_length):
            # Create the current sequence
            current_sequence = tuple(test_df.iloc[i:i+sequence_length]['item_id'].values)
            true_next_item = test_df.iloc[i+sequence_length-1]['next_item_id']
            
            # Skip if true_next_item is NaN (last item in a sequence)
            if pd.isna(true_next_item):
                continue
                
            # Convert to same type for comparison
            true_next_item = int(true_next_item)
            
            if current_sequence in sequence_transitions:
                # Get top N recommendations
                next_items = sorted(sequence_transitions[current_sequence].items(), key=lambda x: x[1], reverse=True)
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


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run all models and evaluations.
    """
    print(f"\n{'='*50}\nImproved Single-User Recommendation Models\n{'='*50}")
    start_time = time.time()
    
    # Configuration
    data_path = "../../data/samples/user_0_processed.csv"
    user_id = 0
    test_size = 0.2
    sequence_length = 2
    decay_factor = 0.9
    popularity_weight = 0.3
    
    # Create models directory if it doesn't exist
    os.makedirs("../../models/single_user", exist_ok=True)
    
    # Load and analyze data
    df = load_and_analyze_data(data_path, user_id)
    
    # Try different train/test split approaches
    split_methods = {
        "chronological": split_train_test_chronological,
        "random": split_train_test_random,
        "session": split_train_test_by_session
    }
    
    all_results = {}
    
    for split_name, split_func in split_methods.items():
        print(f"\n{'='*50}\nUsing {split_name} split\n{'='*50}")
        
        # Split data
        if split_name == "session":
            train_df, test_df = split_func(df, test_size=test_size, session_gap_minutes=30)
        else:
            train_df, test_df = split_func(df, test_size=test_size)
        
        # Model results for this split
        split_results = {}
        
        # Model 1: Basic Transition Matrix
        transition_counts, basic_model = build_basic_transition_matrix(train_df)
        basic_accuracy = evaluate_top_n_recommendations(transition_counts, test_df)
        split_results["Basic Transition Matrix"] = basic_accuracy
        
        # Model 2: Higher-Order Markov Chain
        sequence_transitions, higher_order_model = build_higher_order_markov(train_df, sequence_length)
        higher_order_accuracy = evaluate_sequence_model_top_n(sequence_transitions, test_df, sequence_length)
        split_results[f"Higher-Order Markov (n={sequence_length})"] = higher_order_accuracy
        
        # Model 3: Time-Decay Weighted Matrix
        time_decay_weights, time_decay_model = build_time_decay_matrix(train_df, decay_factor)
        time_decay_accuracy = evaluate_top_n_recommendations(time_decay_weights, test_df)
        split_results[f"Time-Decay Matrix (decay={decay_factor})"] = time_decay_accuracy
        
        # Model 4: Hybrid Model
        hybrid_probs, hybrid_model = build_hybrid_model(train_df, popularity_weight)
        hybrid_accuracy = evaluate_top_n_recommendations(hybrid_probs, test_df)
        split_results[f"Hybrid Model (pop_weight={popularity_weight})"] = hybrid_accuracy
        
        # Save results for this split
        all_results[split_name] = split_results
        
        # Print summary for this split
        print(f"\n{'='*50}\nSummary for {split_name} split\n{'='*50}")
        for model_name, accuracies in split_results.items():
            print(f"{model_name}:")
            for n, acc in accuracies.items():
                print(f"  Top-{n}: {acc:.4f}")
    
    # Print overall summary
    print(f"\n{'='*50}\nOverall Summary\n{'='*50}")
    for split_name, split_results in all_results.items():
        print(f"\n{split_name.capitalize()} Split:")
        for model_name, accuracies in split_results.items():
            best_n = max(accuracies.items(), key=lambda x: x[1])
            print(f"  {model_name}: Best accuracy {best_n[1]:.4f} with Top-{best_n[0]}")
    
    # Find best overall model
    best_split = None
    best_model = None
    best_n = None
    best_accuracy = 0
    
    for split_name, split_results in all_results.items():
        for model_name, accuracies in split_results.items():
            for n, acc in accuracies.items():
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_split = split_name
                    best_model = model_name
                    best_n = n
    
    print(f"\nBest overall: {best_model} with {best_split} split, Top-{best_n} accuracy: {best_accuracy:.4f}")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

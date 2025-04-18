import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
import joblib

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_user_data(data_path, user_id=0):
    """
    Load and preprocess data for a specific user.
    
    Args:
        data_path: Path to the CSV file
        user_id: User ID to filter for (default: 0)
        
    Returns:
        Preprocessed DataFrame for the specified user
    """
    print(f"\n{'='*50}\nLoading data for user {user_id}\n{'='*50}")
    start_time = time.time()
    
    # Load the data
    print(f"Reading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Filter for the specified user if needed
    if 'user_id' in df.columns:
        df = df[df['user_id'] == user_id].copy()
    
    # Sort by timestamp to ensure chronological order
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Print basic statistics
    print(f"Data shape: {df.shape}")
    print(f"Number of unique items: {df['item_id'].nunique()}")
    print(f"Number of unique next items: {df['next_item_id'].nunique()}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    return df


def split_train_test(df, test_size=0.2):
    """
    Split the data into training and test sets, maintaining chronological order.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data to use for testing
        
    Returns:
        train_df, test_df: Training and test DataFrames
    """
    print(f"\n{'='*50}\nSplitting data into train/test sets\n{'='*50}")
    
    # Calculate split index to maintain chronological order
    split_idx = int(len(df) * (1 - test_size))
    
    # Split the data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Training set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
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
        transition_counts[current_item][next_item] += 1
    
    # Create a model that predicts based on most frequent next item
    most_likely_next = {}
    for current_item, next_items in transition_counts.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created transition matrix for {len(most_likely_next)} items")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return most_likely_next


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
        return {}
    
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
    
    return most_likely_next


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
        transition_weights[current_item][next_item] += position_weight
    
    # Create a model that predicts based on highest weighted next item
    most_likely_next = {}
    for current_item, next_items in transition_weights.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created time-decay model for {len(most_likely_next)} items")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return most_likely_next


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
    
    return most_likely_next


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
        transition_counts[current_item][next_item] += 1
    
    most_likely_next = {}
    for current_item, next_items in transition_counts.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    # Build fallback model (item popularity)
    item_counts = Counter(train_df['next_item_id'])
    popular_items = [item for item, _ in item_counts.most_common()]
    
    print(f"Created fallback model with {len(most_likely_next)} primary transitions")
    print(f"Fallback list contains {len(popular_items)} items")
    print(f"Model built in {time.time() - start_time:.2f} seconds")
    
    return most_likely_next, popular_items


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_basic_model(model, test_df):
    """
    Evaluate the basic transition matrix model.
    
    Args:
        model: Dictionary mapping items to their most likely next items
        test_df: Test DataFrame
        
    Returns:
        Accuracy of the model
    """
    print(f"\n{'='*50}\nEvaluating basic transition matrix\n{'='*50}")
    start_time = time.time()
    
    correct = 0
    total = 0
    
    # Loop through all but the last row (which has no next item)
    for i in range(len(test_df) - 1):
        current_item = test_df.iloc[i]['item_id']
        true_next_item = test_df.iloc[i]['next_item_id']
        
        # Skip if true_next_item is NaN (last item in a sequence)
        if pd.isna(true_next_item):
            continue
            
        # Convert to same type for comparison
        true_next_item = int(true_next_item)
        
        if current_item in model:
            total += 1
            predicted_next = model[current_item]
            # Convert to same type for comparison
            if isinstance(predicted_next, float):
                predicted_next = int(predicted_next)
                
            if predicted_next == true_next_item:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    
    return accuracy


def evaluate_higher_order_model(model, test_df, sequence_length=2):
    """
    Evaluate the higher-order Markov chain model.
    
    Args:
        model: Dictionary mapping sequences to their most likely next items
        test_df: Test DataFrame
        sequence_length: Number of previous items in the sequence
        
    Returns:
        Accuracy of the model
    """
    print(f"\n{'='*50}\nEvaluating higher-order Markov chain (sequence length: {sequence_length})\n{'='*50}")
    start_time = time.time()
    
    correct = 0
    total = 0
    
    # We need at least sequence_length+1 items to evaluate
    if len(test_df) <= sequence_length:
        print("Not enough data for evaluation with this sequence length")
        return 0.0
    
    for i in range(len(test_df) - sequence_length):
        # Create the current sequence
        current_sequence = tuple(test_df.iloc[i:i+sequence_length]['item_id'].values)
        # The next item is what we want to predict
        true_next_item = test_df.iloc[i+sequence_length-1]['next_item_id']
        
        # Skip if true_next_item is NaN (last item in a sequence)
        if pd.isna(true_next_item):
            continue
            
        # Convert to same type for comparison
        true_next_item = int(true_next_item)
        
        if current_sequence in model:
            total += 1
            predicted_next = model[current_sequence]
            # Convert to same type for comparison
            if isinstance(predicted_next, float):
                predicted_next = int(predicted_next)
                
            if predicted_next == true_next_item:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    
    return accuracy


def evaluate_fallback_model(primary_model, fallback_list, test_df):
    """
    Evaluate the fallback model.
    
    Args:
        primary_model: Dictionary mapping items to their most likely next items
        fallback_list: List of items sorted by popularity
        test_df: Test DataFrame
        
    Returns:
        Accuracy of the model
    """
    print(f"\n{'='*50}\nEvaluating fallback model\n{'='*50}")
    start_time = time.time()
    
    correct = 0
    total = 0
    primary_hits = 0
    fallback_hits = 0
    
    for i in range(len(test_df) - 1):
        current_item = test_df.iloc[i]['item_id']
        true_next_item = test_df.iloc[i]['next_item_id']
        
        # Skip if true_next_item is NaN (last item in a sequence)
        if pd.isna(true_next_item):
            continue
            
        # Convert to same type for comparison
        true_next_item = int(true_next_item)
        
        total += 1
        
        # Try primary model first
        if current_item in primary_model:
            predicted_item = primary_model[current_item]
            # Convert to same type for comparison
            if isinstance(predicted_item, float):
                predicted_item = int(predicted_item)
                
            if predicted_item == true_next_item:
                correct += 1
                primary_hits += 1
                continue
        
        # Fallback to popularity model - check if in top 10 popular items
        top_n = min(10, len(fallback_list))
        fallback_top_items = [int(item) if isinstance(item, float) else item for item in fallback_list[:top_n]]
        
        if true_next_item in fallback_top_items:
            correct += 1
            fallback_hits += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Primary model hits: {primary_hits}/{total} ({primary_hits/total:.4f})")
    print(f"Fallback model hits: {fallback_hits}/{total} ({fallback_hits/total:.4f})")
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    
    return accuracy


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run all models and evaluations.
    """
    print(f"\n{'='*50}\nSingle-User Recommendation Models\n{'='*50}")
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
    
    # Load and preprocess data
    df = load_user_data(data_path, user_id)
    
    # Ensure item_id and next_item_id are of the same type (int)
    df['item_id'] = df['item_id'].astype('int32')
    # Keep next_item_id as Int32 (nullable integer) to handle NaN values
    df['next_item_id'] = df['next_item_id'].astype('Int32')
    
    # Split data chronologically
    train_df, test_df = split_train_test(df, test_size)
    
    # Print some debug information
    print(f"\nData types in training data:")
    print(train_df.dtypes)
    print(f"\nSample of training data:")
    print(train_df.head(3))
    
    # Build and evaluate models
    results = {}
    
    # Model 1: Basic Transition Matrix
    basic_model = build_basic_transition_matrix(train_df)
    basic_accuracy = evaluate_basic_model(basic_model, test_df)
    results["Basic Transition Matrix"] = basic_accuracy
    
    # Model 2: Higher-Order Markov Chain
    higher_order_model = build_higher_order_markov(train_df, sequence_length)
    higher_order_accuracy = evaluate_higher_order_model(higher_order_model, test_df, sequence_length)
    results[f"Higher-Order Markov (n={sequence_length})"] = higher_order_accuracy
    
    # Model 3: Time-Decay Weighted Matrix
    time_decay_model = build_time_decay_matrix(train_df, decay_factor)
    time_decay_accuracy = evaluate_basic_model(time_decay_model, test_df)
    results[f"Time-Decay Matrix (decay={decay_factor})"] = time_decay_accuracy
    
    # Model 4: Hybrid Model
    hybrid_model = build_hybrid_model(train_df, popularity_weight)
    hybrid_accuracy = evaluate_basic_model(hybrid_model, test_df)
    results[f"Hybrid Model (pop_weight={popularity_weight})"] = hybrid_accuracy
    
    # Model 5: Fallback Model
    fallback_model, popular_items = build_fallback_model(train_df)
    fallback_accuracy = evaluate_fallback_model(fallback_model, popular_items, test_df)
    results["Fallback Model"] = fallback_accuracy
    
    # Print summary of results
    print(f"\n{'='*50}\nSummary of Results\n{'='*50}")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    # Save the best model
    best_model_name = max(results.items(), key=lambda x: x[1])[0]
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
    
    # Save models
    joblib.dump(basic_model, f"../../models/single_user/basic_transition_matrix_user_{user_id}.joblib")
    joblib.dump(higher_order_model, f"../../models/single_user/higher_order_markov_user_{user_id}.joblib")
    joblib.dump(time_decay_model, f"../../models/single_user/time_decay_matrix_user_{user_id}.joblib")
    joblib.dump(hybrid_model, f"../../models/single_user/hybrid_model_user_{user_id}.joblib")
    joblib.dump((fallback_model, popular_items), f"../../models/single_user/fallback_model_user_{user_id}.joblib")
    
    print(f"\nAll models saved to ../../models/single_user/")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

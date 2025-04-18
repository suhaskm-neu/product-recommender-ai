import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

def train_next_item_predictor():
    """
    Load sample data and train a minimal next item prediction model.
    Uses a memory-efficient approach for large numbers of unique items.
    """
    # Configuration
    # sample_data_path = "../data/samples/multi_user_sample.csv"
    sample_data_path = "../data/samples/user_0_processed.csv"
    
    # Load the sample data
    print(f"Loading data from: {sample_data_path}")
    start_time = time.time()
    df = pd.read_csv(sample_data_path)
    print(f"Data shape: {df.shape}")
    
    # Print some basic statistics
    print(f"Number of unique items: {df['item_id'].nunique()}")
    print(f"Number of unique next items: {df['next_item_id'].nunique()}")
    
    # Create a transition probability matrix approach (more suitable for next-item prediction)
    print("\nBuilding transition probability matrix...")
    transition_counts = defaultdict(lambda: defaultdict(int))
    
    # Count transitions from each item to next item
    for i in range(len(df) - 1):
        current_item = df.iloc[i]['item_id']
        next_item = df.iloc[i]['next_item_id']
        transition_counts[current_item][next_item] += 1
    
    # Create a simple model that predicts based on most frequent next item
    most_likely_next = {}
    for current_item, next_items in transition_counts.items():
        if next_items:  # If there are any transitions from this item
            most_likely_next[current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    print(f"Created transition probabilities for {len(most_likely_next)} items")
    
    # Memory optimization: Take a subset of the most frequent items for ML model training
    top_items_count = 100  # Reduced from 1000 to focus on most common patterns
    
    # Get the most frequent items
    top_items = df['item_id'].value_counts().nlargest(top_items_count).index.tolist()
    
    # Filter data to only include rows where item_id is in the top items
    filtered_df = df[df['item_id'].isin(top_items)].copy()
    print(f"Filtered data shape: {filtered_df.shape}")
    
    if filtered_df.empty:
        print("Error: No data left after filtering. Try increasing top_items_count.")
        return None
    
    # Feature engineering: add previous item features
    filtered_df['prev_item_id'] = filtered_df['item_id'].shift(1)
    filtered_df['prev_view_time'] = filtered_df['view_time'].shift(1)
    filtered_df['prev_click_rate'] = filtered_df['click_rate'].shift(1)
    
    # Drop first row which has NaN for previous features
    filtered_df = filtered_df.dropna()
    
    # Prepare features and target
    X = filtered_df[['item_id', 'prev_item_id', 'view_time', 'click_rate', 'prev_view_time', 'prev_click_rate']]
    y = filtered_df['next_item_id'].astype(int)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Scale the features (important for view_time and click_rate)
    scaler = StandardScaler()
    # Only scale the continuous features, not the item IDs
    continuous_features = ['view_time', 'click_rate', 'prev_view_time', 'prev_click_rate']
    X_train_cont = scaler.fit_transform(X_train[continuous_features])
    X_test_cont = scaler.transform(X_test[continuous_features])
    
    # Combine back with the categorical features
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[continuous_features] = X_train_cont
    X_test_scaled[continuous_features] = X_test_cont
    
    # Train a Random Forest model (better for this type of problem than KNN)
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate ML model
    rf_accuracy = model.score(X_test_scaled, y_test)
    print(f"Random Forest model accuracy: {rf_accuracy:.4f}")
    
    # Evaluate transition matrix approach
    correct_predictions = 0
    total_predictions = 0
    
    for i, row in X_test.iterrows():
        current_item = row['item_id']
        true_next_item = y_test[i]
        
        if current_item in most_likely_next:
            total_predictions += 1
            if most_likely_next[current_item] == true_next_item:
                correct_predictions += 1
    
    if total_predictions > 0:
        transition_accuracy = correct_predictions / total_predictions
        print(f"Transition matrix approach accuracy: {transition_accuracy:.4f}")
    else:
        print("No predictions made with transition matrix approach")
    
    # Test prediction with both approaches
    n_samples = min(5, len(X_test))
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    print("\nSample predictions:")
    for idx in sample_indices:
        x = X_test_scaled.iloc[idx:idx+1]
        current_item = X_test.iloc[idx]['item_id']
        true_next_item = y_test.iloc[idx]
        
        # RF model prediction
        rf_pred = model.predict(x)[0]
        
        # Transition matrix prediction
        trans_pred = most_likely_next.get(current_item, "Unknown")
        
        print(f"Current item: {current_item}")
        print(f"  True next: {true_next_item}")
        print(f"  RF model prediction: {rf_pred}")
        print(f"  Transition matrix prediction: {trans_pred}")
        print()
    
    # Save both models
    os.makedirs('../models', exist_ok=True)
    
    # Save Random Forest model
    rf_model_path = '../models/rf_next_item_predictor.joblib'
    joblib.dump(model, rf_model_path)
    
    # Save transition matrix as a dictionary
    trans_model_path = '../models/transition_matrix.joblib'
    joblib.dump(most_likely_next, trans_model_path)
    
    print(f"\nModels saved to:")
    print(f"- Random Forest: {rf_model_path}")
    print(f"- Transition Matrix: {trans_model_path}")
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
    
    return model, most_likely_next

if __name__ == "__main__":
    train_next_item_predictor()
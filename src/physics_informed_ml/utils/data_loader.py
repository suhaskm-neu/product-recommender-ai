"""
Data loading and preprocessing utilities for physics-informed models.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path=None, test_size=0.2, random_state=42):
    """
    Load and preprocess dataset for physics-informed models
    
    Args:
        data_path (str): Path to data file, if None will use default dataset
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, test_df)
    """
    # Set default paths relative to the project root
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')
    
    if data_path is None:
        # Default to user_item_interactions.csv in the data directory
        data_path = os.path.join(project_root, "data/user_item_interactions.csv")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded dataset from {data_path}")
    except FileNotFoundError:
        # Try the sample dataset if main dataset not found
        sample_path = os.path.join(project_root, "data/user_0_processed.csv")
        print(f"Main dataset not found, trying {sample_path}")
        df = pd.read_csv(sample_path)
    
    # Print dataset info
    print(f"Loaded dataset with {len(df)} records and {df['item_id'].nunique()} unique items")
    print(f"Columns: {', '.join(df.columns)}")
    
    # Convert timestamp to datetime depending on its format
    if df['timestamp'].dtype == 'object':
        # String format timestamp (e.g., '2025-02-17 22:52:36.850723')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        # Numeric format (Unix timestamp)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Sort by user_id and timestamp
    df = df.sort_values(by=['user_id', 'timestamp'])
    
    # Handle missing values
    if 'view_time' in df.columns:
        df['view_time'] = df['view_time'].fillna(df['view_time'].median())
    else:
        # If view_time is missing, create a default column
        print("Warning: 'view_time' column not found, creating dummy column")
        df['view_time'] = 0.5  # Default middle value
    
    if 'click_rate' in df.columns:
        df['click_rate'] = df['click_rate'].fillna(0)
    else:
        # If click_rate is missing, create a default column
        print("Warning: 'click_rate' column not found, creating dummy column")
        df['click_rate'] = 0.0  # Default zero value
    
    # Ensure numeric types
    df['view_time'] = pd.to_numeric(df['view_time'], errors='coerce').fillna(0)
    df['click_rate'] = pd.to_numeric(df['click_rate'], errors='coerce').fillna(0)
    
    # Normalize features to [0,1] range to avoid numerical issues
    if df['view_time'].max() > df['view_time'].min():
        df['view_time'] = (df['view_time'] - df['view_time'].min()) / (df['view_time'].max() - df['view_time'].min())
    
    if df['click_rate'].max() > df['click_rate'].min():
        df['click_rate'] = (df['click_rate'] - df['click_rate'].min()) / (df['click_rate'].max() - df['click_rate'].min() + 1e-8)
    
    # Split data - maintain chronological ordering within users
    users = df['user_id'].unique()
    
    if len(users) > 1:
        # If we have multiple users, split by user
        train_users, test_users = train_test_split(users, test_size=test_size, random_state=random_state)
        
        train_df = df[df['user_id'].isin(train_users)]
        test_df = df[df['user_id'].isin(test_users)]
    else:
        # If only one user, split chronologically
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    
    print(f"Split data: {len(train_df)} training records, {len(test_df)} testing records")
    
    return train_df, test_df

def prepare_sequence_data(df, include_features=True):
    """
    Prepare sequential data for next-item prediction
    
    Args:
        df (pandas.DataFrame): DataFrame with user-item interactions
        include_features (bool): Whether to include additional features
        
    Returns:
        tuple: X (features), y (target items)
    """
    # Group by user to create sequences
    sequences = []
    for user_id, user_df in df.groupby('user_id'):
        # Sort by timestamp to ensure chronological order
        user_df = user_df.sort_values('timestamp')
        
        # Extract item sequence
        items = user_df['item_id'].values
        
        # Create (current_item, next_item) pairs
        for i in range(len(items) - 1):
            if include_features:
                # Include current item features
                row = user_df.iloc[i]
                
                # Extract all relevant features
                features = {
                    'user_id': user_id,
                    'current_item': items[i],
                    'next_item': items[i+1],
                    'view_time': row['view_time'],
                    'click_rate': row['click_rate'],
                    'timestamp': row['timestamp']
                }
                
                # Add time delta if available (for sequence i+1)
                if i < len(items) - 2:
                    next_row = user_df.iloc[i+1]
                    features['time_delta'] = (next_row['timestamp'] - row['timestamp']).total_seconds()
                else:
                    features['time_delta'] = 0.0
                    
                sequences.append(features)
            else:
                # Simple pairs without features
                sequences.append((items[i], items[i+1]))
    
    if include_features:
        # Convert to DataFrame
        return pd.DataFrame(sequences)
    else:
        # Return list of tuples
        return sequences

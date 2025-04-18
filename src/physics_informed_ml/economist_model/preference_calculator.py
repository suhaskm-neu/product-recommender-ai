"""
Implementation of the economist's user preference model.

This module implements the economist's equation:
p(t) = 1 / (1 + e^(2*vt + 2*clr))

Where:
- p(t) is the probability the user will buy/interact with the item
- vt is the view time
- clr is the click rate
"""

import numpy as np
import pandas as pd

def calculate_user_preference(view_time, click_rate):
    """
    Calculate user preference using the economist's equation:
    p(t) = 1 / (1 + e^(2*vt + 2*clr))
    
    Args:
        view_time (float): User's view time for an item
        click_rate (float): User's click rate for an item
        
    Returns:
        float: Calculated user preference probability
    """
    # Ensure inputs are numeric
    view_time = float(view_time)
    click_rate = float(click_rate)
    
    # Apply the economist's equation
    return 1.0 / (1.0 + np.exp(2 * view_time + 2 * click_rate))

def augment_dataset_with_preference(df):
    """
    Add user preference as a new feature to the dataset
    
    Args:
        df (pandas.DataFrame): DataFrame with view_time and click_rate columns
        
    Returns:
        pandas.DataFrame: DataFrame with added user_preference column
    """
    # Create a copy to avoid modifying the original
    df_augmented = df.copy()
    
    # Apply the preference calculation to each row
    df_augmented['user_preference'] = df_augmented.apply(
        lambda row: calculate_user_preference(row['view_time'], row['click_rate']), 
        axis=1
    )
    
    # Verify the preference is within [0,1] range
    min_pref = df_augmented['user_preference'].min()
    max_pref = df_augmented['user_preference'].max()
    
    print(f"User preference range: [{min_pref:.4f}, {max_pref:.4f}]")
    
    # Check for potential numerical issues
    if np.isnan(df_augmented['user_preference']).any():
        print("Warning: NaN values detected in user_preference")
        df_augmented['user_preference'] = df_augmented['user_preference'].fillna(0.5)
    
    if (df_augmented['user_preference'] <= 0).any() or (df_augmented['user_preference'] >= 1).any():
        print("Warning: User preference values outside [0,1] range detected")
        # Clip values to ensure they're in range
        df_augmented['user_preference'] = df_augmented['user_preference'].clip(0.001, 0.999)
    
    return df_augmented

def analyze_preference_distribution(df):
    """
    Analyze the distribution of user preferences
    
    Args:
        df (pandas.DataFrame): DataFrame with user_preference column
        
    Returns:
        dict: Statistics about preference distribution
    """
    if 'user_preference' not in df.columns:
        df = augment_dataset_with_preference(df)
    
    stats = {
        'count': len(df),
        'min': df['user_preference'].min(),
        'max': df['user_preference'].max(),
        'mean': df['user_preference'].mean(),
        'median': df['user_preference'].median(),
        'std': df['user_preference'].std(),
        'quartiles': [
            df['user_preference'].quantile(0.25),
            df['user_preference'].quantile(0.5),
            df['user_preference'].quantile(0.75)
        ]
    }
    
    # Calculate distribution by item popularity
    item_counts = df['item_id'].value_counts()
    top_items = item_counts.index[:100]  # Top 100 items
    
    # Calculate average preference for top items vs others
    top_mask = df['item_id'].isin(top_items)
    stats['top_items_avg_pref'] = df.loc[top_mask, 'user_preference'].mean()
    stats['other_items_avg_pref'] = df.loc[~top_mask, 'user_preference'].mean()
    
    # Print key findings
    print(f"Preference Analysis:")
    print(f"  Mean preference: {stats['mean']:.4f}")
    print(f"  Top items avg preference: {stats['top_items_avg_pref']:.4f}")
    print(f"  Other items avg preference: {stats['other_items_avg_pref']:.4f}")
    
    return stats

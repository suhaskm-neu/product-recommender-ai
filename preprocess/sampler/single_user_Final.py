import pandas as pd
import os
import time
from datetime import datetime
import gc

def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*50)
    print(f"{title}")
    print("="*50)

def process_single_user_data(data_path: str, user_id: int = 0, output_dir: str = None):
    """
    Process and save data for a single user.
    
    Args:
        data_path: Path to the processed data CSV
        user_id: ID of the user to extract (default: 0)
        output_dir: Directory to save the output file. If None, uses base_dir/data/samples
    
    Returns:
        Path to the saved file
    """
    print_section_header(f"Processing Data for User {user_id}")
    start_time = time.time()
    
    try:
        # Set default output directory if not provided
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, 'data', 'samples')
        
        # Check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Specify dtypes for memory optimization
        dtypes = {
            'user_id': 'int32',
            'item_id': 'int32',
            'timestamp': 'float32',  # Unix timestamp in seconds
            'view_time': 'float32',
            'click_rate': 'float32'
        }
        
        # Load only needed columns
        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path, usecols=list(dtypes.keys()), dtype=dtypes)
        print(f"Initial data shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Filter for specific user and sort by timestamp
        df = df[df['user_id'] == user_id].sort_values('timestamp').reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No data found for user_id: {user_id}")
            
        print(f"Data shape after filtering user {user_id}: {df.shape}")
        
        # Display time range in both Unix timestamp and human-readable format
        min_timestamp = int(df['timestamp'].min())
        max_timestamp = int(df['timestamp'].max())
        print(f"Time range (Unix timestamp): {min_timestamp} to {max_timestamp}")
        print(f"Time range (Human readable): {datetime.fromtimestamp(min_timestamp)} to {datetime.fromtimestamp(max_timestamp)}")
        print(f"Number of unique items: {df['item_id'].nunique()}")
        
        # Generate next_item_id and ensure it's an integer type
        df['next_item_id'] = df['item_id'].shift(-1)
        # Convert next_item_id to int32 where not null
        df['next_item_id'] = df['next_item_id'].astype('Int32')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the processed data
        output_path = os.path.join(output_dir, f"user_{user_id}_processed.csv")
        df.to_csv(output_path, index=False)
        
        process_time = time.time() - start_time
        print(f"Processing completed in {process_time:.2f} seconds")
        print(f"Processed data saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        raise

def main():
    # Get base directory and set up paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'processed_data.csv')
    
    # Process data for user 0 (can be changed to any user ID)
    user_id = 0
    output_path = process_single_user_data(data_path, user_id)
    
    print_section_header("Summary")
    print(f"Successfully processed and saved data for user {user_id}")
    print(f"Output file: {output_path}")

if __name__ == "__main__":
    main()

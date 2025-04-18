#!/usr/bin/env python
"""

Runner script for the hybrid social-physics model tests

This script executes the hybrid model with top-5 and top-10 configurations
as requested and provides a comprehensive evaluation.
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

def main():
    """Run tests for the hybrid social-physics model"""
    parser = argparse.ArgumentParser(description='Run Hybrid Social-Physics Model Tests')
    
    parser.add_argument('--data-file', type=str, default='user_0_processed.csv',
                      help='Name of the data file (default: user_0_processed.csv)')
    parser.add_argument('--data-dir', type=str, default=None,
                      help='Path to data directory (default: project_root/data/samples)')
    parser.add_argument('--use-social', action='store_true',
                      help='Whether to use social network data if available')
    parser.add_argument('--results-dir', type=str, default='results',
                      help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = os.path.join(project_root, 'data', 'samples')
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(project_root, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Set file paths
    data_path = os.path.join(args.data_dir, args.data_file)
    
    # Check if social data exists (user's social connections)
    social_path = None
    if args.use_social:
        social_file = f"social_connections_{args.data_file.split('_')[1]}.csv"
        potential_social_path = os.path.join(args.data_dir, social_file)
        if os.path.exists(potential_social_path):
            social_path = potential_social_path
    
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"hybrid_model_results_{timestamp}.json")
    
    # Import test module - using relative import
    sys.path.insert(0, os.path.dirname(current_dir))  # Add parent directory to beginning of path
    from hybrid_social_physics.test_hybrid_model import main as test_main
    
    # Create custom argument namespace for test script
    test_args = argparse.Namespace(
        data_path=data_path,
        item_counts=[5, 10],  # Test with both top-5 and top-10
        output=results_file,
        test_size=0.2,
        random_split=False
    )
    
    # Set social path if available
    test_args.social_path = social_path
    
    # Run the test
    print(f"Running hybrid model test with data: {data_path}")
    if social_path:
        print(f"Using social data: {social_path}")
    print(f"Testing with top item counts: {test_args.item_counts}")
    print(f"Results will be saved to: {results_file}")
    
    test_main(test_args)
    
    print("\nTest execution complete. To run again with different parameters:")
    print(f"python {os.path.basename(__file__)} --data-file <file_name> [--use-social] [--data-dir <directory>]")

if __name__ == "__main__":
    main()

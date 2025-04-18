"""
Test script for the Hybrid Social-Physics Recommendation Model

This script evaluates the hybrid model performance by:
1. Loading and preprocessing interaction data
2. Filtering to focus on top-N items 
3. Training the hybrid model
4. Evaluating with both exact match and top-k metrics

The script focuses on top-5 and top-10 models as specified.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import json

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # physics_informed_ml dir
project_dir = os.path.dirname(os.path.dirname(parent_dir))  # project dir
sys.path.append(project_dir)

# Import from current package
from model import HybridSocialPhysicsModel
from data_utils import (
    load_interactions_data,
    load_social_data,
    filter_top_n_items,
    train_test_split,
    prepare_test_sequences,
    print_section
)

def evaluate_model(model, test_sequences, top_ks=[1, 3, 5, 10]):
    """
    Evaluate model using exact match and top-k accuracy metrics
    
    Args:
        model: Trained HybridSocialPhysicsModel
        test_sequences: List of test sequence dictionaries
        top_ks: List of k values for top-k accuracy
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    print_section("Model Evaluation")
    
    exact_correct = 0
    top_k_correct = {k: 0 for k in top_ks}
    total = len(test_sequences)
    
    # Store predictions for analysis
    all_predictions = []
    
    for i, seq in enumerate(test_sequences):
        user_id = seq['user_id']
        current_item = seq['current_item']
        actual_next = seq['next_item']
        
        # Get additional features
        view_time = seq.get('view_time')
        click_rate = seq.get('click_rate')
        time_delta = seq.get('time_delta', 1.0)
        
        # Evaluate exact prediction
        prediction = model.predict(
            user_id, current_item, 
            view_time=view_time, 
            click_rate=click_rate,
            time_delta=time_delta
        )
        
        # Get top-k predictions
        top_k_max = max(top_ks)
        top_k_predictions = model.predict_top_k(
            user_id, current_item, k=top_k_max,
            view_time=view_time, 
            click_rate=click_rate,
            time_delta=time_delta
        )
        
        # Check if exact prediction is correct
        if prediction == actual_next:
            exact_correct += 1
            
        # Check top-k accuracy
        for k in top_ks:
            if actual_next in top_k_predictions[:k]:
                top_k_correct[k] += 1
                
        # Store predictions for analysis
        all_predictions.append({
            'user_id': user_id,
            'current_item': current_item,
            'actual_next': actual_next,
            'predicted': prediction,
            'top_k': top_k_predictions[:top_k_max],
            'is_correct_exact': prediction == actual_next,
            'is_correct_top_k': {k: actual_next in top_k_predictions[:k] for k in top_ks}
        })
        
        # Show progress
        if (i+1) % 10 == 0 or (i+1) == total:
            print(f"Evaluated {i+1}/{total} sequences")
    
    # Calculate metrics
    exact_accuracy = exact_correct / total * 100 if total > 0 else 0
    top_k_accuracy = {k: (top_k_correct[k] / total * 100) if total > 0 else 0 for k in top_ks}
    
    # Print results
    print("\nEvaluation Results:")
    print(f"- Total test sequences: {total}")
    print(f"- Exact match accuracy: {exact_accuracy:.2f}%")
    for k in top_ks:
        print(f"- Top-{k} accuracy: {top_k_accuracy[k]:.2f}%")
    
    # Return metrics
    return {
        'exact_accuracy': exact_accuracy,
        'top_k_accuracy': top_k_accuracy,
        'predictions': all_predictions
    }

def main(args):
    """Main function for testing the hybrid model"""
    # Print current timestamp
    start_time = datetime.now()
    print(f"Starting hybrid model testing at: {start_time}")
    
    # Load data
    interactions_df = load_interactions_data(args.data_path)
    social_df = load_social_data(args.social_path) if args.social_path else None
    
    # Test with different model configurations
    results = {}
    
    for item_count in args.item_counts:
        print_section(f"Testing with Top {item_count} Items")
        
        # Filter to top items
        filtered_df, top_items = filter_top_n_items(interactions_df, n=item_count)
        
        # Split data for training and testing
        train_df, test_df = train_test_split(
            filtered_df, test_size=args.test_size, 
            chronological=not args.random_split
        )
        
        # Prepare test sequences
        test_sequences = prepare_test_sequences(test_df)
        
        # Define model weight configurations to test
        model_configs = [
            (0.6, 0.3, "Default Weights"),
            (0.7, 0.2, "High Physics"),
            (0.5, 0.4, "High Social"),
            (0.8, 0.1, "Physics Dominant"),
            (0.4, 0.5, "Social Dominant")
        ]
        
        config_results = {}
        
        for physics_weight, social_weight, config_name in model_configs:
            print_section(f"Model Configuration: {config_name} (Physics: {physics_weight}, Social: {social_weight})")
            
            # Create and train model
            model = HybridSocialPhysicsModel(
                top_n_items=item_count,
                preference_evolution_weight=physics_weight,
                social_weight=social_weight
            )
            
            model.fit(train_df, social_df)
            
            # Evaluate model
            eval_results = evaluate_model(model, test_sequences)
            
            # Save results for this configuration
            config_results[config_name] = {
                'physics_weight': physics_weight,
                'social_weight': social_weight,
                'exact_accuracy': eval_results['exact_accuracy'],
                'top_k_accuracy': eval_results['top_k_accuracy']
            }
        
        # Store results for this item count
        results[f"top_{item_count}"] = config_results
    
    # Print overall summary
    print_section("Overall Summary")
    for item_count_key, item_results in results.items():
        print(f"\n{item_count_key.replace('_', ' ').title()}:")
        for config_name, metrics in item_results.items():
            print(f"  {config_name}:")
            print(f"    - Exact: {metrics['exact_accuracy']:.2f}%")
            print(f"    - Top-1: {metrics['top_k_accuracy'][1]:.2f}%")
            print(f"    - Top-5: {metrics['top_k_accuracy'][5]:.2f}%")
            if 10 in metrics['top_k_accuracy']:
                print(f"    - Top-10: {metrics['top_k_accuracy'][10]:.2f}%")
    
    # Print execution time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\nTest completed in {duration:.2f} seconds")
    
    # Save results to file
    if args.output:
        results_path = args.output
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Hybrid Social-Physics Recommendation Model')
    
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to interactions data CSV')
    parser.add_argument('--social-path', type=str, default=None,
                        help='Path to social connections data CSV')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    parser.add_argument('--item-counts', type=int, nargs='+', default=[5, 10],
                        help='Number of top items to focus on (can specify multiple)')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data for testing')
    parser.add_argument('--random-split', action='store_true',
                        help='Use random train/test split instead of chronological')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

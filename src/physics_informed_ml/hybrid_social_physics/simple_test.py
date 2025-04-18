#!/usr/bin/env python
"""
Simple test script for the Hybrid Social-Physics Model

This script tests the hybrid model with top-5 and top-10 configurations
as requested, using a more direct approach to avoid import issues.
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
import time
from datetime import datetime

# Set up proper import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # physics_informed_ml directory
sys.path.append(parent_dir)

# Import local modules
from model import HybridSocialPhysicsModel
from data_utils import (
    load_interactions_data, 
    filter_top_n_items,
    train_test_split,
    prepare_test_sequences,
    print_section
)

def test_with_item_count(data_path, item_count, chronological=True):
    """Run test for a specific item count"""
    print_section(f"Testing Hybrid Social-Physics Model with Top {item_count} Items")
    
    # 1. Load data
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} interactions from {data_path}")
    
    # 2. Ensure required columns exist
    required_cols = ['user_id', 'item_id', 'timestamp']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found")
            return
    
    # 3. Filter to top-N items
    filtered_df, top_items = filter_top_n_items(df, n=item_count)
    print(f"Filtered to top {item_count} items: {top_items}")
    
    # 4. Split into train/test
    train_df, test_df = train_test_split(
        filtered_df, test_size=0.2, chronological=chronological
    )
    
    # 5. Create test sequences
    test_sequences = prepare_test_sequences(test_df)
    
    # 6. Define weight configurations to test
    weights = [
        (0.6, 0.3, "Default Weights"),
        (0.7, 0.2, "High Physics"),
        (0.5, 0.4, "High Social"),
        (0.8, 0.1, "Physics Dominant"),
        (0.4, 0.5, "Social Dominant")
    ]
    
    # 7. Test each configuration
    results = {}
    for physics_weight, social_weight, config_name in weights:
        print_section(f"Configuration: {config_name}")
        print(f"Physics weight: {physics_weight}, Social weight: {social_weight}")
        
        # Create and train model
        model = HybridSocialPhysicsModel(
            top_n_items=item_count,
            preference_evolution_weight=physics_weight,
            social_weight=social_weight
        )
        
        # Train model
        model.fit(train_df)
        
        # Test model
        eval_results = evaluate_model(model, test_sequences)
        results[config_name] = eval_results
    
    return results

def evaluate_model(model, test_sequences):
    """Evaluate model using top-k metrics"""
    top_ks = [1, 3, 5, 10]
    exact_correct = 0
    top_k_correct = {k: 0 for k in top_ks}
    total = len(test_sequences)
    
    print(f"Evaluating model on {total} test sequences...")
    
    for seq in test_sequences:
        user_id = seq['user_id']
        current_item = seq['current_item']
        actual_next = seq['next_item']
        
        # Get additional features
        view_time = seq.get('view_time')
        click_rate = seq.get('click_rate')
        
        # Get exact prediction
        prediction = model.predict(
            user_id, current_item, 
            view_time=view_time, 
            click_rate=click_rate
        )
        
        # Check exact match
        if prediction == actual_next:
            exact_correct += 1
        
        # Get top-k predictions
        top_k_max = max(top_ks)
        top_k_preds = model.predict_top_k(
            user_id, current_item, k=top_k_max,
            view_time=view_time, click_rate=click_rate
        )
        
        # Check top-k accuracy
        for k in top_ks:
            if actual_next in top_k_preds[:k]:
                top_k_correct[k] += 1
    
    # Calculate metrics
    exact_accuracy = exact_correct / total * 100 if total > 0 else 0
    top_k_accuracy = {k: (top_k_correct[k] / total * 100) if total > 0 else 0 for k in top_ks}
    
    # Print results
    print("\nEvaluation Results:")
    print(f"- Exact match accuracy: {exact_accuracy:.2f}%")
    for k in top_ks:
        print(f"- Top-{k} accuracy: {top_k_accuracy[k]:.2f}%")
    
    return {
        'exact_accuracy': exact_accuracy,
        'top_k_accuracy': top_k_accuracy
    }

def main():
    """Main function"""
    # Start timing
    start_time = time.time()
    
    # Set paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    data_dir = os.path.join(project_root, 'data', 'samples')
    data_path = os.path.join(data_dir, 'user_0_processed.csv')
    
    # Test with top-5 and top-10 items
    results = {}
    
    # Run top-5 test
    results['top_5'] = test_with_item_count(data_path, 5)
    
    # Run top-10 test
    results['top_10'] = test_with_item_count(data_path, 10)
    
    # Print overall summary
    print_section("Overall Performance Summary")
    
    for model_type, model_results in results.items():
        print(f"\n{model_type.replace('_', ' ').title()}:")
        for config_name, metrics in model_results.items():
            print(f"  {config_name}:")
            print(f"    - Exact: {metrics['exact_accuracy']:.2f}%")
            print(f"    - Top-1: {metrics['top_k_accuracy'][1]:.2f}%")
            print(f"    - Top-5: {metrics['top_k_accuracy'][5]:.2f}%")
            if 10 in metrics['top_k_accuracy']:
                print(f"    - Top-10: {metrics['top_k_accuracy'][10]:.2f}%")
    
    # Print execution time
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nTest completed in {duration:.2f} seconds")

if __name__ == "__main__":
    main()

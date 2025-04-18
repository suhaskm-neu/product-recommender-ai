"""
Main script for physics-informed recommendation models.

This script implements both the economist's preference model and the 
mathematician's differential equation model for predicting user preferences
and next-item recommendations.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import argparse
from datetime import datetime

# Set up path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Import custom modules - use relative imports
from utils.data_loader import load_and_preprocess_data, prepare_sequence_data
from utils.evaluation import evaluate_model, evaluate_detailed
from economist_model.preference_calculator import augment_dataset_with_preference, analyze_preference_distribution
from economist_model.model_trainer import PreferenceEnhancedMarkovModel
from differential_equation.model_trainer import DifferentialMarkovModel

# Try importing MLflow, but it's optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Metrics will be logged to console only.")

def setup_mlflow():
    """Set up MLflow tracking if available"""
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("physics_informed_models")
        return True
    return False

def train_preference_model(train_df, test_df, top_n_items=100, preference_weight=0.5):
    """
    Train the economist's preference-enhanced Markov model
    
    Args:
        train_df (DataFrame): Training data
        test_df (DataFrame): Testing data
        top_n_items (int): Number of top items to consider
        preference_weight (float): Weight for preference in prediction
        
    Returns:
        tuple: (model, evaluation results)
    """
    print("\n" + "="*80)
    print("Training Preference-Enhanced Markov Model")
    print("="*80)
    
    # Add preference feature to datasets
    print("Calculating user preferences using economist's equation")
    train_df_with_pref = augment_dataset_with_preference(train_df)
    test_df_with_pref = augment_dataset_with_preference(test_df)
    
    # Analyze preference distribution
    print("\nPreference Distribution Analysis:")
    pref_stats = analyze_preference_distribution(train_df_with_pref)
    
    # Track start time
    start_time = time.time()
    
    # Create and train model
    model = PreferenceEnhancedMarkovModel(top_n_items=top_n_items, preference_weight=preference_weight)
    
    # Fit the model
    model.fit(
        train_df_with_pref['item_id'].tolist(),
        train_df_with_pref['item_id'].tolist(),
        train_df_with_pref['user_preference'].tolist()
    )
    
    # Calculate training time
    train_time = time.time() - start_time
    print(f"Model trained in {train_time:.2f} seconds")
    
    # Prepare test data
    test_sequences = prepare_sequence_data(test_df_with_pref, include_features=False)
    
    # Evaluate model
    print("\nEvaluating model performance:")
    results = evaluate_model(model, test_sequences)
    
    # Save model
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "preference_enhanced_markov.pkl")
    model.save(model_path)
    
    # Log with MLflow if available
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="preference_enhanced_markov"):
            # Log parameters
            mlflow.log_param("model_type", "preference_enhanced_markov")
            mlflow.log_param("top_n_items", top_n_items)
            mlflow.log_param("preference_weight", preference_weight)
            mlflow.log_param("training_samples", len(train_df))
            mlflow.log_param("test_samples", len(test_df))
            
            # Log metrics
            mlflow.log_metric("training_time", train_time)
            mlflow.log_metric("exact_match_accuracy", results['exact_match_accuracy'])
            for k, acc in results['top_k_accuracy'].items():
                mlflow.log_metric(f"top_{k}_accuracy", acc)
            
            # Log model artifact
            mlflow.log_artifact(model_path)
    
    return model, results

def train_differential_model(train_df, test_df, top_n_items=100, preference_evolution_weight=0.5):
    """
    Train the mathematician's differential equation Markov model
    
    Args:
        train_df (DataFrame): Training data
        test_df (DataFrame): Testing data
        top_n_items (int): Number of top items to consider
        preference_evolution_weight (float): Weight for preference evolution
        
    Returns:
        tuple: (model, evaluation results)
    """
    print("\n" + "="*80)
    print("Training Differential Equation Markov Model")
    print("="*80)
    
    # Calculate time intervals
    print("Preparing time and feature data for differential equation model")
    
    # Ensure timestamps are properly formatted
    if train_df['timestamp'].dtype == 'object':
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    if test_df['timestamp'].dtype == 'object':
        test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    
    # Convert to Unix timestamp for numerical operations
    train_df['timestamp_unix'] = train_df['timestamp'].astype(int) / 10**9
    test_df['timestamp_unix'] = test_df['timestamp'].astype(int) / 10**9
    
    # Calculate time deltas
    train_df['time_delta'] = train_df.groupby('user_id')['timestamp_unix'].diff().fillna(1.0)
    test_df['time_delta'] = test_df.groupby('user_id')['timestamp_unix'].diff().fillna(1.0)
    
    # Handle non-positive time deltas (important for the differential equation)
    train_df['time_delta'] = train_df['time_delta'].apply(lambda x: max(x, 0.001))
    test_df['time_delta'] = test_df['time_delta'].apply(lambda x: max(x, 0.001))
    
    print(f"Time delta statistics: min={train_df['time_delta'].min():.2f}, " 
          f"mean={train_df['time_delta'].mean():.2f}, max={train_df['time_delta'].max():.2f}")
    
    # Track start time
    start_time = time.time()
    
    # Create and train model
    model = DifferentialMarkovModel(
        top_n_items=top_n_items, 
        preference_evolution_weight=preference_evolution_weight
    )
    
    # Fit the model
    model.fit(
        train_df['item_id'].tolist(),
        train_df['item_id'].tolist(),
        train_df['view_time'].tolist(),
        train_df['click_rate'].tolist(),
        train_df['timestamp_unix'].tolist()
    )
    
    # Calculate training time
    train_time = time.time() - start_time
    print(f"Model trained in {train_time:.2f} seconds")
    
    # Prepare test data (simple sequences for basic evaluation)
    test_sequences = prepare_sequence_data(test_df, include_features=False)
    
    # Evaluate model
    print("\nEvaluating model performance:")
    results = evaluate_model(model, test_sequences)
    
    # Save model
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "differential_markov.pkl")
    model.save(model_path)
    
    # Log with MLflow if available
    if MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="differential_markov"):
            # Log parameters
            mlflow.log_param("model_type", "differential_markov")
            mlflow.log_param("top_n_items", top_n_items)
            mlflow.log_param("preference_evolution_weight", preference_evolution_weight)
            mlflow.log_param("training_samples", len(train_df))
            mlflow.log_param("test_samples", len(test_df))
            
            # Log metrics
            mlflow.log_metric("training_time", train_time)
            mlflow.log_metric("exact_match_accuracy", results['exact_match_accuracy'])
            for k, acc in results['top_k_accuracy'].items():
                mlflow.log_metric(f"top_{k}_accuracy", acc)
            
            # Log model artifact
            mlflow.log_artifact(model_path)
    
    return model, results

def compare_models(preference_results, differential_results):
    """
    Compare the performance of both models
    
    Args:
        preference_results (dict): Results from preference model
        differential_results (dict): Results from differential model
    """
    print("\n" + "="*80)
    print("Model Comparison")
    print("="*80)
    
    print(f"{'Metric':<20} {'Preference Model':<20} {'Differential Model':<20}")
    print("-" * 60)
    
    print(f"{'Exact Match':<20} {preference_results['exact_match_accuracy']:.4f}{'':>13} {differential_results['exact_match_accuracy']:.4f}{'':>13}")
    
    for k in sorted(preference_results['top_k_accuracy'].keys()):
        pref_acc = preference_results['top_k_accuracy'][k]
        diff_acc = differential_results['top_k_accuracy'][k]
        print(f"{'Top-' + str(k) + ' Accuracy':<20} {pref_acc:.4f}{'':>13} {diff_acc:.4f}{'':>13}")
    
    # Determine the winner
    pref_win = 0
    diff_win = 0
    
    # Compare exact match
    if preference_results['exact_match_accuracy'] > differential_results['exact_match_accuracy']:
        pref_win += 1
    elif differential_results['exact_match_accuracy'] > preference_results['exact_match_accuracy']:
        diff_win += 1
    
    # Compare top-k
    for k in preference_results['top_k_accuracy']:
        if preference_results['top_k_accuracy'][k] > differential_results['top_k_accuracy'][k]:
            pref_win += 1
        elif differential_results['top_k_accuracy'][k] > preference_results['top_k_accuracy'][k]:
            diff_win += 1
    
    print("\nOverall Winner:", end=" ")
    if pref_win > diff_win:
        print("Preference Model")
    elif diff_win > pref_win:
        print("Differential Model")
    else:
        print("Tie")

def main(args):
    """
    Main function to run the models
    
    Args:
        args: Command-line arguments
    """
    # Set up MLflow
    use_mlflow = setup_mlflow()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    train_df, test_df = load_and_preprocess_data(args.data_path, args.test_size, args.random_state)
    
    # Print dataset statistics
    print(f"\nTraining set: {len(train_df)} records, {train_df['user_id'].nunique()} users, {train_df['item_id'].nunique()} items")
    print(f"Testing set: {len(test_df)} records, {test_df['user_id'].nunique()} users, {test_df['item_id'].nunique()} items")
    
    # Train models
    preference_model, preference_results = train_preference_model(
        train_df, test_df, args.top_n_items, args.preference_weight
    )
    
    differential_model, differential_results = train_differential_model(
        train_df, test_df, args.top_n_items, args.preference_evolution_weight
    )
    
    # Compare model performance
    compare_models(preference_results, differential_results)
    
    print("\nTraining and evaluation complete. Models saved in the 'models' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train physics-informed recommendation models")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to the dataset (defaults to built-in dataset)")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for testing")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Model parameters
    parser.add_argument("--top_n_items", type=int, default=100,
                        help="Number of most frequent items to consider")
    parser.add_argument("--preference_weight", type=float, default=0.5,
                        help="Weight for preference in the economist's model")
    parser.add_argument("--preference_evolution_weight", type=float, default=0.5,
                        help="Weight for preference evolution in the differential model")
    
    args = parser.parse_args()
    main(args)

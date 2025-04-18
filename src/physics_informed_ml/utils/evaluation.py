"""
Evaluation utilities for physics-informed recommendation models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_sequences, k_values=[1, 3, 5, 10]):
    """
    Evaluate the model using accuracy and top-k metrics
    
    Args:
        model: The trained model (must have predict and predict_top_k methods)
        test_sequences (list): List of (current_item, next_item) pairs or DataFrame with sequence data
        k_values (list): List of k values for top-k evaluation
        
    Returns:
        dict: Evaluation metrics
    """
    results = {
        'exact_match_accuracy': 0.0,
        'top_k_accuracy': {k: 0.0 for k in k_values}
    }
    
    # Convert DataFrame to sequence pairs if needed
    if isinstance(test_sequences, pd.DataFrame):
        sequence_pairs = []
        for _, row in test_sequences.iterrows():
            sequence_pairs.append((row['current_item'], row['next_item']))
        test_sequences = sequence_pairs
    
    total = len(test_sequences)
    if total == 0:
        print("Warning: Empty test sequence")
        return results
        
    correct = 0
    top_k_correct = {k: 0 for k in k_values}
    
    for current_item, actual_next_item in test_sequences:
        # Get exact prediction
        predicted_next_item = model.predict(current_item)
        
        if predicted_next_item == actual_next_item:
            correct += 1
            
        # For top-k evaluation (if the model supports it)
        if hasattr(model, 'predict_top_k'):
            top_k_predictions = model.predict_top_k(current_item, max(k_values))
            
            # Check if actual item is in each k-sized prediction set
            for k in k_values:
                if actual_next_item in top_k_predictions[:k]:
                    top_k_correct[k] += 1
    
    results['exact_match_accuracy'] = correct / total
    
    for k in k_values:
        if hasattr(model, 'predict_top_k'):
            results['top_k_accuracy'][k] = top_k_correct[k] / total
        else:
            results['top_k_accuracy'][k] = np.nan
    
    # Print evaluation results
    print(f"Evaluation Results:")
    print(f"  Exact match accuracy: {results['exact_match_accuracy']:.4f}")
    for k in k_values:
        if not np.isnan(results['top_k_accuracy'][k]):
            print(f"  Top-{k} accuracy: {results['top_k_accuracy'][k]:.4f}")
    
    return results

def evaluate_detailed(y_true, y_pred):
    """
    Compute detailed classification metrics for multi-class prediction
    
    Args:
        y_true (array-like): True class labels
        y_pred (array-like): Predicted class labels
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    return metrics

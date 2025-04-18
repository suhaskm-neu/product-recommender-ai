import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import networkx as nx
import joblib
from sklearn.model_selection import train_test_split

"""
SOCIAL NETWORK INFLUENCED RECOMMENDATION MODEL

This implementation extends the Markov model approach by incorporating social network influence.
The model combines:
1. User's own interaction history (Markov transition probabilities)
2. Friends' preferences (social network influence)

The social influence is weighted based on:
- Connection strength between users
- Similarity in item preferences
- Recency of friends' interactions
"""

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def load_data(interactions_path, social_network_path):
    """
    Load user interaction data and social network data.
    
    Args:
        interactions_path: Path to the CSV file with user interaction data
        social_network_path: Path to the CSV file with social network data
        
    Returns:
        Tuple of (interactions DataFrame, social network DataFrame)
    """
    print_section("Loading Data")
    
    # Load user interactions
    print(f"Loading user interactions from: {interactions_path}")
    interactions_df = pd.read_csv(interactions_path)
    
    # Load social network data
    print(f"Loading social network data from: {social_network_path}")
    social_df = pd.read_csv(social_network_path)
    
    # Print basic statistics
    print(f"\nUser Interactions:")
    print(f"- Shape: {interactions_df.shape}")
    print(f"- Unique users: {interactions_df['user_id'].nunique()}")
    print(f"- Unique items: {interactions_df['item_id'].nunique()}")
    
    print(f"\nSocial Network:")
    print(f"- Shape: {social_df.shape}")
    print(f"- Unique users: {social_df['user_id'].nunique()}")
    print(f"- Average connections per user: {social_df.groupby('user_id').size().mean():.2f}")
    
    return interactions_df, social_df

def build_user_transition_matrices(df):
    """
    Build transition matrices for each user.
    
    Args:
        df: DataFrame with user interactions
        
    Returns:
        Dictionary mapping users to their transition matrices
    """
    print_section("Building User Transition Matrices")
    start_time = time.time()
    
    # Initialize user transition matrices
    user_transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Process each user's interactions
    for user_id, user_df in df.groupby('user_id'):
        # Sort by timestamp to ensure correct sequence
        user_df = user_df.sort_values('timestamp')
        
        # Count transitions from each item to next item
        for i in range(len(user_df) - 1):
            current_item = user_df.iloc[i]['item_id']
            next_item = user_df.iloc[i]['next_item_id']
            if not pd.isna(next_item):
                user_transitions[user_id][current_item][next_item] += 1
    
    # Convert to most likely next item for each user and item
    user_models = {}
    for user_id, transitions in user_transitions.items():
        user_models[user_id] = {}
        for current_item, next_items in transitions.items():
            if next_items:  # If there are any transitions from this item
                user_models[user_id][current_item] = max(next_items.items(), key=lambda x: x[1])[0]
    
    elapsed_time = time.time() - start_time
    print(f"Built transition matrices for {len(user_models)} users in {elapsed_time:.2f} seconds")
    
    return user_models, user_transitions

def build_social_network_graph(social_df):
    """
    Build a NetworkX graph from social network data.
    
    Args:
        social_df: DataFrame with social connections
        
    Returns:
        NetworkX graph representing the social network
    """
    print_section("Building Social Network Graph")
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (users)
    unique_users = social_df['user_id'].unique()
    G.add_nodes_from(unique_users)
    
    # Add edges (connections) with influence weights as attributes
    for _, row in social_df.iterrows():
        G.add_edge(row['user_id'], row['friend_id'], weight=row['influence_weight'])
    
    # Print network statistics
    print(f"Created social network with {G.number_of_nodes()} users and {G.number_of_edges()} connections")
    
    return G

def calculate_user_item_preferences(df):
    """
    Calculate user preferences for items based on interaction frequency and engagement.
    
    Args:
        df: DataFrame with user interactions
        
    Returns:
        Dictionary mapping users to their item preferences
    """
    print_section("Calculating User Item Preferences")
    
    # Define a helper function for calculating engagement
    def calculate_engagement(group):
        return (group['view_time'] * group['click_rate']).mean()
    
    # Initialize user preferences as a regular dictionary for better serialization
    user_preferences = {}
    
    # Calculate preferences based on interaction frequency and engagement metrics
    for user_id, user_df in df.groupby('user_id'):
        # Count item occurrences
        item_counts = user_df['item_id'].value_counts()
        
        # Calculate average engagement (view_time * click_rate) per item
        item_engagement = user_df.groupby('item_id').apply(calculate_engagement)
        
        # Normalize counts and engagement
        if len(item_counts) > 0:
            norm_counts = item_counts / item_counts.max()
            
            if len(item_engagement) > 0 and item_engagement.max() > 0:
                norm_engagement = item_engagement / item_engagement.max()
                
                # Initialize preferences dictionary for this user
                user_preferences[user_id] = {}
                
                # Combine frequency and engagement (equal weights)
                for item_id in norm_counts.index:
                    engagement = norm_engagement.get(item_id, 0)
                    user_preferences[user_id][item_id] = 0.5 * norm_counts[item_id] + 0.5 * engagement
    
    print(f"Calculated item preferences for {len(user_preferences)} users")
    
    return user_preferences

def predict_with_social_influence(user_id, item_id, user_models, user_transitions, 
                                 social_graph, user_preferences, social_weight=0.3):
    """
    Predict the next item with social network influence.
    
    Args:
        user_id: ID of the target user
        item_id: Current item ID
        user_models: Dictionary of user transition matrices
        user_transitions: Dictionary of user transition counts
        social_graph: NetworkX graph of social connections
        user_preferences: Dictionary of user item preferences
        social_weight: Weight given to social influence (0-1)
        
    Returns:
        Predicted next item ID
    """
    # Get the user's own prediction if available
    own_prediction = None
    own_confidence = 0
    
    if user_id in user_models and item_id in user_models[user_id]:
        own_prediction = user_models[user_id][item_id]
        
        # Calculate confidence based on transition counts
        if user_id in user_transitions and item_id in user_transitions[user_id]:
            total_transitions = sum(user_transitions[user_id][item_id].values())
            max_count = max(user_transitions[user_id][item_id].values())
            own_confidence = max_count / total_transitions if total_transitions > 0 else 0
    
    # If user has no friends or social weight is 0, return own prediction
    if user_id not in social_graph or social_weight == 0:
        return own_prediction
    
    # Get friends and their influence weights
    friends = list(social_graph.neighbors(user_id))
    
    if not friends:
        return own_prediction
    
    # Get influence weights
    influence_weights = {}
    for friend_id in friends:
        if social_graph.has_edge(user_id, friend_id):
            influence_weights[friend_id] = social_graph[user_id][friend_id].get('weight', 1.0)
    
    # Normalize influence weights
    total_weight = sum(influence_weights.values())
    if total_weight > 0:
        for friend_id in influence_weights:
            influence_weights[friend_id] /= total_weight
    
    # Collect friends' predictions
    friend_predictions = defaultdict(float)
    
    for friend_id in friends:
        # Check if friend has a prediction for this item
        if friend_id in user_models and item_id in user_models[friend_id]:
            friend_next_item = user_models[friend_id][item_id]
            friend_weight = influence_weights.get(friend_id, 0)
            
            # Add to predictions with friend's influence weight
            friend_predictions[friend_next_item] += friend_weight
    
    # If no friend predictions, return own prediction
    if not friend_predictions:
        return own_prediction
    
    # Combine own prediction and friend predictions
    combined_predictions = defaultdict(float)
    
    # Add own prediction with weight
    if own_prediction is not None:
        combined_predictions[own_prediction] += (1 - social_weight) * own_confidence
    
    # Add friend predictions with weight
    for item, weight in friend_predictions.items():
        combined_predictions[item] += social_weight * weight
    
    # Return item with highest combined weight
    if combined_predictions:
        return max(combined_predictions.items(), key=lambda x: x[1])[0]
    
    return own_prediction

def evaluate_social_recommendations(test_df, user_models, user_transitions, 
                                   social_graph, user_preferences, social_weights=[0, 0.3, 0.5, 0.7, 1.0]):
    """
    Evaluate the recommendation model with different social influence weights.
    
    Args:
        test_df: Test DataFrame
        user_models: Dictionary of user transition matrices
        user_transitions: Dictionary of user transition counts
        social_graph: NetworkX graph of social connections
        user_preferences: Dictionary of user item preferences
        social_weights: List of social influence weights to test
        
    Returns:
        Dictionary of accuracy results for each social weight
    """
    print_section("Evaluating Social Recommendations")
    
    results = {}
    
    for social_weight in social_weights:
        print(f"\nEvaluating with social_weight = {social_weight}")
        correct_predictions = 0
        total_predictions = 0
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            true_next_item = row['next_item_id']
            
            if pd.isna(true_next_item):
                continue
            
            # Make prediction with social influence
            predicted_item = predict_with_social_influence(
                user_id, item_id, user_models, user_transitions, 
                social_graph, user_preferences, social_weight
            )
            
            if predicted_item is not None:
                total_predictions += 1
                if predicted_item == true_next_item:
                    correct_predictions += 1
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Made {total_predictions} predictions with {correct_predictions} correct")
        print(f"Accuracy: {accuracy:.4f}")
        
        results[social_weight] = accuracy
    
    return results

def visualize_social_influence_results(results, output_path=None):
    """
    Visualize the impact of social influence on recommendation accuracy.
    
    Args:
        results: Dictionary of accuracy results for each social weight
        output_path: Path to save the visualization
    """
    print_section("Visualizing Results")
    
    # Extract weights and accuracies
    weights = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(weights, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Social Influence Weight')
    plt.ylabel('Accuracy')
    plt.title('Impact of Social Influence on Recommendation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(weights)
    
    # Add value labels
    for i, (x, y) in enumerate(zip(weights, accuracies)):
        plt.text(x, y + 0.01, f'{y:.4f}', ha='center', va='bottom')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    plt.close()

def run_social_recommendation_model(interactions_path, social_network_path, test_size=0.2, random_state=42):
    """
    Run the social recommendation model pipeline.
    
    Args:
        interactions_path: Path to the user interactions data
        social_network_path: Path to the social network data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of results
    """
    start_time = time.time()
    
    # Step 1: Load data
    interactions_df, social_df = load_data(interactions_path, social_network_path)
    
    # Step 2: Build user transition matrices
    user_models, user_transitions = build_user_transition_matrices(interactions_df)
    
    # Step 3: Build social network graph
    social_graph = build_social_network_graph(social_df)
    
    # Step 4: Calculate user item preferences
    user_preferences = calculate_user_item_preferences(interactions_df)
    
    # Step 5: Split data for evaluation
    # Get unique user IDs
    unique_users = interactions_df['user_id'].unique()
    
    # Split users into train and test sets
    train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=random_state)
    
    # Split data based on user assignment
    train_df = interactions_df[interactions_df['user_id'].isin(train_users)].copy()
    test_df = interactions_df[interactions_df['user_id'].isin(test_users)].copy()
    
    print(f"Training users: {len(train_users)}, Test users: {len(test_users)}")
    print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Step 6: Evaluate with different social weights
    social_weights = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    results = evaluate_social_recommendations(
        test_df, user_models, user_transitions, social_graph, user_preferences, social_weights
    )
    
    # Step 7: Visualize results
    output_dir = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/plots"
    os.makedirs(output_dir, exist_ok=True)
    visualize_social_influence_results(
        results, 
        output_path=f"{output_dir}/social_influence_impact.png"
    )
    
    # Step 8: Save models
    models_dir = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/models/social"
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(user_models, f"{models_dir}/user_transition_models.joblib")
    joblib.dump(social_graph, f"{models_dir}/social_graph.joblib")
    joblib.dump(user_preferences, f"{models_dir}/user_preferences.joblib")
    
    print(f"\nModels saved to:")
    print(f"- User Transition Models: {models_dir}/user_transition_models.joblib")
    print(f"- Social Graph: {models_dir}/social_graph.joblib")
    print(f"- User Preferences: {models_dir}/user_preferences.joblib")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    
    return results

def main():
    """
    Main function to run the social recommendation model.
    """
    # Configuration
    interactions_path = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/data/samples/multi_user_sample.csv"
    social_network_path = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/data/social_network/social_connections.csv"
    
    # Check if social network data exists, if not generate it
    if not os.path.exists(social_network_path):
        print("Social network data not found. Generating synthetic social network...")
        from social_network_generator import generate_social_network
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(social_network_path), exist_ok=True)
        
        # Generate social network
        generate_social_network(
            user_data_path=interactions_path,
            output_path=social_network_path,
            connection_probability=0.15,
            seed=42
        )
    
    # Run the social recommendation model
    results = run_social_recommendation_model(
        interactions_path=interactions_path,
        social_network_path=social_network_path,
        test_size=0.2,
        random_state=42
    )
    
    # Print summary
    print_section("Summary of Results")
    print("Impact of Social Influence Weight on Recommendation Accuracy:")
    for weight, accuracy in results.items():
        print(f"Social Weight {weight:.1f}: {accuracy:.4f}")
    
    # Find optimal social weight
    optimal_weight = max(results.items(), key=lambda x: x[1])[0]
    print(f"\nOptimal social influence weight: {optimal_weight}")
    print(f"Best accuracy: {results[optimal_weight]:.4f}")

if __name__ == "__main__":
    main()

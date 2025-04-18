import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import joblib
from collections import defaultdict
import networkx as nx

from social_recommendation import predict_with_social_influence
from social_network_generator import generate_social_network, visualize_social_network

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*70}\n{title}\n{'='*70}")

def load_models(models_dir="/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/models/social"):
    """
    Load the saved social recommendation models.
    
    Args:
        models_dir: Directory containing the saved models
        
    Returns:
        Tuple of (user_models, social_graph, user_preferences)
    """
    print_section("Loading Models")
    
    user_models_path = f"{models_dir}/user_transition_models.joblib"
    social_graph_path = f"{models_dir}/social_graph.joblib"
    user_preferences_path = f"{models_dir}/user_preferences.joblib"
    
    # Check if models exist
    if not all(os.path.exists(path) for path in [user_models_path, social_graph_path, user_preferences_path]):
        print("Models not found. Please run social_recommendation.py first.")
        return None, None, None
    
    # Load models
    print(f"Loading user transition models from: {user_models_path}")
    user_models = joblib.load(user_models_path)
    
    print(f"Loading social graph from: {social_graph_path}")
    social_graph = joblib.load(social_graph_path)
    
    print(f"Loading user preferences from: {user_preferences_path}")
    user_preferences = joblib.load(user_preferences_path)
    
    print(f"Loaded models for {len(user_models)} users with a social network of {social_graph.number_of_nodes()} users")
    
    return user_models, social_graph, user_preferences

def load_user_transitions(interactions_path):
    """
    Load user interaction data and build transition counts.
    
    Args:
        interactions_path: Path to the user interactions data
        
    Returns:
        Dictionary of user transition counts
    """
    print_section("Loading User Transitions")
    
    # Load user interactions
    print(f"Loading user interactions from: {interactions_path}")
    df = pd.read_csv(interactions_path)
    
    # Initialize user transition counts
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
    
    print(f"Built transition counts for {len(user_transitions)} users")
    
    return user_transitions

def test_social_recommendations(user_id, item_id, user_models, user_transitions, social_graph, user_preferences):
    """
    Test social recommendations for a specific user and item with different social weights.
    
    Args:
        user_id: ID of the target user
        item_id: Current item ID
        user_models: Dictionary of user transition matrices
        user_transitions: Dictionary of user transition counts
        social_graph: NetworkX graph of social connections
        user_preferences: Dictionary of user item preferences
    """
    print_section(f"Testing Social Recommendations for User {user_id}, Item {item_id}")
    
    # Check if user and item exist in the data
    if user_id not in user_models:
        print(f"User {user_id} not found in the model")
        return
    
    if item_id not in user_models[user_id]:
        print(f"Item {item_id} not found for user {user_id}")
        return
    
    # Get user's own prediction
    own_prediction = user_models[user_id][item_id]
    print(f"User's own prediction (no social influence): {own_prediction}")
    
    # Get friends
    if user_id in social_graph:
        friends = list(social_graph.neighbors(user_id))
        print(f"User has {len(friends)} friends")
        
        # Print friends' predictions
        print("\nFriends' predictions:")
        for friend_id in friends:
            if friend_id in user_models and item_id in user_models[friend_id]:
                friend_prediction = user_models[friend_id][item_id]
                influence_weight = social_graph[user_id][friend_id].get('weight', 0)
                print(f"Friend {friend_id}: {friend_prediction} (influence: {influence_weight:.4f})")
    else:
        print("User has no friends in the social network")
    
    # Test with different social weights
    print("\nPredictions with different social weights:")
    social_weights = [0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    for weight in social_weights:
        prediction = predict_with_social_influence(
            user_id, item_id, user_models, user_transitions, 
            social_graph, user_preferences, weight
        )
        print(f"Social weight {weight:.1f}: {prediction}")

def visualize_user_network(user_id, social_graph, output_path=None, max_depth=2):
    """
    Visualize a user's social network neighborhood.
    
    Args:
        user_id: ID of the target user
        social_graph: NetworkX graph of social connections
        output_path: Path to save the visualization
        max_depth: Maximum depth of neighbors to include
    """
    print_section(f"Visualizing Social Network for User {user_id}")
    
    if user_id not in social_graph:
        print(f"User {user_id} not found in the social network")
        return
    
    # Extract the user's neighborhood
    neighborhood = {user_id}
    current_depth = 0
    frontier = {user_id}
    
    while current_depth < max_depth:
        next_frontier = set()
        for node in frontier:
            neighbors = set(social_graph.neighbors(node))
            next_frontier.update(neighbors)
        
        neighborhood.update(next_frontier)
        frontier = next_frontier
        current_depth += 1
    
    # Create subgraph of the neighborhood
    subgraph = social_graph.subgraph(neighborhood)
    
    # Visualize
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    
    # Draw nodes with different colors for the target user
    node_colors = ['red' if node == user_id else 'lightblue' for node in subgraph.nodes()]
    node_sizes = [800 if node == user_id else 500 for node in subgraph.nodes()]
    
    # Draw the network
    nx.draw(subgraph, pos, with_labels=True, node_color=node_colors, 
            node_size=node_sizes, edge_color='gray', linewidths=0.5,
            font_size=8, font_weight='bold')
    
    plt.title(f"Social Network for User {user_id} (Depth {max_depth})")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    else:
        plt.show()
    
    plt.close()

def compare_item_predictions(item_id, user_models, social_graph):
    """
    Compare predictions for the same item across different users.
    
    Args:
        item_id: Item ID to compare
        user_models: Dictionary of user transition matrices
        social_graph: NetworkX graph of social connections
    """
    print_section(f"Comparing Predictions for Item {item_id} Across Users")
    
    # Find users who have this item in their model
    users_with_item = []
    for user_id, model in user_models.items():
        if item_id in model:
            users_with_item.append(user_id)
    
    if not users_with_item:
        print(f"No users have item {item_id} in their models")
        return
    
    print(f"Found {len(users_with_item)} users with item {item_id}")
    
    # Group users by their predictions
    prediction_groups = defaultdict(list)
    for user_id in users_with_item:
        prediction = user_models[user_id][item_id]
        prediction_groups[prediction].append(user_id)
    
    # Print prediction groups
    print(f"\nUsers grouped by their predictions for item {item_id}:")
    for prediction, users in prediction_groups.items():
        print(f"Prediction {prediction}: {len(users)} users")
        if len(users) <= 5:  # Only print all users if there are 5 or fewer
            print(f"  Users: {users}")
        else:
            print(f"  Sample users: {users[:5]}...")
    
    # Analyze social connections within prediction groups
    print("\nAnalyzing social connections within prediction groups:")
    for prediction, users in prediction_groups.items():
        if len(users) <= 1:
            continue
            
        # Count connections within the group
        connections = 0
        for i, user1 in enumerate(users):
            for user2 in users[i+1:]:
                if social_graph.has_edge(user1, user2):
                    connections += 1
        
        max_possible = (len(users) * (len(users) - 1)) / 2
        connection_density = connections / max_possible if max_possible > 0 else 0
        
        print(f"Prediction {prediction} group:")
        print(f"  Users: {len(users)}")
        print(f"  Internal connections: {connections} out of {max_possible:.0f} possible ({connection_density:.4f})")

def main():
    """
    Main function to test the social recommendation model.
    """
    # Configuration
    interactions_path = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/data/samples/multi_user_sample.csv"
    output_dir = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    user_models, social_graph, user_preferences = load_models()
    
    # If models don't exist, exit
    if user_models is None:
        print("Please run social_recommendation.py first to generate the models.")
        return
    
    # Load user transitions
    user_transitions = load_user_transitions(interactions_path)
    
    # Select a random user and item for testing
    sample_users = list(user_models.keys())
    if not sample_users:
        print("No users found in the models.")
        return
    
    sample_user = sample_users[0]  # Take the first user
    
    # Find an item for this user
    user_items = list(user_models[sample_user].keys())
    if not user_items:
        print(f"No items found for user {sample_user}.")
        return
    
    sample_item = user_items[0]  # Take the first item
    
    # Test social recommendations
    test_social_recommendations(
        user_id=sample_user,
        item_id=sample_item,
        user_models=user_models,
        user_transitions=user_transitions,
        social_graph=social_graph,
        user_preferences=user_preferences
    )
    
    # Visualize user's social network
    visualize_user_network(
        user_id=sample_user,
        social_graph=social_graph,
        output_path=f"{output_dir}/user_{sample_user}_network.png",
        max_depth=1
    )
    
    # Compare item predictions across users
    compare_item_predictions(
        item_id=sample_item,
        user_models=user_models,
        social_graph=social_graph
    )

if __name__ == "__main__":
    main()

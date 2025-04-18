import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import random
import os
import matplotlib.pyplot as plt

def generate_social_network(user_data_path, output_path=None, connection_probability=0.1, seed=42):
    """
    Generate a synthetic social network based on user interaction data.
    
    Args:
        user_data_path: Path to the CSV file with user interaction data
        output_path: Path to save the generated social network data
        connection_probability: Probability of connection between users (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX graph representing the social network
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Load user data
    print(f"Loading user data from: {user_data_path}")
    df = pd.read_csv(user_data_path)
    
    # Get unique users
    unique_users = df['user_id'].unique()
    num_users = len(unique_users)
    print(f"Found {num_users} unique users")
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes (users)
    for user in unique_users:
        G.add_node(user)
    
    # Add edges (connections) based on probability
    edges_added = 0
    for i, user1 in enumerate(unique_users):
        for user2 in unique_users[i+1:]:
            if random.random() < connection_probability:
                G.add_edge(user1, user2)
                edges_added += 1
    
    print(f"Created social network with {num_users} users and {edges_added} connections")
    
    # Calculate network statistics
    avg_degree = sum(dict(G.degree()).values()) / num_users
    print(f"Average connections per user: {avg_degree:.2f}")
    
    # Generate influence weights (how much each friend influences a user)
    influence_weights = {}
    for user in G.nodes():
        friends = list(G.neighbors(user))
        if friends:
            # Generate random weights and normalize them to sum to 1
            weights = np.random.dirichlet(np.ones(len(friends)))
            influence_weights[user] = dict(zip(friends, weights))
    
    # Save the network data if output path is provided
    if output_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert network to DataFrame format for saving
        edges = []
        for user1, user2 in G.edges():
            weight = influence_weights.get(user1, {}).get(user2, 0)
            edges.append({
                'user_id': user1,
                'friend_id': user2,
                'influence_weight': weight
            })
            # Add the reverse direction as well
            weight = influence_weights.get(user2, {}).get(user1, 0)
            edges.append({
                'user_id': user2,
                'friend_id': user1,
                'influence_weight': weight
            })
        
        edges_df = pd.DataFrame(edges)
        edges_df.to_csv(output_path, index=False)
        print(f"Saved social network data to: {output_path}")
    
    return G, influence_weights

def visualize_social_network(G, output_path=None, max_nodes=100):
    """
    Visualize the social network.
    
    Args:
        G: NetworkX graph representing the social network
        output_path: Path to save the visualization
        max_nodes: Maximum number of nodes to visualize
    """
    # If the network is too large, take a subset for visualization
    if len(G) > max_nodes:
        nodes = list(G.nodes())
        subset_nodes = random.sample(nodes, max_nodes)
        G = G.subgraph(subset_nodes)
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    
    # Draw the network
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, edge_color='gray', linewidths=0.5,
            font_size=8, font_weight='bold')
    
    plt.title(f"Social Network Visualization (showing {len(G)} users)")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    plt.close()

def analyze_social_network(G):
    """
    Analyze the social network and print key metrics.
    
    Args:
        G: NetworkX graph representing the social network
    """
    print("\nSocial Network Analysis:")
    print(f"Number of users (nodes): {G.number_of_nodes()}")
    print(f"Number of connections (edges): {G.number_of_edges()}")
    
    # Calculate degree statistics
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)
    max_degree = max(degrees)
    
    print(f"Average connections per user: {avg_degree:.2f}")
    print(f"Maximum connections for a user: {max_degree}")
    
    # Calculate connected components
    connected_components = list(nx.connected_components(G))
    print(f"Number of connected components: {len(connected_components)}")
    
    # Calculate largest connected component
    largest_cc = max(connected_components, key=len)
    print(f"Size of largest connected component: {len(largest_cc)} users ({len(largest_cc)/G.number_of_nodes()*100:.2f}%)")
    
    # Calculate clustering coefficient (measure of how nodes tend to cluster together)
    clustering_coef = nx.average_clustering(G)
    print(f"Average clustering coefficient: {clustering_coef:.4f}")

def main():
    """
    Main function to generate and analyze a synthetic social network.
    """
    # Configuration
    data_path = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/data/samples/multi_user_sample.csv"
    output_dir = "/Users/suhaskm/Desktop/Big Data/Main Project/product-recommender-ai/data/social_network"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate social network
    G, influence_weights = generate_social_network(
        user_data_path=data_path,
        output_path=f"{output_dir}/social_connections.csv",
        connection_probability=0.15,
        seed=42
    )
    
    # Analyze the network
    analyze_social_network(G)
    
    # Visualize the network
    visualize_social_network(
        G=G,
        output_path=f"{output_dir}/social_network_visualization.png",
        max_nodes=50
    )
    
    print("\nSocial network generation complete!")

if __name__ == "__main__":
    main()

import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def read_book_text(filepath):
    """
    Read the book text file.

    Args:
        filepath (str): Path to the book text file

    Returns:
        str: Content of the book
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return ""


def calculate_character_proximity(text, characters, window_size=100):
    """
    Calculate proximity between characters in the text.

    Args:
        text (str): Full text of the book
        characters (list): List of characters to analyze
        window_size (int): Number of words to consider as proximity

    Returns:
        dict: Proximity matrix between characters
    """
    # Preprocess text
    words = text.split()

    # Initialize proximity matrix
    proximity = {char1: {char2: 0 for char2 in characters}
                 for char1 in characters}

    # Iterate through words to find character proximities
    for i, word in enumerate(words):
        # Check if current word is a character name
        current_chars = [char for char in characters if char in word]

        if current_chars:
            # Look within window size before and after
            start = max(0, i - window_size)
            end = min(len(words), i + window_size)

            window_words = words[start:end]

            # Find other characters in the window
            for current_char in current_chars:
                for other_char in characters:
                    if other_char != current_char:
                        if any(other_char in w for w in window_words):
                            proximity[current_char][other_char] += 1
                            proximity[other_char][current_char] += 1

    return proximity


def create_network(proximity):
    """
    Create a networkx graph from proximity matrix.

    Args:
        proximity (dict): Proximity matrix between characters

    Returns:
        networkx.Graph: Weighted graph of character relationships
    """
    G = nx.Graph()

    # Add nodes
    for char in proximity:
        G.add_node(char)

    # Add weighted edges
    for char1 in proximity:
        for char2 in proximity[char1]:
            if char1 != char2 and proximity[char1][char2] > 0:
                G.add_edge(char1, char2, weight=proximity[char1][char2])

    return G


def invert_edge_weights(G):
    """
    Invert edge weights while maintaining proportionality.

    Args:
        G (networkx.Graph): Input graph

    Returns:
        networkx.Graph: Graph with inverted edge weights
    """
    # Create a copy of the graph
    H = G.copy()

    # Get current weights
    weights = [d['weight'] for (u, v, d) in G.edges(data=True)]

    # Invert weights while maintaining proportionality
    max_weight = max(weights)
    min_weight = min(weights)

    for u, v in H.edges():
        current_weight = H[u][v]['weight']

        # Linear transformation to invert weights
        inverted_weight = max_weight + min_weight - current_weight
        H[u][v]['weight'] = inverted_weight

    return H


def calculate_betweenness(G):
    """
    Calculate betweenness centrality for each node.

    Args:
        G (networkx.Graph): Input graph

    Returns:
        dict: Betweenness centrality for each node
    """
    return nx.betweenness_centrality(G, weight='weight')


def calculate_circuit_rank(G):
    """
    Calculate circuit rank of the graph.

    Args:
        G (networkx.Graph): Input graph

    Returns:
        int: Circuit rank of the graph
    """
    # Number of edges
    e = G.number_of_edges()

    # Number of vertices
    v = G.number_of_nodes()

    # Number of connected components
    c = nx.number_connected_components(G)

    # Circuit rank calculation
    return e - v + c


def visualize_network(G, betweenness):
    """
    Visualize the network with node sizes based on betweenness centrality.

    Args:
        G (networkx.Graph): Input graph
        betweenness (dict): Betweenness centrality for each node
    """
    plt.figure(figsize=(12, 8))

    # Normalize edge weights for visualization
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]

    # Normalize node sizes based on betweenness
    node_sizes = [betweenness[node] * 5000 for node in G.nodes()]

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=[w / max(weights) * 5 for w in weights],
                           edge_color='gray', alpha=0.6)
    nx.draw_networkx_labels(G, pos)

    plt.title("Game of Thrones Character Network")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    # Define characters
    characters = [
        'Arya', 'Bran', 'Brienne', 'Catelyn', 'Cersei',
        'Daenerys', 'Jaime', 'Melisandre', 'Petyr', 'Robert'
    ]

    # Read book text (you'll need to replace with actual book file path)
    book_text = read_book_text('public/A-Game-Of-Thrones.txt')

    # Calculate character proximities
    proximity = calculate_character_proximity(book_text, characters)

    # Create initial network
    G = create_network(proximity)

    # Invert edge weights
    G_inverted = invert_edge_weights(G)

    # Calculate betweenness centrality
    betweenness = calculate_betweenness(G_inverted)

    # Calculate circuit rank
    circuit_rank = calculate_circuit_rank(G_inverted)

    # Print betweenness centrality (sorted)
    print("Betweenness Centrality (sorted):")
    for char, centrality in sorted(betweenness.items(), key=lambda x: x[1], reverse=True):
        print(f"{char}: {centrality}")

    print(f"\nCircuit Rank: {circuit_rank}")

    # Visualize network
    visualize_network(G_inverted, betweenness)


if __name__ == "__main__":
    main()